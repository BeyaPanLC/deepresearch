import os
import time
import streamlit as st
from openai import OpenAI, OpenAIError
import time, concurrent.futures
from fpdf import FPDF
import httpx
import textwrap
import re           
from typing import List, Optional, Union
import logging       

TOTAL_EST_SECS = 30 * 60            # show progress for up to 30 minutes
REFRESH_EVERY  = 0.5   

OPENAI_TIMEOUT = httpx.Timeout(  # seconds
    connect=60.0,  read=1800.0, write=1800.0,  pool=None
)

openai_client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    timeout=OPENAI_TIMEOUT,          # 30-min read window
    max_retries=3   
    )


system_message = """
You are Deep Research, an advanced research agent created by OpenAI. Your task is to deliver a rigorous, data-driven report suitable for private-equity due-diligence.

## Research Objective

Generate a commercial due diligence report on the company described below, focusing on customer feedback, competitive positioning, potential investment risks, as well as additional key questions included below. Prioritize unbiased qualitative insights from customer reviews, Reddit discussions, and credible news sources.
- **Target Company:** [Name & URL]
- **Key Competitors:** [List 3‑5 competitors]
- **Comparison Criteria (Table Columns):** [Optional—list preferred criteria for the competitive table; default set provided in Section 4]
- **Additional Key Questions:** [Any specific exploratory questions]

---

### 1) Company Overview
Provide a concise 1‑2 paragraph summary addressing:
- Primary market proposition and unique differentiators.
- Key insights from preliminary competitive and customer feedback analysis.

---

### 2) Market Category Analysis
Present a structured 3‑4 paragraph analysis covering:
- Definition and market scope of the relevant industry/category.
- Current industry trends and innovations.
- Typical customer demographics and consumer behaviors.
- Major opportunities and challenges within the market.

---

### 3) Voice of the Customer
Summarize key customer feedback sourced from unbiased platforms (e.g., Reddit, review sites, credible media). Organize feedback by clear themes such as:
- Product satisfaction (usability, reliability, effectiveness).
- Service quality and user interactions (human or AI).
- Customer support experience.
- Value perception and pricing.

---

### 4) Competitive Overview
#### Summary
Provide a 1‑2 paragraph synthesis highlighting the target company's relative strengths, weaknesses, and distinctive factors compared to competitors.

#### Competitor Comparison Table
Using the **Comparison Criteria** specified in the Analysis Parameters, construct a table comparing the target company with 3‑5 key competitors. If no criteria are provided, include the relevant subset of columns below (adapt naming to specific company/category being research):
- Primary Product / Service Offering
- Positioning / Pricing / Value Proposition
- Distribution Channels (e.g., e‑commerce, retail, DTC, wholesale)
- Customer Satisfaction & Loyalty / Brand Strength / Awareness
- Size & Scale (revenue, user base, locations)
- Growth Trajectory
- Funding & Key Investors

Quantify wherever available and supplement with qualitative ratings (e.g., **High, Medium, Low**) accompanied by brief (5-10 word) explanations.

---

Output Format
Return a fully formatted report following the structure above.
Do not reveal chain-of-thought; output only the polished report ready for investor review.
Keep your output short, within 10000 words and within 20 sources


"""






# ---------- 1.  Helper --------------------------------------------------------
def generate_due_diligence_report_stream(
        query: str,
        system_template: str,
        client: OpenAI,
        model: str = "o3-deep-research",
) -> str:
    """Stream Deep-Research output. Ignores events that carry no text."""
    chunks = []

    try:
        stream = client.responses.create(
            model=model,
            stream=True,
            timeout=OPENAI_TIMEOUT,
            input=[
                {"role": "developer",
                 "content":[{"type":"input_text","text":system_template}]},
                {"role": "user",
                 "content":[{"type":"input_text","text":query}]},
            ],
            reasoning={"summary": "auto"},
            tools=[
                {"type":"web_search_preview"},
                {"type":"code_interpreter",
                 "container":{"type":"auto","file_ids":[]}},
            ],
        )

        for ev in stream:
            text = None

            # ───── ResponseCreatedEvent ──────────────────────────────
            if getattr(ev, "response", None):
                out = ev.response.output
                if out and getattr(out[-1], "content", None):
                    text = out[-1].content[0].text

            # ───── ResponseOutputItemAddedEvent ──────────────────────
            elif getattr(ev, "item", None) and getattr(ev.item, "content", None):
                text = ev.item.content[0].text

            # ───── ResponseOutputItemDeltaEvent ──────────────────────
            elif getattr(ev, "delta", None) and getattr(ev.delta, "content", None):
                text = ev.delta.content[0].text

            # ───── All other events (e.g. ResponseReasoningItem) ─────
            #   have no .content → skip silently
            if text:
                chunks.append(text)

                # optional live preview every ~20 tokens
                if len(chunks) % 20 == 0:
                    st.session_state.partial = "".join(chunks)
                    st.experimental_rerun()

        return "".join(chunks)

    except OpenAIError as e:
        return f"❌ **OpenAI error:** {e}"
    except Exception as e:
        return f"⚠️ **Unexpected error:** {e}"



def md_to_pdf(md: Union[str, bytes, bytearray, None]) -> bytes:
    """
    Simple, forgiving Markdown → PDF converter.
    Ignores all errors and produces a PDF no matter what.
    """
    
    # ── 1. Just get some text, ignore everything else ──────────────────
    if md is None:
        md = "Empty document"
    
    if isinstance(md, (bytes, bytearray)):
        try:
            md = md.decode("utf-8", errors="ignore")
        except:
            md = "Could not decode content"
    
    if not isinstance(md, str):
        md = str(md)
    
    if not md.strip():
        md = "Empty document"
    
    # Just keep basic characters, ignore everything else
    try:
        md = ''.join(char for char in md if ord(char) < 256)
        md = md.encode("latin-1", errors="ignore").decode("latin-1")
    except:
        md = "Text encoding failed"

    # ── 2. Basic PDF setup ─────────────────────────────────────────────
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        try:
            pdf.set_font("Helvetica", size=11)
        except:
            try:
                pdf.set_font("Arial", size=11)
            except:
                pdf.set_font("Times", size=11)
        pdf.add_page()
        
        full_width = pdf.w - pdf.l_margin - pdf.r_margin
        if full_width <= 0:
            full_width = 180  # fallback width
    except:
        # If PDF setup fails completely, return minimal PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Times", size=11)
        full_width = 180

    # ── 3. Super simple helpers ────────────────────────────────────────
    def safe_write(text: str, bold: bool = False, size: int = 11):
        """Write text, ignore all errors"""
        if not text:
            return
        try:
            if bold:
                pdf.set_font("Helvetica", style="B", size=size)
            else:
                pdf.set_font("Helvetica", size=size)
        except:
            pass
        
        try:
            # Split long words crudely
            words = text.split()
            safe_words = []
            for word in words:
                if len(word) > 50:  # rough cut
                    safe_words.extend([word[i:i+50] for i in range(0, len(word), 50)])
                else:
                    safe_words.append(word)
            
            safe_text = " ".join(safe_words)[:1000]  # limit length
            pdf.multi_cell(0, 6, txt=safe_text)
        except:
            # Last resort
            try:
                pdf.cell(0, 6, txt=text[:100])
                pdf.ln()
            except:
                pass

    # ── 4. Parse markdown super simply ─────────────────────────────────
    try:
        lines = md.splitlines()[:500]  # limit lines
    except:
        lines = ["Could not parse content"]
    
    for line in lines:
        try:
            line = line.strip()
            if not line:
                try:
                    pdf.ln(4)
                except:
                    pass
                continue
            
            # Headings
            if line.startswith("#"):
                heading_text = line.lstrip("#").strip()
                safe_write(heading_text, bold=True, size=14)
                try:
                    pdf.ln(2)
                except:
                    pass
                continue
            
            # Bullets
            if line.startswith(("-", "*", "+")):
                bullet_text = line[1:].strip()
                safe_write(f"• {bullet_text}")
                continue
            
            # Regular text
            safe_write(line)
            
        except:
            # Skip problematic lines silently
            continue

    # ── 5. Return PDF bytes no matter what ─────────────────────────────
    try:
        out = pdf.output(dest="S")
        if isinstance(out, bytes):
            return out
        elif isinstance(out, bytearray):
            return bytes(out)
        else:
            return str(out).encode("latin-1", errors="ignore")
    except:
        # Create absolute minimal PDF as last resort
        try:
            minimal_pdf = FPDF()
            minimal_pdf.add_page()
            minimal_pdf.set_font("Times", size=12)
            minimal_pdf.cell(0, 10, "PDF Generation Error - Minimal Output")
            return bytes(minimal_pdf.output(dest="S"))
        except:
            # Return some bytes that might work
            return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n190\n%%EOF"


# -----------------------------------------------------------------------------
st.set_page_config(page_title="Brand Research", layout="wide")
st.title("📊 Company Specific Due-Diligence Report")

default_hint = ("We are a private-equity firm evaluating the attractiveness "
                "of Good Culture, a cottage-cheese company.")
user_query = st.text_area(
    "Describe the company you want to research:",
    value=default_hint, height=150)

run = st.button("🚀 Run Analysis", type="primary")

if run and user_query.strip():
    if "start" not in st.session_state:              # preserves timer on rerun
        st.session_state.start = time.time()

    prog        = st.progress(0, text="Initializing…")
    placeholder = st.empty()                         # where report appears

    # run the blocking call in a thread so UI stays live
    with concurrent.futures.ThreadPoolExecutor() as ex:
        future = ex.submit(
            generate_due_diligence_report_stream,
            user_query, system_message, openai_client
        )

        # keep updating the bar until the call finishes
        while not future.done():
            elapsed = time.time() - st.session_state.start
            pct     = min(int(elapsed / TOTAL_EST_SECS * 100), 95)
            m, s    = divmod(int(elapsed), 60)
            prog.progress(
                pct,
                text=f"Running ({m:02d}:{s:02d} / 30:00)…"
            )
            time.sleep(REFRESH_EVERY)

    report_md = future.result()
    prog.progress(100, text="Complete!")
    st.session_state.pop("start", None)              # reset for next run

    # ── show & download ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📝 Deep Research Report")
    placeholder.markdown(report_md, unsafe_allow_html=True)

    pdf_bytes = md_to_pdf(report_md)
    st.download_button(
        "📄 Download PDF",
        data=pdf_bytes,
        file_name="due_diligence_report.pdf",
        mime="application/pdf",
        help="Save the report as a PDF you can share.",
    )

elif run:
    st.warning("Please enter a query before clicking **Run Analysis**.")

