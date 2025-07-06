import os
import time
import streamlit as st
from openai import OpenAI, OpenAIError
import time, concurrent.futures
from fpdf import FPDF
import httpx
import textwrap
import re                  

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



def md_to_pdf(md) -> bytes:
    """
    Robust Markdown → PDF (Helvetica, Latin-1 only).

    • Accepts str / bytes / bytearray.
    • All non-Latin-1 (unsupported Unicode) characters will be stripped from the output (they are not transliterated or replaced).
    • Supports #/##/### headings, -* bullets, blank lines, paragraphs.
    • Hard-splits any over-wide word so fpdf2 never fails.
    • Always returns bytes for Streamlit download_button.
    """

    # ── 1. normalise to Latin-1 str (removes unsupported Unicode chars) ──
    if isinstance(md, (bytes, bytearray)):
        try:
            md = md.decode("utf-8")
        except UnicodeDecodeError:
            st.warning("Some characters could not be decoded from UTF-8 and were replaced. These will be omitted from the PDF.")
            md = md.decode("utf-8", errors="replace")
    md = md.encode("latin-1", "ignore").decode("latin-1")

    # ── 2. base PDF setup ────────────────────────────────────────────────
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", size=11)       # font FIRST for width metrics
    pdf.add_page()

    full_width = pdf.w - pdf.l_margin - pdf.r_margin  # printable width (pt)

    # helper: split one very-long token into <full_width> chunks
    def split_long_word(word: str, col_width: float) -> List[str]:
        parts, chunk = [], ""
        for ch in word:
            if pdf.get_string_width(chunk + ch) > col_width:
                if chunk:
                    parts.append(chunk)
                chunk = ch
            else:
                chunk += ch
        if chunk:
            parts.append(chunk)
        return parts
    def write_wrapped(text: str, indent_level: int = 0):
        indent_pt  = indent_level * 4
        col_width  = full_width - indent_pt
        # Cache for token widths
        width_cache = {}
        # 1) break every token that exceeds the column width
        safe_parts = []
        for token in text.split():
            if token in width_cache:
                token_width = width_cache[token]
            else:
                token_width = pdf.get_string_width(token)
                width_cache[token] = token_width
            if token_width > col_width:
                safe_parts.extend(split_long_word(token, col_width))
            else:
                safe_parts.append(token)
        safe_line = " ".join(safe_parts)
        # 2) set cursor to indent & write within the column
        pdf.set_x(pdf.l_margin + indent_pt)
        pdf.multi_cell(col_width, 6, txt=safe_line)

    # ── 3. very-light Markdown renderer ─────────────────────────────────
    for raw in md.splitlines():
        line = raw.rstrip()

        # Headings -------------------------------------------------------
        h = re.match(r"^(#{1,3})\s+(.*)$", line)
        if h:
            level, title = len(h.group(1)), h.group(2)
            pdf.set_font("Helvetica", style="B", size={1:16, 2:14, 3:12}[level])
            pdf.multi_cell(0, 8, txt=title)
            pdf.ln(1)
            pdf.set_font("Helvetica", size=11)
            continue

        # Bullets --------------------------------------------------------
        b = re.match(r"^\s*[-*]\s+(.*)$", line)
        if b:
            pdf.set_x(pdf.l_margin)        # reset to left margin
            pdf.cell(5, 6, "•")            # bullet mark
            write_wrapped(b.group(1), indent_level=1)
            continue

        # Blank line -----------------------------------------------------
        if line.strip() == "":
            pdf.ln(4)
            continue

        # Paragraphs -----------------------------------------------------
        write_wrapped(line)

    out = pdf.output(dest="S")
    if isinstance(out, bytearray):
        out = bytes(out)
    elif isinstance(out, bytes):
        pass  # already bytes, do nothing
    elif isinstance(out, str):
        out = out.encode("latin-1")
    else:
        raise TypeError("Unexpected output type from FPDF: {}. Expected bytes, str, or bytearray.".format(type(out)))
    return out


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

