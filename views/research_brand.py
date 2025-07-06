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
    Robust Markdown → PDF converter with comprehensive error handling.

    Args:
        md: Markdown content as str, bytes, bytearray, or None
        
    Returns:
        bytes: PDF content ready for download
        
    Raises:
        ValueError: If input is invalid or empty
        RuntimeError: If PDF generation fails
        
    Features:
        • Handles None/empty inputs gracefully
        • Supports #/##/### headings, -* bullets, blank lines, paragraphs
        • Strips unsupported Unicode characters (Latin-1 only)
        • Hard-splits over-wide words to prevent fpdf failures
        • Comprehensive error handling and logging
        • Memory-efficient processing for large documents
    """
    
    def _log_warning(msg: str):
        """Log warning with fallback if streamlit not available"""
        if HAS_STREAMLIT:
            st.warning(msg)
        else:
            logging.warning(msg)
    
    def _log_error(msg: str):
        """Log error with fallback if streamlit not available"""
        if HAS_STREAMLIT:
            st.error(msg)
        else:
            logging.error(msg)

    # ── 1. Input validation and normalization ──────────────────────────
    if md is None:
        raise ValueError("Input markdown content cannot be None")
    
    if isinstance(md, (bytes, bytearray)):
        if len(md) == 0:
            raise ValueError("Input markdown content cannot be empty")
        try:
            md = md.decode("utf-8")
        except UnicodeDecodeError as e:
            _log_warning(f"UTF-8 decode error: {e}. Attempting with error replacement.")
            try:
                md = md.decode("utf-8", errors="replace")
            except Exception as e2:
                raise ValueError(f"Failed to decode input bytes: {e2}")
    
    if not isinstance(md, str):
        raise ValueError(f"Input must be str, bytes, or bytearray, got {type(md)}")
    
    if not md.strip():
        raise ValueError("Input markdown content cannot be empty or whitespace-only")
    
    # Convert to Latin-1, removing unsupported Unicode
    try:
        original_len = len(md)
        md = md.encode("latin-1", "ignore").decode("latin-1")
        if len(md) < original_len:
            _log_warning(f"Removed {original_len - len(md)} unsupported Unicode characters from content")
    except Exception as e:
        raise ValueError(f"Failed to process text encoding: {e}")

    # ── 2. PDF setup with error handling ───────────────────────────────
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Test font availability
        try:
            pdf.set_font("Helvetica", size=11)
        except Exception:
            # Fallback to Arial if Helvetica not available
            try:
                pdf.set_font("Arial", size=11)
                font_name = "Arial"
            except Exception:
                # Last resort - use default font
                pdf.set_font("Times", size=11)
                font_name = "Times"
        else:
            font_name = "Helvetica"
            
        pdf.add_page()
        
        # Calculate printable width safely
        try:
            full_width = pdf.w - pdf.l_margin - pdf.r_margin
            if full_width <= 0:
                raise ValueError("Invalid page dimensions")
        except Exception as e:
            raise RuntimeError(f"Failed to calculate page dimensions: {e}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PDF: {e}")

    # ── 3. Helper functions with robust error handling ─────────────────
    def split_long_word(word: str, col_width: float) -> List[str]:
        """Split a word that's too wide for the column"""
        if not word or col_width <= 0:
            return [word] if word else []
        
        parts, chunk = [], ""
        try:
            for ch in word:
                test_width = pdf.get_string_width(chunk + ch)
                if test_width > col_width and chunk:
                    parts.append(chunk)
                    chunk = ch
                else:
                    chunk += ch
            if chunk:
                parts.append(chunk)
        except Exception as e:
            _log_warning(f"Error splitting word '{word[:20]}...': {e}")
            # Fallback: split by character count
            chunk_size = max(1, int(col_width / 10))  # rough estimate
            parts = [word[i:i+chunk_size] for i in range(0, len(word), chunk_size)]
        
        return parts if parts else [word]

    def write_wrapped(text: str, indent_level: int = 0):
        """Write text with word wrapping and indentation"""
        if not text:
            return
            
        try:
            indent_pt = max(0, indent_level * 4)
            col_width = max(10, full_width - indent_pt)  # minimum 10pt width
            
            # Process tokens with caching
            width_cache = {}
            safe_parts = []
            
            for token in text.split():
                if not token:
                    continue
                    
                try:
                    if token in width_cache:
                        token_width = width_cache[token]
                    else:
                        token_width = pdf.get_string_width(token)
                        width_cache[token] = token_width
                    
                    if token_width > col_width:
                        safe_parts.extend(split_long_word(token, col_width))
                    else:
                        safe_parts.append(token)
                except Exception as e:
                    _log_warning(f"Error processing token '{token[:20]}...': {e}")
                    safe_parts.append(token)  # Add anyway
            
            if safe_parts:
                safe_line = " ".join(safe_parts)
                pdf.set_x(pdf.l_margin + indent_pt)
                pdf.multi_cell(col_width, 6, txt=safe_line)
                
        except Exception as e:
            _log_error(f"Error writing wrapped text: {e}")
            # Fallback: write without wrapping
            try:
                pdf.multi_cell(0, 6, txt=text[:1000])  # Truncate if too long
            except Exception:
                pass  # Give up on this text

    # ── 4. Markdown processing with error handling ─────────────────────
    try:
        lines = md.splitlines()
        total_lines = len(lines)
        
        for line_num, raw in enumerate(lines, 1):
            try:
                line = raw.rstrip()
                
                # Skip extremely long lines (potential memory issue)
                if len(line) > 10000:
                    _log_warning(f"Skipping extremely long line {line_num} ({len(line)} characters)")
                    continue

                # Headings
                h = re.match(r"^(#{1,6})\s+(.*)$", line)  # Support up to h6
                if h:
                    level = min(len(h.group(1)), 3)  # Cap at h3 for formatting
                    title = h.group(2)[:200]  # Limit title length
                    
                    size_map = {1: 16, 2: 14, 3: 12}
                    try:
                        pdf.set_font(font_name, style="B", size=size_map[level])
                        pdf.multi_cell(0, 8, txt=title)
                        pdf.ln(1)
                        pdf.set_font(font_name, size=11)
                    except Exception as e:
                        _log_warning(f"Error formatting heading on line {line_num}: {e}")
                        write_wrapped(title)  # Fallback
                    continue

                # Bullets
                b = re.match(r"^\s*[-*+]\s+(.*)$", line)  # Support +, -, *
                if b:
                    bullet_text = b.group(1)
                    try:
                        pdf.set_x(pdf.l_margin)
                        pdf.cell(5, 6, "•")
                        write_wrapped(bullet_text, indent_level=1)
                    except Exception as e:
                        _log_warning(f"Error formatting bullet on line {line_num}: {e}")
                        write_wrapped(f"• {bullet_text}")  # Fallback
                    continue

                # Blank lines
                if not line.strip():
                    try:
                        pdf.ln(4)
                    except Exception:
                        pass  # Skip if fails
                    continue

                # Regular paragraphs
                write_wrapped(line)
                
            except Exception as e:
                _log_warning(f"Error processing line {line_num}: {e}")
                continue  # Skip problematic lines
                
    except Exception as e:
        raise RuntimeError(f"Failed to process markdown content: {e}")

    # ── 5. Generate and return PDF bytes ───────────────────────────────
    try:
        out = pdf.output(dest="S")
        
        # Handle different output types from different fpdf versions
        if isinstance(out, bytes):
            return out
        elif isinstance(out, bytearray):
            return bytes(out)
        elif isinstance(out, str):
            return out.encode("latin-1")
        else:
            # Last resort conversion
            return str(out).encode("latin-1")
            
    except Exception as e:
        raise RuntimeError(f"Failed to generate PDF output: {e}")


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

