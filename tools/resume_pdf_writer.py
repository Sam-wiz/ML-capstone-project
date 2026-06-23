"""
tools/resume_pdf_writer.py

Pipeline:
  1. GPT-4o-mini applies minimal targeted edits to the full original resume
     and returns structured JSON matching the template schema.
  2. The JSON can be rendered to:
     - HTML (for in-app preview, includes footer watermark)
     - PDF  (for download, footer removed)
"""

import json
import html as _html_mod
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ─── character safety ─────────────────────────────────────────────────────────
def _s(t: str) -> str:
    """Replace common unicode chars then encode to latin-1 safely."""
    subs = {
        "–": "-", "—": "-", "→": "->", "•": "-",
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "…": "...", "·": "-",
    }
    text = t or ""
    for ch, rep in subs.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _h(t: str) -> str:
    """HTML-escape a string."""
    return _html_mod.escape(t or "")


# ─── schema ───────────────────────────────────────────────────────────────────
_SCHEMA = """{
  "name": "<candidate full name>",
  "contact": "<phone> | <email> | <LinkedIn> | <GitHub> | <Portfolio>",
  "sections": [
    {
      "heading": "<section name>",
      "type": "<entries | skills | bullets>",
      "items": "<see per-type format>"
    }
  ]
}

type=entries  ->  items is a list of:
  {"title": "<company/project>", "right": "<dates or tech>",
   "subtitle": "<role/degree or null>", "right2": "<location or null>",
   "bullets": ["<bullet text, no leading dash>"]}

type=skills   ->  items is a list of:
  {"label": "<category>", "value": "<values>"}

type=bullets  ->  items is a plain list of strings.

Keep all values clean — no trailing dashes or extra characters."""

_SYSTEM = f"""You are an expert resume editor.

Given the candidate's FULL resume and AI-suggested improvements, apply MINIMAL edits:
- Update summary/objective if provided
- Strengthen matching bullet points with better metrics or keywords
- Add missing skills
- Keep ALL other content verbatim: jobs, dates, companies, education, projects, etc.
- Do NOT invent or remove anything

Return the complete updated resume as valid JSON matching:
{_SCHEMA}

Return ONLY the JSON object — no markdown fences, no explanation."""


# ─── LLM call ─────────────────────────────────────────────────────────────────
def _clean_json(obj):
    if isinstance(obj, str):
        return obj.rstrip(" \t—–-").strip()
    if isinstance(obj, list):
        return [_clean_json(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    return obj


def generate_resume_structured(
    tailored_cv: dict,
    original_text: str,
    role_title: str = "",
    company_name: str = "",
    job_description: str = "",
) -> dict:
    """Call GPT-4o-mini to apply edits and return the structured resume dict."""
    if not original_text.strip():
        raise ValueError("No original resume text provided.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_retries=3)

    prompt = (
        f"TARGET ROLE: {role_title or 'Not specified'}\n"
        f"COMPANY: {company_name or 'Not specified'}\n\n"
        f"AI-SUGGESTED IMPROVEMENTS:\n"
        f"- Updated summary: {tailored_cv.get('summary', '')}\n"
        f"- Improved bullets:\n"
        + "\n".join(f"  * {b}" for b in tailored_cv.get("tailored_bullets", []))
        + f"\n- Skills to add: {', '.join(tailored_cv.get('skills_section', []))}\n"
        f"- Rationale: {'; '.join(tailored_cv.get('changes_made', []))}\n\n"
        f"JOB DESCRIPTION:\n{(job_description or '')[:1000]}\n\n"
        f"ORIGINAL RESUME:\n{'='*60}\n{original_text}\n{'='*60}\n\n"
        f"Return the complete updated resume as JSON."
    )

    resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return _clean_json(json.loads(raw.strip()))


# ─── HTML preview renderer ────────────────────────────────────────────────────
def render_resume_html(data: dict) -> str:
    """Render the structured resume dict to styled HTML for in-app preview."""
    parts = []
    W = """
    <style>
      .rv-wrap {
        font-family: 'Times New Roman', Georgia, serif;
        max-width: 720px; margin: 0 auto;
        padding: 36px 44px 28px;
        background: #fff;
        border: 1px solid #dde1e7;
        border-radius: 6px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        color: #1a1a1a;
        font-size: 12.5px;
        line-height: 1.45;
      }
      .rv-name {
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        margin: 0 0 5px;
        letter-spacing: 0.01em;
      }
      .rv-contact {
        text-align: center;
        color: #6b7280;
        font-size: 11px;
        margin: 0 0 8px;
      }
      .rv-hrule {
        border: none;
        border-top: 1px solid #c8cbd0;
        margin: 6px 0 12px;
      }
      .rv-section-rule {
        border: none;
        border-top: 0.5px solid #d1d5db;
        margin: 2px 0 6px;
      }
      .rv-heading {
        font-size: 13px;
        font-weight: 700;
        margin: 10px 0 0;
      }
      .rv-entry { margin-bottom: 7px; }
      .rv-row1 {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 1px;
      }
      .rv-title  { font-weight: 700; font-size: 12.5px; }
      .rv-right  { color: #6b7280; font-size: 11px; white-space: nowrap; margin-left: 10px; }
      .rv-row2   { display: flex; justify-content: space-between; align-items: baseline; }
      .rv-sub    { font-style: italic; color: #444; font-size: 11.5px; }
      .rv-right2 { color: #6b7280; font-size: 11px; white-space: nowrap; margin-left: 10px; }
      .rv-bullets { margin: 3px 0 0 0; padding: 0; list-style: none; }
      .rv-bullets li {
        padding-left: 14px;
        position: relative;
        margin-bottom: 2px;
        font-size: 12px;
        color: #2d2d2d;
      }
      .rv-bullets li::before {
        content: '\2013';
        position: absolute;
        left: 0;
        color: #555;
      }
      .rv-skill-row { display: flex; margin-bottom: 2px; font-size: 12px; }
      .rv-skill-label { font-weight: 700; width: 120px; flex-shrink: 0; color: #1a1a1a; }
      .rv-skill-val   { color: #2d2d2d; }
      .rv-footer {
        text-align: center;
        font-style: italic;
        font-size: 10px;
        color: #9ca3af;
        margin-top: 20px;
        padding-top: 10px;
        border-top: 0.5px solid #e5e7eb;
      }
    </style>
    <div class="rv-wrap">
    """
    parts.append(W)

    parts.append(f'<p class="rv-name">{_h(data.get("name",""))}</p>')
    parts.append(f'<p class="rv-contact">{_h(data.get("contact",""))}</p>')
    parts.append('<hr class="rv-hrule">')

    for section in data.get("sections", []):
        heading = section.get("heading", "")
        stype   = section.get("type", "entries")
        items   = section.get("items", [])

        parts.append(f'<p class="rv-heading">{_h(heading)}</p>')
        parts.append('<hr class="rv-section-rule">')

        if stype == "entries":
            for item in items:
                parts.append('<div class="rv-entry">')
                parts.append('<div class="rv-row1">')
                parts.append(f'<span class="rv-title">{_h(item.get("title",""))}</span>')
                parts.append(f'<span class="rv-right">{_h(item.get("right",""))}</span>')
                parts.append('</div>')
                if item.get("subtitle") or item.get("right2"):
                    parts.append('<div class="rv-row2">')
                    parts.append(f'<span class="rv-sub">{_h(item.get("subtitle") or "")}</span>')
                    parts.append(f'<span class="rv-right2">{_h(item.get("right2") or "")}</span>')
                    parts.append('</div>')
                bullets = item.get("bullets", [])
                if bullets:
                    parts.append('<ul class="rv-bullets">')
                    for b in bullets:
                        parts.append(f'<li>{_h(b)}</li>')
                    parts.append('</ul>')
                parts.append('</div>')

        elif stype == "skills":
            for row in items:
                if isinstance(row, dict):
                    parts.append(
                        f'<div class="rv-skill-row">'
                        f'<span class="rv-skill-label">{_h(row.get("label",""))}</span>'
                        f'<span class="rv-skill-val">{_h(row.get("value",""))}</span>'
                        f'</div>'
                    )

        elif stype == "bullets":
            parts.append('<ul class="rv-bullets">')
            for b in items:
                parts.append(f'<li>{_h(str(b))}</li>')
            parts.append('</ul>')

    parts.append(
        '<p class="rv-footer">'
        'Generated by Job Application Assistant &mdash; review all content before submitting'
        '</p>'
    )
    parts.append('</div>')
    return "".join(parts)


# ─── PDF renderer ─────────────────────────────────────────────────────────────
L_MARGIN = 15.0
R_MARGIN = 15.0
T_MARGIN = 12.0
B_MARGIN = 10.0
PAGE_W   = 210.0
USABLE_W = PAGE_W - L_MARGIN - R_MARGIN

SZ_NAME    = 17;  SZ_CONTACT = 8.5; SZ_SECTION = 9.5
SZ_BODY    = 8.5; SZ_SMALL   = 8.2

LH_NAME    = 8.5; LH_CONTACT = 4.5; LH_SECTION = 5.0
LH_BODY    = 4.5; LH_SMALL   = 4.3

COL_BLACK = (20,  20,  20)
COL_DARK  = (50,  50,  50)
COL_GRAY  = (110, 110, 110)
COL_RULE  = (180, 180, 180)


class ResumePDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_margins(L_MARGIN, T_MARGIN, R_MARGIN)
        self.set_auto_page_break(auto=True, margin=B_MARGIN)
        self.add_page()

    def _rule(self, t=0.25):
        self.set_draw_color(*COL_RULE)
        self.set_line_width(t)
        y = self.get_y()
        self.line(L_MARGIN, y, PAGE_W - R_MARGIN, y)

    def render_name(self, name):
        self.set_font("Times", "B", SZ_NAME)
        self.set_text_color(*COL_BLACK)
        self.cell(USABLE_W, LH_NAME, _s(name), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def render_contact(self, contact):
        self.set_font("Times", "", SZ_CONTACT)
        self.set_text_color(*COL_GRAY)
        self.cell(USABLE_W, LH_CONTACT, _s(contact), align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(1.0); self._rule(0.4); self.ln(2.0)

    def render_section_heading(self, heading):
        self.ln(0.5)
        self.set_font("Times", "B", SZ_SECTION)
        self.set_text_color(*COL_BLACK)
        self.cell(USABLE_W, LH_SECTION, _s(heading), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self._rule(0.25); self.ln(1.0)

    def render_entry_header(self, title, right, subtitle=None, right2=None):
        y = self.get_y()
        self.set_font("Times", "", SZ_SMALL); self.set_text_color(*COL_GRAY)
        rw1 = self.get_string_width(_s(right)) + 1
        self.set_xy(L_MARGIN, y)
        self.cell(USABLE_W, LH_BODY, _s(right), align="R", border=0)
        self.set_xy(L_MARGIN, y)
        self.set_font("Times", "B", SZ_BODY); self.set_text_color(*COL_BLACK)
        self.cell(USABLE_W - rw1, LH_BODY, _s(title), border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        if subtitle or right2:
            y2 = self.get_y()
            rw2 = 0
            if right2:
                self.set_font("Times", "I", SZ_SMALL); self.set_text_color(*COL_GRAY)
                rw2 = self.get_string_width(_s(right2)) + 1
                self.set_xy(L_MARGIN, y2)
                self.cell(USABLE_W, LH_SMALL, _s(right2), align="R", border=0)
            if subtitle:
                self.set_xy(L_MARGIN, y2)
                self.set_font("Times", "I", SZ_SMALL); self.set_text_color(*COL_DARK)
                self.cell(USABLE_W - rw2, LH_SMALL, _s(subtitle), border=0,
                          new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                self.set_y(y2 + LH_SMALL)

    def render_bullet(self, text, indent=4.0):
        self.set_font("Times", "", SZ_BODY); self.set_text_color(*COL_DARK)
        dash = "- "
        dw = self.get_string_width(dash)
        x0 = L_MARGIN + indent
        self.set_xy(x0, self.get_y())
        self.cell(dw, LH_BODY, dash, border=0)
        self.set_x(x0 + dw)
        self.multi_cell(USABLE_W - indent - dw, LH_BODY, _s(text.strip()),
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def render_skill_row(self, label, value):
        self.set_font("Times", "B", SZ_BODY); self.set_text_color(*COL_BLACK)
        lw = 32.0
        self.set_x(L_MARGIN)
        self.cell(lw, LH_BODY, _s(label), border=0)
        self.set_font("Times", "", SZ_BODY); self.set_text_color(*COL_DARK)
        self.multi_cell(USABLE_W - lw, LH_BODY, _s(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def render_resume_pdf_from_data(data: dict) -> bytes:
    """Render structured resume dict to PDF bytes. No footer (clean download)."""
    pdf = ResumePDF()
    pdf.render_name(data.get("name", ""))
    pdf.render_contact(data.get("contact", ""))

    for section in data.get("sections", []):
        heading = section.get("heading", "")
        stype   = section.get("type", "entries")
        items   = section.get("items", [])

        pdf.render_section_heading(heading)

        if stype == "entries":
            for idx, item in enumerate(items):
                pdf.render_entry_header(
                    title    = item.get("title", ""),
                    right    = item.get("right", ""),
                    subtitle = item.get("subtitle") or None,
                    right2   = item.get("right2") or None,
                )
                for b in item.get("bullets", []):
                    pdf.render_bullet(b)
                if idx < len(items) - 1:
                    pdf.ln(1.0)

        elif stype == "skills":
            for row in items:
                if isinstance(row, dict):
                    pdf.render_skill_row(row.get("label", ""), row.get("value", ""))
                else:
                    pdf.render_bullet(str(row))

        elif stype == "bullets":
            for b in items:
                pdf.render_bullet(str(b))

        pdf.ln(0.5)

    return bytes(pdf.output())


# ─── public API (backward-compat) ─────────────────────────────────────────────
def generate_resume_pdf(
    tailored_cv: dict,
    original_text: str = "",
    role_title: str = "",
    company_name: str = "",
    job_description: str = "",
) -> bytes:
    data = generate_resume_structured(tailored_cv, original_text, role_title,
                                      company_name, job_description)
    return render_resume_pdf_from_data(data)
