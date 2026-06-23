"""
tools/pdf_parser.py — PDF resume parser using pdfplumber.

pdfplumber is chosen over PyPDF2/pypdf because it preserves layout,
handles multi-column resumes, and correctly extracts tables (skills grids).
"""

import io
import pdfplumber


def parse_resume_pdf(file_bytes: bytes) -> str:
    """
    Extract plain text from a PDF resume.
    Returns cleaned, newline-separated text ready to paste into the pipeline.
    """
    text_parts = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # extract_text preserves reading order better than naive bbox sorting
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                text_parts.append(page_text)

    raw = "\n\n".join(text_parts)

    # Light cleanup: collapse 3+ blank lines, strip trailing spaces per line
    lines = raw.splitlines()
    cleaned, blank_run = [], 0
    for line in lines:
        stripped = line.rstrip()
        if stripped == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(stripped)

    return "\n".join(cleaned).strip()
