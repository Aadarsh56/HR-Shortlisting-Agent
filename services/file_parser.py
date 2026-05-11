"""
File parser service for extracting text from PDF and DOCX files.

Supports:
  • PDF  — Multi-strategy extraction: word-level (best for LaTeX/Overleaf),
            HTML-stripped, blocks-sorted, and plain-text fallback.
            The strategy producing the highest-quality result is used.
  • DOCX — python-docx walking body in document order, including all table
            cells (handles professional 2-column resume templates).
"""

import os
import re

import fitz  # PyMuPDF
from docx import Document


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _pdf_words_text(doc) -> str:
    """Word-level extraction — best for LaTeX/Overleaf multi-column PDFs.

    Groups words into lines by Y-coordinate proximity (4-pt buckets), then
    sorts lines top-to-bottom. Within each line, words are sorted
    left-to-right. Avoids the column-mixing problem of naive block extraction.
    """
    parts = []
    for page in doc:
        words = page.get_text("words")  # (x0,y0,x1,y1,word,block,line,word_no)
        if not words:
            continue
        # Group words whose y0 values are within 4pts of each other (same visual line)
        lines: dict[int, list] = {}
        for w in words:
            key = round(w[1] / 4)
            lines.setdefault(key, []).append(w)
        # Sort line groups by y position, then words left-to-right within each line
        for key in sorted(lines.keys()):
            line_words = sorted(lines[key], key=lambda w: w[0])
            line_text = " ".join(w[4] for w in line_words).strip()
            if line_text:
                parts.append(line_text)
    return "\n".join(parts).strip()


def _pdf_html_text(doc) -> str:
    """HTML-mode extraction — handles LaTeX font encoding and ligatures well.

    PyMuPDF's HTML renderer correctly resolves ToUnicode tables that confuse
    raw text extraction in LaTeX/Overleaf PDFs. Strips all HTML tags.
    """
    html_parts = []
    for page in doc:
        html = page.get_text("html")
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"[ \t]+", " ", clean)
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        if clean.strip():
            html_parts.append(clean.strip())
    return "\n".join(html_parts).strip()


def _pdf_blocks_text(doc) -> str:
    """Blocks-mode extraction sorted by reading order."""
    parts = []
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0]))
        for b in blocks:
            if b[6] == 0:
                line = b[4].strip()
                if line:
                    parts.append(line)
    return "\n".join(parts).strip()


def _pdf_plain_text(doc) -> str:
    """Plain text fallback."""
    return "\n".join(page.get_text("text") for page in doc).strip()


def _score_text(text: str) -> int:
    """Heuristic quality score for extracted text.

    Rewards longer output with common resume keywords.
    Penalises garbled output with a high ratio of 1-character tokens.
    """
    if not text or len(text) < 50:
        return 0
    score = len(text)
    signals = [
        "experience", "education", "skills", "project", "python",
        "engineer", "university", "email", "@", "github", "work",
        "summary", "certification", "degree",
    ]
    for s in signals:
        if s.lower() in text.lower():
            score += 500
    tokens = text.split()
    if tokens:
        short_ratio = sum(1 for t in tokens if len(t) <= 1) / len(tokens)
        score -= int(short_ratio * 5000)
    return score


def _extract_pdf_text(doc) -> str:
    """Try four extraction strategies; return whichever scores highest."""
    strategies = [
        _pdf_words_text,   # Best for LaTeX / Overleaf / multi-column
        _pdf_html_text,    # Good for ToUnicode / ligature issues
        _pdf_blocks_text,  # Good for simple single-column PDFs
        _pdf_plain_text,   # Last resort
    ]
    best_text, best_score = "", -1
    for fn in strategies:
        try:
            text = fn(doc)
            score = _score_text(text)
            if score > best_score:
                best_score = score
                best_text = text
        except Exception:
            continue
    return best_text


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        doc = fitz.open(file_path)
        text = _extract_pdf_text(doc)
        doc.close()
        return text
    except Exception as e:
        return f"[ERROR] Failed to parse PDF: {e}"


def parse_pdf_bytes(file_bytes: bytes, filename: str = "upload.pdf") -> str:
    """Extract text from PDF bytes (for Streamlit uploads)."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = _extract_pdf_text(doc)
        doc.close()
        return text
    except Exception as e:
        return f"[ERROR] Failed to parse PDF '{filename}': {e}"


# ── DOCX helpers ──────────────────────────────────────────────────────────────

def _extract_docx_text(doc) -> str:
    """Extract ALL text from a python-docx Document in document order.

    Handles:
      • Regular paragraphs
      • Tables (including merged cells — duplicates suppressed)
      • Text boxes / drawing shapes (w:txbxContent)
    """
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    from docx.table import Table

    parts = []

    def _para_text(element) -> str:
        try:
            return Paragraph(element, doc).text.strip()
        except Exception:
            return ""

    def _walk(parent):
        for element in parent:
            tag = element.tag

            if tag == qn("w:p"):
                text = _para_text(element)
                if text:
                    parts.append(text)

            elif tag == qn("w:tbl"):
                try:
                    table = Table(element, doc)
                    seen: set[str] = set()
                    for row in table.rows:
                        row_texts = []
                        for cell in row.cells:
                            ct = cell.text.strip()
                            if ct and ct not in seen:
                                seen.add(ct)
                                row_texts.append(ct)
                        if row_texts:
                            parts.append("  |  ".join(row_texts))
                except Exception:
                    pass

            # Text boxes and drawing shapes
            elif tag in (qn("w:drawing"), qn("w:pict")) or tag.endswith('AlternateContent'):
                for txbx in element.iter(qn("w:txbxContent")):
                    for para in txbx.iter(qn("w:p")):
                        text = _para_text(para)
                        if text:
                            parts.append(text)

    _walk(doc.element.body)
    return "\n".join(parts).strip()


def parse_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        return _extract_docx_text(doc)
    except Exception as e:
        return f"[ERROR] Failed to parse DOCX: {e}"


def parse_docx_bytes(file_bytes: bytes, filename: str = "upload.docx") -> str:
    """Extract text from DOCX bytes (for Streamlit uploads)."""
    import io
    try:
        doc = Document(io.BytesIO(file_bytes))
        return _extract_docx_text(doc)
    except Exception as e:
        return f"[ERROR] Failed to parse DOCX '{filename}': {e}"


# ── Auto-detect ────────────────────────────────────────────────────────────────

def detect_and_parse(file_path: str) -> str:
    """Auto-detect file type and extract text."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return parse_docx(file_path)
    else:
        return f"[ERROR] Unsupported file type: {ext}"


def detect_and_parse_bytes(file_bytes: bytes, filename: str) -> str:
    """Auto-detect file type from filename and extract text from bytes."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return parse_pdf_bytes(file_bytes, filename)
    elif ext in (".docx", ".doc"):
        return parse_docx_bytes(file_bytes, filename)
    else:
        return f"[ERROR] Unsupported file type: {ext}"
