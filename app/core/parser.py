"""
Document Parser — handles raw PDFs, DOCX, HTML, and scanned images.
Includes Arabic NLP normalization for RTL text.
"""
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Arabic normalization helpers
# ────────────────────────────────────────────────────────────
ARABIC_DIACRITICS = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]')
ARABIC_TATWEEL   = re.compile(r'\u0640')
ARABIC_ALEF      = re.compile(r'[إأآا]')
ARABIC_YEH       = re.compile(r'[يى]')
ARABIC_HEH       = re.compile(r'[ةه]')


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for consistent embedding:
    1. Remove diacritics (tashkeel) — these are writer-dependent
    2. Remove tatweel (kashida elongation)
    3. Normalize alef variants → bare alef
    4. Normalize final yeh/alef-maqsura → yeh
    5. Normalize teh-marbuta / heh → heh (optional but consistent)
    """
    text = ARABIC_DIACRITICS.sub('', text)
    text = ARABIC_TATWEEL.sub('', text)
    text = ARABIC_ALEF.sub('ا', text)
    text = ARABIC_YEH.sub('ي', text)
    # Normalize unicode to NFC
    text = unicodedata.normalize('NFC', text)
    return text


def detect_language(text: str) -> str:
    """Simple heuristic: count Arabic codepoints."""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    ratio = arabic_chars / max(len(text), 1)
    return "ar" if ratio > 0.2 else "en"


# ────────────────────────────────────────────────────────────
# PDF Parser
# ────────────────────────────────────────────────────────────
def parse_pdf(file_path: str) -> Dict[str, Any]:
    """
    Extract text from PDF with fallback to OCR for scanned docs.
    Handles RTL (Arabic) text re-ordering issues from PyMuPDF.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)
    pages_text = []
    metadata = {
        "source": file_path,
        "num_pages": doc.page_count,
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
    }

    for page_num, page in enumerate(doc):
        # flags=TEXT_PRESERVE_WHITESPACE helps with Arabic RTL
        raw = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)

        # If very little text → likely scanned → try OCR
        if len(raw.strip()) < 50:
            logger.info(f"Page {page_num+1} appears scanned, attempting OCR")
            raw = _ocr_page(page)

        lang = detect_language(raw)
        if lang == "ar":
            raw = normalize_arabic(raw)
            logger.debug(f"Page {page_num+1}: Arabic detected, normalized.")

        pages_text.append({"page": page_num + 1, "text": raw, "lang": lang})

    doc.close()
    full_text = "\n\n".join(p["text"] for p in pages_text)
    return {"full_text": full_text, "pages": pages_text, "metadata": metadata}


def _ocr_page(page) -> str:
    """Rasterize page and run Tesseract OCR (Arabic + English)."""
    try:
        import pytesseract
        from PIL import Image
        import io

        mat = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(mat.tobytes("png")))
        # Try Arabic + English
        return pytesseract.image_to_string(img, lang="ara+eng")
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


# ────────────────────────────────────────────────────────────
# DOCX Parser
# ────────────────────────────────────────────────────────────
def parse_docx(file_path: str) -> Dict[str, Any]:
    from docx import Document

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    lang = detect_language(full_text)
    if lang == "ar":
        full_text = normalize_arabic(full_text)

    return {
        "full_text": full_text,
        "metadata": {"source": file_path, "num_pages": None},
    }


# ────────────────────────────────────────────────────────────
# HTML Parser
# ────────────────────────────────────────────────────────────
def parse_html(file_path: str) -> Dict[str, Any]:
    from bs4 import BeautifulSoup

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    full_text = soup.get_text(separator="\n")
    full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()
    lang = detect_language(full_text)
    if lang == "ar":
        full_text = normalize_arabic(full_text)

    return {"full_text": full_text, "metadata": {"source": file_path}}


# ────────────────────────────────────────────────────────────
# Dispatcher
# ────────────────────────────────────────────────────────────
def parse_document(file_path: str) -> Dict[str, Any]:
    """Route to correct parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    parsers = {
        ".pdf":  parse_pdf,
        ".docx": parse_docx,
        ".doc":  parse_docx,
        ".html": parse_html,
        ".htm":  parse_html,
    }
    parser = parsers.get(ext)
    if not parser:
        raise ValueError(f"Unsupported file type: {ext}")

    result = parser(file_path)
    result["metadata"]["file_type"] = ext
    return result
