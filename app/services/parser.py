import fitz  # PyMuPDF
import re
import arabic_reshaper
from bidi.algorithm import get_display
from typing import List, Dict, Any
from pathlib import Path


class PDFParser:
    """
    Custom PDF parser that handles:
    - Multi-column layouts
    - Arabic/RTL text (with reshaping + bidi)
    - Noise removal and cleaning
    """

    def __init__(self):
        self._url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self._whitespace_pattern = re.compile(r"\s{2,}")
        self._noise_pattern = re.compile(r"[^\w\s\u0600-\u06FF\u0750-\u077F.,;:()\-+@/]")

    def _is_arabic(self, text: str) -> bool:
        """Detect if text contains significant Arabic content."""
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        return arabic_chars / max(len(text), 1) > 0.2

    def _fix_arabic(self, text: str) -> str:
        """Reshape and apply bidi algorithm to Arabic text."""
        try:
            reshaped = arabic_reshaper.reshape(text)
            return get_display(reshaped)
        except Exception:
            return text  # fallback — return raw if reshaping fails

    def _clean_text(self, text: str) -> str:
        """Remove URLs and noise, but preserve newlines so the splitter can use section breaks."""
        text = self._url_pattern.sub(" ", text)
        # Collapse spaces/tabs on the same line, but keep newlines intact
        text = re.sub(r"[^\S\n]+", " ", text)
        # Normalise 3+ consecutive newlines down to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        return text

    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a single PDF file and return structured data.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dict with keys: filename, text, pages, language
        """
        path = Path(file_path)
        doc = fitz.open(file_path)

        full_text_blocks = []
        total_pages = len(doc)

        for page_num, page in enumerate(doc):
            # Extract text blocks sorted by reading order (top-to-bottom, left-to-right)
            blocks = page.get_text("blocks", sort=True)

            for block in blocks:
                # block = (x0, y0, x1, y1, text, block_no, block_type)
                if block[6] == 0:  # type 0 = text block (not image)
                    raw_text = block[4]
                    if raw_text.strip():
                        full_text_blocks.append(raw_text)

        raw_combined = "\n".join(full_text_blocks)

        # Handle Arabic content
        if self._is_arabic(raw_combined):
            raw_combined = self._fix_arabic(raw_combined)
            language = "arabic"
        else:
            language = "english"

        cleaned = self._clean_text(raw_combined)
        doc.close()

        return {
            "filename": path.name,
            "text": cleaned,
            "pages": total_pages,
            "language": language,
            "source": str(path),
        }

    def parse_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Parse all PDF files in a directory.

        Args:
            dir_path: Path to directory containing PDFs.

        Returns:
            List of parsed document dicts.
        """
        dir_path = Path(dir_path)
        pdf_files = list(dir_path.glob("*.pdf"))

        if not pdf_files:
            return []

        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.parse_pdf(str(pdf_file))
                documents.append(doc)
                print(f"[Parser] ✅ Parsed: {pdf_file.name} ({doc['language']})")
            except Exception as e:
                print(f"[Parser] ❌ Failed to parse {pdf_file.name}: {e}")

        return documents
