"""Optional OCR for scanned pages.

OCR is a plug-in so the core install stays light: parsers only need something with a
`recognize(image) -> str` method. The default engine uses Tesseract through `pytesseract`
(`pip install .[ocr]` plus the `tesseract` binary); when either is missing there is simply no
engine and scanned pages are reported with a warning instead of failing the whole document.
"""

from __future__ import annotations

import shutil
from typing import Any, Protocol


class OcrEngine(Protocol):
    def recognize(self, image: Any) -> str:
        """Return the text found in a PIL image."""


class TesseractOcr:
    def __init__(self, language: str = "eng") -> None:
        import pytesseract  # imported lazily: it is an optional dependency

        self._pytesseract = pytesseract
        self.language = language

    def recognize(self, image: Any) -> str:
        return self._pytesseract.image_to_string(image, lang=self.language)


def default_ocr(language: str = "eng") -> OcrEngine | None:
    """Tesseract if both the Python wrapper and the binary are installed, else None."""
    if shutil.which("tesseract") is None:
        return None
    try:
        return TesseractOcr(language)
    except ImportError:
        return None
