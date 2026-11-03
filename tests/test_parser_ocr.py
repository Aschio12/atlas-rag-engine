import pdfgen
import pytest

from app.ingest.files import Kind
from app.ingest.parsers import ParseError, default_ocr, parse
from app.ingest.parsers.ocr import TesseractOcr


class FakeOcr:
    def __init__(self, text="Scanned invoice number 42\n\nTotal due: 300 EUR"):
        self.text = text
        self.calls = 0
        self.sizes = []

    def recognize(self, image):
        self.calls += 1
        self.sizes.append(image.size)
        return self.text


def test_scanned_pdf_uses_ocr_text_and_records_it():
    ocr = FakeOcr()
    doc = parse(pdfgen.scanned_pdf(), Kind.PDF, ocr=ocr)
    assert doc.text == "Scanned invoice number 42\n\nTotal due: 300 EUR"
    assert doc.metadata["ocr_pages"] == [1]
    assert any("OCR was used for 1 page" in w for w in doc.warnings)
    assert ocr.calls == 1
    assert ocr.sizes[0][0] > 1000  # rendered at a resolution OCR can read


def test_scanned_pdf_without_an_engine_is_a_clear_error():
    with pytest.raises(ParseError, match="no extractable text"):
        parse(pdfgen.scanned_pdf(), Kind.PDF)


def test_mixed_document_ocrs_only_the_pages_that_need_it():
    ocr = FakeOcr("Text from the scan")
    doc = parse(pdfgen.mixed_pdf(), Kind.PDF, ocr=ocr)
    assert ocr.calls == 1  # the text page and the blank page are not OCRed
    assert doc.metadata["ocr_pages"] == [2]
    assert doc.text == "Page one has a real text layer.\n\nText from the scan"
    assert [p.page for p in doc.pages] == [1, 2, 3]
    assert doc.page_at(doc.text.index("Text from")) == 2


def test_mixed_document_without_an_engine_keeps_the_text_and_warns():
    doc = parse(pdfgen.mixed_pdf(), Kind.PDF)
    assert doc.text == "Page one has a real text layer."
    assert any("Page 2 has no text layer and OCR is not available" in w for w in doc.warnings)


def test_ocr_failure_degrades_to_a_warning():
    class Broken:
        def recognize(self, image):
            raise RuntimeError("tesseract crashed")

    doc = parse(pdfgen.mixed_pdf(), Kind.PDF, ocr=Broken())
    assert doc.text == "Page one has a real text layer."
    assert "OCR failed on page 2" in doc.warnings


def test_empty_ocr_result_is_treated_as_no_text():
    with pytest.raises(ParseError, match="no extractable text"):
        parse(pdfgen.scanned_pdf(), Kind.PDF, ocr=FakeOcr("   "))


def test_blank_pages_without_images_are_never_ocred():
    ocr = FakeOcr()
    with pytest.raises(ParseError):
        parse(pdfgen.blank_pdf(), Kind.PDF, ocr=ocr)
    assert ocr.calls == 0


def test_default_ocr_is_none_without_the_tesseract_binary(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert default_ocr() is None


def test_default_ocr_is_none_without_pytesseract(monkeypatch):
    import builtins

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/tesseract")
    real_import = builtins.__import__

    def no_pytesseract(name, *a, **k):
        if name == "pytesseract":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pytesseract)
    assert default_ocr() is None


def test_tesseract_adapter_passes_the_language_and_image(monkeypatch):
    import sys
    import types

    calls = []
    fake = types.SimpleNamespace(
        image_to_string=lambda img, lang: calls.append((img, lang)) or "ok"
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    engine = TesseractOcr("deu")
    assert engine.recognize("IMG") == "ok"
    assert calls == [("IMG", "deu")]
