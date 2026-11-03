"""Recognising what an uploaded file really is.

Type is decided from the file's bytes first (PDF and Office files have magic numbers) and only
then from the extension or the declared content type, so renaming `evil.exe` to `evil.pdf` is
rejected instead of being fed to a parser.
"""

from __future__ import annotations

import enum
import hashlib
import io
import re
import unicodedata
import zipfile
from pathlib import PurePosixPath, PureWindowsPath

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"


class Kind(enum.StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    HTML = "html"
    MARKDOWN = "markdown"
    CSV = "csv"
    TEXT = "text"


class UnsupportedFile(ValueError):
    """The file is not one of the supported formats (or its contents contradict its name)."""


_BY_EXTENSION = {
    "pdf": Kind.PDF,
    "docx": Kind.DOCX,
    "pptx": Kind.PPTX,
    "html": Kind.HTML,
    "htm": Kind.HTML,
    "xhtml": Kind.HTML,
    "md": Kind.MARKDOWN,
    "markdown": Kind.MARKDOWN,
    "csv": Kind.CSV,
    "tsv": Kind.CSV,
    "txt": Kind.TEXT,
    "text": Kind.TEXT,
    "log": Kind.TEXT,
    "rst": Kind.TEXT,
}
_BY_CONTENT_TYPE = {
    "application/pdf": Kind.PDF,
    "text/html": Kind.HTML,
    "application/xhtml+xml": Kind.HTML,
    "text/markdown": Kind.MARKDOWN,
    "text/x-markdown": Kind.MARKDOWN,
    "text/csv": Kind.CSV,
    "text/tab-separated-values": Kind.CSV,
    "text/plain": Kind.TEXT,
}
_BOMS = (b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")
_HTML_START = re.compile(rb"^\s*(<!doctype\s+html|<html|<head|<body)", re.IGNORECASE)

SUPPORTED_DESCRIPTION = "PDF, DOCX, PPTX, HTML, Markdown, CSV and plain text"


def extension_of(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()


def _office_kind(data: bytes) -> Kind:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        raise UnsupportedFile("The file looks like a ZIP archive but is corrupt") from None
    if "word/document.xml" in names:
        return Kind.DOCX
    if "ppt/presentation.xml" in names:
        return Kind.PPTX
    raise UnsupportedFile(f"Unsupported archive. Supported formats: {SUPPORTED_DESCRIPTION}")


def detect_kind(data: bytes, filename: str | None = None, content_type: str | None = None) -> Kind:
    if not data.strip():
        raise UnsupportedFile("The file is empty")
    ext = extension_of(filename)
    declared = _BY_EXTENSION.get(ext)

    head = data[:1024].lstrip()
    if head.startswith(PDF_MAGIC):
        detected = Kind.PDF
    elif data.startswith(ZIP_MAGIC):
        detected = _office_kind(data)
    else:
        detected = None

    if detected is not None:
        if declared in (Kind.PDF, Kind.DOCX, Kind.PPTX) and declared != detected:
            raise UnsupportedFile(f"The file contents are {detected.value}, not .{ext}")
        if declared is not None and declared not in (Kind.PDF, Kind.DOCX, Kind.PPTX):
            raise UnsupportedFile(f"The file contents are {detected.value}, not .{ext}")
        return detected

    if declared in (Kind.PDF, Kind.DOCX, Kind.PPTX):
        raise UnsupportedFile(f"The file is not a valid .{ext} document")
    if not data.startswith(_BOMS) and b"\x00" in data[:8192]:
        raise UnsupportedFile(f"Binary files are not supported. Supported: {SUPPORTED_DESCRIPTION}")
    if declared is not None:
        return declared
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime in _BY_CONTENT_TYPE:
        return _BY_CONTENT_TYPE[mime]
    if ext and ext not in _BY_EXTENSION and not mime.startswith("text/"):
        raise UnsupportedFile(f"Unsupported file type .{ext}. Supported: {SUPPORTED_DESCRIPTION}")
    if _HTML_START.match(data[:512]):
        return Kind.HTML
    return Kind.TEXT


def sanitize_filename(name: str | None, default: str = "document") -> str:
    """A display-safe filename: no directories, no control characters, bounded length."""
    if not name:
        return default
    base = PureWindowsPath(PurePosixPath(name).name).name  # strip both / and \ prefixes
    base = unicodedata.normalize("NFC", base)
    base = "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"|?*')
    base = base.strip(" .")
    if len(base) > 200:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: 200 - len(ext) - 1] + dot + ext) if dot and len(ext) < 10 else base[:200]
    return base or default


def title_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0] if "." in filename.strip(".") else filename
    return re.sub(r"[_\s]+", " ", stem).strip() or filename


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_text(data: bytes) -> tuple[str, str]:
    """Decode bytes to text, returning (text, encoding_name). Never raises."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace"), "utf-8"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace"), "utf-16"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), "cp1252"
