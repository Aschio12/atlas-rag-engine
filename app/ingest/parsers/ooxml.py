"""Safe access to Office Open XML packages (DOCX, PPTX).

They are ZIP files of XML, so both layers need guarding:
* ZIP bombs: limits on member count, total/uncompressed size and compression ratio.
* XML attacks (entity expansion, external entities): parsed with `defusedxml`, DTDs forbidden.
"""

from __future__ import annotations

import io
import posixpath
import zipfile
from xml.etree.ElementTree import Element

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import ParseError as XmlParseError
from defusedxml.ElementTree import fromstring

from app.ingest.parsers.base import ParseError

MAX_MEMBERS = 5_000
MAX_TOTAL_BYTES = 300 * 1024 * 1024
MAX_PART_BYTES = 100 * 1024 * 1024
MAX_RATIO = 200  # uncompressed / compressed, for parts larger than 1 MB

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
}


def q(tag: str) -> str:
    """'w:p' -> '{namespace}p' for ElementTree lookups."""
    prefix, local = tag.split(":")
    return f"{{{NS[prefix]}}}{local}"


def open_package(data: bytes, label: str) -> zipfile.ZipFile:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ParseError(f"The {label} file is corrupt (not a valid ZIP package)") from None
    infos = zf.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ParseError(f"The {label} file has too many parts")
    if sum(i.file_size for i in infos) > MAX_TOTAL_BYTES:
        raise ParseError(f"The {label} file expands to an unreasonable size")
    for i in infos:
        if i.file_size > 1024 * 1024 and i.file_size > MAX_RATIO * max(i.compress_size, 1):
            raise ParseError(f"The {label} file looks like a compression bomb")
        if i.flag_bits & 0x1:
            raise ParseError(f"The {label} file is password protected")
    return zf


def read_xml(zf: zipfile.ZipFile, name: str, *, required: bool = True) -> Element | None:
    try:
        info = zf.getinfo(name)
    except KeyError:
        if required:
            raise ParseError(f"The file is missing a required part ({name})") from None
        return None
    if info.file_size > MAX_PART_BYTES:
        raise ParseError(f"A part of the file is too large ({name})")
    try:
        with zf.open(info) as f:
            raw = f.read(MAX_PART_BYTES + 1)
        if len(raw) > MAX_PART_BYTES:
            raise ParseError(f"A part of the file is too large ({name})")
        return fromstring(raw, forbid_dtd=True)
    except DefusedXmlException:
        raise ParseError("The file contains disallowed XML (DTD or entities)") from None
    except (XmlParseError, zipfile.BadZipFile, RuntimeError, OSError):
        raise ParseError(f"The file is corrupt (cannot read {name})") from None


def resolve(base_dir: str, target: str) -> str:
    """Resolve a relationship target against the folder of the part that declares it."""
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(base_dir, target))


def relationships(zf: zipfile.ZipFile, part: str) -> dict[str, tuple[str, str]]:
    """Relationships of a part: {rId: (resolved target path, relationship type suffix)}."""
    folder, filename = posixpath.split(part)
    root = read_xml(zf, posixpath.join(folder, "_rels", filename + ".rels"), required=False)
    out: dict[str, tuple[str, str]] = {}
    if root is None:
        return out
    for rel in root.findall(q("rel:Relationship")):
        if rel.get("TargetMode") == "External":
            continue
        out[rel.get("Id", "")] = (
            resolve(folder, rel.get("Target", "")),
            (rel.get("Type") or "").rsplit("/", 1)[-1],
        )
    return out


def core_properties(zf: zipfile.ZipFile) -> dict[str, str]:
    """Title, author and dates from docProps/core.xml (all optional)."""
    root = read_xml(zf, "docProps/core.xml", required=False)
    if root is None:
        return {}
    fields = {
        "title": "dc:title",
        "author": "dc:creator",
        "subject": "dc:subject",
        "created": "dcterms:created",
        "modified": "dcterms:modified",
    }
    props = {}
    for key, tag in fields.items():
        el = root.find(q(tag))
        if el is not None and (el.text or "").strip():
            props[key] = el.text.strip()
    return props
