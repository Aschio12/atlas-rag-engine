"""Builds real PDFs with fpdf2 for the parser tests."""

from __future__ import annotations

import io

from fpdf import FPDF
from PIL import Image, ImageDraw

LOREM = (
    "Our company continued to grow steadily throughout the year, expanding into new markets "
    "and improving margins across every product line while keeping customer satisfaction high."
)


class _Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "", 8)
        self.set_y(6)
        self.cell(0, 5, "ACME Corp Confidential", align="C", new_x="LMARGIN", new_y="NEXT")

    def footer(self):
        self.set_font("Helvetica", "", 8)
        self.set_y(-12)
        self.cell(0, 5, f"Page {self.page_no()}", align="C", new_x="LMARGIN", new_y="NEXT")


def _heading(pdf: FPDF, text: str, size: int):
    pdf.set_font("Helvetica", "B", size)
    pdf.multi_cell(0, size * 0.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _para(pdf: FPDF, text: str):
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)


def report_pdf(title: str = "Annual Report 2025") -> bytes:
    pdf = _Report()
    pdf.set_title(title)
    pdf.set_author("ACME Finance")
    pdf.set_auto_page_break(auto=True, margin=18)

    pdf.add_page()
    _heading(pdf, "Annual Report", 22)
    _para(pdf, LOREM + " " + LOREM)
    _heading(pdf, "Financial results", 16)
    _para(pdf, "Revenue rose by twelve percent compared to the previous period. " + LOREM)
    pdf.set_font("Helvetica", "", 11)
    for item in ("- Record quarterly revenue", "- New offices opened", "- Headcount up 8 percent"):
        pdf.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    _heading(pdf, "Outlook", 13)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 5.5, "Demand for our inter-", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5.5, "national offerings remains strong.", new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    _heading(pdf, "Regional breakdown", 16)
    pdf.set_font("Helvetica", "", 11)
    with pdf.table(text_align="LEFT") as table:
        for row in (
            ("Region", "Revenue", "Growth"),
            ("EMEA", "$4.0M", "12%"),
            ("APAC", "$2.5M", "9%"),
        ):
            r = table.row()
            for cell in row:
                r.cell(cell)
    pdf.ln(6)
    _para(pdf, "The table above summarises regional performance. " + LOREM)

    pdf.add_page()
    _heading(pdf, "Two column section", 16)
    pdf.set_font("Helvetica", "", 10)
    left = ("LEFTCOLUMN " + LOREM + " ") * 3
    right = ("RIGHTCOLUMN " + LOREM + " ") * 3
    top = pdf.get_y()
    pdf.set_xy(10, top)
    pdf.multi_cell(88, 5, left, new_x="LEFT", new_y="NEXT")
    pdf.set_xy(112, top)
    pdf.multi_cell(88, 5, right, new_x="LEFT", new_y="NEXT")
    return bytes(pdf.output())


def scanned_pdf() -> bytes:
    img = Image.new("RGB", (800, 600), "white")
    ImageDraw.Draw(img).text((50, 50), "Scanned invoice number 42", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    pdf = FPDF()
    pdf.add_page()
    pdf.image(buf, x=10, y=10, w=190)
    return bytes(pdf.output())


def mixed_pdf() -> bytes:
    """Page 1 has text, page 2 is a scan, page 3 is blank."""
    img = Image.new("RGB", (400, 300), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Page one has a real text layer.")
    pdf.add_page()
    pdf.image(buf, x=10, y=10, w=100)
    pdf.add_page()
    return bytes(pdf.output())


def blank_pdf() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    return bytes(pdf.output())
