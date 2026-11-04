"""One-off builder for the *binary* golden inputs (DOCX, PPTX, PDF).

The generated files are committed: the golden tests parse those exact bytes, so a change in a
library's output can never silently change the inputs. Re-run only to add or alter a fixture:

    python tests/golden/build_inputs.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # tests/ (for pdfgen and the DOCX/PPTX builders)

import pdfgen  # noqa: E402
from test_parser_docx import build_docx  # noqa: E402
from test_parser_pptx import build_pptx  # noqa: E402


def main() -> None:
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)
    (inputs / "handbook.docx").write_bytes(build_docx())
    (inputs / "deck.pptx").write_bytes(build_pptx())
    (inputs / "report.pdf").write_bytes(pdfgen.report_pdf())
    print("wrote handbook.docx, deck.pptx, report.pdf")


if __name__ == "__main__":
    main()
