"""The parse stage end to end: upload through the API, run the worker, read the result back."""

import io

import pdfgen
import pytest
from test_parser_docx import build_docx
from test_parser_pptx import build_pptx

from app.db.models import DocumentStatus

CSV = b"name,plan,price\nAda,Pro,29\nLin,Free,0\n"
HTML = b"<title>FAQ</title><body><h1>Refunds</h1><p>Within 14 days.</p></body>"
MD = b"# Shipping\n\nWe ship worldwide.\n\n## Costs\n\nFree over 50 EUR."

CASES = {
    "txt": ("notes.txt", b"Plain notes about onboarding.", "text/plain", "Plain notes", 0, 0),
    "md": ("ship.md", MD, "text/markdown", "We ship worldwide.", 2, 0),
    "csv": ("plans.csv", CSV, "text/csv", "name: Ada | plan: Pro", 0, 0),
    "html": ("faq.html", HTML, "text/html", "Within 14 days.", 1, 0),
    "docx": ("handbook.docx", build_docx(), None, "Staff get 25 days", 3, 0),
    "pptx": ("deck.pptx", build_pptx(), None, "Mention the EMEA launch.", 2, 4),
    "pdf": ("report.pdf", pdfgen.report_pdf(), "application/pdf", "Regional breakdown", 5, 3),
}


def _upload(name, data, ctype):
    return {"file": (name, io.BytesIO(data), ctype or "application/octet-stream")}


@pytest.mark.parametrize("kind", list(CASES))
async def test_every_format_goes_from_upload_to_ready(client, app, space, kind):
    name, data, ctype, snippet, n_headings, n_pages = CASES[kind]
    h = space["headers"]
    doc = (await client.post(space["url"], files=_upload(name, data, ctype), headers=h)).json()

    assert await app.state.worker.run_once() is True

    final = (await client.get(f"{space['url']}/{doc['id']}", headers=h)).json()
    assert final["status"] == "ready"
    assert final["job"]["state"] == "done"
    assert final["job"]["progress"] == 100
    assert final["metadata"]["parse"]["characters"] > 0
    assert final["metadata"]["parse"]["headings"] == n_headings

    parsed = (await client.get(f"{space['url']}/{doc['id']}/parsed", headers=h)).json()
    assert snippet in parsed["text"]
    assert parsed["characters"] == len(parsed["text"])
    assert len(parsed["headings"]) == n_headings
    assert len(parsed["pages"]) == n_pages
    for heading in parsed["headings"]:
        assert 0 <= heading["char_start"] < len(parsed["text"])


async def test_parsed_can_be_requested_without_the_text(client, app, space):
    h = space["headers"]
    doc = (await client.post(space["url"], files=_upload(*CASES["md"][:3]), headers=h)).json()
    await app.state.worker.run_once()
    r = await client.get(f"{space['url']}/{doc['id']}/parsed?include_text=false", headers=h)
    assert r.json()["text"] is None
    assert r.json()["characters"] > 0
    assert len(r.json()["headings"]) == 2


async def test_parsed_is_404_before_the_worker_has_run(client, space):
    h = space["headers"]
    doc = (await client.post(space["url"], files=_upload(*CASES["txt"][:3]), headers=h)).json()
    r = await client.get(f"{space['url']}/{doc['id']}/parsed", headers=h)
    assert r.status_code == 404
    assert "not been parsed" in r.json()["detail"]


async def test_parsed_is_hidden_from_other_tenants(client, app, signup, space):
    h = space["headers"]
    doc = (await client.post(space["url"], files=_upload(*CASES["txt"][:3]), headers=h)).json()
    await app.state.worker.run_once()
    eve = await signup("eve@example.com")
    r = await client.get(f"{space['url']}/{doc['id']}/parsed", headers=eve)
    assert r.status_code == 404


async def test_unreadable_pdf_fails_the_job_with_a_clear_message(client, app, space):
    h = space["headers"]
    broken = b"%PDF-1.4\nthis is not really a pdf\n%%EOF"
    doc = (await client.post(space["url"], files=_upload("x.pdf", broken, None), headers=h)).json()
    await app.state.worker.run_once()
    final = (await client.get(f"{space['url']}/{doc['id']}", headers=h)).json()
    assert final["status"] == "failed"
    assert final["job"]["state"] == "failed"
    assert final["job"]["attempts"] == 1  # permanent: not retried
    assert "could not be read" in final["job"]["error"]
    assert (await client.get(f"{space['url']}/{doc['id']}/parsed", headers=h)).status_code == 404


async def test_pdf_with_no_text_explains_what_to_do(client, app, space):
    h = space["headers"]
    doc = (
        await client.post(
            space["url"], files=_upload("scan.pdf", pdfgen.scanned_pdf(), None), headers=h
        )
    ).json()
    await app.state.worker.run_once()
    final = (await client.get(f"{space['url']}/{doc['id']}", headers=h)).json()
    assert final["status"] == "failed"
    assert "OCR" in final["job"]["error"]


async def test_scanned_pdf_succeeds_when_an_ocr_engine_is_available(client, app, space):
    class FakeOcr:
        def recognize(self, image):
            return "Invoice 42 total 300 EUR"

    app.state.worker.ocr = FakeOcr()
    h = space["headers"]
    doc = (
        await client.post(
            space["url"], files=_upload("scan.pdf", pdfgen.scanned_pdf(), None), headers=h
        )
    ).json()
    await app.state.worker.run_once()
    final = (await client.get(f"{space['url']}/{doc['id']}", headers=h)).json()
    assert final["status"] == "ready"
    assert final["metadata"]["parse"]["ocr_pages"] == [1]
    assert final["metadata"]["parse"]["warnings"] == ["OCR was used for 1 page(s)"]
    parsed = (await client.get(f"{space['url']}/{doc['id']}/parsed", headers=h)).json()
    assert parsed["text"] == "Invoice 42 total 300 EUR"


async def test_a_missing_original_fails_cleanly(client, app, space):
    h = space["headers"]
    doc = (await client.post(space["url"], files=_upload(*CASES["txt"][:3]), headers=h)).json()
    folder = app.state.blobs.root / space["ws"] / space["col"] / doc["id"]
    for f in folder.iterdir():
        f.unlink()
    await app.state.worker.run_once()
    final = (await client.get(f"{space['url']}/{doc['id']}", headers=h)).json()
    assert final["status"] == "failed"
    assert "missing from storage" in final["job"]["error"]


async def test_reprocessing_overwrites_the_parsed_result(client, app, space):
    h = space["headers"]
    doc = (await client.post(space["url"], files=_upload(*CASES["md"][:3]), headers=h)).json()
    await app.state.worker.run_once()
    job_url = f"/v1/workspaces/{space['ws']}/jobs/{doc['job']['id']}"
    # Force the job to be re-run (as a re-index would)
    from sqlalchemy import update

    from app.db.models import IngestJob, JobState

    async with app.state.sessionmaker() as s:
        await s.execute(update(IngestJob).values(state=JobState.FAILED, error="x"))
        await s.commit()
    assert (await client.post(job_url + "/retry", headers=h)).status_code == 200
    await app.state.worker.run_once()
    final = (await client.get(f"{space['url']}/{doc['id']}", headers=h)).json()
    assert final["status"] == DocumentStatus.READY.value
    assert (await client.get(f"{space['url']}/{doc['id']}/parsed", headers=h)).status_code == 200
