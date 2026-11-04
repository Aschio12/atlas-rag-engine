import io

import pytest
from sqlalchemy import func, select

from app.db.models import Chunk

MD = (
    "# Handbook\n\n"
    + "\n\n".join(
        f"Paragraph {i} explains the leave policy for team {i} in some detail." for i in range(40)
    )
    + "\n\n## Expenses\n\n"
    + "\n\n".join(f"Expense rule {i} covers travel category {i} and its limits." for i in range(40))
)


async def collection_with(client, space, **body):
    r = await client.post(
        f"/v1/workspaces/{space['ws']}/collections",
        json={"name": f"c-{len(str(body))}-{abs(hash(str(body))) % 9999}", **body},
        headers=space["headers"],
    )
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    return cid, f"/v1/workspaces/{space['ws']}/collections/{cid}/documents"


async def ingest(client, app, space, url, text=MD, name="handbook.md"):
    files = {"file": (name, io.BytesIO(text.encode()), "text/markdown")}
    doc = (await client.post(url, files=files, headers=space["headers"])).json()
    assert await app.state.worker.run_once() is True
    final = await get(client, space, f"{url}/{doc['id']}")
    return doc["id"], final


SMALL = {"target_tokens": 64, "overlap_tokens": 8, "max_tokens": 128}


async def get(client, space, url):
    r = await client.get(url, headers=space["headers"])
    assert r.status_code == 200, r.text
    return r.json()


async def test_ingestion_stores_chunks_with_offsets_pages_and_breadcrumbs(client, app, space):
    _, url = await collection_with(client, space, chunking_config=SMALL)
    doc_id, final = await ingest(client, app, space, url)
    assert final["status"] == "ready" and final["job"]["state"] == "done"

    r = await client.get(f"{url}/{doc_id}/chunks?limit=200", headers=space["headers"])
    body = r.json()
    assert body["total"] > 10
    parsed = await get(client, space, f"{url}/{doc_id}/parsed")
    assert [c["ordinal"] for c in body["items"]] == list(range(body["total"]))
    for c in body["items"]:
        assert parsed["text"][c["char_start"] : c["char_end"]] == c["text"]  # exact offsets
        assert c["kind"] == "chunk" and c["parent_id"] is None
    assert body["items"][0]["heading_path"] == ["Handbook"]
    assert body["items"][-1]["heading_path"] == ["Handbook", "Expenses"]


async def test_the_document_records_a_chunking_summary(client, app, space):
    _, url = await collection_with(client, space, chunking_config=SMALL)
    _, final = await ingest(client, app, space, url)
    summary = final["metadata"]["chunking"]
    assert summary["strategy"] == "recursive"
    assert summary["chunks"] > 10 and summary["parents"] == 0
    assert 0 < summary["avg_tokens"] <= summary["max_tokens"] <= 128


async def test_chunk_pagination_and_listing_requires_access(client, app, signup, space):
    _, url = await collection_with(client, space, chunking_config=SMALL)
    doc_id, _ = await ingest(client, app, space, url)
    page = (
        await client.get(f"{url}/{doc_id}/chunks?limit=3&offset=2", headers=space["headers"])
    ).json()
    assert [c["ordinal"] for c in page["items"]] == [2, 3, 4]
    assert (
        await client.get(f"{url}/{doc_id}/chunks?limit=0", headers=space["headers"])
    ).status_code == 422
    assert (await client.get(f"{url}/{doc_id}/chunks")).status_code == 401
    eve = await signup("eve@example.com")
    assert (await client.get(f"{url}/{doc_id}/chunks", headers=eve)).status_code == 404


async def test_parent_child_collection_stores_a_linked_hierarchy(client, app, space):
    cfg = {**SMALL, "parent_tokens": 256, "child_tokens": 64}
    _, url = await collection_with(
        client, space, chunking_strategy="parent_child", chunking_config=cfg
    )
    doc_id, final = await ingest(client, app, space, url)
    summary = final["metadata"]["chunking"]
    assert summary["parents"] >= 3 and summary["chunks"] > summary["parents"]

    everything = (
        await client.get(f"{url}/{doc_id}/chunks?limit=200", headers=space["headers"])
    ).json()
    by_id = {c["id"]: c for c in everything["items"]}
    children = [c for c in everything["items"] if c["kind"] == "child"]
    assert children and all(by_id[c["parent_id"]]["kind"] == "parent" for c in children)
    for c in children:
        parent = by_id[c["parent_id"]]
        assert parent["char_start"] <= c["char_start"] and c["char_end"] <= parent["char_end"]

    only_children = (
        await client.get(f"{url}/{doc_id}/chunks?kind=child&limit=200", headers=space["headers"])
    ).json()
    assert only_children["total"] == summary["chunks"]
    assert all(c["kind"] == "child" for c in only_children["items"])
    only_parents = (
        await client.get(f"{url}/{doc_id}/chunks?kind=parent", headers=space["headers"])
    ).json()
    assert only_parents["total"] == summary["parents"]


async def test_contextual_collection_stores_context_and_usage(client, app, space):
    app.state.providers._llm = __import__("app.providers.fake", fromlist=["FakeLLM"]).FakeLLM(
        reply="From the company handbook."
    )
    _, url = await collection_with(
        client,
        space,
        chunking_strategy="contextual",
        chunking_config={**SMALL, "context_max_chunks": 5},
    )
    doc_id, final = await ingest(client, app, space, url)
    ctx = final["metadata"]["chunking"]["context"]
    assert ctx["calls"] == 5 and ctx["skipped"] == final["metadata"]["chunking"]["chunks"] - 5
    items = (await get(client, space, f"{url}/{doc_id}/chunks?limit=200"))["items"]
    assert [c["context"] for c in items[:5]] == ["From the company handbook."] * 5
    assert all(c["context"] is None for c in items[5:])


async def test_semantic_collection_uses_the_embedder(client, app, space):
    _, url = await collection_with(
        client, space, chunking_strategy="semantic", chunking_config=SMALL
    )
    _, final = await ingest(client, app, space, url)
    assert final["status"] == "ready"
    assert final["metadata"]["chunking"]["strategy"] == "semantic"


async def test_a_provider_outage_is_retried_with_a_clear_message(client, app, space):
    from app.providers.errors import ProviderError

    class Down:
        async def embed(self, texts):
            raise ProviderError("ollama", "connection refused: secret-host:11434")

    app.state.providers._embedder = Down()
    _, url = await collection_with(
        client, space, chunking_strategy="semantic", chunking_config=SMALL
    )
    doc_id, final = await ingest(client, app, space, url)
    assert final["job"]["state"] == "queued"  # will retry
    assert final["job"]["error"] == "The 'ollama' provider failed; will retry"
    assert "secret-host" not in str(final)


async def test_rechunking_replaces_the_old_chunks(client, app, space):
    _, url = await collection_with(client, space, chunking_config=SMALL)
    doc_id, _ = await ingest(client, app, space, url)
    job_id = (await get(client, space, f"{url}/{doc_id}"))["job"]["id"]
    async with app.state.sessionmaker() as s:
        before = (await s.execute(select(func.count()).select_from(Chunk))).scalar_one()
        from sqlalchemy import update

        from app.db.models import IngestJob, JobState

        await s.execute(update(IngestJob).values(state=JobState.FAILED))
        await s.commit()
    await client.post(f"/v1/workspaces/{space['ws']}/jobs/{job_id}/retry", headers=space["headers"])
    await app.state.worker.run_once()
    async with app.state.sessionmaker() as s:
        after = (await s.execute(select(func.count()).select_from(Chunk))).scalar_one()
    assert after == before  # replaced, not duplicated


async def test_deleting_a_document_deletes_its_chunks(client, app, space):
    _, url = await collection_with(client, space, chunking_config=SMALL)
    doc_id, _ = await ingest(client, app, space, url)
    await client.delete(f"{url}/{doc_id}", headers=space["headers"])
    async with app.state.sessionmaker() as s:
        assert (await s.execute(select(func.count()).select_from(Chunk))).scalar_one() == 0


# --- preview --------------------------------------------------------------------------------


def preview_url(space):
    return f"/v1/workspaces/{space['ws']}/chunking/preview"


async def test_preview_returns_chunks_and_stats_without_storing_anything(client, app, space):
    r = await client.post(
        preview_url(space),
        json={"text": MD, "strategy": "recursive", "config": SMALL},
        headers=space["headers"],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stats"]["chunks"] == len(body["items"]) > 10
    assert body["stats"]["max_tokens"] <= 128
    assert body["items"][0]["id"] is None and body["items"][0]["heading_path"] == ["Handbook"]
    async with app.state.sessionmaker() as s:
        assert (await s.execute(select(func.count()).select_from(Chunk))).scalar_one() == 0


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic", "parent_child"])
async def test_preview_supports_every_non_llm_strategy(client, space, strategy):
    cfg = {**SMALL, "parent_tokens": 256, "child_tokens": 64}
    r = await client.post(
        preview_url(space),
        json={"text": MD, "strategy": strategy, "config": cfg},
        headers=space["headers"],
    )
    assert r.status_code == 200, r.text
    assert r.json()["stats"]["strategy"] == strategy
    if strategy == "parent_child":
        assert r.json()["stats"]["parents"] > 0


async def test_preview_refuses_the_llm_strategy(client, space):
    r = await client.post(
        preview_url(space), json={"text": MD, "strategy": "contextual"}, headers=space["headers"]
    )
    assert r.status_code == 400
    assert "calls the LLM" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "x" * 200_001},
        {"text": "hi", "strategy": "nope"},
        {"text": "hi", "config": {"target_tokens": 1}},
        {"text": "hi", "format": "pdf"},
    ],
)
async def test_preview_validates_input(client, space, payload):
    r = await client.post(preview_url(space), json=payload, headers=space["headers"])
    assert r.status_code == 422


async def test_preview_of_unreadable_text_is_a_400_and_needs_auth(client, space):
    r = await client.post(preview_url(space), json={"text": " \x00\x07 "}, headers=space["headers"])
    assert r.status_code == 400
    assert (await client.post(preview_url(space), json={"text": "hi"})).status_code == 401


# --- enrichment (2.5) -----------------------------------------------------------------------


async def test_default_enrichment_adds_language_and_keywords_for_free(client, app, space):
    _, url = await collection_with(client, space, chunking_config=SMALL)
    doc_id, final = await ingest(client, app, space, url)
    e = final["metadata"]["enrichment"]
    assert e["language"] == "en" and final["metadata"]["language"] == "en"
    assert (e["summaries"], e["questions"]) == (0, 0) and "llm" not in e
    items = (await get(client, space, f"{url}/{doc_id}/chunks?limit=200"))["items"]
    assert all(c["language"] == "en" for c in items)
    assert any("leave policy" in " ".join(c["keywords"]) for c in items)
    assert all(c["summary"] is None and c["questions"] == [] for c in items)


async def test_llm_enrichment_adds_summaries_and_questions_with_a_cap(client, app, space):
    import json

    from app.providers.fake import FakeLLM

    reply = json.dumps(
        {"summary": "Explains leave.", "questions": ["What is the leave policy for my team?"]}
    )
    app.state.providers._llm = FakeLLM(reply=reply)
    cfg = {"summaries": True, "questions": True, "max_chunks": 4}
    _, url = await collection_with(client, space, chunking_config=SMALL, enrichment=cfg)
    doc_id, final = await ingest(client, app, space, url)
    e = final["metadata"]["enrichment"]
    assert e["summaries"] == 4 and e["questions"] == 4
    assert e["llm"]["calls"] == 4 and e["llm"]["skipped"] > 0
    items = (await get(client, space, f"{url}/{doc_id}/chunks?limit=200"))["items"]
    assert items[0]["summary"] == "Explains leave."
    assert items[0]["questions"] == ["What is the leave policy for my team?"]
    assert items[4]["summary"] is None  # beyond the cap


async def test_enrichment_failures_retry_with_a_clear_message(client, app, space):
    from app.providers.errors import ProviderError

    class Down:
        model = "x"

        async def complete(self, *a, **k):
            raise ProviderError("ollama", "connection refused")

    app.state.providers._llm = Down()
    cfg = {"summaries": True, "concurrency": 1}
    _, url = await collection_with(client, space, chunking_config=SMALL, enrichment=cfg)
    _, final = await ingest(client, app, space, url)
    assert final["job"]["state"] == "queued"
    assert final["job"]["error"] == "The 'ollama' provider failed; will retry"


async def test_parent_chunks_are_not_enriched(client, app, space):
    cfg = {**SMALL, "parent_tokens": 256, "child_tokens": 64}
    _, url = await collection_with(
        client, space, chunking_strategy="parent_child", chunking_config=cfg
    )
    doc_id, _ = await ingest(client, app, space, url)
    items = (await get(client, space, f"{url}/{doc_id}/chunks?limit=200"))["items"]
    parents = [c for c in items if c["kind"] == "parent"]
    children = [c for c in items if c["kind"] == "child"]
    assert parents and all(c["language"] is None and c["keywords"] == [] for c in parents)
    assert all(c["language"] == "en" for c in children)


async def test_rerunning_enrichment_removes_stale_fields(client, app, space):
    from sqlalchemy import update

    from app.db.models import Chunk, IngestJob, JobState

    _, url = await collection_with(client, space, chunking_config=SMALL)
    doc_id, _ = await ingest(client, app, space, url)
    async with app.state.sessionmaker() as s:  # pretend an older run left an LLM summary behind
        for c in (await s.execute(select(Chunk))).scalars():
            c.meta = {**c.meta, "summary": "stale", "questions": ["Stale question here?"]}
        await s.execute(update(IngestJob).values(state=JobState.FAILED))
        await s.commit()
    job_id = (await get(client, space, f"{url}/{doc_id}"))["job"]["id"]
    await client.post(f"/v1/workspaces/{space['ws']}/jobs/{job_id}/retry", headers=space["headers"])
    await app.state.worker.run_once()
    items = (await get(client, space, f"{url}/{doc_id}/chunks?limit=200"))["items"]
    assert all(c["summary"] is None and c["questions"] == [] for c in items)
