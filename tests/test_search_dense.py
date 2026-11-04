"""Dense retrieval through the API: documents are ingested by the real worker pipeline into the
in-memory vector store, then searched."""

import io
import json
import uuid

import pytest
from sqlalchemy import select

from app.db.models import Chunk, Role, User, WorkspaceMember
from app.ingest.vectors import vector_collection
from app.providers.base import Point
from app.providers.errors import ProviderError
from app.providers.fake import FakeEmbedder, FakeLLM

SMALL = {"target_tokens": 64, "overlap_tokens": 8, "max_tokens": 128}
PASTA = "Boil salted water, cook the spaghetti for nine minutes and toss with garlic and olive oil"
REFUND = "Refunds are issued within 14 days of purchase to the original payment method"
SHIPPING = "Orders ship from the warehouse in Rotterdam within two business days"
FILLER = " ".join(f"Detail {i} is handled by the customer care team." for i in range(8))
POLICY = (
    "# Policies\n\nThis handbook lists the customer policies.\n\n"
    f"## Refunds\n\n{REFUND}. {FILLER}\n\n"
    f"## Shipping\n\n{SHIPPING}. {FILLER}\n\n"
    f"## Warranty\n\nHardware is covered by a two year limited warranty. {FILLER}"
)
RECIPES = (
    "# Recipes\n\nSimple dishes.\n\n"
    f"## Pasta\n\n{PASTA}. {FILLER}\n\n"
    f"## Soup\n\nSimmer onions, carrots and celery with vegetable stock for forty minutes. {FILLER}"
)


def search_url(space):
    return f"/v1/workspaces/{space['ws']}/search"


async def make_collection(client, space, name="docs", **body):
    r = await client.post(
        f"/v1/workspaces/{space['ws']}/collections",
        json={"name": name, "chunking_config": SMALL, **body},
        headers=space["headers"],
    )
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    return cid, f"/v1/workspaces/{space['ws']}/collections/{cid}/documents"


async def ingest(client, app, space, url, text=POLICY, name="policy.md"):
    files = {"file": (name, io.BytesIO(text.encode()), "text/markdown")}
    r = await client.post(url, files=files, headers=space["headers"])
    assert r.status_code == 201, r.text
    while await app.state.worker.run_once():
        pass
    return r.json()["id"]


async def find(client, space, query, **body):
    r = await client.post(
        search_url(space), json={"query": query, **body}, headers=space["headers"]
    )
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture
async def indexed(client, app, space):
    cid, url = await make_collection(client, space)
    doc_id = await ingest(client, app, space, url)
    return {**space, "cid": cid, "url": url, "doc": doc_id}


async def test_the_best_passage_comes_first_with_everything_needed_to_cite_it(client, indexed):
    out = await find(client, indexed, "Refunds are issued within 14 days of purchase")
    top = out["hits"][0]
    assert "Refunds are issued" in top["text"]
    assert top["rank"] == 1 and top["matched"] == "chunk" and top["question"] is None
    assert top["document_id"] == indexed["doc"] and top["collection_id"] == indexed["cid"]
    assert top["title"] == "policy" and top["heading_path"][0] == "Policies"
    assert top["char_end"] > top["char_start"] >= 0 and top["language"] == "en"
    assert top["scores"] == {"dense": top["score"]} and 0.5 < top["score"] <= 1.0001
    assert top["reasons"][0].startswith("dense cosine similarity")
    assert out["mode"] == "dense" and out["query"].startswith("Refunds")


async def test_results_are_ranked_by_descending_similarity(client, indexed):
    out = await find(client, indexed, "warranty on hardware", top_k=10)
    scores = [h["score"] for h in out["hits"]]
    assert scores == sorted(scores, reverse=True) and len(scores) > 1
    assert [h["rank"] for h in out["hits"]] == list(range(1, len(scores) + 1))
    assert "warranty" in out["hits"][0]["text"].lower()


async def test_top_k_limits_the_number_of_hits(client, indexed):
    assert len((await find(client, indexed, "shipping", top_k=1))["hits"]) == 1
    everything = await find(client, indexed, "shipping", top_k=50)
    assert len({h["chunk_id"] for h in everything["hits"]}) == len(everything["hits"])


async def test_min_score_drops_weak_matches(client, indexed):
    query = REFUND
    loose = await find(client, indexed, query, top_k=50)
    scores = [h["score"] for h in loose["hits"]]
    assert scores[0] > scores[-1]
    cutoff = scores[2]
    strict = await find(client, indexed, query, top_k=50, min_score=cutoff)
    assert len(strict["hits"]) == sum(1 for x in scores if x >= cutoff) < len(loose["hits"])
    assert all(h["score"] >= cutoff for h in strict["hits"])
    resolve = next(s for s in strict["trace"]["stages"] if s["name"] == "resolve")
    assert resolve["details"]["below_min_score"] == len(loose["hits"]) - len(strict["hits"])


async def test_the_trace_records_every_stage(client, indexed):
    out = await find(
        client, indexed, "shipping", filters=[{"field": "language", "op": "eq", "value": "EN"}]
    )
    stages = {s["name"]: s for s in out["trace"]["stages"]}
    assert list(stages) == ["scope", "embed_query", "dense_search", "resolve"]
    assert stages["scope"]["details"]["filters"] == ["language eq 'en'"]
    listed = {c["id"] for c in stages["scope"]["details"]["collections"]}
    assert indexed["cid"] in listed
    assert stages["dense_search"]["details"]["comparable_scores"] is True
    assert stages["resolve"]["details"]["returned"] == len(out["hits"])
    assert all(s["took_ms"] >= 0 for s in stages.values())


# ---- filters -------------------------------------------------------------------------------


async def test_filtering_by_document_searches_only_that_document(client, app, indexed):
    other = await ingest(client, app, indexed, indexed["url"], RECIPES, "recipes.md")
    both = await find(client, indexed, "cook", top_k=50)
    assert {h["document_id"] for h in both["hits"]} == {indexed["doc"], other}
    only = await find(
        client, indexed, "cook", top_k=50,
        filters=[{"field": "document_id", "op": "eq", "value": other}],
    )  # fmt: skip
    assert {h["document_id"] for h in only["hits"]} == {other}
    without = await find(
        client, indexed, "cook", top_k=50,
        filters=[{"field": "document_id", "op": "ne", "value": other}],
    )  # fmt: skip
    assert {h["document_id"] for h in without["hits"]} == {indexed["doc"]}


async def test_filtering_by_heading_and_title(client, app, indexed):
    await ingest(client, app, indexed, indexed["url"], RECIPES, "recipes.md")
    by_heading = await find(
        client, indexed, "water", top_k=50,
        filters=[{"field": "heading", "op": "eq", "value": "Soup"}],
    )  # fmt: skip
    assert by_heading["hits"] and all("Soup" in h["heading_path"] for h in by_heading["hits"])
    by_title = await find(
        client, indexed, "water", top_k=50,
        filters=[{"field": "title", "op": "in", "value": ["policy"]}],
    )  # fmt: skip
    assert {h["title"] for h in by_title["hits"]} == {"policy"}


async def test_filtering_by_source_file_type_and_language(client, indexed):
    def f(field, op, value):
        return {"field": field, "op": op, "value": value}

    every = [
        f("source_type", "eq", "file"),
        f("file_type", "eq", "markdown"),
        f("language", "in", ["en", "de"]),
    ]
    assert (await find(client, indexed, "refunds", filters=every))["hits"]
    for field, value in (("source_type", "url"), ("file_type", "pdf"), ("language", "de")):
        none = await find(client, indexed, "refunds", filters=[f(field, "eq", value)])
        assert none["hits"] == [], field
    not_md = await find(client, indexed, "refunds", filters=[f("file_type", "ne", "markdown")])
    assert not_md["hits"] == []


async def test_filtering_by_creation_date_treats_a_bare_date_as_a_whole_day(client, indexed):
    def f(op, value):
        return [{"field": "created_at", "op": op, "value": value}]

    assert (await find(client, indexed, "refunds", filters=f("gte", "2000-01-01")))["hits"]
    assert (await find(client, indexed, "refunds", filters=f("lte", "2999-12-31")))["hits"]
    assert (await find(client, indexed, "refunds", filters=f("lt", "2000-01-01")))["hits"] == []
    assert (await find(client, indexed, "refunds", filters=f("gt", "2999-12-31")))["hits"] == []
    # today is included by "up to today" because a bare date stands for the whole day
    r = await client.get(f"{indexed['url']}/{indexed['doc']}", headers=indexed["headers"])
    created_day = r.json()["created_at"][:10]
    assert (await find(client, indexed, "refunds", filters=f("lte", created_day)))["hits"]
    assert (await find(client, indexed, "refunds", filters=f("gt", created_day)))["hits"] == []


async def test_page_filters_skip_passages_without_a_page(client, indexed):
    # markdown has no pages, so a page condition can match nothing, but never errors
    out = await find(
        client, indexed, "refunds", filters=[{"field": "page", "op": "gte", "value": 1}]
    )
    assert out["hits"] == []


async def test_filters_cannot_escape_the_workspace_or_collection(client, indexed):
    for name in ("workspace_id", "collection_id"):
        r = await client.post(
            search_url(indexed),
            json={"query": "refunds", "filters": [{"field": name, "op": "eq", "value": "x"}]},
            headers=indexed["headers"],
        )
        assert r.status_code == 422


@pytest.mark.parametrize(
    "bad",
    [
        {"field": "language", "op": "gte", "value": "en"},
        {"field": "language", "op": "in", "value": "en"},
        {"field": "page", "op": "eq", "value": "three"},
        {"field": "created_at", "op": "gte", "value": "yesterday-ish"},
        {"field": "document_id", "op": "eq", "value": "nope"},
        {"field": "nonsense", "op": "eq", "value": "x"},
    ],
)
async def test_bad_filters_are_422(client, indexed, bad):
    r = await client.post(
        search_url(indexed), json={"query": "x", "filters": [bad]}, headers=indexed["headers"]
    )
    assert r.status_code == 422 and r.json()["errors"]


async def test_contradictory_filters_are_422_with_a_readable_message(client, indexed):
    r = await client.post(
        search_url(indexed),
        json={
            "query": "x",
            "filters": [
                {"field": "page", "op": "gte", "value": 5},
                {"field": "page", "op": "lt", "value": 2},
            ],
        },
        headers=indexed["headers"],
    )
    assert r.status_code == 422
    assert "leave nothing" in json.dumps(r.json()["errors"])


# ---- scoping and access ---------------------------------------------------------------------


async def test_a_search_covers_every_collection_unless_narrowed(client, app, indexed):
    cid2, url2 = await make_collection(client, indexed, "cooking")
    doc2 = await ingest(client, app, indexed, url2, RECIPES, "recipes.md")
    everywhere = await find(client, indexed, PASTA, top_k=50)
    assert {h["collection_id"] for h in everywhere["hits"]} == {indexed["cid"], cid2}
    assert everywhere["hits"][0]["document_id"] == doc2
    narrowed = await find(client, indexed, PASTA, top_k=50, collection_ids=[indexed["cid"]])
    assert {h["collection_id"] for h in narrowed["hits"]} == {indexed["cid"]}
    dup = await find(client, indexed, "cook", collection_ids=[cid2, cid2])
    assert len(dup["trace"]["stages"][0]["details"]["collections"]) == 1


async def test_another_workspaces_data_is_never_returned_or_addressable(
    client, app, indexed, signup
):
    outsider_headers = await signup("eve@example.com")
    ws = (
        await client.post("/v1/workspaces", json={"name": "Eve"}, headers=outsider_headers)
    ).json()
    eve = {"headers": outsider_headers, "ws": ws["id"]}
    cid, url = await make_collection(client, eve)
    await ingest(client, app, eve, url, "# Notes\n\nRefunds are secret in Eve's workspace.", "n.md")

    mine = await find(client, indexed, "Refunds are issued", top_k=50)
    assert {h["collection_id"] for h in mine["hits"]} == {indexed["cid"]}
    assert all("Eve" not in h["text"] for h in mine["hits"])
    theirs = await find(client, eve, "Refunds", top_k=50)
    assert {h["collection_id"] for h in theirs["hits"]} == {cid}

    # naming someone else's collection looks exactly like naming one that does not exist
    r = await client.post(
        search_url(indexed),
        json={"query": "x", "collection_ids": [cid]},
        headers=indexed["headers"],
    )
    assert r.status_code == 404
    r = await client.post(
        search_url(indexed),
        json={"query": "x", "collection_ids": [str(uuid.uuid4())]},
        headers=indexed["headers"],
    )
    assert r.status_code == 404
    # and a non-member cannot search the workspace at all
    r = await client.post(search_url(indexed), json={"query": "x"}, headers=outsider_headers)
    assert r.status_code == 404


async def test_viewers_can_search_but_anonymous_callers_cannot(client, app, indexed, signup):
    headers = await signup("vera@example.com")
    async with app.state.sessionmaker() as s:
        uid = (
            await s.execute(select(User.id).where(User.email == "vera@example.com"))
        ).scalar_one()
        s.add(WorkspaceMember(workspace_id=uuid.UUID(indexed["ws"]), user_id=uid, role=Role.VIEWER))
        await s.commit()
    r = await client.post(search_url(indexed), json={"query": "refunds"}, headers=headers)
    assert r.status_code == 200 and r.json()["hits"]
    assert (await client.post(search_url(indexed), json={"query": "refunds"})).status_code == 401


async def test_an_empty_workspace_or_unindexed_collection_returns_nothing(client, space):
    assert (await find(client, space, "anything"))["hits"] == []  # collection exists, no documents
    ws = (
        await client.post("/v1/workspaces", json={"name": "Bare"}, headers=space["headers"])
    ).json()
    bare = {"headers": space["headers"], "ws": ws["id"]}
    out = await find(client, bare, "anything")
    assert out["hits"] == [] and out["trace"]["stages"][0]["details"]["collections"] == []


# ---- correctness of what is served ------------------------------------------------------------


async def chunk_of(app, doc_id):
    async with app.state.sessionmaker() as s:
        rows = await s.execute(select(Chunk).where(Chunk.document_id == uuid.UUID(doc_id)))
        return rows.scalars().first()


async def test_points_from_a_superseded_index_version_are_not_served(client, app, indexed):
    chunk = await chunk_of(app, indexed["doc"])
    store = app.state.providers.vector_store
    name = vector_collection(uuid.UUID(indexed["cid"]))
    query = "zebra zebra zebra"
    (vec,) = await FakeEmbedder().embed([query])
    stale_id = str(uuid.uuid4())
    await store.upsert(
        name,
        [
            Point(
                stale_id,
                vec,  # a perfect match for the query, but left over from an older run
                {
                    "workspace_id": indexed["ws"], "collection_id": indexed["cid"],
                    "document_id": indexed["doc"], "chunk_id": str(chunk.id),
                    "kind": "question", "question": "?", "index_version": "an-older-run",
                },
            )
        ],
    )  # fmt: skip
    out = await find(client, indexed, query, top_k=50)
    resolve = next(s for s in out["trace"]["stages"] if s["name"] == "resolve")
    assert resolve["details"]["stale"] == 1
    # the stale point shares its chunk with a valid point, which must not be hidden by it
    mine = [h for h in out["hits"] if h["chunk_id"] == str(chunk.id)]
    assert len(mine) == 1 and mine[0]["matched"] == "chunk" and mine[0]["score"] < 0.99


async def test_points_whose_chunk_no_longer_exists_are_skipped(client, app, indexed):
    store = app.state.providers.vector_store
    name = vector_collection(uuid.UUID(indexed["cid"]))
    ghost = str(uuid.uuid4())
    (vec,) = await FakeEmbedder().embed(["Refunds are issued within 14 days of purchase"])
    await store.upsert(
        name,
        [
            Point(
                ghost,
                vec,
                {
                    "workspace_id": indexed["ws"], "collection_id": indexed["cid"],
                    "chunk_id": ghost, "kind": "chunk", "index_version": "x",
                },
            )
        ],
    )  # fmt: skip
    out = await find(client, indexed, "Refunds are issued within 14 days of purchase")
    assert ghost not in {h["chunk_id"] for h in out["hits"]}
    resolve = next(s for s in out["trace"]["stages"] if s["name"] == "resolve")
    assert resolve["details"]["missing"] == 1
    assert out["hits"] and "Refunds" in out["hits"][0]["text"]


async def test_questions_lead_to_their_chunk_and_never_duplicate_it(client, app, space):
    reply = json.dumps({"questions": ["How long do I have to request a refund?"]})
    app.state.providers._llm = FakeLLM(reply=reply)
    cid, url = await make_collection(
        client, space, "q", enrichment={"questions": True, "max_chunks": 10}
    )
    await ingest(client, app, space, url)
    out = await find(
        client, {**space, "cid": cid}, "How long do I have to request a refund?", top_k=50
    )
    top = out["hits"][0]
    assert top["matched"] == "question"
    assert top["question"] == "How long do I have to request a refund?"
    assert "question this passage answers" in top["reasons"][1]
    assert len({h["chunk_id"] for h in out["hits"]}) == len(out["hits"])
    resolve = next(s for s in out["trace"]["stages"] if s["name"] == "resolve")
    assert resolve["details"]["folded_into_chunk"] > 0


async def test_collections_with_different_models_are_flagged_as_not_comparable(client, app, space):
    cid1, url1 = await make_collection(client, space, "a")
    cid2, url2 = await make_collection(client, space, "b", embed_model="other-model")
    await ingest(client, app, space, url1)
    await ingest(client, app, space, url2, RECIPES, "recipes.md")
    out = await find(client, space, "boil the spaghetti", top_k=20)
    stages = {s["name"]: s for s in out["trace"]["stages"]}
    assert len(stages["embed_query"]["details"]["models"]) == 2
    assert stages["dense_search"]["details"]["comparable_scores"] is False
    assert {h["collection_id"] for h in out["hits"]} == {cid1, cid2}


async def test_an_embedding_failure_is_a_502_not_a_crash(client, indexed, app):
    class Broken(FakeEmbedder):
        async def embed(self, texts):
            raise ProviderError("fake", "embedder is down")

    app.state.providers._embedder = Broken(model=app.state.settings.resolved_embed_model)
    r = await client.post(search_url(indexed), json={"query": "x"}, headers=indexed["headers"])
    assert r.status_code == 502 and r.json()["provider"] == "fake"


@pytest.mark.parametrize(
    "body",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "x" * 2001},
        {"query": "x", "top_k": 0},
        {"query": "x", "top_k": 51},
        {"query": "x", "min_score": 2},
        {"query": "x", "mode": "psychic"},
        {"query": "x", "collection_ids": ["nope"]},
        {"query": "x", "filters": [{"field": "language", "op": "ne", "value": "en"}] * 21},
        {},
    ],
)
async def test_invalid_requests_are_422(client, indexed, body):
    r = await client.post(search_url(indexed), json=body, headers=indexed["headers"])
    assert r.status_code == 422


async def test_the_query_is_trimmed(client, indexed):
    out = await find(client, indexed, "   refunds   ")
    assert out["query"] == "refunds"


async def test_the_filterable_fields_are_listed(client, indexed):
    r = await client.get(search_url(indexed) + "/fields", headers=indexed["headers"])
    assert r.status_code == 200
    by_name = {f["name"]: f for f in r.json()}
    assert set(by_name) >= {"language", "page", "created_at", "document_id", "heading"}
    assert "gte" in by_name["page"]["ops"] and "eq" not in by_name["created_at"]["ops"]
