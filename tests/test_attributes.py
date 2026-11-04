import io
import json
import uuid

import pytest
from search_helpers import POLICY

from app.ingest.attributes import MAX_KEYS, validate_attributes
from app.ingest.vectors import vector_collection
from app.providers.errors import ProviderError

# ---- validation -------------------------------------------------------------------------------


def test_valid_attributes_are_normalised():
    out = validate_attributes(
        {
            "year": 2024, "score": 0.5, "author": "  Chen ", "reviewed": True,
            "tags": ["rag", " eval ", "rag"],
        }
    )  # fmt: skip
    assert out == {
        "year": 2024, "score": 0.5, "author": "Chen", "reviewed": True, "tags": ["rag", "eval"],
    }  # fmt: skip


def test_none_means_no_attributes():
    assert validate_attributes(None) == {}


@pytest.mark.parametrize(
    "bad",
    [
        [],
        "year",
        {"Year": 1},
        {"1year": 1},
        {"a-b": 1},
        {"x" * 41: 1},
        {"year": None},
        {"year": {"nested": 1}},
        {"year": float("inf")},
        {"year": float("nan")},
        {"author": "  "},
        {"author": "x" * 201},
        {"tags": []},
        {"tags": ["ok", 3]},
        {"tags": ["x" * 101]},
        {"tags": [f"t{i}" for i in range(21)]},
        {f"k{i}": i for i in range(MAX_KEYS + 1)},
    ],
)
def test_invalid_attributes_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_attributes(bad)


# ---- through the API --------------------------------------------------------------------------


def docs_url(space):
    return space["url"]


async def upload(client, space, attributes=None, text="# Note\n\nHello attributes world."):
    data = {"attributes": attributes} if attributes is not None else {}
    files = {"file": ("note.md", io.BytesIO(text.encode()), "text/markdown")}
    return await client.post(docs_url(space), files=files, data=data, headers=space["headers"])


async def test_attributes_can_be_set_on_upload_and_read_back(client, space):
    r = await upload(client, space, json.dumps({"year": 2024, "tags": ["a"]}))
    assert r.status_code == 201 and r.json()["attributes"] == {"year": 2024, "tags": ["a"]}
    got = await client.get(f"{docs_url(space)}/{r.json()['id']}", headers=space["headers"])
    assert got.json()["attributes"] == {"year": 2024, "tags": ["a"]}
    assert (await upload(client, space, None, "# Other\n\nNo attributes.")).json()[
        "attributes"
    ] == {}


@pytest.mark.parametrize("raw", ["not json", "[1]", '{"Bad": 1}', '{"x": null}'])
async def test_bad_upload_attributes_are_400(client, space, raw):
    r = await upload(client, space, raw)
    assert r.status_code == 400


async def test_attributes_can_be_set_on_text_documents(client, space):
    r = await client.post(
        docs_url(space) + "/text",
        json={"title": "Policy", "text": "Refunds in 14 days.", "attributes": {"year": 2023}},
        headers=space["headers"],
    )
    assert r.status_code == 201 and r.json()["attributes"] == {"year": 2023}
    bad = await client.post(
        docs_url(space) + "/text",
        json={"title": "Policy 2", "text": "x", "attributes": {"Year": 1}},
        headers=space["headers"],
    )
    assert bad.status_code == 422


async def test_attributes_reach_the_vector_payload(client, app, space):
    r = await upload(client, space, json.dumps({"year": 2024, "author": "Chen"}), POLICY)
    while await app.state.worker.run_once():
        pass
    store = app.state.providers.vector_store
    name = vector_collection(uuid.UUID(space["col"]))
    payloads = [p.payload for p in store._data[name].values()]
    assert payloads and all(p["attributes"] == {"year": 2024, "author": "Chen"} for p in payloads)
    assert r.json()["attributes"]["year"] == 2024


async def test_patching_updates_the_row_and_the_index_in_place(client, app, space):
    r = await upload(client, space, json.dumps({"year": 2020}), POLICY)
    doc_id = r.json()["id"]
    while await app.state.worker.run_once():
        pass
    store = app.state.providers.vector_store
    name = vector_collection(uuid.UUID(space["col"]))
    before = {p.id for p in store._data[name].values()}
    patched = await client.patch(
        f"{docs_url(space)}/{doc_id}",
        json={"title": "Renamed", "attributes": {"year": 2025, "tags": ["new"]}},
        headers=space["headers"],
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["title"] == "Renamed" and body["attributes"] == {"year": 2025, "tags": ["new"]}
    points = list(store._data[name].values())
    assert {p.id for p in points} == before  # nothing was re-embedded
    assert all(p.payload["title"] == "Renamed" for p in points)
    assert all(p.payload["attributes"] == {"year": 2025, "tags": ["new"]} for p in points)
    again = await client.get(f"{docs_url(space)}/{doc_id}", headers=space["headers"])
    assert again.json()["attributes"] == {"year": 2025, "tags": ["new"]}


async def test_patch_replaces_attributes_and_keeps_omitted_fields(client, space):
    doc_id = (await upload(client, space, json.dumps({"year": 1, "a": "b"}))).json()["id"]
    only_title = await client.patch(
        f"{docs_url(space)}/{doc_id}", json={"title": "T2"}, headers=space["headers"]
    )
    assert only_title.json()["attributes"] == {"year": 1, "a": "b"}
    cleared = await client.patch(
        f"{docs_url(space)}/{doc_id}", json={"attributes": {}}, headers=space["headers"]
    )
    assert cleared.json()["attributes"] == {} and cleared.json()["title"] == "T2"


async def test_a_patch_that_changes_nothing_does_not_touch_the_index(client, app, space):
    doc_id = (await upload(client, space, json.dumps({"year": 1}))).json()["id"]
    calls = []

    async def spy(*a, **kw):
        calls.append(a)

    app.state.providers.vector_store.set_payload = spy
    r = await client.patch(
        f"{docs_url(space)}/{doc_id}",
        json={"attributes": {"year": 1}},
        headers=space["headers"],
    )
    assert r.status_code == 200 and calls == []


async def test_a_vector_store_failure_leaves_the_document_unchanged(client, app, space):
    doc_id = (await upload(client, space, json.dumps({"year": 1}))).json()["id"]

    async def down(*a, **kw):
        raise ProviderError("memory", "store is down")

    app.state.providers.vector_store.set_payload = down
    r = await client.patch(
        f"{docs_url(space)}/{doc_id}", json={"attributes": {"year": 2}}, headers=space["headers"]
    )
    assert r.status_code == 502
    got = await client.get(f"{docs_url(space)}/{doc_id}", headers=space["headers"])
    assert got.json()["attributes"] == {"year": 1}


@pytest.mark.parametrize(
    "body", [{"title": ""}, {"title": "   "}, {"attributes": {"Bad": 1}}, {"attributes": [1]}]
)
async def test_invalid_patches_are_422(client, space, body):
    doc_id = (await upload(client, space)).json()["id"]
    r = await client.patch(f"{docs_url(space)}/{doc_id}", json=body, headers=space["headers"])
    assert r.status_code == 422


async def test_patching_needs_member_access_and_the_right_collection(client, space, signup):
    doc_id = (await upload(client, space)).json()["id"]
    other = await signup("mallory@example.com")
    r = await client.patch(f"{docs_url(space)}/{doc_id}", json={"title": "x"}, headers=other)
    assert r.status_code == 404
    missing = await client.patch(
        f"{docs_url(space)}/{uuid.uuid4()}", json={"title": "x"}, headers=space["headers"]
    )
    assert missing.status_code == 404
