"""Filtering searches by user-defined document attributes."""

import json

from search_helpers import POLICY, RECIPES, find


async def add(client, app, space, text, name, attributes):
    files = {"file": (name, __import__("io").BytesIO(text.encode()), "text/markdown")}
    r = await client.post(
        space["url"],
        files=files,
        data={"attributes": json.dumps(attributes)},
        headers=space["headers"],
    )
    assert r.status_code == 201, r.text
    while await app.state.worker.run_once():
        pass
    return r.json()["id"]


async def test_searching_by_attribute_values_and_ranges(client, app, space):
    old = await add(
        client, app, space, POLICY, "old.md", {"year": 2019, "author": "Chen", "tags": ["policy"]}
    )
    new = await add(
        client,
        app,
        space,
        RECIPES,
        "new.md",
        {"year": 2024, "author": "Li", "tags": ["food", "easy"]},
    )
    sp = {**space}

    async def docs(filters):
        out = await find(
            client, sp, "customer care spaghetti", mode="hybrid", top_k=50, filters=filters
        )
        return {h["document_id"] for h in out["hits"]}

    f = lambda field, op, value: [{"field": field, "op": op, "value": value}]  # noqa: E731
    assert await docs(f("attr.year", "gt", 2023)) == {new}
    assert await docs(f("attr.year", "lte", 2019)) == {old}
    assert await docs(
        [
            {"field": "attr.year", "op": "gte", "value": 2000},
            {"field": "attr.year", "op": "lt", "value": 2020},
        ]
    ) == {old}
    assert await docs(f("attr.author", "eq", "Li")) == {new}
    assert await docs(f("attr.author", "ne", "Li")) == {old}
    assert await docs(f("attr.tags", "eq", "easy")) == {new}
    assert await docs(f("attr.tags", "in", ["policy", "food"])) == {old, new}
    assert await docs(f("attr.year", "gt", 2030)) == set()
    assert await docs(f("attr.missing", "eq", "x")) == set()


async def test_editing_attributes_changes_what_a_search_finds_immediately(client, app, space):
    doc = await add(client, app, space, POLICY, "p.md", {"year": 2019})
    year = lambda op, v: [{"field": "attr.year", "op": op, "value": v}]  # noqa: E731
    assert (await find(client, space, "refunds", filters=year("gte", 2024)))["hits"] == []
    r = await client.patch(
        f"{space['url']}/{doc}", json={"attributes": {"year": 2025}}, headers=space["headers"]
    )
    assert r.status_code == 200
    out = await find(client, space, "refunds", filters=year("gte", 2024))
    assert out["hits"] and {h["document_id"] for h in out["hits"]} == {doc}


async def test_attributes_survive_reindexing(client, app, space):
    doc = await add(client, app, space, POLICY, "p.md", {"year": 2024})
    base = f"/v1/workspaces/{space['ws']}/collections/{space['col']}"
    r = await client.post(f"{base}/documents/{doc}/reindex", headers=space["headers"])
    assert r.status_code in (200, 202), r.text
    while await app.state.worker.run_once():
        pass
    out = await find(
        client, space, "refunds", filters=[{"field": "attr.year", "op": "eq", "value": 2024}]
    )
    assert out["hits"]
