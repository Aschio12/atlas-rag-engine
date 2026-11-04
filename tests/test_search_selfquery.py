"""Self-query inside the search pipeline: filters read out of the question."""

import io
import json
from datetime import date

import pytest
from search_helpers import POLICY, RECIPES, ScriptedLLM, find, search_url, use_llm


def reply(query=None, filters=()):
    data = {"filters": list(filters)}
    if query is not None:
        data["query"] = query
    return json.dumps(data)


def f(field, op, value):
    return {"field": field, "op": op, "value": value}


def stage(out, name):
    return next(s for s in out["trace"]["stages"] if s["name"] == name)


def stage_names(out):
    return [s["name"] for s in out["trace"]["stages"]]


async def add(client, app, space, text, name, attributes):
    files = {"file": (name, io.BytesIO(text.encode()), "text/markdown")}
    data = {"attributes": json.dumps(attributes)}
    r = await client.post(space["url"], files=files, data=data, headers=space["headers"])
    assert r.status_code == 201, r.text
    while await app.state.worker.run_once():
        pass
    return r.json()["id"]


@pytest.fixture
async def papers(client, app, space):
    old = await add(client, app, space, POLICY, "old.md", {"year": 2019, "author": "Chen"})
    new = await add(client, app, space, RECIPES, "new.md", {"year": 2024, "author": "Li"})
    return {**space, "old": old, "new": new}


QUESTION = "recipes after 2023 about spaghetti"


async def ask(client, papers, **kw):
    kw.setdefault("mode", "hybrid")
    kw.setdefault("top_k", 50)
    return await find(client, papers, QUESTION, self_query={"enabled": True}, **kw)


async def test_a_filter_read_from_the_question_narrows_the_search(client, app, papers):
    llm = use_llm(app, ScriptedLLM(self_query=reply("spaghetti", [f("attr.year", "gt", 2023)])))
    out = await ask(client, papers)
    assert {h["document_id"] for h in out["hits"]} == {papers["new"]}
    assert [(c["field"], c["op"], c["value"]) for c in out["inferred_filters"]] == [
        ("attr.year", "gt", 2023)
    ]
    details = stage(out, "self_query")["details"]
    assert details["original"] == QUESTION and details["query"] == "spaghetti"
    assert details["inferred"] == ["attr.year gt 2023"] and details["rejected"] == []
    assert out["query"] == QUESTION and out["warnings"] == []
    assert "attr.year (number" in llm.calls[0][0].content


async def test_the_cleaned_query_is_what_gets_searched(client, app, papers):
    use_llm(app, ScriptedLLM(self_query=reply("spaghetti", [f("attr.year", "gt", 2023)])))
    out = await ask(client, papers, mode="sparse")
    top = out["hits"][0]
    assert "matched terms: spaghetti" in top["reasons"][0]


async def test_the_whole_pipeline_runs_in_order(client, app, papers):
    use_llm(app, ScriptedLLM(self_query=reply("spaghetti", [f("attr.year", "gt", 2023)])))
    out = await ask(client, papers)
    assert stage_names(out) == [
        "scope", "self_query", "embed_query", "dense_search", "sparse_search", "resolve",
        "fusion", "result",
    ]  # fmt: skip


async def test_without_the_switch_the_model_is_never_asked(client, app, papers):
    llm = use_llm(app, ScriptedLLM(self_query=reply("x", [f("attr.year", "gt", 2023)])))
    out = await find(client, papers, QUESTION, mode="hybrid", top_k=50)
    assert llm.calls == [] and out["inferred_filters"] == []
    assert {h["document_id"] for h in out["hits"]} == {papers["old"], papers["new"]}


async def test_inferred_filters_that_match_nothing_fall_back_to_an_unfiltered_search(
    client, app, papers
):
    use_llm(app, ScriptedLLM(self_query=reply("spaghetti", [f("attr.year", "gt", 2030)])))
    out = await ask(client, papers)
    assert out["hits"] and out["inferred_filters"] == []
    assert "No passages matched the inferred filters" in out["warnings"][0]
    assert "attr.year gt 2030" in out["warnings"][0]
    names = stage_names(out)
    assert "self_query_fallback" in names
    assert names.count("resolve") == 2  # searched twice: filtered, then without the filters


async def test_the_fallback_can_be_switched_off(client, app, papers):
    use_llm(app, ScriptedLLM(self_query=reply("spaghetti", [f("attr.year", "gt", 2030)])))
    out = await find(
        client, papers, QUESTION, mode="hybrid", top_k=50,
        self_query={"enabled": True, "fallback": False},
    )  # fmt: skip
    assert out["hits"] == [] and out["warnings"] == []
    assert [c["field"] for c in out["inferred_filters"]] == ["attr.year"]


async def test_explicit_filters_win_over_inferred_ones(client, app, papers):
    use_llm(
        app,
        ScriptedLLM(
            self_query=reply(
                "spaghetti", [f("attr.year", "gt", 2023), f("attr.author", "eq", "Li")]
            )
        ),
    )
    explicit = [f("attr.year", "eq", 2019)]
    out = await find(
        client, papers, QUESTION, mode="hybrid", top_k=50, filters=explicit,
        self_query={"enabled": True, "fallback": False},
    )  # fmt: skip
    rejected = stage(out, "self_query")["details"]["rejected"]
    assert len(rejected) == 1 and rejected[0]["condition"]["field"] == "attr.year"
    assert "conflicts" in rejected[0]["reason"]
    # what is left (author Li) combined with the explicit year 2019 selects nothing
    assert [c["field"] for c in out["inferred_filters"]] == ["attr.author"] and out["hits"] == []
    # with the fallback, the inferred part is dropped and the explicit filter alone is applied
    again = await ask(client, papers, filters=explicit)
    assert again["inferred_filters"] == []
    assert {h["document_id"] for h in again["hits"]} == {papers["old"]}


async def test_values_that_no_document_has_are_not_used(client, app, papers):
    use_llm(
        app,
        ScriptedLLM(
            self_query=reply(
                "spaghetti", [f("attr.author", "eq", "Nobody"), f("attr.year", "gt", 2023)]
            )
        ),
    )
    out = await ask(client, papers)
    rejected = stage(out, "self_query")["details"]["rejected"]
    assert "no document has 'Nobody'" in rejected[0]["reason"]
    assert [c["field"] for c in out["inferred_filters"]] == ["attr.year"]


async def test_the_model_cannot_reach_beyond_the_workspace(client, app, papers):
    use_llm(
        app,
        ScriptedLLM(
            self_query=reply(
                "spaghetti",
                [f("workspace_id", "ne", papers["ws"]), f("collection_id", "ne", papers["col"])],
            )
        ),
    )
    out = await ask(client, papers)
    assert out["inferred_filters"] == [] and out["hits"]
    reasons = [r["reason"] for r in stage(out, "self_query")["details"]["rejected"]]
    assert len(reasons) == 2 and all("unknown filter field" in r for r in reasons)


async def test_a_broken_model_means_a_plain_search_with_a_warning(client, app, papers):
    use_llm(app, ScriptedLLM(self_query="I cannot do that"))
    out = await ask(client, papers)
    assert out["inferred_filters"] == [] and "Self-query failed" in out["warnings"][0]
    assert {h["document_id"] for h in out["hits"]} == {papers["old"], papers["new"]}


async def test_todays_date_reaches_the_model_so_relative_dates_can_be_resolved(
    client, app, papers, monkeypatch
):
    monkeypatch.setattr("app.retrieve.pipeline.today", lambda: date(2031, 5, 17))
    llm = use_llm(app, ScriptedLLM(self_query=reply()))
    await ask(client, papers)
    assert "Today is 2031-05-17" in llm.calls[0][0].content


async def test_a_rewritten_question_is_what_self_query_reads(client, app, papers):
    llm = use_llm(
        app,
        ScriptedLLM(
            rewrite=json.dumps({"query": "spaghetti recipes after 2023"}),
            self_query=reply("spaghetti recipes", [f("attr.year", "gt", 2023)]),
        ),
    )
    out = await find(
        client, papers, "and the newer ones?", mode="hybrid", top_k=50,
        history=[{"role": "user", "content": "Show me spaghetti recipes"}],
        transforms={"rewrite": True}, self_query={"enabled": True},
    )  # fmt: skip
    assert "<question>spaghetti recipes after 2023</question>" in llm.calls[1][0].content
    assert stage_names(out)[:3] == ["scope", "rewrite", "self_query"]
    assert {h["document_id"] for h in out["hits"]} == {papers["new"]}


async def test_expansions_use_the_cleaned_question(client, app, papers):
    llm = use_llm(
        app,
        ScriptedLLM(
            self_query=reply("spaghetti", [f("attr.year", "gt", 2023)]),
            step_back=json.dumps({"query": "What dishes are there?"}),
        ),
    )
    out = await ask(client, papers, transforms={"step_back": True})
    assert "<question>spaghetti</question>" in llm.calls[1][0].content
    assert stage(out, "transform")["details"]["primary"] == "spaghetti"


async def test_invalid_self_query_options_are_422(client, papers):
    for body in ({"enabled": "maybe"}, {"fallback": "sometimes"}):
        r = await client.post(
            search_url(papers), json={"query": "x", "self_query": body}, headers=papers["headers"]
        )
        assert r.status_code == 422


async def test_the_field_catalogue_lists_attributes_with_their_values(client, papers):
    r = await client.get(search_url(papers) + "/fields", headers=papers["headers"])
    assert r.status_code == 200
    by_name = {x["name"]: x for x in r.json()}
    year = by_name["attr.year"]
    assert (year["type"], year["documents"], year["minimum"], year["maximum"]) == (
        "number",
        2,
        2019,
        2024,
    )
    assert year["values"] == [2019, 2024] and year["complete"] is True and "gt" in year["ops"]
    assert by_name["attr.author"]["values"] == ["Chen", "Li"]
    assert by_name["language"]["values"] == ["en"] and by_name["file_type"]["values"] == [
        "markdown"
    ]
    assert by_name["page"]["type"] == "page" and "gte" in by_name["page"]["ops"]
    assert "eq" not in by_name["created_at"]["ops"]
