import json
from datetime import date

import pytest
from search_helpers import ScriptedLLM

from app.providers.errors import ProviderError
from app.retrieve.catalogue import summarise
from app.retrieve.filters import MetadataFilter
from app.retrieve.selfquery import MAX_INFERRED, describe, infer_filters, offered_fields

TODAY = date(2026, 11, 4)

DOCS = [
    ({"year": 2019, "author": "Chen", "tags": ["rag"]}, {"kind": "pdf", "language": "en"}, "file"),
    (
        {"year": 2024, "author": "Li", "tags": ["eval", "rag"]},
        {"kind": "pdf", "language": "DE"},
        "file",
    ),
    ({"year": 2022}, {"kind": "markdown", "language": "en"}, "url"),
    ({}, {}, "text"),
]  # fmt: skip


def reply(query=None, filters=()):
    data = {"filters": list(filters)}
    if query is not None:
        data["query"] = query
    return json.dumps(data)


def f(field, op, value):
    return {"field": field, "op": op, "value": value}


CATALOGUE = summarise(DOCS)


QUESTION = "papers after 2023 about retrieval"


async def infer(question=QUESTION, explicit=(), raw=None, **replies):
    """Ask with `question`; the model answers `raw`, or a JSON reply built from `replies`
    (its `query` and `filters`)."""
    llm = ScriptedLLM(self_query=raw if raw is not None else reply(**replies))
    out = await infer_filters(llm, question, CATALOGUE, list(explicit), today=TODAY)
    return out, llm


def by_name(name):
    return next(i for i in CATALOGUE if i.name == name)


# ---- catalogue ------------------------------------------------------------------------------


def test_attributes_are_typed_and_summarised_from_the_documents():
    year = by_name("attr.year")
    assert (year.type, year.documents, year.minimum, year.maximum) == ("number", 3, 2019, 2024)
    assert year.values == [2019, 2022, 2024] and year.complete
    assert "gt" in year.ops and "gt" not in by_name("attr.author").ops
    author = by_name("attr.author")
    assert author.type == "text" and author.values == ["Chen", "Li"] and author.documents == 2
    tags = by_name("attr.tags")
    assert tags.type == "tags" and tags.values == ["eval", "rag"] and tags.documents == 2


def test_builtin_fields_list_the_values_that_exist():
    assert by_name("language").values == ["de", "en"] and by_name("language").complete
    assert by_name("file_type").values == ["markdown", "pdf"]
    assert by_name("source_type").values == ["file", "text", "url"]
    assert by_name("created_at").type == "date" and by_name("page").type == "page"
    assert by_name("title").values == [] and "eq" in by_name("title").ops


def test_an_attribute_the_documents_disagree_about_is_left_out():
    mixed = summarise([({"x": 1}, {}, "file"), ({"x": "one"}, {}, "file")])
    assert "attr.x" not in {i.name for i in mixed}


def test_long_value_lists_are_cut_and_marked_incomplete():
    many = summarise([({"author": f"person {i:02d}"}, {}, "file") for i in range(40)])
    author = next(i for i in many if i.name == "attr.author")
    assert len(author.values) == 25 and author.complete is False


def test_an_empty_workspace_still_lists_the_builtin_fields():
    names = {i.name for i in summarise([])}
    assert {"language", "page", "created_at", "document_id"} <= names
    assert not any(n.startswith("attr.") for n in names)


def test_the_model_sees_only_fields_it_can_use_and_their_values():
    offered = {i.name for i in offered_fields(CATALOGUE)}
    assert offered == {
        "source_type", "file_type", "language", "page", "created_at",
        "attr.year", "attr.author", "attr.tags",
    }  # fmt: skip
    line = describe(by_name("attr.year"))
    assert line.startswith("- attr.year (number; 3 documents; from 2019 to 2024; ops: eq")
    assert "all values: Chen, Li" in describe(by_name("attr.author"))


# ---- inferring filters ------------------------------------------------------------------------


async def test_a_clear_restriction_becomes_a_filter_and_a_cleaned_query():
    out, llm = await infer(query="retrieval papers", filters=[f("attr.year", "gt", 2023)])
    assert [(c.field, c.op, c.value) for c in out.conditions] == [("attr.year", "gt", 2023)]
    assert out.query == "retrieval papers" and out.rejected == [] and out.warnings == []
    prompt = llm.calls[0][0].content
    assert "Today is 2026-11-04" in prompt and "attr.year (number" in prompt
    assert f"<question>{QUESTION}</question>" in prompt
    assert "document_id" not in prompt and "workspace_id" not in prompt


async def test_without_a_filter_the_original_query_is_kept_even_if_the_model_rewrites_it():
    out, _ = await infer(query="a better phrased question", filters=[])
    assert out.query == QUESTION and out.conditions == []


async def test_a_useless_cleaned_query_is_ignored():
    for junk in ("", "ab", "...", None, 5):
        out, _ = await infer(
            raw=json.dumps({"query": junk, "filters": [f("attr.year", "gt", 2023)]})
        )
        assert out.query == "papers after 2023 about retrieval" and len(out.conditions) == 1


@pytest.mark.parametrize(
    ("condition", "reason"),
    [
        (
            f("attr.invented", "eq", "x"),
            "not offered",
        ),  # a valid name nobody has used  # valid name, but no such attribute
        (f("workspace_id", "eq", "x"), "unknown filter field"),
        (f("collection_id", "eq", "x"), "unknown filter field"),
        (f("document_id", "eq", "6f1c3a52-8f2e-4d0b-a3c4-5b7d9e0f1a2b"), "not offered"),
        (f("title", "eq", "Secret"), "not offered"),
        (f("attr.year", "like", 2023), "Input should be"),
        (f("attr.year", "gt", "recent"), "needs a number"),
        (f("language", "gte", "en"), "not supported"),
        (f("language", "eq", "fr"), "no document has 'fr'"),
        (f("attr.author", "eq", "Nobody"), "no document has 'Nobody'"),
        ({"field": "attr.year"}, "Field required"),
        ("attr.year > 2023", "Input should be"),
    ],
)
async def test_unusable_conditions_are_dropped_with_a_reason(condition, reason):
    out, _ = await infer(filters=[condition, f("attr.year", "gt", 2023)])
    assert [c.field for c in out.conditions] == ["attr.year"]
    (rejected,) = out.rejected
    assert reason in rejected["reason"]


async def test_values_are_checked_case_insensitively_and_exclusions_are_not_blocked():
    out, _ = await infer(filters=[f("language", "eq", "DE"), f("attr.author", "ne", "Nobody")])
    assert [c.field for c in out.conditions] == ["language", "attr.author"]


async def test_explicit_filters_win_over_inferred_ones():
    explicit = [MetadataFilter(field="attr.year", op="eq", value=2019)]
    out, _ = await infer(
        explicit=explicit, filters=[f("attr.year", "gt", 2023), f("language", "eq", "en")]
    )
    assert [c.field for c in out.conditions] == ["language"]
    assert "conflicts" in out.rejected[0]["reason"]


async def test_inferred_filters_conflicting_with_each_other_are_dropped():
    out, _ = await infer(filters=[f("attr.year", "gte", 2023), f("attr.year", "lt", 2000)])
    assert len(out.conditions) == 1 and "conflicts" in out.rejected[0]["reason"]


async def test_the_number_of_inferred_filters_is_capped():
    many = [f("attr.author", "ne", f"x{i}") for i in range(MAX_INFERRED + 3)]
    out, _ = await infer(filters=[f("attr.year", "gt", 1), *many])
    assert len(out.conditions) == MAX_INFERRED
    assert any("too many" in r["reason"] for r in out.rejected)


@pytest.mark.parametrize("bad", ["no json at all", "[1, 2]", '{"filters": "oops"}', "{broken"])
async def test_unusable_replies_leave_the_query_alone_with_a_warning_or_nothing(bad):
    out, _ = await infer(raw=bad)
    assert out.query == QUESTION and out.conditions == []


async def test_a_failing_model_degrades_to_no_inferred_filters():
    llm = ScriptedLLM(self_query=ProviderError("llm", "HTTP 500"))
    out = await infer_filters(llm, "x question", CATALOGUE, [], today=TODAY)
    assert (
        out.conditions == [] and "Self-query failed" in out.warnings[0] and "500" in out.warnings[0]
    )


async def test_a_workspace_with_nothing_to_filter_on_skips_the_model():
    llm = ScriptedLLM(self_query="never used")
    out = await infer_filters(llm, "question here", [], [], today=TODAY)
    assert llm.calls == [] and out.conditions == []
