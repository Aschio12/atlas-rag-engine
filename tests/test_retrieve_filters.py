import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.providers.base import Range
from app.retrieve.filters import (
    FIELDS,
    MAX_CONDITIONS,
    FilterError,
    MetadataFilter,
    compile_filters,
)


def cond(field, op, value):
    return MetadataFilter(field=field, op=op, value=value)


def compiled(*conditions):
    return compile_filters([cond(*c) for c in conditions])


def epoch(*args):
    return int(datetime(*args, tzinfo=UTC).timestamp())


def test_no_conditions_compile_to_an_empty_filter():
    out = compile_filters([])
    assert out.filters == {} and out.exclude == {} and not out


def test_equals_and_any_of_become_positive_conditions():
    out = compiled(("language", "eq", "EN"), ("source_type", "in", ["file", "WEB"]))
    assert out.filters == {"language": "en", "source_type": ["file", "web"]}
    assert out.exclude == {} and out


def test_not_equals_and_not_in_are_merged_into_one_exclusion_per_field():
    out = compiled(
        ("language", "ne", "de"), ("language", "nin", ["fr", "de"]), ("file_type", "ne", "pdf")
    )
    assert out.exclude == {"language": ["de", "fr"], "file_type": "pdf"}
    assert out.filters == {}


def test_text_keeps_its_case_but_codes_are_lowercased():
    out = compiled(("title", "eq", "Annual Report"), ("file_type", "eq", "PDF"))
    assert out.filters == {"title": "Annual Report", "file_type": "pdf"}


def test_heading_filters_the_array_field_in_the_payload():
    assert compiled(("heading", "eq", "Pricing")).filters == {"heading_path": "Pricing"}


def test_ids_are_normalised_and_validated():
    raw = uuid.uuid4()
    out = compiled(("document_id", "in", [str(raw).upper(), str(raw)]))
    assert out.filters == {"document_id": [str(raw), str(raw)]}
    with pytest.raises(ValidationError, match="not a valid id"):
        cond("document_id", "eq", "nope")


def test_page_ranges_merge_into_one_range():
    out = compiled(("page", "gte", 2), ("page", "lt", 9))
    assert out.filters == {"page": Range(gte=2, lt=9)}


def test_integers_accept_whole_floats_but_not_booleans_or_fractions():
    assert compiled(("page", "eq", 3.0)).filters == {"page": 3}
    for bad in (True, 2.5, "3"):
        with pytest.raises(ValidationError):
            cond("page", "eq", bad)


def test_dates_become_unix_seconds_utc():
    out = compiled(("created_at", "gte", "2023-06-01T12:30:00Z"))
    assert out.filters == {"created_at": Range(gte=epoch(2023, 6, 1, 12, 30))}
    naive = compiled(("created_at", "gte", "2023-06-01T12:30:00"))
    assert naive.filters == out.filters  # no zone means UTC
    offset = compiled(("created_at", "gte", "2023-06-01T14:30:00+02:00"))
    assert offset.filters == out.filters
    assert compiled(("created_at", "gte", 1_700_000_000)).filters["created_at"].gte == 1_700_000_000


def test_a_bare_date_means_the_whole_day_at_the_end_it_is_used_for():
    start = epoch(2023, 12, 31)
    last_second = epoch(2024, 1, 1) - 1
    # "from 2023-12-31" and "before 2023-12-31" use the first instant of the day ...
    assert compiled(("created_at", "gte", "2023-12-31")).filters["created_at"].gte == start
    assert compiled(("created_at", "lt", "2023-12-31")).filters["created_at"].lt == start
    # ... "up to" and "after" use its last second, so the day is wholly included or excluded
    assert compiled(("created_at", "lte", "2023-12-31")).filters["created_at"].lte == last_second
    assert compiled(("created_at", "gt", "2023-12-31")).filters["created_at"].gt == last_second


def test_a_full_datetime_is_never_stretched_to_the_end_of_the_day():
    out = compiled(("created_at", "lte", "2023-12-31T00:00:00"))
    assert out.filters["created_at"].lte == epoch(2023, 12, 31)


def test_bad_dates_are_rejected():
    for bad in ("last tuesday", "2023-13-01", True):
        with pytest.raises(ValidationError):
            cond("created_at", "gte", bad)


def test_dates_cannot_be_compared_by_equality():
    with pytest.raises(ValidationError, match="not supported"):
        cond("created_at", "eq", "2023-01-01")


def test_keyword_fields_have_no_ranges():
    with pytest.raises(ValidationError, match="not supported"):
        cond("language", "gte", "en")


def test_list_and_single_values_must_match_the_operator():
    with pytest.raises(ValidationError, match="list"):
        cond("language", "in", "en")
    with pytest.raises(ValidationError, match="single"):
        cond("language", "eq", ["en"])
    with pytest.raises(ValidationError, match="between 1 and"):
        cond("language", "in", [])
    with pytest.raises(ValidationError, match="between 1 and"):
        cond("language", "in", [str(i) for i in range(101)])


def test_only_known_fields_can_be_filtered_on():
    for name in ("workspace_id", "collection_id", "kind", "index_version", "chunk_id", "text"):
        with pytest.raises(ValidationError):
            cond(name, "eq", "x")
    assert set(FIELDS) == {
        "document_id", "source_id", "source_type", "file_type", "language", "title",
        "heading", "page", "created_at",
    }  # fmt: skip


def test_text_values_are_trimmed_and_bounded():
    assert compiled(("title", "eq", "  Report  ")).filters == {"title": "Report"}
    with pytest.raises(ValidationError):
        cond("title", "eq", "   ")
    with pytest.raises(ValidationError):
        cond("title", "eq", "x" * 501)
    with pytest.raises(ValidationError, match="text"):
        cond("language", "eq", 5)


@pytest.mark.parametrize(
    "conditions",
    [
        [("language", "eq", "en"), ("language", "in", ["en", "de"])],
        [("language", "eq", "en"), ("language", "eq", "de")],
        [("page", "eq", 3), ("page", "gte", 1)],
        [("page", "gte", 1), ("page", "gte", 2)],
        [("page", "gte", 5), ("page", "lt", 2)],
        [("created_at", "gte", "2024-01-01"), ("created_at", "lte", "2023-01-01")],
    ],
)
def test_redundant_or_contradictory_conditions_are_rejected(conditions):
    with pytest.raises(FilterError):
        compiled(*conditions)


def test_a_single_point_range_is_allowed():
    assert compiled(("page", "gte", 4), ("page", "lte", 4)).filters == {"page": Range(gte=4, lte=4)}


def test_too_many_conditions_are_rejected():
    many = [("language", "ne", f"x{i}") for i in range(MAX_CONDITIONS + 1)]
    with pytest.raises(FilterError, match="at most"):
        compiled(*many)


def test_the_summary_describes_each_condition_for_the_trace():
    out = compiled(("language", "in", ["en", "de"]), ("page", "lte", 10), ("title", "ne", "X"))
    assert out.summary == ("language in ['en', 'de']", "page lte 10", "title ne 'X'")


# ---- user-defined attributes -------------------------------------------------------------------


def test_attribute_conditions_target_the_nested_payload_key():
    out = compiled(
        ("attr.year", "gte", 2020), ("attr.year", "lt", 2025), ("attr.author", "eq", "Chen"),
        ("attr.tags", "in", ["rag", "eval"]), ("attr.draft", "eq", False),
    )  # fmt: skip
    assert out.filters == {
        "attributes.year": Range(gte=2020, lt=2025),
        "attributes.author": "Chen",  # text keeps its case
        "attributes.tags": ["rag", "eval"],
        "attributes.draft": False,
    }
    assert out.summary[0] == "attr.year gte 2020"


def test_attribute_exclusions_merge_like_any_other_field():
    out = compiled(("attr.author", "ne", "Chen"), ("attr.author", "nin", ["Li", "Chen"]))
    assert out.exclude == {"attributes.author": ["Chen", "Li"]}


def test_attribute_ranges_need_numbers_and_lists_need_whole_values():
    with pytest.raises(ValidationError, match="needs a number"):
        cond("attr.author", "gte", "Chen")
    with pytest.raises(ValidationError, match="needs a number"):
        cond("attr.year", "gt", True)
    with pytest.raises(ValidationError, match="text or whole numbers"):
        cond("attr.score", "in", [0.5])
    assert compiled(("attr.score", "gte", 0.5)).filters["attributes.score"] == Range(gte=0.5)


@pytest.mark.parametrize(
    "name", ["attr.Year", "attr.", "attr.a-b", "attr.1x", "attr." + "x" * 41, "attribute.year"]
)
def test_malformed_attribute_names_are_rejected(name):
    with pytest.raises(ValidationError, match="unknown filter field"):
        cond(name, "eq", "x")


def test_booleans_are_only_allowed_for_attributes():
    assert compiled(("attr.reviewed", "eq", True)).filters == {"attributes.reviewed": True}
    with pytest.raises(ValidationError):
        cond("language", "eq", True)


def test_attribute_values_are_trimmed_and_bounded():
    assert compiled(("attr.author", "eq", "  Chen ")).filters == {"attributes.author": "Chen"}
    with pytest.raises(ValidationError):
        cond("attr.author", "eq", "  ")
    with pytest.raises(ValidationError):
        cond("attr.author", "eq", "x" * 501)


def test_attribute_conditions_conflict_like_other_fields():
    with pytest.raises(FilterError):
        compiled(("attr.year", "eq", 2020), ("attr.year", "gte", 2019))
    with pytest.raises(FilterError):
        compiled(("attr.year", "gte", 2025), ("attr.year", "lt", 2020))
