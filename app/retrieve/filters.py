"""The metadata filter language for retrieval (Phase 3.1).

A search request carries a list of conditions, all of which must hold:

    {"field": "language",   "op": "in",  "value": ["en", "de"]}
    {"field": "created_at", "op": "gte", "value": "2023-01-01"}
    {"field": "page",       "op": "lte", "value": 10}

Only the fields below can be filtered on. That is deliberate: a field is filterable when it is
stored (and indexed) in the vector payload, and the allow-list is what keeps a request, or an LLM
writing filters for it in Phase 3.7, from reaching `workspace_id`, `collection_id` or anything
else the server fixes itself. Workspace and collection scoping is never part of this language.

`compile_filters` turns the conditions into the store's `filters` and `exclude` mappings.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from app.providers.base import Filters, Range

Op = Literal["eq", "ne", "in", "nin", "gt", "gte", "lt", "lte"]
SET_OPS = ("eq", "in")  # positive conditions: the point must match
NOT_OPS = ("ne", "nin")  # negative conditions: the point must not match
RANGE_OPS = ("gt", "gte", "lt", "lte")

MAX_CONDITIONS = 20
MAX_VALUES = 100
MAX_TEXT = 500

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FilterError(ValueError):
    """The filter conditions are invalid or contradict each other."""


@dataclass(frozen=True)
class FieldSpec:
    key: str  # payload field in the vector store
    kind: Literal["keyword", "lower", "uuid", "integer", "datetime"]
    description: str

    @property
    def ops(self) -> tuple[str, ...]:
        if self.kind in ("keyword", "lower", "uuid"):
            return (*SET_OPS, *NOT_OPS)
        if self.kind == "integer":
            return (*SET_OPS, *NOT_OPS, *RANGE_OPS)
        return RANGE_OPS  # datetimes are compared by range only


FIELDS: dict[str, FieldSpec] = {
    "document_id": FieldSpec("document_id", "uuid", "Only these documents"),
    "source_id": FieldSpec("source_id", "uuid", "Documents synced from a source"),
    "source_type": FieldSpec("source_type", "lower", "file, url, text, github or web"),
    "file_type": FieldSpec("file_type", "lower", "pdf, docx, pptx, html, markdown, ..."),
    "language": FieldSpec("language", "lower", "Language code, e.g. en"),
    "title": FieldSpec("title", "keyword", "Exact document title"),
    "heading": FieldSpec("heading_path", "keyword", "A section heading the passage is under"),
    "page": FieldSpec("page", "integer", "Page number (PDF)"),
    "created_at": FieldSpec("created_at", "datetime", "When the document was added"),
}

FilterField = Literal[
    "document_id",
    "source_id",
    "source_type",
    "file_type",
    "language",
    "title",
    "heading",
    "page",
    "created_at",
]


class MetadataFilter(BaseModel):
    """One condition. `value` is a single value, or a non-empty list for `in` and `nin`."""

    field: FilterField
    op: Op = "eq"
    value: str | int | float | list[str | int | float]

    @field_validator("value", mode="before")
    @classmethod
    def _no_booleans(cls, value: Any) -> Any:
        # Left alone, pydantic would quietly turn true into 1.
        if isinstance(value, bool) or (
            isinstance(value, list) and any(isinstance(v, bool) for v in value)
        ):
            raise ValueError("booleans are not valid filter values")
        return value

    @model_validator(mode="after")
    def _check(self) -> MetadataFilter:
        spec = FIELDS[self.field]
        if self.op not in spec.ops:
            raise ValueError(
                f"'{self.op}' is not supported for '{self.field}' (use: {', '.join(spec.ops)})"
            )
        many = self.op in ("in", "nin")
        if many != isinstance(self.value, list):
            raise ValueError(
                f"'{self.op}' takes {'a list of values' if many else 'a single value'}"
            )
        values = self.value if isinstance(self.value, list) else [self.value]
        if many and not 0 < len(values) <= MAX_VALUES:
            raise ValueError(f"'{self.op}' takes between 1 and {MAX_VALUES} values")
        for v in values:
            _coerce(self.field, self.op, v)  # raises ValueError with a readable message
        return self


def _end_of_day(day: date) -> int:
    nxt = datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=1)
    return int(nxt.timestamp()) - 1


def _coerce_datetime(op: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError("a date must be an ISO 8601 string or Unix seconds")
    if not isinstance(value, str):
        return int(value)
    text = value.strip()
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{value!r} is not an ISO 8601 date or datetime") from None
    if _DATE_ONLY.match(text) and op in ("lte", "gt"):
        # "up to 2023-12-31" and "after 2023-12-31" mean the whole day, not its first second.
        return _end_of_day(moment.date())
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp())


def _coerce(name: str, op: str, value: Any) -> Any:
    """Validate one value for a field and convert it to what the payload stores."""
    kind = FIELDS[name].kind
    if kind == "datetime":
        return _coerce_datetime(op, value)
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise ValueError(f"'{name}' takes whole numbers")
        return value
    if not isinstance(value, str):
        raise ValueError(f"'{name}' takes text values")
    text = value.strip()
    if not text or len(text) > MAX_TEXT:
        raise ValueError(f"'{name}' values must be 1 to {MAX_TEXT} characters")
    if kind == "uuid":
        try:
            return str(uuid.UUID(text))
        except ValueError:
            raise ValueError(f"{value!r} is not a valid id") from None
    return text.lower() if kind == "lower" else text


@dataclass(frozen=True)
class CompiledFilter:
    filters: Filters = field(default_factory=dict)
    exclude: Filters = field(default_factory=dict)
    summary: tuple[str, ...] = ()  # human-readable, for the retrieval trace

    def __bool__(self) -> bool:
        return bool(self.filters or self.exclude)


def _one(values: list[Any]) -> Any:
    return values[0] if len(values) == 1 else values


def compile_filters(conditions: list[MetadataFilter]) -> CompiledFilter:
    """Combine conditions into store filters.

    Per field there can be at most one positive condition (`eq` or `in`) or one set of range
    bounds, not both, and each bound at most once: anything else is either redundant or
    contradictory and is rejected rather than guessed at. Negative conditions (`ne`, `nin`)
    on one field are merged into a single exclusion.
    """
    if len(conditions) > MAX_CONDITIONS:
        raise FilterError(f"at most {MAX_CONDITIONS} filter conditions are allowed")
    must: dict[str, Any] = {}
    bounds: dict[str, dict[str, Any]] = {}
    banned: dict[str, list[Any]] = {}
    summary: list[str] = []

    for c in conditions:
        spec = FIELDS[c.field]
        raw = c.value if isinstance(c.value, list) else [c.value]
        values = [_coerce(c.field, c.op, v) for v in raw]
        shown = values[0] if len(values) == 1 else values
        summary.append(f"{c.field} {c.op} {shown!r}")
        if c.op in SET_OPS:
            if spec.key in must or spec.key in bounds:
                raise FilterError(f"conflicting conditions on '{c.field}'; combine them with 'in'")
            must[spec.key] = _one(values)
        elif c.op in NOT_OPS:
            banned.setdefault(spec.key, []).extend(values)
        else:
            if spec.key in must or c.op in bounds.get(spec.key, {}):
                raise FilterError(f"conflicting conditions on '{c.field}'")
            bounds.setdefault(spec.key, {})[c.op] = values[0]

    for key, parts in bounds.items():
        low = parts.get("gte", parts.get("gt"))
        high = parts.get("lte", parts.get("lt"))
        if low is not None and high is not None and low > high:
            raise FilterError(f"the bounds on '{key}' leave nothing: {low} > {high}")
        must[key] = Range(**parts)
    exclude = {key: _one(list(dict.fromkeys(vals))) for key, vals in banned.items()}
    return CompiledFilter(must, exclude, tuple(summary))
