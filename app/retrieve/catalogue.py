"""What can be filtered on, and with which values (Phase 3.7).

The catalogue lists the built-in filter fields and the document attributes that actually exist
in a workspace, with the values seen. It serves two readers: people exploring the API
(`GET .../search/fields`) and the self-query step, which shows it to the model so that filters
are written against real field names and real values instead of guesses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Collection, Document
from app.retrieve.filters import ATTR_PREFIX, FIELDS, RANGE_OPS, SET_OPS, FieldSpec

MAX_SCANNED_DOCUMENTS = 5000
MAX_VALUES = 25  # distinct values listed per field


@dataclass
class FieldInfo:
    name: str  # as used in a filter: "language", "attr.year"
    type: str  # text | number | boolean | tags | date | id | page
    ops: list[str]
    description: str
    values: list[Any] = field(default_factory=list)  # distinct values seen, up to MAX_VALUES
    complete: bool = False  # `values` is every value there is
    minimum: float | None = None
    maximum: float | None = None
    documents: int | None = None  # documents that have it (attributes only)


_STATIC_TYPES = {
    "document_id": "id",
    "source_id": "id",
    "source_type": "text",
    "file_type": "text",
    "language": "text",
    "title": "text",
    "heading": "text",
    "page": "page",
    "created_at": "date",
}
# built-in fields whose values come from the documents themselves
_FROM_DOCUMENTS = ("source_type", "file_type", "language")


def _attr_type(values: list[Any]) -> str | None:
    """One type for an attribute, or None when documents disagree about it."""
    kinds = set()
    for v in values:
        if isinstance(v, bool):
            kinds.add("boolean")
        elif isinstance(v, int | float):
            kinds.add("number")
        elif isinstance(v, list):
            kinds.add("tags")
        else:
            kinds.add("text")
    return kinds.pop() if len(kinds) == 1 else None


def _ops_for(kind: str) -> list[str]:
    if kind == "number":
        return [*SET_OPS, "ne", "nin", *RANGE_OPS]
    return [*SET_OPS, "ne", "nin"]  # text, boolean and tags: equality only


def _ordered(values: set) -> list:
    return sorted(values, key=lambda v: (isinstance(v, str), v))


def _from_spec(name: str, spec: FieldSpec) -> FieldInfo:
    return FieldInfo(name, _STATIC_TYPES[name], list(spec.ops), spec.description)


def summarise(documents: list[tuple[dict, dict, str]]) -> list[FieldInfo]:
    """Build the catalogue from (attributes, metadata, source_type) of documents."""
    infos = {name: _from_spec(name, spec) for name, spec in FIELDS.items()}

    seen: dict[str, set] = {name: set() for name in _FROM_DOCUMENTS}
    for _attributes, meta, source_type in documents:
        for name, value in (
            ("source_type", source_type),
            ("file_type", (meta or {}).get("kind")),
            ("language", (meta or {}).get("language")),
        ):
            if isinstance(value, str) and value:
                seen[name].add(value.lower())
    for name, values in seen.items():
        infos[name].values = _ordered(values)[:MAX_VALUES]
        infos[name].complete = len(values) <= MAX_VALUES

    collected: dict[str, list[Any]] = {}
    for attributes, _meta, _source in documents:
        for key, value in (attributes or {}).items():
            collected.setdefault(key, []).append(value)
    for key in sorted(collected):
        values = collected[key]
        kind = _attr_type(values)
        if kind is None:  # documents disagree: a filter on it would be unreliable
            continue
        flat = {x for v in values for x in (v if isinstance(v, list) else [v])}
        info = FieldInfo(
            f"{ATTR_PREFIX}{key}",
            kind,
            _ops_for(kind),
            f"Document attribute '{key}'",
            values=_ordered(flat)[:MAX_VALUES],
            complete=len(flat) <= MAX_VALUES,
            documents=len(values),
        )
        if kind == "number":
            info.minimum, info.maximum = min(flat), max(flat)
        infos[info.name] = info
    return list(infos.values())


async def build_catalogue(
    session: AsyncSession,
    workspace_id: uuid.UUID,
    collection_ids: list[uuid.UUID] | None = None,
) -> list[FieldInfo]:
    query = (
        select(Document.attributes, Document.meta, Document.source_type)
        .join(Collection, Collection.id == Document.collection_id)
        .where(Collection.workspace_id == workspace_id)
        .order_by(Document.created_at.desc(), Document.id)
        .limit(MAX_SCANNED_DOCUMENTS)
    )
    if collection_ids is not None:
        query = query.where(Collection.id.in_(collection_ids))
    rows = (await session.execute(query)).all()
    return summarise([(a or {}, m or {}, s) for a, m, s in rows])
