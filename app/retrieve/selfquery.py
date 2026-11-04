"""Self-query (Phase 3.7): turn "papers after 2023 about retrieval" into a query and filters.

A model reads the question and the catalogue of fields that exist in the workspace and proposes
a cleaned query plus filter conditions. Nothing it says is trusted:

* every condition goes through the same validation as a caller's filter (known field, allowed
  operator, right value type), so it cannot name `workspace_id` or anything else off the list;
* attributes it invents, and values a complete catalogue says do not exist, are dropped;
* a condition that contradicts the caller's explicit filters or another inferred one is dropped
  (explicit filters always win);
* at most `MAX_INFERRED` conditions are kept, and the pipeline can fall back to searching without
  them when they leave nothing (`fallback`), because a wrongly inferred filter would otherwise
  turn "no match" into a silent failure.

Everything dropped is reported with its reason in the trace.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, ValidationError

from app.providers.base import LLM, ChatMessage, Usage
from app.providers.errors import ProviderError
from app.retrieve.catalogue import FieldInfo
from app.retrieve.filters import FilterError, MetadataFilter, compile_filters

MAX_INFERRED = 5
# Fields shown to the model: ids, titles and headings are exact strings it cannot know.
OFFERED = ("source_type", "file_type", "language", "page", "created_at")
MAX_QUERY_CHARS = 500


class SelfQueryOptions(BaseModel):
    enabled: bool = False
    fallback: bool = True  # search again without the inferred filters if they match nothing


@dataclass
class SelfQueryResult:
    query: str  # the cleaned query when filters were found, else the original
    conditions: list[MetadataFilter] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


PROMPT = """You turn a search question into a cleaned search query plus metadata filters.

Today is {today}.

Fields you may filter on:
{fields}

<question>{query}</question>

Rules:
- Add a filter only when the question clearly restricts the search by one of these fields. When
  unsure, add none.
- Use exactly the field names and operators listed. Dates use ISO format (YYYY-MM-DD).
- For a number field: "after 2023" is gt 2023, "since 2023" or "from 2023" is gte 2023, "before
  2023" is lt 2023, "in 2023" is eq 2023.
- When a field lists "all values", only use values from that list.
- "query" is the question without the words that became filters; keep every topic word. If
  nothing became a filter, return the question unchanged.

Reply with JSON only:
{{"query": "...", "filters": [{{"field": "...", "op": "...", "value": ...}}]}}"""


def describe(info: FieldInfo) -> str:
    parts = [info.type]
    if info.documents is not None:
        parts.append(f"{info.documents} documents")
    if info.minimum is not None:
        parts.append(f"from {info.minimum:g} to {info.maximum:g}")
    if info.values and info.type != "number":
        label = "all values" if info.complete else "examples"
        parts.append(f"{label}: {', '.join(str(v) for v in info.values[:12])}")
    parts.append(f"ops: {', '.join(info.ops)}")
    return f"- {info.name} ({'; '.join(parts)})"


def offered_fields(catalogue: list[FieldInfo]) -> list[FieldInfo]:
    return [i for i in catalogue if i.name in OFFERED or i.name.startswith("attr.")]


def _parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in the reply")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data


def _check_value(info: FieldInfo, cond: MetadataFilter) -> str | None:
    """Reject values a complete catalogue proves nobody has."""
    if not info.complete or not info.values or info.type == "number":
        return None
    if cond.op not in ("eq", "in", "ne", "nin"):
        return None
    wanted = cond.value if isinstance(cond.value, list) else [cond.value]
    known = {str(v).lower() for v in info.values}
    missing = [w for w in wanted if str(w).lower() not in known]
    if missing and cond.op in ("eq", "in"):
        return f"no document has {', '.join(map(repr, missing))} for {info.name}"
    return None


async def infer_filters(
    llm: LLM,
    query: str,
    catalogue: list[FieldInfo],
    explicit: list[MetadataFilter],
    *,
    today: date,
) -> SelfQueryResult:
    result = SelfQueryResult(query)
    fields = offered_fields(catalogue)
    if not fields:
        return result
    prompt = PROMPT.format(
        today=today.isoformat(), fields="\n".join(describe(i) for i in fields), query=query
    )
    try:
        resp = await llm.complete([ChatMessage("user", prompt)], temperature=0.0, max_tokens=400)
        result.usage = resp.usage
        data = _parse(resp.text)
    except (ProviderError, ValueError) as exc:
        result.warnings.append(f"Self-query failed ({exc}); searched without inferred filters")
        return result

    by_name = {i.name: i for i in fields}
    proposed = data.get("filters")
    for raw in proposed if isinstance(proposed, list) else []:
        if len(result.conditions) >= MAX_INFERRED:
            result.rejected.append({"condition": raw, "reason": "too many inferred filters"})
            continue
        try:
            cond = MetadataFilter.model_validate(raw)
        except ValidationError as exc:
            reason = "; ".join(e["msg"] for e in exc.errors())
            result.rejected.append({"condition": raw, "reason": reason})
            continue
        info = by_name.get(cond.field)
        if info is None:
            result.rejected.append({"condition": raw, "reason": f"{cond.field} is not offered"})
            continue
        if reason := _check_value(info, cond):
            result.rejected.append({"condition": raw, "reason": reason})
            continue
        try:
            compile_filters([*explicit, *result.conditions, cond])
        except FilterError as exc:
            reason = f"conflicts with other filters ({exc})"
            result.rejected.append({"condition": raw, "reason": reason})
            continue
        result.conditions.append(cond)

    cleaned = data.get("query")
    if result.conditions and isinstance(cleaned, str):
        text = " ".join(cleaned.split())
        if 3 <= len(text) <= MAX_QUERY_CHARS and any(ch.isalnum() for ch in text):
            result.query = text
    return result
