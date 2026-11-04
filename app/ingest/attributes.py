"""User-defined document attributes (Phase 3.7): the metadata that lets people say "papers after
2023 by Chen" and have it mean something.

Attributes are a small flat map the user attaches to a document: `{"year": 2024, "author":
"Chen", "tags": ["rag", "eval"], "reviewed": true}`. They are stored with the document and copied
into every vector payload (under `attributes`), so retrieval can filter on them inside the vector
store. Keys are short lowercase identifiers; values are text, numbers, booleans or a list of
texts (tags). Store dates as numbers (a year, or Unix seconds) so they can be compared.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any

from pydantic import AfterValidator

KEY = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
MAX_KEYS = 20
MAX_TEXT = 200
MAX_TAGS = 20
MAX_TAG = 100


def _value(key: str, value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"attribute '{key}': numbers must be finite")
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text or len(text) > MAX_TEXT:
            raise ValueError(f"attribute '{key}': text must be 1 to {MAX_TEXT} characters")
        return text
    if isinstance(value, list):
        if not value or len(value) > MAX_TAGS:
            raise ValueError(f"attribute '{key}': a list must hold 1 to {MAX_TAGS} items")
        tags = []
        for item in value:
            if not isinstance(item, str) or not item.strip() or len(item.strip()) > MAX_TAG:
                raise ValueError(
                    f"attribute '{key}': list items must be texts of 1 to {MAX_TAG} characters"
                )
            tags.append(item.strip())
        return list(dict.fromkeys(tags))
    raise ValueError(f"attribute '{key}': use text, a number, true/false or a list of texts")


def validate_attributes(raw: Any) -> dict[str, Any]:
    """Check and normalise a map of attributes (text is trimmed, tags de-duplicated)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("attributes must be an object")
    if len(raw) > MAX_KEYS:
        raise ValueError(f"at most {MAX_KEYS} attributes are allowed")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not KEY.match(key):
            raise ValueError(
                f"attribute name {key!r} must be lowercase letters, digits and underscores, "
                "starting with a letter (at most 40 characters)"
            )
        out[key] = _value(key, value)
    return out


Attributes = Annotated[dict[str, Any], AfterValidator(validate_attributes)]
