"""What every connector produces and how it is configured."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SourceItem:
    """One file or page found at a source, already downloaded."""

    uri: str  # stable identity: the same item gets the same uri on every sync
    data: bytes
    content_type: str | None = None
    filename: str | None = None
    title: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemFailure:
    """An item that could not be fetched. Reported, never fatal to the whole sync."""

    uri: str
    error: str


@dataclass
class ItemUnchanged:
    """An item known to be identical to what is already stored (skipped without downloading)."""

    uri: str


@dataclass
class SyncPlan:
    """Reported by a connector once discovery is finished."""

    discovered: int = 0
    complete: bool = True  # False if a cap or truncation means some items were not even listed
    warnings: list[str] = field(default_factory=list)
    uris: set[str] = field(default_factory=set)  # every uri listed, fetched or not


class Connector(Protocol):
    plan: SyncPlan

    def items(self) -> AsyncIterator[SourceItem | ItemFailure | ItemUnchanged]:
        """Yield items one at a time (politely paced). `plan` is filled in as discovery runs."""
