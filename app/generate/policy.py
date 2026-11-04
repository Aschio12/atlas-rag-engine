"""Per-workspace content rules (Phase 4.6).

A workspace owner can shape what the assistant does without touching code: house rules added to
the system prompt ("answer formally, in German"), phrases the assistant must not discuss, a
disclaimer for every answer, and defaults for PII redaction and injection handling.

Deliberately plain: blocked terms are case-insensitive whole phrases, not regular expressions, so
a policy can never be turned into a denial-of-service by a pathological pattern. Rules are applied
by the answering service: to the question before anything is retrieved, and to the answer before
it is returned. A request can make handling *stricter* than the policy (turn PII redaction on) but
never looser.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

PiiMode = Literal["off", "context", "all"]
InjectionMode = Literal["off", "flag", "drop"]
PII_ORDER = {"off": 0, "context": 1, "all": 2}
INJECTION_ORDER = {"off": 0, "flag": 1, "drop": 2}


class WorkspacePolicy(BaseModel):
    instructions: str = Field("", max_length=2000, description="House rules for the assistant")
    blocked_terms: list[str] = Field(
        default_factory=list,
        max_length=50,
        description="Phrases (case-insensitive) the assistant will not discuss",
    )
    refusal_message: str = Field(
        "I can't help with that in this workspace.", min_length=1, max_length=300
    )
    disclaimer: str = Field("", max_length=500, description="Appended to every answer")
    pii: PiiMode = "off"
    injection: InjectionMode = "flag"
    abstain_below: float | None = Field(
        None, ge=0, le=1, description="Answer only above this confidence (default: the server's)"
    )

    @field_validator("blocked_terms")
    @classmethod
    def _terms(cls, terms: list[str]) -> list[str]:
        cleaned = []
        for term in terms:
            term = " ".join(term.split())
            if not 2 <= len(term) <= 100:
                raise ValueError("each blocked term must be 2 to 100 characters")
            if term.casefold() not in {t.casefold() for t in cleaned}:
                cleaned.append(term)
        return cleaned

    @field_validator("instructions", "disclaimer", "refusal_message")
    @classmethod
    def _trim(cls, value: str) -> str:
        return value.strip()


def violated_term(policy: WorkspacePolicy, text: str) -> str | None:
    """The first blocked term that occurs in `text` as a whole phrase, if any."""
    for term in policy.blocked_terms:
        pattern = r"(?<!\w)" + re.escape(term).replace(r"\ ", r"\s+") + r"(?!\w)"
        if re.search(pattern, text, re.IGNORECASE):
            return term
    return None


def stricter_pii(*modes: str | None) -> PiiMode:
    return max((m for m in modes if m), key=PII_ORDER.__getitem__, default="off")  # type: ignore[return-value]


def stricter_injection(*modes: str | None) -> InjectionMode:
    return max((m for m in modes if m), key=INJECTION_ORDER.__getitem__, default="flag")  # type: ignore[return-value]
