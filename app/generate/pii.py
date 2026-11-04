"""PII redaction (Phase 4.6): keep personal data out of prompts, and out of answers.

Detects emails, phone numbers, payment card numbers, IBANs, US social security numbers and IPv4
addresses, and replaces each with a placeholder (`[EMAIL_1]`). The same value always gets the
same placeholder within one request, so an answer can still say "[EMAIL_1] and [EMAIL_2] are
different people" without either address leaving the server.

Detectors are pattern-based with validation where a checksum exists (Luhn for cards, mod 97 for
IBANs), which keeps false positives rare: order numbers and dates are not cards or phones. They are
not exhaustive: names, postal addresses and free-form identifiers are not detected, and a
language model's judgement is out of scope here. Treat redaction as a safety net with known gaps,
not as anonymisation.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

KINDS = ("email", "phone", "card", "iban", "ssn", "ip")

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
_SSN = re.compile(r"(?<![\w-])(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?![\w-])")
_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d ().-]{7,}\d(?![\w-])")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


def iban_valid(value: str) -> bool:
    compact = value.replace(" ", "")
    if not 15 <= len(compact) <= 34:
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(c, 36)) for c in rearranged)
    return int(number) % 97 == 1


@dataclass(frozen=True)
class Match:
    kind: str
    start: int
    end: int
    value: str


def _candidates(text: str, kinds: frozenset[str]):
    if "email" in kinds:
        for m in _EMAIL.finditer(text):
            yield Match("email", m.start(), m.end(), m.group(0).lower())
    if "iban" in kinds:
        for m in _IBAN.finditer(text):
            if iban_valid(m.group(0)):
                yield Match("iban", m.start(), m.end(), m.group(0).replace(" ", ""))
    if "card" in kinds:
        for m in _CARD.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19 and luhn_valid(digits) and len(set(digits)) > 1:
                yield Match("card", m.start(), m.end(), digits)
    if "ssn" in kinds:
        for m in _SSN.finditer(text):
            yield Match("ssn", m.start(), m.end(), m.group(0))
    if "ip" in kinds:
        for m in _IP.finditer(text):
            if all(int(part) <= 255 for part in m.group(0).split(".")):
                yield Match("ip", m.start(), m.end(), m.group(0))
    if "phone" in kinds:
        for m in _PHONE.finditer(text):
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            if 9 <= len(digits) <= 15 and not _ISO_DATE.match(raw) and not _IP.fullmatch(raw):
                yield Match("phone", m.start(), m.end(), digits)


class Redactor:
    """Redacts text, handing out consistent placeholders for the lifetime of the instance."""

    def __init__(self, kinds: tuple[str, ...] = KINDS):
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown PII kinds: {', '.join(sorted(unknown))}")
        self._kinds = frozenset(kinds)
        self._tokens: dict[tuple[str, str], str] = {}
        self._counts: Counter[str] = Counter()  # occurrences replaced, by kind

    @property
    def found(self) -> dict[str, int]:
        return dict(self._counts)

    def _token(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self._tokens:
            n = sum(1 for k, _ in self._tokens if k == kind) + 1
            self._tokens[key] = f"[{kind.upper()}_{n}]"
        return self._tokens[key]

    def redact(self, text: str) -> str:
        matches = sorted(
            _candidates(text, self._kinds), key=lambda m: (m.start, -(m.end - m.start))
        )
        chosen: list[Match] = []
        for m in matches:  # leftmost, longest wins; overlapping weaker matches are dropped
            if not chosen or m.start >= chosen[-1].end:
                chosen.append(m)
        out, last = [], 0
        for m in chosen:
            out.append(text[last : m.start])
            out.append(self._token(m.kind, m.value))
            self._counts[m.kind] += 1
            last = m.end
        out.append(text[last:])
        return "".join(out)
