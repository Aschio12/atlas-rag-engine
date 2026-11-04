"""Offline language identification: no model, no dependency, no network.

Two cheap signals cover the languages people actually ingest:
* **Script**: Hiragana/Katakana means Japanese, Hangul Korean, Han alone Chinese, and Cyrillic,
  Arabic, Hebrew, Devanagari, Thai and Greek identify themselves.
* **Function words**: for Latin-script text, the share of tokens that are very common words of a
  language (the, de, und, ...) is a surprisingly strong fingerprint.

It is meant for metadata and filtering ("only English chunks"), not for legal-grade detection:
short or mixed text yields `None` rather than a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_TOKENS = 6
MIN_SHARE = 0.12  # share of tokens that must be function words of the winning language

_WORDS: dict[str, str] = {
    "en": "the of and to in is that it for on with as was are be by this at from or an have not "
    "but they which you we can will has had their been there more all would about what when",
    "es": "de la que el en y a los se del las un por con no una su para es al lo como más pero "
    "sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde",
    "fr": "de la le et les des en un du une que est pour qui dans a par plus pas au sur ne se "
    "ce il sont avec ou mais comme nous son vous cette ont été aux leur elle fait être",
    "de": "der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als auch "
    "es an werden aus er hat dass sie nach wird bei einer um am sind noch wie einem über",
    "it": "di e il la che in a per un è non una con si da sono come le del i più ma al della "
    "lo gli dei nel anche alla se ha questo tra era suo loro dalla molto quando",
    "pt": "de a o que e do da em um para é com não uma os no se na por mais as dos como mas foi "
    "ao ele das tem à seu sua ou ser quando muito há nos já está eu também só pelo",
    "nl": "de van het een en in is dat op te zijn met voor niet aan er ook als bij maar om dan "
    "ze uit door over ze naar wordt heeft worden deze tot nog wel meer",
    "sv": "och i att det som en på är av för med till den har de inte om ett men var jag sig "
    "från så kan eller vi då hade ut han hon dem vid sin denna",
    "id": "yang dan di dari untuk dengan pada ini itu dalam tidak akan adalah atau juga ke oleh "
    "ada karena sebagai dapat mereka telah saya kami lebih sudah jika",
    "tr": "ve bir bu da de için ile olarak çok daha gibi ancak kadar sonra en ne var olan ben "
    "değil mi ki o her ama ise veya bunu",
}
STOPWORDS: dict[str, frozenset[str]] = {k: frozenset(v.split()) for k, v in _WORDS.items()}
ALL_STOPWORDS: frozenset[str] = frozenset().union(*STOPWORDS.values())

_SCRIPTS = [
    ("ja", re.compile(r"[぀-ヿ]")),
    ("ko", re.compile(r"[가-힯ᄀ-ᇿ]")),
    ("zh", re.compile(r"[一-鿿㐀-䶿]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
    ("ar", re.compile(r"[؀-ۿ]")),
    ("he", re.compile(r"[֐-׿]")),
    ("hi", re.compile(r"[ऀ-ॿ]")),
    ("th", re.compile(r"[฀-๿]")),
    ("el", re.compile(r"[Ͱ-Ͽ]")),
]
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class LanguageGuess:
    code: str  # ISO 639-1
    confidence: float  # 0..1; share of evidence for the winner


def detect_language(text: str) -> LanguageGuess | None:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 12:
        return None
    # 1. dominant non-Latin script
    counts = {code: len(rx.findall(text)) for code, rx in _SCRIPTS}
    script, hits = max(counts.items(), key=lambda kv: kv[1])
    if hits / len(letters) >= 0.3:
        if script == "zh" and counts["ja"] > 0:
            script = "ja"  # Japanese mixes Han with kana
        return LanguageGuess(script, min(1.0, hits / len(letters) + 0.2))
    # 2. Latin script: function-word share
    tokens = [t.lower() for t in _TOKEN.findall(text)]
    if len(tokens) < MIN_TOKENS:
        return None
    shares = {
        code: sum(1 for t in tokens if t in words) / len(tokens)
        for code, words in STOPWORDS.items()
    }
    best, share = max(shares.items(), key=lambda kv: (kv[1], kv[0] == "en"))
    if share < MIN_SHARE:
        return None
    runner_up = max((v for k, v in shares.items() if k != best), default=0.0)
    return LanguageGuess(best, round(min(1.0, 0.5 + (share - runner_up) * 2), 3))


def dominant_language(codes: list[str | None]) -> str | None:
    """The most common detected language across chunks (None if nothing was detected)."""
    found = [c for c in codes if c]
    if not found:
        return None
    return max(set(found), key=lambda c: (found.count(c), c))
