"""Snapshot helpers for the golden-file tests.

A golden test pins the *exact* output of the parsers and chunkers for fixed input files, so any
change in extraction or splitting shows up as a reviewable diff instead of a silent regression.
Intentional changes: run `python tests/golden/regenerate.py`, read the diff, commit it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.db.models import ChunkingStrategy
from app.ingest.chunking import ChunkingConfig, build_chunker
from app.ingest.files import detect_kind
from app.ingest.parsers import ParsedDocument, parse
from app.providers.fake import FakeEmbedder, FakeLLM

HERE = Path(__file__).parent
INPUTS = HERE / "inputs"
EXPECTED = HERE / "expected"

CONFIG = ChunkingConfig(
    target_tokens=64, overlap_tokens=8, max_tokens=128, parent_tokens=256, child_tokens=64
)
# Deterministic strategies only: contextual needs a real LLM to mean anything.
STRATEGIES = [s for s in ChunkingStrategy if s is not ChunkingStrategy.CONTEXTUAL]


class _Providers:
    """Just enough of Providers for the offline strategies."""

    embedder = FakeEmbedder()
    llm = FakeLLM()


def input_files() -> list[Path]:
    return sorted(p for p in INPUTS.iterdir() if p.is_file())


def parse_file(path: Path) -> ParsedDocument:
    data = path.read_bytes()
    return parse(data, detect_kind(data, path.name), filename=path.name)


def parsed_snapshot(parsed: ParsedDocument) -> dict:
    return json.loads(json.dumps(parsed.to_dict(), ensure_ascii=False))


def chunk_snapshot(parsed: ParsedDocument, strategy: ChunkingStrategy) -> list[dict]:
    chunker = build_chunker(strategy, _Providers())
    drafts = asyncio.run(chunker.chunk(parsed, CONFIG))
    return [
        {
            "start": d.char_start,
            "end": d.char_end,
            "kind": d.kind,
            "page": d.page,
            "heading_path": d.heading_path,
            "parent": d.parent,
            "tokens": d.token_count,
        }
        for d in drafts
    ]


def snapshot(path: Path) -> dict:
    parsed = parse_file(path)
    return {
        "parsed": parsed_snapshot(parsed),
        "chunks": {s.value: chunk_snapshot(parsed, s) for s in STRATEGIES},
    }


def dump(data: dict) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def expected_path(path: Path) -> Path:
    return EXPECTED / f"{path.name}.json"
