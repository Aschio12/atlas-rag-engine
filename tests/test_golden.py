"""Golden-file tests: fixed input files, pinned parser and chunker output (Phase 2.9).

If one fails because you changed extraction or chunking on purpose, run
`python tests/golden/regenerate.py`, read the diff of tests/golden/expected, and commit it.
"""

import difflib
import json
import sys
from pathlib import Path

import pytest
from chunk_helpers import check_chunks

sys.path.insert(0, str(Path(__file__).parent / "golden"))
import golden  # noqa: E402

from app.ingest.chunking.base import HeadingIndex  # noqa: E402

FILES = golden.input_files()
IDS = [p.name for p in FILES]
HINT = "Intentional change? run: python tests/golden/regenerate.py and review the diff."


def test_every_input_has_a_snapshot_and_there_are_no_orphans():
    expected = {p.name for p in golden.EXPECTED.glob("*.json")}
    assert expected == {f"{p.name}.json" for p in FILES}, HINT


def test_the_fixture_set_covers_every_supported_kind():
    from app.ingest.files import Kind, detect_kind

    kinds = {detect_kind(p.read_bytes(), p.name) for p in FILES}
    assert kinds == set(Kind)


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_parsing_matches_the_golden_snapshot(path):
    expected = json.loads(golden.expected_path(path).read_text(encoding="utf-8"))["parsed"]
    actual = golden.parsed_snapshot(golden.parse_file(path))
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                json.dumps(expected, indent=1, ensure_ascii=False, sort_keys=True).splitlines(),
                json.dumps(actual, indent=1, ensure_ascii=False, sort_keys=True).splitlines(),
                "expected",
                "actual",
                lineterm="",
                n=2,
            )  # fmt: skip
        )
        pytest.fail(f"{path.name}: parsed output changed.\n{HINT}\n{diff[:3000]}")


@pytest.mark.parametrize("strategy", golden.STRATEGIES, ids=[s.value for s in golden.STRATEGIES])
@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_chunking_matches_the_golden_snapshot(path, strategy):
    expected = json.loads(golden.expected_path(path).read_text(encoding="utf-8"))["chunks"]
    parsed = golden.parse_file(path)
    actual = golden.chunk_snapshot(parsed, strategy)
    assert actual == expected[strategy.value], (
        f"{path.name} / {strategy.value}: chunks changed. {HINT}"
    )


@pytest.mark.parametrize("strategy", golden.STRATEGIES, ids=[s.value for s in golden.STRATEGIES])
@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_golden_chunks_obey_the_boundary_rules(path, strategy):
    """Independent of the snapshots: no text lost, offsets exact, sizes bounded."""
    import asyncio

    from app.ingest.chunking import build_chunker

    parsed = golden.parse_file(path)
    drafts = asyncio.run(build_chunker(strategy, golden._Providers()).chunk(parsed, golden.CONFIG))
    searchable = [d for d in drafts if d.kind != "parent"]
    parents = [d for d in drafts if d.kind == "parent"]
    check_chunks(parsed, searchable, golden.CONFIG)
    if parents:
        check_chunks(
            parsed, parents, golden.CONFIG, max_tokens=int(golden.CONFIG.parent_tokens * 1.25)
        )
    headings = HeadingIndex(parsed.headings)
    for d in drafts:
        assert d.heading_path == headings.path_at(d.char_start)


def test_snapshots_are_byte_for_byte_current():
    """Catches a hand-edited or stale snapshot file (formatting included)."""
    for path in FILES:
        on_disk = golden.expected_path(path).read_text(encoding="utf-8")
        assert on_disk == golden.dump(golden.snapshot(path)), f"{path.name} is stale. {HINT}"
