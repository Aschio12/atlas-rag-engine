"""The retrieval benchmark: its metrics plumbing, the committed golden set, and a regression gate
against the committed baseline (fake providers, so the numbers are exactly reproducible up to
tie-breaking between equal scores)."""

import json
import re
import uuid
from pathlib import Path

import pytest

from app.retrieve.bench import (
    Config,
    Evidence,
    GoldenItem,
    build_and_run,
    judge,
    load_golden,
    main,
    run_benchmark,
    satisfies,
    standard_configs,
)
from app.retrieve.options import RetrievalOptions
from app.retrieve.types import Hit, RetrievalResult, RetrievalTrace

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "phase3"
CORPUS, GOLDEN, BASELINE = ROOT / "corpus", ROOT / "golden.jsonl", ROOT / "baseline.json"
# Equal scores are ordered by random chunk ids, which can swap one question's result (1/36 = 0.028
# on a 36-question set) from run to run; a real regression moves several questions.
TOLERANCE = 0.045


def hit(title, text, rank=1):
    return Hit(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), collection_id=uuid.uuid4(), rank=rank,
        ordinal=0, score=1.0, text=text, title=title, page=None, char_start=0, char_end=len(text),
        heading_path=[], parent_id=None, language="en", source_uri=None, matched="chunk",
    )  # fmt: skip


def item(*evidence, id="q1", tag="t"):
    return GoldenItem(id, "question?", tuple(Evidence(*e) for e in evidence), tag)


# ---- judging ----------------------------------------------------------------------------------


def test_a_hit_satisfies_evidence_from_the_right_document_with_the_phrase():
    ev = Evidence("rate-limits", "Pro plan allows 600 requests")
    assert satisfies(hit("rate limits", "The  PRO plan allows 600\nrequests per minute."), ev)
    assert not satisfies(hit("billing", "The Pro plan allows 600 requests"), ev)  # wrong document
    assert not satisfies(hit("rate limits", "The Free plan allows 60 requests"), ev)  # wrong text


def test_judge_marks_relevant_hits_and_the_evidence_each_one_covers():
    it = item(("a", "alpha"), ("b", "beta"))
    hits = [hit("x", "alpha"), hit("a", "alpha and more"), hit("b", "the beta passage")]
    relevance, satisfied = judge(hits, it)
    assert relevance == [False, True, True] and satisfied == [set(), {0}, {1}]


# ---- the golden set ---------------------------------------------------------------------------


def test_the_committed_golden_set_loads_and_is_varied():
    items = load_golden(GOLDEN)
    assert len(items) >= 30 and len({i.id for i in items}) == len(items)
    assert {i.tag for i in items} == {"keyword", "paraphrase", "multi"}
    assert any(len(i.evidence) > 1 for i in items)


def test_every_evidence_phrase_really_occurs_in_its_document():
    docs = {p.stem: re.sub(r"\s+", " ", p.read_text()).casefold() for p in CORPUS.glob("*.md")}
    assert len(docs) >= 10
    for it in load_golden(GOLDEN):
        for e in it.evidence:
            assert e.document in docs, f"{it.id}: no document named {e.document}"
            phrase = re.sub(r"\s+", " ", e.contains).casefold()
            assert phrase in docs[e.document], f"{it.id}: {e.contains!r} not in {e.document}"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json\n", "line 1"),
        ('{"id": "a", "question": "q", "evidence": []}\n', "evidence"),
        (
            '{"id": "a", "question": " ", "evidence": [{"document": "d", "contains": "c"}]}\n',
            "question",
        ),
        ('{"id": "a", "question": "q"}\n', "line 1"),
        (
            '{"id": "a", "question": "q", "evidence": [{"document": "d", "contains": "c"}]}\n'
            '{"id": "a", "question": "r", "evidence": [{"document": "d", "contains": "c"}]}\n',
            "unique",
        ),
    ],
)
def test_a_malformed_golden_file_is_rejected_with_its_line(tmp_path, content, message):
    path = tmp_path / "g.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_golden(path)


def test_blank_lines_in_a_golden_file_are_ignored(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(
        '\n{"id": "a", "question": "q", "evidence": [{"document": "d", "contains": "c"}]}\n\n'
    )
    assert [i.id for i in load_golden(path)] == ["a"]


# ---- running and scoring (with a scripted pipeline) -------------------------------------------


class Scripted:
    """A pipeline whose answers are fixed per (question, mode)."""

    def __init__(self, answers):
        self.answers, self.calls = answers, []

    async def run(self, question, workspace_id, collections, options):
        self.calls.append((question, options.mode))
        return RetrievalResult(self.answers[options.mode], RetrievalTrace(question))


async def test_metrics_are_computed_per_question_and_averaged():
    items = [
        item(("a", "alpha"), id="one", tag="x"),
        item(("a", "alpha"), ("b", "beta"), id="two", tag="y"),
    ]
    pipeline = Scripted(
        {
            "dense": [hit("zzz", "noise"), hit("a", "alpha here"), hit("b", "beta here")],
            "sparse": [hit("a", "alpha here")],
        }
    )
    configs = [
        Config("dense", RetrievalOptions(mode="dense")),
        Config("sparse", RetrievalOptions(mode="sparse")),
    ]
    report = await run_benchmark(pipeline, uuid.uuid4(), [], items, configs, ks=(1, 3))
    dense, sparse = report.configs
    q1, q2 = dense.queries
    assert (q1.first_relevant_rank, q1.evidence_found) == (2, 1)
    assert (
        q1.metrics["mrr"] == 0.5 and q1.metrics["recall@1"] == 0.0 and q1.metrics["recall@3"] == 1.0
    )
    assert (
        q2.metrics["recall@1"] == 0.0 and q2.metrics["recall@3"] == 1.0
    )  # both pieces in the top 3
    assert (
        dense.metrics["mrr"] == 0.5
        and dense.metrics["hit@1"] == 0.0
        and dense.metrics["hit@3"] == 1.0
    )
    assert sparse.metrics["mrr"] == 1.0 and sparse.metrics["recall@3"] == pytest.approx(0.75)
    assert set(dense.by_tag) == {"x", "y"} and dense.by_tag["x"]["recall@3"] == 1.0
    assert dense.latency_ms["p95"] >= dense.latency_ms["mean"] >= 0
    assert len(pipeline.calls) == 4


async def test_a_question_nothing_answers_scores_zero():
    pipeline = Scripted({"dense": [hit("zzz", "noise")]})
    report = await run_benchmark(
        pipeline, uuid.uuid4(), [], [item(("a", "alpha"))],
        [Config("dense", RetrievalOptions(mode="dense"))], ks=(1, 5),
    )  # fmt: skip
    (q,) = report.configs[0].queries
    assert q.first_relevant_rank is None and q.metrics["mrr"] == 0.0 and q.metrics["ndcg@5"] == 0.0


async def test_an_empty_golden_set_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        await run_benchmark(Scripted({}), uuid.uuid4(), [], [], standard_configs())


def test_standard_configs_compare_the_four_ways_of_searching():
    names = [c.name for c in standard_configs()]
    assert names == ["dense", "sparse", "hybrid", "hybrid+rerank"]
    assert [c.name for c in standard_configs(rerank=False)] == ["dense", "sparse", "hybrid"]
    assert (
        standard_configs()[3].options.rerank.enabled
        and standard_configs()[2].options.mode == "hybrid"
    )
    assert all(c.options.top_k == 10 for c in standard_configs())


async def test_the_report_renders_a_table_per_config_and_lists_misses():
    items = [item(("a", "alpha"), id="hit", tag="x"), item(("a", "gamma"), id="miss", tag="y")]
    pipeline = Scripted({"dense": [hit("a", "alpha")]})
    report = await run_benchmark(
        pipeline, uuid.uuid4(), [], items,
        [Config("dense", RetrievalOptions(mode="dense"))], ks=(1, 5), meta={"embedding": "fake"},
    )  # fmt: skip
    text = report.to_markdown()
    assert "- **embedding**: fake" in text and "| **dense** | 0.500 |" in text
    assert "| x | 1.00 / 1.00 |" in text and "| y | 0.00 / 0.00 |" in text
    assert "- `miss` (y): question?" in text and "`hit`" not in text.split("did not answer")[1]
    data = report.to_dict(detail=True)
    assert data["configs"]["dense"]["metrics"]["recall@5"] == 0.5
    assert data["configs"]["dense"]["queries"][1] == {
        "id": "miss", "tag": "y", "first_relevant_rank": None, "evidence": "0/1",
    }  # fmt: skip
    assert "queries" not in report.to_dict()["configs"]["dense"]


# ---- the real corpus: a regression gate ---------------------------------------------------------


async def test_benchmark_numbers_match_the_committed_baseline_and_hybrid_beats_dense():
    measured = await build_and_run(CORPUS, GOLDEN)
    baseline = json.loads(BASELINE.read_text())
    current = measured.to_dict()
    assert current["meta"]["questions"] == baseline["meta"]["questions"]
    assert set(current["configs"]) == set(baseline["configs"])
    for name, expected in baseline["configs"].items():
        got = current["configs"][name]["metrics"]
        for metric, value in expected["metrics"].items():
            assert got[metric] == pytest.approx(value, abs=TOLERANCE), f"{name} {metric}"
    # a structural expectation rather than a number: fusing in BM25 must not hurt keyword and
    # multi-part questions relative to vector-only search
    by_name = {c.name: c for c in measured.configs}
    for tag in ("keyword", "multi"):
        assert by_name["hybrid"].by_tag[tag]["mrr"] >= by_name["dense"].by_tag[tag]["mrr"]
    assert by_name["hybrid"].metrics["recall@5"] > by_name["dense"].metrics["recall@5"]


def test_the_cli_writes_the_report_and_the_metrics(tmp_path, capsys):
    out, js = tmp_path / "r.md", tmp_path / "r.json"
    assert main(["--out", str(out), "--json", str(js)]) == 0
    assert "# Retrieval benchmark" in out.read_text() and "hybrid+rerank" in capsys.readouterr().out
    data = json.loads(js.read_text())
    assert data["meta"]["embedding"] == "fake:fake-embed" and "queries" in data["configs"]["dense"]


async def test_a_corpus_without_documents_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no .md files"):
        await build_and_run(tmp_path, GOLDEN)
