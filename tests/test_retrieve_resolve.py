import uuid
from types import SimpleNamespace

import pytest

from app.providers.base import ScoredPoint
from app.retrieve.dense import (
    MAX_FETCH,
    MIN_FETCH,
    OVERFETCH,
    Candidate,
    DenseRetriever,
    collapse,
    fetch_size,
    is_current,
)
from app.retrieve.types import RetrievalTrace


def cand(chunk, score, *, pid=None, col=None, **payload):
    point = ScoredPoint(pid or str(chunk), score, {"chunk_id": str(chunk), **payload})
    return Candidate(point, col or SimpleNamespace(id=uuid.uuid4()))


def test_fetch_size_overfetches_within_bounds():
    assert fetch_size(1) == MIN_FETCH
    assert fetch_size(10) == 10 * OVERFETCH
    assert fetch_size(50) == 50 * OVERFETCH
    assert fetch_size(10_000) == MAX_FETCH


def test_collapse_keeps_the_best_point_of_each_chunk_in_score_order():
    a, b = uuid.uuid4(), uuid.uuid4()
    own_a = cand(a, 0.5)
    question_a = cand(a, 0.9, pid="q1", kind="question")
    own_b = cand(b, 0.7)
    kept, folded = collapse([own_a, question_a, own_b])
    assert kept == [question_a, own_b] and folded == 1


def test_collapse_breaks_ties_by_point_id_so_results_are_stable():
    a, b = uuid.uuid4(), uuid.uuid4()
    first = cand(a, 0.5, pid="a")
    second = cand(b, 0.5, pid="b")
    assert collapse([second, first])[0] == [first, second]


def test_collapse_ignores_points_without_a_usable_chunk_id():
    bad = Candidate(ScoredPoint("not-a-uuid", 0.9, {}), SimpleNamespace(id=uuid.uuid4()))
    good = cand(uuid.uuid4(), 0.1)
    assert collapse([bad, good]) == ([good], 0)


def row(collection_id, version):
    meta = {"indexing": {"version": version}} if version else {}
    return SimpleNamespace(), SimpleNamespace(collection_id=collection_id, meta=meta)


def test_a_point_is_current_only_when_its_chunk_exists_in_its_collection_at_that_version():
    col = SimpleNamespace(id=uuid.uuid4())
    c = cand(uuid.uuid4(), 0.5, col=col, index_version="v2")
    assert is_current(c, None) == "missing"
    assert is_current(c, row(uuid.uuid4(), "v2")) == "missing"  # chunk of another collection
    assert is_current(c, row(col.id, "v1")) == "stale"
    assert is_current(c, row(col.id, None)) == "stale"  # document never finished indexing
    assert is_current(c, row(col.id, "v2")) is None


async def test_searching_collections_of_another_workspace_is_a_programming_error():
    retriever = DenseRetriever(session=None, providers=None)
    foreign = SimpleNamespace(workspace_id=uuid.uuid4())
    with pytest.raises(ValueError, match="belong to the workspace"):
        await retriever.search("q", uuid.uuid4(), [foreign])


def test_a_trace_records_stage_timings():
    import time

    trace = RetrievalTrace("q")
    stage = trace.add("step", time.perf_counter(), count=3)
    assert trace.stages == [stage] and stage.details == {"count": 3} and stage.took_ms >= 0
