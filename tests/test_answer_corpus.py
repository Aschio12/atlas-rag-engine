"""Grounded answering over the benchmark corpus, with the fake providers.

Not a quality claim: the fake embedder is a hashed bag of words and the fake LLM copies the
best-matching sentences. It checks the wiring end to end on realistic documents: answerable
questions are answered with a citation to the right document, and questions the corpus cannot
answer are met with an abstention instead of an invented answer. Thresholds leave room for the
one-question wobble that random chunk ids cause between equal scores."""

from pathlib import Path

import pytest

from app.retrieve.bench import _normalise, _squash, load_golden

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "phase3"

UNANSWERABLE = [
    "What is the capital of France?",
    "How do I bake sourdough bread at home?",
    "Who won the 2018 football World Cup?",
    "Explain the theory of general relativity",
    "What is the airspeed velocity of an unladen swallow?",
    "Which programming language is best for beginners?",
    "How tall is Mount Everest?",
    "What is the plot of Hamlet?",
]


@pytest.fixture
async def corpus(client, app, space):
    for path in sorted((ROOT / "corpus").glob("*.md")):
        files = {"file": (path.name, path.read_bytes(), "text/markdown")}
        r = await client.post(space["url"], files=files, headers=space["headers"])
        assert r.status_code == 201, r.text
    while await app.state.worker.run_once():
        pass
    return space


async def ask(client, space, question):
    url = f"/v1/workspaces/{space['ws']}/answer"
    r = await client.post(url, json={"question": question}, headers=space["headers"])
    assert r.status_code == 200, r.text
    return r.json()


def cites_the_evidence(answer, item):
    sources = {s["label"]: s for s in answer["sources"]}
    return any(
        _normalise(sources[c["label"]]["title"]) == _normalise(e.document)
        and _squash(e.contains) in _squash(sources[c["label"]]["text"])
        for c in answer["citations"]
        for e in item.evidence
    )


async def test_answerable_questions_are_answered_and_cite_the_right_document(client, corpus):
    items = load_golden(ROOT / "golden.jsonl")
    answered = cited = 0
    for item in items:
        answer = await ask(client, corpus, item.question)
        if answer["abstained"]:
            continue
        answered += 1
        assert answer["citations"] and answer["grounding"]["invalid"] == []
        cited += cites_the_evidence(answer, item)
    assert answered >= len(items) - 4, f"answered only {answered} of {len(items)}"
    assert cited >= answered - 4, f"only {cited} of {answered} answers cited the evidence"


async def test_questions_the_corpus_cannot_answer_are_met_with_an_abstention(client, corpus):
    for question in UNANSWERABLE:
        answer = await ask(client, corpus, question)
        assert answer["abstained"] is True, (question, answer["answer"], answer["confidence"])
        assert answer["citations"] == [] and answer["confidence"]["level"] == "none"
        assert answer["answer"].startswith("I couldn't find this in your documents")
