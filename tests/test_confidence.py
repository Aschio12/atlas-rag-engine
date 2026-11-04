import uuid

import pytest

from app.generate.citations import Grounding, ground
from app.generate.confidence import (
    DEFAULT_ABSTAIN_BELOW,
    Confidence,
    assess,
    coverage,
    finalize,
    level_of,
    retrieval_signal,
)
from app.retrieve.types import ContextBlock, Hit

REFUNDS = "Refunds are issued within 14 days of purchase to the original payment method."


def hit(scores):
    return Hit(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), collection_id=uuid.uuid4(), rank=1,
        ordinal=0, score=1.0, text="t", title="T", page=None, char_start=0, char_end=1,
        heading_path=[], parent_id=None, language="en", source_uri=None, matched="chunk",
        scores=scores,
    )  # fmt: skip


def block(text=REFUNDS, label="S1"):
    return ContextBlock(
        label=label, text=text, document_id=uuid.uuid4(), collection_id=uuid.uuid4(), title="T",
        page=None, heading_path=[], char_start=0, char_end=len(text), chunk_ids=[],
        retrieved_chunk_ids=[], score=1.0, tokens=10,
    )  # fmt: skip


def test_levels_follow_the_thresholds():
    assert level_of(0.9) == "high" and level_of(0.7) == "high"
    assert level_of(0.5) == "medium" and level_of(0.45) == "medium"
    assert level_of(0.3) == "low" and level_of(DEFAULT_ABSTAIN_BELOW) == "low"
    assert level_of(0.1) == "none" and level_of(0.0) == "none"
    assert level_of(0.3, abstain_below=0.4) == "none"


def test_coverage_is_the_share_of_question_words_found_in_the_sources():
    blocks = [block()]
    assert coverage("When are refunds issued?", blocks) == 1.0
    assert coverage("refunds elephants", blocks) == 0.5
    assert coverage("elephants giraffes", blocks) == 0.0
    assert coverage("what is the of", blocks) is None  # nothing but stopwords


def test_the_reranker_score_is_preferred_then_dense_similarity():
    assert retrieval_signal([hit({"rerank": 0.9, "dense": 0.2}), hit({"rerank": 0.4})]) == (
        0.9, "reranker",
    )  # fmt: skip
    assert retrieval_signal([hit({"dense": 0.6}), hit({"dense[hyde]": 0.8})]) == (
        0.8, "dense similarity",
    )  # fmt: skip
    assert retrieval_signal([hit({"sparse": 12.0})]) == (None, "")
    assert retrieval_signal([hit({"rerank": 1.7})])[0] == 1.0  # clamped


def test_a_relevant_context_scores_high():
    c = assess("When are refunds issued?", [hit({"rerank": 0.95})], [block()])
    assert c.level == "high" and c.score == pytest.approx(0.65 * 0.95 + 0.35 * 1.0)
    assert c.signals == {"retrieval": 0.95, "coverage": 1.0} and not c.abstain
    assert any("reranker" in r for r in c.reasons) and any("100%" in r for r in c.reasons)


def test_a_context_that_does_not_mention_the_question_abstains():
    c = assess("battery warranty on the X200 model", [hit({"rerank": 0.05})], [block()])
    assert c.abstain and c.level == "none" and c.score < DEFAULT_ABSTAIN_BELOW


def test_similar_looking_sources_do_not_rescue_a_question_they_never_mention():
    # high dense similarity, but none of the question's key words occur in the sources
    c = assess("quantum chromodynamics gluon", [hit({"dense": 0.4})], [block()])
    assert c.score == pytest.approx(0.65 * 0.4) and c.level == "low"


def test_without_a_semantic_signal_coverage_alone_is_capped():
    c = assess("When are refunds issued?", [hit({"sparse": 9.0})], [block()])
    assert c.score == pytest.approx(0.8) and c.level == "high"
    assert any("keyword search only" in r for r in c.reasons)


def test_no_context_means_no_confidence():
    c = assess("anything", [], [])
    assert c.abstain and c.score == 0.0 and c.reasons == ["no passage was retrieved"]


def test_a_question_without_key_words_stays_neutral():
    c = assess("what is it", [hit({"rerank": 0.8})], [block()])
    assert c.score == pytest.approx(0.8) and c.signals["coverage"] is None


def test_the_abstain_threshold_is_adjustable():
    args = ("refunds elephants", [hit({"rerank": 0.3})], [block()])
    assert not assess(*args).abstain
    assert assess(*args, abstain_below=0.6).abstain


def test_a_well_grounded_answer_keeps_the_confidence_and_a_poor_one_loses_it():
    blocks = [block()]
    before = assess("When are refunds issued?", [hit({"rerank": 0.9})], blocks)
    good = finalize(before, ground("Refunds are issued within 14 days of purchase [S1].", blocks))
    bad = finalize(
        before, ground("Elephants are very large grey animals. Nothing is cited.", blocks)
    )
    assert good.score > bad.score and good.level == "high" and bad.level in ("low", "medium")
    assert good.signals["grounding"] == 1.0 and bad.signals["grounding"] == 0.0
    assert any("1 of 1 sentences cite" in r for r in good.reasons)


def test_weak_and_invalid_citations_are_reported_and_penalised():
    blocks = [block()]
    before = Confidence(0.8, "high")
    g = ground("The warranty covers batteries for ten years [S1][S5].", blocks)
    after = finalize(before, g)
    reasons = " ".join(after.reasons)
    assert "does not match the claim" in reasons and "do not exist" in reasons
    assert after.score < before.score


def test_finalize_leaves_the_score_alone_when_there_is_nothing_to_judge():
    before = Confidence(0.6, "medium", {"retrieval": 0.6}, ["r"])
    assert finalize(before, None) is before
    assert finalize(before, Grounding(answer="Short.")) is before
