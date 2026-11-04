import uuid

import pytest

from app.retrieve.context import (
    MAX_GAP,
    Block,
    Joiner,
    Piece,
    add_blocks,
    coalesce,
    cosine,
    dedupe,
    edge_order,
    jaccard,
    mmr_order,
    shingles,
)
from app.retrieve.types import Hit

DOC = uuid.uuid4()
TEXT = "".join(chr(ord("a") + i % 26) for i in range(200))  # a document whose offsets are visible


def piece(start, end, ordinal=0, doc=DOC):
    return Piece(uuid.uuid4(), doc, start, end, TEXT[start:end], ordinal)


def hit(text="some passage of text", score=1.0, start=0, doc=DOC):
    return Hit(
        chunk_id=uuid.uuid4(), document_id=doc, collection_id=uuid.uuid4(), rank=1, ordinal=0,
        score=score, text=text, title="T", page=None, char_start=start,
        char_end=start + len(text), heading_path=[], parent_id=None, language="en",
        source_uri=None, matched="chunk",
    )  # fmt: skip


# ---- similarity ------------------------------------------------------------------------------


def test_shingles_are_word_triples_and_ignore_case_and_punctuation():
    assert shingles("The quick, brown fox!") == {
        ("the", "quick", "brown"),
        ("quick", "brown", "fox"),
    }
    assert shingles("two words") == {("two", "words")}
    assert shingles("") == set()


def test_jaccard_and_cosine():
    assert jaccard({1, 2, 3}, {2, 3, 4}) == 0.5
    assert jaccard(set(), {1}) == 0.0
    assert cosine([1, 0], [1, 0]) == 1.0 and cosine([1, 0], [0, 1]) == 0.0
    assert cosine([0, 0], [1, 1]) == 0.0


# ---- dedupe ----------------------------------------------------------------------------------


def test_near_copies_of_a_better_ranked_passage_are_dropped_and_reported():
    base = "Refunds are issued within fourteen days of purchase to the original payment method"
    first, copy, other = (
        hit(base),
        hit(base + " today"),
        hit("Orders ship from the warehouse in Rotterdam"),
    )
    kept, dropped = dedupe([first, other, copy], 0.7)
    assert kept == [first, other]
    assert dropped[0]["chunk_id"] == str(copy.chunk_id)
    assert dropped[0]["duplicate_of"] == str(first.chunk_id) and dropped[0]["similarity"] >= 0.7


def test_a_higher_threshold_keeps_more():
    a = hit("alpha beta gamma delta epsilon zeta eta theta iota kappa")
    b = hit("alpha beta gamma delta epsilon zeta eta theta iota lambda")
    assert len(dedupe([a, b], 0.5)[0]) == 1
    assert len(dedupe([a, b], 0.95)[0]) == 2


# ---- MMR -------------------------------------------------------------------------------------


def test_mmr_with_lambda_one_is_pure_relevance():
    rel = {"a": 0.9, "b": 0.8, "c": 0.1}
    assert mmr_order(["c", "a", "b"], rel, lambda x, y: 1.0, lam=1.0) == ["a", "b", "c"]


def test_mmr_prefers_a_different_passage_over_a_redundant_one():
    rel = {"a": 1.0, "a2": 0.95, "b": 0.6}
    near = {frozenset(("a", "a2")): 0.99}
    sim = lambda x, y: near.get(frozenset((x, y)), 0.1)  # noqa: E731
    assert mmr_order(["a", "a2", "b"], rel, sim, lam=0.5) == ["a", "b", "a2"]
    assert mmr_order(["a", "a2", "b"], rel, sim, lam=1.0) == ["a", "a2", "b"]


def test_mmr_ties_keep_the_incoming_order_and_handle_flat_relevance():
    assert mmr_order(["x", "y", "z"], {"x": 1, "y": 1, "z": 1}, lambda a, b: 0.0, lam=0.7) == [
        "x",
        "y",
        "z",
    ]
    assert mmr_order([], {}, lambda a, b: 0.0, lam=0.5) == []


def test_mmr_orders_every_candidate_exactly_once():
    keys = list("abcdef")
    out = mmr_order(keys, {k: i for i, k in enumerate(keys)}, lambda a, b: 0.5, lam=0.3)
    assert sorted(out) == sorted(keys)


# ---- ordering --------------------------------------------------------------------------------


def test_edge_order_puts_the_best_first_and_second_best_last():
    assert edge_order([1, 2, 3, 4, 5]) == [1, 3, 5, 4, 2]
    assert edge_order([1, 2]) == [1, 2] and edge_order([1]) == [1] and edge_order([]) == []


# ---- merging exact text ----------------------------------------------------------------------


def test_overlapping_pieces_merge_into_one_exact_slice():
    runs = coalesce([piece(30, 70), piece(0, 40), piece(60, 90)])
    ((start, end, text, members),) = runs
    assert (start, end, text) == (0, 90, TEXT[0:90]) and len(members) == 3


def test_touching_pieces_merge_but_a_gap_keeps_them_apart():
    assert len(coalesce([piece(0, 40), piece(40, 80)])) == 1
    runs = coalesce([piece(0, 40), piece(41, 80)])
    assert [(r[0], r[1]) for r in runs] == [(0, 40), (41, 80)]


def test_a_contained_piece_adds_nothing():
    ((_, end, text, members),) = coalesce([piece(0, 100), piece(20, 50)])
    assert end == 100 and text == TEXT[:100] and len(members) == 2


def test_pieces_of_different_documents_are_never_merged_by_add_blocks():
    other = uuid.uuid4()
    a = _block(piece(0, 40), DOC)
    b = _block(piece(10, 50, doc=other), other)
    assert len(add_blocks([a], [b])) == 2


def _block(p, doc, order=0, score=1.0):
    h = hit(p.text, score, start=p.start, doc=doc)
    return Block(doc, p.start, p.end, p.text, [p], [h.chunk_id], score, h, order)


def test_add_blocks_merges_overlapping_blocks_of_a_document_and_keeps_the_best_score():
    a, b = (
        _block(piece(0, 50), DOC, order=1, score=0.4),
        _block(piece(40, 90), DOC, order=0, score=0.9),
    )
    (merged,) = add_blocks([a], [b])
    assert (merged.start, merged.end, merged.text) == (0, 90, TEXT[:90])
    assert merged.score == 0.9 and merged.order == 0
    assert len(merged.retrieved) == 2 and "merged overlapping passages" in merged.notes


def test_a_new_block_bridging_two_existing_ones_joins_all_three():
    a, c = _block(piece(0, 30), DOC), _block(piece(60, 90), DOC)
    bridge = _block(piece(25, 65), DOC)
    (merged,) = add_blocks([a, c], [bridge])
    assert (merged.start, merged.end, merged.text) == (0, 90, TEXT[:90])


@pytest.mark.parametrize("n", [1, 2, 5])
def test_block_tokens_follow_the_text(n):
    p = piece(0, 40 * n)
    assert _block(p, DOC).tokens > 0


# ---- joining across a blank gap -----------------------------------------------------------------

SPACED = "alpha beta gamma. delta epsilon zeta.   eta theta iota kappa lambda"
ONE = SPACED.index(" delta")  # end of the first sentence; one blank character follows
TWO = SPACED.index("   eta")  # end of the second; three blank characters follow
ETA = TWO + 3


def spaced(start, end):
    return Piece(uuid.uuid4(), DOC, start, end, SPACED[start:end], 0)


def test_a_whitespace_gap_is_bridged_with_the_documents_own_text():
    first, second = spaced(0, ONE), spaced(ONE + 1, TWO)
    ((start, end, text, members),) = coalesce([first, second], Joiner({DOC: SPACED}))
    assert (start, end) == (0, TWO) and text == SPACED[:TWO] and len(members) == 2


def test_a_wider_blank_gap_up_to_the_limit_is_still_bridged():
    first, second = spaced(0, TWO), spaced(ETA, len(SPACED))
    assert 0 < ETA - TWO <= MAX_GAP
    ((_, _, text, _),) = coalesce([first, second], Joiner({DOC: SPACED}))
    assert text == SPACED


def test_a_gap_that_skips_real_content_is_not_bridged():
    runs = coalesce([spaced(0, 5), spaced(11, 30)], Joiner({DOC: SPACED}))  # "beta " is skipped
    assert len(runs) == 2


def test_a_gap_wider_than_the_limit_is_not_bridged_even_if_blank():
    text = "a" * 10 + " " * (MAX_GAP + 2) + "b" * 10
    first = Piece(uuid.uuid4(), DOC, 0, 10, text[:10], 0)
    second = Piece(uuid.uuid4(), DOC, 10 + MAX_GAP + 2, len(text), text[10 + MAX_GAP + 2 :], 1)
    assert len(coalesce([first, second], Joiner({DOC: text}))) == 2


def test_without_the_document_text_only_overlap_or_touch_merges():
    assert len(coalesce([spaced(0, ONE), spaced(ONE + 1, TWO)])) == 2
    assert len(coalesce([spaced(0, ONE), spaced(ONE, TWO)])) == 1
    assert len(coalesce([spaced(0, 20), spaced(15, TWO)])) == 1


def test_add_blocks_bridges_a_blank_gap_between_a_new_and_an_existing_block():
    a, b = _block(spaced(0, ONE), DOC), _block(spaced(ONE + 1, TWO), DOC)
    (merged,) = add_blocks([a], [b], Joiner({DOC: SPACED}))
    assert merged.text == SPACED[:TWO]
    assert len(add_blocks([a], [b])) == 2  # no text: stays apart
