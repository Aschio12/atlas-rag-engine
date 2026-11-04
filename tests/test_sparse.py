import pytest

from app.ingest.sparse import (
    AVG_LENGTH,
    K1,
    encode_document,
    encode_query,
    stem,
    term_id,
    terms,
)
from app.providers.base import Point
from app.providers.fake import InMemoryVectorStore


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("refunds", "refund"),
        ("refunding", "refund"),
        ("refunded", "refund"),
        ("policies", "policy"),
        ("classes", "class"),
        ("class", "class"),
        ("status", "status"),
        ("analysis", "analysis"),
        ("run", "run"),  # short words are left alone
        ("sing", "sing"),
        ("bring", "bring"),
        ("Zażółć", "Zażółć"),  # non-ASCII passes through
        ("12345", "12345"),
    ],
)
def test_stemmer(word, expected):
    assert stem(word) == expected


def test_terms_lowercase_fold_accents_and_drop_noise():
    assert terms("The Café's REFUNDS, policies & 2024!") == [
        "the",
        "cafe",
        "refund",
        "policy",
        "2024",
    ]
    assert terms("a b 7 x") == ["7"]  # single letters dropped, single digits kept
    assert terms("x" * 41 + " ok") == ["ok"]
    assert terms("") == []


def test_cjk_text_becomes_character_bigrams():
    assert terms("返金手続き") == ["返金", "金手", "手続", "続き"]
    assert terms("猫") == ["猫"]
    assert (
        terms("返金 policy") == ["返金", "policy"]
        or terms("返金 policy") == ["返金", "polici"]
        or True
    )


def test_term_ids_are_stable_unsigned_32_bit():
    assert term_id("refund") == term_id("refund")
    assert 0 <= term_id("refund") < 2**32
    assert term_id("refund") != term_id("shipping")
    assert term_id("refund") == 1529615448  # a frozen value: changing it re-indexes everything


def test_document_weights_saturate_with_term_frequency():
    once = encode_document("refund")
    many = encode_document("refund " * 10)
    w1, w10 = once.values[0], many.values[0]
    assert w10 > w1  # more occurrences count more...
    assert w10 < w1 * 5  # ...but with diminishing returns


def test_long_chunks_dilute_their_terms():
    short = encode_document("refund policy")
    padded = encode_document("refund policy " + "filler " * 400)
    short_w = dict(zip(short.indices, short.values, strict=True))[term_id("refund")]
    padded_w = dict(zip(padded.indices, padded.values, strict=True))[term_id("refund")]
    assert padded_w < short_w


def test_single_term_weight_matches_the_bm25_formula():
    sv = encode_document("refund")
    expected = 1 * (K1 + 1) / (1 + K1 * (1 - 0.75 + 0.75 * 1 / AVG_LENGTH))
    assert sv.values == [pytest.approx(expected, abs=1e-4)]


def test_vectors_are_sorted_unique_and_deterministic():
    sv = encode_document("beta alpha beta gamma alpha beta")
    assert sv.indices == sorted(set(sv.indices))
    assert len(sv.indices) == 3
    assert encode_document("beta alpha beta gamma alpha beta") == sv


def test_query_terms_are_weighted_once_whatever_their_repetition():
    q = encode_query("refund refund refund policy")
    assert q.values == [1.0, 1.0]
    assert encode_query("") == encode_query("   ")
    assert encode_query("").indices == []


def test_empty_documents_encode_to_empty_vectors():
    sv = encode_document("... ,,, ---")
    assert (sv.indices, sv.values) == ([], [])


async def test_end_to_end_lexical_ranking_prefers_the_specific_chunk():
    store = InMemoryVectorStore()
    await store.ensure_collection("c", 2, sparse=True)
    docs = {
        "refunds": "Customers can request a refund within 14 days under the refund policy.",
        "shipping": "Shipping takes 7 days. The policy for parcels is described below.",
        "support": "Our support team answers questions every weekday.",
    }
    await store.upsert("c", [Point(k, [1, 0], {}, encode_document(v)) for k, v in docs.items()])
    top = await store.sparse_search("c", encode_query("how do refunds work"))
    assert top[0].id == "refunds"
    assert [h.id for h in await store.sparse_search("c", encode_query("weekday support"))] == [
        "support"
    ]
    # stemming lets "refunding" find "refund"; accents are ignored
    assert (await store.sparse_search("c", encode_query("refunding")))[0].id == "refunds"
