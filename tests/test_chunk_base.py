import pytest
from pydantic import ValidationError

from app.ingest.chunking import ChunkingConfig, estimate_tokens
from app.ingest.chunking.base import HeadingIndex, make_draft, sentence_spans, trim
from app.ingest.parsers import Heading, PageSpan, ParsedDocument


def sentences(text):
    return [text[s:e] for s, e in sentence_spans(text)]


# --- config ---------------------------------------------------------------------------------


def test_defaults_are_valid_and_min_tokens_is_a_quarter_of_target():
    cfg = ChunkingConfig()
    assert (cfg.target_tokens, cfg.overlap_tokens, cfg.max_tokens) == (400, 50, 800)
    assert cfg.min_tokens == 100


@pytest.mark.parametrize(
    "bad",
    [
        {"target_tokens": 10},
        {"target_tokens": 400, "overlap_tokens": 300},  # more than half
        {"target_tokens": 400, "max_tokens": 300},
        {"parent_tokens": 300, "child_tokens": 200},
        {"breakpoint_percentile": 10},
        {"unknown_field": 1},
    ],
)
def test_invalid_configs_are_rejected(bad):
    with pytest.raises(ValidationError):
        ChunkingConfig(**bad)


# --- token estimate -------------------------------------------------------------------------


def test_estimate_tokens_basics():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hi") == 1
    assert estimate_tokens("a" * 400) == 100
    # many short words cost more than characters/4 suggests
    assert estimate_tokens("a b c d e f g h i j") == 13


def test_estimate_tokens_counts_cjk_characters_individually():
    assert estimate_tokens("日本語のテキスト") >= 8
    assert estimate_tokens("日本語" * 100) >= 300


def test_estimate_is_monotonic_when_text_grows():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    sizes = [estimate_tokens(text[:n]) for n in range(0, len(text), 25)]
    assert sizes == sorted(sizes)


# --- sentences ------------------------------------------------------------------------------


def test_splits_sentences_and_keeps_spans_exact():
    text = "First one. Second one! Is this third? Yes."
    assert sentences(text) == ["First one.", "Second one!", "Is this third?", "Yes."]
    for s, e in sentence_spans(text):
        assert text[s:e] == text[s:e].strip()


def test_abbreviations_and_decimals_do_not_split():
    text = "Dr. Smith paid 3.5 euros, e.g. for tea. Next sentence."
    assert sentences(text) == ["Dr. Smith paid 3.5 euros, e.g. for tea.", "Next sentence."]


def test_list_markers_do_not_split():
    text = "1. Install the app\n2. Run setup\n3. Done."
    assert sentences(text) == ["1. Install the app", "2. Run setup", "3. Done."]


def test_every_line_break_is_a_boundary():
    text = "name: Ada | plan: Pro\nname: Lin | plan: Free"
    assert sentences(text) == ["name: Ada | plan: Pro", "name: Lin | plan: Free"]


def test_closing_quotes_stay_with_the_sentence():
    assert sentences('He said "stop." Then left.') == ['He said "stop."', "Then left."]


def test_sentence_spans_respect_a_window():
    text = "One. Two. Three. Four."
    spans = sentence_spans(text, 5, 16)
    assert [text[s:e] for s, e in spans] == ["Two.", "Three."]


def test_no_text_is_lost_between_sentences():
    text = "Alpha beta.  Gamma delta!\n\nEpsilon? Zeta"
    joined = "".join(text[s:e] for s, e in sentence_spans(text))
    assert joined.replace(" ", "").replace("\n", "") == text.replace(" ", "").replace("\n", "")


def test_trim():
    assert trim("  hi  ", 0, 6) == (2, 4)
    assert trim("   ", 0, 3) is None
    assert trim("abc", 2, 2) is None


# --- headings and drafts --------------------------------------------------------------------


def test_heading_index_builds_breadcrumbs():
    idx = HeadingIndex(
        [
            Heading(1, "Guide", 0),
            Heading(2, "Setup", 50),
            Heading(3, "Linux", 80),
            Heading(2, "Use", 120),
        ]
    )
    assert idx.path_at(10) == ["Guide"]
    assert idx.path_at(60) == ["Guide", "Setup"]
    assert idx.path_at(90) == ["Guide", "Setup", "Linux"]
    assert idx.path_at(130) == ["Guide", "Use"]  # a sibling replaces its predecessor's subtree
    assert HeadingIndex([Heading(1, "A", 10)]).path_at(3) == []


def test_make_draft_trims_and_describes_the_passage():
    text = "Intro\n\n   Body text here.  \n\nEnd"
    parsed = ParsedDocument(
        text=text,
        pages=[PageSpan(1, 0, 7), PageSpan(2, 9, len(text))],
        headings=[Heading(1, "Intro", 0, 1)],
    )
    d = make_draft(parsed, HeadingIndex(parsed.headings), 7, 28)
    assert d.text == "Body text here."
    assert text[d.char_start : d.char_end] == d.text
    assert (d.page, d.heading_path, d.kind) == (2, ["Intro"], "chunk")
    assert d.token_count == estimate_tokens(d.text)
    assert d.indexable and d.embed_text == d.text


def test_make_draft_returns_none_for_blank_spans():
    parsed = ParsedDocument(text="a   \n  b")
    assert make_draft(parsed, HeadingIndex([]), 1, 6) is None


def test_context_is_prepended_for_embedding_and_parents_are_not_indexable():
    parsed = ParsedDocument(text="Some passage.")
    d = make_draft(parsed, HeadingIndex([]), 0, 13, context="From the refund policy.")
    assert d.embed_text == "From the refund policy.\n\nSome passage."
    parent = make_draft(parsed, HeadingIndex([]), 0, 13, kind="parent")
    assert not parent.indexable
