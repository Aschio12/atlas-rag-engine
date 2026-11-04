from app.generate.events import (
    NOT_FOUND,
    CitationScanner,
    Event,
    PrefixGate,
    split_answer,
)


def feed_all(scanner, pieces):
    return [scanner.feed(p) for p in pieces]


def test_citations_are_found_as_they_complete_even_when_split_across_chunks():
    s = CitationScanner({"S1", "S2"})
    assert feed_all(s, ["Refunds take 14 days [S", "1]. Orders ship", " fast [S2][", "S1]."]) == [
        [], ["S1"], ["S2"], [],
    ]  # fmt: skip
    assert s.seen == ["S1", "S2"]


def test_each_label_is_reported_once_and_unknown_labels_never():
    s = CitationScanner({"S1"})
    assert s.feed("A [S1] and again [S1] and [S5] and [S1, S9]") == ["S1"]
    assert s.feed("more [S1]") == []


def test_grouped_markers_report_every_new_label():
    assert CitationScanner({"S1", "S2", "S3"}).feed("Claim [S1, S3].") == ["S1", "S3"]


def test_an_unclosed_bracket_is_held_but_ordinary_text_is_not():
    s = CitationScanner({"S1"})
    assert s.feed("see [the docs") == []  # not a marker, and not held forever
    assert s.feed(" for more. Then [S1]") == ["S1"]
    s = CitationScanner({"S1"})
    assert s.feed("[") == [] and s.feed("S1") == [] and s.feed("]") == ["S1"]


def test_a_very_long_unclosed_bracket_is_dropped_not_held():
    s = CitationScanner({"S1"})
    s.feed("[" + "x" * 40)
    assert s.feed(" [S1]") == ["S1"]


def test_the_gate_passes_ordinary_replies_through_unchanged():
    g = PrefixGate()
    out = [g.feed(p) for p in ["Refunds ", "are issued ", "within 14 days."]]
    assert "".join(out) == "Refunds are issued within 14 days." and not g.suppressed


def test_the_gate_holds_back_a_possible_not_found_until_it_is_sure():
    g = PrefixGate()
    assert g.feed("NO") == "" and g.feed("T_FOU") == ""
    assert g.feed("ND The documents are silent.") == "" and g.suppressed is True
    assert g.feed("more text") == ""  # nothing of an abstention is ever shown


def test_a_reply_that_only_starts_like_not_found_is_released_whole():
    g = PrefixGate()
    assert g.feed("NO") == ""
    assert g.feed("TE that refunds take 14 days.") == "NOTE that refunds take 14 days."
    assert g.feed(" Next.") == " Next." and not g.suppressed


def test_the_gate_is_case_insensitive_and_ignores_leading_space():
    g = PrefixGate()
    assert g.feed("  not_found: nothing here") == "" and g.suppressed


def test_a_short_reply_is_released_when_the_stream_ends():
    g = PrefixGate()
    assert g.feed("No") == ""
    assert g.flush() == "No" and g.flush() == ""
    g = PrefixGate()
    g.feed("NOT_F")
    assert g.flush() == "NOT_F"


def test_flush_after_a_suppressed_not_found_releases_nothing():
    g = PrefixGate()
    g.feed(NOT_FOUND)
    assert g.flush() == "" and g.suppressed


def test_split_answer_cuts_text_into_pieces_that_rejoin():
    text = "Refunds are issued within 14 days [S1]. " * 3
    pieces = split_answer(text, 10)
    assert "".join(pieces) == text and all(len(p) <= 10 for p in pieces)
    assert split_answer("", 10) == []


def test_events_format_as_server_sent_events():
    e = Event("token", {"text": 'héllo "x"\n'})
    assert e.sse() == 'event: token\ndata: {"text": "héllo \\"x\\"\\n"}\n\n'
