import pytest

from app.ingest.files import Kind
from app.ingest.parsers import ParseError, parse

PAGE = b"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>  Help   Center </title>
  <meta name="description" content="How to get help">
  <meta name="author" content="Support Team">
  <style>body { color: red }</style>
  <script>var secret = "do not index me";</script>
</head>
<body>
  <nav><a href="/">Home</a> <a href="/pricing">Pricing</a></nav>
  <header><h1>Getting   started</h1></header>
  <main>
    <p>Welcome to the <b>help</b> center.<br>Second line.</p>
    <h2>Install</h2>
    <ul><li>Download the app</li><li>Run <code>setup</code></li></ul>
    <h2>Pricing table</h2>
    <table>
      <tr><th>Plan</th><th>Price</th></tr>
      <tr><td>Free</td><td>$0</td></tr>
      <tr><td>Pro</td><td>$29</td></tr>
    </table>
    <pre>line 1
    indented line 2</pre>
    <div hidden>hidden text</div>
    <p aria-hidden="true">decorative</p>
  </main>
  <form><input name=q><button>Search</button></form>
  <footer>&copy; 2026 Acme. All rights reserved.</footer>
  <noscript>enable javascript</noscript>
</body>
</html>"""


@pytest.fixture(scope="module")
def doc():
    return parse(PAGE, Kind.HTML)


def test_boilerplate_is_removed(doc):
    for noise in ["secret", "color: red", "Home", "Pricing\n", "Search", "rights reserved",
                  "enable javascript", "hidden text", "decorative"]:  # fmt: skip
        assert noise not in doc.text, noise


def test_title_language_and_metadata(doc):
    assert doc.title == "Help Center"
    assert doc.metadata["language"] == "en"
    assert doc.metadata["description"] == "How to get help"
    assert doc.metadata["author"] == "Support Team"
    assert doc.metadata["parser"] == "html"


def test_headings_with_exact_offsets(doc):
    assert [(h.level, h.text) for h in doc.headings] == [
        (1, "Getting started"),
        (2, "Install"),
        (2, "Pricing table"),
    ]
    for h in doc.headings:
        assert doc.text[h.char_start : h.char_start + len(h.text)] == h.text


def test_paragraph_whitespace_and_inline_markup(doc):
    assert "Welcome to the help center. Second line." in doc.text


def test_lists_become_bullets_on_consecutive_lines(doc):
    assert "- Download the app\n- Run setup" in doc.text


def test_tables_become_rows(doc):
    assert "Plan | Price\nFree | $0\nPro | $29" in doc.text


def test_pre_keeps_layout(doc):
    assert "line 1\n    indented line 2" in doc.text


def test_title_falls_back_to_og_title_then_h1():
    og = parse(b'<meta property="og:title" content="OG"><p>x', Kind.HTML)
    assert og.title == "OG"
    h1 = parse(b"<h1>Only heading</h1><p>x", Kind.HTML)
    assert h1.title == "Only heading"
    assert parse(b"<p>just text", Kind.HTML).title is None


def test_declared_legacy_charset_is_honoured():
    raw = '<meta charset="iso-8859-2"><p>Zażółć gęślą jaźń</p>'.encode("iso-8859-2")
    doc = parse(raw, Kind.HTML)
    assert "Zażółć gęślą jaźń" in doc.text
    assert doc.metadata["encoding"] == "iso-8859-2"


def test_malformed_html_does_not_crash():
    doc = parse(
        b"<div><p>unclosed <b>bold <i>italic</div><h2>after</h2>text<td>stray cell", Kind.HTML
    )
    assert "unclosed bold italic" in doc.text
    assert doc.headings[0].text == "after"


def test_entities_are_decoded():
    assert "5 < 6 & café" in parse(b"<p>5 &lt; 6 &amp; caf&eacute;</p>", Kind.HTML).text


def test_page_that_is_only_boilerplate_is_an_error():
    with pytest.raises(ParseError, match="no readable text"):
        parse(
            b"<html><head><title>x</title></head><body><script>1</script><nav>menu</nav></body></html>",
            Kind.HTML,
        )


def test_deeply_nested_input_is_handled():
    html = b"<div>" * 2000 + b"deep" + b"</div>" * 2000
    assert parse(html, Kind.HTML).text == "deep"


def test_nested_hidden_elements_are_fully_skipped():
    html = b"<p>before</p><div hidden><div>inner</div>still hidden</div><p>after</p>"
    assert parse(html, Kind.HTML).text == "before\n\nafter"
