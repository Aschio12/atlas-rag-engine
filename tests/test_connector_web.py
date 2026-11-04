import gzip

import pytest

from app.ingest.connectors.base import ItemFailure, SourceItem
from app.ingest.connectors.web import (
    Robots,
    SyncAborted,
    WebConnector,
    WebSourceConfig,
    extract_links,
    in_scope,
    load_robots,
    normalize_url,
    parse_sitemap,
    scope_prefix,
)
from app.ingest.fetch import Fetched, FetchError

HOST = "https://docs.example.com"


class Site:
    """A fake website: path -> (body, content type) or an int HTTP status."""

    def __init__(self, pages):
        self.pages = {k if k.startswith("http") else HOST + k: v for k, v in pages.items()}
        self.requests: list[str] = []

    async def fetch(self, url, *, max_bytes=0, **_):
        self.requests.append(url)
        entry = self.pages.get(url)
        if entry is None or isinstance(entry, int):
            raise FetchError(f"HTTP {entry or 404}", status=entry or 404)
        body, ctype = entry
        return Fetched(
            url=url,
            data=body if isinstance(body, bytes) else body.encode(),
            content_type=ctype,
            filename=None,
        )


def html(*links, title="Page"):
    anchors = "".join(f'<a href="{h}">l</a>' for h in links)
    return (f"<html><title>{title}</title><body>{anchors}text</body></html>", "text/html")


def sitemap(*urls):
    body = "".join(f"<url><loc>{u}</loc></url>" for u in urls)
    return (
        f'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{body}</urlset>',
        "application/xml",
    )


async def collect(connector):
    return [i async for i in connector.items()]


class Sleeps:
    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, seconds):
        self.calls.append(seconds)


def cfg(**kw):
    return WebSourceConfig(url=kw.pop("url", HOST + "/"), **kw)


# --- urls and scope -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTPS://Docs.Example.COM:443/a#frag", "https://docs.example.com/a"),
        ("http://example.com", "http://example.com/"),
        ("http://example.com:8080/x?y=1", "http://example.com:8080/x?y=1"),
        ("ftp://example.com/", None),
        ("mailto:a@b.c", None),
        ("javascript:alert(1)", None),
        ("http://", None),
        ("http://example.com:bad/", None),
    ],
)
def test_normalize_url(raw, expected):
    assert normalize_url(raw) == expected


def test_normalize_can_drop_the_query():
    assert normalize_url("https://a.com/x?page=2", keep_query=False) == "https://a.com/x"


@pytest.mark.parametrize(
    ("start", "prefix"),
    [
        ("https://a.com", "/"),
        ("https://a.com/", "/"),
        ("https://a.com/docs/", "/docs/"),
        ("https://a.com/docs/intro.html", "/docs/"),
        ("https://a.com/docs/sitemap.xml", "/"),
        ("https://a.com/sitemap.xml.gz".replace(".gz", ""), "/"),
    ],
)
def test_scope_prefix(start, prefix):
    assert scope_prefix(start) == prefix


def test_in_scope_rules():
    c = cfg(url=HOST + "/docs/", exclude=["*/private/*"], include=["/docs/*"])
    ok = lambda u: in_scope(u, HOST + "/docs/", c)  # noqa: E731
    assert ok(HOST + "/docs/guide")
    assert not ok("https://evil.example.com/docs/guide")  # other host
    assert not ok(HOST + "/blog/post")  # outside the folder
    assert not ok(HOST + "/docs/logo.png") and not ok(HOST + "/docs/app.js")  # assets
    assert not ok(HOST + "/docs/private/x")  # excluded
    assert not in_scope(HOST + "/docs/other", HOST + "/docs/", cfg(include=["/docs/guide*"]))


def test_a_different_port_is_a_different_site():
    c = cfg(url="http://a.com:8080/")
    assert in_scope("http://a.com:8080/x", "http://a.com:8080/", c)
    assert not in_scope("http://a.com:9090/x", "http://a.com:8080/", c)
    assert in_scope("http://a.com:80/x", "http://a.com/", cfg(url="http://a.com/"))


# --- robots ---------------------------------------------------------------------------------

ROBOTS = """User-agent: *
Disallow: /private/
Crawl-delay: 3
Sitemap: https://docs.example.com/map.xml

User-agent: AtlasBot
Disallow: /secret/
Crawl-delay: 4
"""


def test_robots_rules_prefer_our_user_agent_section():
    r = Robots(ROBOTS)
    assert not r.allowed(HOST + "/secret/x")
    assert r.allowed(HOST + "/private/x")  # that rule is for other bots
    assert r.crawl_delay == 4.0
    assert r.sitemaps == ["https://docs.example.com/map.xml"]


def test_robots_deny_all_and_defaults():
    assert not Robots(None, allow_all=False).allowed(HOST + "/")
    assert Robots(None).allowed(HOST + "/anything") and Robots(None).crawl_delay == 0.0


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (404, "allow"),
        (410, "allow"),
        (401, "deny"),
        (403, "deny"),
        (500, "abort"),
        (503, "abort"),
        (429, "abort"),
    ],
)
async def test_robots_txt_status_handling(status, outcome):
    site = Site({"/robots.txt": status})
    if outcome == "abort":
        with pytest.raises(SyncAborted, match="robots.txt"):
            await load_robots(site.fetch, HOST)
    else:
        robots = await load_robots(site.fetch, HOST)
        assert robots.allowed(HOST + "/x") is (outcome == "allow")


async def test_robots_blocked_by_the_ssrf_guard_aborts():
    async def fetch(url, **_):
        raise FetchError("That address is not allowed", blocked=True)

    with pytest.raises(SyncAborted):
        await load_robots(fetch, "http://169.254.169.254")


# --- sitemaps -------------------------------------------------------------------------------


def test_parse_sitemap_urlset_and_index():
    pages, maps = parse_sitemap(sitemap(HOST + "/a", HOST + "/b")[0].encode())
    assert pages == [HOST + "/a", HOST + "/b"] and maps == []
    index = (
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        "<sitemap><loc> https://docs.example.com/s1.xml </loc></sitemap></sitemapindex>"
    )
    assert parse_sitemap(index.encode()) == ([], ["https://docs.example.com/s1.xml"])


def test_parse_sitemap_gzip_and_errors():
    raw = sitemap(HOST + "/a")[0].encode()
    assert parse_sitemap(gzip.compress(raw))[0] == [HOST + "/a"]
    for bad in (b"not xml", b"<html/>", b"\x1f\x8bgarbage"):
        with pytest.raises(ValueError):
            parse_sitemap(bad)


def test_parse_sitemap_rejects_entity_attacks_and_gzip_bombs():
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]>'
        b"<urlset><url><loc>&a;</loc></url></urlset>"
    )
    with pytest.raises(ValueError):
        parse_sitemap(bomb)
    big = gzip.compress(
        b"<urlset>" + b"<url><loc>https://a.com/x</loc></url>" * 1_000_000 + b"</urlset>"
    )
    with pytest.raises(ValueError, match="too large"):
        parse_sitemap(big)


# --- links ----------------------------------------------------------------------------------


def test_extract_links_resolves_relative_urls_and_skips_junk():
    page = (
        '<base href="/docs/"><a href="a.html">1</a><a href="/top">2</a><a href="https://x.com/y#z">3</a>'
        '<a href="mailto:a@b.c">4</a><a href="javascript:void(0)">5</a>'
        '<a rel="nofollow" href="/n">6</a>'
        "<a>7</a>"
    )
    assert extract_links(page.encode(), HOST + "/page") == [
        HOST + "/docs/a.html",
        HOST + "/top",
        "https://x.com/y#z",
    ]


# --- sitemap mode ---------------------------------------------------------------------------


async def test_sitemap_crawl_yields_pages_with_pacing():
    site = Site(
        {
            "/robots.txt": 404,
            "/sitemap.xml": sitemap(HOST + "/a", HOST + "/b", "https://other.com/c"),
            "/a": html(title="A"),
            "/b": html(title="B"),
        }
    )
    sleeps = Sleeps()
    conn = WebConnector(cfg(delay_s=0.7), site.fetch, sleep=sleeps)
    items = await collect(conn)
    assert [i.uri for i in items] == [HOST + "/a", HOST + "/b"]
    assert isinstance(items[0], SourceItem) and items[0].meta["crawled"] is True
    assert conn.plan.discovered == 2 and conn.plan.complete is True
    assert conn.plan.uris == {HOST + "/a", HOST + "/b"}
    assert sleeps.calls == [0.7]  # between pages, not before the first


async def test_crawl_delay_from_robots_overrides_a_smaller_setting():
    site = Site(
        {
            "/robots.txt": ("User-agent: *\nCrawl-delay: 4\n", "text/plain"),
            "/sitemap.xml": sitemap(HOST + "/a", HOST + "/b"),
            "/a": html(),
            "/b": html(),
        }
    )
    sleeps = Sleeps()
    await collect(WebConnector(cfg(delay_s=0.5), site.fetch, sleep=sleeps))
    assert sleeps.calls == [4.0]


async def test_robots_disallowed_pages_are_never_requested():
    site = Site(
        {
            "/robots.txt": ("User-agent: *\nDisallow: /private/\n", "text/plain"),
            "/sitemap.xml": sitemap(HOST + "/ok", HOST + "/private/x"),
            "/ok": html(),
        }
    )
    conn = WebConnector(cfg(), site.fetch, sleep=Sleeps())
    items = await collect(conn)
    assert [i.uri for i in items] == [HOST + "/ok"]
    assert HOST + "/private/x" not in site.requests
    assert any("robots.txt" in w for w in conn.plan.warnings)
    assert HOST + "/private/x" not in conn.plan.uris


async def test_robots_sitemap_directive_and_nested_indexes_are_followed():
    index = (
        '<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<sitemap><loc>{HOST}/s1.xml</loc></sitemap></sitemapindex>"
    )
    site = Site(
        {
            "/robots.txt": (f"Sitemap: {HOST}/index.xml\n", "text/plain"),
            "/index.xml": (index, "xml"),
            "/s1.xml": sitemap(HOST + "/deep"),
            "/deep": html(),
        }
    )
    items = await collect(WebConnector(cfg(), site.fetch, sleep=Sleeps()))
    assert [i.uri for i in items] == [HOST + "/deep"]


async def test_the_start_url_can_be_a_sitemap_file():
    site = Site({"/robots.txt": 404, "/maps/pages.xml": sitemap(HOST + "/a"), "/a": html()})
    items = await collect(
        WebConnector(cfg(url=HOST + "/maps/pages.xml"), site.fetch, sleep=Sleeps())
    )
    assert [i.uri for i in items] == [HOST + "/a"]


async def test_failures_are_reported_per_page_and_do_not_stop_the_crawl():
    site = Site(
        {
            "/robots.txt": 404,
            "/sitemap.xml": sitemap(HOST + "/gone", HOST + "/fine"),
            "/fine": html(),
        }
    )
    items = await collect(WebConnector(cfg(), site.fetch, sleep=Sleeps()))
    assert isinstance(items[0], ItemFailure) and items[0].uri == HOST + "/gone"
    assert isinstance(items[1], SourceItem)


async def test_max_pages_truncates_and_marks_the_plan_incomplete():
    urls = [f"{HOST}/p{i}" for i in range(6)]
    site = Site(
        {"/robots.txt": 404, "/sitemap.xml": sitemap(*urls), **{f"/p{i}": html() for i in range(6)}}
    )
    conn = WebConnector(cfg(max_pages=3), site.fetch, sleep=Sleeps())
    items = await collect(conn)
    assert len(items) == 3 and conn.plan.complete is False
    assert any("only the first 3" in w for w in conn.plan.warnings)


async def test_a_missing_sitemap_is_a_warning_not_a_crash():
    site = Site({"/robots.txt": 404})
    conn = WebConnector(cfg(), site.fetch, sleep=Sleeps())
    assert await collect(conn) == []
    assert conn.plan.warnings and "Sitemap" in conn.plan.warnings[0]


# --- links mode -----------------------------------------------------------------------------


async def test_links_mode_follows_in_scope_links_to_the_depth_limit():
    site = Site(
        {
            "/robots.txt": 404,
            "/": html("/a", "/b?page=2", "https://other.com/x", "/logo.png", "#top"),
            "/a": html("/a/deep", "/"),
            "/b": html("/b/deep"),
            "/a/deep": html("/a/deeper"),
            "/b/deep": html(),
        }
    )
    conn = WebConnector(cfg(mode="links", max_depth=2), site.fetch, sleep=Sleeps())
    items = await collect(conn)
    assert sorted(i.uri for i in items) == sorted(
        [HOST + "/", HOST + "/a", HOST + "/b", HOST + "/a/deep", HOST + "/b/deep"]
    )
    assert HOST + "/a/deeper" not in site.requests  # depth 3
    assert "https://other.com/x" not in site.requests and HOST + "/logo.png" not in site.requests
    assert HOST + "/b?page=2" not in site.requests  # query strings are dropped: no crawler traps


async def test_links_mode_visits_each_page_once_and_respects_the_page_cap():
    site = Site(
        {
            "/robots.txt": 404,
            "/": html("/a", "/b", "/c"),
            "/a": html("/", "/b"),
            "/b": html("/a"),
            "/c": html(),
        }
    )
    conn = WebConnector(cfg(mode="links", max_depth=3, max_pages=3), site.fetch, sleep=Sleeps())
    items = await collect(conn)
    assert len(items) == 3 and len({i.uri for i in items}) == 3
    assert conn.plan.complete is False


async def test_links_mode_stays_inside_the_starting_folder():
    site = Site({"/robots.txt": 404, "/docs/": html("/docs/a", "/blog/x"), "/docs/a": html()})
    items = await collect(
        WebConnector(cfg(url=HOST + "/docs/", mode="links"), site.fetch, sleep=Sleeps())
    )
    assert sorted(i.uri for i in items) == [HOST + "/docs/", HOST + "/docs/a"]


async def test_links_are_not_extracted_from_non_html_pages():
    site = Site({"/robots.txt": 404, "/": ("see <a href='/a'>a</a>", "text/plain"), "/a": html()})
    items = await collect(WebConnector(cfg(mode="links", max_depth=2), site.fetch, sleep=Sleeps()))
    assert [i.uri for i in items] == [HOST + "/"]


async def test_links_mode_honours_robots():
    site = Site(
        {
            "/robots.txt": ("User-agent: *\nDisallow: /a\n", "text/plain"),
            "/": html("/a", "/b"),
            "/b": html(),
        }
    )
    items = await collect(WebConnector(cfg(mode="links", max_depth=1), site.fetch, sleep=Sleeps()))
    assert sorted(i.uri for i in items) == [HOST + "/", HOST + "/b"]
    assert HOST + "/a" not in site.requests


def test_config_validation():
    for bad in (
        {"url": "x"},
        {"max_pages": 0},
        {"max_pages": 501},
        {"max_depth": 9},
        {"delay_s": -1},
        {"mode": "x"},
        {"nope": 1},
    ):
        with pytest.raises(ValueError):
            WebSourceConfig(**{"url": HOST + "/", **bad})
