"""Website connector (Phase 2.7): crawl a sitemap, or follow links, politely.

Good manners are part of the design, because this makes the server visit other people's sites:
* `robots.txt` is always honoured (disallowed pages are skipped, Crawl-delay is respected, and a
  site that denies access to robots.txt or errors is not crawled);
* requests are paced (`delay_s`, at least the site's Crawl-delay) and capped (`max_pages`);
* the crawl never leaves the starting host;
* every request goes through the SSRF-safe fetcher.
"""

from __future__ import annotations

import asyncio
import fnmatch
import gzip
import io
import logging
import posixpath
import zlib
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import ParseError as XmlParseError
from defusedxml.ElementTree import fromstring
from pydantic import BaseModel, ConfigDict, Field

from app.ingest.connectors.base import ItemFailure, SourceItem, SyncPlan
from app.ingest.fetch import Fetched, FetchError

log = logging.getLogger("atlas.connectors.web")

USER_AGENT_TOKEN = "AtlasBot"
MAX_SITEMAPS = 20
MAX_SITEMAP_DEPTH = 3
MAX_SITEMAP_BYTES = 20 * 1024 * 1024
MAX_LINKS_PER_PAGE = 5000
MAX_CRAWL_DELAY_S = 10.0
SKIPPED_EXTENSIONS = frozenset(
    "png jpg jpeg gif svg webp ico bmp tiff mp4 mp3 wav avi mov webm zip gz tgz tar rar 7z "
    "css js map woff woff2 ttf eot exe dmg iso bin xls xlsx ppt".split()
)

Fetch = Callable[..., Awaitable[Fetched]]
Sleep = Callable[[float], Awaitable[None]]


class SyncAborted(Exception):
    """The whole sync cannot proceed (e.g. the site's robots.txt is unavailable)."""


class WebSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=8, max_length=2000, examples=["https://docs.example.com/"])
    mode: Literal["sitemap", "links"] = Field(
        default="sitemap",
        description="sitemap: read sitemap.xml (url may be the site, a folder or a sitemap). "
        "links: follow links from the url.",
    )
    max_pages: int = Field(default=50, ge=1, le=500)
    max_depth: int = Field(default=1, ge=0, le=3, description="Link depth (links mode)")
    include: list[str] = Field(default_factory=list, max_length=20, description="Path globs")
    exclude: list[str] = Field(default_factory=list, max_length=20, description="Path globs")
    delay_s: float = Field(default=0.5, ge=0.0, le=MAX_CRAWL_DELAY_S)
    prune: bool = Field(
        default=True, description="Remove documents whose page is gone (only after a full sync)"
    )


# --- URL helpers ----------------------------------------------------------------------------


def normalize_url(url: str, *, keep_query: bool = True) -> str | None:
    """Canonical form used as a page's identity: lowercase scheme and host, no fragment, no
    default port, '/' for an empty path. None if it is not a plain http(s) URL."""
    try:
        parts = urlsplit(url.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not host:
        return None
    default = 443 if parts.scheme == "https" else 80
    netloc = host.lower() + (f":{port}" if port and port != default else "")
    return urlunsplit(
        (parts.scheme, netloc, parts.path or "/", parts.query if keep_query else "", "")
    )


def scope_prefix(start_url: str) -> str:
    """Path prefix a crawl stays inside: the starting folder (the whole host for a root URL
    or a sitemap file)."""
    path = urlsplit(start_url).path or "/"
    if path.endswith("/"):
        return path
    if path.lower().endswith((".xml", ".xml.gz", "robots.txt")):
        return "/"
    return posixpath.dirname(path).rstrip("/") + "/"


def in_scope(url: str, start_url: str, cfg: WebSourceConfig) -> bool:
    start, target = urlsplit(start_url), urlsplit(url)
    if (target.hostname or "").lower() != (start.hostname or "").lower():
        return False
    if (target.port or 0) != (start.port or 0) and not (
        {target.port, start.port} <= {None, 80, 443}
    ):
        return False
    path = target.path or "/"
    if not path.startswith(scope_prefix(start_url)):
        return False
    ext = path.rsplit(".", 1)[-1].lower() if "." in path.rsplit("/", 1)[-1] else ""
    if ext in SKIPPED_EXTENSIONS:
        return False
    if cfg.include and not any(fnmatch.fnmatchcase(path, g) for g in cfg.include):
        return False
    return not any(fnmatch.fnmatchcase(path, g) for g in cfg.exclude)


# --- robots.txt -----------------------------------------------------------------------------


class Robots:
    def __init__(self, text: str | None, *, allow_all: bool = True):
        self._parser = RobotFileParser()
        self._allow_all = allow_all
        # Always parse, even an empty file: until it has parsed something the stdlib parser
        # answers "not allowed" to everything, which would make a site without robots.txt
        # uncrawlable.
        self._parser.parse((text or "").splitlines())

    def allowed(self, url: str) -> bool:
        if not self._allow_all:
            return False
        return self._parser.can_fetch(USER_AGENT_TOKEN, url)

    @property
    def crawl_delay(self) -> float:
        delay = self._parser.crawl_delay(USER_AGENT_TOKEN) if self._allow_all else None
        return float(delay) if delay else 0.0

    @property
    def sitemaps(self) -> list[str]:
        return list(self._parser.site_maps() or [])


async def load_robots(fetch: Fetch, origin: str) -> Robots:
    """Fetch and parse robots.txt per the usual conventions: missing means 'anything goes',
    forbidden means 'nothing', and an unreachable or failing robots.txt means 'not now'."""
    try:
        got = await fetch(f"{origin}/robots.txt", max_bytes=512 * 1024)
    except FetchError as e:
        if e.blocked:
            raise SyncAborted(str(e)) from None
        if e.status in (401, 403):
            return Robots(None, allow_all=False)
        if e.status is not None and e.status < 500 and e.status != 429:
            return Robots(None)  # 404/410 and friends: no robots.txt, no restrictions
        raise SyncAborted(f"robots.txt could not be read ({e}); not crawling the site") from None
    return Robots(got.data.decode("utf-8", errors="replace"))


# --- sitemaps -------------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(data: bytes) -> tuple[list[str], list[str]]:
    """Return (page urls, child sitemap urls). Handles <urlset>, <sitemapindex> and gzip."""
    if data[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                data = gz.read(MAX_SITEMAP_BYTES + 1)
        except (OSError, EOFError, zlib.error):
            raise ValueError("the sitemap is a corrupt gzip file") from None
        if len(data) > MAX_SITEMAP_BYTES:
            raise ValueError("the sitemap is too large")
    try:
        root = fromstring(data, forbid_dtd=True)
    except (XmlParseError, DefusedXmlException):
        raise ValueError("the sitemap is not valid XML") from None
    pages: list[str] = []
    sitemaps: list[str] = []
    kind = _local(root.tag)
    if kind not in ("urlset", "sitemapindex"):
        raise ValueError("this is not a sitemap")
    for entry in root:
        loc = next((c.text for c in entry if _local(c.tag) == "loc" and c.text), None)
        if loc:
            (sitemaps if _local(entry.tag) == "sitemap" else pages).append(loc.strip())
    return pages, sitemaps


# --- links ----------------------------------------------------------------------------------


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.base: str | None = None
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "base" and a.get("href") and self.base is None:
            self.base = a["href"]
        elif tag == "a" and a.get("href") and "nofollow" not in (a.get("rel") or "").lower():
            if len(self.hrefs) < MAX_LINKS_PER_PAGE:
                self.hrefs.append(a["href"])


def extract_links(html: bytes, page_url: str) -> list[str]:
    parser = _Links()
    parser.feed(html.decode("utf-8", errors="replace"))
    base = urljoin(page_url, parser.base) if parser.base else page_url
    return [
        urljoin(base, h)
        for h in parser.hrefs
        if not h.lower().startswith(("javascript:", "mailto:", "tel:"))
    ]


# --- the connector --------------------------------------------------------------------------


class WebConnector:
    def __init__(
        self,
        config: WebSourceConfig,
        fetch: Fetch,
        *,
        max_bytes: int = 25 * 1024 * 1024,
        sleep: Sleep = asyncio.sleep,
    ):
        self.config = config
        self._fetch = fetch
        self._max_bytes = max_bytes
        self._sleep = sleep
        self.plan = SyncPlan()
        self._start = normalize_url(config.url) or config.url

    @property
    def _origin(self) -> str:
        parts = urlsplit(self._start)
        return f"{parts.scheme}://{parts.netloc}"

    async def items(self) -> AsyncIterator[SourceItem | ItemFailure]:
        robots = await load_robots(self._fetch, self._origin)
        delay = min(max(self.config.delay_s, robots.crawl_delay), MAX_CRAWL_DELAY_S)
        if self.config.mode == "sitemap":
            urls = await self._sitemap_urls(robots)
            self.plan.discovered = len(urls)
            self.plan.uris = set(urls)
            queue = deque(urls)
            follow = False
        else:
            queue = deque([self._start])
            self.plan.uris = {self._start}
            self.plan.discovered = 1
            follow = True
        depth = {self._start: 0}
        first = True
        fetched = 0
        while queue:
            url = queue.popleft()
            if not robots.allowed(url):
                self.plan.warnings.append(f"Skipped (robots.txt): {url}")
                self.plan.uris.discard(url)
                continue
            if not first:
                await self._sleep(delay)
            first = False
            try:
                got = await self._fetch(url, max_bytes=self._max_bytes)
            except FetchError as e:
                yield ItemFailure(url, str(e))
                continue
            fetched += 1
            yield SourceItem(
                uri=url,
                data=got.data,
                content_type=got.content_type,
                filename=got.filename,
                meta={"url": url, "final_url": got.url, "crawled": True},
            )
            if follow and depth[url] < self.config.max_depth and "html" in (got.content_type or ""):
                for link in extract_links(got.data, got.url):
                    norm = normalize_url(link, keep_query=False)
                    if (
                        not norm
                        or norm in self.plan.uris
                        or not in_scope(norm, self._start, self.config)
                    ):
                        continue
                    if len(self.plan.uris) >= self.config.max_pages:
                        self.plan.complete = False
                        break
                    self.plan.uris.add(norm)
                    self.plan.discovered += 1
                    depth[norm] = depth[url] + 1
                    queue.append(norm)

    async def _sitemap_urls(self, robots: Robots) -> list[str]:
        start_path = urlsplit(self._start).path.lower()
        if start_path.endswith((".xml", ".xml.gz")):
            roots = [self._start]
        else:
            roots = robots.sitemaps or [f"{self._origin}/sitemap.xml"]
        seen_maps: set[str] = set()
        found: list[str] = []
        queue = deque((r, 0) for r in roots)
        while queue and len(seen_maps) < MAX_SITEMAPS:
            sitemap_url, level = queue.popleft()
            if sitemap_url in seen_maps:
                continue
            seen_maps.add(sitemap_url)
            try:
                got = await self._fetch(sitemap_url, max_bytes=MAX_SITEMAP_BYTES)
                pages, children = parse_sitemap(got.data)
            except (FetchError, ValueError) as e:
                self.plan.warnings.append(f"Sitemap {sitemap_url}: {e}")
                continue
            for child in children:
                if level < MAX_SITEMAP_DEPTH:
                    queue.append((child, level + 1))
            for page in pages:
                norm = normalize_url(page)
                if norm and in_scope(norm, self._start, self.config) and norm not in found:
                    found.append(norm)
        if not found and not self.plan.warnings:
            self.plan.warnings.append("The sitemap lists no pages inside the requested scope")
        if len(seen_maps) >= MAX_SITEMAPS and queue:
            self.plan.complete = False
            self.plan.warnings.append("Some sitemaps were not read (too many)")
        if len(found) > self.config.max_pages:
            self.plan.complete = False
            self.plan.warnings.append(
                f"The sitemap lists {len(found)} pages; only the first "
                f"{self.config.max_pages} are read"
            )
            found = found[: self.config.max_pages]
        return found
