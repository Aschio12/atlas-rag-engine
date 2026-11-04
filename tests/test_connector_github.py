import httpx
import pytest
from pydantic import ValidationError

from app.ingest.connectors.base import ItemFailure, ItemUnchanged, SourceItem
from app.ingest.connectors.github import (
    GithubConnector,
    GithubSourceConfig,
    priority,
    wanted,
)
from app.ingest.connectors.web import SyncAborted

COMMIT = "c0ffee" * 6 + "abcd"
FILES = {
    "README.md": b"# Project\n\nWelcome.",
    "docs/guide.md": b"# Guide\n\nHow to use it.",
    "docs/deep/ref.rst": b"Reference\n=========\n\ntext",
    "src/app.py": b"def main():\n    return 1\n",
    "src/app.min.js": b"var a=1;",
    "node_modules/pkg/index.js": b"module.exports = 1",
    "package-lock.json": b"{}",
    "assets/logo.png": b"\x89PNG....",
    "LICENSE": b"MIT License",
    "big.py": b"x = 1\n" * 200_000,
    "empty.md": b"",
}


def tree():
    return {
        "sha": "treesha",
        "truncated": False,
        "tree": [{"path": "docs", "type": "tree"}]
        + [
            {"path": p, "type": "blob", "size": len(b), "sha": f"sha-{i}"}
            for i, (p, b) in enumerate(FILES.items())
        ],
    }


class FakeGitHub:
    def __init__(self, *, private=False, truncated=False, rate_limited=False):
        self.private, self.truncated, self.rate_limited = private, truncated, rate_limited
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        auth = request.headers.get("authorization")
        if self.rate_limited:
            return httpx.Response(403, headers={"x-ratelimit-remaining": "0"}, json={})
        if self.private and auth != "Bearer tok":
            return httpx.Response(404, json={})
        path = request.url.path
        if path == "/repos/o/r":
            return httpx.Response(200, json={"default_branch": "main"})
        if path == "/repos/o/r/commits/main" or path == "/repos/o/r/commits/v1.0":
            return httpx.Response(200, json={"sha": COMMIT})
        if path.startswith("/repos/o/r/commits/"):
            return httpx.Response(404, json={})
        if path == f"/repos/o/r/git/trees/{COMMIT}":
            t = tree()
            t["truncated"] = self.truncated
            return httpx.Response(200, json=t)
        for p, body in FILES.items():
            if path in (f"/o/r/{COMMIT}/{p}", f"/repos/o/r/contents/{p}"):
                if "contents" in path:
                    assert request.url.params["ref"] == COMMIT
                    assert request.headers["accept"] == "application/vnd.github.raw"
                return httpx.Response(200, content=body)
        return httpx.Response(404)


def connector(handler=None, token="", **cfg):
    handler = handler or FakeGitHub()
    return GithubConnector(
        GithubSourceConfig(repo="o/r", **cfg),
        api_url="https://api.test",
        raw_url="https://raw.test",
        token=token,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def collect(conn):
    return [i async for i in conn.items()]


# --- config ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "repo"),
    [
        ("fastapi/fastapi", "fastapi/fastapi"),
        ("https://github.com/fastapi/fastapi", "fastapi/fastapi"),
        ("https://github.com/fastapi/fastapi.git", "fastapi/fastapi"),
        ("https://github.com/a-b/c.d/", "a-b/c.d"),
        ("  o/r  ", "o/r"),
    ],
)
def test_repo_forms_are_normalised(raw, repo):
    assert GithubSourceConfig(repo=raw).repo == repo


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "justaname",
        "a/b/c",
        "https://gitlab.com/a/b",
        "a/../b",
        "../b",
        "a b/c",
        "a/b;rm -rf",
        "http://github.com.evil.com/a/b",
    ],
)
def test_bad_repos_are_rejected(raw):
    with pytest.raises(ValidationError):
        GithubSourceConfig(repo=raw)


def test_config_cleans_paths_and_extensions_and_checks_bounds():
    c = GithubSourceConfig(repo="o/r", paths=["/docs/", "src//"], extensions=[".MD", "py"])
    assert c.paths == ["docs", "src"] and c.extensions == ["md", "py"]
    assert (c.owner, c.name) == ("o", "r")
    for bad in (
        {"paths": ["../etc"]},
        {"max_files": 0},
        {"max_files": 1001},
        {"max_file_kb": 0},
        {"nope": 1},
    ):
        with pytest.raises(ValidationError):
            GithubSourceConfig(repo="o/r", **bad)


# --- which files ----------------------------------------------------------------------------


def test_wanted_files():
    c = GithubSourceConfig(repo="o/r")
    yes = ["README.md", "docs/guide.md", "src/app.py", "LICENSE", "Makefile", "x/y/z.yaml", "a.rst"]
    no = ["node_modules/x.js", "src/app.min.js", "package-lock.json", "logo.png", "vendor/a.go",
          "dist/b.js", "random", "poetry.lock", "a.exe"]  # fmt: skip
    assert all(wanted(p, 100, c) for p in yes), [p for p in yes if not wanted(p, 100, c)]
    assert not any(wanted(p, 100, c) for p in no), [p for p in no if wanted(p, 100, c)]


def test_size_limits_and_empty_files():
    c = GithubSourceConfig(repo="o/r", max_file_kb=1)
    assert wanted("a.md", 1024, c) and not wanted("a.md", 1025, c) and not wanted("a.md", 0, c)


def test_path_and_extension_filters():
    docs_only = GithubSourceConfig(repo="o/r", paths=["docs"])
    assert wanted("docs/a.md", 5, docs_only) and not wanted("README.md", 5, docs_only)
    assert not wanted("docs2/a.md", 5, docs_only)  # a folder prefix, not a string prefix
    no_code = GithubSourceConfig(repo="o/r", include_code=False)
    assert wanted("a.md", 5, no_code) and not wanted("a.py", 5, no_code)
    only_py = GithubSourceConfig(repo="o/r", extensions=["py"])
    assert wanted("a.py", 5, only_py) and not wanted("a.md", 5, only_py)


def test_priority_puts_readmes_then_docs_then_code_first():
    paths = ["src/b.py", "docs/guide.md", "README.md", "a.py", "docs/deep/x.md"]
    assert sorted(paths, key=priority) == [
        "README.md",
        "docs/guide.md",
        "docs/deep/x.md",
        "a.py",
        "src/b.py",
    ]


# --- the sync -------------------------------------------------------------------------------


async def test_downloads_wanted_files_pinned_to_one_commit():
    gh = FakeGitHub()
    conn = connector(gh)
    items = await collect(conn)
    assert all(isinstance(i, SourceItem) for i in items)
    paths = [i.meta["path"] for i in items]
    assert paths == ["README.md", "docs/guide.md", "docs/deep/ref.rst", "LICENSE", "src/app.py"]
    assert conn.plan.discovered == 5 and conn.plan.complete is True
    readme = items[0]
    assert readme.uri == "https://github.com/o/r/blob/main/README.md"
    assert readme.data == FILES["README.md"] and readme.filename == "README.md"
    assert readme.meta == {
        "repo": "o/r",
        "ref": "main",
        "path": "README.md",
        "commit": COMMIT,
        "blob_sha": "sha-0",
    }
    assert items[-1].title == "src/app.py" and items[0].title is None  # code titled by path
    raw_urls = [str(r.url) for r in gh.requests if r.url.host == "raw.test"]
    assert all(f"/o/r/{COMMIT}/" in u for u in raw_urls)  # never a moving branch name


async def test_a_token_switches_to_the_api_and_is_sent_only_there():
    gh = FakeGitHub(private=True)
    items = await collect(connector(gh, token="tok"))
    assert len(items) == 5
    assert all(r.headers["authorization"] == "Bearer tok" for r in gh.requests)
    assert not [r for r in gh.requests if r.url.host == "raw.test"]  # contents via the API


async def test_private_repo_without_a_token_explains_what_to_do():
    with pytest.raises(SyncAborted, match="GITHUB_TOKEN"):
        await collect(connector(FakeGitHub(private=True)))


async def test_rate_limit_stops_the_sync_with_advice():
    with pytest.raises(SyncAborted, match="rate limit"):
        await collect(connector(FakeGitHub(rate_limited=True)))


async def test_unknown_branch_and_explicit_ref():
    with pytest.raises(SyncAborted, match="'nope' was not found"):
        await collect(connector(ref="nope"))
    items = await collect(connector(ref="v1.0"))
    assert items[0].uri == "https://github.com/o/r/blob/v1.0/README.md"


async def test_unchanged_files_are_not_downloaded_again():
    gh = FakeGitHub()
    known = {"https://github.com/o/r/blob/main/README.md": "sha-0"}
    conn = GithubConnector(
        GithubSourceConfig(repo="o/r"),
        api_url="https://api.test",
        raw_url="https://raw.test",
        client=httpx.AsyncClient(transport=httpx.MockTransport(gh)),
        known=known,
    )
    items = await collect(conn)
    assert isinstance(items[0], ItemUnchanged) and items[0].uri.endswith("README.md")
    assert not any(r.url.path.endswith("README.md") for r in gh.requests)
    assert conn.plan.uris >= set(known)  # still listed, so it is not pruned


async def test_the_file_cap_keeps_the_most_important_files_and_flags_the_plan():
    conn = connector(max_files=2)
    items = await collect(conn)
    assert [i.meta["path"] for i in items] == ["README.md", "docs/guide.md"]
    assert conn.plan.complete is False and any("only the first 2" in w for w in conn.plan.warnings)


async def test_a_truncated_tree_marks_the_plan_incomplete():
    conn = connector(FakeGitHub(truncated=True))
    await collect(conn)
    assert conn.plan.complete is False and any("too large" in w for w in conn.plan.warnings)


async def test_a_file_that_fails_to_download_is_reported_not_fatal():
    class Flaky(FakeGitHub):
        def __call__(self, request):
            if request.url.path.endswith("/docs/guide.md"):
                return httpx.Response(500)
            return super().__call__(request)

    items = await collect(connector(Flaky()))
    failures = [i for i in items if isinstance(i, ItemFailure)]
    assert len(failures) == 1 and "HTTP 500" in failures[0].error
    assert len([i for i in items if isinstance(i, SourceItem)]) == 4


async def test_network_errors_abort_with_a_clear_message():
    def boom(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(SyncAborted, match="Could not reach GitHub"):
        await collect(connector(boom))
