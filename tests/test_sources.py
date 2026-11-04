import asyncio
import hashlib
import uuid

import httpx
import pytest
from sqlalchemy import select
from test_connector_web import HOST, Site, html, sitemap

from app.db.models import Source, SourceStatus, SourceType
from app.ingest.fetch import FetchError
from app.ingest.vectors import vector_collection

SMALL = {"target_tokens": 64, "overlap_tokens": 8, "max_tokens": 128}


@pytest.fixture(autouse=True)
def _no_real_dns(app):
    async def resolve(host, port):
        return ["93.184.216.34"]  # every test host "resolves" to a public address

    app.state.url_resolver = resolve


@pytest.fixture
async def col(client, space):
    r = await client.post(
        f"/v1/workspaces/{space['ws']}/collections",
        json={"name": "Docs", "chunking_config": SMALL},
        headers=space["headers"],
    )
    cid = r.json()["id"]
    base = f"/v1/workspaces/{space['ws']}/collections/{cid}"
    return {"id": cid, "base": base, "docs": base + "/documents", "sources": base + "/sources"}


def serve(app, site: Site):
    app.state.fetch_url = lambda url, **kw: site.fetch(url, **kw)


def page(title, body="Some content about the product and how it works."):
    return (
        f"<html><title>{title}</title><body><h1>{title}</h1><p>{body}</p></body></html>",
        "text/html",
    )


def web_site(**extra):
    pages = {
        "/robots.txt": 404,
        "/sitemap.xml": sitemap(HOST + "/a", HOST + "/b", HOST + "/c"),
        "/a": page("Alpha guide", "Alpha explains the refund policy in detail."),
        "/b": page("Beta guide", "Beta explains the shipping process in detail."),
        "/c": page("Gamma guide", "Gamma explains the support hours in detail."),
    }
    pages.update(extra)
    return Site(pages)


async def create(client, space, col, body, expect=202):
    r = await client.post(col["sources"], json=body, headers=space["headers"])
    assert r.status_code == expect, r.text
    return r.json()


async def sync_done(client, app, space, col, source_id):
    await app.state.sync_runner.wait(uuid.UUID(source_id))
    r = await client.get(f"{col['sources']}/{source_id}", headers=space["headers"])
    return r.json()


async def run_all(app):
    while await app.state.worker.run_once():
        pass


def web_body(**cfg):
    return {"type": "web", "config": {"url": HOST + "/", "delay_s": 0, **cfg}}


async def docs_of(client, space, col):
    r = await client.get(col["docs"] + "?limit=100", headers=space["headers"])
    return r.json()["items"]


# --- web: creating and syncing --------------------------------------------------------------


async def test_a_website_is_crawled_into_documents_and_fully_indexed(client, app, space, col):
    serve(app, web_site())
    created = await create(client, space, col, web_body())
    assert created["type"] == "web" and created["name"] == "docs.example.com"
    src = await sync_done(client, app, space, col, created["id"])
    assert src["status"] == "done" and src["error"] is None
    assert (src["stats"]["created"], src["stats"]["discovered"], src["stats"]["failed"]) == (
        3,
        3,
        0,
    )
    assert src["document_count"] == 3 and src["last_synced_at"] is not None

    docs = await docs_of(client, space, col)
    assert {d["source_uri"] for d in docs} == {HOST + "/a", HOST + "/b", HOST + "/c"}
    assert all(d["source_type"] == "url" for d in docs)
    await run_all(app)
    docs = await docs_of(client, space, col)
    assert {d["title"] for d in docs} == {
        "Alpha guide",
        "Beta guide",
        "Gamma guide",
    }  # parsed titles
    assert all(d["status"] == "ready" for d in docs)
    st = (await client.get(col["base"] + "/index", headers=space["headers"])).json()
    assert st["in_sync"] is True and st["documents"]["ready"] == 3


async def test_a_second_sync_of_an_unchanged_site_changes_nothing(client, app, space, col):
    serve(app, web_site())
    s = await create(client, space, col, web_body())
    await sync_done(client, app, space, col, s["id"])
    await run_all(app)
    r = await client.post(f"{col['sources']}/{s['id']}/sync", headers=space["headers"])
    assert r.status_code == 202
    src = await sync_done(client, app, space, col, s["id"])
    assert (src["stats"]["created"], src["stats"]["updated"], src["stats"]["unchanged"]) == (
        0,
        0,
        3,
    )
    assert await app.state.worker.run_once() is False  # no new jobs


async def test_a_changed_page_is_updated_and_reindexed_in_place(client, app, space, col):
    site = web_site()
    serve(app, site)
    s = await create(client, space, col, web_body())
    await sync_done(client, app, space, col, s["id"])
    await run_all(app)
    before = {d["source_uri"]: d for d in await docs_of(client, space, col)}

    site.pages[HOST + "/a"] = page(
        "Alpha guide", "Alpha now describes a brand new warranty policy."
    )
    await client.post(f"{col['sources']}/{s['id']}/sync", headers=space["headers"])
    src = await sync_done(client, app, space, col, s["id"])
    assert (src["stats"]["updated"], src["stats"]["unchanged"], src["stats"]["created"]) == (
        1,
        2,
        0,
    )
    await run_all(app)
    after = {d["source_uri"]: d for d in await docs_of(client, space, col)}
    assert after[HOST + "/a"]["id"] == before[HOST + "/a"]["id"]  # same document, new content
    assert after[HOST + "/a"]["content_hash"] != before[HOST + "/a"]["content_hash"]
    parsed = (
        await client.get(
            f"{col['docs']}/{after[HOST + '/a']['id']}/parsed", headers=space["headers"]
        )
    ).json()
    assert "warranty" in parsed["text"]
    assert (await client.get(col["base"] + "/index", headers=space["headers"])).json()["in_sync"]


async def test_pages_that_disappear_are_pruned_with_their_vectors(client, app, space, col):
    site = web_site()
    serve(app, site)
    s = await create(client, space, col, web_body())
    await sync_done(client, app, space, col, s["id"])
    await run_all(app)
    store = app.state.providers.vector_store
    name = vector_collection(uuid.UUID(col["id"]))
    gone_doc = next(d for d in await docs_of(client, space, col) if d["source_uri"].endswith("/c"))

    site.pages[HOST + "/sitemap.xml"] = sitemap(HOST + "/a", HOST + "/b")
    await client.post(f"{col['sources']}/{s['id']}/sync", headers=space["headers"])
    src = await sync_done(client, app, space, col, s["id"])
    assert src["stats"]["removed"] == 1 and src["document_count"] == 2
    assert await store.count(name, {"document_id": gone_doc["id"]}) == 0
    assert (
        await client.get(f"{col['docs']}/{gone_doc['id']}", headers=space["headers"])
    ).status_code == 404


async def test_nothing_is_pruned_after_an_incomplete_or_failing_sync(client, app, space, col):
    site = web_site()
    serve(app, site)
    s = await create(client, space, col, web_body())
    await sync_done(client, app, space, col, s["id"])
    # a page now errors: removing documents on partial information would be wrong
    site.pages[HOST + "/sitemap.xml"] = sitemap(HOST + "/a", HOST + "/b")
    site.pages[HOST + "/a"] = 500
    await client.post(f"{col['sources']}/{s['id']}/sync", headers=space["headers"])
    src = await sync_done(client, app, space, col, s["id"])
    assert src["stats"]["failed"] == 1 and src["stats"]["removed"] == 0
    assert src["document_count"] == 3


async def test_per_page_failures_are_recorded_and_the_rest_still_sync(client, app, space, col):
    serve(app, web_site(**{"/b": 404}))
    s = await create(client, space, col, web_body())
    src = await sync_done(client, app, space, col, s["id"])
    assert src["status"] == "done"
    assert (src["stats"]["created"], src["stats"]["failed"]) == (2, 1)
    assert (
        src["stats"]["errors"][0]["item"] == HOST + "/b"
        and "404" in src["stats"]["errors"][0]["error"]
    )


async def test_a_sync_where_everything_fails_is_a_failed_sync(client, app, space, col):
    serve(app, web_site(**{"/a": 500, "/b": 500, "/c": 500}))
    s = await create(client, space, col, web_body())
    src = await sync_done(client, app, space, col, s["id"])
    assert src["status"] == "failed" and "All 3 items failed" in src["error"]
    assert src["last_synced_at"] is None


async def test_unsupported_and_duplicate_pages_are_skipped_or_unchanged(client, app, space, col):
    site = web_site()
    site.pages[HOST + "/sitemap.xml"] = sitemap(HOST + "/a", HOST + "/dup", HOST + "/blob")
    site.pages[HOST + "/dup"] = site.pages[HOST + "/a"]  # same content, another address
    site.pages[HOST + "/blob"] = (b"\x00\x01\x02binary", "application/octet-stream")
    serve(app, site)
    s = await create(client, space, col, web_body())
    src = await sync_done(client, app, space, col, s["id"])
    assert (src["stats"]["created"], src["stats"]["unchanged"], src["stats"]["skipped"]) == (
        1,
        1,
        1,
    )


async def test_a_site_whose_robots_txt_is_broken_is_not_crawled(client, app, space, col):
    serve(app, web_site(**{"/robots.txt": 503}))
    s = await create(client, space, col, web_body())
    src = await sync_done(client, app, space, col, s["id"])
    assert src["status"] == "failed" and "robots.txt" in src["error"]
    assert await docs_of(client, space, col) == []


async def test_links_mode_through_the_api(client, app, space, col):
    site = Site({"/robots.txt": 404, "/": html("/a", "/b"), "/a": html(), "/b": html()})
    serve(app, site)
    s = await create(client, space, col, web_body(mode="links", max_depth=1))
    src = await sync_done(client, app, space, col, s["id"])
    assert src["stats"]["created"] == 1 or src["stats"]["created"] >= 1
    assert src["stats"]["discovered"] == 3


# --- creation rules -------------------------------------------------------------------------


async def test_internal_addresses_are_refused_at_creation(client, space, col):
    body = {"type": "web", "config": {"url": "http://169.254.169.254/latest/"}}
    r = await create(client, space, col, body, expect=400)
    assert "not allowed" in r["detail"]


async def test_names_are_unique_and_collections_have_a_source_cap(
    client, app, settings, space, col
):
    serve(app, web_site())
    first = await create(client, space, col, {**web_body(), "name": "Docs site"})
    await sync_done(client, app, space, col, first["id"])
    dup = await create(client, space, col, {**web_body(), "name": "Docs site"}, expect=409)
    assert "already has a source" in dup["detail"]
    settings.source_max_per_collection = 1
    capped = await create(client, space, col, {**web_body(), "name": "Another"}, expect=409)
    assert "at most 1" in capped["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"type": "ftp", "config": {}},
        {"type": "web", "config": {"url": "nope"}},
        {"type": "web", "config": {"url": HOST + "/", "max_pages": 9999}},
        {"type": "github", "config": {"repo": "not a repo"}},
        {"type": "github", "config": {"repo": "o/r", "max_files": 0}},
        {"type": "web"},
        {"config": {}},
    ],
)
async def test_invalid_sources_are_rejected(client, space, col, body):
    assert (
        await client.post(col["sources"], json=body, headers=space["headers"])
    ).status_code == 422


async def test_a_sync_cannot_be_started_twice(client, app, space, col):
    gate = asyncio.Event()

    async def slow(url, **kw):
        await gate.wait()
        raise FetchError("HTTP 404", status=404)

    app.state.fetch_url = slow
    s = await create(client, space, col, web_body())
    await asyncio.sleep(0.05)
    r = await client.post(f"{col['sources']}/{s['id']}/sync", headers=space["headers"])
    assert r.status_code == 409 and "already running" in r.json()["detail"]
    gate.set()
    await app.state.sync_runner.wait()


# --- delete ---------------------------------------------------------------------------------


async def test_disconnecting_a_source_keeps_its_documents_by_default(client, app, space, col):
    serve(app, web_site())
    s = await create(client, space, col, web_body())
    await sync_done(client, app, space, col, s["id"])
    r = await client.delete(f"{col['sources']}/{s['id']}", headers=space["headers"])
    assert r.status_code == 204
    docs = await docs_of(client, space, col)
    assert len(docs) == 3
    assert (
        await client.get(f"{col['sources']}/{s['id']}", headers=space["headers"])
    ).status_code == 404


async def test_disconnecting_with_delete_documents_removes_everything_it_made(
    client, app, space, col
):
    serve(app, web_site())
    s = await create(client, space, col, web_body())
    await sync_done(client, app, space, col, s["id"])
    await run_all(app)
    # one manual document must survive
    import io

    files = {"file": ("mine.md", io.BytesIO(b"# Mine\n\nMy own note."), "text/markdown")}
    await client.post(col["docs"], files=files, headers=space["headers"])
    r = await client.delete(
        f"{col['sources']}/{s['id']}?delete_documents=true", headers=space["headers"]
    )
    assert r.status_code == 204
    docs = await docs_of(client, space, col)
    assert [d["title"] for d in docs] == ["mine"]
    store = app.state.providers.vector_store
    assert {
        p.payload["document_id"]
        for p in store._data[vector_collection(uuid.UUID(col["id"]))].values()
    } <= {docs[0]["id"]}


async def test_deleting_a_source_cancels_a_running_sync(client, app, space, col):
    gate = asyncio.Event()

    async def slow(url, **kw):
        await gate.wait()
        raise FetchError("HTTP 404", status=404)

    app.state.fetch_url = slow
    s = await create(client, space, col, web_body())
    await asyncio.sleep(0.05)
    assert app.state.sync_runner.is_running(uuid.UUID(s["id"]))
    r = await client.delete(f"{col['sources']}/{s['id']}", headers=space["headers"])
    assert r.status_code == 204
    assert not app.state.sync_runner.is_running(uuid.UUID(s["id"]))


async def test_deleting_the_collection_removes_its_sources(client, app, space, col):
    serve(app, web_site())
    s = await create(client, space, col, web_body())
    await sync_done(client, app, space, col, s["id"])
    await client.delete(col["base"], headers=space["headers"])
    async with app.state.sessionmaker() as session:
        assert (await session.execute(select(Source))).first() is None


# --- permissions and housekeeping -----------------------------------------------------------


async def test_roles_and_tenancy(client, app, signup, space, col):
    from app.db.models import Role, User, WorkspaceMember

    serve(app, web_site())
    s = await create(client, space, col, web_body())
    await sync_done(client, app, space, col, s["id"])
    eve = await signup("eve@example.com")
    one = f"{col['sources']}/{s['id']}"
    for method, url in [
        ("GET", col["sources"]),
        ("GET", one),
        ("POST", one + "/sync"),
        ("DELETE", one),
    ]:
        assert (await client.request(method, url, headers=eve)).status_code == 404

    viewer = await signup("viewer@example.com")
    async with app.state.sessionmaker() as session:
        uid = (
            await session.execute(select(User.id).where(User.email == "viewer@example.com"))
        ).scalar_one()
        session.add(
            WorkspaceMember(workspace_id=uuid.UUID(space["ws"]), user_id=uid, role=Role.VIEWER)
        )
        await session.commit()
    assert (await client.get(col["sources"], headers=viewer)).status_code == 200
    assert (
        await client.post(f"{col['sources']}/{s['id']}/sync", headers=viewer)
    ).status_code == 403
    assert (await client.post(col["sources"], json=web_body(), headers=viewer)).status_code == 403
    assert (await client.get(col["sources"])).status_code == 401


async def test_a_restart_marks_interrupted_syncs_failed(client, app, space, col):
    async with app.state.sessionmaker() as session:
        session.add(Source(collection_id=uuid.UUID(col["id"]), type=SourceType.WEB, name="stuck",
                           config={"url": HOST + "/"}, status=SourceStatus.RUNNING))  # fmt: skip
        await session.commit()
    assert await app.state.sync_runner.recover() == 1
    listing = (await client.get(col["sources"], headers=space["headers"])).json()["items"]
    assert listing[0]["status"] == "failed" and "restart" in listing[0]["error"]
    assert await app.state.sync_runner.recover() == 0


# --- github ---------------------------------------------------------------------------------


class Repo:
    """A mutable fake GitHub repository."""

    def __init__(self, files):
        self.files = dict(files)
        self.commit = "a" * 40
        self.downloads: list[str] = []

    def sha(self, path):
        return hashlib.sha1(self.files[path]).hexdigest()

    def __call__(self, request):
        path = request.url.path
        if path == "/repos/o/r":
            return httpx.Response(200, json={"default_branch": "main"})
        if path.startswith("/repos/o/r/commits/"):
            return httpx.Response(200, json={"sha": self.commit})
        if path.startswith("/repos/o/r/git/trees/"):
            tree = [
                {"path": p, "type": "blob", "size": len(b), "sha": self.sha(p)}
                for p, b in self.files.items()
            ]
            return httpx.Response(200, json={"truncated": False, "tree": tree})
        for p, body in self.files.items():
            if path == f"/o/r/{self.commit}/{p}":
                self.downloads.append(p)
                return httpx.Response(200, content=body)
        return httpx.Response(404)


def github(app, repo):
    app.state.github_client = httpx.AsyncClient(transport=httpx.MockTransport(repo))
    app.state.settings.github_api_url = "https://api.test"
    app.state.settings.github_raw_url = "https://raw.test"


GITHUB_BODY = {"type": "github", "config": {"repo": "https://github.com/o/r"}}
REPO_FILES = {
    "README.md": b"# Atlas Demo\n\nA demo repository about retrieval.",
    "docs/guide.md": b"# Setup Guide\n\nInstall the package and run the server.",
    "src/app.py": b"def handler(event):\n    return {'ok': True}\n",
    "logo.png": b"\x89PNG....",
}


async def test_a_github_repo_is_synced_titled_and_indexed(client, app, space, col):
    repo = Repo(REPO_FILES)
    github(app, repo)
    s = await create(client, space, col, GITHUB_BODY)
    assert s["name"] == "o/r" and s["type"] == "github"
    src = await sync_done(client, app, space, col, s["id"])
    assert (
        src["status"] == "done" and src["stats"]["created"] == 3 and src["stats"]["discovered"] == 3
    )
    await run_all(app)
    docs = {d["source_uri"]: d for d in await docs_of(client, space, col)}
    assert set(docs) == {
        "https://github.com/o/r/blob/main/README.md",
        "https://github.com/o/r/blob/main/docs/guide.md",
        "https://github.com/o/r/blob/main/src/app.py",
    }
    assert (
        docs["https://github.com/o/r/blob/main/README.md"]["title"] == "Atlas Demo"
    )  # parsed title
    code = docs["https://github.com/o/r/blob/main/src/app.py"]
    assert code["title"] == "src/app.py" and code["kind"] == "code"
    assert code["source_type"] == "github" and code["metadata"]["commit"] == repo.commit
    assert code["metadata"]["parse"]["code_language"] == "python"
    assert (await client.get(col["base"] + "/index", headers=space["headers"])).json()["in_sync"]


async def test_a_github_resync_downloads_only_what_changed_and_prunes_deleted_files(
    client, app, space, col
):
    repo = Repo(REPO_FILES)
    github(app, repo)
    s = await create(client, space, col, GITHUB_BODY)
    await sync_done(client, app, space, col, s["id"])
    await run_all(app)
    repo.downloads.clear()

    repo.files["docs/guide.md"] = b"# Setup Guide\n\nNow with Docker instructions as well."
    del repo.files["src/app.py"]
    repo.commit = "b" * 40
    await client.post(f"{col['sources']}/{s['id']}/sync", headers=space["headers"])
    src = await sync_done(client, app, space, col, s["id"])
    assert (src["stats"]["updated"], src["stats"]["unchanged"], src["stats"]["removed"]) == (
        1,
        1,
        1,
    )
    assert repo.downloads == ["docs/guide.md"]  # README (same blob sha) was not fetched again
    await run_all(app)
    assert src["document_count"] == 2
    docs = await docs_of(client, space, col)
    assert {d["title"] for d in docs} == {"Atlas Demo", "Setup Guide"}


async def test_github_errors_fail_the_sync_with_advice(client, app, space, col):
    app.state.github_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(404))
    )
    s = await create(client, space, col, GITHUB_BODY)
    src = await sync_done(client, app, space, col, s["id"])
    assert src["status"] == "failed" and "GITHUB_TOKEN" in src["error"]
