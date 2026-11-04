"""GitHub repository connector (Phase 2.7): documentation and source files of a repo.

Uses the public REST API for the file list (one request for the whole tree) and either the API
(when a token is configured, which also unlocks private repositories and a higher rate limit) or
raw.githubusercontent.com for the contents. The sync is pinned to one commit, so a push in the
middle of a sync cannot produce a mixed snapshot, and files whose blob SHA is unchanged since the
last sync are not downloaded again.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from posixpath import basename
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.ingest.connectors.base import ItemFailure, ItemUnchanged, SourceItem, SyncPlan
from app.ingest.connectors.web import SyncAborted
from app.ingest.files import CODE_LANGUAGES

log = logging.getLogger("atlas.connectors.github")

DOC_EXTENSIONS = frozenset({"md", "mdx", "markdown", "rst", "txt", "adoc"})
KNOWN_NAMES = frozenset(
    {"README", "LICENSE", "CONTRIBUTING", "CHANGELOG", "Dockerfile", "Makefile"}
)
SKIP_DIRS = frozenset(
    {"node_modules", "vendor", "dist", "build", ".git", "__pycache__", ".venv", "venv", ".next",
     "coverage", "third_party", "site-packages", ".idea", ".vscode"}
)  # fmt: skip
SKIP_FILES = re.compile(
    r"(\.min\.(js|css)$|package-lock\.json$|yarn\.lock$|pnpm-lock\.yaml$|poetry\.lock$"
    r"|Cargo\.lock$|go\.sum$|\.lock$)"
)
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_URL = re.compile(r"^https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$")


class GithubSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo: str = Field(examples=["fastapi/fastapi"], description="owner/name or a github.com URL")
    ref: str | None = Field(
        default=None, max_length=200, description="Branch or tag (default: the default branch)"
    )
    paths: list[str] = Field(default_factory=list, max_length=20, description="Only these folders")
    extensions: list[str] | None = Field(default=None, max_length=50)
    include_code: bool = Field(default=True, description="Also index source files, not just docs")
    max_files: int = Field(default=200, ge=1, le=1000)
    max_file_kb: int = Field(default=512, ge=1, le=2048)
    prune: bool = Field(default=True, description="Remove documents whose file was deleted")

    @field_validator("repo")
    @classmethod
    def _repo(cls, v: str) -> str:
        v = v.strip()
        if m := _URL.match(v):
            v = f"{m.group(1)}/{m.group(2)}"
        if not _REPO.match(v) or v.split("/")[1] in {".", ".."} or v.split("/")[0] in {".", ".."}:
            raise ValueError("must look like 'owner/name' or https://github.com/owner/name")
        return v

    @field_validator("paths")
    @classmethod
    def _paths(cls, v: list[str]) -> list[str]:
        cleaned = [p.strip().strip("/") for p in v if p.strip().strip("/")]
        if any(".." in p.split("/") for p in cleaned):
            raise ValueError("paths must not contain '..'")
        return cleaned

    @field_validator("extensions")
    @classmethod
    def _extensions(cls, v: list[str] | None) -> list[str] | None:
        return None if v is None else [e.lower().lstrip(".") for e in v if e.strip()]

    @property
    def owner(self) -> str:
        return self.repo.split("/")[0]

    @property
    def name(self) -> str:
        return self.repo.split("/")[1]


def wanted(path: str, size: int, cfg: GithubSourceConfig) -> bool:
    """Should this repository file be indexed?"""
    parts = path.split("/")
    if any(p in SKIP_DIRS for p in parts[:-1]) or SKIP_FILES.search(path):
        return False
    if cfg.paths and not any(path == p or path.startswith(p + "/") for p in cfg.paths):
        return False
    if size > cfg.max_file_kb * 1024 or size == 0:
        return False
    name = parts[-1]
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if cfg.extensions is not None:
        return ext in cfg.extensions
    if ext in DOC_EXTENSIONS:
        return True
    if not ext:
        return name.split(".")[0] in KNOWN_NAMES
    return cfg.include_code and ext in CODE_LANGUAGES


def priority(path: str) -> tuple[int, int, str]:
    """Docs before code, README first, shallow before deep: so a cap keeps what matters."""
    name = basename(path)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    kind = 0 if name.upper().startswith("README") else 1 if ext in DOC_EXTENSIONS else 2
    return (kind, path.count("/"), path)


class GithubConnector:
    def __init__(
        self,
        config: GithubSourceConfig,
        *,
        api_url: str = "https://api.github.com",
        raw_url: str = "https://raw.githubusercontent.com",
        token: str = "",
        client: httpx.AsyncClient | None = None,
        known: dict[str, str] | None = None,
        pause_s: float = 0.0,
    ):
        self.config = config
        self._api = api_url.rstrip("/")
        self._raw = raw_url.rstrip("/")
        self._token = token
        self._client = client or httpx.AsyncClient(timeout=30, follow_redirects=True)
        self._owns_client = client is None
        self._known = known or {}  # uri -> blob sha stored by the previous sync
        self._pause = pause_s
        self.plan = SyncPlan()

    # -- HTTP --------------------------------------------------------------------------------
    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        h = {"Accept": accept, "User-Agent": "AtlasBot/0.1", "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def _get(
        self, url: str, *, accept: str = "application/vnd.github+json"
    ) -> httpx.Response:
        try:
            resp = await self._client.get(url, headers=self._headers(accept))
        except httpx.HTTPError as e:
            raise SyncAborted(f"Could not reach GitHub ({type(e).__name__})") from None
        if resp.status_code in (403, 429) and (
            resp.headers.get("x-ratelimit-remaining") == "0" or resp.status_code == 429
        ):
            raise SyncAborted(
                "GitHub's rate limit was reached; configure GITHUB_TOKEN or retry later"
            )
        if resp.status_code == 401:
            raise SyncAborted("GitHub rejected the configured token")
        return resp

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- discovery and download --------------------------------------------------------------
    async def items(self) -> AsyncIterator[SourceItem | ItemFailure | ItemUnchanged]:
        cfg = self.config
        repo_url = f"{self._api}/repos/{cfg.owner}/{cfg.name}"
        r = await self._get(repo_url)
        if r.status_code == 404:
            raise SyncAborted(
                f"Repository {cfg.repo} was not found (it may be private: configure GITHUB_TOKEN)"
            )
        if r.status_code >= 400:
            raise SyncAborted(f"GitHub answered HTTP {r.status_code} for the repository")
        ref = cfg.ref or r.json().get("default_branch") or "HEAD"

        c = await self._get(f"{repo_url}/commits/{quote(ref, safe='')}")
        if c.status_code in (404, 422):
            raise SyncAborted(f"Branch or tag '{ref}' was not found in {cfg.repo}")
        if c.status_code >= 400:
            raise SyncAborted(f"GitHub answered HTTP {c.status_code} for the commit")
        commit = c.json()["sha"]

        t = await self._get(f"{repo_url}/git/trees/{commit}?recursive=1")
        if t.status_code >= 400:
            raise SyncAborted(f"GitHub answered HTTP {t.status_code} for the file list")
        tree = t.json()
        if tree.get("truncated"):
            self.plan.complete = False
            self.plan.warnings.append("The repository is too large for GitHub to list every file")

        files = [
            e
            for e in tree.get("tree", [])
            if e.get("type") == "blob" and wanted(e["path"], e.get("size", 0), cfg)
        ]
        files.sort(key=lambda e: priority(e["path"]))
        if len(files) > cfg.max_files:
            self.plan.complete = False
            self.plan.warnings.append(
                f"{len(files)} files match; only the first {cfg.max_files} are read"
            )
            files = files[: cfg.max_files]
        self.plan.discovered = len(files)
        uris = {self._uri(ref, e["path"]): e for e in files}
        self.plan.uris = set(uris)

        for first, (uri, entry) in enumerate(uris.items()):
            if self._known.get(uri) == entry["sha"]:
                yield ItemUnchanged(uri)
                continue
            if first and self._pause:
                await asyncio.sleep(self._pause)
            path = entry["path"]
            resp = await self._download(commit, path)
            if resp.status_code != 200:
                yield ItemFailure(uri, f"HTTP {resp.status_code} while downloading {path}")
                continue
            is_doc = path.rsplit(".", 1)[-1].lower() in DOC_EXTENSIONS
            yield SourceItem(
                uri=uri,
                data=resp.content,
                filename=basename(path),
                title=None if is_doc else path,  # docs take their own title once parsed
                meta={
                    "repo": cfg.repo,
                    "ref": ref,
                    "path": path,
                    "commit": commit,
                    "blob_sha": entry["sha"],
                },
            )

    def _uri(self, ref: str, path: str) -> str:
        return f"https://github.com/{self.config.repo}/blob/{quote(ref, safe='/')}/{quote(path)}"

    async def _download(self, commit: str, path: str) -> httpx.Response:
        cfg = self.config
        if self._token:
            url = f"{self._api}/repos/{cfg.owner}/{cfg.name}/contents/{quote(path)}?ref={commit}"
            return await self._get(url, accept="application/vnd.github.raw")
        return await self._get(
            f"{self._raw}/{cfg.owner}/{cfg.name}/{commit}/{quote(path)}", accept="*/*"
        )
