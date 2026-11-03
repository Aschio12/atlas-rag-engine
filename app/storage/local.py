"""Local-disk blob store. Writes are atomic (temp file + rename) and all disk work runs in a
thread so it never blocks the event loop."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

from app.storage.base import BlobNotFound, split_key


class LocalBlobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = self.root.joinpath(*split_key(key)).resolve()
        # Belt and braces: split_key already forbids '..', symlinks could still escape.
        if not path.is_relative_to(self.root):
            raise ValueError("path escapes the storage root")
        return path

    async def put(self, key: str, data: bytes) -> None:
        path = self._path(key)
        await asyncio.to_thread(self._put, path, data)

    @staticmethod
    def _put(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    async def get(self, key: str) -> bytes:
        path = self._path(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except (FileNotFoundError, IsADirectoryError):
            raise BlobNotFound(key) from None

    async def exists(self, key: str) -> bool:
        path = self._path(key)
        return await asyncio.to_thread(path.is_file)

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await asyncio.to_thread(self._unlink, path)

    @staticmethod
    def _unlink(path: Path) -> None:
        with contextlib.suppress(FileNotFoundError, IsADirectoryError):
            path.unlink()

    async def delete_prefix(self, prefix: str) -> int:
        path = self._path(prefix)
        return await asyncio.to_thread(self._rmtree, path)

    @staticmethod
    def _rmtree(path: Path) -> int:
        if not path.is_dir():
            return 0
        count = sum(1 for p in path.rglob("*") if p.is_file())
        shutil.rmtree(path, ignore_errors=True)
        return count
