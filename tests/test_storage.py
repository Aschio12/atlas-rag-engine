import pytest

from app.storage import BlobNotFound, BlobStore, InvalidKey, LocalBlobStore


@pytest.fixture
def store(tmp_path):
    return LocalBlobStore(tmp_path / "blobs")


def test_implements_the_protocol(store):
    assert isinstance(store, BlobStore)


async def test_put_get_exists_delete_roundtrip(store):
    await store.put("ws/col/doc/original", b"hello")
    assert await store.exists("ws/col/doc/original")
    assert await store.get("ws/col/doc/original") == b"hello"
    await store.delete("ws/col/doc/original")
    assert not await store.exists("ws/col/doc/original")


async def test_put_replaces_atomically_and_leaves_no_temp_files(store, tmp_path):
    await store.put("a/b", b"one")
    await store.put("a/b", b"two")
    assert await store.get("a/b") == b"two"
    assert [p.name for p in (tmp_path / "blobs" / "a").iterdir()] == ["b"]


async def test_get_missing_raises(store):
    with pytest.raises(BlobNotFound):
        await store.get("nope/missing")


async def test_delete_missing_is_not_an_error(store):
    await store.delete("nope/missing")


async def test_delete_prefix_removes_the_whole_folder(store):
    for key in ("w/c/d1/original", "w/c/d1/parsed.json", "w/c/d2/original", "w/other/d3/original"):
        await store.put(key, b"x")
    assert await store.delete_prefix("w/c") == 3
    assert not await store.exists("w/c/d1/original")
    assert await store.exists("w/other/d3/original")
    assert await store.delete_prefix("w/c") == 0


@pytest.mark.parametrize(
    "key",
    ["", "/abs", "a/../b", "..", "a//b", "a/b/", "a/./b", "a/b c", "a/\x00", ".hidden/x", "a/é"],
)
async def test_rejects_unsafe_keys(store, key):
    with pytest.raises(InvalidKey):
        await store.put(key, b"x")
    with pytest.raises(InvalidKey):
        await store.get(key)


async def test_symlink_escape_is_blocked(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "blobs").mkdir(exist_ok=True)
    (tmp_path / "blobs" / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        await store.put("link/file", b"x")
    assert not (outside / "file").exists()
