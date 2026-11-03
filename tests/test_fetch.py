import httpx
import pytest

from app.ingest.fetch import FetchError, fetch_url, validate_url

PUBLIC = "93.184.216.34"


def resolver_for(mapping: dict[str, list[str]]):
    async def resolve(host: str, port: int) -> list[str]:
        if host not in mapping:
            raise FetchError(f"Could not resolve host '{host}'")
        return mapping[host]

    return resolve


RESOLVE = resolver_for({"example.com": [PUBLIC], "cdn.example.com": [PUBLIC]})


def transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "gopher://example.com",
        "//example.com/x",
        "http://",
        "http://user:pass@example.com/",
        "http://example.com:notaport/",
        "http://" + "a" * 2100,
    ],
)
async def test_rejects_bad_schemes_and_shapes(url):
    with pytest.raises(FetchError) as e:
        await validate_url(url, resolver=RESOLVE)
    assert e.value.blocked


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",  # resolves to loopback below
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://100.64.0.1/",  # carrier-grade NAT
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[fd00::1]/",
        "http://[fe80::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://224.0.0.1/",
        "http://internal.example.com/",  # DNS name that points inside
        "http://mixed.example.com/",  # one public + one private answer
    ],
)
async def test_blocks_private_and_internal_targets(url):
    resolve = resolver_for(
        {
            "localhost": ["127.0.0.1", "::1"],
            "internal.example.com": ["10.1.2.3"],
            "mixed.example.com": [PUBLIC, "192.168.0.9"],
        }
    )
    with pytest.raises(FetchError) as e:
        await validate_url(url, resolver=resolve)
    assert e.value.blocked
    assert "not allowed" in str(e.value)


async def test_public_hosts_pass_and_fragments_are_dropped():
    url = await validate_url("https://example.com/a/b?q=1#section", resolver=RESOLVE)
    assert str(url) == "https://example.com/a/b?q=1"


async def test_allow_private_skips_the_address_check():
    url = await validate_url("http://127.0.0.1:8080/x", allow_private=True)
    assert url.host == "127.0.0.1"


async def test_unresolvable_host_is_an_error():
    with pytest.raises(FetchError, match="resolve"):
        await validate_url("http://nope.invalid/", resolver=RESOLVE)


async def test_fetches_a_page():
    def handler(request):
        assert request.headers["user-agent"].startswith("AtlasBot")
        return httpx.Response(200, text="<html>hi</html>", headers={"content-type": "text/html"})

    got = await fetch_url(
        "https://example.com/docs/guide.html",
        max_bytes=1000,
        resolver=RESOLVE,
        transport=transport(handler),
    )
    assert got.data == b"<html>hi</html>"
    assert got.content_type == "text/html"
    assert got.filename == "guide.html"
    assert got.url == "https://example.com/docs/guide.html"


async def test_filename_from_content_disposition():
    def handler(request):
        return httpx.Response(
            200, content=b"x", headers={"content-disposition": 'attachment; filename="r%20q3.pdf"'}
        )

    got = await fetch_url(
        "https://example.com/download?id=7",
        max_bytes=1000,
        resolver=RESOLVE,
        transport=transport(handler),
    )
    assert got.filename == "r q3.pdf"


async def test_follows_redirects_and_revalidates_each_hop():
    def handler(request):
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "https://cdn.example.com/final.txt"})
        return httpx.Response(200, text="done")

    got = await fetch_url(
        "https://example.com/start",
        max_bytes=1000,
        resolver=RESOLVE,
        transport=transport(handler),
    )
    assert got.url == "https://cdn.example.com/final.txt"
    assert got.data == b"done"


async def test_redirect_to_an_internal_address_is_blocked():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    with pytest.raises(FetchError) as e:
        await fetch_url(
            "https://example.com/start",
            max_bytes=1000,
            resolver=RESOLVE,
            transport=transport(handler),
        )
    assert e.value.blocked


async def test_redirect_to_a_non_http_scheme_is_blocked():
    def handler(request):
        return httpx.Response(302, headers={"location": "file:///etc/passwd"})

    with pytest.raises(FetchError) as e:
        await fetch_url(
            "https://example.com/", max_bytes=1000, resolver=RESOLVE, transport=transport(handler)
        )
    assert e.value.blocked


async def test_redirect_loop_stops():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://example.com/again"})

    with pytest.raises(FetchError, match="redirects"):
        await fetch_url(
            "https://example.com/", max_bytes=1000, resolver=RESOLVE, transport=transport(handler)
        )


async def test_redirect_without_location_is_an_error():
    def handler(request):
        return httpx.Response(302)

    with pytest.raises(FetchError, match="redirect"):
        await fetch_url(
            "https://example.com/", max_bytes=1000, resolver=RESOLVE, transport=transport(handler)
        )


async def test_http_errors_are_reported():
    def handler(request):
        return httpx.Response(404)

    with pytest.raises(FetchError, match="404") as e:
        await fetch_url(
            "https://example.com/", max_bytes=1000, resolver=RESOLVE, transport=transport(handler)
        )
    assert not e.value.blocked


async def test_declared_oversize_is_refused_before_reading():
    def handler(request):
        return httpx.Response(200, content=b"x", headers={"content-length": "5000000"})

    with pytest.raises(FetchError, match="larger"):
        await fetch_url(
            "https://example.com/", max_bytes=1024, resolver=RESOLVE, transport=transport(handler)
        )


async def test_streamed_oversize_is_cut_off():
    def handler(request):
        return httpx.Response(200, content=b"y" * 5000)  # no honest content-length needed

    with pytest.raises(FetchError, match="larger"):
        await fetch_url(
            "https://example.com/", max_bytes=1024, resolver=RESOLVE, transport=transport(handler)
        )


async def test_network_errors_are_wrapped():
    def handler(request):
        raise httpx.ConnectError("boom")

    with pytest.raises(FetchError, match="ConnectError"):
        await fetch_url(
            "https://example.com/", max_bytes=1000, resolver=RESOLVE, transport=transport(handler)
        )


async def test_timeouts_are_wrapped():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(FetchError, match="too long"):
        await fetch_url(
            "https://example.com/", max_bytes=1000, resolver=RESOLVE, transport=transport(handler)
        )
