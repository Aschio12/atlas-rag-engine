"""Fetching a user-supplied URL safely (Phase 2.1).

A server that fetches arbitrary URLs is an SSRF vector: without care a user can make it read
cloud metadata endpoints (169.254.169.254), localhost admin ports or private-network services.
Defences, all enforced here:

* only http/https, no credentials in the URL;
* the host is resolved and **every** address must be globally routable (loopback, private,
  link-local, multicast, reserved and shared ranges are refused);
* redirects are followed manually (max 5) and each hop is re-validated;
* the body is streamed with a hard size cap and an overall timeout;
* proxies/netrc from the environment are ignored.

Known limit: validation and connection are two DNS lookups, so a hostile DNS server could
answer differently the second time (DNS rebinding). For hostile multi-tenant deployments also
route this service's egress through a filtering proxy (see Phase 10 hardening).
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

import httpx

MAX_REDIRECTS = 5
USER_AGENT = "AtlasBot/0.1 (+https://github.com/Aschio12/atlas-rag-engine)"

Resolver = Callable[[str, int], Awaitable[list[str]]]


class FetchError(Exception):
    """The URL cannot be fetched. `blocked` means it was refused by policy, not a network fault."""

    def __init__(self, message: str, *, blocked: bool = False):
        super().__init__(message)
        self.blocked = blocked


@dataclass(frozen=True)
class Fetched:
    url: str  # final URL after redirects, without fragment
    data: bytes
    content_type: str | None
    filename: str | None


async def system_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise FetchError(f"Could not resolve host '{host}'") from None
    return sorted({info[4][0] for info in infos})


def _is_public(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


async def validate_url(
    url: str, *, allow_private: bool = False, resolver: Resolver = system_resolver
) -> httpx.URL:
    if len(url) > 2000:
        raise FetchError("The URL is too long", blocked=True)
    try:
        parsed = urlsplit(url.strip())
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        raise FetchError("The URL is not valid", blocked=True) from None
    if parsed.scheme not in ("http", "https") or not host:
        raise FetchError("Only http:// and https:// URLs are supported", blocked=True)
    if parsed.username or parsed.password:
        raise FetchError("URLs with embedded credentials are not allowed", blocked=True)
    port = port or (443 if parsed.scheme == "https" else 80)

    if not allow_private:
        try:
            addresses = [str(ipaddress.ip_address(host))]  # an IP literal needs no lookup
        except ValueError:
            addresses = await resolver(host, port)
        if not addresses or not all(_is_public(a) for a in addresses):
            raise FetchError(
                "That address is not allowed: it points to a private or internal network",
                blocked=True,
            )
    return httpx.URL(url.strip()).copy_with(fragment=None)


_DISPOSITION_NAME = re.compile(r"""filename\*?=(?:UTF-8'')?"?([^";]+)"?""", re.IGNORECASE)


def _filename(url: httpx.URL, disposition: str | None) -> str | None:
    if disposition and (m := _DISPOSITION_NAME.search(disposition)):
        return unquote(m.group(1)).strip() or None
    last = url.path.rstrip("/").rsplit("/", 1)[-1]
    return unquote(last) or None


async def fetch_url(
    url: str,
    *,
    max_bytes: int,
    timeout_s: float = 15.0,
    allow_private: bool = False,
    resolver: Resolver = system_resolver,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Fetched:
    try:
        return await asyncio.wait_for(
            _fetch(url, max_bytes, timeout_s, allow_private, resolver, transport),
            timeout=timeout_s * 2,
        )
    except TimeoutError:
        raise FetchError("The remote server took too long to respond") from None


async def _fetch(url, max_bytes, timeout_s, allow_private, resolver, transport) -> Fetched:
    current = await validate_url(url, allow_private=allow_private, resolver=resolver)
    async with httpx.AsyncClient(
        timeout=timeout_s,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/*,application/pdf,*/*;q=0.5"},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            try:
                async with client.stream("GET", current) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise FetchError("The remote server sent an invalid redirect")
                        target = str(current.join(location))
                        current = await validate_url(
                            target, allow_private=allow_private, resolver=resolver
                        )
                        continue
                    if resp.status_code >= 400:
                        raise FetchError(f"The remote server answered with HTTP {resp.status_code}")
                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > max_bytes:
                        raise FetchError(_too_big(max_bytes))
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf += chunk
                        if len(buf) > max_bytes:
                            raise FetchError(_too_big(max_bytes))
                    return Fetched(
                        url=str(current),
                        data=bytes(buf),
                        content_type=resp.headers.get("content-type"),
                        filename=_filename(current, resp.headers.get("content-disposition")),
                    )
            except httpx.TimeoutException:
                raise FetchError("The remote server took too long to respond") from None
            except httpx.HTTPError as e:
                raise FetchError(f"Could not fetch the URL ({type(e).__name__})") from None
        raise FetchError(f"Too many redirects (more than {MAX_REDIRECTS})")


def _too_big(limit: int) -> str:
    return f"The remote file is larger than the {limit // (1024 * 1024)} MB limit"
