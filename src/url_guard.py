"""
url_guard.py — SSRF guard for The Bailey's agent fetch tools.

Threat model: Guinevere fetches untrusted web content. A prompt injection in
that content can direct her fetch tool at internal services that trust
localhost implicitly — ChromaDB (:8100, her own memory, deletable via REST),
Ollama (:11434), SearXNG (:8889), llama-server (:8080), Fish Speech (:7300),
and every node on the tailnet (100.64.0.0/10), including the NAS.

The trust boundary is not Guinevere. It is the content she reads.

Usage — in the fetch tool handler(s) in tool_implementations.py, before any
HTTP request is made:

    from src.url_guard import assert_url_allowed, GuardedURLError

    try:
        assert_url_allowed(url)
    except GuardedURLError as e:
        return {"error": f"URL blocked by policy: {e}"}

IMPORTANT — redirects: httpx/requests follow redirects by default, and a
public URL can 302 to http://localhost:8100/. Two options:
  a) disable redirects and re-validate each Location hop manually, or
  b) use httpx with follow_redirects=False in a loop calling
     assert_url_allowed() on every hop (helper provided: guarded_fetch_url).
Do NOT validate once and then follow redirects blind.

IMPORTANT — DNS rebinding: we resolve here and check every A/AAAA record,
but the HTTP client resolves again. For a homelab threat model this is an
acceptable residual risk; closing it fully means pinning the connection to
the vetted IP (httpx transport with a custom resolver). Noted, not done.

Pure stdlib. No dependencies.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

__all__ = ["assert_url_allowed", "GuardedURLError", "next_hop_or_none"]


class GuardedURLError(Exception):
    """Raised when a URL fails SSRF policy."""


ALLOWED_SCHEMES = {"http", "https"}

# Hostnames rejected before DNS is even consulted.
BLOCKED_HOSTNAMES = {
    "localhost",
    "camelot",            # this box
    "the-forge",
    "sauron",
    "watchtower",
    "optiplex",           # add/adjust to match your actual hostnames
    "metadata.google.internal",
}

BLOCKED_NETWORKS = [
    ipaddress.ip_network(n)
    for n in (
        "127.0.0.0/8",        # loopback
        "10.0.0.0/8",         # RFC1918
        "172.16.0.0/12",      # RFC1918
        "192.168.0.0/16",     # RFC1918 (the lab LAN lives here)
        "100.64.0.0/10",      # CGNAT — the entire tailnet
        "169.254.0.0/16",     # link-local / cloud metadata
        "0.0.0.0/8",
        "224.0.0.0/4",        # multicast
        "240.0.0.0/4",        # reserved
        "::1/128",            # v6 loopback
        "fc00::/7",           # v6 ULA
        "fe80::/10",          # v6 link-local
        "::ffff:0:0/96",      # v4-mapped v6 (re-checked below, but belt+suspenders)
    )
]


def _ip_blocked(ip: ipaddress._BaseAddress) -> bool:
    # Unwrap v4-mapped v6 so ::ffff:127.0.0.1 is judged as 127.0.0.1
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return any(ip in net for net in BLOCKED_NETWORKS)


def assert_url_allowed(url: str) -> None:
    """Raise GuardedURLError unless `url` is a public http(s) target.

    Checks scheme, hostname denylist, literal-IP ranges, and every address
    the hostname resolves to. Returns None on success.
    """
    try:
        parsed = urlparse(url)
    except ValueError as e:
        raise GuardedURLError(f"unparseable URL: {e}") from e

    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise GuardedURLError(f"scheme '{parsed.scheme}' not allowed")

    host = parsed.hostname
    if not host:
        raise GuardedURLError("no hostname in URL")

    if host.lower().rstrip(".") in BLOCKED_HOSTNAMES:
        raise GuardedURLError(f"hostname '{host}' is on the denylist")

    # Literal IP in the URL?
    try:
        ip = ipaddress.ip_address(host)
        if _ip_blocked(ip):
            raise GuardedURLError(f"IP {ip} is in a blocked range")
        return  # public literal IP — allowed
    except ValueError:
        pass  # not a literal IP; resolve it

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise GuardedURLError(f"DNS resolution failed for '{host}': {e}") from e

    for info in infos:
        addr = ipaddress.ip_address(info[4][0])
        if _ip_blocked(addr):
            raise GuardedURLError(
                f"'{host}' resolves to blocked address {addr}"
            )


def next_hop_or_none(response) -> str | None:
    """Given an httpx/requests response with follow_redirects disabled,
    return the absolute redirect target to re-validate, or None."""
    if response.status_code in (301, 302, 303, 307, 308):
        loc = response.headers.get("location")
        if loc:
            from urllib.parse import urljoin
            return urljoin(str(response.url), loc)
    return None


# --- Optional convenience: fully guarded fetch with manual redirect walk ---
# Uncomment if you want a ready-made replacement for raw client.get(url).
#
# import httpx
#
# def guarded_fetch_url(url: str, *, max_redirects: int = 5, **kw) -> "httpx.Response":
#     for _ in range(max_redirects + 1):
#         assert_url_allowed(url)
#         resp = httpx.get(url, follow_redirects=False, **kw)
#         nxt = next_hop_or_none(resp)
#         if nxt is None:
#             return resp
#         url = nxt
#     raise GuardedURLError("too many redirects")


if __name__ == "__main__":
    # Smoke tests
    cases = [
        ("https://example.com", True),
        ("http://localhost:8100/api/v1/collections", False),
        ("http://127.0.0.1:11434/api/embed", False),
        ("http://192.168.0.169/", False),          # NAS
        ("http://100.79.19.107/", False),          # NAS via tailnet
        ("http://100.118.94.13:7000/", False),     # The Bailey itself
        ("http://169.254.169.254/latest/meta-data", False),
        ("http://[::1]:8080/", False),
        ("http://[::ffff:127.0.0.1]:8100/", False),
        ("ftp://example.com/file", False),
        ("http://camelot:7000/", False),
    ]
    failures = 0
    for url, should_pass in cases:
        try:
            assert_url_allowed(url)
            ok = should_pass
        except GuardedURLError:
            ok = not should_pass
        print(f"{'PASS' if ok else 'FAIL'}  {url}")
        failures += 0 if ok else 1
    raise SystemExit(failures)
