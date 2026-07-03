"""Test that _get_public_url blocks 302 redirects to internal targets.

Covers the redirect-bypass gap: a public URL 302s to localhost/tailnet and
the guard must catch it on the second hop, not just the first.
"""

import ipaddress
import httpx
import pytest
from unittest.mock import patch

from services.search.content import _get_public_url, _is_private_address


# --- Tailnet range coverage ---

def test_tailnet_cgnat_range_is_private():
    """100.64.0.0/10 (CGNAT / Tailscale) must be treated as private."""
    assert _is_private_address(ipaddress.ip_address("100.118.94.13"))  # camelot
    assert _is_private_address(ipaddress.ip_address("100.84.12.22"))   # optiplex
    assert _is_private_address(ipaddress.ip_address("100.64.0.1"))     # bottom of range
    assert _is_private_address(ipaddress.ip_address("100.127.255.254"))  # top of range


# --- 302 redirect to internal targets ---

def _mock_response(status_code, url, headers=None):
    req = httpx.Request("GET", url)
    return httpx.Response(status_code, request=req, headers=headers or {})


def _public_ip_resolver(host):
    """Return a public IP for any hostname so _public_http_url passes hop 1."""
    return [ipaddress.ip_address("93.184.216.34")]


def test_302_to_localhost_blocked():
    """Public URL that 302s to localhost:8100 (ChromaDB) must be blocked."""
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return _mock_response(302, url, {"location": "http://localhost:8100/api/v1/collections"})
        return _mock_response(200, url)

    with patch("services.search.content.httpx.get", side_effect=fake_get), \
         patch("services.search.content._resolve_hostname_ips", side_effect=_public_ip_resolver):
        with pytest.raises(httpx.RequestError, match="Blocked private"):
            _get_public_url("https://attacker.example.com/redirect", headers={}, timeout=5)

    assert len(calls) == 1  # stopped after first hop, never fetched localhost


def test_302_to_tailnet_blocked():
    """Public URL that 302s to a tailnet IP must be blocked."""
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return _mock_response(302, url, {"location": "http://100.118.94.13:7000/"})
        return _mock_response(200, url)

    with patch("services.search.content.httpx.get", side_effect=fake_get), \
         patch("services.search.content._resolve_hostname_ips", side_effect=_public_ip_resolver):
        with pytest.raises(httpx.RequestError, match="Blocked private"):
            _get_public_url("https://attacker.example.com/redir2", headers={}, timeout=5)

    assert len(calls) == 1


def test_302_chain_public_to_public_allowed():
    """Public URL that 302s to another public URL should succeed."""
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        if len(calls) == 1:
            return _mock_response(302, url, {"location": "https://www.example.com/final"})
        return _mock_response(200, url)

    with patch("services.search.content.httpx.get", side_effect=fake_get), \
         patch("services.search.content._resolve_hostname_ips", side_effect=_public_ip_resolver):
        resp = _get_public_url("https://example.com/start", headers={}, timeout=5)

    assert resp.status_code == 200
    assert len(calls) == 2
