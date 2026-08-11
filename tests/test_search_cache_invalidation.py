"""Regression test for invalidate_search_cache key construction.

The write path (`searxng_search_results`) stores a cache entry under
``generate_cache_key(f"{query}|{count}|{time_filter}")`` where ``count`` is the
admin-configured result count (``_get_result_count()``, default **5**) — it
replaces the caller's default of 10 with the configured value before building
the key.

The original ``invalidate_search_cache`` hardcoded ``f"{query}|10|None"``, so it
never matched the key the write path actually produced (``|5|None`` by default)
and silently failed to invalidate anything — a contract violation of its own
docstring ("invalidate ... just the given query"). The fix derives the count
from ``_get_result_count()`` so invalidation matches the stored default entry.
"""
import json
import pytest
from datetime import datetime, timedelta

from src.search import core
from src.search.cache import generate_cache_key, cleanup_cache


def test_invalidate_uses_configured_count_not_hardcoded_10(tmp_path, monkeypatch):
    query = "python tutorial"
    result_count = 5  # documented default of _get_result_count()

    # Pin the configured count and redirect the cache dir to keep the test hermetic.
    monkeypatch.setattr(core, "_get_result_count", lambda: result_count)
    monkeypatch.setattr(core, "SEARCH_CACHE_DIR", tmp_path)

    # Reproduce exactly what searxng_search_results writes for a default search:
    # the caller's default count of 10 is replaced by result_count, time_filter=None.
    write_key = generate_cache_key(f"{query}|{result_count}|None")
    cache_file = tmp_path / f"{write_key}.cache"
    cache_file.write_text("{}", encoding="utf-8")
    core.search_cache_index[write_key] = None

    try:
        core.invalidate_search_cache(query)

        assert not cache_file.exists(), (
            "invalidate_search_cache failed to remove the entry the write path "
            "stored under the configured result count — it used a mismatched key."
        )
        assert write_key not in core.search_cache_index
    finally:
        core.search_cache_index.pop(write_key, None)


def test_cleanup_honors_per_entry_expiry(tmp_path):
    """A reference-query cache entry (24h expiry) must survive a cleanup
    that would have killed it under the old blanket 1-hour max_age."""
    index = {}
    now = datetime.now()

    # Entry with 24h expiry, written 2 hours ago (old behavior would kill it)
    key_alive = "alive_key"
    cache_file_alive = tmp_path / f"{key_alive}.cache"
    cache_file_alive.write_text(json.dumps({
        "timestamp": (now - timedelta(hours=2)).isoformat(),
        "expiry": (now + timedelta(hours=22)).isoformat(),
        "data": [],
    }), encoding="utf-8")
    index[key_alive] = now - timedelta(hours=2)

    # Entry with expired expiry
    key_dead = "dead_key"
    cache_file_dead = tmp_path / f"{key_dead}.cache"
    cache_file_dead.write_text(json.dumps({
        "timestamp": (now - timedelta(hours=25)).isoformat(),
        "expiry": (now - timedelta(hours=1)).isoformat(),
        "data": [],
    }), encoding="utf-8")
    index[key_dead] = now - timedelta(hours=25)

    cleanup_cache(tmp_path, index, timedelta(hours=24))

    assert cache_file_alive.exists(), "Live 24h entry was wrongly reaped"
    assert key_alive in index
    assert not cache_file_dead.exists(), "Expired entry was not reaped"
    assert key_dead not in index


def test_cleanup_reaps_orphaned_expired_files(tmp_path):
    """Files on disk but not in the index (orphans from restart) should be
    reaped if their embedded expiry has passed."""
    index = {}
    now = datetime.now()

    orphan = tmp_path / "orphan_abc.cache"
    orphan.write_text(json.dumps({
        "timestamp": (now - timedelta(hours=3)).isoformat(),
        "expiry": (now - timedelta(hours=1)).isoformat(),
        "data": [],
    }), encoding="utf-8")

    # Orphan with future expiry — should be re-indexed, not deleted
    live_orphan = tmp_path / "live_orphan.cache"
    live_orphan.write_text(json.dumps({
        "timestamp": now.isoformat(),
        "expiry": (now + timedelta(hours=10)).isoformat(),
        "data": [],
    }), encoding="utf-8")

    cleanup_cache(tmp_path, index, timedelta(hours=24))

    assert not orphan.exists(), "Expired orphan was not reaped"
    assert live_orphan.exists(), "Live orphan was wrongly reaped"
    assert "live_orphan" in index, "Live orphan was not re-indexed"
