"""Search and content caching with LRU eviction."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from core.constants import DATA_DIR

logger = logging.getLogger(__name__)

# Cache directories
CACHE_DIR = Path(DATA_DIR) / "cache"
SEARCH_CACHE_DIR = CACHE_DIR / "search"
CONTENT_CACHE_DIR = CACHE_DIR / "content"
CACHE_MAX_ENTRIES = 1000

# Create cache directories. Guarded so an unwritable path (e.g. a read-only
# mount) degrades to no-disk-cache instead of crashing module import.
try:
    SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CONTENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
except OSError as _e:
    logger.warning("Search cache directory unavailable (%s); disk cache disabled", _e)

# Track cache size for LRU eviction
search_cache_index: Dict[str, datetime] = {}
content_cache_index: Dict[str, datetime] = {}

# Cache metrics (shared across modules)
cache_metrics = {"hits": 0, "misses": 0, "evictions": 0}


def generate_cache_key(data: str) -> str:
    """Generate a unique cache key using SHA-256 hash."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _read_expiry(cache_file: Path) -> datetime | None:
    """Read the expiry timestamp from a cache file's JSON. Returns None on failure."""
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("expiry")
        return datetime.fromisoformat(raw) if raw else None
    except Exception:
        return None


def cleanup_cache(cache_dir: Path, cache_index: Dict[str, datetime], max_age: timedelta):
    """Remove expired cache entries (honoring per-entry expiry) and enforce LRU.

    Per-entry expiry is read from the cache file's JSON ``expiry`` field.
    ``max_age`` is used as a ceiling for entries without an embedded expiry.
    Orphaned files (on disk but not in the index) are also reaped if expired.
    """
    now = datetime.now()
    files_in_dir = {f.stem: f for f in cache_dir.glob("*.cache")}

    # Phase 1: check indexed entries
    to_remove = []
    for key, timestamp in list(cache_index.items()):
        if key not in files_in_dir:
            to_remove.append(key)
            continue
        expiry = _read_expiry(files_in_dir[key])
        if expiry and now > expiry:
            to_remove.append(key)
            files_in_dir[key].unlink(missing_ok=True)
        elif not expiry and now - timestamp > max_age:
            # No embedded expiry — fall back to index timestamp + max_age
            to_remove.append(key)
            files_in_dir[key].unlink(missing_ok=True)

    for key in to_remove:
        cache_index.pop(key, None)
        cache_metrics["evictions"] += 1

    # Phase 2: reap orphans (files not in the index, e.g. after restart)
    indexed_keys = set(cache_index.keys())
    for key, filepath in files_in_dir.items():
        if key in indexed_keys:
            continue
        expiry = _read_expiry(filepath)
        if expiry and now > expiry:
            filepath.unlink(missing_ok=True)
            cache_metrics["evictions"] += 1
        elif not expiry:
            # No expiry and not indexed — stale orphan, remove
            filepath.unlink(missing_ok=True)
            cache_metrics["evictions"] += 1
        # else: orphan with future expiry — re-index it
        else:
            cache_index[key] = now

    # Phase 3: LRU enforcement
    if len(cache_index) > CACHE_MAX_ENTRIES:
        sorted_items = sorted(cache_index.items(), key=lambda x: x[1])
        excess_count = len(cache_index) - CACHE_MAX_ENTRIES
        for key, _ in sorted_items[:excess_count]:
            cache_index.pop(key, None)
            cache_file = cache_dir / f"{key}.cache"
            cache_file.unlink(missing_ok=True)
            cache_metrics["evictions"] += 1
