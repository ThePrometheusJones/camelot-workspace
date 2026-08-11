"""Cache duration helpers for the search pipeline."""

import re
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Cache duration helpers
# ----------------------------------------------------------------------
def _is_news_query(query: str) -> bool:
    """Lightweight heuristic to decide if a query is news-oriented."""
    news_terms = {"news", "latest", "breaking", "today", "today's", "current", "updates", "happening"}
    if not isinstance(query, str):
        return False
    tokens = set(re.findall(r"\b\w+\b", query.lower()))
    return bool(tokens & news_terms)


def _cache_duration_for_query(query: str) -> timedelta:
    """News queries -> 30 minutes, reference queries -> 24 hours."""
    if _is_news_query(query):
        return timedelta(minutes=30)
    return timedelta(hours=24)
