"""Regression: cache duration helpers must tolerate a non-string query."""
import importlib.machinery
import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "services" / "search" / "query.py"


def _load():
    loader = importlib.machinery.SourceFileLoader("odysseus_search_query", str(_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def test_is_news_query_handles_none():
    q = _load()
    assert q._is_news_query(None) is False


def test_is_news_query_detects_news():
    q = _load()
    assert q._is_news_query("latest news today") is True
    assert q._is_news_query("python tutorial") is False
