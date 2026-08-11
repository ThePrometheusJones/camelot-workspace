"""Cache duration helpers via src.search.query shim must work."""
import services.search.query as q


def test_is_news_query_handles_non_string():
    assert q._is_news_query(None) is False


def test_is_news_query_works():
    assert q._is_news_query("latest news today") is True
    assert q._is_news_query("python tutorial") is False
