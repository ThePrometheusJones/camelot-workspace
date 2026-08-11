"""Semantic reranking: cosine similarity dominates, keyword fallback works."""
import numpy as np
import services.search.ranking as ranking


class _FakeEmbedder:
    """Deterministic embedder: returns a vector where the first dimension
    encodes the position in the input list, making cosine similarity
    controllable by test data."""

    def __init__(self, vectors):
        self._vectors = vectors

    def encode(self, texts, normalize_embeddings=True):
        vecs = np.array(self._vectors[: len(texts)], dtype="float32")
        if normalize_embeddings and vecs.size > 0:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            vecs = vecs / norms
        return vecs


def test_semantic_rank_prefers_paraphrase_over_keyword_stuffed(monkeypatch):
    # Query vector = [1, 0]. Result A is similar ([0.9, 0.1]), B is orthogonal ([0.1, 0.9]).
    fake = _FakeEmbedder([
        [1.0, 0.0],   # query
        [0.9, 0.1],   # result A (relevant paraphrase)
        [0.1, 0.9],   # result B (off-topic keyword match)
    ])
    monkeypatch.setattr(ranking, "_get_embedder", lambda: fake)

    results = [
        # A first in input — gets vector [0.9, 0.1] (high cosine with query)
        {"title": "Expert guide to layer offloading", "snippet": "how to split model layers across devices",
         "url": "https://example.com/a"},
        # B second — gets vector [0.1, 0.9] (low cosine with query)
        {"title": "Off-topic keyword match", "snippet": "gpu offload flags mentioned here",
         "url": "https://example.com/b"},
    ]
    ranked = ranking.rank_search_results("llama.cpp expert offload flags", results)
    # Semantic similarity should rank A above B despite B having more keyword overlap
    assert ranked[0]["url"] == "https://example.com/a"


def test_fallback_to_keyword_when_embedder_unavailable(monkeypatch):
    monkeypatch.setattr(ranking, "_get_embedder", lambda: None)

    results = [
        {"title": "Python tutorial", "snippet": "learn python basics", "url": "https://example.com/a"},
        {"title": "Java tutorial", "snippet": "learn java basics", "url": "https://example.com/b"},
    ]
    ranked = ranking.rank_search_results("python tutorial", results)
    # Keyword ranker should prefer the result with matching title terms
    assert ranked[0]["url"] == "https://example.com/a"


def test_fallback_on_embedder_exception(monkeypatch):
    class _BrokenEmbedder:
        def encode(self, *a, **kw):
            raise RuntimeError("GPU on fire")

    monkeypatch.setattr(ranking, "_get_embedder", lambda: _BrokenEmbedder())

    results = [
        {"title": "Python tutorial", "snippet": "learn python", "url": "https://example.com/a"},
    ]
    ranked = ranking.rank_search_results("python tutorial", results)
    assert len(ranked) == 1
