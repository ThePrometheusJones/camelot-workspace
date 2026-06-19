"""Vault Recall — Tier 2 long-term memory provider for Camelot Workspace.

Read-only provider that queries Guinevere's Obsidian vault via the existing
vault_recall.db (sqlite-vec + nomic-embed-text embeddings). Writes are handled
by the meditation promotion path, not by this provider.

The DB lives at ~/.hermes/vault_recall.db and is populated by the Hermes
vault_recall plugin during indexing runs. This provider only reads from it.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import struct
from typing import Any, Dict, List, Optional

import requests

from src.memory_provider import MemoryProvider, MemoryRecord, MemorySearchHit

logger = logging.getLogger(__name__)

# ponytail: env vars only, no constructor overrides — these are the ones that get used
_DB_PATH = os.environ.get("VAULT_RECALL_DB", os.path.expanduser("~/.hermes/vault_recall.db"))
_OLLAMA_URL = os.environ.get("VAULT_RECALL_OLLAMA_URL", "http://localhost:11434")
_EMBED_MODEL = os.environ.get("VAULT_RECALL_EMBED_MODEL", "nomic-embed-text")
_TOP_K = int(os.environ.get("VAULT_RECALL_TOP_K", "5"))
_MAX_CONTEXT_CHARS = int(os.environ.get("VAULT_RECALL_MAX_CHARS", "2000"))


def _embed_query(query: str, ollama_url: str, model: str) -> list[float]:
    """Embed a single query via Ollama /api/embed."""
    resp = requests.post(
        f"{ollama_url}/api/embed",
        json={"model": model, "input": [f"search_query: {query}"]},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


def _vec_to_bytes(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


class VaultRecallProvider(MemoryProvider):
    """Read-only long-term memory from Guinevere's Obsidian vault."""

    provider_id = "vault_recall"
    display_name = "Obsidian Vault (long-term)"

    def __init__(self):
        self._db: Optional[sqlite3.Connection] = None
        self._available = False

    async def initialize(self) -> None:
        if not os.path.exists(_DB_PATH):
            logger.warning("vault_recall: DB not found at %s", _DB_PATH)
            return

        try:
            import sqlite_vec
        except ImportError:
            logger.warning("vault_recall: sqlite-vec not installed")
            return

        try:
            self._db = sqlite3.connect(_DB_PATH, check_same_thread=False)
            self._db.enable_load_extension(True)
            sqlite_vec.load(self._db)
            self._db.enable_load_extension(False)

            chunk_count = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            file_count = self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            self._available = chunk_count > 0

            logger.info(
                "vault_recall: connected — %d files, %d chunks",
                file_count, chunk_count,
            )
        except Exception as e:
            logger.warning("vault_recall: init failed: %s", e)
            self._available = False

    async def shutdown(self) -> None:
        if self._db:
            self._db.close()
            self._db = None
            self._available = False

    def _search(self, query: str, top_k: int = 5) -> list[dict]:
        """Vector similarity search against vault chunks."""
        if not self._db or not self._available:
            return []

        try:
            query_emb = _embed_query(query, _OLLAMA_URL, _EMBED_MODEL)
        except Exception as e:
            logger.debug("vault_recall: embed failed: %s", e)
            return []

        query_bytes = _vec_to_bytes(query_emb)

        try:
            rows = self._db.execute(
                """
                SELECT c.content, f.path, cv.distance
                FROM chunks_vec cv
                JOIN chunks c ON c.id = cv.rowid
                JOIN files f ON f.id = c.file_id
                WHERE cv.embedding MATCH ? AND k = ?
                ORDER BY cv.distance
                """,
                (query_bytes, top_k),
            ).fetchall()
        except Exception as e:
            logger.debug("vault_recall: search failed: %s", e)
            return []

        return [
            {"content": r[0], "file": r[1], "distance": r[2]}
            for r in rows
        ]

    # -- MemoryProvider interface --

    async def recall(
        self,
        query: str,
        *,
        owner: Optional[str] = None,
        top_k: int = 5,
    ) -> List[MemorySearchHit]:
        results = self._search(query, min(top_k, _TOP_K))
        hits = []
        for r in results:
            # Convert cosine distance to similarity score (0-1)
            score = max(0.0, 1.0 - r["distance"])
            hits.append(
                MemorySearchHit(
                    memory=MemoryRecord(
                        id=f"vault:{r['file']}",
                        text=r["content"],
                        category="vault",
                        source="obsidian",
                        metadata={"file": r["file"], "distance": r["distance"]},
                    ),
                    provider_id=self.provider_id,
                    score=score,
                )
            )
        return hits

    # ponytail: ABC requires these stubs — vault is read-only, writes go through meditation
    async def remember(self, text, *, owner=None, session_id=None, category="fact", source="user", metadata=None):
        raise NotImplementedError("vault_recall is read-only")

    async def list_memories(self, *, owner=None, limit=100):
        return []

    async def delete(self, memory_id, *, owner=None):
        return False

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._available:
            return []
        return [
            {
                "name": "vault_search",
                "description": (
                    "Search Guinevere's long-term memory vault for facts, decisions, "
                    "identity, project context, and knowledge from past sessions. "
                    "Use when the question references history, people, projects, or "
                    "decisions that may be stored in the Obsidian vault."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in the memory vault.",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name != "vault_search":
            raise KeyError(f"vault_recall does not expose tool {name}")

        query = arguments.get("query", "")
        if not query:
            return {"results": [], "note": "Empty query."}

        results = self._search(query, self._top_k)
        if not results:
            return {"results": [], "note": "No matching memories found in the vault."}

        return {
            "results": [
                {
                    "file": r["file"],
                    "content": r["content"][:500],
                    "similarity": round(1.0 - r["distance"], 4),
                }
                for r in results
            ]
        }
