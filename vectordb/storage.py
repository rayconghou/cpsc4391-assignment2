from __future__ import annotations

import sqlite3
from pathlib import Path


class DocumentStore:
    """SQLite-backed store: document text + persisted TF-IDF vectors (cache)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "corpus_fingerprint TEXT NOT NULL, "
            "x_norm BLOB NOT NULL, "
            "vectorizer BLOB NOT NULL)"
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add(self, text: str) -> int:
        cur = self._conn.execute("INSERT INTO documents (text) VALUES (?)", (text,))
        self._conn.commit()
        return int(cur.lastrowid)

    def load_all_ordered(self) -> list[tuple[int, str]]:
        cur = self._conn.execute("SELECT id, text FROM documents ORDER BY id")
        return [(int(r[0]), str(r[1])) for r in cur.fetchall()]

    def load_embedding_cache(self) -> tuple[str, bytes, bytes] | None:
        cur = self._conn.execute(
            "SELECT corpus_fingerprint, x_norm, vectorizer FROM embedding_cache WHERE id = 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        return (str(row[0]), bytes(row[1]), bytes(row[2]))

    def save_embedding_cache(
        self, corpus_fingerprint: str, x_norm_blob: bytes, vectorizer_blob: bytes
    ) -> None:
        self._conn.execute("DELETE FROM embedding_cache")
        self._conn.execute(
            "INSERT INTO embedding_cache (id, corpus_fingerprint, x_norm, vectorizer) "
            "VALUES (1, ?, ?, ?)",
            (corpus_fingerprint, x_norm_blob, vectorizer_blob),
        )
        self._conn.commit()

    def clear_embedding_cache(self) -> None:
        self._conn.execute("DELETE FROM embedding_cache")
        self._conn.commit()
