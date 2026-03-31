from __future__ import annotations

import sqlite3
from pathlib import Path


class DocumentStore:
    """SQLite-backed append-only text store (ids + raw text)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL)"
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
