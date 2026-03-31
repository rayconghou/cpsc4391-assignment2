from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from vectordb.core import VectorDB


def test_query_empty_db_returns_empty(tmp_path: Path) -> None:
    db_path = tmp_path / "v.sqlite"
    db = VectorDB(db_path)
    try:
        assert db.query("anything", k=5) == []
    finally:
        db.close()


def test_k_non_positive_raises(tmp_path: Path) -> None:
    db = VectorDB(tmp_path / "v.sqlite")
    try:
        db.add("hello")
        with pytest.raises(ValueError):
            db.query("hello", k=0)
        with pytest.raises(ValueError):
            db.query("hello", k=-1)
    finally:
        db.close()


def test_add_query_ranking_and_k_cap(tmp_path: Path) -> None:
    db_path = tmp_path / "v.sqlite"
    db = VectorDB(
        db_path,
        max_features=100,
        min_df=1,
        num_tables=4,
        num_hyperplanes=6,
        random_seed=1,
    )
    try:
        db.add("the cat sat on the mat")
        db.add("dogs run in the park")
        db.add("the cat sleeps")
        r = db.query("cat and mat", k=2)
        assert len(r) == 2
        # Strongest match should be the cat/mat document
        assert "cat" in r[0].text.lower()
        r_all = db.query("cat", k=100)
        assert len(r_all) == 3
    finally:
        db.close()


def test_embedding_cache_survives_reopen(tmp_path: Path) -> None:
    """Vectors + fitted vectorizer persist in SQLite; reopen skips refit path."""
    db_path = tmp_path / "persist.sqlite"
    db = VectorDB(
        db_path,
        max_features=100,
        min_df=1,
        num_tables=4,
        num_hyperplanes=6,
        random_seed=1,
    )
    try:
        db.add("alpha beta")
        db.add("gamma delta")
        r1 = db.query("beta", k=2)
    finally:
        db.close()

    db2 = VectorDB(
        db_path,
        max_features=100,
        min_df=1,
        num_tables=4,
        num_hyperplanes=6,
        random_seed=1,
    )
    try:
        r2 = db2.query("beta", k=2)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2, strict=True):
            assert a.id == b.id and a.text == b.text
            assert abs(a.score - b.score) < 1e-9
    finally:
        db2.close()

    con = sqlite3.connect(db_path)
    try:
        assert con.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0] == 1
    finally:
        con.close()


def test_duplicate_texts_distinct_ids(tmp_path: Path) -> None:
    db = VectorDB(tmp_path / "d.sqlite")
    try:
        a = db.add("same")
        b = db.add("same")
        assert a != b
    finally:
        db.close()


def test_cli_add_query_json(tmp_path: Path) -> None:
    db_file = tmp_path / "cli.sqlite"
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    r1 = subprocess.run(
        [sys.executable, "-m", "vectordb", "add", "--db", str(db_file), "--quiet", "hello world"],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert r1.returncode == 0
    assert r1.stdout.strip().isdigit()

    r2 = subprocess.run(
        [
            sys.executable,
            "-m",
            "vectordb",
            "query",
            "--db",
            str(db_file),
            "-k",
            "1",
            "--json",
            "hello",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert r2.returncode == 0
    data = json.loads(r2.stdout)
    assert len(data) == 1
    assert data[0]["text"] == "hello world"
    assert "score" in data[0]
