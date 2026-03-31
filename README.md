# Vector database (CPSC 4391 Assignment 2)

Local **TF-IDF** embeddings (L2-normalized), **random hyperplane LSH** for approximate candidate retrieval, **cosine similarity** (dot product on unit vectors) for reranking, and **SQLite** for persistent raw text + integer ids. The **fitted TF-IDF vectorizer** and **L2-normalized document matrix** are also **persisted** in SQLite (`embedding_cache`); on each **add** or **query** the corpus is checked and embeddings are **recomputed only** when the text or TF-IDF settings no longer match the cache (the LSH tables are always **rebuilt in memory** from the matrix).

## Requirements

- Python 3.10+ recommended (tested on 3.13)

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest
```

## Python API

```python
from pathlib import Path
from vectordb import VectorDB

db = VectorDB(Path("store.sqlite"), random_seed=0, num_tables=10, num_hyperplanes=16)
doc_id = db.add("the cat sat on the mat")
results = db.query("cat on mat", k=3)
for r in results:
    print(r.id, r.score, r.text)
db.close()
```

- `k <= 0` raises `ValueError`.
- If there are no documents, `query` returns `[]`.
- If `k` is larger than the number of stored documents, all documents are returned, ranked.

## CLI

Use the **same** `--seed`, `--lsh-tables`, `--lsh-bits`, and `--max-features` for `add` and `query` against one database so TF-IDF and LSH stay aligned.

```bash
python -m vectordb add --db ./data.sqlite "hello world"
python -m vectordb query --db ./data.sqlite -k 5 "hello"
python -m vectordb add --db ./data.sqlite --quiet "another doc"    # prints only new id
python -m vectordb query --db ./data.sqlite -k 3 --json "hello"
```

## Layout

- `vectordb/core.py` — `VectorDB` orchestration
- `vectordb/lsh.py` — hyperplane LSH
- `vectordb/storage.py` — SQLite document table + embedding cache (vector matrix + vectorizer blobs)
- `vectordb/cli.py` — `add` / `query` subcommands

## Submission zip

Include source and `requirements.txt`. Do **not** bundle large third-party wheels; document `pip install -r requirements.txt` instead.
