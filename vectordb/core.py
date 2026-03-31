from __future__ import annotations

import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
from numpy.random import RandomState
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from vectordb.lsh import HyperplaneLSH
from vectordb.models import QueryResult
from vectordb.storage import DocumentStore


def _csr_to_bytes(x: sparse.csr_matrix) -> bytes:
    buf = io.BytesIO()
    sparse.save_npz(buf, x, compressed=True)
    return buf.getvalue()


def _csr_from_bytes(data: bytes) -> sparse.csr_matrix:
    return sparse.load_npz(io.BytesIO(data))


class VectorDB:
    """
    TF-IDF embeddings, L2-normalized; LSH for candidate generation;
    cosine (dot on unit vectors) for reranking. SQLite stores document text
    and a persisted cache of the fitted vectorizer plus L2-normalized sparse
    document matrix; the cache is invalidated when the corpus or TF-IDF
    settings no longer match.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        max_features: int | None = 5000,
        min_df: int | float = 1,
        max_df: float = 1.0,
        ngram_range: tuple[int, int] = (1, 1),
        num_tables: int = 10,
        num_hyperplanes: int = 16,
        random_seed: int = 0,
    ) -> None:
        self._store = DocumentStore(db_path)
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
        )
        self._rng = RandomState(random_seed)
        self._num_tables = num_tables
        self._num_hyperplanes = num_hyperplanes
        self._seed = random_seed

        self._X_norm: sparse.csr_matrix | None = None
        self._row_doc_ids: np.ndarray | None = None
        self._row_texts: list[str] | None = None
        self._lsh: HyperplaneLSH | None = None

    def close(self) -> None:
        self._store.close()

    def _corpus_fingerprint(self, rows: list[tuple[int, str]]) -> str:
        v = self._vectorizer
        payload = {
            "rows": rows,
            "tfidf": {
                "max_features": v.max_features,
                "min_df": v.min_df,
                "max_df": v.max_df,
                "ngram_range": list(v.ngram_range),
            },
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _rebuild(self) -> None:
        rows = self._store.load_all_ordered()
        if not rows:
            self._store.clear_embedding_cache()
            self._X_norm = None
            self._row_doc_ids = None
            self._row_texts = None
            self._lsh = None
            return

        doc_ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        fp = self._corpus_fingerprint(rows)

        cached = self._store.load_embedding_cache()
        if cached is not None:
            stored_fp, x_blob, vec_blob = cached
            if stored_fp == fp:
                try:
                    X_norm = _csr_from_bytes(x_blob)
                    loaded = pickle.loads(vec_blob)
                    if not isinstance(loaded, TfidfVectorizer):
                        raise TypeError("cached vectorizer has wrong type")
                    self._vectorizer = loaded
                    if not sparse.isspmatrix_csr(X_norm):
                        X_norm = X_norm.tocsr()
                    self._X_norm = X_norm
                    self._row_doc_ids = np.asarray(doc_ids, dtype=np.int64)
                    self._row_texts = texts
                    lsh = HyperplaneLSH(
                        self._num_tables,
                        self._num_hyperplanes,
                        RandomState(self._seed),
                    )
                    lsh.fit(self._X_norm)
                    self._lsh = lsh
                    return
                except Exception:
                    pass

        X = self._vectorizer.fit_transform(texts)
        X_norm = normalize(X, norm="l2", copy=True)
        if not sparse.isspmatrix_csr(X_norm):
            X_norm = X_norm.tocsr()

        self._X_norm = X_norm
        self._row_doc_ids = np.asarray(doc_ids, dtype=np.int64)
        self._row_texts = texts

        lsh = HyperplaneLSH(
            self._num_tables,
            self._num_hyperplanes,
            RandomState(self._seed),
        )
        lsh.fit(X_norm)
        self._lsh = lsh

        try:
            self._store.save_embedding_cache(
                fp,
                _csr_to_bytes(X_norm),
                pickle.dumps(self._vectorizer, protocol=pickle.HIGHEST_PROTOCOL),
            )
        except Exception:
            pass

    def add(self, text: str) -> int:
        doc_id = self._store.add(text)
        self._rebuild()
        return doc_id

    def query(self, query_text: str, k: int) -> list[QueryResult]:
        if k <= 0:
            raise ValueError("k must be a positive integer")

        self._rebuild()
        if self._X_norm is None or self._row_doc_ids is None or self._row_texts is None:
            return []

        n_docs = self._X_norm.shape[0]
        if n_docs == 0:
            return []

        q = self._vectorizer.transform([query_text])
        q_norm = normalize(q, norm="l2", copy=True)
        if not sparse.isspmatrix_csr(q_norm):
            q_norm = q_norm.tocsr()

        assert self._lsh is not None
        candidates = self._lsh.query_bucket_indices(q_norm)
        # Fallback: if we do not have enough distinct candidates, score everyone.
        if len(candidates) < min(k, n_docs):
            candidates = set(range(n_docs))

        idx_list = sorted(candidates)
        # Batch similarity: dot product on unit vectors == cosine similarity
        raw_sim = self._X_norm[idx_list] @ q_norm.T
        if sparse.issparse(raw_sim):
            sims = raw_sim.toarray().ravel()
        else:
            sims = np.asarray(raw_sim, dtype=np.float64).ravel()

        order = sorted(
            range(len(idx_list)),
            key=lambda j: (-float(sims[j]), int(self._row_doc_ids[idx_list[j]])),
        )

        take = min(k, n_docs)
        out: list[QueryResult] = []
        for j in order[:take]:
            row = idx_list[j]
            out.append(
                QueryResult(
                    id=int(self._row_doc_ids[row]),
                    text=self._row_texts[row],
                    score=float(sims[j]),
                )
            )
        return out
