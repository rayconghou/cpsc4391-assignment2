from __future__ import annotations

import numpy as np
from numpy.random import RandomState
from scipy import sparse


def _signature_bits(
    vectors: sparse.csr_matrix, hyperplanes: np.ndarray
) -> np.ndarray:
    """
    vectors: (n_docs, n_features) CSR float
    hyperplanes: (n_hyperplanes, n_features) float64
    returns bits (n_docs, n_hyperplanes) int8 in {0,1}
    """
    dots = vectors @ hyperplanes.T
    return (dots >= 0).astype(np.int8)


class HyperplaneLSH:
    """
    Random hyperplane LSH for cosine similarity on L2-normalized vectors
    (dot product equals cosine). Multiple independent hash tables reduce
    false negatives.
    """

    def __init__(
        self,
        num_tables: int,
        num_hyperplanes: int,
        rng: RandomState,
    ) -> None:
        if num_tables < 1:
            raise ValueError("num_tables must be >= 1")
        if num_hyperplanes < 1:
            raise ValueError("num_hyperplanes must be >= 1")
        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes
        self._rng = rng
        self._hyperplanes: np.ndarray | None = None
        # buckets[table_idx][signature_tuple] -> list of row indices
        self._buckets: list[dict[tuple[int, ...], list[int]]] = []

    def fit(self, X_norm: sparse.csr_matrix) -> None:
        """Build hash tables for row vectors (already L2-normalized)."""
        n_docs, n_features = X_norm.shape
        if n_docs == 0:
            self._hyperplanes = None
            self._buckets = []
            return

        self._hyperplanes = self._rng.randn(
            self.num_tables, self.num_hyperplanes, n_features
        ).astype(np.float64, copy=False)

        self._buckets = []
        for t in range(self.num_tables):
            bits = _signature_bits(X_norm, self._hyperplanes[t])
            table: dict[tuple[int, ...], list[int]] = {}
            for i in range(n_docs):
                key = tuple(int(x) for x in bits[i].tolist())
                table.setdefault(key, []).append(i)
            self._buckets.append(table)

    def query_bucket_indices(self, q_norm: sparse.csr_matrix) -> set[int]:
        """
        Return the union of row indices that share at least one bucket
        with q_norm across all tables (same signature per table).
        """
        if self._hyperplanes is None or q_norm.shape[0] != 1:
            return set()

        candidates: set[int] = set()
        for t in range(self.num_tables):
            hp = self._hyperplanes[t]
            dots = q_norm @ hp.T
            flat = np.asarray(dots >= 0).astype(np.int8).ravel()
            key = tuple(int(x) for x in flat.tolist())
            for idx in self._buckets[t].get(key, []):
                candidates.add(idx)
        return candidates
