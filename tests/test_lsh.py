from __future__ import annotations

import numpy as np
from numpy.random import RandomState
from scipy import sparse
from sklearn.preprocessing import normalize

from vectordb.lsh import HyperplaneLSH, _signature_bits


def test_signature_bits_stable_with_seed() -> None:
    rng = RandomState(42)
    n_docs, n_features = 5, 20
    X = sparse.random(n_docs, n_features, density=0.3, random_state=rng, dtype=np.float64)
    X = normalize(X, norm="l2", copy=True).tocsr()
    hp = rng.randn(8, n_features).astype(np.float64)
    b1 = _signature_bits(X, hp)
    b2 = _signature_bits(X, hp)
    assert np.array_equal(b1, b2)


def test_hyperplane_lsh_same_vector_in_candidates() -> None:
    """Identical rows should land in the same buckets; query sees itself."""
    rng = RandomState(0)
    texts_like = 4
    n_features = 32
    row = rng.randn(1, n_features)
    X = sparse.vstack([sparse.csr_matrix(row) for _ in range(texts_like)])
    X = normalize(X, norm="l2", copy=True).tocsr()

    lsh = HyperplaneLSH(num_tables=5, num_hyperplanes=8, rng=RandomState(0))
    lsh.fit(X)
    q = X[0]
    cand = lsh.query_bucket_indices(q)
    assert 0 in cand
