r"""Pure-Python facility-location submodular selection.

This module replaces the ``apricot-select`` dependency used by
``rnapolis.distiller``.  It implements a greedy (and optionally lazy-greedy)
maximisation of the standard facility-location function

.. math::

    f(S) = \sum_{i \in V} \max_{j \in S} s(i, j)

where :math:`s` is a non-negative similarity matrix.  The ``ranking`` and
``gains`` attributes exposed by :class:`FacilityLocationSelection` mirror the
interface expected by the rest of the codebase.
"""

import heapq
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def _marginal_gain_dense(
    similarity: np.ndarray, current_values: np.ndarray, idx: int
) -> float:
    """Marginal gain of adding *idx* for a dense similarity matrix."""
    return float(np.maximum(similarity[idx], current_values).sum() - current_values.sum())


def _marginal_gain_sparse(
    similarity: csr_matrix, current_values: np.ndarray, idx: int
) -> float:
    """Marginal gain of adding *idx* for a sparse similarity matrix."""
    start = similarity.indptr[idx]
    end = similarity.indptr[idx + 1]
    if start == end:
        return 0.0
    indices = similarity.indices[start:end]
    data = similarity.data[start:end]
    return float(np.maximum(data, current_values[indices]).sum() - current_values[indices].sum())


def _lazy_greedy_dense(
    similarity: np.ndarray, n_samples: int
) -> Tuple[List[int], List[float]]:
    """Lazy greedy maximisation of facility location with a dense matrix."""
    n = similarity.shape[0]
    k = min(n_samples, n)
    current_values = np.zeros(n, dtype=np.float64)
    selected = np.zeros(n, dtype=bool)

    # Initial modular upper bound: each element covers the whole row.
    initial_gains = np.asarray(similarity.sum(axis=1)).ravel()
    heap = [(-float(initial_gains[i]), i) for i in range(n)]
    heapq.heapify(heap)

    ranking: List[int] = []
    gains: List[float] = []
    for _ in range(k):
        while True:
            neg_gain, idx = heapq.heappop(heap)
            if selected[idx]:
                continue
            gain = _marginal_gain_dense(similarity, current_values, idx)
            if not heap or gain >= -heap[0][0]:
                break
            heapq.heappush(heap, (-gain, idx))

        selected[idx] = True
        ranking.append(int(idx))
        gains.append(float(gain))
        current_values = np.maximum(current_values, similarity[idx])

    return ranking, gains


def _lazy_greedy_sparse(
    similarity: csr_matrix, n_samples: int
) -> Tuple[List[int], List[float]]:
    """Lazy greedy maximisation of facility location with a sparse matrix."""
    n = similarity.shape[0]
    k = min(n_samples, n)
    current_values = np.zeros(n, dtype=np.float64)
    selected = np.zeros(n, dtype=bool)

    initial_gains = np.asarray(similarity.sum(axis=1)).ravel()
    heap = [(-float(initial_gains[i]), i) for i in range(n)]
    heapq.heapify(heap)

    ranking: List[int] = []
    gains: List[float] = []
    for _ in range(k):
        while True:
            neg_gain, idx = heapq.heappop(heap)
            if selected[idx]:
                continue
            gain = _marginal_gain_sparse(similarity, current_values, idx)
            if not heap or gain >= -heap[0][0]:
                break
            heapq.heappush(heap, (-gain, idx))

        selected[idx] = True
        ranking.append(int(idx))
        gains.append(float(gain))

        start = similarity.indptr[idx]
        end = similarity.indptr[idx + 1]
        if start < end:
            indices = similarity.indices[start:end]
            data = similarity.data[start:end]
            current_values[indices] = np.maximum(current_values[indices], data)

    return ranking, gains


def _build_similarity(
    X: np.ndarray,
    metric: str,
    n_neighbors: Optional[int],
) -> csr_matrix | np.ndarray:
    """Convert a data matrix into a non-negative similarity matrix.

    This mirrors the behaviour of ``apricot`` for the two modes used by
    RNApolis: ``metric="precomputed"`` and ``metric="euclidean"`` (either
    dense or with ``n_neighbors``).
    """
    if metric == "precomputed":
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise ValueError("Precomputed similarity matrix must be square.")
        return np.asarray(X, dtype=np.float64)

    if metric == "euclidean" and n_neighbors is None:
        # apricot uses squared Euclidean distances for the dense case.
        distances = pairwise_distances(X, metric="euclidean", squared=True)
    elif metric == "euclidean":
        n = X.shape[0]
        k = min(max(1, int(n_neighbors)), n)
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(X)
        distances = nn.kneighbors_graph(X, mode="distance")
    else:
        # The rest of RNApolis does not use other metrics, but keep the
        # generic distance -> similarity conversion for compatibility.
        distances = pairwise_distances(X, metric=metric)

    if isinstance(distances, csr_matrix):
        if distances.nnz == 0:
            return distances.astype(np.float64)
        max_distance = float(distances.data.max())
        distances.data = max_distance - distances.data
        return distances.astype(np.float64)

    max_distance = float(distances.max())
    return max_distance - np.asarray(distances, dtype=np.float64)


class FacilityLocationSelection:
    """Greedy facility-location subset selection.

    This is a lightweight replacement for ``apricot.FacilityLocationSelection``
    that avoids the ``numba``/``llvmlite`` dependency chain.

    Parameters
    ----------
    n_samples : int
        Number of representatives to select.
    metric : str, optional
        Either ``"precomputed"`` (a square similarity matrix) or a metric
        accepted by scikit-learn.  Default is ``"euclidean"``.
    optimizer : str, optional
        ``"lazy"`` (default) or ``"naive"``.  Both produce the same selected
        set up to tie breaking; ``"lazy"`` is faster for larger inputs.
    n_neighbors : int or None, optional
        If given, build a sparse k-nearest-neighbour similarity graph.
    verbose : bool, optional
        Ignored; kept for API compatibility.

    Attributes
    ----------
    ranking : np.ndarray
        Indices of selected elements in greedy order.
    gains : np.ndarray
        Marginal gain of each selected element.
    """

    def __init__(
        self,
        n_samples: int,
        metric: str = "euclidean",
        optimizer: str = "lazy",
        n_neighbors: Optional[int] = None,
        verbose: bool = False,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        self.n_samples = int(n_samples)
        self.metric = metric
        self.optimizer = optimizer
        self.n_neighbors = n_neighbors
        self.verbose = verbose
        self.ranking: Optional[np.ndarray] = None
        self.gains: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
        sample_cost: Optional[np.ndarray] = None,
    ) -> "FacilityLocationSelection":
        """Run greedy facility-location selection.

        Parameters
        ----------
        X : np.ndarray
            Data matrix or precomputed square similarity matrix.
        y, sample_weight, sample_cost
            Ignored; present for compatibility with apricot's signature.

        Returns
        -------
        FacilityLocationSelection
            The fitted selector.
        """
        similarity = _build_similarity(X, self.metric, self.n_neighbors)
        n = similarity.shape[0]
        k = min(self.n_samples, n)

        if k == 0:
            self.ranking = np.array([], dtype=int)
            self.gains = np.array([], dtype=float)
            return self

        if self.optimizer == "naive":
            if isinstance(similarity, csr_matrix):
                ranking, gains = _naive_greedy_sparse(similarity, k)
            else:
                ranking, gains = _naive_greedy_dense(similarity, k)
        else:
            if isinstance(similarity, csr_matrix):
                ranking, gains = _lazy_greedy_sparse(similarity, k)
            else:
                ranking, gains = _lazy_greedy_dense(similarity, k)

        self.ranking = np.array(ranking, dtype=int)
        self.gains = np.array(gains, dtype=float)
        return self


def _naive_greedy_dense(
    similarity: np.ndarray, n_samples: int
) -> Tuple[List[int], List[float]]:
    """Naive greedy maximisation of facility location with a dense matrix."""
    n = similarity.shape[0]
    k = min(n_samples, n)
    current_values = np.zeros(n, dtype=np.float64)
    selected = np.zeros(n, dtype=bool)

    ranking: List[int] = []
    gains: List[float] = []
    for _ in range(k):
        best_gain = -1.0
        best_idx = 0
        for i in range(n):
            if selected[i]:
                continue
            gain = _marginal_gain_dense(similarity, current_values, i)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        selected[best_idx] = True
        ranking.append(int(best_idx))
        gains.append(float(best_gain))
        current_values = np.maximum(current_values, similarity[best_idx])

    return ranking, gains


def _naive_greedy_sparse(
    similarity: csr_matrix, n_samples: int
) -> Tuple[List[int], List[float]]:
    """Naive greedy maximisation of facility location with a sparse matrix."""
    n = similarity.shape[0]
    k = min(n_samples, n)
    current_values = np.zeros(n, dtype=np.float64)
    selected = np.zeros(n, dtype=bool)

    ranking: List[int] = []
    gains: List[float] = []
    for _ in range(k):
        best_gain = -1.0
        best_idx = 0
        for i in range(n):
            if selected[i]:
                continue
            gain = _marginal_gain_sparse(similarity, current_values, i)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        selected[best_idx] = True
        ranking.append(int(best_idx))
        gains.append(float(best_gain))

        start = similarity.indptr[best_idx]
        end = similarity.indptr[best_idx + 1]
        if start < end:
            indices = similarity.indices[start:end]
            data = similarity.data[start:end]
            current_values[indices] = np.maximum(current_values[indices], data)

    return ranking, gains
