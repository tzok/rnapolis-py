import numpy as np
import pytest

from rnapolis.facility_location import FacilityLocationSelection


def test_precomputed_lazy_selection():
    similarity = np.array(
        [
            [0.0, 0.8, 0.2],
            [0.8, 0.0, 0.7],
            [0.2, 0.7, 0.0],
        ],
        dtype=np.float64,
    )

    selector = FacilityLocationSelection(
        n_samples=2, metric="precomputed", optimizer="lazy"
    )
    selector.fit(similarity)

    assert len(selector.ranking) == 2
    assert len(selector.gains) == 2
    assert set(selector.ranking) <= set(range(3))
    assert selector.gains[0] >= selector.gains[1]
    # The first selected element should be the one with the highest row sum.
    expected_first = int(similarity.sum(axis=1).argmax())
    assert selector.ranking[0] == expected_first


def test_precomputed_naive_matches_lazy():
    rng = np.random.default_rng(0)
    n = 6
    similarity = rng.random((n, n))
    similarity = (similarity + similarity.T) / 2
    np.fill_diagonal(similarity, 0.0)

    lazy = FacilityLocationSelection(n_samples=4, metric="precomputed", optimizer="lazy")
    naive = FacilityLocationSelection(n_samples=4, metric="precomputed", optimizer="naive")

    lazy.fit(similarity)
    naive.fit(similarity)

    assert np.allclose(lazy.gains, naive.gains)
    assert np.array_equal(lazy.ranking, naive.ranking)


def test_embedding_sparse_selection():
    embedding = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )

    selector = FacilityLocationSelection(
        n_samples=3, metric="euclidean", optimizer="lazy", n_neighbors=3
    )
    selector.fit(embedding)

    assert len(selector.ranking) == 3
    assert len(selector.gains) == 3
    assert set(selector.ranking) <= set(range(5))
    assert selector.gains[0] >= selector.gains[1] >= selector.gains[2]


def test_n_samples_capped_to_ground_set():
    similarity = np.eye(3, dtype=np.float64) - np.eye(3, dtype=np.float64)

    selector = FacilityLocationSelection(
        n_samples=10, metric="precomputed", optimizer="lazy"
    )
    selector.fit(similarity)

    assert len(selector.ranking) == 3
    assert len(selector.gains) == 3


def test_non_square_precomputed_raises():
    selector = FacilityLocationSelection(
        n_samples=2, metric="precomputed", optimizer="lazy"
    )
    with pytest.raises(ValueError):
        selector.fit(np.zeros((3, 4), dtype=np.float64))
