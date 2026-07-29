import numpy as np
import pytest

import rinnd


def make_data(n_points=32, dimension=4):
    return np.arange(n_points * dimension, dtype=np.float32).reshape(n_points, dimension)


def test_builds_neighbor_graph_and_queries():
    data = make_data()
    index = rinnd.RINND(data, n_neighbors=6, n_trees=2, n_iters=3, random_state=42)

    indices, distances = index.neighbor_graph
    assert indices.shape == (len(data), 6)
    assert distances.shape == (len(data), 6)
    assert indices.dtype == np.int32
    assert distances.dtype == np.float32

    query_indices, query_distances = index.query(data[:3], k=4)
    assert query_indices.shape == (3, 4)
    assert query_distances.shape == (3, 4)
    assert np.all(np.isfinite(query_distances))


def test_single_query_matches_batch_query():
    data = make_data()
    index = rinnd.RINND(data, n_neighbors=6, n_trees=2, n_iters=3, random_state=42)
    query = data[0]

    batch_indices, batch_distances = index.query(query[None, :], k=4)
    single_indices, single_distances = index.query_one(query, k=4)

    np.testing.assert_array_equal(single_indices, batch_indices[0])
    np.testing.assert_allclose(single_distances, batch_distances[0])


def test_cosine_direct_mode_and_simd_info():
    data = make_data().astype(np.float32)
    index = rinnd.RINND(
        data,
        metric="cosine",
        normalize=True,
        cosine_distance_mode="direct",
        n_neighbors=6,
        n_trees=2,
        n_iters=3,
        random_state=42,
    )

    indices, distances = index.query(data[:2], k=3)
    assert indices.shape == (2, 3)
    assert np.all((distances >= 0.0) & (distances <= 2.0))
    assert isinstance(rinnd.simd_info(), str)


def test_rejects_unknown_metric_and_query_dimension():
    data = make_data()
    with pytest.raises(ValueError, match="Unknown metric"):
        rinnd.RINND(data, metric="not-a-metric")

    index = rinnd.RINND(data, n_neighbors=6, n_trees=2, n_iters=3)
    with pytest.raises(ValueError, match="dimension"):
        index.query(np.zeros((1, data.shape[1] + 1), dtype=np.float32), k=2)
