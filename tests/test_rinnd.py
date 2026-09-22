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


def test_n_jobs_applies_to_construction_preparation_and_batch_query():
    data = make_data(n_points=64)
    index = rinnd.RINND(
        data,
        n_neighbors=6,
        n_trees=2,
        n_iters=3,
        random_state=42,
        n_jobs=1,
    )

    assert index.is_prepared is False
    index.prepare()
    assert index.is_prepared is True
    indices, distances = index.query(data[:8], k=4)
    assert indices.shape == (8, 4)
    assert np.all(np.isfinite(distances))


@pytest.mark.parametrize("n_jobs", [None, -1, 2])
def test_n_jobs_accepts_all_cores_and_positive_limits(n_jobs):
    data = make_data()
    index = rinnd.RINND(
        data,
        graph_only=True,
        n_neighbors=6,
        n_trees=2,
        n_iters=3,
        random_state=42,
        n_jobs=n_jobs,
    )

    indices, distances = index.neighbor_graph
    assert indices.shape == (len(data), 6)
    assert np.all(np.isfinite(distances))


@pytest.mark.parametrize("n_jobs", [0, -2])
def test_n_jobs_rejects_invalid_values(n_jobs):
    with pytest.raises(ValueError, match="n_jobs must be -1 or a positive integer"):
        rinnd.RINND(make_data(), n_jobs=n_jobs)


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


def test_search_preparation_is_lazy_idempotent_and_automatic():
    data = make_data(n_points=64)
    index = rinnd.RINND(data, n_neighbors=6, n_trees=2, n_iters=3, random_state=42)

    assert index.is_prepared is False
    indices, _ = index.neighbor_graph
    assert indices.shape == (len(data), 6)
    assert index.is_prepared is False

    index.prepare()
    assert index.is_prepared is True
    first_stats = dict(index.build_stats)
    index.prepare()
    assert dict(index.build_stats) == first_stats

    lazy_index = rinnd.RINND(data, n_neighbors=6, n_trees=2, n_iters=3, random_state=42)
    lazy_index.query(data[:1], k=2)
    assert lazy_index.is_prepared is True


def test_pre_normalized_cosine_input_contract():
    data = make_data().astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    index = rinnd.RINND(
        data,
        metric="cosine",
        normalize=True,
        input_normalized=True,
        cosine_distance_mode="direct",
        n_neighbors=6,
        n_trees=2,
        n_iters=3,
    )
    assert index.is_prepared is False

    with pytest.raises(ValueError, match="requires metric='cosine' and normalize=True"):
        rinnd.RINND(data, input_normalized=True, n_neighbors=6, n_trees=2, n_iters=3)


def test_graph_only_retains_graph_and_disables_search():
    data = make_data()
    common = dict(n_neighbors=6, n_trees=2, n_iters=3, random_state=42)
    index = rinnd.RINND(data, graph_only=True, **common)

    indices, distances = index.neighbor_graph
    assert indices.shape == (len(data), 6)
    assert np.all((indices >= 0) & (indices < len(data)))
    assert np.all(np.isfinite(distances))
    assert index.is_prepared is False

    storage = dict(index.storage_info())
    assert storage["graph_only"] is True
    assert storage["retained_fp32_bytes"] == 0
    assert storage["graph_bytes"] == indices.nbytes + distances.nbytes
    assert storage["allocated_bytes"] == storage["graph_bytes"]

    disabled = "search is disabled.*graph_only=True"
    with pytest.raises(ValueError, match=disabled):
        index.prepare()
    with pytest.raises(ValueError, match=disabled):
        index.query(data[:1], k=2)
    with pytest.raises(ValueError, match=disabled):
        _ = index.search_graph
    with pytest.raises(ValueError, match=disabled):
        index.export_search_tree()


def test_graph_only_accepts_transformed_non_contiguous_input():
    data = np.asfortranarray(make_data())
    index = rinnd.RINND(
        data,
        metric="cosine",
        normalize=True,
        cosine_distance_mode="direct",
        graph_only=True,
        n_neighbors=6,
        n_trees=2,
        n_iters=3,
    )
    indices, distances = index.neighbor_graph
    assert indices.shape == (len(data), 6)
    assert np.all(np.isfinite(distances))

    with pytest.raises(ValueError, match="incompatible with quantized cosine modes"):
        rinnd.RINND(
            data,
            metric="cosine",
            normalize=True,
            cosine_distance_mode="int8",
            graph_only=True,
            n_neighbors=6,
            n_trees=2,
            n_iters=3,
        )


def test_graph_only_accepts_breadth_first_forest():
    data = make_data()
    common = dict(
        graph_only=True,
        tree_batch_width=4,
        n_neighbors=6,
        n_trees=4,
        n_iters=3,
        random_state=42,
    )
    default_index = rinnd.RINND(data, **common)
    explicit_index = rinnd.RINND(
        data,
        tree_build_strategy="breadth_first",
        **common,
    )

    indices, distances = default_index.neighbor_graph
    expected_indices, expected_distances = explicit_index.neighbor_graph
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(distances, expected_distances)
    assert indices.shape == (len(data), 6)
    assert np.all((indices >= 0) & (indices < len(data)))
    assert np.all(np.isfinite(distances))
    assert np.all(distances[:, 1:] >= distances[:, :-1])


def test_breadth_first_forest_requires_graph_only():
    data = make_data()
    with pytest.raises(ValueError, match="currently requires graph_only=True"):
        rinnd.RINND(data, tree_build_strategy="breadth_first")

    with pytest.raises(ValueError, match="tree_batch_width must be in 1..=8"):
        rinnd.RINND(
            data,
            graph_only=True,
            tree_build_strategy="breadth_first",
            tree_batch_width=0,
        )


def test_graph_only_accepts_explicit_depth_first_forest():
    data = make_data()
    index = rinnd.RINND(
        data,
        graph_only=True,
        tree_build_strategy="depth_first",
        n_neighbors=6,
        n_trees=2,
        n_iters=3,
    )
    assert index.neighbor_graph[0].shape == (len(data), 6)
