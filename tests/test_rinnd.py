import numpy as np
import pytest

import rinnd


def test_named_distances_reports_supported_names():
    assert isinstance(rinnd.named_distances, dict)
    assert "euclidean" in rinnd.named_distances
    assert "cosine" in rinnd.named_distances
    assert "symmetric_kullback_liebler" in rinnd.named_distances
    assert "not-a-metric" not in rinnd.named_distances


def make_data(n_points=32, dimension=4):
    return np.arange(n_points * dimension, dtype=np.float32).reshape(
        n_points, dimension
    )


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


@pytest.mark.parametrize("graph_only", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_candidate_phase_statistics_partition_candidate_time(graph_only, n_jobs):
    index = rinnd.RINND(
        make_data(n_points=300),
        n_neighbors=30,
        n_trees=2,
        n_iters=3,
        random_state=42,
        graph_only=graph_only,
        n_jobs=n_jobs,
    )
    stats = dict(index.build_stats)
    phases = stats["candidate_phases"]
    assert set(phases) == {"initialization", "forward", "reverse", "mark", "release"}
    count = len(stats["updates"])
    assert count > 0
    assert len(stats["candidate_seconds"]) == count
    for values in phases.values():
        assert len(values) == count
        assert np.isfinite(values).all()
        assert np.all(np.asarray(values) >= 0.0)
    for iteration, total in enumerate(stats["candidate_seconds"]):
        assert sum(values[iteration] for values in phases.values()) <= total
    if n_jobs == 1:
        assert phases["reverse"] == [0.0] * count
    expected = (
        stats["forest_seconds"]
        + stats["leaf_initialization_seconds"]
        + sum(stats["candidate_seconds"])
        + sum(stats["update_seconds"])
    )
    assert stats["nn_descent_seconds"] == pytest.approx(expected)
    assert dict(index.build_stats) == stats


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
    iteration_count = len(first_stats["updates"])
    assert len(first_stats["update_generation_seconds"]) == iteration_count
    assert len(first_stats["update_application_seconds"]) == iteration_count
    for total, generation, application in zip(
        first_stats["update_seconds"],
        first_stats["update_generation_seconds"],
        first_stats["update_application_seconds"],
    ):
        assert np.isfinite([total, generation, application]).all()
        assert generation >= 0.0 and application >= 0.0
        assert generation + application <= total
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


def test_search_index_uses_breadth_first_forest_by_default():
    data = make_data()
    common = dict(
        n_neighbors=6, n_trees=4, n_iters=3, random_state=42, tree_batch_width=4
    )
    default_index = rinnd.RINND(data, **common)
    explicit_index = rinnd.RINND(data, tree_build_strategy="breadth_first", **common)
    graph_index = rinnd.RINND(data, graph_only=True, **common)

    indices, distances = default_index.neighbor_graph
    for other in (explicit_index, graph_index):
        other_indices, other_distances = other.neighbor_graph
        np.testing.assert_array_equal(indices, other_indices)
        np.testing.assert_array_equal(distances, other_distances)

    query_indices, _ = default_index.query(data[:10], k=6)
    assert np.all(query_indices[:, 0] == np.arange(10))

    legacy = rinnd.RINND(data, tree_build_strategy="depth_first", **common)
    assert legacy.query(data[:10], k=6)[0].shape == (10, 6)


def test_breadth_first_forest_rejects_invalid_batch_width():
    data = make_data()
    with pytest.raises(ValueError, match="tree_batch_width must be in 1..=8"):
        rinnd.RINND(data, tree_build_strategy="breadth_first", tree_batch_width=0)

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


def clustered_data(n_points=4000, n_queries=60, dimension=48, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(30, dimension)).astype(np.float32) * 3
    data = centers[rng.integers(0, 30, n_points)] + rng.normal(size=(n_points, dimension))
    queries = centers[rng.integers(0, 30, n_queries)] + rng.normal(size=(n_queries, dimension))
    return data.astype(np.float32), queries.astype(np.float32)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("quantization", ["sq8", "sq4"])
@pytest.mark.parametrize("rerank_precision", ["float32", "float16"])
def test_sq_quantization_searches_and_reranks(metric, quantization, rerank_precision):
    data, queries = clustered_data()
    kwargs = dict(normalize=True, cosine_distance_mode="direct") if metric == "cosine" else {}
    index = rinnd.RINND(
        data, metric=metric, n_neighbors=15, random_state=42, quantization=quantization,
        rerank_precision=rerank_precision, **kwargs
    )
    info = index.storage_info()
    assert info["encoding"] == f"{quantization}_affine"
    # Code rows are padded to whole 64-byte lines.
    assert info["code_bytes"] >= len(data) * 64
    if metric == "euclidean":
        exact = np.sqrt(((queries[:, None] - data[None]) ** 2).sum(-1))
    else:
        unit = lambda values: values / np.linalg.norm(values, axis=1, keepdims=True)
        exact = 1.0 - unit(queries) @ unit(data).T
    true = np.argsort(exact, axis=1)[:, :10]

    # Sixteen levels per feature need more candidates on this 48-feature data.
    width = 40 if quantization == "sq8" else 80
    found = np.vstack(
        [
            index.query_one_indices_quantized(
                query, k=10, epsilon=0.05, candidate_width=width, rerank_width=width
            )
            for query in queries
        ]
    )
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, true)])
    assert recall > 0.95

    indices, distances = index.query(queries, k=10, epsilon=0.05)
    # Candidates are reranked with the exact distance (from float16 values,
    # which carry about three significant digits).
    tolerance = dict(rtol=1e-4, atol=1e-5) if rerank_precision == "float32" else dict(rtol=5e-3, atol=2e-3)
    np.testing.assert_allclose(distances, np.take_along_axis(exact, indices, axis=1), **tolerance)
    assert np.all(np.diff(distances, axis=1) >= 0)


def test_sq_quantization_rejects_unsupported_settings():
    data, _ = clustered_data(n_points=200)
    with pytest.raises(ValueError):
        rinnd.RINND(data, metric="manhattan", n_neighbors=10, quantization="sq8")
    with pytest.raises(ValueError):
        rinnd.RINND(data, metric="cosine", n_neighbors=10, quantization="sq8")
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, quantization="sq2")
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, rerank_precision="float16")
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, quantization="sq8", rerank_precision="bfloat16")
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, quantization_clip=0.01)
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, quantization="sq4", quantization_clip=0.6)


def test_sq4_clipping_survives_outliers():
    data, queries = clustered_data()
    data = data.copy()
    data[0, :] = 1e4  # an outlier that would stretch every feature's code range
    unit_true = np.argsort(((queries[:, None] - data[None]) ** 2).sum(-1), axis=1)[:, :10]
    recalls = {}
    for clip in (0.0, None):
        index = rinnd.RINND(data, n_neighbors=15, random_state=42, quantization="sq4", quantization_clip=clip)
        found = np.vstack(
            [
                index.query_one_indices_quantized(q, k=10, epsilon=0.05, candidate_width=40, rerank_width=40)
                for q in queries
            ]
        )
        recalls[clip] = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, unit_true)])
    # Without clipping sixteen levels span the outlier and the codes say little.
    assert recalls[None] > 0.9
    assert recalls[None] > recalls[0.0] + 0.1


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("dims", [12, 0.99])
def test_sq8_quantization_dims_projects_low_rank_data(metric, dims):
    rng = np.random.default_rng(0)
    basis = rng.normal(size=(8, 64)).astype(np.float32)
    data = (rng.normal(size=(4000, 8)) @ basis + 0.05 * rng.normal(size=(4000, 64))).astype(np.float32)
    queries = (rng.normal(size=(60, 8)) @ basis + 0.05 * rng.normal(size=(60, 64))).astype(np.float32)
    kwargs = dict(normalize=True, cosine_distance_mode="direct") if metric == "cosine" else {}
    index = rinnd.RINND(
        data, metric=metric, n_neighbors=15, random_state=42, quantization="sq8",
        quantization_dims=dims, **kwargs
    )
    info = index.storage_info()
    assert info["encoding"] == "sq8_affine_pca"
    # One 64-byte line per row (and a norm for euclidean) instead of 64 features.
    assert info["code_bytes"] <= (64 + 4) * len(data)
    if metric == "euclidean":
        exact = np.sqrt(((queries[:, None] - data[None]) ** 2).sum(-1))
    else:
        unit = lambda values: values / np.linalg.norm(values, axis=1, keepdims=True)
        exact = 1.0 - unit(queries) @ unit(data).T
    true = np.argsort(exact, axis=1)[:, :10]
    indices, distances = index.query(queries, k=10, epsilon=0.05)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(indices, true)])
    assert recall > 0.9
    np.testing.assert_allclose(
        distances, np.take_along_axis(exact, indices, axis=1), rtol=1e-4, atol=1e-5
    )


def test_sq8_quantization_dims_skips_projection_for_spread_variance():
    data = np.random.default_rng(0).normal(size=(2000, 32)).astype(np.float32)
    index = rinnd.RINND(data, n_neighbors=10, quantization="sq8", quantization_dims=0.95)
    assert index.storage_info()["encoding"] == "sq8_affine"


@pytest.mark.parametrize("dims", [0.0, 2.5, -4.0])
def test_sq8_quantization_dims_rejects_invalid_values(dims):
    data, _ = clustered_data(n_points=200)
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, quantization="sq8", quantization_dims=dims)
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, quantization_dims=8)


def shifted_queries(data_dim=32, n_points=3000, n_sample=600, n_test=100, seed=0):
    """Data and queries from different distributions, with exact sample neighbours."""
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_points, data_dim)).astype(np.float32)
    shift = np.linspace(0, 2, data_dim).astype(np.float32)
    queries = (rng.normal(size=(n_sample + n_test, data_dim)) * 0.6 + shift).astype(np.float32)
    return data, queries[:n_sample], queries[n_sample:]


def exact_neighbors(data, queries, k, metric="euclidean"):
    if metric == "euclidean":
        d = ((queries[:, None] - data[None]) ** 2).sum(-1)
    else:
        unit = lambda values: values / np.linalg.norm(values, axis=1, keepdims=True)
        d = 1.0 - unit(queries) @ unit(data).T
    return np.argsort(d, axis=1)[:, :k].astype(np.int32)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("quantization", [None, "sq8"])
def test_query_sample_adapts_the_index(metric, quantization):
    data, sample, tests = shifted_queries()
    kwargs = dict(normalize=True, cosine_distance_mode="direct") if metric == "cosine" else {}
    if quantization:
        kwargs["quantization"] = quantization
    neighbors = exact_neighbors(data, sample, 30, metric)
    true = exact_neighbors(data, tests, 10, metric)

    def recall(index):
        if quantization:
            found = np.vstack([index.query_one_indices_quantized(q, k=10, epsilon=0.02, candidate_width=30, rerank_width=30) for q in tests])
        else:
            found = np.vstack([index.query_one_indices(q, k=10, epsilon=0.1) for q in tests])
        return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, true)])

    plain = rinnd.RINND(data, metric=metric, n_neighbors=10, random_state=42, **kwargs)
    adapted = rinnd.RINND(
        data, metric=metric, n_neighbors=10, random_state=42, query_sample=sample,
        query_neighbors=neighbors, **kwargs
    )
    adapted.prepare()
    assert len(adapted.search_graph[1]) > len(plain.search_graph[1])
    assert recall(adapted) > 0.9
    # The queries are far from the data, which the unadapted index finds hard.
    assert recall(adapted) > recall(plain) + 0.1


def test_query_sample_options_and_validation():
    data, sample, _ = shifted_queries(n_points=500, n_sample=100)
    neighbors = exact_neighbors(data, sample, 20)
    no_edges = rinnd.RINND(
        data, n_neighbors=10, random_state=42, query_sample=sample, query_neighbors=neighbors,
        query_sample_options={"edges": 0},
    )
    plain = rinnd.RINND(data, n_neighbors=10, random_state=42)
    no_edges.prepare()
    plain.prepare()
    # Without query edges only the search tree changes.
    np.testing.assert_array_equal(no_edges.search_graph[1], plain.search_graph[1])
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, query_sample=sample)
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, query_neighbors=neighbors)
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, query_sample=sample[:, :5], query_neighbors=neighbors)
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, query_sample=sample, query_neighbors=neighbors + 1000)
    with pytest.raises(ValueError):
        rinnd.RINND(data, n_neighbors=10, query_sample=sample, query_neighbors=neighbors,
                    query_sample_options={"bogus": 3})


def mips_data(n_points=4000, n_queries=60, dimension=48, seed=0):
    rng = np.random.default_rng(seed)
    # Varied norms, so inner product order differs from euclidean and cosine order.
    scale = rng.uniform(0.3, 2.0, size=(n_points, 1))
    data = rng.normal(size=(n_points, dimension)) * scale
    queries = rng.normal(size=(n_queries, dimension))
    return data.astype(np.float32), queries.astype(np.float32)


@pytest.mark.parametrize(
    "quantization,rerank_precision",
    [(None, "float32"), ("sq8", "float32"), ("sq8", "float16"), ("sq4", "float16")],
)
def test_mips_finds_largest_inner_products(quantization, rerank_precision):
    data, queries = mips_data()
    kwargs = dict(quantization=quantization, rerank_precision=rerank_precision) if quantization else {}
    index = rinnd.RINND(data, metric="mips", n_neighbors=15, random_state=42, **kwargs)
    inner = queries @ data.T
    true = np.argsort(-inner, axis=1)[:, :10]
    if quantization:
        width = 40 if quantization == "sq8" else 80
        found = np.vstack(
            [
                index.query_one_indices_quantized(
                    query, k=10, epsilon=0.3, candidate_width=width, rerank_width=width
                )
                for query in queries
            ]
        )
        assert index.storage_info()["encoding"] == f"{quantization}_affine"
    else:
        found = np.vstack([index.query_one_indices(query, k=10, epsilon=0.3) for query in queries])
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, true)])
    assert recall > 0.95

    # Distances are negative inner products.
    indices, distances = index.query(queries, k=10, epsilon=0.3)
    want = -np.take_along_axis(inner, indices, axis=1)
    atol = 1e-2 if rerank_precision == "float32" else 0.05 * np.abs(inner).max() / 100
    np.testing.assert_allclose(distances, want, rtol=1e-4, atol=atol)
    assert np.all(np.diff(distances, axis=1) >= 0)
    one_indices, one_distances = index.query_one(queries[0], k=10, epsilon=0.3)
    np.testing.assert_array_equal(one_indices, indices[0])
    np.testing.assert_allclose(one_distances, distances[0], rtol=1e-6)


def test_mips_learn_queries_and_validation():
    data, queries = mips_data(n_points=3000, n_queries=200)
    index = rinnd.RINND(data, metric="mips", n_neighbors=10, random_state=42)
    # The graph is of the augmented points, not an inner-product k-NN graph.
    with pytest.raises(NotImplementedError):
        index.neighbor_graph

    sample, tests = queries[:150], queries[150:]
    neighbors = np.argsort(-(sample @ data.T), axis=1)[:, :20].astype(np.int32)
    adapted = rinnd.RINND(
        data, metric="mips", n_neighbors=10, random_state=42, quantization="sq8",
        query_sample=sample, query_neighbors=neighbors,
    )
    true = np.argsort(-(tests @ data.T), axis=1)[:, :10]
    found = np.vstack(
        [
            adapted.query_one_indices_quantized(q, k=10, epsilon=0.3, candidate_width=40, rerank_width=40)
            for q in tests
        ]
    )
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, true)]) > 0.95

    with pytest.raises(ValueError):
        rinnd.RINND(data, metric="mips", n_neighbors=10, quantization="sq8", quantization_dims=8)
    with pytest.raises(ValueError):
        rinnd.RINND(data, metric="mips", n_neighbors=10, graph_only=True)
    with pytest.raises(ValueError):
        index.query(queries[:, :5], k=10)


def test_diversify_alpha_and_vertex_order():
    data, queries = clustered_data(n_points=3000)
    base = dict(n_neighbors=15, random_state=42)
    plain = rinnd.RINND(data, **base)
    relaxed = rinnd.RINND(data, diversify_alpha=1.2, **base)
    rcm = rinnd.RINND(data, vertex_order="rcm", **base)
    for index in (plain, relaxed, rcm):
        index.prepare()
    # alpha-pruning keeps more edges; the vertex order does not change the graph.
    assert len(relaxed.search_graph[1]) > len(plain.search_graph[1])
    assert len(rcm.search_graph[1]) == len(plain.search_graph[1])
    true = np.argsort(((queries[:, None] - data[None]) ** 2).sum(-1), axis=1)[:, :10]
    for index in (relaxed, rcm):
        found = np.vstack([index.query_one_indices(q, k=10, epsilon=0.1) for q in queries])
        assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, true)]) > 0.95
    with pytest.raises(ValueError):
        rinnd.RINND(data, diversify_alpha=0.9, **base)
    with pytest.raises(ValueError):
        rinnd.RINND(data, vertex_order="random", **base)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_progressive_build_matches_exact_build(metric):
    data, _ = clustered_data(n_points=1500, dimension=64)
    kwargs = dict(normalize=True, cosine_distance_mode="direct") if metric == "cosine" else {}
    base = dict(metric=metric, n_neighbors=10, graph_only=True, random_state=42, **kwargs)
    if metric == "euclidean":
        squared = (data**2).sum(1)
        d = squared[:, None] - 2 * data @ data.T + squared[None]
    else:
        unit = data / np.linalg.norm(data, axis=1, keepdims=True)
        d = 1.0 - unit @ unit.T
    np.fill_diagonal(d, np.inf)
    true = np.argsort(d, axis=1)[:, :10]

    graphs = {}
    for mode in (True, False):
        index = rinnd.RINND(data, progressive_build=mode, **base)
        stats = dict(index.build_stats)
        assert (stats["approximate_iterations"] > 0) == mode
        assert (stats["code_seconds"] > 0) == mode
        graphs[mode] = index.neighbor_graph
    recall = {
        mode: np.mean([len(set(a) & set(b)) / 10 for a, b in zip(indices, true)])
        for mode, (indices, _) in graphs.items()
    }
    assert recall[True] > 0.97 and recall[True] >= recall[False] - 0.01
    # Both graphs hold exact distances.
    exact = {}
    for indices, distances in graphs.values():
        for point, (row, values) in enumerate(zip(indices, distances)):
            for neighbor, value in zip(row, values):
                exact.setdefault((point, neighbor), value)
                assert np.isclose(exact[(point, neighbor)], value, rtol=1e-4, atol=1e-5)

    # 'auto' (the default) leaves narrow data exact.
    for mode in ("auto", None):
        kwargs = {} if mode is None else dict(progressive_build=mode)
        assert dict(rinnd.RINND(data, **kwargs, **base).build_stats)["approximate_iterations"] == 0
    with pytest.raises(ValueError, match="progressive_build"):
        rinnd.RINND(data, progressive_build="always", **base)


def test_normalize_copies_any_layout_in_parallel():
    data, _ = clustered_data(n_points=2000, dimension=40)
    unit = data / np.linalg.norm(data, axis=1, keepdims=True)
    d = 1.0 - unit @ unit.T
    np.fill_diagonal(d, np.inf)
    true = np.argsort(d, axis=1)[:, :8]
    base = dict(metric="cosine", normalize=True, cosine_distance_mode="direct", n_neighbors=8, random_state=7)
    # Reversing the features keeps cosine distances, through negative strides.
    for layout in (data, np.asfortranarray(data), data[:, ::-1]):
        indices, distances = rinnd.RINND(layout, graph_only=True, n_jobs=4, **base).neighbor_graph
        assert np.mean([len(set(a) & set(b)) / 8 for a, b in zip(indices, true)]) > 0.97
        assert np.allclose(distances, np.take_along_axis(d, indices, axis=1), atol=1e-5)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_hub_search_tree(metric):
    data, queries = clustered_data(n_points=3000)
    kwargs = dict(normalize=True, cosine_distance_mode="direct") if metric == "cosine" else {}
    base = dict(metric=metric, n_neighbors=15, random_state=42, **kwargs)
    graph = rinnd.RINND(data, graph_only=True, **base).neighbor_graph
    index = rinnd.RINND(data, search_tree_leaf_size=40, **base)
    # The forest only gives leaves: the search index's graph is the graph-only one.
    np.testing.assert_array_equal(index.neighbor_graph[0], graph[0])
    index.prepare()
    assert dict(index.build_stats)["search_tree_seconds"] > 0
    _, _, children, leaf_indices = index.export_search_tree()
    assert sorted(leaf_indices) == list(range(len(data)))
    leaves = children[children[:, 0] <= 0]
    assert np.all(leaves[:, 0] - leaves[:, 1] <= 40)
    exact = exact_neighbors(data, queries, 10, metric)
    found = np.vstack([index.query_one_indices(q, k=10, epsilon=0.1) for q in queries])
    assert np.mean([len(set(a) & set(b)) / 10 for a, b in zip(found, exact)]) > 0.95

    with pytest.raises(ValueError, match="search_tree_leaf_size"):
        rinnd.RINND(data, search_tree_leaf_size=0, **base)
    with pytest.raises(TypeError):
        rinnd.RINND(data, n_search_trees=2, **base)
