import json

import h5py
import numpy as np
import pytest

from ground_truth import exact_neighbors, generate, sample_ids
from benchmark_recall import (
    graph_phase_seconds,
    graph_recall,
    main,
    paired_gate,
    query_ids,
)


def test_public_graph_distances_use_original_vectors():
    from benchmark_recall import validate_graph_distances

    data = np.array([[0.0], [1.0003], [2.0]], dtype=np.float32)
    indices = np.array([[1, 2], [0, 2], [0, 1]])
    distances = np.array([[1.0003, 2.0], [1.0003, 0.9997], [2.0, 0.9997]])
    assert (
        validate_graph_distances(data, [0], indices, distances, "euclidean", {})[
            "edges"
        ]
        == 2
    )
    distances[0, 0] = 1.0
    with pytest.raises(ValueError, match="original-vector"):
        validate_graph_distances(data, [0], indices, distances, "euclidean", {})


def test_graph_phase_seconds_are_disjoint_and_complete():
    record = {
        "elapsed_seconds": 15.0,
        "export_seconds": 0.5,
        "build_stats": {
            "updates": [100, 10],
            "forest_seconds": 1.0,
            "leaf_initialization_seconds": 2.0,
            "candidate_seconds": [1.0, 1.0],
            "update_seconds": [4.0, 3.0],
            "update_generation_seconds": [2.0, 1.0],
            "update_application_seconds": [1.0, 1.0],
            "sort_seconds": 0.5,
            "distance_correction_seconds": 0.5,
        },
    }
    phases = graph_phase_seconds(record)
    assert set(phases) == {
        "forest",
        "leaf_initialization",
        "candidates",
        "update_generation",
        "update_application",
        "update_bookkeeping",
        "sort",
        "distance_correction",
        "export",
        "other",
    }
    assert phases["update_generation"] == 3.0
    assert phases["update_application"] == 2.0
    assert phases["update_bookkeeping"] == 2.0
    assert phases["other"] == 1.5
    assert sum(phases.values()) == record["elapsed_seconds"]
    record["build_stats"]["update_seconds"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="exceed total update"):
        graph_phase_seconds(record)
    record["build_stats"]["update_seconds"] = [1.0]
    with pytest.raises(ValueError, match="iteration count"):
        graph_phase_seconds(record)


def test_feature_matrix_enumerates_all_combinations():
    from feature_matrix import FEATURES, features

    combinations = [tuple(features(mask)) for mask in range(16)]
    assert len(set(combinations)) == 16
    assert combinations[0] == ()
    assert combinations[15] == FEATURES
    for bit, feature in enumerate(FEATURES):
        assert features(1 << bit) == [feature]
    with pytest.raises(ValueError):
        features(16)


def test_candidate_profile_features_and_timing_partition():
    from feature_matrix import FEATURES, candidate_phase_seconds, profile_features

    assert profile_features("wide", "candidate-membership") == list(FEATURES)
    assert profile_features("packed", "candidate-membership") == list(FEATURES) + [
        "candidate-membership"
    ]
    assert profile_features("wide", "candidate-workspace") == list(FEATURES)
    assert profile_features("packed", "candidate-workspace") == list(FEATURES) + [
        "candidate-workspace"
    ]
    assert profile_features("wide", "candidate-combined") == list(FEATURES)
    assert profile_features("packed", "candidate-combined") == list(FEATURES) + [
        "candidate-membership",
        "candidate-workspace",
    ]
    phases = {
        name: [0.1, 0.2]
        for name in ("initialization", "forward", "reverse", "mark", "release")
    }
    record = {
        "build_stats": {"candidate_seconds": [1.0, 2.0], "candidate_phases": phases}
    }
    totals = candidate_phase_seconds(record)
    assert sum(totals.values()) == pytest.approx(3.0)
    assert totals["other"] == pytest.approx(1.5)
    phases["forward"][0] = 5.0
    with pytest.raises(ValueError, match="exceed candidate"):
        candidate_phase_seconds(record)
    phases["forward"] = []
    with pytest.raises(ValueError, match="iteration count"):
        candidate_phase_seconds(record)


@pytest.mark.parametrize(
    "action,handler",
    [("build-profile", "build"), ("profile-updates", "profile_updates")],
)
def test_combined_candidate_profile_cli(action, handler, monkeypatch, tmp_path):
    import feature_matrix

    calls = []
    monkeypatch.setattr(feature_matrix, handler, calls.append)
    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_matrix.py",
            action,
            "--artifacts",
            str(tmp_path),
            "--candidate-feature",
            "candidate-combined",
        ],
    )
    feature_matrix.main()
    assert len(calls) == 1
    assert calls[0].candidate_feature == "candidate-combined"
    assert feature_matrix.profile_features("packed", calls[0].candidate_feature) == (
        list(feature_matrix.FEATURES) + ["candidate-membership", "candidate-workspace"]
    )


def test_update_profile_features_and_repeat_equality():
    from copy import deepcopy
    from feature_matrix import FEATURES, check_profile_fingerprint, profile_features

    assert profile_features("wide") == list(FEATURES)
    assert profile_features("packed") == list(FEATURES) + ["packed-updates"]
    report = {
        "dataset_sha256": "data",
        "records": [
            {
                "case": "graph-k15",
                "seed": 42,
                "graph_sha256": "same",
                "build_stats": {"updates": [10, 1]},
            }
        ],
    }
    seen = {}
    check_profile_fingerprint(seen, report)
    check_profile_fingerprint(seen, deepcopy(report))
    changed = deepcopy(report)
    changed["records"][0]["graph_sha256"] = "different"
    with pytest.raises(ValueError, match="across repeated builds"):
        check_profile_fingerprint(seen, changed)
    changed = deepcopy(report)
    changed["records"][0]["build_stats"]["updates"] = [9, 1]
    with pytest.raises(ValueError, match="across repeated builds"):
        check_profile_fingerprint(seen, changed)


def test_feature_matrix_requires_paired_identical_outputs():
    from copy import deepcopy
    from feature_matrix import compare

    baseline = {
        "dataset_sha256": "data",
        "mode": "graph",
        "threads": 8,
        "parameters": {},
        "graph_only": True,
        "detail": {},
        "records": [
            {
                "case": "graph-k15",
                "seed": 42,
                "repetition": 0,
                "elapsed_seconds": 2,
                "graph_sha256": "same",
                "recalls": {"strict": [1.0]},
            }
        ],
    }
    candidate = deepcopy(baseline)
    candidate["records"][0]["elapsed_seconds"] = 1
    assert compare(baseline, candidate) == {"graph-k15": 2.0}
    candidate["records"][0]["graph_sha256"] = "different"
    with pytest.raises(ValueError, match="outputs or recalls differ"):
        compare(baseline, candidate)
    candidate = deepcopy(baseline)
    candidate["records"][0]["seed"] = 43
    with pytest.raises(ValueError, match="unpaired"):
        compare(baseline, candidate)
    candidate = deepcopy(baseline)
    candidate["threads"] = 1
    with pytest.raises(ValueError, match="incompatible"):
        compare(baseline, candidate)


def test_feature_matrix_confirmation_requires_complete_evidence():
    from copy import deepcopy
    from feature_matrix import confirmation_endpoint

    pairs = []
    for seed in range(100, 110):
        baseline = [
            {
                "seed": seed,
                "repetition": repetition,
                "elapsed_seconds": 2.0,
                "recalls": {"strict": [0.9, 1.0]},
            }
            for repetition in range(5)
        ]
        candidate = deepcopy(baseline)
        for record in candidate:
            record["elapsed_seconds"] = 1.0
        pairs.append((baseline, candidate))
    result = confirmation_endpoint(pairs)
    assert result["recall"]["strict"]["status"] == "non_inferior"
    assert result["speedup_lower_bound"] == pytest.approx(2)
    assert result["timing_status"] == "non_regressing"
    with pytest.raises(ValueError, match="ten paired seeds"):
        confirmation_endpoint(pairs[:9])
    incomplete = deepcopy(pairs)
    incomplete[0][1].pop()
    with pytest.raises(ValueError, match="five paired repetitions"):
        confirmation_endpoint(incomplete)
    slower = deepcopy(pairs)
    for baseline, candidate in slower:
        for record in candidate:
            record["elapsed_seconds"] = 3
            record["recalls"] = {"strict": [0.8, 0.9]}
    result = confirmation_endpoint(slower)
    assert result["recall"]["strict"]["status"] == "blocked"
    assert result["timing_status"] == "inconclusive_or_regressing"


def test_graph_profile_summary_checks_counts_and_preserves_pairing():
    from copy import deepcopy
    from feature_matrix import graph_profile_summary

    record = {
        "case": "graph-k15",
        "seed": 42,
        "repetition": 0,
        "elapsed_seconds": 15.0,
        "export_seconds": 0.5,
        "graph_sha256": "same",
        "recalls": {},
        "build_stats": {
            "updates": [100, 10],
            "forest_seconds": 1.0,
            "leaf_initialization_seconds": 2.0,
            "candidate_seconds": [1.0, 1.0],
            "update_seconds": [4.0, 3.0],
            "update_generation_seconds": [2.0, 1.0],
            "update_application_seconds": [1.0, 1.0],
            "sort_seconds": 0.5,
            "distance_correction_seconds": 0.5,
        },
        "resources": {
            "process_cpu_seconds": 60.0,
            "process_peak_rss_bytes": 10000,
            "involuntary_context_switches": 10,
            "host_load_before": [1, 2, 3],
            "host_load_after": [2, 3, 4],
        },
    }
    baseline = {
        "dataset": "fixture",
        "dataset_sha256": "data",
        "mode": "profile",
        "threads": 8,
        "parameters": {},
        "graph_only": True,
        "detail": {"corpus_size": 100},
        "records": [record],
    }
    candidate = deepcopy(baseline)
    candidate["records"][0]["elapsed_seconds"] = 14.0
    result = graph_profile_summary([(baseline, candidate)] * 3)
    assert result["pairs"] == 3
    assert result["delivery"]["median_paired_speedup"] == pytest.approx(15 / 14)
    assert result["phases"]["update_generation"]["median_paired_speedup"] == 1
    assert result["phases"]["update_bookkeeping"]["wide_median_seconds"] == 2
    assert result["resources"]["wide"]["median_cpu_seconds_per_wall_second"] == 4
    candidate["records"][0]["build_stats"]["updates"][0] += 1
    with pytest.raises(ValueError, match="iteration updates"):
        graph_profile_summary([(baseline, candidate)])
    candidate["records"][0]["graph_sha256"] = "different"
    with pytest.raises(ValueError, match="outputs or recalls"):
        graph_profile_summary([(baseline, candidate)])


def test_graph_recall_penalizes_missing_duplicate_self_and_invalid_ids():
    data = np.arange(8, dtype=float).reshape(-1, 1)
    truth_ids, truth_distances = exact_neighbors(
        data, [0], np.arange(8), "euclidean", 5
    )
    scores = graph_recall(
        data,
        [0],
        truth_ids,
        truth_distances,
        np.array([[1, 1, 0, -1, 99]]),
        "euclidean",
        5,
    )
    assert scores == {"strict": [0.2], "tie_aware": [0.2]}


def test_graph_recall_accepts_tied_alternatives_but_reports_id_overlap():
    data = np.ones((8, 2))
    truth_ids, truth_distances = exact_neighbors(
        data, [0], np.arange(8), "euclidean", 3
    )
    scores = graph_recall(
        data, [0], truth_ids, truth_distances, np.array([[4, 5, 6]]), "euclidean", 3
    )
    assert scores == {"strict": [0.0], "tie_aware": [1.0]}


def test_query_splits_are_disjoint_and_fixed():
    tuning = query_ids(100, "tuning", 0)
    confirmation = query_ids(100, "confirmation", 0)
    assert not set(tuning).intersection(confirmation)
    assert set(tuning).union(confirmation) == set(range(100))
    np.testing.assert_array_equal(query_ids(100, "tuning", 5), tuning[:5])
    with pytest.raises(ValueError, match="only allowed"):
        query_ids(100, "confirmation", 5)


def test_paired_gate_blocks_loss_and_insufficient_evidence():
    assert paired_gate([0.9] * 3, [0.91] * 3)["status"] == "insufficient_seeds"
    assert paired_gate([0.9] * 10, [0.8995] * 10)["status"] == "non_inferior"
    assert paired_gate([0.9] * 10, [0.89] * 10)["status"] == "blocked"
    candidate = np.linspace(0.895, 0.906, 10)
    normal = paired_gate([0.9] * 10, candidate)
    corrected = paired_gate([0.9] * 10, candidate, family_size=16)
    assert corrected["lower_bound"] < normal["lower_bound"]


@pytest.mark.parametrize("mode", ["graph", "query", "profile"])
def test_measurement_cli_end_to_end(tmp_path, monkeypatch, mode):
    pytest.importorskip("rinnd")
    source = tmp_path / "fixture.hdf5"
    reference = tmp_path / "reference.hdf5"
    output = tmp_path / "run.json"
    values = np.random.default_rng(7).normal(size=(48, 4)).astype(np.float32)
    with h5py.File(source, "w") as dataset:
        dataset.attrs["distance"] = "euclidean"
        dataset["train"] = values
        dataset["test"] = values[:20]
        dataset["distances"] = np.sort(
            np.linalg.norm(
                values[:20, None, :].astype(np.float64) - values[None, :, :], axis=2
            ),
            axis=1,
        )[:, :10]
    argv = [
        "benchmark_recall.py",
        mode,
        str(source),
        str(output),
        "--seeds",
        "42",
        "--repeats",
        "1",
        "--threads",
        "1",
        "--epsilons",
        "0.1",
    ]
    if mode != "profile":
        generate(source, reference, size=35)
        argv += ["--reference", str(reference)]
    else:
        argv += ["--degrees", "30"]
    monkeypatch.setattr("sys.argv", argv)
    main()
    result = json.loads(output.read_text())
    assert result["promotion_status"] == "exploratory_only"
    assert result["provenance"]["binary_hashes"]
    assert result["records"]
    for record in result["records"]:
        assert record["elapsed_seconds"] > 0
        for recalls in record["recalls"].values():
            assert all(0 <= value <= 1 for value in recalls)
    if mode in ("graph", "profile"):
        assert all(len(record["graph_sha256"]) == 64 for record in result["records"])
        for record in result["records"]:
            assert sum(record["phase_seconds"].values()) == pytest.approx(
                record["elapsed_seconds"]
            )
            assert all(value >= 0 for value in record["phase_seconds"].values())
            assert record["resources"]["process_cpu_seconds"] > 0
            assert record["resources"]["process_peak_rss_bytes"] > 0
            assert len(record["resources"]["host_load_before"]) == 3
        if mode == "profile":
            assert result["detail"] == {
                "corpus_size": len(values),
                "recall_status": "not_evaluated",
            }
            assert all(record["recalls"] == {} for record in result["records"])
            assert [record["returned_bytes"] for record in result["records"]] == [
                len(values) * 30 * 8
            ]
    else:
        assert result["detail"]["invalid_truth_rows"] == []
        assert result["detail"]["total_invalid_truth_rows"] == 0
    with pytest.raises(SystemExit):
        main()


def test_reference_rejects_overwriting_source_or_unrelated_file(tmp_path):
    source = tmp_path / "source.hdf5"
    unrelated = tmp_path / "unrelated.hdf5"
    for path in (source, unrelated):
        with h5py.File(path, "w") as dataset:
            dataset["preserve"] = [1, 2]
    with pytest.raises(ValueError, match="source dataset"):
        generate(source, source)
    with pytest.raises(ValueError, match="not a reference artifact"):
        generate(source, unrelated)
    with h5py.File(unrelated, "r") as dataset:
        assert list(dataset) == ["preserve"]


@pytest.mark.parametrize("metric", ["euclidean", "angular"])
@pytest.mark.parametrize("reference_block", [1, 7, 100])
def test_exact_neighbors_matches_independent_distances(metric, reference_block):
    data = np.random.default_rng(5).normal(size=(40, 8)).astype(np.float32)
    data[3] = data[2]
    original_ids = np.arange(len(data))[::-1]
    query_rows = np.array([0, 2, 3, 19])
    indices, distances = exact_neighbors(
        data, query_rows, original_ids, metric, 15, reference_block
    )
    values = data.astype(np.float64)
    for row, query_row in enumerate(query_rows):
        query = values[query_row]
        expected = np.array(
            [
                (
                    np.linalg.norm(query - vector)
                    if metric == "euclidean"
                    else np.clip(
                        1
                        - np.dot(query, vector)
                        / (np.linalg.norm(query) * np.linalg.norm(vector)),
                        0,
                        2,
                    )
                )
                for vector in values
            ]
        )
        expected[query_row] = np.inf
        ordered = np.lexsort((original_ids, expected))[:15]
        np.testing.assert_array_equal(indices[row], ordered)
        np.testing.assert_allclose(distances[row], expected[ordered], atol=1e-14)


def test_duplicate_ties_and_self_exclusion():
    data = np.ones((40, 4))
    original_ids = np.arange(40)[::-1]
    indices, distances = exact_neighbors(
        data, np.array([39]), original_ids, "euclidean", 30, 9
    )
    np.testing.assert_array_equal(indices[0], np.arange(38, 8, -1))
    assert not distances.any()


def test_nested_sampling():
    assert set(sample_ids(100, 15, 42)) <= set(sample_ids(100, 30, 42))


def test_invalid_oracle_inputs():
    with pytest.raises(ValueError, match="finite"):
        exact_neighbors(np.full((4, 2), np.nan), [0], np.arange(4), "euclidean", 2)
    with pytest.raises(ValueError, match="query row"):
        exact_neighbors(np.ones((4, 2)), [-1], np.arange(4), "euclidean", 2)


def test_angular_zero_vectors_have_explicit_unit_distance():
    indices, distances = exact_neighbors(
        np.zeros((4, 2)), [0], np.arange(4), "angular", 2
    )
    np.testing.assert_array_equal(indices, [[1, 2]])
    np.testing.assert_array_equal(distances, [[1.0, 1.0]])


@pytest.mark.parametrize("size", [35, 0])
def test_reference_manifest_resume_and_provenance(tmp_path, size):
    source = tmp_path / "fixture.hdf5"
    output = tmp_path / "exact.hdf5"
    with h5py.File(source, "w") as dataset:
        dataset.attrs["distance"] = "euclidean"
        dataset["train"] = np.random.default_rng(2).normal(size=(45, 4))
    generate(source, output, size=size, sample_rows=8, query_block=4)
    with h5py.File(output, "r+") as reference:
        expected = reference["neighbors"][:]
        reference.attrs["complete"] = False
        reference.attrs["completed_rows"] = 4
        reference["neighbors"][4:] = -1
    generate(source, output, size=size, sample_rows=8, query_block=4)
    with h5py.File(output, "r") as reference:
        assert reference.attrs["complete"]
        np.testing.assert_array_equal(reference["neighbors"][:], expected)
    with pytest.raises(ValueError, match="manifest differs"):
        generate(source, output, size=size, sample_rows=8, seed=9, query_block=4)
