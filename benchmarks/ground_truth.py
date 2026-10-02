"""Reproducible exhaustive training-graph references, never approximate oracles."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import scipy
from scipy.spatial.distance import cdist

DATASETS = (
    "mnist-784-euclidean",
    "fashion-mnist-784-euclidean",
    "glove-100-angular",
    "nytimes-256-angular",
)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_ids(count, size, seed):
    if not 0 < size <= count:
        raise ValueError("sample size must be between 1 and corpus size")
    return np.sort(np.random.default_rng(seed).permutation(count)[:size])


def validate_data(data, metric):
    if metric not in ("euclidean", "angular"):
        raise ValueError("metric must be euclidean or angular")
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("expected a finite two-dimensional dataset")


def metric_distances(queries, reference, metric):
    distances = cdist(
        queries, reference, metric="cosine" if metric == "angular" else "euclidean"
    )
    if metric == "angular":
        distances[~np.any(queries != 0, axis=1), :] = 1.0
        distances[:, ~np.any(reference != 0, axis=1)] = 1.0
    return distances


def exact_neighbors(data, query_rows, original_ids, metric, k=30, reference_block=8192):
    """Return corpus-local IDs, ordered by (float64 distance, original row ID)."""
    if not 0 < k < len(data) or reference_block < 1:
        raise ValueError("require 0 < k < corpus size and positive reference_block")
    query_rows = np.asarray(query_rows, dtype=np.int64)
    if np.any(query_rows < 0) or np.any(query_rows >= len(data)):
        raise ValueError("query row outside corpus")
    if len(original_ids) != len(data) or len(np.unique(original_ids)) != len(data):
        raise ValueError("original IDs must be unique and match corpus size")
    queries = np.asarray(data[query_rows], dtype=np.float64)
    validate_data(queries, metric)
    best_ids = np.empty((len(queries), 0), dtype=np.int64)
    best_distances = np.empty((len(queries), 0), dtype=np.float64)
    for start in range(0, len(data), reference_block):
        stop = min(start + reference_block, len(data))
        reference = np.asarray(data[start:stop], dtype=np.float64)
        validate_data(reference, metric)
        distances = metric_distances(queries, reference, metric)
        self_mask = (query_rows >= start) & (query_rows < stop)
        distances[np.flatnonzero(self_mask), query_rows[self_mask] - start] = np.inf
        merged_distances = np.concatenate((best_distances, distances), axis=1)
        merged_ids = np.concatenate(
            (best_ids, np.broadcast_to(np.arange(start, stop), distances.shape)), axis=1
        )
        width = min(k, merged_distances.shape[1])
        next_ids = np.empty((len(queries), width), dtype=np.int64)
        next_distances = np.empty((len(queries), width), dtype=np.float64)
        for row in range(len(queries)):
            cutoff = np.partition(merged_distances[row], width - 1)[width - 1]
            eligible = np.flatnonzero(merged_distances[row] <= cutoff)
            order = np.lexsort(
                (
                    original_ids[merged_ids[row, eligible]],
                    merged_distances[row, eligible],
                )
            )
            selected = eligible[order[:width]]
            next_ids[row] = merged_ids[row, selected]
            next_distances[row] = merged_distances[row, selected]
        best_ids, best_distances = next_ids, next_distances
    if not np.isfinite(best_distances).all():
        raise ValueError("reference contains non-finite neighbor distances")
    return best_ids, best_distances


def generate(
    source,
    output,
    size=10000,
    sample_rows=2048,
    seed=42,
    k=30,
    query_block=128,
    reference_block=8192,
):
    """size=0 evaluates sampled rows exhaustively against the full training corpus."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("output must not overwrite the source dataset")
    if output.exists():
        with h5py.File(output, "r") as existing:
            if "manifest" not in existing.attrs:
                raise ValueError("existing output is not a reference artifact")
    if query_block < 1 or reference_block < 1:
        raise ValueError("block sizes must be positive")
    started = time.perf_counter()
    with h5py.File(source, "r") as dataset:
        metric = str(dataset.attrs["distance"])
        train = dataset["train"]
        corpus_ids = (
            sample_ids(len(train), size, seed) if size else np.arange(len(train))
        )
        data = np.asarray(train[corpus_ids] if size else train[:], dtype=np.float32)
    validate_data(data, metric)
    if not 0 < k < len(data):
        raise ValueError("require 0 < k < corpus size")
    query_rows = (
        np.arange(len(data)) if size else sample_ids(len(data), sample_rows, seed + 1)
    )
    manifest = {
        "schema": 1,
        "source_sha256": file_hash(source),
        "dataset": source.stem,
        "generator_sha256": file_hash(__file__),
        "metric": metric,
        "size": size,
        "sample_rows": len(query_rows),
        "seed": seed,
        "k": k,
        "input_dtype": "float32",
        "distance_dtype": "float64",
        "self": "exclude same row only",
        "angular_zero_distance": 1.0,
        "zero_rows": int(np.count_nonzero(~np.any(data != 0, axis=1))),
        "ties": "distance then original row ID",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "query_block": query_block,
        "reference_block": reference_block,
    }
    encoded = json.dumps(manifest, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "r+" if output.exists() else "x") as result:
        if "manifest" in result.attrs:
            if result.attrs["manifest"] != encoded:
                raise ValueError(
                    "existing reference manifest differs; choose a new output path"
                )
            np.testing.assert_array_equal(result["corpus_ids"][:], corpus_ids)
            np.testing.assert_array_equal(result["query_rows"][:], query_rows)
        else:
            result.attrs["manifest"] = encoded
            result.attrs["completed_rows"] = 0
            result.attrs["complete"] = False
            result.attrs["compute_seconds"] = 0.0
            result.create_dataset("corpus_ids", data=corpus_ids)
            result.create_dataset("query_rows", data=query_rows)
            result.create_dataset("neighbors", (len(query_rows), k), dtype="i8")
            result.create_dataset("distances", (len(query_rows), k), dtype="f8")
        for start in range(
            int(result.attrs["completed_rows"]), len(query_rows), query_block
        ):
            stop = min(start + query_block, len(query_rows))
            block_started = time.perf_counter()
            neighbors, distances = exact_neighbors(
                data, query_rows[start:stop], corpus_ids, metric, k, reference_block
            )
            result["neighbors"][start:stop] = neighbors
            result["distances"][start:stop] = distances
            result.flush()
            result.attrs["compute_seconds"] += time.perf_counter() - block_started
            result.attrs["completed_rows"] = stop
            result.flush()
            if start == 0 or stop == len(query_rows) or stop % (query_block * 10) == 0:
                print(f"{source.stem}: exact rows {stop}/{len(query_rows)}", flush=True)
        result.attrs["complete"] = True
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "output": str(output),
                    "elapsed_seconds": elapsed,
                    "compute_seconds": float(result.attrs["compute_seconds"]),
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="ann-benchmarks dataset HDF5")
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--size",
        type=int,
        default=10000,
        help="subset size; 0 selects full-corpus samples",
    )
    parser.add_argument("--sample-rows", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--query-block", type=int, default=128)
    parser.add_argument("--reference-block", type=int, default=8192)
    generate(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
