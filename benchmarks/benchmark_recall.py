"""Measure graph delivery and official-test query recall without mixing workloads."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import h5py
import numpy as np
from scipy.stats import t as student_t

from ground_truth import file_hash, metric_distances

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ann-benchmarks"))


def graph_recall(data, query_rows, truth_ids, truth_distances, returned, metric, k):
    """Both measures penalize missing/duplicate/self IDs; ties use a fixed 1e-12 tolerance."""
    if len(returned) != len(query_rows) or truth_ids.shape[1] < k:
        raise ValueError("reference and returned graph shapes do not match")
    strict = np.zeros(len(query_rows))
    tied = np.zeros(len(query_rows))
    for row, query_row in enumerate(query_rows):
        candidates = np.asarray(returned[row, :k])
        if not np.issubdtype(candidates.dtype, np.integer):
            raise ValueError("neighbor IDs must be integers")
        candidates = np.unique(
            candidates[
                (candidates >= 0) & (candidates < len(data)) & (candidates != query_row)
            ]
        )
        strict[row] = np.intersect1d(candidates, truth_ids[row, :k]).size / k
        if candidates.size:
            distances = metric_distances(
                data[query_row : query_row + 1].astype(np.float64),
                data[candidates].astype(np.float64),
                metric,
            )[0]
            tied[row] = (
                np.count_nonzero(distances <= truth_distances[row, k - 1] + 1e-12) / k
            )
    return {"strict": strict.tolist(), "tie_aware": tied.tolist()}


def query_ids(count, split, limit):
    ordered = np.random.default_rng(20260930).permutation(count)
    boundary = count // 5
    selected = {
        "tuning": ordered[:boundary],
        "confirmation": ordered[boundary:],
        "all": ordered,
    }[split]
    if limit:
        if split != "tuning" or limit > len(selected) or limit < 1:
            raise ValueError("query limits are only allowed within the tuning split")
        selected = selected[:limit]
    if not len(selected):
        raise ValueError("empty query split")
    return selected


def provenance():
    import rinnd

    package = Path(rinnd.__file__).resolve()
    return {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
        "source_hashes": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in (
                Path(__file__),
                ROOT / "benchmarks/ground_truth.py",
                ROOT / "ann-benchmarks/ann_benchmarks/algorithms/rinnd/module.py",
            )
        },
        "rinnd_module": str(package),
        "binary_hashes": {
            str(path): file_hash(path) for path in package.parent.glob("*.so")
        },
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "simd": rinnd.simd_info(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "RAYON_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "RUSTFLAGS",
            )
        },
    }


def measure_graph(args, dataset, metric):
    import rinnd

    profiling = args.mode == "profile"
    if profiling:
        data = np.asarray(dataset["train"], dtype=np.float32)
        detail = {"recall_status": "not_evaluated", "corpus_size": len(data)}
    else:
        if args.reference is None:
            raise ValueError("graph mode requires --reference")
        with h5py.File(args.reference, "r") as reference:
            manifest = json.loads(reference.attrs["manifest"])
            if not reference.attrs["complete"] or reference.attrs[
                "completed_rows"
            ] != len(reference["query_rows"]):
                raise ValueError("reference generation is incomplete")
            if (
                manifest["source_sha256"] != args.dataset_hash
                or manifest["metric"] != metric
            ):
                raise ValueError("reference belongs to a different dataset")
            corpus_ids = reference["corpus_ids"][:]
            rows = reference["query_rows"][:]
            truth_ids, truth_distances = (
                reference["neighbors"][:],
                reference["distances"][:],
            )
        data = np.asarray(dataset["train"][corpus_ids], dtype=np.float32)
        detail = {
            "reference_sha256": file_hash(args.reference),
            "reference_manifest": manifest,
        }
    records = []
    for seed in args.seeds:
        for k in (15, 30):
            for repetition in range(args.repeats):
                gc.collect()
                started = time.perf_counter()
                index = rinnd.RINND(
                    data,
                    metric="cosine" if metric == "angular" else metric,
                    normalize=metric == "angular",
                    graph_only=args.graph_only,
                    n_neighbors=k,
                    random_state=seed,
                    n_jobs=args.threads,
                    **args.parameters,
                )
                built = time.perf_counter()
                indices, distances = index.neighbor_graph
                finished = time.perf_counter()
                graph_hash = hashlib.sha256(
                    indices.tobytes() + distances.tobytes()
                ).hexdigest()
                recalls = (
                    {}
                    if profiling
                    else graph_recall(
                        data, rows, truth_ids, truth_distances, indices[rows], metric, k
                    )
                )
                record = {
                    "case": f"graph-k{k}",
                    "seed": seed,
                    "repetition": repetition,
                    "elapsed_seconds": finished - started,
                    "export_seconds": finished - built,
                    "returned_bytes": indices.nbytes + distances.nbytes,
                    "graph_sha256": graph_hash,
                    "build_stats": dict(index.build_stats),
                    "recalls": recalls,
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in ("case", "seed", "elapsed_seconds")
                        }
                        | {
                            "recall": {
                                key: float(np.mean(value))
                                for key, value in recalls.items()
                            }
                        }
                    ),
                    flush=True,
                )
                del index, indices, distances
    return records, detail


def measure_query(args, dataset, metric):
    from ann_benchmarks.algorithms.rinnd.module import RINND
    from ann_benchmarks.distance import metrics
    from ann_benchmarks.plotting.metrics import get_recall_values, knn_threshold

    data = np.asarray(dataset["train"], dtype=np.float32)
    rows = query_ids(len(dataset["test"]), args.split, args.query_limit)
    queries = np.asarray(dataset["test"], dtype=np.float32)[rows]
    all_truth = dataset["distances"][:]
    truth = all_truth[rows]
    if truth.shape[1] < 10:
        raise ValueError("official query ground truth has fewer than 10 neighbors")
    records = []
    for seed in args.seeds:
        parameters = dict(args.parameters, random_state=seed, n_jobs=args.threads)
        adapter = RINND(metric, parameters)
        started = time.perf_counter()
        adapter.fit(data)
        build_seconds = time.perf_counter() - started
        for epsilon in args.epsilons:
            adapter.set_query_arguments(epsilon)
            adapter.query(queries[0], 10)
            for repetition in range(args.repeats):
                times = np.empty(len(queries))
                observed = np.full((len(queries), 10), np.nan)
                result_hash = hashlib.sha256()
                for row, query in enumerate(queries):
                    started = time.perf_counter()
                    indices = adapter.query(query, 10)
                    times[row] = time.perf_counter() - started
                    indices = np.asarray(indices)
                    if (
                        not np.issubdtype(indices.dtype, np.integer)
                        or np.any(indices < 0)
                        or np.any(indices >= len(data))
                    ):
                        raise ValueError("query returned invalid indices")
                    if len(indices) > 10 or len(np.unique(indices)) != len(indices):
                        raise ValueError("query returned excess or duplicate indices")
                    result_hash.update(
                        np.asarray([len(indices)], dtype="<i8").tobytes()
                    )
                    result_hash.update(np.asarray(indices, dtype="<i8").tobytes())
                    with np.errstate(invalid="ignore", divide="ignore"):
                        observed[row, : len(indices)] = [
                            metrics[metric].distance(query, data[neighbor])
                            for neighbor in indices
                        ]
                mean, _, counts = get_recall_values(truth, observed, 10, knn_threshold)
                record = {
                    "case": f"query-k10-epsilon{epsilon}",
                    "seed": seed,
                    "repetition": repetition,
                    "elapsed_seconds": float(np.mean(times)),
                    "qps": float(1 / np.mean(times)),
                    "query_ready_build_seconds": build_seconds,
                    "query_sha256": result_hash.hexdigest(),
                    "latency_seconds": times.tolist(),
                    "recalls": {"official": (counts / 10).tolist()},
                    "label": str(adapter),
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "case": record["case"],
                            "seed": seed,
                            "qps": record["qps"],
                            "recall": mean,
                            "build_seconds": build_seconds,
                        }
                    ),
                    flush=True,
                )
        del adapter
        gc.collect()
    return records, {
        "query_rows": rows.tolist(),
        "split": args.split,
        "invalid_truth_rows": rows[~np.isfinite(truth[:, 9])].tolist(),
        "total_invalid_truth_rows": int(
            np.count_nonzero(~np.isfinite(all_truth[:, 9]))
        ),
    }


def paired_gate(
    baseline, candidate, margin=0.001, alpha=0.05, family_size=1, minimum_seeds=10
):
    """Paired seed-level Student-t bound; inference is conditional on the evaluated rows."""
    baseline, candidate = np.asarray(baseline, dtype=float), np.asarray(
        candidate, dtype=float
    )
    if baseline.shape != candidate.shape or baseline.ndim != 1 or not baseline.size:
        raise ValueError("expected equally sized nonempty seed means")
    if not np.isfinite(baseline).all() or not np.isfinite(candidate).all():
        raise ValueError("non-finite recall")
    if margin < 0 or not 0 < alpha < 1 or family_size < 1:
        raise ValueError("invalid gate parameters")
    difference = candidate - baseline
    if len(difference) < minimum_seeds:
        return {"status": "insufficient_seeds", "seeds": len(difference)}
    error = float(
        student_t.ppf(1 - alpha / family_size, len(difference) - 1)
        * np.std(difference, ddof=1)
        / np.sqrt(len(difference))
    )
    lower = float(np.mean(difference) - error)
    return {
        "status": "non_inferior" if lower >= -margin else "blocked",
        "lower_bound": lower,
        "mean_difference": float(np.mean(difference)),
        "margin": margin,
        "seeds": len(difference),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("graph", "query", "profile"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parameters", type=json.loads, default={})
    parser.add_argument("--ordinary-index", action="store_false", dest="graph_only")
    parser.add_argument(
        "--epsilons", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.6, 1.0]
    )
    parser.add_argument(
        "--split", choices=("tuning", "confirmation", "all"), default="tuning"
    )
    parser.add_argument("--query-limit", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a new run path")
    if args.mode == "profile" and args.reference is not None:
        parser.error("profile mode uses the full corpus without a recall reference")
    if args.threads < 1 or args.repeats < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("positive thread/repetition counts and distinct seeds required")
    if not isinstance(args.parameters, dict):
        parser.error("parameters must be a JSON object")
    reserved = {
        "metric",
        "normalize",
        "graph_only",
        "random_state",
        "n_jobs",
        "verbose",
        "input_normalized",
    }
    if args.mode in ("graph", "profile"):
        reserved.add("n_neighbors")
    if reserved.intersection(args.parameters):
        parser.error("parameters override protocol-controlled fields")
    args.dataset_hash = file_hash(args.source)
    metadata = provenance()
    started = time.perf_counter()
    with h5py.File(args.source, "r") as dataset:
        metric = str(dataset.attrs["distance"])
        records, detail = (measure_query if args.mode == "query" else measure_graph)(
            args, dataset, metric
        )
    result = {
        "schema": 1,
        "dataset": args.source.stem,
        "dataset_sha256": args.dataset_hash,
        "mode": args.mode,
        "metric": metric,
        "threads": args.threads,
        "graph_only": args.graph_only,
        "parameters": args.parameters,
        "provenance": metadata,
        "detail": detail,
        "records": records,
        "wall_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "promotion_status": "exploratory_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, allow_nan=False)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
