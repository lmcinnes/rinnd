"""Build isolated feature combinations and run the existing recall benchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

import numpy as np
from scipy.stats import t as student_t

from benchmark_recall import ROOT, paired_gate
from ground_truth import DATASETS, file_hash

FEATURES = (
    "batched-euclidean",
    "batched-angular",
    "compact-candidates",
    "batched-leaves",
)


def features(mask):
    if not 0 <= mask < 16:
        raise ValueError("feature mask must be between 0 and 15")
    return [feature for bit, feature in enumerate(FEATURES) if mask & (1 << bit)]


def source_hash():
    paths = [ROOT / "Cargo.toml", ROOT / "Cargo.lock"]
    paths += sorted((ROOT / "crates").glob("**/*.rs"))
    paths += sorted((ROOT / "crates").glob("**/Cargo.toml"))
    return hashlib.sha256(
        json.dumps(
            {
                str(path.relative_to(ROOT)): file_hash(path)
                for path in paths
                if path.exists()
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def execute(command, log, env=None):
    with log.open("w") as output:
        result = subprocess.run(
            command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def build(args):
    for mask in args.masks:
        folder = args.artifacts / f"mask-{mask:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder / "build.json"
        identity = {
            "features": features(mask),
            "source_sha256": source_hash(),
            "python": str(args.python.resolve()),
            "rustflags": os.environ.get("RUSTFLAGS"),
        }
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved["identity"] != identity:
                raise ValueError(
                    f"build inputs changed; use a new artifact directory: {folder}"
                )
            for path, expected in saved["binary_hashes"].items():
                if file_hash(Path(path)) != expected:
                    raise ValueError(f"binary changed: {path}")
            print(f"Reusing mask {mask:02d}", flush=True)
            continue
        selected = [f"rinnd-core/{feature}" for feature in features(mask)]
        command = [
            "uv",
            "run",
            "--directory",
            str(ROOT / "ann-benchmarks"),
            "--with",
            "maturin",
            "maturin",
            "build",
            "--release",
            "--manifest-path",
            str(ROOT / "crates/rinnd/Cargo.toml"),
            "--no-default-features",
            "--interpreter",
            str(args.python),
            "--out",
            str(folder / "wheels"),
        ]
        if selected:
            command += ["--features", ",".join(selected)]
        execute(command, folder / "build.log")
        wheels = list((folder / "wheels").glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError(f"expected one wheel in {folder}")
        execute(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(args.python),
                "--target",
                str(folder / "python"),
                "--no-deps",
                "--reinstall",
                str(wheels[0]),
            ],
            folder / "install.log",
        )
        execute(
            [
                "cargo",
                "test",
                "-p",
                "rinnd-core",
                "--release",
                "--no-default-features",
                "--features",
                ",".join(["std", "rayon"] + features(mask)),
                "--quiet",
            ],
            folder / "test.log",
        )
        binaries = list((folder / "python/rinnd").glob("*.so"))
        if not binaries:
            raise ValueError(f"no extension found in {folder}")
        with manifest.open("x") as output:
            json.dump(
                {
                    "identity": identity,
                    "binary_hashes": {str(path): file_hash(path) for path in binaries},
                    "wheel_sha256": file_hash(wheels[0]),
                },
                output,
                indent=2,
            )
        print(f"Built and tested mask {mask:02d}: {features(mask)}", flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for name in args.datasets:
            order = list(args.masks)
            if len(order) == 2:
                if seed % 2:
                    order.reverse()
            else:
                random.Random(f"{seed}:{name}:{args.mode}").shuffle(order)
            for mask in order:
                folder = args.artifacts / f"mask-{mask:02d}"
                build_info = json.loads((folder / "build.json").read_text())
                for path, expected in build_info["binary_hashes"].items():
                    if file_hash(Path(path)) != expected:
                        raise ValueError(f"binary changed: {path}")
                result = args.output / f"{args.mode}-{name}-m{mask:02d}-s{seed}.json"
                command = [
                    str(args.python),
                    str(ROOT / "benchmarks/benchmark_recall.py"),
                    args.mode,
                    str(ROOT / "ann-benchmarks/data" / f"{name}.hdf5"),
                    str(result),
                    "--threads",
                    str(args.threads),
                    "--seeds",
                    str(seed),
                    "--repeats",
                    str(args.repeats),
                ]
                if args.mode == "graph":
                    command += [
                        "--reference",
                        str(args.references / f"{name}-{args.reference_suffix}.hdf5"),
                    ]
                elif args.mode == "query":
                    command += [
                        "--split",
                        args.split,
                        "--query-limit",
                        str(args.query_limit),
                    ]
                identity = {
                    "command": command,
                    "build": build_info,
                    "harness": {
                        str(path): file_hash(path)
                        for path in (
                            ROOT / "benchmarks/benchmark_recall.py",
                            ROOT / "benchmarks/ground_truth.py",
                            ROOT
                            / "ann-benchmarks/ann_benchmarks/algorithms/rinnd/module.py",
                        )
                    },
                    "affinity": sorted(os.sched_getaffinity(0)),
                    "threads": {
                        key: os.environ.get(key)
                        for key in (
                            "OPENBLAS_NUM_THREADS",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "RAYON_NUM_THREADS",
                        )
                    },
                }
                job = result.with_suffix(".job.json")
                if job.exists():
                    if json.loads(job.read_text()) != identity:
                        raise ValueError(f"job inputs changed: {job}")
                else:
                    with job.open("x") as output:
                        json.dump(identity, output, indent=2)
                if not result.exists():
                    execute(
                        command,
                        result.with_suffix(".log"),
                        dict(os.environ, PYTHONPATH=str(folder / "python")),
                    )
                report = json.loads(result.read_text())
                if report["provenance"]["binary_hashes"] != build_info["binary_hashes"]:
                    raise ValueError(f"unexpected imported binary: {result}")
                print(f"Finished {result.name}", flush=True)


def compare(baseline, candidate):
    keys = ("dataset_sha256", "mode", "threads", "parameters", "graph_only", "detail")
    if any(baseline[key] != candidate[key] for key in keys):
        raise ValueError("incompatible benchmark inputs")

    def indexed(report):
        records = report["records"]
        result = {
            (record["case"], record["seed"], record["repetition"]): record
            for record in records
        }
        if len(result) != len(records):
            raise ValueError("duplicate records")
        return result

    before, after = indexed(baseline), indexed(candidate)
    if before.keys() != after.keys() or not before:
        raise ValueError("unpaired records")
    ratios = {}
    hash_key = "query_sha256" if baseline["mode"] == "query" else "graph_sha256"
    for key, record in before.items():
        other = after[key]
        if record[hash_key] != other[hash_key] or record["recalls"] != other["recalls"]:
            raise ValueError(f"outputs or recalls differ: {key}")
        ratios.setdefault(key[0], []).append(
            record["elapsed_seconds"] / other["elapsed_seconds"]
        )
    return {case: statistics.median(values) for case, values in ratios.items()}


def summarize(args):
    for name in args.datasets:
        for mask in args.masks:
            collected = {}
            for seed in args.seeds:
                prefix = f"{args.mode}-{name}"
                before = json.loads(
                    (args.output / f"{prefix}-m00-s{seed}.json").read_text()
                )
                after = json.loads(
                    (args.output / f"{prefix}-m{mask:02d}-s{seed}.json").read_text()
                )
                for case, ratio in compare(before, after).items():
                    collected.setdefault(case, []).append(ratio)
            print(
                json.dumps(
                    {
                        "dataset": name,
                        "mask": mask,
                        "features": features(mask),
                        "identical": True,
                        "speedups": {
                            case: round(statistics.median(values), 4)
                            for case, values in collected.items()
                        },
                    }
                ),
                flush=True,
            )


def confirmation_endpoint(pairs, recall_family=36, timing_family=28):
    if len(pairs) < 10:
        raise ValueError("confirmation requires at least ten paired seeds")
    seed_ids = set()
    before_means, after_means, log_speedups, tail_speedups = {}, {}, [], []
    for baseline, candidate in pairs:
        before = sorted(baseline, key=lambda record: record["repetition"])
        after = sorted(candidate, key=lambda record: record["repetition"])
        if len(before) < 5 or len(before) != len(after):
            raise ValueError("confirmation requires at least five paired repetitions")
        if [record["repetition"] for record in before] != list(range(len(before))):
            raise ValueError("missing or duplicate repetitions")
        if [(record["seed"], record["repetition"]) for record in before] != [
            (record["seed"], record["repetition"]) for record in after
        ]:
            raise ValueError("unpaired confirmation records")
        seed = before[0]["seed"]
        if seed in seed_ids or any(record["seed"] != seed for record in before):
            raise ValueError("duplicate or mixed seeds")
        seed_ids.add(seed)
        for metric in before[0]["recalls"]:
            before_means.setdefault(metric, []).append(
                float(np.mean([record["recalls"][metric] for record in before]))
            )
            after_means.setdefault(metric, []).append(
                float(np.mean([record["recalls"][metric] for record in after]))
            )
        log_speedups.append(
            np.log(
                statistics.median(record["elapsed_seconds"] for record in before)
                / statistics.median(record["elapsed_seconds"] for record in after)
            )
        )
        if "latency_seconds" in before[0]:
            tail_speedups.append(
                statistics.median(
                    np.quantile(record["latency_seconds"], 0.99) for record in before
                )
                / statistics.median(
                    np.quantile(record["latency_seconds"], 0.99) for record in after
                )
            )
    if not before_means:
        raise ValueError("confirmation requires recall measurements")
    error = float(
        student_t.ppf(1 - 0.05 / timing_family, len(pairs) - 1)
        * np.std(log_speedups, ddof=1)
        / np.sqrt(len(pairs))
    )
    speedup = float(np.exp(np.mean(log_speedups)))
    lower = float(np.exp(np.mean(log_speedups) - error))
    return {
        "recall": {
            metric: paired_gate(
                before_means[metric], after_means[metric], family_size=recall_family
            )
            for metric in before_means
        },
        "speedup": speedup,
        "speedup_lower_bound": lower,
        "timing_status": (
            "non_regressing" if lower >= 1 else "inconclusive_or_regressing"
        ),
        "median_p99_speedup": (
            statistics.median(tail_speedups) if tail_speedups else None
        ),
    }


def confirm(args):
    endpoints = []
    for name in DATASETS:
        for mode in ("graph", "query"):
            cases = {}
            for seed in args.seeds:
                prefix = f"{mode}-{name}"
                before = json.loads(
                    (args.output / f"{prefix}-m00-s{seed}.json").read_text()
                )
                after = json.loads(
                    (args.output / f"{prefix}-m15-s{seed}.json").read_text()
                )
                compare(before, after)
                if mode == "query" and before["detail"]["split"] != "confirmation":
                    raise ValueError("confirmation requires held-out queries")
                if (
                    mode == "graph"
                    and before["detail"]["reference_manifest"]["size"] != 0
                ):
                    raise ValueError("confirmation requires full-corpus sampled truth")
                for case in dict.fromkeys(
                    record["case"] for record in before["records"]
                ):
                    cases.setdefault(case, []).append(
                        (
                            [
                                record
                                for record in before["records"]
                                if record["case"] == case
                            ],
                            [
                                record
                                for record in after["records"]
                                if record["case"] == case
                            ],
                        )
                    )
            for case, pairs in cases.items():
                endpoints.append(
                    {"dataset": name, "case": case, **confirmation_endpoint(pairs)}
                )
    if (
        len(endpoints) != 28
        or sum(len(endpoint["recall"]) for endpoint in endpoints) != 36
    ):
        raise ValueError("unexpected multiplicity family")
    result = {
        "candidate_mask": 15,
        "recall_family": 36,
        "timing_family": 28,
        "recall_status": (
            "non_inferior"
            if all(
                gate["status"] == "non_inferior"
                for endpoint in endpoints
                for gate in endpoint["recall"].values()
            )
            else "blocked"
        ),
        "timing_status": (
            "non_regressing"
            if all(
                endpoint["timing_status"] == "non_regressing" for endpoint in endpoints
            )
            else "blocked"
        ),
        "inference": "Paired seed means conditional on sampled rows; exact full graph fingerprints also checked.",
        "endpoints": endpoints,
    }
    with (args.output / "confirmation.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "run", "summarize", "confirm"))
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/results/feature-matrix-screen"
    )
    parser.add_argument(
        "--python", type=Path, default=ROOT / "ann-benchmarks/.venv/bin/python"
    )
    parser.add_argument(
        "--masks", type=int, nargs="+", default=list(range(16)), choices=range(16)
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS)
    )
    parser.add_argument(
        "--mode", choices=("graph", "profile", "query"), default="graph"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--split", choices=("tuning", "confirmation"), default="tuning")
    parser.add_argument("--query-limit", type=int, default=200)
    parser.add_argument(
        "--references", type=Path, default=ROOT / "benchmarks/results/ground_truth"
    )
    parser.add_argument("--reference-suffix", default="n10000-k30")
    args = parser.parse_args()
    args.artifacts, args.output = args.artifacts.resolve(), args.output.resolve()
    {"build": build, "run": run, "summarize": summarize, "confirm": confirm}[
        args.action
    ](args)


if __name__ == "__main__":
    main()
