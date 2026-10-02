# RINND

A high-performance Rust implementation of the [NN-Descent](https://dl.acm.org/doi/10.1145/1963405.1963487) algorithm for approximate k-nearest neighbor graph construction and search.
This is an AI enabled port and optimization of [PyNNDescent](https://github.com/lmcinnes/pynndescent) designed to take advantage of the benefits that a rust compiled backend can offer.

## Crate Structure

```
rinnd/
├── crates/
│   ├── rinnd-core/         # Core algorithm library
│   │   ├── src/
│   │   │   ├── distance/   # 30+ distance metrics with SIMD acceleration
│   │   │   ├── heap/       # Neighbor and candidate heaps
│   │   │   ├── graph/      # k-NN graph and CSR search graph
│   │   │   ├── tree/       # Random projection trees
│   │   │   ├── nndescent/  # NN-Descent algorithm
│   │   │   └── search/     # Greedy graph search
│   │   └── benches/        # Criterion benchmarks
│   ├── rinnd-simd/         # SIMD distance kernels (AVX2, AVX-512)
│   └── rinnd/              # Python bindings (PyO3 + maturin)
```

## Features

- **Fast k-NN graph construction** via NN-Descent with random projection tree initialization
- **30+ distance metrics**: Euclidean, Cosine, Inner Product, Manhattan, Chebyshev, Minkowski, Canberra, Bray-Curtis, Hamming, Jaccard, Dice, Correlation, Hellinger, Jensen-Shannon, and many more
- **SIMD acceleration**: AVX2+FMA for Euclidean, Cosine, Inner Product, and Manhattan distances
- **Parallel execution**: Multi-threaded via Rayon
- **Python bindings**: Drop-in use from Python via PyO3/maturin
- **Graph diversification and pruning** for improved recall

## Building

```bash
# Build all crates (release mode recommended)
cargo build --release

# Run tests
cargo test --release

# Run benchmarks
cargo bench --release
```

### Default Optimizations

Default Rust and Python builds enable `batched-euclidean`, `batched-angular`,
`compact-candidates`, `batched-leaves`, `candidate-membership`, and
`candidate-workspace`. These enable distance batching, compact reverse-candidate
storage, batched leaf initialization, vectorizable duplicate checks, and candidate
array reuse with worker-side update-bucket allocation. Unsupported SIMD platforms
retain the existing distance fallbacks.

To disable all six optimizations for a core-only build:

```bash
cargo build -p rinnd-core --release --no-default-features --features std,rayon
```

Rust dependencies can use `default-features = false, features = ["std", "rayon"]`
and add individual optimization features as needed. Cargo features are additive;
another dependency enabling core defaults will re-enable the optimizations.
For Python:

```bash
maturin build --release -m crates/rinnd/Cargo.toml --no-default-features
```

The installed wheel determines the features; there is no runtime toggle.
See [benchmark notes](benchmarks/README.md)
for the measurements and remaining confirmation limitations.

### Experimental Packed Updates

The opt-in `packed-updates` feature stores NN-descent iteration updates in eight
bytes instead of twelve. It encodes a 14-bit source-row offset, two 6-bit candidate
slots, a new/old-list bit, and the original 32-bit distance. Global IDs and public
distances are unchanged; candidate widths above 64 use the unpacked fallback.
Leaf initialization is unchanged. This reduces update-buffer payload by one third,
not total index memory, and the extra candidate lookups can make updates slower.
It is not enabled by default.

```bash
cargo build -p rinnd-core --release --features packed-updates
maturin build --release -m crates/rinnd/Cargo.toml --features packed-updates
```

The existing heap benchmark can compare both formats in the same binary:

```bash
cargo bench -p rinnd-core --bench heap --features bench-internals -- update_replay
```

Direct packing and its buffer-tuning experiment were removed on 2026-10-02:
the roughly 1% best-case gain was inconsistent and sometimes increased memory use.
All builds use fixed 16,384-row source blocks. Candidate-reference packing above
remains opt-in. Historical
[direct-packing results](benchmarks/README.md#direct-packing-and-buffer-size-2026-10-01)
are retained.

### FP16 Experiments (Removed)

The experimental FP16 update and vector-storage features were removed on 2026-10-02.
Full-corpus screening found no delivery speedup and recall losses above the 0.001
allowance at some endpoints. The historical
[FP16 results](benchmarks/README.md#fp16-construction-screen-2026-10-01) are retained.

### Candidate Membership

The default-enabled `candidate-membership` feature uses a vectorizable integer equality
reduction for candidate-heap duplicate checks at widths of 16 or more, retaining
the scalar early-exit scan below that. Candidate insertion order, heap priorities,
RNG consumption and FP32 distances are unchanged. No update packing is enabled.

Independent full-corpus paired screening found 1.030x graph-delivery speedup for GloVe k30 and
1.017x for NYTimes k30, with exact graph outputs. Degree-15 and image delivery were
near neutral. All builds expose
per-iteration candidate subphase timings in `build_stats["candidate_phases"]`;
these are components of `candidate_seconds`, not additional build time.
See the [candidate results](benchmarks/README.md#candidate-profiling-and-membership-2026-10-01)
for phase attribution, variability and reproduction commands.

### Candidate Workspace

The default-enabled `candidate-workspace` feature reuses the four candidate
index/priority arrays across NN-descent iterations. Arrays are reset before each
build; reverse-edge buckets are still released each iteration. Candidate order,
flags, RNG state and FP32 graph distances are unchanged.

Initial independent screening found roughly 1-2% delivery improvements on GloVe and
NYTimes, but mixed image results, including a 2% slowdown on Fashion-MNIST k15.
Priority arrays remain live during updates, so lower allocation cost does not imply
lower memory use. Final workspace destruction is included in the last candidate
release phase and in complete delivery time.
See the [workspace results](benchmarks/README.md#candidate-workspace-reuse-2026-10-01).

The initial combination of `candidate-membership,candidate-workspace` gave 1.046x paired
graph-delivery speedup on GloVe k30 and 1.037x on NYTimes k30 in the combined
full-corpus screen, with exact graph outputs. That initial screen found a
Fashion-MNIST k15 regression (0.972x median paired ratio).
See the [combined results](benchmarks/README.md#combined-candidate-features-2026-10-02)
for timing ranges, memory costs and reproduction commands. `candidate-combined`
is only a benchmark-runner option, not a Cargo feature.

The subsequent [Fashion-MNIST investigation](benchmarks/README.md#fashion-mnist-allocation-regression-2026-10-02)
localized the extra time to update-bucket allocation. With `candidate-workspace`,
these buckets are now allocated on Rayon workers, preserving producer order.
Direct before/after k15 delivery improved from 0.928 to 0.903 seconds with exact
outputs; peak RSS fell from 391 to 372 MB on this host. Both features, including
this fix, were promoted to defaults on 2026-10-02. Full graph indices and distance
bits remained identical in the tested builds, so graph quality was unchanged.
Timing varies and some workloads use slightly more peak memory; this is not a
universal speed or memory guarantee. Candidate-reference packing remains opt-in.

## Python Bindings

### Install

```bash
# Requires maturin: pip install maturin
maturin develop --release -m crates/rinnd/Cargo.toml
```

### Usage

```python
import numpy as np
import rinnd

# Check whether a named distance is supported
assert "euclidean" in rinnd.named_distances

# Build index
data = np.random.rand(10000, 128).astype(np.float32)
index = rinnd.RINND(data, metric="euclidean", n_neighbors=15)

# Get the k-NN graph
indices, distances = index.neighbor_graph

# Query new points
query = np.random.rand(100, 128).astype(np.float32)
indices, distances = index.query(query, k=10)

# Check SIMD support
print(rinnd.simd_info())
```

Pass `n_jobs=-1` to use all logical cores available to the process, or a
positive integer to give an index its own pool with exactly that many worker
threads:

```python
index = rinnd.RINND(data, n_neighbors=15, n_jobs=4)
```

An explicit `n_jobs` setting applies to index construction, lazy preparation,
and queries. Omitting `n_jobs`, or passing `None`, preserves the existing Rayon
global-pool behavior, including `RAYON_NUM_THREADS`. Values of `0` and values
less than `-1` are rejected.

Construction produces the k-NN graph without building query-only structures.
Accessing `neighbor_graph` does not trigger that extra work. Call
`index.prepare()` explicitly when query latency must not include preparation;
otherwise the first query prepares the index automatically. Quantized index
modes currently prepare eagerly because encoding depends on the final physical
layout.

For cosine data that is already row-normalized, set `input_normalized=True` to
avoid repeating normalization:

```python
index = rinnd.RINND(
    normalized_data,
    metric="cosine",
    normalize=True,
    input_normalized=True,
    n_neighbors=15,
)
```

RINND still normalizes query vectors when `normalize=True`. The input array is
copied into Rust-owned memory because retaining borrowed NumPy memory after the
constructor returns would be unsafe.

To construct and retain only the k-NN graph, use `graph_only=True`:

```python
index = rinnd.RINND(data, n_neighbors=15, graph_only=True)
indices, distances = index.neighbor_graph
```

Graph-only construction borrows a C-contiguous input array synchronously when
no normalization is needed, then retains only the graph arrays. It does not
retain the input data or query structures, so `query()`, `prepare()`,
`search_graph`, and search-tree export are disabled. `storage_info()` reports
`graph_only=True`, graph allocation bytes, and zero retained FP32 data bytes.
Graph-only indexes use breadth-first RP forest construction with a tree batch
width of 6 by default. Set `tree_build_strategy="depth_first"` to use the
legacy builder, or set `tree_batch_width` to tune breadth-first batching.

## Rust Usage

Add `rinnd-core` as a dependency in your `Cargo.toml`:

```toml
[dependencies]
rinnd-core = { path = "crates/rinnd-core" }
```

```rust
use rinnd_core::index::NNDescentBuilder;
use rinnd_core::distance::SquaredEuclidean;

let data: Vec<f32> = /* your data */;
let n_points = 10000;
let dim = 128;

let index = NNDescentBuilder::new(&data, n_points, dim)
    .n_neighbors(15)
    .build::<SquaredEuclidean>();

let (indices, distances) = index.query(&query_data, n_queries, k, epsilon);
```

## License

BSD-2-Clause. See [LICENSE](LICENSE) for details.
