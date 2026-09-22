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
