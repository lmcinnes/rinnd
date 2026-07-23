# RINND

A high-performance Rust implementation of the [NN-Descent](https://dl.acm.org/doi/10.1145/1963405.1963487) algorithm for approximate k-nearest neighbor graph construction and search.

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
