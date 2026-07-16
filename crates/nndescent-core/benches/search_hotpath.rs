//! Benchmarks for greedy search hot path.
//!
//! This isolates search-loop overhead (visited checks, heap ops, neighbor scans)
//! by comparing a zero-cost distance implementation against real cosine distance.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nndescent_core::distance::{Cosine, Distance};
use nndescent_core::graph::SearchGraph;
use nndescent_core::heap::{BoundedHeap, CandidateHeap};
use nndescent_core::rng::FastRng;
use nndescent_core::search::{greedy_search_with_workspace, SearchWorkspace};
use nndescent_core::visited::VisitedSet;

#[derive(Clone)]
struct ZeroDistance;

impl Distance<f32> for ZeroDistance {
    #[inline]
    fn distance(&self, _a: &[f32], _b: &[f32]) -> f32 {
        // Constant positive value to keep branch behavior realistic.
        0.5
    }

    fn name(&self) -> &'static str {
        "zero"
    }
}

fn build_data(n_points: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_points * dim];
    for i in 0..n_points {
        for j in 0..dim {
            let x = ((i * 131 + j * 17) % 1024) as f32 / 1024.0;
            out[i * dim + j] = x;
        }
    }
    out
}

fn build_random_knn_graph(n_points: usize, degree: usize) -> SearchGraph {
    let mut rng = FastRng::new(42);
    let mut dense = vec![-1i32; n_points * degree];

    for i in 0..n_points {
        for j in 0..degree {
            let mut v = rng.next_index(n_points) as i32;
            if v == i as i32 {
                v = ((v as usize + 1) % n_points) as i32;
            }
            dense[i * degree + j] = v;
        }
    }

    SearchGraph::from_dense(&dense, n_points, degree)
}

fn bench_search_hotpath(c: &mut Criterion) {
    let n_points = 20_000;
    let dim = 100;
    let degree = 40;
    let k = 10;
    let epsilon = 0.1;
    let min_distance = 1.0;

    let data = build_data(n_points, dim);
    let graph = build_random_knn_graph(n_points, degree);
    let query = data[0..dim].to_vec();

    let mut group = c.benchmark_group("search_hotpath");

    group.bench_function("zero_distance", |b| {
        let distance = ZeroDistance;
        let mut workspace = SearchWorkspace::new(n_points, k);
        let mut rng = FastRng::new(123);
        b.iter(|| {
            let (idx, dist) = greedy_search_with_workspace(
                black_box(&query),
                black_box(&data),
                dim,
                black_box(&graph),
                None,
                &distance,
                k,
                epsilon,
                min_distance,
                &mut rng,
                &mut workspace,
            );
            black_box((idx, dist));
        });
    });

    group.bench_function("cosine_distance", |b| {
        let distance = Cosine;
        let mut workspace = SearchWorkspace::new(n_points, k);
        let mut rng = FastRng::new(123);
        b.iter(|| {
            let (idx, dist) = greedy_search_with_workspace(
                black_box(&query),
                black_box(&data),
                dim,
                black_box(&graph),
                None,
                &distance,
                k,
                epsilon,
                min_distance,
                &mut rng,
                &mut workspace,
            );
            black_box((idx, dist));
        });
    });

    group.finish();
}

fn bench_search_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_primitives");

    // Roughly matches observed per-query scale from profile counters.
    let n_points = 20_000;
    let mut ids = Vec::with_capacity(512);
    for i in 0..470 {
        ids.push((i * 37 % n_points) as i32);
    }
    for i in 0..42 {
        // Add repeated ids to emulate revisit rate.
        ids.push((i * 37 % n_points) as i32);
    }

    group.bench_function("visited_check_and_mark_512", |b| {
        let mut visited = VisitedSet::new(n_points);
        b.iter(|| {
            visited.clear();
            let mut already = 0usize;
            for &idx in &ids {
                if visited.check_and_mark(idx) {
                    already += 1;
                }
            }
            black_box(already);
        });
    });

    group.bench_function("candidate_heap_57_push_20_pop", |b| {
        b.iter(|| {
            let mut h = CandidateHeap::with_capacity(96);
            for i in 0..57 {
                h.push((i as f32) * 0.01, i as i32);
            }
            let mut acc = 0.0f32;
            for _ in 0..20 {
                if let Some((d, _)) = h.pop() {
                    acc += d;
                }
            }
            black_box(acc);
        });
    });

    group.bench_function("bounded_heap_58_push_k10", |b| {
        b.iter(|| {
            let mut h = BoundedHeap::new(10);
            let mut inserted = 0usize;
            for i in 0..58 {
                // Monotone-ish decreasing pattern causes non-trivial insertion churn.
                let d = 1.0 - (i as f32) * 0.0125;
                if h.push(d, i as i32) {
                    inserted += 1;
                }
            }
            black_box((inserted, h.max_distance()));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_search_hotpath, bench_search_primitives);
criterion_main!(benches);
