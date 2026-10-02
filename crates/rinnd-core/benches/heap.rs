//! Benchmarks for heap data structures.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rinnd_core::heap::NeighborHeap;
use rinnd_core::rng::TauRand;

fn bench_neighbor_heap_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_heap_push");

    for n_neighbors in [10, 30, 50, 100].iter() {
        let n_points = 1000;

        group.bench_with_input(
            BenchmarkId::new("random_push", n_neighbors),
            n_neighbors,
            |bench, &k| {
                let mut rng = TauRand::new(42);
                bench.iter(|| {
                    let mut heap = NeighborHeap::new(n_points, k);
                    // Push random neighbors
                    for i in 0..n_points {
                        for _ in 0..k * 2 {
                            let neighbor = (rng.next_int() as usize) % n_points;
                            let dist = (rng.next_int() as f32).abs() / (i32::MAX as f32);
                            heap.checked_flagged_push(i, neighbor as i32, dist, true);
                        }
                    }
                    black_box(heap)
                })
            },
        );
    }

    group.finish();
}

fn bench_neighbor_heap_deheap(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbor_heap_deheap");

    for n_neighbors in [10, 30, 50].iter() {
        let n_points = 1000;

        group.bench_with_input(
            BenchmarkId::new("deheap_sort", n_neighbors),
            n_neighbors,
            |bench, &k| {
                let mut rng = TauRand::new(42);

                // Pre-build heap
                let mut template_heap = NeighborHeap::new(n_points, k);
                for i in 0..n_points {
                    for _ in 0..k * 2 {
                        let neighbor = (rng.next_int() as usize) % n_points;
                        let dist = (rng.next_int() as f32).abs() / (i32::MAX as f32);
                        template_heap.checked_flagged_push(i, neighbor as i32, dist, true);
                    }
                }

                bench.iter(|| {
                    let heap = template_heap.clone();
                    for point in 0..n_points {
                        black_box(heap.deheap_sort(point));
                    }
                    black_box(heap)
                })
            },
        );
    }

    group.finish();
}

fn bench_update_replay(criterion: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        use criterion::BatchSize;
        use rinnd_core::distance::{Distance, SquaredEuclidean};
        use rinnd_core::nndescent::{benchmark, CandidateSets};
        use rinnd_core::rng::FastRng;

        let mut group = criterion.benchmark_group("update_replay");
        group.sample_size(10);
        for n_points in [256, 4096] {
            let dimension = 100;
            let mut rng = FastRng::new(42);
            let data: Vec<f32> = (0..n_points * dimension).map(|_| rng.next_float()).collect();
            for degree in [15, 30] {
                let mut template = NeighborHeap::new(n_points, degree);
                for point in 0..n_points {
                    for _ in 0..degree {
                        let neighbor = rng.next_u64() as usize % n_points;
                        let distance = SquaredEuclidean.distance(
                            &data[point * dimension..(point + 1) * dimension],
                            &data[neighbor * dimension..(neighbor + 1) * dimension],
                        );
                        template.checked_flagged_push(point, neighbor as i32, distance, true);
                    }
                }
                for threads in [1, 8] {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                    let candidates = pool.install(|| CandidateSets::build_from_graph(
                        &mut template.clone(), degree, &mut FastRng::new(42),
                    ));
                    for packed in [false, true] {
                        let layout = if packed { "packed" } else { "wide" };
                        let label = format!("{layout}/n{n_points}-k{degree}-t{threads}");
                        let stats = pool.install(|| benchmark::buffer_stats(
                            &mut template.clone(), &candidates, &data, dimension, &SquaredEuclidean, packed,
                        ));
                        eprintln!("{label}: {stats:?}");
                        group.bench_function(label, |bench| {
                            bench.iter_batched(
                                || template.clone(),
                                |mut graph| pool.install(|| {
                                    black_box(benchmark::replay(
                                        &mut graph, &candidates, &data, dimension, &SquaredEuclidean, packed,
                                    ));
                                    black_box(graph)
                                }),
                                BatchSize::LargeInput,
                            );
                        });
                    }
                }
            }
        }
        group.finish();
    }
    #[cfg(not(feature = "bench-internals"))]
    let _ = criterion;
}

fn bench_candidate_membership(criterion: &mut Criterion) {
    #[cfg(feature = "bench-internals")]
    {
        use criterion::BatchSize;
        use rinnd_core::nndescent::CandidateSets;
        use rinnd_core::rng::FastRng;

        let mut group = criterion.benchmark_group("candidate_membership");
        for width in [15, 30, 60] {
            let mut rng = FastRng::new(89);
            let updates: Vec<(f32, i32)> = (0..width * 4)
                .map(|_| (rng.next_float(), (rng.next_u64() % (width * 3) as u64) as i32)).collect();
            for vector in [false, true] {
                group.bench_function(format!("{}-k{width}", if vector { "reduction" } else { "scalar" }), |bench| {
                    bench.iter_batched(|| (vec![f32::INFINITY; width], vec![-1; width]),
                        |(mut priorities, mut indices)| {
                            CandidateSets::replay_heap_updates(&mut priorities, &mut indices, black_box(&updates), vector);
                            black_box((priorities, indices))
                        }, BatchSize::SmallInput);
                });
            }
        }
        group.finish();
    }
    #[cfg(not(feature = "bench-internals"))]
    let _ = criterion;
}

criterion_group!(
    benches,
    bench_neighbor_heap_push,
    bench_neighbor_heap_deheap,
    bench_update_replay,
    bench_candidate_membership,
);
criterion_main!(benches);
