//! Benchmarks for distance functions.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nndescent_core::distance::{AlternativeDot, Cosine, DirectNormalizedCosine, Distance, Euclidean, InnerProduct, SquaredEuclidean};
use nndescent_core::distance::quantized::{quantized_i8_alternative_dot, quantized_i8_dot};

fn generate_vectors(n: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let b: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.1).cos()).collect();
    (a, b)
}

fn bench_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean");
    
    for dim in [32, 64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(1, *dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(Euclidean.distance(&a, &b))
            })
        });
    }
    
    group.finish();
}

fn bench_squared_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("squared_euclidean");
    
    for dim in [32, 64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(1, *dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(SquaredEuclidean.distance(&a, &b))
            })
        });
    }
    
    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine");
    
    for dim in [32, 64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(1, *dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(Cosine.distance(&a, &b))
            })
        });
    }
    
    group.finish();
}

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");
    
    for dim in [32, 64, 128, 256, 512, 768, 1024].iter() {
        let (a, b) = generate_vectors(1, *dim);
        
        group.bench_with_input(BenchmarkId::new("scalar", dim), dim, |bench, _| {
            bench.iter(|| {
                black_box(InnerProduct.distance(&a, &b))
            })
        });
    }
    
    group.finish();
}

fn bench_alternative_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("alternative_dot");

    for dim in [100, 128, 784] {
        let (a, b) = generate_vectors(1, dim);

        group.bench_with_input(BenchmarkId::new("distance", dim), &dim, |bench, _| {
            bench.iter(|| black_box(AlternativeDot.distance(black_box(&a), black_box(&b))))
        });
        group.bench_with_input(BenchmarkId::new("direct", dim), &dim, |bench, _| {
            bench.iter(|| black_box(DirectNormalizedCosine.distance(black_box(&a), black_box(&b))))
        });
    }

    group.finish();
}

fn bench_quantized_i8_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_i8_dot");

    for dim in [100, 128, 784] {
        let a: Vec<i8> = (0..dim).map(|i| ((i * 37) % 255) as i16 - 127).map(|v| v as i8).collect();
        let b: Vec<i8> = (0..dim).map(|i| ((i * 73 + 11) % 255) as i16 - 127).map(|v| v as i8).collect();
        let inv_norm_a = (quantized_i8_dot(&a, &a) as f32).sqrt().recip();
        let inv_norm_b = (quantized_i8_dot(&b, &b) as f32).sqrt().recip();

        group.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bench, _| {
            bench.iter(|| black_box(quantized_i8_dot(black_box(&a), black_box(&b))))
        });
        group.bench_with_input(BenchmarkId::new("alternative_dot", dim), &dim, |bench, _| {
            bench.iter(|| black_box(quantized_i8_alternative_dot(
                black_box(&a), black_box(&b), inv_norm_a, inv_norm_b,
            )))
        });
    }

    group.finish();
}

fn bench_random_candidate_angular(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_candidate_angular");
    let n_points = 50_000usize;
    let dim = 100usize;
    let n_candidates = 1_000usize;
    let (float_data, _) = generate_vectors(n_points, dim);
    let float_query = &float_data[..dim];
    let quantized_data: Vec<i8> = float_data
        .iter()
        .map(|&value| (value * 127.0).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let quantized_query = &quantized_data[..dim];
    let query_inv_norm = (quantized_i8_dot(quantized_query, quantized_query) as f32).sqrt().recip();
    let candidate_inv_norms: Vec<f32> = quantized_data
        .chunks_exact(dim)
        .map(|point| (quantized_i8_dot(point, point) as f32).sqrt().recip())
        .collect();
    let candidate_ids: Vec<usize> = (0..n_candidates)
        .scan(0x1234_5678u32, |state, _| {
            *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            Some(*state as usize % n_points)
        })
        .collect();

    group.bench_function("float_direct_dot", |bench| {
        bench.iter(|| {
            let mut total = 0.0f32;
            for &id in black_box(&candidate_ids) {
                let point = &float_data[id * dim..(id + 1) * dim];
                total += InnerProduct.distance(black_box(float_query), black_box(point));
            }
            black_box(total)
        })
    });
    group.bench_function("i8_direct_dot", |bench| {
        bench.iter(|| {
            let mut total = 0.0f32;
            for &id in black_box(&candidate_ids) {
                let point = &quantized_data[id * dim..(id + 1) * dim];
                total -= quantized_i8_dot(black_box(quantized_query), black_box(point)) as f32
                    * query_inv_norm
                    * candidate_inv_norms[id];
            }
            black_box(total)
        })
    });

    group.finish();
}

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");
    
    let n_points = 1000;
    let dim = 128;
    let (data, _) = generate_vectors(n_points, dim);
    let query = &data[0..dim];
    
    group.bench_function("euclidean_1000x128", |bench| {
        bench.iter(|| {
            let mut total = 0.0f32;
            for i in 0..n_points {
                let point = &data[i * dim..(i + 1) * dim];
                total += SquaredEuclidean.distance(query, point);
            }
            black_box(total)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_euclidean,
    bench_squared_euclidean,
    bench_cosine,
    bench_inner_product,
    bench_alternative_dot,
    bench_quantized_i8_dot,
    bench_random_candidate_angular,
    bench_batch_distances,
);
criterion_main!(benches);
