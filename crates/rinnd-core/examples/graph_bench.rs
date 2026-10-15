//! Time a graph-only build (the UMAP-style path) and print one JSON line.
//!
//! Usage: graph_bench <data.f32> <n> <dim> <k> <seed> <forest> [out_indices.i32]
//!   forest = depth | bfs<W> (breadth-first, batch width W in 1..=8)
//! Candidate construction follows the build's features; with `blocked-candidates`,
//! RINND_BLOCKED_CANDIDATES=0 selects the original algorithm at runtime.

use rinnd_core::distance::SquaredEuclidean;
use rinnd_core::index::NNDescentBuilder;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!("usage: graph_bench <data.f32> <n> <dim> <k> <seed> <forest> [out.i32]");
        std::process::exit(2);
    }
    let n: usize = a[2].parse().unwrap();
    let dim: usize = a[3].parse().unwrap();
    let k: usize = a[4].parse().unwrap();
    let seed: u64 = a[5].parse().unwrap();
    let forest = a[6].as_str();
    let bytes = std::fs::read(&a[1]).unwrap();
    let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(data.len(), n * dim);

    let mut builder = NNDescentBuilder::new(&data, n, dim).n_neighbors(k).random_seed(seed);
    builder = match forest {
        "depth" => builder.depth_first_tree_build(),
        f if f.starts_with("bfs") => builder.breadth_first_tree_batch_width(f[3..].parse().unwrap()),
        _ => panic!("forest must be depth or bfs<W>"),
    };
    let started = Instant::now();
    let result = builder.build_graph_with_distance(SquaredEuclidean, None);
    let wall = started.elapsed().as_secs_f64();
    let s = &result.stats.nn_descent;
    if let Some(path) = a.get(7) {
        std::fs::write(path, result.graph.indices.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
    let sum = |v: &Vec<f64>| v.iter().sum::<f64>();
    let phase = |f: fn(&rinnd_core::nndescent::CandidateBuildStats) -> f64| s.candidate_phases.iter().map(f).sum::<f64>();
    println!(
        "{{\"forest_mode\":\"{}\",\"cand_mode\":\"{}\",\"n\":{},\"dim\":{},\"k\":{},\"seed\":{},\"wall\":{:.4},\
\"forest\":{:.4},\"leaf_init\":{:.4},\"candidates\":{:.4},\"cand_init\":{:.4},\"cand_forward\":{:.4},\"cand_reverse\":{:.4},\
\"cand_mark\":{:.4},\"generation\":{:.4},\"application\":{:.4},\"iters\":{},\"updates\":{:?}}}",
        forest, std::env::var("RINND_BLOCKED_CANDIDATES").unwrap_or_else(|_| "default".into()),
        n, dim, k, seed, wall, s.forest_seconds, s.leaf_initialization_seconds, sum(&s.candidate_seconds),
        phase(|p| p.initialization_seconds), phase(|p| p.forward_seconds), phase(|p| p.reverse_seconds),
        phase(|p| p.mark_seconds), sum(&s.update_generation_seconds), sum(&s.update_application_seconds),
        s.updates.len(), s.updates,
    );
}
