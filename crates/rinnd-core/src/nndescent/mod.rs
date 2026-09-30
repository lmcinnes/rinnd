//! NN-Descent algorithm implementation.
//!
//! This module implements the core NN-Descent algorithm for approximate
//! k-nearest neighbor graph construction.

mod candidates;
mod update;

pub use candidates::CandidateSets;
pub use update::{apply_updates, UpdateArray};

use crate::distance::Distance;
use crate::heap::NeighborHeap;
use crate::rng::FastRng;
use crate::tree::{
    build_breadth_first_leaf_forest, build_rp_forest, build_rp_leaf_forest, rptree_leaf_array,
    FlatTree,
};

use rayon::prelude::*;

#[derive(Clone, Debug, Default)]
pub struct NNDescentStats {
    pub forest_seconds: f64,
    pub leaf_initialization_seconds: f64,
    pub candidate_seconds: Vec<f64>,
    pub update_seconds: Vec<f64>,
    pub update_generation_seconds: Vec<f64>,
    pub update_application_seconds: Vec<f64>,
    pub updates: Vec<usize>,
}

impl NNDescentStats {
    pub fn total_seconds(&self) -> f64 {
        self.forest_seconds
            + self.leaf_initialization_seconds
            + self.candidate_seconds.iter().sum::<f64>()
            + self.update_seconds.iter().sum::<f64>()
    }
}

/// NN-Descent algorithm parameters.
#[derive(Clone, Debug)]
pub struct NNDescentParams {
    /// Number of neighbors to find
    pub n_neighbors: usize,
    /// Number of RP trees for initialization
    pub n_trees: usize,
    /// Maximum leaf size in RP trees
    pub leaf_size: usize,
    /// Maximum candidates per iteration
    pub max_candidates: usize,
    /// Maximum iterations
    pub n_iters: usize,
    /// Convergence threshold (fraction of updates)
    pub delta: f32,
    /// Whether to use angular trees
    pub angular: bool,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for NNDescentParams {
    fn default() -> Self {
        Self {
            n_neighbors: 30,
            n_trees: 8,
            leaf_size: 64,
            max_candidates: 60,
            n_iters: 10,
            delta: 0.001,
            angular: false,
            max_depth: 200,
            verbose: false,
        }
    }
}

impl NNDescentParams {
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            ..Default::default()
        }
    }
}

/// Run the NN-Descent algorithm to build a k-NN graph.
///
/// # Arguments
/// * `data` - Flattened data array (n_points × dim)
/// * `n_points` - Number of data points
/// * `dim` - Dimension of each point
/// * `distance` - Distance function
/// * `params` - Algorithm parameters
/// * `rng` - Random number generator (FastRng for performance)
///
/// # Returns
/// A `NeighborHeap` containing the k-nearest neighbors for each point.
pub fn nn_descent<D: Distance<f32> + Sync>(
    data: &[f32],
    n_points: usize,
    dim: usize,
    distance: &D,
    params: &NNDescentParams,
    rng: &mut FastRng,
) -> (NeighborHeap, Vec<FlatTree>, NNDescentStats) {
    nn_descent_impl(data, n_points, dim, distance, params, rng, true, None)
}

pub(crate) fn nn_descent_graph<D: Distance<f32> + Sync>(
    data: &[f32],
    n_points: usize,
    dim: usize,
    distance: &D,
    params: &NNDescentParams,
    rng: &mut FastRng,
    breadth_first_batch_width: Option<usize>,
) -> (NeighborHeap, Vec<FlatTree>, NNDescentStats) {
    nn_descent_impl(
        data,
        n_points,
        dim,
        distance,
        params,
        rng,
        false,
        breadth_first_batch_width,
    )
}

fn nn_descent_impl<D: Distance<f32> + Sync>(
    data: &[f32],
    n_points: usize,
    dim: usize,
    distance: &D,
    params: &NNDescentParams,
    rng: &mut FastRng,
    retain_trees: bool,
    breadth_first_batch_width: Option<usize>,
) -> (NeighborHeap, Vec<FlatTree>, NNDescentStats) {
    use std::time::Instant;

    let effective_max_candidates = params.max_candidates.min(60).min(params.n_neighbors);

    let t_forest_start = Instant::now();

    if params.verbose {
        println!("Building RP forest with {} trees...", params.n_trees);
    }

    let (forest, leaf_array) = if retain_trees {
        let forest = build_rp_forest(
            data,
            n_points,
            dim,
            params.n_trees,
            params.leaf_size,
            rng,
            params.angular,
            params.max_depth,
        );
        let leaf_array = rptree_leaf_array(&forest);
        (forest, leaf_array)
    } else if let Some(batch_width) = breadth_first_batch_width {
        let leaves = build_breadth_first_leaf_forest(
            data,
            n_points,
            dim,
            params.n_trees,
            params.leaf_size,
            rng,
            params.angular,
            params.max_depth,
            batch_width,
        );
        (Vec::new(), leaves)
    } else {
        let leaves = build_rp_leaf_forest(
            data,
            n_points,
            dim,
            params.n_trees,
            params.leaf_size,
            rng,
            params.angular,
            params.max_depth,
        );
        (Vec::new(), leaves)
    };

    let t_forest = t_forest_start.elapsed();

    // Initialize neighbor graph from tree leaves
    let mut neighbor_graph = NeighborHeap::new(n_points, params.n_neighbors);
    let t_init_start = Instant::now();

    if params.verbose {
        println!("Initializing graph from {} leaves...", leaf_array.len());
    }

    initialize_from_leaves(&mut neighbor_graph, &leaf_array, data, dim, distance, params.leaf_size);

    let t_init = t_init_start.elapsed();

    if params.verbose {
        println!(
            "Running NN-descent for up to {} iterations...",
            params.n_iters
        );
    }

    let mut candidate_seconds = Vec::with_capacity(params.n_iters);
    let mut update_seconds = Vec::with_capacity(params.n_iters);
    let mut update_generation_seconds = Vec::with_capacity(params.n_iters);
    let mut update_application_seconds = Vec::with_capacity(params.n_iters);
    let mut updates = Vec::with_capacity(params.n_iters);
    let mut actual_iters = 0;

    // NN-descent iterations
    for iter in 0..params.n_iters {
        actual_iters = iter + 1;

        // Build candidate sets (separating new from old)
        // This also marks neighbors that appear in new_candidates as old
        let t_cand_start = Instant::now();
        let candidates =
            CandidateSets::build_from_graph(&mut neighbor_graph, effective_max_candidates, rng);
        candidate_seconds.push(t_cand_start.elapsed().as_secs_f64());

        // Generate and apply updates
        let t_upd_start = Instant::now();
        let (n_changes, generation_seconds, application_seconds) =
            update_iteration(&mut neighbor_graph, &candidates, data, dim, distance);
        update_seconds.push(t_upd_start.elapsed().as_secs_f64());
        update_generation_seconds.push(generation_seconds);
        update_application_seconds.push(application_seconds);
        updates.push(n_changes);

        if params.verbose {
            println!("Iteration {}: {} updates", iter + 1, n_changes);
        }

        // Check convergence
        let threshold = (params.delta * params.n_neighbors as f32 * n_points as f32) as usize;
        if n_changes <= threshold {
            if params.verbose {
                println!("Converged after {} iterations", iter + 1);
            }
            break;
        }

        // Note: Neighbors are marked as old inside CandidateSets::build_from_graph
        // (only those that appear in new_candidates, matching PyNNDescent behavior)
    }

    if params.verbose {
        let t_candidates_total: f64 = candidate_seconds.iter().sum();
        let t_updates_total: f64 = update_seconds.iter().sum();
        println!("\n=== Timing Breakdown ===");
        println!(
            "Forest building:    {:>8.3}ms",
            t_forest.as_secs_f64() * 1000.0
        );
        println!(
            "Leaf initialization:{:>8.3}ms",
            t_init.as_secs_f64() * 1000.0
        );
        println!(
            "Candidate building: {:>8.3}ms ({} iters)",
            t_candidates_total * 1000.0,
            actual_iters
        );
        println!(
            "Update iterations:  {:>8.3}ms ({} iters)",
            t_updates_total * 1000.0,
            actual_iters
        );
        let total =
            t_forest.as_secs_f64() + t_init.as_secs_f64() + t_candidates_total + t_updates_total;
        println!("Total measured:     {:>8.3}ms", total * 1000.0);
    }

    let stats = NNDescentStats {
        forest_seconds: t_forest.as_secs_f64(),
        leaf_initialization_seconds: t_init.as_secs_f64(),
        candidate_seconds,
        update_seconds,
        update_generation_seconds,
        update_application_seconds,
        updates,
    };

    (neighbor_graph, forest, stats)
}

fn leaf_update_capacity(block_size: usize, max_leaf_size: usize, n_threads: usize) -> usize {
    block_size
        .saturating_mul(max_leaf_size)
        .saturating_mul(max_leaf_size)
        .checked_div(n_threads.saturating_mul(2))
        .unwrap_or(0)
        .clamp(1024, 1 << 20)
}

/// Initialize the neighbor graph from RP tree leaves (parallel version).
///
/// This uses a block-based approach similar to PyNNDescent for efficient
/// parallel updates without locks.
fn initialize_from_leaves<D: Distance<f32> + Sync>(
    graph: &mut NeighborHeap,
    leaves: &[Vec<i32>],
    data: &[f32],
    dim: usize,
    distance: &D,
    leaf_size: usize,
) {
    let leaves: Vec<&[i32]> = leaves.iter().flat_map(|leaf| {
        let valid_len = leaf.iter().position(|&point| point < 0).unwrap_or(leaf.len());
        leaf[..valid_len].chunks(leaf_size.max(1))
    }).collect();
    let n_points = graph.n_points;
    let n_leaves = leaves.len();

    // Process in blocks - generate updates in parallel, apply in parallel by vertex block
    let n_threads = rayon::current_num_threads();
    let block_size = (n_threads * 64).max(128);

    // Pre-allocate update storage per thread
    // Estimate max updates: block_size * leaf_size^2 / 2 per thread
    let max_leaf_size = leaves.iter().map(|l| l.len()).max().unwrap_or(0);
    let updates_per_thread = leaf_update_capacity(block_size, max_leaf_size, n_threads);

    let vertex_block_size = (n_points + n_threads - 1) / n_threads;

    // Process leaves in blocks
    for block_start in (0..n_leaves).step_by(block_size) {
        let block_end = (block_start + block_size).min(n_leaves);
        let leaf_block = &leaves[block_start..block_end];

        // Generate updates in parallel (each thread handles a slice of leaves)
        let updates: Vec<Vec<(i32, i32, f32)>> = (0..n_threads)
            .into_par_iter()
            .map(|t| {
                let mut thread_updates = Vec::with_capacity(updates_per_thread);
                let leaves_per_thread = (leaf_block.len() + n_threads - 1) / n_threads;
                let start_leaf = t * leaves_per_thread;
                let end_leaf = (start_leaf + leaves_per_thread).min(leaf_block.len());

                for leaf_idx in start_leaf..end_leaf {
                    let leaf = leaf_block[leaf_idx];

                    if cfg!(feature = "batched-leaves") && D::USE_DISTANCE_FOUR && dim >= 32 {
                        let valid_len = leaf.iter().position(|&point| point < 0).unwrap_or(leaf.len());
                        for position in 0..valid_len {
                            let point = leaf[position];
                            let point_threshold = graph.max_distance(point as usize);
                            candidate_distances(
                                point, &leaf[position + 1..valid_len], false, data, dim, distance,
                                |neighbor, value| {
                                    if value < point_threshold.max(graph.max_distance(neighbor as usize)) {
                                        thread_updates.push((point, neighbor, value));
                                    }
                                },
                            );
                        }
                        continue;
                    }

                    for i in 0..leaf.len() {
                        let p = leaf[i];
                        if p < 0 {
                            break;
                        }
                        let p_usize = p as usize;
                        let point_p = &data[p_usize * dim..(p_usize + 1) * dim];

                        for j in (i + 1)..leaf.len() {
                            let q = leaf[j];
                            if q < 0 {
                                break;
                            }
                            let q_usize = q as usize;
                            let point_q = &data[q_usize * dim..(q_usize + 1) * dim];

                            let d = distance.distance(point_p, point_q);

                            // Filter by threshold (like PyNNDescent)
                            let max_threshold =
                                graph.max_distance(p_usize).max(graph.max_distance(q_usize));
                            if d < max_threshold {
                                thread_updates.push((p, q, d));
                            }
                        }
                    }
                }

                thread_updates
            })
            .collect();

        // Apply updates in parallel by vertex block (avoid write conflicts)
        let updates_ref = &updates;
        (0..n_threads).into_par_iter().for_each(|t| {
            let v_block_start = t * vertex_block_size;
            let v_block_end = (v_block_start + vertex_block_size).min(n_points);

            // Get mutable access to this thread's vertex block
            // SAFETY: Each thread writes to disjoint vertex blocks
            let graph_ptr = graph as *const NeighborHeap as *mut NeighborHeap;
            let graph_mut = unsafe { &mut *graph_ptr };

            for thread_updates in updates_ref.iter() {
                for &(p, q, d) in thread_updates {
                    let p_usize = p as usize;
                    let q_usize = q as usize;

                    if p_usize >= v_block_start && p_usize < v_block_end {
                        graph_mut.checked_flagged_push(p_usize, q, d, true);
                    }
                    if q_usize >= v_block_start && q_usize < v_block_end {
                        graph_mut.checked_flagged_push(q_usize, p, d, true);
                    }
                }
            }
        });
    }
}

/// A potential update to the neighbor graph, stored as compact 12-byte struct.
#[derive(Clone, Copy)]
#[repr(C)]
struct PotentialUpdate {
    point: i32,
    neighbor: i32,
    distance: f32,
}

#[inline(always)]
fn candidate_distances<D: Distance<f32>>(
    point: i32,
    candidates: &[i32],
    skip_self: bool,
    data: &[f32],
    dim: usize,
    distance: &D,
    mut apply: impl FnMut(i32, f32),
) {
    let query = &data[point as usize * dim..(point as usize + 1) * dim];
    let mut offset = 0;
    if D::USE_DISTANCE_FOUR && dim >= 32 {
        while offset + 4 <= candidates.len() {
            let group = &candidates[offset..offset + 4];
            if group.iter().all(|&neighbor| neighbor >= 0 && (!skip_self || neighbor != point)) {
                let vectors = std::array::from_fn(|position| {
                    let start = group[position] as usize * dim;
                    &data[start..start + dim]
                });
                let distances = distance.distance_four(query, vectors);
                for position in 0..4 {
                    apply(group[position], distances[position]);
                }
                offset += 4;
            } else {
                let neighbor = candidates[offset];
                if neighbor >= 0 && (!skip_self || neighbor != point) {
                    let start = neighbor as usize * dim;
                    apply(neighbor, distance.distance(query, &data[start..start + dim]));
                }
                offset += 1;
            }
        }
    }
    for &neighbor in &candidates[offset..] {
        if neighbor >= 0 && (!skip_self || neighbor != point) {
            let start = neighbor as usize * dim;
            apply(neighbor, distance.distance(query, &data[start..start + dim]));
        }
    }
}

/// Run one iteration of NN-descent updates using block-based processing.
///
/// Matches PyNNDescent's `process_candidates` / `generate_graph_update_array`:
/// - Processes candidate rows in blocks of BLOCK_SIZE (16384)
/// - Within each block, rows are split across threads
/// - Each thread writes to pre-allocated update storage (no dynamic allocation)
/// - Updates applied via block-based vertex ownership
///
/// This avoids the per-point Vec allocation of the previous approach, which
/// created n_points separate allocations per iteration.
fn update_iteration<D: Distance<f32> + Sync>(
    graph: &mut NeighborHeap,
    candidates: &CandidateSets,
    data: &[f32],
    dim: usize,
    distance: &D,
) -> (usize, f64, f64) {
    use std::time::Instant;

    let mut generation_seconds = 0.0;
    let mut application_seconds = 0.0;
    let n_points = graph.n_points;
    let n_threads = rayon::current_num_threads().max(1);
    let max_candidates = candidates.max_candidates;

    const BLOCK_SIZE: usize = 16384;

    let rows_per_block = BLOCK_SIZE.min(n_points);
    let max_updates_per_thread = ((max_candidates * max_candidates
        + max_candidates * (max_candidates - 1) / 2)
        * rows_per_block
        / n_threads)
        + 1024;

    // Per-thread, per-destination-block update buckets
    // thread_buckets[gen_thread][dest_block] = Vec<PotentialUpdate>
    let vertex_block_size = (n_points + n_threads - 1) / n_threads;
    let n_vertex_blocks = n_threads; // One vertex block per thread for application elision

    let mut thread_buckets: Vec<Vec<Vec<PotentialUpdate>>> = (0..n_threads)
        .map(|_| {
            (0..n_vertex_blocks)
                .map(|_| Vec::with_capacity(max_updates_per_thread / n_vertex_blocks + 64))
                .collect()
        })
        .collect();

    use std::sync::atomic::{AtomicUsize, Ordering};
    let total_changes = AtomicUsize::new(0);

    let n_blocks = (n_points + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for block_idx in 0..n_blocks {
        let block_start = block_idx * BLOCK_SIZE;
        let block_end = ((block_idx + 1) * BLOCK_SIZE).min(n_points);
        let block_len = block_end - block_start;
        let rows_per_thread = (block_len + n_threads - 1) / n_threads;

        // Clear all buckets
        for thread_buck in thread_buckets.iter_mut() {
            for bucket in thread_buck.iter_mut() {
                bucket.clear();
            }
        }

        let thread_buckets_ref = &thread_buckets;

        // Generate updates and bucket by destination vertex block
        let generation_started = Instant::now();
        (0..n_threads).into_par_iter().for_each(|t| {
            let row_start = block_start + t * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(block_end);

            // SAFETY: Each thread writes to its own thread_buckets[t]
            let buckets_ptr = thread_buckets_ref.as_ptr() as *mut Vec<Vec<PotentialUpdate>>;
            let local_buckets = unsafe { &mut *buckets_ptr.add(t) };

            for i in row_start..row_end {
                let new_cands = candidates.get_new(i);
                let old_cands = candidates.get_old(i);

                for j in 0..max_candidates {
                    let p = new_cands[j];
                    if p < 0 {
                        continue;
                    }
                    let p_usize = p as usize;
                    let data_p = &data[p_usize * dim..(p_usize + 1) * dim];
                    let thresh_p = graph.max_distance(p_usize);

                    if D::USE_DISTANCE_FOUR && dim >= 32 {
                        for (neighbors, skip_self) in
                            [(&new_cands[j + 1..], false), (old_cands, true)]
                        {
                            candidate_distances(
                                p, neighbors, skip_self, data, dim, distance,
                                |neighbor, value| {
                                    let neighbor_index = neighbor as usize;
                                    if value <= thresh_p.max(graph.max_distance(neighbor_index)) {
                                        let update = PotentialUpdate {
                                            point: p,
                                            neighbor,
                                            distance: value,
                                        };
                                        let point_block = p_usize / vertex_block_size;
                                        let neighbor_block = neighbor_index / vertex_block_size;
                                        local_buckets[point_block.min(n_vertex_blocks - 1)].push(update);
                                        if point_block != neighbor_block {
                                            local_buckets[neighbor_block.min(n_vertex_blocks - 1)].push(update);
                                        }
                                    }
                                },
                            );
                        }
                        continue;
                    }

                    for k in (j + 1)..max_candidates {
                        let q = new_cands[k];
                        if q < 0 {
                            continue;
                        }
                        let q_usize = q as usize;
                        let max_thresh = thresh_p.max(graph.max_distance(q_usize));
                        let d =
                            distance.distance(data_p, &data[q_usize * dim..(q_usize + 1) * dim]);

                        if d <= max_thresh {
                            let update = PotentialUpdate {
                                point: p,
                                neighbor: q,
                                distance: d,
                            };
                            let p_block = p_usize / vertex_block_size;
                            let q_block = q_usize / vertex_block_size;
                            local_buckets[p_block.min(n_vertex_blocks - 1)].push(update);
                            if p_block != q_block {
                                local_buckets[q_block.min(n_vertex_blocks - 1)].push(update);
                            }
                        }
                    }

                    for k in 0..max_candidates {
                        let q = old_cands[k];
                        if q < 0 || p == q {
                            continue;
                        }
                        let q_usize = q as usize;
                        let max_thresh = thresh_p.max(graph.max_distance(q_usize));
                        let d =
                            distance.distance(data_p, &data[q_usize * dim..(q_usize + 1) * dim]);

                        if d <= max_thresh {
                            let update = PotentialUpdate {
                                point: p,
                                neighbor: q,
                                distance: d,
                            };
                            let p_block = p_usize / vertex_block_size;
                            let q_block = q_usize / vertex_block_size;
                            local_buckets[p_block.min(n_vertex_blocks - 1)].push(update);
                            if p_block != q_block {
                                local_buckets[q_block.min(n_vertex_blocks - 1)].push(update);
                            }
                        }
                    }
                }
            }
        });

        generation_seconds += generation_started.elapsed().as_secs_f64();

        // Apply updates - each thread only reads buckets destined for its vertex block
        let thread_buckets_apply = &thread_buckets;

        let application_started = Instant::now();
        (0..n_threads).into_par_iter().for_each(|t| {
            let v_block_start = t * vertex_block_size;
            let v_block_end = (v_block_start + vertex_block_size).min(n_points);

            let graph_ptr = graph as *const NeighborHeap as *mut NeighborHeap;
            let graph_mut = unsafe { &mut *graph_ptr };

            let mut local_changes = 0usize;

            // Only scan updates destined for this vertex block
            for src_thread in 0..n_threads {
                let bucket = &thread_buckets_apply[src_thread][t];

                for update in bucket.iter() {
                    let p = update.point as usize;
                    let q = update.neighbor;
                    let d = update.distance;

                    // Apply symmetrically, but only for vertices in our block
                    if p >= v_block_start && p < v_block_end {
                        if graph_mut.checked_flagged_push(p, q, d, true) {
                            local_changes += 1;
                        }
                    }
                    let q_usize = q as usize;
                    if q_usize >= v_block_start && q_usize < v_block_end {
                        if graph_mut.checked_flagged_push(q_usize, update.point, d, true) {
                            local_changes += 1;
                        }
                    }
                }
            }

            total_changes.fetch_add(local_changes, Ordering::Relaxed);
        });
        application_seconds += application_started.elapsed().as_secs_f64();
    }

    (total_changes.load(Ordering::Relaxed), generation_seconds, application_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::SquaredEuclidean;

    fn create_test_data(n: usize, dim: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                data.push((i * dim + j) as f32 * 0.1);
            }
        }
        data
    }

    #[test]
    fn leaf_update_reserve_is_bounded_for_degenerate_leaves() {
        assert_eq!(leaf_update_capacity(512, 64, 8), 131072);
        assert_eq!(leaf_update_capacity(512, 73429, 8), 1 << 20);
        assert_eq!(leaf_update_capacity(512, usize::MAX, 8), 1 << 20);
        assert_eq!(leaf_update_capacity(128, 0, 1), 1024);
    }

    #[test]
    fn leaf_initialization_matches_ordered_reference() {
        for n_points in [3, 37] {
            let dim = 100;
            let data: Vec<f32> = (0..n_points * dim)
                .map(|position| ((position * 17) % 23) as f32)
                .collect();
            let leaves: Vec<Vec<i32>> = (0..600)
                .map(|leaf_index| {
                    let mut leaf: Vec<i32> = (0..n_points.min(15))
                        .map(|offset| ((leaf_index + offset) % n_points) as i32)
                        .collect();
                    leaf.push(-1);
                    leaf
                })
                .collect();
            for degree in [15, 30] {
                let mut expected = NeighborHeap::new(n_points, degree);
                for leaf in &leaves {
                    for (position, &point) in leaf.iter().take_while(|&&point| point >= 0).enumerate() {
                        for &neighbor in leaf[position + 1..].iter().take_while(|&&neighbor| neighbor >= 0) {
                            let distance = SquaredEuclidean.distance(
                                &data[point as usize * dim..(point as usize + 1) * dim],
                                &data[neighbor as usize * dim..(neighbor as usize + 1) * dim],
                            );
                            expected.checked_flagged_push(point as usize, neighbor, distance, true);
                            expected.checked_flagged_push(neighbor as usize, point, distance, true);
                        }
                    }
                }
                for threads in [1, 2, 4, 8] {
                    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                    let mut actual = NeighborHeap::new(n_points, degree);
                    pool.install(|| initialize_from_leaves(&mut actual, &leaves, &data, dim, &SquaredEuclidean, 64));
                    assert_eq!(actual.indices, expected.indices, "threads={threads}");
                    assert_eq!(actual.distances, expected.distances, "threads={threads}");
                    assert_eq!(actual.flags, expected.flags, "threads={threads}");
                }
            }
        }
    }

    #[test]
    fn oversized_leaf_initialization_matches_bounded_groups() {
        let n_points = 1025;
        let dim = 100;
        let data = create_test_data(n_points, dim);
        let mut leaf: Vec<i32> = (0..n_points as i32).collect();
        leaf.extend([-1, i32::MAX]);
        let groups: Vec<Vec<i32>> = leaf[..n_points].chunks(16).map(|group| group.to_vec()).collect();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut expected = NeighborHeap::new(n_points, 15);
            let mut actual = NeighborHeap::new(n_points, 15);
            pool.install(|| {
                initialize_from_leaves(&mut expected, &groups, &data, dim, &SquaredEuclidean, 16);
                initialize_from_leaves(&mut actual, &[leaf.clone()], &data, dim, &SquaredEuclidean, 16);
            });
            assert_eq!(actual.indices, expected.indices);
            assert_eq!(actual.distances, expected.distances);
            assert_eq!(actual.flags, expected.flags);
            for point in 0..n_points {
                assert!(actual.indices[point * 15..(point + 1) * 15].iter()
                    .all(|&neighbor| neighbor < 0 || neighbor as usize / 16 == point / 16));
            }
        }
    }

    #[test]
    fn batched_updates_match_individual_distances() {
        check_batched_updates(SquaredEuclidean, false);
        check_batched_updates(crate::distance::Cosine, true);
        check_batched_updates(crate::distance::AlternativeDot, true);
        check_batched_updates(crate::distance::DirectNormalizedCosine, true);
        check_batched_updates(crate::distance::InnerProduct, false);
        check_batched_updates(crate::distance::Dot, true);
    }

    fn check_batched_updates<D: Distance<f32>>(distance: D, angular: bool) {
        #[derive(Clone)]
        struct Individual<D>(D);

        impl<D: Distance<f32>> Distance<f32> for Individual<D> {
            fn distance(&self, query: &[f32], candidate: &[f32]) -> f32 {
                self.0.distance(query, candidate)
            }

            fn name(&self) -> &'static str {
                self.0.name()
            }
        }

        for dimension in [31, 100, 784] {
            let mut data_rng = FastRng::new(13);
            let mut data: Vec<f32> = (0..128 * dimension).map(|_| (data_rng.next_float() - 0.5) / dimension as f32).collect();
            data[..dimension].fill(0.0);
            for degree in [15, 30] {
                let params = NNDescentParams {
                    n_neighbors: degree, n_trees: 2, leaf_size: 20, max_candidates: degree,
                    n_iters: 5, delta: 0.001, angular, max_depth: 100, verbose: false,
                };
                for threads in [1, 4] {
                    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                        let (expected, _, expected_stats) = nn_descent(
                            &data, 128, dimension, &Individual(distance.clone()), &params, &mut FastRng::new(42),
                        );
                        let (actual, _, actual_stats) = nn_descent(
                            &data, 128, dimension, &distance, &params, &mut FastRng::new(42),
                        );
                        assert_eq!(actual.indices, expected.indices);
                        assert_eq!(actual.distances, expected.distances);
                        assert_eq!(actual.flags, expected.flags);
                        assert_eq!(actual_stats.updates, expected_stats.updates);
                        let iterations = actual_stats.updates.len();
                        assert_eq!(actual_stats.update_seconds.len(), iterations);
                        assert_eq!(actual_stats.update_generation_seconds.len(), iterations);
                        assert_eq!(actual_stats.update_application_seconds.len(), iterations);
                        for iteration in 0..iterations {
                            let generation = actual_stats.update_generation_seconds[iteration];
                            let application = actual_stats.update_application_seconds[iteration];
                            assert!(generation.is_finite() && generation >= 0.0);
                            assert!(application.is_finite() && application >= 0.0);
                            assert!(generation + application <= actual_stats.update_seconds[iteration]);
                        }
                    });
                }
            }
        }
    }

    #[test]
    fn test_nn_descent_basic() {
        let n_points = 100;
        let dim = 10;
        let data = create_test_data(n_points, dim);
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        let params = NNDescentParams {
            n_neighbors: 10,
            n_trees: 2,
            leaf_size: 20,
            max_candidates: 20,
            n_iters: 5,
            delta: 0.001,
            angular: false,
            max_depth: 100,
            verbose: false,
        };

        let (graph, _forest, _stats) =
            nn_descent(&data, n_points, dim, &distance, &params, &mut rng);

        // Check that each point has neighbors
        for point in 0..n_points {
            let (indices, distances, _) = graph.get_row(point);

            // Should have some valid neighbors
            let valid_count = indices.iter().filter(|&&x| x >= 0).count();
            assert!(valid_count > 0, "Point {} has no valid neighbors", point);

            // Distances should be finite for valid neighbors
            for (&idx, &dist) in indices.iter().zip(distances.iter()) {
                if idx >= 0 {
                    assert!(dist.is_finite(), "Infinite distance for point {}", point);
                }
            }
        }
    }

    #[test]
    fn test_nn_descent_self_not_neighbor() {
        let n_points = 50;
        let dim = 5;
        let data = create_test_data(n_points, dim);
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        let params = NNDescentParams::new(5);

        let (graph, _, _) = nn_descent(&data, n_points, dim, &distance, &params, &mut rng);

        // No point should have itself as a neighbor
        for point in 0..n_points {
            let (indices, _, _) = graph.get_row(point);
            assert!(
                !indices.contains(&(point as i32)),
                "Point {} has itself as neighbor",
                point
            );
        }
    }
}
