//! NN-Descent algorithm implementation.
//!
//! This module implements the core NN-Descent algorithm for approximate
//! k-nearest neighbor graph construction.

mod candidates;
mod update;

pub use candidates::{CandidateBuildStats, CandidateSets};
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
    pub candidate_phases: Vec<CandidateBuildStats>,
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
    let mut candidate_phases = Vec::with_capacity(params.n_iters);
    let mut update_seconds = Vec::with_capacity(params.n_iters);
    let mut update_generation_seconds = Vec::with_capacity(params.n_iters);
    let mut update_application_seconds = Vec::with_capacity(params.n_iters);
    let mut updates = Vec::with_capacity(params.n_iters);
    let mut actual_iters = 0;
    #[cfg(feature = "candidate-workspace")]
    let mut candidate_workspace = candidates::CandidateWorkspace::default();

    // NN-descent iterations
    for iter in 0..params.n_iters {
        actual_iters = iter + 1;

        // Build candidate sets (separating new from old)
        // This also marks neighbors that appear in new_candidates as old
        let t_cand_start = Instant::now();
        #[cfg(feature = "candidate-workspace")]
        let (candidates, phases) = CandidateSets::build_from_graph_reusing(
            &mut neighbor_graph, effective_max_candidates, rng, &mut candidate_workspace,
        );
        #[cfg(not(feature = "candidate-workspace"))]
        let (candidates, phases) =
            CandidateSets::build_from_graph_profiled(&mut neighbor_graph, effective_max_candidates, rng);
        candidate_seconds.push(t_cand_start.elapsed().as_secs_f64());
        candidate_phases.push(phases);

        // Generate and apply updates
        let t_upd_start = Instant::now();
        let update_result = update_iteration(&mut neighbor_graph, &candidates, data, dim, distance);
        let (n_changes, generation_seconds, application_seconds) = update_result;
        update_seconds.push(t_upd_start.elapsed().as_secs_f64());
        update_generation_seconds.push(generation_seconds);
        update_application_seconds.push(application_seconds);
        updates.push(n_changes);
        #[cfg(feature = "candidate-workspace")]
        candidate_workspace.recycle(candidates);

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

    #[cfg(feature = "candidate-workspace")]
    {
        let started = Instant::now();
        drop(candidate_workspace);
        let elapsed = started.elapsed().as_secs_f64();
        if let (Some(total), Some(phases)) = (candidate_seconds.last_mut(), candidate_phases.last_mut()) {
            *total += elapsed;
            phases.release_seconds += elapsed;
        }
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
        candidate_phases,
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
                                |_, neighbor, value| {
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

const UPDATE_BLOCK_SIZE: usize = 16384;

trait UpdateRecord: Copy + Send + Sync {
    fn encode(
        update: PotentialUpdate,
        row_offset: usize,
        point_slot: usize,
        neighbor_slot: usize,
        neighbor_is_old: bool,
    ) -> Self;

    fn decode(self, candidates: &CandidateSets, block_start: usize) -> PotentialUpdate;
}

impl UpdateRecord for PotentialUpdate {
    #[inline(always)]
    fn encode(update: Self, _: usize, _: usize, _: usize, _: bool) -> Self {
        update
    }

    #[inline(always)]
    fn decode(self, _: &CandidateSets, _: usize) -> Self {
        self
    }
}

#[inline(always)]
fn bucket_update<Record: UpdateRecord>(
    buckets: &mut [Vec<Record>],
    update: PotentialUpdate,
    row_offset: usize,
    point_slot: usize,
    neighbor_slot: usize,
    neighbor_is_old: bool,
    vertex_block_size: usize,
) {
    let point_block = update.point as usize / vertex_block_size;
    let neighbor_block = update.neighbor as usize / vertex_block_size;
    let record = Record::encode(
        update, row_offset, point_slot, neighbor_slot, neighbor_is_old,
    );
    buckets[point_block].push(record);
    if point_block != neighbor_block {
        buckets[neighbor_block].push(record);
    }
}

#[cfg(any(feature = "packed-updates", feature = "bench-internals", test))]
#[derive(Clone, Copy)]
#[repr(transparent)]
struct CandidateRefUpdate(u64);

#[cfg(any(feature = "packed-updates", feature = "bench-internals", test))]
impl CandidateRefUpdate {
    const MAX_ROWS: usize = 1 << 14;
    const MAX_CANDIDATES: usize = 1 << 6;

    fn supports(block_size: usize, max_candidates: usize) -> bool {
        block_size <= Self::MAX_ROWS && max_candidates <= Self::MAX_CANDIDATES
    }

    #[inline(always)]
    fn pack(
        row_offset: usize,
        point_slot: usize,
        neighbor_slot: usize,
        neighbor_is_old: bool,
        distance: f32,
    ) -> Self {
        assert!(row_offset < Self::MAX_ROWS);
        assert!(point_slot < Self::MAX_CANDIDATES);
        assert!(neighbor_slot < Self::MAX_CANDIDATES);
        Self(
            u64::from(distance.to_bits())
                | ((row_offset as u64) << 32)
                | ((point_slot as u64) << 46)
                | ((neighbor_slot as u64) << 52)
                | (u64::from(neighbor_is_old) << 58),
        )
    }

    #[inline(always)]
    fn unpack(self, candidates: &CandidateSets, block_start: usize) -> PotentialUpdate {
        let row = block_start + ((self.0 >> 32) & 0x3fff) as usize;
        let point_slot = ((self.0 >> 46) & 0x3f) as usize;
        let neighbor_slot = ((self.0 >> 52) & 0x3f) as usize;
        let neighbors = if self.0 & (1 << 58) != 0 {
            candidates.get_old(row)
        } else {
            candidates.get_new(row)
        };
        PotentialUpdate {
            point: candidates.get_new(row)[point_slot],
            neighbor: neighbors[neighbor_slot],
            distance: f32::from_bits(self.0 as u32),
        }
    }
}

#[cfg(any(feature = "packed-updates", feature = "bench-internals", test))]
impl UpdateRecord for CandidateRefUpdate {
    #[inline(always)]
    fn encode(
        update: PotentialUpdate,
        row_offset: usize,
        point_slot: usize,
        neighbor_slot: usize,
        neighbor_is_old: bool,
    ) -> Self {
        Self::pack(row_offset, point_slot, neighbor_slot, neighbor_is_old, update.distance)
    }

    #[inline(always)]
    fn decode(self, candidates: &CandidateSets, block_start: usize) -> PotentialUpdate {
        self.unpack(candidates, block_start)
    }
}

#[inline(always)]
fn candidate_distances<D: Distance<f32>>(
    point: i32,
    candidates: &[i32],
    skip_self: bool,
    data: &[f32],
    dim: usize,
    distance: &D,
    mut apply: impl FnMut(usize, i32, f32),
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
                    apply(offset + position, group[position], distances[position]);
                }
                offset += 4;
            } else {
                let neighbor = candidates[offset];
                if neighbor >= 0 && (!skip_self || neighbor != point) {
                    let start = neighbor as usize * dim;
                    apply(offset, neighbor, distance.distance(query, &data[start..start + dim]));
                }
                offset += 1;
            }
        }
    }
    for (position, &neighbor) in candidates.iter().enumerate().skip(offset) {
        if neighbor >= 0 && (!skip_self || neighbor != point) {
            let start = neighbor as usize * dim;
            apply(position, neighbor, distance.distance(query, &data[start..start + dim]));
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
    #[cfg(feature = "packed-updates")]
    if CandidateRefUpdate::supports(UPDATE_BLOCK_SIZE, candidates.max_candidates) {
        return update_iteration_with::<D, CandidateRefUpdate>(graph, candidates, data, dim, distance);
    }
    update_iteration_with::<D, PotentialUpdate>(graph, candidates, data, dim, distance)
}

fn update_iteration_with<D: Distance<f32> + Sync, Record: UpdateRecord>(
    graph: &mut NeighborHeap,
    candidates: &CandidateSets,
    data: &[f32],
    dim: usize,
    distance: &D,
) -> (usize, f64, f64) {
    update_iteration_observed::<D, Record>(graph, candidates, data, dim, distance, |_, _| {})
}

fn update_iteration_observed<D: Distance<f32> + Sync, Record: UpdateRecord>(
    graph: &mut NeighborHeap,
    candidates: &CandidateSets,
    data: &[f32],
    dim: usize,
    distance: &D,
    mut observe: impl FnMut(&[Vec<Vec<Record>>], usize),
) -> (usize, f64, f64) {
    use std::time::Instant;

    let mut generation_seconds = 0.0;
    let mut application_seconds = 0.0;
    let n_points = graph.n_points;
    let n_threads = rayon::current_num_threads().max(1);
    let max_candidates = candidates.max_candidates;

    if n_points == 0 || graph.k == 0 || max_candidates == 0 {
        return (0, 0.0, 0.0);
    }

    let rows_per_block = UPDATE_BLOCK_SIZE.min(n_points);
    let max_updates_per_thread = ((max_candidates * max_candidates
        + max_candidates * (max_candidates - 1) / 2)
        * rows_per_block
        / n_threads)
        + 1024;

    // Per-thread, per-destination-block update buckets
    // thread_buckets[gen_thread][dest_block] = Vec<Record>
    let vertex_block_size = n_points.div_ceil(n_threads).max(1);
    let n_vertex_blocks = n_points.div_ceil(vertex_block_size);

    #[cfg(feature = "candidate-workspace")]
    let mut thread_buckets: Vec<Vec<Vec<Record>>> = (0..n_threads)
        .into_par_iter()
        .map(|_| {
            (0..n_vertex_blocks)
                .map(|_| Vec::with_capacity(max_updates_per_thread / n_vertex_blocks + 64))
                .collect()
        })
        .collect();
    #[cfg(not(feature = "candidate-workspace"))]
    let mut thread_buckets: Vec<Vec<Vec<Record>>> = (0..n_threads)
        .map(|_| {
            (0..n_vertex_blocks)
                .map(|_| Vec::with_capacity(max_updates_per_thread / n_vertex_blocks + 64))
                .collect()
        })
        .collect();

    let mut total_changes = 0;

    let n_blocks = n_points.div_ceil(UPDATE_BLOCK_SIZE);

    for block_idx in 0..n_blocks {
        let block_start = block_idx * UPDATE_BLOCK_SIZE;
        let block_end = ((block_idx + 1) * UPDATE_BLOCK_SIZE).min(n_points);
        let block_len = block_end - block_start;
        let rows_per_thread = (block_len + n_threads - 1) / n_threads;

        // Clear all buckets
        for thread_buck in thread_buckets.iter_mut() {
            for bucket in thread_buck.iter_mut() {
                bucket.clear();
            }
        }

        // Generate updates and bucket by destination vertex block
        let generation_started = Instant::now();
        thread_buckets.par_iter_mut().enumerate().for_each(|(t, local_buckets)| {
            let row_start = block_start + t * rows_per_thread;
            let row_end = (row_start + rows_per_thread).min(block_end);

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
                        for (neighbors, slot_base, skip_self) in
                            [(&new_cands[j + 1..], j + 1, false), (old_cands, 0, true)]
                        {
                            candidate_distances(
                                p, neighbors, skip_self, data, dim, distance,
                                |slot, neighbor, value| {
                                    let neighbor_index = neighbor as usize;
                                    if value <= thresh_p.max(graph.max_distance(neighbor_index)) {
                                        bucket_update(local_buckets, PotentialUpdate {
                                            point: p,
                                            neighbor,
                                            distance: value,
                                        }, i - block_start, j, slot_base + slot, skip_self, vertex_block_size);
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
                            bucket_update(local_buckets, PotentialUpdate {
                                point: p,
                                neighbor: q,
                                distance: d,
                            }, i - block_start, j, k, false, vertex_block_size);
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
                            bucket_update(local_buckets, PotentialUpdate {
                                point: p,
                                neighbor: q,
                                distance: d,
                            }, i - block_start, j, k, true, vertex_block_size);
                        }
                    }
                }
            }
        });

        generation_seconds += generation_started.elapsed().as_secs_f64();
        observe(&thread_buckets, block_start);

        // Apply updates - each thread only reads buckets destined for its vertex block
        let thread_buckets_apply = &thread_buckets;

        let application_started = Instant::now();
        let degree = graph.k;
        let entries_per_block = vertex_block_size * degree;
        total_changes += graph.indices.par_chunks_mut(entries_per_block)
            .zip(graph.distances.par_chunks_mut(entries_per_block))
            .zip(graph.flags.par_chunks_mut(entries_per_block))
            .enumerate()
            .map(|(block, ((indices, distances), flags))| {
                let vertex_start = block * vertex_block_size;
                let vertex_end = vertex_start + indices.len() / degree;
                let mut local_changes = 0usize;

                for producer in thread_buckets_apply {
                    for update in &producer[block] {
                        let update = update.decode(candidates, block_start);
                        for (point, neighbor) in [
                            (update.point as usize, update.neighbor),
                            (update.neighbor as usize, update.point),
                        ] {
                            if point >= vertex_start && point < vertex_end {
                                let start = (point - vertex_start) * degree;
                                let end = start + degree;
                                if NeighborHeap::checked_flagged_push_row(
                                    &mut indices[start..end],
                                    &mut distances[start..end],
                                    &mut flags[start..end],
                                    neighbor,
                                    update.distance,
                                    true,
                                ) {
                                    local_changes += 1;
                                }
                            }
                        }
                    }
                }
                local_changes
            })
            .sum::<usize>();
        application_seconds += application_started.elapsed().as_secs_f64();
    }

    (total_changes, generation_seconds, application_seconds)
}

#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod benchmark {
    use super::*;

    #[derive(Debug, Default)]
    pub struct UpdateBufferStats {
        pub records: usize,
        pub duplicate_records: usize,
        pub nonempty_buckets: usize,
        pub peak_live_bytes: usize,
        pub peak_capacity_bytes: usize,
        pub bucket_metadata_bytes: usize,
    }

    pub fn replay<D: Distance<f32>>(
        graph: &mut NeighborHeap,
        candidates: &CandidateSets,
        data: &[f32],
        dim: usize,
        distance: &D,
        packed: bool,
    ) -> (usize, f64, f64) {
        if packed && CandidateRefUpdate::supports(UPDATE_BLOCK_SIZE, candidates.max_candidates) {
            update_iteration_with::<D, CandidateRefUpdate>(graph, candidates, data, dim, distance)
        } else {
            update_iteration_with::<D, PotentialUpdate>(graph, candidates, data, dim, distance)
        }
    }

    pub fn buffer_stats<D: Distance<f32>>(
        graph: &mut NeighborHeap,
        candidates: &CandidateSets,
        data: &[f32],
        dim: usize,
        distance: &D,
        packed: bool,
    ) -> UpdateBufferStats {
        if packed && CandidateRefUpdate::supports(UPDATE_BLOCK_SIZE, candidates.max_candidates) {
            collect::<D, CandidateRefUpdate>(graph, candidates, data, dim, distance)
        } else {
            collect::<D, PotentialUpdate>(graph, candidates, data, dim, distance)
        }
    }

    fn collect<D: Distance<f32>, Record: UpdateRecord>(
        graph: &mut NeighborHeap,
        candidates: &CandidateSets,
        data: &[f32],
        dim: usize,
        distance: &D,
    ) -> UpdateBufferStats {
        let threads = rayon::current_num_threads().max(1);
        let vertex_span = (graph.n_points + threads - 1) / threads;
        let mut stats = UpdateBufferStats::default();
        update_iteration_observed::<D, Record>(graph, candidates, data, dim, distance, |buckets, block_start| {
            let mut live = 0;
            let mut capacity = 0;
            for producer in buckets {
                for (destination, bucket) in producer.iter().enumerate() {
                    live += bucket.len();
                    capacity += bucket.capacity();
                    stats.nonempty_buckets += usize::from(!bucket.is_empty());
                    for record in bucket {
                        let update = record.decode(candidates, block_start);
                        let point_owner = update.point as usize / vertex_span;
                        let neighbor_owner = update.neighbor as usize / vertex_span;
                        stats.duplicate_records += usize::from(
                            point_owner != neighbor_owner && destination == neighbor_owner,
                        );
                    }
                }
            }
            stats.records += live;
            stats.peak_live_bytes = stats.peak_live_bytes.max(live * std::mem::size_of::<Record>());
            stats.peak_capacity_bytes = stats.peak_capacity_bytes.max(capacity * std::mem::size_of::<Record>());
            stats.bucket_metadata_bytes = std::mem::size_of_val(buckets)
                + buckets.iter().map(|producer| producer.capacity() * std::mem::size_of::<Vec<Record>>()).sum::<usize>();
        });
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::SquaredEuclidean;

    #[test]
    fn packed_update_layout_and_limits() {
        assert_eq!(std::mem::size_of::<PotentialUpdate>(), 12);
        assert_eq!(std::mem::size_of::<CandidateRefUpdate>(), 8);
        assert_eq!(std::mem::align_of::<CandidateRefUpdate>(), std::mem::align_of::<u64>());
        for width in [0, 1, 60, 64] {
            assert!(CandidateRefUpdate::supports(16384, width));
        }
        assert!(!CandidateRefUpdate::supports(16385, 60));
        assert!(!CandidateRefUpdate::supports(16384, 65));
        assert_eq!(CandidateRefUpdate::pack(16383, 63, 63, true, f32::from_bits(u32::MAX)).0,
            (1u64 << 59) - 1);
    }

    #[test]
    fn packed_update_fields_preserve_all_distance_bits() {
        for row in [0, 1, 16383] {
            for point_slot in [0, 1, 59, 63] {
                for neighbor_slot in [0, 1, 59, 63] {
                    for old in [false, true] {
                        for bits in [0, 1, 0x80000000, 0x3f800000, 0x7f7fffff,
                            0xff7fffff, 0x7f800000, 0xff800000, 0x7fc01234, u32::MAX] {
                            let packed = CandidateRefUpdate::pack(row, point_slot, neighbor_slot,
                                old, f32::from_bits(bits));
                            assert_eq!(packed.0 as u32, bits);
                            assert_eq!(((packed.0 >> 32) & 0x3fff) as usize, row);
                            assert_eq!(((packed.0 >> 46) & 0x3f) as usize, point_slot);
                            assert_eq!(((packed.0 >> 52) & 0x3f) as usize, neighbor_slot);
                            assert_eq!(packed.0 & (1 << 58) != 0, old);
                            assert_eq!(packed.0 >> 59, 0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packed_update_decodes_candidate_ids_without_narrowing() {
        let block_start = UPDATE_BLOCK_SIZE;
        let n_vertices = block_start + UPDATE_BLOCK_SIZE;
        let mut candidates = CandidateSets {
            new_indices: vec![-1; n_vertices * 64],
            old_indices: vec![-1; n_vertices * 64],
            n_vertices,
            max_candidates: 64,
        };
        for offset in [0, UPDATE_BLOCK_SIZE - 1] {
            let start = (block_start + offset) * 64;
            candidates.new_indices[start] = 0;
            candidates.new_indices[start + 63] = i32::MAX;
            candidates.old_indices[start + 63] = i32::MAX - 1;
            for old in [false, true] {
                let update = CandidateRefUpdate::pack(offset, 0, 63, old, -0.0)
                    .unpack(&candidates, block_start);
                assert_eq!(update.point, 0);
                assert_eq!(update.neighbor, i32::MAX - i32::from(old));
                assert_eq!(update.distance.to_bits(), (-0.0f32).to_bits());
            }
        }
    }

    fn ordered_update_reference(
        graph: &mut NeighborHeap,
        candidates: &CandidateSets,
        data: &[f32],
        dimension: usize,
    ) -> usize {
        let mut changes = 0;
        for block_start in (0..graph.n_points).step_by(UPDATE_BLOCK_SIZE) {
            let block_end = (block_start + UPDATE_BLOCK_SIZE).min(graph.n_points);
            let mut updates = Vec::new();
            for row in block_start..block_end {
                let new = candidates.get_new(row);
                let old = candidates.get_old(row);
                for (slot, &point) in new.iter().enumerate() {
                    if point < 0 {
                        continue;
                    }
                    for (neighbors, skip_self) in [(&new[slot + 1..], false), (old, true)] {
                        for &neighbor in neighbors {
                            if neighbor < 0 || (skip_self && neighbor == point) {
                                continue;
                            }
                            let point_start = point as usize * dimension;
                            let neighbor_start = neighbor as usize * dimension;
                            let value = SquaredEuclidean.distance(
                                &data[point_start..point_start + dimension],
                                &data[neighbor_start..neighbor_start + dimension],
                            );
                            if value <= graph.max_distance(point as usize)
                                .max(graph.max_distance(neighbor as usize)) {
                                updates.push(PotentialUpdate { point, neighbor, distance: value });
                            }
                        }
                    }
                }
            }
            for update in updates {
                changes += usize::from(graph.checked_flagged_push(
                    update.point as usize, update.neighbor, update.distance, true,
                ));
                changes += usize::from(graph.checked_flagged_push(
                    update.neighbor as usize, update.point, update.distance, true,
                ));
            }
        }
        changes
    }

    fn assert_same_heap(actual: &NeighborHeap, expected: &NeighborHeap) {
        assert_eq!(actual.indices, expected.indices);
        assert_eq!(actual.flags, expected.flags);
        assert!(actual.distances.iter().zip(&expected.distances)
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
    }

    #[test]
    fn packed_updates_match_ordered_replay() {
        for (n_points, dimension, width) in [
            (0, 7, 6), (3, 7, 0), (3, 7, 1), (19, 7, 6), (19, 35, 6),
            (19, 100, 60), (19, 35, 64), (19, 35, 65), (UPDATE_BLOCK_SIZE + 7, 35, 6),
        ] {
            let data: Vec<f32> = (0..n_points * dimension)
                .map(|position| ((position * 17) % 23) as f32).collect();
            let mut candidates = CandidateSets {
                new_indices: vec![-1; n_points * width],
                old_indices: vec![-1; n_points * width],
                n_vertices: n_points,
                max_candidates: width,
            };
            for row in 0..n_points {
                for slot in 0..width {
                    if (row + slot) % 7 != 0 {
                        candidates.new_indices[row * width + slot] =
                            ((row * 11 + slot * 3) % n_points) as i32;
                    }
                    if (row + slot) % 5 != 0 {
                        candidates.old_indices[row * width + slot] =
                            ((row * 5 + slot * 7) % n_points) as i32;
                    }
                }
            }
            let initial = NeighborHeap::new(n_points, 5);
            let mut expected = initial.clone();
            let expected_changes = ordered_update_reference(&mut expected, &candidates, &data, dimension);
            for threads in [1, 4, 8] {
                rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                    let mut wide = initial.clone();
                    let changes = update_iteration_with::<_, PotentialUpdate>(
                        &mut wide, &candidates, &data, dimension, &SquaredEuclidean,
                    ).0;
                    assert_eq!(changes, expected_changes);
                    assert_same_heap(&wide, &expected);
                    if CandidateRefUpdate::supports(UPDATE_BLOCK_SIZE, width) {
                        let mut packed = initial.clone();
                        let changes = update_iteration_with::<_, CandidateRefUpdate>(
                            &mut packed, &candidates, &data, dimension, &SquaredEuclidean,
                        ).0;
                        assert_eq!(changes, expected_changes);
                        assert_same_heap(&packed, &expected);
                    }
                    let mut selected = initial.clone();
                    let changes = update_iteration(
                        &mut selected, &candidates, &data, dimension, &SquaredEuclidean,
                    ).0;
                    assert_eq!(changes, expected_changes);
                    assert_same_heap(&selected, &expected);
                });
            }
        }
    }

    #[test]
    fn packed_updates_preserve_convergence() {
        check_packed_convergence(SquaredEuclidean);
        check_packed_convergence(crate::distance::AlternativeDot);
        check_packed_convergence(crate::distance::DirectNormalizedCosine);
        check_packed_convergence(crate::distance::InnerProduct);
    }

    #[cfg(feature = "bench-internals")]
    #[test]
    fn packed_update_buffer_accounting_matches_routing() {
        let mut candidates = CandidateSets {
            new_indices: vec![-1; 16],
            old_indices: vec![-1; 16],
            n_vertices: 8,
            max_candidates: 2,
        };
        candidates.new_indices[..4].copy_from_slice(&[0, 7, 2, 3]);
        let data = create_test_data(8, 1);
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                let mut wide = NeighborHeap::new(8, 1);
                let mut packed = wide.clone();
                let wide_stats = benchmark::buffer_stats(&mut wide, &candidates, &data, 1, &SquaredEuclidean, false);
                let packed_stats = benchmark::buffer_stats(&mut packed, &candidates, &data, 1, &SquaredEuclidean, true);
                let duplicates = usize::from(threads > 1);
                assert_same_heap(&wide, &packed);
                assert_eq!(wide_stats.records, 2 + duplicates);
                assert_eq!(wide_stats.duplicate_records, duplicates);
                assert_eq!(packed_stats.records, wide_stats.records);
                assert_eq!(packed_stats.duplicate_records, duplicates);
                assert_eq!(packed_stats.peak_live_bytes * 3, wide_stats.peak_live_bytes * 2);
                assert_eq!(packed_stats.peak_capacity_bytes * 3, wide_stats.peak_capacity_bytes * 2);
                assert_eq!(packed_stats.bucket_metadata_bytes, wide_stats.bucket_metadata_bytes);
                assert_eq!(packed_stats.nonempty_buckets, wide_stats.nonempty_buckets);
            });
        }
    }

    fn check_packed_convergence<D: Distance<f32>>(distance: D) {
        let n_points = 128;
        let dimension = 35;
        let mut rng = FastRng::new(7);
        let data: Vec<f32> = (0..n_points * dimension)
            .map(|_| (rng.next_float() - 0.5) / dimension as f32).collect();
        for degree in [15, 30] {
            let mut initial = NeighborHeap::new(n_points, degree);
            for point in 0..n_points {
                for offset in 1..=degree {
                    let neighbor = (point + offset * 13) % n_points;
                    let value = distance.distance(
                        &data[point * dimension..(point + 1) * dimension],
                        &data[neighbor * dimension..(neighbor + 1) * dimension],
                    );
                    initial.checked_flagged_push(point, neighbor as i32, value, true);
                }
            }
            for threads in [1, 4] {
                rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(|| {
                    let mut wide = initial.clone();
                    let mut packed = initial.clone();
                    let mut wide_rng = FastRng::new(42);
                    let mut packed_rng = FastRng::new(42);
                    let mut converged = false;
                    for _ in 0..10 {
                        let wide_candidates = CandidateSets::build_from_graph(&mut wide, degree, &mut wide_rng);
                        let packed_candidates = CandidateSets::build_from_graph(&mut packed, degree, &mut packed_rng);
                        assert_eq!(wide_candidates.new_indices, packed_candidates.new_indices);
                        assert_eq!(wide_candidates.old_indices, packed_candidates.old_indices);
                        let wide_changes = update_iteration_with::<_, PotentialUpdate>(
                            &mut wide, &wide_candidates, &data, dimension, &distance,
                        ).0;
                        let packed_changes = update_iteration_with::<_, CandidateRefUpdate>(
                            &mut packed, &packed_candidates, &data, dimension, &distance,
                        ).0;
                        assert_eq!(wide_changes, packed_changes);
                        assert_same_heap(&wide, &packed);
                        if wide_changes == 0 {
                            converged = true;
                            break;
                        }
                    }
                    assert!(converged, "metric={}", distance.name());
                });
            }
        }
    }

    #[test]
    #[should_panic]
    fn packed_update_rejects_truncated_rows() {
        CandidateRefUpdate::pack(16384, 0, 0, false, 1.0);
    }

    #[test]
    #[should_panic]
    fn packed_update_rejects_truncated_slots() {
        CandidateRefUpdate::pack(0, 64, 0, false, 1.0);
    }

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
