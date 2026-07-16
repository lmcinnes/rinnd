//! Greedy graph search for approximate nearest neighbors.

use crate::distance::Distance;
use crate::distance::quantized::quantized_i8_dot;
use crate::graph::SearchGraph;
use crate::heap::{BoundedHeap, CandidateHeap};
use crate::rng::FastRng;
use crate::tree::FlatTree;
use crate::visited::VisitedSet;

#[inline(always)]
fn prefetch_point(data: &[f32], point_idx: i32, dim: usize) {
    if point_idx < 0 {
        return;
    }
    let base = point_idx as usize * dim;
    if base >= data.len() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

        // Pull an upcoming candidate row into L1 before distance evaluation.
        unsafe {
            _mm_prefetch(data.as_ptr().add(base) as *const i8, _MM_HINT_T0);
        }
    }
}

#[inline(always)]
fn prefetch_quantized_point(data: &[i8], point_idx: i32, dim: usize) {
    if point_idx < 0 {
        return;
    }
    let base = point_idx as usize * dim;
    if base >= data.len() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm_prefetch, _MM_HINT_T0};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(data.as_ptr().add(base), _MM_HINT_T0);
    }
}

const PREFETCH_LOOKAHEAD: usize = 4;

/// Instrumentation counters collected during a single search.
#[derive(Clone, Copy, Debug, Default)]
pub struct SearchStats {
    /// Number of distance function evaluations performed.
    pub distance_evaluations: u64,
    /// Number of seed candidates from tree initialization.
    pub tree_seed_candidates: u64,
    /// Number of accepted random seeds.
    pub random_seed_candidates: u64,
    /// Number of neighbor entries scanned from adjacency lists.
    pub neighbor_candidates_scanned: u64,
    /// Number of visited check-and-mark operations.
    pub visited_checks: u64,
    /// Number of visited checks that were already visited.
    pub visited_already_seen: u64,
    /// Number of pushes into the seed heap.
    pub seed_pushes: u64,
    /// Number of pops from the seed heap.
    pub seed_pops: u64,
    /// Number of attempts to insert into the result heap.
    pub result_push_attempts: u64,
    /// Number of successful inserts into the result heap.
    pub result_push_inserted: u64,
    /// Number of distance bound recomputations after improved result heap max.
    pub bound_updates: u64,
    /// Number of iterations of the main greedy while-loop.
    pub main_loop_iterations: u64,
}

/// Reusable state for greedy graph search.
///
/// Reusing this across repeated queries avoids allocating/initializing
/// visited sets and heaps on every call.
#[derive(Clone, Debug)]
pub struct SearchWorkspace {
    visited: VisitedSet,
    result_heap: BoundedHeap,
    seed_set: CandidateHeap,
    k: usize,
}

impl SearchWorkspace {
    /// Create workspace sized for `n_points` and `k`.
    pub fn new(n_points: usize, k: usize) -> Self {
        Self {
            visited: VisitedSet::new(n_points),
            result_heap: BoundedHeap::new(k),
            seed_set: CandidateHeap::with_capacity(k.saturating_mul(4).max(32)),
            k,
        }
    }

    #[inline]
    fn reset(&mut self, n_points: usize, k: usize) {
        if self.visited.capacity() != n_points {
            self.visited = VisitedSet::new(n_points);
        } else {
            self.visited.clear();
        }

        if self.k != k {
            self.result_heap = BoundedHeap::new(k);
            self.k = k;
        } else {
            self.result_heap.clear();
        }

        self.seed_set.clear();
    }
}

#[inline]
fn greedy_search_heap_with_workspace<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> BoundedHeap {
    let n_points = graph.n_vertices;
    workspace.reset(n_points, k);

    let visited = &mut workspace.visited;
    let result_heap = &mut workspace.result_heap;
    let seed_set = &mut workspace.seed_set;

    for tree in trees {
        let (start, end) = tree.search(query, rng);
        for &idx in &tree.indices[start..end] {
            if idx >= 0 && !visited.check_and_mark(idx) {
                let point = &data[idx as usize * dim..(idx as usize + 1) * dim];
                let d = distance.distance(query, point);
                result_heap.push(d, idx);
                seed_set.push(d, idx);
            }
        }
    }

    let n_initial = seed_set.len();
    let n_random = k.saturating_sub(n_initial);
    for _ in 0..n_random {
        let idx = rng.next_index(n_points) as i32;
        if !visited.check_and_mark(idx) {
            let point = &data[idx as usize * dim..(idx as usize + 1) * dim];
            let d = distance.distance(query, point);
            result_heap.push(d, idx);
            seed_set.push(d, idx);
        }
    }

    let mut max_distance = result_heap.max_distance();
    let mut distance_bound = max_distance + epsilon * (max_distance - min_distance);

    while let Some((d_vertex, vertex)) = seed_set.pop() {
        if d_vertex >= distance_bound {
            break;
        }

        let neighbors = graph.neighbors(vertex as usize);
        for (ni, &neighbor) in neighbors.iter().enumerate() {
            if let Some(&prefetch_neighbor) = neighbors.get(ni + PREFETCH_LOOKAHEAD) {
                prefetch_point(data, prefetch_neighbor, dim);
            }
            if neighbor < 0 {
                continue;
            }
            if visited.check_and_mark(neighbor) {
                continue;
            }

            let point = &data[neighbor as usize * dim..(neighbor as usize + 1) * dim];
            let d = distance.distance(query, point);
            if d < distance_bound {
                let inserted = result_heap.push(d, neighbor);
                seed_set.push(d, neighbor);
                if inserted {
                    max_distance = result_heap.max_distance();
                    distance_bound = max_distance + epsilon * (max_distance - min_distance);
                }
            }
        }
    }

    result_heap.clone()
}

/// Greedy search using caller-provided reusable workspace.
pub fn greedy_search_with_workspace<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> (Vec<i32>, Vec<f32>) {
    greedy_search_heap_with_workspace(
        query, data, dim, graph, trees, distance, k, epsilon, min_distance, rng, workspace,
    )
    .into_sorted()
}

/// Greedy search that returns only indices, avoiding distance-result
/// allocation when callers do not consume distances.
pub fn greedy_search_indices_with_workspace<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> Vec<i32> {
    greedy_search_heap_with_workspace(
        query, data, dim, graph, trees, distance, k, epsilon, min_distance, rng, workspace,
    )
    .into_sorted_indices()
}

/// Greedy graph search over a symmetrically quantized signed-int8 dataset.
///
/// The float query is retained only for RP-tree entry selection. Candidate
/// distances use the quantized query and precomputed integer-domain norms.
pub fn greedy_search_quantized_i8_with_workspace(
    tree_query: &[f32],
    query: &[i8],
    query_inv_norm: f32,
    data: &[i8],
    inv_norms: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> (Vec<i32>, Vec<f32>) {
    let n_points = graph.n_vertices;
    workspace.reset(n_points, k);

    let distance_to = |idx: i32| {
        let idx = idx as usize;
        let point = &data[idx * dim..(idx + 1) * dim];
        let similarity = quantized_i8_dot(query, point) as f32
            * query_inv_norm
            * inv_norms[idx];
        1.0 - similarity.clamp(-1.0, 1.0)
    };

    for tree in trees {
        let (start, end) = tree.search(tree_query, rng);
        for &idx in &tree.indices[start..end] {
            if idx >= 0 && !workspace.visited.check_and_mark(idx) {
                let distance = distance_to(idx);
                workspace.result_heap.push(distance, idx);
                workspace.seed_set.push(distance, idx);
            }
        }
    }

    let n_random = k.saturating_sub(workspace.seed_set.len());
    for _ in 0..n_random {
        let idx = rng.next_index(n_points) as i32;
        if !workspace.visited.check_and_mark(idx) {
            let distance = distance_to(idx);
            workspace.result_heap.push(distance, idx);
            workspace.seed_set.push(distance, idx);
        }
    }

    let mut max_distance = workspace.result_heap.max_distance();
    let mut distance_bound = max_distance + epsilon * (max_distance - min_distance);
    while let Some((vertex_distance, vertex)) = workspace.seed_set.pop() {
        if vertex_distance >= distance_bound {
            break;
        }

        let neighbors = graph.neighbors(vertex as usize);
        for (position, &neighbor) in neighbors.iter().enumerate() {
            if let Some(&upcoming) = neighbors.get(position + PREFETCH_LOOKAHEAD) {
                prefetch_quantized_point(data, upcoming, dim);
            }
            if neighbor < 0 || workspace.visited.check_and_mark(neighbor) {
                continue;
            }

            let distance = distance_to(neighbor);
            if distance < distance_bound {
                let inserted = workspace.result_heap.push(distance, neighbor);
                workspace.seed_set.push(distance, neighbor);
                if inserted {
                    max_distance = workspace.result_heap.max_distance();
                    distance_bound = max_distance + epsilon * (max_distance - min_distance);
                }
            }
        }
    }

    workspace.result_heap.clone().into_sorted()
}

/// Greedy search using caller-provided reusable workspace with instrumentation.
pub fn greedy_search_with_workspace_stats<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> (Vec<i32>, Vec<f32>, SearchStats) {
    let n_points = graph.n_vertices;
    workspace.reset(n_points, k);
    let mut stats = SearchStats::default();

    let visited = &mut workspace.visited;
    let result_heap = &mut workspace.result_heap;
    let seed_set = &mut workspace.seed_set;

    // Initialize from every retained search tree, deduplicating overlapping leaves.
    for tree in trees {
        let (start, end) = tree.search(query, rng);
        for &idx in &tree.indices[start..end] {
            if idx < 0 {
                continue;
            }
            stats.visited_checks += 1;
            if visited.check_and_mark(idx) {
                stats.visited_already_seen += 1;
                continue;
            }
            {
                let point = &data[idx as usize * dim..(idx as usize + 1) * dim];
                stats.distance_evaluations += 1;
                let d = distance.distance(query, point);
                stats.result_push_attempts += 1;
                let inserted = result_heap.push(d, idx);
                if inserted {
                    stats.result_push_inserted += 1;
                }
                seed_set.push(d, idx);
                stats.seed_pushes += 1;
                stats.tree_seed_candidates += 1;
            }
        }
    }

    // Add random seeds if we don't have enough
    let n_initial = seed_set.len();
    let n_random = k.saturating_sub(n_initial);
    for _ in 0..n_random {
        let idx = rng.next_index(n_points) as i32;
        stats.visited_checks += 1;
        if visited.check_and_mark(idx) {
            stats.visited_already_seen += 1;
            continue;
        }
        {
            let point = &data[idx as usize * dim..(idx as usize + 1) * dim];
            stats.distance_evaluations += 1;
            let d = distance.distance(query, point);
            stats.result_push_attempts += 1;
            let inserted = result_heap.push(d, idx);
            if inserted {
                stats.result_push_inserted += 1;
            }
            seed_set.push(d, idx);
            stats.seed_pushes += 1;
            stats.random_seed_candidates += 1;
        }
    }

    let mut max_distance = result_heap.max_distance();
    let mut distance_bound = max_distance + epsilon * (max_distance - min_distance);

    // Greedy search
    while let Some((d_vertex, vertex)) = seed_set.pop() {
        stats.seed_pops += 1;
        stats.main_loop_iterations += 1;
        if d_vertex >= distance_bound {
            break;
        }

        // Explore neighbors
        let neighbors = graph.neighbors(vertex as usize);
        for (ni, &neighbor) in neighbors.iter().enumerate() {
            if let Some(&prefetch_neighbor) = neighbors.get(ni + PREFETCH_LOOKAHEAD) {
                prefetch_point(data, prefetch_neighbor, dim);
            }
            stats.neighbor_candidates_scanned += 1;
            if neighbor < 0 {
                continue;
            }

            stats.visited_checks += 1;
            if visited.check_and_mark(neighbor) {
                stats.visited_already_seen += 1;
                continue;
            }

            let point = &data[neighbor as usize * dim..(neighbor as usize + 1) * dim];
            stats.distance_evaluations += 1;
            let d = distance.distance(query, point);

            if d < distance_bound {
                stats.result_push_attempts += 1;
                let inserted = result_heap.push(d, neighbor);
                if inserted {
                    stats.result_push_inserted += 1;
                }
                seed_set.push(d, neighbor);
                stats.seed_pushes += 1;
                if inserted {
                    max_distance = result_heap.max_distance();
                    distance_bound = max_distance + epsilon * (max_distance - min_distance);
                    stats.bound_updates += 1;
                }
            }
        }
    }

    let (indices, distances) = result_heap.clone().into_sorted();
    (indices, distances, stats)
}

/// Greedy search on a k-NN graph.
///
/// # Arguments
/// * `query` - Query point
/// * `data` - All data points (flattened)
/// * `dim` - Dimension of points
/// * `graph` - Search graph (CSR format)
/// * `trees` - Search trees used for initialization
/// * `distance` - Distance function
/// * `k` - Number of neighbors to return
/// * `epsilon` - Search expansion factor (0.0 = exact search on graph, higher = more exploration)
/// * `rng` - Random number generator
///
/// # Returns
/// (indices, distances) of the k nearest neighbors found.
pub fn greedy_search<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
) -> (Vec<i32>, Vec<f32>) {
    let mut workspace = SearchWorkspace::new(graph.n_vertices, k);
    greedy_search_with_workspace(
        query,
        data,
        dim,
        graph,
        trees,
        distance,
        k,
        epsilon,
        min_distance,
        rng,
        &mut workspace,
    )
}

/// Batch search for multiple queries.
#[cfg(feature = "rayon")]
pub fn batch_search<D: Distance<f32> + Sync>(
    queries: &[f32],
    n_queries: usize,
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    seed: u64,
) -> (Vec<i32>, Vec<f32>) {
    use rayon::prelude::*;

    let results: Vec<(Vec<i32>, Vec<f32>)> = (0..n_queries)
        .into_par_iter()
        .map(|i| {
            let query = &queries[i * dim..(i + 1) * dim];
            let mut rng = FastRng::new(seed.wrapping_add(i as u64));
            greedy_search(query, data, dim, graph, tree, distance, k, epsilon, min_distance, &mut rng)
        })
        .collect();

    // Flatten results
    let mut all_indices = Vec::with_capacity(n_queries * k);
    let mut all_distances = Vec::with_capacity(n_queries * k);

    for (indices, distances) in results {
        // Pad with -1 if fewer than k results
        all_indices.extend(indices.iter().copied());
        all_distances.extend(distances.iter().copied());
        
        let padding = k.saturating_sub(indices.len());
        all_indices.extend(std::iter::repeat(-1).take(padding));
        all_distances.extend(std::iter::repeat(f32::INFINITY).take(padding));
    }

    (all_indices, all_distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::SquaredEuclidean;

    fn create_simple_graph() -> (Vec<f32>, SearchGraph) {
        // 4 points in 2D arranged in a square
        let data = vec![
            0.0, 0.0,  // 0
            1.0, 0.0,  // 1
            0.0, 1.0,  // 2
            1.0, 1.0,  // 3
        ];

        // Each point connected to its neighbors
        let neighbors = vec![
            1, 2,  // 0 -> 1, 2
            0, 3,  // 1 -> 0, 3
            0, 3,  // 2 -> 0, 3
            1, 2,  // 3 -> 1, 2
        ];

        let graph = SearchGraph::from_dense(&neighbors, 4, 2);

        (data, graph)
    }

    #[test]
    fn test_greedy_search_basic() {
        let (data, graph) = create_simple_graph();
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        // Query near point 0
        let query = vec![0.1, 0.1];
        let (indices, distances) = greedy_search(
            &query, &data, 2, &graph, None, &distance, 2, 0.1, 0.0, &mut rng,
        );

        // Should find point 0 as closest
        assert!(!indices.is_empty());
        // Point 0 should be among the results
        assert!(indices.contains(&0) || distances[0] < 1.0);
    }

    #[test]
    fn test_indices_only_matches_full_search() {
        let (data, graph) = create_simple_graph();
        let distance = SquaredEuclidean;
        let query = [0.1, 0.1];
        let mut full_rng = FastRng::new(42);
        let mut indices_rng = FastRng::new(42);
        let mut full_workspace = SearchWorkspace::new(graph.n_vertices, 3);
        let mut indices_workspace = SearchWorkspace::new(graph.n_vertices, 3);

        let (expected, _) = greedy_search_with_workspace(
            &query, &data, 2, &graph, None, &distance, 3, 0.1, 0.0,
            &mut full_rng, &mut full_workspace,
        );
        let actual = greedy_search_indices_with_workspace(
            &query, &data, 2, &graph, None, &distance, 3, 0.1, 0.0,
            &mut indices_rng, &mut indices_workspace,
        );

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_quantized_i8_search_finds_exact_point() {
        let (_, graph) = create_simple_graph();
        let quantized_data = [0i8, 0, 127, 0, 0, 127, 127, 127];
        let inv_norms = [0.0, 1.0 / 127.0, 1.0 / 127.0, 1.0 / (2.0f32 * 127.0 * 127.0).sqrt()];
        let query = [127i8, 127];
        let query_inv_norm = (2.0f32 * 127.0 * 127.0).sqrt().recip();
        let mut rng = FastRng::new(42);
        let mut workspace = SearchWorkspace::new(graph.n_vertices, 1);

        let (indices, distances) = greedy_search_quantized_i8_with_workspace(
            &[1.0, 1.0],
            &query,
            query_inv_norm,
            &quantized_data,
            &inv_norms,
            2,
            &graph,
            None,
            1,
            0.1,
            0.0,
            &mut rng,
            &mut workspace,
        );

        assert_eq!(indices, vec![3]);
        assert!(distances[0].abs() < 1e-6);
    }

    #[test]
    fn test_greedy_search_exact_point() {
        let (data, graph) = create_simple_graph();
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        // Query exactly at point 3
        let query = vec![1.0, 1.0];
        let (indices, distances) = greedy_search(
            &query, &data, 2, &graph, None, &distance, 1, 0.1, 0.0, &mut rng,
        );

        // Should find point 3
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 3);
        assert!((distances[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_greedy_search_returns_k() {
        let (data, graph) = create_simple_graph();
        let distance = SquaredEuclidean;
        let mut rng = FastRng::new(42);

        let query = vec![0.5, 0.5];
        let (indices, distances) = greedy_search(
            &query, &data, 2, &graph, None, &distance, 4, 0.1, 0.0, &mut rng,
        );

        // Should find all 4 points
        assert_eq!(indices.len(), 4);
        assert_eq!(distances.len(), 4);

        // Distances should be sorted
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }
}
