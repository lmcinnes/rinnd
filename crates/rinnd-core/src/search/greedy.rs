//! Production greedy graph search for approximate nearest neighbors.

use crate::distance::affine::{AffineCodes, AffineQuery};
use crate::distance::{quantized::quantized_i8_dot, Distance};
use crate::graph::SearchGraph;
use crate::heap::{BoundedHeap, CandidateHeap};
use crate::rng::FastRng;
use crate::tree::{FlatTree, QuantizedFlatTree};
use crate::visited::VisitedSet;

/// Hint the CPU to fetch row `index` of `data` (`dim` values per row): whole
/// rows of up to 1 KiB, otherwise the first 8 cache lines (the hardware
/// prefetcher streams the rest once the row is being read).
#[inline(always)]
fn prefetch<T>(data: &[T], index: i32, dim: usize) {
    if index < 0 || index as usize * dim >= data.len() {
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_mm_prefetch, _MM_HINT_T0};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let base = data.as_ptr().add(index as usize * dim) as *const i8;
        let bytes = dim * core::mem::size_of::<T>();
        let span = if bytes <= 1024 { bytes } else { 512 };
        let mut offset = 0;
        while offset < span {
            _mm_prefetch(base.add(offset), _MM_HINT_T0);
            offset += 64;
        }
        if span > 0 {
            // Rows need not start on a cache line, so the span may end on one
            // more line than the stride covers.
            _mm_prefetch(base.add(span - 1), _MM_HINT_T0);
        }
    }
}

/// Crate-visible alias of [`prefetch`] for other search stages (reranking).
#[inline(always)]
pub(crate) fn prefetch_row<T>(data: &[T], index: i32, dim: usize) {
    prefetch(data, index, dim);
}

/// Reusable state for greedy graph search.
#[derive(Clone, Debug)]
pub struct SearchWorkspace {
    visited: VisitedSet,
    result_heap: BoundedHeap,
    seed_set: CandidateHeap,
    tree_frontier: Vec<(f32, usize)>,
    tree_leaves: Vec<(usize, usize)>,
    pending: Vec<i32>,
    k: usize,
    /// The search's results, sorted (see [`SearchWorkspace::sort_results`]).
    pub(crate) results: Vec<(f32, i32)>,
    /// Scratch for reranking.
    pub(crate) reranked: Vec<(f32, i32)>,
}

impl SearchWorkspace {
    pub fn new(n_points: usize, k: usize) -> Self {
        Self {
            visited: VisitedSet::new(n_points),
            result_heap: BoundedHeap::new(k),
            seed_set: CandidateHeap::with_capacity(k.saturating_mul(4).max(32)),
            tree_frontier: Vec::with_capacity(32),
            tree_leaves: Vec::with_capacity(8),
            pending: Vec::with_capacity(64),
            k,
            results: Vec::new(),
            reranked: Vec::new(),
        }
    }

    /// Move the result heap's entries into `results`, nearest first (ties by
    /// index), keeping both allocations for the next query.
    pub(crate) fn sort_results(&mut self) {
        self.result_heap.drain_sorted_into(&mut self.results);
    }

    #[inline]
    fn reset(&mut self, n_points: usize, k: usize) {
        if self.visited.capacity() == n_points {
            self.visited.clear();
        } else {
            self.visited = VisitedSet::new(n_points);
        }
        if self.k == k {
            self.result_heap.clear();
        } else {
            self.result_heap = BoundedHeap::new(k);
            self.k = k;
        }
        self.seed_set.clear();
    }
}

/// Seed a float search from the search trees' leaves: gather the unvisited leaf
/// points, prefetch their rows, then score them (four at a time where the
/// distance has a four-way kernel).
fn initialize_fp32<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    distance: &D,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) {
    let mut pending = std::mem::take(&mut workspace.pending);
    pending.clear();
    for tree in trees {
        tree.search_leaves(
            query,
            rng,
            tree_leaf_budget,
            &mut workspace.tree_frontier,
            &mut workspace.tree_leaves,
        );
        for &(start, end) in &workspace.tree_leaves {
            for &idx in &tree.indices[start..end] {
                if idx >= 0 && !workspace.visited.check_and_mark(idx) {
                    prefetch(data, idx, dim);
                    pending.push(idx);
                }
            }
        }
    }
    let row = |idx: i32| &data[idx as usize * dim..(idx as usize + 1) * dim];
    let mut rest = pending.as_slice();
    if D::USE_DISTANCE_FOUR && dim >= 32 {
        let mut chunks = pending.chunks_exact(4);
        for chunk in &mut chunks {
            let candidates = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let distances = distance.distance_four(query, candidates.map(row));
            for slot in 0..4 {
                workspace.result_heap.push(distances[slot], candidates[slot]);
                workspace.seed_set.push(distances[slot], candidates[slot]);
            }
        }
        rest = chunks.remainder();
    }
    for &idx in rest {
        let d = distance.distance(query, row(idx));
        workspace.result_heap.push(d, idx);
        workspace.seed_set.push(d, idx);
    }
    workspace.pending = pending;
}

fn initialize_quantized<F: Fn(i32) -> f32>(
    tree_query: &[f32],
    trees: &[FlatTree],
    quantized_trees: Option<&[QuantizedFlatTree]>,
    quantized_tree_query: &[i8],
    tree_query_scale: f32,
    tree_leaf_budget: usize,
    distance_to: &F,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) {
    let tree_count = quantized_trees.map_or(trees.len(), |value| value.len());
    for tree_index in 0..tree_count {
        let indices = if let Some(quantized) = quantized_trees {
            let tree = &quantized[tree_index];
            tree.search_leaves(
                quantized_tree_query,
                tree_query_scale,
                rng,
                tree_leaf_budget,
                &mut workspace.tree_frontier,
                &mut workspace.tree_leaves,
            );
            &tree.indices
        } else {
            let tree = &trees[tree_index];
            tree.search_leaves(
                tree_query,
                rng,
                tree_leaf_budget,
                &mut workspace.tree_frontier,
                &mut workspace.tree_leaves,
            );
            &tree.indices
        };
        for &(start, end) in &workspace.tree_leaves {
            for &idx in &indices[start..end] {
                if idx >= 0 && !workspace.visited.check_and_mark(idx) {
                    let d = distance_to(idx);
                    workspace.result_heap.push(d, idx);
                    workspace.seed_set.push(d, idx);
                }
            }
        }
    }
}

fn fill_random<F: Fn(i32) -> f32>(
    n_points: usize,
    k: usize,
    distance_to: &F,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) {
    for _ in 0..k.saturating_sub(workspace.seed_set.len()) {
        let idx = rng.next_index(n_points) as i32;
        if !workspace.visited.check_and_mark(idx) {
            let d = distance_to(idx);
            workspace.result_heap.push(d, idx);
            workspace.seed_set.push(d, idx);
        }
    }
}

/// The search radius: the worst retained distance relaxed by `epsilon`, or no
/// bound until the result heap is full.
#[inline(always)]
fn distance_bound(heap: &BoundedHeap, epsilon: f32, min_distance: f32) -> f32 {
    if !heap.is_full() {
        return f32::INFINITY;
    }
    let maximum = heap.max_distance();
    maximum + epsilon * (maximum - min_distance)
}

#[inline]
fn consider_neighbor(
    neighbor: i32,
    value: f32,
    epsilon: f32,
    min_distance: f32,
    bound: &mut f32,
    workspace: &mut SearchWorkspace,
) {
    if value < *bound {
        let inserted = workspace.result_heap.push(value, neighbor);
        workspace.seed_set.push(value, neighbor);
        if inserted {
            *bound = distance_bound(&workspace.result_heap, epsilon, min_distance);
        }
    }
}

fn traverse<T, F: Fn(i32) -> f32>(
    data: Option<&[T]>,
    dim: usize,
    graph: &SearchGraph,
    epsilon: f32,
    min_distance: f32,
    distance_to: &F,
    workspace: &mut SearchWorkspace,
) {
    let mut bound = distance_bound(&workspace.result_heap, epsilon, min_distance);
    let mut pending = std::mem::take(&mut workspace.pending);
    while let Some((vertex_distance, vertex)) = workspace.seed_set.pop() {
        if workspace.result_heap.is_full() && vertex_distance >= bound {
            break;
        }
        if let Some((_, next)) = workspace.seed_set.peek() {
            // The closest remaining candidate is likely expanded next.
            graph.prefetch_search_row(next);
        }
        // Gather the unvisited neighbors first, prefetching their vectors, so
        // that their memory accesses overlap.
        pending.clear();
        for &neighbor in graph.search_row(vertex as usize) {
            if neighbor < 0 || workspace.visited.check_and_mark(neighbor) {
                continue;
            }
            if let Some(values) = data {
                prefetch(values, neighbor, dim);
            }
            pending.push(neighbor);
        }
        for &neighbor in &pending {
            let d = distance_to(neighbor);
            if d < bound {
                let inserted = workspace.result_heap.push(d, neighbor);
                workspace.seed_set.push(d, neighbor);
                if inserted {
                    bound = distance_bound(&workspace.result_heap, epsilon, min_distance);
                }
            }
        }
    }
    workspace.pending = pending;
}

fn traverse_four<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    epsilon: f32,
    min_distance: f32,
    distance: &D,
    workspace: &mut SearchWorkspace,
) {
    let mut bound = distance_bound(&workspace.result_heap, epsilon, min_distance);
    let mut pending = std::mem::take(&mut workspace.pending);
    let row = |candidate: i32| &data[candidate as usize * dim..(candidate as usize + 1) * dim];
    while let Some((vertex_distance, vertex)) = workspace.seed_set.pop() {
        if workspace.result_heap.is_full() && vertex_distance >= bound {
            break;
        }
        if let Some((_, next)) = workspace.seed_set.peek() {
            // The closest remaining candidate is likely expanded next.
            graph.prefetch_search_row(next);
        }
        // Gather the unvisited neighbors first, prefetching their vectors, so
        // that their memory accesses overlap.
        pending.clear();
        for &neighbor in graph.search_row(vertex as usize) {
            if neighbor < 0 || workspace.visited.check_and_mark(neighbor) {
                continue;
            }
            prefetch(data, neighbor, dim);
            pending.push(neighbor);
        }
        let mut chunks = pending.chunks_exact(4);
        for chunk in &mut chunks {
            let candidates = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let distances = distance.distance_four(query, candidates.map(row));
            for position in 0..4 {
                consider_neighbor(candidates[position], distances[position], epsilon,
                                  min_distance, &mut bound, workspace);
            }
        }
        for &neighbor in chunks.remainder() {
            let value = distance.distance(query, row(neighbor));
            consider_neighbor(neighbor, value, epsilon, min_distance, &mut bound, workspace);
        }
    }
    workspace.pending = pending;
}

fn search<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> BoundedHeap {
    workspace.reset(graph.n_vertices, k);
    initialize_fp32(
        query,
        data,
        dim,
        trees,
        tree_leaf_budget,
        distance,
        rng,
        workspace,
    );
    let distance_to =
        |idx: i32| distance.distance(query, &data[idx as usize * dim..(idx as usize + 1) * dim]);
    // Random seeds only when the trees give none (see greedy_search_affine_into).
    if workspace.seed_set.is_empty() {
        fill_random(graph.n_vertices, k, &distance_to, rng, workspace);
    }
    if D::USE_DISTANCE_FOUR && dim >= 32 {
        traverse_four(query, data, dim, graph, epsilon, min_distance, distance, workspace);
    } else {
        traverse(
            Some(data), dim, graph, epsilon, min_distance, &distance_to, workspace,
        );
    }
    workspace.result_heap.clone()
}

pub fn greedy_search_with_workspace_initialized<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> (Vec<i32>, Vec<f32>) {
    search(
        query,
        data,
        dim,
        graph,
        trees,
        tree_leaf_budget,
        distance,
        k,
        epsilon,
        min_distance,
        rng,
        workspace,
    )
    .into_sorted()
}

pub fn greedy_search_with_workspace<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> (Vec<i32>, Vec<f32>) {
    greedy_search_with_workspace_initialized(
        query,
        data,
        dim,
        graph,
        trees,
        tree_leaf_budget,
        distance,
        k,
        epsilon,
        min_distance,
        rng,
        workspace,
    )
}

pub fn greedy_search_indices_with_workspace_initialized<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> Vec<i32> {
    search(
        query,
        data,
        dim,
        graph,
        trees,
        tree_leaf_budget,
        distance,
        k,
        epsilon,
        min_distance,
        rng,
        workspace,
    )
    .into_sorted_indices()
}

pub fn greedy_search_indices_with_workspace<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> Vec<i32> {
    greedy_search_indices_with_workspace_initialized(
        query,
        data,
        dim,
        graph,
        trees,
        tree_leaf_budget,
        distance,
        k,
        epsilon,
        min_distance,
        rng,
        workspace,
    )
}

pub fn greedy_search_quantized_i8_with_workspace_initialized(
    tree_query: &[f32],
    query: &[i8],
    query_inv_norm: f32,
    data: &[i8],
    inv_norms: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
    quantized_trees: Option<&[QuantizedFlatTree]>,
    tree_query_scale: f32,
) -> (Vec<i32>, Vec<f32>) {
    workspace.reset(graph.n_vertices, k);
    let distance_to = |idx: i32| {
        let idx = idx as usize;
        let similarity = quantized_i8_dot(query, &data[idx * dim..(idx + 1) * dim]) as f32
            * query_inv_norm
            * inv_norms[idx];
        1.0 - similarity.clamp(-1.0, 1.0)
    };
    initialize_quantized(
        tree_query,
        trees,
        quantized_trees,
        query,
        tree_query_scale,
        tree_leaf_budget,
        &distance_to,
        rng,
        workspace,
    );
    fill_random(graph.n_vertices, k, &distance_to, rng, workspace);
    traverse(
        Some(data),
        dim,
        graph,
        epsilon,
        min_distance,
        &distance_to,
        workspace,
    );
    workspace.result_heap.clone().into_sorted()
}

pub fn greedy_search_quantized_i8_with_workspace(
    tree_query: &[f32],
    query: &[i8],
    query_inv_norm: f32,
    data: &[i8],
    inv_norms: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> (Vec<i32>, Vec<f32>) {
    greedy_search_quantized_i8_with_workspace_initialized(
        tree_query,
        query,
        query_inv_norm,
        data,
        inv_norms,
        dim,
        graph,
        trees,
        tree_leaf_budget,
        k,
        epsilon,
        min_distance,
        rng,
        workspace,
        None,
        0.0,
    )
}

pub fn greedy_search_quantized_custom_with_workspace_initialized<F: Fn(i32) -> f32>(
    tree_query: &[f32],
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
    quantized_trees: Option<&[QuantizedFlatTree]>,
    quantized_tree_query: &[i8],
    tree_query_scale: f32,
    distance_to: F,
) -> (Vec<i32>, Vec<f32>) {
    workspace.reset(graph.n_vertices, k);
    initialize_quantized(
        tree_query,
        trees,
        quantized_trees,
        quantized_tree_query,
        tree_query_scale,
        tree_leaf_budget,
        &distance_to,
        rng,
        workspace,
    );
    fill_random(graph.n_vertices, k, &distance_to, rng, workspace);
    traverse::<i8, _>(
        None,
        0,
        graph,
        epsilon,
        min_distance,
        &distance_to,
        workspace,
    );
    workspace.result_heap.clone().into_sorted()
}

/// Traverse with affine 8-bit code distances: gather each expanded vertex's
/// unvisited neighbors (prefetching their code rows), then score them four
/// at a time.
fn traverse_affine(
    query: &AffineQuery,
    codes: &AffineCodes,
    graph: &SearchGraph,
    epsilon: f32,
    min_distance: f32,
    workspace: &mut SearchWorkspace,
) {
    let mut bound = distance_bound(&workspace.result_heap, epsilon, min_distance);
    let mut pending = std::mem::take(&mut workspace.pending);
    while let Some((vertex_distance, vertex)) = workspace.seed_set.pop() {
        if workspace.result_heap.is_full() && vertex_distance >= bound {
            break;
        }
        if let Some((_, next)) = workspace.seed_set.peek() {
            graph.prefetch_search_row(next);
        }
        pending.clear();
        for &neighbor in graph.search_row(vertex as usize) {
            if neighbor < 0 || workspace.visited.check_and_mark(neighbor) {
                continue;
            }
            codes.prefetch(neighbor, CODE_PREFETCH_LINES);
            pending.push(neighbor);
        }
        let mut chunks = pending.chunks_exact(4);
        for chunk in &mut chunks {
            let candidates = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let distances = codes.distance_four(query, candidates);
            for position in 0..4 {
                consider_neighbor(candidates[position], distances[position], epsilon,
                                  min_distance, &mut bound, workspace);
            }
        }
        for &neighbor in chunks.remainder() {
            let value = codes.distance(query, neighbor);
            consider_neighbor(neighbor, value, epsilon, min_distance, &mut bound, workspace);
        }
    }
    workspace.pending = pending;
}

/// Lines of a code row prefetched when it is queued for scoring: whole rows of
/// up to 1 KiB (the hardware prefetcher streams the rest of longer ones).
const CODE_PREFETCH_LINES: usize = 16;

/// Seed an affine-code search from the search trees' leaves: gather the unvisited
/// leaf points, prefetch their code rows, then score them four at a time.
fn initialize_affine(
    tree_query: &[f32],
    query: &AffineQuery,
    codes: &AffineCodes,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) {
    let mut pending = std::mem::take(&mut workspace.pending);
    pending.clear();
    for tree in trees {
        tree.search_leaves(
            tree_query,
            rng,
            tree_leaf_budget,
            &mut workspace.tree_frontier,
            &mut workspace.tree_leaves,
        );
        for &(start, end) in &workspace.tree_leaves {
            for &idx in &tree.indices[start..end] {
                if idx >= 0 && !workspace.visited.check_and_mark(idx) {
                    codes.prefetch(idx, CODE_PREFETCH_LINES);
                    pending.push(idx);
                }
            }
        }
    }
    let mut chunks = pending.chunks_exact(4);
    for chunk in &mut chunks {
        let candidates = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let distances = codes.distance_four(query, candidates);
        for slot in 0..4 {
            workspace.result_heap.push(distances[slot], candidates[slot]);
            workspace.seed_set.push(distances[slot], candidates[slot]);
        }
    }
    for &idx in chunks.remainder() {
        let d = codes.distance(query, idx);
        workspace.result_heap.push(d, idx);
        workspace.seed_set.push(d, idx);
    }
    workspace.pending = pending;
}

/// Greedy search over affine codes, leaving the best `k` by code distance in
/// the workspace's result heap (see [`SearchWorkspace::sort_results`]).
///
/// `tree_query` routes the float32 search trees; `query` is the same query
/// prepared for `codes` (in the coded space, which may be a projection).
/// Searches start from the trees' leaves only (as PyNNDescent's do): until the
/// result heap is full nothing bounds the search, so it fills from the leaves'
/// graph neighbourhoods rather than from random points, which are needed only
/// when the trees give no seeds.
pub fn greedy_search_affine_into(
    tree_query: &[f32],
    query: &AffineQuery,
    codes: &AffineCodes,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) {
    workspace.reset(graph.n_vertices, k);
    initialize_affine(tree_query, query, codes, trees, tree_leaf_budget, rng, workspace);
    if workspace.seed_set.is_empty() {
        let distance_to = |idx: i32| codes.distance(query, idx);
        fill_random(graph.n_vertices, k, &distance_to, rng, workspace);
    }
    traverse_affine(query, codes, graph, epsilon, min_distance, workspace);
}

/// Greedy search over affine codes, returning the best `k` by code distance
/// (rerank them with exact distances afterwards).
pub fn greedy_search_affine_with_workspace_initialized(
    tree_query: &[f32],
    query: &AffineQuery,
    codes: &AffineCodes,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    rng: &mut FastRng,
    workspace: &mut SearchWorkspace,
) -> (Vec<i32>, Vec<f32>) {
    greedy_search_affine_into(
        tree_query, query, codes, graph, trees, tree_leaf_budget, k, epsilon, min_distance, rng,
        workspace,
    );
    workspace.sort_results();
    (
        workspace.results.iter().map(|&(_, idx)| idx).collect(),
        workspace.results.iter().map(|&(d, _)| d).collect(),
    )
}

pub fn greedy_search<D: Distance<f32>>(
    query: &[f32],
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
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
        tree_leaf_budget,
        distance,
        k,
        epsilon,
        min_distance,
        rng,
        &mut workspace,
    )
}

#[cfg(feature = "rayon")]
pub fn batch_search<D: Distance<f32> + Sync>(
    queries: &[f32],
    n_queries: usize,
    data: &[f32],
    dim: usize,
    graph: &SearchGraph,
    trees: &[FlatTree],
    tree_leaf_budget: usize,
    distance: &D,
    k: usize,
    epsilon: f32,
    min_distance: f32,
    seed: u64,
) -> (Vec<i32>, Vec<f32>) {
    use rayon::prelude::*;
    let results: Vec<_> = (0..n_queries)
        .into_par_iter()
        .map(|index| {
            let mut rng = FastRng::new(seed.wrapping_add(index as u64));
            greedy_search(
                &queries[index * dim..(index + 1) * dim],
                data,
                dim,
                graph,
                trees,
                tree_leaf_budget,
                distance,
                k,
                epsilon,
                min_distance,
                &mut rng,
            )
        })
        .collect();
    let (mut indices, mut distances) = (
        Vec::with_capacity(n_queries * k),
        Vec::with_capacity(n_queries * k),
    );
    for (result_indices, result_distances) in results {
        let padding = k.saturating_sub(result_indices.len());
        indices.extend(result_indices);
        distances.extend(result_distances);
        indices.extend(std::iter::repeat(-1).take(padding));
        distances.extend(std::iter::repeat(f32::INFINITY).take(padding));
    }
    (indices, distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::SquaredEuclidean;

    #[test]
    fn batched_traversal_matches_individual_results() {
        check_batched_traversal(SquaredEuclidean);
        check_batched_traversal(crate::distance::Cosine);
        check_batched_traversal(crate::distance::AlternativeDot);
        check_batched_traversal(crate::distance::DirectNormalizedCosine);
        check_batched_traversal(crate::distance::InnerProduct);
        check_batched_traversal(crate::distance::Dot);
    }

    fn check_batched_traversal<D: Distance<f32>>(distance: D) {
        #[derive(Clone)]
        struct Individual<D>(D);

        impl<D: Distance<f32>> Distance<f32> for Individual<D> {
            fn distance(&self, query: &[f32], candidate: &[f32]) -> f32 {
                self.0.distance(query, candidate)
            }

            fn name(&self) -> &'static str { self.0.name() }
        }

        let points = 97;
        let mut graph = SearchGraph::new(points);
        for point in 0..points {
            for offset in 0..19 {
                graph.indices.push(((point * 7 + offset * 3) % points) as i32);
            }
            graph.indices.push(-1);
            graph.indices.push(graph.indices[graph.indices.len() - 2]);
            graph.indptr[point + 1] = graph.indices.len() as i32;
        }
        for dimension in [31, 100, 784] {
            let mut rng = FastRng::new(77);
            let mut data: Vec<f32> = (0..points * dimension).map(|_| (rng.next_float() - 0.5) / dimension as f32).collect();
            data[..dimension].fill(0.0);
            let mut workspace = SearchWorkspace::new(points, 10);
            for epsilon in [0.0, 0.1, 0.3, 0.6, 1.0] {
                for query_row in 0..8 {
                    let query = &data[query_row * dimension..(query_row + 1) * dimension];
                    let expected = greedy_search_with_workspace(
                        query, &data, dimension, &graph, &[], 1, &Individual(distance.clone()), 10,
                        epsilon, 0.0, &mut FastRng::new(42), &mut workspace,
                    );
                    let actual = greedy_search_with_workspace(
                        query, &data, dimension, &graph, &[], 1, &distance, 10,
                        epsilon, 0.0, &mut FastRng::new(42), &mut workspace,
                    );
                    assert_eq!(actual.0, expected.0);
                    assert_eq!(actual.1.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                               expected.1.iter().map(|value| value.to_bits()).collect::<Vec<_>>());
                }
            }
        }
    }

    #[test]
    fn padded_rows_match_csr_search() {
        let points = 97;
        let mut graph = SearchGraph::new(points);
        for point in 0..points {
            // Degrees vary (up to 40, so rows span several cache lines) and rows
            // contain -1 entries and repeated neighbors.
            for offset in 0..(point % 41) {
                graph.indices.push(((point * 7 + offset * 3) % points) as i32);
                if offset % 5 == 4 {
                    graph.indices.push(-1);
                }
            }
            graph.indices.push(((point + 1) % points) as i32);
            graph.indptr[point + 1] = graph.indices.len() as i32;
        }
        let mut padded = graph.clone();
        padded.build_padded();
        for vertex in 0..points {
            let expected: Vec<i32> =
                graph.neighbors(vertex).iter().copied().filter(|&neighbor| neighbor >= 0).collect();
            let row = padded.search_row(vertex);
            assert_eq!(row, &expected[..]);
            // Slot 0 holds the degree, at the start of a cache line.
            assert_eq!((row.as_ptr() as usize - 4) % 64, 0);
        }

        let dimension = 100;
        let mut rng = FastRng::new(5);
        let data: Vec<f32> = (0..points * dimension).map(|_| rng.next_float() - 0.5).collect();
        let mut workspace = SearchWorkspace::new(points, 10);
        for epsilon in [0.0, 0.1, 0.5] {
            for query_row in 0..8 {
                let query = &data[query_row * dimension..(query_row + 1) * dimension];
                let expected = greedy_search_with_workspace(
                    query, &data, dimension, &graph, &[], 1, &SquaredEuclidean, 10, epsilon,
                    0.0, &mut FastRng::new(42), &mut workspace,
                );
                let actual = greedy_search_with_workspace(
                    query, &data, dimension, &padded, &[], 1, &SquaredEuclidean, 10, epsilon,
                    0.0, &mut FastRng::new(42), &mut workspace,
                );
                assert_eq!(actual, expected);
            }
        }
    }
}
