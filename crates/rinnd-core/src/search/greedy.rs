//! Production greedy graph search for approximate nearest neighbors.

use crate::distance::{quantized::quantized_i8_dot, Distance};
use crate::graph::SearchGraph;
use crate::heap::{BoundedHeap, CandidateHeap};
use crate::rng::FastRng;
use crate::tree::{FlatTree, QuantizedFlatTree};
use crate::visited::VisitedSet;

const PREFETCH_LOOKAHEAD: usize = 4;

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
        _mm_prefetch(
            data.as_ptr().add(index as usize * dim) as *const i8,
            _MM_HINT_T0,
        );
    }
}

/// Reusable state for greedy graph search.
#[derive(Clone, Debug)]
pub struct SearchWorkspace {
    visited: VisitedSet,
    result_heap: BoundedHeap,
    seed_set: CandidateHeap,
    tree_frontier: Vec<(f32, usize)>,
    tree_leaves: Vec<(usize, usize)>,
    k: usize,
}

impl SearchWorkspace {
    pub fn new(n_points: usize, k: usize) -> Self {
        Self {
            visited: VisitedSet::new(n_points),
            result_heap: BoundedHeap::new(k),
            seed_set: CandidateHeap::with_capacity(k.saturating_mul(4).max(32)),
            tree_frontier: Vec::with_capacity(32),
            tree_leaves: Vec::with_capacity(8),
            k,
        }
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
                    let d = distance
                        .distance(query, &data[idx as usize * dim..(idx as usize + 1) * dim]);
                    workspace.result_heap.push(d, idx);
                    workspace.seed_set.push(d, idx);
                }
            }
        }
    }
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
            let maximum = workspace.result_heap.max_distance();
            *bound = maximum + epsilon * (maximum - min_distance);
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
    let mut max_distance = workspace.result_heap.max_distance();
    let mut distance_bound = max_distance + epsilon * (max_distance - min_distance);
    while let Some((vertex_distance, vertex)) = workspace.seed_set.pop() {
        if workspace.result_heap.is_full() && vertex_distance >= distance_bound {
            break;
        }
        let neighbors = graph.neighbors(vertex as usize);
        for (position, &neighbor) in neighbors.iter().enumerate() {
            if let (Some(values), Some(&upcoming)) =
                (data, neighbors.get(position + PREFETCH_LOOKAHEAD))
            {
                prefetch(values, upcoming, dim);
            }
            if neighbor < 0 || workspace.visited.check_and_mark(neighbor) {
                continue;
            }
            let d = distance_to(neighbor);
            if d < distance_bound {
                let inserted = workspace.result_heap.push(d, neighbor);
                workspace.seed_set.push(d, neighbor);
                if inserted {
                    max_distance = workspace.result_heap.max_distance();
                    distance_bound = max_distance + epsilon * (max_distance - min_distance);
                }
            }
        }
    }
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
    let maximum = workspace.result_heap.max_distance();
    let mut bound = maximum + epsilon * (maximum - min_distance);
    while let Some((vertex_distance, vertex)) = workspace.seed_set.pop() {
        if workspace.result_heap.is_full() && vertex_distance >= bound {
            break;
        }
        let neighbors = graph.neighbors(vertex as usize);
        let mut pending = [0; 4];
        let mut count = 0;
        for (position, &neighbor) in neighbors.iter().enumerate() {
            if let Some(&upcoming) = neighbors.get(position + PREFETCH_LOOKAHEAD) {
                prefetch(data, upcoming, dim);
            }
            if neighbor < 0 || workspace.visited.check_and_mark(neighbor) {
                continue;
            }
            pending[count] = neighbor;
            count += 1;
            if count == 4 {
                let vectors = pending.map(|candidate| {
                    &data[candidate as usize * dim..(candidate as usize + 1) * dim]
                });
                let distances = distance.distance_four(query, vectors);
                for candidate in 0..4 {
                    consider_neighbor(pending[candidate], distances[candidate], epsilon, min_distance,
                                      &mut bound, workspace);
                }
                count = 0;
            }
        }
        for &neighbor in &pending[..count] {
            let value = distance.distance(query, &data[neighbor as usize * dim..(neighbor as usize + 1) * dim]);
            consider_neighbor(neighbor, value, epsilon, min_distance, &mut bound, workspace);
        }
    }
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
    fill_random(graph.n_vertices, k, &distance_to, rng, workspace);
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
}
