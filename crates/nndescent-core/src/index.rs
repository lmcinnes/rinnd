//! Main NNDescent index structure and builder.

use crate::distance::{Distance, SquaredEuclidean, Euclidean, Cosine, InnerProduct, Metric};
use crate::graph::{NeighborGraph, SearchGraph};
use crate::heap::NeighborHeap;
use crate::nndescent::{nn_descent, NNDescentParams};
use crate::rng::FastRng;
use crate::search::{
    greedy_search_indices_with_workspace_initialized,
    greedy_search_quantized_custom_with_workspace_initialized,
    greedy_search_quantized_i8_with_workspace_initialized,
    greedy_search_with_workspace_initialized,
    SearchWorkspace,
};
use crate::tree::{FlatTree, QuantizedFlatTree};
use rayon::prelude::*;

/// The main NNDescent index for approximate nearest neighbor search.
///
/// This struct holds all the data needed for querying:
/// - The original data points
/// - The search graph (diversified k-NN graph)
/// - Search tree for initialization
/// - Distance function
pub struct NNDescentIndex<D: Distance<f32>> {
    /// Data points (flattened, n_points × dim)
    pub data: Vec<f32>,
    /// Number of data points
    pub n_points: usize,
    /// Dimension of data points
    pub dim: usize,
    /// Distance function
    pub distance: D,
    /// Distance correction function (e.g., sqrt for squared euclidean)
    pub distance_correction: Option<fn(f32) -> f32>,
    /// Number of neighbors in the graph
    pub n_neighbors: usize,
    /// The k-NN graph (indices, shape n_points × n_neighbors)
    pub neighbor_indices: Vec<i32>,
    /// The k-NN graph (distances, shape n_points × n_neighbors)
    pub neighbor_distances: Vec<f32>,
    /// Search graph (CSR format)
    pub search_graph: SearchGraph,
    /// RP trees retained for query initialization.
    pub search_trees: Vec<FlatTree>,
    /// Maximum leaves visited per retained RP tree during query initialization.
    pub search_tree_leaf_budget: usize,
    /// Vertex ordering (for tree leaf order)
    pub vertex_order: Vec<usize>,
    /// Minimum distance in graph (for epsilon scaling)
    pub min_distance: f32,
    /// RNG seed for search
    rng_seed: u64,
}

fn resolve_quantized_candidate_width(k: usize, candidate_width: usize, rerank_width: usize) -> usize {
    let resolved = if candidate_width == 0 {
        k.max(rerank_width)
    } else {
        candidate_width
    };
    assert!(resolved >= k, "quantized candidate width must be at least k");
    assert!(rerank_width <= resolved, "quantized rerank width must not exceed candidate width");
    assert!(candidate_width == 0 || rerank_width == 0 || rerank_width >= k,
        "quantized rerank width must be zero or at least k");
    resolved
}

#[inline]
fn should_rerank_quantized(k: usize, candidate_width: usize, rerank_width: usize) -> bool {
    if candidate_width == 0 {
        // Preserve the old constructor/API behavior when only rerank_width was
        // supplied: widths at or below k did not invoke exact reranking.
        rerank_width > k
    } else {
        rerank_width > 0
    }
}

impl<D: Distance<f32>> NNDescentIndex<D> {
    /// Bytes in retained FP32 arrays used by production APIs, FP32 graph/tree
    /// initialization, or optional exact reranking.
    pub fn retained_fp32_bytes(&self) -> usize {
        let tree_floats: usize = self.search_trees.iter()
            .map(|tree| tree.hyperplanes.len() + tree.offsets.len())
            .sum();
        (self.data.len() + self.neighbor_distances.len() + tree_floats)
            * std::mem::size_of::<f32>()
    }

    fn rerank_quantized_candidates(
        &self,
        query: &[f32],
        approximate_indices: &[i32],
        k: usize,
        rerank_width: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut reranked: Vec<(f32, i32)> = approximate_indices
            .iter()
            .copied()
            .filter(|&idx| idx >= 0)
            .take(rerank_width)
            .map(|idx| {
                let point = &self.data[idx as usize * self.dim..(idx as usize + 1) * self.dim];
                (self.distance.distance(query, point), idx)
            })
            .collect();
        reranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        reranked.truncate(k);
        (
            reranked.iter().map(|&(_, idx)| idx).collect(),
            reranked.iter().map(|&(distance, _)| distance).collect(),
        )
    }

    fn original_to_internal(&self) -> Vec<usize> {
        let mut inverse = vec![0; self.n_points];
        for (internal_id, &original_id) in self.vertex_order.iter().enumerate() {
            inverse[original_id] = internal_id;
        }
        inverse
    }

    /// Return the search graph in original input-ID order.
    pub fn search_graph_original_order(&self) -> SearchGraph {
        let mut graph = self.search_graph.clone();
        graph.reorder(&self.original_to_internal());
        graph
    }

    /// Return the first retained search tree with leaf IDs mapped back to
    /// original input IDs.
    pub fn export_search_tree(&self) -> Option<FlatTree> {
        self.search_trees.first().map(|tree| {
            let mut exported = tree.clone();
            for idx in &mut exported.indices {
                debug_assert!(*idx >= 0 && (*idx as usize) < self.vertex_order.len());
                *idx = self.vertex_order[*idx as usize] as i32;
            }
            exported
        })
    }

    #[inline]
    fn process_query_with_workspace(
        &self,
        query: &[f32],
        query_id: usize,
        k: usize,
        epsilon: f32,
        workspace: &mut SearchWorkspace,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut rng = FastRng::new(self.rng_seed.wrapping_add(query_id as u64));

        let (mut indices, mut distances) = greedy_search_with_workspace_initialized(
            query,
            &self.data,
            self.dim,
            &self.search_graph,
            &self.search_trees,
            self.search_tree_leaf_budget,
            &self.distance,
            k,
            epsilon,
            self.min_distance,
            &mut rng,
            workspace,
        );

        for idx in &mut indices {
            if *idx >= 0 {
                *idx = self.vertex_order[*idx as usize] as i32;
            }
        }

        if let Some(correction) = self.distance_correction {
            for d in &mut distances {
                *d = correction(*d);
            }
        }

        while indices.len() < k {
            indices.push(-1);
            distances.push(f32::INFINITY);
        }

        indices.truncate(k);
        distances.truncate(k);
        (indices, distances)
    }

    #[inline]
    fn process_query_indices_with_workspace(
        &self,
        query: &[f32],
        query_id: usize,
        k: usize,
        epsilon: f32,
        workspace: &mut SearchWorkspace,
    ) -> Vec<i32> {
        let mut rng = FastRng::new(self.rng_seed.wrapping_add(query_id as u64));
        let mut indices = greedy_search_indices_with_workspace_initialized(
            query,
            &self.data,
            self.dim,
            &self.search_graph,
            &self.search_trees,
            self.search_tree_leaf_budget,
            &self.distance,
            k,
            epsilon,
            self.min_distance,
            &mut rng,
            workspace,
        );

        for idx in &mut indices {
            if *idx >= 0 {
                *idx = self.vertex_order[*idx as usize] as i32;
            }
        }

        indices.resize(k, -1);
        indices.truncate(k);
        indices
    }

    #[inline]
    fn process_query(&self, query: &[f32], query_id: usize, k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>) {
        let mut workspace = SearchWorkspace::new(self.n_points, k);
        self.process_query_with_workspace(query, query_id, k, epsilon, &mut workspace)
    }

    /// Query for a single nearest-neighbor request using a reusable workspace.
    pub fn query_one_with_workspace(
        &self,
        query: &[f32],
        k: usize,
        epsilon: f32,
        workspace: &mut SearchWorkspace,
    ) -> (Vec<i32>, Vec<f32>) {
        self.process_query_with_workspace(query, 0, k, epsilon, workspace)
    }

    /// Query one vector and return only original input IDs, avoiding distance
    /// result allocation and correction.
    pub fn query_one_indices_with_workspace(
        &self,
        query: &[f32],
        k: usize,
        epsilon: f32,
        workspace: &mut SearchWorkspace,
    ) -> Vec<i32> {
        self.process_query_indices_with_workspace(query, 0, k, epsilon, workspace)
    }

    /// Query one vector against a signed-int8 search representation that is
    /// stored in this index's internal physical vertex order.
    ///
    /// `candidate_width` is the approximate result-heap capacity. A zero width
    /// preserves the legacy behavior by resolving to `max(k, rerank_width)`;
    /// otherwise it must be at least `k` and at least `rerank_width`.
    /// `rerank_width` controls only how many retained approximate finalists are
    /// evaluated and sorted with the index's exact FP32 distance.
    pub fn query_one_quantized_i8_with_workspace(
        &self,
        tree_query: &[f32],
        query: &[i8],
        query_inv_norm: f32,
        quantized_data: &[i8],
        inv_norms: &[f32],
        quantized_trees: Option<&[QuantizedFlatTree]>,
        tree_query_scale: f32,
        k: usize,
        candidate_width: usize,
        rerank_width: usize,
        epsilon: f32,
        workspace: &mut SearchWorkspace,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut rng = FastRng::new(self.rng_seed);
        let search_k = resolve_quantized_candidate_width(k, candidate_width, rerank_width);
        let (mut indices, mut distances) = greedy_search_quantized_i8_with_workspace_initialized(
            tree_query,
            query,
            query_inv_norm,
            quantized_data,
            inv_norms,
            self.dim,
            &self.search_graph,
            &self.search_trees,
            self.search_tree_leaf_budget,
            search_k,
            epsilon,
            self.min_distance,
            &mut rng,
            workspace,
            quantized_trees,
            tree_query_scale,
        );

        if should_rerank_quantized(k, candidate_width, rerank_width) {
            (indices, distances) = self.rerank_quantized_candidates(
                tree_query, &indices, k, rerank_width,
            );
        }

        for idx in &mut indices {
            if *idx >= 0 {
                *idx = self.vertex_order[*idx as usize] as i32;
            }
        }
        indices.resize(k, -1);
        distances.resize(k, f32::INFINITY);
        indices.truncate(k);
        distances.truncate(k);
        (indices, distances)
    }

    /// Query a quantized representation whose candidate distance requires a
    /// representation-specific reconstruction kernel.
    pub fn query_one_quantized_custom_with_workspace<F>(
        &self,
        tree_query: &[f32],
        k: usize,
        candidate_width: usize,
        rerank_width: usize,
        epsilon: f32,
        workspace: &mut SearchWorkspace,
        quantized_tree_query: &[i8],
        quantized_trees: Option<&[QuantizedFlatTree]>,
        tree_query_scale: f32,
        distance_to: F,
    ) -> (Vec<i32>, Vec<f32>)
    where
        F: Fn(i32) -> f32,
    {
        let mut rng = FastRng::new(self.rng_seed);
        let search_k = resolve_quantized_candidate_width(k, candidate_width, rerank_width);
        let (mut indices, mut distances) = greedy_search_quantized_custom_with_workspace_initialized(
            tree_query, &self.search_graph, &self.search_trees, self.search_tree_leaf_budget,
            search_k, epsilon, self.min_distance, &mut rng, workspace,
            quantized_trees, quantized_tree_query, tree_query_scale,
            distance_to,
        );

        if should_rerank_quantized(k, candidate_width, rerank_width) {
            (indices, distances) = self.rerank_quantized_candidates(
                tree_query, &indices, k, rerank_width,
            );
        }
        for idx in &mut indices {
            if *idx >= 0 { *idx = self.vertex_order[*idx as usize] as i32; }
        }
        indices.resize(k, -1);
        distances.resize(k, f32::INFINITY);
        indices.truncate(k);
        distances.truncate(k);
        (indices, distances)
    }

    /// Query for a single nearest-neighbor request.
    pub fn query_one(&self, query: &[f32], k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>) {
        self.process_query(query, 0, k, epsilon)
    }

    /// Query for the k nearest neighbors of query points.
    ///
    /// # Arguments
    /// * `queries` - Query points (flattened, n_queries × dim)
    /// * `n_queries` - Number of query points
    /// * `k` - Number of neighbors to return
    /// * `epsilon` - Search expansion factor (0.0-0.5 recommended)
    ///
    /// # Returns
    /// (indices, distances) where:
    /// - indices: shape (n_queries × k), neighbor indices in original data order
    /// - distances: shape (n_queries × k), distances to neighbors
    pub fn query(
        &self,
        queries: &[f32],
        n_queries: usize,
        k: usize,
        epsilon: f32,
    ) -> (Vec<i32>, Vec<f32>) {
        let results: Vec<(Vec<i32>, Vec<f32>)> = if n_queries <= 4 {
            let mut workspace = SearchWorkspace::new(self.n_points, k);
            (0..n_queries)
                .map(|i| {
                    let query = &queries[i * self.dim..(i + 1) * self.dim];
                    self.process_query_with_workspace(query, i, k, epsilon, &mut workspace)
                })
                .collect()
        } else {
            (0..n_queries)
                .into_par_iter()
                .map(|i| {
                    let query = &queries[i * self.dim..(i + 1) * self.dim];
                    self.process_query(query, i, k, epsilon)
                })
                .collect()
        };

        let mut all_indices = Vec::with_capacity(n_queries * k);
        let mut all_distances = Vec::with_capacity(n_queries * k);
        for (indices, distances) in results {
            all_indices.extend_from_slice(&indices);
            all_distances.extend_from_slice(&distances);
        }

        (all_indices, all_distances)
    }

    /// Get the neighbor graph (k-NN graph before search preparation).
    pub fn neighbor_graph(&self) -> Option<NeighborGraph> {
        // This would need to be stored if we want to return it
        None
    }
}

/// Builder for NNDescentIndex.
pub struct NNDescentBuilder<'a> {
    data: &'a [f32],
    n_points: usize,
    dim: usize,
    metric: Metric,
    n_neighbors: usize,
    n_trees: usize,
    n_search_trees: usize,
    search_tree_leaf_budget: usize,
    leaf_size: Option<usize>,
    max_candidates: Option<usize>,
    n_iters: Option<usize>,
    delta: f32,
    random_seed: u64,
    verbose: bool,
    diversify_prob: f32,
    pruning_degree_multiplier: f32,
}

impl<'a> NNDescentBuilder<'a> {
    /// Create a new builder.
    ///
    /// # Arguments
    /// * `data` - Flattened data array (n_points × dim)
    /// * `n_points` - Number of data points
    /// * `dim` - Dimension of each point
    pub fn new(data: &'a [f32], n_points: usize, dim: usize) -> Self {
        Self {
            data,
            n_points,
            dim,
            metric: Metric::Euclidean,
            n_neighbors: 30,
            n_trees: 8,
            n_search_trees: 1,
            search_tree_leaf_budget: 1,
            leaf_size: None,
            max_candidates: None,
            n_iters: None,
            delta: 0.001,
            random_seed: 42,
            verbose: false,
            diversify_prob: 1.0,
            pruning_degree_multiplier: 1.5,
        }
    }

    /// Set the distance metric.
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the distance metric from a string.
    pub fn metric_str(mut self, metric: &str) -> Self {
        if let Some(m) = Metric::from_str(metric) {
            self.metric = m;
        }
        self
    }

    /// Set the number of neighbors.
    pub fn n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n;
        self
    }

    /// Set the number of RP trees.
    pub fn n_trees(mut self, n: usize) -> Self {
        self.n_trees = n;
        self
    }

    /// Set the number of construction RP trees retained for query seeding.
    ///
    /// This is independent of `n_trees`, which controls construction. Values
    /// larger than the constructed forest are clamped to the forest size.
    pub fn n_search_trees(mut self, n: usize) -> Self {
        self.n_search_trees = n;
        self
    }

    /// Set the number of leaves visited in each retained search tree.
    pub fn search_tree_leaf_budget(mut self, n: usize) -> Self {
        self.search_tree_leaf_budget = n.max(1);
        self
    }

    /// Set the leaf size for RP trees.
    pub fn leaf_size(mut self, size: usize) -> Self {
        self.leaf_size = Some(size);
        self
    }

    /// Set the maximum candidates per iteration.
    pub fn max_candidates(mut self, n: usize) -> Self {
        self.max_candidates = Some(n);
        self
    }

    /// Set the number of NN-descent iterations.
    pub fn n_iters(mut self, n: usize) -> Self {
        self.n_iters = Some(n);
        self
    }

    /// Set the convergence delta.
    pub fn delta(mut self, d: f32) -> Self {
        self.delta = d;
        self
    }

    /// Set the random seed.
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Enable verbose output.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Set the diversify probability (0.0 = no diversification, 1.0 = full).
    pub fn diversify_prob(mut self, prob: f32) -> Self {
        self.diversify_prob = prob;
        self
    }

    /// Set the pruning degree multiplier (max degree = multiplier * n_neighbors).
    pub fn pruning_degree_multiplier(mut self, mult: f32) -> Self {
        self.pruning_degree_multiplier = mult;
        self
    }

    /// Build the index with Euclidean distance.
    pub fn build_euclidean(self) -> NNDescentIndex<SquaredEuclidean> {
        self.build_with_distance(SquaredEuclidean, Some(|d: f32| d.sqrt()))
    }

    /// Build the index with Cosine distance.
    pub fn build_cosine(self) -> NNDescentIndex<Cosine> {
        self.build_with_distance(Cosine, None)
    }

    /// Build the index with Inner Product distance.
    pub fn build_inner_product(self) -> NNDescentIndex<InnerProduct> {
        self.build_with_distance(InnerProduct, None)
    }

    /// Build the index with a custom distance function.
    pub fn build_with_distance<D: Distance<f32>>(
        self,
        distance: D,
        correction: Option<fn(f32) -> f32>,
    ) -> NNDescentIndex<D> {
        let angular = matches!(
            self.metric,
            Metric::Cosine
                | Metric::InnerProduct
                | Metric::Dot
                | Metric::Correlation
                | Metric::TrueAngular
                | Metric::TSSS
        );

        // Compute default parameters based on data size
        let leaf_size = self.leaf_size.unwrap_or_else(|| {
            (5 * self.n_neighbors).min(256).max(60)
        });
        let max_candidates = self.max_candidates.unwrap_or_else(|| {
            self.n_neighbors.min(60)
        });
        let n_iters = self.n_iters.unwrap_or_else(|| {
            ((self.n_points as f64).log2().ceil() as usize).max(5)
        });

        let params = NNDescentParams {
            n_neighbors: self.n_neighbors,
            n_trees: self.n_trees,
            leaf_size,
            max_candidates,
            n_iters,
            delta: self.delta,
            angular,
            max_depth: 200,
            verbose: self.verbose,
        };

        let mut rng = FastRng::new(self.random_seed);

        // Run NN-descent
        let (mut neighbor_heap, forest) = nn_descent(
            self.data,
            self.n_points,
            self.dim,
            &distance,
            &params,
            &mut rng,
        );

        // Sort the heap so neighbors are in ascending distance order
        // (matches PyNNDescent's deheap_sort behavior)
        neighbor_heap.sort_all();

        // Build search graph with diversification and pruning (matching PyNNDescent pipeline)
        let (mut search_graph, min_distance) = SearchGraph::from_dense_diversified(
            &neighbor_heap.indices,
            &neighbor_heap.distances,
            self.data,
            self.n_points,
            self.n_neighbors,
            self.dim,
            &distance,
            self.diversify_prob,
            self.pruning_degree_multiplier,
        );

        // Apply distance correction to stored distances if needed
        let neighbor_distances = if let Some(corr) = correction {
            neighbor_heap.distances.iter().map(|&d| corr(d)).collect()
        } else {
            neighbor_heap.distances.clone()
        };

        // Use the first retained tree for physical search-time layout and keep
        // the requested prefix of the forest for query seeding. This preserves
        // the current one-tree behavior by default while allowing entry-quality
        // experiments independently of construction forest size.
        let mut search_trees: Vec<FlatTree> = forest
            .into_iter()
            .take(self.n_search_trees)
            .collect();
        let tree_order: Vec<usize> = search_trees
            .first()
            .map(|tree| tree.indices.iter().map(|&idx| idx as usize).collect())
            .unwrap_or_else(|| (0..self.n_points).collect());
        let vertex_order = tree_order;

        assert_eq!(
            vertex_order.len(),
            self.n_points,
            "search tree order must include every point"
        );
        let mut old_to_new = vec![usize::MAX; self.n_points];
        for (new_idx, &old_idx) in vertex_order.iter().enumerate() {
            assert!(
                old_idx < self.n_points,
                "search tree contains an out-of-range point ID"
            );
            assert_eq!(
                old_to_new[old_idx],
                usize::MAX,
                "search tree order contains a duplicate point ID"
            );
            old_to_new[old_idx] = new_idx;
        }

        let mut reordered_data = Vec::with_capacity(self.data.len());
        for &old_idx in &vertex_order {
            let start = old_idx * self.dim;
            reordered_data.extend_from_slice(&self.data[start..start + self.dim]);
        }

        search_graph.reorder(&vertex_order);
        for tree in &mut search_trees {
            tree.remap_indices(&old_to_new);
        }

        NNDescentIndex {
            data: reordered_data,
            n_points: self.n_points,
            dim: self.dim,
            distance,
            distance_correction: correction,
            n_neighbors: self.n_neighbors,
            neighbor_indices: neighbor_heap.indices,
            neighbor_distances,
            search_graph,
            search_trees,
            search_tree_leaf_budget: self.search_tree_leaf_budget,
            vertex_order,
            min_distance,
            rng_seed: self.random_seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize, dim: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(n * dim);
        for i in 0..n {
            for j in 0..dim {
                // Create data with some structure
                data.push(((i * dim + j) as f32 * 0.1).sin());
            }
        }
        data
    }

    #[test]
    fn test_builder_basic() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);

        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(2)
            .n_iters(3)
            .verbose(false)
            .build_euclidean();

        assert_eq!(index.n_points, n);
        assert_eq!(index.dim, dim);
    }

    #[test]
    fn test_query_basic() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);

        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(2)
            .n_iters(3)
            .build_euclidean();

        // Query with first point - should find itself or very close neighbors
        let query = &data[0..dim];
        let (indices, distances) = index.query(query, 1, 5, 0.1);

        assert_eq!(indices.len(), 5);
        assert_eq!(distances.len(), 5);

        // First result should be very close (ideally the query point itself)
        assert!(distances[0] < 1.0);
    }

    #[test]
    fn test_query_multiple() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);

        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(2)
            .n_iters(3)
            .build_euclidean();

        // Query with multiple points
        let queries = &data[0..dim * 3]; // First 3 points
        let (indices, distances) = index.query(queries, 3, 5, 0.1);

        assert_eq!(indices.len(), 15); // 3 queries × 5 neighbors
        assert_eq!(distances.len(), 15);
    }

    #[test]
    fn test_search_layout_uses_tree_order_and_preserves_original_results() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);

        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(2)
            .n_iters(3)
            .build_euclidean();

        assert_eq!(index.vertex_order.len(), n);
        let mut sorted_order = index.vertex_order.clone();
        sorted_order.sort_unstable();
        assert_eq!(sorted_order, (0..n).collect::<Vec<_>>());

        for (new_idx, &old_idx) in index.vertex_order.iter().enumerate() {
            assert_eq!(
                &index.data[new_idx * dim..(new_idx + 1) * dim],
                &data[old_idx * dim..(old_idx + 1) * dim],
            );
        }

        let tree = index.search_trees.first().expect("search tree");
        assert_eq!(tree.indices, (0..n as i32).collect::<Vec<_>>());

        let exported_graph = index.search_graph_original_order();
        let mut round_trip_graph = exported_graph.clone();
        round_trip_graph.reorder(&index.vertex_order);
        assert_eq!(round_trip_graph.indptr, index.search_graph.indptr);
        assert_eq!(round_trip_graph.indices, index.search_graph.indices);

        let query_id = 37usize;
        let query = &data[query_id * dim..(query_id + 1) * dim];
        let (indices, _) = index.query_one(query, 10, 0.1);
        assert!(indices.contains(&(query_id as i32)));
    }

    #[test]
    fn test_multiple_search_trees_are_retained_and_remapped() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);

        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(3)
            .n_search_trees(2)
            .n_iters(3)
            .build_euclidean();

        assert_eq!(index.search_trees.len(), 2);
        for tree in &index.search_trees {
            let mut indices = tree.indices.clone();
            indices.sort_unstable();
            assert_eq!(indices, (0..n as i32).collect::<Vec<_>>());
        }

        let query_id = 37usize;
        let query = &data[query_id * dim..(query_id + 1) * dim];
        let (indices, _) = index.query_one(query, 10, 0.1);
        assert!(indices.contains(&(query_id as i32)));
    }

    #[test]
    fn test_exported_tree_uses_original_input_ids() {
        let n = 100;
        let dim = 10;
        let data = create_test_data(n, dim);
        let index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(2)
            .n_iters(3)
            .build_euclidean();

        let tree = index.export_search_tree().expect("retained search tree");
        let mut ids = tree.indices;
        ids.sort_unstable();
        assert_eq!(ids, (0..n as i32).collect::<Vec<_>>());
    }
}
