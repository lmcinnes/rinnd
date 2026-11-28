//! Main NNDescent index structure and builder.

use crate::distance::affine::{AffineCodes, AffineQuery, HalfQuery, HalfRows};
use crate::distance::{Cosine, Distance, Euclidean, InnerProduct, Metric, SquaredEuclidean};
use crate::graph::{NeighborGraph, SearchGraph};
use crate::nndescent::{nn_descent_graph, NNDescentParams, NNDescentStats, ProgressiveBuild};
use crate::tree::build_hub_tree;
use crate::query_sample::{query_induced_edges, query_search_tree, QuerySample, QuerySampleOptions};
use crate::rng::FastRng;
use crate::search::{
    greedy_search_affine_into,
    greedy_search_indices_with_workspace_initialized,
    greedy_search_quantized_custom_with_workspace_initialized,
    greedy_search_quantized_i8_with_workspace_initialized,
    greedy_search_with_workspace_initialized, prefetch_row, SearchWorkspace,
};
use crate::tree::{FlatTree, QuantizedFlatTree};
use rayon::prelude::*;
use std::borrow::Cow;
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct BuildStats {
    pub nn_descent: NNDescentStats,
    pub sort_seconds: f64,
    pub distance_correction_seconds: f64,
    pub search_graph_seconds: f64,
    pub layout_seconds: f64,
    /// Building a hub search tree (zero for construction trees).
    pub search_tree_seconds: f64,
    pub raw_graph_seconds: f64,
    pub total_seconds: f64,
}

/// Result of building only the k-NN graph, without query search structures.
#[derive(Clone, Debug)]
pub struct GraphBuildResult {
    pub graph: NeighborGraph,
    pub stats: BuildStats,
}

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
    /// The tree searches start from: the hub tree built by [`Self::prepare`]
    /// (see [`crate::tree::build_hub_tree`]), or a query sample's tree.
    pub search_tree: Option<FlatTree>,
    /// Maximum leaves of the search tree visited during query initialization.
    pub search_tree_leaf_budget: usize,
    /// Vertex ordering (for tree leaf order)
    pub vertex_order: Vec<usize>,
    /// Minimum distance in graph (for epsilon scaling)
    pub min_distance: f32,
    /// RNG seed for search
    rng_seed: u64,
    /// Construction timings for diagnostics.
    pub build_stats: BuildStats,
    /// Proxy distances retained until the search graph is prepared.
    construction_distances: Option<Vec<f32>>,
    diversify_prob: f32,
    diversify_relax: f32,
    reverse_diversify: bool,
    pruning_degree_multiplier: f32,
    vertex_order_method: VertexOrder,
    search_tree_leaf_size: usize,
    /// Whether random projection trees split by angle (angular metrics).
    angular_trees: bool,
    prepared: bool,
}

/// Leaf size of the hub search tree unless set: the number of points a search
/// starts from (32-64 searched fastest on VIBE's datasets).
pub const DEFAULT_HUB_LEAF_SIZE: usize = 48;

/// How [`NNDescentIndex::prepare`] numbers the points (and lays out their rows).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum VertexOrder {
    /// The leaf order of the first search tree.
    #[default]
    Tree,
    /// Reverse Cuthill-McKee on the search graph, so that the rows a search
    /// reads when it expands a vertex sit close together in memory.
    Rcm,
}

fn resolve_quantized_candidate_width(
    k: usize,
    candidate_width: usize,
    rerank_width: usize,
) -> usize {
    let resolved = if candidate_width == 0 {
        k.max(rerank_width)
    } else {
        candidate_width
    };
    assert!(
        resolved >= k,
        "quantized candidate width must be at least k"
    );
    assert!(
        rerank_width <= resolved,
        "quantized rerank width must not exceed candidate width"
    );
    assert!(
        candidate_width == 0 || rerank_width == 0 || rerank_width >= k,
        "quantized rerank width must be zero or at least k"
    );
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
    /// Whether the query search graph and physical layout have been prepared.
    pub fn is_prepared(&self) -> bool {
        self.prepared
    }

    /// Build query-only structures. Calling this more than once is a no-op.
    pub fn prepare(&mut self) {
        self.prepare_inner(None);
    }

    /// Build query-only structures adapted to a sample of the queries (see
    /// [`crate::query_sample`]): query-induced edges join the search graph and
    /// searches start from a tree over the sample queries. `sample` holds
    /// queries in this index's (original) data space and the original indices
    /// of their nearest data points. A no-op once prepared.
    pub fn prepare_with_query_sample(&mut self, sample: &QuerySample, options: &QuerySampleOptions) {
        assert_eq!(sample.queries.len(), sample.n_queries * self.dim, "query sample has the wrong shape");
        assert_eq!(sample.neighbors.len(), sample.n_queries * sample.width, "query neighbors have the wrong shape");
        self.prepare_inner(Some((sample, options)));
    }

    fn prepare_inner(&mut self, sample: Option<(&QuerySample, &QuerySampleOptions)>) {
        if self.prepared {
            return;
        }

        let search_graph_start = Instant::now();
        let extra_edges = sample.filter(|(_, options)| options.edges > 0).map(|(sample, options)| {
            query_induced_edges(&self.data, self.dim, self.n_points, &self.distance, sample, options)
        });
        let construction_distances = self
            .construction_distances
            .as_deref()
            .unwrap_or(&self.neighbor_distances);
        let (mut search_graph, min_distance) = SearchGraph::from_dense_diversified(
            &self.neighbor_indices,
            construction_distances,
            &self.data,
            self.n_points,
            self.n_neighbors,
            self.dim,
            &self.distance,
            self.diversify_prob,
            self.diversify_relax,
            self.reverse_diversify,
            self.pruning_degree_multiplier,
            extra_edges.as_deref(),
        );
        self.build_stats.search_graph_seconds = search_graph_start.elapsed().as_secs_f64();

        // A query sample's tree replaces the hub tree for searches, which then
        // only sets the 'tree' layout.
        let sample_tree = sample.filter(|(_, options)| options.tree_leaf_size > 0);
        if sample_tree.is_none() || self.vertex_order_method == VertexOrder::Tree {
            let started = Instant::now();
            self.search_tree = Some(build_hub_tree(
                &self.data,
                self.n_points,
                self.dim,
                &self.neighbor_indices,
                self.search_tree_leaf_size,
                200,
                self.angular_trees,
            ));
            self.build_stats.search_tree_seconds = started.elapsed().as_secs_f64();
            self.build_stats.total_seconds += self.build_stats.search_tree_seconds;
        }

        let layout_start = Instant::now();
        let vertex_order: Vec<usize> = match (self.vertex_order_method, self.search_tree.as_ref()) {
            (VertexOrder::Rcm, _) => search_graph.reverse_cuthill_mckee(),
            (VertexOrder::Tree, Some(tree)) => tree.indices.iter().map(|&idx| idx as usize).collect(),
            (VertexOrder::Tree, None) => (0..self.n_points).collect(),
        };
        assert_eq!(vertex_order.len(), self.n_points);

        let mut old_to_new = vec![usize::MAX; self.n_points];
        for (new_idx, &old_idx) in vertex_order.iter().enumerate() {
            assert!(old_idx < self.n_points);
            assert_eq!(old_to_new[old_idx], usize::MAX);
            old_to_new[old_idx] = new_idx;
        }

        let mut reordered_data = Vec::with_capacity(self.data.len());
        for &old_idx in &vertex_order {
            let start = old_idx * self.dim;
            reordered_data.extend_from_slice(&self.data[start..start + self.dim]);
        }

        search_graph.reorder(&vertex_order);
        search_graph.build_padded();
        if let Some(tree) = &mut self.search_tree {
            tree.remap_indices(&old_to_new);
        }
        if let Some((sample, options)) = sample_tree {
            let mut rng = FastRng::new(self.rng_seed ^ 0x9e37_79b9_7f4a_7c15);
            let mut tree =
                query_search_tree(sample, self.dim, self.n_points, options, self.angular_trees, &mut rng);
            tree.remap_indices(&old_to_new);
            self.search_tree = Some(tree);
        }

        self.data = reordered_data;
        self.search_graph = search_graph;
        self.vertex_order = vertex_order;
        self.min_distance = min_distance;
        self.construction_distances = None;
        self.prepared = true;
        self.build_stats.layout_seconds = layout_start.elapsed().as_secs_f64();
        self.build_stats.total_seconds +=
            self.build_stats.search_graph_seconds + self.build_stats.layout_seconds;
    }

    /// Bytes in retained FP32 arrays used by production APIs, FP32 graph/tree
    /// initialization, or optional exact reranking.
    pub fn retained_fp32_bytes(&self) -> usize {
        let tree_floats: usize = self
            .search_tree
            .as_ref()
            .map_or(0, |tree| tree.hyperplanes.len() + tree.offsets.len());
        (self.data.len() + self.neighbor_distances.len() + tree_floats) * std::mem::size_of::<f32>()
    }

    fn rerank_quantized_candidates(
        &self,
        query: &[f32],
        approximate_indices: &[i32],
        k: usize,
        rerank_width: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let candidates: Vec<i32> = approximate_indices
            .iter()
            .copied()
            .filter(|&idx| idx >= 0)
            .take(rerank_width)
            .collect();
        // Request every finalist's row up front so their memory accesses
        // overlap, then score four at a time.
        for &idx in &candidates {
            prefetch_row(&self.data, idx, self.dim);
        }
        let row = |idx: i32| &self.data[idx as usize * self.dim..(idx as usize + 1) * self.dim];
        let mut reranked: Vec<(f32, i32)> = Vec::with_capacity(candidates.len());
        let mut chunks = candidates.chunks_exact(4);
        for chunk in &mut chunks {
            let distances = self
                .distance
                .distance_four(query, [row(chunk[0]), row(chunk[1]), row(chunk[2]), row(chunk[3])]);
            reranked.extend(distances.into_iter().zip(chunk.iter().copied()));
        }
        for &idx in chunks.remainder() {
            reranked.push((self.distance.distance(query, row(idx)), idx));
        }
        reranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        reranked.truncate(k);
        (
            reranked.iter().map(|&(_, idx)| idx).collect(),
            reranked.iter().map(|&(distance, _)| distance).collect(),
        )
    }

    /// Query over affine codes (see [`AffineCodes`]), reranking the best
    /// `rerank_width` candidates with the exact distance.
    ///
    /// `query` must be `tree_query` prepared for `codes`. Widths follow
    /// [`Self::query_one_quantized_i8_with_workspace`]. Returned distances are
    /// exact (corrected, e.g. square-rooted for euclidean) when reranked.
    pub fn query_one_affine_with_workspace(
        &self,
        tree_query: &[f32],
        query: &AffineQuery,
        codes: &AffineCodes,
        k: usize,
        candidate_width: usize,
        rerank_width: usize,
        epsilon: f32,
        workspace: &mut SearchWorkspace,
    ) -> (Vec<i32>, Vec<f32>) {
        self.query_one_codes(
            tree_query, query, codes, None, k, candidate_width, rerank_width, epsilon, workspace,
        )
    }

    /// As [`Self::query_one_affine_with_workspace`], reranking from `half`
    /// (float16 rows of this index's data, with `tree_query` prepared for them)
    /// when given. Only the returned vectors are allocated.
    pub fn query_one_codes(
        &self,
        tree_query: &[f32],
        query: &AffineQuery,
        codes: &AffineCodes,
        half: Option<(&HalfRows, &HalfQuery)>,
        k: usize,
        candidate_width: usize,
        rerank_width: usize,
        epsilon: f32,
        workspace: &mut SearchWorkspace,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut rng = FastRng::new(self.rng_seed);
        let search_k = resolve_quantized_candidate_width(k, candidate_width, rerank_width);
        greedy_search_affine_into(
            tree_query,
            query,
            codes,
            &self.search_graph,
            self.search_tree.as_slice(),
            self.search_tree_leaf_budget,
            search_k,
            epsilon,
            self.min_distance,
            &mut rng,
            workspace,
        );
        workspace.sort_results();
        let order = |a: &(f32, i32), b: &(f32, i32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        let mut finalists = std::mem::take(&mut workspace.reranked);
        finalists.clear();
        let reranked = should_rerank_quantized(k, candidate_width, rerank_width);
        if reranked {
            let count = rerank_width.min(workspace.results.len());
            let candidates = &workspace.results[..count];
            // Request every finalist's row up front so their memory accesses
            // overlap, then score four at a time.
            match half {
                Some((rows, prepared)) => {
                    for &(_, idx) in candidates {
                        rows.prefetch(idx);
                    }
                    let mut chunks = candidates.chunks_exact(4);
                    for chunk in &mut chunks {
                        let ids = [chunk[0].1, chunk[1].1, chunk[2].1, chunk[3].1];
                        let distances = rows.distance_four(prepared, ids);
                        finalists.extend(distances.into_iter().zip(ids));
                    }
                    for &(_, idx) in chunks.remainder() {
                        finalists.push((rows.distance(prepared, idx), idx));
                    }
                }
                None => {
                    for &(_, idx) in candidates {
                        prefetch_row(&self.data, idx, self.dim);
                    }
                    let row =
                        |idx: i32| &self.data[idx as usize * self.dim..(idx as usize + 1) * self.dim];
                    let mut chunks = candidates.chunks_exact(4);
                    for chunk in &mut chunks {
                        let ids = [chunk[0].1, chunk[1].1, chunk[2].1, chunk[3].1];
                        let distances = self.distance.distance_four(tree_query, ids.map(row));
                        finalists.extend(distances.into_iter().zip(ids));
                    }
                    for &(_, idx) in chunks.remainder() {
                        finalists.push((self.distance.distance(tree_query, row(idx)), idx));
                    }
                }
            }
            if finalists.len() > k && k > 0 {
                finalists.select_nth_unstable_by(k - 1, order);
                finalists.truncate(k);
            }
            finalists.sort_unstable_by(order);
        } else {
            finalists.extend_from_slice(&workspace.results[..k.min(workspace.results.len())]);
        }
        let correction = if reranked { self.distance_correction } else { None };
        let mut indices = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        for &(distance, idx) in finalists.iter().take(k) {
            indices.push(self.vertex_order[idx as usize] as i32);
            distances.push(correction.map_or(distance, |correct| correct(distance)));
        }
        indices.resize(k, -1);
        distances.resize(k, f32::INFINITY);
        workspace.reranked = finalists;
        (indices, distances)
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
        self.search_tree.as_ref().map(|tree| {
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
            self.search_tree.as_slice(),
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
            self.search_tree.as_slice(),
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
    fn process_query(
        &self,
        query: &[f32],
        query_id: usize,
        k: usize,
        epsilon: f32,
    ) -> (Vec<i32>, Vec<f32>) {
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
            self.search_tree.as_slice(),
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
            (indices, distances) =
                self.rerank_quantized_candidates(tree_query, &indices, k, rerank_width);
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
        let (mut indices, mut distances) =
            greedy_search_quantized_custom_with_workspace_initialized(
                tree_query,
                &self.search_graph,
                self.search_tree.as_slice(),
                self.search_tree_leaf_budget,
                search_k,
                epsilon,
                self.min_distance,
                &mut rng,
                workspace,
                quantized_trees,
                quantized_tree_query,
                tree_query_scale,
                distance_to,
            );

        if should_rerank_quantized(k, candidate_width, rerank_width) {
            (indices, distances) =
                self.rerank_quantized_candidates(tree_query, &indices, k, rerank_width);
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
    data: Cow<'a, [f32]>,
    n_points: usize,
    dim: usize,
    metric: Metric,
    n_neighbors: usize,
    n_trees: usize,
    search_tree_leaf_budget: usize,
    leaf_size: Option<usize>,
    max_candidates: Option<usize>,
    n_iters: Option<usize>,
    delta: f32,
    random_seed: u64,
    verbose: bool,
    diversify_prob: f32,
    diversify_relax: f32,
    reverse_diversify: bool,
    pruning_degree_multiplier: f32,
    vertex_order: VertexOrder,
    search_tree_leaf_size: Option<usize>,
    progressive: ProgressiveBuild,
    code_trees: bool,
    breadth_first_batch_width: Option<usize>,
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
            data: Cow::Borrowed(data),
            n_points,
            dim,
            metric: Metric::Euclidean,
            n_neighbors: 30,
            n_trees: 8,
            search_tree_leaf_budget: 1,
            leaf_size: None,
            max_candidates: None,
            n_iters: None,
            delta: 0.001,
            random_seed: 42,
            verbose: false,
            diversify_prob: 1.0,
            diversify_relax: 1.0,
            reverse_diversify: false,
            pruning_degree_multiplier: 1.5,
            vertex_order: VertexOrder::Tree,
            search_tree_leaf_size: None,
            progressive: ProgressiveBuild::Auto,
            code_trees: true,
            breadth_first_batch_width: Some(6),
        }
    }

    /// Create a builder that transfers owned data into the finished index.
    pub fn from_vec(data: Vec<f32>, n_points: usize, dim: usize) -> NNDescentBuilder<'static> {
        NNDescentBuilder {
            data: Cow::Owned(data),
            n_points,
            dim,
            metric: Metric::Euclidean,
            n_neighbors: 30,
            n_trees: 8,
            search_tree_leaf_budget: 1,
            leaf_size: None,
            max_candidates: None,
            n_iters: None,
            delta: 0.001,
            random_seed: 42,
            verbose: false,
            diversify_prob: 1.0,
            diversify_relax: 1.0,
            reverse_diversify: false,
            pruning_degree_multiplier: 1.5,
            vertex_order: VertexOrder::Tree,
            search_tree_leaf_size: None,
            progressive: ProgressiveBuild::Auto,
            code_trees: true,
            breadth_first_batch_width: Some(6),
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

    /// Build the forest breadth first (hybrid builder), `width` trees per margin pass.
    pub fn breadth_first_tree_batch_width(mut self, width: usize) -> Self {
        assert!(
            (1..=8).contains(&width),
            "tree batch width must be in 1..=8"
        );
        self.breadth_first_batch_width = Some(width);
        self
    }

    /// Use the legacy depth-first forest construction.
    pub fn depth_first_tree_build(mut self) -> Self {
        self.breadth_first_batch_width = None;
        self
    }

    /// Set the number of search tree leaves visited to start a search.
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

    /// Relax diversification: an edge `i -> j` is dropped only when a retained
    /// neighbour `c` of `i` has `relax * d(c, j) < d(i, j)` (default 1). Values
    /// above 1 keep more long edges (alpha-pruning; for squared distances
    /// `relax = alpha^2`).
    pub fn diversify_relax(mut self, relax: f32) -> Self {
        self.diversify_relax = relax;
        self
    }

    /// Whether the search graph's reverse edges are diversified too (default
    /// false: as in PyNNDescent, they are all kept and the degree cap prunes
    /// them, which gives a denser graph that searches better).
    pub fn reverse_diversify(mut self, reverse: bool) -> Self {
        self.reverse_diversify = reverse;
        self
    }

    /// Set whether the leaf join and first iterations run on 8-bit codes
    /// (default: from 512 features up, for euclidean and dot-ordered
    /// distances; see [`crate::nndescent::progressive`]).
    pub fn progressive_build(mut self, mode: ProgressiveBuild) -> Self {
        self.progressive = mode;
        self
    }

    /// Leaf size of the hub search tree (default [`DEFAULT_HUB_LEAF_SIZE`]).
    pub fn search_tree_leaf_size(mut self, size: usize) -> Self {
        self.search_tree_leaf_size = Some(size.max(1));
        self
    }

    /// Whether the breadth-first forest splits 8-bit codes rather than the
    /// data, in progressive builds and from
    /// [`crate::nndescent::CODE_TREE_MIN_DIM`] features (default true).
    #[doc(hidden)]
    pub fn code_trees(mut self, enabled: bool) -> Self {
        self.code_trees = enabled;
        self
    }

    /// Set how prepared indexes number their points (default: tree order).
    pub fn vertex_order(mut self, order: VertexOrder) -> Self {
        self.vertex_order = order;
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

    /// Build only the sorted k-NN graph and discard data and RP trees.
    pub fn build_graph_with_distance<D: Distance<f32>>(
        self,
        distance: D,
        correction: Option<fn(f32) -> f32>,
    ) -> GraphBuildResult {
        let build_start = Instant::now();
        let angular = matches!(
            self.metric,
            Metric::Cosine
                | Metric::InnerProduct
                | Metric::Dot
                | Metric::Correlation
                | Metric::TrueAngular
                | Metric::TSSS
        );
        let leaf_size = self
            .leaf_size
            .unwrap_or_else(|| (5 * self.n_neighbors).min(256).max(60));
        let max_candidates = self
            .max_candidates
            .unwrap_or_else(|| self.n_neighbors.min(60));
        let n_iters = self
            .n_iters
            .unwrap_or_else(|| ((self.n_points as f64).log2().ceil() as usize).max(5));
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
            progressive: self.progressive,
            code_trees: self.code_trees,
        };
        let mut rng = FastRng::new(self.random_seed);
        let (mut neighbor_heap, nn_descent_stats) = nn_descent_graph(
            &self.data,
            self.n_points,
            self.dim,
            &distance,
            &params,
            &mut rng,
            self.breadth_first_batch_width,
        );

        let sort_start = Instant::now();
        neighbor_heap.sort_all();
        let sort_seconds = sort_start.elapsed().as_secs_f64();

        let distance_correction_start = Instant::now();
        let distances = if let Some(correction) = correction {
            neighbor_heap
                .distances
                .iter()
                .map(|&distance| correction(distance))
                .collect()
        } else {
            neighbor_heap.distances
        };
        let distance_correction_seconds = distance_correction_start.elapsed().as_secs_f64();
        let raw_graph_seconds =
            nn_descent_stats.total_seconds() + sort_seconds + distance_correction_seconds;

        GraphBuildResult {
            graph: NeighborGraph {
                indices: neighbor_heap.indices,
                distances,
                n_points: self.n_points,
                k: self.n_neighbors,
            },
            stats: BuildStats {
                nn_descent: nn_descent_stats,
                sort_seconds,
                distance_correction_seconds,
                search_graph_seconds: 0.0,
                layout_seconds: 0.0,
                search_tree_seconds: 0.0,
                raw_graph_seconds,
                total_seconds: build_start.elapsed().as_secs_f64(),
            },
        }
    }

    /// Build the index with a custom distance function.
    pub fn build_with_distance<D: Distance<f32>>(
        self,
        distance: D,
        correction: Option<fn(f32) -> f32>,
    ) -> NNDescentIndex<D> {
        let mut index = self.build_unprepared_with_distance(distance, correction);
        index.prepare();
        index
    }

    /// Build only the sorted k-NN graph, deferring query search structures.
    pub fn build_unprepared_with_distance<D: Distance<f32>>(
        self,
        distance: D,
        correction: Option<fn(f32) -> f32>,
    ) -> NNDescentIndex<D> {
        let build_start = Instant::now();
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
        let leaf_size = self
            .leaf_size
            .unwrap_or_else(|| (5 * self.n_neighbors).min(256).max(60));
        let max_candidates = self
            .max_candidates
            .unwrap_or_else(|| self.n_neighbors.min(60));
        let n_iters = self
            .n_iters
            .unwrap_or_else(|| ((self.n_points as f64).log2().ceil() as usize).max(5));

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
            progressive: self.progressive,
            code_trees: self.code_trees,
        };

        let mut rng = FastRng::new(self.random_seed);

        // Run NN-descent
        let (mut neighbor_heap, nn_descent_stats) = nn_descent_graph(
            &self.data,
            self.n_points,
            self.dim,
            &distance,
            &params,
            &mut rng,
            self.breadth_first_batch_width,
        );

        // Sort the heap so neighbors are in ascending distance order
        // (matches PyNNDescent's deheap_sort behavior)
        let sort_start = Instant::now();
        neighbor_heap.sort_all();
        let sort_seconds = sort_start.elapsed().as_secs_f64();

        // Apply distance correction to stored distances if needed
        let distance_correction_start = Instant::now();
        let neighbor_distances = if let Some(corr) = correction {
            neighbor_heap.distances.iter().map(|&d| corr(d)).collect()
        } else {
            neighbor_heap.distances.clone()
        };
        let distance_correction_seconds = distance_correction_start.elapsed().as_secs_f64();

        let raw_graph_seconds =
            nn_descent_stats.total_seconds() + sort_seconds + distance_correction_seconds;
        let build_stats = BuildStats {
            nn_descent: nn_descent_stats,
            sort_seconds,
            distance_correction_seconds,
            search_graph_seconds: 0.0,
            layout_seconds: 0.0,
            search_tree_seconds: 0.0,
            raw_graph_seconds,
            total_seconds: build_start.elapsed().as_secs_f64(),
        };

        let construction_distances = correction.map(|_| neighbor_heap.distances.clone());

        NNDescentIndex {
            data: self.data.into_owned(),
            n_points: self.n_points,
            dim: self.dim,
            distance,
            distance_correction: correction,
            n_neighbors: self.n_neighbors,
            neighbor_indices: neighbor_heap.indices,
            neighbor_distances,
            search_graph: SearchGraph::new(self.n_points),
            search_tree: None,
            search_tree_leaf_budget: self.search_tree_leaf_budget,
            vertex_order: (0..self.n_points).collect(),
            min_distance: f32::INFINITY,
            rng_seed: self.random_seed,
            build_stats,
            construction_distances,
            diversify_prob: self.diversify_prob,
            diversify_relax: self.diversify_relax,
            reverse_diversify: self.reverse_diversify,
            pruning_degree_multiplier: self.pruning_degree_multiplier,
            vertex_order_method: self.vertex_order,
            search_tree_leaf_size: self.search_tree_leaf_size.unwrap_or(DEFAULT_HUB_LEAF_SIZE),
            angular_trees: angular,
            prepared: false,
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
    fn query_sample_adapts_the_search_index() {
        let (n, dim, n_queries, width) = (2000, 16, 400, 30);
        let mut rng = FastRng::new(17);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.next_float() * 2.0 - 1.0).collect();
        // Queries from a shifted distribution, half for the sample, half for testing.
        let queries: Vec<f32> = (0..2 * n_queries * dim).map(|t| rng.next_float() + (t % dim) as f32 * 0.05).collect();
        let brute = |query: &[f32], k: usize| -> Vec<i32> {
            let mut order: Vec<(f32, i32)> = (0..n)
                .map(|i| (SquaredEuclidean.distance(query, &data[i * dim..(i + 1) * dim]), i as i32))
                .collect();
            order.sort_by(|a, b| a.0.total_cmp(&b.0));
            order[..k].iter().map(|&(_, i)| i).collect()
        };
        let (sample_queries, test_queries) = queries.split_at(n_queries * dim);
        let neighbors: Vec<i32> = sample_queries.chunks_exact(dim).flat_map(|q| brute(q, width)).collect();
        let sample = QuerySample { queries: sample_queries, neighbors: &neighbors, n_queries, width };

        let build = || {
            NNDescentBuilder::new(&data, n, dim)
                .n_neighbors(10)
                .random_seed(3)
                .build_unprepared_with_distance(SquaredEuclidean, None)
        };
        let mut plain = build();
        plain.prepare();
        let mut adapted = build();
        adapted.prepare_with_query_sample(&sample, &QuerySampleOptions::default());
        assert!(adapted.search_tree.as_ref().expect("query tree").indices.len() <= n);
        assert!(adapted.search_graph.indices.len() > plain.search_graph.indices.len());

        let recall = |index: &NNDescentIndex<SquaredEuclidean>| {
            let mut hits = 0;
            for query in test_queries.chunks_exact(dim) {
                let (found, _) = index.query_one(query, 10, 0.0);
                let truth = brute(query, 10);
                hits += found.iter().filter(|i| truth.contains(i)).count();
            }
            hits as f64 / (10 * n_queries) as f64
        };
        let (plain_recall, adapted_recall) = (recall(&plain), recall(&adapted));
        assert!(adapted_recall > 0.9, "adapted recall {adapted_recall}");
        assert!(adapted_recall >= plain_recall - 0.01, "{adapted_recall} vs {plain_recall}");
    }

    #[test]
    fn progressive_build_matches_exact_build_quality() {
        let (n, dim, k) = (1500, 64, 10);
        let mut rng = FastRng::new(31);
        let centres: Vec<f32> = (0..15 * dim).map(|_| rng.next_float() * 4.0 - 2.0).collect();
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let centre = rng.next_index(15);
            data.extend((0..dim).map(|t| centres[centre * dim + t] + rng.next_float() - 0.5));
        }
        let build = |mode: ProgressiveBuild| {
            NNDescentBuilder::new(&data, n, dim)
                .n_neighbors(k)
                .random_seed(1)
                .progressive_build(mode)
                .build_graph_with_distance(SquaredEuclidean, None)
        };
        let row = |i: usize| &data[i * dim..(i + 1) * dim];
        let recall = |graph: &NeighborGraph| {
            let mut hits = 0;
            for point in (0..n).step_by(15) {
                let mut exact: Vec<(f32, usize)> =
                    (0..n).filter(|&q| q != point).map(|q| (SquaredEuclidean.distance(row(point), row(q)), q)).collect();
                exact.sort_by(|a, b| a.0.total_cmp(&b.0));
                let truth: Vec<i32> = exact[..k].iter().map(|&(_, q)| q as i32).collect();
                hits += graph.indices[point * k..(point + 1) * k].iter().filter(|q| truth.contains(q)).count();
            }
            hits as f64 / (n.div_ceil(15) * k) as f64
        };
        let progressive = build(ProgressiveBuild::Always);
        let exact = build(ProgressiveBuild::Never);
        assert!(progressive.stats.nn_descent.approximate_iterations > 0);
        assert_eq!(exact.stats.nn_descent.approximate_iterations, 0);
        let (progressive_recall, exact_recall) = (recall(&progressive.graph), recall(&exact.graph));
        assert!(progressive_recall > 0.97, "progressive recall {progressive_recall}");
        assert!(progressive_recall >= exact_recall - 0.01, "{progressive_recall} vs {exact_recall}");
        // The graph holds exact distances.
        for point in 0..n {
            for slot in 0..k {
                let q = progressive.graph.indices[point * k + slot];
                if q >= 0 {
                    let d = SquaredEuclidean.distance(row(point), row(q as usize));
                    assert!((progressive.graph.distances[point * k + slot] - d).abs() <= 1e-4 * (1.0 + d));
                }
            }
        }
        // Auto leaves narrow data exact.
        assert_eq!(build(ProgressiveBuild::Auto).stats.nn_descent.approximate_iterations, 0);
    }

    #[test]
    fn code_tree_forest_matches_float_forest_quality() {
        // From CODE_TREE_MIN_DIM features the forest splits 8-bit codes, also
        // without a progressive build.
        let (n, dim, k) = (1500, crate::nndescent::CODE_TREE_MIN_DIM, 10);
        let mut rng = FastRng::new(37);
        let centres: Vec<f32> = (0..15 * dim).map(|_| rng.next_float() * 4.0 - 2.0).collect();
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let centre = rng.next_index(15);
            data.extend((0..dim).map(|t| centres[centre * dim + t] + rng.next_float() - 0.5));
        }
        let build = |code_trees: bool| {
            NNDescentBuilder::new(&data, n, dim)
                .n_neighbors(k)
                .random_seed(3)
                .code_trees(code_trees)
                .build_graph_with_distance(SquaredEuclidean, None)
        };
        let row = |i: usize| &data[i * dim..(i + 1) * dim];
        let recall = |graph: &NeighborGraph| {
            let mut hits = 0;
            for point in (0..n).step_by(15) {
                let mut exact: Vec<(f32, usize)> =
                    (0..n).filter(|&q| q != point).map(|q| (SquaredEuclidean.distance(row(point), row(q)), q)).collect();
                exact.sort_by(|a, b| a.0.total_cmp(&b.0));
                let truth: Vec<i32> = exact[..k].iter().map(|&(_, q)| q as i32).collect();
                hits += graph.indices[point * k..(point + 1) * k].iter().filter(|q| truth.contains(q)).count();
            }
            hits as f64 / (n.div_ceil(15) * k) as f64
        };
        let (code, float) = (build(true), build(false));
        assert_eq!(code.stats.nn_descent.approximate_iterations, 0);
        let (code_recall, float_recall) = (recall(&code.graph), recall(&float.graph));
        assert!(code_recall > 0.97 && code_recall >= float_recall - 0.01, "{code_recall} vs {float_recall}");
    }

    #[test]
    fn diversify_options_and_rcm_order() {
        let (n, dim) = (2000, 3);
        let mut rng = FastRng::new(23);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.next_float() * 2.0 - 1.0).collect();
        let build = |relax: f32, order: VertexOrder| {
            let mut index = NNDescentBuilder::new(&data, n, dim)
                .n_neighbors(10)
                .random_seed(3)
                .diversify_relax(relax)
                .vertex_order(order)
                .build_unprepared_with_distance(SquaredEuclidean, None);
            index.prepare();
            index
        };
        let tree = build(1.0, VertexOrder::Tree);
        let relaxed = build(1.44, VertexOrder::Tree);
        assert!(relaxed.search_graph.indices.len() > tree.search_graph.indices.len());
        // Diversifying the reverse edges too thins the graph.
        let mut thinned = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .random_seed(3)
            .reverse_diversify(true)
            .build_unprepared_with_distance(SquaredEuclidean, None);
        thinned.prepare();
        assert!(thinned.search_graph.indices.len() < tree.search_graph.indices.len());

        let rcm = build(1.0, VertexOrder::Rcm);
        let mut order = rcm.vertex_order.clone();
        order.sort_unstable();
        assert_eq!(order, (0..n).collect::<Vec<_>>());
        assert_ne!(rcm.vertex_order, tree.vertex_order);
        assert_eq!(rcm.search_graph.indices.len(), tree.search_graph.indices.len());
        // Graph neighbours get closer numbers than in tree order.
        let mean_gap = |index: &NNDescentIndex<SquaredEuclidean>| {
            let graph = &index.search_graph;
            let total: usize = (0..n)
                .flat_map(|v| graph.neighbors(v).iter().map(move |&u| v.abs_diff(u as usize)))
                .sum();
            total as f64 / graph.indices.len() as f64
        };
        // In a random order the mean gap is about n / 3.
        assert!(mean_gap(&rcm) < 0.2 * n as f64 / 3.0, "{} vs {}", mean_gap(&rcm), mean_gap(&tree));
        assert!(mean_gap(&rcm) < mean_gap(&tree));
        // The same results, in original ids.
        for query in data.chunks_exact(dim).take(50) {
            assert_eq!(rcm.query_one(query, 5, 0.1).0[0], tree.query_one(query, 5, 0.1).0[0]);
        }
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

        let tree = index.search_tree.as_ref().expect("search tree");
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
    fn test_prepare_builds_the_hub_search_tree() {
        let n = 300;
        let dim = 10;
        let data = create_test_data(n, dim);

        let mut index = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(10)
            .n_trees(3)
            .n_iters(3)
            .search_tree_leaf_size(20)
            .build_unprepared_with_distance(SquaredEuclidean, None);
        assert!(index.search_tree.is_none());
        index.prepare();
        let tree = index.search_tree.as_ref().expect("hub tree");
        let mut indices = tree.indices.clone();
        indices.sort_unstable();
        assert_eq!(indices, (0..n as i32).collect::<Vec<_>>());
        assert!(tree.get_all_leaves().iter().all(|&(start, end)| end - start <= 20));
        assert!(index.build_stats.search_tree_seconds > 0.0);

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

    #[test]
    fn test_build_graph_with_distance_retains_only_graph() {
        let n = 50;
        let dim = 8;
        let k = 5;
        let data = create_test_data(n, dim);

        let result = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(k)
            .n_trees(2)
            .n_iters(3)
            .build_graph_with_distance(SquaredEuclidean, Some(|distance| distance.sqrt()));

        assert_eq!(result.graph.n_points, n);
        assert_eq!(result.graph.k, k);
        assert_eq!(result.graph.indices.len(), n * k);
        assert_eq!(result.graph.distances.len(), n * k);
        assert!(result
            .graph
            .indices
            .iter()
            .all(|&index| index >= 0 && index < n as i32));
        assert!(result
            .graph
            .distances
            .iter()
            .all(|distance| distance.is_finite()));
        assert!(result.stats.raw_graph_seconds > 0.0);
        assert_eq!(result.stats.search_graph_seconds, 0.0);
        assert_eq!(result.stats.layout_seconds, 0.0);
    }

    #[test]
    fn custom_distances_preserve_graph_outputs() {
        #[derive(Clone)]
        struct ExactSquared;
        impl Distance<f32> for ExactSquared {
            const USE_DISTANCE_FOUR: bool = SquaredEuclidean::USE_DISTANCE_FOUR;
            fn distance(&self, first: &[f32], second: &[f32]) -> f32 {
                SquaredEuclidean.distance(first, second)
            }
            fn distance_four(&self, query: &[f32], candidates: [&[f32]; 4]) -> [f32; 4] {
                SquaredEuclidean.distance_four(query, candidates)
            }
            fn name(&self) -> &'static str { "custom_exact_squared" }
        }
        let points = 50;
        let dimension = 8;
        let data: Vec<f32> = (0..points * dimension).map(|position| (position % 97) as f32 * 100000.0).collect();
        let build = || NNDescentBuilder::new(&data, points, dimension).n_neighbors(10).n_trees(2).n_iters(3);
        let custom = build().build_graph_with_distance(ExactSquared, None);
        let built_in = build().build_graph_with_distance(SquaredEuclidean, None);
        assert_eq!(custom.graph.indices, built_in.graph.indices);
        assert_eq!(custom.graph.distances, built_in.graph.distances);
    }

    #[test]
    fn graph_builders_export_original_distances_in_sorted_order() {
        fn check<D: Distance<f32>>(metric: D, correction: fn(f32) -> f32) {
            let points = 80;
            let dimension = 35;
            let degree = 15;
            let data = create_test_data(points, dimension);
            let build = || NNDescentBuilder::new(&data, points, dimension)
                .n_neighbors(degree).n_trees(2).n_iters(4);
            let graph = build().build_graph_with_distance(metric.clone(), Some(correction));
            let index = build().build_unprepared_with_distance(metric.clone(), Some(correction));
            for (indices, distances) in [
                (&graph.graph.indices, &graph.graph.distances),
                (&index.neighbor_indices, &index.neighbor_distances),
            ] {
                for point in 0..points {
                    let row = &distances[point * degree..(point + 1) * degree];
                    assert!(row.windows(2).all(|pair| pair[0] <= pair[1]));
                    for slot in 0..degree {
                        let neighbor = indices[point * degree + slot] as usize;
                        let exact = correction(metric.distance(
                            &data[point * dimension..(point + 1) * dimension],
                            &data[neighbor * dimension..(neighbor + 1) * dimension],
                        ));
                        assert_eq!(row[slot].to_bits(), exact.to_bits());
                    }
                }
            }
        }
        check(SquaredEuclidean, f32::sqrt);
        check(crate::distance::AlternativeDot, crate::distance::correct_alternative_cosine);
        check(crate::distance::DirectNormalizedCosine, |value| value);
    }

    #[test]
    fn test_build_graph_with_breadth_first_forest() {
        let n = 100;
        let dim = 12;
        let k = 8;
        let data = create_test_data(n, dim);

        let result = NNDescentBuilder::new(&data, n, dim)
            .n_neighbors(k)
            .n_trees(4)
            .n_iters(3)
            .breadth_first_tree_batch_width(4)
            .build_graph_with_distance(SquaredEuclidean, Some(|distance| distance.sqrt()));

        assert_eq!(result.graph.indices.len(), n * k);
        assert_eq!(result.graph.distances.len(), n * k);
        assert!(result.graph.indices.iter().all(|&index| index >= 0));
        assert!(result
            .graph
            .distances
            .iter()
            .all(|distance| distance.is_finite()));
        assert!(result
            .graph
            .distances
            .chunks(k)
            .all(|row| row.windows(2).all(|pair| pair[0] <= pair[1])));
    }
}
