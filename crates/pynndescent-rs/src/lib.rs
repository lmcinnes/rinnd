//! Python bindings for nndescent-rs using PyO3.
//!
//! This crate provides Python-compatible classes that mirror the PyNNDescent API.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Mutex;
use std::time::Instant;

use nndescent_core::index::{NNDescentBuilder, NNDescentIndex};
use nndescent_core::distance::*;
use nndescent_core::distance::quantized::{
    quantized_i8_alternative_dot,
    quantized_i8_dot,
    quantized_u8_sq_euclidean,
};
use nndescent_core::graph::SearchGraph;
use nndescent_core::search::SearchWorkspace;
use nndescent_core::tree::{FlatTree, QuantizedFlatTree};

/// NNDescent index for approximate nearest neighbor search.
///
/// This is the main class for building and querying k-NN graphs.
///
/// Parameters
/// ----------
/// data : numpy.ndarray
///     2D array of shape (n_samples, n_features) containing the data points.
/// metric : str, default='euclidean'
///     Distance metric to use. Options: 'euclidean', 'l2', 'cosine',
///     'inner_product', 'dot'.
/// n_neighbors : int, default=30
///     Number of neighbors to compute.
/// n_trees : int, default=8
///     Number of random projection trees to build.
/// leaf_size : int, optional
///     Maximum leaf size for RP trees.
/// max_candidates : int, optional
///     Maximum number of candidates per iteration.
/// n_iters : int, optional
///     Number of NN-descent iterations.
/// delta : float, default=0.001
///     Convergence threshold (early stopping if fewer than delta*n*k updates).
/// random_state : int, optional
///     Random seed for reproducibility.
/// verbose : bool, default=False
///     Whether to print progress information.
///
/// Attributes
/// ----------
/// neighbor_graph : tuple of (indices, distances)
///     The k-NN graph as a tuple of 2D arrays.
///
/// Examples
/// --------
/// >>> from pynndescent import NNDescent
/// >>> import numpy as np
/// >>> data = np.random.randn(1000, 128).astype(np.float32)
/// >>> index = NNDescent(data, n_neighbors=15)
/// >>> indices, distances = index.query(data[:10], k=5)
#[pyclass(name = "NNDescent")]
pub struct PyNNDescent {
    n_points: usize,
    dim: usize,
    /// Whether queries should be L2-normalized before search.
    normalize: bool,
    /// Index parameters
    n_neighbors: usize,
    /// The internal index (type-erased)
    index_data: Box<dyn AnyIndex>,
    /// Reusable scratch space for single-query normalization.
    query_scratch: Mutex<Vec<f32>>,
}

/// Trait for type-erased index operations.
trait AnyIndex: Send + Sync {
    fn query(&self, queries: &[f32], n_queries: usize, k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>);
    fn query_one(&self, query: &[f32], k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>);
    fn query_one_indices(&self, query: &[f32], k: usize, epsilon: f32) -> Vec<i32>;
    fn query_one_quantized_widths(
        &self,
        _query: &[f32],
        _k: usize,
        _epsilon: f32,
        _candidate_width: usize,
        _rerank_width: usize,
    ) -> PyResult<(Vec<i32>, Vec<f32>)> {
        Err(PyValueError::new_err(
            "query-time quantized widths require a quantized cosine_distance_mode",
        ))
    }
    fn neighbor_indices(&self) -> &[i32];
    fn neighbor_distances(&self) -> &[f32];
    fn search_graph_original_order(&self) -> SearchGraph;
    fn search_graph_min_distance(&self) -> f32;
    fn export_search_tree(&self) -> Option<FlatTree>;
    fn storage_info(&self) -> (&'static str, usize, usize, usize, f64);
    fn quantized_tree_storage_info(&self) -> (usize, usize, usize, usize, usize, usize) {
        (0, 0, 0, 0, 0, 0)
    }
    fn retained_fp32_tree_topology_bytes(&self) -> usize {
        0
    }
    fn released_fp32_data_bytes(&self) -> usize {
        0
    }
}

struct AnyIndexWithWorkspace<D: Distance<f32> + Send + Sync> {
    index: NNDescentIndex<D>,
    workspace: Mutex<SearchWorkspace>,
}

struct QuantizedI8Index {
    index: NNDescentIndex<DirectNormalizedCosine>,
    quantized_data: Vec<i8>,
    inv_norms: Vec<f32>,
    workspace: Mutex<SearchWorkspace>,
    query_scratch: Mutex<Vec<i8>>,
    candidate_width: usize,
    rerank_width: usize,
    encoding: QuantizedI8Encoding,
    global_multiplier: f32,
    global_dequant_scale: f32,
    quantized_trees: Vec<QuantizedFlatTree>,
    released_fp32_tree_bytes: usize,
    released_fp32_data_bytes: usize,
    encode_seconds: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QuantizedI8Encoding {
    PerVectorCosine,
    GlobalSymmetric,
    PerVectorSymmetric,
}

impl QuantizedI8Index {
    fn new(
        mut index: NNDescentIndex<DirectNormalizedCosine>,
        candidate_width: usize,
        rerank_width: usize,
        encoding: QuantizedI8Encoding,
        quantize_trees: bool,
    ) -> Self {
        let encode_start = Instant::now();
        let dim = index.dim;
        let mut quantized_data = Vec::with_capacity(index.data.len());
        let mut inv_norms = Vec::with_capacity(index.n_points);
        let global_max_abs = index
            .data
            .iter()
            .fold(0.0f32, |value, &x| value.max(x.abs()));
        let global_multiplier = if global_max_abs > 0.0 {
            127.0 / global_max_abs
        } else {
            0.0
        };
        let global_dequant_scale = if global_multiplier > 0.0 {
            global_multiplier.recip()
        } else {
            0.0
        };

        for row in index.data.chunks_exact(dim) {
            let multiplier = match encoding {
                QuantizedI8Encoding::PerVectorCosine | QuantizedI8Encoding::PerVectorSymmetric => {
                    let max_abs = row.iter().fold(0.0f32, |value, &x| value.max(x.abs()));
                    if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 }
                }
                QuantizedI8Encoding::GlobalSymmetric => global_multiplier,
            };
            let start = quantized_data.len();
            quantized_data.extend(row.iter().map(|&x| {
                (x * multiplier).round().clamp(-127.0, 127.0) as i8
            }));
            match encoding {
                QuantizedI8Encoding::PerVectorCosine => {
                    let norm_sq: i32 = quantized_data[start..]
                        .iter().map(|&x| i32::from(x) * i32::from(x)).sum();
                    inv_norms.push(if norm_sq > 0 { (norm_sq as f32).sqrt().recip() } else { 0.0 });
                }
                QuantizedI8Encoding::PerVectorSymmetric => {
                    inv_norms.push(if multiplier > 0.0 { multiplier.recip() } else { 0.0 });
                }
                QuantizedI8Encoding::GlobalSymmetric => inv_norms.push(global_dequant_scale),
            }
        }

        let n_points = index.n_points;
        let initial_k = index.n_neighbors.max(1);
        let released_fp32_tree_bytes = if quantize_trees {
            index.search_trees.iter().map(|tree| {
                tree.hyperplanes.len() * std::mem::size_of::<f32>()
                    + tree.offsets.len() * std::mem::size_of::<f32>()
                    + tree.children.len() * std::mem::size_of::<[i32; 2]>()
                    + tree.indices.len() * std::mem::size_of::<i32>()
            }).sum()
        } else {
            0
        };
        let quantized_trees = if quantize_trees {
            std::mem::take(&mut index.search_trees)
                .into_iter()
                .map(QuantizedFlatTree::from_owned_flat_tree)
                .collect()
        } else {
            Vec::new()
        };
        let release_fp32_data = quantize_trees && rerank_width == 0;
        let released_fp32_data_bytes = if release_fp32_data {
            index.data.len() * std::mem::size_of::<f32>()
        } else {
            0
        };
        if release_fp32_data {
            drop(std::mem::take(&mut index.data));
        }
        Self {
            index,
            quantized_data,
            inv_norms,
            workspace: Mutex::new(SearchWorkspace::new(n_points, initial_k)),
            query_scratch: Mutex::new(vec![0; dim]),
            candidate_width,
            rerank_width,
            encoding,
            global_multiplier,
            global_dequant_scale,
            quantized_trees,
            released_fp32_tree_bytes,
            released_fp32_data_bytes,
            encode_seconds: encode_start.elapsed().as_secs_f64(),
        }
    }

    #[inline]
    fn exact_rerank_requested(k: usize, candidate_width: usize, rerank_width: usize) -> bool {
        if candidate_width == 0 {
            rerank_width > k
        } else {
            rerank_width > 0
        }
    }

    fn ensure_rerank_available(
        &self,
        k: usize,
        candidate_width: usize,
        rerank_width: usize,
    ) -> PyResult<()> {
        if self.released_fp32_data_bytes > 0
            && Self::exact_rerank_requested(k, candidate_width, rerank_width)
        {
            return Err(PyValueError::new_err(
                "exact reranking is unavailable because FP32 vector data was released",
            ));
        }
        Ok(())
    }

    fn tree_query_scale(&self, query: &[f32]) -> f32 {
        match self.encoding {
            QuantizedI8Encoding::PerVectorCosine | QuantizedI8Encoding::PerVectorSymmetric => {
                let max_abs = query.iter().fold(0.0f32, |value, &x| value.max(x.abs()));
                if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 }
            }
            QuantizedI8Encoding::GlobalSymmetric => self.global_dequant_scale,
        }
    }

    fn routed_trees(&self) -> Option<&[QuantizedFlatTree]> {
        if self.quantized_trees.is_empty() {
            None
        } else {
            Some(&self.quantized_trees)
        }
    }

    fn quantize_query(&self, query: &[f32], quantized_query: &mut Vec<i8>) -> f32 {
        let multiplier = match self.encoding {
            QuantizedI8Encoding::PerVectorCosine | QuantizedI8Encoding::PerVectorSymmetric => {
                let max_abs = query.iter().fold(0.0f32, |value, &x| value.max(x.abs()));
                if max_abs > 0.0 { 127.0 / max_abs } else { 0.0 }
            }
            QuantizedI8Encoding::GlobalSymmetric => self.global_multiplier,
        };
        quantized_query.resize(query.len(), 0);
        for (output, &value) in quantized_query.iter_mut().zip(query) {
            *output = (value * multiplier).round().clamp(-127.0, 127.0) as i8;
        }
        let norm_sq: i32 = quantized_query
            .iter()
            .map(|&x| i32::from(x) * i32::from(x))
            .sum();
        let query_inv_norm = match self.encoding {
            QuantizedI8Encoding::PerVectorCosine if norm_sq > 0 => {
                (norm_sq as f32).sqrt().recip()
            }
            QuantizedI8Encoding::PerVectorCosine => 0.0,
            QuantizedI8Encoding::GlobalSymmetric => self.global_dequant_scale,
            QuantizedI8Encoding::PerVectorSymmetric => if multiplier > 0.0 { multiplier.recip() } else { 0.0 },
        };
        query_inv_norm
    }

    fn query_one_quantized(
        &self,
        query: &[f32],
        k: usize,
        epsilon: f32,
    ) -> (Vec<i32>, Vec<f32>) {
        self.query_one_quantized_widths(
            query, k, epsilon, self.candidate_width, self.rerank_width,
        )
    }

    fn query_one_quantized_widths(
        &self,
        query: &[f32],
        k: usize,
        epsilon: f32,
        candidate_width: usize,
        rerank_width: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        // Always acquire these mutexes in scratch-then-workspace order. The
        // scratch guard stays alive because the traversal borrows its buffer.
        let mut quantized_query = self.query_scratch.lock().expect("quantized query mutex poisoned");
        let query_inv_norm = self.quantize_query(query, &mut quantized_query);
        let mut workspace = self.workspace.lock().expect("query workspace mutex poisoned");
        self.index.query_one_quantized_i8_with_workspace(
            query,
            &quantized_query,
            query_inv_norm,
            &self.quantized_data,
            &self.inv_norms,
            self.routed_trees(),
            self.tree_query_scale(query),
            k,
            candidate_width,
            rerank_width,
            epsilon,
            &mut workspace,
        )
    }

}

impl AnyIndex for QuantizedI8Index {
    fn query(&self, queries: &[f32], n_queries: usize, k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>) {
        let mut all_indices = Vec::with_capacity(n_queries * k);
        let mut all_distances = Vec::with_capacity(n_queries * k);
        for query in queries.chunks_exact(self.index.dim).take(n_queries) {
            let (indices, distances) = self.query_one_quantized(query, k, epsilon);
            all_indices.extend(indices);
            all_distances.extend(distances);
        }
        (all_indices, all_distances)
    }

    fn query_one(&self, query: &[f32], k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>) {
        self.query_one_quantized(query, k, epsilon)
    }

    fn query_one_indices(&self, query: &[f32], k: usize, epsilon: f32) -> Vec<i32> {
        self.query_one_quantized(query, k, epsilon).0
    }

    fn query_one_quantized_widths(
        &self,
        query: &[f32],
        k: usize,
        epsilon: f32,
        candidate_width: usize,
        rerank_width: usize,
    ) -> PyResult<(Vec<i32>, Vec<f32>)> {
        self.ensure_rerank_available(k, candidate_width, rerank_width)?;
        Ok(self.query_one_quantized_widths(
            query, k, epsilon, candidate_width, rerank_width,
        ))
    }

    fn neighbor_indices(&self) -> &[i32] { &self.index.neighbor_indices }
    fn neighbor_distances(&self) -> &[f32] { &self.index.neighbor_distances }
    fn search_graph_original_order(&self) -> SearchGraph { self.index.search_graph_original_order() }
    fn search_graph_min_distance(&self) -> f32 { self.index.min_distance }
    fn export_search_tree(&self) -> Option<FlatTree> { self.index.export_search_tree() }
    fn storage_info(&self) -> (&'static str, usize, usize, usize, f64) {
        let encoding = match self.encoding {
            QuantizedI8Encoding::PerVectorCosine => "int8_per_vector_cosine",
            QuantizedI8Encoding::GlobalSymmetric => "sq8u_global_symmetric",
            QuantizedI8Encoding::PerVectorSymmetric => "sq8p_per_vector_symmetric",
        };
        let code_bytes = self.quantized_data.len();
        let metadata_bytes = match self.encoding {
            QuantizedI8Encoding::PerVectorCosine | QuantizedI8Encoding::PerVectorSymmetric =>
                self.inv_norms.len() * std::mem::size_of::<f32>(),
            QuantizedI8Encoding::GlobalSymmetric =>
                self.inv_norms.len() * std::mem::size_of::<f32>(),
        };
        (
            encoding,
            code_bytes,
            metadata_bytes,
            self.index.retained_fp32_bytes(),
            self.encode_seconds,
        )
    }

    fn quantized_tree_storage_info(&self) -> (usize, usize, usize, usize, usize, usize) {
        let code_bytes = self.quantized_trees.iter().map(QuantizedFlatTree::code_bytes).sum();
        let scale_bytes = self.quantized_trees.iter().map(QuantizedFlatTree::scale_bytes).sum();
        let offset_bytes = self.quantized_trees.iter().map(QuantizedFlatTree::offset_bytes).sum();
        let topology_bytes = self.quantized_trees.iter().map(|tree| {
            tree.children.len() * std::mem::size_of::<[i32; 2]>()
                + tree.indices.len() * std::mem::size_of::<i32>()
        }).sum();
        (
            code_bytes,
            scale_bytes,
            offset_bytes,
            topology_bytes,
            0,
            self.released_fp32_tree_bytes,
        )
    }

    fn retained_fp32_tree_topology_bytes(&self) -> usize {
        self.index.search_trees.iter().map(|tree| {
            tree.children.len() * std::mem::size_of::<[i32; 2]>()
                + tree.indices.len() * std::mem::size_of::<i32>()
        }).sum()
    }

    fn released_fp32_data_bytes(&self) -> usize {
        self.released_fp32_data_bytes
    }

}

impl<D: Distance<f32> + Send + Sync> AnyIndexWithWorkspace<D> {
    fn new(mut index: NNDescentIndex<D>, min_distance_override: Option<f32>) -> Self {
        if let Some(v) = min_distance_override {
            index.min_distance = v;
        }
        let initial_k = index.n_neighbors.max(1);
        Self {
            workspace: Mutex::new(SearchWorkspace::new(index.n_points, initial_k)),
            index,
        }
    }
}

impl<D: Distance<f32> + Send + Sync> AnyIndex for AnyIndexWithWorkspace<D> {
    fn query(&self, queries: &[f32], n_queries: usize, k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>) {
        self.index.query(queries, n_queries, k, epsilon)
    }
    fn query_one(&self, query: &[f32], k: usize, epsilon: f32) -> (Vec<i32>, Vec<f32>) {
        let mut workspace = self.workspace.lock().expect("query workspace mutex poisoned");
        self.index
            .query_one_with_workspace(query, k, epsilon, &mut workspace)
    }
    fn query_one_indices(&self, query: &[f32], k: usize, epsilon: f32) -> Vec<i32> {
        let mut workspace = self.workspace.lock().expect("query workspace mutex poisoned");
        self.index
            .query_one_indices_with_workspace(query, k, epsilon, &mut workspace)
    }
    fn neighbor_indices(&self) -> &[i32] {
        &self.index.neighbor_indices
    }
    fn neighbor_distances(&self) -> &[f32] {
        &self.index.neighbor_distances
    }
    fn search_graph_original_order(&self) -> SearchGraph {
        self.index.search_graph_original_order()
    }
    fn search_graph_min_distance(&self) -> f32 {
        self.index.min_distance
    }
    fn export_search_tree(&self) -> Option<FlatTree> {
        self.index.export_search_tree()
    }
    fn storage_info(&self) -> (&'static str, usize, usize, usize, f64) {
        (
            "fp32",
            self.index.data.len() * std::mem::size_of::<f32>(),
            0,
            0,
            0.0,
        )
    }
}

#[pymethods]
impl PyNNDescent {
    #[new]
    #[pyo3(signature = (data, metric="euclidean", n_neighbors=30, n_trees=None, n_search_trees=1, search_tree_leaf_budget=1, leaf_size=None, max_candidates=None, n_iters=None, delta=0.001, random_state=None, diversify_prob=1.0, pruning_degree_multiplier=1.5, verbose=false, normalize=false, cosine_distance_mode="log", quantized_candidate_width=0, quantized_rerank_width=0, tree_quantization="none"))]
    fn new(
        data: PyReadonlyArray2<f32>,
        metric: &str,
        n_neighbors: usize,
        n_trees: Option<usize>,
        n_search_trees: usize,
        search_tree_leaf_budget: usize,
        leaf_size: Option<usize>,
        max_candidates: Option<usize>,
        n_iters: Option<usize>,
        delta: f32,
        random_state: Option<u64>,
        diversify_prob: f32,
        pruning_degree_multiplier: f32,
        verbose: bool,
        normalize: bool,
        cosine_distance_mode: &str,
        quantized_candidate_width: usize,
        quantized_rerank_width: usize,
        tree_quantization: &str,
    ) -> PyResult<Self> {
        let shape = data.shape();
        let n_points = shape[0];
        let dim = shape[1];

        if quantized_candidate_width > 0 && quantized_rerank_width > quantized_candidate_width {
            return Err(PyValueError::new_err(
                "quantized_rerank_width must not exceed quantized_candidate_width when candidate width is explicit",
            ));
        }

        // Copy data to owned vec in C-contiguous (row-major) order
        let mut data_vec: Vec<f32> = if data.is_c_contiguous() {
            data.as_slice().unwrap().to_vec()
        } else {
            // F-contiguous or strided: read element by element in row-major order
            let mut vec = Vec::with_capacity(n_points * dim);
            for i in 0..n_points {
                for j in 0..dim {
                    vec.push(*data.get([i, j]).unwrap());
                }
            }
            vec
        };

        // Parse metric
        let parsed_metric = Metric::from_str(metric)
            .ok_or_else(|| PyValueError::new_err(format!("Unknown metric: {}", metric)))?;

        // For cosine, optional pre-normalization lets query-time distance use a dot-only path.
        let normalize_cosine = normalize && matches!(parsed_metric, Metric::Cosine);
        let (direct_cosine, quantized_encoding) = match cosine_distance_mode {
            "log" => (false, None),
            "direct" => (true, None),
            "int8" => (true, Some(QuantizedI8Encoding::PerVectorCosine)),
            "sq8u" => (true, Some(QuantizedI8Encoding::GlobalSymmetric)),
            "sq8p" => (true, Some(QuantizedI8Encoding::PerVectorSymmetric)),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown cosine_distance_mode: {other}; expected 'log', 'direct', 'int8', 'sq8u', or 'sq8p'"
                )))
            }
        };
        let quantize_trees = match tree_quantization {
            "none" => false,
            "int8" => true,
            other => return Err(PyValueError::new_err(format!(
                "Unknown tree_quantization: {other}; expected 'none' or 'int8'"
            ))),
        };
        if quantize_trees && quantized_encoding.is_none() {
            return Err(PyValueError::new_err(
                "tree_quantization='int8' is available only with a quantized cosine_distance_mode",
            ));
        }
        if quantize_trees && matches!(quantized_encoding, Some(QuantizedI8Encoding::GlobalSymmetric)) {
            return Err(PyValueError::new_err(
                "tree_quantization='int8' is incompatible with cosine_distance_mode='sq8u'; use tree_quantization='none'",
            ));
        }
        if direct_cosine && !normalize_cosine {
            return Err(PyValueError::new_err(
                "direct and quantized cosine modes require metric='cosine' and normalize=True",
            ));
        }
        if normalize_cosine {
            normalize_rows_inplace(&mut data_vec, dim);
        }

        // Build index based on metric
        let index_data = Self::build_index(
            &data_vec,
            n_points,
            dim,
            parsed_metric,
            normalize_cosine,
            direct_cosine,
            quantized_encoding,
            quantized_candidate_width,
            quantized_rerank_width,
            quantize_trees,
            n_neighbors,
            n_trees,
            n_search_trees,
            search_tree_leaf_budget,
            leaf_size,
            max_candidates,
            n_iters,
            delta,
            random_state.unwrap_or(42),
            diversify_prob,
            pruning_degree_multiplier,
            verbose,
        )?;

        Ok(Self {
            n_points,
            dim,
            normalize: normalize_cosine,
            n_neighbors,
            index_data,
            query_scratch: Mutex::new(vec![0.0; dim]),
        })
    }

    /// Query for nearest neighbors.
    ///
    /// Parameters
    /// ----------
    /// query_data : numpy.ndarray
    ///     2D array of shape (n_queries, n_features) containing query points.
    /// k : int, default=10
    ///     Number of neighbors to return.
    /// epsilon : float, default=0.1
    ///     Search expansion factor. Larger values give more accurate results
    ///     but slower queries.
    ///
    /// Returns
    /// -------
    /// indices : numpy.ndarray
    ///     2D array of shape (n_queries, k) containing neighbor indices.
    /// distances : numpy.ndarray
    ///     2D array of shape (n_queries, k) containing distances to neighbors.
    #[pyo3(signature = (query_data, k=10, epsilon=0.1))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        query_data: PyReadonlyArray2<f32>,
        k: usize,
        epsilon: f32,
    ) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f32>>)> {
        let shape = query_data.shape();
        let n_queries = shape[0];
        let query_dim = shape[1];

        if query_dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match data dimension {}",
                query_dim, self.dim
            )));
        }

        let mut query_vec: Vec<f32> = if query_data.is_c_contiguous() {
            query_data.as_slice().unwrap().to_vec()
        } else {
            let mut vec = Vec::with_capacity(n_queries * query_dim);
            for i in 0..n_queries {
                for j in 0..query_dim {
                    vec.push(*query_data.get([i, j]).unwrap());
                }
            }
            vec
        };

        if self.normalize {
            normalize_rows_inplace(&mut query_vec, query_dim);
        }

        let (indices, distances) = py.allow_threads(|| {
            self.index_data.query(&query_vec, n_queries, k, epsilon)
        });

        // Create 2D arrays directly
        let indices_arr = PyArray1::from_vec_bound(py, indices);
        let distances_arr = PyArray1::from_vec_bound(py, distances);

        let indices_2d = indices_arr.reshape([n_queries, k])?;
        let distances_2d = distances_arr.reshape([n_queries, k])?;

        Ok((indices_2d, distances_2d))
    }

    /// Query for nearest neighbors of a single vector with lower Python overhead.
    #[pyo3(signature = (query, k=10, epsilon=0.1))]
    fn query_one<'py>(
        &self,
        py: Python<'py>,
        query: numpy::PyReadonlyArray1<f32>,
        k: usize,
        epsilon: f32,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>)> {
        let qdim = query.shape()[0];
        if qdim != self.dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match data dimension {}",
                qdim, self.dim
            )));
        }

        let query_slice = query.as_slice()?;
        let (indices, distances) = if self.normalize {
            let mut scratch = self
                .query_scratch
                .lock()
                .expect("query scratch mutex poisoned");
            if scratch.len() != qdim {
                scratch.resize(qdim, 0.0);
            }
            scratch.copy_from_slice(query_slice);
            normalize_rows_inplace(&mut scratch[..], qdim);
            py.allow_threads(|| self.index_data.query_one(&scratch[..], k, epsilon))
        } else {
            py.allow_threads(|| self.index_data.query_one(query_slice, k, epsilon))
        };

        let idx = PyArray1::from_vec_bound(py, indices);
        let dist = PyArray1::from_vec_bound(py, distances);
        Ok((idx, dist))
    }

    /// Query a single vector and return only indices.
    ///
    /// This avoids Python-side distance array materialization when callers
    /// only need neighbor ids (common in benchmark single-query loops).
    #[pyo3(signature = (query, k=10, epsilon=0.1))]
    fn query_one_indices<'py>(
        &self,
        py: Python<'py>,
        query: numpy::PyReadonlyArray1<f32>,
        k: usize,
        epsilon: f32,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let qdim = query.shape()[0];
        if qdim != self.dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match data dimension {}",
                qdim, self.dim
            )));
        }

        let query_slice = query.as_slice()?;
        let indices = if self.normalize {
            let mut scratch = self
                .query_scratch
                .lock()
                .expect("query scratch mutex poisoned");
            if scratch.len() != qdim {
                scratch.resize(qdim, 0.0);
            }
            scratch.copy_from_slice(query_slice);
            normalize_rows_inplace(&mut scratch[..], qdim);
            py.allow_threads(|| self.index_data.query_one_indices(&scratch[..], k, epsilon))
        } else {
            py.allow_threads(|| self.index_data.query_one_indices(query_slice, k, epsilon))
        };

        Ok(PyArray1::from_vec_bound(py, indices))
    }

    /// Query a quantized index with independent query-time candidate and FP32
    /// rerank widths. The index and graph are not rebuilt.
    ///
    /// `candidate_width=0` selects legacy compatibility semantics:
    /// `max(k, rerank_width)`. An explicit candidate width must be at least
    /// `k`; rerank width must be zero or in `k..=candidate_width`.
    #[pyo3(signature = (query, k=10, epsilon=0.1, candidate_width=0, rerank_width=0))]
    fn query_one_indices_quantized<'py>(
        &self,
        py: Python<'py>,
        query: numpy::PyReadonlyArray1<f32>,
        k: usize,
        epsilon: f32,
        candidate_width: usize,
        rerank_width: usize,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        validate_quantized_widths(k, candidate_width, rerank_width)?;
        let qdim = query.shape()[0];
        if qdim != self.dim {
            return Err(PyValueError::new_err(format!(
                "Query dimension {} does not match data dimension {}",
                qdim, self.dim
            )));
        }

        let query_slice = query.as_slice()?;
        let result = if self.normalize {
            let mut scratch = self.query_scratch.lock().expect("query scratch mutex poisoned");
            if scratch.len() != qdim { scratch.resize(qdim, 0.0); }
            scratch.copy_from_slice(query_slice);
            normalize_rows_inplace(&mut scratch[..], qdim);
            py.allow_threads(|| self.index_data.query_one_quantized_widths(
                &scratch[..], k, epsilon, candidate_width, rerank_width,
            ))
        } else {
            py.allow_threads(|| self.index_data.query_one_quantized_widths(
                query_slice, k, epsilon, candidate_width, rerank_width,
            ))
        }?;
        Ok(PyArray1::from_vec_bound(py, result.0))
    }

    /// Export current search graph in CSR format.
    #[getter]
    fn search_graph<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>, f32)> {
        let graph = self.index_data.search_graph_original_order();
        let min_distance = self.index_data.search_graph_min_distance();
        Ok((
            PyArray1::from_vec_bound(py, graph.indptr),
            PyArray1::from_vec_bound(py, graph.indices),
            min_distance,
        ))
    }

    /// Return search-representation storage and encoding metadata.
    fn storage_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let (encoding, code_bytes, metadata_bytes, retained_fp32_bytes, encode_seconds) =
            self.index_data.storage_info();
        let info = PyDict::new_bound(py);
        info.set_item("encoding", encoding)?;
        info.set_item("code_bytes", code_bytes)?;
        info.set_item("metadata_bytes", metadata_bytes)?;
        info.set_item("retained_fp32_bytes", retained_fp32_bytes)?;
        info.set_item("encode_seconds", encode_seconds)?;
        let (tree_code_bytes, tree_scale_bytes, tree_offset_bytes, tree_topology_bytes,
            hypothetical_releasable_fp32_tree_bytes, released_fp32_tree_bytes) =
            self.index_data.quantized_tree_storage_info();
        info.set_item("tree_code_bytes", tree_code_bytes)?;
        info.set_item("tree_scale_bytes", tree_scale_bytes)?;
        info.set_item("tree_offset_bytes", tree_offset_bytes)?;
        info.set_item("tree_topology_bytes", tree_topology_bytes)?;
        info.set_item("hypothetical_releasable_fp32_tree_bytes", hypothetical_releasable_fp32_tree_bytes)?;
        info.set_item("released_fp32_tree_bytes", released_fp32_tree_bytes)?;
        info.set_item(
            "released_fp32_data_bytes",
            self.index_data.released_fp32_data_bytes(),
        )?;
        let retained_fp32_tree_topology_bytes =
            self.index_data.retained_fp32_tree_topology_bytes();
        info.set_item(
            "retained_fp32_tree_topology_bytes",
            retained_fp32_tree_topology_bytes,
        )?;
        let allocated_bytes = code_bytes + metadata_bytes + retained_fp32_bytes
            + retained_fp32_tree_topology_bytes
            + tree_code_bytes + tree_scale_bytes + tree_offset_bytes + tree_topology_bytes;
        info.set_item("allocated_bytes", allocated_bytes)?;
        info.set_item(
            "bytes_per_vector",
            allocated_bytes as f64 / self.n_points as f64,
        )?;
        Ok(info)
    }

    /// Get the computed neighbor graph.
    ///
    /// Returns
    /// -------
    /// indices : numpy.ndarray
    ///     2D array of shape (n_samples, n_neighbors) containing neighbor indices.
    /// distances : numpy.ndarray
    ///     2D array of shape (n_samples, n_neighbors) containing neighbor distances.
    #[getter]
    fn neighbor_graph<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f32>>)> {
        // Return the stored neighbor graph (no re-query needed)
        let indices = self.index_data.neighbor_indices().to_vec();
        let distances = self.index_data.neighbor_distances().to_vec();

        let indices_arr = PyArray1::from_vec_bound(py, indices);
        let distances_arr = PyArray1::from_vec_bound(py, distances);

        let indices_2d = indices_arr.reshape([self.n_points, self.n_neighbors])?;
        let distances_2d = distances_arr.reshape([self.n_points, self.n_neighbors])?;

        Ok((indices_2d, distances_2d))
    }

    /// Export the first retained RP tree in original input-ID space.
    ///
    /// Returns `(hyperplanes, offsets, children, leaf_indices)`.
    fn export_search_tree<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray2<i32>>,
        Bound<'py, PyArray1<i32>>,
    )> {
        let tree = self
            .index_data
            .export_search_tree()
            .ok_or_else(|| PyValueError::new_err("no retained search tree is available"))?;
        let children: Vec<i32> = tree.children.into_iter().flatten().collect();
        let hyperplanes = PyArray1::from_vec_bound(py, tree.hyperplanes)
            .reshape([tree.n_nodes, tree.dim])?;
        let offsets = PyArray1::from_vec_bound(py, tree.offsets);
        let children = PyArray1::from_vec_bound(py, children)
            .reshape([tree.n_nodes, 2])?;
        let indices = PyArray1::from_vec_bound(py, tree.indices);
        Ok((hyperplanes, offsets, children, indices))
    }
}

fn normalize_rows_inplace(data: &mut [f32], dim: usize) {
    if dim == 0 {
        return;
    }

    for row in data.chunks_mut(dim) {
        let norm_sq = row.iter().map(|&x| x * x).sum::<f32>();
        if norm_sq <= 1e-12 {
            continue;
        }
        let inv_norm = norm_sq.sqrt().recip();
        for x in row {
            *x *= inv_norm;
        }
    }
}

fn validate_quantized_widths(k: usize, candidate_width: usize, rerank_width: usize) -> PyResult<()> {
    if candidate_width == 0 {
        return Ok(());
    }
    if candidate_width < k {
        return Err(PyValueError::new_err(format!(
            "candidate_width {candidate_width} must be at least k ({k})"
        )));
    }
    if rerank_width > candidate_width {
        return Err(PyValueError::new_err(format!(
            "rerank_width {rerank_width} must not exceed candidate_width {candidate_width}"
        )));
    }
    if rerank_width > 0 && rerank_width < k {
        return Err(PyValueError::new_err(format!(
            "rerank_width {rerank_width} must be zero or at least k ({k})"
        )));
    }
    Ok(())
}

impl PyNNDescent {
    fn build_index(
        data: &[f32],
        n_points: usize,
        dim: usize,
        metric: Metric,
        normalize_cosine: bool,
        direct_cosine: bool,
        quantized_encoding: Option<QuantizedI8Encoding>,
        quantized_candidate_width: usize,
        quantized_rerank_width: usize,
        quantize_trees: bool,
        n_neighbors: usize,
        n_trees: Option<usize>,
        n_search_trees: usize,
        search_tree_leaf_budget: usize,
        leaf_size: Option<usize>,
        max_candidates: Option<usize>,
        n_iters: Option<usize>,
        delta: f32,
        random_seed: u64,
        diversify_prob: f32,
        pruning_degree_multiplier: f32,
        verbose: bool,
    ) -> PyResult<Box<dyn AnyIndex>> {
        // Compute default n_trees matching PyNNDescent: max(3, min(12, round(2*log10(n))))
        let effective_n_trees = n_trees.unwrap_or_else(|| {
            let log_val = 2.0 * (n_points as f64).log10();
            (log_val.round() as usize).clamp(3, 12)
        });

        let mut builder = NNDescentBuilder::new(data, n_points, dim)
            .metric(metric)
            .n_neighbors(n_neighbors)
            .n_trees(effective_n_trees)
            .n_search_trees(n_search_trees)
            .search_tree_leaf_budget(search_tree_leaf_budget)
            .delta(delta)
            .random_seed(random_seed)
            .diversify_prob(diversify_prob)
            .pruning_degree_multiplier(pruning_degree_multiplier)
            .verbose(verbose);

        if let Some(ls) = leaf_size {
            builder = builder.leaf_size(ls);
        }
        if let Some(mc) = max_candidates {
            builder = builder.max_candidates(mc);
        }
        if let Some(ni) = n_iters {
            builder = builder.n_iters(ni);
        }
        // Dispatch to the correct concrete distance type for each metric.
        // Metrics with fast alternatives use proxy distances + correction.
        // Others use the direct distance function.
        macro_rules! build {
            ($dist:expr, $corr:expr, $min_override:expr) => {
                Box::new(AnyIndexWithWorkspace::new(
                    builder.build_with_distance($dist, $corr),
                    $min_override,
                ))
                    as Box<dyn AnyIndex>
            };
        }

        let index_data: Box<dyn AnyIndex> = match metric {
            // Minkowski family
            Metric::Euclidean | Metric::L2 => build!(SquaredEuclidean, Some(|d: f32| d.sqrt()), None),
            Metric::SquaredEuclidean => build!(SquaredEuclidean, None, None),
            Metric::Manhattan => build!(Manhattan, None, None),
            Metric::Chebyshev => build!(Chebyshev, None, None),
            Metric::Canberra => build!(Canberra, None, None),
            Metric::BrayCurtis => build!(BrayCurtis, None, None),
            // Angular / similarity
            Metric::Cosine => {
                if normalize_cosine {
                    if let Some(encoding) = quantized_encoding {
                        Box::new(QuantizedI8Index::new(
                            builder.build_with_distance(DirectNormalizedCosine, None),
                            quantized_candidate_width,
                            quantized_rerank_width,
                            encoding,
                            quantize_trees,
                        )) as Box<dyn AnyIndex>
                    } else if direct_cosine {
                        build!(DirectNormalizedCosine, None, None)
                    } else {
                        build!(AlternativeDot, Some(correct_alternative_cosine), None)
                    }
                } else {
                    build!(AlternativeCosine, Some(correct_alternative_cosine), None)
                }
            }
            Metric::Dot => build!(AlternativeDot, Some(correct_alternative_cosine), None),
            Metric::InnerProduct => build!(AlternativeInnerProduct, Some(correct_alternative_inner_product), None),
            Metric::Correlation => build!(Correlation, None, None),
            Metric::TrueAngular => build!(TrueAngular, None, None),
            Metric::TSSS => build!(TSSS, None, None),
            // Binary / set
            Metric::Hamming => build!(Hamming, None, None),
            Metric::Jaccard => build!(Jaccard, None, None),
            Metric::Dice => build!(Dice, None, None),
            Metric::Matching => build!(Matching, None, None),
            Metric::Kulsinski => build!(Kulsinski, None, None),
            Metric::RogersTanimoto => build!(RogersTanimoto, None, None),
            Metric::RussellRao => build!(RussellRao, None, None),
            Metric::SokalMichener => build!(SokalMichener, None, None),
            Metric::SokalSneath => build!(SokalSneath, None, None),
            Metric::Yule => build!(Yule, None, None),
            // Distribution
            Metric::Hellinger => build!(Hellinger, None, None),
            Metric::JensenShannon => build!(JensenShannon, None, None),
            Metric::SymmetricKL => build!(SymmetricKL, None, None),
        };

        Ok(index_data)
    }
}

/// Get the version of the nndescent-rs library.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Check available SIMD support.
#[pyfunction]
fn simd_info() -> String {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let mut features = Vec::new();
        if is_x86_feature_detected!("avx512f") {
            features.push("AVX-512F");
        }
        if is_x86_feature_detected!("avx2") {
            features.push("AVX2");
        }
        if is_x86_feature_detected!("fma") {
            features.push("FMA");
        }
        if is_x86_feature_detected!("sse4.1") {
            features.push("SSE4.1");
        }

        if features.is_empty() {
            "Scalar (no SIMD)".to_string()
        } else {
            features.join(", ")
        }
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        "Scalar (non-x86 platform)".to_string()
    }
}

/// Benchmark heap push operations.
/// 
/// This function simulates the heap push operations that occur during
/// candidate building in NN-Descent.
#[pyfunction]
fn benchmark_heap_push(
    n_vertices: usize,
    k: usize,
    max_candidates: usize,
    n_iters: usize,
    test_indices: PyReadonlyArray2<i32>,
    test_priorities: numpy::PyReadonlyArray4<f32>,
) -> PyResult<usize> {
    let indices_slice = test_indices.as_slice()?;
    let priorities_slice = test_priorities.as_slice()?;
    
    let mut total_pushes: usize = 0;
    
    // Allocate flat arrays for heaps
    let size = n_vertices * max_candidates;
    let mut heap_priorities = vec![f32::INFINITY; size];
    let mut heap_indices = vec![-1i32; size];
    
    for iter_idx in 0..n_iters {
        // Reset heaps
        for i in 0..size {
            heap_priorities[i] = f32::INFINITY;
            heap_indices[i] = -1;
        }
        
        // Simulate pushing edges (forward + reverse)
        for i in 0..n_vertices {
            for j in 0..k {
                let neighbor = indices_slice[i * k + j];
                if neighbor < 0 {
                    continue;
                }
                
                // Forward edge: push neighbor as candidate for vertex i
                let priority_idx = iter_idx * n_vertices * k * 2 + i * k * 2 + j * 2;
                let priority = priorities_slice[priority_idx];
                
                let offset_i = i * max_candidates;
                checked_heap_push_bench(
                    &mut heap_priorities[offset_i..offset_i + max_candidates],
                    &mut heap_indices[offset_i..offset_i + max_candidates],
                    priority,
                    neighbor,
                );
                total_pushes += 1;
                
                // Reverse edge: push i as candidate for neighbor
                let reverse_priority = priorities_slice[priority_idx + 1];
                let neighbor_idx = neighbor as usize;
                let offset_n = neighbor_idx * max_candidates;
                checked_heap_push_bench(
                    &mut heap_priorities[offset_n..offset_n + max_candidates],
                    &mut heap_indices[offset_n..offset_n + max_candidates],
                    reverse_priority,
                    i as i32,
                );
                total_pushes += 1;
            }
        }
    }
    
    Ok(total_pushes)
}

/// Push to a bounded priority max-heap with duplicate checking.
#[inline]
fn checked_heap_push_bench(
    priorities: &mut [f32],
    indices: &mut [i32],
    priority: f32,
    index: i32,
) {
    // Early exit if priority is worse than current max
    if priority >= priorities[0] {
        return;
    }

    // Check for duplicate (linear scan)
    let n = priorities.len();
    for i in 0..n {
        if indices[i] == index {
            return;
        }
    }

    // Insert by replacing root and sifting down
    priorities[0] = priority;
    indices[0] = index;
    
    // Sift down to maintain max-heap property
    let mut pos = 0;
    loop {
        let left = 2 * pos + 1;
        let right = 2 * pos + 2;
        let mut largest = pos;

        if left < n && priorities[left] > priorities[largest] {
            largest = left;
        }
        if right < n && priorities[right] > priorities[largest] {
            largest = right;
        }

        if largest != pos {
            priorities.swap(pos, largest);
            indices.swap(pos, largest);
            pos = largest;
        } else {
            break;
        }
    }
}

/// Benchmark candidate building from a graph.
///
/// Takes graph indices, distances, and flags, builds candidate sets.
/// Returns tuple of (new_candidates, old_candidates).
#[pyfunction]
fn benchmark_candidate_building<'py>(
    py: Python<'py>,
    graph_indices: PyReadonlyArray2<i32>,
    graph_distances: PyReadonlyArray2<f32>,
    graph_flags: PyReadonlyArray2<u8>,
    max_candidates: usize,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i32>>)> {
    use nndescent_core::heap::NeighborHeap;
    use nndescent_core::nndescent::CandidateSets;
    use nndescent_core::rng::FastRng;
    
    let indices_view = graph_indices.as_array();
    let distances_view = graph_distances.as_array();
    let flags_view = graph_flags.as_array();
    
    let n_vertices = indices_view.shape()[0];
    let k = indices_view.shape()[1];
    
    // Create a NeighborHeap from the input data
    let mut heap = NeighborHeap::new(n_vertices, k);
    
    // Copy data into the heap
    for i in 0..n_vertices {
        for j in 0..k {
            heap.indices[i * k + j] = indices_view[[i, j]];
            heap.distances[i * k + j] = distances_view[[i, j]];
            heap.flags[i * k + j] = flags_view[[i, j]];
        }
    }
    
    let mut rng = FastRng::new(42);
    
    let candidates = CandidateSets::build_from_graph(&mut heap, max_candidates, &mut rng);
    
    // Convert to numpy arrays
    let new_indices = PyArray1::from_vec_bound(py, candidates.new_indices)
        .reshape([n_vertices, max_candidates])?;
    let old_indices = PyArray1::from_vec_bound(py, candidates.old_indices)
        .reshape([n_vertices, max_candidates])?;
    
    Ok((new_indices, old_indices))
}

/// Benchmark distance computations.
///
/// Computes distances for given pairs.
#[pyfunction]
#[pyo3(signature = (data, pairs_i, pairs_j, metric="sqeuclidean"))]
fn benchmark_distances<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
    pairs_i: numpy::PyReadonlyArray1<i32>,
    pairs_j: numpy::PyReadonlyArray1<i32>,
    metric: &str,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let data_view = data.as_array();
    let n_points = data_view.shape()[0];
    let dim = data_view.shape()[1];
    let data_slice = data.as_slice()?;

    let pairs_i_slice = pairs_i.as_slice()?;
    let pairs_j_slice = pairs_j.as_slice()?;
    if pairs_i_slice.len() != pairs_j_slice.len() {
        return Err(PyValueError::new_err(
            "pairs_i and pairs_j must have the same length",
        ));
    }

    let metric = Metric::from_str(metric).ok_or_else(|| {
        PyValueError::new_err(format!(
            "Unknown benchmark distance metric: {} (expected one of: sqeuclidean, euclidean, cosine, dot, inner_product)",
            metric
        ))
    })?;

    if !matches!(
        metric,
        Metric::SquaredEuclidean | Metric::Euclidean | Metric::L2 | Metric::Cosine | Metric::Dot | Metric::InnerProduct
    ) {
        return Err(PyValueError::new_err(
            "benchmark_distances currently supports metrics: sqeuclidean, euclidean, cosine, dot, inner_product",
        ));
    }

    let n_pairs = pairs_i_slice.len();

    let mut results = Vec::with_capacity(n_pairs);
    for k in 0..n_pairs {
        let i = usize::try_from(pairs_i_slice[k]).map_err(|_| {
            PyValueError::new_err(format!("pairs_i[{}] is negative", k))
        })?;
        let j = usize::try_from(pairs_j_slice[k]).map_err(|_| {
            PyValueError::new_err(format!("pairs_j[{}] is negative", k))
        })?;
        if i >= n_points {
            return Err(PyValueError::new_err(format!(
                "pairs_i[{}]={} out of range for n_points={} ",
                k, i, n_points
            )));
        }
        if j >= n_points {
            return Err(PyValueError::new_err(format!(
                "pairs_j[{}]={} out of range for n_points={} ",
                k, j, n_points
            )));
        }

        let vi = &data_slice[i * dim..(i + 1) * dim];
        let vj = &data_slice[j * dim..(j + 1) * dim];
        results.push(metric.distance(vi, vj));
    }

    Ok(PyArray1::from_vec_bound(py, results))
}

/// Benchmark quantized u8 squared-euclidean distances.
///
/// Quantizes all data vectors to u8 once per call, then computes distances
/// between float queries (from pairs_i) and quantized vectors (from pairs_j).
#[pyfunction]
#[pyo3(signature = (data, pairs_i, pairs_j))]
fn benchmark_quantized_distances_u8<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<f32>,
    pairs_i: numpy::PyReadonlyArray1<i32>,
    pairs_j: numpy::PyReadonlyArray1<i32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let data_view = data.as_array();
    let n_points = data_view.shape()[0];
    let dim = data_view.shape()[1];
    let data_slice = data.as_slice()?;

    let pairs_i_slice = pairs_i.as_slice()?;
    let pairs_j_slice = pairs_j.as_slice()?;
    if pairs_i_slice.len() != pairs_j_slice.len() {
        return Err(PyValueError::new_err(
            "pairs_i and pairs_j must have the same length",
        ));
    }

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &x in data_slice {
        if x < min_v {
            min_v = x;
        }
        if x > max_v {
            max_v = x;
        }
    }
    let span = (max_v - min_v).max(1e-12);
    let step = span / 255.0;

    let mut codebook = vec![0.0f32; 256];
    for (i, c) in codebook.iter_mut().enumerate() {
        *c = min_v + step * (i as f32);
    }

    let mut quantized = vec![0u8; data_slice.len()];
    for (i, &x) in data_slice.iter().enumerate() {
        let q = ((x - min_v) / step).round().clamp(0.0, 255.0) as u8;
        quantized[i] = q;
    }

    let n_pairs = pairs_i_slice.len();
    let mut results = Vec::with_capacity(n_pairs);
    for k in 0..n_pairs {
        let i = usize::try_from(pairs_i_slice[k]).map_err(|_| {
            PyValueError::new_err(format!("pairs_i[{}] is negative", k))
        })?;
        let j = usize::try_from(pairs_j_slice[k]).map_err(|_| {
            PyValueError::new_err(format!("pairs_j[{}] is negative", k))
        })?;
        if i >= n_points || j >= n_points {
            return Err(PyValueError::new_err(format!(
                "pair index out of range at {} (i={}, j={}, n_points={})",
                k, i, j, n_points
            )));
        }

        let q = &data_slice[i * dim..(i + 1) * dim];
        let yq = &quantized[j * dim..(j + 1) * dim];
        results.push(quantized_u8_sq_euclidean(q, yq, &codebook));
    }

    Ok(PyArray1::from_vec_bound(py, results))
}

/// Benchmark candidate distances on data already symmetrically quantized to
/// signed int8. Dataset quantization is intentionally outside this call.
#[pyfunction]
#[pyo3(signature = (data, inv_norms, pairs_i, pairs_j, transformed=true))]
fn benchmark_quantized_angular_i8<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<i8>,
    inv_norms: numpy::PyReadonlyArray1<f32>,
    pairs_i: numpy::PyReadonlyArray1<i32>,
    pairs_j: numpy::PyReadonlyArray1<i32>,
    transformed: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let shape = data.shape();
    let n_points = shape[0];
    let dim = shape[1];
    let data = data.as_slice()?;
    let inv_norms = inv_norms.as_slice()?;
    let pairs_i = pairs_i.as_slice()?;
    let pairs_j = pairs_j.as_slice()?;

    if inv_norms.len() != n_points {
        return Err(PyValueError::new_err("inv_norms length must match data rows"));
    }
    if pairs_i.len() != pairs_j.len() {
        return Err(PyValueError::new_err("pairs_i and pairs_j must have the same length"));
    }

    let mut results = Vec::with_capacity(pairs_i.len());
    for (pair, (&i, &j)) in pairs_i.iter().zip(pairs_j).enumerate() {
        let i = usize::try_from(i).map_err(|_| PyValueError::new_err(format!("negative pair index at {pair}")))?;
        let j = usize::try_from(j).map_err(|_| PyValueError::new_err(format!("negative pair index at {pair}")))?;
        if i >= n_points || j >= n_points {
            return Err(PyValueError::new_err(format!("pair index out of range at {pair}")));
        }
        let x = &data[i * dim..(i + 1) * dim];
        let y = &data[j * dim..(j + 1) * dim];
        let value = if transformed {
            quantized_i8_alternative_dot(x, y, inv_norms[i], inv_norms[j])
        } else {
            -(quantized_i8_dot(x, y) as f32 * inv_norms[i] * inv_norms[j])
        };
        results.push(value);
    }

    Ok(PyArray1::from_vec_bound(py, results))
}

/// The pynndescent_rs Python module.
#[pymodule]
fn pynndescent_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNNDescent>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(simd_info, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_heap_push, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_candidate_building, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_distances, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_quantized_distances_u8, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_quantized_angular_i8, m)?)?;
    Ok(())
}

#[cfg(test)]
mod quantization_tests {
    use super::*;

    fn normalized_fixture(n: usize, dim: usize) -> Vec<f32> {
        let mut data: Vec<f32> = (0..n * dim)
            .map(|i| ((i * 37 + 11) as f32 * 0.071).sin() + ((i * 13) as f32 * 0.037).cos() * 0.3)
            .collect();
        normalize_rows_inplace(&mut data, dim);
        data
    }

    fn make_quantized(
        data: &[f32],
        n: usize,
        dim: usize,
        encoding: QuantizedI8Encoding,
        rerank_width: usize,
    ) -> QuantizedI8Index {
        let builder = NNDescentBuilder::new(data, n, dim)
            .n_neighbors(10)
            .n_trees(3)
            .n_search_trees(2)
            .n_iters(3)
            .random_seed(41);
        QuantizedI8Index::new(
            builder.build_with_distance(DirectNormalizedCosine, None),
            0,
            rerank_width,
            encoding,
            false,
        )
    }

    #[test]
    fn sq8p_storage_and_error_bounds_are_preserved() {
        let (n, dim) = (48, 5);
        let data = normalized_fixture(n, dim);

        let sq8p = make_quantized(&data, n, dim, QuantizedI8Encoding::PerVectorSymmetric, 0);
        let sq8p_storage = sq8p.storage_info();
        assert_eq!(sq8p_storage.1, n * dim);
        assert_eq!(sq8p_storage.2, n * std::mem::size_of::<f32>());
        for ((row, codes), &scale) in sq8p.index.data.chunks_exact(dim)
            .zip(sq8p.quantized_data.chunks_exact(dim)).zip(&sq8p.inv_norms) {
            for (&value, &code) in row.iter().zip(codes) {
                assert!((value - code as f32 * scale).abs() <= scale * 0.501 + 1e-7);
            }
        }
    }

    #[test]
    fn retained_quantized_modes_return_sorted_production_results() {
        let (n, dim) = (64, 7);
        let data = normalized_fixture(n, dim);
        let encodings = [
            QuantizedI8Encoding::PerVectorCosine,
            QuantizedI8Encoding::GlobalSymmetric,
            QuantizedI8Encoding::PerVectorSymmetric,
        ];
        let query = &data[9 * dim..10 * dim];
        for encoding in encodings {
            let index = make_quantized(&data, n, dim, encoding, 0);
            let default_result = index.query_one_quantized(query, 6, 0.1);
            let explicit_default = index.query_one_quantized_widths(query, 6, 0.1, 0, 0);
            assert_eq!(default_result, explicit_default, "encoding={encoding:?}");
            assert_eq!(default_result.0.len(), 6);
            assert_eq!(default_result.1.len(), 6);
            assert!(default_result.1.windows(2).all(|pair| pair[0] <= pair[1]));
        }
    }

    #[test]
    fn quantized_candidate_and_rerank_widths_are_independent() {
        let (n, dim) = (96, 9);
        let data = normalized_fixture(n, dim);
        let query = &data[13 * dim..14 * dim];
        let index = make_quantized(
            &data, n, dim, QuantizedI8Encoding::GlobalSymmetric, 0,
        );

        let narrow = index.query_one_quantized_widths(query, 6, 0.08, 6, 0);
        let wide = index.query_one_quantized_widths(query, 6, 0.08, 18, 0);
        let reranked = index.query_one_quantized_widths(query, 6, 0.08, 18, 12);
        assert_eq!(narrow.0.len(), 6);
        assert_eq!(wide.0.len(), 6);
        assert_eq!(reranked.0.len(), 6);
        assert!(reranked.1.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn quantized_width_validation_is_clear_and_legacy_zero_is_accepted() {
        assert!(validate_quantized_widths(10, 0, 5).is_ok());
        assert!(validate_quantized_widths(10, 9, 0).is_err());
        assert!(validate_quantized_widths(10, 20, 21).is_err());
        assert!(validate_quantized_widths(10, 20, 5).is_err());
        assert!(validate_quantized_widths(10, 20, 10).is_ok());
    }
}
