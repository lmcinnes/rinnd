//! Flat (compressed) random projection tree structure.
//!
//! The flat tree format is optimized for search operations, storing
//! all tree data in contiguous arrays.

use super::builder::dot_product;
use crate::distance::quantized::quantized_i8_dot;
use crate::rng::FastRng;

/// A flattened random projection tree for efficient search.
///
/// This matches PyNNDescent's FlatTree structure.
#[derive(Clone, Debug)]
pub struct FlatTree {
    /// Hyperplane vectors for each internal node, shape (n_nodes × dim)
    pub hyperplanes: Vec<f32>,
    /// Hyperplane offsets for each internal node
    pub offsets: Vec<f32>,
    /// Child pointers: children[node] = [left_child, right_child]
    /// Negative values indicate leaves: -children[node] gives the range
    pub children: Vec<[i32; 2]>,
    /// Point indices in leaf order
    pub indices: Vec<i32>,
    /// Dimension of data
    pub dim: usize,
    /// Number of nodes in the tree
    pub n_nodes: usize,
}

/// An opt-in signed-int8 routing representation of a [`FlatTree`].
///
/// Each hyperplane has an independent symmetric scale. Offsets remain FP32;
/// topology and leaf index storage are preserved exactly.
#[derive(Clone, Debug)]
pub struct QuantizedFlatTree {
    pub hyperplanes: Vec<i8>,
    pub scales: Vec<f32>,
    pub offsets: Vec<f32>,
    pub children: Vec<[i32; 2]>,
    pub indices: Vec<i32>,
    pub dim: usize,
    pub n_nodes: usize,
}

impl QuantizedFlatTree {
    fn quantize_hyperplanes(tree: &FlatTree) -> (Vec<i8>, Vec<f32>) {
        let mut hyperplanes = Vec::with_capacity(tree.hyperplanes.len());
        let mut scales = Vec::with_capacity(tree.n_nodes);
        for hyperplane in tree.hyperplanes.chunks_exact(tree.dim) {
            let max_abs = hyperplane
                .iter()
                .fold(0.0f32, |value, &x| value.max(x.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            scales.push(scale);
            hyperplanes.extend(hyperplane.iter().map(|&x| {
                if scale > 0.0 {
                    (x / scale).round().clamp(-127.0, 127.0) as i8
                } else {
                    0
                }
            }));
        }
        (hyperplanes, scales)
    }

    pub fn from_flat_tree(tree: &FlatTree) -> Self {
        let (hyperplanes, scales) = Self::quantize_hyperplanes(tree);
        Self {
            hyperplanes,
            scales,
            offsets: tree.offsets.clone(),
            children: tree.children.clone(),
            indices: tree.indices.clone(),
            dim: tree.dim,
            n_nodes: tree.n_nodes,
        }
    }

    /// Quantize routing parameters while taking ownership of the source tree's
    /// offsets, topology, and leaves. The FP32 hyperplanes are dropped.
    pub fn from_owned_flat_tree(tree: FlatTree) -> Self {
        let (hyperplanes, scales) = Self::quantize_hyperplanes(&tree);
        Self {
            hyperplanes,
            scales,
            offsets: tree.offsets,
            children: tree.children,
            indices: tree.indices,
            dim: tree.dim,
            n_nodes: tree.n_nodes,
        }
    }

    #[inline]
    pub fn margin(&self, node: usize, query: &[i8], query_scale: f32) -> f32 {
        let start = node * self.dim;
        quantized_i8_dot(&self.hyperplanes[start..start + self.dim], query) as f32
            * self.scales[node]
            * query_scale
            + self.offsets[node]
    }

    /// Quantized counterpart of [`FlatTree::search_leaves`], including the
    /// reconstructed absolute margin used for spill-frontier priority.
    pub fn search_leaves(
        &self,
        query: &[i8],
        query_scale: f32,
        rng: &mut FastRng,
        leaf_budget: usize,
        frontier: &mut Vec<(f32, usize)>,
        leaves: &mut Vec<(usize, usize)>,
    ) -> usize {
        frontier.clear();
        leaves.clear();
        if self.children.is_empty() || leaf_budget == 0 {
            return 0;
        }

        if leaf_budget == 1 {
            let mut node = 0usize;
            let mut evaluations = 0;
            while self.children[node][0] > 0 {
                let margin = self.margin(node, query, query_scale);
                evaluations += 1;
                let side = if margin.abs() < 1e-8 {
                    rng.next_bool() as usize
                } else {
                    (margin > 0.0) as usize
                };
                node = self.children[node][side] as usize;
            }
            leaves.push((
                (-self.children[node][0]) as usize,
                (-self.children[node][1]) as usize,
            ));
            return evaluations;
        }

        frontier.push((0.0, 0));
        let mut evaluations = 0;
        while leaves.len() < leaf_budget && !frontier.is_empty() {
            let best = frontier
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.0.total_cmp(&b.0))
                .map(|(index, _)| index)
                .unwrap();
            let (priority, node) = frontier.remove(best);
            if self.children[node][0] <= 0 {
                leaves.push((
                    (-self.children[node][0]) as usize,
                    (-self.children[node][1]) as usize,
                ));
                continue;
            }

            let margin = self.margin(node, query, query_scale);
            evaluations += 1;
            let side = if margin.abs() < 1e-8 {
                rng.next_bool() as usize
            } else {
                (margin > 0.0) as usize
            };
            frontier.push((priority, self.children[node][side] as usize));
            frontier.push((
                priority.max(margin.abs()),
                self.children[node][1 - side] as usize,
            ));
        }
        evaluations
    }

    pub fn code_bytes(&self) -> usize {
        self.hyperplanes.len()
    }
    pub fn scale_bytes(&self) -> usize {
        self.scales.len() * std::mem::size_of::<f32>()
    }
    pub fn offset_bytes(&self) -> usize {
        self.offsets.len() * std::mem::size_of::<f32>()
    }
}

impl FlatTree {
    /// Create a new empty flat tree.
    pub fn new(dim: usize) -> Self {
        Self {
            hyperplanes: Vec::new(),
            offsets: Vec::new(),
            children: Vec::new(),
            indices: Vec::new(),
            dim,
            n_nodes: 0,
        }
    }

    /// Search the tree to find the leaf containing a query point.
    ///
    /// Returns (start, end) indices into `self.indices` for the leaf.
    #[inline]
    pub fn search(&self, point: &[f32], rng: &mut FastRng) -> (usize, usize) {
        let mut node = 0usize;

        while self.children[node][0] > 0 {
            // Get hyperplane for this node
            let hp_start = node * self.dim;
            let hp_end = hp_start + self.dim;
            let hyperplane = &self.hyperplanes[hp_start..hp_end];
            let offset = self.offsets[node];

            // Compute margin (dot product with hyperplane + offset)
            let margin = dot_product(point, hyperplane) + offset;

            // Choose side (with random tie-breaking)
            let side = if margin.abs() < 1e-8 {
                rng.next_bool() as usize
            } else {
                (margin > 0.0) as usize
            };

            node = self.children[node][side] as usize;
        }

        // At a leaf node - extract bounds from children
        // children[node] = [-start, -end]
        let start = (-self.children[node][0]) as usize;
        let end = (-self.children[node][1]) as usize;

        (start, end)
    }

    /// Find up to `leaf_budget` leaves using bounded best-first spill traversal.
    ///
    /// The query-side child inherits its parent's priority. The alternate child
    /// is prioritized by the smallest hyperplane margin that must be crossed to
    /// reach it. A budget of one is equivalent to ordinary hard tree traversal.
    /// Caller-owned scratch buffers avoid per-query allocation.
    pub fn search_leaves(
        &self,
        point: &[f32],
        rng: &mut FastRng,
        leaf_budget: usize,
        frontier: &mut Vec<(f32, usize)>,
        leaves: &mut Vec<(usize, usize)>,
    ) -> usize {
        frontier.clear();
        leaves.clear();
        if self.children.is_empty() || leaf_budget == 0 {
            return 0;
        }

        if leaf_budget == 1 {
            let mut node = 0usize;
            let mut hyperplane_evaluations = 0;
            while self.children[node][0] > 0 {
                let hp_start = node * self.dim;
                let hyperplane = &self.hyperplanes[hp_start..hp_start + self.dim];
                let margin = dot_product(point, hyperplane) + self.offsets[node];
                hyperplane_evaluations += 1;
                let side = if margin.abs() < 1e-8 {
                    rng.next_bool() as usize
                } else {
                    (margin > 0.0) as usize
                };
                node = self.children[node][side] as usize;
            }
            leaves.push((
                (-self.children[node][0]) as usize,
                (-self.children[node][1]) as usize,
            ));
            return hyperplane_evaluations;
        }

        frontier.push((0.0, 0));
        let mut hyperplane_evaluations = 0;
        while leaves.len() < leaf_budget && !frontier.is_empty() {
            let best = frontier
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.0.total_cmp(&b.0))
                .map(|(index, _)| index)
                .unwrap();
            let (priority, node) = frontier.remove(best);

            if self.children[node][0] <= 0 {
                leaves.push((
                    (-self.children[node][0]) as usize,
                    (-self.children[node][1]) as usize,
                ));
                continue;
            }

            let hp_start = node * self.dim;
            let hyperplane = &self.hyperplanes[hp_start..hp_start + self.dim];
            let margin = dot_product(point, hyperplane) + self.offsets[node];
            hyperplane_evaluations += 1;
            let side = if margin.abs() < 1e-8 {
                rng.next_bool() as usize
            } else {
                (margin > 0.0) as usize
            };

            let primary = self.children[node][side] as usize;
            let alternate = self.children[node][1 - side] as usize;
            frontier.push((priority, primary));
            frontier.push((priority.max(margin.abs()), alternate));
        }

        hyperplane_evaluations
    }

    /// Search tree for angular data (using normalized hyperplane comparison).
    #[inline]
    pub fn search_angular(&self, point: &[f32], rng: &mut FastRng) -> (usize, usize) {
        // For angular trees, the search is the same - normalization happens during construction
        self.search(point, rng)
    }

    /// Get the leaf indices for a query.
    pub fn get_leaf_indices(&self, point: &[f32], rng: &mut FastRng) -> &[i32] {
        let (start, end) = self.search(point, rng);
        &self.indices[start..end]
    }

    /// Remap point IDs stored in leaves from an old ID space to a new one.
    ///
    /// `old_to_new[old_id]` must contain the corresponding new point ID.
    pub fn remap_indices(&mut self, old_to_new: &[usize]) {
        for idx in &mut self.indices {
            debug_assert!(*idx >= 0 && (*idx as usize) < old_to_new.len());
            *idx = old_to_new[*idx as usize] as i32;
        }
    }

    /// Get all leaf boundaries.
    ///
    /// Returns a vector of (start, end) pairs for each leaf.
    pub fn get_all_leaves(&self) -> Vec<(usize, usize)> {
        let mut leaves = Vec::new();
        self.collect_leaves(0, &mut leaves);
        leaves
    }

    fn collect_leaves(&self, node: usize, leaves: &mut Vec<(usize, usize)>) {
        if self.children[node][0] <= 0 {
            // Leaf node
            let start = (-self.children[node][0]) as usize;
            let end = (-self.children[node][1]) as usize;
            leaves.push((start, end));
        } else {
            // Internal node - recurse
            self.collect_leaves(self.children[node][0] as usize, leaves);
            self.collect_leaves(self.children[node][1] as usize, leaves);
        }
    }
}

/// Select which side of a hyperplane a point falls on.
///
/// Returns 0 for left, 1 for right.
#[inline]
pub fn select_side(hyperplane: &[f32], offset: f32, point: &[f32], rng: &mut FastRng) -> usize {
    let mut margin = offset;
    for i in 0..hyperplane.len() {
        margin += point[i] * hyperplane[i];
    }

    if margin.abs() < 1e-8 {
        rng.next_bool() as usize
    } else {
        (margin > 0.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::build_rp_tree;

    fn create_test_tree() -> FlatTree {
        // Create a simple tree with 3 nodes (1 internal + 2 leaves)
        // dim = 2
        let dim = 2;
        let mut tree = FlatTree::new(dim);

        // Root node (0): hyperplane [1, 0], offset = 0
        tree.hyperplanes.extend_from_slice(&[1.0, 0.0]);
        tree.offsets.push(0.0);
        tree.children.push([1, 2]); // left child = node 1, right child = node 2

        // Left leaf (node 1): points 0, 1
        tree.hyperplanes.extend_from_slice(&[0.0, 0.0]); // unused for leaf
        tree.offsets.push(0.0);
        tree.children.push([-0, -2]); // indices[0:2]

        // Right leaf (node 2): points 2, 3
        tree.hyperplanes.extend_from_slice(&[0.0, 0.0]); // unused for leaf
        tree.offsets.push(0.0);
        tree.children.push([-2, -4]); // indices[2:4]

        tree.indices = vec![0, 1, 2, 3];
        tree.n_nodes = 3;

        tree
    }

    #[test]
    fn test_search_left() {
        let tree = create_test_tree();
        let mut rng = FastRng::new(42);

        // Point on left side (x < 0)
        let point = vec![-1.0, 0.0];
        let (start, end) = tree.search(&point, &mut rng);

        assert_eq!(start, 0);
        assert_eq!(end, 2);
    }

    #[test]
    fn test_search_right() {
        let tree = create_test_tree();
        let mut rng = FastRng::new(42);

        // Point on right side (x > 0)
        let point = vec![1.0, 0.0];
        let (start, end) = tree.search(&point, &mut rng);

        assert_eq!(start, 2);
        assert_eq!(end, 4);
    }

    #[test]
    fn test_get_leaf_indices() {
        let tree = create_test_tree();
        let mut rng = FastRng::new(42);

        let point = vec![1.0, 0.0];
        let indices = tree.get_leaf_indices(&point, &mut rng);

        assert_eq!(indices, &[2, 3]);
    }

    #[test]
    fn test_get_all_leaves() {
        let tree = create_test_tree();
        let leaves = tree.get_all_leaves();

        assert_eq!(leaves.len(), 2);
        assert!(leaves.contains(&(0, 2)));
        assert!(leaves.contains(&(2, 4)));
    }

    #[test]
    fn test_remap_indices_preserves_leaf_ranges() {
        let mut tree = create_test_tree();
        let children = tree.children.clone();

        tree.remap_indices(&[2, 0, 3, 1]);

        assert_eq!(tree.indices, vec![2, 0, 3, 1]);
        assert_eq!(tree.children, children);
        assert_eq!(tree.get_all_leaves(), vec![(0, 2), (2, 4)]);
    }

    #[test]
    fn test_spill_search_leaf_budget() {
        let tree = create_test_tree();
        let mut rng = FastRng::new(42);
        let mut frontier = Vec::new();
        let mut leaves = Vec::new();

        let evaluations = tree.search_leaves(&[1.0, 0.0], &mut rng, 2, &mut frontier, &mut leaves);

        assert_eq!(evaluations, 1);
        assert_eq!(leaves, vec![(2, 4), (0, 2)]);
    }

    #[test]
    fn test_quantized_margin_reconstruction_sign_and_odd_dimension() {
        let mut tree = FlatTree::new(5);
        tree.hyperplanes = vec![0.25, -0.5, 0.75, -1.0, 0.125];
        tree.offsets = vec![0.2];
        tree.children = vec![[-0, -0]];
        tree.n_nodes = 1;
        let quantized = QuantizedFlatTree::from_flat_tree(&tree);
        let query = [64, -32, 16, -8, 4];
        let query_scale = 0.01;
        let expected = quantized_i8_dot(&quantized.hyperplanes, &query) as f32
            * quantized.scales[0]
            * query_scale
            + 0.2;
        assert_eq!(quantized.margin(0, &query, query_scale), expected);
        assert!(quantized.margin(0, &query, query_scale) > 0.0);
    }

    #[test]
    fn test_quantized_tree_preserves_topology_and_leaves() {
        let tree = create_test_tree();
        let quantized = QuantizedFlatTree::from_flat_tree(&tree);
        assert_eq!(quantized.children, tree.children);
        assert_eq!(quantized.indices, tree.indices);
        assert_eq!(quantized.dim, tree.dim);
        assert_eq!(quantized.n_nodes, tree.n_nodes);
    }

    #[test]
    fn test_owned_quantization_matches_borrowed_quantization() {
        let tree = create_test_tree();
        let borrowed = QuantizedFlatTree::from_flat_tree(&tree);
        let owned = QuantizedFlatTree::from_owned_flat_tree(tree);
        assert_eq!(owned.hyperplanes, borrowed.hyperplanes);
        assert_eq!(owned.scales, borrowed.scales);
        assert_eq!(owned.offsets, borrowed.offsets);
        assert_eq!(owned.children, borrowed.children);
        assert_eq!(owned.indices, borrowed.indices);
        assert_eq!(owned.dim, borrowed.dim);
        assert_eq!(owned.n_nodes, borrowed.n_nodes);
    }

    #[test]
    fn test_generated_angular_tree_quantized_routing_fidelity() {
        let (n, dim) = (512, 17);
        let mut data: Vec<f32> = (0..n * dim)
            .map(|i| {
                ((i * 37 + 11) as f32 * 0.017).sin() + 0.31 * ((i * 13 + 5) as f32 * 0.029).cos()
            })
            .collect();
        for row in data.chunks_exact_mut(dim) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            for value in row {
                *value /= norm;
            }
        }
        let mut build_rng = FastRng::new(73);
        let tree = build_rp_tree(&data, n, dim, 24, &mut build_rng, true, 200);
        let quantized = QuantizedFlatTree::from_flat_tree(&tree);
        let mut identical = 0usize;
        let mut overlap_sum = 0.0f32;
        for query_id in 0..128 {
            let query = &data[query_id * dim..(query_id + 1) * dim];
            let max_abs = query.iter().fold(0.0f32, |value, &x| value.max(x.abs()));
            let scale = max_abs / 127.0;
            let codes: Vec<i8> = query
                .iter()
                .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
                .collect();
            let mut fp_rng = FastRng::new(991);
            let mut q_rng = FastRng::new(991);
            let mut fp_frontier = Vec::new();
            let mut q_frontier = Vec::new();
            let mut fp_leaves = Vec::new();
            let mut q_leaves = Vec::new();
            tree.search_leaves(query, &mut fp_rng, 3, &mut fp_frontier, &mut fp_leaves);
            quantized.search_leaves(&codes, scale, &mut q_rng, 3, &mut q_frontier, &mut q_leaves);
            if fp_leaves == q_leaves {
                identical += 1;
            }
            let common = fp_leaves
                .iter()
                .filter(|leaf| q_leaves.contains(leaf))
                .count();
            overlap_sum += common as f32 / fp_leaves.len() as f32;
        }
        let identity = identical as f32 / 128.0;
        let overlap = overlap_sum / 128.0;
        eprintln!("quantized RP routing fidelity: identity={identity:.6}, overlap={overlap:.6}");
        assert!(identity >= 0.95, "identity={identity}");
        assert!(overlap >= 0.98, "overlap={overlap}");
    }
}
