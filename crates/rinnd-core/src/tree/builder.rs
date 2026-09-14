//! Random projection tree construction.
//!
//! This module implements the tree building algorithm matching PyNNDescent.

use super::flat_tree::FlatTree;
use crate::rng::FastRng;
use rayon::prelude::*;

/// Compute dot product of two slices, using AVX2+FMA if available.
#[inline]
pub(super) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(a, b) };
        }
    }

    // Scalar fallback
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let aa = _mm256_loadu_ps(a.as_ptr().add(idx));
        let bb = _mm256_loadu_ps(b.as_ptr().add(idx));
        sum = _mm256_fmadd_ps(aa, bb, sum);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut result = _mm_cvtss_f32(result);

    // Handle remainder
    let start = chunks * 8;
    for i in start..n {
        result += a[i] * b[i];
    }

    result
}

/// Build a random projection tree.
pub fn build_rp_tree(
    data: &[f32],
    n_points: usize,
    dim: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    angular: bool,
    max_depth: usize,
) -> FlatTree {
    let indices: Vec<i32> = (0..n_points as i32).collect();
    let mut builder = TreeBuilder::new(data, dim, leaf_size, angular, max_depth, true);
    builder.build(&indices, rng);
    builder.into_flat_tree()
}

/// Build only the leaf partitions of a random projection tree.
pub(crate) fn build_rp_leaf_tree(
    data: &[f32],
    n_points: usize,
    dim: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    angular: bool,
    max_depth: usize,
) -> Vec<Vec<i32>> {
    let indices: Vec<i32> = (0..n_points as i32).collect();
    let mut builder = TreeBuilder::new(data, dim, leaf_size, angular, max_depth, false);
    builder.build(&indices, rng);
    builder.into_leaves()
}

/// Build a forest of random projection trees in parallel.
pub fn build_rp_forest(
    data: &[f32],
    n_points: usize,
    dim: usize,
    n_trees: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    angular: bool,
    max_depth: usize,
) -> Vec<FlatTree> {
    let seeds: Vec<u64> = (0..n_trees).map(|_| rng.next_u64()).collect();
    seeds
        .into_par_iter()
        .map(|seed| {
            let mut tree_rng = FastRng::new(seed);
            build_rp_tree(
                data,
                n_points,
                dim,
                leaf_size,
                &mut tree_rng,
                angular,
                max_depth,
            )
        })
        .collect()
}

/// Build forest leaf partitions without retaining query-tree storage.
pub(crate) fn build_rp_leaf_forest(
    data: &[f32],
    n_points: usize,
    dim: usize,
    n_trees: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    angular: bool,
    max_depth: usize,
) -> Vec<Vec<i32>> {
    let seeds: Vec<u64> = (0..n_trees).map(|_| rng.next_u64()).collect();
    seeds
        .into_par_iter()
        .map(|seed| {
            let mut tree_rng = FastRng::new(seed);
            build_rp_leaf_tree(
                data,
                n_points,
                dim,
                leaf_size,
                &mut tree_rng,
                angular,
                max_depth,
            )
        })
        .flatten()
        .collect()
}

/// Extract leaf arrays from a retained forest.
pub fn rptree_leaf_array(forest: &[FlatTree]) -> Vec<Vec<i32>> {
    let mut result = Vec::new();
    for tree in forest {
        for (start, end) in tree.get_all_leaves() {
            result.push(tree.indices[start..end].to_vec());
        }
    }
    result
}

struct TreeBuilder<'a> {
    data: &'a [f32],
    dim: usize,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
    retain_tree: bool,
    hyperplanes: Vec<f32>,
    offsets: Vec<f32>,
    children: Vec<[i32; 2]>,
    leaf_indices: Vec<i32>,
    leaves: Vec<Vec<i32>>,
}

impl<'a> TreeBuilder<'a> {
    fn new(
        data: &'a [f32],
        dim: usize,
        leaf_size: usize,
        angular: bool,
        max_depth: usize,
        retain_tree: bool,
    ) -> Self {
        Self {
            data,
            dim,
            leaf_size,
            angular,
            max_depth,
            retain_tree,
            hyperplanes: Vec::new(),
            offsets: Vec::new(),
            children: Vec::new(),
            leaf_indices: Vec::new(),
            leaves: Vec::new(),
        }
    }

    fn build(&mut self, indices: &[i32], rng: &mut FastRng) {
        let mut indices = indices.to_vec();
        self.build_recursive(&mut indices, 0, rng);
    }

    fn store_leaf(&mut self, indices: &[i32]) -> i32 {
        if !self.retain_tree {
            self.leaves.push(indices.to_vec());
            return -1;
        }
        let node_id = self.children.len() as i32;
        let start = self.leaf_indices.len() as i32;
        self.leaf_indices.extend_from_slice(indices);
        let end = self.leaf_indices.len() as i32;
        self.hyperplanes
            .resize(self.hyperplanes.len() + self.dim, 0.0);
        self.offsets.push(0.0);
        self.children.push([-start, -end]);
        node_id
    }

    fn build_recursive(&mut self, indices: &mut [i32], depth: usize, rng: &mut FastRng) -> i32 {
        if indices.len() <= self.leaf_size || depth >= self.max_depth {
            return self.store_leaf(indices);
        }

        let (hyperplane, offset) = self.make_split(indices, rng);
        let split_pos = self.partition_inplace(indices, &hyperplane, offset, rng);
        if split_pos == 0 || split_pos == indices.len() {
            return self.store_leaf(indices);
        }

        let node_id = if self.retain_tree {
            let node_id = self.children.len() as i32;
            self.hyperplanes.extend_from_slice(&hyperplane);
            self.offsets.push(offset);
            self.children.push([0, 0]);
            node_id
        } else {
            -1
        };

        let (left_indices, right_indices) = indices.split_at_mut(split_pos);
        let left_child = self.build_recursive(left_indices, depth + 1, rng);
        let right_child = self.build_recursive(right_indices, depth + 1, rng);
        if self.retain_tree {
            self.children[node_id as usize] = [left_child, right_child];
        }
        node_id
    }

    fn make_split(&self, indices: &[i32], rng: &mut FastRng) -> (Vec<f32>, f32) {
        let idx1 = rng.next_index(indices.len());
        let mut idx2 = rng.next_index(indices.len());
        while idx2 == idx1 && indices.len() > 1 {
            idx2 = rng.next_index(indices.len());
        }
        let point1 =
            &self.data[indices[idx1] as usize * self.dim..(indices[idx1] as usize + 1) * self.dim];
        let point2 =
            &self.data[indices[idx2] as usize * self.dim..(indices[idx2] as usize + 1) * self.dim];
        if self.angular {
            self.make_angular_split(point1, point2)
        } else {
            self.make_euclidean_split(point1, point2)
        }
    }

    fn make_euclidean_split(&self, point1: &[f32], point2: &[f32]) -> (Vec<f32>, f32) {
        let mut hyperplane = vec![0.0; self.dim];
        let mut norm_sq = 0.0;
        let mut dot_nm = 0.0;
        for i in 0..self.dim {
            let value = point2[i] - point1[i];
            hyperplane[i] = value;
            norm_sq += value * value;
            dot_nm += value * (point1[i] + point2[i]) * 0.5;
        }
        let norm = norm_sq.sqrt();
        let offset = if norm > 1e-8 {
            let inv_norm = norm.recip();
            for value in &mut hyperplane {
                *value *= inv_norm;
            }
            -dot_nm * inv_norm
        } else {
            0.0
        };
        (hyperplane, offset)
    }

    fn make_angular_split(&self, point1: &[f32], point2: &[f32]) -> (Vec<f32>, f32) {
        let norm1 = dot_product(point1, point1).sqrt();
        let norm2 = dot_product(point2, point2).sqrt();
        if norm1 < 1e-8 || norm2 < 1e-8 {
            return (vec![0.0; self.dim], 0.0);
        }
        let mut hyperplane: Vec<f32> = (0..self.dim)
            .map(|i| point2[i] / norm2 - point1[i] / norm1)
            .collect();
        let norm = dot_product(&hyperplane, &hyperplane).sqrt();
        if norm > 1e-8 {
            for value in &mut hyperplane {
                *value /= norm;
            }
        }
        (hyperplane, 0.0)
    }

    fn partition_inplace(
        &self,
        indices: &mut [i32],
        hyperplane: &[f32],
        offset: f32,
        rng: &mut FastRng,
    ) -> usize {
        let mut left = 0;
        let mut right = indices.len();
        while left < right {
            let index = indices[left] as usize;
            let point = &self.data[index * self.dim..(index + 1) * self.dim];
            let margin = dot_product(point, hyperplane) + offset;
            let goes_left = if margin.abs() < 1e-8 {
                !rng.next_bool()
            } else {
                margin < 0.0
            };
            if goes_left {
                left += 1;
            } else {
                right -= 1;
                indices.swap(left, right);
            }
        }
        left
    }

    fn into_flat_tree(self) -> FlatTree {
        FlatTree {
            n_nodes: self.children.len(),
            hyperplanes: self.hyperplanes,
            offsets: self.offsets,
            children: self.children,
            indices: self.leaf_indices,
            dim: self.dim,
        }
    }

    fn into_leaves(self) -> Vec<Vec<i32>> {
        self.leaves
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_single_tree() {
        // Create simple 2D data
        let data: Vec<f32> = vec![
            0.0, 0.0, // point 0
            1.0, 0.0, // point 1
            0.0, 1.0, // point 2
            1.0, 1.0, // point 3
            -1.0, 0.0, // point 4
            0.0, -1.0, // point 5
        ];
        let n_points = 6;
        let dim = 2;
        let leaf_size = 2;
        let mut rng = FastRng::new(42);

        let tree = build_rp_tree(&data, n_points, dim, leaf_size, &mut rng, false, 10);

        // Tree should have all points
        assert_eq!(tree.indices.len(), n_points);

        // All indices should be present
        let mut indices: Vec<i32> = tree.indices.clone();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_build_forest() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let n_points = 50;
        let dim = 2;
        let n_trees = 3;
        let leaf_size = 10;
        let mut rng = FastRng::new(42);

        let forest = build_rp_forest(
            &data, n_points, dim, n_trees, leaf_size, &mut rng, false, 10,
        );

        assert_eq!(forest.len(), n_trees);

        for tree in &forest {
            assert_eq!(tree.indices.len(), n_points);
        }
    }

    #[test]
    fn test_leaf_array() {
        let data: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
        let n_points = 10;
        let dim = 2;
        let leaf_size = 3;
        let mut rng = FastRng::new(42);

        let tree = build_rp_tree(&data, n_points, dim, leaf_size, &mut rng, false, 10);
        let leaves = rptree_leaf_array(&[tree]);

        // All points should appear in exactly one leaf
        let mut all_indices: Vec<i32> = leaves.iter().flatten().copied().collect();
        all_indices.sort();
        let expected: Vec<i32> = (0..n_points as i32).collect();
        assert_eq!(all_indices, expected);
    }

    #[test]
    fn test_leaf_only_tree_matches_retained_tree_partitions() {
        let data: Vec<f32> = (0..800).map(|i| ((i as f32) * 0.17).sin()).collect();
        let mut retained_rng = FastRng::new(42);
        let mut leaf_only_rng = FastRng::new(42);

        let tree = build_rp_tree(&data, 100, 8, 10, &mut retained_rng, true, 20);
        let retained_leaves = rptree_leaf_array(&[tree]);
        let leaf_only_leaves = build_rp_leaf_tree(&data, 100, 8, 10, &mut leaf_only_rng, true, 20);

        assert_eq!(leaf_only_leaves, retained_leaves);
    }

    #[test]
    fn test_search_consistency() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let n_points = 4;
        let dim = 2;
        let leaf_size = 2;
        let mut rng = FastRng::new(42);

        let tree = build_rp_tree(&data, n_points, dim, leaf_size, &mut rng, false, 10);

        // Search for each data point - it should find itself in its leaf
        for i in 0..n_points {
            let point = &data[i * dim..(i + 1) * dim];
            let leaf_indices = tree.get_leaf_indices(point, &mut rng);

            // The point should be in its own leaf (though exact index might vary)
            assert!(!leaf_indices.is_empty());
        }
    }
}
