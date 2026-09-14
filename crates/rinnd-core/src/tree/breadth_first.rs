//! Experimental breadth-first RP forest construction for graph-only indexes.

use super::builder::dot_product;
use crate::rng::FastRng;
use rayon::prelude::*;

const NO_NODE: usize = usize::MAX;
const MAX_TREE_BATCH_WIDTH: usize = 8;

#[derive(Clone, Copy, Debug)]
struct FrontierNode {
    start: usize,
    end: usize,
    depth: usize,
    seed: u64,
}

#[derive(Debug)]
struct Split {
    hyperplane: Vec<f32>,
    offset: f32,
}

struct TreeState {
    permutation: Vec<i32>,
    assignments: Vec<usize>,
    frontier: Vec<FrontierNode>,
    splits: Vec<Option<Split>>,
    leaves: Vec<Vec<i32>>,
}

impl TreeState {
    fn new(n_points: usize, seed: u64) -> Self {
        Self {
            permutation: (0..n_points as i32).collect(),
            assignments: vec![0; n_points],
            frontier: vec![FrontierNode {
                start: 0,
                end: n_points,
                depth: 0,
                seed,
            }],
            splits: Vec::new(),
            leaves: Vec::new(),
        }
    }

    fn prepare_level(
        &mut self,
        data: &[f32],
        dim: usize,
        leaf_size: usize,
        angular: bool,
        max_depth: usize,
    ) -> bool {
        self.splits = self
            .frontier
            .iter()
            .map(|node| {
                if node.end - node.start <= leaf_size || node.depth >= max_depth {
                    None
                } else {
                    make_split(
                        data,
                        dim,
                        &self.permutation[node.start..node.end],
                        node.seed,
                        angular,
                    )
                }
            })
            .collect();
        self.splits.iter().any(Option::is_some)
    }

    fn finish_level(&mut self, tree_slot: usize, batch_width: usize, sides: &[u8]) {
        let mut next_frontier = Vec::with_capacity(self.frontier.len() * 2);
        for (node_index, node) in self.frontier.iter().copied().enumerate() {
            if self.splits[node_index].is_none() {
                self.leaves
                    .push(self.permutation[node.start..node.end].to_vec());
                for &point in &self.permutation[node.start..node.end] {
                    self.assignments[point as usize] = NO_NODE;
                }
                continue;
            }

            let mut left = node.start;
            let mut right = node.end;
            while left < right {
                let point = self.permutation[left] as usize;
                if sides[point * batch_width + tree_slot] == 0 {
                    left += 1;
                } else {
                    right -= 1;
                    self.permutation.swap(left, right);
                }
            }

            if left == node.start || left == node.end {
                self.leaves
                    .push(self.permutation[node.start..node.end].to_vec());
                for &point in &self.permutation[node.start..node.end] {
                    self.assignments[point as usize] = NO_NODE;
                }
                continue;
            }

            let left_index = next_frontier.len();
            next_frontier.push(FrontierNode {
                start: node.start,
                end: left,
                depth: node.depth + 1,
                seed: child_seed(node.seed, false),
            });
            let right_index = next_frontier.len();
            next_frontier.push(FrontierNode {
                start: left,
                end: node.end,
                depth: node.depth + 1,
                seed: child_seed(node.seed, true),
            });
            for &point in &self.permutation[node.start..left] {
                self.assignments[point as usize] = left_index;
            }
            for &point in &self.permutation[left..node.end] {
                self.assignments[point as usize] = right_index;
            }
        }
        self.frontier = next_frontier;
        self.splits.clear();
    }

    fn finish_terminal_level(&mut self) {
        for node in self.frontier.drain(..) {
            self.leaves
                .push(self.permutation[node.start..node.end].to_vec());
        }
        self.splits.clear();
    }
}

/// Build graph-initialization leaves by routing batches of trees breadth first.
pub(crate) fn build_breadth_first_leaf_forest(
    data: &[f32],
    n_points: usize,
    dim: usize,
    n_trees: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    angular: bool,
    max_depth: usize,
    tree_batch_width: usize,
) -> Vec<Vec<i32>> {
    assert_eq!(data.len(), n_points * dim);
    assert!(n_points <= i32::MAX as usize);
    assert!((1..=MAX_TREE_BATCH_WIDTH).contains(&tree_batch_width));

    let seeds: Vec<u64> = (0..n_trees).map(|_| rng.next_u64()).collect();
    let mut forest_leaves = Vec::new();
    for seed_batch in seeds.chunks(tree_batch_width) {
        let batch_width = seed_batch.len();
        let mut trees: Vec<TreeState> = seed_batch
            .iter()
            .copied()
            .map(|seed| TreeState::new(n_points, seed))
            .collect();

        loop {
            let mut any_splits = false;
            for tree in &mut trees {
                any_splits |= tree.prepare_level(data, dim, leaf_size, angular, max_depth);
            }
            if !any_splits {
                for tree in &mut trees {
                    tree.finish_terminal_level();
                }
                break;
            }

            let mut sides = vec![0u8; n_points * batch_width];
            sides
                .par_chunks_mut(batch_width)
                .enumerate()
                .for_each(|(point_index, point_sides)| {
                    let point = &data[point_index * dim..(point_index + 1) * dim];
                    let mut hyperplanes = [None; MAX_TREE_BATCH_WIDTH];
                    for (tree_slot, tree) in trees.iter().enumerate() {
                        let node_index = tree.assignments[point_index];
                        if node_index == NO_NODE {
                            continue;
                        }
                        if let Some(split) = &tree.splits[node_index] {
                            hyperplanes[tree_slot] = Some(split.hyperplane.as_slice());
                        }
                    }
                    let margins = batched_dot_products(point, &hyperplanes, batch_width);
                    for (tree_slot, tree) in trees.iter().enumerate() {
                        let node_index = tree.assignments[point_index];
                        if node_index == NO_NODE {
                            continue;
                        }
                        if let Some(split) = &tree.splits[node_index] {
                            let margin = margins[tree_slot] + split.offset;
                            point_sides[tree_slot] = if margin.abs() < 1e-8 {
                                tie_side(tree.frontier[node_index].seed, point_index)
                            } else {
                                (margin >= 0.0) as u8
                            };
                        }
                    }
                });

            trees
                .par_iter_mut()
                .enumerate()
                .for_each(|(tree_slot, tree)| tree.finish_level(tree_slot, batch_width, &sides));
        }

        for tree in trees {
            forest_leaves.extend(tree.leaves);
        }
    }
    forest_leaves
}

#[inline]
fn batched_dot_products(
    point: &[f32],
    hyperplanes: &[Option<&[f32]>; MAX_TREE_BATCH_WIDTH],
    width: usize,
) -> [f32; MAX_TREE_BATCH_WIDTH] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { batched_dot_products_avx2(point, hyperplanes, width) };
        }
    }

    let mut sums = [0.0; MAX_TREE_BATCH_WIDTH];
    for (dimension, &value) in point.iter().enumerate() {
        for slot in 0..width {
            if let Some(hyperplane) = hyperplanes[slot] {
                sums[slot] += value * hyperplane[dimension];
            }
        }
    }
    sums
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn batched_dot_products_avx2(
    point: &[f32],
    hyperplanes: &[Option<&[f32]>; MAX_TREE_BATCH_WIDTH],
    width: usize,
) -> [f32; MAX_TREE_BATCH_WIDTH] {
    use std::arch::x86_64::*;

    let chunks = point.len() / 8;
    let mut accumulators = [_mm256_setzero_ps(); MAX_TREE_BATCH_WIDTH];
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let values = _mm256_loadu_ps(point.as_ptr().add(offset));
        for slot in 0..width {
            if let Some(hyperplane) = hyperplanes[slot] {
                let plane = _mm256_loadu_ps(hyperplane.as_ptr().add(offset));
                accumulators[slot] = _mm256_fmadd_ps(values, plane, accumulators[slot]);
            }
        }
    }

    let mut sums = [0.0; MAX_TREE_BATCH_WIDTH];
    for slot in 0..width {
        if hyperplanes[slot].is_some() {
            let high = _mm256_extractf128_ps(accumulators[slot], 1);
            let low = _mm256_castps256_ps128(accumulators[slot]);
            let pair = _mm_add_ps(high, low);
            let shuffled = _mm_movehdup_ps(pair);
            let pair = _mm_add_ps(pair, shuffled);
            let shuffled = _mm_movehl_ps(pair, pair);
            sums[slot] = _mm_cvtss_f32(_mm_add_ss(pair, shuffled));
        }
    }
    for dimension in chunks * 8..point.len() {
        for slot in 0..width {
            if let Some(hyperplane) = hyperplanes[slot] {
                sums[slot] += point[dimension] * hyperplane[dimension];
            }
        }
    }
    sums
}

fn make_split(
    data: &[f32],
    dim: usize,
    indices: &[i32],
    seed: u64,
    angular: bool,
) -> Option<Split> {
    let mut rng = FastRng::new(seed);
    let first = rng.next_index(indices.len());
    let mut second = rng.next_index(indices.len());
    while second == first && indices.len() > 1 {
        second = rng.next_index(indices.len());
    }
    let first_index = indices[first] as usize;
    let second_index = indices[second] as usize;
    let point1 = &data[first_index * dim..(first_index + 1) * dim];
    let point2 = &data[second_index * dim..(second_index + 1) * dim];
    if angular {
        make_angular_split(point1, point2)
    } else {
        make_euclidean_split(point1, point2)
    }
}

fn make_angular_split(point1: &[f32], point2: &[f32]) -> Option<Split> {
    let norm1 = dot_product(point1, point1).sqrt();
    let norm2 = dot_product(point2, point2).sqrt();
    if norm1 < 1e-8 || norm2 < 1e-8 {
        return None;
    }
    let mut hyperplane: Vec<f32> = point1
        .iter()
        .zip(point2)
        .map(|(&first, &second)| second / norm2 - first / norm1)
        .collect();
    let norm = dot_product(&hyperplane, &hyperplane).sqrt();
    if norm < 1e-8 {
        return None;
    }
    for value in &mut hyperplane {
        *value /= norm;
    }
    Some(Split {
        hyperplane,
        offset: 0.0,
    })
}

fn make_euclidean_split(point1: &[f32], point2: &[f32]) -> Option<Split> {
    let mut hyperplane = Vec::with_capacity(point1.len());
    let mut norm_squared = 0.0;
    let mut midpoint_dot = 0.0;
    for (&first, &second) in point1.iter().zip(point2) {
        let value = second - first;
        hyperplane.push(value);
        norm_squared += value * value;
        midpoint_dot += value * (first + second) * 0.5;
    }
    let norm = norm_squared.sqrt();
    if norm < 1e-8 {
        return None;
    }
    let inverse_norm = norm.recip();
    for value in &mut hyperplane {
        *value *= inverse_norm;
    }
    Some(Split {
        hyperplane,
        offset: -midpoint_dot * inverse_norm,
    })
}

#[inline]
fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

#[inline]
fn child_seed(seed: u64, right: bool) -> u64 {
    mix64(
        seed ^ if right {
            0xd1b54a32d192ed03
        } else {
            0x94d049bb133111eb
        },
    )
}

#[inline]
fn tie_side(seed: u64, point_index: usize) -> u8 {
    (mix64(seed ^ point_index as u64) & 1) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPoolBuilder;

    fn assert_point_multiplicity(leaves: &[Vec<i32>], n_points: usize, n_trees: usize) {
        let mut counts = vec![0; n_points];
        for point in leaves.iter().flatten() {
            counts[*point as usize] += 1;
        }
        assert_eq!(counts, vec![n_trees; n_points]);
    }

    fn data(n_points: usize, dim: usize) -> Vec<f32> {
        (0..n_points * dim)
            .map(|index| ((index as f32) * 0.017).sin())
            .collect()
    }

    fn build_with_threads(threads: usize, batch_width: usize) -> Vec<Vec<i32>> {
        let data = data(128, 16);
        ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                build_breadth_first_leaf_forest(
                    &data,
                    128,
                    16,
                    4,
                    12,
                    &mut FastRng::new(42),
                    true,
                    20,
                    batch_width,
                )
            })
    }

    #[test]
    fn every_tree_contains_each_point_once() {
        let leaves = build_with_threads(4, 4);
        assert!(leaves
            .iter()
            .all(|leaf| !leaf.is_empty() && leaf.len() <= 12));
        assert_point_multiplicity(&leaves, 128, 4);
    }

    #[test]
    fn output_is_independent_of_threads_and_batch_width() {
        let expected = build_with_threads(1, 1);
        assert_eq!(build_with_threads(4, 1), expected);
        assert_eq!(build_with_threads(4, 2), expected);
        assert_eq!(build_with_threads(4, 4), expected);
    }

    #[test]
    fn batched_dot_matches_individual_dot_products() {
        let point: Vec<f32> = (0..37).map(|index| ((index as f32) * 0.13).sin()).collect();
        let planes: Vec<Vec<f32>> = (0..4)
            .map(|plane| {
                (0..37)
                    .map(|index| (((plane * 37 + index) as f32) * 0.07).cos())
                    .collect()
            })
            .collect();
        let mut references = [None; MAX_TREE_BATCH_WIDTH];
        for (slot, plane) in planes.iter().enumerate() {
            references[slot] = Some(plane.as_slice());
        }

        let batched = batched_dot_products(&point, &references, planes.len());
        for (slot, plane) in planes.iter().enumerate() {
            let expected = dot_product(&point, plane);
            assert!((batched[slot] - expected).abs() <= 1e-5);
        }
    }

    #[test]
    fn handles_euclidean_and_degenerate_data() {
        for angular in [false, true] {
            let data = vec![0.0; 64 * 9];
            let mut rng = FastRng::new(23);
            let leaves =
                build_breadth_first_leaf_forest(&data, 64, 9, 3, 8, &mut rng, angular, 10, 3);
            assert_point_multiplicity(&leaves, 64, 3);
        }
    }

    #[test]
    fn depth_limit_emits_oversized_terminal_leaves() {
        let data: Vec<f32> = (0..128 * 7)
            .map(|index| ((index as f32) * 0.19).sin())
            .collect();
        let mut rng = FastRng::new(29);
        let leaves = build_breadth_first_leaf_forest(&data, 128, 7, 2, 4, &mut rng, true, 1, 2);

        assert_point_multiplicity(&leaves, 128, 2);
        assert!(leaves.iter().any(|leaf| leaf.len() > 4));
    }
}
