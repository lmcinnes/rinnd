//! RP-tree splits on 8-bit codes, for the forest of a progressive build.
//!
//! A split is chosen from two points' rows as for float trees, then moved into
//! code space: with `x ~ offset + scale * c`, the margin `<h, x> + b` is
//! `<h * scale, c> + (<h, offset> + b)`, so a point's side comes from the int8
//! weights `w ~ h * scale / step` against its code row, an integer dot product
//! (one code row against four trees' weights with AVX-512 VNNI). Code rows and
//! weights are a quarter of the float rows and hyperplanes, the memory traffic
//! that bounds forest construction on wide data. Only the leaves are kept.

use super::breadth_first::{attempt_seed, choose_split, make_split, tie_side, Split, MAX_TREE_BATCH_WIDTH, SPLIT_ATTEMPTS};
use super::hybrid_forest::SplitSpace;
use crate::distance::affine::{quantize_i8, AffineCodes};
use crate::distance::int_dot::{dot4_u8, dot_u8, dot_u8_weights4, AlignedBytes, Kernel};
use crate::rng::FastRng;

/// The points of a forest as 8-bit codes (splits are chosen from `data`).
pub(crate) struct CodeSpace<'a> {
    pub data: &'a [f32],
    pub dim: usize,
    pub angular: bool,
    pub codes: &'a AffineCodes,
}

/// A split in code space: a point is on the right when `<w, c> >= threshold`.
pub(crate) struct CodeSplit {
    weights: AlignedBytes,
    threshold: f64,
    tie_seed: u64,
}

impl CodeSpace<'_> {
    fn kernel(&self) -> Kernel {
        self.codes.kernel()
    }

    /// `split` in code space.
    fn code_split(&self, split: &Split) -> CodeSplit {
        let (offsets, scales) = (self.codes.offsets(), self.codes.scales());
        let mut weights = AlignedBytes::zeroed(self.codes.stride());
        let u: Vec<f32> = split.hyperplane.iter().zip(scales).map(|(&h, &s)| h * s).collect();
        let largest = u.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if largest == 0.0 {
            // A zero hyperplane: every point ties (and goes to a random side).
            return CodeSplit { weights, threshold: 0.0, tie_seed: split.tie_seed };
        }
        let inverse = 127.0 / largest;
        quantize_i8(&u, inverse, &mut weights.as_mut_i8_slice()[..self.dim]);
        let constant: f64 =
            split.hyperplane.iter().zip(offsets).map(|(&h, &o)| f64::from(h) * f64::from(o)).sum::<f64>()
                + f64::from(split.offset);
        CodeSplit { weights, threshold: -constant * f64::from(inverse), tie_seed: split.tie_seed }
    }

    #[inline(always)]
    fn weights<'s>(split: &'s CodeSplit) -> &'s [i8] {
        split.weights.as_i8_slice()
    }

    #[inline(always)]
    fn side(split: &CodeSplit, dot: i32, point: usize) -> u8 {
        let margin = f64::from(dot) - split.threshold;
        if margin == 0.0 {
            tie_side(split.tie_seed, point)
        } else {
            (margin > 0.0) as u8
        }
    }

    fn point_side(&self, split: &CodeSplit, point: i32) -> u8 {
        let dot = dot_u8(self.kernel(), self.codes.row(point), Self::weights(split));
        Self::side(split, dot, point as usize)
    }
}

impl SplitSpace for CodeSpace<'_> {
    type Split = CodeSplit;
    type Local = ();

    fn choose_split(&self, ids: &[i32], seed: u64) -> CodeSplit {
        self.code_split(&choose_split(self.data, self.dim, ids, seed, self.angular))
    }

    fn repair_split(&self, ids: &mut [i32], mut swap: impl FnMut(usize, usize), seed: u64) -> (CodeSplit, usize) {
        for attempt in SPLIT_ATTEMPTS + 1..2 * SPLIT_ATTEMPTS {
            let Some(split) = make_split(self.data, self.dim, ids, attempt_seed(seed, attempt), self.angular) else {
                continue;
            };
            let split = self.code_split(&split);
            let (mut left, mut right) = (0, ids.len());
            while left < right {
                if self.point_side(&split, ids[left]) == 0 {
                    left += 1;
                } else {
                    right -= 1;
                    ids.swap(left, right);
                    swap(left, right);
                }
            }
            if left > 0 && left < ids.len() {
                return (split, left);
            }
        }
        let mut rng = FastRng::new(attempt_seed(seed, 2 * SPLIT_ATTEMPTS));
        for position in (1..ids.len()).rev() {
            let other = rng.next_index(position + 1);
            ids.swap(position, other);
            swap(position, other);
        }
        let zero = Split { hyperplane: vec![0.0; self.dim], offset: 0.0, tie_seed: seed };
        (self.code_split(&zero), ids.len() / 2)
    }

    fn batch_sides(&self, point: usize, splits: &[Option<&CodeSplit>; MAX_TREE_BATCH_WIDTH], width: usize, sides: &mut [u8]) {
        let row = self.codes.row(point as i32);
        let mut slots = [0usize; MAX_TREE_BATCH_WIDTH];
        let mut count = 0;
        for (slot, split) in splits[..width].iter().enumerate() {
            if split.is_some() {
                slots[count] = slot;
                count += 1;
            }
        }
        for group in slots[..count].chunks(4) {
            let mut weights: [&[i8]; 4] = [&[]; 4];
            for (position, &slot) in group.iter().enumerate() {
                weights[position] = Self::weights(splits[slot].unwrap());
            }
            for position in group.len()..4 {
                weights[position] = weights[0];
            }
            let dots = dot_u8_weights4(self.kernel(), row, weights, group.len());
            for (position, &slot) in group.iter().enumerate() {
                sides[slot] = Self::side(splits[slot].unwrap(), dots[position], point);
            }
        }
    }

    fn gather(&self, ids: &[i32]) -> ((), Vec<u32>) {
        ((), ids.iter().map(|&id| id as u32).collect())
    }

    fn node_sides(&self, _local: &mut (), _rows: &[u32], ids: &[i32], split: &CodeSplit, sides: &mut Vec<u8>) {
        let weights = Self::weights(split);
        sides.clear();
        let mut groups = ids.chunks_exact(4);
        for group in &mut groups {
            let rows = [0, 1, 2, 3].map(|position| self.codes.row(group[position]));
            let dots = dot4_u8(self.kernel(), rows, weights);
            for position in 0..4 {
                sides.push(Self::side(split, dots[position], group[position] as usize));
            }
        }
        for &id in groups.remainder() {
            sides.push(self.point_side(split, id));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::hybrid_forest::{build_space_forest, FloatSpace};
    use super::*;
    use crate::distance::affine::AffineKind;

    fn clustered(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = FastRng::new(seed);
        let centres: Vec<f32> = (0..20 * dim).map(|_| rng.next_float() * 4.0 - 2.0).collect();
        (0..n)
            .flat_map(|_| {
                let centre = rng.next_index(20);
                (0..dim).map(|t| centres[centre * dim + t] + rng.next_float() - 0.5).collect::<Vec<_>>()
            })
            .collect()
    }

    /// Each point's `k` nearest other points (squared euclidean).
    fn nearest(data: &[f32], dim: usize, k: usize) -> Vec<Vec<i32>> {
        let n = data.len() / dim;
        (0..n)
            .map(|p| {
                let x = &data[p * dim..(p + 1) * dim];
                let mut d: Vec<(f32, i32)> = (0..n)
                    .filter(|&q| q != p)
                    .map(|q| (x.iter().zip(&data[q * dim..(q + 1) * dim]).map(|(a, b)| (a - b) * (a - b)).sum(), q as i32))
                    .collect();
                d.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
                d[..k].iter().map(|&(_, q)| q).collect()
            })
            .collect()
    }

    /// Mean fraction of each point's true neighbours that share a leaf with it.
    fn coverage(leaves: &[Vec<i32>], truth: &[Vec<i32>]) -> f64 {
        let mut mates = vec![std::collections::HashSet::new(); truth.len()];
        for leaf in leaves {
            for &a in leaf {
                mates[a as usize].extend(leaf.iter().copied());
            }
        }
        let found: usize = truth.iter().zip(&mates).map(|(t, m)| t.iter().filter(|q| m.contains(q)).count()).sum();
        found as f64 / truth.iter().map(Vec::len).sum::<usize>() as f64
    }

    #[test]
    fn code_forest_leaves_find_neighbours_like_float_leaves() {
        let (n, dim, trees, leaf_size) = (4000, 96, 6, 40);
        let data = clustered(n, dim, 1);
        let truth = nearest(&data, dim, 10);
        let codes = AffineCodes::encode_for_build(&data, dim, AffineKind::SquaredEuclidean);
        for angular in [false, true] {
            let code_space = CodeSpace { data: &data, dim, angular, codes: &codes };
            let float_space = FloatSpace { data: &data, dim, angular };
            // A small local budget exercises both the breadth-first and the local phase.
            let leaves = build_space_forest(&code_space, n, trees, leaf_size, &mut FastRng::new(5), 200, 4, 300);
            let float_leaves = build_space_forest(&float_space, n, trees, leaf_size, &mut FastRng::new(5), 200, 4, 300);
            let mut counts = vec![0; n];
            for leaf in &leaves {
                assert!(!leaf.is_empty() && leaf.len() <= leaf_size);
                leaf.iter().for_each(|&point| counts[point as usize] += 1);
            }
            assert!(counts.iter().all(|&count| count == trees));
            // Leaves hold as many true neighbours as float leaves do.
            let (code_coverage, float_coverage) = (coverage(&leaves, &truth), coverage(&float_leaves, &truth));
            assert!(code_coverage > 0.95 * float_coverage, "angular={angular}: {code_coverage} vs {float_coverage}");
        }
    }

    #[test]
    fn code_splits_handle_duplicate_and_zero_points() {
        let (n, dim) = (1500, 64);
        let base = clustered(30, dim, 2);
        let mut data: Vec<f32> = (0..n).flat_map(|i| base[(i % 30) * dim..(i % 30 + 1) * dim].to_vec()).collect();
        data[..300 * dim].fill(0.0);
        let codes = AffineCodes::encode_for_build(&data, dim, AffineKind::SquaredEuclidean);
        for angular in [false, true] {
            let space = CodeSpace { data: &data, dim, angular, codes: &codes };
            let leaves = build_space_forest(&space, n, 3, 20, &mut FastRng::new(9), 200, 6, 200);
            assert!(leaves.iter().all(|leaf| !leaf.is_empty() && leaf.len() <= 20), "angular={angular}");
            assert_eq!(leaves.iter().map(Vec::len).sum::<usize>(), 3 * n);
        }
    }
}
