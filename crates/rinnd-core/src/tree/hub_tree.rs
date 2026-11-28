//! Graph-informed ("hub") search tree, after pynndescent's `hub_tree`.
//!
//! A search starts from the points in its query's tree leaf, so a search tree
//! should cut the data where the k-NN graph is sparse. Each node splits on the
//! hyperplane between two of its three highest in-degree points ("hubs"),
//! taking the most balanced of the three hub pairs; when none puts at least
//! [`MIN_SPLIT_BALANCE`] of the points on its smaller side, the most balanced
//! direction is moved to the median margin. Partitions are stable, so leaves
//! (and the vertex order they give) keep the order of the points within a node.
//! Large nodes compute margins in parallel, and sibling subtrees are built in
//! parallel; the tree does not depend on the number of threads.

use super::breadth_first::{batched_dot_products, MAX_TREE_BATCH_WIDTH};
use super::builder::dot_product;
use super::flat_tree::FlatTree;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Smallest accepted fraction of a node's points on the smaller side.
pub const MIN_SPLIT_BALANCE: f32 = 0.1;

/// Margins at or within this of zero are ties, alternated by rank in the node.
const EPS: f32 = 1e-8;

/// Hubs per node: all pairs of them are candidate splits.
const HUBS: usize = 3;
const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

/// Nodes with more points compute margins in parallel and split their children
/// in parallel.
const PARALLEL_MARGINS: usize = 1 << 15;
const PARALLEL_CHILDREN: usize = 1 << 11;

enum Built {
    Leaf { start: usize, end: usize },
    Split { hyperplane: Vec<f32>, offset: f32, left: Box<Built>, right: Box<Built> },
}

struct Builder<'a> {
    data: &'a [f32],
    dim: usize,
    degrees: Vec<u32>,
    leaf_size: usize,
    max_depth: usize,
    angular: bool,
}

/// Build the hub tree of `n_points` rows of `data` from their k-NN graph
/// (`neighbors`, `n_points` rows, negative entries absent).
pub fn build_hub_tree(
    data: &[f32],
    n_points: usize,
    dim: usize,
    neighbors: &[i32],
    leaf_size: usize,
    max_depth: usize,
    angular: bool,
) -> FlatTree {
    assert_eq!(data.len(), n_points * dim);
    assert!(n_points > 0 && n_points <= i32::MAX as usize);
    let counts: Vec<AtomicU32> = (0..n_points).map(|_| AtomicU32::new(0)).collect();
    neighbors.par_chunks(1 << 16).for_each(|chunk| {
        for &neighbor in chunk {
            if neighbor >= 0 && (neighbor as usize) < n_points {
                counts[neighbor as usize].fetch_add(1, Ordering::Relaxed);
            }
        }
    });
    let builder = Builder {
        data,
        dim,
        degrees: counts.into_iter().map(AtomicU32::into_inner).collect(),
        leaf_size: leaf_size.max(1),
        max_depth,
        angular,
    };
    let mut ids: Vec<i32> = (0..n_points as i32).collect();
    let root = builder.build(&mut ids, 0, 0);

    let mut tree = FlatTree {
        hyperplanes: Vec::new(),
        offsets: Vec::new(),
        children: Vec::new(),
        indices: Vec::new(),
        dim,
        n_nodes: 0,
    };
    flatten(root, &mut tree);
    tree.n_nodes = tree.children.len();
    tree.indices = ids;
    tree
}

/// Append `node` and its subtree in pre-order; returns its node number.
fn flatten(node: Built, tree: &mut FlatTree) -> i32 {
    let id = tree.children.len();
    tree.children.push([0, 0]);
    tree.offsets.push(0.0);
    tree.hyperplanes.resize((id + 1) * tree.dim, 0.0);
    match node {
        Built::Leaf { start, end } => tree.children[id] = [-(start as i32), -(end as i32)],
        Built::Split { hyperplane, offset, left, right } => {
            tree.hyperplanes[id * tree.dim..(id + 1) * tree.dim].copy_from_slice(&hyperplane);
            tree.offsets[id] = offset;
            let left = flatten(*left, tree);
            let right = flatten(*right, tree);
            tree.children[id] = [left, right];
        }
    }
    id as i32
}

/// The side of the point of rank `rank` in its node (1 = right, the side of
/// positive margins).
#[inline(always)]
fn side(margin: f32, rank: usize) -> u8 {
    if margin > EPS {
        1
    } else if margin < -EPS {
        0
    } else {
        (rank % 2) as u8
    }
}

impl Builder<'_> {
    fn row(&self, point: i32) -> &[f32] {
        &self.data[point as usize * self.dim..(point as usize + 1) * self.dim]
    }

    /// The node's (up to) three highest in-degree points, ties to lower ids.
    fn hubs(&self, ids: &[i32]) -> Vec<i32> {
        let mut top: Vec<(u32, i32)> = Vec::with_capacity(HUBS + 1);
        for &id in ids {
            let key = (self.degrees[id as usize], id);
            let better = |other: &(u32, i32)| key.0 > other.0 || (key.0 == other.0 && key.1 < other.1);
            if top.len() < HUBS || better(top.last().unwrap()) {
                let position = top.iter().position(|other| better(other)).unwrap_or(top.len());
                top.insert(position, key);
                top.truncate(HUBS);
            }
        }
        top.into_iter().map(|(_, id)| id).collect()
    }

    /// The hyperplane between points `a` and `b` (positive margins nearer `b`),
    /// unit length; `None` when they coincide.
    fn plane(&self, a: i32, b: i32) -> Option<(Vec<f32>, f32)> {
        let (a, b) = (self.row(a), self.row(b));
        let mut hyperplane: Vec<f32>;
        let mut offset = 0.0f32;
        if self.angular {
            let norm = |x: &[f32]| {
                let n = dot_product(x, x).sqrt();
                if n < EPS {
                    1.0
                } else {
                    n
                }
            };
            let (na, nb) = (norm(a), norm(b));
            hyperplane = a.iter().zip(b).map(|(&x, &y)| y / nb - x / na).collect();
        } else {
            hyperplane = a.iter().zip(b).map(|(&x, &y)| y - x).collect();
            offset = -a.iter().zip(b).zip(&hyperplane).map(|((&x, &y), &h)| h * (x + y) * 0.5).sum::<f32>();
        }
        let length = dot_product(&hyperplane, &hyperplane).sqrt();
        if !(length >= EPS) {
            return None;
        }
        hyperplane.iter_mut().for_each(|h| *h /= length);
        Some((hyperplane, offset / length))
    }

    /// Margins of the points `ids` against `planes` (one row per point).
    fn margins(&self, ids: &[i32], planes: &[(Vec<f32>, f32)]) -> Vec<[f32; HUBS]> {
        let mut hyperplanes: [Option<&[f32]>; MAX_TREE_BATCH_WIDTH] = [None; MAX_TREE_BATCH_WIDTH];
        for (slot, (hyperplane, _)) in planes.iter().enumerate() {
            hyperplanes[slot] = Some(hyperplane);
        }
        let one = |&id: &i32| {
            let dots = batched_dot_products(self.row(id), &hyperplanes, planes.len());
            let mut out = [0.0; HUBS];
            for (slot, (_, offset)) in planes.iter().enumerate() {
                out[slot] = dots[slot] + offset;
            }
            out
        };
        if ids.len() >= PARALLEL_MARGINS {
            ids.par_iter().with_min_len(1024).map(one).collect()
        } else {
            ids.iter().map(one).collect()
        }
    }

    /// The split of a node: its hyperplane, offset and each point's side, or
    /// `None` to make it a leaf.
    fn split(&self, ids: &[i32]) -> Option<(Vec<f32>, f32, Vec<u8>)> {
        let hubs = self.hubs(ids);
        let mut planes = Vec::with_capacity(PAIRS.len());
        for &(i, j) in &PAIRS {
            if j < hubs.len() {
                if let Some(plane) = self.plane(hubs[i], hubs[j]) {
                    planes.push(plane);
                }
            }
        }
        if planes.is_empty() {
            return None;
        }
        let n = ids.len();
        let margins = self.margins(ids, &planes);
        let smaller = |sides: &[u8]| {
            let right = sides.iter().map(|&s| s as usize).sum::<usize>();
            right.min(n - right)
        };
        let sides_of = |candidate: usize, shift: f32| -> Vec<u8> {
            margins.iter().enumerate().map(|(rank, m)| side(m[candidate] - shift, rank)).collect()
        };
        // The most balanced candidate (the first on ties).
        let (mut best, mut best_sides, mut best_small) = (0, Vec::new(), 0);
        for candidate in 0..planes.len() {
            let sides = sides_of(candidate, 0.0);
            let small = smaller(&sides);
            if candidate == 0 || small > best_small {
                (best, best_sides, best_small) = (candidate, sides, small);
            }
        }
        let accept = |small: usize| small > 0 && small as f32 >= MIN_SPLIT_BALANCE * n as f32;
        let (hyperplane, offset) = planes.swap_remove(best);
        if accept(best_small) {
            return Some((hyperplane, offset, best_sides));
        }
        // Move the most balanced direction to the median margin.
        let mut values: Vec<f32> = margins.iter().map(|m| m[best]).collect();
        let middle = n / 2;
        let (_, &mut upper, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
        let shift = if n % 2 == 0 {
            let lower = values[..middle].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            0.5 * (lower + upper)
        } else {
            upper
        };
        let sides = sides_of(best, shift);
        accept(smaller(&sides)).then(|| (hyperplane, offset - shift, sides))
    }

    /// Build the subtree of the points `ids` (at positions `base..` of the
    /// final order), partitioning them in place.
    fn build(&self, ids: &mut [i32], base: usize, depth: usize) -> Built {
        let n = ids.len();
        let leaf = Built::Leaf { start: base, end: base + n };
        if n <= self.leaf_size || depth >= self.max_depth {
            return leaf;
        }
        let Some((hyperplane, offset, sides)) = self.split(ids) else {
            return leaf;
        };
        let mut right: Vec<i32> = Vec::with_capacity(n);
        let mut n_left = 0;
        for position in 0..n {
            if sides[position] == 0 {
                ids[n_left] = ids[position];
                n_left += 1;
            } else {
                right.push(ids[position]);
            }
        }
        ids[n_left..].copy_from_slice(&right);
        drop((right, sides));
        let (left_ids, right_ids) = ids.split_at_mut(n_left);
        let (left, right) = if n >= PARALLEL_CHILDREN {
            rayon::join(
                || self.build(left_ids, base, depth + 1),
                || self.build(right_ids, base + n_left, depth + 1),
            )
        } else {
            (self.build(left_ids, base, depth + 1), self.build(right_ids, base + n_left, depth + 1))
        };
        Built::Split { hyperplane, offset, left: Box::new(left), right: Box::new(right) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::FastRng;

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

    fn knn(data: &[f32], dim: usize, k: usize) -> Vec<i32> {
        let n = data.len() / dim;
        let row = |p: usize| &data[p * dim..(p + 1) * dim];
        (0..n)
            .flat_map(|p| {
                let mut d: Vec<(f32, i32)> = (0..n)
                    .filter(|&q| q != p)
                    .map(|q| (row(p).iter().zip(row(q)).map(|(a, b)| (a - b) * (a - b)).sum(), q as i32))
                    .collect();
                d.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
                d[..k].sort_by(|a, b| a.0.total_cmp(&b.0));
                d[..k].iter().map(|&(_, q)| q).collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn hub_tree_routes_points_to_their_leaves_and_keeps_neighbours_together() {
        let (n, dim, k, leaf_size) = (3000, 24, 10, 40);
        let data = clustered(n, dim, 1);
        let neighbors = knn(&data, dim, k);
        for angular in [false, true] {
            let tree = build_hub_tree(&data, n, dim, &neighbors, leaf_size, 200, angular);
            let mut sorted = tree.indices.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..n as i32).collect::<Vec<_>>());
            let leaves = tree.get_all_leaves();
            assert_eq!(tree.n_nodes, 2 * leaves.len() - 1);
            assert!(leaves.iter().all(|&(s, e)| e > s && e - s <= leaf_size));
            // Balanced splits keep the tree shallow: ~n / leaf_size leaves.
            assert!(leaves.len() < 4 * n / leaf_size, "{} leaves", leaves.len());
            // Points route to their own leaves, which hold at least as many of
            // their neighbours as a random projection tree's leaves do.
            let mut rng = FastRng::new(2);
            let mut together = |tree: &FlatTree| {
                let (mut same_leaf, mut routed) = (0, 0);
                for point in 0..n {
                    let (start, end) = tree.search(&data[point * dim..(point + 1) * dim], &mut rng);
                    let leaf = &tree.indices[start..end];
                    routed += leaf.contains(&(point as i32)) as usize;
                    same_leaf += neighbors[point * k..(point + 1) * k].iter().filter(|q| leaf.contains(q)).count();
                }
                (routed, same_leaf)
            };
            let (routed, same_leaf) = together(&tree);
            assert!(routed as f64 > 0.99 * n as f64, "angular={angular}: {routed} of {n}");
            let random = crate::tree::build_rp_tree(&data, n, dim, leaf_size, &mut FastRng::new(3), angular, 200);
            let (_, random_same_leaf) = together(&random);
            assert!(same_leaf as f64 >= 0.95 * random_same_leaf as f64, "angular={angular}: {same_leaf} vs {random_same_leaf}");
        }
    }

    #[test]
    fn hub_tree_does_not_depend_on_threads_and_handles_duplicates() {
        let (n, dim, k) = (2500, 16, 8);
        let base = clustered(25, dim, 3);
        let data: Vec<f32> = (0..n).flat_map(|i| base[(i % 25) * dim..(i % 25 + 1) * dim].to_vec()).collect();
        let neighbors = knn(&data, dim, k);
        let build = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| build_hub_tree(&data, n, dim, &neighbors, 20, 200, false))
        };
        let (one, four) = (build(1), build(4));
        assert_eq!(one.indices, four.indices);
        assert_eq!(one.children, four.children);
        assert_eq!(one.hyperplanes, four.hyperplanes);
        // 25 distinct points: duplicates end in (oversized) leaves rather than looping.
        let mut sorted = one.indices.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..n as i32).collect::<Vec<_>>());
    }
}
