//! Hybrid breadth-first / cache-local RP forest (feature `hybrid-forest`).
//!
//! The breadth-first builder streams the data once per level for a batch of
//! trees, but every point needs its own node's hyperplane from each tree. At
//! deep levels those hyperplanes no longer fit in cache. The depth-first
//! builder has the opposite problem: hyperplanes stay hot but each level
//! gathers its points from the whole dataset.
//!
//! This builder runs the breadth-first levels while nodes are large. Once a
//! node holds at most `local_points` points (sized from a byte budget), its
//! rows are gathered once into a contiguous buffer and the whole subtree is
//! finished from that buffer.
//!
//! The output is identical to `build_breadth_first_leaf_forest`: the same
//! per-node seeds, split construction, dot-product kernel (called per slot,
//! whose result does not depend on batch width), tie rule and two-pointer
//! partition are used, and leaves are emitted in breadth-first order by
//! sorting on (depth, root-to-node path).

use super::breadth_first::{batched_dot_products, child_seed, make_split, tie_side, Split, MAX_TREE_BATCH_WIDTH};
use crate::rng::FastRng;
use rayon::prelude::*;

const NO_NODE: usize = usize::MAX;
const LOCAL_SUBTREE_BYTES: usize = 4 << 20;

/// Points per cache-local subtree.
pub(crate) fn local_points_for(dim: usize) -> usize {
    LOCAL_SUBTREE_BYTES / (4 * dim.max(1))
}

/// Root-to-node path, one bit per level (right = 1), most significant first.
/// Nodes at equal depth compare lexicographically, which is breadth-first order.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
struct Path(Vec<u64>);

impl Path {
    fn child(&self, parent_depth: usize, right: bool) -> Path {
        let mut words = self.0.clone();
        let word = parent_depth / 64;
        if words.len() <= word {
            words.push(0);
        }
        if right {
            words[word] |= 1u64 << (63 - parent_depth % 64);
        }
        Path(words)
    }
}

struct Node {
    start: usize,
    end: usize,
    depth: usize,
    seed: u64,
    path: Path,
}

type Leaf = (usize, Path, Vec<i32>);

struct Tree {
    permutation: Vec<i32>,
    assignments: Vec<usize>,
    frontier: Vec<Node>,
    splits: Vec<Split>,
    leaves: Vec<Leaf>,
    deferred: Vec<(Node, Vec<i32>)>,
}

impl Tree {
    fn new(n_points: usize, seed: u64) -> Self {
        Self {
            permutation: (0..n_points as i32).collect(),
            assignments: vec![0; n_points],
            frontier: vec![Node { start: 0, end: n_points, depth: 0, seed, path: Path::default() }],
            splits: Vec::new(),
            leaves: Vec::new(),
            deferred: Vec::new(),
        }
    }

    fn retire(&mut self, node: &Node) {
        for &point in &self.permutation[node.start..node.end] {
            self.assignments[point as usize] = NO_NODE;
        }
    }

    /// Decide every frontier node: leaf now, defer to the cache-local phase,
    /// or split at this level. Returns whether any node splits.
    fn classify(&mut self, data: &[f32], dim: usize, leaf_size: usize, local_points: usize, angular: bool, max_depth: usize) -> bool {
        let frontier = std::mem::take(&mut self.frontier);
        self.splits.clear();
        for node in frontier {
            let size = node.end - node.start;
            if size <= leaf_size || node.depth >= max_depth {
                self.retire(&node);
                self.leaves.push((node.depth, node.path.clone(), self.permutation[node.start..node.end].to_vec()));
            } else if size <= local_points {
                self.retire(&node);
                let ids = self.permutation[node.start..node.end].to_vec();
                self.deferred.push((node, ids));
            } else {
                match make_split(data, dim, &self.permutation[node.start..node.end], node.seed, angular) {
                    None => {
                        self.retire(&node);
                        self.leaves.push((node.depth, node.path.clone(), self.permutation[node.start..node.end].to_vec()));
                    }
                    Some(split) => {
                        let index = self.frontier.len();
                        for &point in &self.permutation[node.start..node.end] {
                            self.assignments[point as usize] = index;
                        }
                        self.frontier.push(node);
                        self.splits.push(split);
                    }
                }
            }
        }
        !self.frontier.is_empty()
    }

    /// Partition each split node exactly as the breadth-first builder does.
    fn partition(&mut self, tree_slot: usize, batch_width: usize, sides: &[u8]) {
        let frontier = std::mem::take(&mut self.frontier);
        let mut next = Vec::with_capacity(frontier.len() * 2);
        for node in frontier {
            let (mut left, mut right) = (node.start, node.end);
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
                self.retire(&node);
                self.leaves.push((node.depth, node.path.clone(), self.permutation[node.start..node.end].to_vec()));
                continue;
            }
            next.push(Node { start: node.start, end: left, depth: node.depth + 1, seed: child_seed(node.seed, false), path: node.path.child(node.depth, false) });
            next.push(Node { start: left, end: node.end, depth: node.depth + 1, seed: child_seed(node.seed, true), path: node.path.child(node.depth, true) });
        }
        for (index, node) in next.iter().enumerate() {
            for &point in &self.permutation[node.start..node.end] {
                self.assignments[point as usize] = index;
            }
        }
        self.frontier = next;
        self.splits.clear();
    }
}

/// Margins of up to 8 points against one hyperplane. Each point uses exactly
/// the operation sequence of `batched_dot_products` for a single slot (one
/// 8-lane FMA accumulator, the same horizontal reduction, the same scalar
/// tail), so results are bit-identical; batching points instead of trees gives
/// independent FMA chains instead of one latency-bound chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_points_avx2(points: &[&[f32]], plane: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;
    let dim = plane.len();
    let chunks = dim / 8;
    let count = points.len();
    let mut acc = [_mm256_setzero_ps(); 8];
    for chunk in 0..chunks {
        let offset = chunk * 8;
        let h = _mm256_loadu_ps(plane.as_ptr().add(offset));
        for p in 0..count {
            let v = _mm256_loadu_ps(points[p].as_ptr().add(offset));
            acc[p] = _mm256_fmadd_ps(v, h, acc[p]);
        }
    }
    for p in 0..count {
        let high = _mm256_extractf128_ps(acc[p], 1);
        let low = _mm256_castps256_ps128(acc[p]);
        let pair = _mm_add_ps(high, low);
        let shuffled = _mm_movehdup_ps(pair);
        let pair = _mm_add_ps(pair, shuffled);
        let shuffled = _mm_movehl_ps(pair, pair);
        let mut sum = _mm_cvtss_f32(_mm_add_ss(pair, shuffled));
        for d in chunks * 8..dim {
            sum += points[p][d] * plane[d];
        }
        out[p] = sum;
    }
}

/// Margins (dot product only, offset not added) for `rows` of `buffer`.
fn local_margins(buffer: &[f32], dim: usize, rows: &[u32], plane: &[f32], out: &mut Vec<f32>) {
    out.clear();
    out.resize(rows.len(), 0.0);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            for (group, margins) in rows.chunks(8).zip(out.chunks_mut(8)) {
                let mut points: [&[f32]; 8] = [&[]; 8];
                for (slot, &row) in group.iter().enumerate() {
                    points[slot] = &buffer[row as usize * dim..(row as usize + 1) * dim];
                }
                unsafe { dot_points_avx2(&points[..group.len()], plane, margins) };
            }
            return;
        }
    }
    let mut planes: [Option<&[f32]>; MAX_TREE_BATCH_WIDTH] = [None; MAX_TREE_BATCH_WIDTH];
    planes[0] = Some(plane);
    for (margin, &row) in out.iter_mut().zip(rows) {
        *margin = batched_dot_products(&buffer[row as usize * dim..(row as usize + 1) * dim], &planes, 1)[0];
    }
}

/// Finish one deferred subtree from a contiguous copy of its rows.
#[allow(clippy::too_many_arguments)]
fn local_subtree(
    data: &[f32],
    dim: usize,
    buffer: &[f32],
    ids: &mut [i32],
    rows: &mut [u32],
    depth: usize,
    seed: u64,
    path: Path,
    leaf_size: usize,
    angular: bool,
    max_depth: usize,
    out: &mut Vec<Leaf>,
    scratch: &mut (Vec<f32>, Vec<u8>),
) {
    let m = ids.len();
    if m <= leaf_size || depth >= max_depth {
        out.push((depth, path, ids.to_vec()));
        return;
    }
    let Some(split) = make_split(data, dim, ids, seed, angular) else {
        out.push((depth, path, ids.to_vec()));
        return;
    };
    // Sides for every point (a function of the point alone), then the same
    // two-pointer partition as the breadth-first builder.
    let (margins, sides) = scratch;
    local_margins(buffer, dim, rows, &split.hyperplane, margins);
    sides.clear();
    sides.extend(margins.iter().zip(ids.iter()).map(|(&dot, &id)| {
        let margin = dot + split.offset;
        if margin.abs() < 1e-8 { tie_side(seed, id as usize) } else { (margin >= 0.0) as u8 }
    }));
    let (mut left, mut right) = (0, m);
    while left < right {
        if sides[left] == 0 {
            left += 1;
        } else {
            right -= 1;
            ids.swap(left, right);
            rows.swap(left, right);
            sides.swap(left, right);
        }
    }
    if left == 0 || left == m {
        out.push((depth, path, ids.to_vec()));
        return;
    }
    let (left_ids, right_ids) = ids.split_at_mut(left);
    let (left_rows, right_rows) = rows.split_at_mut(left);
    let left_path = path.child(depth, false);
    let right_path = path.child(depth, true);
    local_subtree(data, dim, buffer, left_ids, left_rows, depth + 1, child_seed(seed, false), left_path, leaf_size, angular, max_depth, out, scratch);
    local_subtree(data, dim, buffer, right_ids, right_rows, depth + 1, child_seed(seed, true), right_path, leaf_size, angular, max_depth, out, scratch);
}

/// Drop-in replacement for `build_breadth_first_leaf_forest` with identical output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_hybrid_leaf_forest(
    data: &[f32],
    n_points: usize,
    dim: usize,
    n_trees: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    angular: bool,
    max_depth: usize,
    tree_batch_width: usize,
    local_points: usize,
) -> Vec<Vec<i32>> {
    assert_eq!(data.len(), n_points * dim);
    assert!(n_points <= i32::MAX as usize);
    assert!((1..=MAX_TREE_BATCH_WIDTH).contains(&tree_batch_width));

    let seeds: Vec<u64> = (0..n_trees).map(|_| rng.next_u64()).collect();
    let mut forest_leaves = Vec::new();
    for seed_batch in seeds.chunks(tree_batch_width) {
        let batch_width = seed_batch.len();
        let mut trees: Vec<Tree> = seed_batch.iter().map(|&seed| Tree::new(n_points, seed)).collect();

        // Breadth-first levels over large nodes.
        loop {
            let mut any_splits = false;
            for tree in &mut trees {
                any_splits |= tree.classify(data, dim, leaf_size, local_points, angular, max_depth);
            }
            if !any_splits {
                break;
            }
            let mut sides = vec![0u8; n_points * batch_width];
            sides.par_chunks_mut(batch_width).enumerate().for_each(|(point_index, point_sides)| {
                let mut hyperplanes: [Option<&[f32]>; MAX_TREE_BATCH_WIDTH] = [None; MAX_TREE_BATCH_WIDTH];
                let mut any = false;
                for (slot, tree) in trees.iter().enumerate() {
                    let node = tree.assignments[point_index];
                    if node != NO_NODE {
                        hyperplanes[slot] = Some(tree.splits[node].hyperplane.as_slice());
                        any = true;
                    }
                }
                if !any {
                    return;
                }
                let point = &data[point_index * dim..(point_index + 1) * dim];
                let margins = batched_dot_products(point, &hyperplanes, batch_width);
                for (slot, tree) in trees.iter().enumerate() {
                    let node = tree.assignments[point_index];
                    if node != NO_NODE {
                        let margin = margins[slot] + tree.splits[node].offset;
                        point_sides[slot] = if margin.abs() < 1e-8 {
                            tie_side(tree.frontier[node].seed, point_index)
                        } else {
                            (margin >= 0.0) as u8
                        };
                    }
                }
            });
            trees.par_iter_mut().enumerate().for_each(|(slot, tree)| tree.partition(slot, batch_width, &sides));
        }

        // Cache-local subtrees, then breadth-first output order per tree.
        for tree in trees {
            let mut leaves = tree.leaves;
            let local: Vec<Vec<Leaf>> = tree
                .deferred
                .into_par_iter()
                .map(|(node, mut ids)| {
                    let mut buffer = Vec::with_capacity(ids.len() * dim);
                    for &id in &ids {
                        buffer.extend_from_slice(&data[id as usize * dim..(id as usize + 1) * dim]);
                    }
                    let mut rows: Vec<u32> = (0..ids.len() as u32).collect();
                    let mut out = Vec::new();
                    let mut scratch = (Vec::new(), Vec::new());
                    local_subtree(data, dim, &buffer, &mut ids, &mut rows, node.depth, node.seed, node.path, leaf_size, angular, max_depth, &mut out, &mut scratch);
                    out
                })
                .collect();
            leaves.extend(local.into_iter().flatten());
            leaves.sort_unstable_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
            forest_leaves.extend(leaves.into_iter().map(|leaf| leaf.2));
        }
    }
    forest_leaves
}

#[cfg(test)]
mod tests {
    use super::super::breadth_first::build_breadth_first_leaf_forest;
    use super::*;
    use rayon::ThreadPoolBuilder;

    fn data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = FastRng::new(seed);
        (0..n * dim).map(|_| rng.next_float()).collect()
    }

    fn both(
        data: &[f32], n: usize, dim: usize, trees: usize, leaf: usize, angular: bool, max_depth: usize, width: usize, local: usize, threads: usize,
    ) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
        let pool = ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        pool.install(|| {
            let a = build_breadth_first_leaf_forest(data, n, dim, trees, leaf, &mut FastRng::new(9), angular, max_depth, width);
            let b = build_hybrid_leaf_forest(data, n, dim, trees, leaf, &mut FastRng::new(9), angular, max_depth, width, local);
            (a, b)
        })
    }

    #[test]
    fn hybrid_matches_breadth_first_exactly() {
        let (n, dim) = (6000, 37);
        let x = data(n, dim, 1);
        for &(angular, width, local, threads) in &[
            (false, 6, 800, 1),
            (false, 8, 300, 3),
            (true, 3, 2000, 2),
            (false, 1, 61, 1),     // deferral just above leaf size
            (false, 4, 10_000, 1), // whole tree deferred at the root
            (false, 6, 0, 1),      // never deferred: plain breadth-first path
        ] {
            let (a, b) = both(&x, n, dim, 7, 60, angular, 200, width, local, threads);
            assert_eq!(a, b, "angular={angular} width={width} local={local} threads={threads}");
        }
    }

    #[test]
    fn hybrid_matches_with_depth_limit_and_ties() {
        // Many duplicate points force tie-breaking and depth-limited leaves.
        let (n, dim) = (3000, 16);
        let base = data(40, dim, 2);
        let x: Vec<f32> = (0..n).flat_map(|i| base[(i % 40) * dim..(i % 40 + 1) * dim].to_vec()).collect();
        for &(max_depth, local) in &[(6, 500), (200, 500), (200, 64)] {
            let (a, b) = both(&x, n, dim, 4, 20, false, max_depth, 4, local, 2);
            assert_eq!(a, b, "max_depth={max_depth} local={local}");
        }
    }

    #[test]
    fn path_order_is_breadth_first_order() {
        let root = Path::default();
        let l = root.child(0, false);
        let r = root.child(0, true);
        assert!(l < r);
        assert!(r.child(1, false) > l.child(1, true));
        // Paths deeper than one word still compare correctly.
        let mut deep_a = Path::default();
        let mut deep_b = Path::default();
        for d in 0..70 {
            deep_a = deep_a.child(d, d == 66);
            deep_b = deep_b.child(d, d == 67);
        }
        assert!(deep_a > deep_b);
    }
}
