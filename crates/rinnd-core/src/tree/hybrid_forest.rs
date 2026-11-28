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

use super::breadth_first::{batched_dot_products, child_seed, choose_split, repair_split, tie_side, Split, MAX_TREE_BATCH_WIDTH};
use crate::rng::FastRng;
use rayon::prelude::*;

const NO_NODE: u32 = u32::MAX;
const LOCAL_SUBTREE_BYTES: usize = 1 << 20;
/// Local subtrees copy their rows into a contiguous buffer only for short rows;
/// longer rows are already contiguous enough that the copy costs more than it saves.
const GATHER_MAX_DIM: usize = 128;
const ASSIGN_CHUNK: usize = 16384;

/// Point-indexed node assignments shared across threads; each point is written by
/// exactly one task because permutation ranges are disjoint and hold distinct points.
#[derive(Clone, Copy)]
struct SharedAssignments(*mut u32);
unsafe impl Send for SharedAssignments {}
unsafe impl Sync for SharedAssignments {}

impl SharedAssignments {
    #[inline(always)]
    unsafe fn set(self, point: i32, value: u32) {
        *self.0.add(point as usize) = value;
    }
}

/// Assign `value` to every point in each disjoint permutation range, in parallel.
fn assign_ranges(permutation: &[i32], assignments: &mut [u32], ranges: &[(usize, usize, u32)]) {
    let target = SharedAssignments(assignments.as_mut_ptr());
    ranges.par_iter().for_each(|&(start, end, value)| {
        permutation[start..end].par_chunks(ASSIGN_CHUNK).for_each(|chunk| {
            for &point in chunk {
                // SAFETY: ranges are disjoint and the permutation holds each point once.
                unsafe { target.set(point, value) };
            }
        });
    });
}

/// Points per cache-local subtree.
pub(crate) fn local_points_for(dim: usize) -> usize {
    local_points_for_row_bytes(4 * dim)
}

/// Points per cache-local subtree of rows `row_bytes` long.
pub(crate) fn local_points_for_row_bytes(row_bytes: usize) -> usize {
    LOCAL_SUBTREE_BYTES / row_bytes.max(1)
}

/// Root-to-node path, one bit per level (right = 1), most significant first.
/// Nodes at equal depth compare lexicographically, which is breadth-first order.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
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

/// How a forest splits its nodes. [`FloatSpace`] splits the data rows; other
/// spaces split another form of them. Every decision depends only on a node's
/// points and seed.
pub(crate) trait SplitSpace: Sync {
    /// A node's split.
    type Split: Send + Sync;
    /// A deferred node's rows, as [`Self::node_sides`] reads them, and scratch.
    type Local: Send;

    /// Split the points `ids` (see [`choose_split`]).
    fn choose_split(&self, ids: &[i32], seed: u64) -> Self::Split;

    /// Re-split points left on one side (see [`repair_split`]): partitions
    /// `ids`, mirroring each swap through `swap`, and returns the split and
    /// the size of the left part.
    fn repair_split(&self, ids: &mut [i32], swap: impl FnMut(usize, usize), seed: u64) -> (Self::Split, usize);

    /// The side of `point` for each present split of a batch of `width` trees.
    fn batch_sides(&self, point: usize, splits: &[Option<&Self::Split>; MAX_TREE_BATCH_WIDTH], width: usize, sides: &mut [u8]);

    /// The rows of a deferred node's points `ids`, and each point's row in them.
    fn gather(&self, ids: &[i32]) -> (Self::Local, Vec<u32>);

    /// The side of each of the points `ids` (rows `rows` of `local`).
    fn node_sides(&self, local: &mut Self::Local, rows: &[u32], ids: &[i32], split: &Self::Split, sides: &mut Vec<u8>);
}

/// Splits of the data rows themselves.
pub(crate) struct FloatSpace<'a> {
    pub data: &'a [f32],
    pub dim: usize,
    pub angular: bool,
}

/// A deferred node's rows (copied when short) and margins.
pub(crate) struct FloatLocal {
    buffer: Option<Vec<f32>>,
    margins: Vec<f32>,
}

impl SplitSpace for FloatSpace<'_> {
    type Split = Split;
    type Local = FloatLocal;

    fn choose_split(&self, ids: &[i32], seed: u64) -> Split {
        choose_split(self.data, self.dim, ids, seed, self.angular)
    }

    fn repair_split(&self, ids: &mut [i32], swap: impl FnMut(usize, usize), seed: u64) -> (Split, usize) {
        repair_split(self.data, self.dim, ids, swap, seed, self.angular)
    }

    fn batch_sides(&self, point: usize, splits: &[Option<&Split>; MAX_TREE_BATCH_WIDTH], width: usize, sides: &mut [u8]) {
        let row = &self.data[point * self.dim..(point + 1) * self.dim];
        let hyperplanes = splits.map(|split| split.map(|split| split.hyperplane.as_slice()));
        let margins = batched_dot_products(row, &hyperplanes, width);
        for slot in 0..width {
            if let Some(split) = splits[slot] {
                sides[slot] = float_side(margins[slot] + split.offset, split.tie_seed, point);
            }
        }
    }

    fn gather(&self, ids: &[i32]) -> (FloatLocal, Vec<u32>) {
        // Only short rows are copied: longer rows are already contiguous enough.
        if self.dim > GATHER_MAX_DIM {
            return (FloatLocal { buffer: None, margins: Vec::new() }, ids.iter().map(|&id| id as u32).collect());
        }
        let mut buffer = Vec::with_capacity(ids.len() * self.dim);
        for &id in ids {
            buffer.extend_from_slice(&self.data[id as usize * self.dim..(id as usize + 1) * self.dim]);
        }
        (FloatLocal { buffer: Some(buffer), margins: Vec::new() }, (0..ids.len() as u32).collect())
    }

    fn node_sides(&self, local: &mut FloatLocal, rows: &[u32], ids: &[i32], split: &Split, sides: &mut Vec<u8>) {
        let buffer = local.buffer.as_deref().unwrap_or(self.data);
        local_margins(buffer, self.dim, rows, &split.hyperplane, &mut local.margins);
        sides.clear();
        sides.extend(
            local.margins.iter().zip(ids).map(|(&dot, &id)| float_side(dot + split.offset, split.tie_seed, id as usize)),
        );
    }
}

/// The side of a point with this margin (ties broken per point).
#[inline(always)]
fn float_side(margin: f32, tie_seed: u64, point: usize) -> u8 {
    if margin.abs() < 1e-8 {
        tie_side(tie_seed, point)
    } else {
        (margin >= 0.0) as u8
    }
}

struct Tree<P> {
    permutation: Vec<i32>,
    assignments: Vec<u32>,
    frontier: Vec<Node>,
    splits: Vec<P>,
    leaves: Vec<Leaf>,
    deferred: Vec<(Node, Vec<i32>)>,
}

impl<P: Send + Sync> Tree<P> {
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

    /// Decide every frontier node: leaf now, defer to the cache-local phase,
    /// or split at this level. Returns whether any node splits.
    fn classify<S: SplitSpace<Split = P>>(&mut self, space: &S, leaf_size: usize, local_points: usize, max_depth: usize) -> bool {
        let frontier = std::mem::take(&mut self.frontier);
        self.splits.clear();
        let mut ranges = Vec::with_capacity(frontier.len());
        for node in frontier {
            let size = node.end - node.start;
            if size <= leaf_size || node.depth >= max_depth {
                ranges.push((node.start, node.end, NO_NODE));
                self.leaves.push((node.depth, node.path.clone(), self.permutation[node.start..node.end].to_vec()));
            } else if size <= local_points {
                ranges.push((node.start, node.end, NO_NODE));
                let ids = self.permutation[node.start..node.end].to_vec();
                self.deferred.push((node, ids));
            } else {
                ranges.push((node.start, node.end, self.frontier.len() as u32));
                self.splits.push(space.choose_split(&self.permutation[node.start..node.end], node.seed));
                self.frontier.push(node);
            }
        }
        assign_ranges(&self.permutation, &mut self.assignments, &ranges);
        !self.frontier.is_empty()
    }

    /// Partition each split node exactly as the breadth-first builder does,
    /// re-splitting nodes whose split left one side empty.
    /// Children's assignments are written by the next `classify`.
    fn partition<S: SplitSpace<Split = P>>(&mut self, space: &S, tree_slot: usize, batch_width: usize, sides: &[u8]) {
        let frontier = std::mem::take(&mut self.frontier);
        // Frontier nodes are disjoint ranges in increasing order of `start`.
        let mut slices = Vec::with_capacity(frontier.len());
        let mut rest: &mut [i32] = &mut self.permutation;
        let mut offset = 0;
        for node in &frontier {
            let (_, tail) = std::mem::take(&mut rest).split_at_mut(node.start - offset);
            let (slice, tail) = tail.split_at_mut(node.end - node.start);
            slices.push((slice, node.seed));
            rest = tail;
            offset = node.end;
        }
        let lefts: Vec<usize> = slices
            .into_par_iter()
            .map(|(slice, seed)| {
                let (mut left, mut right) = (0, slice.len());
                while left < right {
                    if sides[slice[left] as usize * batch_width + tree_slot] == 0 {
                        left += 1;
                    } else {
                        right -= 1;
                        slice.swap(left, right);
                    }
                }
                if left == 0 || left == slice.len() {
                    return space.repair_split(slice, |_, _| {}, seed).1;
                }
                left
            })
            .collect();
        let mut next = Vec::with_capacity(frontier.len() * 2);
        self.splits.clear();
        for (node, offset) in frontier.into_iter().zip(lefts) {
            let left = node.start + offset;
            next.push(Node { start: node.start, end: left, depth: node.depth + 1, seed: child_seed(node.seed, false), path: node.path.child(node.depth, false) });
            next.push(Node { start: left, end: node.end, depth: node.depth + 1, seed: child_seed(node.seed, true), path: node.path.child(node.depth, true) });
        }
        self.frontier = next;
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

/// Finish one deferred subtree from its gathered rows.
#[allow(clippy::too_many_arguments)]
fn local_subtree<S: SplitSpace>(
    space: &S,
    local: &mut S::Local,
    ids: &mut [i32],
    rows: &mut [u32],
    depth: usize,
    seed: u64,
    path: Path,
    leaf_size: usize,
    max_depth: usize,
    leaves: &mut Vec<Leaf>,
    sides: &mut Vec<u8>,
) {
    let m = ids.len();
    if m <= leaf_size || depth >= max_depth {
        leaves.push((depth, path, ids.to_vec()));
        return;
    }
    let split = space.choose_split(ids, seed);
    // Sides for every point (a function of the point alone), then the same
    // two-pointer partition as the breadth-first builder.
    space.node_sides(local, rows, ids, &split, sides);
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
    drop(split);
    if left == 0 || left == m {
        left = space.repair_split(ids, |a, b| rows.swap(a, b), seed).1;
    }
    let (left_ids, right_ids) = ids.split_at_mut(left);
    let (left_rows, right_rows) = rows.split_at_mut(left);
    let left_path = path.child(depth, false);
    let right_path = path.child(depth, true);
    local_subtree(space, local, left_ids, left_rows, depth + 1, child_seed(seed, false), left_path, leaf_size, max_depth, leaves, sides);
    local_subtree(space, local, right_ids, right_rows, depth + 1, child_seed(seed, true), right_path, leaf_size, max_depth, leaves, sides);
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
    let space = FloatSpace { data, dim, angular };
    build_space_forest(&space, n_points, n_trees, leaf_size, rng, max_depth, tree_batch_width, local_points)
}

/// The leaves of every tree of the hybrid forest of `space`'s `n_points` points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_space_forest<S: SplitSpace>(
    space: &S,
    n_points: usize,
    n_trees: usize,
    leaf_size: usize,
    rng: &mut FastRng,
    max_depth: usize,
    tree_batch_width: usize,
    local_points: usize,
) -> Vec<Vec<i32>> {
    assert!(n_points <= i32::MAX as usize);
    assert!((1..=MAX_TREE_BATCH_WIDTH).contains(&tree_batch_width));

    let seeds: Vec<u64> = (0..n_trees).map(|_| rng.next_u64()).collect();
    build_trees(space, n_points, &seeds, tree_batch_width, leaf_size, max_depth, local_points)
}

/// Build all trees with shared breadth-first levels and return their leaves in tree order,
/// each in breadth-first order. Margins are computed for `tree_batch_width` trees per pass.
#[allow(clippy::too_many_arguments)]
fn build_trees<S: SplitSpace>(
    space: &S,
    n_points: usize,
    seeds: &[u64],
    tree_batch_width: usize,
    leaf_size: usize,
    max_depth: usize,
    local_points: usize,
) -> Vec<Vec<i32>> {
    let n_trees = seeds.len();
    let mut trees: Vec<Tree<S::Split>> = seeds.par_iter().map(|&seed| Tree::new(n_points, seed)).collect();
    // Points outside every frontier keep stale sides, which partition never reads.
    let mut sides = vec![0u8; n_points * n_trees];

    // Breadth-first levels over large nodes, shared by all trees.
    loop {
        let any_splits = trees
            .par_iter_mut()
            .map(|tree| tree.classify(space, leaf_size, local_points, max_depth))
            .reduce(|| false, |a, b| a | b);
        if !any_splits {
            break;
        }
        sides.par_chunks_mut(n_trees).enumerate().for_each(|(point_index, point_sides)| {
            for (group, group_sides) in trees.chunks(tree_batch_width).zip(point_sides.chunks_mut(tree_batch_width)) {
                let mut splits: [Option<&S::Split>; MAX_TREE_BATCH_WIDTH] = [None; MAX_TREE_BATCH_WIDTH];
                let mut any = false;
                for (slot, tree) in group.iter().enumerate() {
                    let node = tree.assignments[point_index];
                    if node != NO_NODE {
                        splits[slot] = Some(&tree.splits[node as usize]);
                        any = true;
                    }
                }
                if any {
                    space.batch_sides(point_index, &splits, group.len(), group_sides);
                }
            }
        });
        trees.par_iter_mut().enumerate().for_each(|(slot, tree)| tree.partition(space, slot, n_trees, &sides));
    }
    drop(sides);

    // Cache-local subtrees for all trees at once, then breadth-first output order per tree.
    let built: Vec<Vec<Vec<i32>>> = trees
        .into_par_iter()
        .map(|tree| {
            let mut leaves = tree.leaves;
            let local: Vec<Vec<Leaf>> = tree
                .deferred
                .into_par_iter()
                .map(|(node, mut ids)| {
                    let mut out = Vec::new();
                    let (mut local, mut rows) = space.gather(&ids);
                    let mut sides = Vec::new();
                    local_subtree(space, &mut local, &mut ids, &mut rows, node.depth, node.seed, node.path, leaf_size, max_depth, &mut out, &mut sides);
                    out
                })
                .collect();
            leaves.extend(local.into_iter().flatten());
            leaves.sort_unstable_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
            leaves.into_iter().map(|leaf| leaf.2).collect()
        })
        .collect();
    built.into_iter().flatten().collect()
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
    fn hybrid_matches_breadth_first_without_gather() {
        let (n, dim) = (3000, GATHER_MAX_DIM + 72);
        let x = data(n, dim, 3);
        for &(angular, width, local, threads) in &[(false, 6, 500, 2), (true, 4, 200, 3), (false, 5, 10_000, 1)] {
            let (a, b) = both(&x, n, dim, 11, 60, angular, 200, width, local, threads);
            assert_eq!(a, b, "angular={angular} width={width} local={local} threads={threads}");
        }
    }

    #[test]
    fn failed_splits_are_retried_and_leaves_stay_bounded() {
        // Duplicates and zero vectors make many splits fail.
        let (n, dim) = (3000, 16);
        let base = data(40, dim, 2);
        let mut x: Vec<f32> = (0..n).flat_map(|i| base[(i % 40) * dim..(i % 40 + 1) * dim].to_vec()).collect();
        x[..500 * dim].fill(0.0);
        for angular in [false, true] {
            let (a, b) = both(&x, n, dim, 3, 20, angular, 200, 6, 500, 2);
            assert_eq!(a, b);
            assert!(b.iter().all(|leaf| !leaf.is_empty() && leaf.len() <= 20), "angular={angular}");
            let mut points: Vec<i32> = b.iter().flatten().copied().collect();
            points.sort_unstable();
            assert_eq!(points, (0..n as i32).flat_map(|p| [p; 3]).collect::<Vec<_>>());
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
