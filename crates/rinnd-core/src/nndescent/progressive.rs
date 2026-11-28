//! Progressive-precision NN-descent for wide dense data.
//!
//! On data with many features most of a build's time goes into distance
//! evaluations in the RP-tree leaf join and the first NN-descent iterations.
//! Those phases only need to find roughly the right neighbours, so they can run
//! on 8-bit codes ([`AffineCodes::encode_for_build`]) scored with integer dot
//! products (AVX-512 VNNI where available): each point acts as a query, with
//! int8 weights and a 16-byte record ([`AffineCodes::row_queries`]), against
//! the other points' code rows, four at a time. Code and weight rows are padded
//! to whole 64-byte cache lines and start on one.
//!
//! The approximate phase keeps wider heaps ([`APPROX_WIDTH`] times
//! `n_neighbors`): pairs compared on the codes are not compared again, so the
//! true neighbours have to survive the approximate phase somewhere in the heap.
//! Once the update count converges on the codes, every heap entry is re-scored
//! exactly ([`rescore`]), the heaps are cut to `n_neighbors`, and float
//! NN-descent runs to convergence from there. The resulting graph has exact
//! distances.

use super::{CandidateSets, PairSource};
use crate::distance::affine::{AffineCodes, AffineKind, RowQuery};
use crate::distance::int_dot::{dot4_u8, dot_u8, AlignedBytes, Kernel, LINE};
use crate::distance::Distance;
use crate::heap::NeighborHeap;
use rayon::prelude::*;

/// Whether a build runs its early phases on 8-bit codes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ProgressiveBuild {
    /// From [`MIN_DIMENSION`] features up, for distances with a code form.
    #[default]
    Auto,
    /// Whenever the distance has a code form.
    Always,
    /// Never: every distance is exact.
    Never,
}

/// Below this many features the faster code distances do not pay for the
/// wider approximate heaps and the final re-score.
pub const MIN_DIMENSION: usize = 512;

/// Width of the approximate phase's heaps, relative to `n_neighbors`.
pub const APPROX_WIDTH: f32 = 1.25;

/// Code (and weight) rows of up to this many cache lines are requested whole
/// ahead of a join (at 768 features, 4% to 5% faster than none); of longer
/// rows, only the first [`LONG_ROW_PREFETCH_LINES`], as the hardware prefetcher
/// streams the rest (at 4096 features, 16 lines measured 4% to 6% slower).
const PREFETCH_LINES: usize = 16;
const LONG_ROW_PREFETCH_LINES: usize = 4;

/// Lines of a `stride`-byte row to request ahead of a join.
fn prefetch_lines(stride: usize) -> usize {
    let lines = stride / LINE;
    if lines <= PREFETCH_LINES {
        lines
    } else {
        LONG_ROW_PREFETCH_LINES
    }
}

/// The code form of `distance` for this build, if it runs progressively.
pub(crate) fn code_kind<D: Distance<f32>>(
    mode: ProgressiveBuild,
    distance: &D,
    n_points: usize,
    dim: usize,
    n_neighbors: usize,
) -> Option<AffineKind> {
    let kind = distance.build_code_kind()?;
    if n_points <= n_neighbors + 1 {
        return None;
    }
    match mode {
        ProgressiveBuild::Never => None,
        ProgressiveBuild::Always => Some(kind),
        ProgressiveBuild::Auto => (dim >= MIN_DIMENSION).then_some(kind),
    }
}

/// Heap width of the approximate phase.
pub(crate) fn approximate_width(n_neighbors: usize, n_points: usize) -> usize {
    let wide = (APPROX_WIDTH * n_neighbors as f32).round() as usize;
    wide.max(n_neighbors).min(n_points.saturating_sub(1).max(n_neighbors))
}

/// Approximate distances between points from their 8-bit codes: point `p`'s
/// int8 weights against point `q`'s codes, `offset_p + scale_p <w_p, c_q>`
/// (plus `q`'s code norm for squared euclidean distances), in the units of the
/// exact distance's code form.
pub(crate) struct CodePairs {
    codes: AffineCodes,
    weights: AlignedBytes,
    records: Vec<RowQuery>,
    stride: usize,
    squared_euclidean: bool,
    kernel: Kernel,
    prefetch_lines: usize,
}

impl CodePairs {
    /// Encode `data` (`dim` values per row).
    pub(crate) fn encode(data: &[f32], dim: usize, kind: AffineKind) -> Self {
        let codes = AffineCodes::encode_for_build(data, dim, kind);
        let (weights, records) = codes.row_queries(data);
        let stride = codes.stride();
        Self {
            kernel: codes.kernel(),
            stride,
            squared_euclidean: kind == AffineKind::SquaredEuclidean,
            prefetch_lines: prefetch_lines(stride),
            codes,
            weights,
            records,
        }
    }

    /// The codes.
    pub(crate) fn codes(&self) -> &AffineCodes {
        &self.codes
    }

    /// Bytes of codes, weights and records.
    pub(crate) fn storage_bytes(&self) -> usize {
        self.codes.storage_bytes() + self.weights.len() + self.records.len() * std::mem::size_of::<RowQuery>()
    }

    #[inline(always)]
    fn weights(&self, point: i32) -> &[i8] {
        let start = point as usize * self.stride;
        &self.weights.as_i8_slice()[start..start + self.stride]
    }

    #[inline(always)]
    fn finish(&self, record: &RowQuery, dot: i32, neighbor: i32) -> f32 {
        let distance = record.offset + record.scale * dot as f32;
        if self.squared_euclidean {
            distance + self.records[neighbor as usize].norm
        } else {
            distance
        }
    }

    /// Request the lines a join reads of `point`: its code row and record, and
    /// its weights when it acts as the query.
    #[inline(always)]
    fn prefetch_point(&self, point: i32, with_weights: bool) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            if point < 0 {
                return;
            }
            let index = point as usize;
            let codes = self.codes.row(point).as_ptr() as *const i8;
            let weights = self.weights.as_slice().as_ptr().add(index * self.stride) as *const i8;
            for line in 0..self.prefetch_lines {
                _mm_prefetch(codes.add(line * LINE), _MM_HINT_T0);
                if with_weights {
                    _mm_prefetch(weights.add(line * LINE), _MM_HINT_T0);
                }
            }
            _mm_prefetch(self.records.as_ptr().add(index) as *const i8, _MM_HINT_T0);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (point, with_weights);
    }
}

impl PairSource for CodePairs {
    #[inline(always)]
    fn distances(&self, point: i32, candidates: &[i32], skip_self: bool, mut apply: impl FnMut(usize, i32, f32)) {
        let weights = self.weights(point);
        let record = self.records[point as usize];
        let valid = |neighbor: i32| neighbor >= 0 && (!skip_self || neighbor != point);
        let mut offset = 0;
        while offset + 4 <= candidates.len() {
            let group = [candidates[offset], candidates[offset + 1], candidates[offset + 2], candidates[offset + 3]];
            if group.iter().all(|&neighbor| valid(neighbor)) {
                let dots = dot4_u8(self.kernel, group.map(|neighbor| self.codes.row(neighbor)), weights);
                for slot in 0..4 {
                    apply(offset + slot, group[slot], self.finish(&record, dots[slot], group[slot]));
                }
                offset += 4;
            } else {
                let neighbor = group[0];
                if valid(neighbor) {
                    let dot = dot_u8(self.kernel, self.codes.row(neighbor), weights);
                    apply(offset, neighbor, self.finish(&record, dot, neighbor));
                }
                offset += 1;
            }
        }
        for (slot, &neighbor) in candidates.iter().enumerate().skip(offset) {
            if valid(neighbor) {
                let dot = dot_u8(self.kernel, self.codes.row(neighbor), weights);
                apply(slot, neighbor, self.finish(&record, dot, neighbor));
            }
        }
    }

    #[inline(always)]
    fn prefetch_hub(&self, candidates: &CandidateSets, hub: usize, _graph: &NeighborHeap) {
        // New candidates are joined against everything (as queries); old ones
        // only as the other side.
        for &point in candidates.get_new(hub) {
            self.prefetch_point(point, true);
        }
        for &point in candidates.get_old(hub) {
            self.prefetch_point(point, false);
        }
    }

    #[inline(always)]
    fn prefetch_points(&self, points: &[i32]) {
        for &point in points {
            self.prefetch_point(point, true);
        }
    }
}

/// Points in order of first appearance in `leaves` (then any others): a
/// permutation that visits neighbouring points close together.
pub(crate) fn leaf_order(leaves: &[Vec<i32>], n_points: usize) -> Vec<u32> {
    let mut seen = vec![false; n_points];
    let mut order = Vec::with_capacity(n_points);
    for &point in leaves.iter().flatten() {
        if point >= 0 && !std::mem::replace(&mut seen[point as usize], true) {
            order.push(point as u32);
        }
    }
    order.extend((0..n_points as u32).filter(|&point| !seen[point as usize]));
    order
}

/// Raw pointers to a heap's arrays, for writing disjoint rows from threads.
struct RowWriter {
    indices: *mut i32,
    distances: *mut f32,
    flags: *mut u8,
}
// SAFETY: rows are written by one thread each (the order is a permutation).
unsafe impl Send for RowWriter {}
unsafe impl Sync for RowWriter {}

/// Re-score every entry of the approximate graph with `exact` distances and
/// keep each row's `k` nearest (with their new/old flags). Rows are processed
/// in `order` (see [`leaf_order`]): re-scoring is memory bound, and neighbours'
/// rows are shared between points that are close in that order.
pub(crate) fn rescore<S: PairSource>(approximate: &NeighborHeap, exact: &S, k: usize, order: &[u32]) -> NeighborHeap {
    let mut graph = NeighborHeap::new(approximate.n_points, k);
    let writer = RowWriter {
        indices: graph.indices.as_mut_ptr(),
        distances: graph.distances.as_mut_ptr(),
        flags: graph.flags.as_mut_ptr(),
    };
    let writer = &writer;
    order.par_chunks(256).for_each_init(
        || Vec::with_capacity(approximate.k),
        |scored: &mut Vec<(f32, i32, u8)>, chunk| {
            for &point in chunk {
                let point = point as usize;
                let (indices, _, flags) = approximate.get_row(point);
                scored.clear();
                exact.distances(point as i32, indices, true, |slot, neighbor, distance| {
                    scored.push((distance, neighbor, flags[slot]));
                });
                let order = |a: &(f32, i32, u8), b: &(f32, i32, u8)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
                if scored.len() > k {
                    scored.select_nth_unstable_by(k - 1, order);
                    scored.truncate(k);
                }
                // Descending order is a valid max-heap; missing entries
                // (infinite distance) come first.
                scored.sort_unstable_by(|a, b| order(b, a));
                let start = point * k + (k - scored.len());
                for (position, &(distance, neighbor, flag)) in scored.iter().enumerate() {
                    // SAFETY: in bounds, and row `point` belongs to this thread (see RowWriter).
                    unsafe {
                        *writer.indices.add(start + position) = neighbor;
                        *writer.distances.add(start + position) = distance;
                        *writer.flags.add(start + position) = flag;
                    }
                }
            }
        },
    );
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{DirectNormalizedCosine, SquaredEuclidean};
    use crate::nndescent::FloatPairs;
    use crate::rng::FastRng;

    fn clustered(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = FastRng::new(seed);
        let centres: Vec<f32> = (0..20 * dim).map(|_| rng.next_float() * 4.0 - 2.0).collect();
        let mut data = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let centre = rng.next_index(20);
            data.extend((0..dim).map(|t| centres[centre * dim + t] + rng.next_float() - 0.5));
        }
        data
    }

    #[test]
    fn code_pairs_approximate_exact_distances() {
        let (n, dim) = (600, 96);
        let mut data = clustered(n, dim, 3);
        for (kind, normalize) in [(AffineKind::SquaredEuclidean, false), (AffineKind::Dot, true)] {
            if normalize {
                for row in data.chunks_exact_mut(dim) {
                    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    row.iter_mut().for_each(|x| *x /= norm);
                }
            }
            let pairs = CodePairs::encode(&data, dim, kind);
            let candidates: Vec<i32> = (0..n as i32).collect();
            let mut worst = 0.0f32;
            let mut scale = 0.0f32;
            for point in [0, 7, 311] {
                let mut approximate = vec![f32::NAN; n];
                pairs.distances(point, &candidates, true, |slot, _, d| approximate[slot] = d);
                assert!(approximate[point as usize].is_nan(), "self is skipped");
                let x = &data[point as usize * dim..(point as usize + 1) * dim];
                for q in (0..n).filter(|&q| q != point as usize) {
                    let y = &data[q * dim..(q + 1) * dim];
                    let exact = if normalize {
                        DirectNormalizedCosine.distance(x, y)
                    } else {
                        SquaredEuclidean.distance(x, y)
                    };
                    worst = worst.max((approximate[q] - exact).abs());
                    scale = scale.max(exact.abs());
                }
            }
            assert!(worst < 0.02 * scale, "{kind:?}: worst error {worst} of {scale}");
        }
    }

    #[test]
    fn rescore_keeps_the_nearest_with_exact_distances() {
        let (n, dim, width, k) = (200, 64, 12, 8);
        let data = clustered(n, dim, 5);
        let exact = FloatPairs { data: &data, dim, distance: &SquaredEuclidean };
        let mut approximate = NeighborHeap::new(n, width);
        let mut rng = FastRng::new(9);
        for point in 0..n {
            for _ in 0..width {
                let neighbor = rng.next_index(n) as i32;
                if neighbor != point as i32 && !approximate.contains(point, neighbor) {
                    // Approximate distances unrelated to the exact ones.
                    approximate.checked_flagged_push(point, neighbor, rng.next_float(), neighbor % 2 == 0);
                }
            }
        }
        let order: Vec<u32> = (0..n as u32).rev().collect();
        let graph = rescore(&approximate, &exact, k, &order);
        for point in 0..n {
            let (indices, distances, flags) = graph.get_row(point);
            let mut expected: Vec<(f32, i32)> = approximate
                .get_row(point)
                .0
                .iter()
                .filter(|&&q| q >= 0)
                .map(|&q| (SquaredEuclidean.distance(&data[point * dim..(point + 1) * dim], &data[q as usize * dim..(q as usize + 1) * dim]), q))
                .collect();
            expected.sort_by(|a, b| a.0.total_cmp(&b.0));
            expected.truncate(k);
            let mut got: Vec<(f32, i32)> = indices.iter().zip(distances).filter(|(&q, _)| q >= 0).map(|(&q, &d)| (d, q)).collect();
            got.sort_by(|a, b| a.0.total_cmp(&b.0));
            assert_eq!(got.len(), expected.len());
            for (g, e) in got.iter().zip(&expected) {
                assert_eq!(g.1, e.1);
                assert!((g.0 - e.0).abs() <= 1e-4 * (1.0 + e.0));
            }
            // A valid max-heap: the root is the largest.
            assert!(distances.iter().all(|&d| d <= distances[0]));
            for (&q, &flag) in indices.iter().zip(flags) {
                if q >= 0 {
                    assert_eq!(flag, u8::from(q % 2 == 0));
                }
            }
        }
    }
}
