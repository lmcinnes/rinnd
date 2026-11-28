//! Per-feature affine codes (8-bit, or 4-bit packed two to a byte) searched
//! with an integer-weighted query (asymmetric).
//!
//! Each feature `j` gets its own map `x_j ~ offset_j + scale_j * c_j`, with
//! `c_j` in `0..=255` (8 bits) or `0..=15` (4 bits), fitted to the feature's
//! range: the full range, or the range between two quantiles (by default for
//! 4-bit codes, whose sixteen levels are too few to spend on outliers; values
//! outside it get the end codes).
//!
//! Both distances are linear in the codes, with `u = scale * (q - offset)`
//! (squared euclidean) or `u = scale * q` (dot):
//!
//! * squared euclidean: `|q - x|^2 = |q - offset|^2 - 2 <u, c> + |scale c|^2`,
//!   the last term stored per row;
//! * dot (direct cosine on unit vectors): `1 - <q, x> = 1 - <q, offset> - <u, c>`;
//! * maximum inner product: `|q|^2 + M^2 - 2 <q, x>`, the squared euclidean
//!   distance between the zero-padded query and the point augmented to norm
//!   `M` (see [`AffineKind::Mips`]), computed from the dot product alone.
//!
//! A query rounds `u` once to int8 weights `w` (`u ~ step * w`), so each code
//! distance is one integer dot product ([`super::int_dot`]). The rounding's
//! expected effect over the data, `sum_j (step w_j - u_j) mean_j`, is folded
//! into the per-query constant, so it mostly shifts every candidate's distance
//! by the same amount. Code rows are padded to whole 64-byte lines and stored
//! aligned.
//!
//! Code distances guide the graph search only; the best candidates are then
//! reranked with exact distances, or from a float16 copy ([`HalfRows`]).

use super::int_dot::{dot4_u4, dot4_u8, dot_u4, dot_u8, AlignedBytes, Kernel, LINE};

/// The distance the codes approximate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AffineKind {
    /// Squared euclidean distance.
    SquaredEuclidean,
    /// `1 - <q, x>` (cosine distance of unit vectors).
    Dot,
    /// `|q|^2 + norm2 - 2 <q, x>`: for points augmented with
    /// `sqrt(norm2 - |x|^2)` (so all have norm `sqrt(norm2)`) and queries
    /// padded with a zero, the squared euclidean distance, which orders points
    /// by inner product. Codes and rows hold the unaugmented features only.
    Mips { norm2: f32 },
}

/// A query transformed for code distances (see the module documentation).
#[derive(Clone, Debug, Default)]
pub struct AffineQuery {
    /// int8 weights laid out like a code row (4-bit codes: low nibbles'
    /// weights, then high nibbles').
    weights: AlignedBytes,
    /// Unrounded weights (scratch).
    u: Vec<f32>,
    offset: f32,
    scale: f32,
}

impl AffineQuery {
    /// Add `shift` to the dot product the codes approximate (used for the
    /// `<q, mean>` term of projected codes, which is zero for squared
    /// euclidean codes).
    pub fn add_to_dot(&mut self, shift: f32) {
        self.offset -= shift;
    }
}

/// How many principal components projected codes keep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectionDims {
    /// A fixed number of components.
    Components(usize),
    /// The fewest components explaining this fraction of the variance.
    VarianceFraction(f64),
}

/// Projections keeping more than this fraction of the features are skipped:
/// projecting each query would cost more than the shorter codes save.
pub const MAX_PROJECTED_FRACTION: f64 = 2.0 / 3.0;

/// Projection onto the top principal axes of a dataset, stored as int8 with
/// one scale per output column (so projecting a query is cheap).
///
/// Codes of `P^T (x - mean)` then stand in for `x`: squared euclidean
/// distances use the centred projected query `P^T (q - mean)`, and dot
/// products `<q, x> ~ <q, mean> + <P^T q, P^T (x - mean)>`.
#[derive(Clone, Debug)]
pub struct PcaProjection {
    /// Data mean.
    pub mean: Vec<f32>,
    /// `dim x n_components` projection, row-major; column `j` is
    /// `column_scale[j] * projection8[.., j]`.
    pub projection8: Vec<i8>,
    /// Per-column scales of `projection8`.
    pub column_scale: Vec<f32>,
    /// Input dimension.
    pub dim: usize,
    /// Output dimension.
    pub n_components: usize,
    use_avx2: bool,
}

impl PcaProjection {
    /// Fit on up to `sample_size` rows of `data`; None when the requested
    /// components exceed [`MAX_PROJECTED_FRACTION`] of the features.
    pub fn fit(data: &[f32], dim: usize, dims: ProjectionDims, sample_size: usize, seed: u64) -> Option<Self> {
        assert!(dim > 0 && data.len() % dim == 0);
        let n = data.len() / dim;
        if n == 0 {
            return None;
        }
        let rows: Vec<usize> = if n <= sample_size {
            (0..n).collect()
        } else {
            let mut rng = crate::rng::FastRng::new(seed);
            (0..sample_size).map(|_| rng.next_index(n)).collect()
        };
        let mut mean = vec![0.0f64; dim];
        for &row in &rows {
            for (total, &value) in mean.iter_mut().zip(&data[row * dim..(row + 1) * dim]) {
                *total += f64::from(value);
            }
        }
        for total in &mut mean {
            *total /= rows.len() as f64;
        }
        let mut covariance = vec![0.0f64; dim * dim];
        let mut centred = vec![0.0f64; dim];
        for &row in &rows {
            for ((out, &value), &mu) in centred.iter_mut().zip(&data[row * dim..(row + 1) * dim]).zip(&mean) {
                *out = f64::from(value) - mu;
            }
            for i in 0..dim {
                let xi = centred[i];
                for (total, &xj) in covariance[i * dim + i..(i + 1) * dim].iter_mut().zip(&centred[i..]) {
                    *total += xi * xj;
                }
            }
        }
        for i in 0..dim {
            for j in 0..i {
                covariance[i * dim + j] = covariance[j * dim + i];
            }
        }
        let (values, vectors) = crate::linalg::symmetric_eigen(&covariance, dim);
        let n_components = match dims {
            ProjectionDims::Components(count) => count.min(dim),
            ProjectionDims::VarianceFraction(fraction) => {
                let total: f64 = values.iter().map(|&value| value.max(0.0)).sum();
                if total <= 0.0 {
                    dim
                } else {
                    let mut explained = 0.0;
                    let mut count = dim;
                    for (index, &value) in values.iter().enumerate() {
                        explained += value.max(0.0);
                        if explained / total >= fraction {
                            count = index + 1;
                            break;
                        }
                    }
                    count
                }
            }
        };
        if n_components == 0 || n_components as f64 > MAX_PROJECTED_FRACTION * dim as f64 {
            return None;
        }
        let mut column_scale = vec![0.0f32; n_components];
        for (j, scale) in column_scale.iter_mut().enumerate() {
            let max_abs = vectors[j * dim..(j + 1) * dim].iter().fold(0.0f64, |m, &v| m.max(v.abs()));
            *scale = if max_abs > 0.0 { (max_abs / 127.0) as f32 } else { 1.0 };
        }
        let mut projection8 = vec![0i8; dim * n_components];
        for i in 0..dim {
            for j in 0..n_components {
                let value = vectors[j * dim + i] / f64::from(column_scale[j]);
                projection8[i * n_components + j] = value.round().clamp(-127.0, 127.0) as i8;
            }
        }
        Some(Self {
            mean: mean.into_iter().map(|value| value as f32).collect(),
            projection8,
            column_scale,
            dim,
            n_components,
            use_avx2: avx2_available(),
        })
    }

    /// `(x - mean) P` for every row of `data`, `n_components` values per row.
    pub fn project_rows(&self, data: &[f32]) -> Vec<f32> {
        let projection: Vec<f32> = self
            .projection8
            .chunks_exact(self.n_components)
            .flat_map(|row| row.iter().zip(&self.column_scale).map(|(&value, &scale)| f32::from(value) * scale))
            .collect();
        let n = data.len() / self.dim;
        let mut out = vec![0.0f32; n * self.n_components];
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 {
                unsafe { project_rows_avx2(data, &self.mean, &projection, self.dim, self.n_components, &mut out) };
                return out;
            }
        }
        project_rows_body::<false>(data, &self.mean, &projection, self.dim, self.n_components, &mut out);
        out
    }

    /// Project `query` into `out` for codes approximating `kind`; returns the
    /// dot product shift (`<q, mean>` for [`AffineKind::Dot`], else zero).
    pub fn project_query(&self, kind: AffineKind, query: &[f32], out: &mut Vec<f32>) -> f32 {
        assert_eq!(query.len(), self.dim);
        out.clear();
        out.resize(self.n_components, 0.0);
        let mut shift = 0.0f32;
        let centred: Vec<f32> = match kind {
            AffineKind::SquaredEuclidean => query.iter().zip(&self.mean).map(|(&q, &m)| q - m).collect(),
            AffineKind::Dot => {
                shift = query.iter().zip(&self.mean).map(|(&q, &m)| q * m).sum();
                query.to_vec()
            }
            AffineKind::Mips { .. } => panic!("projected codes do not support mips"),
        };
        #[cfg(target_arch = "x86_64")]
        {
            if self.use_avx2 {
                unsafe { project_query_avx2(&centred, &self.projection8, self.n_components, out) };
            } else {
                project_query_body::<false>(&centred, &self.projection8, self.n_components, out);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        project_query_body::<false>(&centred, &self.projection8, self.n_components, out);
        for (value, &scale) in out.iter_mut().zip(&self.column_scale) {
            *value *= scale;
        }
        shift
    }
}

/// Rows of `(data - mean) @ projection`, eight rows at a time so that each
/// projection row is read once per block. Written for auto-vectorization of
/// the inner loop over output columns.
#[inline(always)]
fn project_rows_body<const FUSED: bool>(data: &[f32], mean: &[f32], projection: &[f32], dim: usize, m: usize, out: &mut [f32]) {
    const BLOCK: usize = 8;
    let n = data.len() / dim;
    let mut centred = vec![0.0f32; BLOCK * dim];
    for start in (0..n).step_by(BLOCK) {
        let rows = BLOCK.min(n - start);
        for r in 0..rows {
            let row = &data[(start + r) * dim..(start + r + 1) * dim];
            for ((c, &x), &mu) in centred[r * dim..(r + 1) * dim].iter_mut().zip(row).zip(mean) {
                *c = x - mu;
            }
        }
        let block = &mut out[start * m..(start + rows) * m];
        for i in 0..dim {
            let p = &projection[i * m..(i + 1) * m];
            for r in 0..rows {
                let x = centred[r * dim + i];
                for (o, &pj) in block[r * m..(r + 1) * m].iter_mut().zip(p) {
                    *o = if FUSED { x.mul_add(pj, *o) } else { *o + x * pj };
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn project_rows_avx2(data: &[f32], mean: &[f32], projection: &[f32], dim: usize, m: usize, out: &mut [f32]) {
    project_rows_body::<true>(data, mean, projection, dim, m, out);
}

#[inline(always)]
fn project_query_body<const FUSED: bool>(centred: &[f32], projection8: &[i8], m: usize, out: &mut [f32]) {
    for (i, &x) in centred.iter().enumerate() {
        for (o, &p) in out.iter_mut().zip(&projection8[i * m..(i + 1) * m]) {
            *o = if FUSED { x.mul_add(f32::from(p), *o) } else { *o + x * f32::from(p) };
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn project_query_avx2(centred: &[f32], projection8: &[i8], m: usize, out: &mut [f32]) {
    project_query_body::<true>(centred, projection8, m, out);
}

/// Default 4-bit clipping: each feature's codes span its 0.3% to 99.7% quantiles.
pub const DEFAULT_SQ4_CLIP_QUANTILE: f32 = 0.003;

/// Rows sampled to find clipped feature ranges.
const CLIP_SAMPLE_ROWS: usize = 100_000;

/// Affine 8-bit or 4-bit codes for a set of vectors.
#[derive(Clone, Debug)]
pub struct AffineCodes {
    /// Code rows, `stride` bytes each (zero padded), 64-byte aligned.
    codes: AlignedBytes,
    /// Bytes per code row: a multiple of 64.
    stride: usize,
    /// Per-feature offsets.
    pub offsets: Vec<f32>,
    /// Per-feature scales.
    pub scales: Vec<f32>,
    /// Number of features per code.
    pub dim: usize,
    /// Bits per feature code (8 or 4).
    pub bits: u32,
    /// Mean code at each (nibble, byte) position, laid out like the weights.
    means: Vec<f32>,
    /// Per row `|scale * c|^2` (squared euclidean codes only).
    norms: Vec<f32>,
    n_rows: usize,
    kind: AffineKind,
    kernel: Kernel,
}

impl AffineCodes {
    /// Encode `data` (`dim` values per row) with 8-bit codes over each
    /// feature's full range.
    pub fn encode(data: &[f32], dim: usize, kind: AffineKind) -> Self {
        Self::encode_bits(data, dim, kind, 8, None)
    }

    /// Encode `data` with `bits`-bit codes (8 or 4). With `clip_quantile` the
    /// code range of each feature runs between that quantile and its
    /// complement (of a sample of rows) instead of the full min to max.
    pub fn encode_bits(
        data: &[f32],
        dim: usize,
        kind: AffineKind,
        bits: u32,
        clip_quantile: Option<f32>,
    ) -> Self {
        Self::encode_columns(data, dim, dim, kind, bits, clip_quantile)
    }

    /// As [`Self::encode_bits`] for the first `dim` values of each
    /// `row_stride`-value row of `data` (e.g. without the augmenting feature
    /// of [`AffineKind::Mips`] rows).
    pub fn encode_columns(
        data: &[f32],
        row_stride: usize,
        dim: usize,
        kind: AffineKind,
        bits: u32,
        clip_quantile: Option<f32>,
    ) -> Self {
        assert!(dim > 0 && dim <= row_stride && data.len() % row_stride == 0);
        let (low, high) = match clip_quantile {
            Some(q) if q > 0.0 => clipped_ranges(data, row_stride, dim, q),
            _ => full_ranges(data, row_stride, dim),
        };
        Self::encode_ranges(data, row_stride, dim, kind, bits, &low, &high)
    }

    /// 8-bit codes of `data` (`dim` values per row) for building a k-NN graph:
    /// each feature's codes span its full range, except for features with
    /// outliers (a full range over [`BUILD_CLIP_RATIO`] times the range between
    /// the [`BUILD_CLIP_QUANTILE`] quantiles), which span those quantiles. A few
    /// extreme values would otherwise squeeze nearly every row into a handful of
    /// codes; genuine tails are kept, as clamping them costs recall.
    pub(crate) fn encode_for_build(data: &[f32], dim: usize, kind: AffineKind) -> Self {
        assert!(dim > 0 && data.len() % dim == 0);
        let (low, high) = outlier_clipped_ranges(data, dim, dim);
        Self::encode_ranges(data, dim, dim, kind, 8, &low, &high)
    }

    fn encode_ranges(
        data: &[f32],
        row_stride: usize,
        dim: usize,
        kind: AffineKind,
        bits: u32,
        low: &[f32],
        high: &[f32],
    ) -> Self {
        use rayon::prelude::*;
        assert!(bits == 8 || bits == 4, "codes have 8 or 4 bits");
        let n_rows = data.len() / row_stride;
        let levels = ((1u32 << bits) - 1) as f32;
        let scales: Vec<f32> = low
            .iter()
            .zip(high)
            .map(|(&lo, &hi)| {
                let scale = (hi - lo) / levels;
                if scale > 0.0 && scale.is_finite() {
                    scale
                } else {
                    1.0
                }
            })
            .collect();
        let offsets: Vec<f32> = low.iter().map(|&lo| if lo.is_finite() { lo } else { 0.0 }).collect();
        let inverse: Vec<f32> = scales.iter().map(|&scale| scale.recip()).collect();

        let code_bytes = if bits == 8 { dim } else { dim.div_ceil(2) };
        let stride = code_bytes.div_ceil(LINE) * LINE;
        let halves = if bits == 8 { 1 } else { 2 };
        let mut codes = AlignedBytes::zeroed(n_rows * stride);
        let squares: Vec<f32> = scales.iter().map(|&s| s * s).collect();
        let row_norms: Vec<f32> = codes
            .as_mut_slice()
            .par_chunks_mut(stride)
            .zip(data.par_chunks_exact(row_stride))
            .with_min_len(256)
            .map(|(out, row)| {
                if bits == 8 {
                    return encode_row_u8(&row[..dim], &offsets, &inverse, &squares, &mut out[..dim]);
                }
                let mut norm = 0.0f32;
                for (j, &value) in row[..dim].iter().enumerate() {
                    let code = ((value - offsets[j]) * inverse[j]).clamp(0.0, levels);
                    let code = (code + 0.5) as i32 as f32;
                    norm += squares[j] * code * code;
                    out[j >> 1] |= (code as u8) << (4 * (j & 1));
                }
                norm
            })
            .collect();
        let norms = if kind == AffineKind::SquaredEuclidean { row_norms } else { Vec::new() };
        // Mean code at each (nibble, byte) position, from exact integer sums.
        let width = halves * stride;
        let sums = codes
            .as_slice()
            .par_chunks(stride)
            .with_min_len(256)
            .fold(
                || vec![0u64; width],
                |mut sums, row| {
                    if bits == 8 {
                        add_codes(&mut sums, row);
                    } else {
                        for (byte, &code) in row.iter().enumerate() {
                            sums[byte] += u64::from(code & 15);
                            sums[stride + byte] += u64::from(code >> 4);
                        }
                    }
                    sums
                },
            )
            .reduce(
                || vec![0u64; width],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            );
        let means = sums.iter().map(|&sum| (sum as f64 / n_rows.max(1) as f64) as f32).collect();
        Self {
            codes,
            stride,
            offsets,
            scales,
            dim,
            bits,
            means,
            norms,
            n_rows,
            kind,
            kernel: Kernel::detect(),
        }
    }

    /// Each coded row of `data` (`dim` values per row) prepared as a query
    /// against these 8-bit codes, as [`Self::prepare_query`] prepares one, after
    /// clamping it to the code range (an out-of-range value would otherwise set
    /// the int8 step and round every other weight of the row to zero). Returns
    /// the int8 weights, in rows of [`Self::stride`] bytes, and a record per row
    /// that also holds the row's code norm, so that the distance between rows
    /// `p` and `q` is `offset_p + scale_p <w_p, c_q> + norm_q`.
    pub(crate) fn row_queries(&self, data: &[f32]) -> (AlignedBytes, Vec<RowQuery>) {
        use rayon::prelude::*;
        assert_eq!(self.bits, 8, "row queries need 8-bit codes");
        assert_eq!(data.len(), self.n_rows * self.dim);
        let (dim, stride) = (self.dim, self.stride);
        let high: Vec<f32> = self
            .offsets
            .iter()
            .zip(&self.scales)
            .map(|(&offset, &scale)| offset + 255.0 * scale)
            .collect();
        let mut weights = AlignedBytes::zeroed(self.n_rows * stride);
        let mut records = vec![RowQuery::default(); self.n_rows];
        weights
            .as_mut_i8_slice()
            .par_chunks_mut(stride)
            .zip(data.par_chunks_exact(dim))
            .zip(records.par_iter_mut())
            .enumerate()
            .with_min_len(256)
            .for_each_init(
                || vec![0.0f32; dim],
                |u, (index, ((weights, row), record))| {
                    let squared_euclidean = self.kind == AffineKind::SquaredEuclidean;
                    let (total, correction, step) = row_query_weights_i8(
                        row,
                        &self.offsets,
                        &self.scales,
                        &high,
                        &self.means[..dim],
                        squared_euclidean,
                        u,
                        &mut weights[..dim],
                    );
                    let (offset, scale) = if self.kind == AffineKind::SquaredEuclidean {
                        (total + 2.0 * correction, -2.0 * step)
                    } else {
                        (1.0 - total + correction, -step)
                    };
                    let norm = self.norms.get(index).copied().unwrap_or(0.0);
                    *record = RowQuery { norm, offset, scale, _pad: 0.0 };
                },
            );
        (weights, records)
    }

    /// The code dot product kernel.
    pub(crate) fn kernel(&self) -> Kernel {
        self.kernel
    }

    /// Each feature's value at code 0.
    pub(crate) fn offsets(&self) -> &[f32] {
        &self.offsets
    }

    /// Each feature's value per code step.
    pub(crate) fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// The approximated distance.
    pub fn kind(&self) -> AffineKind {
        self.kind
    }

    /// Bytes per (padded) code row.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Bytes of code storage, including per-row norms.
    pub fn storage_bytes(&self) -> usize {
        self.codes.len() + self.norms.len() * core::mem::size_of::<f32>()
    }

    /// The (padded) code row of vector `index`.
    #[inline(always)]
    pub fn row(&self, index: i32) -> &[u8] {
        let start = index as usize * self.stride;
        &self.codes.as_slice()[start..start + self.stride]
    }

    /// Hint the CPU to fetch what the code distance of `index` reads: up to
    /// `max_lines` lines of its row (and its norm).
    #[inline(always)]
    pub fn prefetch(&self, index: i32, max_lines: usize) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            if index < 0 || index as usize >= self.n_rows {
                return;
            }
            let base = self.codes.as_slice().as_ptr().add(index as usize * self.stride) as *const i8;
            let lines = (self.stride / LINE).min(max_lines);
            for line in 0..lines {
                _mm_prefetch(base.add(line * LINE), _MM_HINT_T0);
            }
            if !self.norms.is_empty() {
                _mm_prefetch(self.norms.as_ptr().add(index as usize) as *const i8, _MM_HINT_T0);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (index, max_lines);
    }

    /// Transform `query` (in the coded space) into `out`.
    pub fn prepare_query(&self, query: &[f32], out: &mut AffineQuery) {
        assert_eq!(query.len(), self.dim);
        let halves = if self.bits == 8 { 1 } else { 2 };
        let width = halves * self.stride;
        out.u.clear();
        out.u.resize(width, 0.0);
        let mut total = 0.0f32;
        let mut query_norm2 = 0.0f32;
        for (j, &q) in query.iter().enumerate() {
            let position = if self.bits == 8 { j } else { (j & 1) * self.stride + (j >> 1) };
            let (offset, scale) = (self.offsets[j], self.scales[j]);
            match self.kind {
                AffineKind::SquaredEuclidean => {
                    let e = q - offset;
                    total += e * e;
                    out.u[position] = e * scale;
                }
                AffineKind::Dot => {
                    total += q * offset;
                    out.u[position] = q * scale;
                }
                AffineKind::Mips { .. } => {
                    total += q * offset;
                    query_norm2 += q * q;
                    out.u[position] = q * scale;
                }
            }
        }
        let max_abs = out.u.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let step = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inverse = step.recip();
        if out.weights.len() != width {
            out.weights.reset_zeroed(width);
        }
        let weights = out.weights.as_mut_i8_slice();
        let mut correction = 0.0f32;
        for ((w, &u), &mean) in weights.iter_mut().zip(&out.u).zip(&self.means) {
            let rounded = (u * inverse).round();
            *w = rounded as i8;
            correction += (step * rounded - u) * mean;
        }
        match self.kind {
            AffineKind::SquaredEuclidean => {
                out.offset = total + 2.0 * correction;
                out.scale = -2.0 * step;
            }
            AffineKind::Dot => {
                out.offset = 1.0 - total + correction;
                out.scale = -step;
            }
            AffineKind::Mips { norm2 } => {
                out.offset = query_norm2 + norm2 - 2.0 * total + 2.0 * correction;
                out.scale = -2.0 * step;
            }
        }
    }

    /// Code distance from a prepared query to vector `index`.
    #[inline(always)]
    pub fn distance(&self, query: &AffineQuery, index: i32) -> f32 {
        let row = self.row(index);
        let weights = query.weights.as_i8_slice();
        let dot = if self.bits == 8 {
            dot_u8(self.kernel, row, weights)
        } else {
            dot_u4(self.kernel, row, weights)
        };
        let mut d = query.offset + query.scale * dot as f32;
        if let Some(&norm) = self.norms.get(index as usize) {
            d += norm;
        }
        d
    }

    /// Code distances from a prepared query to four vectors.
    #[inline(always)]
    pub fn distance_four(&self, query: &AffineQuery, indices: [i32; 4]) -> [f32; 4] {
        let rows = indices.map(|index| self.row(index));
        let weights = query.weights.as_i8_slice();
        let dots = if self.bits == 8 {
            dot4_u8(self.kernel, rows, weights)
        } else {
            dot4_u4(self.kernel, rows, weights)
        };
        let mut out = [0.0f32; 4];
        for slot in 0..4 {
            out[slot] = query.offset + query.scale * dots[slot] as f32;
            if !self.norms.is_empty() {
                out[slot] += self.norms[indices[slot] as usize];
            }
        }
        out
    }
}

/// Lanes of the row kernels below: elementwise work over 16-value chunks and
/// sums in 16 independent accumulators vectorize (a single running sum, a
/// serial chain, would not). So that they vectorize, the kernels also clamp by
/// comparison (`f32::max` and `f32::min` handle NaN by branching) and round with
/// [`ROUND_MAGIC`] (`f32::round` is a libm call on baseline x86-64, and an `as`
/// cast to an integer saturates element by element).
const ROW_LANES: usize = 16;

/// `1.5 * 2^23`: for `|v| < 2^22`, `v + ROUND_MAGIC` is rounded (half to even)
/// to an integer whose low bits are those of the float's bits, and subtracting
/// `ROUND_MAGIC` again gives the rounded value.
const ROUND_MAGIC: f32 = 12_582_912.0;

/// `v` clamped to `low..=high`; NaN becomes `low`.
#[inline(always)]
fn clamp_select(v: f32, low: f32, high: f32) -> f32 {
    let v = if v > low { v } else { low };
    if v < high {
        v
    } else {
        high
    }
}

/// 8-bit codes of `row` into `out` (rounded half up); returns `|scale * c|^2`.
fn encode_row_u8(row: &[f32], offsets: &[f32], inverse: &[f32], squares: &[f32], out: &mut [u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: AVX2 and FMA are available.
        return unsafe { encode_row_u8_avx2(row, offsets, inverse, squares, out) };
    }
    encode_row_u8_body(row, offsets, inverse, squares, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn encode_row_u8_avx2(row: &[f32], offsets: &[f32], inverse: &[f32], squares: &[f32], out: &mut [u8]) -> f32 {
    encode_row_u8_body(row, offsets, inverse, squares, out)
}

#[inline(always)]
fn encode_row_u8_body(row: &[f32], offsets: &[f32], inverse: &[f32], squares: &[f32], out: &mut [u8]) -> f32 {
    let mut lanes = [0.0f32; ROW_LANES];
    let whole = row.len() / ROW_LANES * ROW_LANES;
    for start in (0..whole).step_by(ROW_LANES) {
        let end = start + ROW_LANES;
        let (row, offsets, inverse) = (&row[start..end], &offsets[start..end], &inverse[start..end]);
        let (squares, out) = (&squares[start..end], &mut out[start..end]);
        for t in 0..ROW_LANES {
            let (code, rounded) = code_u8(row[t], offsets[t], inverse[t]);
            out[t] = code;
            lanes[t] += squares[t] * rounded * rounded;
        }
    }
    let mut norm: f32 = lanes.iter().sum();
    for j in whole..row.len() {
        let (code, rounded) = code_u8(row[j], offsets[j], inverse[j]);
        out[j] = code;
        norm += squares[j] * rounded * rounded;
    }
    norm
}

/// The 8-bit code of `x`, as a byte and a float.
#[inline(always)]
fn code_u8(x: f32, offset: f32, inverse: f32) -> (u8, f32) {
    let shifted = clamp_select((x - offset) * inverse, 0.0, 255.0) + ROUND_MAGIC;
    (shifted.to_bits() as u8, shifted - ROUND_MAGIC)
}

/// `row` as int8 query `weights` for 8-bit codes, via [`row_query_u8`] and
/// [`round_weights_i8`] (`u` is scratch); returns the data term, the rounding
/// correction and the weight step.
#[allow(clippy::too_many_arguments)]
fn row_query_weights_i8(
    row: &[f32],
    offsets: &[f32],
    scales: &[f32],
    high: &[f32],
    means: &[f32],
    squared_euclidean: bool,
    u: &mut [f32],
    weights: &mut [i8],
) -> (f32, f32, f32) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: AVX2 and FMA are available.
        return unsafe { row_query_weights_i8_avx2(row, offsets, scales, high, means, squared_euclidean, u, weights) };
    }
    row_query_weights_i8_body(row, offsets, scales, high, means, squared_euclidean, u, weights)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn row_query_weights_i8_avx2(
    row: &[f32],
    offsets: &[f32],
    scales: &[f32],
    high: &[f32],
    means: &[f32],
    squared_euclidean: bool,
    u: &mut [f32],
    weights: &mut [i8],
) -> (f32, f32, f32) {
    row_query_weights_i8_body(row, offsets, scales, high, means, squared_euclidean, u, weights)
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn row_query_weights_i8_body(
    row: &[f32],
    offsets: &[f32],
    scales: &[f32],
    high: &[f32],
    means: &[f32],
    squared_euclidean: bool,
    u: &mut [f32],
    weights: &mut [i8],
) -> (f32, f32, f32) {
    let (total, max_abs) = row_query_u8(row, offsets, scales, high, squared_euclidean, u);
    let step = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    (total, round_weights_i8(u, step, means, weights), step)
}

/// `row` (clamped to `offsets..=high`) as unrounded query weights `u` for
/// 8-bit codes; returns the query constant's data term (`|x - offset|^2`, or
/// `<x, offset>` for dot codes) and `max |u|`.
#[inline(always)]
fn row_query_u8(
    row: &[f32],
    offsets: &[f32],
    scales: &[f32],
    high: &[f32],
    squared_euclidean: bool,
    u: &mut [f32],
) -> (f32, f32) {
    let mut totals = [0.0f32; ROW_LANES];
    let mut maxima = [0.0f32; ROW_LANES];
    let whole = row.len() / ROW_LANES * ROW_LANES;
    let one = |x: f32, offset: f32, scale: f32, high: f32, u: &mut f32, total: &mut f32, max: &mut f32| {
        let x = clamp_select(x, offset, high);
        let (term, weight) = if squared_euclidean {
            let e = x - offset;
            (e * e, e * scale)
        } else {
            (x * offset, x * scale)
        };
        *total += term;
        *u = weight;
        let magnitude = weight.abs();
        *max = if magnitude > *max { magnitude } else { *max };
    };
    for start in (0..whole).step_by(ROW_LANES) {
        let end = start + ROW_LANES;
        let (row, offsets, scales) = (&row[start..end], &offsets[start..end], &scales[start..end]);
        let (high, u) = (&high[start..end], &mut u[start..end]);
        for t in 0..ROW_LANES {
            one(row[t], offsets[t], scales[t], high[t], &mut u[t], &mut totals[t], &mut maxima[t]);
        }
    }
    let (mut total, mut max) = (0.0f32, 0.0f32);
    for j in whole..row.len() {
        one(row[j], offsets[j], scales[j], high[j], &mut u[j], &mut total, &mut max);
    }
    (total + totals.iter().sum::<f32>(), maxima.iter().fold(max, |m, &v| m.max(v)))
}

/// Round `u / step` to int8 `weights` (half to even, clamped to +-127);
/// returns the rounding's expected effect `sum_j (step w_j - u_j) mean_j`.
#[inline(always)]
fn round_weights_i8(u: &[f32], step: f32, means: &[f32], weights: &mut [i8]) -> f32 {
    let inverse = step.recip();
    let round = |u: f32| {
        let shifted = clamp_select(u * inverse, -127.0, 127.0) + ROUND_MAGIC;
        (shifted.to_bits() as i8, shifted - ROUND_MAGIC)
    };
    let mut lanes = [0.0f32; ROW_LANES];
    let whole = u.len() / ROW_LANES * ROW_LANES;
    for start in (0..whole).step_by(ROW_LANES) {
        let end = start + ROW_LANES;
        let (u, means, weights) = (&u[start..end], &means[start..end], &mut weights[start..end]);
        for t in 0..ROW_LANES {
            let (weight, rounded) = round(u[t]);
            weights[t] = weight;
            lanes[t] += (step * rounded - u[t]) * means[t];
        }
    }
    let mut correction: f32 = lanes.iter().sum();
    for j in whole..u.len() {
        let (weight, rounded) = round(u[j]);
        weights[j] = weight;
        correction += (step * rounded - u[j]) * means[j];
    }
    correction
}

/// `u * inverse` rounded (half to even, clamped to +-127) to int8 `weights`.
pub(crate) fn quantize_i8(u: &[f32], inverse: f32, weights: &mut [i8]) {
    for (weight, &value) in weights.iter_mut().zip(u) {
        *weight = (clamp_select(value * inverse, -127.0, 127.0) + ROUND_MAGIC).to_bits() as i8;
    }
}

/// Add each code byte of `row` to its running sum.
fn add_codes(sums: &mut [u64], row: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: AVX2 is available.
        return unsafe { add_codes_avx2(sums, row) };
    }
    add_codes_body(sums, row)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn add_codes_avx2(sums: &mut [u64], row: &[u8]) {
    add_codes_body(sums, row)
}

#[inline(always)]
fn add_codes_body(sums: &mut [u64], row: &[u8]) {
    sums.iter_mut().zip(row).for_each(|(sum, &code)| *sum += u64::from(code));
}

fn full_ranges(data: &[f32], row_stride: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    let empty = || (vec![f32::INFINITY; dim], vec![f32::NEG_INFINITY; dim]);
    data.par_chunks_exact(row_stride)
        .with_min_len(1024)
        .fold(empty, |(mut low, mut high), row| {
            for ((lo, hi), &value) in low.iter_mut().zip(high.iter_mut()).zip(&row[..dim]) {
                *lo = lo.min(value);
                *hi = hi.max(value);
            }
            (low, high)
        })
        .reduce(empty, |(mut low, mut high), (other_low, other_high)| {
            low.iter_mut().zip(other_low).for_each(|(lo, other)| *lo = lo.min(other));
            high.iter_mut().zip(other_high).for_each(|(hi, other)| *hi = hi.max(other));
            (low, high)
        })
}

/// Per-feature `q` and `1 - q` quantiles of (an evenly spaced sample of) the rows.
fn clipped_ranges(data: &[f32], row_stride: usize, dim: usize, q: f32) -> (Vec<f32>, Vec<f32>) {
    quantile_ranges(data, row_stride, dim, q, CLIP_SAMPLE_ROWS)
}

/// Per-feature `q` and `1 - q` quantiles of at most `max_rows` evenly spaced
/// rows. Columns are gathered a cache line (16 features) at a time.
fn quantile_ranges(
    data: &[f32],
    row_stride: usize,
    dim: usize,
    q: f32,
    max_rows: usize,
) -> (Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    const BLOCK: usize = LINE / 4;
    let n_rows = data.len() / row_stride;
    let step = n_rows.div_ceil(max_rows).max(1);
    let sample: Vec<usize> = (0..n_rows).step_by(step).collect();
    let n = sample.len();
    if n == 0 {
        return (vec![0.0; dim], vec![0.0; dim]);
    }
    let low_rank = ((q as f64) * (n - 1) as f64).round() as usize;
    let high_rank = (((1.0 - q as f64) * (n - 1) as f64).round() as usize).max(low_rank);
    let ranges: Vec<(f32, f32)> = (0..dim.div_ceil(BLOCK))
        .into_par_iter()
        .flat_map_iter(|block| {
            let first = block * BLOCK;
            let width = BLOCK.min(dim - first);
            let mut columns = vec![0.0f32; width * n];
            for (position, &row) in sample.iter().enumerate() {
                let values = &data[row * row_stride + first..row * row_stride + first + width];
                for (offset, &value) in values.iter().enumerate() {
                    columns[offset * n + position] = value;
                }
            }
            columns
                .chunks_exact_mut(n)
                .map(|column| {
                    let (_, &mut hi, _) = column.select_nth_unstable_by(high_rank, f32::total_cmp);
                    let (_, &mut lo, _) = column.select_nth_unstable_by(low_rank, f32::total_cmp);
                    (lo, hi)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    ranges.into_iter().unzip()
}

/// Quantile clipped by the build codes (see [`AffineCodes::encode_for_build`]).
pub const BUILD_CLIP_QUANTILE: f32 = 1e-3;
/// A feature is clipped only when its full range is this many times its
/// quantile range: an outlier, not a tail (a Gaussian feature's full range is
/// about 1.5 times its 0.1%-99.9% range).
pub const BUILD_CLIP_RATIO: f32 = 4.0;
const BUILD_CLIP_SAMPLE_ROWS: usize = 50_000;

fn outlier_clipped_ranges(data: &[f32], row_stride: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut low, mut high) = full_ranges(data, row_stride, dim);
    if data.len() / row_stride <= 1000 {
        return (low, high);
    }
    let (quantile_low, quantile_high) =
        quantile_ranges(data, row_stride, dim, BUILD_CLIP_QUANTILE, BUILD_CLIP_SAMPLE_ROWS);
    for j in 0..dim {
        let quantile_range = quantile_high[j] - quantile_low[j];
        if quantile_range > 0.0 && high[j] - low[j] > BUILD_CLIP_RATIO * quantile_range {
            low[j] = quantile_low[j];
            high[j] = quantile_high[j];
        }
    }
    (low, high)
}

/// A row prepared as a query against 8-bit codes (see
/// [`AffineCodes::row_queries`]): one aligned 16-byte record, so a distance
/// reads a single cache line of scalars per row.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
pub(crate) struct RowQuery {
    /// `|scale * c|^2` of the row's codes (squared euclidean codes only).
    pub norm: f32,
    pub offset: f32,
    pub scale: f32,
    _pad: f32,
}

/// A float16 copy of the data for reranking: `data / scale`, with `scale` the
/// largest absolute value, so every stored value lies in [-1, 1] and nothing
/// overflows. Rows are padded to whole 64-byte lines and stored aligned.
#[derive(Clone, Debug)]
pub struct HalfRows {
    bytes: AlignedBytes,
    /// Values per (padded) row: a multiple of 32.
    stride: usize,
    /// Features per row.
    pub dim: usize,
    /// The data's largest absolute value.
    pub scale: f32,
    n_rows: usize,
    kind: AffineKind,
    simd: HalfSimd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HalfSimd {
    Avx512,
    F16c,
    Portable,
}

/// A query prepared for [`HalfRows`] distances.
#[derive(Clone, Debug, Default)]
pub struct HalfQuery {
    /// The scaled query, padded to a row's length.
    values: Vec<f32>,
    /// `|q|^2 + norm2` for [`AffineKind::Mips`], else zero.
    offset: f32,
}

impl HalfRows {
    /// Store `data` (`dim` values per row) as float16 for `kind` distances.
    pub fn encode(data: &[f32], dim: usize, kind: AffineKind) -> Self {
        Self::encode_columns(data, dim, dim, kind)
    }

    /// As [`Self::encode`] for the first `dim` values of each
    /// `row_stride`-value row of `data`.
    pub fn encode_columns(data: &[f32], row_stride: usize, dim: usize, kind: AffineKind) -> Self {
        assert!(dim > 0 && dim <= row_stride && data.len() % row_stride == 0);
        let n_rows = data.len() / row_stride;
        let max_abs = data
            .chunks_exact(row_stride)
            .flat_map(|row| &row[..dim])
            .fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs > 0.0 && max_abs.is_finite() { max_abs } else { 1.0 };
        let inverse = scale.recip();
        let stride = dim.div_ceil(LINE / 2) * (LINE / 2);
        let mut bytes = AlignedBytes::zeroed(n_rows * stride * 2);
        {
            let values = half_slice_mut(&mut bytes);
            for (row_index, row) in data.chunks_exact(row_stride).enumerate() {
                let out = &mut values[row_index * stride..row_index * stride + dim];
                for (o, &v) in out.iter_mut().zip(&row[..dim]) {
                    *o = f32_to_f16_bits(v * inverse);
                }
            }
        }
        Self {
            bytes,
            stride,
            dim,
            scale,
            n_rows,
            kind,
            simd: HalfSimd::detect(),
        }
    }

    /// Bytes of storage.
    pub fn storage_bytes(&self) -> usize {
        self.bytes.len()
    }

    #[inline(always)]
    fn row(&self, index: i32) -> &[u16] {
        let start = index as usize * self.stride;
        &half_slice(&self.bytes)[start..start + self.stride]
    }

    /// Hint the CPU to fetch row `index`.
    #[inline(always)]
    pub fn prefetch(&self, index: i32) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            if index < 0 || index as usize >= self.n_rows {
                return;
            }
            let base = self.bytes.as_slice().as_ptr().add(index as usize * self.stride * 2) as *const i8;
            for line in 0..(self.stride * 2) / LINE {
                _mm_prefetch(base.add(line * LINE), _MM_HINT_T0);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = index;
    }

    /// Prepare `query` (`dim` values) for [`Self::distance_four`].
    pub fn prepare_query(&self, query: &[f32], out: &mut HalfQuery) {
        assert_eq!(query.len(), self.dim);
        out.values.clear();
        out.values.resize(self.stride, 0.0);
        out.offset = 0.0;
        // |q - s y|^2 = s^2 |q / s - y|^2 and <q, s y> = <s q, y>.
        let factor = match self.kind {
            AffineKind::SquaredEuclidean => self.scale.recip(),
            AffineKind::Dot => self.scale,
            AffineKind::Mips { norm2 } => {
                out.offset = norm2 + query.iter().map(|&q| q * q).sum::<f32>();
                self.scale
            }
        };
        for (o, &q) in out.values.iter_mut().zip(query) {
            *o = q * factor;
        }
    }

    #[inline(always)]
    fn finish(&self, sum: f32, offset: f32) -> f32 {
        match self.kind {
            AffineKind::SquaredEuclidean => sum * self.scale * self.scale,
            AffineKind::Dot => 1.0 - sum.clamp(-1.0, 1.0),
            AffineKind::Mips { .. } => offset - 2.0 * sum,
        }
    }

    /// Distances (squared euclidean, `1 - dot` clamped like
    /// `DirectNormalizedCosine`, or the [`AffineKind::Mips`] distance) from a
    /// prepared query to four rows.
    #[inline(always)]
    pub fn distance_four(&self, query: &HalfQuery, indices: [i32; 4]) -> [f32; 4] {
        let rows = indices.map(|index| self.row(index));
        let values = query.values.as_slice();
        let sums = match self.simd {
            #[cfg(target_arch = "x86_64")]
            HalfSimd::Avx512 => unsafe { half_sums_avx512(self.kind, values, rows) },
            #[cfg(target_arch = "x86_64")]
            HalfSimd::F16c => unsafe { half_sums_f16c(self.kind, values, rows) },
            _ => rows.map(|row| half_sum_portable(self.kind, values, row)),
        };
        sums.map(|sum| self.finish(sum, query.offset))
    }

    /// The distance from a prepared query to row `index`.
    #[inline(always)]
    pub fn distance(&self, query: &HalfQuery, index: i32) -> f32 {
        self.distance_four(query, [index; 4])[0]
    }
}

impl HalfSimd {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return HalfSimd::Avx512;
            }
            if is_x86_feature_detected!("f16c") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return HalfSimd::F16c;
            }
        }
        HalfSimd::Portable
    }
}

#[inline(always)]
fn half_slice(bytes: &AlignedBytes) -> &[u16] {
    let raw = bytes.as_slice();
    // 64-byte aligned, so u16 aligned.
    unsafe { core::slice::from_raw_parts(raw.as_ptr() as *const u16, raw.len() / 2) }
}

#[inline(always)]
fn half_slice_mut(bytes: &mut AlignedBytes) -> &mut [u16] {
    let raw = bytes.as_mut_slice();
    unsafe { core::slice::from_raw_parts_mut(raw.as_mut_ptr() as *mut u16, raw.len() / 2) }
}

/// Round an f32 to the nearest float16 (ties to even), as its bits.
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let x = value.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exponent = ((x >> 23) & 0xff) as i32;
    let mantissa = x & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let e = exponent - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rest = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = if rest > halfway || (rest == halfway && half & 1 == 1) { half + 1 } else { half };
        return sign | rounded as u16;
    }
    let half = ((e as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    // A carry out of the mantissa correctly increments the exponent.
    let rounded = if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | rounded as u16
}

/// The f32 value of float16 bits.
#[inline(always)]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    if exponent == 0 {
        let magnitude = mantissa as f32 * (1.0 / 16_777_216.0);
        return if sign != 0 { -magnitude } else { magnitude };
    }
    if exponent == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13));
    }
    f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13))
}

#[inline(always)]
fn half_sum_portable(kind: AffineKind, query: &[f32], row: &[u16]) -> f32 {
    match kind {
        AffineKind::SquaredEuclidean => query
            .iter()
            .zip(row)
            .map(|(&q, &y)| {
                let e = q - f16_bits_to_f32(y);
                e * e
            })
            .sum(),
        AffineKind::Dot | AffineKind::Mips { .. } => {
            query.iter().zip(row).map(|(&q, &y)| q * f16_bits_to_f32(y)).sum()
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn half_sums_avx512(kind: AffineKind, query: &[f32], rows: [&[u16]; 4]) -> [f32; 4] {
    use core::arch::x86_64::*;
    let n = query.len();
    let mut acc = [_mm512_setzero_ps(); 4];
    let mut offset = 0;
    while offset < n {
        let q = _mm512_loadu_ps(query.as_ptr().add(offset));
        for r in 0..4 {
            let y = _mm512_cvtph_ps(_mm256_loadu_si256(rows[r].as_ptr().add(offset).cast()));
            acc[r] = match kind {
                AffineKind::SquaredEuclidean => {
                    let e = _mm512_sub_ps(q, y);
                    _mm512_fmadd_ps(e, e, acc[r])
                }
                AffineKind::Dot | AffineKind::Mips { .. } => _mm512_fmadd_ps(q, y, acc[r]),
            };
        }
        offset += 16;
    }
    acc.map(|a| _mm512_reduce_add_ps(a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn half_sums_f16c(kind: AffineKind, query: &[f32], rows: [&[u16]; 4]) -> [f32; 4] {
    use core::arch::x86_64::*;
    let n = query.len();
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut offset = 0;
    while offset < n {
        let q = _mm256_loadu_ps(query.as_ptr().add(offset));
        for r in 0..4 {
            let y = _mm256_cvtph_ps(_mm_loadu_si128(rows[r].as_ptr().add(offset).cast()));
            acc[r] = match kind {
                AffineKind::SquaredEuclidean => {
                    let e = _mm256_sub_ps(q, y);
                    _mm256_fmadd_ps(e, e, acc[r])
                }
                AffineKind::Dot | AffineKind::Mips { .. } => _mm256_fmadd_ps(q, y, acc[r]),
            };
        }
        offset += 8;
    }
    acc.map(|a| {
        let mut lanes = [0.0f32; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), a);
        lanes.iter().sum()
    })
}

fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::FastRng;

    fn sample(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = FastRng::new(seed);
        (0..n * dim).map(|_| rng.next_float() * 2.0 - 1.0).collect()
    }

    fn exact(kind: AffineKind, row: &[f32], query: &[f32]) -> f32 {
        match kind {
            AffineKind::SquaredEuclidean => row.iter().zip(query).map(|(x, q)| (x - q) * (x - q)).sum::<f32>(),
            AffineKind::Dot => 1.0 - row.iter().zip(query).map(|(x, q)| x * q).sum::<f32>(),
            AffineKind::Mips { norm2 } => {
                query.iter().map(|q| q * q).sum::<f32>() + norm2
                    - 2.0 * row.iter().zip(query).map(|(x, q)| x * q).sum::<f32>()
            }
        }
    }

    #[test]
    fn code_distances_approximate_exact_distances() {
        for dim in [5, 8, 37, 128, 200] {
            let data = sample(200, dim, 3);
            let query = sample(1, dim, 4);
            for kind in [AffineKind::SquaredEuclidean, AffineKind::Dot] {
                for (bits, clip) in [(8, None), (4, None), (4, Some(DEFAULT_SQ4_CLIP_QUANTILE))] {
                    let codes = AffineCodes::encode_bits(&data, dim, kind, bits, clip);
                    assert_eq!(codes.stride() % LINE, 0);
                    assert_eq!(codes.row(0).as_ptr() as usize % LINE, 0);
                    let mut prepared = AffineQuery::default();
                    codes.prepare_query(&query, &mut prepared);
                    // Half a code step per feature, plus the int8 rounding of the query.
                    let step = 2.0 / ((1u32 << bits) - 1) as f32;
                    let tolerance = (step * 0.5 + 0.01) * (dim as f32).sqrt() * 2.0 + 0.02;
                    let mut total_error = 0.0;
                    for index in 0..200 {
                        let row = &data[index * dim..(index + 1) * dim];
                        let approx = codes.distance(&prepared, index as i32);
                        let error = (approx - exact(kind, row, &query)).abs();
                        total_error += error;
                        assert!(error < tolerance * 3.0, "{kind:?} {bits} bits dim {dim}: error {error}");
                    }
                    assert!(total_error / 200.0 < tolerance, "{kind:?} {bits} bits dim {dim}");
                    let four = codes.distance_four(&prepared, [3, 1, 4, 1]);
                    for (value, index) in four.iter().zip([3, 1, 4, 1]) {
                        assert!((*value - codes.distance(&prepared, index)).abs() <= 1e-5 * (1.0 + value.abs()));
                    }
                }
            }
        }
    }

    #[test]
    fn code_distances_rank_like_exact_distances() {
        let (n, dim) = (500, 64);
        let data = sample(n, dim, 7);
        let query = sample(1, dim, 8);
        for bits in [8, 4] {
            let codes = AffineCodes::encode_bits(&data, dim, AffineKind::SquaredEuclidean, bits, None);
            let mut prepared = AffineQuery::default();
            codes.prepare_query(&query, &mut prepared);
            let mut by_exact: Vec<usize> = (0..n).collect();
            by_exact.sort_by(|&a, &b| {
                exact(AffineKind::SquaredEuclidean, &data[a * dim..(a + 1) * dim], &query)
                    .total_cmp(&exact(AffineKind::SquaredEuclidean, &data[b * dim..(b + 1) * dim], &query))
            });
            let mut by_code: Vec<usize> = (0..n).collect();
            by_code.sort_by(|&a, &b| codes.distance(&prepared, a as i32).total_cmp(&codes.distance(&prepared, b as i32)));
            let top: std::collections::HashSet<_> = by_exact[..10].iter().collect();
            let found = by_code[..40].iter().filter(|i| top.contains(i)).count();
            assert!(found >= 9, "{bits} bits: {found} of the 10 nearest in the 40 nearest by code");
        }
    }

    #[test]
    fn clipping_spends_codes_on_the_bulk_of_each_feature() {
        let (n, dim) = (1000, 4);
        let mut data = sample(n, dim, 9);
        data[0] = 1000.0; // one outlier in feature 0
        let full = AffineCodes::encode_bits(&data, dim, AffineKind::SquaredEuclidean, 4, None);
        let clipped = AffineCodes::encode_bits(&data, dim, AffineKind::SquaredEuclidean, 4, Some(0.01));
        assert!(full.scales[0] > 50.0);
        assert!(clipped.scales[0] < 0.2);
        // The outlier gets the top code.
        assert_eq!(clipped.row(0)[0] & 15, 15);
    }

    #[test]
    fn half_rows_rerank_like_exact_distances() {
        for dim in [3, 32, 100] {
            let mut data = sample(50, dim, 5);
            for v in data.iter_mut() {
                *v *= 7.0;
            }
            let query = sample(1, dim, 6);
            for kind in [AffineKind::SquaredEuclidean, AffineKind::Dot] {
                let rows = HalfRows::encode(&data, dim, kind);
                let mut prepared = HalfQuery::default();
                rows.prepare_query(&query, &mut prepared);
                for index in 0..50 {
                    let row = &data[index * dim..(index + 1) * dim];
                    let mut want = exact(kind, row, &query);
                    if kind == AffineKind::Dot {
                        want = 1.0 - (1.0 - want).clamp(-1.0, 1.0);
                    }
                    let got = rows.distance(&prepared, index as i32);
                    assert!((got - want).abs() <= 2e-3 * (1.0 + want.abs()) * dim as f32, "{kind:?} dim {dim}: {got} vs {want}");
                }
                let portable = rows.row(7);
                let sum = half_sum_portable(kind, &prepared.values, portable);
                let finished = rows.finish(sum, prepared.offset);
                assert!((finished - rows.distance(&prepared, 7)).abs() <= 1e-4 * (1.0 + sum.abs()));
            }
        }
    }

    #[test]
    fn mips_distances_match_the_augmented_squared_euclidean_distance() {
        let (n, dim) = (300, 40);
        let mut data = sample(n, dim, 12);
        for (row, v) in data.chunks_exact_mut(dim).enumerate() {
            let scale = 0.2 + (row % 7) as f32 * 0.3;
            v.iter_mut().for_each(|x| *x *= scale);
        }
        let norms: Vec<f32> = data.chunks_exact(dim).map(|r| r.iter().map(|x| x * x).sum()).collect();
        let norm2 = norms.iter().fold(0.0f32, |m, &v| m.max(v));
        // Augmented rows: the features, then sqrt(norm2 - |x|^2).
        let mut augmented = Vec::with_capacity(n * (dim + 1));
        for (row, &norm) in data.chunks_exact(dim).zip(&norms) {
            augmented.extend_from_slice(row);
            augmented.push((norm2 - norm).max(0.0).sqrt());
        }
        let query: Vec<f32> = sample(1, dim, 13).iter().map(|q| q * 1.5).collect();
        let mut padded = query.clone();
        padded.push(0.0);
        let augmented_l2 = |i: usize| -> f32 {
            augmented[i * (dim + 1)..(i + 1) * (dim + 1)].iter().zip(&padded).map(|(x, q)| (x - q) * (x - q)).sum()
        };
        let kind = AffineKind::Mips { norm2 };
        for (bits, clip) in [(8, None), (4, Some(DEFAULT_SQ4_CLIP_QUANTILE))] {
            let codes = AffineCodes::encode_columns(&augmented, dim + 1, dim, kind, bits, clip);
            assert_eq!(codes.dim, dim);
            let mut prepared = AffineQuery::default();
            codes.prepare_query(&query, &mut prepared);
            let mut by_exact: Vec<usize> = (0..n).collect();
            by_exact.sort_by(|&a, &b| augmented_l2(a).total_cmp(&augmented_l2(b)));
            let mut by_code: Vec<usize> = (0..n).collect();
            by_code.sort_by(|&a, &b| codes.distance(&prepared, a as i32).total_cmp(&codes.distance(&prepared, b as i32)));
            let top: std::collections::HashSet<_> = by_exact[..10].iter().collect();
            let found = by_code[..40].iter().filter(|i| top.contains(i)).count();
            assert!(found >= 9, "{bits} bits: {found} of the 10 nearest in the 40 nearest by code");
            let mean_error: f32 = (0..n)
                .map(|i| (codes.distance(&prepared, i as i32) - augmented_l2(i)).abs() / augmented_l2(i))
                .sum::<f32>()
                / n as f32;
            assert!(mean_error < if bits == 8 { 0.01 } else { 0.05 }, "{bits} bits: mean relative error {mean_error}");
        }
        let rows = HalfRows::encode_columns(&augmented, dim + 1, dim, kind);
        let mut prepared = HalfQuery::default();
        rows.prepare_query(&query, &mut prepared);
        for i in 0..n {
            let (got, want) = (rows.distance(&prepared, i as i32), augmented_l2(i));
            assert!((got - want).abs() <= 1e-3 * want.max(1.0), "row {i}: {got} vs {want}");
            assert!((got - exact(kind, &data[i * dim..(i + 1) * dim], &query)).abs() <= 1e-3 * want.max(1.0));
        }
    }

    #[test]
    fn float16_conversion_round_trips() {
        for &v in &[0.0f32, -0.0, 1.0, -1.0, 0.5, 0.333_333, 1e-3, -2.5e-5, 6.0e-8, 65504.0, 1e-9] {
            let back = f16_bits_to_f32(f32_to_f16_bits(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 6.0e-8, "{v} -> {back}");
        }
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert!(f16_bits_to_f32(f32_to_f16_bits(1e6)).is_infinite());
    }

    #[test]
    fn pca_projection_preserves_low_rank_geometry() {
        let (n, dim, rank) = (2000, 48, 4);
        let mut rng = FastRng::new(11);
        let basis: Vec<f32> = (0..rank * dim).map(|_| rng.next_float() * 2.0 - 1.0).collect();
        let mut data = vec![0.0f32; n * dim];
        for row in 0..n {
            let weights: Vec<f32> = (0..rank).map(|_| rng.next_float() * 4.0 - 2.0).collect();
            for t in 0..dim {
                let signal: f32 = (0..rank).map(|r| weights[r] * basis[r * dim + t]).sum();
                data[row * dim + t] = 0.5 + signal + 0.01 * (rng.next_float() - 0.5);
            }
        }
        let projection =
            PcaProjection::fit(&data, dim, ProjectionDims::VarianceFraction(0.99), 1000, 1).unwrap();
        assert_eq!(projection.n_components, rank);
        let projected = projection.project_rows(&data);
        let (x, y) = (&data[..dim], &data[dim..2 * dim]);
        let (px, py) = (&projected[..rank], &projected[rank..2 * rank]);
        let exact_l2: f32 = x.iter().zip(y).map(|(a, b)| (a - b) * (a - b)).sum();
        let projected_l2: f32 = px.iter().zip(py).map(|(a, b)| (a - b) * (a - b)).sum();
        assert!((exact_l2 - projected_l2).abs() < 0.02 * exact_l2 + 0.01);

        let mut query = Vec::new();
        let shift = projection.project_query(AffineKind::Dot, x, &mut query);
        let exact_dot: f32 = x.iter().zip(y).map(|(a, b)| a * b).sum();
        let approx_dot = shift + query.iter().zip(py).map(|(a, b)| a * b).sum::<f32>();
        assert!((exact_dot - approx_dot).abs() < 0.02 * exact_dot.abs() + 0.02);
        let shift = projection.project_query(AffineKind::SquaredEuclidean, x, &mut query);
        assert_eq!(shift, 0.0);
        for (a, b) in query.iter().zip(px) {
            assert!((a - b).abs() < 1e-3 * (1.0 + b.abs()));
        }

        // Keeping more than two thirds of the features is not worth projecting.
        assert!(PcaProjection::fit(&data, dim, ProjectionDims::Components(40), 1000, 1).is_none());
        assert_eq!(
            PcaProjection::fit(&data, dim, ProjectionDims::Components(10), 1000, 1).unwrap().n_components,
            10
        );
        let isotropic = sample(2000, dim, 5);
        assert!(PcaProjection::fit(&isotropic, dim, ProjectionDims::VarianceFraction(0.95), 1000, 1).is_none());
    }

    #[test]
    fn constant_features_are_encoded() {
        let data = vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
        for bits in [8, 4] {
            let codes = AffineCodes::encode_bits(&data, 2, AffineKind::SquaredEuclidean, bits, None);
            assert_eq!(codes.scales[0], 1.0);
            let mut prepared = AffineQuery::default();
            codes.prepare_query(&[1.0, 3.0], &mut prepared);
            // At most half a code step (scale / 2) from the stored value, plus query rounding.
            let half_step = codes.scales[1] / 2.0;
            assert!(codes.distance(&prepared, 1) <= half_step * half_step * 1.001 + 0.02);
        }
    }
}
