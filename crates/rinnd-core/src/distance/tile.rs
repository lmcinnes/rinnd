//! AVX-512VL/AVX2 register-tiled pairwise kernels for dense blocks (leaf initialization).
//!
//! Each pair keeps the accumulator layout, reduction order and scalar tail of the
//! corresponding single-pair kernel, so results are bit-identical to `distance`.
//! AVX-512VL is used only for its 32 vector registers (6x4 tiles of 256-bit lanes).

/// Squared Euclidean for all pairs `i < j`; false if AVX2/FMA is unavailable.
pub(crate) fn l2_upper(rows: &[&[f32]], out: &mut [f32]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if rows.len() >= 2 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        if avx512_tiles() {
            unsafe { upper_avx512::<false>(rows, out) };
        } else {
            unsafe { upper_avx2::<false>(rows, out) };
        }
        return true;
    }
    let _ = (rows, out);
    false
}

/// Dot products for all pairs `i < j`; false if unsupported or not profitable.
pub(crate) fn dot_upper(rows: &[&[f32]], out: &mut [f32]) -> bool {
    // At d = 100 tiles were neutral to slower in leaf init (GloVe), for both tile widths.
    const MIN_DIM: usize = 128;
    #[cfg(target_arch = "x86_64")]
    if rows.len() >= 2
        && rows[0].len() >= MIN_DIM
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        if avx512_tiles() {
            unsafe { upper_avx512::<true>(rows, out) };
        } else {
            unsafe { upper_avx2::<true>(rows, out) };
        }
        return true;
    }
    let _ = (rows, out);
    false
}

#[cfg(target_arch = "x86_64")]
fn avx512_tiles() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl")
}

const TQ: usize = 4;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn upper_avx2<const DOT: bool>(rows: &[&[f32]], out: &mut [f32]) {
    upper::<DOT, 2>(rows, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma", enable = "avx512f", enable = "avx512vl")]
unsafe fn upper_avx512<const DOT: bool>(rows: &[&[f32]], out: &mut [f32]) {
    upper::<DOT, 6>(rows, out)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn upper<const DOT: bool, const TP: usize>(rows: &[&[f32]], out: &mut [f32]) {
    let m = rows.len();
    debug_assert!(out.len() >= m * m);
    let row = |index: usize| rows[index.min(m - 1)];
    let mut tile = [[0.0f32; TQ]; TP];
    for a0 in (0..m).step_by(TP) {
        let ps: [&[f32]; TP] = std::array::from_fn(|a| row(a0 + a));
        let mut b0 = a0 + 1;
        while b0 < m {
            let qs: [&[f32]; TQ] = std::array::from_fn(|b| row(b0 + b));
            if DOT {
                dot_tile::<TP>(ps, qs, &mut tile);
            } else {
                l2_tile::<TP>(ps, qs, &mut tile);
            }
            for a in 0..TP {
                let i = a0 + a;
                for b in 0..TQ {
                    let j = b0 + b;
                    if i < m && j < m && j > i {
                        out[i * m + j] = tile[a][b];
                    }
                }
            }
            b0 += TQ;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_tile<const TP: usize>(ps: [&[f32]; TP], qs: [&[f32]; TQ], out: &mut [[f32; TQ]; TP]) {
    use std::arch::x86_64::*;
    let dim = ps[0].len();
    let end = dim / 32 * 32;
    let mut lanes = [[[_mm256_setzero_ps(); 4]; TQ]; TP];
    for lane in 0..4 {
        let mut acc = [[_mm256_setzero_ps(); TQ]; TP];
        let mut offset = lane * 8;
        while offset < end {
            let qv: [__m256; TQ] = std::array::from_fn(|b| _mm256_loadu_ps(qs[b].as_ptr().add(offset)));
            for a in 0..TP {
                let pv = _mm256_loadu_ps(ps[a].as_ptr().add(offset));
                for b in 0..TQ {
                    acc[a][b] = _mm256_fmadd_ps(pv, qv[b], acc[a][b]);
                }
            }
            offset += 32;
        }
        for a in 0..TP {
            for b in 0..TQ {
                lanes[a][b][lane] = acc[a][b];
            }
        }
    }
    for a in 0..TP {
        for b in 0..TQ {
            let s = lanes[a][b];
            let mut dot = dot_hsum(_mm256_add_ps(_mm256_add_ps(s[0], s[1]), _mm256_add_ps(s[2], s[3])));
            for i in end..dim {
                dot += ps[a][i] * qs[b][i];
            }
            out[a][b] = dot;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn l2_tile<const TP: usize>(ps: [&[f32]; TP], qs: [&[f32]; TQ], out: &mut [[f32; TQ]; TP]) {
    use std::arch::x86_64::*;
    let dim = ps[0].len();
    let end = dim / 8 * 8;
    let mut acc = [[_mm256_setzero_ps(); TQ]; TP];
    let mut offset = 0;
    while offset < end {
        let qv: [__m256; TQ] = std::array::from_fn(|b| _mm256_loadu_ps(qs[b].as_ptr().add(offset)));
        for a in 0..TP {
            let pv = _mm256_loadu_ps(ps[a].as_ptr().add(offset));
            for b in 0..TQ {
                let difference = _mm256_sub_ps(pv, qv[b]);
                acc[a][b] = _mm256_fmadd_ps(difference, difference, acc[a][b]);
            }
        }
        offset += 8;
    }
    for a in 0..TP {
        for b in 0..TQ {
            let mut sum = l2_hsum(acc[a][b]);
            for i in end..dim {
                let difference = ps[a][i] - qs[b][i];
                sum += difference * difference;
            }
            out[a][b] = sum;
        }
    }
}

/// Same reduction as `alternatives::horizontal_sum_avx`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn dot_hsum(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
    let pairs = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let odd = _mm_shuffle_ps(pairs, pairs, 0x55);
    _mm_cvtss_f32(_mm_add_ss(pairs, odd))
}

/// Same reduction as `euclidean::hsum256_ps_avx`.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn l2_hsum(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let sum = _mm_add_ps(_mm256_extractf128_ps(value, 1), _mm256_castps256_ps128(value));
    let sums = _mm_add_ps(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(_mm_add_ss(sums, _mm_movehl_ps(sums, sums)))
}

#[cfg(test)]
mod tests {
    use crate::distance::{AlternativeDot, DirectNormalizedCosine, Distance, SquaredEuclidean};
    use crate::rng::FastRng;

    fn check<D: Distance<f32>>(distance: D) {
        let mut rng = FastRng::new(17);
        for dim in [8, 31, 100, 128, 131, 256, 784] {
            for m in [2, 3, 5, 8, 9, 17, 60, 64] {
                let mut data: Vec<f32> = (0..m * dim).map(|_| rng.next_float() - 0.5).collect();
                for row in data.chunks_mut(dim) {
                    let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                    row.iter_mut().for_each(|x| *x /= norm);
                }
                let rows: Vec<&[f32]> = data.chunks(dim).collect();
                let mut out = vec![f32::NAN; m * m];
                if !distance.distance_upper(&rows, &mut out) {
                    continue;
                }
                for i in 0..m {
                    for j in i + 1..m {
                        let expected = distance.distance(rows[i], rows[j]);
                        assert_eq!(out[i * m + j].to_bits(), expected.to_bits(), "dim={dim} m={m} ({i},{j})");
                    }
                }
            }
        }
    }

    #[test]
    fn tiled_upper_matches_single_pair_distances() {
        check(SquaredEuclidean);
        check(AlternativeDot);
        check(DirectNormalizedCosine);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_and_avx512_tiles_agree() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && super::avx512_tiles()) {
            return;
        }
        let mut rng = FastRng::new(5);
        for dim in [8, 31, 100, 256, 784] {
            for m in [2, 7, 13, 60] {
                let data: Vec<f32> = (0..m * dim).map(|_| rng.next_float() - 0.5).collect();
                let rows: Vec<&[f32]> = data.chunks(dim).collect();
                let run = |avx512: bool, dot: bool| {
                    let mut out = vec![0.0f32; m * m];
                    unsafe {
                        match (avx512, dot) {
                            (true, true) => super::upper_avx512::<true>(&rows, &mut out),
                            (true, false) => super::upper_avx512::<false>(&rows, &mut out),
                            (false, true) => super::upper_avx2::<true>(&rows, &mut out),
                            (false, false) => super::upper_avx2::<false>(&rows, &mut out),
                        }
                    }
                    out.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
                };
                assert_eq!(run(true, true), run(false, true), "dot dim={dim} m={m}");
                assert_eq!(run(true, false), run(false, false), "l2 dim={dim} m={m}");
            }
        }
    }
}
