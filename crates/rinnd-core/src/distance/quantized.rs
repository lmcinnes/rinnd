//! Quantized distance functions.
//!
//! These compute distances between a full-precision float query vector and
//! a quantized (uint8 or uint4) database vector, using a lookup table to
//! dequantize on the fly.
//!
//! Three distance types are supported for each quantization level:
//!   - Squared Euclidean
//!   - Alternative Cosine (log-transformed)
//!   - Alternative Dot (log-transformed)
//!
//! AVX2 SIMD is used where available, processing 8 floats at a time by
//! gathering dequantized values through the lookup table.

const FLOAT32_MAX: f32 = f32::MAX;

// ────────────────────────────────────────────────────────────────
//  Symmetric signed-int8 distances
// ────────────────────────────────────────────────────────────────

/// Integer dot product for two symmetrically quantized vectors.
///
/// Values are widened before multiplication, so the result is exact for
/// dimensions up to the practical limits of an `i32` accumulator.
#[inline]
pub fn quantized_i8_dot(x: &[i8], y: &[i8]) -> i32 {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { quantized_i8_dot_avx2(x, y) };
        }
    }

    x.iter()
        .zip(y)
        .map(|(&a, &b)| i32::from(a) * i32::from(b))
        .sum()
}

/// Reconstruct an inner product for per-dimension symmetric signed-int8 codes.
///
/// `scales[i]` is the dequantization scale shared by the query and every
/// database vector in dimension `i`, so each integer product is multiplied by
/// `scales[i]²`. This remains correct for negative inner products.
#[inline]
pub fn quantized_i8_per_dimension_dot(x: &[i8], y: &[i8], scales: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len(), scales.len());
    x.iter()
        .zip(y)
        .zip(scales)
        .map(|((&a, &b), &scale)| i32::from(a) as f32 * i32::from(b) as f32 * scale * scale)
        .sum()
}

/// Encode one signed SQ4 value in the low four bits.
///
/// Values use a biased representation: signed `-7..=7` maps to nibbles
/// `0..=14`; nibble `15` is reserved and decodes as zero defensively.
#[inline(always)]
pub fn encode_signed_i4(value: i8) -> u8 {
    debug_assert!((-7..=7).contains(&value));
    (value + 7) as u8
}

/// Decode one biased signed SQ4 nibble.
#[inline(always)]
pub fn decode_signed_i4(nibble: u8) -> i8 {
    match nibble & 0x0f {
        15 => 0,
        value => value as i8 - 7,
    }
}

/// Exact integer dot product for two nibble-packed signed SQ4 vectors.
///
/// Dimension zero occupies the low nibble, dimension one the high nibble.
/// `dim` is explicit so an odd final dimension never consumes the padding
/// nibble.
#[inline]
pub fn quantized_i4_dot(x: &[u8], y: &[u8], dim: usize) -> i32 {
    debug_assert!(x.len() >= dim.div_ceil(2));
    debug_assert!(y.len() >= dim.div_ceil(2));
    let mut result = 0i32;
    for i in 0..dim {
        let shift = (i & 1) * 4;
        let a = decode_signed_i4(x[i / 2] >> shift);
        let b = decode_signed_i4(y[i / 2] >> shift);
        result += i32::from(a) * i32::from(b);
    }
    result
}

/// Alternative angular distance for two symmetrically quantized vectors.
///
/// `inv_norm_x` and `inv_norm_y` are precomputed reciprocal L2 norms in the
/// integer domain. Per-vector quantization scales cancel during cosine
/// normalization and therefore are not needed in the search loop.
#[inline]
pub fn quantized_i8_alternative_dot(x: &[i8], y: &[i8], inv_norm_x: f32, inv_norm_y: f32) -> f32 {
    let similarity = quantized_i8_dot(x, y) as f32 * inv_norm_x * inv_norm_y;
    if similarity <= 0.0 {
        FLOAT32_MAX
    } else {
        -similarity.min(1.0).log2()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantized_i8_dot_avx2(x: &[i8], y: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_si256();
    let ones = _mm256_set1_epi16(1);
    let chunks = x.len() / 32;

    for chunk in 0..chunks {
        let offset = chunk * 32;
        let vx = _mm256_loadu_si256(x.as_ptr().add(offset) as *const __m256i);
        let vy = _mm256_loadu_si256(y.as_ptr().add(offset) as *const __m256i);

        let x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
        let x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vx));
        let y_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vy));
        let y_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(vy));

        let products_lo = _mm256_mullo_epi16(x_lo, y_lo);
        let products_hi = _mm256_mullo_epi16(x_hi, y_hi);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(products_lo, ones));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(products_hi, ones));
    }

    let hi = _mm256_extracti128_si256::<1>(sum);
    let lo = _mm256_castsi256_si128(sum);
    let sum128 = _mm_add_epi32(lo, hi);
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32::<0x4E>(sum128));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32::<0xB1>(sum64));
    let mut result = _mm_cvtsi128_si32(sum32);

    for i in chunks * 32..x.len() {
        result += i32::from(x[i]) * i32::from(y[i]);
    }
    result
}

// ────────────────────────────────────────────────────────────────
//  uint8 quantized distances
// ────────────────────────────────────────────────────────────────

/// Squared Euclidean between float query `x` and quantized uint8 `y`.
///
/// `codebook[y[i]]` gives the dequantized float value for dimension i.
#[inline]
pub fn quantized_u8_sq_euclidean(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { quantized_u8_sq_euclidean_avx2(x, y, codebook) };
        }
    }

    let mut result = 0.0f32;
    for i in 0..x.len() {
        let yi = codebook[y[i] as usize];
        let diff = x[i] - yi;
        result += diff * diff;
    }
    result
}

/// Alternative cosine between float query `x` and quantized uint8 `y`.
///
/// Returns log₂((‖x‖·‖y‖) / (x·y)) mapped through (sim+1)/2 to keep
/// values non-negative even for negative cosine similarities.
#[inline]
pub fn quantized_u8_alternative_cosine(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { quantized_u8_alt_cosine_avx2(x, y, codebook) };
        }
    }

    let mut dot = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        let qy = codebook[y[i] as usize];
        dot += x[i] * qy;
        norm_x += x[i] * x[i];
        norm_y += qy * qy;
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        return 0.0;
    } else if norm_x == 0.0 || norm_y == 0.0 {
        return FLOAT32_MAX;
    } else if dot <= 0.0 {
        return FLOAT32_MAX;
    }

    let sim = dot / (norm_x * norm_y).sqrt();
    -((sim + 1.0) / 2.0).log2()
}

/// Alternative dot between float query `x` and quantized uint8 `y`.
///
/// y is assumed to be from a normalized dataset.
/// Returns -log₂(x·y / ‖y‖).
#[inline]
pub fn quantized_u8_alternative_dot(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());

    let mut dot = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        let qy = codebook[y[i] as usize];
        dot += x[i] * qy;
        norm_y += qy * qy;
    }

    if dot <= 0.0 {
        FLOAT32_MAX
    } else {
        -(dot / norm_y.sqrt()).log2()
    }
}

// ────────────────────────────────────────────────────────────────
//  uint4 quantized distances (nibble-packed: 2 values per byte)
// ────────────────────────────────────────────────────────────────

/// Extract the float value for dimension `i` from a nibble-packed byte array.
#[inline(always)]
fn dequant_u4(y: &[u8], i: usize, codebook: &[f32]) -> f32 {
    let byte = y[i / 2];
    let idx = if i % 2 == 0 {
        byte & 0x0F // lower nibble
    } else {
        (byte >> 4) & 0x0F // upper nibble
    };
    codebook[idx as usize]
}

/// Squared Euclidean between float query `x` and quantized uint4 `y`.
#[inline]
pub fn quantized_u4_sq_euclidean(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    let mut result = 0.0f32;
    for i in 0..x.len() {
        let diff = x[i] - dequant_u4(y, i, codebook);
        result += diff * diff;
    }
    result
}

/// Alternative cosine between float query `x` and quantized uint4 `y`.
#[inline]
pub fn quantized_u4_alternative_cosine(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_x = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        let qy = dequant_u4(y, i, codebook);
        dot += x[i] * qy;
        norm_x += x[i] * x[i];
        norm_y += qy * qy;
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        return 0.0;
    } else if norm_x == 0.0 || norm_y == 0.0 {
        return FLOAT32_MAX;
    } else if dot <= 0.0 {
        return FLOAT32_MAX;
    }

    let sim = dot / (norm_x * norm_y).sqrt();
    -((sim + 1.0) / 2.0).log2()
}

/// Alternative dot between float query `x` and quantized uint4 `y`.
#[inline]
pub fn quantized_u4_alternative_dot(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_y = 0.0f32;
    for i in 0..x.len() {
        let qy = dequant_u4(y, i, codebook);
        dot += x[i] * qy;
        norm_y += qy * qy;
    }

    if dot <= 0.0 {
        FLOAT32_MAX
    } else {
        -(dot / norm_y.sqrt()).log2()
    }
}

// ────────────────────────────────────────────────────────────────
//  AVX2+FMA SIMD implementations for uint8 quantized distances
// ────────────────────────────────────────────────────────────────

/// AVX2+FMA squared Euclidean for uint8 quantized vectors.
///
/// Strategy: process 8 dimensions at a time.
/// For each group of 8:
///   1. Gather 8 codebook entries via `_mm256_i32gather_ps` using u8 indices.
///   2. Subtract query floats, FMA to accumulate squared differences.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn quantized_u8_sq_euclidean_avx2(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    let mut vsum = _mm256_setzero_ps();

    for c in 0..chunks {
        let idx = c * 8;

        // Load 8 indices and widen to i32 for gather
        let indices = _mm256_set_epi32(
            y[idx + 7] as i32,
            y[idx + 6] as i32,
            y[idx + 5] as i32,
            y[idx + 4] as i32,
            y[idx + 3] as i32,
            y[idx + 2] as i32,
            y[idx + 1] as i32,
            y[idx] as i32,
        );

        // Gather dequantized values from codebook
        let vy = _mm256_i32gather_ps::<4>(codebook.as_ptr(), indices);

        // Load query values
        let vx = _mm256_loadu_ps(x.as_ptr().add(idx));

        // diff = x - y_dequant
        let diff = _mm256_sub_ps(vx, vy);
        // accumulate diff²
        vsum = _mm256_fmadd_ps(diff, diff, vsum);
    }

    let mut result = hsum256_ps(vsum);

    // Scalar remainder
    let start = chunks * 8;
    for i in start..n {
        let yi = codebook[y[i] as usize];
        let diff = x[i] - yi;
        result += diff * diff;
    }

    result
}

/// AVX2+FMA alternative cosine for uint8 quantized vectors.
///
/// Gathers dequantized values and computes dot, norm_x, norm_y in one pass.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn quantized_u8_alt_cosine_avx2(x: &[f32], y: &[u8], codebook: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = x.len();
    let chunks = n / 8;
    let mut vdot = _mm256_setzero_ps();
    let mut vnorm_x = _mm256_setzero_ps();
    let mut vnorm_y = _mm256_setzero_ps();

    for c in 0..chunks {
        let idx = c * 8;

        let indices = _mm256_set_epi32(
            y[idx + 7] as i32,
            y[idx + 6] as i32,
            y[idx + 5] as i32,
            y[idx + 4] as i32,
            y[idx + 3] as i32,
            y[idx + 2] as i32,
            y[idx + 1] as i32,
            y[idx] as i32,
        );

        let vy = _mm256_i32gather_ps::<4>(codebook.as_ptr(), indices);
        let vx = _mm256_loadu_ps(x.as_ptr().add(idx));

        vdot = _mm256_fmadd_ps(vx, vy, vdot);
        vnorm_x = _mm256_fmadd_ps(vx, vx, vnorm_x);
        vnorm_y = _mm256_fmadd_ps(vy, vy, vnorm_y);
    }

    let mut dot = hsum256_ps(vdot);
    let mut norm_x = hsum256_ps(vnorm_x);
    let mut norm_y = hsum256_ps(vnorm_y);

    let start = chunks * 8;
    for i in start..n {
        let qy = codebook[y[i] as usize];
        dot += x[i] * qy;
        norm_x += x[i] * x[i];
        norm_y += qy * qy;
    }

    if norm_x == 0.0 && norm_y == 0.0 {
        return 0.0;
    } else if norm_x == 0.0 || norm_y == 0.0 {
        return FLOAT32_MAX;
    } else if dot <= 0.0 {
        return FLOAT32_MAX;
    }

    let sim = dot / (norm_x * norm_y).sqrt();
    -((sim + 1.0) / 2.0).log2()
}

/// Horizontal sum of 8 floats in __m256.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_i8_dot_matches_scalar_for_remainders() {
        for len in [1, 7, 31, 32, 33, 100, 128, 784] {
            let x: Vec<i8> = (0..len)
                .map(|i| ((i * 37) % 255) as i16 - 127)
                .map(|v| v as i8)
                .collect();
            let y: Vec<i8> = (0..len)
                .map(|i| ((i * 73 + 11) % 255) as i16 - 127)
                .map(|v| v as i8)
                .collect();
            let expected: i32 = x
                .iter()
                .zip(&y)
                .map(|(&a, &b)| i32::from(a) * i32::from(b))
                .sum();
            assert_eq!(quantized_i8_dot(&x, &y), expected, "len={len}");
        }
    }

    #[test]
    fn test_quantized_i8_alternative_dot_identical() {
        let x = [12i8, -35, 72, 4, -8];
        let norm = (quantized_i8_dot(&x, &x) as f32).sqrt();
        let distance = quantized_i8_alternative_dot(&x, &x, norm.recip(), norm.recip());
        assert!(distance.abs() < 1e-6, "distance={distance}");
    }

    #[test]
    fn test_per_dimension_i8_reconstructs_negative_inner_product() {
        let x = [2i8, -3, 4];
        let y = [-5i8, -2, -1];
        let scales = [0.5f32, 0.25, 2.0];
        let expected = -10.0 * 0.25 + 6.0 * 0.0625 - 4.0 * 4.0;
        assert!((quantized_i8_per_dimension_dot(&x, &y, &scales) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_signed_i4_encoding_and_odd_dimension_dot() {
        for value in -7i8..=7 {
            assert_eq!(decode_signed_i4(encode_signed_i4(value)), value);
        }
        assert_eq!(decode_signed_i4(15), 0);

        let x = [
            encode_signed_i4(-7) | (encode_signed_i4(3) << 4),
            encode_signed_i4(-2) | (encode_signed_i4(7) << 4),
        ];
        let y = [
            encode_signed_i4(2) | (encode_signed_i4(-4) << 4),
            encode_signed_i4(-5) | (encode_signed_i4(-7) << 4),
        ];
        // The high nibble of the last byte is padding and must be ignored.
        assert_eq!(quantized_i4_dot(&x, &y, 3), -14 - 12 + 10);
    }

    fn make_codebook_256() -> Vec<f32> {
        // Simple identity codebook: index i → value i as float
        (0..256).map(|i| i as f32 / 255.0).collect()
    }

    fn make_codebook_16() -> Vec<f32> {
        (0..16).map(|i| i as f32 / 15.0).collect()
    }

    #[test]
    fn test_u8_sq_euclidean_identical() {
        let cb = make_codebook_256();
        let x = vec![0.0 / 255.0, 128.0 / 255.0, 255.0 / 255.0];
        let y = vec![0u8, 128u8, 255u8];
        let d = quantized_u8_sq_euclidean(&x, &y, &cb);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_u8_sq_euclidean_basic() {
        let cb = make_codebook_256();
        let x = vec![0.0; 4];
        let y = vec![1u8, 1u8, 1u8, 1u8];
        // Each dequantized y_i = 1/255, diff = 1/255
        // sq_euc = 4 * (1/255)^2
        let expected = 4.0 * (1.0 / 255.0) * (1.0 / 255.0);
        let d = quantized_u8_sq_euclidean(&x, &y, &cb);
        assert!((d - expected).abs() < 1e-8);
    }

    #[test]
    fn test_u4_sq_euclidean_identical() {
        let cb = make_codebook_16();
        // 4 dimensions packed into 2 bytes
        // dim0 = idx 0 (lower nibble byte 0), dim1 = idx 7 (upper nibble byte 0)
        // dim2 = idx 15 (lower nibble byte 1), dim3 = idx 8 (upper nibble byte 1)
        let y = vec![0x70u8, 0x8Fu8]; // nibbles: 0, 7, 15, 8
        let x = vec![0.0 / 15.0, 7.0 / 15.0, 15.0 / 15.0, 8.0 / 15.0];
        let d = quantized_u4_sq_euclidean(&x, &y, &cb);
        assert!(d.abs() < 1e-6);
    }

    #[test]
    fn test_u8_alt_cosine_identical() {
        let cb: Vec<f32> = (0..256).map(|i| (i as f32 + 1.0) / 256.0).collect();
        let y: Vec<u8> = (0..8).collect();
        let x: Vec<f32> = y.iter().map(|&i| cb[i as usize]).collect();
        let d = quantized_u8_alternative_cosine(&x, &y, &cb);
        // Cosine similarity of identical vectors should be ~1
        // -log₂((1+1)/2) = -log₂(1) = 0
        assert!(d.abs() < 1e-4, "expected ~0, got {}", d);
    }
}
