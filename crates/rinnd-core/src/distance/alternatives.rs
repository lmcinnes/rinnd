//! Alternative/proxy distance functions and correction functions.
//!
//! These distances are cheaper to compute during graph construction.
//! A correction function is applied to final results to recover the true distance.

use super::traits::Distance;

const FLOAT32_MAX: f32 = f32::MAX;

/// Alternative dot product distance using log transform.
///
/// For pre-normalized vectors: d_alt(a, b) = -log₂(a·b)
/// Returns FLOAT32_MAX for non-positive dot products.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlternativeDot;

#[inline]
fn normalized_dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    for i in 0..chunks {
        let idx = i * 4;
        dot += a[idx] * b[idx]
            + a[idx + 1] * b[idx + 1]
            + a[idx + 2] * b[idx + 2]
            + a[idx + 3] * b[idx + 3];
    }
    let start = chunks * 4;
    for i in 0..remainder {
        dot += a[start + i] * b[start + i];
    }

    dot
}

#[inline]
fn alternative_dot_from_similarity(dot: f32) -> f32 {
    if dot <= 0.0 {
        FLOAT32_MAX
    } else {
        -dot.log2()
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn alternative_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    alternative_dot_from_similarity(normalized_dot_avx2(a, b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn normalized_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();
    let chunks = a.len() / 32;

    for chunk in 0..chunks {
        let idx = chunk * 32;
        let a0 = _mm256_loadu_ps(a.as_ptr().add(idx));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(idx));
        let a1 = _mm256_loadu_ps(a.as_ptr().add(idx + 8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(idx + 8));
        let a2 = _mm256_loadu_ps(a.as_ptr().add(idx + 16));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(idx + 16));
        let a3 = _mm256_loadu_ps(a.as_ptr().add(idx + 24));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(idx + 24));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);
    }

    let sum01 = _mm256_add_ps(sum0, sum1);
    let sum23 = _mm256_add_ps(sum2, sum3);
    let mut dot = horizontal_sum_avx(_mm256_add_ps(sum01, sum23));

    for i in chunks * 32..a.len() {
        dot += a[i] * b[i];
    }

    dot
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx")]
unsafe fn horizontal_sum_avx(value: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let high = _mm256_extractf128_ps(value, 1);
    let low = _mm256_castps256_ps128(value);
    let sum = _mm_add_ps(low, high);
    let high_pairs = _mm_movehl_ps(sum, sum);
    let pairs = _mm_add_ps(sum, high_pairs);
    let odd = _mm_shuffle_ps(pairs, pairs, 0x55);
    _mm_cvtss_f32(_mm_add_ss(pairs, odd))
}

impl Distance<f32> for AlternativeDot {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { alternative_dot_avx2(a, b) };
            }
        }

        alternative_dot_from_similarity(normalized_dot_scalar(a, b))
    }

    fn name(&self) -> &'static str {
        "alternative_dot"
    }
}

/// Direct cosine distance for vectors that have already been L2-normalized.
///
/// This is an explicit alternative search geometry to `AlternativeDot`. It is
/// monotonic with cosine similarity but avoids a logarithm per candidate.
#[derive(Clone, Copy, Debug, Default)]
pub struct DirectNormalizedCosine;

impl Distance<f32> for DirectNormalizedCosine {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        let similarity = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { normalized_dot_avx2(a, b) }
        } else {
            normalized_dot_scalar(a, b)
        };

        #[cfg(not(target_arch = "x86_64"))]
        let similarity = normalized_dot_scalar(a, b);

        1.0 - similarity.clamp(-1.0, 1.0)
    }

    fn name(&self) -> &'static str {
        "direct_normalized_cosine"
    }
}

/// Alternative inner product distance using reciprocal transform.
///
/// d_alt(a, b) = 1 / (a·b) for positive inner products, FLOAT32_MAX otherwise.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlternativeInnerProduct;

impl Distance<f32> for AlternativeInnerProduct {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut dot = 0.0f32;
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;

        for i in 0..chunks {
            let idx = i * 4;
            dot += a[idx] * b[idx]
                + a[idx + 1] * b[idx + 1]
                + a[idx + 2] * b[idx + 2]
                + a[idx + 3] * b[idx + 3];
        }
        let start = chunks * 4;
        for i in 0..remainder {
            dot += a[start + i] * b[start + i];
        }

        if dot <= 0.0 {
            FLOAT32_MAX
        } else {
            1.0 / dot
        }
    }

    fn name(&self) -> &'static str {
        "alternative_inner_product"
    }
}

/// Alternative cosine using log transform.
///
/// d_alt(a, b) = log₂(‖a‖·‖b‖ / (a·b))
/// Returns FLOAT32_MAX for non-positive similarities.
#[derive(Clone, Copy, Debug, Default)]
pub struct AlternativeCosine;

impl Distance<f32> for AlternativeCosine {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a == 0.0 && norm_b == 0.0 {
            0.0
        } else if norm_a == 0.0 || norm_b == 0.0 {
            FLOAT32_MAX
        } else if dot <= 0.0 {
            FLOAT32_MAX
        } else {
            ((norm_a * norm_b).sqrt() / dot).log2()
        }
    }

    fn name(&self) -> &'static str {
        "alternative_cosine"
    }
}

/// Proxy inner product: hybrid log-cosine + magnitude normalization.
///
/// Used internally during graph construction; results should be reranked
/// with true inner product distance.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProxyInnerProduct;

impl Distance<f32> for ProxyInnerProduct {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());

        let mut ip = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a.len() {
            ip += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if norm_a == 0.0 || norm_b == 0.0 {
            return FLOAT32_MAX;
        }

        let cosine_part = -(ip / (norm_a * norm_b).sqrt()).log2();
        if ip >= 0.0 {
            cosine_part + 1.0 / ip.sqrt()
        } else {
            FLOAT32_MAX
        }
    }

    fn name(&self) -> &'static str {
        "proxy_inner_product"
    }
}

// ---- Correction functions ----

/// Correction for alternative cosine/dot: d = 1 - 2^(-d_alt)
#[inline]
pub fn correct_alternative_cosine(d: f32) -> f32 {
    1.0 - 2.0f32.powf(-d)
}

/// Correction for alternative inner product: d = -1/d_alt
#[inline]
pub fn correct_alternative_inner_product(d: f32) -> f32 {
    if d >= FLOAT32_MAX {
        0.0
    } else {
        -1.0 / d
    }
}

/// Correction for true angular from alternative cosine:
/// d = 1 - arccos(2^(-d_alt)) / π
#[inline]
pub fn true_angular_from_alt_cosine(d: f32) -> f32 {
    1.0 - (2.0f32.powf(-d).acos()) / std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alternative_dot_identical_normalized() {
        // Normalized vector dot with itself = 1 → -log₂(1) = 0
        let norm = 1.0 / 3.0f32.sqrt();
        let a = vec![norm, norm, norm];
        let d = AlternativeDot.distance(&a, &a);
        assert!(d.abs() < 1e-5);
    }

    #[test]
    fn test_alternative_dot_matches_scalar_for_remainders() {
        for dim in [1, 7, 8, 31, 32, 100, 128] {
            let a: Vec<f32> = (0..dim).map(|i| ((i * 13 + 5) as f32).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i * 7 + 3) as f32).cos()).collect();
            let expected = alternative_dot_from_similarity(normalized_dot_scalar(&a, &b));
            let actual = AlternativeDot.distance(&a, &b);

            if expected == FLOAT32_MAX {
                assert_eq!(actual, FLOAT32_MAX);
            } else {
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "dim={dim}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn test_direct_normalized_cosine() {
        let a = [1.0, 0.0, 0.0];
        let same = DirectNormalizedCosine.distance(&a, &a);
        let orthogonal = DirectNormalizedCosine.distance(&a, &[0.0, 1.0, 0.0]);
        let opposite = DirectNormalizedCosine.distance(&a, &[-1.0, 0.0, 0.0]);
        assert!(same.abs() < 1e-6);
        assert!((orthogonal - 1.0).abs() < 1e-6);
        assert!((opposite - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_alternative_inner_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0, 1.0];
        // dot = 6 → 1/6 ≈ 0.1667
        let d = AlternativeInnerProduct.distance(&a, &b);
        assert!((d - 1.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_correct_alternative_cosine() {
        // d_alt = 0 → 1 - 2^0 = 0 (identical)
        assert!((correct_alternative_cosine(0.0)).abs() < 1e-6);
        // d_alt = 1 → 1 - 2^(-1) = 0.5
        assert!((correct_alternative_cosine(1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_correct_alternative_inner_product() {
        // d_alt = 0.5 → -1/0.5 = -2
        assert!((correct_alternative_inner_product(0.5) - (-2.0)).abs() < 1e-6);
        // d_alt = FLOAT32_MAX → 0.0
        assert!((correct_alternative_inner_product(FLOAT32_MAX)).abs() < 1e-6);
    }
}
