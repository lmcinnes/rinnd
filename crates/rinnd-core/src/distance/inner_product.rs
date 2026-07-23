//! Inner product (dot product) distance implementation.
//!
//! Inner product is used for maximum inner product search (MIPS),
//! commonly used with embeddings where higher dot product indicates
//! more similarity.

use super::traits::Distance;

/// Inner product distance: negative of dot product.
///
/// Since we want smaller values to indicate "closer" (more similar),
/// we negate the inner product so that higher similarity results in
/// lower distance.
///
/// Formula: d(a, b) = -Σ(aᵢ × bᵢ)
///
/// Note: For normalized vectors, this is equivalent to cosine similarity.
/// Use this for maximum inner product search (MIPS) problems.
#[derive(Clone, Copy, Debug, Default)]
pub struct InnerProduct;

#[inline]
fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
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

    dot
}

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let mut vdot = _mm256_setzero_ps();

    for i in 0..chunks {
        let idx = i * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(idx));
        let vb = _mm256_loadu_ps(b.as_ptr().add(idx));
        vdot = _mm256_fmadd_ps(va, vb, vdot);
    }

    let mut dot = hsum256_ps(vdot);
    for i in (chunks * 8)..n {
        dot += a[i] * b[i];
    }
    dot
}

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    scalar_dot(a, b)
}

impl Distance<f32> for InnerProduct {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product(a, b);
        // Negate so that higher similarity = lower distance
        -dot
    }

    fn name(&self) -> &'static str {
        "inner_product"
    }
}

/// Dot distance used by pynndescent for normalized vectors.
///
/// Formula: 1 - dot(a, b), clamped to 1.0 when dot <= 0.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dot;

impl Distance<f32> for Dot {
    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot = dot_product(a, b);
        if dot <= 0.0 {
            1.0
        } else {
            1.0 - dot
        }
    }

    fn name(&self) -> &'static str {
        "dot"
    }
}

/// Raw dot product (not negated).
///
/// This computes the standard dot product without negation.
/// Useful when you need the actual inner product value.
#[derive(Clone, Copy, Debug, Default)]
pub struct DotProduct;

impl DotProduct {
    /// Compute the dot product of two vectors.
    #[inline]
    pub fn compute(a: &[f32], b: &[f32]) -> f32 {
        dot_product(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inner_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // dot = 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let d = InnerProduct.distance(&a, &b);
        assert!((d - (-32.0)).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        // Orthogonal vectors have dot product 0
        let d = InnerProduct.distance(&a, &b);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_parallel() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![2.0, 0.0, 0.0];

        // Parallel vectors: dot = 2
        let d = InnerProduct.distance(&a, &b);
        assert!((d - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_antiparallel() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];

        // Anti-parallel vectors: dot = -1, distance = 1
        let d = InnerProduct.distance(&a, &b);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let dot = DotProduct::compute(&a, &b);
        assert!((dot - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_self() {
        let a = vec![3.0, 4.0];

        // Self inner product = squared norm = 9 + 16 = 25
        let d = InnerProduct.distance(&a, &a);
        assert!((d - (-25.0)).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_large_vectors() {
        let dim = 512;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 + 0.5).collect();

        // Just verify it produces a reasonable value
        let d = InnerProduct.distance(&a, &b);
        assert!(d.is_finite());
    }

    #[test]
    fn test_inner_product_zeros() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];

        let d = InnerProduct.distance(&a, &b);
        assert!((d - 0.0).abs() < 1e-6);
    }
}
