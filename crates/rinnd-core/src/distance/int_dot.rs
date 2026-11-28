//! Integer dot products of unsigned code bytes with signed 8-bit query weights,
//! the inner loop of quantized search (see [`super::affine`]).
//!
//! 8-bit codes hold one value per byte; 4-bit codes hold two (the low nibble is
//! the even feature, the high nibble the odd one), and their weights are two
//! rows of the code row's length: low nibbles' weights, then high nibbles'.
//!
//! Code rows and weights are padded to whole 64-byte lines and stored 64-byte
//! aligned ([`AlignedBytes`]), so a row touches the fewest cache lines. With
//! AVX-512 VNNI each line costs one `vpdpbusd` (two for 4-bit codes); without
//! it a portable loop is used, compiled for AVX2 where available.

/// Bytes per cache line, the unit of code row storage.
pub const LINE: usize = 64;

/// One cache line.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct Line(pub [u8; LINE]);

/// Bytes stored in whole, 64-byte aligned lines.
#[derive(Clone, Debug, Default)]
pub struct AlignedBytes {
    lines: Vec<Line>,
}

impl AlignedBytes {
    /// `n_bytes` zero bytes (rounded up to whole lines).
    pub fn zeroed(n_bytes: usize) -> Self {
        Self {
            lines: vec![Line([0; LINE]); n_bytes.div_ceil(LINE)],
        }
    }

    /// Make this `n_bytes` long (rounded up to whole lines), all zero.
    pub fn reset_zeroed(&mut self, n_bytes: usize) {
        let n_lines = n_bytes.div_ceil(LINE);
        self.lines.clear();
        self.lines.resize(n_lines, Line([0; LINE]));
    }

    /// Total stored bytes (a multiple of [`LINE`]).
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.lines.len() * LINE
    }

    /// Whether nothing is stored.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        // Lines are plain bytes, laid out contiguously.
        unsafe { core::slice::from_raw_parts(self.lines.as_ptr() as *const u8, self.len()) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.lines.as_mut_ptr() as *mut u8, self.len()) }
    }

    /// The bytes as signed values (for query weights).
    #[inline(always)]
    pub fn as_i8_slice(&self) -> &[i8] {
        unsafe { core::slice::from_raw_parts(self.lines.as_ptr() as *const i8, self.len()) }
    }

    #[inline(always)]
    pub fn as_mut_i8_slice(&mut self) -> &mut [i8] {
        unsafe { core::slice::from_raw_parts_mut(self.lines.as_mut_ptr() as *mut i8, self.len()) }
    }
}

/// Which implementation computes the dot products.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kernel {
    /// AVX-512 VNNI `vpdpbusd` (needs whole-line rows).
    Vnni,
    /// The portable loop compiled for AVX2.
    Avx2,
    /// The portable loop.
    Portable,
}

impl Kernel {
    /// The fastest kernel this CPU supports.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni")
            {
                return Kernel::Vnni;
            }
            if is_x86_feature_detected!("avx2") {
                return Kernel::Avx2;
            }
        }
        Kernel::Portable
    }
}

/// `sum_t row[t] * weights[t]` for 8-bit codes (`weights.len() == row.len()`).
#[inline(always)]
pub fn dot_u8(kernel: Kernel, row: &[u8], weights: &[i8]) -> i32 {
    debug_assert_eq!(row.len(), weights.len());
    match kernel {
        #[cfg(target_arch = "x86_64")]
        Kernel::Vnni => unsafe { vnni::dot4_u8([row; 4], weights, 1)[0] },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => unsafe { avx2::dot_u8(row, weights) },
        _ => portable_dot_u8(row, weights),
    }
}

/// [`dot_u8`] of four rows.
#[inline(always)]
pub fn dot4_u8(kernel: Kernel, rows: [&[u8]; 4], weights: &[i8]) -> [i32; 4] {
    match kernel {
        #[cfg(target_arch = "x86_64")]
        Kernel::Vnni => unsafe { vnni::dot4_u8(rows, weights, 4) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => unsafe { avx2::dot4_u8(rows, weights) },
        _ => rows.map(|row| portable_dot_u8(row, weights)),
    }
}

/// `sum_t lo(row[t]) * weights[t] + hi(row[t]) * weights[n + t]` for 4-bit
/// codes (`n == row.len()`, `weights.len() == 2 * n`).
#[inline(always)]
pub fn dot_u4(kernel: Kernel, row: &[u8], weights: &[i8]) -> i32 {
    debug_assert_eq!(2 * row.len(), weights.len());
    match kernel {
        #[cfg(target_arch = "x86_64")]
        Kernel::Vnni => unsafe { vnni::dot4_u4([row; 4], weights, 1)[0] },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => unsafe { avx2::dot_u4(row, weights) },
        _ => portable_dot_u4(row, weights),
    }
}

/// [`dot_u4`] of four rows.
#[inline(always)]
pub fn dot4_u4(kernel: Kernel, rows: [&[u8]; 4], weights: &[i8]) -> [i32; 4] {
    match kernel {
        #[cfg(target_arch = "x86_64")]
        Kernel::Vnni => unsafe { vnni::dot4_u4(rows, weights, 4) },
        #[cfg(target_arch = "x86_64")]
        Kernel::Avx2 => unsafe { avx2::dot4_u4(rows, weights) },
        _ => rows.map(|row| portable_dot_u4(row, weights)),
    }
}

#[inline(always)]
fn portable_dot_u8(row: &[u8], weights: &[i8]) -> i32 {
    row.iter().zip(weights).map(|(&c, &w)| i32::from(c) * i32::from(w)).sum()
}

#[inline(always)]
fn portable_dot_u4(row: &[u8], weights: &[i8]) -> i32 {
    let (low, high) = weights.split_at(row.len());
    row.iter()
        .zip(low)
        .zip(high)
        .map(|((&c, &lo), &hi)| i32::from(c & 15) * i32::from(lo) + i32::from(c >> 4) * i32::from(hi))
        .sum()
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{portable_dot_u4, portable_dot_u8};

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_u8(row: &[u8], weights: &[i8]) -> i32 {
        portable_dot_u8(row, weights)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot4_u8(rows: [&[u8]; 4], weights: &[i8]) -> [i32; 4] {
        rows.map(|row| portable_dot_u8(row, weights))
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_u4(row: &[u8], weights: &[i8]) -> i32 {
        portable_dot_u4(row, weights)
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot4_u4(rows: [&[u8]; 4], weights: &[i8]) -> [i32; 4] {
        rows.map(|row| portable_dot_u4(row, weights))
    }
}

#[cfg(target_arch = "x86_64")]
mod vnni {
    use core::arch::x86_64::*;

    /// Dot products of the first `count` (1 or 4) of `rows`; the rest are zero.
    /// Rows must be whole lines long.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn dot4_u8(rows: [&[u8]; 4], weights: &[i8], count: usize) -> [i32; 4] {
        let n = weights.len();
        debug_assert_eq!(n % super::LINE, 0);
        let w = weights.as_ptr();
        let mut acc = [_mm512_setzero_si512(); 4];
        let mut offset = 0;
        while offset < n {
            let wv = _mm512_loadu_si512(w.add(offset).cast());
            for r in 0..count {
                let c = _mm512_loadu_si512(rows[r].as_ptr().add(offset).cast());
                acc[r] = _mm512_dpbusd_epi32(acc[r], c, wv);
            }
            offset += super::LINE;
        }
        let mut out = [0i32; 4];
        for r in 0..count {
            out[r] = _mm512_reduce_add_epi32(acc[r]);
        }
        out
    }

    /// As [`dot4_u8`] for 4-bit codes (weights: low nibbles', then high nibbles').
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn dot4_u4(rows: [&[u8]; 4], weights: &[i8], count: usize) -> [i32; 4] {
        let n = weights.len() / 2;
        debug_assert_eq!(n % super::LINE, 0);
        let w_low = weights.as_ptr();
        let w_high = w_low.add(n);
        let mask = _mm512_set1_epi8(0x0F);
        let mut acc = [_mm512_setzero_si512(); 4];
        let mut offset = 0;
        while offset < n {
            let wl = _mm512_loadu_si512(w_low.add(offset).cast());
            let wh = _mm512_loadu_si512(w_high.add(offset).cast());
            for r in 0..count {
                let c = _mm512_loadu_si512(rows[r].as_ptr().add(offset).cast());
                let low = _mm512_and_si512(c, mask);
                let high = _mm512_and_si512(_mm512_srli_epi16::<4>(c), mask);
                acc[r] = _mm512_dpbusd_epi32(acc[r], low, wl);
                acc[r] = _mm512_dpbusd_epi32(acc[r], high, wh);
            }
            offset += super::LINE;
        }
        let mut out = [0i32; 4];
        for r in 0..count {
            out[r] = _mm512_reduce_add_epi32(acc[r]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::FastRng;

    fn random_bytes(n: usize, seed: u64) -> AlignedBytes {
        let mut rng = FastRng::new(seed);
        let mut out = AlignedBytes::zeroed(n);
        for b in out.as_mut_slice().iter_mut() {
            *b = (rng.next_float() * 256.0) as u8;
        }
        out
    }

    #[test]
    fn aligned_bytes_are_line_aligned_and_zeroed() {
        let bytes = AlignedBytes::zeroed(70);
        assert_eq!(bytes.len(), 128);
        assert_eq!(bytes.as_slice().as_ptr() as usize % LINE, 0);
        assert!(bytes.as_slice().iter().all(|&b| b == 0));
    }

    #[test]
    fn kernels_match_the_portable_loops() {
        let mut kernels = vec![Kernel::Portable];
        let detected = Kernel::detect();
        if detected != Kernel::Portable {
            kernels.push(Kernel::Avx2);
        }
        if detected == Kernel::Vnni {
            kernels.push(Kernel::Vnni);
        }
        for n in [64, 128, 384, 1024] {
            let rows: Vec<AlignedBytes> = (0..4).map(|i| random_bytes(n, 10 + i)).collect();
            let weights = random_bytes(2 * n, 99);
            let w = weights.as_i8_slice();
            let views = [rows[0].as_slice(), rows[1].as_slice(), rows[2].as_slice(), rows[3].as_slice()];
            let expect8 = views.map(|r| portable_dot_u8(r, &w[..n]));
            let expect4 = views.map(|r| portable_dot_u4(r, w));
            for &kernel in &kernels {
                assert_eq!(dot4_u8(kernel, views, &w[..n]), expect8, "{kernel:?} n={n}");
                assert_eq!(dot4_u4(kernel, views, w), expect4, "{kernel:?} n={n}");
                assert_eq!(dot_u8(kernel, views[2], &w[..n]), expect8[2]);
                assert_eq!(dot_u4(kernel, views[3], w), expect4[3]);
            }
            // Spot check the portable loops themselves.
            let direct: i32 = (0..n).map(|t| i32::from(views[0][t]) * i32::from(w[t])).sum();
            assert_eq!(expect8[0], direct);
        }
    }
}
