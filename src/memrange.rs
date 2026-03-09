//! SIMD-accelerated byte-range search.
//!
//! `memrange(lo, hi, haystack)` finds the index of the first byte `b` in
//! `haystack` satisfying `lo <= b <= hi`.
//!
//! **Core insight**: `clamp(x, lo, hi) == x` iff `lo <= x <= hi`.
//! In vector ops: `cmpeq(min(max(x, lo), hi), x)` — 3 SIMD ops per chunk.
//!
//! Provides SSE2/AVX2 paths on x86_64, NEON on aarch64, and a scalar
//! fallback for other targets or short haystacks.

/// Find the first byte in `haystack` in the inclusive range `[lo, hi]`.
///
/// Returns `None` if `lo > hi`, the haystack is empty, or no byte is in range.
#[inline]
pub(crate) fn memrange(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
    if lo > hi || haystack.is_empty() {
        return None;
    }
    imp::memrange(lo, hi, haystack)
}

/// Scalar fallback: simple byte-at-a-time loop.
#[inline]
fn memrange_fallback(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| b >= lo && b <= hi)
}

// ---------------------------------------------------------------------------
// x86_64: SSE2 (16 B) + AVX2 (32 B)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
mod imp {
    use super::memrange_fallback;
    use core::arch::x86_64::*;

    #[inline]
    pub(super) fn memrange(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime.
            unsafe { memrange_avx2(lo, hi, haystack) }
        } else {
            // SAFETY: SSE2 is baseline on x86_64.
            unsafe { memrange_sse2(lo, hi, haystack) }
        }
    }

    // -- SSE2 (16 bytes per iteration) --

    const SSE2_WIDTH: usize = 16;

    #[target_feature(enable = "sse2")]
    unsafe fn memrange_sse2(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
        let len = haystack.len();
        if len < SSE2_WIDTH {
            return memrange_fallback(lo, hi, haystack);
        }

        let ptr = haystack.as_ptr();
        let v_lo = _mm_set1_epi8(lo as i8);
        let v_hi = _mm_set1_epi8(hi as i8);

        // First unaligned load: [0, 16)
        let chunk = _mm_loadu_si128(ptr as *const __m128i);
        let bits = test_sse2(chunk, v_lo, v_hi);
        if bits != 0 {
            return Some(bits.trailing_zeros() as usize);
        }

        // Aligned middle loop.
        let aligned_start = (ptr as usize + SSE2_WIDTH) & !(SSE2_WIDTH - 1);
        let aligned_end = (ptr as usize + len) & !(SSE2_WIDTH - 1);
        let mut p = aligned_start as *const __m128i;
        let p_end = aligned_end as *const __m128i;

        while p < p_end {
            let chunk = _mm_load_si128(p);
            let bits = test_sse2(chunk, v_lo, v_hi);
            if bits != 0 {
                let offset = (p as usize) - (ptr as usize);
                return Some(offset + bits.trailing_zeros() as usize);
            }
            p = p.add(1);
        }

        // Final unaligned load: [len-16, len)
        let chunk = _mm_loadu_si128(ptr.add(len - SSE2_WIDTH) as *const __m128i);
        let bits = test_sse2(chunk, v_lo, v_hi);
        if bits != 0 {
            return Some(len - SSE2_WIDTH + bits.trailing_zeros() as usize);
        }

        None
    }

    /// Test 16 bytes: returns a bitmask where bit `i` is set if byte `i`
    /// is in `[lo, hi]`.
    #[inline(always)]
    unsafe fn test_sse2(chunk: __m128i, v_lo: __m128i, v_hi: __m128i) -> i32 {
        let clamped = _mm_min_epu8(_mm_max_epu8(chunk, v_lo), v_hi);
        _mm_movemask_epi8(_mm_cmpeq_epi8(clamped, chunk))
    }

    // -- AVX2 (32 bytes per iteration) --

    const AVX2_WIDTH: usize = 32;

    #[target_feature(enable = "avx2")]
    unsafe fn memrange_avx2(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
        let len = haystack.len();
        if len < AVX2_WIDTH {
            // Fall through to SSE2 for 16–31 byte inputs.
            return memrange_sse2(lo, hi, haystack);
        }

        let ptr = haystack.as_ptr();
        let v_lo = _mm256_set1_epi8(lo as i8);
        let v_hi = _mm256_set1_epi8(hi as i8);

        // First unaligned load: [0, 32)
        let chunk = _mm256_loadu_si256(ptr as *const __m256i);
        let bits = test_avx2(chunk, v_lo, v_hi);
        if bits != 0 {
            return Some(bits.trailing_zeros() as usize);
        }

        // Aligned middle loop.
        let aligned_start = (ptr as usize + AVX2_WIDTH) & !(AVX2_WIDTH - 1);
        let aligned_end = (ptr as usize + len) & !(AVX2_WIDTH - 1);
        let mut p = aligned_start as *const __m256i;
        let p_end = aligned_end as *const __m256i;

        while p < p_end {
            let chunk = _mm256_load_si256(p);
            let bits = test_avx2(chunk, v_lo, v_hi);
            if bits != 0 {
                let offset = (p as usize) - (ptr as usize);
                return Some(offset + bits.trailing_zeros() as usize);
            }
            p = p.add(1);
        }

        // Final unaligned load: [len-32, len)
        let chunk = _mm256_loadu_si256(ptr.add(len - AVX2_WIDTH) as *const __m256i);
        let bits = test_avx2(chunk, v_lo, v_hi);
        if bits != 0 {
            return Some(len - AVX2_WIDTH + bits.trailing_zeros() as usize);
        }

        None
    }

    /// Test 32 bytes: returns a bitmask where bit `i` is set if byte `i`
    /// is in `[lo, hi]`.
    #[inline(always)]
    unsafe fn test_avx2(chunk: __m256i, v_lo: __m256i, v_hi: __m256i) -> i32 {
        let clamped = _mm256_min_epu8(_mm256_max_epu8(chunk, v_lo), v_hi);
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(clamped, chunk))
    }
}

// ---------------------------------------------------------------------------
// aarch64: NEON (16 B)
// ---------------------------------------------------------------------------
// Following memchr's approach: NEON is mandatory in ARMv8-A and enabled by
// default in all standard Rust aarch64 targets.  Use compile-time
// `target_feature = "neon"` (not runtime detection).  Exotic targets
// compiled without NEON fall through to the scalar fallback.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod imp {
    use super::memrange_fallback;
    use core::arch::aarch64::*;

    const NEON_WIDTH: usize = 16;

    #[inline]
    pub(super) fn memrange(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
        // SAFETY: NEON availability guaranteed by cfg(target_feature = "neon").
        unsafe { memrange_neon(lo, hi, haystack) }
    }

    /// NEON: 16 bytes per iteration.
    ///
    /// Match extraction uses u64 lane inspection instead of a movemask
    /// (NEON has no direct movemask equivalent).  Each matched byte is
    /// 0xFF in the comparison result; we extract two u64 halves and use
    /// `trailing_zeros() / 8` to find the first set byte.
    #[target_feature(enable = "neon")]
    unsafe fn memrange_neon(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
        let len = haystack.len();
        if len < NEON_WIDTH {
            return memrange_fallback(lo, hi, haystack);
        }

        unsafe {
            let ptr = haystack.as_ptr();
            let v_lo = vdupq_n_u8(lo);
            let v_hi = vdupq_n_u8(hi);

            // First load: [0, 16)
            let chunk = vld1q_u8(ptr);
            if let Some(pos) = test_neon(chunk, v_lo, v_hi) {
                return Some(pos);
            }

            // Aligned middle loop.
            let aligned_start = (ptr as usize + NEON_WIDTH) & !(NEON_WIDTH - 1);
            let aligned_end = (ptr as usize + len) & !(NEON_WIDTH - 1);
            let mut p = aligned_start as *const u8;
            let p_end = aligned_end as *const u8;

            while p < p_end {
                let chunk = vld1q_u8(p);
                if let Some(pos) = test_neon(chunk, v_lo, v_hi) {
                    let offset = (p as usize) - (ptr as usize);
                    return Some(offset + pos);
                }
                p = p.add(NEON_WIDTH);
            }

            // Final load: [len-16, len)
            let chunk = vld1q_u8(ptr.add(len - NEON_WIDTH));
            if let Some(pos) = test_neon(chunk, v_lo, v_hi) {
                return Some(len - NEON_WIDTH + pos);
            }

            None
        }
    }

    /// Test 16 bytes: returns `Some(pos)` of the first byte in `[lo, hi]`,
    /// or `None` if no byte matches.
    #[inline(always)]
    unsafe fn test_neon(chunk: uint8x16_t, v_lo: uint8x16_t, v_hi: uint8x16_t) -> Option<usize> {
        unsafe {
            let clamped = vminq_u8(vmaxq_u8(chunk, v_lo), v_hi);
            let eq = vceqq_u8(clamped, chunk);

            // Extract first match position via u64 lane inspection.
            // Each matching byte is 0xFF (all bits set), non-matching is 0x00.
            // On little-endian aarch64, byte 0 occupies the LSB of lane 0.
            let eq_u64 = vreinterpretq_u64_u8(eq);
            let lo64 = vgetq_lane_u64::<0>(eq_u64);
            if lo64 != 0 {
                return Some((lo64.trailing_zeros() / 8) as usize);
            }
            let hi64 = vgetq_lane_u64::<1>(eq_u64);
            if hi64 != 0 {
                return Some(8 + (hi64.trailing_zeros() / 8) as usize);
            }
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback for other architectures (or aarch64 without NEON)
// ---------------------------------------------------------------------------
#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
)))]
mod imp {
    use super::memrange_fallback;

    #[inline]
    pub(super) fn memrange(lo: u8, hi: u8, haystack: &[u8]) -> Option<usize> {
        memrange_fallback(lo, hi, haystack)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::memrange;

    #[test]
    fn empty_haystack() {
        assert_eq!(memrange(0, 255, &[]), None);
    }

    #[test]
    fn inverted_range() {
        assert_eq!(memrange(10, 5, &[7]), None);
    }

    #[test]
    fn single_byte_exact() {
        assert_eq!(memrange(5, 5, &[5]), Some(0));
        assert_eq!(memrange(5, 5, &[4]), None);
        assert_eq!(memrange(5, 5, &[6]), None);
    }

    #[test]
    fn digit_range() {
        assert_eq!(memrange(b'0', b'9', b"hello world 42!"), Some(12));
    }

    #[test]
    fn no_match() {
        let hay = vec![b'A'; 128];
        assert_eq!(memrange(b'0', b'9', &hay), None);
    }

    #[test]
    fn match_first_byte() {
        assert_eq!(memrange(b'0', b'9', b"5hello"), Some(0));
    }

    #[test]
    fn match_last_byte() {
        assert_eq!(memrange(b'0', b'9', b"hello5"), Some(5));
    }

    #[test]
    fn full_range() {
        assert_eq!(memrange(0, 255, &[42]), Some(0));
    }

    #[test]
    fn boundary_values() {
        // lo and hi themselves must match.
        assert_eq!(memrange(10, 20, &[10]), Some(0));
        assert_eq!(memrange(10, 20, &[20]), Some(0));
        assert_eq!(memrange(10, 20, &[9]), None);
        assert_eq!(memrange(10, 20, &[21]), None);
    }

    #[test]
    fn high_byte_range() {
        // Range in the upper half of u8.
        let hay = [0x00, 0x7F, 0x80, 0xFE, 0xFF];
        assert_eq!(memrange(0x80, 0xFF, &hay), Some(2));
        assert_eq!(memrange(0xFE, 0xFF, &hay), Some(3));
        assert_eq!(memrange(0xFF, 0xFF, &hay), Some(4));
    }

    #[test]
    fn various_lengths() {
        // Test lengths around SIMD boundaries (16, 32).
        for len in 1..=128 {
            let mut hay = vec![b'A'; len];
            // No match.
            assert_eq!(memrange(b'0', b'9', &hay), None, "len={len}");
            // Match at last position.
            hay[len - 1] = b'5';
            assert_eq!(memrange(b'0', b'9', &hay), Some(len - 1), "len={len}");
            // Match at first position.
            hay[0] = b'3';
            assert_eq!(memrange(b'0', b'9', &hay), Some(0), "len={len}");
        }
    }

    #[test]
    fn alignment_offsets() {
        // Test at every alignment offset within a large buffer.
        let mut buf = vec![b'A'; 256 + 64];
        for offset in 0..64 {
            let hay = &mut buf[offset..offset + 128];
            for b in hay.iter_mut() {
                *b = b'A';
            }
            for pos in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 127] {
                hay[pos] = b'5';
                assert_eq!(
                    memrange(b'0', b'9', hay),
                    Some(pos),
                    "offset={offset}, pos={pos}",
                );
                hay[pos] = b'A';
            }
        }
    }

    #[test]
    fn compare_with_scalar() {
        // Compare SIMD result against a known-correct scalar implementation.
        let scalar = |lo: u8, hi: u8, hay: &[u8]| -> Option<usize> {
            hay.iter().position(|&b| b >= lo && b <= hi)
        };

        // Deterministic pseudo-random data (LCG).
        let mut data = vec![0u8; 1024];
        let mut state: u32 = 0xDEAD_BEEF;
        for b in data.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *b = (state >> 16) as u8;
        }

        let ranges: &[(u8, u8)] = &[
            (b'0', b'9'),
            (b'a', b'z'),
            (b'A', b'F'),
            (0, 31),
            (128, 255),
            (100, 100),
            (0, 0),
            (255, 255),
        ];

        for &(lo, hi) in ranges {
            for start in 0..64 {
                for len in [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 128, 256, 512] {
                    if start + len > data.len() {
                        continue;
                    }
                    let hay = &data[start..start + len];
                    let expected = scalar(lo, hi, hay);
                    let actual = memrange(lo, hi, hay);
                    assert_eq!(
                        actual, expected,
                        "lo={lo}, hi={hi}, start={start}, len={len}",
                    );
                }
            }
        }
    }
}
