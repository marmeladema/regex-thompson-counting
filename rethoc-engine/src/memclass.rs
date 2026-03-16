//! SIMD-accelerated byte-class membership search.
//!
//! `memclass(table, haystack)` finds the index of the first byte in
//! `haystack` that belongs to a precomputed set of up to 8 bytes.
//!
//! **Core technique**: split each byte into two 4-bit nibbles, use
//! SIMD shuffle (`vtbl`/`vpshufb`) to look up each nibble in a
//! precomputed 16-entry table, AND the results.  A byte matches iff
//! both its low and high nibble lookups share a common bit.
//!
//! The lookup tables are built once at regex compile time and stored
//! on the [`MemclassTable`] struct.  The per-vector cost is constant
//! (2 shuffles + 1 AND + mask extraction) regardless of how many
//! bytes are in the class (up to 8).
//!
//! Provides SSE2/AVX2 paths on x86_64, NEON on aarch64, and a scalar
//! fallback for other targets or short haystacks.

/// Precomputed lookup tables for SIMD byte-class membership testing.
///
/// Each target byte is assigned a unique bit position (0–7).  The
/// low-nibble table maps `byte & 0xF` to the OR of bit positions of
/// all target bytes with that low nibble.  The high-nibble table does
/// the same for `byte >> 4`.  A byte is a member iff
/// `lo_table[b & 0xF] & hi_table[b >> 4] != 0`.
#[derive(Clone, Debug)]
pub struct MemclassTable {
    /// Lookup table indexed by low nibble (byte & 0xF).
    pub(crate) lo: [u8; 16],
    /// Lookup table indexed by high nibble (byte >> 4).
    pub(crate) hi: [u8; 16],
}

impl MemclassTable {
    /// Build lookup tables for the given set of target bytes.
    ///
    /// # Panics
    ///
    /// Panics if `bytes` is empty or contains more than 8 distinct values.
    pub fn new(bytes: &[u8]) -> Self {
        assert!(!bytes.is_empty(), "memclass: need at least 1 byte");
        assert!(bytes.len() <= 8, "memclass: at most 8 bytes supported");

        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];

        for (i, &b) in bytes.iter().enumerate() {
            let bit = 1u8 << i;
            lo[(b & 0x0F) as usize] |= bit;
            hi[(b >> 4) as usize] |= bit;
        }

        Self { lo, hi }
    }

    /// Scalar membership test for a single byte.
    #[inline]
    pub fn contains(&self, b: u8) -> bool {
        self.lo[(b & 0x0F) as usize] & self.hi[(b >> 4) as usize] != 0
    }
}

/// Find the first byte in `haystack` that belongs to the class defined
/// by `table`.
///
/// Returns `None` if the haystack is empty or no byte matches.
///
/// Uses SIMD acceleration (NEON on aarch64, SSE2/AVX2 on x86_64) when
/// available, falling back to a scalar loop for short haystacks or
/// unsupported targets.
#[inline]
pub fn memclass(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
    if haystack.is_empty() {
        return None;
    }
    imp::memclass(table, haystack)
}

/// Scalar byte-at-a-time implementation, exposed for benchmarking.
#[doc(hidden)]
#[inline]
pub fn memclass_scalar(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
    if haystack.is_empty() {
        return None;
    }
    memclass_fallback(table, haystack)
}

/// Scalar fallback: simple byte-at-a-time loop.
#[inline]
fn memclass_fallback(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&b| table.contains(b))
}

// ---------------------------------------------------------------------------
// x86_64: SSE2 (16 B) + AVX2 (32 B)
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
mod imp {
    use super::{MemclassTable, memclass_fallback};
    use core::arch::x86_64::*;

    #[inline]
    pub(super) fn memclass(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 detected at runtime.
            unsafe { memclass_avx2(table, haystack) }
        } else {
            // SAFETY: SSE2 is baseline on x86_64.
            unsafe { memclass_sse2(table, haystack) }
        }
    }

    // -- SSE2 (16 bytes per iteration) --

    const SSE2_WIDTH: usize = 16;

    #[target_feature(enable = "sse2", enable = "ssse3")]
    unsafe fn memclass_sse2(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
        let len = haystack.len();
        if len < SSE2_WIDTH {
            return memclass_fallback(table, haystack);
        }

        let ptr = haystack.as_ptr();
        let v_lo_table = _mm_loadu_si128(table.lo.as_ptr() as *const __m128i);
        let v_hi_table = _mm_loadu_si128(table.hi.as_ptr() as *const __m128i);
        let v_mask_lo = _mm_set1_epi8(0x0F);

        // First unaligned load: [0, 16)
        let chunk = _mm_loadu_si128(ptr as *const __m128i);
        let bits = test_sse2(chunk, v_lo_table, v_hi_table, v_mask_lo);
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
            let bits = test_sse2(chunk, v_lo_table, v_hi_table, v_mask_lo);
            if bits != 0 {
                let offset = (p as usize) - (ptr as usize);
                return Some(offset + bits.trailing_zeros() as usize);
            }
            p = p.add(1);
        }

        // Final unaligned load: [len-16, len)
        let chunk = _mm_loadu_si128(ptr.add(len - SSE2_WIDTH) as *const __m128i);
        let bits = test_sse2(chunk, v_lo_table, v_hi_table, v_mask_lo);
        if bits != 0 {
            return Some(len - SSE2_WIDTH + bits.trailing_zeros() as usize);
        }

        None
    }

    /// Test 16 bytes: returns a bitmask where bit `i` is set if byte `i`
    /// is a member of the class.
    #[inline(always)]
    unsafe fn test_sse2(
        chunk: __m128i,
        v_lo_table: __m128i,
        v_hi_table: __m128i,
        v_mask_lo: __m128i,
    ) -> i32 {
        // Split each byte into nibbles.
        let lo_nibbles = _mm_and_si128(chunk, v_mask_lo);
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(chunk, 4), v_mask_lo);
        // Shuffle-lookup each nibble in its table.
        let lo_bits = _mm_shuffle_epi8(v_lo_table, lo_nibbles);
        let hi_bits = _mm_shuffle_epi8(v_hi_table, hi_nibbles);
        // A byte matches iff both nibble lookups share a bit.
        let matched = _mm_and_si128(lo_bits, hi_bits);
        // Convert to bitmask: any non-zero byte → bit set.
        // _mm_cmpeq_epi8 with zero gives 0xFF for zero bytes;
        // we want the inverse (non-zero = match).
        let zero = _mm_setzero_si128();
        let is_zero = _mm_cmpeq_epi8(matched, zero);
        // Invert: matched bytes have 0 in is_zero → bit clear in movemask.
        // So we invert the movemask.
        !_mm_movemask_epi8(is_zero) & 0xFFFF
    }

    // -- AVX2 (32 bytes per iteration) --

    const AVX2_WIDTH: usize = 32;

    #[target_feature(enable = "avx2")]
    unsafe fn memclass_avx2(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
        let len = haystack.len();
        if len < AVX2_WIDTH {
            return memclass_sse2(table, haystack);
        }

        let ptr = haystack.as_ptr();
        // Broadcast 16-byte tables to both 128-bit lanes of a 256-bit register.
        let lo_128 = _mm_loadu_si128(table.lo.as_ptr() as *const __m128i);
        let hi_128 = _mm_loadu_si128(table.hi.as_ptr() as *const __m128i);
        let v_lo_table = _mm256_broadcastsi128_si256(lo_128);
        let v_hi_table = _mm256_broadcastsi128_si256(hi_128);
        let v_mask_lo = _mm256_set1_epi8(0x0F);

        // First unaligned load: [0, 32)
        let chunk = _mm256_loadu_si256(ptr as *const __m256i);
        let bits = test_avx2(chunk, v_lo_table, v_hi_table, v_mask_lo);
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
            let bits = test_avx2(chunk, v_lo_table, v_hi_table, v_mask_lo);
            if bits != 0 {
                let offset = (p as usize) - (ptr as usize);
                return Some(offset + bits.trailing_zeros() as usize);
            }
            p = p.add(1);
        }

        // Final unaligned load: [len-32, len)
        let chunk = _mm256_loadu_si256(ptr.add(len - AVX2_WIDTH) as *const __m256i);
        let bits = test_avx2(chunk, v_lo_table, v_hi_table, v_mask_lo);
        if bits != 0 {
            return Some(len - AVX2_WIDTH + bits.trailing_zeros() as usize);
        }

        None
    }

    /// Test 32 bytes: returns a bitmask where bit `i` is set if byte `i`
    /// is a member of the class.
    #[inline(always)]
    unsafe fn test_avx2(
        chunk: __m256i,
        v_lo_table: __m256i,
        v_hi_table: __m256i,
        v_mask_lo: __m256i,
    ) -> i32 {
        let lo_nibbles = _mm256_and_si256(chunk, v_mask_lo);
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), v_mask_lo);
        let lo_bits = _mm256_shuffle_epi8(v_lo_table, lo_nibbles);
        let hi_bits = _mm256_shuffle_epi8(v_hi_table, hi_nibbles);
        let matched = _mm256_and_si256(lo_bits, hi_bits);
        let zero = _mm256_setzero_si256();
        let is_zero = _mm256_cmpeq_epi8(matched, zero);
        !_mm256_movemask_epi8(is_zero)
    }
}

// ---------------------------------------------------------------------------
// aarch64: NEON (16 B)
// ---------------------------------------------------------------------------
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod imp {
    use super::{MemclassTable, memclass_fallback};
    use core::arch::aarch64::*;

    const NEON_WIDTH: usize = 16;

    #[inline]
    pub(super) fn memclass(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
        // SAFETY: NEON availability guaranteed by cfg(target_feature = "neon").
        unsafe { memclass_neon(table, haystack) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn memclass_neon(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
        let len = haystack.len();
        if len < NEON_WIDTH {
            return memclass_fallback(table, haystack);
        }

        unsafe {
            let ptr = haystack.as_ptr();
            let v_lo_table = vld1q_u8(table.lo.as_ptr());
            let v_hi_table = vld1q_u8(table.hi.as_ptr());
            let v_mask_lo = vdupq_n_u8(0x0F);

            // First load: [0, 16)
            let chunk = vld1q_u8(ptr);
            if let Some(pos) = test_neon(chunk, v_lo_table, v_hi_table, v_mask_lo) {
                return Some(pos);
            }

            // Aligned middle loop.
            let aligned_start = (ptr as usize + NEON_WIDTH) & !(NEON_WIDTH - 1);
            let aligned_end = (ptr as usize + len) & !(NEON_WIDTH - 1);
            let mut p = aligned_start as *const u8;
            let p_end = aligned_end as *const u8;

            while p < p_end {
                let chunk = vld1q_u8(p);
                if let Some(pos) = test_neon(chunk, v_lo_table, v_hi_table, v_mask_lo) {
                    let offset = (p as usize) - (ptr as usize);
                    return Some(offset + pos);
                }
                p = p.add(NEON_WIDTH);
            }

            // Final load: [len-16, len)
            let chunk = vld1q_u8(ptr.add(len - NEON_WIDTH));
            if let Some(pos) = test_neon(chunk, v_lo_table, v_hi_table, v_mask_lo) {
                return Some(len - NEON_WIDTH + pos);
            }

            None
        }
    }

    /// Test 16 bytes: returns `Some(pos)` of the first matching byte.
    #[inline(always)]
    unsafe fn test_neon(
        chunk: uint8x16_t,
        v_lo_table: uint8x16_t,
        v_hi_table: uint8x16_t,
        v_mask_lo: uint8x16_t,
    ) -> Option<usize> {
        unsafe {
            // Split into nibbles.
            let lo_nibbles = vandq_u8(chunk, v_mask_lo);
            let hi_nibbles = vshrq_n_u8(chunk, 4);
            // Shuffle-lookup each nibble.
            let lo_bits = vqtbl1q_u8(v_lo_table, lo_nibbles);
            let hi_bits = vqtbl1q_u8(v_hi_table, hi_nibbles);
            // Match = both nibble lookups share a bit.
            let matched = vandq_u8(lo_bits, hi_bits);

            // Find first non-zero byte via u64 lane inspection.
            let matched_u64 = vreinterpretq_u64_u8(matched);
            let lo64 = vgetq_lane_u64::<0>(matched_u64);
            if lo64 != 0 {
                return Some((lo64.trailing_zeros() / 8) as usize);
            }
            let hi64 = vgetq_lane_u64::<1>(matched_u64);
            if hi64 != 0 {
                return Some(8 + (hi64.trailing_zeros() / 8) as usize);
            }
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback for other architectures
// ---------------------------------------------------------------------------
#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
)))]
mod imp {
    use super::{MemclassTable, memclass_fallback};

    #[inline]
    pub(super) fn memclass(table: &MemclassTable, haystack: &[u8]) -> Option<usize> {
        memclass_fallback(table, haystack)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // ── MemclassTable construction ──

    #[test]
    fn test_table_single_byte() {
        let t = MemclassTable::new(&[b'a']);
        assert!(t.contains(b'a'));
        assert!(!t.contains(b'b'));
        assert!(!t.contains(0));
    }

    #[test]
    fn test_table_case_insensitive_pair() {
        let t = MemclassTable::new(&[b'a', b'A']);
        assert!(t.contains(b'a'));
        assert!(t.contains(b'A'));
        assert!(!t.contains(b'b'));
    }

    #[test]
    fn test_table_six_bytes() {
        // 3 case-insensitive letters
        let t = MemclassTable::new(&[b'a', b'A', b'b', b'B', b'c', b'C']);
        for &b in &[b'a', b'A', b'b', b'B', b'c', b'C'] {
            assert!(t.contains(b), "should contain {}", b as char);
        }
        assert!(!t.contains(b'd'));
        assert!(!t.contains(b'D'));
        assert!(!t.contains(0));
        assert!(!t.contains(255));
    }

    #[test]
    fn test_table_eight_bytes() {
        let t = MemclassTable::new(&[b'a', b'A', b'b', b'B', b'c', b'C', b'd', b'D']);
        for &b in &[b'a', b'A', b'b', b'B', b'c', b'C', b'd', b'D'] {
            assert!(t.contains(b));
        }
        assert!(!t.contains(b'e'));
    }

    #[test]
    #[should_panic]
    fn test_table_empty_panics() {
        MemclassTable::new(&[]);
    }

    #[test]
    #[should_panic]
    fn test_table_too_many_panics() {
        MemclassTable::new(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    // ── memclass search ──

    #[test]
    fn test_empty_haystack() {
        let t = MemclassTable::new(&[b'x']);
        assert_eq!(memclass(&t, &[]), None);
    }

    #[test]
    fn test_no_match() {
        let t = MemclassTable::new(&[b'x', b'y']);
        assert_eq!(memclass(&t, b"abcdefghijklmnop"), None);
    }

    #[test]
    fn test_first_byte_matches() {
        let t = MemclassTable::new(&[b'a', b'A']);
        assert_eq!(memclass(&t, b"abcdef"), Some(0));
    }

    #[test]
    fn test_last_byte_matches() {
        let t = MemclassTable::new(&[b'f']);
        assert_eq!(memclass(&t, b"abcdef"), Some(5));
    }

    #[test]
    fn test_match_in_middle() {
        let t = MemclassTable::new(&[b'x', b'X']);
        let hay = b"0123456789abcdefXhij";
        assert_eq!(memclass(&t, hay), Some(16));
    }

    #[test]
    fn test_six_byte_class() {
        let t = MemclassTable::new(&[b'a', b'A', b'b', b'B', b'c', b'C']);
        assert_eq!(memclass(&t, b"0123456789B"), Some(10));
        assert_eq!(memclass(&t, b"0123456789"), None);
    }

    // ── Large haystacks (exercise SIMD paths) ──

    #[test]
    fn test_large_no_match() {
        let t = MemclassTable::new(&[b'x']);
        let hay = vec![b'a'; 1024];
        assert_eq!(memclass(&t, &hay), None);
    }

    #[test]
    fn test_large_match_at_start() {
        let t = MemclassTable::new(&[b'x']);
        let mut hay = vec![b'a'; 1024];
        hay[0] = b'x';
        assert_eq!(memclass(&t, &hay), Some(0));
    }

    #[test]
    fn test_large_match_at_end() {
        let t = MemclassTable::new(&[b'x']);
        let mut hay = vec![b'a'; 1024];
        hay[1023] = b'x';
        assert_eq!(memclass(&t, &hay), Some(1023));
    }

    #[test]
    fn test_large_match_past_simd_boundary() {
        let t = MemclassTable::new(&[b'x', b'X']);
        let mut hay = vec![b'a'; 1024];
        hay[500] = b'X';
        assert_eq!(memclass(&t, &hay), Some(500));
    }

    #[test]
    fn test_all_bytes_match() {
        let t = MemclassTable::new(&[b'a']);
        let hay = vec![b'a'; 256];
        assert_eq!(memclass(&t, &hay), Some(0));
    }

    // ── Scalar vs SIMD consistency ──

    #[test]
    fn test_scalar_matches_simd() {
        let t = MemclassTable::new(&[b'a', b'A', b'b', b'B', b'x', b'X']);
        for pos in 0..256 {
            let mut hay = vec![b'.'; 256];
            hay[pos] = b'X';
            let scalar = memclass_scalar(&t, &hay);
            let simd = memclass(&t, &hay);
            assert_eq!(
                scalar, simd,
                "mismatch at pos {pos}: scalar={scalar:?}, simd={simd:?}"
            );
        }
    }

    // ── Boundary values ──

    #[test]
    fn test_byte_zero() {
        let t = MemclassTable::new(&[0u8]);
        assert_eq!(memclass(&t, &[1, 2, 3, 0, 5]), Some(3));
    }

    #[test]
    fn test_byte_255() {
        let t = MemclassTable::new(&[255u8]);
        assert_eq!(memclass(&t, &[0, 1, 2, 255]), Some(3));
    }

    #[test]
    fn test_same_nibble_different_bytes() {
        // 0x1A and 0x2A share low nibble 0xA.
        // 0x1B and 0x1A share high nibble 0x1.
        // Only 0x1A and 0x2A should match.
        let t = MemclassTable::new(&[0x1A, 0x2A]);
        assert!(t.contains(0x1A));
        assert!(t.contains(0x2A));
        // 0x1B shares high nibble with 0x1A and low nibble with neither → no match.
        assert!(!t.contains(0x1B));
        // 0x2B shares high nibble with 0x2A but not low nibble → no match.
        assert!(!t.contains(0x2B));
    }

    #[test]
    fn test_no_false_positives_nibble_collision() {
        // Adversarial: bytes that share nibbles with targets but aren't targets.
        // Targets: 0x12, 0x34
        // 0x14 shares low nibble with 0x34, high nibble with 0x12 → potential false positive.
        let t = MemclassTable::new(&[0x12, 0x34]);
        assert!(t.contains(0x12));
        assert!(t.contains(0x34));
        // 0x14: lo=4, hi=1.  lo_table[4] has bit for 0x34, hi_table[1] has bit for 0x12.
        // AND = bit_for_0x34 & bit_for_0x12 = different bits → 0.  No false positive.
        assert!(!t.contains(0x14));
        // 0x32: lo=2, hi=3.  lo_table[2] has bit for 0x12, hi_table[3] has bit for 0x34.
        // AND = different bits → 0.
        assert!(!t.contains(0x32));
    }
}
