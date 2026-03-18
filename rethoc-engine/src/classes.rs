//! Byte-class types for character class matching.
//!
//! Three representations are used:
//!
//! - [`ByteClassBits`] — compact 32-byte bit-packed class for custom
//!   character classes stored on the heap.
//! - [`ByteClass`] — legacy 256-byte boolean table used at compile time
//!   for class construction, then converted to [`ByteClassBits`].
//! - [`static_classes`] — predefined `[bool; 256]` tables in `.rodata`
//!   for common classes (`\d`, `\w`, `\s` and negations).
//!
//! The [`ClassIdx`] newtype indexes into `Regex::classes` (a
//! `Box<[ByteClassBits]>`) for custom classes only — wildcard and
//! predefined classes are stored directly on the NFA state.

use std::ops::Index;

use regex_syntax::hir;

/// Compact bit-packed byte-class for custom character classes.
///
/// Stores 256 membership bits in 4 × u64 = 32 bytes (8× smaller than
/// `[bool; 256]`).  Used by [`State::ByteClassCustom`](crate::State::ByteClassCustom).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ByteClassBits(pub(crate) [u64; 4]);

impl ByteClassBits {
    /// Build from a `[bool; 256]` membership array.
    pub(crate) fn from_bools(bools: &[bool; 256]) -> Self {
        let mut bits = [0u64; 4];
        for (i, &b) in bools.iter().enumerate() {
            if b {
                bits[i >> 6] |= 1u64 << (i & 63);
            }
        }
        Self(bits)
    }

    /// Test whether `byte` is a member of this class.
    #[inline(always)]
    pub(crate) fn contains(&self, byte: u8) -> bool {
        self.0[(byte >> 6) as usize] & (1u64 << (byte & 63)) != 0
    }
}

/// Legacy 256-byte boolean class table.  Used at compile time for
/// class construction and predefined-class detection, then converted
/// to [`ByteClassBits`] for storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ByteClass(pub(crate) [bool; 256]);

impl ByteClass {
    /// A class that matches no byte value.
    pub(crate) const NONE: Self = Self([false; 256]);

    /// Convert to compact bit-packed representation.
    pub(crate) fn to_bits(self) -> ByteClassBits {
        ByteClassBits::from_bools(&self.0)
    }

    /// Test whether this class matches every byte (wildcard).
    pub(crate) fn is_all(&self) -> bool {
        self.0.iter().all(|&b| b)
    }

    /// Build from a `regex-syntax` HIR character class.
    ///
    /// Handles both `Class::Bytes` (infallible) and `Class::Unicode`
    /// (fallible — returns `None` if any codepoint exceeds 0xFF).
    ///
    /// `regex-syntax` may produce `Class::Unicode` for ASCII-only
    /// patterns like `(a|b)` or `[ab]`.  If all ranges fit in a
    /// single byte (0x00..=0xFF), they are lowered to a byte table;
    /// otherwise the class is rejected.
    pub(crate) fn from_hir_class(class: &hir::Class) -> Option<Self> {
        match class {
            hir::Class::Bytes(bc) => {
                let mut table = Self::NONE;
                for range in bc.ranges() {
                    for b in range.start()..=range.end() {
                        table.0[b as usize] = true;
                    }
                }
                Some(table)
            }
            hir::Class::Unicode(uc) => {
                let ranges = uc.ranges();
                // Reject multi-byte Unicode classes (codepoints > 0xFF).
                if !ranges
                    .iter()
                    .all(|r| (r.start() as u32) <= 0xFF && (r.end() as u32) <= 0xFF)
                {
                    return None;
                }
                let mut table = Self::NONE;
                for range in ranges {
                    for b in (range.start() as u8)..=(range.end() as u8) {
                        table.0[b as usize] = true;
                    }
                }
                Some(table)
            }
        }
    }
}

/// `class[byte]` — test whether a byte matches this class.
impl Index<u8> for ByteClass {
    type Output = bool;

    #[inline]
    fn index(&self, byte: u8) -> &bool {
        &self.0[byte as usize]
    }
}

// ---------------------------------------------------------------------------
// Predefined static byte-class tables
// ---------------------------------------------------------------------------

/// Static `[bool; 256]` tables for common character classes.
///
/// Used by [`State::ByteClassStatic`](crate::State::ByteClassStatic) — no
/// heap allocation, lives in `.rodata`.
pub(crate) mod static_classes {
    /// `\d` — ASCII digits `[0-9]`.
    pub(crate) static DIGIT: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'0';
        while i <= b'9' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    /// `\D` — non-digit bytes.
    pub(crate) static NOT_DIGIT: [bool; 256] = {
        let mut t = [true; 256];
        let mut i = b'0';
        while i <= b'9' {
            t[i as usize] = false;
            i += 1;
        }
        t
    };

    /// `\w` — ASCII word characters `[a-zA-Z0-9_]`.
    pub(crate) static WORD: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'a';
        while i <= b'z' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'A';
        while i <= b'Z' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'0';
        while i <= b'9' {
            t[i as usize] = true;
            i += 1;
        }
        t[b'_' as usize] = true;
        t
    };

    /// `\W` — non-word bytes.
    pub(crate) static NOT_WORD: [bool; 256] = {
        let mut t = [true; 256];
        let mut i = b'a';
        while i <= b'z' {
            t[i as usize] = false;
            i += 1;
        }
        i = b'A';
        while i <= b'Z' {
            t[i as usize] = false;
            i += 1;
        }
        i = b'0';
        while i <= b'9' {
            t[i as usize] = false;
            i += 1;
        }
        t[b'_' as usize] = false;
        t
    };

    /// `\s` — ASCII whitespace `[\t\n\x0B\x0C\r ]`.
    pub(crate) static SPACE: [bool; 256] = {
        let mut t = [false; 256];
        t[b'\t' as usize] = true;
        t[b'\n' as usize] = true;
        t[0x0B] = true; // vertical tab
        t[0x0C] = true; // form feed
        t[b'\r' as usize] = true;
        t[b' ' as usize] = true;
        t
    };

    /// `\S` — non-whitespace bytes.
    pub(crate) static NOT_SPACE: [bool; 256] = {
        let mut t = [true; 256];
        t[b'\t' as usize] = false;
        t[b'\n' as usize] = false;
        t[0x0B] = false;
        t[0x0C] = false;
        t[b'\r' as usize] = false;
        t[b' ' as usize] = false;
        t
    };

    // ── Hexadecimal ──

    /// `[0-9a-f]` — lowercase hex digits.
    pub(crate) static HEX_LOWER: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'0';
        while i <= b'9' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'a';
        while i <= b'f' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    /// `[0-9A-F]` — uppercase hex digits.
    pub(crate) static HEX_UPPER: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'0';
        while i <= b'9' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'A';
        while i <= b'F' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    /// `[0-9a-fA-F]` — case-insensitive hex digits.
    pub(crate) static HEX: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'0';
        while i <= b'9' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'a';
        while i <= b'f' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'A';
        while i <= b'F' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    // ── Alphabetic ──

    /// `[a-z]` — ASCII lowercase letters.
    pub(crate) static LOWER: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'a';
        while i <= b'z' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    /// `[A-Z]` — ASCII uppercase letters.
    pub(crate) static UPPER: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'A';
        while i <= b'Z' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    /// `[a-zA-Z]` — ASCII letters.
    pub(crate) static ALPHA: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'a';
        while i <= b'z' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'A';
        while i <= b'Z' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    // ── Alphanumeric ──

    /// `[0-9a-z]` — ASCII lowercase alphanumeric.
    pub(crate) static ALNUM_LOWER: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'0';
        while i <= b'9' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'a';
        while i <= b'z' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    /// `[0-9A-Z]` — ASCII uppercase alphanumeric.
    pub(crate) static ALNUM_UPPER: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'0';
        while i <= b'9' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'A';
        while i <= b'Z' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    /// `[0-9a-zA-Z]` — ASCII alphanumeric.
    pub(crate) static ALNUM: [bool; 256] = {
        let mut t = [false; 256];
        let mut i = b'0';
        while i <= b'9' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'a';
        while i <= b'z' {
            t[i as usize] = true;
            i += 1;
        }
        i = b'A';
        while i <= b'Z' {
            t[i as usize] = true;
            i += 1;
        }
        t
    };

    // ── Wildcard ──

    /// Wildcard table — all bytes match.  Used for reference comparison
    /// during class detection, not stored on states (Wildcard has its
    /// own State variant).
    pub(crate) static ALL: [bool; 256] = [true; 256];

    /// All known static tables for detection.
    static KNOWN: &[&[bool; 256]] = &[
        &ALL,
        &DIGIT,
        &NOT_DIGIT,
        &WORD,
        &NOT_WORD,
        &SPACE,
        &NOT_SPACE,
        &HEX_LOWER,
        &HEX_UPPER,
        &HEX,
        &LOWER,
        &UPPER,
        &ALPHA,
        &ALNUM_LOWER,
        &ALNUM_UPPER,
        &ALNUM,
    ];

    /// Try to match a `[bool; 256]` table against a known static class.
    /// Returns a reference to the static table if it matches, or `None`.
    pub(crate) fn detect(table: &[bool; 256]) -> Option<&'static [bool; 256]> {
        KNOWN.iter().copied().find(|s| *s == table)
    }
}

/// Index into the custom byte-class tables ([`Regex::classes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ClassIdx(pub(crate) u16);

impl ClassIdx {
    #[inline]
    pub(crate) fn idx(self) -> usize {
        self.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ByteClassBits: bit manipulation ──

    #[test]
    fn test_bits_empty() {
        let bits = ByteClassBits([0; 4]);
        for b in 0..=255u8 {
            assert!(!bits.contains(b), "empty class should not contain {b}");
        }
    }

    #[test]
    fn test_bits_full() {
        let bits = ByteClassBits([u64::MAX; 4]);
        for b in 0..=255u8 {
            assert!(bits.contains(b), "full class should contain {b}");
        }
    }

    #[test]
    fn test_bits_single_byte_zero() {
        let mut bools = [false; 256];
        bools[0] = true;
        let bits = ByteClassBits::from_bools(&bools);
        assert!(bits.contains(0));
        assert!(!bits.contains(1));
        assert!(!bits.contains(255));
    }

    #[test]
    fn test_bits_single_byte_255() {
        let mut bools = [false; 256];
        bools[255] = true;
        let bits = ByteClassBits::from_bools(&bools);
        assert!(bits.contains(255));
        assert!(!bits.contains(0));
        assert!(!bits.contains(254));
    }

    #[test]
    fn test_bits_word_boundaries() {
        // Test bytes at u64 word boundaries: 0, 63, 64, 127, 128, 191, 192, 255.
        let boundary_bytes = [0u8, 63, 64, 127, 128, 191, 192, 255];
        let mut bools = [false; 256];
        for &b in &boundary_bytes {
            bools[b as usize] = true;
        }
        let bits = ByteClassBits::from_bools(&bools);
        for b in 0..=255u8 {
            let expected = boundary_bytes.contains(&b);
            assert_eq!(
                bits.contains(b),
                expected,
                "byte {b}: expected {expected}, got {}",
                bits.contains(b)
            );
        }
    }

    #[test]
    fn test_bits_each_word_one_bit() {
        // One bit set in each of the 4 u64 words.
        let mut bools = [false; 256];
        bools[10] = true; // word 0
        bools[70] = true; // word 1
        bools[130] = true; // word 2
        bools[200] = true; // word 3
        let bits = ByteClassBits::from_bools(&bools);
        assert!(bits.contains(10));
        assert!(bits.contains(70));
        assert!(bits.contains(130));
        assert!(bits.contains(200));
        assert!(!bits.contains(11));
        assert!(!bits.contains(71));
        assert!(!bits.contains(131));
        assert!(!bits.contains(201));
    }

    #[test]
    fn test_bits_from_bools_roundtrip() {
        // Every even byte is set.
        let mut bools = [false; 256];
        for i in (0..256).step_by(2) {
            bools[i] = true;
        }
        let bits = ByteClassBits::from_bools(&bools);
        for b in 0..=255u8 {
            assert_eq!(bits.contains(b), b % 2 == 0, "byte {b}");
        }
    }

    #[test]
    fn test_bits_contiguous_range() {
        // [a-z]
        let mut bools = [false; 256];
        for b in b'a'..=b'z' {
            bools[b as usize] = true;
        }
        let bits = ByteClassBits::from_bools(&bools);
        for b in 0..=255u8 {
            assert_eq!(bits.contains(b), b >= b'a' && b <= b'z', "byte {b}");
        }
    }

    // ── ByteClass: legacy table ──

    #[test]
    fn test_byteclass_none() {
        let c = ByteClass::NONE;
        for b in 0..=255u8 {
            assert!(!c[b]);
        }
    }

    #[test]
    fn test_byteclass_is_all() {
        assert!(ByteClass([true; 256]).is_all());
        assert!(!ByteClass::NONE.is_all());
        let mut almost = [true; 256];
        almost[42] = false;
        assert!(!ByteClass(almost).is_all());
    }

    #[test]
    fn test_byteclass_to_bits_consistency() {
        // Build a class, convert to bits, verify every byte agrees.
        let mut bools = [false; 256];
        for b in b'A'..=b'Z' {
            bools[b as usize] = true;
        }
        bools[b'_' as usize] = true;
        let class = ByteClass(bools);
        let bits = class.to_bits();
        for b in 0..=255u8 {
            assert_eq!(bits.contains(b), class[b], "mismatch at byte {b}");
        }
    }

    // ── static_classes::detect ──

    #[test]
    fn test_detect_wildcard() {
        let table = [true; 256];
        let detected = static_classes::detect(&table);
        assert!(detected.is_some());
        assert!(std::ptr::eq(detected.unwrap(), &static_classes::ALL));
    }

    #[test]
    fn test_detect_digit() {
        assert!(std::ptr::eq(
            static_classes::detect(&static_classes::DIGIT).unwrap(),
            &static_classes::DIGIT
        ));
    }

    #[test]
    fn test_detect_word() {
        assert!(std::ptr::eq(
            static_classes::detect(&static_classes::WORD).unwrap(),
            &static_classes::WORD
        ));
    }

    #[test]
    fn test_detect_space() {
        assert!(std::ptr::eq(
            static_classes::detect(&static_classes::SPACE).unwrap(),
            &static_classes::SPACE
        ));
    }

    #[test]
    fn test_detect_negations() {
        assert!(static_classes::detect(&static_classes::NOT_DIGIT).is_some());
        assert!(static_classes::detect(&static_classes::NOT_WORD).is_some());
        assert!(static_classes::detect(&static_classes::NOT_SPACE).is_some());
    }

    #[test]
    fn test_detect_hex() {
        assert!(static_classes::detect(&static_classes::HEX_LOWER).is_some());
        assert!(static_classes::detect(&static_classes::HEX_UPPER).is_some());
        assert!(static_classes::detect(&static_classes::HEX).is_some());
    }

    #[test]
    fn test_detect_alpha() {
        assert!(static_classes::detect(&static_classes::LOWER).is_some());
        assert!(static_classes::detect(&static_classes::UPPER).is_some());
        assert!(static_classes::detect(&static_classes::ALPHA).is_some());
    }

    #[test]
    fn test_detect_alnum() {
        assert!(static_classes::detect(&static_classes::ALNUM_LOWER).is_some());
        assert!(static_classes::detect(&static_classes::ALNUM_UPPER).is_some());
        assert!(static_classes::detect(&static_classes::ALNUM).is_some());
    }

    #[test]
    fn test_detect_custom_returns_none() {
        // A class that doesn't match any predefined table.
        let mut table = [false; 256];
        table[b'x' as usize] = true;
        table[b'y' as usize] = true;
        assert!(static_classes::detect(&table).is_none());
    }

    #[test]
    fn test_detect_near_miss() {
        // DIGIT with one extra byte — should NOT match.
        let mut table = static_classes::DIGIT;
        table[b'a' as usize] = true;
        assert!(static_classes::detect(&table).is_none());
    }

    // ── Static table correctness ──

    #[test]
    fn test_static_digit_correctness() {
        for b in 0..=255u8 {
            assert_eq!(
                static_classes::DIGIT[b as usize],
                b.is_ascii_digit(),
                "DIGIT mismatch at {b}"
            );
        }
    }

    #[test]
    fn test_static_word_correctness() {
        for b in 0..=255u8 {
            let expected = b.is_ascii_alphanumeric() || b == b'_';
            assert_eq!(
                static_classes::WORD[b as usize],
                expected,
                "WORD mismatch at {b}"
            );
        }
    }

    #[test]
    fn test_static_space_correctness() {
        for b in 0..=255u8 {
            let expected = matches!(b, b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ');
            assert_eq!(
                static_classes::SPACE[b as usize],
                expected,
                "SPACE mismatch at {b}"
            );
        }
    }

    #[test]
    fn test_static_hex_lower_correctness() {
        for b in 0..=255u8 {
            let expected = b.is_ascii_digit() || (b >= b'a' && b <= b'f');
            assert_eq!(
                static_classes::HEX_LOWER[b as usize],
                expected,
                "HEX_LOWER mismatch at {b}"
            );
        }
    }

    #[test]
    fn test_static_alpha_correctness() {
        for b in 0..=255u8 {
            assert_eq!(
                static_classes::ALPHA[b as usize],
                b.is_ascii_alphabetic(),
                "ALPHA mismatch at {b}"
            );
        }
    }

    #[test]
    fn test_static_alnum_correctness() {
        for b in 0..=255u8 {
            assert_eq!(
                static_classes::ALNUM[b as usize],
                b.is_ascii_alphanumeric(),
                "ALNUM mismatch at {b}"
            );
        }
    }

    #[test]
    fn test_static_negation_complement() {
        // Every negation table should be the exact complement.
        for b in 0..=255u8 {
            let i = b as usize;
            assert_eq!(static_classes::NOT_DIGIT[i], !static_classes::DIGIT[i]);
            assert_eq!(static_classes::NOT_WORD[i], !static_classes::WORD[i]);
            assert_eq!(static_classes::NOT_SPACE[i], !static_classes::SPACE[i]);
        }
    }

    // ── ClassIdx ──

    #[test]
    fn test_class_idx_roundtrip() {
        let idx = ClassIdx(42);
        assert_eq!(idx.idx(), 42);
        let idx = ClassIdx(0);
        assert_eq!(idx.idx(), 0);
        let idx = ClassIdx(u16::MAX);
        assert_eq!(idx.idx(), u16::MAX as usize);
    }
}
