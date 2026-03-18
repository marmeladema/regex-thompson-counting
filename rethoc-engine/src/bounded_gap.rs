//! Bounded-gap analysis helpers for detecting `Anchor (Gap Anchor)* Gap?`
//! patterns in `regex-syntax` HIR.
//!
//! These helpers operate on HIR before NFA construction.  They detect
//! whether a pattern can be compiled as a bounded-gap chain and extract
//! the structural parameters needed for plan construction.
//!
//! Phase 0 of the bounded-gap engine: analysis scaffolding only, no
//! runtime types or matching logic.

use regex_syntax::hir::{Hir, HirKind};

use crate::classes::{ByteClass, ByteClassBits};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Classification of a gap body predicate.
///
/// Version 1 gap predicates are always single-byte.  `Any` is kept
/// separate to enable a meaningful runtime fast path where bad-byte
/// tracking is skipped entirely for `.{0,K}` / `[\s\S]{0,K}` style gaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GapBodyClass {
    /// Matches any single byte (wildcard).
    Any,
    /// Matches a specific set of bytes, stored as a 256-bit inline set.
    /// Negation is already normalized at compile time.
    ByteClass(ByteClassBits),
}

/// Fixed-length information for an anchor fragment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedLengthInfo {
    /// The exact byte length consumed by this anchor.
    pub(crate) len: u16,
}

// ---------------------------------------------------------------------------
// HIR Flattening
// ---------------------------------------------------------------------------

/// Strip non-semantic HIR wrappers (captures, singleton concats).
///
/// Peels away `Capture`, single-child `Concat`, and single-child
/// `Alternation` to expose the structurally relevant node underneath.
fn strip_wrappers(hir: &Hir) -> &Hir {
    match hir.kind() {
        HirKind::Capture(cap) => strip_wrappers(&cap.sub),
        HirKind::Concat(children) if children.len() == 1 => strip_wrappers(&children[0]),
        HirKind::Alternation(children) if children.len() == 1 => strip_wrappers(&children[0]),
        _ => hir,
    }
}

/// Flatten a top-level concat HIR into a linear sequence of pieces.
///
/// Non-semantic wrappers (captures, singleton concats) are stripped
/// from the outermost level.  The result is a flat view suitable for
/// partitioning into alternating anchor/gap segments.
///
/// If the outermost node (after stripping) is not a `Concat`, a
/// single-element vector is returned.
pub(crate) fn flatten_top_level_concat(hir: &Hir) -> Vec<&Hir> {
    let stripped = strip_wrappers(hir);
    match stripped.kind() {
        HirKind::Concat(children) => children.iter().collect(),
        _ => vec![stripped],
    }
}

// ---------------------------------------------------------------------------
// Gap Body Recognition
// ---------------------------------------------------------------------------

/// Recognize whether an HIR fragment is a Version 1 gap body: a single
/// byte-consuming predicate atom.
///
/// Returns `Some(GapBodyClass)` if the body qualifies, `None` otherwise.
///
/// Qualifying bodies:
/// - `.` (wildcard)
/// - `[\s\S]`, `[^]` (all-bytes class)
/// - `[^/]`, `\s`, `[A-Za-z0-9_]` (specific byte class)
/// - Single-byte literal like `a`
///
/// Non-qualifying bodies:
/// - Multi-byte literals (`ab`)
/// - Bodies containing assertions (`\b`)
/// - Alternations, repetitions, or other compound structures
pub(crate) fn classify_gap_body(hir: &Hir) -> Option<GapBodyClass> {
    match hir.kind() {
        // Single-byte literal → byte class with one byte set.
        HirKind::Literal(lit) if lit.0.len() == 1 => {
            let mut table = ByteClass::NONE;
            table.0[lit.0[0] as usize] = true;
            Some(GapBodyClass::ByteClass(table.to_bits()))
        }
        // Byte or Unicode class → check for wildcard or specific class.
        HirKind::Class(class) => {
            let table = ByteClass::from_hir_class(class)?;
            if table.is_all() {
                Some(GapBodyClass::Any)
            } else {
                Some(GapBodyClass::ByteClass(table.to_bits()))
            }
        }
        // Capture is just a wrapper — recurse.
        HirKind::Capture(cap) => classify_gap_body(&cap.sub),
        // Anything else is not a valid gap body.
        _ => None,
    }
}

/// Recognize whether a top-level HIR piece is a Version 1 gap: a
/// bounded repetition whose body is a single-byte predicate.
///
/// Returns `Some((min, max, class))` if the piece is a valid gap,
/// `None` otherwise.
///
/// Requires:
/// - `Repetition` with finite `max`
/// - Body qualifies under [`classify_gap_body`]
pub(crate) fn classify_gap(hir: &Hir) -> Option<(u32, u32, GapBodyClass)> {
    let hir = strip_wrappers(hir);
    match hir.kind() {
        HirKind::Repetition(rep) => {
            let max = rep.max?; // None = unbounded → not a gap
            let body_class = classify_gap_body(&rep.sub)?;
            Some((rep.min, max, body_class))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Fixed-Length Detection
// ---------------------------------------------------------------------------

/// Compute whether an HIR fragment has exact fixed byte length.
///
/// Uses the `minimum_len` / `maximum_len` properties computed by
/// `regex-syntax`.  Returns `None` if the fragment has variable length,
/// is unbounded, matches nothing, or has length 0 (empty).
///
/// Note: assertions (`Look` nodes) have zero length and do not affect
/// the byte count.  A fragment like `\bfoo` has fixed length 3 (from
/// the 3-byte literal) — the assertion-free check is separate.
pub(crate) fn compute_fixed_length(hir: &Hir) -> Option<FixedLengthInfo> {
    let props = hir.properties();
    let min = props.minimum_len()?; // None if matches nothing
    let max = props.maximum_len()?; // None if unbounded
    if min == max && min > 0 {
        let len = u16::try_from(min).ok()?; // reject if > 65535
        Some(FixedLengthInfo { len })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Assertion-Free Check
// ---------------------------------------------------------------------------

/// Check whether an HIR fragment is free of all assertions (`Look` nodes).
///
/// Version 1 anchors must be assertion-free.  Top-level `^` / `$` are
/// handled separately by hoisting into plan flags before individual
/// anchor fragments are checked.
pub(crate) fn is_assertion_free(hir: &Hir) -> bool {
    hir.properties().look_set().is_empty()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_hir;

    /// Helper: parse a pattern to HIR using the same settings as the engine.
    fn hir(pattern: &str) -> Hir {
        parse_hir(pattern).expect("test pattern should parse")
    }

    // -- flatten_top_level_concat -------------------------------------------

    #[test]
    fn test_flatten_simple_concat() {
        // foo.{0,10}bar → 3 pieces: Literal("foo"), Repetition, Literal("bar")
        let h = hir(r"foo.{0,10}bar");
        let pieces = flatten_top_level_concat(&h);
        assert_eq!(pieces.len(), 3, "expected 3 concat pieces");
    }

    #[test]
    fn test_flatten_wrapped_in_capture() {
        // (foo.{0,10}bar) → same 3 pieces after stripping capture
        let h = hir(r"(foo.{0,10}bar)");
        let pieces = flatten_top_level_concat(&h);
        assert_eq!(pieces.len(), 3, "capture wrapper should be stripped");
    }

    #[test]
    fn test_flatten_single_literal() {
        // "foo" → 1 piece (the whole literal, or 3 single-byte literals)
        let h = hir(r"foo");
        let pieces = flatten_top_level_concat(&h);
        // After HIR optimization, "foo" is a single Literal node with 3 bytes.
        assert_eq!(pieces.len(), 1, "single literal should be one piece");
    }

    #[test]
    fn test_flatten_alternation_at_top() {
        // "foo|bar" → 1 piece (the alternation itself)
        let h = hir(r"foo|bar");
        let pieces = flatten_top_level_concat(&h);
        assert_eq!(pieces.len(), 1, "alternation should be one piece");
    }

    #[test]
    fn test_flatten_multi_gap_chain() {
        // foo.{0,10}bar.{0,20}baz → 5 pieces
        let h = hir(r"foo.{0,10}bar.{0,20}baz");
        let pieces = flatten_top_level_concat(&h);
        assert_eq!(
            pieces.len(),
            5,
            "expected 5 concat pieces for 3-anchor chain"
        );
    }

    // -- classify_gap_body --------------------------------------------------

    #[test]
    fn test_gap_body_wildcard() {
        // "." with dot_matches_new_line(true) → Any
        let h = hir(r".");
        assert_eq!(classify_gap_body(&h), Some(GapBodyClass::Any));
    }

    #[test]
    fn test_gap_body_all_bytes_class() {
        // [\s\S] should match all bytes → Any
        let h = hir(r"[\s\S]");
        assert_eq!(classify_gap_body(&h), Some(GapBodyClass::Any));
    }

    #[test]
    fn test_gap_body_negated_slash() {
        // [^/] → ByteClass with '/' excluded
        let h = hir(r"[^/]");
        match classify_gap_body(&h) {
            Some(GapBodyClass::ByteClass(bits)) => {
                assert!(!bits.contains(b'/'), "'/' should not be in [^/] class");
                assert!(bits.contains(b'a'), "'a' should be in [^/] class");
            }
            other => panic!("expected ByteClass, got {:?}", other),
        }
    }

    #[test]
    fn test_gap_body_whitespace() {
        // \s → ByteClass with whitespace bytes
        let h = hir(r"\s");
        match classify_gap_body(&h) {
            Some(GapBodyClass::ByteClass(bits)) => {
                assert!(bits.contains(b' '), "space should be in \\s class");
                assert!(bits.contains(b'\t'), "tab should be in \\s class");
                assert!(!bits.contains(b'a'), "'a' should not be in \\s class");
            }
            other => panic!("expected ByteClass, got {:?}", other),
        }
    }

    #[test]
    fn test_gap_body_single_byte_literal() {
        // "a" → ByteClass with just 'a'
        let h = hir(r"a");
        match classify_gap_body(&h) {
            Some(GapBodyClass::ByteClass(bits)) => {
                assert!(bits.contains(b'a'), "'a' should be set");
                assert!(!bits.contains(b'b'), "'b' should not be set");
            }
            other => panic!("expected ByteClass, got {:?}", other),
        }
    }

    #[test]
    fn test_gap_body_multi_byte_literal_rejected() {
        // "ab" → None (not a single atom)
        let h = hir(r"ab");
        assert_eq!(classify_gap_body(&h), None);
    }

    #[test]
    fn test_gap_body_assertion_rejected() {
        // \b → None
        let h = hir(r"\b");
        assert_eq!(classify_gap_body(&h), None);
    }

    #[test]
    fn test_gap_body_simple_alternation_lowered_to_class() {
        // a|b → regex-syntax normalizes to [ab] → ByteClass
        let h = hir(r"a|b");
        match classify_gap_body(&h) {
            Some(GapBodyClass::ByteClass(bits)) => {
                assert!(bits.contains(b'a'));
                assert!(bits.contains(b'b'));
                assert!(!bits.contains(b'c'));
            }
            other => panic!("expected ByteClass for a|b, got {:?}", other),
        }
    }

    #[test]
    fn test_gap_body_multi_byte_alternation_rejected() {
        // (ab|cd) → alternation of multi-byte literals, not a single atom
        let h = hir(r"ab|cd");
        assert_eq!(classify_gap_body(&h), None);
    }

    #[test]
    fn test_gap_body_word_class() {
        // \w → ByteClass
        let h = hir(r"\w");
        match classify_gap_body(&h) {
            Some(GapBodyClass::ByteClass(bits)) => {
                assert!(bits.contains(b'a'));
                assert!(bits.contains(b'Z'));
                assert!(bits.contains(b'0'));
                assert!(bits.contains(b'_'));
                assert!(!bits.contains(b' '));
            }
            other => panic!("expected ByteClass, got {:?}", other),
        }
    }

    // -- classify_gap -------------------------------------------------------

    #[test]
    fn test_classify_gap_bounded_wildcard() {
        // .{0,100} → (0, 100, Any)
        let h = hir(r".{0,100}");
        assert_eq!(classify_gap(&h), Some((0, 100, GapBodyClass::Any)));
    }

    #[test]
    fn test_classify_gap_exact_count() {
        // .{254} → (254, 254, Any)
        let h = hir(r".{254}");
        assert_eq!(classify_gap(&h), Some((254, 254, GapBodyClass::Any)));
    }

    #[test]
    fn test_classify_gap_constrained() {
        // [^/]{0,20} → (0, 20, ByteClass(...))
        let h = hir(r"[^/]{0,20}");
        match classify_gap(&h) {
            Some((0, 20, GapBodyClass::ByteClass(bits))) => {
                assert!(!bits.contains(b'/'));
                assert!(bits.contains(b'a'));
            }
            other => panic!("expected constrained gap, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_gap_unbounded_rejected() {
        // .* → None (unbounded)
        let h = hir(r".*");
        assert_eq!(classify_gap(&h), None);
    }

    #[test]
    fn test_classify_gap_multi_byte_body_rejected() {
        // (ab){0,10} → None (not a single-byte body)
        let h = hir(r"(ab){0,10}");
        assert_eq!(classify_gap(&h), None);
    }

    #[test]
    fn test_classify_gap_zero_length_optimized_away() {
        // .{0,0} → the HIR optimizer collapses this to Empty, so
        // classify_gap correctly returns None.  The {0,0} normalization
        // step in the detection pipeline handles this at the chain level.
        let h = hir(r".{0,0}");
        assert_eq!(classify_gap(&h), None);
    }

    // -- compute_fixed_length -----------------------------------------------

    #[test]
    fn test_fixed_length_literal() {
        let h = hir(r"foo");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 3 }));
    }

    #[test]
    fn test_fixed_length_alternation_same_lengths() {
        // (abc|def) → fixed length 3
        let h = hir(r"abc|def");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 3 }));
    }

    #[test]
    fn test_fixed_length_alternation_different_lengths() {
        // (abc|de) → None
        let h = hir(r"abc|de");
        assert_eq!(compute_fixed_length(&h), None);
    }

    #[test]
    fn test_fixed_length_exact_repetition() {
        // a{3} → fixed length 3
        let h = hir(r"a{3}");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 3 }));
    }

    #[test]
    fn test_fixed_length_variable_repetition() {
        // a{2,4} → None
        let h = hir(r"a{2,4}");
        assert_eq!(compute_fixed_length(&h), None);
    }

    #[test]
    fn test_fixed_length_unbounded() {
        // a+ → None
        let h = hir(r"a+");
        assert_eq!(compute_fixed_length(&h), None);
    }

    #[test]
    fn test_fixed_length_empty() {
        // Empty pattern → None (length 0 is rejected as non-empty is required)
        let h = hir(r"");
        assert_eq!(compute_fixed_length(&h), None);
    }

    #[test]
    fn test_fixed_length_complex_anchor() {
        // foo\d{4} → fixed length 7
        let h = hir(r"foo\d{4}");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 7 }));
    }

    #[test]
    fn test_fixed_length_case_insensitive() {
        // (?i)file → fixed length 4
        let h = hir(r"(?i)file");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 4 }));
    }

    #[test]
    fn test_fixed_length_class_repetition() {
        // [A-Z]{2}admin → fixed length 7
        let h = hir(r"[A-Z]{2}admin");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 7 }));
    }

    #[test]
    fn test_fixed_length_with_assertion() {
        // \bfoo → fixed length 3 (assertion is zero-width)
        // Note: this has fixed length but is NOT assertion-free.
        let h = hir(r"\bfoo");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 3 }));
    }

    // -- is_assertion_free --------------------------------------------------

    #[test]
    fn test_assertion_free_literal() {
        let h = hir(r"foo");
        assert!(is_assertion_free(&h));
    }

    #[test]
    fn test_assertion_free_alternation() {
        let h = hir(r"GET|POST");
        assert!(is_assertion_free(&h));
    }

    #[test]
    fn test_assertion_free_class_repetition() {
        let h = hir(r"foo\d{4}");
        assert!(is_assertion_free(&h));
    }

    #[test]
    fn test_assertion_free_case_insensitive() {
        let h = hir(r"(?i)file");
        assert!(is_assertion_free(&h));
    }

    #[test]
    fn test_not_assertion_free_word_boundary() {
        let h = hir(r"\bfoo");
        assert!(!is_assertion_free(&h));
    }

    #[test]
    fn test_not_assertion_free_start_anchor() {
        let h = hir(r"^foo");
        assert!(!is_assertion_free(&h));
    }

    #[test]
    fn test_not_assertion_free_end_anchor() {
        let h = hir(r"foo$");
        assert!(!is_assertion_free(&h));
    }

    #[test]
    fn test_not_assertion_free_word_boundary_negated() {
        let h = hir(r"foo\B");
        assert!(!is_assertion_free(&h));
    }
}
