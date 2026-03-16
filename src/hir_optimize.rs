//! HIR normalization pass — simplifies redundant constructs before
//! NFA compilation.
//!
//! Applied in [`parse_hir`](crate::parse_hir) so that all code paths
//! (estimation, compilation, CLI) see the optimized tree.
//!
//! # Transformations
//!
//! 1. **Strip captures** — `(X)` → `X`.  This engine does not track
//!    capture groups, so the wrapper is pure overhead and hides
//!    structure from later passes.
//!
//! 2. **Collapse nested quantifiers** — redundant nesting of `?`, `*`,
//!    `+` is flattened to the minimal equivalent:
//!
//!    | Outer | Inner | Result |
//!    |-------|-------|--------|
//!    | `?`   | `?`   | `?`    |
//!    | `?`   | `+`   | `*`    |
//!    | `?`   | `*`   | `*`    |
//!    | `+`   | `?`   | `*`    |
//!    | `+`   | `+`   | `+`    |
//!    | `+`   | `*`   | `*`    |
//!    | `*`   | `?`   | `*`    |
//!    | `*`   | `+`   | `*`    |
//!    | `*`   | `*`   | `*`    |
//!
//! 3. **Deduplicate alternation branches** — removes duplicate
//!    sub-expressions and collapses multiple empty branches to one.
//!
//! The pass is bottom-up: children are optimized before their parents,
//! so nested redundancies are caught in a single traversal.  The
//! `regex_syntax` smart constructors (`Hir::concat`, `Hir::alternation`,
//! `Hir::repetition`) provide additional normalization for free
//! (flattening, adjacent-literal merging, `{0,0}` → empty, etc.).

use std::mem;

use regex_syntax::hir::{Hir, HirKind, Repetition};

/// Optimize an HIR tree by stripping captures, collapsing nested
/// quantifiers, and deduplicating alternation branches.
pub(crate) fn optimize(hir: Hir) -> Hir {
    match hir.into_kind() {
        // Leaf nodes — pass through unchanged.
        HirKind::Empty => Hir::empty(),
        HirKind::Literal(lit) => Hir::literal(lit.0),
        HirKind::Class(cls) => Hir::class(cls),
        HirKind::Look(look) => Hir::look(look),

        // Strip captures — our engine ignores capture groups.
        HirKind::Capture(cap) => optimize(*cap.sub),

        // Concatenation — optimize children in place, let smart
        // constructor flatten/merge.
        HirKind::Concat(mut subs) => {
            optimize_children(&mut subs);
            Hir::concat(subs)
        }

        // Alternation — optimize children in place, deduplicate,
        // reconstruct.
        HirKind::Alternation(mut subs) => {
            optimize_children(&mut subs);
            dedup_branches(&mut subs);
            Hir::alternation(subs)
        }

        // Repetition — optimize sub, then try to collapse nested
        // quantifiers.
        HirKind::Repetition(rep) => {
            let sub = optimize(*rep.sub);
            collapse_repetition(rep.min, rep.max, rep.greedy, sub)
        }
    }
}

/// Optimize each child in `subs` in place, reusing the Vec allocation.
fn optimize_children(subs: &mut [Hir]) {
    for sub in subs.iter_mut() {
        let owned = mem::replace(sub, Hir::empty());
        *sub = optimize(owned);
    }
}

/// Build a `Hir::repetition` from parts, attempting to collapse
/// nested quantifiers first.
///
/// If `sub` is itself a repetition and both inner and outer are
/// unbounded quantifiers (`?`, `*`, `+`), collapse them:
///
/// - Both `min ≥ 1` → `+`  (at least one match required on both levels)
/// - Otherwise → `*`        (zero matches possible on at least one level)
fn collapse_repetition(outer_min: u32, outer_max: Option<u32>, greedy: bool, sub: Hir) -> Hir {
    // Only collapse unbounded quantifiers: ?(0,1), +(1,∞), *(0,∞).
    if !is_simple_quantifier(outer_min, outer_max) {
        return Hir::repetition(Repetition {
            min: outer_min,
            max: outer_max,
            greedy,
            sub: Box::new(sub),
        });
    }

    // Check if sub is also a simple quantifier.
    if let HirKind::Repetition(inner_rep) = sub.kind()
        && is_simple_quantifier(inner_rep.min, inner_rep.max)
    {
        // Both are simple quantifiers — collapse.
        //
        // Combined semantics:
        // - min: 1 only if BOTH require at least one (both +).
        // - max: unbounded unless BOTH are ? (max=1).
        let combined_min = if outer_min >= 1 && inner_rep.min >= 1 {
            1
        } else {
            0
        };
        let combined_max = match (outer_max, inner_rep.max) {
            (Some(1), Some(1)) => Some(1), // ?+? → ?
            _ => None,                     // any ∞ side → ∞
        };

        // Extract the innermost sub.
        let inner_sub = inner_rep.sub.clone();
        return Hir::repetition(Repetition {
            min: combined_min,
            max: combined_max,
            greedy,
            sub: inner_sub,
        });
    }

    // No collapse possible — just wrap.
    Hir::repetition(Repetition {
        min: outer_min,
        max: outer_max,
        greedy,
        sub: Box::new(sub),
    })
}

/// Returns `true` if `(min, max)` represents one of the three simple
/// unbounded quantifiers: `?` (0,1), `+` (1,∞), `*` (0,∞).
fn is_simple_quantifier(min: u32, max: Option<u32>) -> bool {
    matches!((min, max), (0, Some(1)) | (1, None) | (0, None))
}

/// Remove duplicate branches from an alternation in place.
///
/// Uses `Hir`'s `PartialEq` to detect equal sub-trees.  Preserves
/// the first occurrence of each unique branch and the relative order.
fn dedup_branches(subs: &mut Vec<Hir>) {
    let mut i = 0;
    while i < subs.len() {
        if subs[..i].iter().any(|s| s == &subs[i]) {
            subs.remove(i);
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: parse a pattern to HIR (without optimization), then
    /// optimize and return the result.
    fn opt(pattern: &str) -> Hir {
        use regex_syntax::ast::parse::ParserBuilder;
        use regex_syntax::hir::translate::TranslatorBuilder;
        let ast = ParserBuilder::new().build().parse(pattern).unwrap();
        let hir = TranslatorBuilder::new()
            .unicode(false)
            .utf8(false)
            .dot_matches_new_line(true)
            .build()
            .translate(pattern, &ast)
            .unwrap();
        optimize(hir)
    }

    /// Helper: parse a pattern to HIR without optimization (raw).
    fn raw(pattern: &str) -> Hir {
        use regex_syntax::ast::parse::ParserBuilder;
        use regex_syntax::hir::translate::TranslatorBuilder;
        let ast = ParserBuilder::new().build().parse(pattern).unwrap();
        TranslatorBuilder::new()
            .unicode(false)
            .utf8(false)
            .dot_matches_new_line(true)
            .build()
            .translate(pattern, &ast)
            .unwrap()
    }

    /// Assert that optimizing `input` produces the same HIR as the
    /// raw parse of `expected`.
    fn assert_opt(input: &str, expected: &str) {
        let optimized = opt(input);
        let expected_hir = raw(expected);
        assert_eq!(
            optimized, expected_hir,
            "\noptimize({input:?}) produced:\n  {optimized:?}\n\
             expected (from {expected:?}):\n  {expected_hir:?}"
        );
    }

    // ── Capture stripping ──

    #[test]
    fn test_strip_capture_simple() {
        assert_opt("(a)", "a");
    }

    #[test]
    fn test_strip_capture_nested() {
        assert_opt("((a))", "a");
    }

    #[test]
    fn test_strip_capture_in_concat() {
        assert_opt("(a)(b)", "ab");
    }

    #[test]
    fn test_strip_capture_preserves_quantifier() {
        assert_opt("(a){3}", "a{3}");
    }

    // ── Nested quantifier collapsing ──

    #[test]
    fn test_collapse_optional_optional() {
        // (a?)? → a?
        assert_opt("(a?)?", "a?");
    }

    #[test]
    fn test_collapse_plus_plus() {
        // (a+)+ → a+
        assert_opt("(a+)+", "a+");
    }

    #[test]
    fn test_collapse_star_star() {
        // (a*)* → a*
        assert_opt("(a*)*", "a*");
    }

    #[test]
    fn test_collapse_plus_star() {
        // (a+)* → a*
        assert_opt("(a+)*", "a*");
    }

    #[test]
    fn test_collapse_star_plus() {
        // (a*)+ → a*
        assert_opt("(a*)+", "a*");
    }

    #[test]
    fn test_collapse_optional_plus() {
        // (a?)+ → a*
        assert_opt("(a?)+", "a*");
    }

    #[test]
    fn test_collapse_optional_star() {
        // (a?)* → a*
        assert_opt("(a?)*", "a*");
    }

    #[test]
    fn test_collapse_plus_optional() {
        // (a+)? → a*
        assert_opt("(a+)?", "a*");
    }

    #[test]
    fn test_collapse_star_optional() {
        // (a*)? → a*
        assert_opt("(a*)?", "a*");
    }

    // ── Triple nesting ──

    #[test]
    fn test_collapse_triple_optional() {
        // ((a?)?)? → a?  (two rounds of collapsing)
        assert_opt("((a?)?)?", "a?");
    }

    #[test]
    fn test_collapse_triple_plus() {
        assert_opt("((a+)+)+", "a+");
    }

    #[test]
    fn test_collapse_triple_star() {
        assert_opt("((a*)*)*", "a*");
    }

    // ── Non-simple quantifiers are NOT collapsed ──

    #[test]
    fn test_no_collapse_bounded() {
        // (a{2,4})+ — inner is bounded, not a simple quantifier
        // Should NOT collapse.
        let optimized = opt("(a{2,4})+");
        // The outer + wraps a{2,4}.  Verify it's still a Repetition
        // wrapping a Repetition.
        match optimized.kind() {
            HirKind::Repetition(outer) => {
                assert_eq!(outer.min, 1);
                assert_eq!(outer.max, None);
                match outer.sub.kind() {
                    HirKind::Repetition(inner) => {
                        assert_eq!(inner.min, 2);
                        assert_eq!(inner.max, Some(4));
                    }
                    other => panic!("expected inner Repetition, got {other:?}"),
                }
            }
            other => panic!("expected outer Repetition, got {other:?}"),
        }
    }

    // ── Alternation dedup ──

    #[test]
    fn test_dedup_alternation_equal_branches() {
        // ab|ab|cd → ab|cd
        assert_opt("ab|ab|cd", "ab|cd");
    }

    #[test]
    fn test_dedup_alternation_empty_branches() {
        // (||||a) has 5 empty branches + a.  Dedup to (?:|a).
        // Use non-capturing group in expected since captures are stripped.
        assert_opt("(||||a)", "(?:|a)");
    }

    #[test]
    fn test_dedup_alternation_all_same() {
        // a|a|a → a
        assert_opt("a|a|a", "a");
    }

    // ── Combined: capture strip + quantifier collapse ──

    #[test]
    fn test_capture_then_collapse() {
        // ((a+))+ → a+  (strip capture exposes inner +, then collapse)
        assert_opt("((a+))+", "a+");
    }

    #[test]
    fn test_deep_capture_and_collapse() {
        // (((a?))?)? → a?
        assert_opt("(((a?)?)?)", "a?");
    }
}
