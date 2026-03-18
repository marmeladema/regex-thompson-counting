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

/// Optimize an HIR tree in a single pass: strip captures, collapse
/// nested quantifiers, deduplicate alternation branches, merge adjacent
/// same-body bounded repetitions, and strip semantically irrelevant
/// leading/trailing `Repetition { min: 0 }` from the top-level
/// unanchored concatenation.
///
/// Gated by [`RegexConfig::optimize_hir`](crate::RegexConfig::optimize_hir).
/// When the flag is false, `build()` uses the raw HIR from `parse_hir()`
/// directly — `hir2postfix` handles `Capture` nodes by recursing.
pub(crate) fn optimize(hir: Hir) -> Hir {
    let result = optimize_inner(hir);
    // Strip irrelevant boundary gaps at the outermost level only.
    // This is NOT safe inside nested concats (the surrounding
    // context — groups, alternations, anchors — may make the gap
    // semantically relevant).
    if !matches!(result.kind(), HirKind::Concat(_)) {
        return result;
    }
    match result.into_kind() {
        HirKind::Concat(mut subs) => {
            strip_leading_gaps(&mut subs);
            strip_trailing_gaps(&mut subs);
            Hir::concat(subs)
        }
        _ => unreachable!(),
    }
}

/// Recursive optimization pass (everything except top-level gap strip).
fn optimize_inner(hir: Hir) -> Hir {
    match hir.into_kind() {
        HirKind::Empty => Hir::empty(),
        HirKind::Literal(lit) => Hir::literal(lit.0),
        HirKind::Class(cls) => Hir::class(cls),
        HirKind::Look(look) => Hir::look(look),

        // Strip captures — our engine ignores capture groups.
        HirKind::Capture(cap) => optimize_inner(*cap.sub),

        // Concatenation — optimize children, merge adjacent repetitions.
        HirKind::Concat(mut subs) => {
            optimize_children(&mut subs);
            merge_adjacent_repetitions(&mut subs);
            Hir::concat(subs)
        }

        // Alternation — optimize children, deduplicate.
        HirKind::Alternation(mut subs) => {
            optimize_children(&mut subs);
            dedup_branches(&mut subs);
            Hir::alternation(subs)
        }

        // Repetition — optimize sub, collapse nested quantifiers.
        HirKind::Repetition(rep) => {
            let sub = optimize_inner(*rep.sub);
            collapse_repetition(rep.min, rep.max, rep.greedy, sub)
        }
    }
}

/// Optimize each child in `subs` in place, reusing the Vec allocation.
fn optimize_children(subs: &mut [Hir]) {
    for sub in subs.iter_mut() {
        let owned = mem::replace(sub, Hir::empty());
        *sub = optimize_inner(owned);
    }
}

/// Merge adjacent bounded repetitions with identical bodies in a
/// `Concat` child list.
///
/// E.g. `.{0,1000}.{0,1000}.{0,1000}` → `.{0,3000}`.
///
/// Only merges genuine bounded ranges (finite `max`, `max > 1`,
/// `min < max`) — not `?` `(0,1)`, `*` `(0,∞)`, or `+` `(1,∞)`.
/// Replaces consumed siblings with `Hir::empty()` so the caller can
/// rely on the `Hir::concat` smart constructor to strip them.
fn merge_adjacent_repetitions(subs: &mut [Hir]) {
    let mut i = 0;
    while i < subs.len() {
        let dominated = if let HirKind::Repetition(rep) = subs[i].kind()
            && let Some(rep_max) = rep.max
            && rep_max > 1
            && rep.min < rep_max
        {
            // Scan forward for mergeable siblings.
            let mut merged_min = rep.min;
            let mut merged_max = rep_max;
            let body = rep.sub.clone();
            let greedy = rep.greedy;
            let mut j = i + 1;
            while j < subs.len() {
                if let HirKind::Repetition(rep2) = subs[j].kind()
                    && let Some(rep2_max) = rep2.max
                    && rep2_max > 1
                    && rep2.min < rep2_max
                    && *rep2.sub == *body
                {
                    merged_min = merged_min.saturating_add(rep2.min);
                    merged_max = merged_max.saturating_add(rep2_max);
                    j += 1;
                } else {
                    break;
                }
            }
            if j > i + 1 {
                // Replace the first repetition with the merged one.
                subs[i] = Hir::repetition(Repetition {
                    min: merged_min,
                    max: Some(merged_max),
                    greedy,
                    sub: body,
                });
                // Blank out the consumed siblings.
                for s in &mut subs[i + 1..j] {
                    *s = Hir::empty();
                }
                Some(j)
            } else {
                None
            }
        } else {
            None
        };
        i = dominated.unwrap_or(i + 1);
    }
}

/// Strip semantically irrelevant leading gaps from a `Concat`.
///
/// In an unanchored existence-only engine, a leading `Repetition { min: 0 }`
/// is irrelevant: the engine tries every position, so allowing 0–K bytes
/// of anything before the real pattern adds no constraint.
///
/// Stops when the first child is not a droppable repetition (including
/// when it's a `Look::Start` anchor, which makes the gap meaningful).
fn strip_leading_gaps(subs: &mut Vec<Hir>) {
    while let Some(first) = subs.first() {
        if let HirKind::Repetition(rep) = first.kind()
            && rep.min == 0
        {
            subs.remove(0);
            continue;
        }
        break;
    }
}

/// Strip semantically irrelevant trailing gaps from a `Concat`.
///
/// Same reasoning as [`strip_leading_gaps`]: a trailing
/// `Repetition { min: 0 }` after the real pattern adds no constraint
/// for existence matching, unless followed by a `Look::End` anchor.
fn strip_trailing_gaps(subs: &mut Vec<Hir>) {
    while let Some(last) = subs.last() {
        if let HirKind::Repetition(rep) = last.kind()
            && rep.min == 0
        {
            subs.pop();
            continue;
        }
        break;
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

    // ── Leading gap stripping ──

    #[test]
    fn test_strip_leading_wildcard_gap() {
        // .{0,10}foo → foo (leading wildcard gap irrelevant unanchored)
        assert_opt(".{0,10}foo", "foo");
    }

    #[test]
    fn test_strip_leading_class_gap() {
        // [^/]{0,20}foo → foo
        assert_opt("[^/]{0,20}foo", "foo");
    }

    #[test]
    fn test_strip_leading_star() {
        // a*foo → foo (a* = a{0,∞}, min=0)
        assert_opt("a*foo", "foo");
    }

    #[test]
    fn test_strip_leading_optional() {
        // a?foo → foo (a? = a{0,1}, min=0)
        assert_opt("a?foo", "foo");
    }

    #[test]
    fn test_strip_leading_multi_byte_body() {
        // (abc){0,5}foo → foo (multi-byte body, still min=0)
        assert_opt("(abc){0,5}foo", "foo");
    }

    #[test]
    fn test_strip_leading_chained() {
        // .{0,5}.{0,10}foo → merged to .{0,15}foo → stripped → foo
        assert_opt(".{0,5}.{0,10}foo", "foo");
    }

    #[test]
    fn test_no_strip_leading_min_nonzero() {
        // .{3,10}foo → NOT stripped (min=3 is meaningful)
        assert_opt(".{3,10}foo", ".{3,10}foo");
    }

    #[test]
    fn test_no_strip_leading_plus() {
        // a+foo → NOT stripped (a+ = a{1,∞}, min=1)
        assert_opt("a+foo", "a+foo");
    }

    #[test]
    fn test_no_strip_leading_anchored_start() {
        // ^.{0,10}foo → NOT stripped (^ makes gap meaningful)
        assert_opt("^.{0,10}foo", "^.{0,10}foo");
    }

    #[test]
    fn test_no_strip_leading_anchored_startlf() {
        // (?m:^).{0,10}foo → NOT stripped
        assert_opt("(?m:^).{0,10}foo", "(?m:^).{0,10}foo");
    }

    // ── Trailing gap stripping ──

    #[test]
    fn test_strip_trailing_wildcard_gap() {
        // foo.{0,10} → foo
        assert_opt("foo.{0,10}", "foo");
    }

    #[test]
    fn test_strip_trailing_class_gap() {
        // foo[^/]{0,20} → foo
        assert_opt("foo[^/]{0,20}", "foo");
    }

    #[test]
    fn test_strip_trailing_star() {
        // fooa* → foo
        assert_opt("fooa*", "foo");
    }

    #[test]
    fn test_strip_trailing_optional() {
        // fooa? → foo
        assert_opt("fooa?", "foo");
    }

    #[test]
    fn test_no_strip_trailing_min_nonzero() {
        // foo.{3,10} → NOT stripped
        assert_opt("foo.{3,10}", "foo.{3,10}");
    }

    #[test]
    fn test_no_strip_trailing_anchored_end() {
        // foo.{0,10}$ → NOT stripped ($ makes gap meaningful)
        assert_opt("foo.{0,10}$", "foo.{0,10}$");
    }

    #[test]
    fn test_no_strip_trailing_anchored_endlf() {
        // foo.{0,10}(?m:$) → NOT stripped
        assert_opt("foo.{0,10}(?m:$)", "foo.{0,10}(?m:$)");
    }

    // ── Both sides ──

    #[test]
    fn test_strip_both_sides() {
        // .{0,5}foo.{0,5} → foo
        assert_opt(".{0,5}foo.{0,5}", "foo");
    }

    #[test]
    fn test_strip_leading_only() {
        // .{0,5}foo.{3,5} → foo.{3,5}
        assert_opt(".{0,5}foo.{3,5}", "foo.{3,5}");
    }

    #[test]
    fn test_strip_trailing_only() {
        // .{3,5}foo.{0,5} → .{3,5}foo
        assert_opt(".{3,5}foo.{0,5}", ".{3,5}foo");
    }

    // ── Nested anchors / groups ──

    #[test]
    fn test_no_strip_nested_start_anchor_in_alternation() {
        // (^.{0,10}foo)|bar → branch 1 has ^, NOT stripped; branch 2 unchanged
        // The top-level is an Alternation, not a Concat — no stripping at all
        // Captures stripped by optimize, so expected uses non-capturing syntax.
        assert_opt("(^.{0,10}foo)|bar", "(?:^.{0,10}foo)|bar");
    }

    #[test]
    fn test_no_strip_inside_nested_concat() {
        // Nested a?a? inside a group must NOT be stripped
        // ^(a?a?)?$ — the a? inside the group is semantically relevant
        assert_opt("^(a?a?)?$", "^(?:a?a?)?$");
    }

    #[test]
    fn test_no_strip_inside_alternation_branch() {
        // (a?x|b?y) — a? and b? inside branches are NOT stripped
        // (they're in nested concats, not top-level)
        assert_opt("(a?x|b?y)", "(?:a?x|b?y)");
    }

    #[test]
    fn test_strip_top_level_optional_group_unanchored() {
        // (foo){0,3}bar → bar (optional group stripped, capture stripped)
        assert_opt("(foo){0,3}bar", "bar");
    }

    #[test]
    fn test_no_strip_top_level_optional_group_anchored() {
        // ^(foo){0,3}bar → NOT stripped (^ present; capture stripped)
        assert_opt("^(foo){0,3}bar", "^(?:foo){0,3}bar");
    }

    // ── Edge cases ──

    #[test]
    fn test_strip_to_single_literal() {
        // .{0,100}x.{0,100} → x
        assert_opt(".{0,100}x.{0,100}", "x");
    }

    #[test]
    fn test_strip_chained_optional_groups() {
        // (ab){0,5}(cd){0,5}xyz → xyz
        assert_opt("(ab){0,5}(cd){0,5}xyz", "xyz");
    }

    #[test]
    fn test_strip_mixed_min_zero() {
        // .{0,10}.{5,20}foo → .{5,20}foo (only .{0,10} stripped, .{5,20} has min>0)
        // Note: with merging, .{0,10}.{5,20} does NOT merge (min < max required
        // for both, and the body is the same, but the merge conditions are
        // rep_max > 1 && rep.min < rep_max — both satisfy, so they DO merge
        // to .{5,30}). After merge: .{5,30}foo → NOT stripped (min=5).
        assert_opt(".{0,10}.{5,20}foo", ".{5,30}foo");
    }

    // ── Interaction: stripping does NOT happen recursively ──

    #[test]
    fn test_no_recursive_strip_in_repetition_body() {
        // (.{0,5}foo)+ → the .{0,5} is inside a repetition body,
        // not at the top-level concat — must NOT be stripped
        assert_opt("(.{0,5}foo)+", "(?:.{0,5}foo)+");
    }

    #[test]
    fn test_no_recursive_strip_in_alternation() {
        // .{0,5}foo|bar → top-level is Alternation (not Concat)
        // so no stripping at all (stripping only applies to top-level Concat)
        assert_opt(".{0,5}foo|bar", ".{0,5}foo|bar");
    }
}
