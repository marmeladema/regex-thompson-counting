//! Fuzz target: oracle differential.
//!
//! Generates a structured regex pattern and many targeted inputs, then
//! compares `rethoc` (all eligible tiers) against the `regex` crate.
//!
//! Run with:
//! ```sh
//! cargo +nightly fuzz run fuzz_match
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;

use regex_thompson_counting::fuzz_gen::{generate_inputs, generate_pattern, FuzzRng};
use regex_thompson_counting::{MatcherMemory, RegexBuilder};

fuzz_target!(|data: &[u8]| {
    // Split fuzzer data: first half drives pattern generation,
    // second half drives input generation + extra random inputs.
    let mid = data.len() / 2;
    let (pattern_seed, input_seed) = data.split_at(mid);

    let (pattern, ast) = generate_pattern(&mut FuzzRng::new(pattern_seed));
    let inputs = generate_inputs(&mut FuzzRng::new(input_seed), &ast);

    // Parse with regex-syntax in byte mode.
    let hir = match parse_hir_bytes(&pattern) {
        Some(h) => h,
        None => return,
    };

    // Compile with rethoc.
    let mut builder = RegexBuilder::default();
    let re = match builder.build(&hir) {
        Ok(r) => r,
        Err(_) => return,
    };

    // Compile with the regex crate oracle.
    let full = format!("(?s-u){}", pattern);
    let oracle = match regex::bytes::Regex::new(&full) {
        Ok(o) => o,
        Err(_) => return,
    };

    let mut memory = MatcherMemory::default();

    for input in &inputs {
        let expected = oracle.is_match(input);

        // Default (highest-eligible) tier.
        let mut matcher = memory.matcher(&re);
        matcher.chunk(input);
        let actual = matcher.finish();
        assert_eq!(
            actual,
            expected,
            "oracle mismatch (default tier) for `{}` on input len={}: ours={}, oracle={}",
            pattern,
            input.len(),
            actual,
            expected
        );

        // NFA (always available).
        if let Ok(mut m) = memory.matcher_for_tier(&re, 0) {
            m.chunk(input);
            let nfa_result = m.finish();
            assert_eq!(
                nfa_result,
                expected,
                "NFA mismatch for `{}` on input len={}: nfa={}, oracle={}",
                pattern,
                input.len(),
                nfa_result,
                expected
            );
        }

        // Each eligible DFA tier.
        for tier in 1..=4u8 {
            if let Ok(mut m) = memory.matcher_for_tier(&re, tier) {
                m.chunk(input);
                let tier_result = m.finish();
                assert_eq!(
                    tier_result,
                    expected,
                    "Tier {} mismatch for `{}` on input len={}: tier{}={}, oracle={}",
                    tier,
                    pattern,
                    input.len(),
                    tier,
                    tier_result,
                    expected
                );
            }
        }
    }

    // Second pass: recompile without unrolling.
    builder.max_unroll_states(0);
    if let Ok(re_no_unroll) = builder.build(&hir) {
        for input in &inputs {
            let expected = oracle.is_match(input);

            let mut matcher = memory.matcher(&re_no_unroll);
            matcher.chunk(input);
            let actual = matcher.finish();
            assert_eq!(
                actual,
                expected,
                "oracle mismatch (no-unroll) for `{}` on input len={}: ours={}, oracle={}",
                pattern,
                input.len(),
                actual,
                expected
            );

            for tier in 0..=4u8 {
                if let Ok(mut m) = memory.matcher_for_tier(&re_no_unroll, tier) {
                    m.chunk(input);
                    let tier_result = m.finish();
                    assert_eq!(
                        tier_result,
                        expected,
                        "Tier {} mismatch (no-unroll) for `{}` on input len={}: \
                         tier{}={}, oracle={}",
                        tier,
                        pattern,
                        input.len(),
                        tier,
                        tier_result,
                        expected
                    );
                }
            }
        }
    }
});

fn parse_hir_bytes(pattern: &str) -> Option<regex_thompson_counting::Hir> {
    use regex_syntax::ast::parse::ParserBuilder;
    use regex_syntax::hir::translate::TranslatorBuilder;

    let ast = ParserBuilder::new().build().parse(pattern).ok()?;
    TranslatorBuilder::new()
        .unicode(false)
        .utf8(false)
        .dot_matches_new_line(true)
        .build()
        .translate(pattern, &ast)
        .ok()
}
