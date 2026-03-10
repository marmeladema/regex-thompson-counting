//! Fuzz target: cross-tier differential.
//!
//! The NFA simulator is the oracle — all eligible DFA tiers must produce
//! the same `is_match` result.  This catches tier-specific bugs without
//! needing an external crate.
//!
//! Run with:
//! ```sh
//! cargo +nightly fuzz run fuzz_differential
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;

use regex_thompson_counting::fuzz_gen::{generate_inputs, generate_pattern, FuzzRng};
use regex_thompson_counting::{MatcherMemory, RegexBuilder};

fuzz_target!(|data: &[u8]| {
    let mid = data.len() / 2;
    let (pattern_seed, input_seed) = data.split_at(mid);

    let (pattern, ast) = generate_pattern(&mut FuzzRng::new(pattern_seed));
    let inputs = generate_inputs(&mut FuzzRng::new(input_seed), &ast);

    let hir = match parse_hir_bytes(&pattern) {
        Some(h) => h,
        None => return,
    };

    let mut builder = RegexBuilder::default();
    let re = match builder.build(&hir) {
        Ok(r) => r,
        Err(_) => return,
    };

    let mut memory = MatcherMemory::default();

    for input in &inputs {
        // NFA is the ground truth.
        let mut m = match memory.matcher_for_tier(&re, 0) {
            Ok(m) => m,
            Err(_) => continue,
        };
        m.chunk(input);
        let nfa_result = m.finish();

        for tier in 1..=4u8 {
            if let Ok(mut m) = memory.matcher_for_tier(&re, tier) {
                m.chunk(input);
                let tier_result = m.finish();
                assert_eq!(
                    tier_result,
                    nfa_result,
                    "Tier {} disagrees with NFA for `{}` on input len={}: \
                     tier{}={}, nfa={}",
                    tier,
                    pattern,
                    input.len(),
                    tier,
                    tier_result,
                    nfa_result
                );
            }
        }
    }

    // Second pass: no unrolling.
    builder.max_unroll_states(0);
    if let Ok(re_no_unroll) = builder.build(&hir) {
        for input in &inputs {
            let mut m = match memory.matcher_for_tier(&re_no_unroll, 0) {
                Ok(m) => m,
                Err(_) => continue,
            };
            m.chunk(input);
            let nfa_result = m.finish();

            for tier in 1..=4u8 {
                if let Ok(mut m) = memory.matcher_for_tier(&re_no_unroll, tier) {
                    m.chunk(input);
                    let tier_result = m.finish();
                    assert_eq!(
                        tier_result,
                        nfa_result,
                        "Tier {} disagrees with NFA (no-unroll) for `{}` on input len={}: \
                         tier{}={}, nfa={}",
                        tier,
                        pattern,
                        input.len(),
                        tier,
                        tier_result,
                        nfa_result
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
