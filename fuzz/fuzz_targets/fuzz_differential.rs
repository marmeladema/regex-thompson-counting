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

use std::cell::RefCell;

thread_local! {
    static CURRENT_PATTERN: RefCell<String> = RefCell::new(String::new());
    static CURRENT_INPUT: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static CURRENT_PHASE: RefCell<String> = RefCell::new(String::new());
    static HOOK_INSTALLED: RefCell<bool> = RefCell::new(false);
}

fn install_panic_hook() {
    HOOK_INSTALLED.with(|h| {
        if !*h.borrow() {
            *h.borrow_mut() = true;
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                CURRENT_PATTERN.with(|p| {
                    CURRENT_PHASE.with(|ph| {
                        CURRENT_INPUT.with(|inp| {
                            let pat = p.borrow();
                            let phase = ph.borrow();
                            let input = inp.borrow();
                            eprintln!("\n╔══════════════════════════════════════════════════");
                            eprintln!("║ FUZZ CRASH");
                            eprintln!("║ Pattern: `{}`", *pat);
                            eprintln!("║ Phase:   {}", *phase);
                            if !input.is_empty() {
                                if let Ok(s) = std::str::from_utf8(&input) {
                                    eprintln!("║ Input:   {:?} (len={})", s, input.len());
                                } else {
                                    eprintln!("║ Input:   {:?} (len={})", *input, input.len());
                                }
                            }
                            eprintln!("╚══════════════════════════════════════════════════\n");
                        });
                    });
                });
                prev(info);
            }));
        }
    });
}

fn set_phase(phase: &str) {
    CURRENT_PHASE.with(|p| *p.borrow_mut() = phase.to_string());
}

fn set_input(input: &[u8]) {
    CURRENT_INPUT.with(|i| {
        let mut v = i.borrow_mut();
        v.clear();
        v.extend_from_slice(input);
    });
}

fuzz_target!(|data: &[u8]| {
    install_panic_hook();

    let mid = data.len() / 2;
    let (pattern_seed, input_seed) = data.split_at(mid);

    let (pattern, ast) = generate_pattern(&mut FuzzRng::new(pattern_seed));
    let inputs = generate_inputs(&mut FuzzRng::new(input_seed), &ast);

    CURRENT_PATTERN.with(|p| *p.borrow_mut() = pattern.clone());
    CURRENT_INPUT.with(|i| i.borrow_mut().clear());

    let hir = match parse_hir_bytes(&pattern) {
        Some(h) => h,
        None => return,
    };

    let mut builder = RegexBuilder::default();
    set_phase("compilation (default unroll)");
    let re = match builder.build(&hir) {
        Ok(r) => r,
        Err(_) => return,
    };

    // Computational budget: skip patterns that are too expensive under
    // ASAN instrumentation.  Patterns with many NFA states (nested
    // optionals) or large total input volume cause timeouts.
    let info = re.info();
    let num_states = info.memory.num_states;
    let total_input_bytes: usize = inputs.iter().map(|i| i.len()).sum();
    let max_input_len = inputs.iter().map(|i| i.len()).max().unwrap_or(0);
    if num_states > 50
        || total_input_bytes > 1000
        || max_input_len > 200
        || num_states * total_input_bytes > 15_000
    {
        return;
    }

    let mut memory = MatcherMemory::default();

    for input in &inputs {
        set_input(input);

        // NFA is the ground truth.
        set_phase("match (NFA)");
        let mut m = match memory.matcher_for_tier(&re, 0) {
            Ok(m) => m,
            Err(_) => continue,
        };
        m.chunk(input);
        let nfa_result = m.finish();

        for tier in 1..=4u8 {
            if let Ok(mut m) = memory.matcher_for_tier(&re, tier) {
                set_phase(&format!("match (Tier {})", tier));
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
    set_phase("compilation (no-unroll)");
    set_input(&[]);
    builder.max_unroll_states(0);
    let re_no_unroll = match builder.build(&hir) {
        Ok(r) => r,
        Err(_) => return,
    };

    // Re-check budget: no-unroll compilation may promote patterns to
    // higher tiers with more expensive matching (e.g. Tier 4 with
    // nested counter programs).
    let info_nu = re_no_unroll.info();
    let num_states_nu = info_nu.memory.num_states;
    if num_states_nu > 50 || num_states_nu * total_input_bytes > 15_000 {
        return;
    }

    for input in &inputs {
        set_input(input);

        set_phase("match (no-unroll NFA)");
        let mut m = match memory.matcher_for_tier(&re_no_unroll, 0) {
            Ok(m) => m,
            Err(_) => continue,
        };
        m.chunk(input);
        let nfa_result = m.finish();

        for tier in 1..=4u8 {
            if let Ok(mut m) = memory.matcher_for_tier(&re_no_unroll, tier) {
                set_phase(&format!("match (no-unroll Tier {})", tier));
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
