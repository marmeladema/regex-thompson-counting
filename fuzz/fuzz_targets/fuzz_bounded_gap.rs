//! Fuzz target: bounded-gap differential.
//!
//! Generates patterns that are eligible for the bounded-gap engine,
//! then compares bounded-gap execution directly against tier 0 (NFA),
//! tier 1 (lazy DFA), and tier 2 (differential-counter DFA).
//!
//! Unlike `fuzz_match` and `fuzz_differential` which exercise all pattern
//! shapes, this target ensures every iteration produces a bounded-gap
//! pattern — no wasted iterations on patterns the specialisation rejects.
//!
//! Run with:
//! ```sh
//! cargo +nightly fuzz run fuzz_bounded_gap
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;

use rethoc_engine::fuzz_gen::{generate_bounded_gap_pattern, generate_inputs, FuzzRng};
use rethoc_engine::{MatcherMemory, RegexBuilder};

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
                            eprintln!("║ FUZZ CRASH (fuzz_bounded_gap)");
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

    // Generate a pattern guaranteed to be bounded-gap eligible.
    let (pattern, ast) = generate_bounded_gap_pattern(&mut FuzzRng::new(pattern_seed));
    let inputs = generate_inputs(&mut FuzzRng::new(input_seed), &ast);

    CURRENT_PATTERN.with(|p| *p.borrow_mut() = pattern.clone());
    CURRENT_INPUT.with(|i| i.borrow_mut().clear());

    // Parse and compile.
    let hir = match parse_hir_bytes(&pattern) {
        Some(h) => h,
        None => return,
    };

    set_phase("compilation");
    let mut builder = RegexBuilder::default();
    let re = match builder.build(&hir) {
        Ok(r) => r,
        Err(_) => return,
    };

    // Skip if no bounded-gap plan (some patterns may get optimised
    // away, e.g. all gaps normalised to {0,0}).
    if !re.has_bounded_gap_plan() {
        return;
    }

    // Computational budget.
    let total_input_bytes: usize = inputs.iter().map(|i| i.len()).sum();
    let max_input_len = inputs.iter().map(|i| i.len()).max().unwrap_or(0);
    if total_input_bytes > 2000 || max_input_len > 300 {
        return;
    }

    let mut memory = MatcherMemory::default();

    for input in &inputs {
        set_input(input);

        // NFA (tier 0) is the oracle.
        set_phase("match (NFA)");
        let mut m = match memory.matcher_for_tier(&re, 0) {
            Ok(m) => m,
            Err(_) => continue,
        };
        m.chunk(input);
        let nfa_result = m.finish();

        // Bounded-gap engine: full chunk.
        set_phase("match (BoundedGap full-chunk)");
        let mut m = memory
            .bounded_gap_matcher(&re)
            .expect("bounded-gap matcher must be available");
        m.chunk(input);
        let bg_result = m.finish();
        assert_eq!(
            bg_result,
            nfa_result,
            "BoundedGap (full-chunk) disagrees with NFA for `{}` on input len={}: \
             bg={}, nfa={}",
            pattern,
            input.len(),
            bg_result,
            nfa_result
        );

        // Bounded-gap engine: byte-at-a-time.
        set_phase("match (BoundedGap byte-at-a-time)");
        let mut m = memory
            .bounded_gap_matcher(&re)
            .expect("bounded-gap matcher must be available");
        for &b in input.iter() {
            m.chunk(&[b]);
        }
        let bg_byte_result = m.finish();
        assert_eq!(
            bg_byte_result,
            nfa_result,
            "BoundedGap (byte-at-a-time) disagrees with NFA for `{}` on input len={}: \
             bg={}, nfa={}",
            pattern,
            input.len(),
            bg_byte_result,
            nfa_result
        );

        // Tier 1 (lazy DFA) if eligible.
        if let Ok(mut m) = memory.matcher_for_tier(&re, 1) {
            set_phase("match (Tier 1)");
            m.chunk(input);
            let t1_result = m.finish();
            assert_eq!(
                t1_result,
                nfa_result,
                "Tier 1 disagrees with NFA for `{}` on input len={}: \
                 tier1={}, nfa={}",
                pattern,
                input.len(),
                t1_result,
                nfa_result
            );
        }

        // Tier 2 (differential counters) if eligible.
        if let Ok(mut m) = memory.matcher_for_tier(&re, 2) {
            set_phase("match (Tier 2)");
            m.chunk(input);
            let t2_result = m.finish();
            assert_eq!(
                t2_result,
                nfa_result,
                "Tier 2 disagrees with NFA for `{}` on input len={}: \
                 tier2={}, nfa={}",
                pattern,
                input.len(),
                t2_result,
                nfa_result
            );
        }
    }
});

fn parse_hir_bytes(pattern: &str) -> Option<rethoc_engine::Hir> {
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
