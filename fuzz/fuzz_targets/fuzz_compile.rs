//! Fuzz target: compilation robustness.
//!
//! Generates structured regex patterns and compiles them.  The pattern
//! must either compile successfully or return an `Err` — it must never
//! panic, hang, or OOM.  If compilation succeeds, we also verify that
//! `.info()` and `.memory_size()` don't panic.
//!
//! Run with:
//! ```sh
//! cargo +nightly fuzz run fuzz_compile
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;

use rethoc_engine::fuzz_gen::{generate_pattern, FuzzRng};
use rethoc_engine::RegexBuilder;

use std::cell::RefCell;

thread_local! {
    static CURRENT_PATTERN: RefCell<String> = RefCell::new(String::new());
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
                        let pat = p.borrow();
                        let phase = ph.borrow();
                        eprintln!("\n╔══════════════════════════════════════════════════");
                        eprintln!("║ FUZZ CRASH");
                        eprintln!("║ Pattern: `{}`", *pat);
                        eprintln!("║ Phase:   {}", *phase);
                        eprintln!("╚══════════════════════════════════════════════════\n");
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

fuzz_target!(|data: &[u8]| {
    install_panic_hook();

    let (pattern, _ast) = generate_pattern(&mut FuzzRng::new(data));

    CURRENT_PATTERN.with(|p| *p.borrow_mut() = pattern.clone());

    let hir = match parse_hir_bytes(&pattern) {
        Some(h) => h,
        None => return,
    };

    let mut builder = RegexBuilder::default();
    set_phase("compilation (default unroll)");
    match builder.build(&hir) {
        Ok(re) => {
            // These must not panic.
            set_phase("info/memory_size (default unroll)");
            let _ = re.memory_size();
            let _ = re.info();
            let _ = re.min_tier();
        }
        Err(_) => {
            // Expected for some patterns (too many counters, etc.).
        }
    }

    // Also try with unrolling disabled.
    set_phase("compilation (no-unroll)");
    builder.max_unroll_states(0);
    match builder.build(&hir) {
        Ok(re) => {
            set_phase("info/memory_size (no-unroll)");
            let _ = re.memory_size();
            let _ = re.info();
            let _ = re.min_tier();
        }
        Err(_) => {}
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
