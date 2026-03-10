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

use regex_thompson_counting::fuzz_gen::{generate_pattern, FuzzRng};
use regex_thompson_counting::RegexBuilder;

fuzz_target!(|data: &[u8]| {
    let (pattern, _ast) = generate_pattern(&mut FuzzRng::new(data));

    let hir = match parse_hir_bytes(&pattern) {
        Some(h) => h,
        None => return,
    };

    let mut builder = RegexBuilder::default();
    match builder.build(&hir) {
        Ok(re) => {
            // These must not panic.
            let _ = re.memory_size();
            let _ = re.info();
            let _ = re.min_tier();
        }
        Err(_) => {
            // Expected for some patterns (too many counters, etc.).
        }
    }

    // Also try with unrolling disabled.
    builder.max_unroll_states(0);
    match builder.build(&hir) {
        Ok(re) => {
            let _ = re.memory_size();
            let _ = re.info();
            let _ = re.min_tier();
        }
        Err(_) => {}
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
