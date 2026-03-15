use gungraun::{
    library_benchmark, library_benchmark_group, main, Callgrind, FlamegraphConfig,
    LibraryBenchmarkConfig,
};
use regex_thompson_counting::{MatcherMemory, Regex, RegexBuilder};
use std::hint::black_box;

/// Pathological bounded-repetition pattern: three sequential `.{0,1000}`
/// followed by a literal `a`.  Compiled with merging disabled to preserve
/// the 3-counter Tier 3 structure for profiling.  With default merging,
/// these would become `.{0,3000}a` (Tier 2, single counter).
const PATTERN: &str = r".{0,1000}.{0,1000}.{0,1000}a";

/// Haystack size for callgrind profiling.  With compilation factored out
/// into `setup`, we use 128 KB — large enough to expose the super-linear
/// DFA state growth in the match-at-end case while keeping callgrind
/// runtime under a few minutes.
const HAYSTACK_SIZE: usize = 128 * 1024;

fn parse_hir(pattern: &str) -> regex_syntax::hir::Hir {
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

/// Compile the regex once in setup; returns `(Regex, haystack)`.
/// Callgrind only measures the benchmark function, not setup.
fn setup_no_match() -> (Regex, Vec<u8>) {
    let hir = parse_hir(PATTERN);
    let re = RegexBuilder::default()
        .merge_repetitions(false)
        .build(&hir)
        .unwrap();
    let haystack = vec![b'x'; HAYSTACK_SIZE];
    (re, haystack)
}

fn setup_match_at_end() -> (Regex, Vec<u8>) {
    let hir = parse_hir(PATTERN);
    let re = RegexBuilder::default()
        .merge_repetitions(false)
        .build(&hir)
        .unwrap();
    let mut haystack = vec![b'x'; HAYSTACK_SIZE];
    *haystack.last_mut().unwrap() = b'a';
    (re, haystack)
}

// ---- No-match benchmark: pure simulation cost, no early exit ----

#[library_benchmark]
#[bench::no_match(setup = setup_no_match)]
fn bench_no_match((re, haystack): (Regex, Vec<u8>)) {
    let mut mem = MatcherMemory::default();
    let mut matcher = mem.matcher(&re);
    matcher.chunk(black_box(&haystack));
    let matched = matcher.finish();
    assert!(!matched);
    black_box(matched);
}

// ---- Match-at-end benchmark: full simulation + final match ----

#[library_benchmark]
#[bench::match_at_end(setup = setup_match_at_end)]
fn bench_match_at_end((re, haystack): (Regex, Vec<u8>)) {
    let mut mem = MatcherMemory::default();
    let mut matcher = mem.matcher(&re);
    matcher.chunk(black_box(&haystack));
    let matched = matcher.finish();
    assert!(matched);
    black_box(matched);
}

library_benchmark_group!(name = pathological_no_match, benchmarks = bench_no_match);
library_benchmark_group!(
    name = pathological_match_at_end,
    benchmarks = bench_match_at_end
);

main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().flamegraph(FlamegraphConfig::default()));
    library_benchmark_groups = pathological_no_match, pathological_match_at_end
);
