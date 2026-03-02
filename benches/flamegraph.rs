use gungraun::{
    Callgrind, FlamegraphConfig, LibraryBenchmarkConfig, library_benchmark,
    library_benchmark_group, main,
};
use regex_thompson_counting::{MatcherMemory, RegexBuilder};
use std::hint::black_box;

// aws-keys "quick" pattern from rebar benchmark
const AWS_PATTERN: &str = r"((?:ASIA|AKIA|AROA|AIDA)([A-Z0-7]{16}))";

// grep/every-line: empty pattern (matches every line)
const GREP_EVERY_LINE_PATTERN: &str = "";

/// Truncation size for callgrind profiling.  The full haystack is ~7 MB;
/// running it under valgrind would take many minutes.  1 MB is enough to
/// produce a representative instruction-count profile.
const HAYSTACK_LIMIT: usize = 1_000_000;

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

fn load_haystack(path: &str) -> Vec<u8> {
    let data = std::fs::read(path).expect("failed to read haystack file");
    data[..data.len().min(HAYSTACK_LIMIT)].to_vec()
}

fn aws_haystack() -> Vec<u8> {
    load_haystack(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/rebar/benchmarks/haystacks/wild/cpython-226484e4.py"
    ))
}

fn grep_haystack() -> Vec<u8> {
    load_haystack(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/bench/rebar/benchmarks/haystacks/rust-src-tools-3b0d4813.txt"
    ))
}

/// Iterate lines like bstr::ByteSlice::lines() — split on `\n`,
/// strip optional trailing `\r`.
fn for_each_line(haystack: &[u8], mut f: impl FnMut(&[u8])) {
    let mut rest = haystack;
    while let Some(pos) = memchr::memchr(b'\n', rest) {
        let line = if pos > 0 && rest[pos - 1] == b'\r' {
            &rest[..pos - 1]
        } else {
            &rest[..pos]
        };
        f(line);
        rest = &rest[pos + 1..];
    }
    if !rest.is_empty() {
        let line = if rest.last() == Some(&b'\r') {
            &rest[..rest.len() - 1]
        } else {
            rest
        };
        f(line);
    }
}

// ---- aws-keys/quick benchmark (NFA path, has counters) ----

#[library_benchmark]
fn bench_aws_keys_quick() {
    let haystack = aws_haystack();
    let hir = parse_hir(AWS_PATTERN);
    let re = RegexBuilder::default().build(&hir).unwrap();
    let mut mem = MatcherMemory::default();
    let mut matcher = mem.matcher(&re);
    matcher.chunk(black_box(&haystack));
    let matched = matcher.finish();
    black_box(matched);
}

// ---- grep/every-line benchmark (DFA path, per-line iteration) ----

#[library_benchmark]
fn bench_grep_every_line() {
    let haystack = grep_haystack();
    let hir = parse_hir(GREP_EVERY_LINE_PATTERN);
    let re = RegexBuilder::default().build(&hir).unwrap();
    let mut mem = MatcherMemory::default();
    let mut count = 0usize;
    for_each_line(&haystack, |line| {
        let mut m = mem.matcher(&re);
        m.chunk(black_box(line));
        if m.finish() {
            count += 1;
        }
    });
    black_box(count);
}

// ---- Tier 1 DFA on same haystack for per-byte cost comparison ----

/// Same haystack, DFA-eligible equivalent of the aws-keys pattern.
/// The `{16}` repetition is expanded into 16 copies of the character class
/// so the pattern remains counter-free and uses the Tier 1 DFA path.
const AWS_PREFIX_PATTERN: &str = r"(?:ASIA|AKIA|AROA|AIDA)[A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7][A-Z0-7]";

#[library_benchmark]
fn bench_aws_prefix_tier1() {
    let haystack = aws_haystack();
    let hir = parse_hir(AWS_PREFIX_PATTERN);
    let re = RegexBuilder::default().build(&hir).unwrap();
    let mut mem = MatcherMemory::default();
    let mut matcher = mem.matcher(&re);
    matcher.chunk(black_box(&haystack));
    let matched = matcher.finish();
    black_box(matched);
}

library_benchmark_group!(name = aws_keys_group, benchmarks = bench_aws_keys_quick);
library_benchmark_group!(name = aws_prefix_group, benchmarks = bench_aws_prefix_tier1);
library_benchmark_group!(name = grep_group, benchmarks = bench_grep_every_line);

// main! must be at file scope — it generates fn main() internally.
main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default().flamegraph(FlamegraphConfig::default()));
    library_benchmark_groups = aws_keys_group, aws_prefix_group, grep_group
);
