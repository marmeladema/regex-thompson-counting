//! Criterion benchmarks comparing rethoc tiers and the `regex` crate on a
//! pathological bounded-repetition pattern.
//!
//! Pattern: `.{0,1000}.{0,1000}.{0,1000}a`
//!
//! Three sequential bounded repetitions with a wildcard body — this creates
//! massive NFA state space (~3000 active states) that punishes engines
//! without efficient counting.  rethoc compiles this to Tier 3 by default
//! (conditional DFA with 3 counters, 14 NFA states).
//!
//! We benchmark four engines:
//!
//! - **rethoc/tier3** — Conditional DFA with range-compressed counters.
//!   O(body_length) per counter per byte.
//! - **rethoc/tier4** — Counter-program DFA.  More general but slower on
//!   this pattern because it tracks full counter contexts.
//! - **rethoc/nfa** — Pure Thompson NFA simulation (tier 0).  Baseline
//!   for rethoc without any DFA acceleration.
//! - **regex** — The `regex` crate (for external comparison).
//!
//! Run with: `cargo bench --bench pathological`

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use regex_thompson_counting::{MatcherMemory, RegexBuilder};

const PATTERN: &str = r".{0,1000}.{0,1000}.{0,1000}a";

fn parse_hir(pattern: &str) -> regex_thompson_counting::Hir {
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

// ---------------------------------------------------------------------------
// Compilation benchmark
// ---------------------------------------------------------------------------

fn bench_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathological/compile");

    group.bench_function("rethoc", |b| {
        let hir = parse_hir(PATTERN);
        b.iter(|| {
            black_box(RegexBuilder::default().build(black_box(&hir)).unwrap());
        })
    });

    group.bench_function("regex", |b| {
        b.iter(|| {
            black_box(
                regex::bytes::RegexBuilder::new(black_box(PATTERN))
                    .unicode(false)
                    .dot_matches_new_line(true)
                    .build()
                    .unwrap(),
            );
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Match benchmarks
// ---------------------------------------------------------------------------

/// Sizes for match benchmarks.
const SIZES: &[usize] = &[1024, 4 * 1024, 16 * 1024, 64 * 1024, 128 * 1024];

/// Maximum input size for slow engines (tier 4 and NFA).
///
/// Both have O(N * max_count) per-byte cost on this pattern, making
/// larger inputs impractical (1 KB already takes ~4 s for NFA and
/// ~1.8 s for tier 4, so 10 samples at 4 KB would exceed 10 minutes).
const SLOW_MAX_SIZE: usize = 1024;

fn bench_no_match(c: &mut Criterion) {
    let hir = parse_hir(PATTERN);
    let rethoc_re = RegexBuilder::default().build(&hir).unwrap();
    let regex_re = regex::bytes::RegexBuilder::new(PATTERN)
        .unicode(false)
        .dot_matches_new_line(true)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("pathological/no_match");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES {
        let hay = vec![b'x'; size];
        group.throughput(Throughput::Bytes(size as u64));

        // rethoc tier 3 (default — conditional DFA, range-compressed)
        group.bench_with_input(BenchmarkId::new("rethoc/tier3", size), &hay, |b, hay| {
            let mut mem = MatcherMemory::default();
            // Warm up the DFA cache.
            let mut m = mem.matcher_for_tier(&rethoc_re, 3).unwrap();
            m.chunk(hay);
            m.finish();
            b.iter(|| {
                let mut m = mem.matcher_for_tier(&rethoc_re, 3).unwrap();
                m.chunk(black_box(hay));
                black_box(m.finish())
            })
        });

        // rethoc tier 4 (counter-program DFA) — only for small sizes.
        if size <= SLOW_MAX_SIZE {
            group.bench_with_input(BenchmarkId::new("rethoc/tier4", size), &hay, |b, hay| {
                let mut mem = MatcherMemory::default();
                let mut m = mem.matcher_for_tier(&rethoc_re, 4).unwrap();
                m.chunk(hay);
                m.finish();
                b.iter(|| {
                    let mut m = mem.matcher_for_tier(&rethoc_re, 4).unwrap();
                    m.chunk(black_box(hay));
                    black_box(m.finish())
                })
            });
        }

        // rethoc NFA (tier 0 — pure Thompson simulation) — only for small sizes.
        if size <= SLOW_MAX_SIZE {
            group.bench_with_input(BenchmarkId::new("rethoc/nfa", size), &hay, |b, hay| {
                let mut mem = MatcherMemory::default();
                let mut m = mem.matcher_for_tier(&rethoc_re, 0).unwrap();
                m.chunk(hay);
                m.finish();
                b.iter(|| {
                    let mut m = mem.matcher_for_tier(&rethoc_re, 0).unwrap();
                    m.chunk(black_box(hay));
                    black_box(m.finish())
                })
            });
        }

        // regex crate
        group.bench_with_input(BenchmarkId::new("regex", size), &hay, |b, hay| {
            let _ = regex_re.is_match(hay);
            b.iter(|| black_box(regex_re.is_match(black_box(hay))))
        });
    }

    group.finish();
}

fn bench_match_at_end(c: &mut Criterion) {
    let hir = parse_hir(PATTERN);
    let rethoc_re = RegexBuilder::default().build(&hir).unwrap();
    let regex_re = regex::bytes::RegexBuilder::new(PATTERN)
        .unicode(false)
        .dot_matches_new_line(true)
        .build()
        .unwrap();

    let mut group = c.benchmark_group("pathological/match_at_end");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &size in SIZES {
        let mut hay = vec![b'x'; size];
        hay[size - 1] = b'a';
        group.throughput(Throughput::Bytes(size as u64));

        // rethoc tier 3 (default — conditional DFA, range-compressed)
        group.bench_with_input(BenchmarkId::new("rethoc/tier3", size), &hay, |b, hay| {
            let mut mem = MatcherMemory::default();
            let mut m = mem.matcher_for_tier(&rethoc_re, 3).unwrap();
            m.chunk(hay);
            m.finish();
            b.iter(|| {
                let mut m = mem.matcher_for_tier(&rethoc_re, 3).unwrap();
                m.chunk(black_box(hay));
                black_box(m.finish())
            })
        });

        // rethoc tier 4 (counter-program DFA) — only for small sizes.
        if size <= SLOW_MAX_SIZE {
            group.bench_with_input(BenchmarkId::new("rethoc/tier4", size), &hay, |b, hay| {
                let mut mem = MatcherMemory::default();
                let mut m = mem.matcher_for_tier(&rethoc_re, 4).unwrap();
                m.chunk(hay);
                m.finish();
                b.iter(|| {
                    let mut m = mem.matcher_for_tier(&rethoc_re, 4).unwrap();
                    m.chunk(black_box(hay));
                    black_box(m.finish())
                })
            });
        }

        // rethoc NFA (tier 0 — pure Thompson simulation) — only for small sizes.
        if size <= SLOW_MAX_SIZE {
            group.bench_with_input(BenchmarkId::new("rethoc/nfa", size), &hay, |b, hay| {
                let mut mem = MatcherMemory::default();
                let mut m = mem.matcher_for_tier(&rethoc_re, 0).unwrap();
                m.chunk(hay);
                m.finish();
                b.iter(|| {
                    let mut m = mem.matcher_for_tier(&rethoc_re, 0).unwrap();
                    m.chunk(black_box(hay));
                    black_box(m.finish())
                })
            });
        }

        // regex crate
        group.bench_with_input(BenchmarkId::new("regex", size), &hay, |b, hay| {
            let _ = regex_re.is_match(hay);
            b.iter(|| black_box(regex_re.is_match(black_box(hay))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_compile, bench_no_match, bench_match_at_end);
criterion_main!(benches);
