//! Criterion benchmarks for pathological bounded-repetition patterns.
//!
//! # Non-nested pattern (`.{0,1000}.{0,1000}.{0,1000}a`)
//!
//! Three sequential bounded repetitions with a wildcard body — this creates
//! massive NFA state space (~3000 active states) that punishes engines
//! without efficient counting.  With default compilation, rethoc merges
//! the three repetitions into `.{0,3000}a` (single counter, Tier 2).
//! The tier3/tier4/nfa benchmarks disable merging to preserve the
//! 3-counter structure and exercise those tiers directly.
//!
//! Benchmarked engines: rethoc/default, rethoc/tier3, rethoc/tier4,
//! rethoc/nfa, regex.
//!
//! # Nested pattern (`(.{0,1000}a){0,1000}b`)
//!
//! Nested bounded repetition requiring Tier 4.  The inner `.{0,1000}`
//! counter is nested inside the outer `(...){0,1000}` counter, creating
//! an O(max_inner × max_outer) context space.  This pattern is not
//! eligible for Tier 3 (nested counters) or Tier 2 (overlapping bytes).
//!
//! Benchmarked engines: rethoc/tier4, regex.
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
/// Both have O(N * max_count) per-byte cost on this pattern.  After
/// hash-based dedup optimizations, 1 KB takes ~54 ms for NFA and
/// ~145 ms for tier 4, making 4 KB practical (~1 s worst case per
/// sample for tier 4).
const SLOW_MAX_SIZE: usize = 4 * 1024;

fn bench_no_match(c: &mut Criterion) {
    let hir = parse_hir(PATTERN);
    // Default compilation: merges .{0,1000}.{0,1000}.{0,1000} → .{0,3000}
    // (single counter, Tier 2).
    let rethoc_merged = RegexBuilder::default()
        .max_estimated_states(4096)
        .build(&hir)
        .unwrap();
    assert_eq!(rethoc_merged.min_tier(), 2);
    // No-merge compilation: preserves 3 separate counters for tier3/tier4/nfa
    // benchmarks that need the multi-counter structure.
    let rethoc_re = RegexBuilder::default()
        .merge_repetitions(false)
        .max_estimated_states(4096)
        .build(&hir)
        .unwrap();
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

        // rethoc default (merged repetitions → single counter, Tier 2)
        group.bench_with_input(BenchmarkId::new("rethoc/default", size), &hay, |b, hay| {
            let mut mem = MatcherMemory::default();
            let mut m = mem.matcher(&rethoc_merged);
            m.chunk(hay);
            m.finish();
            b.iter(|| {
                let mut m = mem.matcher(&rethoc_merged);
                m.chunk(black_box(hay));
                black_box(m.finish())
            })
        });

        // rethoc tier 2 (merged - differential-counter DFA, 1 counter)
        group.bench_with_input(BenchmarkId::new("rethoc/tier2", size), &hay, |b, hay| {
            let mut mem = MatcherMemory::default();
            // Warm up the DFA cache.
            let mut m = mem.matcher_for_tier(&rethoc_merged, 2).unwrap();
            m.chunk(hay);
            m.finish();
            b.iter(|| {
                let mut m = mem.matcher_for_tier(&rethoc_merged, 2).unwrap();
                m.chunk(black_box(hay));
                black_box(m.finish())
            })
        });

        // rethoc tier 3 (no-merge — conditional DFA, range-compressed, 3 counters)
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
    let rethoc_merged = RegexBuilder::default()
        .max_estimated_states(4096)
        .build(&hir)
        .unwrap();
    assert_eq!(rethoc_merged.min_tier(), 2);
    let rethoc_re = RegexBuilder::default()
        .merge_repetitions(false)
        .max_estimated_states(4096)
        .build(&hir)
        .unwrap();
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

        // rethoc default (merged repetitions → single counter, Tier 2)
        group.bench_with_input(BenchmarkId::new("rethoc/default", size), &hay, |b, hay| {
            let mut mem = MatcherMemory::default();
            let mut m = mem.matcher(&rethoc_merged);
            m.chunk(hay);
            m.finish();
            b.iter(|| {
                let mut m = mem.matcher(&rethoc_merged);
                m.chunk(black_box(hay));
                black_box(m.finish())
            })
        });

        // rethoc tier 2 (merged - differential-counter DFA, 1 counter)
        group.bench_with_input(BenchmarkId::new("rethoc/tier2", size), &hay, |b, hay| {
            let mut mem = MatcherMemory::default();
            let mut m = mem.matcher_for_tier(&rethoc_merged, 2).unwrap();
            m.chunk(hay);
            m.finish();
            b.iter(|| {
                let mut m = mem.matcher_for_tier(&rethoc_merged, 2).unwrap();
                m.chunk(black_box(hay));
                black_box(m.finish())
            })
        });

        // rethoc tier 3 (no-merge — conditional DFA, range-compressed, 3 counters)
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

// ---------------------------------------------------------------------------
// Nested pattern benchmarks
// ---------------------------------------------------------------------------

const NESTED_PATTERN: &str = r"(.{0,1000}a){0,1000}b";

/// Sizes for nested benchmarks.  Tier 4 on this pattern is O(N * max_inner *
/// max_outer) so only small inputs are practical.
const NESTED_SIZES: &[usize] = &[256, 1024];

fn bench_nested_match_at_end(c: &mut Criterion) {
    let hir = parse_hir(NESTED_PATTERN);
    let rethoc_re = RegexBuilder::default().build(&hir).unwrap();

    // The regex crate cannot compile this nested pattern within its default
    // 10 MB NFA size limit (or even 100 MB), so we only benchmark rethoc.
    let regex_re = regex::bytes::RegexBuilder::new(NESTED_PATTERN)
        .unicode(false)
        .dot_matches_new_line(true)
        .size_limit(500 * (1 << 20))
        .build()
        .ok();

    let mut group = c.benchmark_group("pathological_nested/match_at_end");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &size in NESTED_SIZES {
        let mut hay = vec![b'x'; size];
        hay[size - 1] = b'b';
        group.throughput(Throughput::Bytes(size as u64));

        // rethoc tier 4 (counter-program DFA)
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

        // regex crate (only if it could compile the pattern)
        if let Some(ref regex_re) = regex_re {
            group.bench_with_input(BenchmarkId::new("regex", size), &hay, |b, hay| {
                let _ = regex_re.is_match(hay);
                b.iter(|| black_box(regex_re.is_match(black_box(hay))))
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compile,
    bench_no_match,
    bench_match_at_end,
    bench_nested_match_at_end,
);
criterion_main!(benches);
