//! Criterion benchmarks comparing rethoc vs the `regex` crate on a
//! pathological bounded-repetition pattern.
//!
//! Pattern: `.{0,1000}.{0,1000}.{0,1000}a`
//!
//! Three sequential bounded repetitions with a wildcard body — this creates
//! massive NFA state space (~3000 active states) that punishes engines
//! without efficient counting.  rethoc compiles this to Tier 3
//! (conditional DFA with 3 counters, 14 NFA states, 1177 bytes).
//!
//! **Key findings**:
//!
//! - **Compilation**: rethoc is ~140× faster (3 µs vs 430 µs).
//!
//! - **No-match**: the `regex` crate extracts `a` as a literal prefilter
//!   and uses memchr to scan the input, returning instantly (~16 ns for
//!   1 KB).  rethoc has no prefilter for this pattern and runs the full
//!   Tier 3 DFA at every byte position (~5.4 ms for 1 KB).
//!
//! - **Match-at-end**: both engines must actually simulate the pattern.
//!   rethoc's counting DFA is ~2.5× faster than the `regex` crate's
//!   backtracker / NFA on this workload, but both exhibit super-linear
//!   scaling.
//!
//! Run with: `cargo bench --bench pathological`

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

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

/// Sizes for match benchmarks.  Both engines are super-linear on this
/// pattern, so we cap at 64 KB to keep total bench time reasonable.
const SIZES: &[usize] = &[1024, 4 * 1024, 16 * 1024, 64 * 1024];

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

        group.bench_with_input(BenchmarkId::new("rethoc", size), &hay, |b, hay| {
            let mut mem = MatcherMemory::default();
            let mut m = mem.matcher(&rethoc_re);
            m.chunk(hay);
            m.finish();
            b.iter(|| {
                let mut m = mem.matcher(&rethoc_re);
                m.chunk(black_box(hay));
                black_box(m.finish())
            })
        });

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

        group.bench_with_input(BenchmarkId::new("rethoc", size), &hay, |b, hay| {
            let mut mem = MatcherMemory::default();
            let mut m = mem.matcher(&rethoc_re);
            m.chunk(hay);
            m.finish();
            b.iter(|| {
                let mut m = mem.matcher(&rethoc_re);
                m.chunk(black_box(hay));
                black_box(m.finish())
            })
        });

        group.bench_with_input(BenchmarkId::new("regex", size), &hay, |b, hay| {
            let _ = regex_re.is_match(hay);
            b.iter(|| black_box(regex_re.is_match(black_box(hay))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_compile, bench_no_match, bench_match_at_end);
criterion_main!(benches);
