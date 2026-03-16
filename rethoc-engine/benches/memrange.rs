//! Criterion benchmarks for `memrange`: SIMD (NEON/SSE2/AVX2) vs scalar.
//!
//! Measures wall-clock throughput (bytes/sec) to quantify the real-world
//! benefit of SIMD byte-range search over the scalar fallback.
//!
//! Run with: `cargo bench --bench memrange`

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use rethoc_engine::{memrange_scalar, memrange_simd};

/// Build a haystack of `size` bytes filled with `fill`, optionally placing
/// a match byte at `match_pos`.
fn make_haystack(size: usize, fill: u8, match_pos: Option<usize>) -> Vec<u8> {
    let mut hay = vec![fill; size];
    if let Some(pos) = match_pos {
        hay[pos] = b'5'; // in the b'0'..=b'9' range
    }
    hay
}

// ---------------------------------------------------------------------------
// Benchmark: vary scenario (no_match, match_at_end, match_early)
// ---------------------------------------------------------------------------

fn bench_no_match(c: &mut Criterion) {
    let size: usize = 1_000_000;
    let hay = make_haystack(size, b'A', None);

    let mut group = c.benchmark_group("memrange/no_match");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("simd", |b| {
        b.iter(|| memrange_simd(black_box(b'0'), black_box(b'9'), black_box(&hay)))
    });
    group.bench_function("scalar", |b| {
        b.iter(|| memrange_scalar(black_box(b'0'), black_box(b'9'), black_box(&hay)))
    });

    group.finish();
}

fn bench_match_at_end(c: &mut Criterion) {
    let size: usize = 1_000_000;
    let hay = make_haystack(size, b'A', Some(size - 1));

    let mut group = c.benchmark_group("memrange/match_at_end");
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_function("simd", |b| {
        b.iter(|| memrange_simd(black_box(b'0'), black_box(b'9'), black_box(&hay)))
    });
    group.bench_function("scalar", |b| {
        b.iter(|| memrange_scalar(black_box(b'0'), black_box(b'9'), black_box(&hay)))
    });

    group.finish();
}

fn bench_match_early(c: &mut Criterion) {
    let size: usize = 1_000_000;
    let hay = make_haystack(size, b'A', Some(64));

    let mut group = c.benchmark_group("memrange/match_early");
    group.throughput(Throughput::Bytes(64_u64)); // only 64 bytes scanned

    group.bench_function("simd", |b| {
        b.iter(|| memrange_simd(black_box(b'0'), black_box(b'9'), black_box(&hay)))
    });
    group.bench_function("scalar", |b| {
        b.iter(|| memrange_scalar(black_box(b'0'), black_box(b'9'), black_box(&hay)))
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: vary haystack size (fixed scenario: no match)
// ---------------------------------------------------------------------------

fn bench_sizes(c: &mut Criterion) {
    let sizes: &[usize] = &[64, 1024, 64 * 1024, 1_000_000];

    let mut group = c.benchmark_group("memrange/sizes");

    for &size in sizes {
        let hay = make_haystack(size, b'A', None);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("simd", size), &hay, |b, hay| {
            b.iter(|| memrange_simd(black_box(b'0'), black_box(b'9'), black_box(hay)))
        });
        group.bench_with_input(BenchmarkId::new("scalar", size), &hay, |b, hay| {
            b.iter(|| memrange_scalar(black_box(b'0'), black_box(b'9'), black_box(hay)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_no_match,
    bench_match_at_end,
    bench_match_early,
    bench_sizes
);
criterion_main!(benches);
