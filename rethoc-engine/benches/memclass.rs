use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rethoc_engine::memclass::{MemclassTable, memclass, memclass_scalar};

fn make_haystack(size: usize) -> Vec<u8> {
    // Fill with bytes that DON'T match any of our test classes.
    // Use 0x00 which has low/high nibbles that won't collide with
    // typical ASCII targets.
    vec![0x00; size]
}

fn bench_no_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("memclass/no_match");

    // Benchmark different class sizes (number of target bytes).
    let cases: &[(&str, &[u8])] = &[
        ("2_bytes", &[b'a', b'A']),
        ("4_bytes", &[b'a', b'A', b'b', b'B']),
        ("6_bytes", &[b'a', b'A', b'b', b'B', b'c', b'C']),
        ("8_bytes", &[b'a', b'A', b'b', b'B', b'c', b'C', b'd', b'D']),
    ];

    for &size in &[1024, 16384, 65536] {
        let hay = make_haystack(size);
        group.throughput(Throughput::Bytes(size as u64));

        for &(name, bytes) in cases {
            let table = MemclassTable::new(bytes);

            group.bench_with_input(
                BenchmarkId::new(format!("simd/{name}"), size),
                &hay,
                |b, hay| {
                    b.iter(|| {
                        assert_eq!(memclass(&table, hay), None);
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("scalar/{name}"), size),
                &hay,
                |b, hay| {
                    b.iter(|| {
                        assert_eq!(memclass_scalar(&table, hay), None);
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_match_at_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("memclass/match_at_end");

    let cases: &[(&str, &[u8])] = &[
        ("2_bytes", &[b'a', b'A']),
        ("4_bytes", &[b'a', b'A', b'b', b'B']),
        ("6_bytes", &[b'a', b'A', b'b', b'B', b'c', b'C']),
        ("8_bytes", &[b'a', b'A', b'b', b'B', b'c', b'C', b'd', b'D']),
    ];

    for &size in &[1024, 16384, 65536] {
        let mut hay = make_haystack(size);
        hay[size - 1] = b'C'; // match on last byte
        group.throughput(Throughput::Bytes(size as u64));

        for &(name, bytes) in cases {
            let table = MemclassTable::new(bytes);

            group.bench_with_input(
                BenchmarkId::new(format!("simd/{name}"), size),
                &hay,
                |b, hay| {
                    b.iter(|| {
                        assert_eq!(memclass(&table, hay), Some(size - 1));
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("scalar/{name}"), size),
                &hay,
                |b, hay| {
                    b.iter(|| {
                        assert_eq!(memclass_scalar(&table, hay), Some(size - 1));
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_vs_memchr3(c: &mut Criterion) {
    let mut group = c.benchmark_group("memclass/vs_memchr3");

    // Compare memclass with 3 bytes against memchr3 with the same 3 bytes.
    let bytes_3 = [b'a', b'b', b'c'];
    let table = MemclassTable::new(&bytes_3);

    for &size in &[1024, 16384, 65536] {
        let hay = make_haystack(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("memclass_simd", size), &hay, |b, hay| {
            b.iter(|| {
                assert_eq!(memclass(&table, hay), None);
            });
        });

        group.bench_with_input(BenchmarkId::new("memchr3", size), &hay, |b, hay| {
            b.iter(|| {
                assert_eq!(memchr::memchr3(b'a', b'b', b'c', hay), None);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_no_match,
    bench_match_at_end,
    bench_vs_memchr3
);
criterion_main!(benches);
