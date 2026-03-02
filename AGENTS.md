# Agent Reference: Benchmarks & Profiling

Quick-reference for running benchmarks, callgrind profiling, and source annotation.
Avoids having to rediscover paths and flags each time.

## 1. Callgrind Benchmarks (gungraun/iai-callgrind)

The benchmark binary is at `benches/flamegraph.rs`. It uses the `gungraun` harness
which wraps `valgrind --tool=callgrind` for deterministic instruction counts.

### Run all benchmarks

```bash
cargo bench --bench flamegraph
```

This runs all three benchmarks and prints instruction counts with deltas vs the
previous run. Output files land under `target/gungraun/`.

### Available benchmarks

```
flamegraph::aws_keys_group::bench_aws_keys_quick      # Tier 3 counting DFA (has {16})
flamegraph::aws_prefix_group::bench_aws_prefix_tier1   # Tier 1 DFA (expanded, no counters)
flamegraph::grep_group::bench_grep_every_line           # per-line iteration (empty pattern)
```

List them with:
```bash
cargo bench --bench flamegraph -- --list
```

### Typical instruction counts (1 MB haystack)

| Benchmark | Instructions | Instr/byte |
|---|---|---|
| `bench_aws_prefix_tier1` | ~29M | ~29 |
| `bench_aws_keys_quick` | ~54M | ~54 |
| `bench_grep_every_line` | ~7.9M | — |

## 2. Callgrind Source Annotation

After `cargo bench --bench flamegraph`, callgrind output files are at:

```
target/gungraun/regex-thompson-counting/flamegraph/<group>/<bench>/callgrind.<bench>.out
```

For example:
```
target/gungraun/regex-thompson-counting/flamegraph/aws_prefix_group/bench_aws_prefix_tier1/callgrind.bench_aws_prefix_tier1.out
```

The `.out.old` file is the previous run (used for delta comparison).

### Annotate with source (top functions)

```bash
callgrind_annotate --auto=yes --inclusive=no --threshold=95 \
  target/gungraun/regex-thompson-counting/flamegraph/aws_prefix_group/bench_aws_prefix_tier1/callgrind.bench_aws_prefix_tier1.out
```

This prints:
1. A summary table of top functions by instruction count
2. Auto-annotated source for each hot function (line-by-line Ir counts)

### Filter to a specific source file

Pipe through grep to isolate a file:

```bash
callgrind_annotate --auto=yes --inclusive=no --threshold=99 \
  target/gungraun/regex-thompson-counting/flamegraph/aws_prefix_group/bench_aws_prefix_tier1/callgrind.bench_aws_prefix_tier1.out \
  2>&1 | grep -A 200 'Auto-annotated source: src/dfa/tier1.rs'
```

Replace the path suffix to target `src/lib.rs`, `src/dfa/tier3.rs`, etc.

### Find the hot DFA driver loop

```bash
callgrind_annotate --auto=yes --inclusive=no --threshold=99 \
  <callgrind.out> 2>&1 | grep -B2 -A 20 'fn step\|fn chunk\|fn transition'
```

## 3. Rebar Benchmarks

Rebar is in `bench/rebar/`. The engine wrapper is `bench/rebar/engines/rethoc/`.

### Rebuild the rebar engine (required after code changes)

```bash
cd bench/rebar/engines/rethoc && cargo build --release
```

### Run rebar benchmarks

```bash
cd bench/rebar && ./target/release/rebar measure -f '<filter>' -e '^(rethoc|rust/regex)$'
```

Replace `<filter>` with a benchmark name pattern (e.g. `aws`, `sqli`).

## 4. CLI Diagnostics (`rethoc`)

### Pattern info (NFA states, byte classes, memory, execution tier)

```bash
cargo run --release -- info '<pattern>'
```

Example:
```bash
cargo run --release -- info '(?i)\bselect\b'
```

### Match testing

```bash
cargo run --release -- match '<pattern>' 'input1' 'input2' ...
```

Note: `match` takes literal string arguments, NOT file paths.

### Debug mode (step-by-step trace)

```bash
cargo run --release -- match --debug '<pattern>' 'input'
```

### Graphviz DOT output

```bash
cargo run --release -- dot '<pattern>'
```

## 5. Standard Checks

```bash
cargo test                          # all tests (currently 255)
cargo clippy -- -D clippy::all      # strict clippy
cargo fmt                           # format
```

## 6. Key Callgrind Output Paths

| Benchmark | Callgrind output |
|---|---|
| aws_prefix_tier1 | `target/gungraun/regex-thompson-counting/flamegraph/aws_prefix_group/bench_aws_prefix_tier1/callgrind.bench_aws_prefix_tier1.out` |
| aws_keys_quick | `target/gungraun/regex-thompson-counting/flamegraph/aws_keys_group/bench_aws_keys_quick/callgrind.bench_aws_keys_quick.out` |
| grep_every_line | `target/gungraun/regex-thompson-counting/flamegraph/grep_group/bench_grep_every_line/callgrind.bench_grep_every_line.out` |

Flamegraph SVGs are next to the `.out` files (same directory, `.svg` extension).
