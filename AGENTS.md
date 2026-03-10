# AGENTS.md — Coding Agent Reference

Thompson NFA regex engine with multi-tier counting DFA acceleration.
Rust 2024 edition. Binary: `rethoc`. All source under `src/`.

## Build / Test / Lint

```bash
cargo build                         # debug build
cargo build --release               # release build (needed for benchmarks + bless_memory.py)
cargo test                          # run all 348 tests
cargo test test_counting            # run a single test by exact name
cargo test test_word_boundary       # run tests matching a substring
cargo test -- --nocapture           # show stdout/stderr from tests
cargo test -- --list                # list all test names
cargo clippy -- -D clippy::all      # strict clippy (treat all warnings as errors)
cargo fmt                           # format (default rustfmt, no custom config)
cargo fmt -- --check                # check formatting without modifying
```

After changing the NFA compiler or `State` layout, update memory assertions:
```bash
python3 scripts/bless_memory.py             # update all stale memory values in tests
python3 scripts/bless_memory.py --check     # CI-friendly: exits 1 if any stale (no writes)
python3 scripts/bless_memory.py --dry-run   # show what would change
```
Requires `target/release/rethoc` (auto-builds if missing).
Does NOT touch `min_tier:` values — those are left alone to catch tier regressions.

## Project Layout

```
src/lib.rs          # Core NFA compiler, matcher, counter pool, ALL tests (~9200 lines)
src/main.rs         # CLI binary `rethoc` (info, match, dot subcommands)
src/info.rs         # Diagnostic/serialization types (RegexInfo, MemoryInfo, etc.)
src/memrange.rs     # SIMD byte-range prefilter (SSE2/AVX2/NEON + scalar fallback)
src/dfa/mod.rs      # Shared DFA infrastructure (DfaMemory, DfaState, epsilon_closure)
src/dfa/tier1.rs    # Tier 1: Lazy DFA, counter-free patterns
src/dfa/tier2.rs    # Tier 2: Differential counters, fixed-length bodies (Becchi-style)
src/dfa/tier3.rs    # Tier 3: Conditional transitions, non-nested variable-length bodies
src/dfa/tier4.rs    # Tier 4: Counter programs, nested repetitions
benches/flamegraph.rs  # Callgrind benchmarks (gungraun harness)
scripts/bless_memory.py  # Update memory assertions in match_tests!
```

All tests are inline in `src/lib.rs` (`#[cfg(test)] mod tests`) and `src/memrange.rs`.
The `tests/` directory is empty — there are no integration tests.

## Architecture

Tiers are selected at compile time based on pattern analysis:
- **Tier 0 (NFA)**: Pure Thompson simulation. Fallback for CRLF assertions or zero-width counter bodies.
- **Tier 1**: Standard lazy DFA. Counter-free patterns. Supports deferred assertions (`\b`, `\B`, `EndLF`).
- **Tier 2**: Becchi-style differential counters. Non-nested, fixed-length-body repetitions. O(1) per byte.
- **Tier 3**: Conditional transitions. Non-nested counters with variable-length bodies.
- **Tier 4**: Compiled counter programs. Nested repetitions. Most general DFA tier.

## Code Style

### Imports

Three groups separated by blank lines, in this order:
1. Standard library (`use std::...`)
2. Third-party crates (`regex_syntax`, `ahash`, `indexmap`, `serde`, `memchr`)
3. Crate-internal (`mod`, `use crate::`, `use super::`, `pub use`)

No wildcard imports in production code. `use super::*` is acceptable only in test modules.
DFA submodules use `use crate::` for lib.rs items and `use super::` for `dfa/mod.rs` items.

### Naming Conventions

| Category | Convention | Examples |
|----------|-----------|----------|
| Types | `CamelCase` | `ByteClass`, `DfaState`, `CounterPool` |
| Index newtypes | `*Idx` suffix | `StateIdx(u32)`, `ClassIdx(usize)`, `CounterIdx(u8)` |
| Diagnostic types | `*Info` suffix | `RegexInfo`, `MemoryInfo`, `CounterInfo` |
| Cache/Matcher pairs | `*Cache` / `*Matcher` | `DfaCache`/`DfaMatcher`, `Tier2DfaCache`/`Tier2DfaMatcher` |
| Functions/methods | `snake_case` | `epsilon_closure`, `intern_class`, `hir2postfix` |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_COUNTERS`, `DFA_MAX_STATES`, `SENTINEL` |
| Struct fields | `snake_case` | Keyword-colliding fields get trailing `_`: `byte_`, `match_` (with `#[serde(rename)]`) |
| Tests | `test_` prefix + `snake_case` | `test_counter_ctx_empty`, `test_nested_counting` |

### Index Newtypes

Every index into an internal array is a newtype wrapping a primitive with an `.idx() -> usize` method
and sentinel constants (`const NONE`, `const DEAD`, `const UNPOPULATED`):
```rust
pub(crate) struct StateIdx(pub(crate) u32);  // NONE = u32::MAX
pub(crate) struct DfaStateId(u32);           // DEAD = u32::MAX, UNPOPULATED = u32::MAX - 1
```

### Visibility

- `pub` — only at the crate API boundary (`Regex`, `RegexBuilder`, `Error`, re-exported info types)
- `pub(crate)` — internal types shared across modules
- `pub(super)` — DFA-internal types shared between tier files
- Private by default for everything else

### Error Handling

- Custom `Error` enum at crate root (4 variants), manually implements `Display` and `std::error::Error`.
- Use `?` operator in compilation paths (`hir2postfix`, `RegexBuilder::build`).
- CLI: `unwrap_or_else(|e| { eprintln!(...); process::exit(1); })` — never bare `unwrap()` in user-facing code.
- Tests: prefer `.expect("descriptive message")` over `.unwrap()`.
- `debug_assert!()` liberally for internal invariants (zero-cost in release).
- `panic!()` only for truly impossible states (exactly once in codebase, in `patch()`).

### Documentation

- Module-level `//!` doc comments on all modules with architectural overviews.
- `///` doc comments on all public and `pub(crate)` types, methods, fields, and enum variants.
- Markdown formatting with backtick code, `[TypeName]` cross-references, and ASCII diagrams.
- Section separators in long files:
  ```rust
  // ---------------------------------------------------------------------------
  // Section Name
  // ---------------------------------------------------------------------------
  ```

### Lint Attributes

Minimal. No crate-level lint configuration. No `clippy.toml` or `rustfmt.toml`.
Only these targeted suppressions exist:
- `#[allow(dead_code)]` on a few unused-but-intentional public API methods
- `#[allow(clippy::too_many_arguments)]` on `epsilon_closure()` (8 params including closures)
- `#[allow(clippy::type_complexity)]` on one local binding with a complex tuple type

### Unsafe Code

Exclusively in `src/memrange.rs`, behind `#[cfg(target_arch = ...)]` gates. Every `unsafe` call
site has a `// SAFETY:` comment. A scalar fallback is always available. Zero unsafe elsewhere.

### Tests

Data-driven tests use the `match_tests!` macro in `src/lib.rs` (line ~4616). Each entry specifies:
- `pattern`: regex pattern string
- `memory`: expected compiled memory footprint in bytes
- `min_tier`: minimum DFA tier the pattern should qualify for
- `unroll_limit`: optional override (defaults to `DEFAULT_MAX_UNROLL_STATES`)
- `inputs`: list of `("input", expected_bool)` pairs

The macro generates one `#[test]` per entry that compiles the pattern, asserts tier and memory,
compares against the `regex` crate as oracle, and exercises ALL eligible tiers (NFA through Tier 4).
It also re-runs with unrolling disabled to force counter-based execution paths.

### Code Organization

- Struct definition immediately followed by `impl` block (no blank line between them).
- Trait impls (`Display`, `Index`, `Deref`) grouped after the inherent `impl`.
- Tests at the very bottom of each file in `#[cfg(test)] mod tests`.

## CLI Quick Reference

```bash
cargo run --release -- info '<pattern>'                 # NFA states, memory, tier, counters
cargo run --release -- --format json info '<pattern>'   # JSON output (for scripting)
cargo run --release -- match '<pattern>' 'input' ...    # match testing (literal strings, not files)
cargo run --release -- match --debug '<pattern>' 'input' # step-by-step NFA trace
cargo run --release -- dot '<pattern>'                  # Graphviz DOT output
cargo run --release -- --tier 2 match '<pattern>' 'input' # force a specific tier
```

## Benchmarks

```bash
cargo bench --bench flamegraph          # run all callgrind benchmarks (gungraun)
cargo bench --bench flamegraph -- --list  # list available benchmarks
```

Callgrind output: `target/gungraun/regex-thompson-counting/flamegraph/<group>/<bench>/callgrind.<bench>.out`

Criterion wall-clock benchmarks (SIMD `memrange` throughput):

```bash
cargo bench --bench memrange            # run all memrange benchmarks
cargo bench --bench memrange -- "no_match"  # filter by name substring
```

Compares NEON (aarch64) / SSE2+AVX2 (x86_64) against the scalar fallback.
Reports throughput in GiB/s.  HTML reports in `target/criterion/`.

Rebar (comparative benchmarks against `rust/regex`):

All rebar commands run from `bench/rebar/`.  The rebar binary is at
`bench/rebar/target/release/rebar`.  The rethoc engine source is at
`bench/rebar/engines/rethoc/` (depends on the root crate via path).

rethoc supports rebar models `compile`, `count`, and `grep`.
It does **not** support `count-spans`, `count-captures`, or `grep-captures`
(no match position / capture reporting), so benchmarks using those models
will error for rethoc.

```bash
# Build both engines (from bench/rebar/)
./target/release/rebar build -e '^(rethoc|rust/regex)$'

# Run a specific benchmark and compare
./target/release/rebar measure -f '<filter>' -e '^(rethoc|rust/regex)$' | tee /tmp/results.csv
./target/release/rebar cmp /tmp/results.csv

# Example filters:
#   '^curated/09-aws-keys/quick$'       single benchmark
#   '^curated/09-aws-keys/'             all aws-keys variants
#   '.'                                 everything (slow)

# Sanity-check (verify correctness without timing)
./target/release/rebar measure -f '<filter>' -e '^(rethoc|rust/regex)$' --test
```
