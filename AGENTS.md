# AGENTS.md — Coding Agent Reference

Thompson NFA regex engine with multi-tier counting DFA acceleration.
Rust 2024 edition. Binary: `rethoc`. All source under `src/`.

## Key Engine Properties

These properties are fundamental invariants that must be considered in
**every** bug fix, feature addition, and code change:

1. **Streaming.** The engine processes input via `chunk(&[u8])` calls —
   callers may split input across arbitrarily many chunks.  All matcher
   state (DFA state, counter contexts, deferred assertions, match flags)
   must survive across chunk boundaries.  Never assume the full input is
   available in a single slice.  Test multi-chunk scenarios when adding
   or modifying matcher logic.

2. **Untrusted patterns.** The regex pattern is attacker-controlled.
   Compilation must not panic, hang, or use unbounded memory.  All
   compile-time limits (`MAX_COUNTERS`, `max_repetition`, `DFA_MAX_STATES`,
   etc.) exist to bound resource usage.  New pattern features must have
   corresponding limits.  The `fuzz_compile` target exercises this.

3. **Untrusted input.** The input payload is attacker-controlled.
   Matching must run in time proportional to the input length (no
   backtracking, no exponential blowup).  Per-byte work must be bounded
   by the pattern's compiled structure, not by input content.  This is
   the core guarantee of the Thompson NFA approach.  The `fuzz_match`
   and `fuzz_differential` targets exercise this.

## Build / Test / Lint

```bash
cargo build                         # debug build
cargo build --release               # release build (needed for benchmarks + bless_memory.py)
cargo test                          # run all ~382 tests (excludes ignored fuzz tests)
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
src/lib.rs          # Core NFA compiler, matcher, counter pool, ALL tests (~9800 lines)
src/main.rs         # CLI binary `rethoc` (info, match, dot, dump subcommands)
src/dump.rs         # Human-readable dump (DumpRegex, DumpState Display wrappers)
src/info.rs         # Diagnostic/serialization types (RegexInfo, MemoryInfo, etc.)
src/memrange.rs     # SIMD byte-range prefilter (SSE2/AVX2/NEON + scalar fallback)
src/fuzz_gen.rs     # Grammar-aware pattern + input generator for fuzzing
src/dfa/mod.rs      # Shared DFA infrastructure (DfaMemory, DfaState, epsilon_closure)
src/dfa/tier1.rs    # Tier 1: Lazy DFA, counter-free patterns
src/dfa/tier2.rs    # Tier 2: Differential counters, fixed-length bodies (Becchi-style)
src/dfa/tier3.rs    # Tier 3: Conditional transitions, non-nested variable-length bodies
src/dfa/tier4.rs    # Tier 4: Counter programs, nested repetitions
fuzz/Cargo.toml                    # cargo-fuzz workspace configuration
fuzz/fuzz_targets/fuzz_match.rs    # Fuzz target: oracle differential (rethoc vs regex crate)
fuzz/fuzz_targets/fuzz_differential.rs  # Fuzz target: cross-tier differential (NFA as oracle)
fuzz/fuzz_targets/fuzz_compile.rs  # Fuzz target: compilation robustness
benches/flamegraph.rs      # Callgrind benchmarks (gungraun harness)
benches/pathological.rs    # Criterion benchmarks: pathological bounded repetitions
benches/pathological_profile.rs  # Callgrind profile for pathological patterns
scripts/bless_memory.py    # Update memory assertions in match_tests!
```

All tests are inline in `src/lib.rs` (`#[cfg(test)] mod tests`) and `src/memrange.rs`.
The `tests/` directory is empty — there are no integration tests.

## Architecture

Tiers are selected at compile time based on pattern analysis:
- **Tier 0 (NFA)**: Pure Thompson simulation. Fallback for CRLF assertions or zero-width counter bodies.
- **Tier 1**: Standard lazy DFA. Counter-free patterns. Supports deferred assertions (`\b`, `\B`, `EndLF`).
- **Tier 2**: Becchi-style differential counters. Non-nested, fixed-length-body repetitions. O(1) per byte.
- **Tier 3**: Conditional transitions. Non-nested counters with variable-length
  bodies. Two sub-paths selected at compile time: **range-compressed**
  (O(num\_origins) per byte) when all `CounterInstance` nodes are
  epsilon-reachable from the start state, otherwise **per-instance** fallback
  (O(live\_instances) per byte). Patterns with `^`, consuming prefixes, or
  assertions blocking re-seeding use the per-instance path.
- **Tier 4**: Compiled counter programs. Nested repetitions. Most general DFA tier.

## Design Principles

1. **Precompute at build time.** Everything that can be computed or preprocessed
   from the NFA structure alone must be done in `RegexBuilder::build()` and
   stored on the `Regex` struct (e.g. `Tier2Analysis`, `Tier3Analysis`,
   `counter_info`, `byte_classes`).  Matcher hot paths should never recompute
   invariant data.

2. **Reuse memory across matches.** Cache and scratch structures (`Tier*DfaCache`,
   `DfaMemory`, `CounterPool`) are allocated once and reused for every match
   invocation.  `clear()` / `reset_for_new_match()` methods reset logical state
   without deallocating.  Avoid allocating in the per-byte matching loop.

3. **Prefer flat data structures during matching.** Nested collections like
   `Vec<Vec<_>>` or `HashMap<_, Vec<_>>` make memory reuse difficult (inner
   containers are separately heap-allocated and cannot be bulk-cleared).  Use
   flat arrays with index ranges (e.g. `body_interior: Box<[u32]>` +
   `body_ranges: Box<[(usize, usize)]>`) or arena-style pools instead.

## Code Style

### Imports

Three groups separated by blank lines, in this order:
1. Standard library (`use std::...`)
2. Third-party crates (`regex_syntax`, `ahash`, `hashbrown`, `indexmap`, `serde`, `memchr`)
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

**Key implication for debugging:** A test entry with `min_tier: 1` and a simple pattern like
`\b\w{3,5}\b` will be tested on Tier 1 with default unrolling, BUT ALSO on Tier 2/3 when the
macro re-runs with `unroll_limit=0` (which forces counter-based execution, promoting the pattern
to higher tiers). So a Tier 3 bug can cause failures in tests whose `min_tier` is 1. When you
see "Tier3 chunk mismatch" in a test failure, it's the `unroll_limit=0` re-run that failed.

**NEVER set `unroll_limit: 0` in test entries.** The macro ALREADY re-runs every
test with `unroll_limit=0` automatically. Setting it explicitly just skips the
default-unrolling run and redundantly tests `unroll_limit=0` twice. Use the
default (omit `unroll_limit` entirely) so both code paths are exercised. The
`memory` and `min_tier` values should reflect default unrolling.

### Code Organization

- Struct definition immediately followed by `impl` block (no blank line between them).
- Trait impls (`Display`, `Index`, `Deref`) grouped after the inherent `impl`.
- Tests at the very bottom of each file in `#[cfg(test)] mod tests`.

## CLI Quick Reference

```bash
cargo run --release -- info '<pattern>'                 # NFA states, memory, tier, counters
cargo run --release -- info --format json '<pattern>'   # JSON output (for scripting)
cargo run --release -- match '<pattern>' 'input' ...    # match testing (literal strings, not files)
cargo run --release -- match --debug '<pattern>' 'input' # step-by-step NFA trace
cargo run --release -- dot '<pattern>'                  # Graphviz DOT output
cargo run --release -- match --tier 2 '<pattern>' 'input' # force a specific tier
cargo run --release -- info --unroll-limit 0 '<pattern>'  # disable unrolling (force counters)
cargo run --release -- match --unroll-limit 0 '<pattern>' 'input' # match with counters only
cargo run --release -- dump '<pattern>'                 # NFA states, counters, byte classes, aux arrays
cargo run --release -- dump --dfa '<pattern>'           # + tier-specific DFA analysis (Tier 2/3)
cargo run --release -- dump --format debug '<pattern>'  # Rust {:#?} of the Regex struct
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

Criterion wall-clock benchmarks (pathological pattern, rethoc vs `regex` crate):

```bash
cargo bench --bench pathological            # run all pathological benchmarks
cargo bench --bench pathological -- "compile"   # compile time only
cargo bench --bench pathological -- "match_at_end"  # match-at-end only
```

Two pattern groups:

- **Non-nested**: `.{0,1000}.{0,1000}.{0,1000}a` — three sequential bounded
  repetitions with wildcard bodies.  Compares compilation time, no-match
  (prefilter-dominated), and match-at-end (actual simulation) across
  rethoc tiers (tier3/tier4/nfa) and the `regex` crate, sizes up to 64 KB.
- **Nested**: `(.{0,1000}a){0,1000}b` — nested bounded repetitions
  (Tier 4 only).  Sizes limited to 256 and 1024 bytes.

HTML reports in `target/criterion/`.

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

## Fuzzing

Two complementary approaches share a common grammar-aware pattern generator
(`src/fuzz_gen.rs`) and the `regex` crate as correctness oracle.

### Bug-Fix Workflow

When fuzzing discovers bugs, follow this discipline:

1. **One bug at a time.** Investigate, fix, and test one bug before moving
   to the next.
2. **Reproduce with `--debug`.** Use the CLI to trace matcher state
   byte-by-byte:
   ```bash
   # Default tier (may unroll counters away):
   cargo run --release -- match --debug --chunk-size 1 '<pattern>' '<input>'
   # Force counter-based execution (no unrolling) — essential when the
   # bug is in a counter tier:
   cargo run --release -- match --debug --chunk-size 1 --unroll-limit 0 '<pattern>' '<input>'
   # Force a specific tier (0=NFA, 1-4=DFA tiers):
   cargo run --release -- match --debug --chunk-size 1 --tier 3 '<pattern>' '<input>'
   # Combine: force tier 3 without unrolling:
   cargo run --release -- match --debug --chunk-size 1 --tier 3 --unroll-limit 0 '<pattern>' '<input>'
   ```
   The `[init]` line prints full `Debug` (struct fields, NFA states, cache
   details).  Each subsequent `[after chunk ...]` line prints compact
   `Display` output:  DFA state ID, NFA state set, match flags, and
   tier-specific counter/context summaries.  Tier 3 shows multi-line
   counter entries (origins, ranges) and post-break tails.  Compare the
   NFA oracle (`--tier 0`) against a DFA tier to pinpoint where they
   diverge.
3. **Commit each fix separately.** Each bug fix gets its own commit with a
   descriptive message.  Always ask for confirmation before committing.
   Include the regression test **and** the post-mortem in the same commit
   as the code fix — they are all part of the same logical change.
4. **Add a regression test** as an entry in the `match_tests!` macro (not a
   handwritten test function).  The macro automatically tests all eligible
   tiers and re-runs with `unroll_limit=0`.
5. **Run the full test suite** (`cargo test`) after each fix to ensure no
   regressions.
6. **Write a post-mortem** as part of the fix.  Create a Markdown file in
   `docs/bugs/` (e.g. `docs/bugs/001-tier3-range-merge.md`) documenting:
   - **Bug summary** — pattern, input, expected vs actual, affected tier(s).
   - **Root cause** — what went wrong and why.
   - **Investigation narrative** — how you found it: which CLI commands,
     which `--debug` output lines revealed the divergence, what hypotheses
     were tested and ruled out.
   - **What was hard** — what made the investigation slow or confusing
     (e.g. misleading output, missing information in traces, state that
     was difficult to inspect).
   - **Tooling ideas** — concrete suggestions for CLI improvements, new
     `--debug` output fields, dump enhancements, or new scripts that
     would have shortened the investigation.

   These post-mortems accumulate into a knowledge base that drives future
   tooling improvements and helps onboard new contributors to the
   debugging workflow.
7. **Commit infrastructure improvements separately** from bug fixes (e.g.
   fuzz target enhancements, new scripts).

### Pattern Generator (`src/fuzz_gen.rs`)

The generator maps a deterministic byte-seed into a structured AST, then
renders it to a regex string.  Controlled features:

| Feature | Range | Notes |
|---------|-------|-------|
| Atoms | Literals, `.`, byte classes `[a-c]` | From a fixed printable pool |
| Assertions | `\b`, `\B` | ~18% of atoms; zero-width, no repetition |
| Repetitions | `?`, `*`, `+`, `{n,m}` | Bounded max 50 |
| Nesting depth | 0–3 | Depth ≥2 exercises Tier 4 |
| Alternation | 2–4 branches | |
| Concatenation | 2–5 pieces | |
| Anchoring | `^...$`, `^...`, `...$`, unanchored | Random per pattern |

Input generation is **pattern-aware**: the AST is walked to produce positive
candidates with correct literals and valid repeat counts, then mutated to
create boundary-condition near-misses (truncated, extended, byte-flipped,
off-by-one on counters).  Fixed edge cases (empty, `\x00`, `\n`, `\xff`,
repeated chars at counter boundaries) are always included.

### proptest (property-based, in `cargo test`)

Two `#[ignore]`d tests in `src/lib.rs` that run 100 random patterns each:

```bash
cargo test test_fuzz -- --ignored              # run both fuzz tests (~60-120s)
cargo test test_fuzz_oracle -- --ignored       # oracle differential only
cargo test test_fuzz_differential -- --ignored # cross-tier differential only
```

- **`test_fuzz_oracle`** — generates a pattern + targeted inputs, compiles
  with both rethoc and the `regex` crate, asserts all tiers agree with the
  oracle.  Also re-runs with `unroll_limit=0` to exercise counter-based paths.
- **`test_fuzz_differential_tiers`** — same pattern + inputs, but uses the
  NFA as oracle instead of the regex crate.  Every eligible DFA tier must
  agree with the NFA.  Catches tier-specific bugs independently.

Both tests are `#[ignore]` so `cargo test` stays fast (~0.7s for the 380
data-driven tests).  Run them explicitly when changing the NFA compiler,
DFA tiers, or counter logic.

### cargo-fuzz (coverage-guided, for deep exploration)

Three libFuzzer targets in `fuzz/fuzz_targets/`:

```bash
cargo +nightly fuzz run fuzz_match              # oracle differential (rethoc vs regex crate)
cargo +nightly fuzz run fuzz_differential       # cross-tier differential (NFA as oracle)
cargo +nightly fuzz run fuzz_compile            # compilation robustness (no panics/hangs)

# Useful flags
cargo +nightly fuzz run fuzz_match -- -timeout=10   # per-input timeout (seconds)
cargo +nightly fuzz run fuzz_match -- -jobs=4       # parallel fuzzing (4 workers)
cargo +nightly fuzz run fuzz_match -- -runs=100000  # stop after N iterations

# Reproduce a crash artifact
cargo +nightly fuzz run fuzz_match fuzz/artifacts/fuzz_match/<artifact>
```

- **`fuzz_match`** — the primary target.  For each seed: generate pattern,
  compile with both engines, test many inputs against all eligible tiers
  (default unroll + no-unroll), assert agreement with the regex crate.
- **`fuzz_differential`** — same structure but NFA is the oracle.  Useful
  for tier-specific bugs without external dependencies.
- **`fuzz_compile`** — generates patterns and compiles them.  Must not
  panic, hang, or OOM.  Also exercises `.info()` and `.memory_size()`.

Corpus and artifacts are in `fuzz/corpus/` and `fuzz/artifacts/` (gitignored).

### Decoding Fuzz Seeds

The `fuzz_match` and `fuzz_differential` targets split the seed in half:
first half drives pattern generation, second half drives input generation.
Both `generate_pattern` and `generate_inputs` from `src/fuzz_gen.rs` are
`pub` functions.  To decode a crash artifact:

```rust
use regex_thompson_counting::fuzz_gen::{generate_pattern, generate_inputs, FuzzRng};
let data: &[u8] = &[/* seed bytes */];
let mid = data.len() / 2;
let (pattern, ast) = generate_pattern(&mut FuzzRng::new(&data[..mid]));
let inputs = generate_inputs(&mut FuzzRng::new(&data[mid..]), &ast);
```

### When to Fuzz

- After any change to the NFA compiler, DFA tiers, or counter logic.
- After fixing a bug found by fuzzing (to confirm the fix and find
  related bugs).
- Periodically during development to catch regressions early.

Start with `fuzz_match` (most coverage), then `fuzz_differential` (catches
tier-specific bugs), then `fuzz_compile` (robustness).
