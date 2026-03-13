# Bug 51: break_deferred_asserts OR semantics (AND interning)

## Bug Summary

- **Pattern**: `^c{2,49}((a?\B)?\b)*\b$`
- **Input**: `"ccccccccccccccc"` (15 c's)
- **Expected**: `true` (NFA and regex crate both match)
- **Actual**: `false` (Tier 3 false negative)
- **Affected tier**: Tier 3 (both range-compressed and per-instance paths)
- **Fuzz artifact**: `fuzz/artifacts/fuzz_match/minimized-from-3a09a1c3e8440fb13a76633049f917d96ba4647d`

## Root Cause

The `break_deferred_asserts` field on `Tier3OriginKind::Increment` is a flat
list of NFA assertion state indices that gate the break path's match.  These
assertions come from **different NFA epsilon paths** leading to the Match
state — they represent independent entry points with **OR semantics** (any
one passing means the match fires).

For this pattern, the counter `c{2,49}` has a break path that goes through
the `((a?\B)?\b)*` group.  The NFA epsilon closure from the break point
finds two assertion paths to Match:
1. `\B` (non-word boundary) — reachable via the `a?\B` branch
2. `\b` (word boundary) — reachable via the outer `\b` at the end

These are **contradictory** assertions — `\B` and `\b` cannot both pass at
the same byte position.  But that's fine: only one needs to pass.

**The bug**: All assertion state indices in `break_deferred_asserts` were
interned as a **single assertion chain** via `arena.intern(break_deferred_asserts)`,
producing one `AssertChainId` shared across all entries.  The chain evaluation
function `eval_assert_chain` walks the chain and requires **all** assertions
to pass (AND semantics).  For contradictory assertions like `\B` and `\b`,
this is impossible — the chain always fails, producing a false negative.

The bug manifested in three code sites:
1. **Effect compiler** (`tier3_effects.rs`): `arena.intern(break_deferred_asserts)`
   created a single combined chain instead of per-entry chains.
2. **Deposit site in `step_slow` macro** (two locations): Used the single
   `break_assert_chain_id` to create one `PendingEffect`.
3. **Deposit site in `chunk()` tail resolution**: Same single-chain usage.

## Investigation Narrative

The bug was found by `fuzz_match` (oracle differential against regex crate).
The minimized artifact decoded to:
- Pattern: `^c{2,49}((a?\B)?\b)*\b$`
- Input: `"ccccccccccccccc"` (15 c's)

Reproduction confirmed the divergence:
```
cargo run --release -- match --tier 3 --unroll-limit 0 '^c{2,49}((a?\B)?\b)*\b$' 'ccccccccccccccc'
# → NO MATCH (incorrect)

cargo run --release -- match --tier 0 '^c{2,49}((a?\B)?\b)*\b$' 'ccccccccccccccc'
# → MATCH (correct)
```

The `--debug` trace showed that `break_deferred_asserts` contained two entries
(`\B` at state X, `\b` at state Y`) but they were interned as a single chain.
At runtime, the `PendingEffect` was deposited with this combined chain, and
`eval_assert_chain` walked both assertions sequentially — `\B` fails when
`\b` would pass, and vice versa, so the chain always fails.

The fix was immediately clear: each entry in `break_deferred_asserts` represents
an independent path and must be interned as a separate 1-element chain, with
one `PendingEffect` emitted per chain (OR semantics via multiple effects).

## Fix

**Structural change**: Replaced the single `break_assert_chain_id: AssertChainId`
field with `break_deferred_chain_ids: Box<[AssertChainId]>`, a parallel array
to `break_deferred_asserts` where each entry is a 1-element chain interned
independently.

**Sites fixed**:
1. `compute_tier3_analysis` post-pass: intern each assertion as a separate
   1-element chain → populate `break_deferred_chain_ids`.
2. `step_slow` macro, deposit site 1 (post-break-tail CInc handoff): loop
   over `break_deferred_chain_ids` instead of single chain.
3. `step_slow` macro, deposit site 2 (counter entry break): same loop fix.
4. `chunk()` tail resolution, deposit site 3: same loop fix.
5. Effect compiler (`tier3_effects.rs`): intern per-entry chains in a loop
   instead of interning the whole list as one chain.
6. `dump.rs`: updated display to show per-entry chain IDs.

## What Was Hard

The conceptual difficulty was recognizing that a flat list of assertions
has **OR semantics** (any one passing suffices) while the assertion chain
interning mechanism provides **AND semantics** (all must pass).  The
`break_deferred_asserts` field name doesn't signal whether the entries are
independent (OR) or conjunctive (AND).

The bug was also present in three separate deposit sites in the runtime
matcher plus the effect compiler — four code locations that all needed the
same fix.

## Tooling Ideas

1. **Assertion chain semantics annotation**: The `AssertChainId` type could
   carry a discriminant (AND vs OR) to make the semantics explicit and
   catch misuse at compile time.
2. **Dump output for assertion chains**: The `--dfa` dump could show the
   full assertion chain contents (not just the ID) to make it easier to
   spot combined chains that should be separate.
3. **Debug trace for eval_assert_chain**: Log each assertion evaluation
   (pass/fail) within a chain, making it visible when contradictory
   assertions cause unconditional failure.
