# Bug 27: Tier 3 break seeds applied unconditionally through deferred assertions

## Bug summary

- **Pattern:** `^.{0,13}\b.{2,2}$`
- **Input:** `abc` (3 chars, all word characters)
- **Expected:** NO MATCH (`\b` between word chars never passes)
- **Actual:** MATCH (Tier 3 false positive)
- **Affected tier:** Tier 3 (per-instance path), when break seeds traverse a
  deferred assertion (e.g. `\b`, `\B`) on the path from trigger counter's break
  to the seeded counter's CI

## Root cause

The pattern has c0 (`.{0,13}`) and c1 (`.{2,2}`), with `\b` (state 5,
`Assert(WordAscii)`) between c0's break output and c1's CI.  The precomputed
break seed table contained `trigger:c0 → seed c1 at origin:6` — but the `\b`
on the path was ignored.

In `compute_tier3_analysis`, the break seed walk (Phase 1, lines 326-358)
followed Assert states unconditionally:
```rust
State::Assert { out, .. } => ci_stack.push(out),
```

This created a break seed that would unconditionally seed c1 whenever c0 broke.
At runtime, when c0 broke on byte 'b' (at position 1 of "abc"), the break seed
seeded c1 at origin 6 without checking whether `\b` passes at that position
(it doesn't: 'a' and 'b' are both word chars).

With the spurious seed, c1 accumulated count=2 (matching `.{2,2}`) and then
broke with `break_is_match_at_end=true` (the `$` assertion).  This set
`match_at_end`, and `finish()` returned true.

## Fix

1. **Extended `Tier3BreakSeed`** with a `deferred_asserts: Box<[StateIdx]>`
   field.  During break seed computation (Phase 1 and Phase 2), the walk now
   tracks non-End Assert state indices encountered on the path from the
   trigger's break output to the seeded counter's CI.  End asserts are excluded
   (they're handled separately by `break_is_match_at_end`).

2. **Runtime gating in `step_slow_impl!`**: when applying break seeds, the code
   looks up the corresponding analysis-level break seed and evaluates its
   deferred asserts with the current byte context (`at_end=false`,
   `prev=state.prev_byte_representative()`, `next=Some(byte)`).  If any assert
   fails, the seed is skipped.

3. **Passed `byte` parameter to `step_slow_impl!`** methods so the runtime
   assert evaluation has access to the current byte.

## Investigation narrative

1. **Fuzz discovery.** `cargo +nightly fuzz run fuzz_differential` found crash
   artifact `crash-27943dd493b5370e5c5560848e7ccaa5750d5fa9`.  Pattern:
   `^.{0,13}\b.{2,2}$`.

2. **Brute-force input testing.** Tested various 3-4 char inputs. Found that
   `abc`, `abcd`, and `ab ` all showed NFA=NO MATCH, Tier 3=MATCH.  Inputs
   with a word boundary (e.g. ` ab`, `a b`) correctly matched on both.

3. **Debug trace.** After chunk #1 ('b'), `mae=true` appeared.  After chunk #2
   ('c'), `mae=true` persisted.  The `match_at_end` flag was incorrectly set.

4. **NFA dump analysis.** `dump --dfa --unroll-limit 0` showed:
   - c1's `break_is_match_at_end: true` (correct: `$ → Match`)
   - `break_seeds: trigger:c0 → seed c1 at origin:6` (missing `\b` gate)
   - State 5: `Assert(WordAscii)` sits between c0's break and c1's CI

5. **Identified the gap.** The break seed walk in `compute_tier3_analysis`
   followed Assert states without recording them.  The seed was applied
   unconditionally at runtime.

6. **Fix approach.** Extended `Tier3BreakSeed` to carry deferred assert state
   indices.  Modified both Phase 1 and Phase 2 walks to track Assert states on
   the path.  At runtime, evaluate the asserts before applying the seed.

## What was hard

- **Initially thought this was a Bug 26 regression.** The pattern was discovered
  from the same fuzz run as Bug 26.  First instinct was that the mid-input
  deferred assert resolution from Bug 26 caused the false positive.  Testing
  against the Bug 25 commit showed the bug was pre-existing.

- **Break seeds are a separate mechanism from break_deferred_asserts.** The
  Tier 3 architecture has multiple break-related mechanisms: `break_is_match`,
  `break_is_match_at_end`, `break_deferred_asserts`, `break_consuming_states`,
  and `break_seeds`.  The first four handle the CInc's own break path.
  `break_seeds` handle downstream counter seeding.  The `break_closure`
  function correctly tracks deferred asserts for the CInc's own break path,
  but the break_seed computation was a separate walk with no assert tracking.

## Tooling ideas

- **Show break seed deferred asserts in `dump --dfa` output.** Currently the
  dump shows `break_seeds: trigger:c0 → seed c1 at origin:6` but doesn't
  indicate gating assertions.  Adding `(gated by \b@5)` would make the
  missing gate immediately visible.
