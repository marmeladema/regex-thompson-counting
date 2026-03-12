# Bug 26: Tier 3 counter-break deferred asserts not resolved mid-input

## Bug summary

- **Pattern:** `.{2,38}c{4,10}\B`
- **Input:** `cccccca` (6 c's + a)
- **Expected:** MATCH (`\B` between last 'c' and 'a' passes — both are word chars)
- **Actual:** NO MATCH (Tier 3 false negative)
- **Affected tier:** Tier 3 (per-instance path), when a counter breaks with
  `break_deferred_asserts` and the deferred assert would pass mid-input

## Root cause

When counter c1 (`c{4,10}`) breaks with count >= 4, its break path includes a
deferred assert: state 6 (`Assert(WordAsciiNegate) → 7 (Match)`).  This assert
was stored in `verified_deferred_asserts` during `step_slow_impl!`.

However, `verified_deferred_asserts` was **only evaluated at end-of-input** in
`is_match_at_end_inner()`.  At end-of-input, `\B` uses `(at_end=true, prev,
next=None)`: prev='c' (word), next=end (non-word) → word boundary → `\B` fails.

The correct behavior: when the next byte arrives ('a'), the deferred assert
should be resolved with `(at_end=false, prev='c', next='a')`: both word chars →
non-boundary → `\B` passes → Match is reached.

The Tier 1 DFA handles this via `resolve_deferred()` which resolves DFA-state
deferred asserts at transition time (line 1396 in tier3.rs).  But counter-break
deferred asserts are runtime-only (produced per-instance when a counter breaks)
and are not stored in the DFA state.  They lived in `verified_deferred_asserts`
with no mid-input resolution path.

## Fix

Added mid-input resolution of `verified_deferred_asserts` at the start of each
byte in the `chunk()` loop, before the transition is computed:

```rust
if !self.verified_deferred_asserts.is_empty()
    && self.current != DfaStateId::DEAD
{
    let prev = state.prev_byte_representative();
    for &assert_idx in &self.verified_deferred_asserts {
        if kind.eval(false, false, prev, Some(b)) == AssertEval::Pass
            && Self::can_reach_match_mid(out, prev, Some(b), regex)
        {
            self.ever_matched = true;
            break;
        }
    }
}
```

Added `can_reach_match_mid()`: walks epsilon transitions from the assert's
`out` state, evaluating any intermediate asserts with mid-input context
(`at_end=false, next=Some(byte)`).  Returns true if `Match` is reachable with
all assertions passing.

**First fix attempt (failed):** Used `state_can_reach_match[out.idx()]` instead
of `can_reach_match_mid()`.  This caused 5 test failures because
`state_can_reach_match` is a static flag that doesn't evaluate runtime assert
conditions.  Patterns like `\B\b$` (contradictory asserts) or `\B$` (must be
at end) were incorrectly reported as matching because `state_can_reach_match`
saw a static path to `Match` through the intervening asserts.

## Investigation narrative

1. **Fuzz discovery.** `cargo +nightly fuzz run fuzz_differential` found
   crash artifact `crash-5fe5e5eb46489248e2c2baaf99e8ee3acdb8c0f9`.

2. **Decode seed.** Pattern: `.{2,38}c{4,10}\B`.  Reproduced with input
   `cccccca`: NFA=MATCH, Tier 3=NO MATCH.

3. **Byte-by-byte debug traces.** NFA trace showed deferred `\B` resolved at
   chunk #6 ('a') with `matched=true`.  Tier 3 trace showed `deferred: [6]`
   at chunk #5 but `matched=false` at chunk #6 — the deferred assert vanished
   without being evaluated.

4. **Found the gap.** `verified_deferred_asserts` was cleared at the start of
   `step_slow_impl!` (line 2392) before counter-break asserts from the
   *previous* step could be resolved with the current byte.

5. **Compared with Tier 1.** Tier 1's DFA state deferred asserts are resolved
   via `resolve_deferred(byte)` during transition computation.  Counter-break
   asserts have no equivalent path.

6. **First fix: `state_can_reach_match`.** Failed on 5 tests (`\B\b$`,
   `\B$`).  Static reachability doesn't evaluate intermediate asserts.

7. **Final fix: `can_reach_match_mid`.** Dynamic walk with mid-input assert
   evaluation.  All 441 tests pass.

## What was hard

- **Two resolution contexts.** Deferred asserts must be evaluated both
  mid-input (when the next byte arrives) AND at end-of-input (when no more
  bytes follow).  The existing `is_match_at_end_inner()` only handled the
  end-of-input case.  The mid-input case requires different `eval()` arguments
  (`at_end=false, next=Some(b)`).

- **Static vs dynamic reachability.** The first instinct was to use
  `state_can_reach_match` for the post-assert walk.  This is a static
  precomputation that ignores runtime assert conditions.  For patterns with
  chained or contradictory asserts (`\B\b$`, `\B$`), it produces false
  positives.  The fix required a dynamic walk (`can_reach_match_mid`) that
  evaluates each intermediate assert with the current byte context.

## Tooling ideas

- **Show deferred assert resolution in `--debug` Display output.** Currently
  the Display shows `deferred: [6]` but doesn't indicate whether or when the
  assert was resolved.  Adding `resolved_mid: [6@'a']` or similar would make
  the resolution visible.

- **Dedicated `resolve_counter_break_asserts()` method.** Factor the resolution
  logic into a method that's called both mid-input (from `chunk()`) and at
  end-of-input (from `finish()`), to avoid code duplication and ensure both
  paths stay in sync.
