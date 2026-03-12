# Bug 31: Tier 3 chained target_deferred_asserts false positive

## Bug Summary

- **Pattern:** `^.{7,7}(.\b\Ba?(a?a?)?)?$`
- **Input:** `cy1aacc ` (8 bytes, trailing space)
- **Expected:** NO MATCH (`\b\B` is always contradictory)
- **Actual (before fix):** MATCH (Tier 3 false positive)
- **Affected tier:** Tier 3 (per-instance path, only with `--unroll-limit 0`)
- **Found by:** `fuzz_differential` during fuzzing after Bug 30 fix
- **Related:** Regression from Bug 30 fix (`target_deferred_asserts`)

## Root Cause

The Bug 30 fix introduced `target_deferred_asserts`: a precomputed array
recording non-End Assert states on the epsilon path from each consuming
state's target.  When a post-break tail advances through a consuming state,
these asserts are deposited into `verified_deferred_asserts` for evaluation.

The computation walked the epsilon path from the target and **followed
through** Assert states, recording every non-End Assert encountered.  For
state 4 (`ByteClass`, the `.` in `(.\b\Ba?...)`), the target epsilon path is:

```
state 5: Assert(WordAscii)     → state 6
state 6: Assert(WordAsciiNegate) → state 8
state 8: Split → 7 | 13
```

Both `\b@5` and `\B@6` were recorded in `target_deferred_asserts[4]`.

At runtime, `resolve_verified_deferred_asserts` evaluated each assert
**independently**:
- `\b@5`: prev=space(non-word), next=None(non-word) → FAILS
- `\B@6`: prev=space(non-word), next=None(non-word) → PASSES

`\B@6` passes, and `can_reach_match_at_end(state 8, ...)` reaches
`Assert(End) → Match`.  Result: false positive.

But in the NFA, `\B@6` is only reachable if `\b@5` passes first.  The
two assertions are **chained** (sequential on the same path), not
independent alternatives.  `\b\B` is always contradictory — it can never
pass.

## Fix

Changed the `target_deferred_asserts` computation (Step 6c in
`compute_tier3_analysis`) to **not follow through** non-End Assert states.
The walk records the first Assert on each path but does not push the
Assert's `out` onto the stack.

This means `target_deferred_asserts[4] = [\b@5]` (only the first assert).
At runtime:
- `resolve_verified_deferred_asserts` evaluates `\b@5` → fails → done
- If `\b@5` had passed, `can_reach_match_at_end(state 6, ...)` would
  properly evaluate `\B@6` as a downstream assertion in the epsilon walk,
  preserving the chain semantics.

This approach is correct because `can_reach_match_at_end` (and
`can_reach_match_mid`) already handle chained assertions: they walk the
epsilon path from the assert's `out`, evaluating each subsequent assertion
and only following `out` if it passes.

## Investigation Narrative

1. Fuzz crash within minutes of resuming after Bug 30 commit.
2. Decoded: pattern `^.{7,7}(.\b\Ba?(a?a?)?)?$`, input `cy1aacc `.
3. NFA=NO MATCH, Tier 3=MATCH. Tier 2 correct (NO MATCH).
4. `--debug` trace showed `deferred: [\b@5, \B@6]` after the tail at
   state 4 advanced.  Both were deposited by the Bug 30
   `target_deferred_asserts` mechanism.
5. Recognized that `\b\B` is contradictory but `\B` was being evaluated
   independently.
6. Root cause: the Step 6c epsilon walk followed through Assert states,
   recording every assert as if on independent paths.
7. Fix: stop the walk at non-End Assert states.  The downstream chain
   is handled by `can_reach_match_at_end` at runtime.

## What Was Hard

Nothing — the bug was a direct consequence of an incorrect assumption in
the Bug 30 fix.  Once I saw `\b@5, \B@6` both deposited independently,
the fix was obvious: don't follow through Assert states in the walk.

## Tooling Ideas

None new.  The existing `--debug` output showing `deferred: [\b@5, \B@6]`
immediately pointed to the problem.
