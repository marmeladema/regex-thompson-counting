# Bug 43 — Tier 3 false positive: pending break tails promoted despite chained contradictory assertions

## Bug summary

- **Pattern:** `^.{2,26}\b\B(a?)?$`
- **Input:** `"cc"` (or any input of length 2–26)
- **Expected:** `false` (NFA oracle) — `\b\B` is a contradiction, can never match
- **Actual (Tier 3):** `true` (false positive)
- **Affected tier:** Tier 3 (per-instance path)

## Root cause

This is a follow-on to Bug 42.  Bug 42 introduced `pending_break_tails`
and `pending_break_mae` to defer tail deposit until deferred assertions
are resolved.  The resolution function `any_deferred_assert_passes` only
checked whether the **first** assertion in the chain passed (the entry
point recorded in `break_deferred_asserts`), without walking the
downstream epsilon transitions to evaluate subsequent assertions.

The Tier 3 analysis (`break_closure`) records only the first assertion
on a break path as the entry point in `break_deferred_asserts`.  For
the pattern `\b\B$`, the break path is:

    CInc break → Assert(\b, state 4) → Assert(\B, state 5) → Split → ... → Match

Only state 4 (`\b`) is in `break_deferred_asserts`.  State 5 (`\B`) is
expected to be evaluated dynamically when the entry point is resolved.
`resolve_verified_deferred_asserts` handles this correctly by calling
`can_reach_match_at_end` (or `can_reach_match_mid`), which walks epsilon
transitions from the entry assertion's `out` and evaluates all
intermediate assertions.

But `any_deferred_assert_passes` — the function used by Bug 42's pending
tail resolution — only evaluated `kind.eval(...)` on the entry assertion
without walking downstream.  For `\b` at EOI with `prev='c'` (word
char), `\b` passes (word→EOI = boundary).  The function returned `true`
and the pending tails were promoted, even though `\B` (at the next state)
would fail at the same position.

## Fix

Replaced `any_deferred_assert_passes` with `resolve_deferred_for_pending`,
which performs a full epsilon walk from the entry assertion's `out` state,
evaluating all intermediate assertions.  Returns `true` only if some
consuming state (Byte, ByteClass, etc.) or Match is reachable through the
full assertion chain.  This correctly blocks promotion when `\b` passes
but `\B` on the downstream path fails.

## Investigation narrative

1. **Fuzz discovery:** Continuing `fuzz_differential` after fixing Bug 42
   immediately found a new crash: `^.{2,26}\b\B((a?a?)?(a?a?)?)?...` on
   `"ccdac0d"` — Tier 3 true, NFA false.

2. **Pattern analysis:** Noticed `\b\B` — an inherently contradictory
   assertion pair.  No position can be both a word boundary and a
   non-boundary.  This should NEVER match.

3. **Minimization:** Simplified to `^.{2,26}\b\B(a?)?$` which still
   exhibited the false positive on any input of length 2–26.

4. **Dump inspection:** `dump --dfa` showed `break_deferred_asserts: [4]`
   — only `\b` recorded, not `\B`.  The comment in `break_closure`
   explains this is by design: "Subsequent assertions deeper in the chain
   are evaluated dynamically."

5. **Trace with pending_tails display:** Added `pending_tails: [6]` to
   the Display impl.  After the counter break, pending tails were
   correctly deferred (Bug 42 working).  But in `finish()`, the
   `any_deferred_assert_passes` function only checked `\b` (which passes
   at EOI) and returned `true`, causing the pending tails to be promoted.

6. **Root cause:** `any_deferred_assert_passes` evaluated `kind.eval()`
   on the entry assertion but did not walk epsilon transitions to check
   `\B`.  Fixed by replacing it with a full epsilon walk.

## What was hard

- The bug was introduced by the Bug 42 fix itself.  The
  `any_deferred_assert_passes` function was too permissive — it treated
  the entry assertion as sufficient for promotion, not realizing that the
  design relies on dynamic evaluation of the full chain.

- Understanding the `break_closure` design intent (only record the first
  assertion, handle deeper ones dynamically) was necessary to see why
  `break_deferred_asserts` only had one entry for `\b\B`.

## Tooling ideas

- The `break_closure` function could include a note in the dump output
  about chained assertions (e.g. "deferred chain: \b → \B → ... →
  consuming states") to make the design intent visible during debugging.
