# Bug 46: Per-tail deferred assertion gating

## Bug summary

- **Pattern**: `^.{2,10}\by?(\B ?(a?a?)?(a?a?)?)?$`
- **Input**: `"xa1  y "` (7 chars)
- **Expected**: `false` (NFA oracle)
- **Actual**: `true` (Tier 3 false positive)
- **Affected tier**: Tier 3 (conditional transitions)

## Root cause

The break path from the counter `.{2,10}` goes through `\b` (state 4)
first, then branches.  Some consuming tails — specifically the `Byte(' ')`
at state 8 — are behind BOTH `\b` AND `\B` (state 7) on the break path.

At compile time, `break_consuming_tails()` collected all consuming states
reachable from the break output, and `break_deferred_asserts` recorded
the top-level assertion entry points.  But the per-tail deferred assertion
information was lost: ALL pending break tails were promoted (or rejected)
as a group based on the single global `resolve_deferred_for_pending()`
check.

At runtime, when `\b` passes (e.g., at a word→non-word boundary), the
global check said "assertions pass" and promoted ALL pending tails —
including state 8 (`Byte(' ')`) which also needs `\B` to pass on its
specific path.  Since `\b` and `\B` are contradictory at the same
position, state 8 should never be promoted when `\b` passes.

This is a pre-existing bug, not introduced by Bug 45's changes (confirmed
by reverting Bug 45 and testing — same false positive).

## Investigation narrative

The crash artifact was discovered by `fuzz_differential`.  Reproducing
with `--debug --chunk-size 1 --tier 3 --unroll-limit 0` showed the Tier 3
matcher producing `ever_matched=true` while the NFA said `false`.

Comparing `--tier 0` (NFA) vs `--tier 3` traces, the divergence appeared
after the counter `.{2,10}` broke and the break path's deferred assertions
were resolved.  The `pending_break_tails` contained state 8 (`Byte(' ')`),
which was promoted when `\b` passed — but state 8 is behind both `\b` AND
`\B` in the NFA graph.

Examining `break_consuming_tails()` confirmed that it returned a flat list
of consuming states without tracking which assertions gated each
individual state.  The global `resolve_deferred_for_pending()` method
checked if ANY deferred assertion passed and reached consuming/Match
states, but didn't distinguish which tails were behind which assertions.

## What was hard

- The Bug 46 fix was interleaved with Bug 45 work, making it tricky to
  confirm this was a pre-existing issue vs. a Bug 45 regression.
- The `pending_break_tails` type change from `Vec<StateIdx>` to
  `Vec<(StateIdx, Box<[StateIdx]>)>` required updates at many sites:
  3 deposit sites in `step_slow_impl`, the pre-step resolution in
  `chunk()`, the `finish()` EOI path, Display/Debug impls, and `dump.rs`.
- The original `resolve_deferred_for_pending()` method walked epsilon
  transitions dynamically — replacing it with per-tail static assertion
  lists required a different evaluation model.

## Fix

1. **Modified `break_consuming_tails()`** to track per-tail deferred
   assertions during the epsilon walk.  Each consuming state now carries
   a `Box<[StateIdx]>` of the assertion NFA indices on its specific path
   from the break output.  Returns a third value:
   `Vec<(StateIdx, Box<[StateIdx]>)>`.

2. **Added `break_consuming_deferred` field** to
   `Tier3OriginKind::Increment` — the per-tail deferred assertions
   precomputed at analysis time.

3. **Changed `pending_break_tails`** from `Vec<StateIdx>` to
   `Vec<(StateIdx, Box<[StateIdx]>)>` — each pending tail carries its own
   deferred assertions.

4. **Added `per_tail_asserts_pass()` method** — evaluates ALL assertions
   in a tail's specific list using `kind.eval()`.  Returns `true` only if
   every assertion passes.

5. **Updated all deposit sites** (3 in `step_slow_impl` macro, 1 in
   `chunk()` pre-step) to push per-tail assertions from
   `break_consuming_deferred` instead of plain `StateIdx` values.

6. **Updated resolution sites**:
   - Pre-step `chunk()`: evaluate each tail individually instead of using
     a global `any_assert_passed` gate.  Tails whose assertions fail are
     simply skipped (not promoted).
   - `finish()` EOI: evaluate per-tail assertions at end-of-input for
     each pending tail individually.

7. **Removed `resolve_deferred_for_pending()`** — replaced entirely by
   per-tail evaluation.

8. **Updated Display impl** for `Tier3DfaMatcher` and dump output for
   `Tier3OriginKind::Increment` to show per-tail assertion details.

## Tooling ideas

- When adding new fields to matcher structs or analysis types, also update
  the Display/Debug impls and dump output immediately.  This would have
  prevented the half-edited compilation state.
- A `--debug` trace line showing per-tail assertion evaluation results
  (which asserts passed/failed for each tail) would speed up future
  investigations of assertion-gating bugs.
