# Bug 44 — Pending break tails lost: not in DFA state, can't consume byte

## Bug summary

| Field | Value |
|-------|-------|
| Pattern | `^.{2,13}\Ba?((a?a?)?(a?a?)?)?$` |
| Input | `"yxaayca"` (7 bytes) |
| Expected | `true` (NFA matches) |
| Actual | `false` (Tier 3 false negative) |
| Affected tier | Tier 3 (no-unroll path) |
| Artifact | `crash-4fa1269e5abb8981285ce26083e742abd5f3859c` |

## Root cause

Two interacting bugs conspired to lose pending break tails:

### Bug A: `resolve_deferred_for_pending` incorrectly pruned consuming states

The function walks epsilon transitions from the deferred assertion's output
to check if consuming states or Match are reachable.  It used
`state_can_reach_match[i]` to prune unreachable states.  But
`state_can_reach_match` tracks reachability through **epsilon transitions
only** — consuming states (Byte, ByteClass, etc.) are always `false` in
this array because they require byte consumption.  The function therefore
pruned the very consuming states it was looking for, always returning
`false` for patterns where the break path's deferred assertion leads to
consuming states (which is the common case).

Fix: remove the `state_can_reach_match` pruning from the epsilon walk.
The walk is bounded by the visited set and only follows epsilon edges,
so it terminates without the pruning.

### Bug B: promoted pending tails not consumable by step_slow

When `resolve_deferred_for_pending` returned `true` (after fixing Bug A),
the pending tails were promoted to `post_break_tails`.  But `step_slow_impl`
processes tails by looking them up in the cached transition's `origin_keys`,
which only includes consuming states from the current DFA state.  The
pending tails are consuming states from the counter break path — they're
behind the deferred assertion that blocked the epsilon closure at compile
time, so they're NOT in the DFA state and NOT in `origin_keys`.  Result:
step_slow silently dropped them.

Fix: consume the current byte `b` directly in the pre-step resolution code
using `consume_byte()` + `analysis.targets[]` (the same static data
step_slow uses, but accessed without needing origin_keys).  Results are
stored in `pending_resolved_tails` / `pending_resolved_mae` and injected
into `post_break_tails` / `match_at_end` after step_slow returns (step_slow
clears these at its start, so direct injection before step_slow would be
overwritten).

## Investigation narrative

1. **Fuzz discovery**: `cargo +nightly fuzz run fuzz_differential` found a
   crash within seconds.  The assertion message identified Tier 3 disagreeing
   with NFA on `^.{2,13}\Ba?((a?a?)?(a?a?)?)?$` input len=7.

2. **Decoded seed**: Used a small Rust program with `generate_pattern` /
   `generate_inputs` to extract the pattern and input `"yxaayca"`.

3. **Reproduced**: `cargo run --release -- match --tier 3 --unroll-limit 0`
   confirmed the false negative.

4. **NFA trace**: Byte-by-byte NFA trace showed that after the last byte
   'a', the NFA had state 18 (`Assert(End)`) in its thread set, reaching
   Match at finish().  The counter broke, `\B` passed between 'c' and 'a',
   and the tail chain `a?((a?a?)?(a?a?)?)?$` consumed 'a' and reached `$`.

5. **Tier 3 trace**: The new always-show Display impl (implemented earlier
   this session) immediately revealed the problem:
   - `pending_tails: [5,7,9,12,14]` persisted across ALL steps
   - `tails: []` was always empty
   - `mae=false` throughout
   This showed the pending tails were never being promoted or consumed.

6. **Hypothesis 1**: Bug 42's pending mechanism isn't resolving.  Traced
   the pre-step resolution code.  Added a debug_assert that confirmed
   `resolve_deferred_for_pending` was returning `false` even though
   `verified_deferred_asserts` had entries and `pending_break_tails` was
   non-empty.

7. **Root cause A found**: Examined `resolve_deferred_for_pending`'s epsilon
   walk.  The pruning condition `!self.regex.state_can_reach_match[i]`
   rejected consuming states (Byte, ByteClass) because `state_can_reach_match`
   only tracks epsilon reachability.  The `dump --dfa` output confirmed:
   `state_can_reach_match` showed `.` (false) for states 5, 7, 9, 12, 14.

8. **Fixed Bug A**: Removed the `state_can_reach_match` pruning.
   `resolve_deferred_for_pending` now returned `true`, and the code entered
   the promotion branch.

9. **Still broken**: After fixing Bug A, the tails were promoted to
   `post_break_tails`, but step_slow dropped them (not in `origin_keys`).
   Then `match_at_end` was set in the pre-step code but cleared by
   step_slow's `self.match_at_end = false;`.

10. **Root cause B found**: The promoted tails are consuming states NOT in
    the current DFA state (the deferred assertion blocked the closure).
    step_slow uses the cached transition which only knows about origins in
    the DFA state.

11. **Fixed Bug B**: Added `pending_resolved_tails`/`pending_resolved_mae`
    fields.  Pre-step code consumes `b` directly via `consume_byte()` +
    `analysis.targets[]`, stores results in the pending_resolved fields,
    and injects them after step_slow returns.

## What was hard

- The interaction between Bug 42's pending mechanism and the DFA state's
  origin_keys was non-obvious.  The pending tails are "outside" the DFA
  state — they exist only in the tail tracker, not in the cached transition.
  This meant the standard tail processing in step_slow couldn't handle them.

- `state_can_reach_match` being epsilon-only is documented but easy to
  forget.  Its use in `resolve_deferred_for_pending` silently made the
  function return false for the most common case (assertions leading to
  consuming states).

## Tooling ideas

- The new always-show Display format (implemented just before this bug)
  was immediately useful.  `pending_tails: [5,7,9,12,14]` persisting
  across all steps was the first clue that promotion wasn't working.

- A `--debug-resolve` flag that logs each `resolve_deferred_for_pending`
  call with its inputs (deferred asserts, prev byte, next byte) and result
  would have saved the manual debug_assert step.
