# Bug 34 — Tier 3 false negative: deferred asserts lost on None target in post_break_tails

## Bug Summary

- **Pattern:** `^1{2,30}c{9,48}y\b$`
- **Input:** `"1111111cccccccccccccy"` (7 '1's + 13 'c's + 'y')
- **Expected:** true (NFA matches)
- **Actual:** Tier 3 returns false
- **Affected tiers:** Tier 3 (conditional transitions)
- **Type:** False negative

## Root Cause

When counter c1 breaks after accumulating 13 'c's, the break path leads to
state 7 (`Byte('y')`), which becomes a `post_break_tail`.  On the next byte
('y'), the tail matches and transitions to state 8 (`Assert(\b)`).

The compile-time `analyze_target` function walks epsilon transitions from
state 8 and finds:
- State 8: `Assert(WordAscii)` → state 9
- State 9: `Assert(End)` → state 10
- State 10: `Match`

No consuming states are reachable, no `CounterIncrement` is found.
`analyze_target` returns `None` — meaning "dead target."

In `step_slow_impl!`, when the post_break_tails code processes tail
origin 7, it looks up the transition action for origin 7.  The action
maps to target 8, which has `analyze_target` result `None`.  The code
hits the `Some(None)` arm:

```rust
Some(None) => {
    // "dead" target — only calls check_tail_match_flags!
    check_tail_match_flags!(self, pbo);
}
```

This arm was NOT depositing `target_deferred_asserts[7]` (which contains
state 8, the `\b` assertion).  The Bug 30 fix that added deferred assert
deposits only put the code in the `Advance` arm, not the `None` arm.

Result: the `\b` assertion at state 8 is never recorded in
`verified_deferred_asserts`.  At `finish()`, `resolve_verified_deferred_asserts`
has nothing to evaluate → false negative.

### Why `analyze_target` returns `None`

`analyze_target` classifies targets into three cases:
1. **Increment** — CInc found downstream
2. **Advance** — consuming states found downstream (no CInc)
3. **None** — neither consuming states nor CInc found

For target state 8 (Assert → Assert → Match), there are no consuming
states and no CInc.  This is legitimately "no further byte consumption
needed" — but it's not "dead."  The path reaches Match through
assertions that need runtime evaluation.

The issue is that `None` conflates "unreachable" with "reachable only
through assertions."  The `target_deferred_asserts` mechanism exists
precisely to handle this case, but the deposit code was only in the
`Advance` arm.

## Investigation

1. **Artifact:** `crash-3f4ad8a3fb040898277438610023e4027ea0db22` from
   `fuzz_differential`.  Panic message: "Tier 3 disagrees with NFA ...
   tier3=false, nfa=true."

2. **Reproduction:** `cargo run --release -- match --tier 3 --unroll-limit 0
   '^1{2,30}c{9,48}y\b$' '1111111cccccccccccccy'` confirmed false negative.

3. **NFA trace:** `--tier 0 --debug --chunk-size 1` showed state 8 in the
   active set after 'y', successfully evaluating `\b` and `$` at EOI.

4. **Tier 3 trace:** `--tier 3 --debug --chunk-size 1` showed tail [7] at
   chunk 19, then `nfa={}` with no deferred asserts after chunk 20 ('y').

5. **NFA dump:** Confirmed state 8 is Assert(WordAscii) → 9 (Assert(End))
   → 10 (Match).  `state_can_reach_match[7]` is false (consuming state),
   `state_can_reach_match[8]` is true.

6. **Root cause:** Traced through `analyze_target(8)` → `None`, then
   the `populate` code building `origin_actions[7] = None`, then the
   `step_slow_impl!` macro's `Some(None)` arm — no deferred assert
   deposit.

## Fix

In the `Some(None)` arm of the post_break_tails processing in
`step_slow_impl!`, deposit `target_deferred_asserts[pbo.idx()]` into
`verified_deferred_asserts`, matching what the `Advance` arm already does.

**File:** `src/dfa/tier3.rs`, `step_slow_impl!` macro, `Some(None)` arm.

The `None` action correctly means "no consuming states downstream," but
the tail DID consume a byte and may have deferred assertions on its
target's epsilon path.  These must be deposited for `finish()` to
evaluate them at EOI.

## What Was Hard

The multi-layer indirection made the root cause non-obvious:
1. Compile-time `analyze_target` returns `None` — looks like "dead"
2. Runtime `populate` maps origin 7 to action `None`
3. `step_slow_impl!` treats `Some(None)` as "byte not accepted" when
   really the byte WAS accepted, just the downstream path is assert-only

The `--debug` trace was immediately helpful: seeing `tails: [7]` disappear
with no deferred asserts deposited pinpointed the problem to the tail
processing code.

## Tooling Ideas

- The `--debug` display could show when a tail origin matches but the
  action is `None`, distinguishing "byte consumed, assert-only path"
  from "byte not consumed."
- `analyze_target` could return a distinct variant for "assert-only
  path" vs "truly dead," making the semantics clearer at all call sites.
