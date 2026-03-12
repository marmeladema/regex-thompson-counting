# Bug 23: Tier 3 false positive from unguarded `match_at_end` in non-counting transitions

## Bug summary

- **Pattern:** `c{2,12}c{2,12}f$`
- **Input:** `cccccccdcccf` (12 bytes)
- **Expected:** NO MATCH (NFA result)
- **Actual:** MATCH (Tier 3 result, with `unroll_limit=0`)
- **Affected tier:** Tier 3 (per-instance path)
- **Type:** False positive

The `d` at position 7 breaks all counter threads.  Only 3 `c`'s remain
(positions 8-10) before `f`, which is not enough for `c{2,12}c{2,12}`
(requires at least 4 `c`'s total: 2 for each counter).

## Root cause

DFA state 2 has `nfa_states = {0, 3, 6}`, where:
- State 0: `Byte('c')` — counter 0's body
- State 3: `Byte('c')` — counter 1's body
- State 6: `Byte('f')` — a **post-break tail**, only reachable after both
  counters break with valid counts (>= min)

The transition from state 2 on byte `'f'` is classified as **non-counting**
(`is_counting = false`) because the probe epsilon closure seeds are
`[state_7, start]`.  State 7 (`Assert(End)`) is not a `CounterIncrement`
node, and from start the walk goes CI(c0) -> Byte('c') and stops at the
consuming state without encountering `CounterIncrement`.  So
`encountered_cinc = false`.

Because `!t.is_counting`, the `step_slow_impl!` macro's non-counting
branch (tier3.rs:2519-2530) directly propagated
`t.no_break_is_match_at_end` to `self.match_at_end`.  This was `true`
because state 6 consumed `'f'`, advancing to state 7 (`Assert(End)`),
which set `is_match_at_end` during the epsilon closure computation.

The problem: state 6 is a counter-dependent tail state.  The
`is_match_at_end` from consuming state 6 should only be valid when
counter 1 has actually broken (count >= min=2).  But the non-counting
branch bypassed all counter-free filtering.

At byte 11 (`f`), counter 1's maximum instance count was only 1 (< min=2),
so the break path was not valid.  But the cached DFA transition still
propagated `mae=true` unguarded.

## Investigation narrative

1. **Fuzzer output:** `cargo +nightly fuzz run fuzz_differential` found a
   crash with exit code 77 (assertion failure).  The panic hook printed:
   ```
   Pattern: `c{2,12}c{2,12}f$`
   Phase:   match (no-unroll Tier 3)
   Input:   "cccccccdcccf" (len=12)
   Tier 3 disagrees with NFA: tier3=true, nfa=false
   ```

2. **Debug traces:** Compared NFA (`--tier 0`) and Tier 3 (`--tier 3`)
   byte-by-byte with `--debug --chunk-size 1 --unroll-limit 0`:
   - NFA: after `d` at position 7, all threads reset.  Only state 0 with
     fresh counter survives.  After `ccc` (positions 8-10), counter 0 has
     count 3 and counter 1 has count 1.  `f` at position 11 finds no valid
     break path (c1 count < min=2).  Result: NO MATCH.
   - Tier 3: after `f` at position 11, DFA state 3 with `nfa={0}` has
     `mae=true`.  The `mae` flag came from the DFA transition computation.

3. **Code inspection:** Traced `match_at_end` propagation in
   `step_slow_impl!`.  Found the `!t.is_counting` branch at line 2519
   directly used `no_break_is_match_at_end`, while the counting branch
   (line 2541) correctly used `counter_free_match_at_end` (or
   `clean_counter_free_mae` when contaminated).  The inline fast path
   (line 2843) also correctly used `nb_counter_free_mae`.

4. **Confirmed:** `t.nb_counter_free_mae = false` for this transition
   (counter-free closure from start doesn't reach `Assert(End)`), so
   using it instead of the raw flag is correct.

## What was hard

The transition being classified as **non-counting** was the non-obvious
part.  The pattern has two counters, and the `'f'` byte is downstream of
both.  But the probe epsilon closure from `[state_7, start]` never
encounters a `CounterIncrement` node because (a) state 7 is an assertion,
and (b) from start the walk stops at `Byte('c')` (a consuming state).

This meant the bug was in a code path (`!t.is_counting`) that seemed
irrelevant for counter-dependent patterns.

## Tooling ideas

- The `--debug` output's `mae` flag on the DFA state line was critical for
  spotting the false positive.  Adding a `cf_mae` (counter-free mae) field
  alongside `mae` in the debug output would make it immediately clear when
  the raw and filtered values diverge.

- A consistency check in `step_slow_impl!` that asserts
  `match_at_end <= nb_counter_free_mae` for non-counting transitions would
  have caught this as a debug assertion.

## Fix

Changed the `!t.is_counting` branch in `step_slow_impl!` (tier3.rs:2519)
and `step_from_dead` (tier3.rs:2720) to use `nb_counter_free_mae` (or
`clean_counter_free_mae` when contaminated) instead of the raw
`no_break_is_match_at_end` / `with_break_is_match_at_end`.  This matches
the logic already used in the fast path (tier3.rs:2843).

Removed the now-unused `no_break_is_match_at_end` and
`with_break_is_match_at_end` fields from the `Transition` struct entirely,
since all `match_at_end` decisions now go through counter-free filtered
fields.
