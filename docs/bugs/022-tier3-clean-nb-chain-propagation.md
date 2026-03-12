# Bug 22: Tier 3 false positive — contamination propagates through no-break chain

## Bug summary

- **Pattern:** Multi-counter pattern where contamination from a
  `with_break` transition persists across subsequent non-break steps.
- **Expected:** NO MATCH
- **Actual (Tier 3):** MATCH (false positive)
- **Affected tier:** Tier 3 (conditional transitions)

## Root cause

Bug 19 introduced `clean_counter_free_mae()` to handle contaminated DFA
states.  It works by comparing the `with_break` state's NFA origins
against the `no_break` state's NFA origins — only origins present in both
are considered "clean" for the counter-free match-at-end check.

The reference for "clean" was `self.no_break_current` — the no-break
DFA state chain.  But here's the problem: when a contaminated state
transitions to a `no_break` successor, that successor is computed from
the contaminated state's full NFA set (which includes break-injected
origins).  The no-break successor therefore inherits the contamination.
On the next step, `clean_counter_free_mae()` compares against this
already-contaminated `no_break_current` and finds the break-injected
origins in both — declaring them "clean" when they're not.

The contamination propagates indefinitely through the `no_break_current`
chain, eventually causing `clean_counter_free_mae()` to report a match
on a state where no counter has actually reached its minimum.

## Investigation narrative

This bug was found by fuzzing shortly after Bug 19 was fixed.  The
`--debug` trace showed:

1. Step N: `with_break` transition → `current_has_break_extras=true`
2. Step N+1: `no_break` transition → but `current_has_break_extras`
   remained `true` (contamination propagated)
3. Step N+2: `clean_counter_free_mae()` returned `true` — false positive

The key insight came from examining the NFA state sets in
`no_break_current` across steps.  After a contaminated step, the
`no_break_current` state contained origins that shouldn't have been there
— they entered through the contaminated predecessor's break-injected
states.

The fix introduces a separate `clean_nb` DFA state that always advances
from the *previous* clean state, never from a contaminated one.  On each
step:
- If not contaminated: `clean_nb` tracks `no_break_current` normally.
- If contaminated: `clean_nb` is computed independently by looking up the
  no-break successor of the previous `clean_nb` state.

`clean_counter_free_mae()` uses `clean_nb` as its reference instead of
`no_break_current`.  This ensures the reference chain is never
contaminated by break-injected origins.

The cost is one extra DFA transition lookup per byte while contaminated,
but contamination only occurs in multi-counter patterns and the lookups
hit the cached transition table.

## What was hard

- **Subtle propagation.** The contamination doesn't show up on the step
  where it happens — it shows up on subsequent steps when
  `clean_counter_free_mae()` incorrectly validates origins.  You need to
  trace the `no_break_current` NFA state sets across multiple steps to see
  the inherited contamination.

- **Reference vs subject confusion.** Bug 19's `clean_counter_free_mae()`
  was correct in *concept* (compare against a clean reference) but wrong in
  *implementation* (the reference itself was contaminated).  The fix isn't
  to change the comparison logic, but to maintain a separate clean chain.

- **Cost concern.** Maintaining a parallel DFA state chain raised
  performance questions.  Verifying that the extra lookup was limited to
  contaminated steps (multi-counter patterns only) and that it hit cached
  transitions required reading through the transition cache internals.

## Tooling ideas

- **`clean_nb` in `--debug` output:** Show `clean_nb` alongside
  `no_break_current` in the per-step Display for tier 3.  When they
  diverge, contamination is active.  This would make it immediately
  visible when the clean chain separates from the regular no-break chain.

- **Contamination lifetime tracking:** A counter showing "contaminated
  for N steps" would help gauge whether contamination is transient
  (typically 1-2 steps) or persistent (suggesting a propagation bug).

- **Diff between `no_break_current` and `clean_nb` NFA sets:** Show which
  NFA states are in `no_break_current` but not `clean_nb` — these are the
  contaminating origins.  Would directly identify which break-path states
  are causing the problem.
