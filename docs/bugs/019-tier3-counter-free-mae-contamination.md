# Bug 19: Tier 3 false positive — counter_free_mae contamination in multi-counter patterns

## Bug summary

- **Pattern:** Multi-counter pattern where a `with_break` transition injects
  downstream counter origins into the DFA state before those counters have
  reached their minimums.
- **Expected:** NO MATCH
- **Actual (Tier 3):** MATCH (false positive)
- **Affected tier:** Tier 3 (conditional transitions)

## Root cause

Tier 3's `counter_free_match_at_end` flag is precomputed per DFA
transition.  It answers: "ignoring all counter instances, can the NFA
states in this DFA state reach `$ → Match` through non-counter paths?"
This is used to decide match-at-end without consulting counter values —
a fast path for states where the match doesn't depend on any counter
reaching its minimum.

The flag is computed using the static `reachable_without_break` property,
which identifies NFA states reachable from the start without going
through any counter break path.  The problem: when the current DFA state
was reached via a `with_break` transition, it includes NFA states from
downstream counter break chains.  These states are statically
`reachable_without_break` (they *can* be reached without breaks in
general), but they entered *this particular DFA state* through a break
path — and the upstream counter may not have actually reached its minimum.

Concretely: counter A breaks → its break path reaches counter B's entry →
counter B's suffix states (including `$ → Match` paths) appear in the
DFA state.  The `counter_free_match_at_end` flag sees these suffix states
as `reachable_without_break` and reports a match, even though counter A
hasn't actually been verified to have broken with a sufficient count.

## Investigation narrative

Found by fuzzing.  The `--debug` trace showed `mae` (match-at-end)
appearing on a step where no counter instances had reached their minimum.
Comparing against `--tier 0` (NFA) confirmed the NFA correctly reported
no match.

The key observation: the DFA state's NFA set contained origins that
shouldn't have been there yet — they came from a downstream counter's
suffix that was only reachable *after* the upstream counter broke.  But
the `counter_free_match_at_end` check doesn't know whether origins
entered via break paths or not.

The fix tracks a `current_has_break_extras` flag on the matcher that's
set whenever the DFA transitions to a `with_break` state.  For
multi-counter patterns (`num_counters > 1`), the precomputed
`counter_free_match_at_end` is replaced by `clean_counter_free_mae()`,
which restricts the check to origins also present in the `no_break`
version of the *from* state.  Origins that only appear in the
`with_break` state are excluded.

Single-counter patterns are exempt: once the sole counter breaks, all
break-path origins are valid (there's no downstream counter chain).

## What was hard

- **Static vs dynamic reachability.** The `reachable_without_break`
  property is correct statically — those origins *can* be reached without
  breaks.  The problem is *dynamic* — they entered this DFA state through
  a break, and the break hasn't been verified yet.  This distinction
  between "statically reachable" and "dynamically present via a specific
  path" is subtle and not visible in any existing debug output.

- **Single-counter exemption.** The initial fix applied
  `clean_counter_free_mae()` unconditionally, which caused false negatives
  on single-counter patterns (Bugs 20/21).  Understanding that the
  multi-counter chain problem doesn't apply to single-counter patterns
  required careful reasoning about when break-path contamination matters.

## Tooling ideas

- **Contamination flag in `--debug`:** Show `break_extras` in the per-step
  Display output when the current DFA state was reached via `with_break`.
  This was added as part of the fix (the `break_extras` annotation in the
  Tier 3 Display impl).

- **Origin provenance tracing:** When computing DFA successors, annotate
  which origins came from the `no_break` closure vs the `with_break`
  closure.  Would make contamination immediately visible.
