# Bug 24: Tier 3 clean\_counter\_free\_mae loses mae when clean\_nb has empty NFA states

## Bug summary

- **Pattern:** `^(.{8,8}|f{5,26}|f{5,26})$`
- **Input:** `fffffabc` (8 chars)
- **Expected:** MATCH (via `.{8,8}` alternative)
- **Actual:** NO MATCH (Tier 3 false negative)
- **Affected tier:** Tier 3 (per-instance path), only when contaminated
  (`break_extras && num_counters > 1`)

The single-counter variant `^(.{8,8}|f{5,26})$` matched correctly. The bug
required 2+ counters to trigger the contamination code path.

## Root cause

`clean_counter_free_mae(t)` (Bug 22 infrastructure) verified that a
transition's match-at-end signal was genuinely counter-free by checking
whether the origin existed in `clean_nb`'s NFA states. But `clean_nb` was
already advanced to the **post-transition** state before the check ran.

When the transition's only consuming origin (state 8, from `.{8,8}`) had a
target of `Assert(End) → Match` — a state with no consuming successors — the
post-transition `clean_nb` had `nfa_states = []`. The origin check
`nb_nfa.contains(&8)` failed, and the mae signal was lost.

The mismatch:
- **Origin 8** was in the **pre-transition** `clean_nb` (nfa={8})
- But NOT in the **post-transition** `clean_nb` (nfa={})
- The method checked the post-transition state

## Investigation narrative

1. **Fuzz discovery.** `cargo +nightly fuzz run fuzz_differential` found a
   crash artifact. The crash was an `assert_eq!` failure: a DFA tier
   disagreed with the NFA oracle.

2. **Decode the seed.** Used the `decode_seed` example to extract the pattern
   `^(.{8,8}|f{5,26}|f{5,26})$` and its 53 generated inputs.

3. **Binary search for failing input.** Tested f-string inputs at counter
   boundaries manually via the CLI. Found that `fffffabc`, `fffffXXX`, and
   `ffffffxx` (8-char strings starting with 5+ f's then non-f chars) all
   showed NFA=MATCH, Tier3=NO MATCH.

4. **Confirmed single-counter works.** `^(.{8,8}|f{5,26})$` matched
   correctly — the bug only triggered with 2 counters (contamination path).

5. **Byte-by-byte debug traces.** Compared NFA (`--tier 0 --chunk-size 1`)
   with Tier 3:
   - After byte 4 (5th 'f'): `mae=true, break_extras` — counter break set
     mae correctly.
   - After byte 5 ('a'): `mae` cleared (reset in step_slow), counter
     instances died, `break_extras` persisted.
   - After byte 7 ('c'): NFA showed `nfa={17,0}` (End assert), MATCH.
     Tier 3 showed `state=13 nfa={} matched=false` — no mae, no cf_mae.

   Single-counter trace showed `mae cf_mae` correctly set after byte 7.
   The newly added `cf_mae` debug output (tooling from Bug 23) was
   instrumental in spotting the divergence.

6. **Traced `clean_nb` chain manually.**
   - After byte 4: `clean_nb` = state 10 (nfa={6,9,13})
   - After byte 5: `clean_nb` advanced to nfa={7} (only `.{8,8}` survived)
   - After byte 6: `clean_nb` → nfa={8}
   - After byte 7: `clean_nb` → nfa={} (state 8 consumed 'c' → target 17
     Assert(End), no consuming successors)

7. **Found the mismatch.** `clean_counter_free_mae(t)` checked
   `nb_nfa.contains(&8)` where `nb_nfa` was `clean_nb`'s post-transition
   NFA states (empty). Origin 8 was NOT in the empty set → returned false.

8. **Designed the fix.** Instead of checking origins against `clean_nb`'s
   NFA states (a post-transition artifact), use the clean chain's
   transition's `nb_counter_free_mae` directly. This is computed at
   `populate()` time against the pre-transition clean state and correctly
   indicates whether a counter-free `$ → Match` path exists.

## What was hard

- **The contamination path is deeply indirect.** The bug required tracing
  `clean_nb` through multiple bytes to understand that it advanced past the
  relevant origin. The contamination mechanism (Bug 22) is already complex;
  this bug was in a subtle interaction between `clean_nb` advancement timing
  and the origin check semantics.

- **The origin-vs-NFA-state confusion.** Origins are consuming states in the
  FROM state. `clean_nb`'s NFA states are consuming states in the TO state.
  After a transition, origins from the FROM state are NOT in the TO state's
  NFA set — they've been consumed and replaced by their targets. The check
  was conceptually comparing apples to oranges.

- **Single-counter variant working correctly was misleading.** With one
  counter, the non-contaminated path (`t.nb_counter_free_mae`) was used,
  which works correctly. This made it look like a contamination-specific
  issue, which it was — but the root cause was a design flaw in the
  contamination check, not a missing contamination guard.

## Tooling ideas

1. **Add `clean_nb` NFA states to `--debug` Display output.** Showing the
   clean chain's state alongside the main state would have made the empty
   NFA set immediately visible.

2. **Log `clean_counter_free_mae` inputs.** When the contaminated path
   runs and returns false, log the origin being checked, the clean_nb
   NFA states, and why the check failed. This would have pointed directly
   to the empty-NFA-set issue.

3. **Property test for clean_nb tracking.** Assert that when `clean_nb`
   is not DEAD, it should be a "subset" of the main path's no-break
   state in terms of the origins that matter for mae. This is tricky to
   formalize but would catch future divergences.
