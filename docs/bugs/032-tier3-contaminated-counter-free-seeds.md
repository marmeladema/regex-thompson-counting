# Bug 32: Tier 3 contaminated counter_free_seeds false positive

## Bug summary

- **Pattern:** `^((a{8,13}d*)?.?a?.{6,6}a?)?$`
- **Input:** 50 `'a'` characters
- **Expected:** `false` (NFA oracle)
- **Actual:** `true` (Tier 3 false positive)
- **Affected tier:** Tier 3 (conditional transitions)
- **Artifact:** `fuzz/artifacts/fuzz_differential/crash-b07014e9fada092038af6b7c05add8a36d1edb0a`

## Root cause

### The contamination problem

The pattern has two counters: c0 (`{8,13}`) with body `Byte('a')`, and c1
(`{6,6}`) with body `ByteClass(.)`.  The NFA structure includes optional
paths that make origins 7, 9, 11 globally `reachable_without_break` (from
`^` via the `(a{8,13}d*)?` optional skip).

When c0 first can break (after 8 'a's), the DFA transitions to its
`with_break` state (state 2, NFA set `{1,4,7,9,11,14}`).  The extras
`{4,7,9,14}` come from c0's break chain.  This state is "contaminated" —
it contains NFA states that only appeared because of the break.

### How the false unconditional seed arises

During `populate()` for state 2's transition on `'a'`, the
`counter_free_seeds` computation:
1. Filters origins by `reachable_without_break` — origins 7, 9 pass
   because they ARE globally reachable from `^`.
2. Runs an epsilon closure from those origins' targets.
3. The closure reaches `CI(c1)` at state 13, producing seed `c1@11`.

This makes `c1@11` an unconditional seed: it fires every byte from state 2.

### The break_seed deduplication

The `compute_break_seeds()` function excluded break_seeds that duplicated
unconditional seeds (to prevent double-seeding).  Since `c1@11` was already
an unconditional seed, the break_seed `trigger:c0 → c1@11` was removed.

### The cascading failure

After c0 dies at position 13 (max=13), the DFA state remains contaminated
(contamination propagates via the `from_contaminated` flag).  The
unconditional seed `c1@11` keeps firing every byte, giving c1 a constant
stream of instances.  After 6 bytes, c1 breaks (value 6 ≥ min 6), adding
tail [14] (which has `target_is_match_at_end`).  This sets `match_at_end`
= true, which is sticky.  At end-of-input (position 50), the sticky
`match_at_end` produces a false positive.

### Why the clean_nb chain can't see it

The clean_nb chain (state 3, NFA `{1,11}`) correctly classifies `c1@11` as
break-gated (not unconditional) because:
- From state 3's transition on 'a', the counter_free closure starts from
  cf_targets `{2,12}` + start.
- At position > 0, `Assert(Start)` at state 0 evaluates to `Fail`, so the
  start state contributes nothing.
- The CInc cont paths lead to consuming states directly (no CI traversal).
- Result: `counter_free_seeds = []`, and `c1@11` is properly in
  `break_seeds`.

## Fix (two-part)

### Part 1: Runtime seed filtering in step_slow

On **counting** transitions when the DFA state is contaminated, filter
unconditional seeds against the clean_nb chain's transition seeds.  Only
seeds that also appear in the clean chain are truly counter-free.

Non-counting transitions are NOT filtered because their seeds come from the
full probe closure (`follow_break=true`) and have no `break_seeds` to fall
back on.  Filtering them would kill legitimate downstream counter seeds
(verified by `test_inner_unroll_over_budget` / `^(a{1,17}b){2,3}$` on
`"abab"`).

### Part 2: Stop deduplicating break_seeds against unconditional seeds

`compute_break_seeds()` no longer skips break_seeds that duplicate
unconditional seeds.  This ensures that when Part 1 suppresses a
contaminated unconditional seed, the corresponding break_seed is still
available to fire when the triggering counter actually breaks.

Double-seeding is harmless: `seed()` deduplicates via `contains()`.

## Investigation narrative

1. **Reproduced the crash** using the fuzz artifact.  Decoded the pattern
   and input from the artifact seed.

2. **Ran `--debug --chunk-size 1`** comparing `--tier 3` vs `--tier 0`.
   The NFA showed no match for 50 'a's, while Tier 3 showed
   `match_at_end=true` after just 12 bytes.

3. **Used `dump --dfa`** to examine the Tier 3 analysis.  Identified that
   `reachable_without_break: [1,7,9,11]` — origins 7,9 are in the break
   extras of state 2 but also globally reachable without break.

4. **First fix attempt** filtered ALL unconditional seeds against clean_nb
   when contaminated.  Fixed the crash but caused a regression on
   `^(a{1,17}b){2,3}$` / `"abab"` (inner-unrolled nested counters).

5. **Traced the regression**: the seed `c1@5` in the non-counting
   transition from state 1 on 'b' was legitimate — it came from the
   full probe closure for non-counting transitions, which don't have
   break_seeds.

6. **Refined fix** to only filter counting transitions.  Fixed the
   regression but introduced a false negative on the original pattern for
   14 'a's (which should match).

7. **Root cause of false negative**: suppressing `c1@11` from the counting
   transition was correct, BUT the break_seed `trigger:c0 → c1@11` had
   been removed by `compute_break_seeds()` deduplication.  Without either
   the unconditional seed or break_seed, c1 couldn't be seeded when c0
   actually broke.

8. **Final fix**: removed the deduplication in `compute_break_seeds()`.
   Now both the unconditional seed and break_seed exist in the transition.
   The runtime filter suppresses the unconditional seed when contaminated,
   and the break_seed fires normally when c0 breaks.

## What was hard

- **The multi-layered interaction**: the bug involved three interacting
  mechanisms — `counter_free_seeds` at populate time, `break_seeds`
  deduplication, and runtime contamination tracking.  Fixing one layer
  without understanding the others caused cascading regressions.

- **The `Assert(Start)` subtlety**: the clean_nb chain's
  `counter_free_seeds` were empty because `Assert(Start)` fails at
  position > 0, so the start state doesn't contribute CI seeds to the
  closure.  This is correct behavior but was initially confusing — the
  same transition from the main state (which has break extras) produces
  different counter_free_seeds than the clean equivalent.

- **Non-counting vs counting transitions**: the seed computation paths
  diverge completely between counting (conservative, filtered) and
  non-counting (permissive, all probe seeds).  The fix needed to respect
  this distinction at runtime.

## Tooling ideas

- **Populate trace mode**: a `--debug-populate` flag that logs each
  `populate()` call's inputs and outputs (from_state, byte, seeds,
  break_seeds, counter_free_seeds, is_counting).  Would have revealed the
  empty counter_free_seeds and missing break_seeds immediately.

- **Seed provenance in debug output**: show where each counter instance
  came from (unconditional seed, break_seed, tail handoff, pre_seed) in
  the `--debug` trace.  Would have identified the unconditional seed as
  the source of the false c1 instances.

- **Clean_nb seed comparison**: a debug assertion or trace that compares
  the main transition's seeds against the clean_nb transition's seeds
  when contaminated, flagging discrepancies.
