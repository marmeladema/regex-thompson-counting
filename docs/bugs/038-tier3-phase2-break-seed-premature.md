# Bug 38 — Phase-2 break_seeds fire one byte too early

## Bug summary

| Field | Value |
|-------|-------|
| Pattern | `^e{4,5}e{4,5}ee{4,5}$` |
| Input | `"eeeeeeeeeeee"` (12 e's) |
| Expected | `false` (pattern requires 13-16 e's) |
| Actual | `true` (Tier 3 false positive) |
| Affected tier | Tier 3 (no-unroll path) |
| Artifact | `crash-e43c320080b1d813ce0d59d909b15a02013910e5` |

## Root cause

The `compute_tier3_analysis` function computes `break_seeds` in two phases:

- **Phase 1**: Walk epsilon transitions from a CInc's break target.  When
  a `CounterInstance` is found, seed the downstream counter at its body
  origin.  This handles the common case where the break path reaches the
  next counter through pure epsilon transitions (Split, Assert, CI).

- **Phase 2**: For consuming states found on the break path (stored in
  `break_consuming`), follow their post-consumption targets through
  epsilon to find CIs.  This was meant to handle "multi-hop" break paths
  where a consuming state sits between the trigger counter's break and
  the downstream counter's CI.

Phase 2 seeds were applied at runtime in `step_slow_impl!` on the **same
byte** as the counter break.  But the intervening consuming state hasn't
consumed its byte yet at that point — it won't consume until the *next*
byte (when `post_break_tails` processes it).  This gave the downstream
counter one extra counting step.

### Concrete trace

NFA states for `^e{4,5}e{4,5}ee{4,5}$`:

```
 0: Assert(Start) → 3
 1: Byte('e') → 2          ← c0 body
 2: CInc(c0, {4,5}) → 1|6
 3: CI(c0) → 1
 4: Byte('e') → 5          ← c1 body
 5: CInc(c1, {4,5}) → 4|7
 6: CI(c1) → 4
 7: Byte('e') → 10         ← literal 'e' between c1 and c2
10: CI(c2) → 8
 8: Byte('e') → 9          ← c2 body
 9: CInc(c2, {4,5}) → 8|11
11: Assert(End) → 12
12: Match
```

When c1 breaks at byte 7 (value 4 >= min 4):

| Mechanism | When c2 is seeded | c2's first body byte | c2 iterations for bytes 8-11 |
|-----------|-------------------|---------------------|------------------------------|
| Phase 2 break_seed | Byte 7 (value 0) | Byte 8 | **4** (too many) |
| Tail at state 7 | Byte 8 (state 7 consumes 'e') → Advance → origin 8 tail | Byte 9 | **3** (correct) |
| NFA | Byte 8 (state 7 consumes 'e') → CI(c2) → state 8 | Byte 9 | **3** (correct) |

The phase 2 seed gave c2 four iterations (value 4 >= min 4), enabling break
to `$ → Match`.  The NFA only had three c2 threads (value 3 < min 4), correctly
rejecting.

### Why the tail mechanism is sufficient

The `break_consuming_tails` function already tracks consuming states on the
break path as post-break tails.  When the tail processes state 7 on the next
byte, it consumes 'e', sees the Advance target → origin 8, and adds origin 8
to `next_post_break_tails`.  On the byte after that, origin 8 consumes 'e'
and reaches CInc(c2), triggering the tail→CInc handoff which correctly seeds
c2 with `pbt_value + 1 = 1`.

Phase 2 seeds are always redundant with the tail mechanism and always fire
one byte too early.

## Investigation narrative

1. Reproduced with `--tier 0` (NO MATCH) vs `--tier 3 --unroll-limit 0` (MATCH).

2. Compared debug traces byte-by-byte.  The NFA showed 3 c2 threads maximum
   at end-of-input; Tier 3 showed 4 c2 instances with `[8@4]` enabling match.

3. Traced the divergence to chunk #7: Tier 3 seeded c2@8 with value 0 (via
   break_seeds), but the NFA had state 7 as a pending consuming thread —
   c2 wouldn't start until state 7 consumed on chunk #8.

4. Examined `compute_break_seeds` and identified the two-phase structure.
   Phase 1 handles pure epsilon paths; Phase 2 follows consuming states'
   targets.  Phase 2 seeds bypass the consuming state's byte consumption.

5. Verified the tail mechanism handles the same case correctly: state 7 is
   in `break_consuming_tails`, tracked as a post_break_tail, and processes
   with proper byte timing.

6. Removed phase 2 entirely.  All 454 tests pass.

## What was hard

The initial context summary described the bug as "contamination" where c2 was
seeded one byte too early from the with_break DFA state, but the actual
mechanism was different — it was the `break_seeds` array that fired the
premature seed, not the DFA closure contamination.  The break_seeds are a
separate mechanism from DFA state computation.

Understanding the interaction between three parallel seeding mechanisms
(break_seeds, post_break_tails/Advance, and counter_free_seeds) required
careful reading of the populate and step_slow_impl code to track which
mechanism fires when and what value each seed gets.

## Tooling ideas

- The `--debug` output could show when break_seeds fire (which trigger
  counter broke, which seed was applied, what value) to distinguish
  break_seed effects from tail effects and unconditional seed effects.
- A `dump --dfa --verbose` mode could show which break_seeds are phase-1
  vs phase-2, making it immediately clear that a consuming state was
  being bypassed.
