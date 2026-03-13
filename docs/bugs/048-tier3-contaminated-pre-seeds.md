# Bug 48: Contaminated pre_seeds cause false positive match_at_end

## Bug Summary

- **Pattern**: `^.{0,26}\B((x{7,7}a?)?(a?a?)?)?$`
- **Input**: `"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"` (50 x's)
- **Expected**: `false` (NFA reports no match)
- **Actual**: `true` (Tier 3 false positive)
- **Affected tier**: Tier 3 (per-instance path)
- **Fuzz artifact**: `fuzz/artifacts/fuzz_match/crash-c7f37f3ce15b79086eedd4b19d002fe27f4d8b04`

## Root Cause

The pattern has two counters: c0 (`.{0,26}`, NFA min=1 max=26) and c1
(`x{7,7}`, min=7 max=7), with `\B` between them.

When c0 breaks, the with_break DFA closure includes c1's body state
(state 6, `Byte('x')`) via the break path through `\B`.  DFA state 4 =
`{1, 6, 9, 12, 14}`.  Because state 6 is in the DFA closure and has an
Increment action (it's c1's body: `Byte('x')` → `CInc(c1)`), the
`populate()` function produced **pre_seeds** for c1 at origin 6.

Pre_seeds are resolved seeds applied BEFORE the counter increment loop
on each step.  Unlike unconditional seeds (which have a Bug 32
contamination guard that filters against the clean no-break chain),
pre_seeds had **no contamination guard**.  They were applied
unconditionally every step.

The Bug 40 filter (`reachable_without_break`) didn't help because state 6
IS structurally reachable without a break — it's c1's body state, reached
from `CI(c1) → 6`.  But in this specific DFA state, state 6 entered the
closure only because of c0's break path.

The result: every step on 'x', a new c1 instance was seeded at value 0.
The counter loop incremented it and continued at value 1.  Previous
instances incremented too.  After 7 steps, the oldest instance reached
value 7 (= min = max), broke with `break_is_match_at_end=true`, and set
`self.match_at_end = true`.  This persisted to `finish()`, causing the
false positive.

Meanwhile, the clean no-break chain stayed at DfaStateId(1) = `{1}` (only
c0's body), with `clean_cf_mae=false` and `clean_is_match=false`.  The
counter-free mae guard correctly suppressed the DFA-level mae, but the
counter break's mae bypassed all contamination checks.

## The Contamination Mechanism (step by step)

1. Chunk 0: c0 gets value 1, breaks (1 ≥ min 1).  `\B` between
   'x'(word) and next 'x'(word) → `\B` passes (not a word boundary).
   c1 seeded via deferred break_seed.  DFA → state 2 (with_break).
2. Chunk 1: Deferred seed resolves → c1 at origin 6, value 0.  DFA →
   state 4 `{1,6,9,12,14}`.  `break_extras=true`.  From now on, pre_seeds
   re-seed c1 at origin 6 every step (unguarded).
3. Chunks 2–6: c1 accumulates instances [6@6, 6@5, ..., 6@1].  c0 keeps
   incrementing.
4. Chunk 7: c1's oldest instance (value 6) increments to 7.  `can_break(7, 7)
   = true`.  `break_is_match_at_end=true` → `self.match_at_end = true`.
5. Chunks 8–49: mae stays true.  c1 maintains 6 live instances (oldest
   breaks each step, new one seeded via pre_seeds).  c0 exhausts at
   chunk 25 (value 26 = max), but c1 self-perpetuates through pre_seeds.
6. `finish()`: `match_at_end=true` → reports match.  Wrong.

## Investigation Narrative

1. Reproduced with `--debug --chunk-size 1 --tier 3 --unroll-limit 0`.
   Saw `mae=true` first appear at chunk 7.  NFA (`--tier 0`) showed
   NO MATCH throughout.

2. Examined the dump output: c1's `break_is_match_at_end: true` confirmed
   that c1 breaking would set mae.  `break_seeds: trigger:c0 → seed c1
   at origin:6 (gated by \B@5)` showed the legitimate seeding path.

3. Tracked c1 instance counts: 0 at chunk 0, 1 at chunk 1, growing to 6
   and stabilizing.  The steady-state of 6 instances with the oldest
   breaking each step pointed to continuous re-seeding.

4. Identified three possible seed sources: pre_seeds (before counter
   loop), seeds (after counter loop, Bug 32 guarded), and break_seeds
   (gated on counter_broke).  The Bug 32 guard filters seeds against
   `clean_nb_trans_slot`, but pre_seeds had no such guard.

5. Confirmed by reading `populate()` at line 1636-1652: pre_seeds come
   from resolved seeds where the origin has an Increment action AND
   `reachable_without_break[origin]` is true.  State 6 passes both
   checks (it's c1's body, structurally reachable without break).

6. Verified the clean_nb chain (DfaStateId(1) = `{1}`) would NOT produce
   c1 pre_seeds since state 6 is not in the clean closure.

## Fix

Added a contamination guard to pre_seeds application in the
`step_slow_impl` macro (line ~2846), mirroring the Bug 32 guard for
unconditional seeds:

```rust
let pre_seed_contaminated =
    self.current_has_break_extras && self.regex.num_counters > 1;
for &(counter, origin, value) in t.pre_seeds.iter() {
    if pre_seed_contaminated {
        if let Some(cn_slot) = self.clean_nb_trans_slot {
            let cn_t = &self.cache.transitions[cn_slot];
            if !cn_t.pre_seeds.iter().any(|s| s.0 == counter && s.1 == origin) {
                continue;
            }
        }
    }
    self.$current.seed(counter.idx(), origin, value);
}
```

When the current DFA state is contaminated (has break extras, >1
counter), pre_seeds are filtered against the clean no-break chain's
pre_seeds.  Only seeds that also appear in the clean chain are truly
counter-free and applied.

## What Was Hard

- **Pre_seeds vs seeds distinction**: The Bug 32 guard existed for
  unconditional seeds but not pre_seeds.  Pre_seeds are a separate code
  path (applied before the counter loop) added for L=1 body counters
  where the seed resolves within the same transition.  It was easy to
  miss that the same contamination logic needed to apply to both paths.

- **Structural vs contextual reachability**: `reachable_without_break`
  is a structural property of the NFA — state 6 is always reachable
  from CI(c1) without going through a break.  But in a specific DFA
  state, it may only be present because of a break.  The Bug 40 filter
  catches origins that are ONLY reachable via break paths, but not
  origins that are structurally break-free yet contextually
  break-dependent.

## Tooling Ideas

- **Pre_seeds in debug trace**: The debug trace shows `pending_seeds`
  (deferred break seeds) but doesn't explicitly show pre_seeds being
  applied.  Adding a line like `pre_seeds: [(c1, 6, val=0)]` to the
  trace output would have made the re-seeding source immediately
  visible.

- **Contamination annotations**: Mark seeds/pre_seeds in the trace with
  a contamination indicator (e.g. `[C]` for contaminated, suppressed
  by the guard) so the filtering is visible.
