# Bug 33 — OOM in epsilon walk due to missing visited set

## Bug Summary

- **Pattern:** `^[01]{10,42}(([01]|\b|\B))*((c{1,6}|[01]?))?$`
- **Input:** Various (e.g., `"0101111100010"`)
- **Expected:** Terminates in bounded time
- **Actual:** OOM / infinite loop in `can_reach_match_mid()` and
  `can_reach_match_at_end()`
- **Affected tiers:** All tiers that use deferred assertion resolution
  (Tier 1+ via `can_reach_match_at_end` in `dfa/mod.rs`, Tier 3 via
  `can_reach_match_mid` in `dfa/tier3.rs`)
- **Severity:** Resource exhaustion (not a correctness bug)

## Root Cause

Both `can_reach_match_at_end` (`src/dfa/mod.rs:159`) and
`can_reach_match_mid` (`src/dfa/tier3.rs:3293`) perform a stack-based
epsilon walk over the NFA graph to determine if a `Match` state is
reachable through epsilon transitions (Split, Assert, CounterInstance)
from a given start state.  Neither function maintained a **visited set**,
so they revisited states indefinitely when the NFA contained epsilon
cycles.

### How epsilon cycles form

The pattern `(([01]|\b|\B))*` creates:

```
State 9: Split → 8 | 24      (the * loop)
State 8: Split → 6 | 7       (outer alternation)
State 6: Split → 4 | 5       (inner alternation: [01] vs \b)
State 5: Assert(WordAscii) → 9    ← back to state 9!
State 7: Assert(WordAsciiNegate) → 9  ← also back to 9!
```

The cycle is: `9 → 8 → 6 → 5 → 9` (via `\b`) or `9 → 8 → 7 → 9`
(via `\B`).  At any position in the input, either `\b` or `\B` passes
(they are complementary), so the assertion evaluation always allows the
walk to follow at least one path back into the cycle.

All states in the cycle have `state_can_reach_match = true` (they can
reach `Match` through the `$` assertion at state 25), so the early-exit
optimization doesn't help.

### Why it OOMs instead of just looping

The stack-based walk pushes multiple children per `Split` state:
```rust
State::Split { out, out1 } => {
    stack.push(out);
    stack.push(out1);
}
```

In the cycle, each visit to state 9 pushes states 8 and 24.  State 8
pushes 6 and 7.  States 6 and 5/7 push back to 9.  The stack grows
exponentially with each cycle iteration, consuming memory until OOM.

## Investigation

1. **Artifact discovery:** `cargo +nightly fuzz run fuzz_differential`
   produced artifact `oom-ad2742c19178952a55ded5f020c1093d220ccb33`.

2. **Stack trace analysis:** The libFuzzer OOM trace pointed directly at
   `can_reach_match_mid` at `tier3.rs:3314` — the `stack.push(out1)` in
   the `Split` arm.

3. **Pattern decoding:** Used a temporary test to decode the fuzz seed
   into the pattern and inputs via `generate_pattern`/`generate_inputs`.

4. **NFA dump:** `cargo run --release -- dump` revealed the epsilon cycle
   through states 5→9→8→6→5 and 7→9→8→7.

5. **Root cause:** Immediate — both walk functions lacked visited sets.
   Confirmed that `can_reach_match_at_end` in `dfa/mod.rs` had the
   identical structure and identical bug.

## Fix

Added a `visited: Vec<bool>` bitvec (indexed by NFA state index) to both
functions.  Each state is visited at most once, guaranteeing O(n)
termination where n is the number of NFA states.

**Files changed:**
- `src/dfa/mod.rs`: `can_reach_match_at_end` — added visited set
- `src/dfa/tier3.rs`: `can_reach_match_mid` — added visited set

The `Vec<bool>` allocation is small (one byte per NFA state, typically
< 100 states) and only happens during deferred assertion resolution,
not on every byte.

## What Was Hard

Nothing — this was a straightforward missing-visited-set bug.  The stack
trace pointed directly at the problematic function, and the NFA dump
clearly showed the epsilon cycle.  The fix was mechanical.

## Tooling Ideas

- The `dump --dfa` output could flag NFA epsilon cycles (cycles through
  Split/Assert/CounterInstance edges) as a diagnostic aid.
- A compile-time check could detect patterns where `(\b|\B)*` or similar
  always-passing assertion loops exist and warn about them, though such
  patterns are valid regex.
