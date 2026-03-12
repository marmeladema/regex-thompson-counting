# Bug 35 — Tier 3 false negative: ByteTable skipped in break_consuming_tails

## Bug Summary

- **Pattern:** `^(c{2,48}c?y?b1e{0,0})?((a?a?)?(a?a?)?)?$`
- **Input:** `"cccccccccccccccccccccccccccccccccccccccccyb1"` (41 'c's + "yb1")
- **Expected:** true (NFA matches)
- **Actual:** Tier 3 returns false
- **Affected tiers:** Tier 3 (conditional transitions)
- **Type:** False negative

## Root Cause

When counter c0 (`c{2,48}`) breaks, the break path leads to NFA state 5,
which is a `ByteTable` mapping 'b'→9, 'c'→7, 'y'→8.  This ByteTable
represents the `c?y?b1` suffix after the counter body.

The compile-time `break_consuming_tails` function walks from the CInc
break output through epsilon transitions, collecting consuming states to
use as `post_break_tails`.  It handled `Byte`, `ByteCI`, and `ByteClass`
but had a special case for `ByteTable`:

```rust
State::ByteTable { .. } => {
    // ByteTable has variable targets; conservatively skip.
}
```

This meant ByteTable state 5 was never included in `break_consuming_states`.
When c0 broke at runtime, no post_break_tails were created.  The entire
downstream consuming chain (state 5 → 7/8 → 8 → 9 → 21) was lost.

Without tails tracking state 9 (`Byte('1') → 21`), the
`check_tail_match_flags!` macro never checked `target_is_match_at_end[9]`
(which is true — state 21 leads to `$→Match`).  The `match_at_end` flag
was never set, and `counter_free_nb_mae` was false (all origins are
counter-dependent).  Result: false negative at EOI.

### Why conservative skip was wrong

The original comment said "ByteTable has variable targets" — meaning
different bytes map to different NFA states, making it unclear which
single `targets[out.idx()]` to check for the Increment filter.

But the runtime tail tracking handles ByteTable correctly.  The DFA
transition's `origin_keys`/`origin_actions` mechanism computes the
byte-specific target at populate time.  The `check_tail_match_flags!`
macro uses precomputed per-origin flags that work regardless of the
consuming state type.  Skipping ByteTable was unnecessary.

## Investigation

1. **Artifact:** `crash-c2c273973511bb033ee680d892cc82400463837c` from
   `fuzz_differential`.  Tier 3 = false, NFA = true.

2. **NFA dump:** Confirmed state 5 is ByteTable for `c?y?b1`, state 9
   has `target_is_match_at_end = true`, and `counter_break_can_match[c0]
   = false` (Match only reachable through consuming states).

3. **Byte-by-byte trace:** Tier 3 trace showed no tails at any point.
   After c0 broke and the break extras consumed all remaining bytes,
   the final state had `match_at_end = false` with no mechanism to
   detect the `$ → Match` path.

4. **Root cause:** `break_consuming_tails` at line 2367 — ByteTable
   `=> {}` (empty match arm, skipping the state entirely).

## Fix

Include ByteTable states in `break_consuming_tails` unconditionally,
matching the treatment of `Byte`, `ByteCI`, and `ByteClass` states.

**File:** `src/dfa/tier3.rs`, `break_consuming_tails` function.

The runtime tail tracking already handles ByteTable correctly through
the DFA transition mechanism, so no runtime changes were needed.

## What Was Hard

The investigation was straightforward once the root cause of Bug 34 was
understood — both bugs involve the post_break_tails mechanism.  The
`break_consuming_tails` function was small (30 lines) and the ByteTable
skip was clearly marked with a comment.

## Tooling Ideas

- The `dump --dfa` output could show `break_consuming_states` for each
  Increment target, making it visible when a ByteTable or other state
  type is unexpectedly absent.
- A compile-time warning when `break_consuming_states` is empty for a
  counter whose break path has consuming states (detectable via
  `state_can_reach_match` from the break output) would catch this class
  of bugs.
