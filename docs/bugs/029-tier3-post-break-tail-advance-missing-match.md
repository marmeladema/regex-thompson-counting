# Bug 29: Tier 3 post-break tail Advance branch missing direct match check

## Bug summary

- **Pattern:** `^c{3,5}c{1,5}cee?`
- **Input:** `ccccce` (5 c's + 1 e)
- **Expected:** MATCH (`ccc` + `c` + `c` + `e` + empty `e?`)
- **Actual:** NO MATCH (Tier 3 false negative)
- **Affected tier:** Tier 3 (per-instance path), when a post-break tail consumes
  a byte and its target epsilon-closure includes the Match state directly (not
  via `$ → Match`)

## Root cause

The post-break tail loop in `step_slow_impl!` handles three cases when a tail
encounters a transition:

1. **Advance** (line ~2444): tail consumed the byte, advances to new origins.
2. **Increment**: tail hit a CInc, hands off to counter machinery.
3. **None** (dead): tail didn't consume, check `target_is_match_at_end` and
   `target_is_match`.

The `None` branch correctly checks both `target_is_match_at_end` (for `$ → Match`)
and `target_is_match` (for direct `Match`), as fixed in Bug 25.  However, the
`Advance` branch only checked `target_is_match_at_end` — the `target_is_match`
check was never added.

For the pattern `^c{3,5}c{1,5}cee?`:
- c0 is `c{3,5}`, c1 is `c{1,5}`.  After c1 breaks, the post-break path is:
  state 7 (`Byte('c')`) → state 8 (`Byte('e')`) → state 10 (`Split → 9 | 11`).
- State 11 is `Match`.  So `target_is_match[8] = true`.
- When the tail at state 8 consumes 'e', it **advances** to `new_origins = [9]`
  (Byte('e') for the optional `e?`).  The Advance branch adds state 9 to
  `next_post_break_tails` and checks `target_is_match_at_end[8]` (false), but
  never checks `target_is_match[8]` (true).
- The direct match at state 11 is lost.

Note: if the tail had been killed (None branch) instead of advancing, the
match would have been detected.  The bug only manifests when the tail
*advances* (produces new_origins) while *also* having a direct Match in
its target epsilon-closure.

## Fix

Added `target_is_match` check in the Advance branch of the post-break tail
loop, mirroring the existing check in the None branch (Bug 25):

```rust
// In the Advance branch, after target_is_match_at_end check:
if self.analysis.target_is_match[pbo.idx()] {
    self.ever_matched = true;
}
```

One-line fix (plus comment).

## Investigation narrative

1. **Fuzz discovery.** `cargo +nightly fuzz run fuzz_differential` produced
   crash artifact `crash-7ba04bb92ca51ebce8e028e0d8ce04647e86e27d`.  Decoding:
   pattern `^c{3,5}c{1,5}cee?`.

2. **Input sweep.** Tested `ccccce`, `cccccce`, `ccccccee`, `cccce`, etc.
   Found `ccccce` (5c+1e) and `cccccce` (6c+1e) fail — NFA=MATCH, Tier 3=
   NO MATCH.  Inputs with 2 e's at the end (`ccccccee`, `cccccee`) matched
   correctly.

3. **Debug trace.** `--debug --tier 3 --chunk-size 1` on `ccccce` showed:
   - After chunk #4 ('c'): `tails: [8,7]` — both post-break tails alive.
   - After chunk #5 ('e'): `tails: [9]`, `is_match` on DFA state, but
     `matched=false`.
   - The DFA state correctly had `is_match` (state 11 in epsilon closure),
     but `ever_matched` was never set.

4. **NFA comparison.** NFA trace showed `matched=true` after 'e' — state 10
   (Split → 9|11) includes state 11 (Match).

5. **Code inspection.** Traced `ever_matched` assignments.  The post-break
   tail loop's Advance branch (line 2444) only checked `target_is_match_at_end`
   — the `target_is_match` check that was added in Bug 25's None branch was
   never carried over to the Advance branch.

6. **Pattern analysis.** The 2-e inputs worked because after consuming the
   first 'e' at state 8, the tail at state 9 consumed the second 'e' and
   died (None branch) — triggering the Bug 25 `target_is_match` check for
   state 9 (which also leads to Match via Split at 10).

## What was hard

- **Not hard at all.** This was a classic omission bug — the direct-match
  check existed in one branch but not the other.  The investigation was
  straightforward once the debug trace showed `is_match` without
  `ever_matched`.  Total investigation time: ~10 minutes.

## Tooling ideas

- **Audit similar branches for consistency.** The post-break tail loop has
  three branches (Advance, Increment, None) that share similar logic.  A
  linting pass or code review should ensure all three handle the same set
  of match conditions.  Consider extracting a shared helper function to
  avoid future omissions.
