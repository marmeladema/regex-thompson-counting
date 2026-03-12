# Bug 37: ByteTable tail target_is_match_at_end skipped

## Bug summary

- **Pattern:** `^0{3,43}((a?c)*x(a?a?)?(a?a?)?)?$`
- **Input:** `"000x"` (len=4)
- **Expected:** match (NFA=true)
- **Actual:** Tier 3 false negative (Tier 3=false)
- **Affected tier:** Tier 3 (per-instance path)
- **Artifact:** `fuzz/artifacts/fuzz_differential/crash-42b85b2fb609bd1065e48ba290f2ebd9d8bb597d`

## Root cause

The static `target_is_match_at_end` array, computed during Tier 3
analysis (Step 6), explicitly skipped `ByteTable` states with the
comment "overapproximation would be unsound, and the tail tracker
won't encounter ByteTable origins in practice for tier 3 patterns."

After Bug 35's fix (include ByteTable in `break_consuming_tails`),
ByteTable origins CAN appear as post_break_tails.  The pattern
`^0{3,43}((a?c)*x(a?a?)?(a?a?)?)?$` has ByteTable state 7 with entries
`'a'→6, 'c'→7, 'x'→13` on the counter break path.  When c0 breaks
after consuming 3 zeros, state 7 becomes a post_break_tail.

When tail [7] consumed `'x'`, the Advance action correctly produced
`new_origins: [9, 11, 14, 16]`.  But `check_tail_match_flags!(self, 7)`
checked `target_is_match_at_end[7]` which was `false` (ByteTable skipped).
The `$ → Match` path through state 13 → optional groups → `Assert(End)`
was never detected, causing a false negative at end-of-input.

### Why a simple OR-union over ByteTable entries is unsound

The static `target_is_match_at_end` is a per-origin flag (not per-byte).
For ByteTable state 7:
- `'a' → 6` (Byte 'c'): does NOT reach `$ → Match`
- `'c' → 7` (ByteTable): loops back, does NOT reach `$ → Match`
- `'x' → 13` (Split): DOES reach `$ → Match`

An OR-union would set `target_is_match_at_end[7] = true`, but when
tail [7] consumes `'a'`, the code would falsely set `match_at_end = true`
even though `'a'`'s target doesn't reach `$ → Match`.  This is a false
positive.

### The fix: byte-specific `is_match_at_end` on Advance actions

The Advance actions in `origin_actions` are already computed per-byte
during `populate()`.  The `analyze_target()` function walks the epsilon
closure from the byte-specific target.  By computing `is_match_at_end`
within `analyze_target()` and storing it on the `Advance` variant, the
flag is automatically byte-specific.

The `$ → Match` walk in `analyze_target()` uses a separate epsilon
traversal (mirroring the static `target_is_match_at_end` algorithm):
it follows Split and CI but does NOT follow non-End Assert states
(e.g. `\b`, `\B`).  Only `Assert(End)` with `can_reach_match` sets
the flag.  This prevents false positives from patterns like
`^a{3,34}b{1,6}\B$` where `\B` blocks the `$ → Match` path.

## Investigation narrative

1. Reproduced with `cargo run --release -- match --debug --chunk-size 1
   --tier 3 --unroll-limit 0 '^0{3,43}((a?c)*x(a?a?)?(a?a?)?)?$' '000x'`.

2. Compared Tier 3 and NFA traces.  NFA after chunk 3 ('x') has
   state 20 (Assert End) in its thread set.  Tier 3 after chunk 3
   has tails [9,11,14,16] but no `mae` flag.

3. Checked `dump --dfa`: `target_is_match_at_end: [8, 9, 11, 14, 16]`.
   State 7 (ByteTable) is NOT in this list.

4. Read the static `target_is_match_at_end` computation (lines 533-578):
   ByteTable is explicitly skipped with a comment about unsoundness.
   This was correct before Bug 35, but Bug 35 made ByteTable tails
   possible, invalidating the assumption.

5. First attempt: added `is_match_at_end` and `is_match` fields to
   `Advance` variant, computed inline in `analyze_target`'s DFS.  This
   caused 7 test failures — false positives from `is_match` (the DFS
   follows Assert states, reaching Match behind Assert(End), which
   conflates `is_match` with `is_match_at_end`).

6. Second attempt: removed `is_match` (kept using static array), fixed
   `is_match_at_end` computation to match static array semantics (don't
   follow non-End Assert states).  Still 3 failures — the `is_match_at_end`
   walk was following non-End asserts (the main DFS pushes `out` from
   ALL Assert states for collecting consuming states, but the mae check
   was inline in the same walk).

7. Final fix: separate epsilon walk for `is_match_at_end` that mirrors
   the static algorithm exactly (follow Split/CI only, stop at Assert
   unless it's Assert(End)).  All 453 tests pass.

## What was hard

Three-layer interaction between Bug 35's fix (ByteTable in
`break_consuming_tails`), the static `target_is_match_at_end` array
(which skipped ByteTable), and the tail processing code (which relied
on the static array).  The fix also had to handle non-End Assert states
correctly — the initial attempt that followed all Assert states caused
false positives on patterns with `\B$` or `\b\B` chains.  Each
iteration of the fix required running the full 453-test suite to catch
regressions.

## Tooling ideas

- The `dump --dfa` output could flag when `target_is_match_at_end`
  skips a state that appears in `break_consuming_states`, making the
  gap between the two arrays visible.
- A debug assertion in `check_tail_match_flags!` could cross-check the
  static array against the byte-specific Advance flags for non-ByteTable
  origins, catching any divergence between the two paths.
