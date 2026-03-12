# Bug 39 — Dead target loses match_at_end for ByteTable tails

## Bug summary

| Field | Value |
|-------|-------|
| Pattern | `^(.{6,36}a{0,4}e*c)?$` |
| Input | `"xxxcbd ccac"` (len=11) |
| Expected | `true` |
| Actual | `false` (Tier 3 false negative) |
| Affected tier | Tier 3 (default unrolling) |
| Artifact | `crash-2d6c42a763b5e92e2a342f0d5f9bc8e12389f64b` |

## Root cause

With default unrolling, `a{0,4}` is expanded into four optional `Byte('a')`
states plus `ByteTable` merge nodes (states 8-11).  Only one counter remains
(`c0: {6,36}` for `.{6,36}`), making the pattern Tier 2 eligible.  State 11
is a `ByteTable` with entries `'a'→10, 'c'→16, 'e'→13`.

After c0 breaks, state 11 enters `post_break_tails`.  When the tail
processes byte `'c'`, state 11 maps `'c'→16`.  State 16 is `Assert(End) →
Match`.  The epsilon walk from state 16 finds no consuming states (it
reaches only `Match`), so `analyze_target(16)` returned `None` ("dead
target").

In the tail processing loop, the `Some(None)` branch called
`check_tail_match_flags!`, which checked the **static**
`target_is_match_at_end[11]` array.  This array was computed at build time
for each consuming state, but **skipped ByteTable states** (they have
per-byte targets, so a single static flag is unsound).  The result was
always `false` for ByteTable origins, so `match_at_end` was never set.

The `analyze_target` function already computed `advance_is_match_at_end =
true` (correctly detecting `Assert(End) → Match`), but this value was
discarded in the `advance_origins.is_empty()` branch that returned `None`.

## Fix

Changed `analyze_target` to return `Some(Advance { new_origins: [],
is_match_at_end: true })` instead of `None` when the epsilon walk reaches
`$ → Match` despite having no consuming states downstream.  The `Advance`
variant carries the byte-specific `is_match_at_end` flag (added in Bug 37),
so the tail processing's `Advance` branch correctly sets `match_at_end`.

An `Advance` with empty `new_origins` produces no new tails — the only
effect is the `is_match_at_end` flag propagation.

## Investigation narrative

1. Reproduced: `--tier 0` (MATCH) vs `--tier 3` (NO MATCH) on the
   default-unrolled pattern.  `--tier 3 --unroll-limit 0` (different NFA)
   correctly matched — the bug only affected the unrolled version.

2. Checked `--tier 2` and `--tier 4`: both matched correctly.  Only Tier 3
   had the false negative.

3. Compared debug traces.  NFA showed state 16 (`Assert(End)`) active
   after consuming 'c' at each relevant position; Tier 3 trace showed
   `tails: [11]` but no `mae` flag.

4. Dumped the Tier 3 analysis: `target_is_match_at_end: [14]` — only
   state 14 (`Byte('c')`) was listed.  State 11 (ByteTable) was absent
   because the static array skips ByteTable states.

5. Traced the code path: `analyze_target(16)` found no consuming states
   → returned `None` → tail hit `Some(None)` → `check_tail_match_flags!`
   → static `target_is_match_at_end[11]` = false.

6. Identified that `advance_is_match_at_end` was correctly computed as
   `true` inside `analyze_target` but discarded in the `None` return.

7. Changed the `advance_origins.is_empty()` branch to return `Advance`
   when `advance_is_match_at_end` is true.

## What was hard

The pattern was initially confusing because the fuzz crash said "Tier 3"
but the `info` output showed "Tier 2".  The fuzz target tests all eligible
tiers, and this pattern's unrolled form has both Tier 2 and Tier 3 eligible
— the bug was in the Tier 3 code path applied to the unrolled NFA.

The ByteTable skip in static arrays is a recurring theme (Bugs 35, 37, 39).
Each instance requires a different fix because the information flows through
different code paths.

## Tooling ideas

- Add a static analysis pass that flags ByteTable states where any byte
  maps to a target reachable through `$ → Match` but the static array says
  `false`.  This would catch this class of bugs at compile time.
- The `dump --dfa` output could highlight cases where a target returns
  `None` from `analyze_target` but `advance_is_match_at_end` was true,
  making the information loss visible.
