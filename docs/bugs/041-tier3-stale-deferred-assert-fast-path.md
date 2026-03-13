# Bug 41: Stale verified_deferred_asserts surviving the inline fast path

## Bug Summary

- **Pattern:** `\bc{2,12}\b`
- **Input:** `"cccccccccccczzz"` (12 c's + "zzz")
- **Expected:** NO MATCH (no word boundary between c and z)
- **Actual (Tier 3, unroll_limit=0):** MATCH (false positive)
- **Affected tier:** Tier 3 (also Tier 2 when forced via `--tier 3`)
- **Artifact:** `fuzz/artifacts/fuzz_match/crash-edb2ccb64b91fc1e68220f083bf6603a2527e834`

## Root Cause

When a counter breaks and the break path contains a deferred assertion
(e.g. `\b`), the assertion's NFA state index is deposited into the
`verified_deferred_asserts` list during `step_slow_impl`.  This list is
designed to be resolved on the NEXT byte (Bug 26 mechanism): at the top
of the per-byte loop in `chunk()`, `resolve_verified_deferred_asserts()`
evaluates each deferred assertion with the new byte as context.

The problem: `verified_deferred_asserts` is only cleared inside
`step_slow_impl` (line ~2551).  The **inline fast path** in `chunk()`
(lines ~3276-3311) — taken when there are no live counter instances, no
seeds, and no post-break tails — bypasses `step_slow_impl` entirely and
**never clears the list**.

Flow for the buggy input `"cccccccccccczzz"`:

1. **Byte 11 (12th 'c'):** Counter c0 hits max (12), breaks.  The break
   path goes through `\b@4 → Match`.  `\b@4` is deposited into
   `verified_deferred_asserts`.  No post-break consuming states (Match is
   epsilon-reachable), so `post_break_tails` is empty.

2. **Byte 12 ('z'):** `resolve_verified_deferred_asserts(false, Some(b'z'))`
   evaluates `\b` with prev='c', next='z'.  Both are word characters →
   `\b` **fails**.  `ever_matched` stays false.  Good so far.

   Then the fast-path condition is checked: `!is_counting && no seeds &&
   !has_live_instances && no post_break_tails` → all true.  Fast path
   taken.  **`verified_deferred_asserts` is NOT cleared.**

3. **Bytes 13-14 ('z', 'z'):** Same as byte 12.  Each time,
   `resolve_verified_deferred_asserts` evaluates `\b` with prev='z',
   next='z' → fails.  Fast path taken.  List never cleared.

4. **EOI (`finish()`):** `resolve_verified_deferred_asserts(true, None)`
   evaluates `\b` with prev='z', at_end=true → `\b` **passes** (word →
   end-of-input is a word boundary).  `can_reach_match_at_end(state 5)`
   finds `Match`.  Returns true → **false positive**.

The deferred `\b` assertion from the counter break at position 12 was a
one-shot event: the c→z boundary was the only moment it should have been
evaluated.  When it failed there, the match opportunity was lost.  But
because the list was never cleared, the assertion lingered until EOI where
the different byte context (z→end) caused it to pass.

The same bug class also affected `pending_break_seeds`, which captures
break seed events with deferred assertions.  These are also one-shot: the
`prev_was_word` flag captures the context at break time, but the `next`
byte changes on every evaluation.  Without clearing after resolution, a
stale pending break seed could be re-evaluated on a later byte with a
different `next` context, potentially incorrectly seeding a counter.

## Fix

Added unconditional `self.verified_deferred_asserts.clear()` and
`self.pending_break_seeds.clear()` in `chunk()` immediately after their
resolution, before the fast-path check.  This ensures one-shot resolution
regardless of whether the slow path or fast path runs.  The existing
`clear()` in `step_slow_impl` is now redundant but harmless.

Two lines changed in `src/dfa/tier3.rs`, in the `chunk()` method's
per-byte loop.

## Investigation Narrative

1. Reproduced with `--debug --chunk-size 1 --tier 3 --unroll-limit 0`.
   The trace showed `deferred: [\b@4]` persisting through all 'z' bytes
   and into EOI, where the final `MATCH` appeared.

2. Compared against NFA oracle (`--tier 0`): correctly NO MATCH for
   inputs with trailing word characters after 12 c's.

3. Tested boundary cases: `"cccccccccccc"` (12 c's, no suffix) correctly
   matched (deferred `\b` resolves at EOI as word→end = boundary).
   `"ccccccccccccz"` was the minimal false positive.

4. Traced the code path: `resolve_verified_deferred_asserts` at line 3155
   correctly fails for 'z' bytes.  The issue was the list surviving
   across steps.  Searched for `verified_deferred_asserts.clear()` — found
   it only inside `step_slow_impl`.  The fast path (lines 3276-3310) has
   no clear, confirming the diagnosis.

5. The fix was straightforward: clear after resolution.  Also applied the
   same fix to `pending_break_seeds` which had the identical fast-path
   leakage vulnerability.

## What Was Hard

Nothing — this was a clean, isolated bug with a clear symptom (stale
state surviving across steps).  The `--debug` trace immediately showed
the `deferred: [\b@4]` list persisting when it shouldn't have been.
The fix was obvious once the mechanism was understood.

## Tooling Ideas

- The `--debug` trace could highlight when `verified_deferred_asserts`
  entries survive unchanged across steps (e.g. `deferred: [\b@4] (stale!)`
  when the list wasn't modified during the step).  This would immediately
  flag the "not cleared on fast path" issue.
- A debug assertion could verify that `verified_deferred_asserts` is empty
  at the start of each byte's processing (after resolution), catching
  any future code paths that fail to clear it.
