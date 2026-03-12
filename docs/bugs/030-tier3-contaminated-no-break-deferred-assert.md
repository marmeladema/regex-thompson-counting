# Bug 30: Tier 3 contaminated no_break_current deferred assert false positive

## Bug Summary

- **Pattern:** `^.{0,44}1{3,38}f\ba?$`
- **Input:** `a11f`
- **Expected:** NO MATCH (c1 counted only 2 '1's, needs ≥3 to break)
- **Actual (before fix):** MATCH (Tier 3 false positive)
- **Constraint:** `111f` must still MATCH (c1 counted 3, breaks correctly)
- **Affected tier:** Tier 3 (per-instance path)
- **Found by:** `fuzz_differential` (`crash-3aacd6ec41806d3681ce66291f017f3c3b4f21fa`)

## Root Cause

The NFA for `^.{0,44}1{3,38}f\ba?$` has:

```
 0: Assert(Start) → 4
 1: ByteClass(cls:0) → 2        [c0 body: any byte]
 2: CInc(c0, {1,44}) → cont:1 | break:7
 4: Split → 3 | 7
 7: CI(c1) → 5
 5: Byte('1') → 6               [c1 body: '1']
 6: CInc(c1, {3,38}) → cont:5 | break:8
 8: Byte('f') → 9               [c1 break consuming state]
 9: Assert(WordAscii) → 11      [\b deferred assert]
11: Split → 10 | 12
10: Byte('a') → 12
12: Assert(End) → 13
13: Match
```

When c0 breaks, the DFA takes the `with_break` successor. Because
`break_seeds` contains `trigger:c0 → seed c1 at origin:5`, c1's body is
seeded into the DFA state. But the `with_break` DFA state also picks up
state 8 (`Byte('f')`) from c1's break-path epsilon closure — this is
**contamination** (state 8 is only legitimately reachable when c1 actually
breaks with count ≥ 3).

The epsilon closure from state 8 reaches state 9 (`Assert(WordAscii)`),
which becomes a **deferred assert** in the DFA state's `deferred_asserts`
array. This deferred assert propagates into `no_break_current`.

In `finish()`, `no_break_current`'s deferred asserts are resolved at
end-of-input. For `a11f`:
- `prev` byte = `f` (word char)
- `next` = None (end of input, non-word)
- `\b` passes (word → non-word boundary)
- State 9's `out` (state 11) reaches `Assert(End) → Match`
- Result: **false positive**

But c1 only counted 2 '1's (below min 3), so it should never have broken.
The `\b` in `no_break_current` is a phantom from contamination.

### Why this is hard

The deferred `\b` serves **double duty**:

1. **Legitimate** for `111f`: c1 breaks (count ≥ 3), the post-break tail
   advances through state 8, consuming 'f'. State 9 (`\b`) is on the
   epsilon path from state 8's target. The `\b` must fire at EOI.

2. **Spurious** for `a11f`: state 8 is in the DFA state from contamination
   (c0 break → epsilon closure includes c1's break path). The same `\b`
   fires even though c1 never broke.

Both cases produce the same `\b` deferred assert in `no_break_current`.
We cannot simply filter it out (that breaks the legitimate case), and we
cannot leave it in (that causes the false positive).

### Failed approaches

1. **Use `clean_nb` instead of `no_break_current`:** The clean chain
   excludes break-gated states, so the `\b` at state 9 is absent. This
   fixes `a11f` but breaks `111f` — the legitimate `\b` is also absent
   from `clean_nb`.

2. **Filter by `clean_nb` intersection:** Only resolve deferred asserts
   present in both `no_break_current` and `clean_nb`. Same problem — the
   legitimate `\b` is not in `clean_nb` because state 8→9 is break-gated.

## Fix

Two-part fix that separates the contaminated DFA-level path from the
counter-verified per-instance path:

### Part 1: Precomputed `target_deferred_asserts` (build time)

Added `target_deferred_asserts: Box<[Box<[StateIdx]>]>` to `Tier3Analysis`.
For each consuming NFA state, this records the non-End Assert state indices
that lie on the epsilon path from its byte-consumption target. Computed in
`compute_tier3_analysis()` Step 6c, parallel to `target_is_match` and
`target_is_match_at_end`.

For state 8 (`Byte('f')`): target is state 9 (`Assert(WordAscii)`), which
is a non-End assert on the path → `target_deferred_asserts[8] = [9]`.

### Part 2: Deposit into `verified_deferred_asserts` (runtime)

In the post-break tail `Advance` branch of `step_slow_impl!`, when a tail
at origin `pbo` advances: deposit `target_deferred_asserts[pbo.idx()]` into
`verified_deferred_asserts`. This ensures the `\b` goes through the
per-instance counter-gated path (which already handles mid-input and EOI
resolution via `resolve_verified_deferred_asserts`).

### Part 3: Skip contaminated `no_break_current` deferred asserts (EOI)

In `finish()`, when contaminated (`current_has_break_extras && num_counters > 1`):
- Skip `no_break_current`'s deferred asserts entirely (they may contain
  spurious break-gated asserts).
- Fall back to `clean_nb`'s deferred asserts for counter-free assertions.
- Legitimate counter-break deferred asserts are handled by
  `verified_deferred_asserts` (resolved earlier in `finish()`).

## Investigation Narrative

1. Fuzz artifact decoded to pattern `^.{0,44}1{3,38}f\ba?$`, input `a11f`.
2. NFA returns NO MATCH, Tier 3 returns MATCH → false positive.
3. `--debug --tier 3` trace showed `\b@9` in the DFA state's deferred
   asserts, resolving at EOI.
4. `dump --dfa` revealed state 8 is a `break_consuming_state` of c1,
   only reachable when c1 breaks. But it's in the DFA state from c0's
   break contamination.
5. First fix attempt (use `clean_nb`): fixed `a11f` but broke `111f`.
   `clean_nb` doesn't have the `\b` because it's behind break-gated
   state 8.
6. Second fix attempt (filter by `clean_nb` intersection): same problem.
7. Key insight: the legitimate `\b` comes through the **post-break tail**
   (state 8 advancing after c1 breaks), not through the DFA state's
   epsilon closure. The tail mechanism already feeds into
   `verified_deferred_asserts` (Bug 26). What's missing is recording that
   the `\b` is on state 8's target path.
8. Solution: precompute `target_deferred_asserts` at build time, deposit
   at runtime in the tail `Advance` branch, suppress contaminated DFA
   deferred asserts in `finish()`.

## What Was Hard

The dual nature of the deferred assert was the core difficulty. The same
`\b` at state 9 appears in `no_break_current` via two completely different
mechanisms:

- **Contamination path:** c0 break → epsilon closure includes c1 break
  path → state 8 → state 9. This is DFA-level, not counter-gated.
- **Legitimate path:** c1 actually breaks → post-break tail at state 8
  consumes 'f' → state 9 is on the target epsilon path. This is
  per-instance, counter-gated.

The first two fix attempts tried to filter at the DFA level, which cannot
distinguish these paths. The solution required moving the legitimate assert
into the per-instance path (via `target_deferred_asserts` +
`verified_deferred_asserts`) and suppressing the DFA-level path entirely
when contaminated.

## Tooling Ideas

- `dump --dfa` could show `target_deferred_asserts` alongside
  `target_is_match` and `target_is_match_at_end` to make the assertion
  paths visible without manual NFA tracing.
- `--debug` Display for Tier 3 could annotate which deferred asserts in
  the DFA state are break-gated (from contamination) vs counter-free,
  helping distinguish the two paths during investigation.
