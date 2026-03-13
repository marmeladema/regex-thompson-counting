# Bug 49: Shorter deferred paths not explored in break_consuming_tails

## Bug Summary

- **Pattern**: `^.{7,36}\B(\ba{0,0})?((a?a?)?(a?a?)?)?$`
- **Input**: `"aaaaaaaaaaaaaaaa"` (16 a's)
- **Expected**: `true` (NFA matches)
- **Actual**: `false` (Tier 3 false negative)
- **Affected tier**: Tier 3 (per-instance path)
- **Fuzz artifact**: `fuzz/artifacts/fuzz_differential/crash-4d53cacfa730ee947eacbc4d939a8b0a65639014`

## Root Cause

The `break_consuming_tails()` function walks the NFA from counter break
points to find consuming states (post-break tails), collecting deferred
assertions along each path.  Its visited-state tracking used a 3-state
flag: 0=unvisited, 1=visited-deferred, 2=visited-pure.  Once a node was
visited via ANY deferred path (flag=1), it was skipped on subsequent
visits with different deferred paths.

The pattern has `\B` (state 4) followed by an optional `\b` (state 5)
group.  After the counter breaks:

```
state 4 (\B) → state 6 (Split)
  → state 5 (\b) → state 17 (Split) → ... → tails [7, 9, 12, 14]
  → state 17 (Split) → ... → tails [7, 9, 12, 14]
```

Two paths reach state 17 and the downstream tails:
- **Path A**: `\B@4` → `\b@5` → state 17 — deferred `[4, 5]`
- **Path B**: `\B@4` → state 17 — deferred `[4]`

Due to DFS stack ordering (Split pushes `out1` then `out`, so `out` is
popped first), Path A was explored first.  State 17 was marked as
visited-deferred.  When Path B reached state 17, the check
`visited[17] == 1 && !is_pure` was true (deferred `[4]` is non-empty),
so it was skipped.

All consuming tails got deferred asserts `[4, 5]` — both `\B` AND `\b`.
These assertions are contradictory (one requires a word boundary, the
other requires NOT a word boundary), so `per_tail_asserts_pass()` always
returned false, and the tails were never promoted.

## Investigation Narrative

1. Reproduced the crash artifact: Tier 3 false negative on 16 a's.
   Confirmed with `--tier 0` (NFA MATCH) vs `--tier 3` (NO MATCH).

2. Debug trace showed: at chunk 6, c0 breaks with `\B@4` deferred and
   `pending_break_tails: [7(?@4,5), 9(?@4,5), 12(?@4,5), 14(?@4,5)]`.
   All tails had BOTH `\B@4` and `\b@5`, which is contradictory.

3. Examined the NFA dump: state 4 (`\B`) → state 6 (Split → 5 | 17),
   state 5 (`\b`) → state 17.  Tails are reachable through state 17
   via two paths: one through `\b` (deferred `[4, 5]`) and one
   skipping it (deferred `[4]`).

4. Read `break_consuming_tails()` code.  Traced the DFS stack
   operations manually, confirmed that Path A (through `\b`) was
   explored first due to Stack push order, and Path B was skipped at
   state 17 because visited[17] was already 1 (visited-deferred).

5. The visited tracking only had pure vs deferred granularity, not
   deferred-length granularity.  A shorter deferred path was lost.

## Fix

Replaced the 3-state `visited` array (`u8`: 0/1/2) with a
`best_deferred` array (`u32`: minimum deferred length seen, u32::MAX
for unvisited).  A node is re-explored whenever a shorter deferred path
reaches it:

```rust
let mut best_deferred: Vec<u32> = vec![u32::MAX; n];
// ...
let d_len = deferred.len() as u32;
if d_len >= best_deferred[i] {
    continue;
}
best_deferred[i] = d_len;
```

This ensures downstream consuming states always get the minimal
deferred assert set.  Pure paths (length 0) are still the best case
and subsume all deferred paths.

## What Was Hard

- **The bug only manifests with specific assertion topology**: An
  optional assertion group (`(\b...)?`) where the mandatory assertion
  (`\B`) and optional assertion (`\b`) are different.  This makes
  one path strictly shorter than another through the same intermediate
  Split node.  Most patterns have assertions on a single path, so the
  visited optimization worked correctly.

- **DFS exploration order dependency**: The bug is sensitive to which
  Split branch is explored first.  If `out1` (the `\b` branch) had been
  explored after `out` (the skip branch), the shorter path would have
  been found first and the bug wouldn't trigger.

## Tooling Ideas

- **Per-tail assert details in dump**: The dump already shows
  `break_consuming_deferred`, which was critical for diagnosis.  No
  additional tooling was needed for this bug.

- **Assertion contradiction detection**: A compile-time check could
  warn when a tail's deferred asserts contain contradictory pairs
  (`\b` and `\B`), since such tails can never be promoted.  This
  wouldn't fix the bug but would flag impossible configurations.
