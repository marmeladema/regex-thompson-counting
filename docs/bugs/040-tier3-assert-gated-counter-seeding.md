# Bug 40 — Assertion-gated counter seeding false positive

## Bug summary

- **Pattern:** `^((.1?\B.{3,28}){3,3}|(a?a?)?)$`
- **Input:** `"aaaaaaaaaa"` (10 a's)
- **Expected:** NO MATCH (NFA and `regex` crate agree)
- **Actual (Tier 3):** MATCH (false positive)
- **Affected tier:** Tier 3 (conditional DFA, per-instance path)
- **Artifact:** `fuzz/artifacts/fuzz_match/crash-e57cddb4258f44ab2ef7b576a9dc0d9cb90af3c2`
- **Source fuzzer:** `fuzz_match` (oracle = `regex` crate)

## Root cause

The `{3,3}` outer repetition is unrolled into three counter copies (c0, c1,
c2 for `.{3,28}`).  Each repetition copy has the NFA structure:

```
ByteClass → Split(optional '1') → Assert(\B) → CI(counter) → body → CInc
```

The `with_break` DFA state includes NFA states from both the no-break path
and all counter break paths.  The break path from c0's CInc crosses a
consuming state (state 8, the prefix ByteClass for the second repetition)
before reaching `\B → CI(c1) → c1 body`.  Because the break path crosses
a consuming state, the `break_seeds` epsilon walk (which only follows
non-consuming states) does NOT produce a break_seed entry for c1.

The `with_break` DFA state stores `\B` as a deferred assertion.  During
`populate()`, the deferred assertion is resolved: if `\B` passes between
the previous byte and the transition byte, the resolved closure produces
seeds for c1 and c2 (via the epsilon path through `\B → CI → body`).

These `resolved_seeds` were merged into the transition's unconditional
`seeds` array.  The existing `is_break_gated` check (Bug 17) only filters
resolved seeds that appear in `analysis.break_seeds` — but there IS no
break_seeds entry (the epsilon walk stopped at the consuming state).  So
the resolved c1/c2 seeds passed the filter and became unconditional seeds
on the transition.

Result: c1 and c2 were seeded on EVERY transition from the `with_break`
state, not just when c0 actually broke.  This gave c2 enough extra
counting steps to reach its minimum on short inputs where the NFA
correctly reports no match.

For the 10-a input: the pattern requires 3 reps × (1 prefix byte +
3 body bytes) = 12 bytes minimum.  With 10 bytes it's impossible, but
the over-seeding gave c2 count 3 (≥ min) and `break_is_match_at_end`
fired.

## Fix

Broadened the `is_break_gated` check for resolved seeds in `populate()`:
a resolved seed is now break-gated whenever its origin is NOT in
`reachable_without_break`, regardless of whether it appears in
`analysis.break_seeds`.

```rust
// Before (Bug 17):
let is_break_gated = analysis.break_seeds.iter().any(|bs| {
    bs.counter == s.0 && bs.origin == s.1
        && !analysis.reachable_without_break[s.1.idx()]
});

// After (Bug 40):
let is_break_gated = !analysis.reachable_without_break[s.1.idx()];
```

Origins only reachable through CInc break paths are counter-break-
dependent by definition.  The old check missed cases where the break
path crosses a consuming state (no `break_seeds` entry), but the
resolved seed from deferred assertion resolution still reached the
downstream counter.

The same fix was applied to the `pre_seeds` filter (resolved seeds
on L=1 bodies for counting transitions).

The tail mechanism (`post_break_tails` + tail-to-counter handoff)
correctly handles the actual seeding with proper byte timing: the
tail consumes the prefix byte, advances through `\B` (depositing it
as a verified deferred assert), and seeds the downstream counter on
the next byte.

## Investigation narrative

1. The `fuzz_match` fuzzer found the crash artifact.  Decoded the seed to
   get pattern `^((.1?\B.{3,28}){3,3}|(a?a?)?)$` and input
   `"aaaaaaaaaaaa"` (12 a's).

2. Confirmed the disagreement: Tier 3 said NO MATCH for 10 a's (should be
   NO MATCH — false positive on unfixed code) and also found that 12 a's
   SHOULD match (3 reps × 4 bytes = 12).

3. First attempted fix: blocking non-End Assert traversal in
   `analyze_target()` and `break_consuming_tails()`.  This prevented the
   tail mechanism from crossing `\B`.  Fixed the 10-a false positive but
   introduced a false negative on 12 a's — the tail mechanism was the ONLY
   way to seed c1/c2 (break_seeds was empty, resolved_seeds were filtered).

4. Second attempted fix: re-evaluating deferred asserts at EOI in
   `finish()` before accepting `match_at_end`.  Wrong because
   `verified_deferred_asserts` are assertions that passed mid-input and
   don't need to pass at EOI — `match_at_end` from a counter break is
   valid if the assertion passed when the counter was counting.

5. Traced the seeding path: DFA `populate()` → `resolve_deferred()` →
   `resolved_seeds` → merged into `seeds`.  The `is_break_gated` check
   used `analysis.break_seeds` which was empty (break path crosses
   consuming state 8, epsilon walk didn't find CI).  So resolved c1/c2
   seeds became unconditional.

6. Correct fix: broadened `is_break_gated` to check `reachable_without_break`
   directly.  This subsumes the old `analysis.break_seeds` check and
   correctly identifies all counter-break-dependent seeds.

7. Verified: NFA and Tier 3 agree on all inputs 0-19 a's.  The fuzz artifact
   no longer crashes.  All 456 tests pass.

## What was hard

- **Three failed fix attempts before finding the correct one.**  The
  first two fixes targeted the wrong mechanism (tail traversal and
  finish() assertion re-evaluation).  The root cause was in the seed
  generation during DFA transition compilation — a subtle interaction
  between deferred assertion resolution, the `is_break_gated` filter,
  and the `break_seeds` epsilon walk.

- **Counter-to-counter seeding has THREE independent paths:** (a)
  `break_seeds` (epsilon-only, with deferred assertion gating), (b) the
  tail mechanism (consuming-state tracking with byte-by-byte advancement),
  and (c) resolved_seeds from deferred assertions on the `with_break` DFA
  state.  Path (c) was the problematic one, but path (b) is the correct
  one for this pattern.  Understanding which path does what required
  careful tracing through `populate()`.

- **The false negative from the first fix was caught by the fuzz artifact**
  but on a different input (12 a's instead of 10 a's).  The artifact
  generates multiple inputs from the seed, and the 12-a input exercises
  the legitimate match path.  Without the artifact's broader coverage,
  the false negative might have been missed.

## Tooling ideas

- The `--debug` trace should show the source of each counter seed:
  "seeded via tail handoff at state X" vs "seeded via unconditional
  seed on transition" vs "seeded via break_seed trigger=cN".  This
  would immediately reveal when a counter is being seeded through the
  wrong mechanism.

- A `dump --dfa` mode showing the transition's `seeds`, `break_seeds`,
  `pre_seeds`, and `resolved_seeds` separately (rather than merged) would
  help debug seed-related bugs without reading the `populate()` code.

- The `break_seeds` computation could warn when a CInc break path
  crosses a consuming state without producing any break_seed entries.
  This gap in coverage (no break_seeds, but resolved_seeds can still
  reach the downstream counter) is a known source of bugs.
