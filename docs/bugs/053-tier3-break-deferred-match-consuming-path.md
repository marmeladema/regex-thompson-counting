# Bug 53: Break-deferred Match atom dropped when downstream path has consuming states

## Bug summary

- **Pattern:** `^.{7,8}\B(\b)?a{0,2}$`
- **Input:** `"aaaaaaaa"` (8 bytes)
- **Expected:** MATCH (`.{7}` + `\B` passes word→word + skip `(\b)?` + `a{1}` + `$`)
- **Actual:** NO MATCH (Tier 3 with `--unroll-limit 0`)
- **Affected tier(s):** Tier 3, per-instance path

## Root cause

When counter 0 breaks (value ≥ 7), the break path goes through `\B`
(deferred assertion) to a downstream structure:

```
\B → Split → (\b)? → Split → (CI(c1) → Byte('a') | $ → Match)
```

`compile_target_effects()` emits a guarded `EffectAtom::Match` for the
`\B` deferred chain.  At runtime, `can_reach_match_mid()` walks epsilon
transitions from `\B`'s output and finds no direct Match — only consuming
states and `$` (which fails mid-input).  The `Match` effect is dropped.

The match IS reachable (through counter 1's body → `$` at EOI), but
the effect model cannot represent a match that requires future byte
consumption after the deferred assertion resolves.

## Investigation narrative

1. Discovered during the consecutive-repetition merger, which turns
   `a?a?` into `a{0,2}`.  The merged form forces a counter with
   `--unroll-limit 0`, creating the NFA topology.

2. Minimized to `^.{7,8}\B(\b)?a{0,2}$` on `"aaaaaaaa"`.

3. Multiple fix attempts (in-place MatchAtEnd atom, EndOnly timing,
   can_reach_match_mid consuming-state acceptance) all either introduced
   false positives or had incorrect timing semantics.

4. Root cause analysis in `docs/tier3-bug53-correctness-way-forward.md`
   concluded that the effect model cannot soundly represent mixed
   post-assert topologies (consuming + `$ → Match`).

## Fix

**Correctness-first approach:** reject patterns from Tier 3 when any
break-deferred assertion chain has a mixed post-assert topology.

Implementation:

1. Added `PostAssertTopology` enum in `tier3_effects.rs` classifying
   the graph downstream of each assertion:
   - `DirectMatch`: Match epsilon-reachable without `$`
   - `PureEndMatch`: Match only through `$ → Match`
   - `ConsumingOnly`: only consuming states downstream
   - `MixedMatchAndConsuming`: direct Match + consuming states
   - `MixedEndAndConsuming`: `$ → Match` + consuming states

2. Added `classify_post_assert()` to walk the epsilon graph and
   determine the topology.

3. Added `check_break_deferred_soundness()` called during
   `RegexBuilder::build()` after `compute_tier3_analysis()`.  If any
   chain is `Mixed*`, the pattern is marked `tier3_eligible = false`
   and falls back to Tier 4 / NFA.

4. Added 7 unit tests: 4 for `classify_post_assert` topology cases,
   2 for `check_break_deferred_soundness` accept/reject, 1 match_tests
   regression entry.

## What was hard

Three attempted runtime fixes failed:

- **MatchAtEnd atom + EndOnly timing:** Correct for pure `$ → Match`,
  but `prev_was_word` captured at deposit time becomes stale by EOI.
  For `\B$`, `\B` passes mid-input (word→word) but fails at EOI
  (word→end = boundary).

- **MatchAtEnd atom + skip reachability check mid-input:** `\B` passes
  mid-input and sets `match_at_end`, but this is a false positive for
  patterns like `^c{3,5}\b.{0,3}$` where `\b` passes mid-input
  (word→non-word) but the suffix doesn't match.

- **can_reach_match_mid accepting consuming states:** Produces false
  positives because consuming-state presence doesn't guarantee the
  suffix matches.

The fundamental issue: the current effect model represents only local
boundary-point truths.  A match that requires future byte consumption
is a multi-boundary truth that the model cannot represent.

## Tooling ideas

- The `dump --dfa` output should show `PostAssertTopology` for each
  break-deferred chain, making mixed topologies visible in debugging.
