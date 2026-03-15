# Tier 3 Correctness After Bug 53

## Purpose

This document analyzes Bug 53 and uses it to reassess what the typed-effects
redesign did and did not achieve for Tier 3 correctness.

It focuses on four questions:

1. What does Bug 53 actually say about the current Tier 3 model?
2. Did typed effects improve correctness, or mostly improve structure?
3. What root correctness problem remains after the current fix?
4. What is the best way forward if correctness matters more than preserving
   maximum Tier 3 scope?

## Short Answer

Bug 53 does **not** prove that typed effects were useless.

It **does** prove that typed effects, as currently implemented, are not enough
to make broad Tier 3 semantics easy to trust.

The redesign clearly improved:

- authority boundaries
- OR/AND normalization
- queue lifecycle
- diagnosability of pending-effect bugs

But Bug 53 shows that the redesign mostly improved **representation hygiene**,
not the deeper correctness question:

- can Tier 3 soundly summarize all break-deferred, assertion-gated,
  counter-interacting futures with the current effect vocabulary and timing
  model?

The answer appears to be: **not reliably enough**.

My recommendation is therefore correctness-first and two-phase:

1. **Narrow Tier 3's supported fragment more aggressively** when the
   post-assert topology is mixed or timing-sensitive.
2. Only then decide whether to expand back out with a stronger effect algebra.

## Bug 53 In One Sentence

Bug 53 happened because Tier 3 emitted a guarded `Match` effect for a break
path whose real match required **future byte consumption**, while runtime
validation only knew how to check **epsilon-only** downstream reachability.

Post-mortem:

- `docs/bugs/053-tier3-break-deferred-match-consuming-path.md:20-35`

Current Bug 53-related code:

- `src/dfa/tier3_effects.rs:1021-1060`
- `src/dfa/tier3_effects.rs:1188-1217`
- `src/dfa/tier3.rs:4110-4127`

## What Typed Effects Actually Helped With

It is important not to over-correct the conclusion.

Bug 53 was fixed mostly inside the typed-effects pipeline:

- `EffectTiming::EndOnly` was reintroduced in
  `src/dfa/tier3_effects.rs:102-120`
- `compile_target_effects()` now classifies break-deferred chains using
  `epsilon_match_reachability()` in `src/dfa/tier3_effects.rs:1021-1060`
- `finish()` now has explicit `EndOnly` resolution in
  `src/dfa/tier3.rs:4110-4127`

That is already better than the pre-redesign world where the same bug would
likely have required synchronized fixes across:

- hand-written break-side deposition logic
- ad hoc match flags
- per-tail side channels
- dump-only or analysis-only structures

So the fair conclusion is:

- typed effects improved **maintainability and fix locality**
- typed effects improved **correctness debugging**
- typed effects did **not** yet deliver a strong enough semantic model to make
  broad Tier 3 scope obviously safe

This distinction matters.

## What Bug 53 Says About The Current Model

Bug 53 exposes a deeper limitation than just one wrong atom choice.

The current Tier 3 model still mixes three different questions:

1. **What assertion chain must pass?**
2. **At which boundary should it be evaluated?**
3. **What kind of future is behind it?**
   - immediate epsilon `Match`
   - pure `$ -> Match`
   - future byte consumption
   - a mixture of several of the above

Typed effects improved question 1 a lot.

Bug 53 shows questions 2 and 3 are still underspecified.

### The current fix is pragmatic, but still conceptually awkward

The current implementation uses:

- `NextByte` for effects resolved at the next boundary
- `EndOnly` for effects skipped mid-input and kept until `finish()`

See:

- `src/dfa/tier3_effects.rs:100-120`
- `src/dfa/tier3.rs:3944-3953`
- `src/dfa/tier3.rs:4110-4127`

That fixes the concrete false negative in Bug 53's post-mortem, but the type is
still awkward because `PendingEffect` stores only:

- `timing`
- `guard`
- `atom`
- `prev_was_word`

See `src/dfa/tier3_effects.rs:656-676`.

That means one object is trying to represent both:

- a fact about the boundary where the effect was scheduled, and
- a fact about the much later boundary where it may be resolved

That is survivable only because the currently permitted deferred assertions are
fairly constrained.  It is not a strong general model.

## The Real Root Problem

The root problem after Bug 53 is not "typed effects are bad".

The root problem is:

**Tier 3 still tries to summarize too much future behavior with effects that are
only locally typed, not fully terminalized.**

In practice that means:

- a guarded `Match` effect still needs runtime reachability checks in
  `eval_assert_chain()` / `resolve_pending()`
- compile time still performs a lossy classification of the graph behind an
  assertion chain
- the effect timing model still has to encode subtle cross-boundary semantics

Relevant code:

- runtime reachability check: `src/dfa/tier3_effects.rs:783-859`
- pending-effect resolution: `src/dfa/tier3_effects.rs:878-944`
- mid-input reachability walk: `src/dfa/tier3.rs:3743-3778`
- EOI reachability walk: `src/dfa/mod.rs:161-207`

That is exactly why typed effects improved the implementation's shape without
fully improving its trustworthiness.

## Why The Current Classifier Is Still Too Weak

`epsilon_match_reachability()` returns only:

- `direct_match`
- `match_at_end`

See `src/dfa/tier3_effects.rs:1014-1060`.

That is enough for one narrow question:

- "is `Match` epsilon-reachable without consuming another byte?"

But Bug 53 shows the real topology space is richer:

- direct epsilon `Match`
- pure `$ -> Match`
- consuming continuation only
- consuming continuation plus `$ -> Match`
- direct `Match` plus consuming continuation
- chained assertions followed by one of the above

Reducing all of that to `(direct, mae)` is already a warning sign.

It means the compiler is not producing a truly exact terminal action graph.
Instead, it is producing a partial summary and asking runtime to reconstruct the
rest conservatively.

That is where correctness tension keeps returning.

## The Most Important Correctness Insight

The strongest lesson from Bug 53 is this:

**Tier 3 should stop emitting speculative match effects for paths whose real
success still depends on future consumption, unless the model can represent that
future exactly.**

In other words:

- if a path is epsilon-only, an effect can carry a match signal
- if a path requires future consumption, the effect should only carry the local
  progress obligation (`AddTail`, `AddSeed`, or equivalent)
- if a path mixes both and the model cannot represent the timing exactly,
  Tier 3 should fall back

This is more important than whether the atom is named `Match`, `MatchAtEnd`, or
`EndOnly`.

## What This Means For The Typed-Effects Redesign

The typed-effects redesign should no longer be treated as "the thing that makes
Tier 3 correct".

Instead, it should be treated as:

- a **representation cleanup** that made the remaining semantic problems easier
  to see
- a good substrate for a stricter correctness strategy
- not, by itself, a proof that Tier 3's full current scope is sound

That reframing is healthy.

The right next question is not:

- "how do we extend typed effects so Tier 3 can still cover everything?"

The right next question is:

- "which fragment of Tier 3 can this model represent with strong confidence,
   and where should we fallback?"

## Recommended Way Forward

## 1. Adopt a correctness-first Tier 3 envelope

This is the most important recommendation.

When compile-time analysis discovers a break-deferred assertion chain whose
post-assert graph is not cleanly representable, Tier 3 should not try to be
clever.  It should route the pattern elsewhere.

At minimum, Tier 3 should fallback on any break-deferred chain whose post-assert
classification is one of:

- consuming continuation plus `$ -> Match`
- consuming continuation plus direct epsilon `Match`
- any topology that would require an `EndOnly` effect whose correctness depends
  on more than the currently stored boundary snapshot

Practical trigger:

- add an explicit `Unsupported` or `Mixed` result to the post-assert classifier
- when it appears, mark the pattern Tier 4 or Tier 0 instead of Tier 3

This is the single best way to improve Tier 3 correctness quickly.

## 2. Strengthen the compile-time classifier before strengthening runtime

Today the classifier returns `(direct, mae)`.

It should instead return something more explicit, for example:

```rust
enum PostAssertTopology {
    DirectMatch,
    PureEndMatch,
    ConsumingOnly,
    MixedEndAndConsuming,
    MixedDirectAndConsuming,
    Unsupported,
}
```

The precise names can change, but the important point is that the compiler must
distinguish the cases that are currently blurred together.

Why compile time first:

- runtime cannot reliably recover information the compiler already discarded
- compile-time fallback is much safer than runtime reinterpretation

## 3. Use a stricter rule for effect emission

Once the classifier is richer, the emission rule should be conservative:

- `DirectMatch` -> guarded `Match`
- `PureEndMatch` -> guarded end-only-style match effect, **only if** its timing
  semantics are fully defined and tested
- `ConsumingOnly` -> no match atom at all; rely only on `AddTail` / `AddSeed`
- `Mixed*` -> fallback unless and until the model can represent it exactly

This gives Tier 3 a clear correctness story:

- effects carry only exact local truths
- future-consuming possibilities are represented as future work, not predicted
  match signals

## 4. Treat `EndOnly` as a stopgap, not as a stable abstraction

The current `EndOnly` fix is understandable and probably necessary for the live
bug, but it should not be treated as the final semantic answer.

The abstraction is still muddy because it conflates:

- resolution time
- boundary context capture
- and the question of whether the match belongs to the next boundary or the
  final boundary

Two acceptable ways forward exist:

### Option A: keep Tier 3 narrower

- allow `EndOnly` only in the narrow cases already tested and validated
- fallback when the topology is more complicated

### Option B: replace it with a stronger timing model

For example, the semantic idea is not really "end only forever".  It is closer
to:

- "this effect is valid only if the next relevant boundary is EOI"

That is a different notion than a general queue entry that survives until
`finish()`.

If you do not want to build that stronger timing model now, choose Option A.

## 5. Move correctness checks from comments/debug assertions into real release
fallbacks

Tier 3 already has a number of debug-time constraints on the deferred-assert
universe and on other structural assumptions.

Examples:

- `src/dfa/tier3.rs:489-507`
- `src/dfa/tier3.rs:785-805`
- `src/dfa/tier3.rs:2728-2745`

That is useful, but not enough.

If a condition is required for soundness, it should not exist only as:

- a debug assertion
- a post-mortem sentence
- or a comment in `tier3_effects.rs`

It should be part of release-mode eligibility.

That means:

- compute a `tier3_soundness_ok` style flag at build time
- downgrade to Tier 4 or Tier 0 when a required invariant does not hold

## 6. Add direct compiler-level tests for the exact Bug 53 shape family

The existing `resolve_pending()` tests are useful, but Bug 53 is mainly a
compiler-classification problem, not just a pending-resolution problem.

Add tests that directly exercise `compile_target_effects()`-style logic for
post-assert topologies:

- pure `assert -> Match`
- pure `assert -> $ -> Match`
- pure `assert -> consuming`
- mixed `assert -> ($ | consuming)`
- mixed `assert -> (Match | consuming)`

Each test should assert either:

- exact emitted effects, or
- explicit fallback from Tier 3

This is where the current correctness story is weakest.

## 7. Accept that Proposal 1 may be the right short-term correctness move

Bug 53 is a strong argument that Proposal 2 cleaned up Tier 3 without fully
making its broad fragment trustworthy.

That does **not** mean Proposal 2 was a mistake.

It does mean Proposal 1 deserves renewed weight as a practical correctness
strategy:

- keep Tier 3 for the fragment whose effect semantics are simple and explicit
- route the rest to Tier 4 / Tier 0
- expand back out only when each expansion has its own exact model and tests

That is probably a better correctness story than continuing to stretch Tier 3's
semantic envelope with ad hoc exceptions.

## What Not To Do

Bug 53 does **not** imply that the next step should be a large provenance-aware
state redesign.

Proposal 3 may still be useful for contamination families, but Bug 53 is mainly
about:

- timing
- post-assert topology
- and speculative match signaling

So the best immediate response is:

- stronger classification
- narrower eligibility
- better release-mode fallbacks

not a full provenance-aware rewrite.

## Recommended Sequence

If the project wants the highest correctness return for the least risk, I would
do the following in order:

1. Add an explicit post-assert topology enum in the effect compiler
2. Mark mixed / unsupported topologies as Tier 3-ineligible in release builds
3. Keep `EndOnly` only for the very narrow cases that remain demonstrably sound
4. Add compiler-level regression tests for the Bug 53 topology family
5. Re-run fuzzing with emphasis on break-deferred + downstream-counter patterns
6. Only after that, decide whether Tier 3 should regain some of the excluded
   cases via a stronger timing/effect model

## Bottom Line

Bug 53 shows that the typed-effects redesign was **necessary but not
sufficient**.

It made Tier 3 easier to inspect and fix, but it did not yet give Tier 3 a
semantics strong enough to cover every current pattern shape confidently.

The best way forward is not "more clever Tier 3" by default.

The best way forward is:

- make Tier 3 more conservative where the classifier is lossy
- fallback aggressively on mixed break-deferred topologies
- and only expand the supported fragment when the effect model can describe it
  exactly

That is the clearest route to making Tier 3 feel correct rather than merely
repairable.
