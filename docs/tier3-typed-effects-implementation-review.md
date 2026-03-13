# Tier 3 Typed Effects Implementation Review

## Purpose

This document reviews the current Tier 3 typed-effects implementation after
the migration and Bug 51.

It focuses on:

- correctness risks and possible unsoundness
- structural inconsistencies between the effect model and the runtime
- simplifications that would make the implementation easier to reason about
- practical next steps for making typed effects the real source of truth

The review is based on the current implementation in `src/dfa/tier3.rs`,
`src/dfa/tier3_effects.rs`, the dump output path in `src/dump.rs`, and the
recent post-mortem `docs/bugs/051-tier3-break-deferred-or-semantics.md`.

## Executive Judgment

The typed-effects architecture is still worth keeping.

The main problem is not that the effect model is fundamentally incapable of
describing Tier 3 semantics.  The main problem is that the current codebase is
in a half-normalized state where multiple representations coexist:

- legacy analysis fields on `Tier3OriginKind`
- hand-written pending-effect deposition logic in the matcher
- compiled `CompiledTargetEffects` data that is dumped but not actually used
  as the runtime source of truth

That leaves the implementation vulnerable to semantic drift.  Bug 51 is best
understood as one instance of that broader issue.

My overall assessment is:

- the architecture is **salvageable and directionally correct**
- the current implementation is **not yet clean enough to call sound with
  confidence**
- the highest-value work is to **finish normalization**, not to abandon typed
  effects

## Review Scope

Main code reviewed:

- `src/dfa/tier3_effects.rs`
- `src/dfa/tier3.rs`
- `src/dump.rs`
- `docs/bugs/051-tier3-break-deferred-or-semantics.md`

Historical context considered:

- migration commits from `6023359` through `57c709d`
- Bug 51 fix commit `19014b3`

## Findings

## 1. Compiled typed effects are not the runtime source of truth

**Classification:** maintainability risk with direct correctness impact

The code advertises compiled typed effects as the structured representation of
Tier 3 nonlocal semantics:

- `src/dfa/tier3.rs:163`
- `src/dfa/tier3_effects.rs:1`

But in practice, the matcher still deposits most `PendingEffect` entries by
hand from `Tier3OriginKind` and side tables:

- `src/dfa/tier3.rs:2834`
- `src/dfa/tier3.rs:2913`
- `src/dfa/tier3.rs:2979`
- `src/dfa/tier3.rs:3092`
- `src/dfa/tier3.rs:3312`
- `src/dfa/tier3.rs:3592`
- `src/dfa/tier3.rs:3658`
- `src/dfa/tier3.rs:3730`

Meanwhile, `analysis.target_effects` is compiled and dumped:

- `src/dfa/tier3.rs:165`
- `src/dump.rs:168`

but not used to drive runtime execution.

### Why this matters

This means there are multiple semantic sources:

- the legacy structural analysis
- the hand-written runtime deposition logic
- the compiled effect representation

Any semantic bug has to be fixed in several places, and one representation can
easily drift from another.  Bug 51 is exactly that kind of drift.

### Recommendation

Make one representation authoritative.

Best option:

- transitions and runtime deposit helpers should consume compiled effect
  templates or template IDs, not reconstruct effect semantics manually from
  `Tier3OriginKind`

Short-term fallback:

- if full normalization is not ready, reduce the public/documented role of
  `CompiledTargetEffects` so the code does not imply a stronger guarantee than
  it currently provides

## 2. `required_breaks` is carried but not enforced in `resolve_pending`

**Classification:** latent correctness bug / broken contract

`EffectGuard` is documented as a conjunction of:

- `required_breaks`
- `assert_chain`

See `src/dfa/tier3_effects.rs:168`.

But `resolve_pending()` checks only the assertion chain and completely ignores
`required_breaks`:

- `src/dfa/tier3_effects.rs:586`
- `src/dfa/tier3_effects.rs:618`

### Why this matters

Today this is partly masked because most enqueue sites only create break-gated
effects after the break already happened.  But the type contract is already
false: the guard says one thing and the runtime checks another.

This is especially dangerous because the design documents explicitly rely on
`required_breaks` as the provenance-ready part of the model.

### Recommendation

Either:

1. enforce `required_breaks` during pending resolution by passing a current
   satisfied-break mask, or
2. remove `required_breaks` from the runtime contract until it is genuinely
   enforced

The current in-between state is misleading.

## 3. The compiled effect representation and the runtime manual logic already disagree

**Classification:** maintainability risk, likely future bug

The clearest example is break-deferred match behavior.

In the effect compiler, break-deferred assertions currently compile to guarded
`MatchAtEnd` atoms:

- `src/dfa/tier3_effects.rs:818`
- `src/dfa/tier3_effects.rs:843`
- `src/dfa/tier3_effects.rs:852`

But the runtime manual deposition code emits guarded `Match` atoms for the same
concept:

- `src/dfa/tier3.rs:2925`
- `src/dfa/tier3.rs:3104`
- `src/dfa/tier3.rs:3675`

### Why this matters

Even if the current matcher does not execute `CompiledTargetEffects` directly,
the dump output and internal comments imply that the effect compiler reflects
runtime semantics.  It currently does not.

If the project later switches the runtime to consume compiled effects without
first reconciling this mismatch, a new semantic regression is likely.

### Recommendation

Choose one canonical normalization for break-deferred match behavior and make
both representations follow it.  My read is that the runtime form is the more
trustworthy one today because it was exercised by the migration and bug fixes.

## 4. Bug 51 fixed one OR/AND mistake, but the same pattern may still exist elsewhere

**Classification:** plausible correctness risk

Bug 51 established an important normalization rule:

- a list of assertions may represent alternative paths with OR semantics, not
  one conjunctive chain

See `docs/bugs/051-tier3-break-deferred-or-semantics.md:14`.

The immediate fix covered `break_deferred_asserts`, but there are other places
where the code still collapses path structure.

### 4a. Break seeds may collapse distinct deferred paths

Break seeds are collected in `break_seeds_raw` and later normalized into
`Tier3BreakSeed` records:

- `src/dfa/tier3.rs:410`

This review did not prove a live bug here, but the structure is suspicious:
if the same `(trigger, counter, origin)` is reachable via different deferred
paths, the representation needs to preserve whether those paths are:

- one chain
- several alternatives
- or one path subsuming another

The current `Tier3BreakSeed` model stores just one `deferred_asserts` slice and
one `assert_chain_id`:

- `src/dfa/tier3.rs:262`

### 4b. Per-tail deferred paths may also be over-collapsed

`break_consuming_tails()` stores one deferred chain per tail and prefers a
minimal path length:

- `src/dfa/tier3.rs:2551`
- `src/dfa/tier3.rs:2566`

That is a useful heuristic, but it is not obviously semantics-preserving when a
tail is reachable through multiple alternative deferred paths of similar cost.

### Recommendation

Audit every legacy deferred-assert field and classify it explicitly as one of:

- one conjunctive path
- OR over several alternative paths
- one chain per tail
- one chain per seed

Do not rely on slice shape alone to imply semantics.

## 5. Queue lifecycle in `chunk()` looks unsafe for second-order deferred effects

**Classification:** plausible correctness bug

During pending-effect resolution in `chunk()`, resolved tails can enqueue fresh
`PendingEffect`s while the matcher is still iterating the current pending list:

- `src/dfa/tier3.rs:3596`
- `src/dfa/tier3.rs:3668`
- `src/dfa/tier3.rs:3694`
- `src/dfa/tier3.rs:3734`

Then the queue is unconditionally cleared:

- `src/dfa/tier3.rs:3755`

### Why this matters

If a pending effect resolved at the current boundary discovers a new deferred
effect that should live until the next boundary, clearing the same vector will
drop it.

I have not proven a concrete reproducer from this review alone, but the queue
lifecycle is suspicious enough that it deserves immediate attention.

### Recommendation

Split the queue into two logical buffers:

- `pending_current`
- `pending_next`

Drain only `pending_current`, and move `pending_next` into place after the
boundary is fully processed.

## 6. `EndOnly` exists in the effect model but appears to be dead machinery

**Classification:** simplification / maintainability issue

The effect model has `EffectTiming::EndOnly` and the matcher has a
`pending_effects_end_only` field:

- `src/dfa/tier3_effects.rs:59`
- `src/dfa/tier3.rs:2760`

But the current code appears to enqueue no real `EndOnly` pending effects and
does not resolve that queue in the matching logic.  The queue is only present
in reset/debug code:

- `src/dfa/tier3.rs:3397`
- `src/dfa/tier3.rs:3424`
- `src/dfa/tier3.rs:4094`

The effect compiler does still emit `EndOnly` guarded effects for
break-deferred match-at-end:

- `src/dfa/tier3_effects.rs:846`

but these compiled effects are not the runtime source of truth today.

### Recommendation

Pick one:

- remove `EndOnly` until the runtime really uses it, or
- complete the runtime path so `EndOnly` effects are actually scheduled and
  resolved explicitly

Right now it increases mental overhead without giving real semantic coverage.

## 7. `resolve_pending()` applies reachability gating to a whole bundle, not each atom

**Classification:** latent design risk

`resolve_pending()` decides whether to run downstream match reachability checks
for the entire `PendingEffect` by scanning whether *any* atom is `Match` or
`MatchAtEnd`:

- `src/dfa/tier3_effects.rs:611`

### Why this matters

If the implementation ever creates mixed bundles such as:

- `Match` + `AddTail`
- `Match` + `AddSeed`

then the non-match atoms will inherit match-oriented reachability gating even
if that is not semantically correct.

### Recommendation

Either:

- enforce homogeneous bundles with debug assertions, or
- resolve guards atom-by-atom rather than effect-by-effect

At minimum, document the intended invariant clearly.

## 8. Transition and effect data remain too heavyweight for the hot path

**Classification:** performance and maintainability issue

The transition cache still stores cloned `Tier3OriginKind` values, which now
carry multiple boxed slices and chain metadata:

- `src/dfa/tier3.rs:191`
- `src/dfa/tier3.rs:213`

The runtime also allocates boxed atom arrays while matching:

- `src/dfa/tier3.rs:2842`
- `src/dfa/tier3.rs:2925`
- `src/dfa/tier3.rs:2953`
- `src/dfa/tier3.rs:2988`
- `src/dfa/tier3.rs:3104`
- `src/dfa/tier3.rs:3132`
- `src/dfa/tier3.rs:3316`
- `src/dfa/tier3.rs:3603`
- `src/dfa/tier3.rs:3675`
- `src/dfa/tier3.rs:3702`
- `src/dfa/tier3.rs:3741`

This runs against the project rule that hot matching code should avoid heap
allocation and should reuse precomputed data.

### Recommendation

Move toward:

- compact effect-template IDs stored in transitions
- pre-interned atom bundles or small fixed-size inline representations
- runtime queues that store cheap handles, not freshly boxed slices

This will help both performance and clarity.

## 9. End-of-input seed resolution looks over-approximate

**Classification:** plausible correctness risk

At EOI, resolved seeds from pending effects are checked by scanning all target
actions for any increment on the same counter whose `min` is satisfied:

- `src/dfa/tier3.rs:4028`

This check ignores the seed's specific origin and instead reasons only by
counter ID and threshold.

### Why this matters

A seed at one body origin is not automatically equivalent to being at a break
point for every increment action of that counter.  The current logic may be
safe under additional invariants, but those invariants are not documented here.

### Recommendation

Either:

- document and prove the invariant that makes this reduction safe, or
- route seed-at-EOI behavior through the same normalized target/effect model
  used during ordinary stepping

## 10. The test surface around `tier3_effects.rs` is too shallow

**Classification:** process weakness

Current local tests in `src/dfa/tier3_effects.rs` cover only basic arena and
display behavior:

- `src/dfa/tier3_effects.rs:978`

There are no focused semantic tests for:

- OR versus AND normalization
- runtime enforcement of `required_breaks`
- `resolve_pending()` queue lifecycle
- effect-compiler equivalence against runtime deposition logic

### Recommendation

Add table-driven tests directly in `src/dfa/tier3_effects.rs` for:

- contradictory alternatives (`\b` vs `\B`)
- a true conjunctive chain (`\b -> \B`)
- pending effects that schedule more pending effects
- mixed timing scenarios
- break-gated versus counter-free effects

## Simplifications Worth Making

These are not all correctness fixes, but they would make the implementation
substantially easier to maintain.

### 1. Make one effect deposition helper per family

Right now the same concepts are open-coded in multiple sites:

- target deferred matches
- break deferred matches
- deferred tails

This duplication made Bug 51 a four-site fix.

Suggested helpers:

- `enqueue_target_deferred_matches(...)`
- `enqueue_break_deferred_matches(...)`
- `enqueue_deferred_tails(...)`

### 2. Reduce plural semantic channels in `Tier3OriginKind`

`Tier3OriginKind::Increment` still contains many pre-effect semantic fields:

- `break_deferred_asserts`
- `break_consuming_states`
- `break_consuming_pure`
- `break_consuming_deferred`
- `break_deferred_chain_ids`

See `src/dfa/tier3.rs:213`.

That is a sign that the effect model has not fully absorbed the semantics yet.

### 3. Clarify which semantics belong to analysis versus effect compilation

Some path decisions happen in `break_closure()` and `break_consuming_tails()`,
others are reinterpreted in `compile_target_effects()`, and others are rebuilt
again manually in the matcher.

That layering should be simplified so there is one clear answer to:

- where semantic normalization happens
- where it becomes effect data
- where the matcher merely executes precomputed information

## Priority Action Plan

## Priority 1: Fix the broken contracts

1. Enforce `required_breaks` or remove it from the runtime contract.
2. Split current-boundary and next-boundary pending queues.
3. Reconcile compiled-effect semantics with runtime deposition semantics.

## Priority 2: Finish semantic normalization

1. Audit all deferred-assert fields for OR versus AND meaning.
2. Normalize alternative paths explicitly instead of relying on slice shape.
3. Restore stronger equivalence checks while this work is in flight.

## Priority 3: Make compiled effects authoritative

1. Stop hand-building pending effects from `Tier3OriginKind` in multiple code
   paths.
2. Transition the matcher to consume compiled templates or template IDs.
3. Shrink `Tier3OriginKind` once the effect compiler fully owns nonlocal
   semantics.

## Priority 4: Then optimize

1. Remove hot-path boxed-slice allocations.
2. Intern effect templates and atom bundles.
3. Reduce cloned transition payload size.

## Bottom Line

The typed-effects redesign was the right direction, but the current
implementation is still split across too many semantic layers.

The most important conclusion from this review is:

- **the next step should not be another isolated bug fix**
- **the next step should be completing the normalization so compiled effects
  become the single authoritative model for nonlocal Tier 3 semantics**

If that is done, most of the current problems become much easier to reason
about, test, and optimize.
