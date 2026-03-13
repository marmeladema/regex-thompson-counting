# Tier 3 Typed Effects Implementation Plan

## Purpose

This document turns Proposal 2 (`docs/tier3-redesign-proposal-2-typed-effects.md`)
into an implementation plan detailed enough for a junior engineer to follow.

The plan is intentionally incremental.  The first priority is to preserve
correctness at every step, not to land a heroic rewrite in one patch.

## Target Outcome

At the end of this plan, Tier 3 should:

- represent transition consequences using a typed effect model instead of
  several ad hoc side channels
- handle current intended Tier 3 scope in principle: non-nested counters,
  variable-length bodies, break-path deferred assertions, break-triggered
  seeding, and post-break consuming tails
- remain compatible with both existing Tier 3 storage backends:
  range-compressed and per-instance
- have a clearer path to either stop at Proposal 2 or extend into Proposal 3
  if explicit provenance on active state is still needed

## Non-Goals

This plan does **not** try to do the following immediately:

- unify Tier 3 and Tier 4 implementation code
- redesign Tier 2
- prove the full range-compression story for provenance-aware masks
- optimize for peak performance before correctness is re-established

## Recommended Interpretation of Proposal 2

Implement Proposal 2 as:

- a typed effect language for nonlocal transition consequences
- plus a provenance-ready guard representation
- while keeping the current local counter-step structure (`Advance` vs
  `Increment`) and the current ranged/per-instance storage backends

This is the most practical way to improve Tier 3 without rewriting the
entire tier around a new executor model.

## Current Code to Replace Gradually

The implementation should begin by understanding these parts of the current
Tier 3.

### Compile-time analysis

- `Tier3Analysis` in `src/dfa/tier3.rs:70`
- `Tier3OriginKind` in `src/dfa/tier3.rs:168`
- `compute_tier3_analysis()` in `src/dfa/tier3.rs:251`
- `analyze_target()` in `src/dfa/tier3.rs:2139`
- `break_closure()` in `src/dfa/tier3.rs:2355`
- `break_consuming_tails()` in `src/dfa/tier3.rs:2458`

### Cached transition shape

- `Transition` in `src/dfa/tier3.rs:796`
- `populate()` in `src/dfa/tier3.rs:1471`

### Runtime state and execution

- `Tier3DfaMatcher` in `src/dfa/tier3.rs:2565`
- `step_slow_impl!` in `src/dfa/tier3.rs:2703`
- `chunk()` in `src/dfa/tier3.rs:3330`
- `finish()` in `src/dfa/tier3.rs:3874`

### Shared DFA state

- `epsilon_closure()` in `src/dfa/mod.rs:259`
- `DfaState` / interning logic in `src/dfa/mod.rs:352`

### Tier 4 reference model

- `CounterOp` in `src/dfa/tier4.rs:30`
- `execute_program()` in `src/dfa/tier4.rs:138`

Do not change all of these at once.

## Core Design Choice

### Keep local counter stepping separate from effect execution

Do **not** try to encode everything in one enormous effect IR at first.

The current `Advance` vs `Increment` split maps well onto the existing
counter storage backends, because those backends answer local questions like:

- can this entry continue?
- can this entry break?
- where do continued values go?

Relevant code:

- `Tier3OriginKind` in `src/dfa/tier3.rs:168`
- `CounterStorage` in `src/dfa/tier3.rs:893`

The typed effect system should therefore handle the *nonlocal* and *guarded*
consequences of a transition, while the local counter-step remains a
dedicated data structure.

That means the redesign target is closer to:

- local step description + typed side effects

than to:

- one giant IR that replaces every part of the current slow path

This keeps the migration safer.

## Recommended New Module Layout

### New file

Add a new sibling module:

- `src/dfa/tier3_effects.rs`

Declare it from `src/dfa/mod.rs`.

### Why a new file?

`src/dfa/tier3.rs` is already very large and already mixes:

- compile-time analysis
- cache management
- two runtime backends
- debug formatting

Putting the effect IR and its helper code in a dedicated file makes the
transition safer and easier to review.

## Recommended Data Model

The following types are recommended.  The exact names may differ, but the
model should stay close to this.

### 1. Provenance-ready break mask

```rust
type BreakMask = u64;
```

Why include this now, even if the first version uses only `0`?

- Proposal 2 can support full Tier 3 scope only if break provenance remains
  representable when needed
- the current Tier 3 is already capped to 64 counters for bitmask reasons in
  `src/lib.rs:2261`
- adding the field later is more disruptive than reserving it now

Important rule:

- `0` means counter-free

### 2. Normalized timing enum

```rust
enum EffectTiming {
    Now,
    NextByte,
    EndOnly,
}
```

Do not overload booleans for this.

### 3. Normalized guard struct

Do **not** use a recursive `And(Box<[EffectGuard]>)` shape in the first
implementation.  It is too hard for a junior engineer to normalize and test
correctly.

Use a single normalized guard struct instead:

```rust
struct EffectGuard {
    required_breaks: BreakMask,
    assert_chain: AssertChainId,
}
```

Where:

- `required_breaks == 0` means counter-free
- `assert_chain == AssertChainId::NONE` means no assertion gating

This is much easier to reason about than a tree of guards.

### 4. Assertion-chain arena IDs

Store assertion chains in an arena so runtime structures can use compact IDs
instead of heap-allocating slices during matching.

Recommended types:

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AssertChainId(u32);

impl AssertChainId {
    const NONE: Self = Self(u32::MAX);
}
```

And in a compile-time arena, something like:

```rust
struct AssertChainArena {
    chains: Vec<Box<[StateIdx]>>,
}
```

Later, if needed, this can be flattened further.

### 5. Effect atoms

Keep the atom set intentionally small.

Recommended atoms:

```rust
enum EffectAtom {
    AddSeed {
        counter: CounterIdx,
        origin: StateIdx,
        value: u32,
    },
    AddTail {
        origin: StateIdx,
    },
    Match,
    MatchAtEnd,
}
```

Do **not** encode ordinary local `Advance` or `Increment` motion as atoms in
the first pass.  Keep them in the step description.

### 6. Guarded effects

```rust
struct GuardedEffect {
    timing: EffectTiming,
    guard: EffectGuard,
    atoms: Box<[EffectAtom]>,
}
```

### 7. Local target step

Add a local step type that replaces only the part of `Tier3OriginKind` that
talks about local counter motion.

```rust
enum TargetStep {
    None,
    Advance {
        new_origins: Box<[StateIdx]>,
    },
    Increment {
        counter: CounterIdx,
        advance_origins: Box<[StateIdx]>,
        min: u32,
        max: u32,
        continue_origins: Box<[StateIdx]>,
    },
}
```

### 8. Compiled target effects

Replace the old "action + scattered booleans/lists" with one struct.

```rust
struct CompiledTargetEffects {
    step: TargetStep,
    immediate: Box<[EffectAtom]>,
    on_break_now: Box<[EffectAtom]>,
    guarded: Box<[GuardedEffect]>,
}
```

Semantics:

- `immediate`: unconditional `Now` effects from the target
- `on_break_now`: `Now` effects that are valid only when the incrementing
  instance actually breaks
- `guarded`: all next-byte and end-only obligations, plus any assertion-
  gated effects that cannot be resolved immediately

### 9. Pending effects

Replace the current family of pending vectors with one pending-effect type.

Recommended shape:

```rust
struct PendingEffect {
    timing: EffectTiming,
    guard: EffectGuard,
    atoms: Box<[EffectAtom]>,
    prev_was_word: bool,
}
```

Important note:

- `prev_was_word` must capture the boundary context at the time the effect
  is scheduled, because several existing Tier 3 bugs were timing/context
  bugs rather than pure reachability bugs

### 10. Two operational queues, one conceptual model

At runtime, it is fine to store:

- `pending_next_byte: Vec<PendingEffect>`
- `pending_end_only: Vec<PendingEffect>`

This is still one effect system.  Two vectors are just an execution detail.

## Key Invariants

These invariants should be written down in comments and enforced with
`debug_assert!()` where possible.

### State and origin invariants

- every seed origin is consuming
- every tail origin is consuming
- `TargetStep::Advance` and `TargetStep::Increment` only refer to consuming
  origins
- no pending `AddTail` or `AddSeed` effect points at a non-consuming state

### Timing invariants

- `NextByte` effects are consumed exactly once on the next boundary
- `EndOnly` effects are never evaluated mid-input
- a `Now` effect is never inserted into a pending queue

### Guard invariants

- `required_breaks == 0` means genuinely counter-free
- assertion chains are evaluated in order, not independently
- a chain ID must always refer to the *full required chain for that path*,
  not a flattened set of independent assertions

### Migration invariants

During the shadow-compile phases:

- old analysis and effect-compiled analysis must agree on every currently
  represented fact
- if both runtimes coexist in debug/test builds, their per-step observable
  results must agree

## Implementation Phases

Do the work in the phases below.  Do not skip ahead.

### Phase 0: Establish the safety harness

#### Goal

Create the testing and debugging scaffolding needed before any semantic
rewrite begins.

#### Code changes

- no behavior changes yet
- add comments or TODO markers identifying the legacy channels to remove:
  - `verified_deferred_asserts`
  - `pending_break_seeds`
  - `pending_break_tails`
  - `pending_resolved_tails`
  - `pending_resolved_mae`

#### Tests to run

- `cargo test`

#### Output to inspect manually

Pick 3-5 representative patterns and record the current `--debug` output so
you can compare later.  Include at least one case for:

- contamination
- break seed timing
- pending break tails
- ByteTable-specific behavior

#### Done criteria

- baseline tests pass
- you have a small notebook or scratch file with representative debug traces

### Phase 1: Add the effect module and debug dump support

#### Goal

Introduce the new types without changing behavior.

#### Files to edit

- `src/dfa/mod.rs`
- `src/dfa/tier3_effects.rs` (new)
- `src/dfa/tier3.rs`
- `src/dump.rs`

#### Tasks

1. Add the new module declaration.
2. Add the core types listed in the data model section.
3. Add `Debug` / `Display` helpers for effect types.
4. Add dump support so a compiled regex can show effect data once it exists.

#### Important rule

Do **not** add any runtime allocation in the hot path yet.  This phase is
types only.

#### Tests to run

- `cargo test`

#### Done criteria

- the code compiles
- dump/debug helpers compile
- there is still no behavior change

### Phase 2: Shadow-compile target effects alongside legacy analysis

#### Goal

Compile effect summaries in parallel with the old Tier 3 analysis, without
using them for matching yet.

#### Files to edit

- `src/dfa/tier3.rs`
- `src/dfa/tier3_effects.rs`
- `src/dump.rs`

#### Tasks

1. Extend `Tier3Analysis` with a new field, for example:

   - `target_effects: Box<[Option<CompiledTargetEffects>]>`

2. Add a new compile-time function in `tier3_effects.rs`, for example:

   - `compile_target_effects(target, states, can_reach_match, ...)`

3. Keep calling the legacy analysis helpers:

   - `analyze_target()`
   - `break_closure()`
   - `break_consuming_tails()`

4. Initially, use those helpers to *populate* the new effect structure.
   This is fine for the shadow phase.  The point is to ensure semantic
   equivalence before deleting anything.

5. Add dump output for `target_effects`.

#### What not to do yet

- do not delete legacy analysis fields
- do not change `Transition`
- do not change runtime execution

#### Tests to add

Add unit tests in `src/dfa/tier3_effects.rs` for small hand-built patterns or
compile-time helpers that validate effect compilation for:

- direct match target
- direct `$ -> Match` target
- break path with pure match-at-end
- break path with deferred assertion chain
- break path with consuming tail
- ByteTable target where different bytes produce different effects

#### Tests to run

- `cargo test`

#### Done criteria

- every tier-3-eligible pattern compiles with both legacy and effect data
- dump output shows effect summaries

### Phase 3: Add a shadow lowering adapter from effects back to legacy facts

#### Goal

Prove that the newly compiled effects contain the same information as the old
representation.

#### Files to edit

- `src/dfa/tier3_effects.rs`
- `src/dfa/tier3.rs`

#### Tasks

1. Write a test-only adapter that lowers `CompiledTargetEffects` back into
   legacy categories such as:

   - direct match
   - direct match-at-end
   - break seeds
   - pure break tails
   - deferred tails

2. Compare that lowered form against the existing legacy analysis for every
   target.

3. Put these checks under `#[cfg(test)]` or debug-only validation helpers.

#### Why this matters

If this phase fails, do not proceed.  The entire migration depends on
proving that the effect compiler already captures the old semantics.

#### Tests to run

- `cargo test`

#### Done criteria

- shadow-lowered effect data agrees with legacy data on all existing tests

### Phase 4: Introduce runtime effect application without deleting legacy channels

#### Goal

Start exercising the effect runtime, but continue feeding the old matcher
state so behavior can be compared safely.

#### Files to edit

- `src/dfa/tier3_effects.rs`
- `src/dfa/tier3.rs`

#### Tasks

1. Add an `EffectApplier` helper with methods such as:

   - `apply_now(...)`
   - `schedule_pending(...)`
   - `run_pending_next_byte(...)`
   - `run_pending_end_only(...)`

2. In this phase, let `EffectApplier` feed the legacy fields instead of
   replacing them immediately.

3. Add debug-only comparisons between:

   - legacy pending structures
   - effect-applier-produced pending structures

#### Important rule

Do not replace `step_slow_impl!`, `chunk()`, or `finish()` entirely in one
patch.  Route one semantic channel at a time through the effect applier.

#### Tests to run

- `cargo test`
- a few manual `cargo run --release -- match --debug --chunk-size 1 --tier 3 --unroll-limit 0 ...`

#### Done criteria

- the effect applier can generate legacy-equivalent updates in debug mode

### Phase 5: Migrate break seeds first

#### Why start here?

Break seeds are the smallest isolated pending-effect channel.

#### Current code to replace

- emission in `step_slow_impl!`: `src/dfa/tier3.rs:3142`
- next-byte resolution in `chunk()`: `src/dfa/tier3.rs:3569`
- end-of-input resolution in `finish()`: `src/dfa/tier3.rs:3948`

#### Tasks

1. Replace `pending_break_seeds` with `PendingEffect` entries whose atoms are
   `EffectAtom::AddSeed`.
2. Capture `prev_was_word` when scheduling them.
3. Resolve them via the generic pending-effect path in `chunk()` and
   `finish()`.
4. Delete seed-specific resolution code once the generic path is trusted.

#### Key bug families this phase must preserve

- break-seed deferred assertion gating
- break-seed next-byte timing
- EOI seed resolution

#### Tests to run

- `cargo test`
- re-run any existing regression cases you have for Bugs 27, 28, 38, and 40

#### Done criteria

- `pending_break_seeds` field is gone
- all seed timing behavior goes through pending effects only

### Phase 6: Migrate break tails

#### Why this is the hardest phase

Tail handling is currently split across:

- the slow path
- the pre-step pending resolution in `chunk()`
- `finish()`

This is where most of the special-case complexity lives.

#### Current code to replace

- tail deposition and advancement in `step_slow_impl!`: `src/dfa/tier3.rs:2716`
- pre-step tail resolution in `chunk()`: `src/dfa/tier3.rs:3400`
- end-of-input tail resolution in `finish()`: `src/dfa/tier3.rs:4002`

#### Tasks

1. Replace `pending_break_tails`, `pending_resolved_tails`, and
   `pending_resolved_mae` with pending effects.
2. Represent a tail as an `AddTail` atom plus a guard containing the exact
   assertion chain for that tail.
3. Ensure the assertion chain is evaluated as a full chain, not as a flat
   set of independent assertions.
4. Keep `post_break_tails` temporarily if needed, but only allow effects to
   feed it.
5. Once stable, delete the old reinjection path.

#### Critical gotchas

- per-tail assertion chains must stay per-tail
- a tail behind `\b` then `\B` must require both in order
- byte-specific targets must remain byte-specific for `ByteTable`

#### Tests to run

- `cargo test`
- manual debug runs for Bug 42, Bug 44, Bug 45, Bug 46, and Bug 47 style
  patterns

#### Done criteria

- tail promotion and tail EOI resolution are effect-driven
- `pending_break_tails`, `pending_resolved_tails`, and
  `pending_resolved_mae` are gone

### Phase 7: Migrate deferred assertion match handling

#### Goal

Move `verified_deferred_asserts` and EOI-only match resolution into the same
pending-effect system.

#### Current code to replace

- `resolve_verified_deferred_asserts()` in `src/dfa/tier3.rs:3782`
- deferred resolution portions of `finish()` in `src/dfa/tier3.rs:3920`

#### Tasks

1. Encode assertion-gated `Match` and `MatchAtEnd` as effects instead of as a
   naked list of deferred assert indices.
2. Use `EffectTiming::NextByte` and `EffectTiming::EndOnly` instead of
   separate custom logic.
3. Delete `verified_deferred_asserts` once behavior matches.

#### Tests to run

- `cargo test`
- manual debug runs for contradictory and chained assert patterns

#### Done criteria

- `verified_deferred_asserts` is gone
- `finish()` is materially smaller and easier to read

### Phase 8: Decide whether Proposal 2 is enough or whether Proposal 3 overlap is needed

#### Goal

Make an explicit decision about contamination after the effect migration is
mostly complete.

#### Two possible outcomes

##### Outcome A: Proposal 2 is sufficient

If effects with `required_breaks` on pending and derived material are enough,
you may be able to delete most contamination repair logic without fully
partitioning active state.

##### Outcome B: Proposal 2 needs Proposal 3-style reinforcement

If contamination still leaks because the active state itself remains too
coarse, introduce provenance-aware active entries:

- per-instance entries keyed by `(counter, origin, required_breaks)`
- range entries keyed by `(counter, origin, required_breaks)`

This is where Proposal 2 overlaps with Proposal 3.

#### What to measure before deciding

- number of remaining contamination-specific code paths
- number of places that still need `clean_nb*`
- performance impact of guard evaluation versus mask partitioning

#### Recommended default

Try hard to make Proposal 2 work first.  Escalate to Proposal 3-style active
state partitioning only if tests and profiling show it is necessary.

### Phase 9: Delete legacy structures and simplify the code

#### Goal

Remove the old Tier 3 channels completely so the codebase does not have two
semantic systems permanently layered on top of each other.

#### Legacy structures to remove once replaced

- `break_seeds` field from `Transition`
- `pre_seeds` field from `Transition`, if fully subsumed by effects
- `seeds` field from `Transition`, if fully subsumed by effects
- legacy target booleans that are now represented via effects
- pending legacy matcher fields listed earlier in this document

#### Final cleanup tasks

1. simplify `Transition`
2. simplify `chunk()`
3. simplify `finish()`
4. update `Debug`, `Display`, and dump output to show effect data instead of
   legacy field counts
5. remove temporary shadow adapters and debug-only equivalence bridges

#### Done criteria

- the old side-channel families are gone
- Tier 3 semantics are effect-driven
- debug output exposes the new model clearly

## Detailed File-by-File Checklist

### `src/dfa/mod.rs`

- add the `tier3_effects` module declaration
- keep `DfaState` unchanged initially

### `src/dfa/tier3_effects.rs`

- add IR types
- add compile-time helper functions
- add effect pretty-printing
- add unit tests for effect compilation and guard evaluation

### `src/dfa/tier3.rs`

- extend `Tier3Analysis`
- add effect-aware compile-time population
- route runtime behavior through effect appliers one channel at a time
- shrink matcher state as channels are migrated away

### `src/dump.rs`

- add effect dump support
- add readable output for assertion-chain IDs and guards

### `src/lib.rs`

- keep the existing `match_tests!` harness
- add targeted regression entries when the migration exposes a new subtlety
- if needed, add a small helper to exercise more chunkings for short inputs

## Test Strategy

Use all four layers below.

### 1. Existing regression suite

Always run:

- `cargo test`

This remains the main safety net.

### 2. New effect-compiler unit tests

Add tests that do not depend on the full matcher.  These should validate:

- compiled effect timing
- compiled assertion chains
- byte-specific target effects
- break-gated vs counter-free effect classification

### 3. Shadow equivalence tests

While legacy and effect paths coexist, compare:

- legacy analysis vs shadow-lowered effect analysis
- legacy runtime updates vs effect runtime updates

Use `debug_assert!()` or test-only helpers.

### 4. Manual debug comparisons

For selected tricky patterns, compare:

- Tier 0 debug trace
- current Tier 3 debug trace
- effect-enabled Tier 3 debug trace

Do this especially after Phases 5 through 8.

## Debugging Requirements

The new model must be visible in traces.  Do not repeat the current pattern
of adding semantic state without exposing it.

### Required debug output

For the matcher:

- pending next-byte effects
- pending end-only effects
- for each pending effect: timing, guard, atom list, and stored
  `prev_was_word`

For dump output:

- per-target compiled effects
- assertion-chain contents
- break mask values where present

### Required debug assertions

- no effect atom references a non-consuming origin when it should not
- no `NextByte` effect survives two boundaries
- no `EndOnly` effect is evaluated mid-input
- `required_breaks == 0` is used consistently for counter-free facts

## Common Mistakes To Avoid

### Mistake 1: Replacing everything at once

Do not rewrite `populate()`, `step_slow_impl!`, `chunk()`, and `finish()` in
one patch.

### Mistake 2: Flattening assertion chains again

Do not convert `\b -> \B -> ...` into a flat set of assertion IDs.  Keep a
per-path ordered chain.

### Mistake 3: Losing byte-specific targets

`ByteTable` targets must remain byte-specific from compilation through
execution.  Never fall back to one static per-origin boolean when the target
depends on the consumed byte.

### Mistake 4: Forgetting streaming context

When scheduling `NextByte` and `EndOnly` effects, capture the boundary
context needed to evaluate them later.

### Mistake 5: Leaving two semantic systems around forever

Shadow compilation is good.  Permanent duplication is not.  Once a channel
is effect-driven and validated, delete the legacy version.

## Suggested Commit Breakdown

If you choose to implement this across multiple commits, a good breakdown is:

1. add effect types and debug/dump support
2. shadow-compile target effects
3. add shadow equivalence tests/adapters
4. migrate break seeds
5. migrate break tails
6. migrate deferred assertion match handling
7. remove legacy pending channels
8. contamination decision / optional provenance reinforcement
9. cleanup and doc updates

## Definition of Done

Proposal 2 is implemented successfully when all of the following are true:

- Tier 3 no longer relies on specialized pending vectors for seeds, tails,
  and deferred assertion resolution
- transition consequences are represented through the typed effect model
- current Tier 3 regression tests pass
- debug output makes pending and guarded effects visible
- the code contains a clear, explicit answer to the contamination question:
  either Proposal 2 alone is sufficient, or Proposal 3-style provenance was
  intentionally adopted for active state

## Related Documents

- `docs/tier3-redesign-overview.md`
- `docs/tier3-redesign-proposal-2-typed-effects.md`
- `docs/tier3-redesign-proposal-3-provenance-aware-state.md`
- `docs/tier3-redesign-proposal-4-tier4-derived.md`
