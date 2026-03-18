# Bounded-Gap Engine Patch Plan

## Purpose

This document turns `docs/bounded-gap-engine-design-proposal.md` into a concrete,
implementation-oriented patch plan.

The goal is to add a bounded-gap specialization that:

- is detected from ordinary `regex-syntax` HIR at compile time
- integrates into the existing `Regex` / `RegexBuilder` pipeline
- preserves the current public API shape as much as reasonably possible
- starts narrow and correctness-first
- leaves clear extension seams for later phases

This plan is deliberately more operational than the design proposal. It calls out
ambiguous steps, offers options where the implementation path is not unique, and
states which option is recommended for Version 1.

## Version 1 Scope

Version 1 should support only the narrow, high-confidence fragment from the
design proposal:

- whole-pattern chains of the form `Anchor (Gap Anchor)* Gap?`
- one-byte finite gap bodies only
- fixed-length anchors only
- anchors individually executable by Tier 0 / Tier 1 / Tier 2 only
- no assertions inside gaps
- no internal assertions inside anchors, except whole-pattern `Start` / `End`
  that can be hoisted cleanly into plan flags

Important Version 1 restriction:

- no **leading** gap shape such as `.{0,20}foo`

Everything else falls back to the existing regex engine.

That narrow scope is not a compromise; it is the reason the feature can be added
with a trustworthy correctness story.

## Public API Constraints

The external/public API should not change much.

Concretely, this means:

- keep `RegexBuilder` as the main compile entry point
- keep `Regex` as the main compiled-regex type
- keep `MatcherMemory::matcher()` and `matcher_for_tier()` as the main matching
  entry points
- do not introduce a second public top-level builder or a separate public engine

Some marginal changes are acceptable if coherent with the existing tiers.

The most likely surface changes are:

- a new execution description in `RegexInfo`
- possibly a new `AnyMatcher` variant

Those are acceptable if they remain consistent with how existing tiers are
already exposed.

## Key Design Choices That Affect The Patch Plan

## 1. Dual compile versus plan-only compile

### Option A: dual compile (recommended for Version 1)

- build the ordinary regex as today
- additionally build and store a `BoundedGapPlan` when the specialization probe
  succeeds
- `matcher()` chooses the bounded-gap matcher when the plan is present

Pros:

- simplest fallback story
- easiest differential testing
- no need to make the bounded-gap plan a complete replacement for all existing
  diagnostics immediately

Cons:

- higher per-regex memory until plan-only mode exists

### Option B: plan-only compile when the whole pattern is recognized

- skip building the full generic regex when the plan fully covers the pattern

Pros:

- best memory story for the service deployment model

Cons:

- much more invasive in Version 1
- harder debugging and validation bring-up

### Recommendation

Use **Option A** in Version 1.

The service-memory concern is real, but the fastest safe route is to get the
specialization correct first, then collapse to plan-only compilation later.

## 2. Public matcher exposure

The current engine exposes a public `AnyMatcher` enum in
`rethoc-engine/src/lib.rs:3619-3630`.

That creates a practical API decision.

### Option A: add `AnyMatcher::BoundedGap` (recommended)

Pros:

- honest and coherent with the existing tiered architecture
- easy internal dispatch and debugging

Cons:

- public enum change

### Option B: hide bounded-gap execution behind an existing variant

Pros:

- avoids a public enum variant change

Cons:

- semantically misleading
- makes debugging and diagnostics worse
- fights the current architecture rather than extending it

### Recommendation

Use **Option A**.

This is a marginal public API change, but it is coherent with how the crate
already exposes specialized execution strategies.

Important Version 1 rule:

- `matcher_for_tier()` should remain `0..4`
- do **not** add a new public forced-tier number in Version 1
- the bounded-gap specialization is selected only by `matcher()` / default
  dispatch when the plan is available

That keeps the public tier numbering stable while still exposing the actual
matcher variant honestly.

## 3. Internal anchor API shape

### Option A: semantic `step_event(byte)` only

Pros:

- simple to reason about

Cons:

- throws away anchor-side prefiltering and re-engagement performance
- makes it harder to converge on the right long-term internal API

### Option B: scanner/event adapters (recommended)

Pros:

- preserves prefiltering
- preserves chunk-oriented fast paths
- still compatible with the same semantic event model

Cons:

- slightly more internal machinery

### Recommendation

Use **Option B** as the long-term internal boundary from the start, but allow a
temporary byte-step implementation behind that boundary during bring-up.

Concretely:

- the bounded-gap matcher should depend on a scanner/event-style internal
  abstraction
- the earliest implementations may internally step byte-by-byte to get
  correctness and chunk behavior right
- the feature should not be considered complete until the intended anchor
  classes preserve their prefilter/re-engagement behavior under that boundary

## 4. Anchor support breadth in Version 1

### Option A: allow all fixed-length anchors, including internal assertions

Pros:

- broader coverage immediately

Cons:

- much larger correctness surface
- more chunk / EOI edge cases

### Option B: assertion-free fixed-length anchors only, plus hoisted whole-pattern
`Start` / `End` (recommended)

Pros:

- narrower and easier to verify

Cons:

- some useful anchors deferred to Phase 2a

### Recommendation

Use **Option B** in Version 1.

## 5. Gap support breadth in Version 1

### Option A: allow any gap body that is bounded and locally evaluable

Too broad. Reject.

### Option B: allow only one-byte finite gap predicates (recommended)

Pros:

- clean `GapPredicate + bounds` model
- easy streaming correctness story

Cons:

- some useful but more complex gaps deferred

### Recommendation

Use **Option B** in Version 1.

## 6. Chain-shape breadth in Version 1

### Option A: anchor-terminated chains only

- `Anchor Gap Anchor (Gap Anchor)*`

Pros:

- simplest initial model

Cons:

- misses useful shapes like `CWS.{254}` and `token[^/]{0,20}`

### Option B: allow an optional terminal gap (recommended)

- `Anchor (Gap Anchor)* Gap?`

Pros:

- significantly better coverage for a simple additional case
- terminal-gap semantics are actually simpler than interior-gap semantics

Cons:

- slightly more compile-time and runtime branching
- requires a few more tests around immediate and delayed terminal success

### Recommendation

Use **Option B** in Version 1.

Still reject leading-gap shapes in Version 1.

## Implementation Phases

## Phase 0: preparatory analysis scaffolding

### Goal

Create the internal analysis helpers needed for a narrow, explicit Version 1
probe before adding any new runtime.

### Changes

- add a compile-time helper to flatten top-level concat HIR into a simpler
  internal sequence
- add a helper that recognizes whether a HIR fragment is a Version 1 gap body
- add a helper that computes whether an anchor fragment has exact fixed length
- add a helper that checks whether an anchor fragment is assertion-free in the
  Version 1 sense

### Suggested internal types

```rust
enum GapBodyClass {
    Any,
    ByteClass(ByteClassBits),
}

struct FixedLengthInfo {
    len: u16,
}
```

Version 1 recommendation:

- keep `GapBodyClass` fully self-contained and inline
- do not use `ClassIdx` indirection for gap predicates in Version 1

Reason:

- gap predicates belong to the `BoundedGapPlan`, not to some separate NFA graph
- a plan-local `ClassIdx` indirection would need extra ownership machinery for
  little benefit in the narrow Version 1 chain shapes
- `Any` stays separate because it enables a meaningful runtime fast path where
  bad-byte tracking is skipped entirely

### Files likely affected

- `rethoc-engine/src/lib.rs`
- possibly a new helper module if `lib.rs` becomes too large

### Tests

- unit tests for fixed-length detection
- unit tests for gap-body recognition
- unit tests for top-level concat flattening
- unit tests that zero-length gaps are normalized away before plan construction

### Done criteria

- no runtime changes yet
- detection helpers exist and are directly testable

## Phase 1: add `BoundedGapPlan` types and storage on `Regex`

### Goal

Introduce the new internal plan representation without yet enabling execution.

### Changes

- add `BoundedGapPlan`, `GapPlan`, `AnchorPlan`, `AnchorProgram`,
  `AnchorEngine`, `AnchorLengthInfo`, `CompileMode`
- store `Option<BoundedGapPlan>` on `Regex`
- update `RegexInfo` and diagnostics plumbing to mention bounded-gap availability
  if present

### Important storage decision

Anchors must **not** be stored as `Box<Regex>` or `Arc<Regex>`.

That would create a recursive type graph where `Regex` contains a plan that
contains more `Regex` objects.  It is awkward for ownership, memory accounting,
and diagnostics, and it risks accidentally letting anchors recurse into
top-level specializations.

Instead, anchors should be stored as dedicated, non-recursive `AnchorProgram`
objects.

### Anchor program type

```rust
struct AnchorProgram {
    engine: AnchorEngine,
    prefilter: Option<Prefilter>,
}

enum AnchorEngine {
    Tier0(AnchorNfaProgram),
    Tier1(AnchorDfaProgram),
    Tier2(AnchorTier2Program),
}
```

`AnchorProgram` stores only the execution data needed for event-producing
anchor runners.  It does not carry a nested `BoundedGapPlan`, a public
diagnostics surface, or top-level matcher dispatch.

Each `Anchor*Program` variant should be a private struct containing only the
compiled state that the corresponding tier's anchor scanner needs.  The exact
fields can be refined during implementation, but they should be deliberately
smaller than a full `Regex`.

### Compile mode

Anchors should be compiled in a distinct mode:

```rust
enum CompileMode {
    TopLevel,
    Anchor,
}
```

In `CompileMode::Anchor`:

- bounded-gap detection is disabled
- other future top-level-only specializations are disabled
- the output is an `AnchorProgram`, not a `Regex`

This prevents recursive specialization trees and keeps the ownership graph flat.

### Plan type

The plan type should reflect Version 1 chain shape explicitly:

```rust
struct BoundedGapPlan {
    anchors: Box<[AnchorPlan]>,
    interior_gaps: Box<[GapPlan]>,
    tail_gap: Option<GapPlan>,
    start_anchor: StartAnchorKind,
    end_anchor: EndAnchorKind,
}

struct AnchorPlan {
    program: AnchorProgram,
    length_info: AnchorLengthInfo,
}
```

with:

```rust
enum StartAnchorKind {
    None,
    StartOfInput,
}

enum EndAnchorKind {
    None,
    EndOfInput,
}
```

with the invariant:

- `anchors.len() == interior_gaps.len() + 1`

### Ownership summary

This means `Regex` will own in Version 1 dual compile mode:

- the ordinary compiled regex (for fallback and differential testing)
- plus the bounded-gap plan containing `AnchorProgram`s (not `Regex` objects)

No type recursion.  No reference counting.  No dedup.

### Files likely affected

- `rethoc-engine/src/lib.rs`
- `rethoc-engine/src/info.rs`
- `rethoc-engine/src/dump.rs` if bounded-gap diagnostics are included there

### Tests

- serialization / display tests for new `RegexInfo` execution text
- internal tests that a recognized pattern stores a plan while unrecognized ones
  do not

### Done criteria

- `Regex` can carry a bounded-gap plan
- no matcher selection changes yet

## Phase 2: compile-time specialization probe

### Goal

Detect Version 1 bounded-gap patterns during `RegexBuilder::build()`.

### Recommended builder seam

Add a helper like:

```rust
fn try_build_bounded_gap_plan(&mut self, hir: &Hir) -> Result<Option<BoundedGapPlan>, Error>
```

### Probe steps

1. strip non-semantic wrappers
2. flatten top-level concat
3. partition into alternating anchor / gap pieces, allowing one optional trailing
   gap
4. normalize away zero-length gaps (`{0,0}`)
5. hoist whole-pattern `Start` / `End` when structurally unambiguous
6. validate Version 1 restrictions
7. compile each anchor fragment as an `AnchorProgram` using
   `CompileMode::Anchor`
8. if all steps succeed, return `Some(plan)`
9. otherwise return `None` and continue normal compilation

Zero-gap normalization rule:

- an interior `{0,0}` gap is merged away by concatenating the adjacent anchor
  fragments into one anchor fragment
- a trailing `{0,0}` gap is dropped entirely
- if normalization removes all gaps, the bounded-gap specialization no longer
  applies and the pattern falls back to the normal engine

### Important ambiguity: whether to compile anchors with the same builder config

### Option A: inherit the parent builder config (recommended)

- same unroll limit
- same merge-repetitions setting
- same max-states estimate limits where sensible

This is the most predictable behavior.

### Option B: use separate special anchor settings

This adds too much policy surface for Version 1.

### Recommendation

Use **Option A**.

### Tests

- recognition tests for positive examples
- rejection tests for near misses:
  - assertions inside gaps
  - variable-length anchors
  - empty anchors
  - leading-gap shapes
  - alternating whole chains

- positive tests for terminal-gap shapes such as:
  - `CWS.{254}`
  - `token[^/]{0,20}`

- normalization tests for:
  - `foo.{0,0}bar`
  - `foo[^/]{0,0}`

- hoisting tests for:
  - top-level `^` lowering to `Start`
  - top-level `$` lowering to `End`

- rejection tests for whole-pattern line anchors:
  - `StartLF`
  - `EndLF`

### Done criteria

- recognized patterns store a plan
- all other patterns continue through the normal compiler unchanged

## Phase 3: internal anchor runner adapters

### Goal

Create internal adapters that run anchors and emit end-position events.

### Recommended interface

Use the scanner/event design from the proposal as the intended internal
boundary.

Suggested internal trait:

```rust
trait AnchorScanner {
    fn scan_chunk(
        &mut self,
        input: &[u8],
        base_pos: u64,
        emit: &mut dyn FnMut(u64),
    );

    fn finish_scan(&mut self, emit: &mut dyn FnMut(u64));
}
```

### Implementation plan

- implement `AnchorScanner` for each `AnchorEngine` variant:
  - `AnchorNfaProgram` (Tier 0)
  - `AnchorDfaProgram` (Tier 1)
  - `AnchorTier2Program` (Tier 2)
- Version 1 events carry only end positions because anchor length is fixed

It is acceptable for the earliest bring-up code to implement some of these
scanners via internal byte-step adapters, but the phase is not complete until
the intended anchor classes preserve their prefilter/re-engagement behavior.

### Important note on adapter placement

The anchor scanners should be private structs built directly on top of the
`AnchorProgram` / `Anchor*Program` internals, **not** wrappers around the public
`AnyMatcher` type.

This is the natural consequence of the `AnchorProgram` decision:

- anchors are not `Regex` objects, so there is no public matcher to wrap
- the scanner implementations consume `Anchor*Program` data directly
- this avoids accidental coupling between anchor-scanner semantics and public
  matcher semantics

### Tests

- unit tests for each anchor runner on fixed-length examples
- cross-chunk tests
- tests that event positions are monotonically increasing within a chunk
- tests that prefilter-based scanners emit the same end-position events as naive
  stepping

### Event contract

The event contract should be explicit:

- events are observed in monotonically increasing absolute position order
- at most one end-event is produced per anchor per end position in Version 1

### Done criteria

- fixed-length Tier 0/1/2 anchors can emit correct end-position events

## Phase 4: bounded-gap runtime state and matcher

### Goal

Implement the Version 1 streaming runtime.

### Runtime state

Use the queue-based stage design from the proposal:

```rust
struct GapStageState {
    accepted_prev_ends: VecDeque<u64>,
    bad_positions: VecDeque<u64>,
}
```

and a matcher like:

```rust
struct BoundedGapMatcher<'a> {
    plan: &'a BoundedGapPlan,
    position: u64,
    anchors: Box<[AnchorRunner<'a>]>,
    stages: Box<[GapStageState]>,
    tail_stage: Option<TailGapState>,
    anchor_active: Box<[bool]>,
    matched_here: bool,
    matched: bool,
}
```

Where `matched_here` means:

- the chain is fully satisfied at the current boundary, but may still require
  `finish()` resolution when `end_anchor == EndAnchorKind::EndOfInput`

with an additional tail-gap state when `plan.tail_gap.is_some()`:

```rust
struct TailGapState {
    accepted_final_anchor_ends: VecDeque<u64>,
    bad_positions: VecDeque<u64>,
}
```

### Required invariants

The implementation must enforce the linearity invariants from the proposal:

- monotone accepted-end queues
- monotone bad-position queues
- append-at-most-once / pop-at-most-once
- no scanning of all live predecessors for a single event

The same rules must apply to the terminal-gap queues too.

For interior gaps, the intended destructive-pop validation algorithm is:

1. pop expired accepted ends from the front
2. pop accepted ends invalidated by the most recent bad byte before the
   candidate next-anchor start
3. test the new front against `min_gap`

The destructive pop in step 2 is intentional and correct. Once a bad byte lies
between an accepted end `e` and some candidate start `s`, that same bad byte
will also lie between `e` and every future `s' >= s`, so `e` can never become
valid again.

### Important ambiguity: event processing order inside a byte/chunk

If several anchors emit events ending at the same byte position, the engine must
define a deterministic order.

### Option A: process stage updates left-to-right (recommended)

This matches the chain semantics naturally:

- earlier anchors may enable later anchors at the same end position only if the
  semantics actually allow that through fixed anchor lengths and zero-gap bounds

### Option B: batch all events then resolve repeatedly to a fixpoint

Too complicated for Version 1.

### Recommendation

Use **Option A**, and explicitly document the same-position propagation rule.

### Additional terminal-gap rule

The runtime must also document the same-boundary success rule for trailing gaps:

- if `tail_gap.min_gap == 0`, a newly accepted final-anchor event should cause an
  immediate full-pattern match at that same boundary

This rule must be tested explicitly because it is easy to miss when the
implementation is otherwise event-driven.

### Additional terminal-gap bad-byte ordering rule

At position `p`, if `tail_gap` is present:

1. record the current byte as bad for the terminal gap if its predicate fails
2. then evaluate terminal-gap validity at `p`

This ordering is required because terminal-gap bytes are `(e, p]`, so the
current byte participates in the gap being validated.

Important consequence:

- a bad byte at `p` invalidates every pending terminal-gap entry with `e < p`
- a newly accepted final-anchor end at `e = p` is not invalidated, because
  `(p, p]` is empty

### anchored_start / anchored_end semantics

Version 1 should not model these as booleans.

Required semantics:

- `StartAnchorKind::StartOfInput` is enforced in the bounded-gap layer by
  filtering first-anchor events whose computed start position is not 0
- `EndAnchorKind::EndOfInput` means true whole-input end anchoring only

Do not hoist `StartLF` / `EndLF` in Version 1.

Operational consequence:

- when `end_anchor == EndAnchorKind::EndOfInput`, the matcher must not declare
  a final match during ordinary byte processing
- instead it tracks only whether the chain is fully satisfied at the **current**
  boundary and resolves that candidate in `finish()`
- ordinary early exit is therefore disabled when `end_anchor != None`

### Lazy anchor activation

Include one-way lazy anchor activation in Version 1:

- `anchor[0]` active from the start
- `anchor[i+1]` becomes active once stage `i` receives its first accepted end
- once active, an anchor remains active for the rest of the match

### Tests

- simple `A.{0,K}B` behavior tests
- multi-gap chain tests
- terminal-gap tests like `A.{K}` and `A[^/] {0,K}` equivalents
- gap predicate tests for `Any` and constrained gaps like `[^/]`
- cross-chunk tests where the gap spans chunk boundaries

Specifically add:

- `A.{0,3}` immediate-success tests
- `A.{2,3}` delayed-success tests
- `A[^/]{0,3}` invalidation tests when `/` occurs after `A`
- cross-chunk `A.{254}` tests where `A` ends in one chunk and the trailing gap
  completes in another
- whole-pattern `End` tests such as `foo.{0,3}$` on exact end-of-input matches
- whole-pattern `End` non-early-exit tests where a candidate match is followed by
  more bytes and therefore must not succeed
- rejection tests for multiline top-level `^` / `$` lowering to `StartLF` /
  `EndLF`
- anchored-start filtering tests for first-anchor start position 0 only
- lazy-activation sanity tests showing later anchors stay dormant until enabled

### Done criteria

- `BoundedGapMatcher` is correct on Version 1 recognized patterns
- invariants are exercised by tests

## Phase 5: matcher dispatch integration

### Goal

Make default matcher selection use the bounded-gap plan when available.

### Recommended dispatch rule

In `MatcherMemory::matcher()`:

- if a bounded-gap plan is present, prefer `AnyMatcher::BoundedGap`
- otherwise keep the existing tier dispatch unchanged

### Public API recommendation

- add `AnyMatcher::BoundedGap`
- keep `matcher_for_tier()` unchanged at `0..4`
- do not add a public forced bounded-gap selector in Version 1

### Important ambiguity: whether bounded-gap should outrank Tier 1/2/3 always

### Option A: always prefer bounded-gap when available (recommended for bring-up)

Pros:

- simplest semantics
- easiest to validate the specialization in real use

Cons:

- may occasionally choose bounded-gap when a normal tier would be faster

### Option B: choose based on heuristic cost comparison

Pros:

- potentially better performance tuning

Cons:

- harder to validate initially

### Recommendation

Use **Option A** in Version 1.

Once the specialization is trusted, heuristics can be added later.

### Tests

- dispatch tests showing recognized patterns choose bounded-gap
- existing non-recognized patterns remain unchanged
- `matcher_for_tier()` behavior remains unchanged for `0..4`

## Phase 6: bounded-gap cache in `MatcherMemory`

### Goal

Reuse bounded-gap allocations across matches, consistent with the rest of the
engine.

### Changes

- add a `bounded_gap_cache` field to `MatcherMemory`
- cache reusable queue storage and anchor-runner state there

### Recommendation

Do this in Version 1, not as later polish. Fresh allocation per match is the
wrong default for the intended service setting.

### Tests

- repeated-match tests confirming memory is reused without semantic drift

## Phase 7: prefilter and re-engagement integration

### Goal

Make sure the specialization does not accidentally discard anchor performance.

### Required work

- preserve per-anchor prefilter behavior in anchor scanners
- preserve per-anchor re-engagement when the anchor runner becomes quiescent
- support whole-plan first-anchor prefiltering when no windows are active

### Optional follow-up work

- gap-predicate span skipping for constrained gaps

This should be explicitly treated as a second optimization step, not as a core
correctness dependency.

### Tests

- correctness tests that skipping does not change match results
- tests where anchor candidates are sparse
- tests where anchors match every byte to prove invariants still hold when
  prefilters provide no help

## Phase 8: diagnostics and explainability

### Goal

Make the specialization visible and debuggable.

### Changes

- `RegexInfo` should mention bounded-gap execution when selected
- include:
  - number of anchors
  - number of gaps
  - gap predicates and bounds
  - anchor execution hints
  - anchor fixed lengths

### Optional but recommended

- add a bounded-gap section to debug/dump output

### Important ambiguity: numeric execution tier reporting

`RegexInfo::ExecutionInfo` currently exposes a numeric `tier` plus a human-readable
`tier_name`.

Bounded-gap execution does not fit neatly into the existing `0..4` tier ladder.

### Option A: report `tier = 5` for bounded-gap

Pros:

- simple
- honest that this is a new execution strategy

Cons:

- expands the apparent tier numbering contract in diagnostics

### Option B: keep `tier` as the ordinary fallback tier and add a separate
specialization field

Pros:

- keeps the legacy tier numbering stable

Cons:

- more schema change in diagnostics
- easier to misread if users only look at `tier`

### Recommendation

Use **Option B** if you want to preserve the meaning of `tier = 0..4` strictly in
Version 1.

Concretely:

- keep `execution.tier` as the ordinary underlying engine family that would have
  run without the specialization
- add an optional specialization string such as `"BoundedGap"`
- set `tier_name` to something explicit like:
  - `BoundedGap specialization (anchors via Tier 0/1/2)`

This is the least surprising public-diagnostics story while the specialization is
still new.

### Tests

- text output tests
- JSON info tests

## Phase 9: adversarial validation

### Goal

Prove the specialization is safe under untrusted patterns and payloads.

### Required test families

#### 1. Pathological dense-anchor matches

Examples:

- `(a|b).{0,1000}(a|b).{0,1000}(a|b)` on all-`a`/`b` input
- `(foo|bar).{0,500}(foo|bar)` on repeated matching payloads

Assertions:

- queue sizes remain bounded as expected
- runtime remains linear in input length for fixed plans

#### 2. Cross-chunk correctness

Examples where:

- anchor ends in one chunk
- gap spans several chunks
- next anchor ends in a later chunk

and additionally:

- terminal-gap-only matches complete in a later chunk after the final anchor end

#### 3. Gap predicate invalidation tests

Examples for:

- `[^/]`
- `\s`
- exact bad-byte boundary just before next anchor start
- exact bad-byte boundary inside a trailing gap before terminal success

#### 4. Differential tests against the normal engine

For every recognized bounded-gap pattern in the test corpus:

- default engine result == forced Tier 0 result == `regex` oracle where valid

#### 5. Resource-cap rejection tests

Patterns just below and just above the chosen Version 1 caps.

### Optional but strongly recommended

- ignored property/fuzz test generating small bounded-gap chains and comparing the
  specialization against the normal engine

## Post-Version-1 extension hooks

Version 1 should leave explicit seams for later work, even if it does not
implement them yet.

### Phase 2a hook: fixed-length anchors with internal assertions

- design the anchor admissibility check so that "assertion-free only" is a
  policy gate, not a structural assumption baked into the entire pipeline

### Phase 2 hook: exact small length sets

- design event types so they can later grow from `emit(end_pos)` into
  `emit(end_pos, MatchLengthSet)` without rewriting the whole runtime

### Plan-only compilation hook

- keep the plan storage and diagnostics separate enough that the normal regex can
  eventually become optional when the plan fully covers the pattern

## Required Test Matrix

## Public behavior tests

Add end-to-end tests for:

- recognized bounded-gap patterns
- rejected near-miss patterns
- whole-pattern `Start` / `End` hoisting cases
- whole-pattern `StartLF` / `EndLF` rejection cases
- cross-chunk cases
- terminal-gap cases

## Internal analysis tests

Add unit tests for:

- fixed-length detection
- gap recognition
- chain partitioning
- assertion rejection in gaps
- top-level anchoring extraction

## Internal runner tests

Add tests for:

- Tier 0 anchor event emission
- Tier 1 anchor event emission
- Tier 2 anchor event emission
- scanner/event equivalence with naive stepping

## Adversarial tests

Add tests for:

- queue bounds under dense matches
- bad-position queue expiry
- resource-cap rejection

## Success Criteria

The Version 1 patch series is done when all of the following hold:

1. Recognized Version 1 patterns compile to a `BoundedGapPlan`.
2. Default matcher dispatch selects the bounded-gap specialization when the plan
   is present.
3. The public compile/match API still revolves around `Regex`, `RegexBuilder`,
   `MatcherMemory`, and `AnyMatcher`.
4. `matcher_for_tier()` remains `0..4` and unchanged in behavior.
5. Fixed-length Tier 0/1/2 anchors emit correct end-position events.
6. The bounded-gap matcher is correct for both anchor-terminated and terminal-gap
   chain shapes.
7. The bounded-gap matcher is correct across arbitrary chunk boundaries.
8. `MatcherMemory` reuses bounded-gap state across matches.
9. Queue-size and linearity invariants hold on adversarial Version 1 patterns.
10. Existing patterns not recognized by the probe behave identically to today.
11. `cargo test`, `cargo clippy -- -D clippy::all`, and `cargo fmt -- --check`
   all pass.

## Recommended Execution Order

1. Phase 0 — preparatory analysis helpers
2. Phase 1 — plan types and storage on `Regex`
3. Phase 2 — compile-time probe
4. Phase 3 — internal anchor scanners
5. Phase 4 — bounded-gap runtime
6. Phase 5 — default matcher dispatch
7. Phase 6 — bounded-gap cache in `MatcherMemory`
8. Phase 7 — prefilter / re-engagement integration
9. Phase 8 — diagnostics
10. Phase 9 — adversarial validation

## Final Recommendation

The safest Version 1 is deliberately narrow and explicit:

- dual compile mode
- fixed-length anchors only
- one-byte finite gaps only, including an optional terminal gap
- no internal anchor assertions except hoisted whole-pattern `Start` / `End`
- scanner/event as the internal boundary, with byte-step bring-up allowed behind
  it temporarily
- one-way lazy anchor activation
- bounded-gap cache in `MatcherMemory`
- bounded-gap default dispatch, but no new public forced-tier number

That gives the implementation a clear, coherent first landing point without
forcing large public API changes or speculative support for the hardest cases.

Once that is stable, the two most natural follow-on steps are:

1. fixed-length anchors with deferred assertions (Phase 2a)
2. exact small length-set events for bounded variable-length anchors (Phase 2)
