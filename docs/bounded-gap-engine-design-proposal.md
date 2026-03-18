# Bounded-Gap Engine Design Proposal

## Purpose

This document proposes a new specialized execution strategy for a very common
class of security-analytics regexes:

- `A.{0,100}B`
- `A[\s\S]{0,50}B`
- `A[^/]{0,20}B`
- `A.{0,100}B.{0,50}C`

These patterns are typically not trying to express arbitrary regex structure.
They are trying to express a **bounded proximity constraint**:

- after `A` matches,
- `B` must occur within at most `K` bytes,
- optionally under a constraint on the bytes between them.

This document explores a dedicated bounded-gap engine that is:

- detected from ordinary `regex-syntax` HIR at compile time
- integrated into the current engine without syntax changes
- especially compatible with Tier 0 / Tier 1 / Tier 2 anchor execution
- easier to reason about than stretching Tier 2/Tier 3 to cover these patterns

## Executive Summary

The core recommendation is:

- add a **new bounded-gap specialization**, not a Tier 2 extension

The new engine should detect whole-pattern concat forms like:

```text
Anchor0  (Gap0  Anchor1)*  GapTail?
```

where:

- each interior `Gap` is a bounded repetition of a single-byte predicate between
  two anchors
- `GapTail` is an optional terminal bounded repetition after the last anchor
- each `Anchor` is a separately compiled regex fragment that can report
  **match-ending events** while streaming

This engine would be a good fit for the current architecture because:

- it reuses the current compiler and matcher machinery for anchors
- it is naturally streaming
- it makes the intended semantics explicit
- it avoids treating "bounded proximity between anchors" as generic overlapping
  counters

The most important design choice is:

- **Version 1 should require anchors that are individually executable by Tier 0,
  Tier 1, or Tier 2 and have fixed exact length**

This keeps the first version small, robust, and useful.

Bounded-length variable anchors are still interesting, but they should be a
second phase because they require a richer anchor event API.

## Motivation

The existing engine can already match patterns like:

```regex
(?i)(?:moondust[\s\S]{0,50}(?:alphaBeacon|bravoBeacon|deltaBeacon).{0,100}(?:spur|planet))
```

but the selected execution path is often Tier 3, because the pattern is encoded
as multiple bounded repeats over broad byte classes.

That encoding is semantically valid, but it is not the best abstraction for what
the user usually means.

In security practice, these regexes often mean:

- "the payload fragment `B` should appear within N bytes after `A`"

This is a **windowed event relation**, not fundamentally a counter problem.

That difference matters because a windowed event relation admits a much smaller
runtime state than a generic non-nested counting NFA/DFA.

## Non-Goals

This proposal does **not** try to:

- replace the general regex engine
- replace Tier 3 for arbitrary non-nested counter patterns
- support every regex that merely contains bounded repeats
- add new user-facing syntax

It is a specialization for a specific structural idiom.

## Current Engine Context

Today the engine chooses among:

- Tier 0: NFA simulator
- Tier 1: pure lazy DFA
- Tier 2: differential-counter DFA
- Tier 3: conditional-transition DFA
- Tier 4: counter-program DFA

Relevant code:

- matcher dispatch: `rethoc-engine/src/lib.rs:3462-3500`
- forced tier dispatch: `rethoc-engine/src/lib.rs:3508-3569`
- public matcher enum: `rethoc-engine/src/lib.rs:3619-3630`

The existing tiers are built around either:

- explicit NFA simulation, or
- DFA summaries of NFA progress and counter state

The bounded-gap specialization proposed here is different:

- it is an **event-chain engine** over anchor matches

That is why it should be introduced as a separate specialization rather than as a
minor extension of Tier 2.

## High-Level Model

## Source shape

The source regex must be recognizable as:

```text
Anchor0  (Gap0  Anchor1)*  GapTail?
```

Where:

- each interior `Gap` is a bounded repetition `{min,max}` of a single-byte
  predicate body
- `GapTail` is an optional final bounded repetition of the same kind
- each `Anchor` is a non-empty regex fragment

Version 1 should still reject a **leading** gap shape such as:

```regex
.{0,20}foo
```

because the specialization is meant to anchor matching around explicit anchor
events, not around an unconstrained initial window.

Examples:

- `foo.{0,100}bar`
- `moondust[\s\S]{0,50}(alphaBeacon|bravoBeacon|deltaBeacon).{0,100}(spur|planet)`
- `file[^/]{0,20}admin`
- `CWS.{254}`

## Semantic interpretation

### Interior gap

For an interior gap between `Anchor_i` and `Anchor_{i+1}`:

- if `Anchor_i` ends at byte position `e`
- and `Anchor_{i+1}` starts at byte position `s`
- then the gap length is:

```text
gap_len = s - e - 1
```

and the gap is valid iff:

- `min_gap <= gap_len <= max_gap`, and
- every byte in `(e, s)` satisfies the gap byte predicate

### Terminal gap

For a trailing gap after the final anchor:

- if the final anchor ends at byte position `e`
- and the current position being evaluated as a potential whole-match end is `p`
- then the trailing gap length is:

```text
tail_len = p - e
```

and the trailing gap is valid iff:

- `min_gap <= tail_len <= max_gap`, and
- every byte in `(e, p]` satisfies the gap byte predicate

Important consequence:

- for `A.{0,K}`, the pattern matches immediately when `A` ends, because a
  trailing gap of length 0 is allowed

This should be treated as part of the intended Version 1 semantics.

## Why this should be a separate engine

Trying to express this through Tier 2 / Tier 2.5 style counters forces the
engine to reason about:

- overlapping counter instances
- break subsets
- `counter_reset`
- seed propagation

But the actual semantic problem is much simpler:

- detect anchor-match events
- carry bounded windows / eligibility relations forward
- validate the next anchor against those windows

That is a different computational problem.

## Proposed Compile-Time Representation

## Core plan type

```rust
struct BoundedGapPlan {
    anchors: Box<[AnchorPlan]>,
    interior_gaps: Box<[GapPlan]>,
    tail_gap: Option<GapPlan>,
    start_anchor: StartAnchorKind,
    end_anchor: EndAnchorKind,
}
```

with the invariant:

- `anchors.len() == interior_gaps.len() + 1`

and:

- `tail_gap.is_some()` means the last anchor is followed by a terminal gap
- `tail_gap.is_none()` means the chain ends at the final anchor

### Whole-plan anchor kinds

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

Important Version 1 rule:

- hoisted whole-pattern end anchoring in Version 1 corresponds only to
  `AssertKind::End`
- whole-pattern line-end anchoring (`AssertKind::EndLF`) is **not** supported in
  Version 1 and must fall back to the normal engine

## Gap plan

```rust
struct GapPlan {
    min_gap: u32,
    max_gap: u32,
    predicate: GapPredicate,
}
```

### Gap predicate

```rust
enum GapPredicate {
    Any,
    ByteClass(ByteClassBits),
}
```

In Version 1, the gap predicate should be stored inline as a 256-bit byte set,
with negation already normalized at compile time.

Why this is the recommended simplification:

- `GapPredicate` belongs to the `BoundedGapPlan`, not to some separate NFA graph
- `ClassIdx` would therefore need an additional ownership story or plan-local
  side table
- the number of gaps in a Version 1 plan is small, so storing one inline
  `ByteClassBits` per non-`Any` gap is acceptable
- `Any` remains worth keeping separate because it lets the runtime skip gap-byte
  invalidation tracking entirely for `.{0,K}` / `[\s\S]{0,K}` style gaps

Important note:

- Version 1 only needs single-byte predicates
- it does not need multi-byte gap bodies or arbitrary body regexes

## Anchor plan

```rust
struct AnchorPlan {
    program: AnchorProgram,
    length_info: AnchorLengthInfo,
}
```

### Anchor program

`AnchorProgram` is a dedicated, non-recursive compiled type for anchor
submatchers.  It is **not** a `Regex`.

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

Why a separate type instead of `Box<Regex>`:

- avoids recursive `Regex` ownership
- anchors do not need full `Regex` semantics (no nested specialization probes,
  no top-level dispatch policy, no full public diagnostics surface)
- makes the compilation boundary explicit: anchors are compiled in
  `CompileMode::Anchor`, which disables bounded-gap detection and other
  top-level-only specializations
- easier memory accounting

Each `Anchor*Program` variant stores only the execution data that the
corresponding tier's anchor scanner needs to emit end-position events.  The
exact internal layout can be refined during implementation, but it should be
deliberately smaller than a full `Regex`.

### Length info

For Version 1:

```rust
enum AnchorLengthInfo {
    Fixed(u16),
}
```

For Phase 2:

```rust
enum AnchorLengthInfo {
    Fixed(u16),
    ExactSmallSet { base: u16, mask: u64 },
    ExactSparse(Box<[u16]>),
}
```

The specialization should deliberately avoid depending on Tier 3 / Tier 4
anchors in Version 1.

## Why anchors are compiled as dedicated programs, not as `Regex` objects

This proposal is **not** recommending a separate literal-only engine.

Anchors should be separately compiled internal programs so they can reuse
existing compilation logic for:

- character classes
- case-insensitive literals
- fixed-length bounded repetitions
- simple alternation

That is how repetitions inside `A` and/or `B` can be supported naturally.

However, anchors are **not** full `Regex` objects.

`AnchorProgram` is a deliberately non-recursive compiled type that stores only
the execution data needed for event-producing anchor runners.  It does not carry
a nested `BoundedGapPlan`, a public diagnostics surface, or top-level matcher
dispatch policy.

This is enforced by compiling anchors in `CompileMode::Anchor`, which disables
bounded-gap detection and other top-level-only specializations.  That prevents
recursive specialization trees and keeps the ownership graph flat.

## Detection From `regex-syntax` HIR

## Detection goal

Detect the bounded-gap idiom without new syntax.

That means compiling ordinary HIR into a `BoundedGapPlan` only when the whole
pattern clearly has the right structure.

## Detection pipeline

### Step 1: strip non-semantic wrappers

Ignore wrappers that do not change the concat structure, such as:

- outer non-capturing groups
- captures (API does not report captures anyway)
- singleton concat wrappers

### Step 2: flatten top-level concat

Obtain a linear sequence of HIR pieces.

### Step 3: partition into alternating anchor / gap pieces

Recognize a gap piece only if it is:

- `Repetition { min, max, sub }`
- with finite `max`
- where `sub` is exactly one byte-consuming predicate atom

All other pieces belong to anchors.

### Step 4: validate the chain

Require:

- at least one gap
- no empty anchors
- anchors and gaps alternate exactly

### Step 4a: normalize away zero-length gaps

If a detected gap has:

```text
min_gap = 0, max_gap = 0
```

then it should not become a real gap stage in the bounded-gap plan.

Instead:

- an interior zero-length gap should be normalized away by merging the adjacent
  anchor fragments into one larger anchor fragment
- a trailing zero-length gap should be dropped entirely

Why:

- `A.{0,0}B` is just adjacency, not proximity
- keeping it as a real gap stage adds runtime complexity for no benefit

If normalizing zero-length gaps removes all gaps from the chain, the specialization
should simply not apply and the pattern should fall back to the normal engine.

### Step 5: compile each anchor independently

Each anchor HIR fragment is compiled using the existing builder, then checked for:

- bounded exact length information
- supported execution hint (Tier 0/1/2 only in Version 1)

If any anchor fails, abort the specialization and fall back to the normal engine.

## Gap recognition rules

Version 1 should recognize only gap bodies that lower to one consuming byte
predicate.

Examples that should qualify:

- `.`
- `[\s\S]`
- `[^/]`
- `[A-Za-z0-9_]`
- `\s`

Examples that should not qualify in Version 1:

- `(ab){0,10}`
- `(?:.|\n\r){0,10}` if it does not normalize to one byte predicate
- any body containing assertions or counters

The easiest implementation strategy is to reuse the same kind of "single atom"
recognition already useful for dense bounded-repeat lowering.

## Anchor admissibility

## Version 1 anchor subset

Version 1 should keep anchor admissibility deliberately narrow:

- non-empty
- exact fixed length
- individually executable by Tier 0, Tier 1, or Tier 2

Examples that should qualify:

- `moondust`
- `(alphaBeacon|bravoBeacon|deltaBeacon)`
- `foo\d{4}`
- `(?i)file`
- `[A-Z]{2}admin`

Examples that should not qualify in Version 1:

- `foo\d{2,4}`
- `.*bar`
- anchors requiring Tier 3 / Tier 4 internally

## Why fixed-length anchors first

This is the most important scoping decision.

If an anchor has fixed length `L`, then when it reports a match ending at
position `p`, its start position is known exactly:

```text
start = p + 1 - L
```

That makes streaming validation against prior gap windows straightforward.

For variable bounded-length anchors, the runtime needs to know **which specific
lengths matched ending at this position**, not just the static min/max length of
the anchor.

That is a much richer API and should be deferred to Phase 2.

## Assertion Support

This proposal should be explicit about assertions, because saying only
"reject deferred assertions" is too imprecise.

The real questions are:

- is the assertion inside a gap or inside an anchor?
- does it affect only local anchor matching, or the meaning of the bytes between
  anchors?
- can it be hoisted into a simple whole-plan flag?

## 1. Assertions in gaps

Version 1 should support **no assertions inside gaps**.

This includes:

- `\b`, `\B`
- `^`, `$`
- multiline start/end assertions
- CRLF assertions
- any zero-width assertion embedded in the repeated gap body

Reason:

- the Version 1 gap model is deliberately a one-byte local predicate over the
  bytes strictly between two anchors
- assertions are boundary conditions, not byte predicates
- allowing assertions inside gaps would immediately break the clean
  `GapPredicate + bounds` abstraction

So these should stay unsupported by the bounded-gap engine:

```regex
foo(?:\b.){0,10}bar
foo(?:^.){0,10}bar
foo(?:\s|\b){0,10}bar
```

Path forward:

- only if the gap representation is later generalized from a byte predicate to a
  small boundary-aware program or mini-NFA
- that is a much larger design and should not be part of Version 1

## 2. Whole-pattern anchoring assertions

Top-level anchoring is much easier.

The plan already has:

- `start_anchor`
- `end_anchor`

So these are good early candidates:

- leading `^` when it lowers to `AssertKind::Start`
- trailing `$` when it lowers to `AssertKind::End`

They should be hoisted out of the chain when the HIR makes that unambiguous.

Why this is easy:

- they constrain the whole chain, not the gap body
- they do not require changing the per-gap runtime model

Version 1 recommendation:

- support top-level start/end anchoring only when it lowers cleanly to
  `AssertKind::Start` / `AssertKind::End`

Semantic detail:

- top-level `^` support in Version 1 means true start-of-input anchoring only
- top-level `$` support in Version 1 means true end-of-input anchoring only
- top-level `StartLF` / `EndLF` are intentionally deferred

## 3. Assertions inside anchors

Assertions inside anchors are the interesting middle ground.

In principle, they are **much more supportable** than assertions in gaps,
because anchor semantics are delegated to an internal anchor runner.

That means the bounded-gap engine does not need to understand the assertion
directly; it only needs a trustworthy anchor match-ending event.

Still, there are several sub-cases.

### 3a. Assertion-free anchors

These should be the Version 1 baseline.

Examples:

- `foo`
- `(?:GET|POST)`
- `foo\d{4}`
- `(?i)file`

### 3b. Fixed-length anchors with local assertions

Examples:

- `\bfoo`
- `foo\b`
- `^foo` as the first anchor
- `bar$` as the last anchor

These are potentially supportable because:

- the chosen anchor runner already knows how to evaluate them
- the bounded-gap layer only needs the final match-ending event

However, they widen the Version 1 correctness surface significantly:

- `\b` / `\B` depend on local boundary facts
- `$` / `EndLF` interact with end-of-input event timing
- CRLF assertions are Tier-1/2-ineligible and would force Tier 0 anchor runners

Version 1 recommendation:

- exclude internal anchor assertions at first, except for top-level
  `AssertKind::Start` / `AssertKind::End` that are hoisted into plan flags

This is a scoping choice, not a claim of impossibility.

### 3c. Deferred assertions inside anchors

Examples:

- `\bfoo`
- `foo\B`
- `foo(?m:$)`

These are **not fundamentally incompatible** with the bounded-gap model.

Important clarification:

- they are much less problematic than assertions inside gaps
- if the anchor runner can already support them correctly, the bounded-gap layer
  can in principle treat them as part of the anchor's local semantics

The real reason to exclude them in Version 1 is engineering confidence:

- they enlarge the chunk-boundary and end-of-input event surface
- they require stronger confidence that "matched ending now" remains trustworthy
  under deferred internal anchor conditions

Path forward:

- once the anchor event API is stable, allow fixed-length anchors with deferred
  assertions, provided the selected anchor runner (Tier 0/1/2) already supports
  them correctly
- this is a good Phase 2a extension, separate from variable-length anchors

### 3d. CRLF assertions inside anchors

These are the hardest assertion sub-case for bounded-gap anchors.

Why:

- they are not Tier 1 / Tier 2 eligible today
- they would force Tier 0 anchor runners
- they enlarge the cross-chunk and end-of-input test surface the most

Recommendation:

- explicitly exclude CRLF assertions in Version 1
- consider them only after ordinary fixed-length deferred-anchor assertions are
  already working

## 4. Summary matrix

### Supported in Version 1

- no assertions inside gaps
- top-level `Start` / `End` hoisted to plan flags when structurally obvious
- otherwise assertion-free anchors only

### Not supported in Version 1

- any assertion inside a gap
- internal anchor assertions like `\bfoo`, `foo\b`, `foo$`
- CRLF assertions inside anchors
- whole-pattern `StartLF` / `EndLF`

### Plausible future support

- fixed-length anchors with deferred assertions, via existing Tier 0 / Tier 1 /
  Tier 2 anchor runners
- top-level multiline/start/end variants when they can be hoisted cleanly

### Likely out of scope for a long time

- assertion-aware gaps
- arbitrary zero-width logic inside the repeated gap body

## Runtime Design

## Overview

The bounded-gap matcher runs all anchors as streaming submatchers and links their
match-ending events through per-gap stage state.

The matcher therefore has two responsibilities:

1. run anchor matchers byte-by-byte and collect anchor-end events
2. validate those events against the gap constraints and propagate success to the
   next stage

## Proposed runtime types

```rust
struct BoundedGapMatcher<'a> {
    plan: &'a BoundedGapPlan,
    position: u64,
    anchors: Box<[AnchorRunner<'a>]>,
    stages: Box<[GapStageState]>,
    matched: bool,
}
```

### Anchor runner

```rust
enum AnchorRunner<'a> {
    Tier0(AnchorTier0Runner<'a>),
    Tier1(AnchorTier1Runner<'a>),
    Tier2(AnchorTier2Runner<'a>),
}
```

### Gap stage state

For Version 1 fixed-length anchors, the cleanest state is based on accepted end
positions of the previous anchor, plus an optional terminal-gap stage.

```rust
struct GapStageState {
    accepted_prev_ends: VecDeque<u64>,
    bad_positions: VecDeque<u64>,
}

struct TailGapState {
    accepted_final_anchor_ends: VecDeque<u64>,
    bad_positions: VecDeque<u64>,
}
```

Where:

- `accepted_prev_ends` stores end positions of `Anchor_i` that have already been
  validated by all previous stages and may still participate in an interior gap
- `bad_positions` stores recent positions where the gap predicate failed

For `TailGapState`:

- `accepted_final_anchor_ends` stores validated end positions of the final
  anchor that may still satisfy the terminal gap as more bytes are consumed

## Why end-position queues are the right V1 model

For a fixed-length next anchor of length `L`, a match ending at `p` starts at:

```text
s = p + 1 - L
```

So `Gap_i` is satisfied iff there exists a prior end position `e` in
`accepted_prev_ends` such that:

```text
min_gap <= s - e - 1 <= max_gap
```

and no forbidden gap byte occurred in `(e, s)`.

This makes the stage state naturally queue-based.

## Handling forbidden gap bytes exactly

The key subtlety is that forbidden bytes matter only **before the next anchor
starts**, not inside the bytes consumed by that anchor.

So a single scalar `last_bad_pos` at the current end position is not enough.

The stage must be able to answer:

- what was the most recent forbidden byte position strictly before the candidate
  anchor start `s`?

That is why `bad_positions` must retain recent bad-byte locations rather than
collapsing to one scalar.

## Stage validation rule

Given:

- current position `p`
- fixed next-anchor length `L`
- candidate start `s = p + 1 - L`

let `bad_before_s` be the most recent forbidden-byte position `< s`, if any.

Then a prior end position `e` is valid iff:

```text
min_gap <= s - e - 1 <= max_gap
and
bad_before_s <= e   (or no such bad byte exists)
```

This is the exact streaming form of "all bytes between the anchors satisfy the
gap predicate".

## Runtime step algorithm (Version 1)

For each input byte at absolute position `p`:

### 1. Feed the byte to every anchor runner

Each anchor runner reports whether its anchor matched ending at `p`.

In Version 1 the event shape can be simple:

```rust
struct AnchorEvent {
    matched_now: bool,
}
```

because the anchor length is fixed and known from `AnchorLengthInfo::Fixed`.

### 2. Update bad-byte queues for each gap

For every gap predicate that rejects the current byte, append `p` to the stage's
`bad_positions` queue.

### 3. Process anchor matches left-to-right through the chain

If `Anchor_0` matches ending at `p`:

- record `p` as an accepted prior end for `Gap_0`

If `Anchor_{i+1}` matches ending at `p`:

- compute its start `s`
- query `Gap_i` for whether some accepted end `e` validates the gap
- if yes, record `p` as an accepted end for stage `i+1`

If the final anchor is accepted:

- when `end_anchor == EndAnchorKind::None`, set `matched = true`
- when `end_anchor == EndAnchorKind::EndOfInput`, do **not** set the final match
  result yet; instead record that the chain is satisfied at the current boundary
  and resolve it only in `finish()`

This is the same high-level idea as the existing engine's distinction between
ordinary mid-stream success and end-of-input-only success.

### 4. Expire dead queue entries

For each gap stage, old accepted end positions can be popped when they are too
old to satisfy any future start:

```text
e < current_possible_start - 1 - max_gap
```

Bad-byte positions can also be expired once they are older than every still-live
accepted end.

### 5. Terminal-gap success checks

If `tail_gap` is present, then after interior stage propagation the matcher must
also check whether the current position satisfies the trailing gap for any
accepted end of the final anchor.

Important Version 1 rule:

- if `tail_gap.min_gap == 0`, then any newly accepted final-anchor end position
  should produce an immediate match at the same input boundary

End-of-input caveat:

- if `end_anchor == EndAnchorKind::EndOfInput`, the trailing-gap success must
  still be treated as a candidate success at the current boundary and only become
  a real match in `finish()` if that boundary is the final input boundary

This is the main additional semantic difference introduced by allowing `GapTail`.

### 6. Terminal-gap ordering and bad-byte handling

For terminal gaps, the gap bytes are `(e, p]`, so the current byte position `p`
is part of the gap being validated.

That means the ordering at position `p` must be:

1. update terminal-gap bad-byte tracking for the current byte
2. then evaluate terminal-gap validity at `p`

Important consequence:

- if the current byte fails the terminal gap predicate, it invalidates every
  pending terminal-gap entry with `e < p`, because `p` lies in `(e, p]`
- a newly accepted final-anchor end at `e = p` is **not** invalidated, because
  `(p, p]` is empty

This ordering should be treated as a Version 1 correctness requirement.

## Data-structure invariants required for linearity

This section is critical because the engine is intended for untrusted patterns
and untrusted payloads.

The bounded-gap engine must **not** retain predecessor links or scan all prior
anchor matches for every new anchor event.

The implementation should maintain these invariants:

1. `accepted_prev_ends` is monotone increasing by position
2. each accepted end is appended at most once and popped at most once
3. `bad_positions` is monotone increasing by position
4. each bad position is appended at most once and popped at most once
5. stage validation never walks all live predecessor ends for one event; it may
   only:
   - pop expired queue heads,
   - consult a small number of frontier values,
   - and append the current accepted end at most once

Without those invariants, the specialization would be too risky for adversarial
use.

For interior gaps, one intended validation strategy is:

1. pop expired accepted ends from the front
2. pop accepted ends invalidated by the most recent bad byte before the candidate
   next-anchor start
3. test the new front against `min_gap`

The destructive pop in step 2 is correct because once a bad byte lies between an
accepted end `e` and some candidate next-anchor start `s`, it will also lie
between `e` and every future `s' >= s`. So that `e` can never become valid again.

## Pathological / Worst-Case Behavior

The canonical stress family is:

```regex
(a|b).{0,1000}(a|b).{0,1000}(a|b)
```

on a payload consisting entirely of `a` and `b` bytes.

In that case:

- every anchor can match ending at essentially every byte
- every gap is broad (`.`)
- so a naive implementation would be tempted to keep a quadratic number of
  predecessor relations

That is exactly what the implementation must avoid.

### What happens in the proposed design

For fixed-length anchors of length 1 and gaps `{0,1000}`:

- stage 0 can accept at most one new prior end per byte
- each accepted end remains relevant for only about `max_gap + next_len`
  future positions
- after that it expires permanently

So even if the anchor matches at every byte, `accepted_prev_ends` is bounded by:

```text
O(max_gap + next_anchor_len)
```

not by total input length.

The same logic applies to later stages.

### Queue-size bound (Version 1 fixed-length anchors)

For gap stage `i` followed by an anchor of fixed length `L_{i+1}`:

- each accepted prior end can remain useful for only about
  `max_gap_i + L_{i+1}` future byte positions

Therefore:

```text
|accepted_prev_ends_i| = O(max_gap_i + L_{i+1})
```

if the anchor can emit at most one end-event per byte position, which is exactly
the Version 1 event model.

Similarly, `bad_positions` only needs to retain entries newer than the oldest
still-live accepted end, so it has the same asymptotic bound.

### Whole-matcher memory bound

For a chain with `N - 1` gaps, the Version 1 bounded-gap state should therefore
be bounded by:

```text
O(sum_i (max_gap_i + L_{i+1}))
```

plus the memory of the anchor runners themselves.

This is pattern-dependent but importantly **independent of total input length**.

### Per-byte work bound

If all anchors can match at every byte, then each byte can produce at most:

- one end-event per anchor

So the bounded-gap layer itself should do only:

```text
O(num_anchors + num_gaps)
```

amortized work per byte, plus the cost of stepping the anchor runners.

The crucial implementation rule is:

- never iterate over all prior accepted ends for one anchor event

If a design requires scanning every currently-live predecessor end for a new
event, it is the wrong design for this service setting.

### Overall worst-case runtime

For Version 1, a reasonable target bound is:

```text
O(input_len * (sum anchor_step_costs + num_gaps))
```

where each `anchor_step_cost` is the existing per-byte cost of the chosen anchor
runner (Tier 0 / Tier 1 / Tier 2).

This means the specialization itself should remain linear in input length for a
fixed accepted plan.

### Important practical implication

If the anchors themselves are pathological and can only be executed efficiently by
Tier 3 or Tier 4, the bounded-gap specialization should not admit them in
Version 1.

That is another reason the Version 1 anchor subset should stay narrow.

## Eligibility and resource caps for adversarial safety

Because patterns are untrusted, the bounded-gap specialization should enforce
explicit build-time limits.

Suggested Version 1 caps:

- maximum number of anchors in a chain
- maximum sum of `max_gap` values across the chain
- maximum admissible anchor size for Tier 0/Tier 1/Tier 2 runners
- maximum fixed anchor length

If any limit is exceeded:

- do not build the bounded-gap plan
- fall back to the normal regex engine

This is important because even a linear-time specialization can still be a bad
fit if its per-regex memory footprint is allowed to grow without bound.

## Complexity

If the chain has `N` anchors, runtime is approximately:

```text
O(N * input_len + total_anchor_events)
```

with bounded amortized queue operations per stage.

This is a very favorable shape for the intended use case, because `N` is usually
small and the gap handling itself is O(1)-ish per event.

## Anchor Event API

## Why the current matcher API is insufficient

Today `AnyMatcher` exposes only:

- `chunk(&[u8])`
- `finish() -> bool`
- `ismatch()`

Relevant code:

- `rethoc-engine/src/lib.rs:3632-3667`

That answers whole-pattern existence, not:

- "did this anchor match ending at the current byte?"

The bounded-gap engine therefore needs a new **internal-only** event API.

There is an important performance caveat, though:

- a purely per-byte `step_event(byte)` API is semantically sufficient
- but it is not the right implementation boundary if the engine wants to reuse
  existing anchor prefilters and re-engagement effectively

If the bounded-gap engine feeds every anchor one byte at a time, then each
anchor is forced into byte-by-byte stepping even when the underlying matcher
could have skipped large spans.

So the design should distinguish:

1. a **semantic** per-byte event model, which is easy to reason about
2. a more practical **scanner/event** implementation model, which is needed for
   speed

## Proposed internal anchor API

```rust
trait AnchorStep {
    fn step_event(&mut self, byte: u8) -> AnchorEvent;
    fn finish_event(&mut self) -> AnchorFinishEvent;
}
```

This is a good semantic model, but it should not be the only implementation
shape considered.

## Future optimization shape: chunk scanners

For long-term performance, the bounded-gap engine should eventually support an
internal scanner API that lets anchors consume chunks and emit sparse
end-position events:

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

Where `emit(end_pos)` means:

- the anchor matched ending at absolute byte position `end_pos`

For Version 1 fixed-length anchors, this is enough because the start position is
recoverable from the static anchor length.

For Phase 2 variable-length anchors, the emitted event would need to carry a
length set instead of only `end_pos`.

## Why scanner/event is better than byte-by-byte stepping

It lets each anchor reuse the optimizations it already has or will have:

- literal prefilters
- memchr-style skipping
- start-state re-engagement
- chunk-oriented DFA scanning

Without that, the bounded-gap engine would semantically work but would leave a
large amount of anchor-side performance on the table.

However, this should be treated as a **follow-up optimization direction**, not as
the mandatory Version 1 implementation boundary.

Version 1 recommendation:

- start with a simpler byte-by-byte `step_event(byte)` anchor API
- make correctness, chunk behavior, and end-of-input handling solid first
- add scanner/event anchor runners later once the specialization is trusted

For Version 1 fixed-length anchors:

```rust
struct AnchorEvent {
    matched_now: bool,
}
```

For Phase 2 bounded variable-length anchors:

```rust
struct AnchorEvent {
    matched_lengths: MatchLengthSet,
}
```

Where:

```rust
enum MatchLengthSet {
    Empty,
    Fixed(u16),
    SmallMask { base: u16, mask: u64 },
    Sparse(Box<[u16]>),
}
```

This event reports the **actual lengths that matched ending now**, not the
anchor's static min/max range.

## Integration with Tier 0 / Tier 1 / Tier 2 anchors

## Tier 0 anchors

Tier 0 can support `matched_now` naturally because it already knows whether a
`Match` state was reached during the current byte step.

For Version 1 fixed-length anchors, this is enough semantically.

Performance note:

- Tier 0 already has prefilter and re-engagement logic for normal matching
- a Tier 0 anchor scanner should preserve that behavior instead of degrading to a
  forced byte-at-a-time loop

For Phase 2 variable-length anchors, Tier 0 would need additional provenance to
report exact matched lengths ending at the current position.

## Tier 1 anchors

Tier 1 is a strong fit for fixed-length anchors:

- it already tracks the exact DFA state after each byte
- fixed-length anchored fragments can report `matched_now` cheaply

Performance note:

- Tier 1 is one of the strongest reasons to prefer a scanner/event interface
- a Tier 1 anchor scanner should be able to reuse its current chunk-oriented
  scanning and prefilter behavior, then emit only the actual end-position events

Phase 2 exact-length-set reporting is possible, but would require extra
compile-time metadata in DFA states or transitions.

## Tier 2 anchors

Tier 2 can also support fixed-length anchors well.

This is useful because bounded-gap patterns may include anchors like:

- `foo\d{4}`
- `(?:GET|POST)` if normalized to fixed length

Performance note:

- as with Tier 1, Tier 2 anchor scanners should retain any existing prefilter or
  chunk-level fast-path behavior
- the bounded-gap engine should consume sparse anchor events, not force Tier 2
  anchors into byte-by-byte stepping unless no better path exists

As with Tier 1, Phase 2 variable-length reporting would require extra metadata.

## Version 1 boundary

To keep the first implementation robust, the bounded-gap engine should only
admit anchors that can supply `matched_now` under:

- Tier 0
- Tier 1
- Tier 2

and have fixed exact length.

This avoids immediate dependency on Tier 3 / Tier 4 event-reporting semantics.

## Version 1 anchor activation

Version 1 does not need to run every anchor from the start.

Recommended policy:

- `anchor[0]` is active from position 0
- `anchor[i+1]` becomes active once stage `i` receives its first accepted end
- once activated, an anchor stays active for the rest of the match

This one-way lazy activation reduces wasted work without introducing complex
deactivation/reactivation state.

## anchored_start implementation choice

Top-level `^` should be implemented in the bounded-gap layer, not by recompiling
the first anchor with different semantics.

For a first anchor of fixed length `L`:

- if it reports a match ending at `p`, its start is `p + 1 - L`
- with `start_anchor == StartAnchorKind::StartOfInput`, accept that event only if
  the computed start is 0

This keeps anchor semantics and bounded-gap semantics cleanly separated.

## end_anchor implementation choice

With `end_anchor == EndAnchorKind::EndOfInput`, a chain that is otherwise fully
satisfied at position `p` should be treated only as a **candidate** full match at
the current boundary.

That candidate becomes a real match only in `finish()`, when the engine knows no
further input bytes will arrive.

Operational consequence:

- ordinary early exit must be disabled while `end_anchor != None`
- the matcher should track only whether the chain is fully satisfied at the
  current boundary, not declare success for earlier boundaries that may later be
  invalidated by additional input

## Event ordering contract

Version 1's per-byte event model naturally processes positions in increasing
order, but the contract should still be explicit.

Required contract:

- anchor events are processed in monotonically increasing absolute position order
- at most one end-event is produced per anchor per end position in Version 1

If scanner/event runners are added later, they must preserve the same ordering.

## Matcher memory reuse

The bounded-gap matcher should participate in `MatcherMemory` from the start.

Version 1 recommendation:

- add a reusable bounded-gap cache analogous to the existing tier caches
- cache stage queues, tail-gap queues, and anchor-runner memory/state across
  matches

Fresh allocation per match should not be the intended steady-state design, even
in Version 1.

## Prefilters and Re-Engagement

The bounded-gap specialization should explicitly reuse anchor prefilters.

This is not just an optional optimization. It is one of the main reasons the
specialization could outperform a generic Tier 3 execution on analyst-style
patterns.

## 1. Per-anchor prefilters

Each anchor should be allowed to reuse whatever prefilter strategy it would use
as a standalone regex:

- memchr-style literal scanning
- memclass / memrange scanning
- start-state re-engagement

Operationally, skipped bytes simply mean:

- that anchor emitted no end-position events for those bytes

This composes naturally with the bounded-gap runtime.

## 2. Whole-plan prefiltering on the first anchor

When no stage windows are active yet, only the first anchor can begin a match.

That means the bounded-gap matcher itself can safely treat the first anchor's
candidate positions as a whole-plan start prefilter.

This is directly analogous to the current engine's start-state re-engagement:

- if nothing is in flight yet, skip to the next plausible first-anchor event

This should be part of the design from the beginning.

## 3. Gap-predicate span skipping

Anchor prefilters are most powerful when gaps are unconstrained:

- `.{0,K}`
- `[\s\S]{0,K}`

In those cases, skipped regions are semantically simple:

- no bad gap bytes need to be tracked
- only position advancement and deadline expiry matter

Constrained gaps are trickier:

- `[^/]`
- `\s`
- `[A-Za-z0-9_]`

Here the bounded-gap layer itself still needs to know whether a forbidden byte
occurred in the skipped span.

So a fully optimized implementation should eventually support gap-level span
queries too, for example:

- next byte that violates `GapPredicate`
- next byte that satisfies `GapPredicate`

These could likely reuse the same low-level machinery already present for:

- `Prefilter::Memchr1/2/3`
- `Prefilter::Memclass`
- `Prefilter::Range`

Version 1 does not need to fully optimize this, but the design should acknowledge
that constrained-gap skipping is a separate concern from anchor prefiltering.

## 4. Best-case vs worst-case prefilter benefit

Best case:

- anchors have strong literal prefilters
- gaps are `Any`
- the engine can jump from anchor candidate to anchor candidate while maintaining
  only small window state

Worst case:

- anchors match at every byte
- gaps are broad and therefore do not help prune anything
- e.g. `(a|b).{0,1000}(a|b).{0,1000}(a|b)` on all-`a`/`b` input

In that case, anchor prefilters provide little or no help, so the bounded-gap
engine must still rely on its queue invariants and explicit linear-time design.

That is why prefilters are an important optimization, but not a correctness
crutch.

## Version 1 Supported Patterns

Examples that should be targeted first:

```regex
foo.{0,100}bar
moondust[\s\S]{0,50}(alphaBeacon|bravoBeacon|deltaBeacon).{0,100}(spur|planet)
file[^/]{0,20}admin
GET\s{0,8}/api/.{0,80}token
CWS.{254}
token[^/]{0,20}
```

Examples with repetitions in anchors that should also fit Version 1 if fixed
length:

```regex
foo\d{4}.{0,100}bar
[A-Z]{2}admin[^/]{0,20}login
```

## Version 1 Unsupported Patterns

Examples that should fall back to the normal engine initially:

```regex
foo\d{2,4}.{0,100}bar
\bfoo.{0,20}bar
foo(?:ab|xyz).{0,20}bar   // if length set is variable and Phase 2 not implemented yet
.{0,20}foo                // leading-gap shape
```

These are still interesting, but they should be Phase 2 work.

## Handling Regular Repetitions In Anchors

This was an explicit requirement of the investigation.

## Version 1 answer

Regular repetitions in `A` and/or `B` can be supported **when they preserve a
fixed exact anchor length**.

Examples:

- `foo\d{4}` -> yes
- `[A-Z]{2}admin` -> yes
- `(?:ab|cd){3}` -> yes if normalized to fixed length

## Phase 2 answer

Bounded variable-length anchors can be supported once the engine can report the
**actual matched length set at each end position**.

This is a real extension and should not be hand-waved.

The key point is:

- static compile-time min/max length is not sufficient
- the runtime needs per-end-position exact length events

That is why the design separates Version 1 from Phase 2.

## Integration Into `RegexBuilder`

## Proposed compile-time hook

Inside `RegexBuilder::build()`, before generic postfix/NFA construction becomes
the only path, add a specialization probe such as:

```rust
fn try_build_bounded_gap_plan(&mut self, hir: &Hir) -> Result<Option<BoundedGapPlan>, Error>
```

If it succeeds:

- store the plan on `Regex`
- store enough metadata to select the new matcher kind

If it fails:

- proceed with normal regex compilation

## Deployment options

There are two plausible deployment modes.

### Mode A: dual compile (safer first version)

- compile and keep both:
  - normal `Regex`
  - `BoundedGapPlan`
- execution prefers the bounded-gap matcher when available

Pros:

- easier differential testing
- easier fallback/debugging

Cons:

- higher per-regex memory

### Mode B: plan-only compile when fully recognized

- if the pattern is fully covered by the bounded-gap model, only keep the plan
  and anchor programs

Pros:

- much better for the "millions of regexes" service use case

Cons:

- more invasive changes to `Regex` layout and diagnostics

Recommendation:

- start with Mode A for bring-up
- move to Mode B once correctness is trusted and the memory benefit matters

## Interaction With Existing Dense Unrolling

The current tree already has denser unrolling for bounded single-atom repeats via
`out_exit` on consuming states.

Relevant code:

- dense bounded single-atom repeat lowering: `rethoc-engine/src/lib.rs:2360-2428`

This is complementary to the bounded-gap engine.

Why:

- dense unrolling helps anchors stay in Tier 0/1/2 and avoid Tier 4 internally
- the bounded-gap engine then consumes those anchors as match-event producers

So the two optimizations reinforce each other.

## Debugging and Diagnostics

The specialization should be visible in diagnostics.

Suggested additions:

- new execution tier name, e.g. `BoundedGap`
- info output showing:
  - number of anchors
  - number of gaps
  - gap predicates and bounds
  - each anchor's fixed length and execution hint

This is important because otherwise users will not understand why a pattern did
not go through Tier 3/Tier 4.

## Correctness Invariants

The implementation should enforce these invariants explicitly.

## 1. Non-empty anchors only

Empty anchors complicate same-position propagation and should be rejected in
Version 1.

## 2. Exact anchor-start position must be known

In Version 1 that means fixed exact length.

## 3. Gap predicate must be one-byte and local

No assertions, counters, or multi-byte sub-bodies in gaps.

## 4. Anchors must be independent match-event producers

They should not depend on Tier 3 / Tier 4 internal semantics in Version 1.

## 5. The bounded-gap matcher must remain streaming

All per-stage state must survive arbitrary `chunk()` boundaries.

## Performance Expectations

For the intended pattern family, the bounded-gap engine should beat Tier 3
because it avoids:

- generic counter tracking
- generic break-path reasoning
- generic tail/effect machinery

Its cost is closer to:

- `O(num_anchors * input_len)` for anchor scanning
- plus bounded amortized queue maintenance for stage validation

And in practice it should often do better than a naive per-byte reading of that
bound because anchor scanners may be able to reuse prefilters and re-engagement
to emit sparse match-ending events.

This is especially attractive when:

- the number of anchors is small, and
- the gaps are broad classes like `.` / `[^/]` / `[\s\S]`

which is exactly the common analyst use case.

## Limitations and Open Questions

## 1. Variable-length anchors are the main Phase 2 challenge

This is the most important non-trivial extension.

It requires per-end-position exact length events, not just static anchor length
ranges.

## 2. Assertions in anchors are a staged-support problem, not a fundamental blocker

Assertions inside anchors are not all equally problematic.

- assertion-aware gaps are fundamentally out of scope for Version 1
- fixed-length anchors with local assertions are a later extension once the
  anchor event API is trusted

So the right posture is:

- exclude most anchor assertions at first,
- but treat them as a phased extension path, not as a dead end

## 3. Top-level line-anchor support can be added later, but should not block Version 1

Version 1 already supports whole-pattern `Start` / `End` anchors.

What remains deferred is whole-pattern line anchoring:

- `StartLF`
- `EndLF`

The current use cases are mostly unanchored scan patterns, so deferring those is
acceptable.

## 4. Alternation of whole chains should be deferred

Version 1 should target a single top-level chain, not arbitrary alternation over
chains.

## Recommended Phased Implementation

## Phase 1: fixed-length anchors only

- detect `Anchor Gap Anchor (Gap Anchor)*`
- support one-byte finite gaps
- support anchors with fixed exact length only
- support top-level `^` / `$` when they can be hoisted into plan flags
- support anchor execution via Tier 0 / Tier 1 / Tier 2
- add a new matcher kind for bounded-gap execution

This phase already supports many important patterns.

## Phase 2: exact small length sets for anchors

- add `MatchLengthSet` event reporting
- allow bounded variable-length anchors with exact per-end-position lengths

This is where bounded repetitions inside anchors become much more powerful.

## Phase 2a: fixed-length anchors with internal assertions

- allow fixed-length anchors with local deferred assertions
- start with ordinary word-boundary and end-line deferred assertions
- keep CRLF assertions excluded until the event API and test surface are well
  understood

## Phase 3: plan-only compilation mode

- stop compiling the full generic regex when the bounded-gap plan fully covers
  the source pattern

This is the memory-optimized service deployment mode.

## Final Recommendation

The bounded-gap engine is a promising specialization for the current codebase.

The strongest version of the proposal is:

- **do not** force this into Tier 2 or Tier 2.5 semantics
- **do** detect the idiom from ordinary HIR
- **do** compile it into a dedicated `BoundedGapPlan`
- **do** reuse Tier 0 / Tier 1 / Tier 2 as anchor submatchers

For performance, that should ideally mean:

- scanner/event-style anchor runners that preserve existing anchor prefilters and
  re-engagement behavior
- not a naive byte-at-a-time wrapper around otherwise fast anchor engines

This gives the engine a new execution strategy that better matches the semantics
security analysts actually want, while fitting cleanly into the existing
streaming architecture.

The most practical Version 1 is narrow but useful:

- finite one-byte gaps, including an optional terminal gap
- fixed-length anchors
- anchor runners built from Tier 0 / Tier 1 / Tier 2 only

That is a good place to start before considering richer anchor-length events or
broader chain detection.
