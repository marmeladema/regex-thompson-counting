# Tier 3 Proposal 2: Replace Ad Hoc Logic with a Typed Effect Model

## Goal

Keep a dedicated Tier 3, but redesign its representation so that the
runtime manipulates one explicit effect language instead of several loosely
related booleans, vectors, and special cases.

This is the best option if the project wants to preserve a specialized
non-nested variable-length counter tier without accepting the current level
of semantic drift.

## Thesis

The current Tier 3 implementation does not mainly suffer from "missing
branches".  It suffers from using the wrong abstraction boundary.

Today, semantic facts are scattered across:

- `Tier3OriginKind`
- `Transition`
- static analysis tables like `target_is_match_at_end`
- runtime pending lists in `Tier3DfaMatcher`

Those pieces describe the same underlying phenomenon: the effect of
consuming one byte from one origin under some guard and on some timeline.

This proposal replaces those fragments with a single typed effect model.

## Design Principle

Every semantic consequence of a transition should be represented as an
explicit effect with:

- a payload: what happens
- a guard: under what condition it is valid
- a timing: when it becomes valid

That is the information Tier 3 keeps losing today.

## What the Current Model Loses

The current representation flattens several dimensions that should remain
explicit:

- whether an effect is unconditional or break-gated
- whether an effect is immediate, next-byte, or EOI-only
- whether a tail or seed sits behind one assertion or a whole assertion
  chain
- whether a match-at-end fact belongs to a pure path or a guarded path

This flattening is visible across the bug history:

- contamination bugs: `docs/bugs/019-tier3-counter-free-mae-contamination.md`
- timing bugs: `docs/bugs/028-tier3-break-seed-deferred-assertion-timing.md`
- tail gating bugs: `docs/bugs/042-tier3-pending-break-tails.md`
- chained-assert bugs: `docs/bugs/031-tier3-chained-target-deferred-asserts.md`
- per-tail bugs: `docs/bugs/046-tier3-per-tail-deferred-asserts.md`

## Proposed Representation

The exact type names are flexible, but the model should look roughly like
this.

```rust
enum EffectTiming {
    Now,
    NextByte,
    EndOnly,
}

enum EffectGuard {
    Always,
    CounterBreak { counter: CounterIdx },
    AssertPath { entry: StateIdx, chain: Box<[StateIdx]> },
    And(Box<[EffectGuard]>),
}

enum EffectAtom {
    Seed { counter: CounterIdx, origin: StateIdx, value: u32 },
    ContinueOrigin { origin: StateIdx },
    CounterContinue { counter: CounterIdx, origin: StateIdx, value_delta: u32 },
    AddTail { origin: StateIdx },
    Match,
    MatchAtEnd,
}

struct GuardedEffect {
    timing: EffectTiming,
    guard: EffectGuard,
    atoms: Box<[EffectAtom]>,
}

struct OriginEffect {
    immediate: Box<[EffectAtom]>,
    guarded: Box<[GuardedEffect]>,
}
```

This is only a sketch, but the critical shift is that seeds, tails,
deferred matches, and break-gated material all become instances of one
general idea instead of separate bespoke channels.

## Execution Model

The runtime becomes a simple effect pipeline.

### Phase 0: Resolve pending obligations from the previous boundary

The matcher keeps one queue of pending `GuardedEffect`s.  At the start of a
byte, it evaluates all `NextByte` obligations against the current boundary.
At end-of-input, it evaluates `EndOnly` obligations.

This replaces today's split between:

- `verified_deferred_asserts`
- `pending_break_seeds`
- `pending_break_tails`
- `pending_resolved_tails`
- `pending_resolved_mae`

### Phase 1: Consume the current byte

For each active origin or counter instance, consult the cached
`OriginEffect` for that origin and byte.

### Phase 2: Apply immediate atoms

Apply all `EffectAtom`s whose timing is `Now` and whose guard is already
known to hold.

### Phase 3: Schedule guarded future effects

Effects that need the next boundary or end-of-input are pushed into the
pending obligation queue without special-case code.

## Can Proposal 2 Support the Full Intended Tier 3 Scope?

Yes, in principle.

If Proposal 2 is implemented carefully, it can cover the full feature scope
that Tier 3 is currently intended to handle:

- non-nested counters
- variable-length counter bodies
- deferred assertions on break paths
- post-break consuming tails
- break-triggered downstream seeding
- both current execution backends: range-compressed and per-instance

That scope is larger than Proposal 1's reduced fragment and matches the
current Tier 3 design target in `src/lib.rs:2256` and `src/dfa/tier3.rs:1`.

However, there is an important qualification:

### Typed effects are sufficient only if they preserve provenance where needed

Typed effects solve two major problems immediately:

- they make timing explicit
- they make guard structure explicit

That is enough to replace most of the current ad hoc channels around:

- break seeds
- break tails
- deferred assertion queues
- end-of-input resolution

But the contamination bug family comes from a slightly different source:
the matcher often loses the reason why a state or derived fact is present.

If Proposal 2 is implemented as only a nicer replacement for pending lists,
while live origins, tails, and counter entries still lose their break
history across steps, then Proposal 2 will improve the code a lot but may
still need some contamination-repair logic.

To fully cover current Tier 3 semantics, Proposal 2 therefore needs one of
the following:

1. effects whose guards remain attached to live and pending material across
   multiple steps, for example via a `required_breaks: BreakMask` field, or
2. a provenance-aware state model like Proposal 3.

So the right way to read Proposal 2 is:

- **yes, it can support full intended Tier 3 scope**
- **but only if break provenance is preserved somewhere explicit**

That means a fully realized Proposal 2 naturally overlaps with Proposal 3 on
the provenance dimension.

## Proposal 2 vs Proposal 3

The two proposals are not mutually exclusive, but they solve different parts
of the problem.

### Proposal 2 is the more generic transition language

Typed effects are more generic in the sense that they describe all kinds of
transition consequences in one model:

- immediate actions
- next-byte actions
- end-of-input actions
- assertion-gated actions
- break-gated actions
- seeds, tails, direct matches, and match-at-end

Proposal 2 is therefore the broader semantic cleanup.  It addresses timing,
path guards, and the current sprawl of special-purpose runtime channels.

### Proposal 3 is the more specialized provenance model

Proposal 3 is narrower and more targeted.  It does not try to provide a
general effect language.  Instead, it says that merged runtime state should
explicitly remember which counter breaks were required to make each fact
valid.

That makes Proposal 3 especially strong against contamination bugs, but it
does not by itself solve the broader timing-and-effects problem.

### Which is more generic?

- Proposal 2 is more generic as a representation of transition semantics.
- Proposal 3 is more specialized: it focuses on provenance of live state.

In practice, that means Proposal 2 can stand on its own for a reduced or
carefully engineered full-scope Tier 3, while Proposal 3 is best viewed as
either:

- an alternative implementation strategy for contamination-heavy cases, or
- an extension that strengthens Proposal 2 when provenance still leaks.

### Which is likely to perform better?

My expectation is:

- Proposal 2 is likely better on average because it can keep the current
  DFA-state shape and current counter backends with a smaller risk of state
  explosion.
- Proposal 3 may make some counter-free and break-gated queries simpler, but
  it risks more runtime and cache growth because state and storage may need
  to split by provenance mask.

So Proposal 3 is the more aggressive and likely more expensive option.

### Why consider Proposal 3 at all if Proposal 2 can work?

You would only choose Proposal 3 over Proposal 2 if you conclude one of the
following:

1. the contamination family is so central that provenance must become a
   first-class part of the active state, not just of pending effects
2. the guard-carrying version of Proposal 2 becomes awkward enough that a
   provenance-partitioned state is actually simpler to reason about
3. performance measurements show that direct provenance partitioning is more
   predictable than repeatedly evaluating guard-rich effects

If Proposal 2 fully works, Proposal 3 is not mandatory.  Proposal 3 is best
thought of as the "stronger, more explicit provenance option" rather than
the default path.

### Recommended interpretation

The best default is:

- start with Proposal 2
- make it provenance-ready from day one by reserving a field like
  `required_breaks: BreakMask`
- only escalate toward Proposal 3's full state partitioning if experience
  shows that effect-local provenance is not enough

## Compile-Time Changes

The current analysis is already close to the right raw information, but it
is spread across several functions:

- `analyze_target()`
- `break_closure()`
- `break_consuming_tails()`
- break-seed analysis

Relevant code:

- `src/dfa/tier3.rs:2139`
- `src/dfa/tier3.rs:2355`
- `src/dfa/tier3.rs:2458`

Proposal 2 replaces those with a single effect compiler that walks the
post-consumption target and emits a normalized `OriginEffect`.

That compiler should preserve:

- direct match vs match-at-end
- pure vs assertion-gated paths
- per-tail assertion chains
- break-gated vs unconditional seeds
- byte-specific targets, including `ByteTable`

## How This Addresses the Current Bug Families

### Contamination

Instead of deriving `counter_free_match_at_end` and similar facts from a
coarse DFA state, Proposal 2 makes the guard explicit on each emitted
effect.  A match-at-end fact that requires a counter break remains marked
as such.  It does not need a separate clean-chain repair.

### Deferred assertion timing

Timing is explicit in `EffectTiming`, so next-byte and EOI work use the
same mechanism rather than parallel special cases.

### Tail handling

Tails are just effect atoms with guards and timing, not an out-of-band
state category with its own promotion and reinjection rules.

### ByteTable handling

The effect compiler works from the actual post-consumption target of the
matched byte, so byte-specific targets remain explicit instead of being
forced through a per-origin boolean summary.

## Interaction with Counter Backends

A major advantage of this proposal is that it does not require changing the
counter storage strategy.

The existing backend split can remain:

- range-compressed storage when `all_counters_rangeable`
- per-instance storage otherwise

The backend only needs to answer questions like:

- which instances can continue
- which instances can break
- where to store the resulting values

The effect model governs control-flow semantics, not storage layout.

## Expected Runtime Shape

The matcher state becomes smaller conceptually even if the transition
objects grow somewhat.

Likely long-term matcher fields:

- current DFA state
- counter backend storage
- queue of pending guarded effects
- match flags

Likely fields that disappear or collapse:

- many specialized pending vectors
- tail-specific reinjection paths
- several contamination-specific booleans

## Migration Plan

### Stage 1: Introduce the effect types in parallel

Do not rewrite execution immediately.  First compile effects alongside the
old summaries and compare them in tests.

### Stage 2: Convert one semantic channel at a time

Good first targets:

1. break seeds
2. break tails
3. deferred assertion handling

Each conversion should delete the corresponding special-case path from the
runtime.

### Stage 3: Remove clean-chain repair logic if guards subsume it

If effect guards successfully carry break provenance, the clean no-break
repair chain should no longer be necessary.

### Stage 4: Simplify `finish()`

Once pending EOI effects are just one case of the same effect system,
`finish()` should shrink considerably.

## Main Risks

### Risk 1: Transition objects become too large

An effect-rich transition can be larger than the current compact boolean
summary.  That may increase cache size and population cost.

### Risk 2: Effect normalization becomes subtle

If the effect compiler emits redundant or overlapping guarded effects, the
runtime may do extra work or accidentally reintroduce ambiguity.

### Risk 3: Partial migration keeps both models alive too long

This proposal only helps if the old ad hoc channels are actually removed.
If the effect model is layered on top without deleting the old logic, the
complexity gets worse instead of better.

## Validation Plan

Proposal 2 should be validated with:

1. effect-compiler self-checks against local NFA walks
2. exact step-by-step comparison against a slow reference interpreter
3. differential tests that compare old Tier 3 vs effect-based Tier 3 on the
   subset where both are implemented
4. explicit tests for effect timing and guard evaluation independent of the
   full matcher

## When To Choose This Proposal

Choose Proposal 2 if the project wants:

- to keep a real dedicated Tier 3
- a medium-sized refactor rather than an immediate retreat or a full Tier 4
  unification
- a representation where timing and guard structure are first-class

Do not choose it if the project would rather narrow Tier 3 quickly or is
already ready to unify the counting tiers under one shared semantics.

## Related Documents

- `docs/tier3-redesign-overview.md`
- `docs/tier3-redesign-proposal-1-provable-core.md`
- `docs/tier3-redesign-proposal-3-provenance-aware-state.md`
- `docs/tier3-redesign-proposal-4-tier4-derived.md`
- `docs/tier3-redesign-proposal-2-implementation-plan.md`
