# Tier 3 Proposal 3: Make Provenance Part of the State

## Goal

Eliminate the contamination bug family by explicitly representing where a
state, seed, tail, or deferred assertion came from instead of trying to
recover that information later using a clean reference chain.

This proposal is more invasive than Proposal 2, but it attacks the most
recurrent structural weakness in the current algorithm: loss of provenance.

## Thesis

The binary `no_break` / `with_break` split is not expressive enough once a
merged DFA state contains material that is only valid under some break
history.  The current code tries to repair that by maintaining:

- `current_has_break_extras`
- `clean_nb`
- `clean_nb_cf_mae`
- `clean_nb_is_match`
- contamination-aware seed filtering

Relevant code:

- `src/dfa/tier3.rs:2645`
- `src/dfa/tier3.rs:3640`

That repair machinery exists because the state has forgotten which origins
require which break decisions.

Proposal 3 says the state should never forget that information.

## Core Idea

Every piece of runtime material should carry a provenance requirement:

- which counter breaks had to occur for it to be valid
- whether it is counter-free

The natural representation is a break-requirement mask.

```rust
type BreakMask = u64;

struct TaggedOrigin {
    origin: StateIdx,
    required_breaks: BreakMask,
}

struct TaggedSeed {
    counter: CounterIdx,
    origin: StateIdx,
    value: u32,
    required_breaks: BreakMask,
}

struct TaggedAssert {
    assert_idx: StateIdx,
    required_breaks: BreakMask,
}
```

Under this model:

- counter-free material has `required_breaks = 0`
- material reachable only if counter `c0` broke has `required_breaks = 1 << c0`
- material that depends on two upstream breaks carries both bits

This turns "contamination" from an informal notion into a precise runtime
fact.

## Why the Current Binary Model Fails

The existing design only distinguishes two broad situations:

- the no-break closure
- the with-break closure

But many runtime facts require finer granularity:

- one origin may be valid only if `c0` broke
- another may be valid only if `c1` broke
- another may require both
- a fourth may be entirely counter-free

Once those are all merged into the same DFA state, a later transition can
no longer tell which derived seed, tail, `is_match`, or `is_match_at_end`
fact is safe to treat as unconditional.

That is why the contamination bugs keep returning:

- `docs/bugs/019-tier3-counter-free-mae-contamination.md`
- `docs/bugs/022-tier3-clean-nb-chain-propagation.md`
- `docs/bugs/025-tier3-contaminated-no-break-is-match.md`
- `docs/bugs/030-tier3-contaminated-no-break-deferred-assert.md`
- `docs/bugs/032-tier3-contaminated-counter-free-seeds.md`
- `docs/bugs/048-tier3-contaminated-pre-seeds.md`

## Proposed State Shape

There are two main implementation variants.

### Variant A: Fully tagged origin sets

Store the DFA state as a set of `(origin, required_breaks)` pairs.

This is the most direct model and the easiest to reason about.

### Variant B: Partitioned state

Store one counter-free partition plus zero or more break-gated partitions,
for example:

```rust
struct PartitionedState {
    base: DfaStateId,
    partitions: Box<[(BreakMask, DfaStateId)]>,
}
```

This reuses existing `DfaStateId` machinery more aggressively, but it still
keeps provenance explicit.

The exact representation is less important than the rule: provenance must
survive state merging.

## Runtime Consequences

The runtime should carry provenance through all semantic channels:

- counter instances
- seeds
- tails
- deferred assertions
- match and match-at-end reports

For example, when a counter instance breaks on counter `c`, every effect on
that break path should be tagged with `required_breaks | (1 << c)`.

An effect is usable only when its requirement is satisfied by the break
history of the specific runtime path that produced it.

## Why This Solves the Clean-Chain Problem

With explicit provenance, the matcher does not need to ask questions like:

- "is the current state contaminated?"
- "what would the clean no-break chain have looked like?"
- "should I filter this seed against the clean state?"

Those are all indirect ways of asking for information that should already be
attached to the state or effect.

Counter-free `is_match_at_end` simply means:

- there exists a match-at-end effect with `required_breaks = 0`

Break-gated seeding simply means:

- the seed exists, but with a nonzero requirement mask

No separate clean-chain repair is needed.

## Why Consider This If Proposal 2 Fully Works?

This is the key strategic question, because Proposal 2 and Proposal 3 are
not peers in exactly the same dimension.

### Proposal 2 is broader

Proposal 2 is a general effect model.  It aims to describe:

- immediate actions
- next-byte actions
- end-of-input actions
- assertion-gated actions
- break-gated actions
- seeds, tails, and match signals

So Proposal 2 is the broader cleanup of Tier 3 semantics.

### Proposal 3 is narrower but more forceful about contamination

Proposal 3 is not trying to replace the effect language problem.  It is a
direct answer to one specific structural failure: merged states lose break
provenance.

That means Proposal 3 is:

- less generic than Proposal 2 as a semantic representation
- more explicit than Proposal 2 about how break history is stored

### When Proposal 3 is unnecessary

If Proposal 2 is implemented with guards that remain attached to live and
pending material across steps, for example with a `required_breaks` mask on
effects, tails, seeds, and any other break-derived facts, then Proposal 2
may already solve the contamination family well enough.

In that case, Proposal 3 as a separate redesign is probably unnecessary.

### When Proposal 3 still has value

Proposal 3 becomes attractive only if one of the following turns out to be
true during implementation or benchmarking:

1. typed effects still need too much indirect contamination repair because
   the active state itself remains under-specified
2. guard-heavy effect evaluation becomes harder to reason about than an
   explicit provenance-partitioned state
3. the project wants counter-free vs break-gated facts to be trivial to
   query from the runtime state itself, without re-evaluating guards

### Performance expectation

My expectation is that Proposal 3 is usually the more expensive option.
Making provenance part of the state likely increases:

- cached transition size
- live state count
- storage keys for per-instance or ranged counters

So Proposal 3 should not be viewed as the "default better version" of
Proposal 2.  It is the more explicit but likely heavier model.

### Practical recommendation

The best way to think about Proposal 3 is:

- not as the first thing to build
- not as a more generic replacement for Proposal 2
- but as a stronger follow-on or extension if Proposal 2 proves that timing
  and guard structure are solved while provenance remains the hard part

## Compile-Time Changes

Transition population can no longer stop at "no_break closure" vs
"with_break closure".  It needs to compute provenance-tagged deltas.

For each consumed byte and each origin, the compile-time analysis should
determine:

- which downstream consuming states are reachable without adding new break
  requirements
- which downstream states arise only on the break branch of which counter
- which seeds, tails, and deferred assertions inherit those requirements

This likely means replacing `Transition` with a structure that carries
tagged successor information instead of just two DFA IDs and a handful of
derived booleans.

Relevant code today:

- `src/dfa/tier3.rs:790`
- `src/dfa/tier3.rs:1471`

## Interaction with Counter Storage

The counter backend must become provenance-aware too.

For per-instance storage, each instance would naturally carry a requirement
mask.

For range-compressed storage, the storage key becomes something like:

- `(counter, origin, required_breaks)`

That is a real cost.  It means range compression may remain useful only
when the number of distinct live masks is small.

## Where State Explosion Can Happen

This proposal pays for soundness by making provenance explicit, and the main
risk is combinatorial growth in masks or partitions.

Examples of pressure points:

- many counters whose break paths overlap
- long chains where downstream material depends on several upstream breaks
- patterns where many masks remain simultaneously live

This is why Proposal 3 should be paired with fallback thresholds.  If the
number of live provenance partitions or masks exceeds a cap, the engine
should route the pattern to Tier 4 or Tier 0 instead of forcing Tier 3 to
approximate.

## A Practical Restricted Variant

One practical way to pilot this proposal is to apply exact provenance masks
only when the number of counters is small, for example four or fewer.

That would allow the project to test the semantic value of provenance-aware
state without immediately committing to a large fully general mask space.

If the approach works well, the threshold can be expanded or the mask model
can be compressed further.

## Main Benefits

### 1. The contamination bug family becomes representable

The main question becomes "what requirement mask does this fact carry?"
instead of "is this contaminated in some indirect sense?".

### 2. Counter-free facts become trivial to recognize

Any fact with `required_breaks = 0` is genuinely counter-free.

### 3. Break-gated facts stop impersonating unconditional facts

Seeds, tails, and deferred assertions no longer need special-case filters to
avoid being treated as unconditional.

## Main Risks

### Risk 1: State space growth

Making provenance explicit can increase both cached transition size and live
runtime state.

### Risk 2: The model may still need explicit timing

Provenance alone does not solve next-byte vs EOI timing.  If timing bugs are
equally important, Proposal 3 may still need some of Proposal 2's effect
language ideas.

### Risk 3: Added complexity in range compression

The current range proof is about values at an origin.  A provenance-aware
model asks whether the proof should hold separately per mask partition.

## Validation Plan

Proposal 3 should be validated with:

1. exhaustive small-regex testing where the reference interpreter also
   carries provenance masks
2. assertions that every derived match, seed, or tail has the expected
   requirement mask
3. targeted model-checking of multi-counter break-chain patterns
4. measurements of live mask counts to decide whether the approach stays
   practical

## When To Choose This Proposal

Choose Proposal 3 if:

- the dominant pain is contamination and provenance bugs
- the team is willing to pay a higher implementation and state-management
  cost for a more direct model
- the project still wants a dedicated Tier 3 rather than a Tier 4-derived
  replacement

Do not choose it if the real objective is broader semantic unification or a
quick stability win.

## Related Documents

- `docs/tier3-redesign-overview.md`
- `docs/tier3-redesign-proposal-1-provable-core.md`
- `docs/tier3-redesign-proposal-2-typed-effects.md`
- `docs/tier3-redesign-proposal-4-tier4-derived.md`
- `docs/tier3-redesign-proposal-2-implementation-plan.md`
