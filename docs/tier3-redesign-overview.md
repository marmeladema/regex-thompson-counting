# Tier 3 Redesign Overview

## Purpose

This document summarizes why Tier 3 has become bug-prone, what semantic
information the current algorithm loses, and what redesign directions are
available.  It is an index for the four detailed proposal documents:

- `docs/tier3-redesign-proposal-1-provable-core.md`
- `docs/tier3-redesign-proposal-2-typed-effects.md`
- `docs/tier3-redesign-proposal-3-provenance-aware-state.md`
- `docs/tier3-redesign-proposal-4-tier4-derived.md`

The goal is not just to stop the next fuzz bug.  The goal is to decide
whether Tier 3 should be narrowed, rebuilt, or replaced so that the engine
has a sound long-term story for non-nested variable-length counters.

For the most implementation-ready next step, see:

- `docs/tier3-redesign-proposal-2-implementation-plan.md`

## Current Tier 3 Model

At the top of `src/dfa/tier3.rs`, Tier 3 is described as a DFA with two
successors per counting transition: `no_break` and `with_break`.  At
runtime, live counter values choose which successor is semantically valid.

That description is true, but incomplete.

The actual runtime state also includes:

- counter instances or counter ranges
- post-break consuming tails
- verified deferred assertions
- pending break seeds
- pending break tails
- pending resolved tails and match-at-end flags
- contamination tracking (`current_has_break_extras`)
- a separate clean no-break chain (`clean_nb`, `clean_nb_cf_mae`,
  `clean_nb_is_match`)

The key observation is that Tier 3 is no longer a closed semantic object.
The cached DFA transition does not contain enough information to decide the
next step by itself, so the matcher carries several side channels that try
to reconstruct path information lost by merging.

Relevant code anchors:

- `src/dfa/tier3.rs:1`
- `src/dfa/tier3.rs:70`
- `src/dfa/tier3.rs:790`
- `src/dfa/tier3.rs:2554`
- `src/dfa/tier3.rs:3330`

## Why the Current Design Is Brittle

### 1. Provenance is lost

The DFA state knows which NFA consuming states are present, but not why
they are present in this particular runtime state.  It cannot distinguish:

- counter-free presence
- presence because a counter could break
- presence because a specific counter did break
- presence gated by an unresolved assertion on a break path

This is the root of the contamination bug family:

- `docs/bugs/019-tier3-counter-free-mae-contamination.md`
- `docs/bugs/022-tier3-clean-nb-chain-propagation.md`
- `docs/bugs/025-tier3-contaminated-no-break-is-match.md`
- `docs/bugs/030-tier3-contaminated-no-break-deferred-assert.md`
- `docs/bugs/032-tier3-contaminated-counter-free-seeds.md`
- `docs/bugs/048-tier3-contaminated-pre-seeds.md`

The clean-chain machinery is a repair step for this missing information.

### 2. Timing is not modeled explicitly

Tier 3 has effects that live on three different timelines:

- immediate effects on the current byte
- next-byte effects whose assertions were deferred
- end-of-input-only effects

Those timelines are not represented in one unified way.  Instead they are
split across `verified_deferred_asserts`, `pending_break_seeds`,
`pending_break_tails`, `pending_resolved_tails`, and `finish()` logic.

This drives the recurring timing bugs:

- `docs/bugs/026-tier3-counter-break-deferred-assert-mid-input.md`
- `docs/bugs/028-tier3-break-seed-deferred-assertion-timing.md`
- `docs/bugs/041-tier3-stale-deferred-assert-fast-path.md`
- `docs/bugs/042-tier3-pending-break-tails.md`
- `docs/bugs/045-tier3-pure-break-mae-deferred.md`

### 3. Guard structure is flattened too aggressively

Several Tier 3 summaries started as flat lists or booleans even though the
underlying NFA paths are guarded by path-specific assertions:

- `break_deferred_asserts`
- `target_deferred_asserts`
- `break_consuming_states`
- `break_seeds`

The bug history shows repeated reintroduction of missing structure:

- chained deferred assertions
- per-tail deferred assertions
- pure vs guarded tails
- shortest deferred path selection

See:

- `docs/bugs/031-tier3-chained-target-deferred-asserts.md`
- `docs/bugs/043-tier3-chained-contradictory-asserts.md`
- `docs/bugs/046-tier3-per-tail-deferred-asserts.md`
- `docs/bugs/049-tier3-deferred-path-shortcut.md`

### 4. Byte-specific targets do not fit static per-origin summaries

`ByteTable` states keep exposing places where Tier 3 tried to summarize a
byte-dependent target using a per-origin boolean.  The later additions of
byte-specific `is_match` and `is_match_at_end` in `Tier3OriginKind::Advance`
are examples of the representation being stretched to recover that detail.

See:

- `docs/bugs/035-tier3-byte-table-break-tail.md`
- `docs/bugs/037-tier3-byte-table-tail-mae.md`
- `docs/bugs/039-tier3-dead-target-byte-table-mae.md`
- `docs/bugs/047-tier3-byte-table-tail-direct-match.md`

### 5. The same semantics are implemented in several places

Tier 3 behavior is distributed across:

- `populate()` for transition construction
- `step_slow_impl!` for counting execution
- pre-step pending resolution in `chunk()`
- `finish()` at end-of-input

As a result, fixes are easy to land in one branch and miss another branch.
Bug 50 is a clean example: the same resolved-seed remapping bug existed in
both the counting and non-counting populate paths.

See `docs/bugs/050-tier3-resolved-seed-nonconsume-remap.md`.

## What Makes a Sound Tier 3 Hard

Tier 3 is trying to sit between two cleaner models:

- Tier 2, which works because fixed-length bodies admit a strong,
  regularized differential-counter model
- Tier 4, which works by keeping an explicit counter-program semantics

Tier 3 covers the awkward middle fragment: non-nested counters, but with
arbitrary body structure and length.  In that fragment, the core semantic
question is not just "which NFA states are reachable after this byte?".
It is:

- which NFA states are reachable
- under which break decisions
- with which counter updates
- under which assertion obligations
- and on which timeline those obligations resolve

The current representation only stores part of that answer.

## Proposal Summary

| Proposal | Core idea | Main benefit | Main cost | Confidence |
|----------|-----------|--------------|-----------|------------|
| 1 | Shrink Tier 3 to a provable core | Immediate stability | Reduced Tier 3 coverage | High |
| 2 | Replace ad hoc summaries with typed effects | Keeps dedicated Tier 3 with clearer semantics | Medium refactor | Fairly high |
| 3 | Make provenance part of the state | Directly attacks contamination bugs | High state and implementation complexity | Medium |
| 4 | Rebuild Tier 3 as a specialization of Tier 4 semantics | Best long-term soundness story | Largest refactor | Highest long-term confidence |

## Recommendation

### Short term

Adopt Proposal 1 immediately if the priority is to stop the current bug
stream.  It gives up acceleration on some patterns, but it does so in a
controlled and explainable way.

At the same time, build the validation infrastructure described below.

### Medium term

If a dedicated Tier 3 remains important, Proposal 2 is the best next step.
It keeps the current high-level architecture but replaces the scattered
repair channels with one explicit effect model.

Proposal 3 is worth considering only if contamination is judged to be the
dominant remaining problem and the team is comfortable with a more complex
state representation.

### Long term

Proposal 4 is the strongest architecture.  It replaces "multiple subtly
different semantics" with "one semantics, several execution backends".
That is the cleanest way to stop repeating the same correctness work in
Tier 3 and Tier 4.

## Shared Validation Program

No redesign should rely on fuzzing alone.  Fuzzing is essential, but it is
best at finding counterexamples after the fact.  To gain confidence in
soundness, the engine should also have the following validation layers.

### 1. Transition self-checks

For each compile-time Tier 3 summary, add a test-only recomputation path
that compares the cached summary with a direct local NFA exploration.

Targets:

- `analyze_target()`
- `break_closure()`
- `break_consuming_tails()`
- break-seed analysis

Relevant code:

- `src/dfa/tier3.rs:2139`
- `src/dfa/tier3.rs:2355`
- `src/dfa/tier3.rs:2458`

### 2. Exact small-step reference interpreter

Build a slow, test-only interpreter for the Tier 3 fragment that tracks:

- consuming origin
- per-counter values
- pending assertion obligations
- pending tails and seeds

Compare Tier 3 against that interpreter step by step, not just on final
match/no-match.

### 3. Exhaustive small-regex model checking

For small Tier 3-eligible regexes:

- enumerate patterns systematically
- enumerate short inputs over a reduced alphabet
- enumerate all chunkings of those inputs
- compare every step and final result against Tier 0

This is especially important because streaming chunk boundaries are a core
engine invariant.

### 4. Backend equivalence tests

Where overlap exists, compare:

- Tier 3 ranged vs Tier 3 per-instance
- Tier 3 vs Tier 2 on Tier 2 overlap
- Tier 3 vs Tier 4 on Tier 4 overlap
- optimized vs unoptimized lowering shapes, especially `ByteTable`

### 5. Stronger debug invariants

Add more internal assertions and better debug visibility for:

- consuming-only origins for seeds and tails
- provenance of break-gated material
- per-step sources of `is_match` and `is_match_at_end`
- pending obligation lifetimes

The post-mortems repeatedly show that the problem was not just a missing
test.  It was also that the hidden invariant was hard to observe.

## Decision Rule

Use the following rule when deciding how far to go:

- If the priority is immediate robustness, choose Proposal 1.
- If the priority is keeping a dedicated Tier 3 with manageable risk,
  choose Proposal 2.
- If the priority is explicitly modeling break provenance in Tier 3,
  choose Proposal 3.
- If the priority is one shared semantic foundation for counting DFAs,
  choose Proposal 4.

## Related Documents

- `docs/tier3-redesign-proposal-1-provable-core.md`
- `docs/tier3-redesign-proposal-2-typed-effects.md`
- `docs/tier3-redesign-proposal-3-provenance-aware-state.md`
- `docs/tier3-redesign-proposal-4-tier4-derived.md`
- `docs/tier3-redesign-proposal-2-implementation-plan.md`
- `docs/tier4-performance-analysis.md`
