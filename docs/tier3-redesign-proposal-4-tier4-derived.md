# Tier 3 Proposal 4: Rebuild Tier 3 as a Specialization of Tier 4 Semantics

## Goal

Replace the current hand-maintained Tier 3 semantics with a backend derived
from a single exact counting-DFA semantics, using Tier 4's explicit counter
program model as the starting point.

This is the most ambitious proposal and the one with the strongest
long-term soundness story.

## Thesis

The deepest problem in the current architecture is not just that Tier 3 is
complicated.  It is that the engine effectively maintains several related
but different semantic implementations for counting patterns:

- Tier 2 differential counters
- Tier 3 conditional transitions
- Tier 4 counter programs

Tier 4 is already the most explicit of these.  It does not summarize a
transition as a few booleans.  It compiles a small program that explicitly
says what happens to counters and what gets emitted.

Relevant code:

- `src/dfa/tier4.rs:23`
- `src/dfa/tier4.rs:117`

Proposal 4 takes that idea seriously: one semantic IR, multiple execution
backends.

## Why Tier 4 Is the Right Starting Point

Tier 4 already represents transition meaning in a structured form:

- `Init`
- `Increment`
- `Remove`
- `EmitContinue`
- `EmitMatch`
- `EmitMatchAtEnd`

That is much closer to a closed semantic object than Tier 3's mix of:

- `no_break` / `with_break`
- seed lists
- break-seed lists
- tail lists
- several pending vectors

Tier 4 is therefore a better place to express "what a counting transition
means", even if a specialized backend is later needed for speed.

## Proposed Architecture

### Layer 1: Shared effect-program IR

Define one shared IR for counting transitions.  The IR must be expressive
enough for:

- counter initialization, increment, and removal
- direct match and match-at-end
- deferred assertions and boundary-sensitive evaluation
- emitting surviving consuming origins
- emitting seeds or tails as ordinary program effects

The current Tier 4 ops are close, but likely need extension for assertion
timing.

An illustrative direction:

```rust
enum EffectOp {
    Init { counter: CounterIdx, then: Vec<EffectOp> },
    Increment {
        counter: CounterIdx,
        min: usize,
        max: usize,
        on_continue: Vec<EffectOp>,
        on_break: Vec<EffectOp>,
    },
    Remove { counter: CounterIdx, then: Vec<EffectOp> },
    EvalAssert {
        assert_idx: StateIdx,
        on_pass: Vec<EffectOp>,
        on_defer: Vec<EffectOp>,
    },
    EmitOrigin { origin: StateIdx },
    EmitMatch,
    EmitMatchAtEnd,
}
```

The exact shape may differ, but the point is that deferred-assertion and
boundary behavior become part of the shared IR, not a Tier 3-only side
channel.

### Layer 2: Multiple execution backends over the same IR

Once the IR is shared, different tiers can become backends:

- generic explicit-context executor (current Tier 4 style)
- non-nested per-instance executor
- non-nested range-compressed executor
- fixed-length differential executor, if profitable

The tier decision then becomes "which executor backend is legal and fast"
rather than "which separately maintained semantics should we trust".

## What This Means for Tier 3

In this design, Tier 3 is no longer a separate semantics.

It becomes a specialization of the shared counting IR for the non-nested
fragment:

- when a counter fragment is non-nested and exact per-instance execution is
  cheaper than full Tier 4 contexts, use a specialized executor
- when range compression is provably sound, use a range-compressed backend
- otherwise use the generic Tier 4 executor

This makes Tier 3 a performance optimization layer rather than a parallel
semantic implementation.

## Benefits

### 1. One semantic source of truth

The largest benefit is architectural.  Bugs fixed in the shared IR or its
reference executor benefit all specialized backends.

### 2. Better backend equivalence testing

With a shared IR, equivalence becomes easy to phrase:

- compile one program
- run it through two executors
- compare outputs step by step

That is much stronger than today's tier-to-tier differential testing over
independently implemented semantics.

### 3. Simpler reasoning about correctness

Reasoning shifts from "did Tier 3's summary forget some condition?" to
"does this backend faithfully execute the shared IR?".

That is a much more modular question.

## Missing Pieces Relative to Current Tier 4

Proposal 4 is not free.  The current Tier 4 implementation does not yet
cover everything Tier 3 struggles with, especially deferred assertions and
boundary-sensitive semantics.

The major missing pieces are likely:

- deferred assertion ops in the shared IR
- explicit next-byte and EOI boundary semantics
- support for byte-specific target effects where needed
- a reusable way to encode post-break consuming behavior as ordinary
  program structure instead of matcher-local side state

## Migration Plan

### Phase 1: Define the shared semantic contract

Before changing executors, write down the contract of one counting
transition in shared terms:

- counter updates
- emitted origins
- deferred obligations
- direct match vs match-at-end

### Phase 2: Refactor Tier 4's program compiler toward shared use

Extract the parts of Tier 4 compilation that are generally useful for
counting semantics into shared modules.

### Phase 3: Add deferred-assertion and boundary operations

This is the key semantic expansion required before Tier 3 can be derived
from the same IR.

### Phase 4: Build a non-nested exact executor

Implement a specialized executor for the non-nested fragment that still
faithfully executes the shared IR.

The first backend can be exact per-instance execution, because that is the
easiest to validate against the generic Tier 4 executor.

### Phase 5: Add range compression as an optimization

Only after the exact non-nested executor is in place should range
compression be reintroduced as a backend optimization with its own proof and
equivalence tests.

### Phase 6: Retire the current Tier 3 implementation

Once the derived backend is trusted, remove the old `src/dfa/tier3.rs`
semantic machinery rather than keeping both systems around.

## Expected Costs

### Engineering cost

This is the largest refactor among the proposals.  It touches:

- tier selection
- transition compilation
- deferred assertion handling
- test structure and backend comparison

### Short-term performance risk

The first shared-IR backend may be slower than the hand-tuned current Tier
3 on some patterns.  That is acceptable if the architecture becomes sound
and can then be optimized in a controlled way.

### Longer rollout time

This is not the right choice if the team needs an immediate reduction in
Tier 3 fuzz failures next week.

## Validation Plan

Proposal 4 has the strongest validation story:

1. define a reference generic executor for the shared IR
2. validate all optimized backends against that executor step by step
3. validate the reference executor against Tier 0 on exhaustive small cases
4. fuzz the IR compiler and the backend equivalence relation, not just the
   final match result

This gives the project a layered correctness argument instead of a single
black-box differential test.

## Why This Proposal May Be the Best Long-Term Investment

The current Tier 3 bug stream suggests that each incremental fix is really
teaching the same lesson: the algorithm wants a more explicit semantics than
the current summary form can provide.

Tier 4 already points in that direction.  Proposal 4 finishes the move.

Instead of maintaining:

- one semantics for Tier 2
- another for Tier 3
- another for Tier 4

the engine can maintain:

- one semantics
- several execution strategies

That is the cleanest path away from pattern-specific Tier 3 patching.

## When To Choose This Proposal

Choose Proposal 4 if:

- long-term soundness matters more than short-term implementation size
- the project is willing to refactor toward shared counting semantics
- the current Tier 3 bug rate is evidence that parallel semantic stacks are
  too expensive to maintain

Do not choose it if the project needs an immediate stabilizing measure.

## Related Documents

- `docs/tier3-redesign-overview.md`
- `docs/tier3-redesign-proposal-1-provable-core.md`
- `docs/tier3-redesign-proposal-2-typed-effects.md`
- `docs/tier3-redesign-proposal-3-provenance-aware-state.md`
- `docs/tier4-performance-analysis.md`
