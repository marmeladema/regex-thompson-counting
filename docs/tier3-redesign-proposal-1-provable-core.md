# Tier 3 Proposal 1: Shrink Tier 3 to a Provable Core

## Goal

Reduce Tier 3 bug churn immediately by narrowing the accepted pattern
fragment to cases where the current two-successor model is close to an
exact semantics instead of a lossy summary that requires repair logic.

This proposal is intentionally conservative.  It prefers correctness and
maintainability over keeping the current breadth of Tier 3 acceleration.

## Thesis

The fastest path to a sound implementation is not to patch the current Tier
3 on every new fuzz artifact.  It is to stop using Tier 3 for patterns that
require the algorithm to recover hidden provenance, timing, or guard-path
information after the fact.

In practice, this means reducing Tier 3 to a subset where all three of the
following are true:

1. break effects are local and do not create downstream semantic debt
2. deferred assertions do not escape into a separate runtime channel
3. the DFA state does not need to remember why a state is present

## Why This Proposal Exists

The bug history strongly suggests that several current Tier 3 features are
not just "under-tested".  They are exactly the places where the runtime is
forced to reconstruct information that the representation has already lost.

The main sources of that loss are:

- multi-counter contamination
- break-path deferred assertions
- break-path consuming tails
- break-path downstream counter seeding
- byte-specific control flow hidden behind static booleans

Representative bug documents:

- `docs/bugs/019-tier3-counter-free-mae-contamination.md`
- `docs/bugs/022-tier3-clean-nb-chain-propagation.md`
- `docs/bugs/026-tier3-counter-break-deferred-assert-mid-input.md`
- `docs/bugs/042-tier3-pending-break-tails.md`
- `docs/bugs/046-tier3-per-tail-deferred-asserts.md`
- `docs/bugs/050-tier3-resolved-seed-nonconsume-remap.md`

## Proposed Tier 3 Core Fragment

The core fragment should satisfy all of the following conditions.

### 1. Exactly one counter

Multi-counter patterns are where provenance contamination becomes hard to
avoid.  Once the `with_break` closure can inject material from one counter
into the region of another counter, the matcher needs more than a binary
"clean vs contaminated" distinction.

Requiring exactly one counter removes the entire downstream-break-chain
class of bugs.

### 2. No deferred assertions on any break path

For every `Tier3OriginKind::Increment`, require:

- `break_deferred_asserts.is_empty()`

If a counter break needs a future boundary decision, the current Tier 3
needs `verified_deferred_asserts`, pending queues, and special EOI logic.
That is outside the provable core.

### 3. No consuming states on any break path

For every `Tier3OriginKind::Increment`, require:

- `break_consuming_states.is_empty()`

This removes post-break tail tracking entirely.  If a break path crosses a
consuming state before match or end-of-input, the current representation is
already paying an extra semantic debt that lives outside the DFA state.

### 4. No downstream counter seeding on break paths

Require:

- `analysis.break_seeds.is_empty()`

In a single-counter fragment this should hold automatically if no break
path re-enters the counter and no other counter exists.  The point of this
condition is to make the semantic contract explicit.

### 5. No deferred assertions inside counter bodies

This is already excluded by Tier 3 eligibility in `src/lib.rs:2256` and
should remain excluded in the core fragment.

### 6. No representation-dependent origin tricks

All instance origins must remain consuming states.  This should be guarded
with explicit assertions in debug builds instead of relying on convention.

Bug 50 is the key reminder here:

- `docs/bugs/050-tier3-resolved-seed-nonconsume-remap.md`

## What Tier 3 Looks Like After This Reduction

Once the above restrictions are enforced, the runtime can be dramatically
simplified.

The following state should disappear from `Tier3DfaMatcher`:

- `post_break_tails`
- `next_post_break_tails`
- `verified_deferred_asserts`
- `pending_break_seeds`
- `pending_break_tails`
- `pending_resolved_tails`
- `pending_resolved_mae`
- `current_has_break_extras`
- `clean_nb`
- `clean_nb_cf_mae`
- `clean_nb_is_match`
- `clean_nb_trans_slot`

The resulting execution model becomes much closer to the original Tier 3
story:

1. compute `no_break` and `with_break`
2. inspect the single live counter store
3. choose the correct successor
4. apply local counter updates
5. report direct match or match-at-end from local transition facts

That machine is not only simpler.  It is also much easier to reason about.

## Compile-Time Changes

Introduce a separate eligibility report for the reduced fragment, for
example `Tier3CoreEligibilityReport`, produced during
`RegexBuilder::build()`.

The report should explicitly record why a pattern is rejected, such as:

- more than one counter
- break path has deferred assertions
- break path has consuming tails
- break path reaches another counter entry
- deferred assertion appears in body

This is useful for three reasons:

1. it makes the selection rule explainable
2. it avoids silent, accidental broadening of Tier 3 later
3. it gives the CLI and tests better observability into why a pattern fell
   back to Tier 4 or Tier 0

Relevant code anchors:

- `src/lib.rs:2256`
- `src/lib.rs:3257`
- `src/dfa/tier3.rs:168`

## Runtime Changes

The runtime should become stricter as well as smaller.

Recommended debug assertions:

- every seed origin is consuming
- every range/per-instance origin is consuming
- `analysis.break_seeds` is empty in Tier 3 core mode
- `break_deferred_asserts` is empty on every increment action
- `break_consuming_states` is empty on every increment action

These assertions should fail fast if a compile-time check regresses.

## What Reduced-Scope Tier 3 Still Buys Over Tier 2

The obvious question is: if Proposal 1 narrows Tier 3 this much, why keep
it at all instead of using Tier 2 for the easy cases and Tier 4 or Tier 0
for everything else?

The answer is that even a heavily reduced Tier 3 still covers an important
fragment that Tier 2 cannot express cleanly.

### Tier 2's boundary

Tier 2 is fundamentally a fixed-length-body technique.  Its differential
counter model depends on every counter body consuming a known constant byte
length, and on the phase-clock structure that follows from that.

Relevant code and comments:

- `src/dfa/tier2.rs:1`
- `src/lib.rs:2493`

That makes Tier 2 excellent for patterns like:

- `(?:ab){3,7}`
- `(?:[0-9]{2}){4,8}`
- multiple fixed-length counters whose byte sets remain disjoint

But Tier 2 stops being applicable as soon as a counter body has genuine
variable length, even when the overall repetition is otherwise simple and
well-behaved.

### The gap that remains after Proposal 1

Reduced-scope Tier 3 would still cover patterns with:

- exactly one non-nested counter
- a variable-length body
- no deferred assertions inside the body
- a break path that exits locally to `Match` or `$ -> Match`
- no post-break consuming tails or downstream counter seeding

That is a real fragment, not an academic corner.

Examples of the kind of shape this can still accelerate:

- `(?:ab?){2,5}$`
- `(?:foo|bar?){3,6}`
- `(?:[A-Z][a-z]?){1,8}$`

These are all single-counter, variable-length-body repetitions whose exit
behavior is local.  Tier 2 rejects them because the body length is not
constant.  Proposal 1 would still give them a dedicated accelerator instead
of forcing them all the way down to Tier 4 or Tier 0.

### Concrete gains versus Tier 2

#### 1. A dedicated tier for variable-length bodies

The most direct gain is expressiveness.  Proposal 1 preserves a fast path
for the simplest variable-length-body repetitions, which is exactly the
space between Tier 2 and Tier 4.

Without this reduced Tier 3, the engine would have:

- Tier 2 for fixed-length bodies
- Tier 4 or Tier 0 for everything variable-length

That leaves a useful middle fragment unoptimized.

#### 2. A cheaper runtime than generic Tier 4 for the simple fragment

Tier 4's strength is generality, but it pays for that by carrying explicit
counter contexts and executing counter programs.  For a single non-nested
counter with local break semantics, that machinery is often more general
than necessary.

A reduced Tier 3 can still exploit the fact that there is:

- one counter
- no downstream break chain
- no break-path tail queue
- no break-path deferred-assert queue

That should allow a lighter matcher than the full Tier 4 executor while
still being more expressive than Tier 2.

#### 3. A cleaner place for range compression or single-counter specialization

Tier 2's O(1) differential counters rely on fixed phases.  They do not
extend naturally to variable-length bodies.

Reduced Tier 3 still gives the engine a place to exploit specialized
single-counter structure, such as:

- per-origin range compression
- lighter per-instance bookkeeping
- smaller transition metadata than a generic counter-program executor

Even if that path is not as cheap as Tier 2, it can still be materially
cheaper than a fully general engine.

#### 4. A clearer architectural separation of responsibilities

With Proposal 1, the tiers become easier to explain:

- Tier 2: fixed-length bodies
- Tier 3 core: simple variable-length single-counter bodies
- Tier 4: everything more general

That is a much better architectural story than the current Tier 3, which
tries to be "the general non-nested tier" and then keeps inheriting bugs
from cases that are only barely representable.

### What Proposal 1 does not gain over Tier 2

Proposal 1 is not trying to outperform or replace Tier 2 on Tier 2's home
territory.

It does not aim to beat Tier 2 for:

- fixed-length-body counters
- multi-counter patterns that satisfy Tier 2's phase assumptions
- the strongest O(1) counter-update cases

Tier 2 should remain the preferred engine there.

The point of Proposal 1 is narrower: keep a small, defensible Tier 3 so the
engine still has a specialized answer for variable-length bodies without
dragging along all of current Tier 3's complexity.

### Why this matters strategically

If Proposal 1 succeeds, the project gets an attractive intermediate state:

- Tier 2 remains the high-confidence fixed-length accelerator
- Tier 3 remains useful because it still covers simple variable-length
  repetitions
- Tier 4 becomes the fallback for the genuinely hard cases

That means Proposal 1 is not merely a retreat.  It is a redefinition of
Tier 3 around the part of the problem that Tier 2 fundamentally cannot do.

## Routing and Performance Expectations

Patterns rejected from the Tier 3 core should route to the next sound tier:

- Tier 4 when its eligibility and semantics cover the pattern
- Tier 0 otherwise

This proposal will reduce Tier 3 coverage, but it should also reduce the
time spent carrying correctness fixes across several subtle runtime paths.

The expected performance outcome is:

- simple non-nested single-counter patterns remain accelerated
- complex assertion-heavy or multi-counter patterns fall back to a slower
  but more trustworthy engine

That is a good trade when correctness is the bottleneck.

## Rollout Plan

### Phase 1: Narrow eligibility only

Keep the existing Tier 3 implementation, but restrict which patterns can
reach it.

This produces an immediate stability gain without a large refactor.

### Phase 2: Delete now-dead runtime machinery

Once the reduced fragment is in place and tested, remove the logic that can
no longer be triggered in Tier 3 core mode.

This is important.  If the dead machinery stays around, the code still has
to be understood and maintained.

### Phase 3: Re-admit features only with proof

Do not broaden the fragment because a feature "seems to work" on fuzzing.
Re-admit features only when there is either:

- a clear proof that the two-successor model remains exact, or
- a more expressive runtime representation

## Why This Proposal Is Likely Sound

The reduced fragment removes the places where Tier 3 currently leaks hidden
state into side channels.

Specifically:

- single-counter execution removes multi-counter contamination
- no break-path deferred assertions removes next-byte and EOI obligation
  queues
- no break-path consuming tails removes post-break tail tracking
- no downstream break seeding removes break-triggered seed provenance

What remains is a local, one-counter transfer function with a much smaller
semantic surface.

## Main Risks

### Risk 1: Coverage loss is larger than expected

This proposal may route many previously Tier 3-eligible patterns to Tier 4
or Tier 0.  That could create throughput regressions on some workloads.

### Risk 2: The chosen core is still too broad

Even the reduced fragment must be validated carefully.  "Single counter"
does not automatically imply soundness if some other representation leak is
still present.

### Risk 3: The codebase keeps two Tier 3s mentally

If the old broader semantics remain in comments or dead code, it becomes
easy for future changes to accidentally reintroduce unsupported cases.

## Validation Plan

Proposal 1 should be validated with:

1. exhaustive small-regex comparison against Tier 0 over all chunkings
2. transition-level debug assertions for the reduced invariants
3. explicit tests that rejected patterns do in fact fall back out of Tier 3
4. coverage measurement to quantify how much real-world Tier 3 usage remains

## When To Choose This Proposal

Choose Proposal 1 if the immediate priority is:

- stop Tier 3 fuzz regressions now
- simplify maintenance burden
- create time for a deeper redesign later

Do not choose it if the primary objective is to keep the current breadth of
Tier 3 acceleration at all costs.

## Related Documents

- `docs/tier3-redesign-overview.md`
- `docs/tier3-redesign-proposal-2-typed-effects.md`
- `docs/tier3-redesign-proposal-3-provenance-aware-state.md`
- `docs/tier3-redesign-proposal-4-tier4-derived.md`
