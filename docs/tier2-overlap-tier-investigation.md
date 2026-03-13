# Investigation: Is There Room for a Tier Between Tier 2 and Tier 3?

## Question

Tier 2 currently handles non-nested counters with fixed-length bodies, but it
requires different counter bodies to have pairwise-disjoint byte sets.  Tier 3
handles the general non-nested case, including variable-length bodies, but it
is substantially more complex.

The question is whether there is a worthwhile middle tier for:

- non-nested counters
- fixed-length bodies
- overlapping body byte sets

The standard for "worthwhile" is strict:

- if the algorithm ends up with complexity close to Tier 3, it is probably
  not worth a separate tier
- if fixed length gives enough structure for a materially simpler and faster
  design, it may be worth pursuing

## Short Answer

There is probably **some** room between Tier 2 and Tier 3, but the most
promising step is **not** a brand new general-purpose tier.

The best opportunities are:

1. **Refine Tier 2 eligibility** so some overlapping-body patterns can still
   use the existing Tier 2 runtime when no reachable transition actually lets
   multiple counters co-fire.
2. If profiling shows a real gap, build a **small exact-break-mask Tier 2.5**
   for fixed-length overlaps under strong restrictions.

What is **not** promising is a fully general "fixed-length but overlapping"
tier that also handles arbitrary break-path seeds, consuming tails, and
deferred assertions.  That starts to reconstruct Tier 3's complexity on top
of Tier 2's counters.

## Why Tier 2 Requires Disjoint Byte Sets Today

Tier 2's runtime model is fundamentally binary per counting transition.

Its cached transition stores:

- `no_break`
- `with_break`
- one `counting_mask`
- one `counter_reset`
- one pre-seed list
- one seed list

See `src/dfa/tier2.rs:256`.

At runtime, Tier 2:

1. increments every counter in `counting_mask`
2. asks whether **any** of them can break
3. chooses exactly one successor: `with_break` if any broke, otherwise
   `no_break`

See `src/dfa/tier2.rs:1479`.

This means Tier 2 can distinguish only:

- no counter breaks
- at least one counter breaks

It cannot distinguish:

- counter A breaks, counter B does not
- counters A and B both reach CInc, but only one has `value + 1 >= min`
- A breaks and contributes downstream seeds, B only continues

That is why `src/lib.rs:2493` insists that counter body byte sets be
pairwise disjoint.  The comment gives the exact reason: the disjointness rule
guarantees that at most one counter fires CInc on any DFA transition, so the
binary `with_break` / `no_break` split remains correct.

## Why Fixed Length Still Helps

Even when byte sets overlap, fixed length still provides strong structure.

Tier 2's differential counters rely on a phase clock:

- each counter body has length `L`
- each active instance belongs to one of `L` phases
- each byte advances the phase clock uniformly
- all instances in the same phase increment together

See `src/dfa/tier2.rs:113` and `src/dfa/tier2.rs:617`.

That is a real advantage over Tier 3.  Fixed length means counter **storage**
can remain cheap and regular even when the control-flow semantics become more
complicated.

So the opportunity for a new tier is real, but it is only on the control-flow
side.  The counter-update side is already in good shape.

## What Overlap Actually Breaks

Once byte sets overlap, several new complications appear.

### 1. Break choice becomes a subset, not a boolean

If `k` counters can reach CInc on the same byte, the true control decision is
which subset of those `k` counters actually breaks.  That is a `2^k` problem,
not a 2-state problem.

### 2. Seeds become subset-dependent

If counter A breaks into counter B's entry, B must only be seeded when A
actually breaks.  A unioned `with_break` successor over-approximates this.

This is the same family of issue that forced `break_seeds` and contamination
repairs in Tier 3.  See `src/dfa/tier3.rs:85` and
`docs/bugs/032-tier3-contaminated-counter-free-seeds.md`.

### 3. Match and match-at-end become subset-dependent

If the transition says "some counter broke", but the match or `$ -> Match`
path actually depends on a specific counter breaking, then a unioned
`with_break` successor can report a false positive.

Again, this is a Tier 3 contamination-style problem.

### 4. Deferred assertions and consuming break tails create timing debt

Once a break path contains:

- deferred assertions
- consuming states that need to be processed later

the runtime needs queues or structured pending effects.  That is exactly the
kind of complexity that pushed Tier 3 away from a small binary model.

Bug 18 in Tier 2 is the simplest example of this class even before overlap:
`docs/bugs/018-tier2-break-path-consuming-states.md`.

## The Most Promising Design Space

The design space is easiest to understand if it is split into cases.

## Case A: Overlap Exists Statistically, But Reachable Co-Fire Never Happens

This is the most attractive case.

Example shape:

- the body byte sets overlap as sets
- but along all reachable transitions, at most one counter can actually reach
  CInc on a given byte

In that case, Tier 2's runtime is already sufficient.  The current
disjoint-byte rule is simply too conservative.

### Why this matters

The test `^\w{3}\d{2}$` is already documented as a pattern that must not use
Tier 2 under the current rule because `\w` and `\d` overlap:

- `src/lib.rs:9402`

But this is exactly the sort of pattern that may still be safe for the Tier 2
runtime if a stronger compile-time proof shows that no reachable transition
has multiple counters co-firing.

### Recommendation for this case

Do **not** create a new runtime tier first.  Instead, refine Tier 2
eligibility.

Possible proof targets:

- no reachable transition has `popcount(counting_mask) > 1`
- no reachable DFA state can have multiple counters whose active phases align
  to fire on the same byte
- no reachable closure merges multiple break-relevant counters on one step

This is the highest-ROI path because it preserves Tier 2's existing runtime
cost model.

## Case B: Reachable Co-Fire Exists, But Break Paths Are Pure and Local

This is the best actual candidate for a new middle tier.

The idea is:

- keep Tier 2's differential counters and phase clock
- but store exact successors for the **actual break subset**, not just
  `no_break` and `with_break`

### Restrictions needed to keep it simple

This only stays attractive if break paths are strongly restricted.

Suggested restrictions:

- fixed-length bodies only
- non-nested counters only
- no deferred assertions inside bodies beyond what Tier 2 already handles
- no consuming states on break paths
- no post-break tails
- no arbitrary chained break-path assertions

If break paths are pure epsilon structure leading to:

- other counter entries
- direct `Match`
- `$ -> Match`

then the exact break subset can likely be handled with subset-indexed seed and
match metadata while still using the cheap phase-based counter storage.

### Runtime shape

For a local overlap set of size `k`:

- increment all counters that fire CInc on this transition
- compute the actual `break_mask`
- choose the exact successor for that break mask
- apply subset-specific seeds and match flags

### Complexity

The obvious cost is that a transition no longer has 2 control variants.  It
has up to `2^k` variants.

That means this only looks attractive when `k` is small, for example:

- `k <= 2`
- maybe `k <= 3`

Above that, compile-time and cache size will probably stop being attractive.

### Recommendation for this case

This is the strongest candidate for a true "Tier 2.5".

But it should be deliberately narrow and should fall back immediately when the
local overlap set becomes too large.

## Case C: Reachable Co-Fire Exists and Break Paths Have Seeds, Tails, or Deferred Logic

This is where a new tier starts to lose its value.

Once subset-specific break behavior includes:

- break-gated downstream seeds
- consuming break tails
- next-byte deferred resolution
- end-of-input-only break effects

the tier is no longer just "Tier 2 with a slightly richer break decision".
It starts to need the same kind of semantic machinery that made Tier 3 hard.

At that point, fixed length helps counter storage but does **not** buy a much
simpler control model.

This is the point where a new tier is probably not worth it.

## Candidate Designs

## Candidate 1: Refined Tier 2 Eligibility

### Idea

Keep the existing Tier 2 runtime, but replace the current coarse
`disjoint_bytes` rule with a more precise proof that multiple counters cannot
co-fire on any reachable transition.

### Benefits

- no hot-path runtime changes
- same performance model as Tier 2
- likely the best engineering ROI

### Costs

- compile-time analysis gets more complicated
- the proof needs to be conservative and obviously correct

### Recommendation

This is the first thing to try.

## Candidate 2: Exact-Break-Mask Tier 2.5

### Idea

Keep Tier 2 counter storage, but replace binary `with_break` / `no_break`
with exact subset-based control for a small local overlap set.

### Benefits

- still uses fixed-length differential counters
- should be materially cheaper than Tier 3 per-instance for anchored,
  non-rangeable fixed-length overlap patterns
- avoids contamination from false break unions

### Costs

- transition count grows roughly as `2^k` per overlapping counting transition
- needs a hard local-overlap cap
- still needs careful subset-specific match and seed handling

### Recommendation

Worth pursuing only after profiling, and only with strong restrictions.

## Candidate 3: Generic Overlapping Fixed-Length Tier

### Idea

Support fixed-length overlapping counters even when break paths have:

- seeds into other counters
- consuming tails
- deferred assertions
- subset-specific match-at-end behavior

### Benefits

- more coverage than Candidate 2

### Costs

- control-flow semantics become very close to Tier 3
- bug surface will likely look like Tier 3's contamination and timing bugs
- fixed length only helps the storage side, not enough to justify a whole
  additional tier

### Recommendation

Probably not worth pursuing.

## Where the Performance Opportunity Actually Is

The main niche for an intermediate tier is not all fixed-length overlap
patterns equally.

It is specifically patterns where:

- bodies are fixed length, so Tier 2-style phase counting is attractive
- overlapping byte sets exclude Tier 2 today
- the pattern is anchored or assertion-gated so Tier 3 cannot use its ranged
  fast path and falls to the per-instance backend

This is the most likely place where a small Tier 2.5 could produce a real win.

Relevant Tier 3 rangeability comments:

- `src/dfa/tier3.rs:145`
- `src/dfa/tier3.rs:933`

## Complexity and Performance Expectations

| Design | Runtime | Compile/cache cost | Likely position |
|--------|---------|--------------------|-----------------|
| Tier 2 today | best | lowest | baseline fastest |
| Candidate 1 | same as Tier 2 | higher compile-time proof only | best ROI |
| Candidate 2 | near Tier 2 plus subset logic | `O(2^k)` local expansion | realistic middle tier |
| Candidate 3 | closer to Tier 3 than Tier 2 | high | probably not worth it |
| Tier 3 ranged | `O(num_origins)` per byte | moderate | already good fast path |
| Tier 3 per-instance | `O(live_instances)` per byte | moderate | main performance gap |

The key point is that only Candidate 2 feels like a genuine new tier.  The
others are either:

- just better Tier 2 eligibility, or
- essentially Tier 3 complexity in disguise

## Recommendation

## Recommendation 1: Start with eligibility refinement, not a new tier

The first experiment should be a compile-time analysis that admits some
overlapping fixed-length patterns into Tier 2 when reachable co-fire is
provably impossible.

This is likely the best value for effort.

## Recommendation 2: If needed, prototype a narrow Tier 2.5

If benchmarks still show a real gap, prototype a very narrow exact-break-mask
fixed-length tier with all of the following restrictions:

- fixed-length bodies
- non-nested counters
- no consuming break tails
- no generalized deferred break machinery
- small local overlap cap

The prototype should explicitly target anchored fixed-length overlap cases
that currently fall to Tier 3 per-instance.

## Recommendation 3: Do not pursue a fully general overlap tier

Once the design needs:

- subset-specific seeds
- tails
- deferred-resolution queues
- contamination repairs

it is too close to Tier 3 to justify another execution tier.

## Concrete Next Steps

1. Add a compile-time investigation path for fixed-length overlapping patterns
   that measures the maximum reachable co-fire width instead of just checking
   body byte-set disjointness.
2. Gather examples where the current rule rejects patterns that in practice
   never co-fire.
3. Benchmark anchored fixed-length overlap patterns that currently fall to
   Tier 3 per-instance.
4. Only if those benchmarks are compelling, prototype Candidate 2 with a hard
   overlap cap and pure break-path restrictions.

## Final Judgment

Yes, there is probably useful room between Tier 2 and Tier 3, but mostly in
the form of:

- a **smarter Tier 2 boundary**, and maybe
- a **small, exact-break-mask Tier 2.5**

There is probably **not** enough room for a fully general new tier unless the
engine is willing to reintroduce many of Tier 3's current semantic problems.
