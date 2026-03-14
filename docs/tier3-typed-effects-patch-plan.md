# Tier 3 Typed Effects Patch Plan

## Purpose

This document turns the 10 findings from
`docs/tier3-typed-effects-implementation-review.md` into an ordered sequence
of concrete patches.  Each patch is self-contained: it describes the problem,
lists the exact files and line ranges to change, specifies what to test, and
defines done criteria.

The patches are ordered by dependency and risk.  Later patches may reference
earlier ones, but earlier patches never depend on later ones.

## Relationship to Other Documents

- `docs/tier3-typed-effects-implementation-review.md` — the findings this
  plan is derived from
- `docs/tier3-redesign-proposal-2-implementation-plan.md` — the original
  phased migration plan; this patch plan picks up where the migration left off
- `docs/tier3-redesign-proposal-2-typed-effects.md` — the design proposal
  that motivates the typed effects architecture

## Overall Strategy

The review identified four priority tiers:

1. **Fix broken contracts** — things that are documented as working but are
   not, or structural issues that can produce live bugs
2. **Finish semantic normalization** — eliminate remaining OR/AND ambiguities
   and make the effect model authoritative
3. **Make compiled effects the runtime source of truth** — stop hand-building
   `PendingEffect` entries from `Tier3OriginKind` fields
4. **Optimize** — remove hot-path allocations, intern effect templates,
   shrink transition payload

The patches below follow this order.  Within each priority tier, patches are
ordered to minimize risk: smallest scope first, each patch independently
testable.

### Exact handoff point

This document uses "authoritative at runtime" in a specific, narrow sense:

- `CompiledTargetEffects` becomes the single runtime source of truth for
  **nonlocal Tier 3 semantics**
- local counter stepping still stays in the local step data
  (`TargetStep` / the surviving local fields on `Tier3OriginKind`)

In concrete terms, the handoff happens in **Patch 8A**, not earlier.

- **Patches 1-6** fix contracts, normalize semantics, and strengthen tests.
  They do **not** make compiled effects authoritative.
- **Patch 7** creates a single runtime funnel by replacing the many open-coded
  enqueue sites with helper functions.  After Patch 7, there is one execution
  path, but that path may still read legacy `Tier3OriginKind` fields.  Patch 7
  is therefore **not** the authority flip.
- **Patch 8A** is the authority flip.  At that point, transitions carry
  effect-template IDs, and the helpers from Patch 7 must obtain deferred
  matches, break seeds, deferred tails, and other guarded nonlocal behavior
  from `CompiledTargetEffects` / template tables rather than reconstructing it
  from legacy Tier 3 side fields.
- **Patch 8B** and **Patch 9** make the handoff exclusive by deleting the old
  nonlocal truth and the remaining end-of-input fallback logic.
- **Patch 8C** and **Patch 10** optimize the new model after the semantic
  handoff is already complete.

If a future reader wants one sentence to remember, it is this:

- `CompiledTargetEffects` becomes authoritative when **Patch 8A lands**
  and all nonlocal runtime deposition reads compiled templates instead of
  legacy analysis fields.

---

## Priority 1: Fix Broken Contracts

### Patch 1: Remove or enforce `required_breaks` in `resolve_pending()`

**Finding:** 2

**Problem:**

`EffectGuard` documents `required_breaks` as a conjunctive guard condition
alongside `assert_chain`, but `resolve_pending()` in
`src/dfa/tier3_effects.rs:596-628` completely ignores `required_breaks`.
The function checks only `eval_assert_chain()` at line 618.  This makes the
type contract false.

**Current state:**

Today this is partly masked because enqueue sites only create break-gated
effects after the break already happened.  But the field is documented as a
guard, and any future change that trusts the documentation will introduce a
bug.

**Decision:**

Remove `required_breaks` from `EffectGuard` for now.  The field is never
enforced at runtime, never populated with meaningful values during matching,
and its presence is misleading.  If provenance-aware state (Proposal 3) is
later adopted, `required_breaks` can be reintroduced with proper enforcement.

**Files to change:**

- `src/dfa/tier3_effects.rs`
  - Remove `required_breaks` from `EffectGuard` (line 168)
  - Remove `required_breaks` from `EffectGuard::is_always()` (line ~177)
  - Remove `required_breaks` from all Display impls that render it
  - Update `compile_target_effects()` to stop setting `required_breaks`
  - Update the `test_effect_guard_is_always` test (line 1021)
  - Add a comment at the `EffectGuard` definition explaining why
    `required_breaks` was removed and under what conditions it should return

- `src/dfa/tier3.rs`
  - Update all `PendingEffect` construction sites to stop setting
    `required_breaks` (approximately 10 sites: lines 2842, 2925, 2953,
    2988, 3104, 3132, 3316, 3603, 3675, 3702)

- `src/dump.rs`
  - Update dump output if it renders `required_breaks`

**Alternative (deferred):**

If we later decide to enforce `required_breaks`, the patch is:
1. Add a `satisfied_breaks: BreakMask` parameter to `resolve_pending()`
2. Skip effects whose `required_breaks & !satisfied_breaks != 0`
3. Compute `satisfied_breaks` at each call site

This is the right thing to do if Proposal 3 is adopted.  It is premature now.

**Tests:**

- `cargo test`
- `cargo clippy -- -D clippy::all`

**Done criteria:**

- `required_breaks` does not appear in `EffectGuard`
- `resolve_pending()` has no dead guard paths
- All tests pass

---

### Patch 2: Split pending-effect queues for current vs next boundary

**Finding:** 5

**Problem:**

During pending-effect resolution in `chunk()`, resolved tails and seeds can
enqueue fresh `PendingEffect` entries into `pending_effects_next_byte` (lines
3596, 3668, 3694, 3734 in `src/dfa/tier3.rs`).  Then the queue is
unconditionally cleared at line 3755.  If a resolved effect schedules a new
deferred effect that should live until the next byte boundary, that effect is
dropped.

**Fix:**

Split the queue into two logical buffers:

```rust
pending_effects_current: Vec<PendingEffect>,
pending_effects_next: Vec<PendingEffect>,
```

The resolution loop drains `pending_effects_current`.  Any new effects
deposited during resolution go into `pending_effects_next`.  After
resolution, swap `pending_effects_next` into `pending_effects_current` for
the next byte boundary.

**Files to change:**

- `src/dfa/tier3.rs`
  - Rename `pending_effects_next_byte` to `pending_effects_current`
  - Add `pending_effects_next` field to matcher struct (near line 2760)
  - Update `reset_for_new_match()` (line 3424) to clear both
  - Update all deposit sites in `step_slow_impl!` to push into
    `pending_effects_next` when depositing during resolution, or
    `pending_effects_current` when depositing during the step phase
  - Update the resolution block in `chunk()` (~lines 3585-3755):
    - resolve from `pending_effects_current`
    - any re-enqueue goes to `pending_effects_next`
    - after resolution: clear `pending_effects_current`, swap next→current
  - Update `finish()` to resolve from the correct buffer
  - Update Display/Debug impls to show both buffers

**Design detail — when to push to which buffer:**

- During the step phase (inside `step_slow_impl!`): push to
  `pending_effects_current` for resolution at the upcoming boundary
- During resolution of `pending_effects_current`: push to
  `pending_effects_next` for the *following* boundary
- This ensures resolution never drops newly scheduled effects

**Tests:**

- `cargo test`
- Construct a targeted test case (if possible) where a resolved tail deposits
  a new deferred effect; verify it survives to the next boundary

**Done criteria:**

- `pending_effects_next_byte` is gone
- Two distinct buffers exist with clear ownership rules
- All tests pass

---

### Patch 3: Remove dead `EndOnly` machinery

**Finding:** 6

**Problem:**

`EffectTiming::EndOnly` exists in the type system (line 60 of
`tier3_effects.rs`).  The matcher has a `pending_effects_end_only` field
(line 2762 of `tier3.rs`).  But no code ever pushes effects into that queue.
The queue is only initialized (line 3397), cleared (line 3424), and displayed
(lines 4094, 4218).  The effect compiler emits `EndOnly`-timed guarded
effects (line 846 of `tier3_effects.rs`), but since compiled effects are not
the runtime source of truth, those entries are never executed.

This is dead code that increases cognitive load.

**Fix:**

Remove `EndOnly` from the runtime path entirely:

- Remove `pending_effects_end_only` from the matcher struct
- Remove `EffectTiming::EndOnly` from the enum
- Remove `EndOnly` Display/Debug arms
- Remove `EndOnly`-timed entries from `compile_target_effects()` output

If `EndOnly` is needed in the future (when compiled effects become
authoritative), it can be reintroduced as part of that migration.

**Files to change:**

- `src/dfa/tier3_effects.rs`
  - Remove `EndOnly` variant from `EffectTiming` (line 60)
  - Remove `EndOnly` from Display impl
  - Remove `EndOnly`-timed guarded effects from `compile_target_effects()`
    (around lines 846-853)
  - Update any tests that reference `EndOnly`

- `src/dfa/tier3.rs`
  - Remove `pending_effects_end_only` field (line 2762)
  - Remove its initialization (line 3397)
  - Remove its clear (line 3424)
  - Remove its debug/display output (lines 4094, 4218-4220)

- `src/dump.rs`
  - Remove `EndOnly` rendering if present

**Tests:**

- `cargo test`
- `cargo clippy -- -D clippy::all`

**Done criteria:**

- No `EndOnly` variant exists
- No `pending_effects_end_only` field exists
- All tests pass

---

### Patch 4: Reconcile compiled-effect `MatchAtEnd` with runtime `Match`

**Finding:** 3

**Problem:**

The effect compiler in `compile_target_effects()` emits `MatchAtEnd` atoms
for break-deferred assertion paths (lines 842-843, 851-852 in
`tier3_effects.rs`).  But the runtime matcher hand-builds `PendingEffect`
entries with `EffectAtom::Match` for the same semantic concept (lines 2925,
3104, 3675 in `tier3.rs`).

This semantic drift will produce incorrect behavior if compiled effects ever
become the runtime source of truth without reconciliation.

**Fix:**

Align the effect compiler with the runtime's current behavior, since the
runtime is the tested and exercised path:

- Change `compile_target_effects()` to emit `EffectAtom::Match` for
  break-deferred assertion paths, not `EffectAtom::MatchAtEnd`
- Add a comment explaining that `MatchAtEnd` is reserved for contexts where
  the match is only valid at end-of-input (e.g., `$`-anchored patterns), not
  for deferred assertion paths that may resolve mid-input

**Rationale for aligning compiler→runtime (not the other way):**

The runtime has been exercised by the entire test suite, fuzz soaks, and
bug fixes.  The compiled effect model has only been exercised through dump
output.  Changing the runtime to match the compiler would risk introducing
new regressions.

**Files to change:**

- `src/dfa/tier3_effects.rs`
  - Change `MatchAtEnd` → `Match` in break-deferred guarded effect atoms
    (lines 842-843)
  - Remove the `EndOnly`-timed `MatchAtEnd` emission if not already removed
    by Patch 3
  - Add clarifying comments on `MatchAtEnd` semantics at the `EffectAtom`
    definition

**Tests:**

- `cargo test`
- `cargo run --release -- dump --dfa` on a pattern with break-deferred
  assertions (e.g., `\b\w{3,5}\b`) to verify dump output changed

**Done criteria:**

- Compiled effects and runtime use the same atom type for the same concept
- `MatchAtEnd` in compiled effects is only used where the runtime would also
  use `MatchAtEnd`
- All tests pass

---

## Priority 2: Finish Semantic Normalization

### Patch 5: Audit and document OR vs AND semantics for all deferred-assert fields

**Finding:** 4 (4a, 4b)

**Problem:**

Bug 51 proved that collapsing alternative deferred paths (OR semantics) into
a single assertion chain (AND semantics) causes false negatives.  The
immediate fix covered `break_deferred_asserts`, but the same structural risk
exists in:

- `Tier3BreakSeed.deferred_asserts` (line 263 of `tier3.rs`) — one slice per
  seed, dedup'd by `(trigger, counter, origin)` at line 478
- `break_consuming_deferred` entries (line 253 of `tier3.rs`) — one chain per
  tail, selected by shortest path at line 2566
- `target_deferred_asserts` on the advance path

**Fix:**

This patch is primarily an audit and documentation pass, not a behavior
change.  For each deferred-assert field:

1. Classify it as one of:
   - `PathChain` — one ordered path, AND semantics
   - `AlternativeChains` — OR over several paths
   - `PerTailChain` — one chain per tail
   - `PerSeedChain` — one chain per break-triggered seed

2. Add a doc comment at the field definition stating which classification
   applies and why.

3. If the audit reveals a field that currently uses AND semantics but should
   use OR (like Bug 51), file it as a separate bug and create a follow-up
   patch.

**Fields to audit:**

| Field | Location | Expected semantics |
|-------|----------|-------------------|
| `break_deferred_asserts` | `Tier3OriginKind::Increment` line 241 | `AlternativeChains` (fixed by Bug 51) |
| `break_deferred_chain_ids` | `Tier3OriginKind::Increment` line 258 | One chain per alternative |
| `break_consuming_deferred` | `Tier3OriginKind::Increment` line 253 | `PerTailChain` |
| `Tier3BreakSeed.deferred_asserts` | Line 269 | `PerSeedChain` (but dedup at line 478 may merge distinct paths) |
| `target_deferred_asserts` | Per advance-path origin analysis | `AlternativeChains` or `PathChain` depending on NFA structure |

**Files to change:**

- `src/dfa/tier3.rs`
  - Add doc comments to each field listed above
  - Add doc comments to the `break_seeds_raw` dedup block (lines 478-497)
    explaining whether dedup is semantics-preserving
  - If the audit finds that `Tier3BreakSeed` dedup merges distinct OR-paths,
    add a TODO or immediately fix it

**Tests:**

- `cargo test`

**Done criteria:**

- Every deferred-assert field has a doc comment stating its OR/AND
  classification
- The dedup logic in `break_seeds_raw` has a documented justification
- Any newly discovered OR/AND bugs are filed as TODOs or separate patches

---

### Patch 6: Add semantic tests for `tier3_effects.rs`

**Finding:** 10

**Problem:**

The test module in `tier3_effects.rs` (lines 978-1136) contains only 9 tests,
all covering Display formatting and arena dedup.  There are no tests for:

- `resolve_pending()` behavior
- `eval_assert_chain()` behavior
- `compile_target_effects()` correctness
- OR vs AND normalization
- Queue lifecycle scenarios

**Fix:**

Add table-driven semantic tests directly in `src/dfa/tier3_effects.rs`.
These tests should exercise the effect resolution logic without depending on
the full matcher.

**Tests to add:**

1. **`test_resolve_pending_unconditional_match`** — a pending `Match` with
   `EffectGuard::ALWAYS` resolves successfully regardless of boundary context

2. **`test_resolve_pending_assert_chain_passes`** — a pending `Match` guarded
   by a word-boundary assertion resolves when the boundary condition is met

3. **`test_resolve_pending_assert_chain_fails`** — same guard, boundary
   condition not met, effect is skipped

4. **`test_resolve_pending_contradictory_or_alternatives`** — two pending
   effects for the same match, one guarded by `\b` and one by `\B`; at any
   boundary, exactly one should resolve (OR semantics via multiple effects)

5. **`test_resolve_pending_conjunctive_chain`** — a single pending effect
   guarded by a chain of `\b` then `\B`; should resolve only when both
   conditions are met in order (AND semantics via single chain)

6. **`test_resolve_pending_timing_filter`** — effects with `NextByte` timing
   are skipped when the filter is a different timing

7. **`test_resolve_pending_add_tail`** — resolved `AddTail` atoms appear in
   the returned `ResolvedActions`

8. **`test_resolve_pending_add_seed`** — resolved `AddSeed` atoms appear in
   the returned `ResolvedActions`

9. **`test_resolve_pending_mixed_atoms_reachability`** — a bundle containing
   both `Match` and `AddTail` has match-reachability gating applied to the
   whole bundle (documents Finding 7's current behavior as a known limitation)

**Files to change:**

- `src/dfa/tier3_effects.rs` — add tests to the existing `#[cfg(test)] mod
  tests` block

**Test helpers needed:**

Some tests will need small helper functions to construct `Regex` values or
mock the assertion-evaluation context.  If constructing a full `Regex` is too
heavyweight for unit tests, extract the assertion-evaluation logic into a
function that takes only the needed context (prev byte, next byte, assertion
states) rather than a full `Regex` reference.  If that refactor is too large,
it can be deferred and these tests can use real compiled `Regex` values from
simple patterns.

**Tests:**

- `cargo test`

**Done criteria:**

- At least 8 new semantic tests exist in `tier3_effects.rs`
- Tests cover OR alternatives, AND chains, timing filters, and atom types
- All tests pass

---

## Priority 3: Make Compiled Effects the Runtime Source of Truth

### Patch 7: Extract effect-deposition helpers

**Finding:** 1, Simplification 1

**Problem:**

The matcher deposits `PendingEffect` entries by hand at ~10 sites across
`step_slow_impl!`, `chunk()`, and `finish()`.  Each site reconstructs the
same logic: pick the guard, pick the timing, construct the atoms, push.
Bug 51 was a four-site fix because the same semantic concept was open-coded
in four places.

**Fix:**

Extract three helper methods on the matcher (or as free functions taking
the matcher's pending queue):

```rust
fn enqueue_target_deferred_match(
    queue: &mut Vec<PendingEffect>,
    arena: &AssertChainArena,
    chain_id: AssertChainId,
    prev_was_word: bool,
)

fn enqueue_break_deferred_match(
    queue: &mut Vec<PendingEffect>,
    arena: &AssertChainArena,
    chain_ids: &[AssertChainId],
    prev_was_word: bool,
)

fn enqueue_deferred_tail(
    queue: &mut Vec<PendingEffect>,
    arena: &AssertChainArena,
    origin: StateIdx,
    chain_id: AssertChainId,
    prev_was_word: bool,
)
```

Replace every hand-coded deposit site with a call to one of these helpers.

This patch is intentionally only a funneling patch.  It creates one runtime
deposition path, but it does **not** yet make `CompiledTargetEffects`
authoritative.  The helpers introduced here may still read legacy
`Tier3OriginKind` fields.  That is acceptable for Patch 7 as long as there is
now exactly one place to swap over in Patch 8A.

**Sites to replace:**

| Line(s) | Current logic | Helper to use |
|---------|--------------|---------------|
| 2834-2850 | Target deferred asserts (range-compressed) | `enqueue_target_deferred_match` |
| 2913-2930 | Break deferred asserts (range-compressed) | `enqueue_break_deferred_match` |
| 2943-2960 | Deferred tail (range-compressed) | `enqueue_deferred_tail` |
| 2979-2995 | None-target tail deferred asserts | `enqueue_target_deferred_match` |
| 3092-3110 | Break deferred asserts (per-instance) | `enqueue_break_deferred_match` |
| 3120-3140 | Deferred tail (per-instance) | `enqueue_deferred_tail` |
| 3310-3325 | Break seed with deferred asserts | `enqueue_deferred_seed` (new) |
| 3592-3610 | Target deferred asserts (tail resolution) | `enqueue_target_deferred_match` |
| 3665-3680 | Break deferred asserts (tail resolution) | `enqueue_break_deferred_match` |
| 3692-3710 | Deferred tail (tail resolution) | `enqueue_deferred_tail` |

**Files to change:**

- `src/dfa/tier3.rs`
  - Add helper methods
  - Replace open-coded deposit sites with helper calls

**Tests:**

- `cargo test`
- Manual debug trace comparison for 2-3 patterns to verify behavior is
  identical

**Done criteria:**

- No `PendingEffect` is constructed inline outside the helpers
- All nonlocal runtime deposition flows through a single helper layer
- The helper layer may still read legacy fields; compiled effects are not yet
  authoritative at this stage
- All tests pass
- A future OR/AND bug requires fixing in one place, not four

---

### Patch 8: Add effect-template IDs to transitions

**Findings:** 1, 8

**Problem:**

Transitions cache entire cloned `Tier3OriginKind` values (line 191 of
`tier3.rs`), which carry multiple boxed slices.  The matcher then
reconstructs effect semantics from these fields at every step.  This is both
a performance issue (cloned boxed slices, hot-path heap allocations) and a
correctness risk (reconstruction logic can diverge from the compiled model).

**Fix:**

This is the most invasive patch and should be approached carefully.

Phase A — add template IDs and flip runtime authority:

1. Extend `CompiledTargetEffects` with interned IDs for each effect family
2. Store effect template IDs alongside `Tier3OriginKind` in the transition
   cache
3. The helpers from Patch 7 should consume template IDs rather than raw
   `Tier3OriginKind` fields

**This is the exact runtime handoff point.**

After Phase A lands:

- deferred matches
- break seeds
- deferred tails
- other guarded nonlocal effects

must all be deposited from compiled template data, not reconstructed from the
legacy Tier 3 side fields.  Local counter stepping still stays in the local
step representation; only the nonlocal semantics flip to compiled effects.

Put differently:

- before Patch 8A, `CompiledTargetEffects` is shadow or advisory data
- after Patch 8A, `CompiledTargetEffects` is authoritative for nonlocal
  runtime behavior

Phase B — shrink `Tier3OriginKind`:

1. Once the helpers consume template IDs, remove the legacy fields from
   `Tier3OriginKind::Increment` that are now redundant:
   - `break_deferred_asserts`
   - `break_consuming_states`
   - `break_consuming_pure`
   - `break_consuming_deferred`
   - `break_deferred_chain_ids`
2. Keep only the fields needed for local counter stepping:
    - `counter`, `advance_origins`, `min`, `max`, `continue_origins`
    - `break_is_match`, `break_is_match_at_end`

Phase B is where the old nonlocal truth is deleted.  Patch 8A flips runtime
authority; Patch 8B removes the obsolete representation that would otherwise
allow semantic drift to return.

Phase C — intern atom bundles:

1. Replace `Box<[EffectAtom]>` in `PendingEffect` with a pre-interned
   template ID or a small inline enum
2. For the common case (single `Match` atom), use a zero-allocation inline
   representation

**Files to change:**

- `src/dfa/tier3_effects.rs` — template interning, ID types
- `src/dfa/tier3.rs` — transition storage, helper consumption, field removal
- `src/dump.rs` — dump template IDs

**Tests:**

- `cargo test` after each phase
- Performance comparison before/after (optional but recommended)

**Done criteria (all phases):**

- Transitions store template IDs, not full semantic payloads
- `Tier3OriginKind::Increment` has no break-deferred fields
- Hot-path `PendingEffect` construction does not heap-allocate
- All tests pass

**Authority checkpoints:**

- **Done for Patch 8A:** the runtime helpers read compiled template IDs for all
  nonlocal effects, so `CompiledTargetEffects` is authoritative at runtime
- **Done for Patch 8B:** the legacy nonlocal fields are removed, so there is no
  second semantic source left in `Tier3OriginKind`
- **Done for Patch 8C:** the authoritative model is also the efficient one

---

### Patch 9: Route EOI seed resolution through the effect model

**Finding:** 9

**Problem:**

At end-of-input, resolved seeds are checked by scanning all target actions
for any `Increment` on the same counter whose `min` is satisfied (lines
4028-4044 of `tier3.rs`).  This ignores the seed's specific origin and
reasons only by counter ID and threshold.  The invariant that makes this
over-approximate check correct is not documented.

**Fix:**

Two options, in order of preference:

**Option A (document and validate):**

If the current check is provably safe because any origin that can seed a
given counter always shares the same break-path match reachability, then:

1. Document that invariant as a comment at the check site
2. Add a `debug_assert!()` that validates the invariant at compile time:
   for every `(counter, origin)` pair in the seed set, verify that all
   targets with the same counter agree on `break_is_match` and
   `break_is_match_at_end`

**Option B (use the effect model):**

Route seed-at-EOI resolution through the compiled effect model:

1. Look up the seed's specific origin in `target_effects`
2. Check whether that origin's break path has a `Match` or `MatchAtEnd`
   effect
3. Remove the broad scan over all targets

Option B is preferred if Patch 8 is complete, since the infrastructure will
already exist.

**Files to change:**

- `src/dfa/tier3.rs` — EOI seed resolution block (lines 4028-4044)
- Possibly `src/dfa/tier3_effects.rs` — if adding a helper to query
  break-path match reachability from compiled effects

**Tests:**

- `cargo test`
- Construct a test case with multiple counters where one counter's break path
  is match-reachable and another's is not; verify the seed resolves correctly

**Done criteria:**

- EOI seed resolution is either documented with a validated invariant or
  routed through the effect model
- All tests pass

---

## Priority 4: Optimize

### Patch 10: Remove hot-path heap allocations

**Finding:** 8

**Problem:**

The matcher allocates `Box<[EffectAtom]>` at approximately 10 sites during
the per-byte matching loop (see Finding 8 in the review).  Each allocation
is a small boxed slice, typically containing a single `EffectAtom::Match` or
`EffectAtom::AddTail`.  This violates the project rule that hot matching code
should avoid heap allocation.

**Fix:**

If Patch 8 Phase C (intern atom bundles) is done, this is already solved.

If Patch 8 Phase C is deferred, apply a smaller fix:

1. Replace `atoms: Box<[EffectAtom]>` in `PendingEffect` with an inline
   enum:

   ```rust
   enum EffectAtoms {
       Single(EffectAtom),
       Pair(EffectAtom, EffectAtom),
       Heap(Box<[EffectAtom]>),
   }
   ```

2. The common single-atom case (which covers all current deposit sites)
   becomes zero-allocation.

3. Update `resolve_pending()` to iterate over `EffectAtoms` instead of
   `&[EffectAtom]`.

**Files to change:**

- `src/dfa/tier3_effects.rs` — `EffectAtoms` type, resolution iteration
- `src/dfa/tier3.rs` — all `PendingEffect` construction sites

**Tests:**

- `cargo test`
- Optional: benchmark comparison

**Done criteria:**

- No heap allocation per byte in the common case
- All tests pass

---

### Patch 11: Enforce homogeneous atom bundles or per-atom guard resolution

**Finding:** 7

**Problem:**

`resolve_pending()` applies match-reachability gating to the entire
`PendingEffect` bundle by scanning whether *any* atom is `Match` or
`MatchAtEnd` (line 611 of `tier3_effects.rs`).  Non-match atoms in a mixed
bundle inherit match-oriented reachability gating.

**Current safety:**

Today all bundles are homogeneous (single-atom).  This is safe but
undocumented.

**Fix (lightweight):**

Add a `debug_assert!()` in `resolve_pending()` that validates bundle
homogeneity:

```rust
debug_assert!(
    atoms.iter().all(|a| matches!(a, EffectAtom::Match | EffectAtom::MatchAtEnd))
    || atoms.iter().all(|a| !matches!(a, EffectAtom::Match | EffectAtom::MatchAtEnd)),
    "mixed match/non-match atom bundle is not supported"
);
```

**Fix (full, deferred):**

If mixed bundles are ever needed, switch to per-atom guard evaluation.

**Files to change:**

- `src/dfa/tier3_effects.rs` — add debug assertion in `resolve_pending()`

**Tests:**

- `cargo test`

**Done criteria:**

- Debug assertion documents the homogeneity invariant
- All tests pass

---

## Patch Dependency Graph

```
Patch 1 (remove required_breaks)
  │
  ├── Patch 3 (remove EndOnly) ─── independent
  │
  └── Patch 4 (reconcile Match/MatchAtEnd)
        │
        └── Patch 5 (audit OR/AND semantics)
              │
              └── Patch 6 (semantic tests)

Patch 2 (split pending queues) ─── independent of Patches 1,3,4

Patch 7 (extract helpers) ─── depends on Patches 1-4 being stable
  │
  └── Patch 8A (template IDs + authority flip) ─── depends on Patch 7
        │
        ├── Patch 8B (remove legacy nonlocal fields)
        │     │
        │     └── Patch 9 (EOI seeds) ─── depends on Patch 8A for Option B,
        │                                 and ideally lands after 8B
        │
        └── Patch 8C (intern atom bundles)
              │
              └── Patch 10 (remove allocs) ─── depends on Patch 8C unless
                                               solved as part of 8C

Patch 11 (homogeneous bundles) ─── independent, can be done anytime
```

**Recommended execution order:**

1. Patches 1, 2, 3 — can be done in parallel, all independent
2. Patch 4 — depends on Patch 3 only if `EndOnly` removal affects the same
   lines
3. Patch 5 — audit pass, no behavior change
4. Patch 6 — adds tests, benefits from Patches 1-4 being landed
5. Patch 11 — small, independent, can slot in anywhere
6. Patch 7 — create the single runtime deposition funnel
7. Patch 8A — make `CompiledTargetEffects` authoritative for all nonlocal
   runtime behavior
8. Patch 8B — delete the old nonlocal truth from `Tier3OriginKind`
9. Patch 9 — route the remaining EOI seed logic through the same model
10. Patch 8C and Patch 10 — optimize the now-authoritative model

## Estimated Effort

| Patch | Size | Risk | Effort |
|-------|------|------|--------|
| 1 | Small | Low | 1-2 hours |
| 2 | Medium | Medium | 2-4 hours |
| 3 | Small | Low | 1 hour |
| 4 | Small | Low | 1 hour |
| 5 | Medium | Low (audit only) | 2-3 hours |
| 6 | Medium | Low | 3-4 hours |
| 7 | Medium | Medium | 3-4 hours |
| 8 | Large | High | 8-12 hours (across 3 phases) |
| 9 | Small-Medium | Medium | 2-3 hours |
| 10 | Small-Medium | Low | 2-3 hours |
| 11 | Small | Low | 30 minutes |

Total: approximately 25-37 hours of focused work.

## Validation Strategy

After each patch:

1. `cargo test` — full test suite
2. `cargo clippy -- -D clippy::all` — no new warnings
3. `cargo fmt -- --check` — formatting

After Patches 7-8 (the structural changes):

4. `cargo test test_fuzz -- --ignored` — property-based tests (~2 minutes)
5. Manual `--debug` trace comparison for 3-5 representative patterns
6. Short fuzz soak: `cargo +nightly fuzz run fuzz_match -- -runs=50000`

After all patches:

7. Extended fuzz soak: `cargo +nightly fuzz run fuzz_match -- -runs=500000`
8. `cargo +nightly fuzz run fuzz_differential -- -runs=100000`
9. Memory assertion update: `python3 scripts/bless_memory.py`
