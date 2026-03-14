# Tier 3 Typed Effects Post-Implementation Review

## Purpose

This document reviews the current Tier 3 typed-effects implementation after the
patch-plan work landed.

It focuses on four questions:

1. What parts of `docs/tier3-typed-effects-patch-plan.md` have actually been
   achieved?
2. What gaps still remain between the current code and the intended end state?
3. What simplifications or refactorings are still worth doing?
4. How could the typed-effects model be straightened even further?

This review is deliberately broader than the earlier
`docs/tier3-typed-effects-implementation-review.md`.  That earlier document was
written while the migration was still incomplete and while multiple semantic
representations coexisted.  The current code is much further along, so the
questions have changed from "is typed effects viable?" to "how clean is the
post-migration result, and what should happen next?"

## Scope and Method

Reviewed files:

- `src/dfa/tier3.rs`
- `src/dfa/tier3_effects.rs`
- `src/dump.rs`
- `docs/tier3-typed-effects-patch-plan.md`
- `docs/tier3-typed-effects-implementation-review.md`

Historical context considered:

- the patch-plan commits through `patch 11`
- later follow-up commits through `f2ab52f`

Validation performed during this review:

- static code review of the files above
- `cargo test` on the current tree

Result of `cargo test` during this review:

- 499 passed
- 0 failed
- 2 ignored

That matters because the main remaining questions are architectural and
maintenance-oriented, not "the code is obviously red right now".

## Executive Judgment

The typed-effects migration has been largely successful.

The central goal of the patch plan was to make compiled effects authoritative
for Tier 3's nonlocal runtime behavior.  That goal is now substantially met:

- `Tier3Analysis::target_effects` is explicitly documented as the runtime
  authoritative source for nonlocal target-keyed behavior in
  `src/dfa/tier3.rs:170-178`
- `Tier3Analysis::origin_effects` now carries the origin-keyed match/deferred
  facts that were previously scattered across `target_is_match`,
  `target_is_match_at_end`, and `target_deferred_asserts` in
  `src/dfa/tier3.rs:180-188`
- the runtime step logic reads compiled effects rather than reconstructing
  break-path behavior from the old `break_*` side fields in
  `src/dfa/tier3.rs:2936-3147` and `src/dfa/tier3.rs:3539-3650`
- the pending-effect queue model is materially simpler and safer than the old
  mix of specialized vectors in `src/dfa/tier3.rs:2844-2874`

The implementation is therefore in much better shape than the code reviewed in
`docs/tier3-typed-effects-implementation-review.md`.

However, the migration is not perfectly "finished" in the strongest possible
sense.  The biggest remaining issues are now narrower:

- one important normalization gap still exists for per-tail alternative paths
- end-of-input handling still straddles typed effects and legacy DFA-state
  deferred-assert resolution
- a few model boundaries remain implicit rather than first-class
- several helper, dump, and test improvements would make the implementation
  easier to reason about and harder to regress

My bottom-line assessment is:

- the current implementation is good enough to keep and build on
- the patch plan's main semantic goals were achieved
- the highest-value remaining work is now targeted cleanup and straightening,
  not another major redesign

## What the Current Implementation Got Right

### 1. The authority flip really happened

The patch plan's key handoff point was "make `CompiledTargetEffects`
authoritative for nonlocal runtime behavior".  That is now true in the core
step logic.

Evidence:

- `Tier3Analysis::target_effects` is explicitly documented as runtime
  authoritative in `src/dfa/tier3.rs:170-178`
- transitions store target-state handles, and runtime looks up compiled effects
  through `target_effects[target.idx()]` in `src/dfa/tier3.rs:1040-1047` and
  `src/dfa/tier3.rs:1780-1789`
- step-time break behavior uses `effects.on_break` and `effects.guarded` in
  `src/dfa/tier3.rs:2995-3022` and `src/dfa/tier3.rs:3120-3146`
- resolved-tail behavior uses the same compiled tables in
  `src/dfa/tier3.rs:3538-3650`

This is the single most important improvement over the state captured in the
older implementation-review doc.

### 2. The model got simpler than the original plan in good ways

The current code is actually cleaner than the original sketch in several
respects.

#### Timing simplified successfully

The original plan still contemplated a richer timing space.  The current code
collapsed this to one explicit deferred timing:

- `EffectTiming` contains only `NextByte` in
  `src/dfa/tier3_effects.rs:100-120`
- end-of-input resolution is handled by calling `resolve_pending()` with
  `at_end=true` and `next=None` in `src/dfa/tier3.rs:3930-3939`

That is simpler than carrying a permanent `EndOnly` branch everywhere, and the
current comments explain why the simplification is sound.

#### Single-atom effects removed a whole class of ambiguity

The original design allowed atom bundles.  The current implementation narrows
both `GuardedEffect` and `PendingEffect` to exactly one atom:

- `GuardedEffect` uses `atom: EffectAtom` in
  `src/dfa/tier3_effects.rs:426-439`
- `PendingEffect` does the same in `src/dfa/tier3_effects.rs:671-684`

This is an excellent change.  It deletes the old mixed-bundle ambiguity from
Finding 7 of the prior review and makes `resolve_pending()` materially easier
to reason about.

#### `CompiledOriginEffects` was a good adaptation

The original plan focused mainly on target-keyed effects.  The implementation
added `CompiledOriginEffects` in `src/dfa/tier3_effects.rs:595-658`, and that
was the right move.

It consolidates three previously scattered channels:

- direct origin-keyed `Match`
- origin-keyed `MatchAtEnd`
- origin-keyed deferred target assertions

That addition significantly reduced the amount of ad hoc per-origin logic in
`tier3.rs`.

### 3. Queue lifecycle is substantially healthier

The swap-buffer design in `src/dfa/tier3.rs:2844-2874` and its use in
`src/dfa/tier3.rs:3508-3666` closes one of the most suspicious holes from the
earlier review.

The conceptual model is now clear:

- resolve `pending_effects_current`
- deposit second-order effects into `pending_effects_next`
- clear and swap
- let `step_slow` append first-order effects to the new current queue

This is much easier to trust than the old clear-in-place behavior.

### 4. The test surface is materially better than before

`tier3_effects.rs` now contains semantic tests in
`src/dfa/tier3_effects.rs:1406-1760`, including:

- unconditional resolution
- assert pass/fail
- OR alternatives (`\b` vs `\B`)
- conjunctive chains
- `AddTail`
- `AddSeed`
- `MatchAtEnd`

That is a real improvement over the earlier review's complaint that the module
only had arena/display tests.

## Patch-Plan Status Against the Current Code

This section compares the current code to the intent of
`docs/tier3-typed-effects-patch-plan.md`.

### Patch 1 - remove or enforce `required_breaks`

**Status:** done

- `EffectGuard` now only carries `assert_chain` in
  `src/dfa/tier3_effects.rs:335-338`
- the module docs explain why `required_breaks` was removed in
  `src/dfa/tier3_effects.rs:26-66`

### Patch 2 - split current vs next pending queues

**Status:** done

- `pending_effects_current` / `pending_effects_next` are present in
  `src/dfa/tier3.rs:2863-2871`
- their lifecycle is implemented in `src/dfa/tier3.rs:3508-3666`

### Patch 3 - remove dead `EndOnly`

**Status:** done

- `EffectTiming` only has `NextByte` in `src/dfa/tier3_effects.rs:115-120`
- no end-only pending queue remains on the matcher in `src/dfa/tier3.rs`

### Patch 4 - reconcile `MatchAtEnd` vs `Match`

**Status:** done

- break-deferred guarded effects now emit `EffectAtom::Match` in
  `src/dfa/tier3_effects.rs:1097-1129`
- `MatchAtEnd` is reserved for truly EOI-only match signals in
  `src/dfa/tier3_effects.rs:398-404`

### Patch 5 - audit OR vs AND semantics

**Status:** mostly done, but not fully finished

The code now documents OR/AND semantics in several important places:

- `Tier3BreakSeed.deferred_asserts` in `src/dfa/tier3.rs:272-287`
- `BreakEffects` field semantics in `src/dfa/tier3_effects.rs:289-311`
- `CompiledOriginEffects.deferred_chain_ids` in
  `src/dfa/tier3_effects.rs:617-625`
- break-seed dedup semantics in `src/dfa/tier3.rs:516-542`

However, one meaningful gap remains: per-tail deferred alternatives are still
collapsed to a single chain per tail by `break_consuming_tails()` in
`src/dfa/tier3.rs:2646-2732`.

### Patch 6 - semantic tests in `tier3_effects.rs`

**Status:** mostly done

The module now has semantic tests, but it still lacks:

- direct `compile_target_effects()` tests
- queue-lifecycle / second-order scheduling tests
- tests for non-word-boundary deferred assertions

### Patch 7 - extract deposition helpers

**Status:** partially done

- `enqueue_target_deferred_match()` exists in
  `src/dfa/tier3_effects.rs:721-736`
- guarded-effect enqueue is still open-coded in three runtime sites:
  `src/dfa/tier3.rs:3016-3022`, `src/dfa/tier3.rs:3140-3146`, and
  `src/dfa/tier3.rs:3624-3630`

### Patch 8A / 8B - authority flip and delete old nonlocal truth

**Status:** mostly done

- target-keyed nonlocal behavior is now effect-driven
- `Tier3OriginKind::Increment` no longer stores the old break-side payload; it
  stores only local-step data plus `break_effects_id` in
  `src/dfa/tier3.rs:239-259`

### Patch 8C / Patch 10 - atom/template cleanup and hot-path allocation removal

**Status:** partially done, with a good simplification but not the fully
explicit template shape the plan imagined

- the large boxed atom bundles are gone
- the runtime now carries single atoms by value
- but `origin_targets[i].idx()` is still only a de facto effect-template key,
  not an explicit newtype-backed template ID

### Patch 9 - route EOI seed resolution through the effect model

**Status:** partially done via Option A, not fully done via Option B

- the invariant behind the EOI scan is now documented in
  `src/dfa/tier3.rs:3940-3951`
- it is validated by a debug assertion in `src/dfa/tier3.rs:970-1003`
- but `finish()` still uses a broad scan over `target_effects` rather than a
  purely origin-specific effect lookup in `src/dfa/tier3.rs:3952-3969`

### Patch 11 - homogeneous bundles / per-atom resolution

**Status:** effectively achieved by design

The single-atom `GuardedEffect` / `PendingEffect` shape made the original
mixed-bundle concern disappear.

## Current Architecture: Where Runtime Authority Actually Lives

The post-migration implementation no longer has one single effect table.  It
has two authoritative compiled tables plus one legacy state-based EOI path.

### 1. `target_effects`: target-keyed, nonlocal target consequences

Location:

- `src/dfa/tier3.rs:170-178`
- `src/dfa/tier3_effects.rs:543-552`

This table is the authoritative source for:

- local target step shape (`TargetStep`)
- unconditional immediate target effects
- on-break effects
- guarded target effects

Runtime use sites include:

- post-break tail stepping in `src/dfa/tier3.rs:2936-3023`
- counter-entry stepping in `src/dfa/tier3.rs:3083-3147`
- resolved-tail advancement in `src/dfa/tier3.rs:3539-3650`

### 2. `origin_effects`: origin-keyed fallback and deferred target assertions

Location:

- `src/dfa/tier3.rs:180-188`
- `src/dfa/tier3_effects.rs:595-658`

This table is authoritative for:

- dead-target direct match fallback
- dead-target match-at-end fallback
- target-deferred assertion entry points

Runtime use sites include:

- `check_origin_match_flags!` in `src/dfa/tier3.rs:2917-2926`
- target-deferred deposition in `src/dfa/tier3.rs:2947-2951`,
  `src/dfa/tier3.rs:3030-3034`, `src/dfa/tier3.rs:3561-3565`, and
  `src/dfa/tier3.rs:3645-3648`

### 3. `DfaState::resolve_deferred_at_end()`: remaining legacy EOI path

Location:

- `src/dfa/tier3.rs:3907-3918`

This is the most significant remaining authority split.  `finish()` still
resolves counter-free DFA-state deferred assertions through the old state-based
path before also resolving typed pending effects.

That split is not obviously wrong, but it means the implementation has not
reached the cleanest possible "one semantic execution model" end state.

## Main Findings

## 1. The highest remaining correctness risk is per-tail alternative collapse

**Classification:** high-risk semantic gap

`break_consuming_tails()` still computes exactly one deferred-assert chain per
tail origin.

Relevant code:

- `src/dfa/tier3.rs:2652-2670`
- `src/dfa/tier3.rs:2694-2699`
- `src/dfa/tier3.rs:2718-2728`

Current behavior:

- the traversal keeps `best_deferred[i]`, the minimum deferred-chain length
  seen for a node
- equal-length or longer alternatives are discarded
- `per_tail_asserts[i]` stores only one `Vec<StateIdx>` for each consuming
  state

Why this matters:

- Bug 51 taught that OR-alternative paths must not be flattened into one AND
  chain
- Bug 52 taught the same lesson for break seeds
- the tail path now has the same structural smell: if the same tail is
  reachable through `\b` on one path and `\B` on another, one path wins by
  traversal order rather than semantics

This is the single most important gap left versus the patch plan.

### Recommendation

Change per-tail lowering from:

- one tail -> one chain

into one of these explicit representations:

- one tail -> many chain IDs (OR over chains), or
- many deferred-tail entries with the same `origin` but different `chain_id`

The second option fits the current single-atom/single-guard model nicely and
requires less redesign:

- `DeferredTail { origin, chain_id }` entries may repeat `origin`
- OR semantics become "multiple guarded `AddTail` entries"
- runtime remains unchanged because each `GuardedEffect` is already independent

## 2. The pending-effect context model is still narrower than the collector logic

**Classification:** medium-risk semantic mismatch

`PendingEffect` stores only one piece of boundary context:

- `prev_was_word: bool` in `src/dfa/tier3_effects.rs:667-683`

`resolve_pending()` reconstructs the previous byte as either `b'a'` or `b' '`:

- `src/dfa/tier3_effects.rs:873-881`

That is sufficient for word-boundary-family assertions, but the collectors
currently defer **all non-`End` assertions**, not just word-boundary ones:

- break closure: `src/dfa/tier3.rs:2574-2594`
- per-tail break paths: `src/dfa/tier3.rs:2679-2687`
- origin target deferred asserts: `src/dfa/tier3.rs:765-775`

That means the collection side is broader than the stored boundary model.

### Why this matters

The current representation is obviously suitable for:

- `\b`
- `\B`
- `\<` / `\>`-style word-boundary variants if encoded through wordness only

It is not obviously sufficient for assertions whose truth depends on more than
"previous byte is word/non-word", for example:

- line-start / line-end family assertions in multiline mode
- assertions whose semantics depend on explicit newline context
- any future deferred assertion kind needing exact previous-byte class rather
  than wordness only

The current test surface does not exercise this boundary.

### Recommendation

Pick one of two paths explicitly.

#### Option A: constrain the deferred universe

Add compile-time or debug-time checks that only word-boundary-family asserts may
enter pending effects.

That would make the current `prev_was_word` representation explicitly correct.

#### Option B: widen the boundary context model

Replace `prev_was_word: bool` with a richer boundary snapshot, for example:

```rust
struct PendingBoundary {
    prev_kind: PrevBoundaryKind,
}

enum PrevBoundaryKind {
    Start,
    Word,
    NonWord,
    Newline,
    Other(u8),
}
```

That is more general, but probably only worth doing if multiline/start-line
assertions really do flow through Tier 3 deferred queues.

My recommendation is to do Option A first by proving or asserting the current
invariant.  If the invariant does not hold, then widen the stored context.

## 3. `finish()` still uses two semantic systems

**Classification:** medium-priority architectural gap

`finish()` currently uses three checks in sequence:

1. `last_nb_counter_free_mae` in `src/dfa/tier3.rs:3890-3892`
2. `resolve_deferred_at_end()` on `no_break_current` / `clean_nb` in
   `src/dfa/tier3.rs:3907-3918`
3. typed pending-effect resolution in `src/dfa/tier3.rs:3930-3990`

This is better than the pre-migration state, but it still means EOI semantics
are split across:

- DFA-state deferred assertions
- transition-derived effect tables
- pending effects

### Why this matters

The code is now mostly effect-driven mid-stream, but EOI still requires the
reviewer to remember a second, older model.

That is not just aesthetic debt; it is also the place where future semantic
changes are most likely to drift.

### Recommendation

If you want the cleanest end state, make `finish()` effect-driven too.

The most straightforward route is:

- compile the counter-free deferred assertions that currently live only on DFA
  states into an explicit effect-side structure
- or schedule them into pending effects during stepping so `finish()` only
  drains queues and applies already-compiled effect facts

I would not rush this if the current code is stable, but this is the biggest
remaining place where the implementation still feels half old-model / half
new-model.

## 4. `CompiledOriginEffects` is useful, but still half normalized

**Classification:** high-value simplification opportunity

`CompiledOriginEffects` in `src/dfa/tier3_effects.rs:595-658` is a good table,
but it does not yet use the same structural vocabulary as
`CompiledTargetEffects`.

Current shape:

- `is_match_at_end: bool`
- `is_match: bool`
- `deferred_chain_ids: Box<[AssertChainId]>`

Runtime consequence:

- one custom macro (`check_origin_match_flags!`) for immediate facts in
  `src/dfa/tier3.rs:2917-2926`
- one custom helper for deferred target assertions in
  `src/dfa/tier3_effects.rs:721-736`

This is cleaner than the older arrays, but still not fully straight.

### Better shape

A straighter model would give origin effects the same shape as target effects,
for example:

```rust
struct CompiledOriginEffects {
    immediate: Box<[EffectAtom]>,
    guarded: Box<[GuardedEffect]>,
}
```

Then:

- dead-target fallback would just apply `immediate`
- target-deferred assertions would just enqueue `guarded`
- the runtime would stop having special "origin macro" code at all

This is the single best refactor if the goal is to make typed effects feel like
one coherent model instead of two nearby structures.

## 5. Helper extraction is still incomplete in the hot path

**Classification:** medium-value maintainability cleanup

The runtime still open-codes guarded-effect queueing at three sites:

- `src/dfa/tier3.rs:3016-3022`
- `src/dfa/tier3.rs:3140-3146`
- `src/dfa/tier3.rs:3624-3630`

And it still open-codes immediate/on-break atom application in several places:

- `src/dfa/tier3.rs:2954-2963`
- `src/dfa/tier3.rs:2995-3012`
- `src/dfa/tier3.rs:3120-3137`
- `src/dfa/tier3.rs:3547-3557`
- `src/dfa/tier3.rs:3597-3619`

### Why this matters

These sites are semantically aligned today, but they are the next obvious
places drift could return.

### Recommendation

Extract two more helpers:

```rust
fn enqueue_guarded_effects(
    queue: &mut Vec<PendingEffect>,
    guarded: &[GuardedEffect],
    prev_was_word: bool,
)

fn apply_immediate_atoms(
    atoms: &[EffectAtom],
    ... matcher outputs ...,
)
```

The current code already proved that helper extraction helps.  Finishing that
cleanup would reduce noise in `step_slow_impl!` and make code review easier.

## 6. Some hot-path lookups are still structurally simple rather than explicit

**Classification:** low-to-medium priority performance refactor

The current transition payload uses parallel arrays:

- `origin_keys`
- `origin_targets`

in `src/dfa/tier3.rs:1040-1047`.

At runtime, the code repeatedly finds the matching origin with linear search:

- tail stepping: `src/dfa/tier3.rs:2935-2936`
- counter-entry stepping: `src/dfa/tier3.rs:3083-3088`

This is not a correctness issue, and the arrays are probably small in the
common case.  Still, the typed-effects migration made the lookup boundary much
clearer, so the remaining linear search stands out more than it used to.

### Recommendation

If profiling shows Tier 3 spending meaningful time here, consider one of:

- store a compact per-transition origin -> target-index map
- store `origin_keys` sorted and use binary search
- store a tiny fixed-size open-addressed table specialized for the observed
  origin counts

I would treat this as a post-cleanup optimization, not as something to do
before the semantic/model issues above.

## 7. Several build-time-only tables still live on the runtime analysis object

**Classification:** medium-value memory and clarity issue

The following fields are documented as build-time-only or dump-oriented, yet are
still stored on `Tier3Analysis`:

- `targets` in `src/dfa/tier3.rs:71-82`
- `break_seeds` in `src/dfa/tier3.rs:91-101`
- `target_is_match_at_end` in `src/dfa/tier3.rs:116-125`
- `target_is_match` in `src/dfa/tier3.rs:127-133`
- `target_deferred_asserts` in `src/dfa/tier3.rs:135-142`
- `break_effects` in `src/dfa/tier3.rs:193-205`

Some of these are still used by `populate()`, so they cannot all be deleted
immediately.  But they do represent a lingering split between:

- the runtime authoritative compiled tables
- the build-time scaffolding retained for compilation/dump/debugging

### Recommendation

There are two levels of cleanup possible.

#### Near-term cleanup

Use `target_effects.step` instead of `analysis.targets[...]` in the few places
where `populate()` still only needs to know whether a target is `Advance` or
`Increment`, for example:

- `src/dfa/tier3.rs:1806-1810`
- `src/dfa/tier3.rs:1863-1869`
- `src/dfa/tier3.rs:2020-2023`

That would reduce one more semantic channel without requiring a memory-layout
change.

#### Longer-term cleanup

If runtime memory footprint matters, split dump/build scaffolding from runtime
analysis more aggressively:

- keep only the authoritative compiled tables in the runtime `Regex`
- retain the legacy/build-time structures only under dump/debug-only builds,
  or regenerate dump output from compiled tables

I would treat this as a memory/maintenance tradeoff, not an urgent correctness
fix.

## 8. The dump path still under-represents the authoritative runtime model

**Classification:** medium-value tooling gap

`dump.rs` shows `target_effects` and the assertion-chain arena in
`src/dump.rs:168-214`, but it still emphasizes older analysis tables in the
Tier 3 section:

- `target_is_match`
- `target_is_match_at_end`
- `target_deferred_asserts`
- `break_seeds`
- `targets`
- `break_effects`

See `src/dump.rs:425-649`.

What it does **not** show is the full `origin_effects` table, even though that
is now part of the runtime authority boundary.

### Recommendation

Add an `origin_effects` dump section that shows, per consuming origin:

- `immediate`-equivalent flags (`is_match`, `is_match_at_end`)
- deferred assertion chain IDs

Also consider labeling the older arrays as "build-time scaffolding" in the dump
so a reader does not confuse them with the main runtime path.

## 9. There are a few low-level cleanup items that are still worth doing

**Classification:** low severity, good cleanup value

### `Tier3BreakSeed.assert_chain_id` is populated but not used

- populated in `src/dfa/tier3.rs:956-960`
- `compile_target_effects()` still re-interns `bs.deferred_asserts` in
  `src/dfa/tier3_effects.rs:1147-1155`

Use the precomputed ID or remove the field.

### A few comments are now stale

Examples:

- the module header in `src/dfa/tier3_effects.rs:16-19` still describes timing
  as if there were an "end-of-input only" timing mode
- `resolve_pending()` docs still mention "break mask + assertion chain" in
  `src/dfa/tier3_effects.rs:842-845` even though the break mask was removed

These are not semantic bugs, but they increase audit friction.

### `EffectAtom` looks eligible for `Copy`

`GuardedEffect` enqueue still does `ge.atom.clone()` in
`src/dfa/tier3.rs:3020`, `src/dfa/tier3.rs:3144`, and
`src/dfa/tier3.rs:3628`.

The atom is small and value-like.  If all embedded types are `Copy`, making
`EffectAtom` `Copy` would be a simple cleanup.

## What Gaps Remain Compared to the Patch Plan?

At a high level, the patch plan was implemented successfully.  The remaining
gaps are not "the plan failed" gaps.  They are "the implementation is usable,
but not yet maximally straight" gaps.

The most important remaining deltas are:

1. **Per-tail OR semantics are still not explicit.**
   - This is the only major remaining semantic-normalization gap.

2. **`finish()` is still partly legacy-driven.**
   - Mid-stream behavior is effect-driven; EOI is not fully there.

3. **`CompiledOriginEffects` has not been promoted to full effect vocabulary.**
   - It is an important partial normalization, but still a partial one.

4. **Helper extraction stopped halfway.**
   - Enough to land the migration, not enough to eliminate future drift risk.

5. **Dump/tests/documentation lag the final model.**
   - The code is ahead of the supporting tooling.

## Simplifications and Refactorings Worth Doing Next

This section is intentionally pragmatic.  These are the changes I think would
most improve the code-to-complexity ratio from here.

### 1. Finish helper extraction

This is the smallest high-leverage cleanup.

Add:

- `enqueue_guarded_effects(...)`
- `apply_effect_atom(...)` or `apply_effect_atoms(...)`

That reduces repeated logic in `step_slow_impl!` and in pre-step tail
resolution.

### 2. Promote `CompiledOriginEffects` to the same vocabulary as target effects

Current shape is useful but still custom.

Best next straightening step:

- replace origin booleans + chain IDs with `immediate` and `guarded`
- let target and origin code paths use the same execution helpers

If I had to pick one refactor that best improves conceptual clarity, this is
it.

### 3. Formalize the de facto template key

`Transition.origin_targets` currently stores a raw `StateIdx`, and comments say
that the NFA target index acts as the effect-template lookup key in
`src/dfa/tier3.rs:1040-1047` and `src/dfa/tier3.rs:1780-1789`.

That works, but it is implicit.

Introduce a newtype such as:

```rust
struct TargetEffectsId(u32);
```

Even if it is still backed by the target state's index, naming the concept
would make the authority boundary easier to understand and would leave room for
future dedup or compaction.

### 4. Decide whether `BreakMask` is still a real future or just a ghost

`BreakMask` remains as future scaffolding in `src/dfa/tier3_effects.rs:80-95`.
That is not wrong, but it is now one of the last pieces still carrying
"Proposal 3 maybe later" conceptual weight in the current code.

Two valid choices exist:

- keep it, but document clearly that no current review finding requires it
- remove it entirely until a concrete provenance patch actually needs it

Given the current code, I would lean toward keeping it only if the team still
believes provenance-aware state is a live future direction.

### 5. Consider trimming runtime-retained build scaffolding

If memory footprint matters, move toward retaining only the authoritative
compiled tables in runtime `Regex` state.

This is a second-tier cleanup, but it would tighten the relationship between:

- what the engine stores
- what the engine executes
- what reviewers need to reason about

## How to Straighten the Typed-Effects Model Even Further

If the goal is not just "working and reviewed" but "as straight as possible",
this is the architecture I would aim for.

## Stage 1: one effect vocabulary everywhere

Today there are two compiled effect structures:

- `CompiledTargetEffects`
- `CompiledOriginEffects`

but only the first one fully uses the target-step / immediate / on-break /
guarded vocabulary.

A straighter model would use:

- target-keyed effects for post-consumption targets
- origin-keyed effects for dead-target and origin-local deferred consequences
- the same immediate/guarded atom vocabulary in both places

That gets rid of the last custom booleans and special-case macro logic.

## Stage 2: explicit OR semantics for tails

Represent tail alternatives explicitly rather than by best-path heuristics.

The simplest compatible change is:

- one tail origin may compile into multiple guarded `AddTail` effects
- OR semantics come from multiple entries, not from one chain pretending to
  summarize several paths

That aligns perfectly with the single-atom current model.

## Stage 3: make EOI just another effect-resolution boundary

Right now `finish()` is still special.

A straighter end state would make the last boundary behave like the others:

- no special DFA-state deferred resolution path
- only effect-side resolution with `at_end=true`
- all EOI semantics visible in compiled tables plus pending queues

This is the natural "finish the normalization" endpoint.

## Stage 4: optionally tighten runtime memory/layout

Only after the semantic simplifications above, consider:

- explicit template IDs
- dropping runtime-retained build-only tables
- more compact transition-side lookup structures

Those are good cleanups, but they matter less than the semantic straightening.

## Recommended Next Work Sequence

If the goal is maximum benefit for minimum disruption, I would do the next work
in this order:

1. **Fix per-tail OR semantics**
   - highest remaining semantic risk
2. **Add tests for non-word deferred asserts and queue lifecycle**
   - close the most obvious coverage holes
3. **Extract `enqueue_guarded_effects()` and atom-application helpers**
   - reduce drift risk in hot-path code
4. **Promote `CompiledOriginEffects` to immediate/guarded form**
   - biggest clarity win
5. **Add `origin_effects` to dump output**
   - make the runtime model visible
6. **Optionally unify EOI onto effects only**
   - cleanest final normalization step

## Bottom Line

The patch plan succeeded.

The current implementation is not a half-migrated prototype anymore.  It has a
real effect authority boundary, a safer queue lifecycle, a narrower and cleaner
effect shape, and a substantially better test surface.

What remains is not another major rescue.  What remains is the kind of cleanup
that turns a successful migration into a durable architecture:

- make per-tail OR semantics explicit
- finish removing the last special runtime paths
- promote origin effects into the same vocabulary as target effects
- make dump/tests/comments reflect the actual authority boundary

If those steps are taken, Tier 3 typed effects will stop feeling like "the new
system plus a few old leftovers" and start feeling like one cohesive model.
