# Tier 3 EOI Unification Analysis

## Problem Statement

`finish()` in Tier 3 resolves end-of-input semantics through **two separate
systems**:

1. **DFA-state deferred assertions** — the legacy path inherited from the
   shared DFA infrastructure.  `DfaState::resolve_deferred_at_end()` evaluates
   assertions parked on the DFA state during epsilon closure, then checks
   `can_reach_match_at_end()` for downstream match reachability.

2. **Typed pending effects** — the new effect path.
   `tier3_effects::resolve_pending()` evaluates assertion chains on
   `PendingEffect` entries deposited by the step loop.

Both paths converge in `finish()` at `tier3.rs:3910-3962`, running sequentially.
A match from either path is accepted.

This means EOI semantics are split across two conceptual models.  Mid-input
stepping is fully effect-driven, but EOI still requires the reviewer to reason
about a second, older model.

## What Each Path Handles

### Path 1: DFA-state deferred assertions

**Source:** Assertions parked during `epsilon_closure()` in `dfa/mod.rs:355-358`.

When `epsilon_closure()` evaluates the NFA epsilon closure for a DFA state and
encounters an assertion that returns `AssertEval::Defer` (i.e., it needs the
next byte which is not yet known), the assertion's NFA state index is stored in
`DfaState.deferred_asserts`.

**Mid-input resolution:** At the start of `populate()`, when the next byte IS
known, these deferred assertions are resolved via
`DfaState::resolve_deferred()`.  Assertions that pass contribute their `out`
states to the epsilon closure for the next DFA state.

**EOI resolution:** In `finish()`, any remaining deferred assertions that were
never resolved (because no more input arrived) are evaluated with
`at_end=true, next=None` via `resolve_deferred_at_end()`.

**Scope:** This mechanism is **shared across all 4 DFA tiers** (Tier 1, 2, 3,
4).  It is not Tier-3-specific.

**What it catches in Tier 3:** Counter-free deferred assertions — assertions
that are part of the DFA state's epsilon closure but NOT gated by any counter
break.  For example, in `^a{1,2}\b.+$`, the `\b` might be in the no-break
DFA state's closure if the counter hasn't broken yet.

### Path 2: Typed pending effects

**Source:** `PendingEffect` entries deposited during the step loop by
`enqueue_guarded_effects()`.

These come from counter-break paths: when a counter instance breaks, the
`CompiledTargetEffects::guarded` effects (assertion-gated `Match`, `AddTail`,
`AddSeed` atoms) are deposited into `pending_effects_current`.

**Mid-input resolution:** At the start of each byte's processing,
`resolve_pending(at_end=false, next=Some(b))` evaluates all pending effects.

**EOI resolution:** In `finish()`, `resolve_pending(at_end=true, next=None)`
evaluates any remaining pending effects from the last byte.

**What it catches in Tier 3:** Counter-dependent deferred assertions — those
that fire only when a specific counter breaks.  Also break-path tails and seeds
gated by assertions.

## Why They Are Separate

The DFA-state deferred assertions predate the typed-effects system.  They were
designed as part of the shared DFA infrastructure (Tier 1 was the first tier,
and deferred assertions are how `\b` and `\B` work in the standard lazy DFA).

The typed-effects system was layered on top specifically for Tier 3's
counter-dependent nonlocal behavior.  It was never designed to replace the
DFA-state mechanism — it supplements it.

## What Would Need to Change for Unification

### Option A: Schedule DFA-state deferred assertions as PendingEffects

During `populate()` or during the step loop, convert the DFA state's
`deferred_asserts` into `PendingEffect` entries and deposit them into
`pending_effects_current`.  Then `finish()` would only need
`resolve_pending(at_end=true)` — no separate `resolve_deferred_at_end()`.

**Changes required:**

1. **`epsilon_closure()` in `dfa/mod.rs`:** Currently parks deferred
   assertions on `self.closure_deferred`, which gets stored on the DFA state.
   Would need to either:
   - Still park them (for mid-input resolution during populate), AND
   - Also schedule them as PendingEffects (for EOI)
   - Or stop parking them entirely and only use PendingEffects

2. **`populate()` mid-input resolution (`dfa/mod.rs` and tier-specific
   `populate()`):** Currently calls `resolve_deferred()` which returns extra
   NFA targets to fold into the next DFA state's closure.  This is a
   DFA-state-construction concern, not an effect concern.  If deferred
   assertions were only in the effect queue, we'd need a way to inject their
   resolved NFA targets back into `epsilon_closure()`.

3. **Cross-tier impact:** `DfaState.deferred_asserts` is used by:
   - `DfaMatcher` (Tier 1) — `populate()` and `finish()`
   - `Tier2DfaMatcher` — `populate()` and `finish()`
   - `Tier3DfaMatcher` — `populate()` and `finish()`
   - `Tier4DfaMatcher` — `populate()` and `finish()`

   Any change to how deferred assertions are handled would need to work for
   all tiers, or each tier would need its own variant.

4. **DFA state identity:** Deferred assertions are part of the DFA state's
   identity (they affect `DfaState.deferred_asserts` which is part of the
   hash/eq used for DFA state dedup in the state table).  Removing them from
   the DFA state would change state identity semantics.

**Difficulty: HIGH.** This touches the most fundamental DFA infrastructure.

### Option B: Convert DFA-state deferred assertions to effects at step time

During the Tier 3 step loop (after the DFA transition is computed), inspect
the current DFA state's `deferred_asserts` and deposit corresponding
`PendingEffect` entries.  This avoids changing `epsilon_closure()` or
`populate()`.

**Changes required:**

1. **In the Tier 3 step loop** (both fast path and slow paths): after
   computing the next DFA state, read its `deferred_asserts` and for each,
   create a `PendingEffect { timing: NextByte, guard: chain_for(assert),
   atom: Match, prev_was_word: is_word_byte(current_byte) }`.

2. **DFA-state deferred assertions would still exist** on the DFA state for
   `populate()`'s mid-input resolution.  But `finish()` would no longer call
   `resolve_deferred_at_end()` — the effects would be in the pending queue.

3. **Risk:** Double-firing.  The DFA-state deferred assertions are already
   resolved mid-input by `populate()`.  If we also deposit them as
   PendingEffects, the mid-input `resolve_pending()` might fire them AND
   `populate()` might fire them — double-counting.

   To avoid this, we'd need to deposit them ONLY for the last byte (for EOI
   purposes), or deposit them every byte but ensure the mid-input path
   doesn't double-fire.  This is tricky because we don't know at deposit
   time whether the current byte is the last one.

**Difficulty: MEDIUM.** Tier-3-specific, but has subtle double-firing risks.

### Option C: Duplicate-and-check at EOI only

Keep the current `resolve_deferred_at_end()` call in `finish()` but refactor
it to use the effect vocabulary:

1. In `finish()`, before calling `resolve_pending()`, convert the DFA state's
   `deferred_asserts` into temporary `PendingEffect` entries.
2. Append them to `pending_effects_current`.
3. Then call `resolve_pending(at_end=true)` once for everything.
4. Remove the separate `resolve_deferred_at_end()` call.

**Changes required:**

1. In `Tier3DfaMatcher::finish()`: iterate the relevant DFA state's
   `deferred_asserts`, intern each as a 1-element chain (or look up from
   the arena), create `PendingEffect` entries, and push them to
   `pending_effects_current` before calling `resolve_pending()`.

2. Need access to the arena for interning.  The arena is on
   `self.analysis.assert_chain_arena` — available.

3. Need `prev_was_word` context for the PendingEffect.  The DFA state
   has `prev_was_word` (via `DfaState.prev_was_word`).

4. The `check_reachability` in `eval_assert_chain` handles the downstream
   match check that `resolve_deferred_at_end` does via
   `can_reach_match_at_end`.

**Difficulty: LOW-MEDIUM.**  Tier-3-only change, no cross-tier impact, no
mid-input changes.  The only subtlety is making sure the counter-free
(`clean_nb`) vs contaminated (`no_break_current`) distinction is preserved
when converting.

**This is the recommended approach.**

## Why It Is Deferred

The current code is correct and stable.  The two-path EOI resolution works
because:

1. The DFA-state path handles counter-free deferred assertions (from the
   no-break or clean-nb DFA state).
2. The typed-effect path handles counter-dependent deferred assertions (from
   break-path effects).
3. They don't overlap (counter-free vs counter-dependent), so there's no
   double-firing risk.

Unifying them would improve conceptual clarity but doesn't fix any known bug.
The recommended approach (Option C) is surgical and low-risk, making it a
good candidate for a future cleanup pass.

## Recommended Implementation (Option C)

If this unification is attempted later, the steps are:

1. In `finish()`, after the `last_nb_counter_free_mae` check, convert the
   relevant DFA state's `deferred_asserts` into `PendingEffect` entries:

   ```rust
   // Convert DFA-state deferred asserts to PendingEffects.
   let state = if from_contaminated {
       if self.clean_nb != DfaStateId::DEAD {
           Some(&self.cache.inner.states[self.clean_nb.idx()])
       } else {
           None
       }
   } else if self.no_break_current != DfaStateId::DEAD {
       Some(&self.cache.inner.states[self.no_break_current.idx()])
   } else {
       None
   };
   if let Some(dfa_state) = state {
       let prev_was_word = dfa_state.prev_was_word;
       for &assert_idx in dfa_state.deferred_asserts.iter() {
           let chain_id = self.analysis.assert_chain_arena.intern(&[assert_idx]);
           self.pending_effects_current.push(PendingEffect {
               timing: EffectTiming::NextByte,
               guard: EffectGuard { assert_chain: chain_id },
               atom: EffectAtom::Match,
               prev_was_word,
           });
       }
   }
   ```

2. Remove the two `resolve_deferred_at_end()` calls.

3. The single `resolve_pending(at_end=true)` call handles everything.

4. Test with the full suite + fuzz.  Pay special attention to:
   - Patterns with `\b` or `\B` in counter-free positions
   - Multi-counter patterns where contamination applies
   - The `clean_nb` fallback path

5. The `resolve_deferred_at_end()` method on `DfaState` stays for Tiers 1, 2,
   and 4 — this change is Tier-3-only.
