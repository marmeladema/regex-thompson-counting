# Patch 8E/F/G: CompiledTargetEffects Authority Flip

## Context

Patches 8A–8D took a pragmatic shortcut: they introduced `BreakEffects` as
an interned side table and `BreakEffectsId` on `Tier3OriginKind::Increment`,
routed 3 runtime break-path sites through the table, removed 5 legacy fields,
inlined `PendingEffect.atom`, and replaced cloned origin actions with target
indices.

However, `CompiledTargetEffects` remains **dead at runtime**.  It is compiled
by `compile_all_target_effects()` and stored in `analysis.target_effects`,
but only read by `dump.rs`.  The runtime still reads:

- `analysis.targets[idx]` → `Tier3OriginKind` (4 sites)
- `analysis.break_effects[id]` → `BreakEffects` (3 sites)
- `analysis.break_seeds` → deferred assertion chain lookup (1 site)
- `analysis.target_assert_chain_ids` → target deferred chains (2 sites)
- `analysis.target_is_match_at_end` → EOI tail check (1 site)

The original plan says `CompiledTargetEffects` should be authoritative at
runtime for all nonlocal effects after Patch 8A.  This document describes
three sub-phases to close that gap.

## Pre-requisite: Fix MatchAtEnd divergence

### Problem

`compile_target_effects()` line 1096 gates `MatchAtEnd` in `on_break` on
`!be.has_deferred`:

```rust
if *break_is_match_at_end && !be.has_deferred {
    on_break.push(EffectAtom::MatchAtEnd);
}
```

But the runtime (lines 2992, 3131, 3649) unconditionally sets
`self.match_at_end = true` when `break_is_match_at_end` is true.

### Root cause

`break_closure()` already separates pure paths from deferred paths.
`is_match_at_end` is ONLY set for assertion-free `$ → Match` paths
(line 2557–2559: `if !through_deferred && can_reach_match[out.idx()]`).
The `has_deferred` flag means OTHER branches of the break epsilon closure
have assertions — not that the `$ → Match` sub-path is gated.

A CInc break can have BOTH a pure `$ → Match` path AND deferred assertion
paths on sibling branches.  The `&& !be.has_deferred` condition conflates
"this break closure has any deferred assertions anywhere" with "the
`$ → Match` sub-path requires assertion evaluation."

### Evidence

Pattern `^.{2,5}(\b|$)`, input `"ab"`:
- NFA: **MATCH** (correct — `$` matches at end-of-input)
- Tier 3 runtime: **MATCH** (correct — `break_is_match_at_end = true` fires)
- `CompiledTargetEffects`: would **NOT MATCH** — `MatchAtEnd` suppressed
  because `has_deferred = true` (the `\b` branch has a deferred assertion)

### Fix

Remove `&& !be.has_deferred` from `compile_target_effects()` line 1096.
Make `MatchAtEnd` unconditional in `on_break` when `break_is_match_at_end`
is true, matching the runtime behavior.

---

## Phase 8E — Authority flip

Switch the 4 runtime `analysis.targets` read sites to read from
`analysis.target_effects` instead.  After this phase,
`CompiledTargetEffects` is the runtime source of truth for all nonlocal
effects.

### Sites to change

**Site 1: Post-break-tail processing** (~line 2910)

```rust
// Before:
match pos.map(|i| &self.analysis.targets[t.origin_targets[i].idx()]) {
    Some(Some(Tier3OriginKind::Advance { new_origins, is_match_at_end, is_match })) => { ... }
    Some(Some(Tier3OriginKind::Increment { counter, min, max, ... })) => { ... }
    Some(None) => { ... }  // dead target
    None => {}              // no matching origin
}

// After:
match pos.map(|i| &self.analysis.target_effects[t.origin_targets[i].idx()]) {
    Some(Some(effects)) => {
        match &effects.step {
            TargetStep::Advance { new_origins } => {
                // advance + apply effects.immediate atoms
            }
            TargetStep::Increment { counter, min, max, continue_origins, advance_origins } => {
                // advance + continue + break with effects.on_break + effects.guarded
            }
        }
    }
    Some(None) => { ... }  // dead target (unchanged)
    None => {}              // no matching origin (unchanged)
}
```

**Site 2: Per-instance step** (~line 3090)

Same pattern as Site 1, in the per-instance counter loop.

**Site 3: Effect resolution tail consumption** (~line 3583)

Same pattern but with different targets:
- Tails go to `resolved_tails` (local vec), not `next_post_break_tails`
- Deferred effects go to `pending_effects_next`, not `pending_effects_current`
- Seeds go directly to `ranged_counters`/`inst_counters` (not `$current`/`$next`)

**Site 4: EOI seed scan** (~line 3988)

```rust
// Before:
for target_action in self.analysis.targets.iter().flatten() {
    if let Tier3OriginKind::Increment { counter: tc, min, break_is_match_at_end, break_is_match, .. } = target_action
        && *tc == counter && value >= *min
        && (*break_is_match_at_end || *break_is_match) { return true; }
}

// After:
for effects in self.analysis.target_effects.iter().flatten() {
    if let TargetStep::Increment { counter: tc, min, .. } = &effects.step
        && *tc == counter && value >= *min
        && effects.on_break.iter().any(|a| matches!(a, EffectAtom::Match | EffectAtom::MatchAtEnd))
    { return true; }
}
```

### Atom application logic

For **Advance** targets:

```rust
// Local step
for &new_o in new_origins.iter() {
    self.$next.advance(c_idx, entry, new_o);
}
// Nonlocal: immediate effects
for atom in effects.immediate.iter() {
    match atom {
        EffectAtom::Match => self.ever_matched = true,
        EffectAtom::MatchAtEnd => self.match_at_end = true,
        _ => {} // AddTail/AddSeed not expected for Advance
    }
}
```

For **Increment** targets on break:

```rust
// on_break atoms (skip AddSeed — handled by transition-level break_seeds until 8F)
for atom in effects.on_break.iter() {
    match atom {
        EffectAtom::Match => self.ever_matched = true,
        EffectAtom::MatchAtEnd => self.match_at_end = true,
        EffectAtom::AddTail { origin } => {
            if !tail_vec.contains(origin) { tail_vec.push(*origin); }
        }
        EffectAtom::AddSeed { .. } => {} // Deferred to 8F
    }
}

// guarded effects → PendingEffect (skip AddSeed atoms)
for ge in effects.guarded.iter() {
    debug_assert_eq!(ge.atoms.len(), 1);
    let atom = &ge.atoms[0];
    if matches!(atom, EffectAtom::AddSeed { .. }) { continue; }
    queue.push(PendingEffect {
        timing: ge.timing,
        guard: ge.guard,
        atom: atom.clone(),
        prev_was_word: is_word_byte(byte),
    });
}
```

### What stays the same in 8E

- Transition-level break seeds (`t.break_seeds` loop at line 3310) — unchanged
- `target_assert_chain_ids` for None targets — unchanged
- `target_is_match_at_end` / `target_is_match` for `check_tail_match_flags!` — unchanged
- `analysis.targets` on `Tier3Analysis` — still alive for compile-time use

### Deposition helper decision

The 4 specific helpers from Patch 7 (`enqueue_target_deferred_match`,
`enqueue_break_deferred_match`, `enqueue_deferred_tail`,
`enqueue_deferred_seed`) will be partially superseded by the generic
`on_break` + `guarded` dispatch loop in 8E.

**Decision to make during implementation:** whether to keep the helpers as
inner callees of the atom dispatch (e.g., `EffectAtom::AddTail` →
`push_tail()`), or inline the logic directly.  The rationale and final
choice should be documented here after 8E is implemented.

### Testing

- `cargo test`
- `cargo clippy -- -D clippy::all`
- Manual `--debug` traces for:
  - `^.{2,5}(\b|$)` on `"ab"` (MatchAtEnd + has_deferred case)
  - `\b\w{3,5}\b` on `"abc"` (break-deferred assertions)
  - `^a{1,3}.{2,4}$` on `"aaa"` (multi-counter seeds)

---

## Phase 8F — Absorb transition-level break seeds

After 8E, the only runtime data source outside `target_effects` is
`Transition.break_seeds` + `analysis.break_seeds`.

### Changes

1. **Stop skipping `AddSeed` atoms** in the `on_break` / `guarded` dispatch
   loops from 8E.
2. **Remove the transition-level break seeds loop** (lines ~3310–3337):
   - Ungated seeds: fire per-instance from `effects.on_break` `AddSeed` atoms
   - Gated seeds: deposit per-instance from `effects.guarded` `AddSeed` atoms
   - Deduplication: `seed()` method already deduplicates; PendingEffect
     duplicates resolve harmlessly
3. **Remove `Transition.break_seeds` field** (line ~984) and its population
   in `populate()`.
4. **Remove `compute_break_seeds()` method** on the cache (line ~2150).
5. **Remove `analysis.break_seeds` runtime lookup** (line ~3314):
   the assert chain ID is now in the `GuardedEffect.guard` from
   `CompiledTargetEffects.guarded`.

### Testing

- `cargo test`
- `cargo clippy -- -D clippy::all`
- `cargo test test_fuzz -- --ignored` (structural seed flow change)
- Manual `--debug` traces for patterns with break seeds:
  - `^a{1,17}b{4,13}$` on `"aaabbbb"` (ungated break seed)
  - `^a{1,5}(\b).{3,7}$` on `"aaabbb"` (gated break seed with deferred assertion)

---

## Phase 8G — Remove dead runtime structures

After 8E+8F, clean up structures no longer read at runtime.

### Structures to clean up

| Structure | Used by | Action |
|---|---|---|
| `analysis.break_effects` (Box<[BreakEffects]>) | compile_target_effects() only | Keep on Tier3Analysis for compile-time + dump. Remove break_effects_id from Increment if feasible. |
| `analysis.break_seeds` (Box<[Tier3BreakSeed]>) | compile_target_effects(), break seed walk | Keep for compile-time. Remove assert_chain_id field if no longer needed at runtime. |
| `analysis.targets` (Box<[Option<Tier3OriginKind>]>) | populate(), compile_target_effects() | Keep for compile-time. Update doc comments: build-time only. |
| `Transition.break_seeds` | Removed in 8F | Already gone. |
| Deposition helpers | Partially superseded by generic atom dispatch | Remove unused helpers; keep any still used for None-target path. |

### Additional work

- Update `dump.rs` if output format changes
- `python3 scripts/bless_memory.py` for struct size changes
- Document the helper decision (see Phase 8E section)
- Update `docs/tier3-typed-effects-patch-plan.md` authority checkpoints

### Testing

- `cargo test`
- `cargo clippy -- -D clippy::all`
- `cargo test test_fuzz -- --ignored`
- Short fuzz soak: `cargo +nightly fuzz run fuzz_match -- -runs=50000`

---

## Risks

| Risk | Phase | Mitigation |
|---|---|---|
| MatchAtEnd divergence (compiler bug) | Pre-req | Fix before 8E; add regression test |
| Per-instance seed fires multiple times | 8F | seed() deduplicates; safe but slightly wasteful |
| on_break atom ordering matters | 8E | Shouldn't — Match/MatchAtEnd/AddTail all idempotent |
| GuardedEffect.atoms has > 1 atom | 8E | All current are single-atom; add debug_assert |
| Effect resolution site has different queue/tail targets | 8E | Parameterize via closure args or separate match blocks |
| EOI seed scan behavioral change | 8E | Checking on_break atoms equivalent to raw fields |
