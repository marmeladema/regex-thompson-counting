# Bug 45 — Tier 3 false negative: pure break flags and consuming tails incorrectly gated behind sibling deferred assertions

## Bug summary

- **Pattern:** `^.{1,11}((\B|a?))?$`
- **Input:** `"aa"`, `"cd yc 1a"`, `"aaaaaaaaaaaa"` (any input ≥ 1 char)
- **Expected:** `true` (NFA oracle)
- **Actual (Tier 3):** `false` (false negative)
- **Affected tier:** Tier 3 (per-instance path), only with `unroll_limit=0`

## Root cause

The Bug 42 fix introduced a mechanism to defer counter break results
(`break_is_match_at_end`, `break_is_match`, `break_consuming_states`)
when the break path has deferred assertions (`break_deferred_asserts`).
This correctly prevented false positives from tails deposited before
the deferred assertion was resolved.

However, the fix was **too aggressive**: it deferred ALL break results
when ANY break path had a deferred assertion, even when some results
came from pure paths (no deferred assertion).

Three independent problems in the same root cause:

### Problem 1: Pure `break_is_match_at_end` deferred

`break_is_match_at_end` is set by `break_closure()` only for paths
to `$ → Match` that do NOT pass through any deferred assertion.  It's
a "pure" flag that should fire unconditionally.  But Bug 42's fix
wrapped it in `if !break_deferred_asserts.is_empty() { ... pending ... }`
instead of `self.match_at_end = true`.

In this pattern, the counter break from `.{1,11}` has:
- Pure path: `Split(8) → Assert($)(9) → Match` — sets `break_is_match_at_end = true`
- Deferred path: `Split(8) → Split(7) → Assert(\B)(4) → ...` — adds `\B` to `break_deferred_asserts`

The pure `break_is_match_at_end` was incorrectly routed to
`pending_break_mae` instead of `match_at_end`.

### Problem 2: Pure consuming tails deferred

`break_consuming_states` includes ALL consuming states reachable from
the break epsilon closure, regardless of whether they're behind deferred
assertions.  Bug 42 deferred ALL of them.

In this pattern, state 5 (`Byte('a')`) is reachable from the break path
through `Split(7) → Split(6) → Byte('a')(5)` — a pure path that does NOT
go through `\B`.  But it was added to `pending_break_tails` and gated
behind `\B` resolution.

When `resolve_deferred_for_pending` was called at mid-input, the `\B`
assertion sometimes passed, but the downstream walk from `\B`'s output
only reached `$ → Match` (which fails at mid-input, at_end=false).  The
pending tails were then discarded, even though state 5 was on a
completely independent pure path.

### Problem 3: Fast path skipped pending_resolved injection

The fast path in `chunk()` (no counting transition, no instances, no
tails) sets `match_at_end = cf_mae` and `continue`s, skipping the
`pending_resolved_mae` injection code that runs after `step_slow`.
When pending resolved results existed from the pre-step resolution,
they were silently dropped.

## Fix

Three-part fix, all in `src/dfa/tier3.rs`:

### Part A: Move pure flags out of the deferred gate

At all three counter-break deposit sites (tail→CInc handoff in
`step_slow_impl`, counter increment in `step_slow_impl`, and pre-step
tail→CInc in `chunk()`), `break_is_match` and `break_is_match_at_end`
are applied unconditionally — they're pure flags by definition.
`pending_break_mae` field was removed entirely since it's no longer set.

### Part B: Split consuming tails into pure and deferred

Added `break_consuming_pure` field to `Tier3OriginKind::Increment`:
consuming states reachable from the break path WITHOUT passing through
any non-End Assert state.  Modified `break_consuming_tails()` to track
`through_deferred` per path (like `break_closure()` already does).

At runtime, pure consuming tails are deposited immediately (to
`next_post_break_tails` or `pending_resolved_tails`), while
deferred-only tails go to `pending_break_tails` for assertion gating.

### Part C: Inject pending_resolved in fast path

Added `pending_resolved_mae` and `pending_resolved_tails` injection
after the fast path's state update, before `continue`.

## Investigation narrative

1. **Fuzz discovery:** `cargo +nightly fuzz run fuzz_differential` found
   crash artifact `crash-8d31896476b207826c68bc43f59d0698f2819809` with
   pattern `^.{1,11}((\B|a?))?$` on input `"cd yc 1a"` — Tier 3 said
   `false`, NFA said `true`.

2. **Initial reproduction:** Confirmed with `--debug --chunk-size 1
   --tier 3 --unroll-limit 0`.  Trace showed `pending_mae=true` and
   `deferred: [\B@4]` and `pending_tails: [5]` after every byte, but
   `mae=false` at the end.

3. **Pattern analysis:** `dump --dfa --unroll-limit 0` showed:
   - Counter break: `break_is_match_at_end: true` (pure path)
   - `break_deferred_asserts: [4]` (from `\B` sibling path)
   - `break_consuming_states: [5]` (from pure `Byte('a')` sibling path)
   These come from independent alternation branches.

4. **Root cause (Problem 1):** Identified that `break_is_match_at_end`
   was set from a pure path (`Split → $ → Match`) but deferred behind
   `\B` resolution.  Fix: apply pure flags unconditionally.

5. **Testing revealed Problem 3:** After fixing pure flags, input `"aa"`
   passed but `"aaaaaaaaaaaa"` (12 chars) failed.  Debug trace showed
   that at chunk #11, the fast path fired (no live instances, no tails)
   and overwrote `match_at_end` with `cf_mae` (false), discarding the
   `pending_resolved_mae` from pre-step resolution.

6. **Testing revealed Problem 2:** After fixing the fast path, the
   `"aaaaaaaaaaaa"` input still failed because state 5 (`Byte('a')`)
   — a pure consuming tail — was stuck in `pending_break_tails` and
   gated behind `\B`.  At mid-input, `resolve_deferred_for_pending`
   returned false because `\B`'s downstream path only reached `$`
   (which fails mid-input).  The tail was discarded even though it
   was on a pure path.

7. **Structural fix (Problem 2):** Modified `break_consuming_tails()`
   to track `through_deferred` per path and return `(all, pure)`.
   Added `break_consuming_pure` field.  Runtime deposits pure tails
   immediately and only defers deferred-only tails.

## What was hard

- The three problems manifested with different inputs: `"aa"` (Problem 1),
  `"aaaaaaaaaaaa"` (Problem 3, then Problem 2).  Each required a fresh
  debug trace to identify.

- The fast path interaction (Problem 3) was subtle: `pending_resolved_mae`
  was set correctly in the pre-step code, but the fast path's `continue`
  jumped over the injection point.  The fast path condition didn't account
  for pending resolved results.

- Problem 2 required understanding that `resolve_deferred_for_pending`
  walks from the deferred assertion's output — which in this case only
  reaches `$ → Match`.  At mid-input, `$` fails, so the function returns
  false even when `\B` itself passes.  The consuming tails on sibling
  paths are collateral damage.

## Tooling ideas

- **Trace pending_resolved fields:** The Display impl currently doesn't
  show `pending_resolved_mae` or `pending_resolved_tails` state.  Adding
  them would have revealed Problem 3 immediately.

- **Show fast-path vs slow-path in trace:** A `[fast]` / `[slow]`
  annotation on each byte's trace output would clarify which code path
  ran, making it obvious when the fast path skips logic.

- **Per-path deferred tracking in dump:** `dump --dfa` shows
  `break_consuming_states` but not which are pure vs deferred.  Showing
  `break_consuming_pure` separately would immediately reveal when pure
  tails exist alongside deferred assertions.
