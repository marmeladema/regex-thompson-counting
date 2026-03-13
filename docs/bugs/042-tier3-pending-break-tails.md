# Bug 42 — Tier 3 false positive: unconditional tail deposit from counter break with deferred assertions

## Bug summary

- **Pattern:** `^a{8,8}((a{2,2}\b)?(a?a?)?)?$`
- **Input:** `"aaaaaaaaaaaa"` (12 a's)
- **Expected:** `false` (NFA oracle)
- **Actual (Tier 3):** `true` (false positive)
- **Affected tier:** Tier 3 (per-instance path), only with `unroll_limit=0`

## Root cause

When a counter breaks and its break path has deferred assertions (e.g.
`\b`), the Tier 3 matcher deposited the break path's `break_consuming_states`
(post-break tails) into `next_post_break_tails` **immediately**, before
the deferred assertion was resolved.  The deferred assertion is evaluated
on the NEXT byte, but the tails were already active and tracking state
independently.  If the assertion then failed, the tails persisted anyway.

In the bug pattern, counter c1 (`a{2,2}`) breaks after 2 a's with deferred
`\b` (word boundary).  The break path deposits tails at NFA states 9 and 11
(the `a?` consuming states in `(a?a?)?`).  These tails should only be
active if `\b` passes (i.e., the next byte is a non-word character).  But
with 12 a's, the byte after c1's break is another `a` → `\b` fails (word →
word).  Despite the failure, the tails persisted and consumed subsequent a's.

Each tail's advance through a byte triggered `check_tail_match_flags!`,
which set `match_at_end = true` because `target_is_match_at_end[9]` and
`target_is_match_at_end[11]` are statically true (these states lead to
`$ → Match` via epsilon transitions).  The spurious `match_at_end` flag
survived to `finish()` where it was returned as a match.

## Fix

Added `pending_break_tails` and `pending_break_mae` fields to
`Tier3DfaMatcher`.  When a counter break has non-empty
`break_deferred_asserts`:

1. **Deposit tails to pending list** instead of `next_post_break_tails`.
2. **Defer `break_is_match_at_end`** to `pending_break_mae` instead of
   setting `match_at_end` immediately.
3. **On the next byte** (in `chunk()`), check if any deferred assertion
   passes via `any_deferred_assert_passes()`.  If yes, promote pending
   tails to `post_break_tails` and set `match_at_end`.  If no, discard.
4. **At EOI** (in `finish()`), same evaluation with `at_end=true`.

The `any_deferred_assert_passes` method is a new helper that checks if
the assertion itself passes (using the DFA state's `prev_byte_representative`
and the next byte), without requiring Match to be epsilon-reachable — the
tails themselves will track matching via the existing `check_tail_match_flags`
mechanism.

Both deposit sites were updated: the normal counter-break path in the
`step_slow_impl` macro and the tail→CInc handoff path.

## Investigation narrative

1. **Fuzz discovery:** `cargo +nightly fuzz run fuzz_differential` found a
   crash artifact with pattern `^a{8,8}((a{2,2}\b)?(a?a?)?)?$` on input
   `"aaaaaaaaaaaa"` — Tier 3 said `true`, NFA said `false`.

2. **Debug trace:** `--debug --chunk-size 1 --tier 3 --unroll-limit 0`
   showed `mae` (match_at_end) set from byte 7 onward.  At byte 7, c0
   breaks with `break_is_match_at_end: true` (legitimate — the optional
   group can be skipped entirely).  At byte 9, c1 breaks depositing
   deferred `\b@7` and tails [9,11].

3. **Key observation:** After byte 9, `deferred: [\b@7]` appeared in the
   trace.  On byte 10 (a→a), `\b` should fail.  And indeed the deferred
   assert was cleared.  But `tails: [9,11]` persisted into bytes 10 and 11.

4. **Structural analysis:** Used `dump --dfa --unroll-limit 0` to see
   `target_is_match_at_end: [9, 11]` and c1's `break_deferred_asserts: [7]`,
   `break_consuming_states: [9, 11]`.  This confirmed that tails [9,11]
   from c1's break should be contingent on `\b` passing.

5. **Code audit:** Found that both counter-break deposit sites (normal
   and tail→CInc handoff) added tails to `next_post_break_tails`
   unconditionally, while `break_deferred_asserts` went to
   `verified_deferred_asserts` for later evaluation.  The disconnect
   between deferred assertion evaluation (next byte) and immediate tail
   deposit was the root cause.

## What was hard

- The `match_at_end` flag was set BOTH by c0's legitimate break (byte 7)
  and by c1's spurious tail advances (bytes 8-11).  It was initially
  unclear which source was causing the false positive — only by
  comparing NFA thread counts (which dropped to 1 at byte 10) against
  Tier 3's persistent `mae` flag did the divergence become clear.

- The `check_tail_match_flags!` macro sets `match_at_end` based on a
  static analysis (`target_is_match_at_end`) that doesn't account for
  runtime assertion resolution.  Understanding that the tails' validity
  was upstream (at deposit time) rather than at the flag-check site
  required tracing back through the break-deposit logic.

## Tooling ideas

- The `--debug` output could annotate tails with their provenance
  (which counter break deposited them) and whether they're pending
  on deferred assertions.  E.g. `tails: [9(c1,\b?), 11(c1,\b?)]`.

- A `--debug` mode that shows the `match_at_end` source (which code
  path set it) would help distinguish legitimate vs. spurious flags.
