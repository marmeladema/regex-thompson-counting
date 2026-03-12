# Bug 28: Tier 3 break seed deferred assertions evaluated at wrong position

## Bug summary

- **Pattern:** `^f{3,4}\b.{5,10}a?$`
- **Input:** `fff abcde` (and other inputs where `\b` sits between word/non-word)
- **Expected:** MATCH (NFA matches: `fff` matches `f{3,4}`, `\b` passes at
  `f`/` ` boundary, ` abcde` matches `.{5,10}`, `a?` matches empty)
- **Actual:** NO MATCH (Tier 3 false negative)
- **Affected tier:** Tier 3, when a break seed has `deferred_asserts` (added in
  Bug 27) and the assertion depends on the *next* byte after the break position

## Root cause

Bug 27 added `deferred_asserts` to `Tier3BreakSeed` and evaluated them at break
time in `step_slow_impl!`.  The evaluation used:
- `prev = state.prev_byte_representative()` — the TO state's representative,
  which reflects the byte that just entered that state
- `next = Some(byte)` — the current byte being processed

For the pattern `^f{3,4}\b.{5,10}a?$`, counter c0 is `f{3,4}` and c1 is
`.{5,10}`, with `\b` between c0's break output and c1's CI.  When c0 breaks on
byte `f` (consuming its 3rd `f`), the break seed fires for c1 through `\b`.

At break time, the current byte is `f` (the byte that caused c0's CInc to
fire).  Bug 27's code evaluated `\b(prev=f, next=f)` — which is **always
false** because both sides are word characters.  The assertion should be
evaluated at the *boundary between the break byte and the next input byte*.
But the next input byte isn't available yet at break time.

The result: `\b`-gated break seeds **never** fire when the break byte is a word
character (or equivalently, `\B`-gated seeds never fire when the break byte is
a non-word character).

Bug 26's `verified_deferred_asserts` mechanism correctly resolves DFA-state
deferred asserts at the start of the *next* byte.  But break seeds are a
separate mechanism — they produce counter instances, not DFA transitions.  The
resolved instances from `verified_deferred_asserts` were classified as
"break-gated" (because `!analysis.reachable_without_break[origin]`) and only
appeared in `t.break_seeds`, requiring c0 to break again on the *next* byte.
But c0 may have no live instances left, so the seed was permanently lost.

## Fix

Added a `pending_break_seeds` buffer to `Tier3DfaMatcher` that defers assertion
evaluation to the next byte, mirroring how `verified_deferred_asserts` works
for DFA-state assertions (Bug 26):

1. **`pending_break_seeds: Vec<(CounterIdx, CounterIdx, StateIdx, u32, bool)>`**
   on `Tier3DfaMatcher`.  Each entry stores `(trigger, counter, origin, value,
   prev_was_word)` — everything needed to evaluate the assertion later.

2. **In `step_slow_impl!`**: break seeds with **non-empty** `deferred_asserts`
   no longer evaluate immediately.  Instead, they push an entry to
   `pending_break_seeds` with `prev_was_word = is_word_byte(byte)` (the byte
   at the break position becomes `prev` for the assertion).  Break seeds with
   **empty** `deferred_asserts` still fire immediately (no behavioral change).

3. **In `chunk()` at start of each byte**: iterate `pending_break_seeds`,
   evaluate each seed's deferred asserts with `prev=stored_prev_was_word` and
   `next=current_byte`.  If all assertions pass, seed the counter.  Clear the
   buffer after processing.

4. **In `finish()`**: iterate `pending_break_seeds` with `at_end=true` and
   `next=None`.  If the assertion passes **and** the counter can immediately
   break (value ≥ min) with a match-producing break path, set `match_at_end`.
   This handles the edge case where the deferred break seed is the last thing
   before end-of-input.

## Investigation narrative

1. **Fuzz discovery.** `cargo +nightly fuzz run fuzz_differential` produced
   crash artifact `crash-6e9747575cde07f49820cb579634cbcb39928fa1`.  Decoding
   the seed revealed pattern `^f{3,4}\b.{5,10}a?$`.

2. **Brute-force input testing.** Tested inputs of various lengths.  Found that
   `fff abcde` (NFA=MATCH) produced NO MATCH on Tier 3.  The `\b` between
   `f{3,4}` and `.{5,10}` was the critical path.

3. **Debug trace (`--tier 3 --chunk-size 1`).** Confirmed that when c0 broke
   on the 3rd `f`, the break seed for c1 through `\b` was evaluated but
   failed.  The trace showed `\b` being tested with both sides being `f`
   (word, word) → false.

4. **Identified the timing problem.** The assertion `\b` sits at the boundary
   *after* c0's break byte.  At break time, we know `prev` (the break byte)
   but not `next` (the next input byte).  Bug 27's code used the current byte
   for both sides.

5. **Checked Bug 26 analogy.** Bug 26 solved the same timing problem for
   DFA-state deferred asserts by deferring resolution to the next byte in
   `chunk()`.  Applied the same pattern to break seeds.

6. **Verified `finish()` edge case.** Patterns where the deferred seed must
   resolve at end-of-input (e.g. `^.{2,4}\b$` with input `ab`) need special
   handling in `finish()`.  Added assertion evaluation with `at_end=true` and
   immediate break check.

## What was hard

- **Two layers of deferral.** Bug 27 introduced deferred asserts on break
  seeds.  Bug 28 revealed that those asserts must *themselves* be deferred to
  the next byte.  This creates a chain: break event → store pending seed →
  next byte arrives → evaluate assertion → seed counter.  Reasoning about the
  correct `prev`/`next` values at each stage required careful tracking of
  which byte was "current" at each point.

- **`finish()` edge case.** The pending break seed might never get a "next
  byte" if the break happens on the last input byte.  This required separate
  logic in `finish()` to evaluate with `at_end=true` and check whether the
  newly seeded counter can immediately produce a match via its break path.

- **Break-gated classification blocking resolved seeds.** Even when Bug 26's
  `verified_deferred_asserts` correctly resolved the DFA-state assertion, the
  resulting seed was classified as break-gated (because the origin was not
  epsilon-reachable without a break).  This meant it only appeared in
  `t.break_seeds`, requiring another break from c0 — which had no instances
  left.  The fix bypasses this by operating at the counter level (pending
  break seeds) rather than the DFA-state level.

## Tooling ideas

- **Trace pending break seeds in `--debug` output.** Currently the debug trace
  shows break seed evaluation but not the deferred pending mechanism.  Adding
  `[pending_break_seed] stored (trigger=c0, counter=c1, origin=6, val=0,
  prev_word=true)` and `[pending_break_seed] resolved with next='  ' → \b
  passes → seed c1` lines would make the deferral chain visible.

- **Show assertion evaluation details in `step_slow_impl!` trace.** When a
  break seed's assertion fails, print the actual `prev`/`next` values used
  and the expected assertion type.  This would have immediately revealed that
  `\b(f, f)` was being tested instead of `\b(f, ' ')`.
