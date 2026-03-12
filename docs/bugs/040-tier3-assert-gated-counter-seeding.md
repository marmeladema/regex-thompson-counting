# Bug 40 — Assertion-gated counter seeding false positive

## Bug summary

- **Pattern:** `^((.1?\B.{3,28}){3,3}|(a?a?)?)$`
- **Input:** `"aaaaaaaaaa"` (10 a's)
- **Expected:** NO MATCH (NFA and `regex` crate agree)
- **Actual (Tier 3):** MATCH (false positive)
- **Affected tier:** Tier 3 (conditional DFA, per-instance path)
- **Artifact:** `fuzz/artifacts/fuzz_match/crash-e57cddb4258f44ab2ef7b576a9dc0d9cb90af3c2`
- **Source fuzzer:** `fuzz_match` (oracle = `regex` crate)

## Root cause

The `{3,3}` outer repetition is unrolled into three counter copies (c0, c1,
c2 for `.{3,28}`).  Each repetition copy has the NFA structure:

```
ByteClass → Split(optional '1') → Assert(\B) → CI(counter) → body → CInc
```

Two independent mechanisms contributed to the false positive:

### Problem 1: `analyze_target` follows assertion states unconditionally

The `analyze_target()` function performs a DFS from a target state to find
consuming states (advance origins) and check `is_match_at_end`.  Its
`State::Assert` arm followed all assertion kinds unconditionally:

```rust
State::Assert { out, .. } => stack.push(out),
```

This meant consuming states *behind* a `\B` assertion were included in
`advance_origins`.  The tail mechanism used these origins to seed downstream
counters (c1 from c0's break, c2 from c1's break) without any assertion
gating.  The seeded counters counted freely regardless of whether `\B`
would actually pass.

### Problem 2: `finish()` accepted `match_at_end` without EOI assertion check

When counter c2 reached value 3 (≥ min 3) and `break_is_match_at_end`
fired, it set `self.match_at_end = true`.  In `finish()`, this flag was
accepted unconditionally:

```rust
if self.match_at_end {
    return true;
}
```

The `\B` assertion passes mid-input (between two word characters — no word
boundary), so counter seeding appeared valid during matching.  But at
end-of-input, `\B` *fails* (there IS a word boundary after the last 'a').
The `match_at_end` check ran before `resolve_verified_deferred_asserts`,
so the match was accepted without verifying that `\B` holds at EOI.

### Combined effect

c0 counted through the body, broke, and the tail mechanism seeded c1
through the assertion-gated path (Problem 1).  c1 counted and broke,
seeding c2 the same way.  c2 reached value 3 and set `match_at_end`
(via `break_is_match_at_end`).  `finish()` returned true without checking
whether `\B` passes at EOI (Problem 2).

## Fix

Two complementary changes:

### Fix 1: `analyze_target` — block non-End Assert states

Changed the `State::Assert` arm to only follow `AssertKind::End` (`$`):

```rust
State::Assert { kind, out } => {
    if kind == AssertKind::End {
        stack.push(out);
    }
}
```

Non-End assertions (`\b`, `\B`) are deferred assertions resolved at DFA
transition time.  Consuming states behind them are already handled by the
DFA's epsilon closure and deferred assertion resolution.  Including them
in `advance_origins` caused the tail mechanism to bypass assertion checking.

`$` (End) is still followed because it leads to `Match` (handled by
`advance_is_match_at_end`) and any consuming states after `$` should still
be discovered.

### Fix 2: `finish()` — re-evaluate deferred asserts at EOI

When `match_at_end` is true and `verified_deferred_asserts` is non-empty,
re-evaluate each verified deferred assertion with EOI context (`is_end=true`,
`next_byte=None`) before accepting:

```rust
if self.match_at_end {
    if self.verified_deferred_asserts.is_empty() {
        return true;
    }
    let any_pass = self.verified_deferred_asserts.iter().any(|&assert_idx| {
        if let State::Assert { kind, out } = self.regex.states[assert_idx] {
            kind.eval(false, true, prev, None) == AssertEval::Pass
                && DfaState::can_reach_match_at_end(out, prev, self.regex)
        } else {
            false
        }
    });
    if any_pass {
        return true;
    }
}
```

This is a safety net: even if Problem 1's fix prevents most assertion-gated
seeding, the EOI check catches any remaining cases where `match_at_end` was
set through an assertion-gated path that doesn't hold at end-of-input.

When `verified_deferred_asserts` is empty, `match_at_end` is accepted
unconditionally (no assertion-gated paths involved).

## Investigation narrative

1. The `fuzz_match` fuzzer found the crash artifact.  Decoded the seed to
   get pattern `^((.1?\B.{3,28}){3,3}|(a?a?)?)$` and input `"aaaaaaaaaa"`.

2. Confirmed the disagreement:
   ```
   cargo run --release -- match --tier 0 '<pattern>' 'aaaaaaaaaa'  → NO MATCH
   cargo run --release -- match --tier 3 '<pattern>' 'aaaaaaaaaa'  → MATCH (wrong)
   ```

3. Used `--debug --chunk-size 1` to trace byte-by-byte.  The Tier 3 trace
   showed counters c0/c1/c2 all reaching their minimum value and
   `match_at_end` being set.  The NFA trace showed no active states
   surviving past the first few bytes because `\B` fails at relevant
   positions.

4. Examined the `analyze_target` code and found the unconditional
   `State::Assert { out, .. } => stack.push(out)` — this included
   consuming states behind `\B` in advance origins.

5. Examined `finish()` and found `match_at_end` was checked before any
   deferred assertion verification.

6. Applied both fixes and verified the pattern returns NO MATCH on all
   tiers.

## What was hard

- The pattern has complex structure: outer `{3,3}` unrolled into three
  counter copies, each with an optional literal `1?`, a `\B` assertion,
  and a variable-length `.{3,28}` body.  Plus an alternation with `(a?a?)?`.
  Understanding which NFA states correspond to which counters required
  careful examination of the `dump` output.

- The `\B` assertion is context-dependent: it passes mid-input (between
  word chars) but fails at EOI.  This makes the bug position-dependent —
  the counter seeding appears "correct" during matching but produces a
  wrong result at the boundary.

- Two independent problems needed fixing.  Fix 1 alone might have been
  sufficient for this specific pattern, but Fix 2 provides a necessary
  safety net for other patterns where assertion-gated states enter the
  DFA through paths not controlled by `analyze_target`.

## Tooling ideas

- The `--debug` trace could show which deferred assertions are currently
  verified/pending and their pass/fail status at each byte position.
  This would make it easier to see when an assertion is "contaminating"
  downstream counter seeding.

- A `dump --dfa` mode showing the `advance_origins` for each tail entry
  would help identify when assertion-gated states are incorrectly included
  in seeding paths.

- The `finish()` method's decision tree is complex (match_at_end,
  deferred asserts, counter-free paths).  A `--debug` flag that shows
  which branch of `finish()` fired and why would speed up EOI-related
  bug investigations.
