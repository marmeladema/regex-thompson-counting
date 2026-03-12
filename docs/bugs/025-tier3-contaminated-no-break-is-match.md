# Bug 25: Tier 3 contaminated no\_break\_is\_match false positive

## Bug summary

- **Pattern:** `^.{0,46}x{10,35}f`
- **Input:** `xxxxxxxxxf` (9 x's + f)
- **Expected:** NO MATCH (c1 `x{10,35}` only reached count 9, below min 10)
- **Actual:** MATCH (Tier 3 false positive)
- **Affected tier:** Tier 3 (per-instance path), only when contaminated
  (`break_extras && num_counters > 1`)

## Root cause

The pattern has two counters: c0 (`.{0,46}`) and c1 (`x{10,35}`).  When c0
breaks, the DFA takes the `with_break` successor, which includes NFA state 8
(`Byte('f')`) from c1's break closure.  This state is **break-gated**: it
should only contribute to matching when c1 actually breaks (count >= 10).

When byte `f` arrives, state 8 consumes it and reaches `Match` (state 9).
The DFA transition's `no_break_is_match` flag is set to `true` because
state 8 → Match is in the no-break epsilon closure.  But `no_break_is_match`
doesn't distinguish between counter-free origins and break-gated origins.

In the counting branch of `step_slow_impl!`, the code used:
```rust
let m = if from_contaminated && !any_can_break {
    self.clean_nb_is_match
} else {
    t.no_break_is_match  // contaminated flag!
};
```

When `any_can_break = true` (c0 broke on this byte), the condition
`from_contaminated && !any_can_break` was false, so the contaminated
`t.no_break_is_match` was used.  But c0 breaking doesn't validate c1's
break-gated origins — c1 never reached its minimum.

## Fix

Two changes:

1. **In `step_slow_impl!` (counting branch):** When `from_contaminated`, always
   use `self.clean_nb_is_match` (the clean chain's transition flag) regardless
   of `any_can_break`.  Per-instance break checks already set `ever_matched`
   when an actual counter breaks with `break_is_match=true`, so no signal is
   lost.

2. **`target_is_match` computation:** The initial implementation followed all
   `Assert` states (including `Assert(End)`), making `target_is_match[i]` true
   whenever `Match` was reachable through `Assert(End)`.  This caused 344 test
   failures because it overlapped with `target_is_match_at_end`.  Fixed to NOT
   follow any `Assert` states — `target_is_match` means `Match` is reachable
   unconditionally (no assertion gating).

Supporting infrastructure added:

- **`clean_nb_is_match: bool`** field on `Tier3DfaMatcher`: caches the clean
  chain's transition's `no_break_is_match`.  Updated whenever `clean_nb` is
  advanced (same places as `clean_nb_cf_mae` from Bug 24).

- **`target_is_match: Box<[bool]>`** field on `Tier3Analysis`: precomputed
  per-consuming-state flag, true when consuming a byte leads to `Match` via
  epsilon transitions without going through any `Assert`.  Used in the
  post-break tail tracker's dead-target branch to detect direct matches
  (distinct from `target_is_match_at_end` which handles `$ -> Match`).

## Investigation narrative

1. **Fuzz discovery.** `cargo +nightly fuzz run fuzz_differential` found a
   crash artifact (`crash-80804a2c59883680533c3c522d97d9695bc9d9d5`).

2. **Decode the seed.** Used the `decode_seed` example to extract the pattern
   `^.{0,46}x{10,35}f` and failing input `xxxxxxxxxf`.

3. **Reproduce with `--debug`.** NFA (tier 0) returned NO MATCH; Tier 3
   returned MATCH.  The byte-by-byte trace showed that after 9 x's, c1 had
   instances from `5@0` through `5@8` (max count 9).  When `f` arrived, the
   trace showed `matched=true is_match` — the false positive.

4. **Identified the contamination path.** Used `dump --dfa` to see that state 8
   (`Byte('f') -> Match`) was in the DFA state via c1's break closure
   (`break_extras`).  The transition's `no_break_is_match` was true because
   state 8 consumed `f` and reached `Match`.

5. **Traced the decision logic.** In `step_slow_impl!`, the counting branch
   used `t.no_break_is_match` when `any_can_break` was true (c0 broke).  The
   `from_contaminated && !any_can_break` guard only protected the case where
   no counter broke — but c0 breaking didn't validate c1's origins.

6. **First fix attempt (incomplete).** Changed to `from_contaminated &&
   !any_can_break` → `from_contaminated`.  Also added `target_is_match`
   for the post-break tail tracker.  Initial `target_is_match` computation
   followed `Assert` states, causing 344 memory assertion failures (the new
   `Box<[bool]>` field increased `Tier3Analysis` size).  After `bless_memory.py`
   updated memory values, all 344 failures turned out to be actual test failures
   from `target_is_match` being too permissive (it followed `Assert(End) ->
   Match`, overlapping with `target_is_match_at_end`).

7. **Fixed `target_is_match` to skip Assert states.** The epsilon walk now
   stops at any `Assert` node.  `target_is_match` is true only when `Match`
   is reachable unconditionally.  Re-ran `bless_memory.py` to update the 344
   stale memory values; all tests passed.

## What was hard

- **The `any_can_break` red herring.** The original guard
  `from_contaminated && !any_can_break` looked like it was protecting against
  contamination, but only for the no-break case.  The mental model "if a
  counter broke, the break path is validated" is wrong when there are multiple
  counters — counter A breaking doesn't validate counter B's break-gated
  origins.

- **Two bugs in one fix.** The `target_is_match` computation had its own bug
  (following `Assert(End)`), which manifested as 344 test failures that
  initially looked like the main fix was wrong.  It took a second investigation
  to realize the memory assertion failures were from the new field, and the
  actual test failures after blessing were from the Assert traversal.

## Tooling ideas

- **Show `clean_nb_is_match` in Display output.** Currently `--debug` shows
  `cf_mae` but not the clean chain's `is_match` flag.  Adding it would make
  it immediately visible when the contaminated flag diverges from the clean
  chain's flag.

- **Per-origin `is_match` attribution in transitions.** The transition's
  `no_break_is_match` is a single bool that merges Match reachability from
  all origins.  A per-origin breakdown (like `nb_counter_free_mae` for mae)
  would make contamination bugs easier to spot: you'd see which origin
  contributed the Match signal and whether it was break-gated.
