# Bug 52: Break-seed dedup discards ungated OR-alternative path

## Bug summary

- **Pattern:** `^a{1,2}(\b)?.{4,16}$`
- **Input:** `"aaaaa"` (5 bytes)
- **Expected:** MATCH (`a` + skip `(\b)?` + `aaaa`)
- **Actual:** NO MATCH (counter 1 never seeded)
- **Affected tier(s):** Tier 3 (per-instance path, `--unroll-limit 0`)

## Root cause

When counter 0 breaks, the epsilon closure walks the break path.  The
pattern `(\b)?` compiles to:

```
CInc(c0) break → Split(5) → Assert(\b, 4) → CI(c1, 8)
                           → CI(c1, 8)
```

The Split creates two paths to the same CI(c1):
1. Through `Assert(\b)` → gated seed with `deferred_asserts = [\b@4]`
2. Direct → ungated seed with `deferred_asserts = []`

Two bugs combined to lose the ungated seed:

**Bug A — `ci_visited` suppresses second arrival at CI node.**
The epsilon walk in `compute_tier3_analysis()` Step 4 uses a `ci_visited`
boolean array to prevent infinite loops.  When state 8 (CI) was first
reached via the Assert path (LIFO stack order), it was marked visited.
The second arrival via the direct path was skipped entirely.  The CI
only recorded the gated seed; the ungated seed was never generated.

**Bug B — `dedup_by` compared only the key, not the assertions.**
Even if both seeds had been recorded, the dedup at line ~532 compared
only `(trigger, counter, origin)` and discarded the second entry.  The
comment claimed "a seed's assertion path is uniquely determined by these
three keys" — this is false when alternation (Split) in the epsilon
closure creates multiple paths to the same CI.

## Investigation narrative

1. The `test_fuzz_oracle` property test flagged the pattern
   `^f{6,47}((y|\b|a|\B))?x+{4,16}d?a?((a?a?)?(a?a?)?)?$` with many
   inputs.  All failures were false negatives (ours=false, oracle=true).

2. Reproduced with the CLI:
   ```
   cargo run --release -- match --unroll-limit 0 '^f{6,47}((y|\b|a|\B))?x+{4,16}d?a?...$' 'fff...'
   ```
   Confirmed: default tier (with unrolling) MATCH, NFA MATCH,
   `--unroll-limit 0` NO MATCH.

3. Minimized by testing simpler patterns:
   - `^a{1,2}.{4,16}$` → MATCH (no optional group)
   - `^a{1,2}(.)?.{4,16}$` → MATCH (optional consuming, no assert)
   - `^a{1,2}(a)?.{4,16}$` → MATCH (optional literal, no assert)
   - `^a{1,2}(\b)?.{4,16}$` → **NO MATCH** (optional assert)
   - `^a{1,2}(a|\b)?.{4,16}$` → **NO MATCH** (alternation with assert)

   The trigger is `\b` (or any assertion) inside the optional group
   between two counters.

4. Debug trace (`--debug --chunk-size 1 --unroll-limit 0`) showed:
   - After byte 0: c0=[1@1], pending `AddSeed(c1, origin:6, val=0)`
     gated by chain#0 (\b)
   - After byte 1: c0=0, c1=0, pending still gated by chain#0
   - Bytes 2–4: c1 never seeded, no instances

   The gated seed couldn't fire because `\b` between two word chars
   ('a','a') fails.  The ungated seed (from skipping `(\b)?`) was
   completely absent.

5. `dump --dfa --unroll-limit 0` confirmed: only one break_seed
   (gated), and `target_effects` for state 2 had only a `guarded`
   AddSeed, no `on_break` AddSeed.

6. Traced through the epsilon walk code (Step 4 in
   `compute_tier3_analysis()`): the `ci_visited` check on the LIFO
   stack prevented the second visit to CI(c1).  The Split pushed
   Assert(4) first, then CI(8).  LIFO processed Assert first → reached
   CI(8) → marked visited.  Direct path to CI(8) → already visited →
   skipped.

## What was hard

The pattern minimization was straightforward, and the `--debug` trace
immediately showed the missing ungated seed.  The hardest part was
tracing through the epsilon walk's LIFO ordering to understand why one
path was favored over the other — the stack push order determines which
path reaches the CI first.

## Tooling ideas

- The `dump --dfa` output could show **all raw break_seeds before dedup**
  (or at least a count), making it visible when the dedup discards
  entries with different assertion sets.

## Fix

Two changes in `compute_tier3_analysis()` Step 4:

1. **CI nodes always record seeds** regardless of `ci_visited`.  The
   walk still uses `ci_visited` to prevent re-exploring the downstream
   graph past a CI (which is path-independent), but every arrival at a
   CI — even a re-visit — records its seeds with the arrival path's
   deferred assertions.

2. **Dedup compares assertions, not just the key.**  Changed
   `dedup_by(|a, b| a.0 == b.0 && a.1 == b.1 && a.2 == b.2)` to also
   compare `a.3 == b.3` (the sorted/deduped assertion list).  This
   preserves distinct OR-alternative paths: ungated seeds go to
   `on_break`, gated seeds go to `guarded`.
