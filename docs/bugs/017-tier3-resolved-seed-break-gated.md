# Bug 17: Tier 3 false negative — resolved seeds with reachable_without_break origins incorrectly excluded as break-gated

## Bug summary

- **Pattern:** `^\b(b{2,8})?.{9,9}$`
- **Input:** `"aaaaaaaaa"`
- **Expected:** MATCH (NFA agrees)
- **Actual (Tier 3):** NO MATCH
- **Affected tier:** Tier 3 (range-compressed path)

## Root cause

A prior fix (Bug 14) added a filter to exclude "break-gated" resolved
seeds — seeds that are only reachable through a counter's break path and
should not be unconditionally seeded.  The filter checked whether a
resolved seed appeared in `analysis.break_seeds` and, if so, excluded it
from the unconditional seed list.

The problem: this check was too aggressive.  In the pattern
`^\b(b{2,8})?.{9,9}$`, the `\b` deferred assertion at the start state
gates a Split that leads to two CounterInstance nodes — CI-0 for the
optional `(b{2,8})?` and CI-1 for the required `.{9,9}`.  CI-1's seed
(at the first body byte of `.{9,9}`) appears in `analysis.break_seeds`
because CInc-0's break path *also* reaches CI-1.  But CI-1's origin is
`reachable_without_break` — it's unconditionally reachable from the start
state regardless of whether any counter breaks.

The Bug 14 filter unconditionally excluded the seed, so `.{9,9}` was
never seeded on the first input byte, causing a false negative.

## Investigation narrative

The bug was found by the `fuzz_differential` target comparing Tier 3
against the NFA oracle.  The failing case was:
```
cargo run --release -- match --debug --chunk-size 1 --unroll-limit 0 \
    '^\b(b{2,8})?.{9,9}$' 'aaaaaaaaa'
```

The NFA showed threads advancing through `.{9,9}` from byte 0, while
Tier 3 never created any counter instances — the counter stayed empty on
every step.  This pointed to a seeding problem rather than a counter
increment or break issue.

Examining the `Tier3Analysis` (via `dump --dfa`) confirmed that the seed
for counter 1 appeared in `break_seeds` because of CInc-0's break path.
The fix was to add a `reachable_without_break` check: a seed is only
break-gated when its origin is NOT in `reachable_without_break`.

## What was hard

- **Seed classification is indirect.** The distinction between "break-gated
  seed" and "unconditionally reachable seed" depends on static analysis
  (`break_seeds`, `reachable_without_break`) computed at compile time.
  The `--debug` output at the time showed counter instance counts but not
  *why* seeding was skipped — you couldn't see which seeds were filtered.

- **Multi-counter interaction.** The bug only manifests when one counter's
  break path reaches another counter's entry point, which requires an
  optional counter (`?` or `*`) followed by a required counter.

## Tooling ideas

- **Seed tracing in `--debug`:** When a transition computes seeds, log
  which seeds were accepted and which were filtered (and why: "break-gated",
  "not in pre_seeds", etc.).  Would have immediately shown that the
  `.{9,9}` seed was being excluded.

- **`dump --dfa` seed detail:** Show the `break_seeds` list alongside
  `reachable_without_break` flags so you can visually spot seeds that are
  break-listed but also unconditionally reachable.
