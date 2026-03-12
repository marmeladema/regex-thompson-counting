# Bug 18: Tier 2 false negative — missing break-path consuming states in with_break DFA successor

## Bug summary

- **Pattern:** `^(a\B){4,4}a$`
- **Input:** `"aaaaa"`
- **Expected:** MATCH (NFA agrees)
- **Actual (Tier 2):** NO MATCH
- **Affected tier:** Tier 2 (differential counters, L=1 body)

## Root cause

When a deferred assertion (here `\B`) sits inside an L=1 counter body
and gates the path to `CounterIncrement`, the Phase 1 probe closure can
resolve the assertion and reach CInc.  But the code that built the
`with_break` DFA successor only looked for *epsilon-reachable Match
states* from CInc's break target (`out1`).  It never collected
**consuming states** on the break path.

In `^(a\B){4,4}a$`, after the counter reaches 4 and breaks, the break
path leads to the consuming state for the final `a` before `$`.  Since
this consuming state was never injected into the `with_break` closure,
the DFA had no way to continue matching after the counter broke — it
effectively dead-ended.

Two sub-issues compounded this:

1. The `counter_break_can_match[ci]` precomputed flag only checks whether
   Match is epsilon-reachable from `out1`.  It was being used to gate the
   *entire* break-path walk, not just the epsilon-to-Match check.  When
   the break path had consuming states but no direct epsilon path to
   Match, the gate prevented any break-path processing.

2. There was no mechanism to collect consuming states from break paths
   and inject their byte-consumed targets into the `with_break` closure.

## Investigation narrative

Found by `fuzz_differential` comparing Tier 2 against NFA.  Running:
```
cargo run --release -- match --debug --chunk-size 1 --unroll-limit 0 \
    '^(a\B){4,4}a$' 'aaaaa'
```

The NFA showed 5 threads processing all 5 `a` bytes successfully.  Tier 2
showed the counter incrementing correctly through bytes 0-3 (oldest=4),
but after byte 3 (where the counter should break), the DFA transitioned
to a state with an empty NFA set — it had lost all threads.

The key insight was comparing `--tier 0` (NFA) output against `--tier 2`
output byte by byte.  After the break byte, NFA had a thread at the
consuming state for the final `a`, but Tier 2 had nothing.

The fix introduced a `break_path_consuming_states()` helper that walks
CInc's `out1` through epsilon states and collects all consuming states.
These are consumed against the current byte and their targets injected
into the `with_break` closure exclusively.

## What was hard

- **Two-phase interaction.** The bug requires a deferred assertion inside
  the counter body (Phase 1 resolution) AND a consuming state on the break
  path (Phase 2 successor computation).  This combination is rare —
  assertions inside counter bodies with L=1 are already an edge case, and
  having consuming states after the counter break (rather than just `$`
  or Match) adds another layer.

- **Gate masking the real problem.** The `counter_break_can_match` gate
  made it look like the engine correctly determined "no match possible from
  break", when in reality it was only checking epsilon-to-Match and
  ignoring consuming-state continuations.

## Tooling ideas

- **Break-path detail in `dump --dfa`:** Show what's reachable from each
  CInc's `out1` — both epsilon-to-Match and consuming states.  Would have
  immediately revealed that the consuming `a` state was present but not
  being collected.

- **Phase annotation in `--debug`:** Mark which phase (probe, Phase 1
  resolution, Phase 2 successor) contributed each NFA state to the DFA
  successor.  Would show that the `with_break` successor was missing states
  that the probe or Phase 1 should have contributed.
