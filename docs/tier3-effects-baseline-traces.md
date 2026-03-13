# Tier 3 Effects Migration — Baseline Debug Traces

Captured before Phase 0 of the typed effects implementation.
These traces serve as reference points during the migration.

## Pattern 1: Contamination (`a{2,4}$` on `"aaa"`)

Shows `break_extras=true` after value reaches min, `cf_mae=false` (break-gated MAE).

```
[after chunk #0 "a"] DFA[T3] state=0 nfa={0} break_extras=false live=true
  c0: 1 range(s) [origin:0=0-1]
[after chunk #1 "a"] DFA[T3] state=1 nfa={0} mae=true cf_mae=false break_extras=true
  c0: 1 range(s) [origin:0=0-2]
[after chunk #2 "a"] DFA[T3] state=1 nfa={0} mae=true cf_mae=false break_extras=true
  c0: 1 range(s) [origin:0=0-3]
MATCH
```

## Pattern 2: Break seed timing with deferred assertion (`\b[a-z]{2,3}\b` on `"abc "`)

Shows `deferred: [\b@4]` after value reaches min. Assertion resolved on next byte.

```
[after chunk #0 "a"] state=1 nfa={1} break_extras=false live=true
  c0: 1 inst [1@1]
[after chunk #1 "b"] state=2 nfa={1} break_extras=true live=true
  c0: 1 inst [1@2]
  deferred: [\b@4]
[after chunk #2 "c"] state=2 nfa={1} break_extras=true live=false
  c0: 0 inst []
  deferred: [\b@4]
[after chunk #3 " "] state=0 nfa={} matched=true live=true
  c0: 1 inst [1@0]
MATCH
```

## Pattern 3: Pending break tails (`(a|bb){2,3}c` on `"abbc"`)

Shows `tails: [6]` after `bb` group completes, consumed by `c`.

```
[after chunk #0 "a"] state=0 nfa={0,1} break_extras=false live=true
  c0: 2 range(s) [origin:0=0-1, origin:1=0-1]
[after chunk #1 "b"] state=2 nfa={0,1,2} break_extras=false live=true
  c0: 3 range(s) [origin:2=0-1, origin:0=0-0, origin:1=0-0]
[after chunk #2 "b"] state=3 nfa={0,1,2,6} break_extras=true live=true
  c0: 3 range(s) [origin:0=0-2, origin:1=0-2, origin:2=0-0]
  tails: [6]
[after chunk #3 "c"] state=4 nfa={0,1} is_match=true matched=true
  c0: 2 range(s) [origin:0=0-0, origin:1=0-0]
MATCH
```

## Pattern 4: ByteTable-specific behavior (`[ab]{2,3}c` on `"abc"`)

Shows break tails from ByteTable: `tails: [3]` after second byte.

```
[after chunk #0 "a"] state=0 nfa={0} break_extras=false live=true
  c0: 1 range(s) [origin:0=0-1]
[after chunk #1 "b"] state=1 nfa={0,3} break_extras=true live=true
  c0: 1 range(s) [origin:0=0-2]
  tails: [3]
[after chunk #2 "c"] state=2 nfa={0} is_match=true matched=true
  c0: 1 range(s) [origin:0=0-0]
MATCH
```

## Pattern 5: Break seed with assertion chain (`\b\w{2,4}\b` on `"ab d"`)

Shows deferred `\b` resolved on non-word byte, then re-seeding.

```
[after chunk #0 "a"] state=1 nfa={1} break_extras=false live=true
  c0: 1 inst [1@1]
[after chunk #1 "b"] state=2 nfa={1} break_extras=true live=true
  c0: 1 inst [1@2]
  deferred: [\b@4]
[after chunk #2 " "] state=0 nfa={} matched=true live=true
  c0: 1 inst [1@0]
[after chunk #3 "d"] state=0 nfa={} matched=true live=true
  c0: 1 inst [1@0]
MATCH
```
