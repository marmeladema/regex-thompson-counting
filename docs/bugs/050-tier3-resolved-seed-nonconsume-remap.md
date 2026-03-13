# Bug 50: Resolved seed remapped to non-consuming target

## Bug Summary

- **Pattern**: `^(\b(0*a{2,8}){7,7})?$`
- **Input**: `"00aa00aa00aa00aa00aa00aa00aa"` (28 chars, 7 groups of `00aa`)
- **Expected**: `true` (NFA matches)
- **Actual**: `false` (Tier 3 false negative)
- **Affected tier**: Tier 3 (per-instance path)
- **Fuzz artifact**: `fuzz/artifacts/fuzz_differential/crash-ba6b1f7ce6d7c3fd455563bfa52e75a5049f2826`

## Root Cause

The pattern's `\b` assertion is on the start DFA state as a deferred
assertion.  When processing the first byte ('0'), `populate()` resolves
the deferred `\b` assertion, runs an epsilon closure from the states
behind it, and collects resolved seeds.  The resolved seed for c0 is at
origin StateIdx(2) (`Byte('0')`), the counter body's first consuming
state.

Phase 1 of `populate()` also consumes the resolved closure's NFA states
against the current byte.  State 2 (`Byte('0')`) consumes '0' and
produces target state 3 (`Split → 2 | 4`, the `0*` loop entry).  The
Bug 16 remapping logic then remaps the seed's origin from 2 → 3 so the
instance appears at the "post-consumption" position.

**The problem**: State 3 is a `Split` node, not a consuming state.
Consuming states are the only ones that appear in a transition's
`origin_keys` array (line 1571-1576).  When the instance at origin 3
was processed on the next step, `origin_keys.iter().position(|&k| k ==
origin)` returned `None`, the action was `None`, and the instance was
silently dropped.  All counter instances died after one step.

The Bug 16 remapping was designed for L>1 counter bodies where the
post-consumption target is the second byte of the body (a consuming
state).  For L=1 bodies preceded by a non-consuming loop (`0*`), the
target is a Split node back to the loop entry, which is not consuming.

The bug had TWO manifestation sites: the resolved seed merge in the
counting path (line ~1747) and the non-counting path (line ~1842).
The first byte transition was non-counting because the probe closure
from target 3 (Split → Byte('0'), Byte('a')) did not encounter a CInc
via epsilon transitions alone.  The fix was applied to both paths.

## Investigation Narrative

1. Reproduced: Tier 3 false negative on `"00aa"*7`.  NFA matches,
   Tier 3 doesn't.  Simpler all-'a' inputs (14+) worked correctly.

2. Debug trace showed: after chunk 0 ('0'), c0 had `[3@0, 4@0]` —
   origin 3 (Split) instead of expected origin 2 (Byte('0')).  After
   chunk 1 ('0'), c0 had 0 instances.  Instances died immediately.

3. Traced the seed origin: `ci_origins[3] = [2, 4]` was correct.
   The `start_seeds` were at origins 2 and 4.  But the display showed
   origin 3.  This meant remapping occurred somewhere.

4. Found the Bug 16 remap at line 1738-1741:
   `resolved_body_targets.find(from == s.1).map_or(s.1, |(_, to)| to)`.
   `resolved_body_targets = [(2, 3)]` — origin 2 consumed '0', target
   3.  Seed remapped from 2 → 3.

5. Initially only fixed the counting path (line 1747).  Bug persisted.
   Re-examined and found `is_counting = false` for this transition
   (probe closure from Split→consuming didn't encounter CInc).  The
   non-counting path at line 1842 had the same unconditional remap.
   Fixed both paths.

## Fix

In both the counting and non-counting seed merge paths, added a filter
to the Bug 16 remapping that only applies when the target is a consuming
state (Byte, ByteCI, ByteClass, or ByteTable):

```rust
let remapped_origin = resolved_body_targets
    .iter()
    .find(|&&(from, _)| from == s.1)
    .filter(|&&(_, to)| {
        matches!(
            regex.states[to],
            State::Byte { .. }
                | State::ByteCI { .. }
                | State::ByteClass { .. }
                | State::ByteTable { .. }
        )
    })
    .map_or(s.1, |&(_, to)| to);
```

When the target is non-consuming (Split, Assert, CI, CInc, Match), the
seed keeps its original consuming origin.

## What Was Hard

- **Two code paths for the same logic**: The resolved seed merge exists
  in both the counting and non-counting branches of `populate()`.  The
  counting path (line ~1735) had extensive comments about Bug 14, Bug 16,
  Bug 17, and Bug 40 filters.  The non-counting path (line ~1841) was a
  simpler version with fewer guards.  Fixing only one path left the bug
  in the other.

- **Determining is_counting**: The transition from the start state on
  byte '0' is non-counting because the probe epsilon closure (from
  targets + regex.start) doesn't encounter CInc.  The CInc is deep in
  the body (after `a{2,8}`), unreachable via epsilon only.  This was
  counter-intuitive — the pattern obviously has a counter, but the
  specific DFA transition is classified as non-counting.

- **Confusing display output**: The debug trace showed `3@0` which
  means origin=StateIdx(3), value=0.  This was the key clue, but
  initially it was unclear whether this was a seeding bug or a display
  bug.  Understanding the Instance struct's Display format
  (`origin@value`) was essential.

## Tooling Ideas

- **Seed origin validation**: A debug_assert in `seed()` or `push()`
  that checks the origin is a consuming state would have caught this
  immediately.  Non-consuming origins should never be counter instance
  origins.

- **Transition dump in debug trace**: Showing the transition's
  `origin_keys`, `seeds`, and `pre_seeds` inline in the debug output
  for each step would make it immediately visible what seeds are being
  applied and with what origins.
