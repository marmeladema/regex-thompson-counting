# Tier 2 Overlap Probe — Current Limitations

## What the probe admits

The binary-exactness probe (Patch 2) correctly admits:

1. **Alternation-body patterns** — patterns where overlapping counters are in
   different alternation branches and can never be co-active.  Example:
   `^(.{8,8}|f{5,26}|f{5,26})$`.  Because only one branch fires per match,
   the DFA states never contain consuming states from multiple counter bodies
   simultaneously.

2. **Disjoint-body patterns** — the fast path, no probe needed.

## What the probe rejects (conservatively)

The probe rejects **sequential overlapping-body patterns** like `^\w{300}\d{200}$`
where the counters have overlapping byte sets but are structurally sequential
(c0 must complete before c1 starts).

### Why the probe rejects these

Tier 2's `populate()` computes a `with_break` DFA successor that includes ALL
consuming states reachable when ANY counter breaks.  For `^\w{300}\d{200}$`,
the `with_break` closure includes both `\w` body state (from c0's continue path)
and `\d` body state (from c0's break → c1's entry).

This creates a DFA state containing both body consuming states.  On the next
byte, both counters' CInc nodes are structurally reachable.  The probe enumerates
break subsets for this transition:

- S = ∅: neither counter breaks.  mae = false.
- S = {c0}: c0 breaks (c1 entered).  mae = false (c1 hasn't reached its min).
- S = {c1}: c1 breaks.  mae = true (`$ → Match` reachable).
- S = {c0, c1}: both break.  mae = true.

The S≠∅ group has inconsistent `match_at_end` flags: {c0} has mae=false, {c1}
has mae=true.  The probe correctly identifies this as non-binary-exact: Tier 2's
single `with_break` state cannot represent both outcomes.

### Why the runtime is actually correct for these

At runtime, when the DFA state `[\w_body, \d_body]` is reached, c0's differential
counter has already broken — its instances have been cleared by `counter_reset`.
The CInc(c0) fires but `counter_increment(c0)` returns false (no live instances).
Only c1's increment matters.

The `with_break` DFA state is only selected when `any_can_break` is true, which
means c1 (the only counter with live instances) can break.  So the runtime
effectively computes subset {c1} (or {c0,c1} with c0 vacuous), which always
produces the correct mae=true result.

### What would be needed to admit these

A **counter-lifecycle analysis** that tracks which counters CAN have live
instances at each DFA state.  If the analysis proves that c0 cannot have live
instances in state `[\w_body, \d_body]` (because it only appears after c0
broke and was reset), then the probe could restrict subset enumeration to
subsets where only live-capable counters break.  This would eliminate the
{c0} subset (c0 can't break because it has no instances), making the remaining
subsets {∅, {c1}, {c0,c1}} all binary-exact.

Implementation outline:

1. For each DFA state in the probe's work queue, track a set of
   "possibly-live" counters (conservative over-approximation).
2. A counter becomes possibly-live when its CI is in the epsilon closure.
3. A counter becomes dead when `counter_reset` fires for it.
4. Restrict subset enumeration to subsets where only possibly-live counters
   break.

This is a meaningful extension but requires careful validation (the over-
approximation must not admit false positives).  It belongs to a follow-up
patch series, not to the first conservative refinement.

## The identical-body false accept

The probe incorrectly accepts `^.{0,200}.{0,200}$` where both counters have
identical bodies (`.`).  All subsets produce identical closures (symmetric),
so binary-exactness trivially passes.  But Tier 2's runtime overcounts:
both counters advance simultaneously, and `counter_reset` never fires (both
have body-interior progress at all times), causing an over-long match at
length 401 (should max at 400).

This is addressed by an additional check: if two counters have identical body
byte sets, reject.  This catches the symmetric case without complicating the
probe.
