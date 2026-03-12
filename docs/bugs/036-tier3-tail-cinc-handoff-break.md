# Bug 36: tail→CInc handoff break not setting any_can_break

## Bug summary

- **Pattern:** `^(.{1,3}.{0,0} )+$` (effectively `^(.{1,3} )+$`)
- **Input:** `"ax a "` (len=5)
- **Expected:** match (NFA=true)
- **Actual:** Tier 3 false negative (Tier 3=false)
- **Affected tier:** Tier 3 (per-instance path)
- **Artifact:** `fuzz/artifacts/fuzz_differential/crash-b912e6bbe888466e41d1135983f7e4d7bc555e37`

## Root cause

When a post_break_tail hits a CInc (counter increment) state via the
tail→CInc handoff in `step_slow_impl`, and the newly created instance
can immediately break (value after increment ≥ min), the handoff
correctly added `break_consuming_states` to `next_post_break_tails` and
set `break_is_match`/`break_is_match_at_end` flags.  However, it did
NOT set `any_can_break` or `counter_broke`, which are used later for:

1. **DFA state selection** (line 2742): `if t.is_counting && any_can_break`
   selects `with_break` (which includes break-gated NFA origins like
   state 4, the `Byte(' ')` after counter break).  Without this, the
   DFA selected `no_break`, which has a smaller NFA state set.

2. **Break seed gating** (line 2871): `counter_broke & (1u64 << trigger.idx())`
   gates break_seeds on the triggering counter.  Without `counter_broke`
   being set, break seeds don't fire.

The consequence: after the tail→CInc handoff creates a new
post_break_tail [4] (the `Byte(' ')` state), the DFA goes to the
`no_break` state (nfa={1}) which doesn't include origin 4.  On the next
byte, the tail's origin 4 is not found in the transition's `origin_keys`
(since origin 4 isn't in the DFA state), so the `None => {}` arm drops
the tail entirely.  The `+` loop's ability to continue matching is lost.

### Step-by-step trace (before fix)

NFA structure:
```
0: Assert(Start) → 3
1: ByteClass(cls:0) → 2     ← counter body (matches any byte)
2: CInc(c0, {1,3}) → cont:1 | break:4
3: CI(c0) → 1
4: Byte(' ') → 5
5: Split → 3 | 6            ← loops back to CI or exits
6: Assert(End) → 7
7: Match
```

Input `"ax a "`:

| Byte | Tier 3 state (before fix) | Notes |
|------|---------------------------|-------|
| `a` (0) | state=1 nfa={1,4} c0:[1@1] tails:[4] break_extras | c0 seeded at 1, can break → with_break state |
| `x` (1) | state=1 nfa={1,4} c0:[1@2] tails:[4] break_extras | c0 incremented to 2 |
| ` ` (2) | state=3 nfa={1,4} tails:[1,4] mae break_extras | c0 breaks at 3, tail[4] matches ' ', mae set |
| `a` (3) | state=0 nfa={1} c0:[1@1] tails:[4] **no break_extras** | **BUG:** tail[1]→CInc handoff broke but any_can_break=false, DFA goes to no_break |
| ` ` (4) | state=1 nfa={1,4} c0:[1@2] tails:[4] **no mae** | tail[4]'s origin was NOT in state 0's origin_keys at step 3 → dropped. No mae. **FALSE NEGATIVE** |

After fix, step 3 goes to state=1 (with_break, nfa={1,4}) instead of
state=0, keeping origin 4 in the DFA state so tail [4] survives.

## Investigation narrative

1. Reproduced with `cargo run --release -- match --debug --chunk-size 1
   --tier 3 --unroll-limit 0 '^(.{1,3}.{0,0} )+$' 'ax a '`.

2. Compared Tier 3 trace with NFA trace (`--tier 0`).  Both agree
   through chunks 0-2.  At chunk 3 ('a'), the NFA retains states
   {2,6,0} (active counter threads + potential match paths), while
   Tier 3 drops to state=0 nfa={1} — losing the break-gated origins.

3. Examined the `dump --dfa` output: `reachable_without_break: [1]`,
   `target_is_match_at_end: [4]`, `break_consuming_states: [4]`.
   Confirmed state 4 is only reachable via counter break.

4. Traced through `step_slow_impl` for chunk 3: the tail→CInc handoff
   at origin 1 creates a break (value 0+1=1 ≥ min=1), adds [4] to
   `next_post_break_tails`, but `any_can_break` is only set by the
   counter entry loop which has 0 entries at this step.

5. Confirmed: `any_can_break=false` → `self.current = t.no_break` →
   state 0 (nfa={1}).  At chunk 4, tail [4]'s origin 4 is not in
   state 0's transitions → dropped in `None => {}`.

## What was hard

The bug required understanding the interaction between three mechanisms:
post_break_tails (which carry consuming states across steps), the
tail→CInc handoff (which converts tails into counter instances), and
DFA state selection (which determines which NFA origins are available in
subsequent transitions).  The tail→CInc handoff was correctly computing
break conditions and adding tails, but the DFA state selection was blind
to it because `any_can_break` was only set by the counter entry loop.

The debug trace was helpful but didn't directly show why `any_can_break`
was false — that required reading the source to understand that the
variable was only set in the counter loop, not in the tail handoff.

## Tooling ideas

- The `--debug` output could include the value of `any_can_break` and
  `counter_broke` after each step, making it visible when the DFA state
  selection diverges from what the tail mechanism expects.
- A diagnostic mode that shows "tail [4] dropped: origin not in
  transition" would make the proximate cause immediately visible.

## Fix

Moved the declarations of `any_can_break` and `counter_broke` above the
tail loop so that the tail→CInc Increment arm can set them when
`pbt_value + 1 >= min`.  This ensures the DFA selects `with_break`
(including break-gated origins) and break seeds fire when a tail handoff
produces a counter break.
