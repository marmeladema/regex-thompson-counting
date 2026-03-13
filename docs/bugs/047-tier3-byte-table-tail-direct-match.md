# Bug 47: ByteTable post-break tail reaching Match directly

## Bug summary

- **Pattern**: `c{0,32}.{6,36}a?c?x`
- **Input**: `"aaaaaax"` (7 chars)
- **Expected**: `true` (NFA oracle, regex crate)
- **Actual**: `false` (Tier 3 false negative)
- **Affected tier**: Tier 3 (conditional transitions)

## Root cause

The static `target_is_match` array, computed at analysis time, tracks
whether a consuming state's epsilon closure reaches `Match` directly
(without `$`).  The computation walked epsilon paths from `Byte`,
`ByteCI`, and `ByteClass` states but **skipped `ByteTable` states**
(`_ => None` in the match).  This meant `target_is_match[8]` was always
`false` for state 8 (a ByteTable).

At runtime, when c1 (`.{6,36}`) broke with value >= 6, state 8 became a
post-break tail.  On byte 'x', state 8 (ByteTable) consumed 'x' and its
target was state 12 (Match).  The tail processing code in `step_slow_impl`
checked `self.analysis.target_is_match[pbo.idx()]` — which was `false`
for ByteTable state 8 — and never set `self.ever_matched`.

This is the same class of bug as Bug 37, where `target_is_match_at_end`
similarly skipped ByteTable states.  Bug 37 was fixed by adding a
byte-specific `is_match_at_end` flag to the `Advance` action.  The same
approach was needed for `is_match`.

## Investigation narrative

The crash artifact from `fuzz_match` identified the minimal pattern and
input.  Running `--debug --tier 3` showed `is_match=true` on the DFA
state but `matched=false` — meaning the DFA's static match flag was set
(from the with-break closure including Match) but `ever_matched` was
never set at runtime.

Checking `--debug --tier 0` (NFA) confirmed the match was legitimate:
counter c1 reached count 6, broke, and the post-break path through
state 8 (ByteTable) consumed 'x' reaching Match.

The `dump --dfa` output showed `target_is_match: [11]` — only state 11
(a `Byte('x')`) was in the list, not state 8 (the ByteTable).  Tracing
the `target_is_match` computation confirmed ByteTable was excluded.

## What was hard

Nothing particularly difficult.  This was a straightforward analogue of
Bug 37 (same root cause, same fix pattern).  The comment in the code
even predicted this bug: "a ByteTable tail whose byte-specific target
reaches Match directly without Assert would need a similar byte-specific
flag, **but no such pattern has been found yet**."

## Fix

1. Added `is_match: bool` field to `Tier3OriginKind::Advance` — computed
   per-target in `analyze_target()` with an epsilon walk that finds
   `Match` (mirroring the existing `is_match_at_end` walk but checking
   for direct `Match` instead of `$ -> Match`).

2. Updated the post-break tail processing in `step_slow_impl` to use the
   byte-specific `is_match` from the `Advance` action instead of the
   static `target_is_match` array.

3. Updated the pre-step tail resolution in `chunk()` similarly.

4. Updated `dump.rs` to display the new `is_match` field.

## Tooling ideas

- The `dump --dfa` output should highlight when `target_is_match` excludes
  ByteTable states, since this is a known gap that causes bugs.
