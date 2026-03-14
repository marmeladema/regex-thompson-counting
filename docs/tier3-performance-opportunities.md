# Tier 3 Performance Opportunities

## Purpose

This document reviews the current Tier 3 implementation from a performance
perspective now that the typed-effects migration is largely complete and the
correctness picture is much clearer.

It focuses on three questions:

1. What obvious constant-factor wins are available now?
2. What deeper restructurings could unlock materially larger speedups?
3. Which work should happen first?

This document intentionally does **not** treat end-of-input unification as a
near-term performance prerequisite.  That work touches shared DFA machinery and
is currently deferred for architectural reasons.

## Scope

Reviewed files:

- `src/dfa/tier3.rs`
- `src/dfa/tier3_effects.rs`
- `src/dfa/mod.rs`
- `src/lib.rs`
- `benches/pathological.rs`
- `benches/pathological_profile.rs`

Commands run during this review:

- `cargo bench --bench pathological -- "rethoc/tier3"`
- `cargo bench --bench pathological -- "regex/131072"`
- `cargo bench --bench pathological -- "rethoc/tier4/4096"`
- `cargo run --release -- info --unroll-limit 0 '.{0,1000}.{0,1000}.{0,1000}a'`
- `cargo bench --bench pathological_profile`

## Executive Summary

Tier 3 is already doing the most important algorithmic job correctly: it avoids
the catastrophic blowups that hit both naive simulation and backtracking-style
engines on bounded-repetition pathologies.

The main performance story is now about **constant factors** and
**work-avoidance**, not about rescuing the core asymptotic behavior.

The two most important takeaways are:

1. **The biggest immediate win is better prefiltering for Tier 3.**
   The current pathological benchmark pattern, `.{0,1000}.{0,1000}.{0,1000}a`,
   reports `Prefilter: none`, so Tier 3 simulates every byte even in the all-no-
   match case.  The most practical prefilter work is still streaming-safe
   start-side filtering and re-engagement, not suffix-side filtering.

2. **The biggest hot-loop structural win is replacing origin searches with
   direct indexing.**
   The current step loop repeatedly searches `origin_keys` linearly and dedups
   tails with `Vec::contains`.  That keeps the implementation simple, but it is
   an obvious tax on every byte.

If the goal is a practical optimization sequence, I would prioritize:

1. Tier 3 prefilter re-engagement
2. Better start-byte / nullable-prefix prefilter analysis
3. Tail dedup bitsets / scratch reuse / tiny hot-loop cleanups
4. Direct origin lookup instead of `position()`
5. Re-benchmark
6. Then choose between:
   - a denser rangeable Tier 3 execution kernel, or
   - narrower prefilter specializations for known single-chunk / non-streaming
     use cases

## Current Baseline

### Pathological benchmark context

The existing pathological benchmark uses:

- pattern: `.{0,1000}.{0,1000}.{0,1000}a`
- 3 counters
- Tier 3 range-compressed execution by default
- no deferred assertions
- no prefilter

Evidence:

- benchmark definition: `benches/pathological.rs:3-10`,
  `benches/pathological.rs:30`
- regex info output for the pattern shows:
  - 14 NFA states
  - 3 counters
  - Tier 3 range-compressed
  - `Prefilter: none`

### Measured throughput on the current tree

From `cargo bench --bench pathological -- "rethoc/tier3"`:

- no-match, 64 KiB: about **22.0-22.4 MiB/s**
- no-match, 128 KiB: about **22.3-22.4 MiB/s**
- match-at-end, 64 KiB: about **22.2-22.4 MiB/s**
- match-at-end, 128 KiB: about **22.2-22.3 MiB/s**

Interpretation:

- no-match and match-at-end have very similar throughput
- `finish()` is not the dominant cost on this benchmark
- the steady-state per-byte Tier 3 loop is the main cost center

### Comparison with other engines on the same benchmark

From the targeted Criterion runs:

- `regex` no-match, 128 KiB: about **90 GiB/s**
- `regex` match-at-end, 128 KiB: about **35 KiB/s**
- `rethoc/tier4` no-match, 4 KiB: about **3.85 KiB/s**
- `rethoc/tier4` match-at-end, 4 KiB: about **3.6-4.0 KiB/s**

Interpretation:

- Tier 3 is **orders of magnitude better** than the catastrophic engines on
  the hard match-at-end case
- but Tier 3 is still **orders of magnitude slower** than a good literal-
  prefiltered engine on the all-no-match case

That is exactly the signature of a system whose asymptotics are good but whose
work-avoidance and hot-loop structure still have headroom.

## What Looks Expensive Right Now

## 1. Repeated origin lookup in the step loop

Relevant code:

- tail stepping: `src/dfa/tier3.rs:2977-2979`
- counter-entry stepping: `src/dfa/tier3.rs:3119-3123`
- transition payload: `src/dfa/tier3.rs:1040-1047`

Current shape:

- transitions store parallel arrays `origin_keys` and `origin_targets`
- the runtime searches `origin_keys.iter().position(...)` to find the matching
  target for each active tail or counter entry

Why this matters:

- this work happens in the steady-state per-byte path
- the cost is paid once per active tail and once per active counter entry
- it scales with both live activity and transition origin count

This is the clearest hot-loop constant-factor tax in the current code.

## 2. Tail dedup still uses `Vec::contains`

Relevant code:

- `next_post_break_tails.contains(...)` in `src/dfa/tier3.rs:2984-2985` and
  `src/dfa/tier3.rs:3041-3042`
- `resolved_tails.contains(...)` in `src/dfa/tier3.rs:3573-3575`,
  `src/dfa/tier3.rs:3631-3633`, `src/dfa/tier3.rs:3674-3675`,
  `src/dfa/tier3.rs:3792-3818`
- `actions.tails.contains(...)` in `src/dfa/tier3_effects.rs:919-922`

Current shape:

- tail sets are represented as `Vec<StateIdx>`
- dedup is done with linear membership checks before every push

Why this matters:

- this is conceptually set behavior implemented with repeated linear scans
- it shows up in both the main step path and deferred-effect resolution path
- it is simple, but it is not cheap when tails are numerous or repeatedly
  reintroduced

## 3. Counter storage still does linear dedup/merge

Relevant code:

- per-instance dedup: `src/dfa/tier3.rs:1289-1299`
- per-instance seed/advance/continue: `src/dfa/tier3.rs:1306-1345`
- range merge: `src/dfa/tier3.rs:1523-1547`

Current shape:

- `InstanceCounters` scans linearly to dedup `(value, origin)` pairs
- `RangeCounters` scans linearly to find an existing origin slot to merge into

Why this matters:

- on the common rangeable path, origin counts are usually small, so this is not
  the worst offender
- on the fallback per-instance path, this can become much more expensive when
  counters fan out or are heavily gated

This is more important as a medium-term structural issue than as a tiny cleanup.

## 4. Tier 3 still lacks useful prefiltering on exactly the patterns where it
needs it most

Relevant code:

- prefilter derivation: `src/lib.rs:2782-2852`
- start closure derivation: `src/lib.rs:2854-2892`
- Tier 3 chunk entry prefilter use: `src/dfa/tier3.rs:3478-3507`
- Tier 1/2 re-engaging prefilter loops:
  - `src/dfa/tier1.rs:581-625`
  - `src/dfa/tier2.rs:1691-1726`

Current limitation:

- the only built-in prefilter is derived from the immediate start closure
- `compute_start_closure()` gives up when it sees counters or assertions
- Tier 3 only uses the prefilter once at chunk entry and then permanently drops
  it

Why this matters:

- the pathological benchmark pattern has an obviously discriminative literal,
  `a`, but the current prefilter framework cannot use it because it only knows
  about start-side consuming bytes
- the no-match benchmark haystack is all `x`, so Tier 3 needlessly simulates
  every byte
- `regex` wins the no-match case by enormous margins because it effectively
  reduces the work to fast scanning rather than full simulation

This is the single highest-upside work-avoidance gap in the current design.

## Addendum: prefiltering in a streaming engine

Tier 3 is a streaming matcher, so prefilter opportunities need to be evaluated
under the rule that the engine may receive input across arbitrarily many
`chunk()` calls.

That strongly affects which prefilter ideas are natural and which are not.

### Start-side prefilters fit streaming naturally

Examples:

- current `Prefilter::{Memchr1, Memchr2, Memchr3, Range}` derived from the
  start closure
- re-engaging those filters when the matcher returns to a quiescent start-like
  state
- deriving a stronger start-byte set through simple nullable prefixes, counted
  prefixes, or assertion-tolerant start analysis

Why these fit:

- they are only used when there is no in-flight match state that skipped bytes
  could affect
- once the matcher returns to a safe idle configuration, non-candidate bytes can
  be skipped without losing cross-chunk information

This is why Tier 1 and Tier 2 can safely memchr ahead when back at start, and
why the same style of optimization is attractive for Tier 3.

### End-side or candidate-end prefilters are much less natural

The tempting idea for patterns like `.{0,1000}.{0,1000}.{0,1000}a` is to say:

- "matches end on `a`, so just scan for `a`"

That is not a generally safe streaming interpretation.

Example:

- pattern: `.{0,5}a`
- chunk 1: `xxxxx`
- chunk 2: `a`

If the matcher skipped chunk 1 just because it contained no `a`, it would lose
the partial-match state needed for the cross-chunk match ending in chunk 2.

So a general candidate-end filter is not just "another memchr prefilter" in a
streaming engine.  It only becomes viable with additional machinery such as:

- bounded replay with a rolling suffix window, or
- bulk state summaries for long runs of non-candidate bytes

Both ideas are deeper execution specializations, not near-term prefilter
extensions.

### Practical prefilter opportunities for this engine

The best streaming-friendly prefilter directions are:

1. **Tier 3 prefilter re-engagement** when the matcher returns to a quiescent
   start-like state
2. **Better start-byte derivation** through counters, nullable prefixes, and
   simple assertions instead of giving up as soon as `compute_start_closure()`
   sees them
3. **Literal-island / bounded-prefix style filtering** where a required early
   literal can be used without violating streaming semantics

More speculative directions still exist, but they should be framed as runtime
specializations rather than generic prefilters:

- candidate-end filtering for known single-chunk or non-streaming use cases
- bulk run skipping for long spans of structurally uninteresting bytes

## 5. Deferred-effect resolution still recomputes assertion/reachability work

Relevant code:

- `resolve_pending()`: `src/dfa/tier3_effects.rs:857-924`
- `eval_assert_chain()`: `src/dfa/tier3_effects.rs:789-840`
- `can_reach_match_mid()`: `src/dfa/tier3.rs:3829-3845`
- `can_reach_match_at_end()`: `src/dfa/mod.rs:161-207`

Current shape:

- each pending effect evaluates its guard independently
- match-carrying effects also walk downstream reachability from the assertion
  continuation

Why this matters:

- on assertion-heavy patterns, many effects may share the same chain or the same
  downstream continuation
- the current code re-evaluates those paths effect-by-effect

This is not the main bottleneck on the benchmarked pathological pattern, but it
is likely important on effect-heavy Tier 3 workloads.

## Obvious Wins

These are improvements that look attractive without first redesigning Tier 3.

## 1. Re-engage the prefilter in Tier 3 when the matcher becomes quiescent

Why it is attractive:

- Tier 1, Tier 2, and the NFA already know how to do this
- the control-flow pattern already exists in the codebase
- the Tier 3 chunk loop already tracks enough state to decide whether it is
  safe: no live instances, no tails, no pending effects, clean start-like state

What to change:

- add a Tier-3-specific equivalent of `chunk_prefilter()` from Tier 1/2
- re-enable skipping when the matcher is back in a start-like quiescent state

Expected payoff:

- large on literal-starting Tier 3 workloads
- smaller or none on patterns whose first-byte set is still broad

Risk:

- low-to-medium; the main challenge is getting the re-engagement condition right
  under contamination/tails/pending effects

## 2. Replace tail dedup scans with a matcher-local bitmap or epoch array

What to change:

- maintain `seen_tails: Vec<u32>` or a bitset keyed by NFA state index
- bump an epoch per byte or per resolution pass
- membership becomes O(1) instead of `Vec::contains`

Expected payoff:

- modest but reliable
- especially helpful on patterns that accumulate many tails or repeatedly
  recycle them

Risk:

- low

## 3. Finish the remaining tiny hot-loop cleanups

Examples:

- compute `let byte_is_word = is_word_byte(byte);` once per byte and reuse it
- make `EffectAtom` `Copy` if possible and stop cloning it in enqueue paths
- reuse `resolved_tails` / related scratch vectors instead of recreating them
  per byte

Expected payoff:

- small individually
- worthwhile because the Tier 3 steady-state loop is already down to constant-
  factor work

Risk:

- low

## Medium-Risk Refactors

These are not giant redesigns, but they affect core data flow enough that they
should be treated as deliberate performance work.

## 1. Replace `origin_keys.iter().position(...)` with a direct transition-local
lookup structure

Short-term version:

- replace the parallel arrays with a compact array of `(origin, target)` pairs
- sort by origin and use binary search

Better version:

- add a transition-local origin -> target-index map
- runtime no longer searches linearly for every active entry

Expected payoff:

- meaningful on all Tier 3 patterns
- especially important on patterns with many body origins or many active
  instances/tails

Risk:

- medium; touches transition population and the main step loop

## 2. Replace `ReachScratch` clear/reset with epoch stamping

Current issue:

- reachability walks currently prepare/reset visited state repeatedly

Change:

- use a generation counter and `visited_gen: Vec<u32>` instead of clearing
  boolean arrays

Expected payoff:

- medium on deferred-assert-heavy patterns
- low on the pathological benchmark, which has no assertions

Risk:

- medium-low

## 3. Split effect application into typed fields instead of per-atom enum
dispatch

Current shape:

- `immediate` and `on_break` are `Box<[EffectAtom]>`
- runtime loops over them and matches on the atom enum every time

Possible shape:

```rust
struct AppliedEffects {
    set_match: bool,
    set_match_at_end: bool,
    tails: Box<[StateIdx]>,
    seeds: Box<[(CounterIdx, StateIdx, u32)]>,
}
```

Expected payoff:

- moderate
- mostly reduces branching and makes hot paths simpler to specialize

Risk:

- medium; more compile-time lowering logic

## Deeper Restructurings With the Biggest Upside

These are the changes most likely to unlock substantial speedups, but they are
architectural enough that they should be treated as design work, not just
cleanup.

## 1. Dense-slot execution for rangeable Tier 3

This is the most promising structural optimization.

Idea:

- assign each counter-body consuming origin a dense local slot at compile time
- transitions operate on slot IDs rather than searching by `StateIdx`
- range counters become dense slot arrays instead of sparse origin-search
  structures

What this buys:

- no `origin_keys.iter().position(...)`
- no `RangeCounters` linear search by raw origin state
- more compact and predictable memory access in the hot loop
- better cache locality

Why this is attractive now:

- the typed-effects migration already clarified the target-step/effect boundary
- the current rangeable path is structurally clean enough to support a denser
  execution kernel

Expected payoff:

- high on the common range-compressed Tier 3 path

Risk:

- medium-high; compile-time analysis must assign and carry slot IDs cleanly

## 2. Effect-free or effect-light Tier 3 specialization

Observation:

- the pathological benchmark pattern has no deferred assertions and no real
  effect-heavy behavior, yet it still pays the generic Tier 3 machinery cost

Idea:

- detect at compile time when a Tier 3 regex has:
  - no guarded effects
  - no deferred assertion resolution
  - no post-break tails
- route it through a stripped-down executor that only handles local step logic,
  break decisions, and direct match flags

Why this matters:

- many practical bounded-repetition patterns are structurally simple even if
  they are not Tier 1 or Tier 2 eligible

Expected payoff:

- high on a meaningful subset of Tier 3 workloads

Risk:

- medium-high; adds another execution specialization to maintain

## 3. Stronger candidate-end or mandatory-byte prefilters

This is no longer the highest-priority general direction for Tier 3.

It is still potentially valuable, but it should be treated as a **specialized**
direction rather than the obvious next prefilter extension.

Current limitation:

- the engine only derives start-byte prefilters from `start_closure`
- it does not derive later mandatory bytes or candidate-end filters

Why the pathological benchmark is a perfect example:

- pattern: `.{0,1000}.{0,1000}.{0,1000}a`
- the only really discriminative fact is that a match must end on `a`
- current Tier 3 still simulates every byte, because it cannot exploit that

Potential directions:

### Mandatory-byte prefilter

- derive one or more bytes that must occur somewhere in a match
- use memchr-style scanning to rule out negative chunks quickly

This helps negative cases the most, but it is still trickier than start-side
filtering in a streaming matcher because "byte absent in this chunk" does not
by itself rule out a cross-chunk match.

### Candidate-end prefilter

- derive bytes that can end a match
- scan for those bytes and only run the expensive Tier 3 machinery around those
  candidate endpoints
- for bounded-length patterns, retain only the necessary suffix window

This has large upside for patterns like the pathological benchmark, but it is
best understood as a specialization for:

- known single-chunk matching,
- bounded replay with explicit suffix buffering, or
- other non-generic execution modes

rather than as the natural next step for the streaming engine as a whole.

Expected payoff:

- potentially enormous on the right workload, especially known single-chunk or
  sparse-match cases

Risk:

- high; needs careful streaming/window semantics and likely regex-wide length
  bound information, or else a deliberately restricted execution mode

## 4. Remove the runtime clean-chain tax by encoding more provenance in cached
state or transition data

Current cost center:

- contaminated multi-counter states maintain `clean_nb`, `clean_nb_cf_mae`, and
  `clean_nb_is_match`
- that means extra transition tracking and extra filtering work in the step loop

Idea:

- move more of that distinction into cached state/transition structure instead
  of maintaining a runtime shadow path

Expected payoff:

- meaningful on complex multi-counter patterns

Risk:

- high; this is semantically delicate and should come only after lower-risk
  wins have been exhausted

## On `pathological_profile`

The callgrind-based `benches/pathological_profile.rs` benchmark is still useful,
but in this environment the gungraun summary output was comparison-oriented and
did not produce obviously actionable raw hotspot attribution on its own.

My recommendation is:

- keep using `pathological_profile` for regression tracking
- but for hotspot work, inspect the raw callgrind artifacts directly or add a
  workflow that reports the top functions more transparently

The Criterion benchmark results and the current code structure already give a
clear enough first-pass performance agenda even without that deeper callgrind
read.

## Recommended Order Of Work

### First wave: cheap wins with strong upside

1. Re-engage Tier 3 prefiltering when the matcher returns to a safe quiescent
   state
2. Improve start-byte prefilter derivation through simple counters / nullable
   prefixes / assertion-tolerant start analysis
3. Replace tail dedup `Vec::contains` with an epoch bitmap / seen array
4. Apply tiny hot-loop cleanups (`EffectAtom: Copy`, precompute `byte_is_word`,
   scratch vector reuse)

### Second wave: hot-loop structure

5. Replace origin linear search with a direct transition-local lookup
6. Re-benchmark on `benches/pathological.rs`

### Third wave: choose the bigger bet based on workload goals

If the target workload is mostly scan-heavy / negative cases:

7. first pursue stronger **streaming-safe** prefilters and only then consider
   narrower suffix-side specializations

If the target workload is mostly heavy bounded-repetition simulation:

7. pursue dense-slot rangeable Tier 3 execution

### Fourth wave: only if needed

8. specialized effect-light Tier 3 executor
9. contamination/clean-chain restructuring
10. candidate-end / suffix-window specializations for known single-chunk or
    otherwise restricted execution modes

## Bottom Line

The obvious performance story is not that Tier 3 is algorithmically broken.
It is not.  The obvious story is:

- Tier 3 is already solving the hard asymptotic problem
- the remaining gap is mostly constant-factor cost and work-avoidance

The biggest near-term win is **streaming-safe prefiltering**.

The biggest structural runtime win is **removing origin searches through direct
indexing**.

The most practical prefilter path is:

- re-engage filters when Tier 3 returns to a quiescent start-like state, and
- derive better start-side filters through more permissive start analysis

Suffix-side filtering still has potential, but it should be viewed as a later
specialization for restricted execution modes rather than as the next obvious
step for the general streaming engine.

If the streaming-safe prefilter work and direct-indexing work both land, Tier 3
should retain its current correctness and pathology resistance while moving much
closer to the throughput expected from a high-performance scanning engine.
