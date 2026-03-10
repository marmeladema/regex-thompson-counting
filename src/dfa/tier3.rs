//! Tier 3: DFA with conditional transitions for non-nested counters.
//!
//! Unlike Tier 4 (which unions both CInc paths into one DFA successor and
//! replays compiled counter programs at runtime), Tier 3 precomputes **two
//! DFA successor states** per counting transition: one that follows only
//! the continue path (`NoBreak`), and one that follows both paths
//! (`WithBreak`).  At runtime, the matcher evaluates counter values to
//! pick the correct successor.
//!
//! **Eligibility**: patterns with bounded repetitions where no counter is
//! nested inside another counter's body.  Body can be any length/structure.
//!
//! # Build-time analysis
//!
//! The [`Tier3Analysis`] struct, built once at regex compile time by
//! [`compute_tier3_analysis`], precomputes three tables that eliminate
//! per-transition DFS walks during DFA population:
//!
//! - **Per-target actions** (`targets`): for each NFA state that can be
//!   reached after consuming a byte, the structural action (advance or
//!   increment) is determined once and cached.
//! - **CI-output origins** (`ci_origins`): for each CounterInstance output,
//!   the consuming states reachable through epsilon transitions.
//! - **Break seeds** (`break_seeds`): global table of counter seeds that
//!   are gated on a specific counter's break path being taken.
//!
//! # Terminology: "origin"
//!
//! An **origin** is the NFA consuming state (`Byte`, `ByteCI`, `ByteClass`,
//! or `ByteTable`) where a counter instance is currently parked, waiting to
//! consume the next input byte.
//!
//! When an instance is first created at a CI (`CounterInstance`) node, its
//! origin is the first consuming state in the counter body.  As input
//! bytes are consumed, the instance's origin advances through the body's
//! consuming states.  When the body's last consuming state is consumed,
//! the epsilon closure reaches CInc and the counter value increments.
//!
//! Origins bridge the DFA (which tracks NFA state subsets) and per-counter
//! instance tracking (which tracks repetition counts).  A cached DFA
//! transition maps each origin to an [`OriginAction`] that describes what
//! happens structurally when a byte is consumed there.

use std::fmt;

use crate::{
    AssertKind, CounterIdx, Prefilter, Regex, State, StateIdx, byte_match_ci, is_word_byte,
};

use super::{DfaCache, DfaMemory, DfaStateId};

// ---------------------------------------------------------------------------
// Precomputed NFA analysis (built once at regex compile time)
// ---------------------------------------------------------------------------

/// Precomputed NFA analysis for Tier 3 patterns.
///
/// Built once at regex compile time by [`compute_tier3_analysis`],
/// so that DFA population can look up structural actions directly
/// instead of running DFS walks per transition.
///
/// **Indexed by target state**: each entry in [`targets`] corresponds to
/// a post-consumption NFA state (the state reached *after* a byte is
/// consumed).  For `Byte`/`ByteCI`/`ByteClass` origins, the target is
/// the fixed `.out` field.  For `ByteTable` origins, each distinct
/// non-NONE table entry is a target.  This unifies all consuming-state
/// types under a single lookup.
#[derive(Clone, Debug)]
pub(crate) struct Tier3Analysis {
    /// Per-target-state precomputed origin action.
    ///
    /// Indexed by NFA state index.  `targets[idx]` is `Some(kind)` when
    /// `idx` is a known post-consumption target; `None` for states that
    /// are never reached as byte-consumption targets.
    pub(crate) targets: Box<[Option<Tier3OriginKind>]>,

    /// Per-CI-output consuming states.
    ///
    /// `ci_origins[idx]` lists the consuming NFA states reachable from
    /// state `idx` through epsilon transitions.  Used to resolve CI
    /// seed pairs at DFA populate time.  Empty for non-CI-output states.
    pub(crate) ci_origins: Box<[Box<[StateIdx]>]>,

    /// Global break-triggered seeds.
    ///
    /// Each entry records a seed that is only applied when a specific
    /// counter breaks: the seed for `counter` at `origin` is applied
    /// when `trigger`'s CInc break path is taken.
    pub(crate) break_seeds: Box<[Tier3BreakSeed]>,

    /// Maximum number of distinct consuming NFA states in any single
    /// counter body.  Used as the stride for flat range-compressed
    /// instance storage in [`Tier3DfaMatcher`].  Zero when there are
    /// no counters.
    pub(crate) max_body_origins: usize,

    /// Per-consuming-state flag: true if the NFA state's byte-consumption
    /// target reaches `$ → Match` through epsilon transitions.
    ///
    /// Indexed by NFA state index.  Only meaningful for consuming states;
    /// `false` for all others.  Used by the post-break tail tracker to
    /// detect match-at-end when a tail's action is `Dead` (its target
    /// has no further consuming states but may have `$ → Match`).
    pub(crate) target_is_match_at_end: Box<[bool]>,
}

/// What happens structurally when a byte is consumed at a given target.
///
/// Mirrors [`OriginAction`] but without the `Dead` variant (dead targets
/// are represented as `None` in `Tier3Analysis::targets`).
#[derive(Clone, Debug)]
pub(crate) enum Tier3OriginKind {
    /// Epsilon closure from the target did NOT reach CInc.
    /// The instance keeps its counter value and moves to `new_origins`.
    Advance { new_origins: Box<[StateIdx]> },
    /// Epsilon closure from the target reached CInc.  Counter is
    /// incremented; min/max determine continue vs. break.
    Increment {
        /// Consuming states reachable WITHOUT going through CInc.
        advance_origins: Box<[StateIdx]>,
        min: u32,
        max: u32,
        /// Consuming states on the CInc continue path.
        continue_origins: Box<[StateIdx]>,
        /// True if the break path reaches `Match` directly.
        break_is_match: bool,
        /// True if the break path reaches `Match` through `$`.
        break_is_match_at_end: bool,
        /// Non-counter consuming states on the CInc break path.
        /// These form the "post-counter tail" that must be tracked
        /// at runtime to detect `$ → Match` after additional bytes.
        break_consuming_states: Box<[StateIdx]>,
    },
}

/// A break-triggered seed: applied only when `trigger` counter breaks.
#[derive(Clone, Debug)]
pub(crate) struct Tier3BreakSeed {
    /// The counter whose CInc break path leads to this seed.
    pub(crate) trigger: CounterIdx,
    /// The counter that gets a new instance.
    pub(crate) counter: CounterIdx,
    /// The consuming NFA state where the new instance starts.
    pub(crate) origin: StateIdx,
}

/// Build the [`Tier3Analysis`] for a tier-3-eligible pattern.
///
/// Walks the NFA state array to precompute:
/// 1. Per-target-state origin actions (what happens when a byte is consumed).
/// 2. Per-CI-output consuming states (for resolving CI seed pairs).
/// 3. Global break-triggered seeds (for counter break-path seeding).
/// 4. Max body origins per counter (stride for flat instance storage).
///
/// Uses only pure DFS — no DFA cache or matcher state is needed.
pub(crate) fn compute_tier3_analysis(
    states: &[State],
    byte_tables: &[crate::ByteMap],
    state_can_reach_match: &[bool],
) -> Tier3Analysis {
    let n = states.len();

    // -- Step 1: identify all target states -----------------------------------
    // A "target" is a post-consumption NFA state index.
    let mut is_target = vec![false; n];
    for (i, state) in states.iter().enumerate() {
        // Skip dead states (patched-away ByteTable interior nodes use Match
        // as a tombstone, but their out fields are irrelevant).
        let _ = i;
        match *state {
            State::Byte { out, .. } | State::ByteCI { out, .. } | State::ByteClass { out, .. } => {
                if out != StateIdx::NONE {
                    is_target[out.idx()] = true;
                }
            }
            State::ByteTable { table } => {
                let map = &byte_tables[table.idx()];
                for b in 0u16..256 {
                    let t = map[b as u8];
                    if t != StateIdx::NONE {
                        is_target[t.idx()] = true;
                    }
                }
            }
            _ => {}
        }
    }

    // -- Step 2: compute per-target origin actions ----------------------------
    let mut targets_vec: Vec<Option<Tier3OriginKind>> = vec![None; n];

    for idx in 0..n {
        if !is_target[idx] {
            continue;
        }
        let target = StateIdx(idx as u32);
        targets_vec[idx] = analyze_target(target, states, state_can_reach_match);
    }

    // -- Step 2b: populate break_consuming_states (post-pass) ------------------
    // Now that all targets are known, re-derive each Increment target's
    // CInc break outputs and compute "true tail" consuming states — those
    // whose byte-consumption target does NOT lead to another CInc.
    for idx in 0..n {
        if let Some(Tier3OriginKind::Increment { .. }) = &targets_vec[idx] {
            // Walk from the target state itself (not a consuming state's
            // `out`) through epsilon transitions to find CInc break outputs.
            // `idx` is already the post-consumption target — it may be a
            // CInc directly, or reachable through Split/Assert/CI chains.
            let mut cinc_break_outs = Vec::new();
            let mut stack = vec![StateIdx(idx as u32)];
            let mut visited = vec![false; n];
            while let Some(s) = stack.pop() {
                let i = s.idx();
                if visited[i] {
                    continue;
                }
                visited[i] = true;
                match states[s] {
                    State::Split { out, out1 } => {
                        stack.push(out1);
                        stack.push(out);
                    }
                    State::Assert { out, .. } => stack.push(out),
                    State::CounterInstance { out, .. } => stack.push(out),
                    State::CounterIncrement { out1, .. } => {
                        cinc_break_outs.push(out1);
                    }
                    _ => {}
                }
            }
            if !cinc_break_outs.is_empty() {
                let tails = break_consuming_tails(&cinc_break_outs, states, &targets_vec);
                // Update the Increment with the computed tails.
                if let Some(Tier3OriginKind::Increment {
                    break_consuming_states,
                    ..
                }) = &mut targets_vec[idx]
                {
                    *break_consuming_states = tails.into_boxed_slice();
                }
            }
        }
    }

    // -- Step 3: compute per-CI-output consuming states -----------------------
    let mut ci_origins_vec: Vec<Box<[StateIdx]>> = vec![Box::new([]); n];
    for state in states.iter() {
        if let State::CounterInstance { out, .. } = *state
            && out != StateIdx::NONE
            && ci_origins_vec[out.idx()].is_empty()
        {
            let consuming = consuming_states_from(out, states);
            ci_origins_vec[out.idx()] = consuming.into_boxed_slice();
        }
    }

    // -- Step 4: compute global break seeds -----------------------------------
    // Collect ALL CInc nodes in the NFA, then walk each break path for CI
    // seeds.
    let mut all_cinc_nodes: Vec<(CounterIdx, StateIdx)> = Vec::new();
    for state in states.iter() {
        if let State::CounterIncrement { counter, out1, .. } = *state {
            all_cinc_nodes.push((counter, out1));
        }
    }

    let mut break_seeds_raw: Vec<(CounterIdx, CounterIdx, StateIdx)> = Vec::new();
    for &(trigger, break_target) in &all_cinc_nodes {
        // Phase 1: walk epsilon transitions from break target (Split,
        // Assert, CI) — same as the original code.  Self-referential
        // seeds (counter == trigger) are allowed here because they
        // represent the outer loop re-entering the counter body
        // through epsilon transitions (e.g. `(a{2,3})+`).
        let mut ci_stack = vec![break_target];
        let mut ci_visited = vec![false; n];
        // Collect consuming states reachable via epsilon from the break
        // path — these are the "hop points" for phase 2.
        let mut break_consuming: Vec<StateIdx> = Vec::new();
        while let Some(idx) = ci_stack.pop() {
            let i = idx.idx();
            if ci_visited[i] {
                continue;
            }
            ci_visited[i] = true;
            match states[idx] {
                State::CounterInstance { counter, out } => {
                    let ci_consuming = consuming_states_from(out, states);
                    for c in ci_consuming {
                        break_seeds_raw.push((trigger, counter, c));
                    }
                    ci_stack.push(out);
                }
                State::Split { out, out1 } => {
                    ci_stack.push(out1);
                    ci_stack.push(out);
                }
                State::Assert { out, .. } => ci_stack.push(out),
                State::Byte { .. }
                | State::ByteCI { .. }
                | State::ByteClass { .. }
                | State::ByteTable { .. } => {
                    break_consuming.push(idx);
                }
                _ => {}
            }
        }

        // Phase 2: follow consuming states' targets to find CI nodes
        // reachable after one byte consumption (multi-hop break seeds).
        // Self-referential seeds (counter == trigger) are EXCLUDED here
        // because consuming states on the break path entered the DFA
        // state through a previous with_break closure; re-seeding the
        // trigger counter at val=0 would create imprecise ranges.
        for cs in &break_consuming {
            let target = match states[*cs] {
                State::Byte { out, .. }
                | State::ByteCI { out, .. }
                | State::ByteClass { out, .. } => out,
                State::ByteTable { .. } => continue,
                _ => continue,
            };
            let mut hop_stack = vec![target];
            let mut hop_visited = vec![false; n];
            while let Some(idx) = hop_stack.pop() {
                let i = idx.idx();
                if hop_visited[i] {
                    continue;
                }
                hop_visited[i] = true;
                match states[idx] {
                    State::CounterInstance { counter, out } => {
                        // Only add seeds for DIFFERENT counters to
                        // prevent self-referential re-seeding.
                        if counter != trigger {
                            let ci_consuming = consuming_states_from(out, states);
                            for c in ci_consuming {
                                break_seeds_raw.push((trigger, counter, c));
                            }
                        }
                        hop_stack.push(out);
                    }
                    State::Split { out, out1 } => {
                        hop_stack.push(out1);
                        hop_stack.push(out);
                    }
                    State::Assert { out, .. } => hop_stack.push(out),
                    State::CounterIncrement { .. } => {
                        // Don't follow through another CInc — tier 3
                        // doesn't support nested counters in break paths.
                    }
                    _ => {}
                }
            }
        }
    }
    break_seeds_raw.sort_by_key(|&(t, c, s)| (t.idx(), c.idx(), s.0));
    break_seeds_raw.dedup();

    let break_seeds: Vec<Tier3BreakSeed> = break_seeds_raw
        .into_iter()
        .map(|(trigger, counter, origin)| Tier3BreakSeed {
            trigger,
            counter,
            origin,
        })
        .collect();

    // -- Step 5: compute max body origins per counter -------------------------
    // For each counter, walk from CI.out through the body (stopping at CInc
    // for the same counter) and count the distinct consuming NFA states.
    // The maximum across all counters becomes the stride for the flat
    // range-compressed instance storage.
    let mut max_body_origins: usize = 0;
    for state in states.iter() {
        if let State::CounterInstance { counter, out } = *state {
            let mut count: usize = 0;
            let mut stack = vec![out];
            let mut visited = vec![false; n];
            while let Some(idx) = stack.pop() {
                let i = idx.idx();
                if visited[i] {
                    continue;
                }
                visited[i] = true;
                match states[idx] {
                    State::CounterIncrement { counter: c, .. } if c == counter => {}
                    State::Split { out, out1 } => {
                        stack.push(out1);
                        stack.push(out);
                    }
                    State::Assert { out, .. } | State::CounterInstance { out, .. } => {
                        stack.push(out);
                    }
                    State::Byte { out, .. }
                    | State::ByteCI { out, .. }
                    | State::ByteClass { out, .. } => {
                        count += 1;
                        stack.push(out);
                    }
                    State::ByteTable { table } => {
                        count += 1;
                        for &succ in byte_tables[table.idx()].0.iter() {
                            if succ != StateIdx::NONE {
                                stack.push(succ);
                            }
                        }
                    }
                    _ => {}
                }
            }
            max_body_origins = max_body_origins.max(count);
        }
    }

    // -- Step 6: compute per-consuming-state target_is_match_at_end ----------
    // For each consuming NFA state, check if its byte-consumption target
    // reaches `$ → Match` through epsilon transitions.  This is used by
    // the post-break tail tracker: when a tail's action is Dead (its target
    // has no further consuming states), this flag tells us whether the
    // target state is `$ → Match`.
    let mut target_mae = vec![false; n];
    for (i, state) in states.iter().enumerate() {
        let target = match *state {
            State::Byte { out, .. } | State::ByteCI { out, .. } | State::ByteClass { out, .. } => {
                if out != StateIdx::NONE {
                    Some(out)
                } else {
                    None
                }
            }
            // ByteTable targets vary per byte; skip (overapproximation would
            // be unsound, and the tail tracker won't encounter ByteTable
            // origins in practice for tier 3 patterns).
            _ => None,
        };
        if let Some(t) = target {
            // Walk epsilon transitions from the target to find `$ → Match`.
            let mut estack = vec![t];
            let mut evisited = vec![false; n];
            while let Some(eidx) = estack.pop() {
                let ei = eidx.idx();
                if evisited[ei] {
                    continue;
                }
                evisited[ei] = true;
                match states[eidx] {
                    State::Split { out, out1 } => {
                        estack.push(out1);
                        estack.push(out);
                    }
                    State::Assert { kind, out } => {
                        if kind == AssertKind::End && state_can_reach_match[out.idx()] {
                            target_mae[i] = true;
                        }
                    }
                    State::CounterInstance { out, .. } => estack.push(out),
                    _ => {}
                }
            }
        }
    }

    Tier3Analysis {
        targets: targets_vec.into_boxed_slice(),
        ci_origins: ci_origins_vec.into_boxed_slice(),
        break_seeds: break_seeds.into_boxed_slice(),
        max_body_origins,
        target_is_match_at_end: target_mae.into_boxed_slice(),
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// Cached DFA transition for `(state, byte)`.
///
/// For non-counting transitions, `no_break` and `with_break` are identical.
/// For counting transitions, the matcher picks `with_break` when any
/// incrementing instance has `value+1 >= min`, otherwise `no_break`.
#[derive(Clone)]
struct Transition {
    /// DFA successor when no instance can break (continue-only closure).
    no_break: DfaStateId,
    no_break_is_match: bool,
    no_break_is_match_at_end: bool,
    /// DFA successor when at least one instance can break (both closure).
    with_break: DfaStateId,
    with_break_is_match: bool,
    with_break_is_match_at_end: bool,
    /// True if this transition crosses a CInc node.
    is_counting: bool,
    /// New counter instances from CI nodes reachable WITHOUT following
    /// any CInc break path (always applied).
    /// The third element is the initial counter value (0 for fresh seeds,
    /// 1 for seeds from resolved deferred assertions whose origin already
    /// consumed the resolving byte through a CInc increment).
    seeds: Box<[(CounterIdx, StateIdx, u32)]>,
    /// Additional seeds reachable only through CInc break paths.
    /// Each entry is `(trigger, counter, origin, initial_value)`:
    /// the seed for `counter` at `origin` is only applied when `trigger`
    /// (the counter whose CInc break leads to this CI) actually breaks.
    break_seeds: Box<[(CounterIdx, CounterIdx, StateIdx, u32)]>,
    /// Parallel arrays: `origin_keys[i]` → `origin_actions[i]`.
    origin_keys: Box<[StateIdx]>,
    origin_actions: Box<[OriginAction]>,
    /// True if any origin's byte-consumption target reaches `$ → Match`
    /// without going through a CInc node.  This is the "counter-free"
    /// subset of `no_break_is_match_at_end` — safe to propagate even for
    /// counting transitions, because the `$ → Match` path doesn't depend
    /// on any counter reaching its minimum.
    counter_free_match_at_end: bool,
}

impl Transition {
    fn empty() -> Self {
        Self {
            no_break: DfaStateId::UNPOPULATED,
            no_break_is_match: false,
            no_break_is_match_at_end: false,
            with_break: DfaStateId::UNPOPULATED,
            with_break_is_match: false,
            with_break_is_match_at_end: false,
            is_counting: false,
            seeds: Box::new([]),
            break_seeds: Box::new([]),
            origin_keys: Box::new([]),
            origin_actions: Box::new([]),
            counter_free_match_at_end: false,
        }
    }
}

// ---------------------------------------------------------------------------
// OriginAction
// ---------------------------------------------------------------------------

/// What happens to a counter instance at a given origin when the next
/// byte is consumed and epsilon closure is computed.
#[derive(Clone, Debug)]
enum OriginAction {
    /// The byte did not match — the instance dies.
    Dead,
    /// The byte matched and epsilon closure did NOT reach CInc.
    /// The instance keeps its counter value and moves to `new_origins`.
    Advance { new_origins: Box<[StateIdx]> },
    /// The byte matched and epsilon closure reached CInc.  The counter
    /// value is incremented and the min/max bounds determine whether the
    /// instance continues or breaks.
    ///
    /// `advance_origins` handles the `AdvanceOrIncrement` case (e.g.,
    /// `(a+){2,3}` where a Split before CInc lets the instance stay in
    /// the inner loop).  Empty when there are no such bypass paths.
    Increment {
        /// Origins reached WITHOUT going through CInc (same counter value).
        /// Empty for pure increments; non-empty for AdvanceOrIncrement.
        advance_origins: Box<[StateIdx]>,
        min: u32,
        max: u32,
        /// Origins the instance moves to when continuing (value+1 < max).
        continue_origins: Box<[StateIdx]>,
        /// True if the break path reaches Match directly.
        break_is_match: bool,
        /// True if the break path reaches match-at-end ($ → Match).
        break_is_match_at_end: bool,
        /// Non-counter consuming states on the break path (post-counter tail).
        break_consuming_states: Box<[StateIdx]>,
    },
}

impl OriginAction {
    /// Convert a precomputed [`Tier3OriginKind`] into an [`OriginAction`].
    fn from_precomputed(kind: &Tier3OriginKind) -> Self {
        match kind {
            Tier3OriginKind::Advance { new_origins } => OriginAction::Advance {
                new_origins: new_origins.clone(),
            },
            Tier3OriginKind::Increment {
                advance_origins,
                min,
                max,
                continue_origins,
                break_is_match,
                break_is_match_at_end,
                break_consuming_states,
            } => OriginAction::Increment {
                advance_origins: advance_origins.clone(),
                min: *min,
                max: *max,
                continue_origins: continue_origins.clone(),
                break_is_match: *break_is_match,
                break_is_match_at_end: *break_is_match_at_end,
                break_consuming_states: break_consuming_states.clone(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Range-compressed instance tracking
// ---------------------------------------------------------------------------

/// A contiguous range of counter values at a single origin.
///
/// Represents the set of instances `{(v, origin) | min_val <= v <= max_val}`.
/// At steady state, each origin within a counter body accumulates a
/// contiguous range `[0, max-1]`.
///
/// ## Why values at each origin always form a contiguous range
///
/// This holds for both fixed-length and variable-length bodies.  The
/// argument proceeds by induction on the byte position in the input:
///
/// 1. **Seed injection**: Every step, the unanchored loop injects a
///    fresh seed with value 0 at origin 0 (the body entry point).
///    This establishes value 0 at origin 0 on every step.
///
/// 2. **Advance**: When a byte is consumed, an instance at
///    `(value, origin_i)` moves to `(value, origin_j)` for each
///    `origin_j` reachable from `origin_i` via epsilon closure.
///    Since the set of values at `origin_i` is contiguous `[a, b]`
///    by the inductive hypothesis, the set of values arriving at
///    `origin_j` from this source is also `[a, b]`.  Multiple
///    sources merging at `origin_j` are each contiguous and all
///    anchored at 0, so their union is still contiguous.
///
/// 3. **Continue (increment)**: When a counter body completes, the
///    instance at `(value, final_origin)` becomes `(value + 1, origin_0)`.
///    By the inductive hypothesis, `final_origin` holds `[a, b]`, so
///    origin\_0 receives `[a+1, b+1]`.  Combined with the seed's
///    value 0 already at origin\_0 (from step 1), origin\_0 now has
///    `[0, b+1]` — still contiguous.
///
/// 4. **Break (counter exhausted)**: When `value + 1 > max`, the
///    instance exits the counter entirely.  This removes the top
///    of the range but doesn't fragment it.
///
/// The key insight is that all ranges are "anchored at 0" — they always
/// include value 0 because of the continuous seed injection.  Two ranges
/// that both start at 0 can never have a gap between them; their union
/// is simply `[0, max(b1, b2)]`, which [`RangeCounters::insert`] computes
/// via min/max.  This reduces per-byte cost from O(max\_count) to
/// O(num\_origins) — typically single-digit even for complex bodies.
#[derive(Clone, Debug)]
struct InstanceRange {
    /// The NFA consuming state these instances are waiting at.
    origin: StateIdx,
    /// Minimum counter value (inclusive).
    min_val: u32,
    /// Maximum counter value (inclusive).
    max_val: u32,
}

/// Flat storage for range-compressed counter instances.
///
/// Replaces `Vec<Vec<InstanceRange>>` with a single flat `Vec<InstanceRange>`
/// using fixed-stride slots per counter.  Counter `i` occupies
/// `data[i * stride .. i * stride + counts[i]]` where `stride` is
/// [`Tier3Analysis::max_body_origins`] (the maximum number of distinct
/// consuming NFA states in any counter body).
///
/// This eliminates inner `Vec` heap allocations, making memory reuse
/// trivial: [`clear`](Self::clear) just zeroes the `counts` array.
/// Double-buffering via [`std::mem::swap`] swaps two flat buffers in O(1).
struct RangeCounters {
    /// Flat backing storage.  Length = `num_counters * stride`.
    data: Vec<InstanceRange>,
    /// Number of live entries per counter.
    counts: Vec<u8>,
    /// Fixed number of slots per counter (= `max_body_origins`).
    stride: usize,
}

impl RangeCounters {
    /// Create a new `RangeCounters` for `num_counters` counters with
    /// the given stride (max origins per counter body).
    fn new(num_counters: usize, stride: usize) -> Self {
        let total = num_counters * stride;
        let mut data = Vec::with_capacity(total);
        // Fill with dummy entries — only `counts[i]` entries are live.
        data.resize(
            total,
            InstanceRange {
                origin: StateIdx::NONE,
                min_val: 0,
                max_val: 0,
            },
        );
        Self {
            data,
            counts: vec![0; num_counters],
            stride,
        }
    }

    /// Clear all counters (reset live counts to zero).
    #[inline]
    fn clear(&mut self) {
        for c in &mut self.counts {
            *c = 0;
        }
    }

    /// Returns the number of counters.
    #[inline]
    fn num_counters(&self) -> usize {
        self.counts.len()
    }

    /// Returns the live entries for counter `ci`.
    #[inline]
    fn entries(&self, ci: usize) -> &[InstanceRange] {
        let base = ci * self.stride;
        &self.data[base..base + self.counts[ci] as usize]
    }

    /// Returns true if any counter has live entries.
    #[inline]
    fn any_live(&self) -> bool {
        self.counts.iter().any(|&c| c != 0)
    }

    /// Merge a `[min_val, max_val]` range into counter `ci`.
    ///
    /// If an entry with the same origin already exists, fuses the ranges.
    /// Otherwise, appends a new entry.  Each origin appears at most once.
    #[inline]
    fn insert(&mut self, ci: usize, origin: StateIdx, min_val: u32, max_val: u32) {
        debug_assert!(min_val <= max_val);
        let base = ci * self.stride;
        let count = self.counts[ci] as usize;
        // Linear scan — at most `stride` entries (typically 1-4).
        for i in 0..count {
            let r = &mut self.data[base + i];
            if r.origin == origin {
                r.min_val = r.min_val.min(min_val);
                r.max_val = r.max_val.max(max_val);
                return;
            }
        }
        debug_assert!(
            count < self.stride,
            "RangeCounters overflow: counter {ci} has {count} entries, stride = {}",
            self.stride,
        );
        self.data[base + count] = InstanceRange {
            origin,
            min_val,
            max_val,
        };
        self.counts[ci] = (count + 1) as u8;
    }
}

// ---------------------------------------------------------------------------
// Tier 3 DFA cache
// ---------------------------------------------------------------------------

/// Lazy DFA cache for Tier 3.
pub(crate) struct Tier3DfaCache {
    inner: DfaCache,
    stride: usize,
    transitions: Vec<Transition>,
    start_seeds: Box<[(CounterIdx, StateIdx, u32)]>,
}

impl fmt::Debug for Tier3DfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier3DfaCache")
            .field("num_states", &self.inner.states.len())
            .finish()
    }
}

impl Tier3DfaCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: DfaCache::new(),
            stride: 256,
            transitions: Vec::new(),
            start_seeds: Box::new([]),
        }
    }

    fn intern_state(
        &mut self,
        nfa_states: &[StateIdx],
        deferred_asserts: &[StateIdx],
        is_match: bool,
        is_match_at_end: bool,
        prev_was_word: bool,
    ) -> Option<DfaStateId> {
        let stride = self.stride;
        self.inner.intern_state(
            nfa_states,
            deferred_asserts,
            is_match,
            is_match_at_end,
            prev_was_word,
            || {
                self.transitions
                    .extend(std::iter::repeat_with(Transition::empty).take(stride));
            },
        )
    }

    // -----------------------------------------------------------------------
    // Epsilon closure
    // -----------------------------------------------------------------------

    /// Compute epsilon closure.
    ///
    /// `follow_break`: whether to follow the CInc break path (out1).
    /// The continue path (out) is always followed.
    ///
    /// Also tracks CI traversals: when a CI is visited, consuming states
    /// reachable from CI.out are recorded as seed instances.
    #[allow(clippy::too_many_arguments)]
    fn epsilon_closure(
        &mut self,
        memory: &mut DfaMemory,
        seeds: impl Iterator<Item = StateIdx>,
        regex: &Regex,
        analysis: &Tier3Analysis,
        at_start: bool,
        prev_byte: Option<u8>,
        next_byte: Option<u8>,
        follow_break: bool,
    ) -> ClosureResult {
        let mut encountered_cinc = false;

        let (is_match, is_match_at_end) = memory.epsilon_closure(
            seeds,
            regex,
            at_start,
            false,
            prev_byte,
            next_byte,
            |mem, counter, out| {
                mem.closure_stack.push(out);
                mem.closure_seeds.push((counter, out));
            },
            |mem, _, out, out1, _, _| {
                encountered_cinc = true;
                mem.closure_stack.push(out);
                if follow_break {
                    mem.closure_stack.push(out1);
                }
            },
        );

        // Resolve CI seed pairs to (counter, consuming_state) pairs
        // using the precomputed ci_origins table.
        let mut seed_instances: Vec<(CounterIdx, StateIdx)> = Vec::new();
        for &(counter, ci_out) in &memory.closure_seeds {
            for &c in analysis.ci_origins[ci_out.idx()].iter() {
                seed_instances.push((counter, c));
            }
        }
        seed_instances.sort_by_key(|&(c, s)| (c.idx(), s.0));
        seed_instances.dedup();

        ClosureResult {
            nfa_states: memory.closure_result.as_slice().into(),
            deferred_asserts: memory.closure_deferred.as_slice().into(),
            is_match,
            is_match_at_end,
            encountered_cinc,
            seed_instances: seed_instances.into_boxed_slice(),
        }
    }

    // -----------------------------------------------------------------------
    // Transition computation
    // -----------------------------------------------------------------------

    /// Compute the transition for `(from_state, byte)`.
    fn populate(
        &mut self,
        memory: &mut DfaMemory,
        from: DfaStateId,
        byte: u8,
        regex: &Regex,
        analysis: &Tier3Analysis,
    ) -> Transition {
        // Phase 1: collect targets — NFA states reached after consuming `byte`.
        let mut targets_per_origin: Vec<(StateIdx, Vec<StateIdx>)> = Vec::new();

        let mut resolved_seeds: Vec<(CounterIdx, StateIdx, u32)> = Vec::new();
        let mut resolved_cinc = false;
        let mut resolved_is_match = false;
        // Note: we do NOT track resolved_is_match_at_end here.  That flag
        // means "the deferred assertion passed mid-stream (with next=byte) and
        // $ → Match was reachable."  But the assertion was evaluated with
        // next=Some(byte) — at actual end-of-input, next is None and word-
        // boundary conditions may flip.  finish() correctly re-evaluates via
        // resolve_deferred_at_end().

        if from != DfaStateId::DEAD {
            let from_state = &self.inner.states[from.idx()];

            // Resolve deferred assertions.
            let extra = from_state.resolve_deferred(byte, regex);
            if !extra.is_empty() {
                let resolved_prev = from_state.prev_byte_representative();
                let cr = self.epsilon_closure(
                    memory,
                    extra.into_iter(),
                    regex,
                    analysis,
                    false,
                    resolved_prev,
                    Some(byte),
                    true, // follow_break=true for resolved assert closure
                );
                resolved_seeds = cr
                    .seed_instances
                    .iter()
                    .map(|&(c, s)| (c, s, 0u32))
                    .collect();
                resolved_cinc = cr.encountered_cinc;
                resolved_is_match = cr.is_match;
                // Discard cr.is_match_at_end — see comment above.
                for &idx in cr.nfa_states.iter() {
                    if let Some(t) = consume_byte(idx, byte, regex) {
                        targets_per_origin.push((idx, vec![t]));
                    }
                }
            }

            // Phase 2: consuming states in `from` consume `byte`.
            let nfa_states = self.inner.states[from.idx()].nfa_states.clone();
            for &idx in nfa_states.iter() {
                if let Some(t) = consume_byte(idx, byte, regex) {
                    targets_per_origin.push((idx, vec![t]));
                }
            }
        }

        let all_targets: Vec<StateIdx> = targets_per_origin
            .iter()
            .flat_map(|(_, ts)| ts.iter().copied())
            .collect();

        // Probe: full "both" closure to detect CInc and collect seeds.
        let probe = self.epsilon_closure(
            memory,
            all_targets
                .iter()
                .copied()
                .chain(std::iter::once(regex.start)),
            regex,
            analysis,
            false,
            Some(byte),
            None,
            true, // follow_break=true
        );

        let is_counting = probe.encountered_cinc || resolved_cinc;

        // Build per-origin actions from the precomputed analysis.
        //
        // The analysis is indexed by *target* state (the NFA state
        // reached after byte consumption), not by origin.  Each origin
        // produces exactly one target, so `targets[0]` is the lookup key.
        let mut origin_keys = Vec::new();
        let mut origin_actions = Vec::new();
        for &(origin, ref targets) in &targets_per_origin {
            debug_assert_eq!(targets.len(), 1);
            let target = targets[0];
            let action = match analysis.targets[target.idx()] {
                Some(ref kind) => OriginAction::from_precomputed(kind),
                None => OriginAction::Dead,
            };
            origin_keys.push(origin);
            origin_actions.push(action);
        }

        // Compute seed initial values.  Resolved deferred seeds whose
        // origin consumed the byte and had an Increment action start at
        // value 1 (the resolving byte already counted as one iteration).
        for rs in &mut resolved_seeds {
            if let Some(pos) = origin_keys.iter().position(|&k| k == rs.1)
                && matches!(origin_actions[pos], OriginAction::Increment { .. })
            {
                rs.2 = 1;
            }
        }

        // Compute DFA successors: no_break and with_break.
        if is_counting {
            // Two separate closures.
            let cr_nb = self.epsilon_closure(
                memory,
                all_targets
                    .iter()
                    .copied()
                    .chain(std::iter::once(regex.start)),
                regex,
                analysis,
                false,
                Some(byte),
                None,
                false, // follow_break=false → no_break
            );
            let nb_id = self.intern_closure_result(&cr_nb, byte);

            // with_break reuses the probe closure.
            let wb_id = self.intern_closure_result(&probe, byte);

            let (nb_m, nb_mae) = self.match_flags(nb_id);
            let (wb_m, wb_mae) = self.match_flags(wb_id);

            // Unconditional seeds: reachable without following CInc break
            // paths (from the no_break closure) plus resolved deferred seeds.
            let mut seeds: Vec<(CounterIdx, StateIdx, u32)> = cr_nb
                .seed_instances
                .iter()
                .map(|&(c, s)| (c, s, 0u32))
                .collect();
            for s in &resolved_seeds {
                if !seeds.iter().any(|e| e.0 == s.0 && e.1 == s.1 && e.2 == s.2) {
                    seeds.push(*s);
                }
            }

            // Break-only seeds: reachable only through CInc break paths.
            // Each seed is tagged with the counter whose CInc break leads
            // to it — the seed is only applied when that specific counter's
            // instance actually breaks.
            //
            // Break seeds that duplicate unconditional seeds are excluded:
            // the unconditional seed already fires every time, so the
            // break-gated duplicate is redundant.  Keeping it would cause
            // double-seeding when the trigger counter actually breaks.
            let break_seeds = Self::compute_break_seeds(&seeds, analysis);

            // Compute counter-free match-at-end: true if any origin's
            // target reaches `$ → Match` without going through CInc.
            // Such paths are safe to propagate even for counting
            // transitions — the `$ → Match` doesn't depend on any
            // counter reaching its minimum.
            //
            // We check all non-Increment origins (Dead and Advance).
            // Dead means the target has no consuming states at all;
            // Advance means the target has consuming states but may
            // ALSO reach `$ → Match` via epsilon transitions.  In both
            // cases, the `$ → Match` path doesn't cross any CInc.
            let counter_free_mae =
                origin_keys
                    .iter()
                    .zip(origin_actions.iter())
                    .any(|(&origin, action)| {
                        !matches!(action, OriginAction::Increment { .. })
                            && analysis.target_is_match_at_end[origin.idx()]
                    });

            // Fold resolved deferred assertion matches into both
            // successors' flags.  `resolved_is_match` applies
            // unconditionally (the DFA state that was transitioned FROM
            // already encodes the correct counter-aware path).
            Transition {
                no_break: nb_id,
                no_break_is_match: nb_m || resolved_is_match,
                no_break_is_match_at_end: nb_mae,
                with_break: wb_id,
                with_break_is_match: wb_m || resolved_is_match,
                with_break_is_match_at_end: wb_mae,
                is_counting: true,
                seeds: seeds.into(),
                break_seeds: break_seeds.into(),
                origin_keys: origin_keys.into_boxed_slice(),
                origin_actions: origin_actions.into_boxed_slice(),
                counter_free_match_at_end: counter_free_mae,
            }
        } else {
            // Non-counting: both successors are the same.  All seeds are
            // unconditional (no CInc break distinction).
            let id = self.intern_closure_result(&probe, byte);
            let (m, mae) = self.match_flags(id);

            let mut seeds: Vec<(CounterIdx, StateIdx, u32)> = probe
                .seed_instances
                .iter()
                .map(|&(c, s)| (c, s, 0u32))
                .collect();
            for s in &resolved_seeds {
                if !seeds.iter().any(|e| e.0 == s.0 && e.1 == s.1 && e.2 == s.2) {
                    seeds.push(*s);
                }
            }

            Transition {
                no_break: id,
                no_break_is_match: m || resolved_is_match,
                no_break_is_match_at_end: mae,
                with_break: id,
                with_break_is_match: m || resolved_is_match,
                with_break_is_match_at_end: mae,
                is_counting: false,
                seeds: seeds.into(),
                break_seeds: Box::new([]),
                origin_keys: origin_keys.into_boxed_slice(),
                origin_actions: origin_actions.into_boxed_slice(),
                counter_free_match_at_end: false, // Not used for non-counting transitions.
            }
        }
    }

    /// Intern a closure result into the state table.  Returns DEAD if
    /// empty or if the state cap is reached.
    fn intern_closure_result(&mut self, cr: &ClosureResult, byte: u8) -> DfaStateId {
        if cr.nfa_states.is_empty()
            && cr.deferred_asserts.is_empty()
            && !cr.is_match
            && !cr.is_match_at_end
        {
            return DfaStateId::DEAD;
        }
        let pw = if cr.deferred_asserts.is_empty() {
            false
        } else {
            is_word_byte(byte)
        };
        self.intern_state(
            &cr.nfa_states,
            &cr.deferred_asserts,
            cr.is_match,
            cr.is_match_at_end,
            pw,
        )
        .unwrap_or(DfaStateId::DEAD)
    }

    /// Get match flags for a DFA state ID.
    fn match_flags(&self, id: DfaStateId) -> (bool, bool) {
        if id == DfaStateId::DEAD {
            (false, false)
        } else {
            let s = &self.inner.states[id.idx()];
            (s.is_match, s.is_match_at_end)
        }
    }

    /// Compute break seeds for a counting transition.
    ///
    /// Uses the precomputed [`Tier3Analysis::break_seeds`] table.
    /// Break seeds already present in the unconditional `seeds` list are
    /// excluded — the unconditional seed fires every time, making the
    /// break-gated duplicate redundant.
    ///
    /// Including ALL other precomputed break seeds (not just those reachable
    /// from the current targets) is safe: at runtime, each break seed is
    /// gated on `counter_broke[trigger]`, and unreachable triggers never
    /// break.
    fn compute_break_seeds(
        seeds: &[(CounterIdx, StateIdx, u32)],
        analysis: &Tier3Analysis,
    ) -> Vec<(CounterIdx, CounterIdx, StateIdx, u32)> {
        let mut result: Vec<(CounterIdx, CounterIdx, StateIdx, u32)> = Vec::new();
        for bs in analysis.break_seeds.iter() {
            let entry = (bs.trigger, bs.counter, bs.origin, 0u32);
            // Skip if this seed is already in the unconditional list.
            if seeds
                .iter()
                .any(|e| e.0 == entry.1 && e.1 == entry.2 && e.2 == entry.3)
            {
                continue;
            }
            if !result
                .iter()
                .any(|e| e.0 == entry.0 && e.1 == entry.1 && e.2 == entry.2 && e.3 == entry.3)
            {
                result.push(entry);
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Cache management
    // -----------------------------------------------------------------------

    fn clear(&mut self, memory: &mut DfaMemory, num_nfa_states: usize, stride: usize) {
        self.inner.clear();
        self.stride = stride;
        memory.clear(num_nfa_states);
        self.transitions.clear();
        self.start_seeds = Box::new([]);
    }

    /// Prepare the cache for `regex`.
    pub(crate) fn prepare(
        &mut self,
        memory: &mut DfaMemory,
        regex: &Regex,
        analysis: &Tier3Analysis,
    ) {
        let id = regex.id;
        if self.inner.regex_id == id && self.inner.start_id != DfaStateId::DEAD {
            return;
        }
        self.clear(memory, regex.states.len(), regex.num_byte_classes);
        self.inner.regex_id = id;

        let cr = self.epsilon_closure(
            memory,
            std::iter::once(regex.start),
            regex,
            analysis,
            true,
            None,
            None,
            true, // follow_break=true for start closure
        );
        self.inner.start_id = self
            .intern_state(
                &cr.nfa_states,
                &cr.deferred_asserts,
                cr.is_match,
                cr.is_match_at_end,
                false,
            )
            .expect("start state exceeds DFA_MAX_STATES");
        self.inner.start_is_match = self.inner.states[self.inner.start_id.idx()].is_match;
        self.inner.start_is_match_at_end =
            self.inner.states[self.inner.start_id.idx()].is_match_at_end;
        self.start_seeds = cr
            .seed_instances
            .iter()
            .map(|&(c, s)| (c, s, 0u32))
            .collect();
    }
}

// ---------------------------------------------------------------------------
// ClosureResult
// ---------------------------------------------------------------------------

struct ClosureResult {
    nfa_states: Box<[StateIdx]>,
    deferred_asserts: Box<[StateIdx]>,
    is_match: bool,
    is_match_at_end: bool,
    encountered_cinc: bool,
    seed_instances: Box<[(CounterIdx, StateIdx)]>,
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Try to consume `byte` at NFA state `idx`.
fn consume_byte(idx: StateIdx, byte: u8, regex: &Regex) -> Option<StateIdx> {
    match regex.states[idx] {
        State::Byte { byte: b, out } if byte == b => Some(out),
        State::ByteCI { byte: b, out } if byte_match_ci(byte, b) => Some(out),
        State::ByteClass { class, out } if regex.classes[class][byte] => Some(out),
        State::ByteTable { table } => {
            let t = regex.byte_tables[table][byte];
            if t != StateIdx::NONE { Some(t) } else { None }
        }
        _ => None,
    }
}

/// Find all consuming NFA states reachable from `start` through epsilon
/// transitions.
fn consuming_states_from(start: StateIdx, states: &[State]) -> Vec<StateIdx> {
    let mut result = Vec::new();
    let mut stack = vec![start];
    let mut visited = vec![false; states.len()];
    while let Some(idx) = stack.pop() {
        let i = idx.idx();
        if visited[i] {
            continue;
        }
        visited[i] = true;
        match states[idx] {
            State::Split { out, out1 } => {
                stack.push(out1);
                stack.push(out);
            }
            State::Assert { out, .. } => stack.push(out),
            State::CounterInstance { out, .. } => stack.push(out),
            State::Byte { .. }
            | State::ByteCI { .. }
            | State::ByteClass { .. }
            | State::ByteTable { .. } => {
                result.push(idx);
            }
            _ => {}
        }
    }
    result
}

/// Analyze a single post-consumption target state.
///
/// Performs one DFS from `target` through epsilon transitions (Split,
/// Assert, CI) to determine the structural action:
///
/// - If a `CounterIncrement` is reachable, returns `Increment` with
///   advance origins (consuming states before CInc), continue origins
///   (consuming states on the CInc continue path), and break match flags.
/// - Otherwise, returns `Advance` with consuming states reachable, or
///   `None` if dead (no consuming states reachable).
fn analyze_target(
    target: StateIdx,
    states: &[State],
    can_reach_match: &[bool],
) -> Option<Tier3OriginKind> {
    // Single DFS: walk through Split, Assert, CI.
    // - Consuming states reached without crossing CInc → advance_origins.
    // - CInc encountered → record (min, max, continue_out, break_out).
    let mut advance_origins = Vec::new();
    let mut found_cinc: Option<(usize, usize)> = None;
    let mut cinc_continue_outs = Vec::new();
    let mut cinc_break_outs = Vec::new();

    let mut stack = vec![target];
    let mut visited = vec![false; states.len()];
    while let Some(idx) = stack.pop() {
        let i = idx.idx();
        if visited[i] {
            continue;
        }
        visited[i] = true;
        match states[idx] {
            State::Split { out, out1 } => {
                stack.push(out1);
                stack.push(out);
            }
            State::Assert { out, .. } => stack.push(out),
            State::CounterInstance { out, .. } => stack.push(out),
            State::CounterIncrement {
                min,
                max,
                out,
                out1,
                ..
            } => {
                found_cinc = Some((min, max));
                cinc_continue_outs.push(out);
                cinc_break_outs.push(out1);
            }
            State::Byte { .. }
            | State::ByteCI { .. }
            | State::ByteClass { .. }
            | State::ByteTable { .. } => {
                advance_origins.push(idx);
            }
            State::Match => {}
        }
    }

    if let Some((min, max)) = found_cinc {
        advance_origins.sort_unstable_by_key(|s| s.0);
        advance_origins.dedup();

        // Continue origins: consuming states reachable from CInc continue
        // outputs (no nested CInc in tier 3).
        let mut continue_origins = Vec::new();
        for co in &cinc_continue_outs {
            let consuming = consuming_states_from(*co, states);
            continue_origins.extend(consuming);
        }
        continue_origins.sort_unstable_by_key(|s| s.0);
        continue_origins.dedup();

        // Break match flags: walk from CInc break outputs through epsilon
        // transitions to find Match / ($ → Match).
        let (break_is_match, break_is_match_at_end) =
            break_closure(&cinc_break_outs, states, can_reach_match);

        Some(Tier3OriginKind::Increment {
            advance_origins: advance_origins.into_boxed_slice(),
            min: min as u32,
            max: max as u32,
            continue_origins: continue_origins.into_boxed_slice(),
            break_is_match,
            break_is_match_at_end,
            // Populated in a post-pass by compute_tier3_analysis after all
            // targets are known (break_consuming_tails needs the targets array).
            break_consuming_states: Box::new([]),
        })
    } else if advance_origins.is_empty() {
        None // Dead.
    } else {
        Some(Tier3OriginKind::Advance {
            new_origins: advance_origins.into_boxed_slice(),
        })
    }
}

/// Epsilon closure from CInc break targets only.
///
/// Returns `(is_match, is_match_at_end)` — whether `Match` or `$ → Match`
/// is reachable from the given break-path seeds **without** passing through
/// another `CounterInstance`.
///
/// Stopping at `CounterInstance` is critical for multi-counter patterns:
/// when counter A breaks, the break path may enter counter B's region
/// (via CI → body → CInc → break → `$` → `Match`).  But counter B has
/// not yet accumulated any iterations — its break condition hasn't been
/// met.  Propagating `break_is_match_at_end` through CI would cause a
/// false positive: the break of counter A would immediately claim "match
/// at end" even though counter B's minimum count hasn't been reached.
///
/// The runtime counter machinery handles downstream counters precisely:
/// `break_seeds` creates new counter instances for B when A breaks, and
/// `post_break_tails` tracks non-counter consuming states for deferred
/// `$ → Match` detection after additional bytes are consumed.
fn break_closure(
    break_seeds: &[StateIdx],
    states: &[State],
    can_reach_match: &[bool],
) -> (bool, bool) {
    let mut is_match = false;
    let mut is_match_at_end = false;
    let mut stack: Vec<StateIdx> = break_seeds.to_vec();
    let mut visited = vec![false; states.len()];
    while let Some(idx) = stack.pop() {
        let i = idx.idx();
        if visited[i] {
            continue;
        }
        visited[i] = true;
        match states[idx] {
            State::Split { out, out1 } => {
                stack.push(out1);
                stack.push(out);
            }
            State::Assert { kind, out } => {
                if kind == AssertKind::End && can_reach_match[out.idx()] {
                    is_match_at_end = true;
                }
            }
            State::Match => {
                is_match = true;
            }
            // Do NOT follow through CounterInstance — downstream
            // counters have not accumulated their required count.
            // Their match paths are handled by break_seeds and
            // post_break_tails at runtime.
            State::CounterInstance { .. } => {}
            _ => {}
        }
    }
    (is_match, is_match_at_end)
}

/// Collect consuming NFA states on a CInc break path that are "true tail"
/// states — their byte-consumption target does NOT lead to another CInc.
///
/// These states need runtime tracking to detect deferred `$ → Match` after
/// additional bytes.  Consuming states whose targets lead to CInc are
/// handled by the counter instance seeding/tracking machinery instead.
fn break_consuming_tails(
    break_seeds: &[StateIdx],
    states: &[State],
    targets: &[Option<Tier3OriginKind>],
) -> Vec<StateIdx> {
    let mut result = Vec::new();
    let mut stack: Vec<StateIdx> = break_seeds.to_vec();
    let mut visited = vec![false; states.len()];
    while let Some(idx) = stack.pop() {
        let i = idx.idx();
        if visited[i] {
            continue;
        }
        visited[i] = true;
        match states[idx] {
            State::Split { out, out1 } => {
                stack.push(out1);
                stack.push(out);
            }
            State::Assert { out, .. } => stack.push(out),
            State::CounterInstance { out, .. } => stack.push(out),
            State::Byte { out, .. } | State::ByteCI { out, .. } | State::ByteClass { out, .. } => {
                // Only include if the target's analysis is NOT Increment.
                // Increment targets are handled by counter seeding.
                if !matches!(targets[out.idx()], Some(Tier3OriginKind::Increment { .. })) {
                    result.push(idx);
                }
            }
            State::ByteTable { table: _ } => {
                // ByteTable has variable targets; conservatively skip.
                // Tier 3 patterns with ByteTable post-break tails are rare.
            }
            _ => {}
        }
    }
    result.sort_unstable_by_key(|s| s.0);
    result.dedup();
    result
}

// ---------------------------------------------------------------------------
// Tier 3 Matcher
// ---------------------------------------------------------------------------

/// Tier 3 DFA matcher.
///
/// Counter instances are tracked using range-compressed
/// [`InstanceRange`] entries (one per origin per counter) rather than
/// individual instances.  See the contiguity proof on [`InstanceRange`]
/// for why this is always valid.
pub struct Tier3DfaMatcher<'a> {
    cache: &'a mut Tier3DfaCache,
    memory: &'a mut DfaMemory,
    regex: &'a Regex,
    analysis: &'a Tier3Analysis,
    current: DfaStateId,
    /// Range-compressed instance storage — flat per-counter slots.
    counters: RangeCounters,
    /// Scratch buffer for the next step (double-buffered with `counters`).
    next_counters: RangeCounters,
    ever_matched: bool,
    match_at_end: bool,
    has_live_instances: bool,
    /// NFA consuming states currently active from a counter break path.
    /// These track the post-counter "tail" (e.g. a trailing `.` or
    /// literal before `$ → Match`).  Updated each step: each tail is
    /// advanced through its `OriginAction`.  When a tail's action is
    /// `Dead` and `target_is_match_at_end` is true, `match_at_end` is set.
    ///
    /// This replaces the use of DFA-level `is_match_at_end` for counting
    /// transitions, which is overly optimistic because the DFA state
    /// includes post-break states before counters have actually broken.
    post_break_tails: Vec<StateIdx>,
    /// Scratch buffer for advancing `post_break_tails` (double-buffered).
    next_post_break_tails: Vec<StateIdx>,
    prefilter: Prefilter,
}

impl<'a> Tier3DfaMatcher<'a> {
    pub(crate) fn new(
        cache: &'a mut Tier3DfaCache,
        memory: &'a mut DfaMemory,
        regex: &'a Regex,
        analysis: &'a Tier3Analysis,
    ) -> Self {
        let nc = regex.num_counters;
        let stride = analysis.max_body_origins;

        let mut counters = RangeCounters::new(nc, stride);
        let next_counters = RangeCounters::new(nc, stride);

        for &(counter, origin, value) in cache.start_seeds.iter() {
            counters.insert(counter.idx(), origin, value, value);
        }

        let ever_matched = cache.inner.start_is_match;
        let match_at_end = cache.inner.start_is_match_at_end;
        let has_live_instances = !cache.start_seeds.is_empty();

        Tier3DfaMatcher {
            current: cache.inner.start_id,
            ever_matched,
            match_at_end,
            has_live_instances,
            post_break_tails: Vec::new(),
            next_post_break_tails: Vec::new(),
            cache,
            memory,
            regex,
            analysis,
            counters,
            next_counters,
            prefilter: regex.prefilter,
        }
    }

    /// Slow path: handles counting transitions and instance processing.
    ///
    /// Operates on range-compressed [`InstanceRange`] entries (one per
    /// origin per counter).  This reduces per-byte work from O(max_count)
    /// to O(num_body_origins) per counter.  See the contiguity proof on
    /// [`InstanceRange`] for why range compression is always valid.
    #[inline(never)]
    fn step_slow(&mut self, slot: usize) {
        let t = &self.cache.transitions[slot];

        // Reset next_counters.
        self.next_counters.clear();
        self.match_at_end = false;

        // Advance existing post-break tails through this transition.
        // Each tail is a consuming NFA state from a previous counter
        // break.  If it consumed this byte and its target reaches
        // `$ → Match`, set match_at_end.
        self.next_post_break_tails.clear();
        for &pbo in &self.post_break_tails {
            let action = t
                .origin_keys
                .iter()
                .position(|&k| k == pbo)
                .map(|i| &t.origin_actions[i]);
            match action {
                Some(OriginAction::Advance { new_origins }) => {
                    for &new_o in new_origins.iter() {
                        if !self.next_post_break_tails.contains(&new_o) {
                            self.next_post_break_tails.push(new_o);
                        }
                    }
                }
                Some(OriginAction::Increment { .. }) => {
                    // Tail hit a CInc — the post-break path has entered
                    // a counter body.  Stop tracking: the counter instance
                    // machinery (seeding + per-instance break checks) handles
                    // match detection from here.  We must NOT propagate the
                    // CInc's break_is_match / break_is_match_at_end, because
                    // the tail has no counter value — it can't know if the
                    // counter will reach its min.
                }
                Some(OriginAction::Dead) => {
                    // Tail consumed a byte and its target is dead (no
                    // further consuming states or CInc).  Check whether
                    // the target reaches `$ → Match` via precomputed flag.
                    if self.analysis.target_is_match_at_end[pbo.idx()] {
                        self.match_at_end = true;
                    }
                }
                None => {
                    // Tail state is not in the transition's origin_keys,
                    // meaning this NFA state does not accept the current
                    // byte.  The tail is dead — drop it silently.
                }
            }
        }

        let mut any_can_break = false;
        let num_counters = self.counters.num_counters();
        // Use u64 bitmask — tier 3 patterns have at most a handful of
        // counters (well under 64).
        debug_assert!(num_counters <= 64);
        let mut counter_broke: u64 = 0;

        #[allow(clippy::needless_range_loop)]
        for c_idx in 0..num_counters {
            for range in self.counters.entries(c_idx) {
                let action = t
                    .origin_keys
                    .iter()
                    .position(|&k| k == range.origin)
                    .map(|i| &t.origin_actions[i]);

                match action {
                    Some(OriginAction::Advance { new_origins }) => {
                        for &new_o in new_origins.iter() {
                            self.next_counters
                                .insert(c_idx, new_o, range.min_val, range.max_val);
                        }
                    }
                    Some(OriginAction::Dead) | None => {}
                    Some(OriginAction::Increment {
                        advance_origins,
                        min,
                        max,
                        continue_origins,
                        break_is_match,
                        break_is_match_at_end,
                        break_consuming_states,
                    }) => {
                        // Advance-or-increment: the range survives at
                        // advance_origins with the same values.
                        for &new_o in advance_origins.iter() {
                            self.next_counters
                                .insert(c_idx, new_o, range.min_val, range.max_val);
                        }

                        // CInc fires: new values are [min_val+1, max_val+1].
                        let new_min = range.min_val + 1;
                        let new_max = range.max_val + 1;

                        // Continue: values in [new_min, min(new_max, max-1)].
                        let continue_cap = *max - 1; // max counter value that can continue
                        if new_min <= continue_cap {
                            let cont_max = new_max.min(continue_cap);
                            for &new_o in continue_origins.iter() {
                                self.next_counters.insert(c_idx, new_o, new_min, cont_max);
                            }
                        }

                        // Break: any value >= min triggers break.
                        // The break range is [max(new_min, *min), new_max].
                        if new_max >= *min {
                            any_can_break = true;
                            counter_broke |= 1u64 << c_idx;
                            if *break_is_match {
                                self.ever_matched = true;
                            }
                            if *break_is_match_at_end {
                                self.match_at_end = true;
                            }
                            // Track post-break consuming states for
                            // subsequent match-at-end detection.
                            for &new_o in break_consuming_states.iter() {
                                if !self.next_post_break_tails.contains(&new_o) {
                                    self.next_post_break_tails.push(new_o);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Select DFA successor.
        if t.is_counting && any_can_break {
            self.current = t.with_break;
        } else {
            self.current = t.no_break;
        }
        if !t.is_counting {
            let (m, mae) = if any_can_break {
                (t.with_break_is_match, t.with_break_is_match_at_end)
            } else {
                (t.no_break_is_match, t.no_break_is_match_at_end)
            };
            if m {
                self.ever_matched = true;
            }
            if mae {
                self.match_at_end = true;
            }
        } else {
            // For counting transitions, propagate `no_break_is_match`
            // and `counter_free_match_at_end`, but NOT the full
            // `no_break_is_match_at_end`.
            //
            // The no_break DFA state may contain consuming states that
            // arrived via *previous* counter breaks (e.g., state 7 in
            // `^.{2,3}.{2,3}.$` is reachable after counter 0 breaks).
            // The full no_break closure's `is_match_at_end` includes
            // `$ → Match` paths from those post-break states, but those
            // paths are only valid when the relevant downstream counter
            // has also reached its minimum.  Propagating the full flag
            // would bypass per-instance counter checks, causing false
            // positives at boundary lengths.
            //
            // `counter_free_match_at_end` is the safe subset: it only
            // includes `$ → Match` from origins whose target does NOT
            // go through any CInc (Dead actions with target_is_match_at_end).
            // These paths are always valid regardless of counter state.
            //
            // Counter-dependent `$ → Match` paths are handled precisely
            // by per-instance break checks (`break_is_match_at_end`) and
            // the `post_break_tails` mechanism (target_is_match_at_end
            // gated on actual counter breaks).
            //
            // We do NOT propagate with_break flags here (even when
            // any_can_break) because the with_break DFA state's match flags
            // are overly optimistic — they include paths where ALL CInc
            // break paths are followed regardless of actual counter values.
            // Per-instance break handling above already handles the break
            // case precisely.
            if t.no_break_is_match {
                self.ever_matched = true;
            }
            if t.counter_free_match_at_end {
                self.match_at_end = true;
            }
        }

        // Swap range buffers (O(1) — just swaps the flat Vec pointers).
        std::mem::swap(&mut self.counters, &mut self.next_counters);
        // Swap post-break tail buffers.
        std::mem::swap(&mut self.post_break_tails, &mut self.next_post_break_tails);

        // Seed new instances as single-value ranges.
        for &(counter, origin, value) in t.seeds.iter() {
            self.counters.insert(counter.idx(), origin, value, value);
        }
        // Break seeds gated on the triggering counter.
        for &(trigger, counter, origin, value) in t.break_seeds.iter() {
            if counter_broke & (1u64 << trigger.idx()) != 0 {
                self.counters.insert(counter.idx(), origin, value, value);
            }
        }

        // Update has_live_instances flag.
        self.has_live_instances = self.counters.any_live();
    }

    fn step_from_dead(&mut self, byte: u8) {
        if self.has_live_instances {
            self.counters.clear();
            self.has_live_instances = false;
        }
        self.match_at_end = false;
        self.post_break_tails.clear();

        let trans = self.cache.populate(
            self.memory,
            DfaStateId::DEAD,
            byte,
            self.regex,
            self.analysis,
        );

        // From DEAD, no instances exist, so use no_break successor.
        self.current = trans.no_break;

        if !trans.is_counting {
            if trans.no_break_is_match {
                self.ever_matched = true;
            }
            if trans.no_break_is_match_at_end {
                self.match_at_end = true;
            }
        }

        // Seed instances.
        for &(counter, origin, value) in trans.seeds.iter() {
            self.counters.insert(counter.idx(), origin, value, value);
        }
        if !trans.seeds.is_empty() {
            self.has_live_instances = true;
        }
    }

    #[inline(always)]
    pub fn chunk(&mut self, input: &[u8]) {
        if self.ever_matched {
            return;
        }

        let input = match self.prefilter {
            Prefilter::None => input,
            Prefilter::Memchr1(b) => {
                if let Some(idx) = memchr::memchr(b, input) {
                    self.prefilter = Prefilter::None;
                    &input[idx..]
                } else {
                    return;
                }
            }
            Prefilter::Memchr2(b1, b2) => {
                if let Some(idx) = memchr::memchr2(b1, b2, input) {
                    self.prefilter = Prefilter::None;
                    &input[idx..]
                } else {
                    return;
                }
            }
            Prefilter::Memchr3(b1, b2, b3) => {
                if let Some(idx) = memchr::memchr3(b1, b2, b3, input) {
                    self.prefilter = Prefilter::None;
                    &input[idx..]
                } else {
                    return;
                }
            }
            Prefilter::Range(lo, hi) => {
                if let Some(idx) = crate::memrange::memrange(lo, hi, input) {
                    self.prefilter = Prefilter::None;
                    &input[idx..]
                } else {
                    return;
                }
            }
        };

        let stride = self.cache.stride;

        for &b in input {
            if self.ever_matched {
                return;
            }

            // --- Inline fast path ---
            if self.current == DfaStateId::DEAD {
                self.step_from_dead(b);
                continue;
            }

            let class = if stride == 256 {
                b as usize
            } else {
                self.regex.byte_classes[b as usize] as usize
            };
            let slot = self.current.idx() * stride + class;
            if self.cache.transitions[slot].no_break == DfaStateId::UNPOPULATED {
                let trans =
                    self.cache
                        .populate(self.memory, self.current, b, self.regex, self.analysis);
                self.cache.transitions[slot] = trans;
            }
            let t = &self.cache.transitions[slot];

            if !t.is_counting
                && t.seeds.is_empty()
                && !self.has_live_instances
                && self.post_break_tails.is_empty()
            {
                self.current = t.no_break;
                self.match_at_end = t.no_break_is_match_at_end;
                if t.no_break_is_match {
                    self.ever_matched = true;
                }
                continue;
            }

            self.step_slow(slot);
        }
    }

    pub fn finish(self) -> bool {
        if self.ever_matched {
            return true;
        }
        if self.match_at_end {
            return true;
        }
        // Note: we intentionally do NOT check `state.is_match_at_end` here.
        // For non-counting transitions, `self.match_at_end` already captures
        // the DFA state's match-at-end flag (set in the fast path or
        // step_slow).  For counting transitions, the DFA state's flag is
        // overly optimistic: the with_break closure follows ALL CInc break
        // paths (including downstream counters that haven't actually reached
        // their min), so `is_match_at_end` may be set even when no valid
        // break chain leads to `$ → Match`.  Only `self.match_at_end`
        // (computed per-instance in step_slow) correctly reflects whether a
        // specific counter instance actually broke with enough value.
        if self.current != DfaStateId::DEAD {
            let state = &self.cache.inner.states[self.current.idx()];
            if state.resolve_deferred_at_end(self.regex) {
                return true;
            }
        }
        false
    }

    #[allow(dead_code)]
    pub fn ismatch(&self) -> bool {
        self.ever_matched
    }
}

impl fmt::Debug for Tier3DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier3DfaMatcher")
            .field("current", &self.current)
            .field("ever_matched", &self.ever_matched)
            .finish()
    }
}
