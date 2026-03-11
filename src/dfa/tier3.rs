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
//! transition maps each origin to an [`OriginAction`] (`Option<Tier3OriginKind>`)
//! that describes what happens structurally when a byte is consumed there.

use std::fmt;

use crate::{
    AssertEval, AssertKind, CounterIdx, Prefilter, Regex, State, StateIdx, byte_match_ci,
    is_word_byte,
};

use super::{DfaCache, DfaMemory, DfaState, DfaStateId};

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

    /// Maximum number of live [`Instance`] entries per counter in the
    /// per-instance fallback path.  Equals `max(counter_max * body_origins)`
    /// across all counters.  Used as the stride for the flat
    /// [`InstanceCounters`] storage.  Zero when there are no counters or
    /// when [`all_counters_rangeable`](Self::all_counters_rangeable) is true.
    pub(crate) max_instance_stride: usize,

    /// Per-consuming-state flag: true if the NFA state's byte-consumption
    /// target reaches `$ → Match` through epsilon transitions.
    ///
    /// Indexed by NFA state index.  Only meaningful for consuming states;
    /// `false` for all others.  Used by the post-break tail tracker to
    /// detect match-at-end when a tail's action is `None` (dead — its
    /// target has no further consuming states but may have `$ → Match`).
    pub(crate) target_is_match_at_end: Box<[bool]>,

    /// Per-NFA-state flag: true if the state is reachable from the start
    /// state by following only CInc *continue* paths (not break paths).
    ///
    /// Used by `counter_free_match_at_end` to distinguish consuming
    /// states that are always present (no counter break required) from
    /// states that only enter the DFA state after a counter breaks.
    /// Only the former can contribute to counter-free match-at-end.
    pub(crate) reachable_without_break: Box<[bool]>,

    /// `true` when all `CounterInstance` nodes are epsilon-reachable from
    /// the start state — i.e. reachable by following only `Split` and
    /// `CounterInstance` edges (no consuming states, no `Assert` nodes,
    /// no `CounterIncrement` nodes).
    ///
    /// When this holds, the unanchored start-state re-seeding injects
    /// value 0 into every counter body on every input byte, guaranteeing
    /// that values at each origin always form a contiguous range anchored
    /// at 0.  This enables the range-compressed fast path
    /// ([`RangeCounters`]).
    ///
    /// When `false`, some counter body is gated behind a consuming prefix
    /// or an assertion (e.g. `^`, `\b`) that blocks re-seeding.  Values
    /// can become non-contiguous and range compression would
    /// over-approximate, so the per-instance fallback is used instead.
    pub(crate) all_counters_rangeable: bool,
}

/// What happens structurally when a byte is consumed at a given target.
///
/// [`OriginAction`] is a type alias for `Option<Tier3OriginKind>`: dead targets
/// are represented as `None` in both `Tier3Analysis::targets` and `Transition::origin_actions`.
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
        /// Deferred assertion NFA state indices on the break path to
        /// `$ → Match`.  Non-empty when the break path includes
        /// assertions like `\b` or `\B` before reaching `$ → Match`.
        /// These must be evaluated at end-of-input before confirming
        /// the match.  Empty when `break_is_match_at_end` comes from
        /// a pure `$ → Match` path (no deferred assertions).
        break_deferred_asserts: Box<[StateIdx]>,
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
    start: StateIdx,
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
    //
    // Also compute `max_instance_stride`: for each counter, the worst-case
    // number of live `Instance` entries is `counter_max * body_origins`.
    // The maximum across all counters becomes the stride for the flat
    // per-instance fallback storage.
    let mut max_body_origins: usize = 0;
    let mut max_instance_stride: usize = 0;
    for state in states.iter() {
        if let State::CounterInstance { counter, out } = *state {
            let mut count: usize = 0;
            let mut counter_max: usize = 0;
            let mut stack = vec![out];
            let mut visited = vec![false; n];
            while let Some(idx) = stack.pop() {
                let i = idx.idx();
                if visited[i] {
                    continue;
                }
                visited[i] = true;
                match states[idx] {
                    State::CounterIncrement {
                        counter: c, max, ..
                    } if c == counter => {
                        counter_max = max;
                    }
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
            // For unbounded counters (max = usize::MAX), cap the stride
            // to avoid overflow.  See `UNBOUNDED_INSTANCE_CAP`.
            let per_counter = counter_max
                .saturating_mul(count)
                .min(UNBOUNDED_INSTANCE_CAP);
            max_instance_stride = max_instance_stride.max(per_counter);
        }
    }

    // -- Step 6: compute per-consuming-state target_is_match_at_end ----------
    // For each consuming NFA state, check if its byte-consumption target
    // reaches `$ → Match` through epsilon transitions.  This is used by
    // the post-break tail tracker: when a tail's action is `None` (dead —
    // no further consuming states), this flag tells us whether the target
    // state is `$ → Match`.
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

    // -- Step 7: compute reachable_without_break --------------------------------
    // Transitive closure from start following ALL paths EXCEPT CInc break
    // paths.  Mark consuming states reachable without any counter break.
    // States only reachable via CInc break paths are counter-dependent
    // and should NOT contribute to counter_free_match_at_end.
    //
    // Unlike a single epsilon closure, this walk also follows consuming
    // states' `.out` targets so that multi-byte chains (e.g. unrolled
    // `[0-9]{1,3}\.[0-9]{1,3}...`) are fully traversed.  The `visited`
    // array ensures each state is processed at most once.
    let mut rwb = vec![false; n];
    {
        let mut visited = vec![false; n];
        let mut stack = vec![start];
        while let Some(idx) = stack.pop() {
            let i = idx.idx();
            if i >= n || visited[i] {
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
                State::CounterIncrement { out, .. } => {
                    // Follow only the continue path (out), NOT the break
                    // path (out1).  States behind out1 are counter-dependent.
                    stack.push(out);
                }
                State::Byte { out, .. }
                | State::ByteCI { out, .. }
                | State::ByteClass { out, .. } => {
                    rwb[i] = true;
                    // Follow .out to reach consuming states deeper in the
                    // NFA graph (e.g. unrolled repetition chains).
                    stack.push(out);
                }
                State::ByteTable { table } => {
                    rwb[i] = true;
                    // ByteTable dispatches to multiple targets — push all
                    // non-NONE entries.
                    for &target in &byte_tables[table].0 {
                        if target != StateIdx::NONE {
                            stack.push(target);
                        }
                    }
                }
                State::Match => {}
            }
        }
    }

    // -- Step 8: epsilon-reachability of all CI nodes ----------------------------
    // Walk from `start` following only Split and CounterInstance edges
    // (no consuming states, no Assert, no CInc).  If every CI in the NFA
    // is reachable this way, the unanchored loop re-seeds value 0 into
    // every counter body on every byte, so range compression is safe.
    let all_counters_rangeable = {
        let mut visited = vec![false; n];
        let mut stack = vec![start];
        let mut ci_reached = vec![false; n];
        while let Some(idx) = stack.pop() {
            let i = idx.idx();
            if i >= n || visited[i] {
                continue;
            }
            visited[i] = true;
            match states[idx] {
                State::Split { out, out1 } => {
                    stack.push(out1);
                    stack.push(out);
                }
                State::CounterInstance { out, .. } => {
                    ci_reached[i] = true;
                    stack.push(out);
                }
                // Stop at consuming states, Assert, CInc, Match —
                // these break the epsilon-only path.
                _ => {}
            }
        }
        // Every CI node must have been reached.
        states
            .iter()
            .enumerate()
            .all(|(i, s)| !matches!(s, State::CounterInstance { .. }) || ci_reached[i])
    };

    Tier3Analysis {
        targets: targets_vec.into_boxed_slice(),
        ci_origins: ci_origins_vec.into_boxed_slice(),
        break_seeds: break_seeds.into_boxed_slice(),
        max_body_origins,
        target_is_match_at_end: target_mae.into_boxed_slice(),
        reachable_without_break: rwb.into_boxed_slice(),
        max_instance_stride,
        all_counters_rangeable,
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
    /// Seeds applied BEFORE counter increment.
    ///
    /// From Phase 1 deferred-assertion resolution on counting transitions
    /// with L=1 counter bodies.  The resolved path consumed the body byte,
    /// so the counter must be non-empty when increment fires.  Value is 0
    /// because the increment itself provides the +1.
    pre_seeds: Box<[(CounterIdx, StateIdx, u32)]>,
    /// New counter instances from CI nodes reachable WITHOUT following
    /// any CInc break path (always applied after counter increment).
    /// The third element is the initial counter value (0 for fresh seeds,
    /// 1 for seeds from resolved deferred assertions on non-counting
    /// transitions whose origin already consumed the resolving byte).
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
    /// True if the no-break closure's `is_match_at_end` comes from
    /// counter-free paths (targets of `reachable_without_break` origins,
    /// or epsilon `$ → Match` paths from the start state).
    ///
    /// Used in `finish()` instead of the raw DFA state's `is_match_at_end`,
    /// which may include `$ → Match` from counter-dependent origins that
    /// entered the DFA state via previous counter breaks (Bug 13).
    nb_counter_free_mae: bool,
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
            pre_seeds: Box::new([]),
            seeds: Box::new([]),
            break_seeds: Box::new([]),
            origin_keys: Box::new([]),
            origin_actions: Box::new([]),
            counter_free_match_at_end: false,
            nb_counter_free_mae: false,
        }
    }
}

/// Per-origin action in a cached [`Transition`].
///
/// `None` means the byte did not match at this origin — the instance
/// dies.  `Some(kind)` carries the structural outcome (advance or
/// increment) from [`Tier3Analysis::targets`].
type OriginAction = Option<Tier3OriginKind>;

// ---------------------------------------------------------------------------
// CounterStorage trait — abstracts over range-compressed and per-instance paths
// ---------------------------------------------------------------------------

/// Trait abstracting the counter-specific operations that differ between
/// range-compressed and per-instance slow paths.
///
/// [`RangeCounters`] implements this with range-compressed entries (one
/// `[min_val, max_val]` range per origin), while [`InstanceCounters`]
/// implements it with individual `(value, origin)` pairs.  The
/// [`step_slow_impl`] macro uses these methods for the operations that
/// diverge between the two paths; shared operations (clear, entries,
/// num_counters, any_live) are called directly as inherent methods
/// since both storage types provide them with identical signatures.
///
/// All methods are `#[inline]` to ensure monomorphization produces code
/// equivalent to the previous hand-written per-path methods.
///
/// Methods that logically operate on entries rather than the storage
/// (e.g. `can_continue`, `can_break`) take `&self` to enable
/// trait-method dispatch from the macro, where the concrete type is
/// inferred from the field receiver.
trait CounterStorage {
    /// The element type stored per counter slot.  Must have an `origin:
    /// StateIdx` field accessible in the macro.
    type Entry;

    /// Seed a new entry at counter `ci` with the given origin and value.
    ///
    /// For [`RangeCounters`] this inserts a single-value range `[v, v]`
    /// (merging if the origin already exists).  For [`InstanceCounters`]
    /// this pushes a new instance after a dedup check.
    fn seed(&mut self, ci: usize, origin: StateIdx, value: u32);

    /// Advance an existing entry to a new origin, preserving its
    /// value / range.  Inserts into `self` (the "next" buffer).
    fn advance(&mut self, ci: usize, entry: &Self::Entry, new_origin: StateIdx);

    /// Returns true if the entry has any value that can continue
    /// (i.e. value + 1 < max after incrementing).
    fn can_continue(&self, entry: &Self::Entry, max: u32) -> bool;

    /// Returns true if the entry has any value that triggers a break
    /// (i.e. value + 1 >= min after incrementing).
    fn can_break(&self, entry: &Self::Entry, min: u32) -> bool;

    /// Insert the continued (incremented) portion of an entry at
    /// `new_origin`.  Only called when [`can_continue`](Self::can_continue)
    /// returned true.
    ///
    /// For ranges: inserts `[min_val+1, min(max_val+1, max-1)]`.
    /// For instances: inserts `Instance { value: value+1, origin }`.
    fn insert_continued(&mut self, ci: usize, entry: &Self::Entry, new_origin: StateIdx, max: u32);
}

// ---------------------------------------------------------------------------
// Per-instance tracking (fallback when range compression is unsound)
// ---------------------------------------------------------------------------

/// A single active counter instance.
///
/// Used by the per-instance fallback path when
/// [`Tier3Analysis::all_counters_rangeable`] is `false` — i.e. when at
/// least one counter body is not epsilon-reachable from the start state
/// and value-0 seeds are not injected on every byte.
#[derive(Clone, Debug)]
struct Instance {
    /// Number of completed iterations (0 when freshly seeded at CI).
    value: u32,
    /// The NFA consuming state this instance is waiting at.
    origin: StateIdx,
}

/// Flat storage for per-instance counter tracking.
///
/// Uses a single flat `Vec<Instance>` with
/// fixed-stride slots per counter.  Counter `i` occupies
/// `data[i * stride .. i * stride + counts[i]]` where `stride` is
/// [`Tier3Analysis::max_instance_stride`].  For bounded repetitions this
/// equals `counter_max * body_origins`; for unbounded repetitions the
/// stride is capped at a compile-time constant, and [`push`](Self::push)
/// is a no-op when the slot is full (safe because unbounded counters
/// break at every increment once `value >= min`, so excess instances
/// are redundant).
///
/// This eliminates inner `Vec` heap allocations, making memory reuse
/// trivial: [`clear`](Self::clear) just zeroes the `counts` array.
/// Double-buffering via [`std::mem::swap`] swaps two flat buffers in O(1).
struct InstanceCounters {
    /// Flat backing storage.  Length = `num_counters * stride`.
    data: Vec<Instance>,
    /// Number of live entries per counter.
    counts: Vec<u32>,
    /// Fixed number of slots per counter (= `max_instance_stride`).
    stride: usize,
}

/// Cap for per-counter instance slots when the repetition is unbounded.
///
/// Unbounded repetitions (`{n,}`, `+`, `*`) have `max = usize::MAX`,
/// so `max * body_origins` overflows.  We cap at a generous value that
/// exceeds any realistic number of live instances for a single counter
/// in the non-rangeable path (where seeds are NOT injected every byte).
const UNBOUNDED_INSTANCE_CAP: usize = 4096;

impl InstanceCounters {
    /// Create a new `InstanceCounters` for `num_counters` counters with
    /// the given stride (max instances per counter).
    fn new(num_counters: usize, stride: usize) -> Self {
        let total = num_counters * stride;
        let mut data = Vec::with_capacity(total);
        // Fill with dummy entries — only `counts[i]` entries are live.
        data.resize(
            total,
            Instance {
                value: 0,
                origin: StateIdx::NONE,
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
    fn entries(&self, ci: usize) -> &[Instance] {
        let base = ci * self.stride;
        &self.data[base..base + self.counts[ci] as usize]
    }

    /// Returns true if any counter has live entries.
    #[inline]
    fn any_live(&self) -> bool {
        self.counts.iter().any(|&c| c != 0)
    }

    /// Push an instance into counter `ci`.
    ///
    /// If the counter's slot is full (can only happen for unbounded
    /// repetitions capped at [`UNBOUNDED_INSTANCE_CAP`]), the push is
    /// silently dropped.  This is safe because unbounded counters break
    /// at every increment once `value >= min`, so the oldest instances
    /// (which would have already broken) are the ones effectively lost.
    #[inline]
    fn push(&mut self, ci: usize, inst: Instance) {
        let count = self.counts[ci] as usize;
        if count >= self.stride {
            return;
        }
        let base = ci * self.stride;
        self.data[base + count] = inst;
        self.counts[ci] = (count + 1) as u32;
    }

    /// Returns true if counter `ci` already contains an instance with the
    /// given `value` and `origin`.
    #[inline]
    fn contains(&self, ci: usize, value: u32, origin: StateIdx) -> bool {
        let base = ci * self.stride;
        let count = self.counts[ci] as usize;
        for i in 0..count {
            let inst = &self.data[base + i];
            if inst.value == value && inst.origin == origin {
                return true;
            }
        }
        false
    }
}

impl CounterStorage for InstanceCounters {
    type Entry = Instance;

    #[inline]
    fn seed(&mut self, ci: usize, origin: StateIdx, value: u32) {
        if !self.contains(ci, value, origin) {
            self.push(ci, Instance { value, origin });
        }
    }

    #[inline]
    fn advance(&mut self, ci: usize, entry: &Instance, new_origin: StateIdx) {
        if !self.contains(ci, entry.value, new_origin) {
            self.push(
                ci,
                Instance {
                    value: entry.value,
                    origin: new_origin,
                },
            );
        }
    }

    #[inline]
    fn can_continue(&self, entry: &Instance, max: u32) -> bool {
        entry.value + 1 < max
    }

    #[inline]
    fn can_break(&self, entry: &Instance, min: u32) -> bool {
        entry.value + 1 >= min
    }

    #[inline]
    fn insert_continued(&mut self, ci: usize, entry: &Instance, new_origin: StateIdx, _max: u32) {
        if !self.contains(ci, entry.value + 1, new_origin) {
            self.push(
                ci,
                Instance {
                    value: entry.value + 1,
                    origin: new_origin,
                },
            );
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
/// ## When range compression is sound (contiguity proof)
///
/// Range compression is sound when **every `CounterInstance` node is
/// epsilon-reachable from `regex.start`** — i.e. reachable via only
/// `Split` and `CounterInstance` edges, with no consuming states, no
/// `Assert` nodes, and no `CounterIncrement` nodes on the path.  When
/// this holds, the unanchored `.*` loop at the start injects a fresh
/// seed with value 0 at the counter entry on every byte.  The argument
/// then proceeds by induction:
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
///
/// ## When range compression is unsound
///
/// The proof breaks down when a `CounterInstance` node is **not**
/// epsilon-reachable from `regex.start`.  In that case, value-0 seeds
/// are not injected on every byte, so ranges need not be anchored at 0
/// and can fragment.  Merging disjoint ranges via min/max creates
/// "phantom" values that do not correspond to any real NFA thread,
/// causing false positives.
///
/// Known cases where epsilon-reachability fails:
///
/// - **`^`-anchored patterns** (e.g. `^(a|aaa){4,4}b`): the
///   `Assert(Start)` node blocks re-seeding after byte 0.
/// - **Consuming prefix before counter** (e.g. `c(a|aaa){4,4}b`):
///   seeds only enter the counter when the prefix byte `c` is consumed.
/// - **Any assertion on the path** to a `CounterInstance` node that can
///   evaluate to `Fail` on some bytes.
///
/// The [`Tier3Analysis::all_counters_rangeable`] flag detects these
/// cases at compile time.  When `false`, the matcher falls back to
/// per-instance tracking via [`Instance`] (O(max\_count) per byte
/// but always correct).
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
/// Uses a single flat `Vec<InstanceRange>`
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
    ///
    /// # Panics
    ///
    /// Panics if `stride > 255`.  The per-counter entry count is stored
    /// as `u8` for compactness (one byte of overhead per counter instead
    /// of eight).  A stride of 256+ would overflow the `u8` count on
    /// insertion.  In practice this requires ≥256 distinct consuming NFA
    /// states inside a single counter body, which is unreachable for any
    /// realistic pattern — alternations of single bytes compile to one
    /// `ByteClass`, and even deeply nested multi-byte alternations
    /// rarely exceed a handful of consuming positions.
    fn new(num_counters: usize, stride: usize) -> Self {
        assert!(
            stride <= u8::MAX as usize,
            "RangeCounters: stride {} exceeds u8::MAX (255); the pattern has \
             too many consuming NFA states in a single counter body",
            stride,
        );
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

impl CounterStorage for RangeCounters {
    type Entry = InstanceRange;

    #[inline]
    fn seed(&mut self, ci: usize, origin: StateIdx, value: u32) {
        self.insert(ci, origin, value, value);
    }

    #[inline]
    fn advance(&mut self, ci: usize, entry: &InstanceRange, new_origin: StateIdx) {
        self.insert(ci, new_origin, entry.min_val, entry.max_val);
    }

    #[inline]
    fn can_continue(&self, entry: &InstanceRange, max: u32) -> bool {
        // Incremented range is [min_val+1, max_val+1].
        // Continue cap is max - 1 (highest value that can stay in the loop).
        // Can continue if new_min <= continue_cap, i.e. min_val < max - 1.
        entry.min_val < max - 1
    }

    #[inline]
    fn can_break(&self, entry: &InstanceRange, min: u32) -> bool {
        // Any value in the incremented range [min_val+1, max_val+1] >= min.
        entry.max_val + 1 >= min
    }

    #[inline]
    fn insert_continued(
        &mut self,
        ci: usize,
        entry: &InstanceRange,
        new_origin: StateIdx,
        max: u32,
    ) {
        let new_min = entry.min_val + 1;
        let new_max = entry.max_val + 1;
        let continue_cap = max - 1;
        let cont_max = new_max.min(continue_cap);
        self.insert(ci, new_origin, new_min, cont_max);
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
            let action = analysis.targets[target.idx()].clone();
            origin_keys.push(origin);
            origin_actions.push(action);
        }

        // Compute seed initial values for non-counting transitions.
        // For counting transitions, resolved seeds on L=1 bodies go to
        // pre_seeds (value 0) — the increment provides the +1.
        // For non-counting transitions, bump to value 1 since there's
        // no increment to provide the +1.
        if !is_counting {
            for rs in &mut resolved_seeds {
                if let Some(pos) = origin_keys.iter().position(|&k| k == rs.1)
                    && matches!(origin_actions[pos], Some(Tier3OriginKind::Increment { .. }))
                {
                    rs.2 = 1;
                }
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

            let nb_counter_free_mae = self.counter_free_nb_mae(
                memory,
                nb_mae,
                &targets_per_origin,
                regex,
                analysis,
                byte,
            );

            // Build pre_seeds: resolved seeds on counting transitions whose
            // origin had an Increment action (L=1 body consumed on this
            // transition via deferred assertion resolution).
            let pre_seeds: Vec<(CounterIdx, StateIdx, u32)> = resolved_seeds
                .iter()
                .filter(|rs| {
                    origin_keys
                        .iter()
                        .position(|&k| k == rs.1)
                        .is_some_and(|pos| {
                            matches!(origin_actions[pos], Some(Tier3OriginKind::Increment { .. }))
                        })
                })
                .cloned()
                .collect();

            // Unconditional seeds: reachable without following CInc break
            // paths (from the no_break closure).  Resolved seeds that became
            // pre_seeds are excluded to avoid double-seeding.
            let mut seeds: Vec<(CounterIdx, StateIdx, u32)> = cr_nb
                .seed_instances
                .iter()
                .map(|&(c, s)| (c, s, 0u32))
                .collect();
            // Only merge resolved seeds that are NOT pre_seeds.
            for s in &resolved_seeds {
                let is_pre = pre_seeds.iter().any(|p| p.0 == s.0 && p.1 == s.1);
                if !is_pre && !seeds.iter().any(|e| e.0 == s.0 && e.1 == s.1 && e.2 == s.2) {
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
            // target reaches `$ → Match` without going through CInc,
            // AND the origin itself is reachable from the start state
            // without following any CInc break path.
            //
            // The second condition (reachable_without_break) is critical
            // for multi-counter patterns: an origin like Byte('a') in
            // `^c{2,12}.{6,6}(a?)?$` enters the DFA state only after
            // both counters break.  Its `$ → Match` path is locally
            // counter-free, but the origin is counter-dependent.
            // Propagating match_at_end from it would bypass per-instance
            // counter checks, causing false positives (Bug 13).
            //
            // We check all non-Increment origins (`None` and `Advance`).
            // `None` (dead) means the target has no consuming states at all;
            // `Advance` means the target has consuming states but may
            // ALSO reach `$ → Match` via epsilon transitions.  In both
            // cases, the `$ → Match` path doesn't cross any CInc.
            let counter_free_mae =
                origin_keys
                    .iter()
                    .zip(origin_actions.iter())
                    .any(|(&origin, action)| {
                        !matches!(action, Some(Tier3OriginKind::Increment { .. }))
                            && analysis.target_is_match_at_end[origin.idx()]
                            && analysis.reachable_without_break[origin.idx()]
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
                pre_seeds: pre_seeds.into(),
                seeds: seeds.into(),
                break_seeds: break_seeds.into(),
                origin_keys: origin_keys.into_boxed_slice(),
                origin_actions: origin_actions.into_boxed_slice(),
                counter_free_match_at_end: counter_free_mae,
                nb_counter_free_mae,
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

            let nb_counter_free_mae =
                self.counter_free_nb_mae(memory, mae, &targets_per_origin, regex, analysis, byte);

            Transition {
                no_break: id,
                no_break_is_match: m || resolved_is_match,
                no_break_is_match_at_end: mae,
                with_break: id,
                with_break_is_match: m || resolved_is_match,
                with_break_is_match_at_end: mae,
                is_counting: false,
                pre_seeds: Box::new([]),
                seeds: seeds.into(),
                break_seeds: Box::new([]),
                origin_keys: origin_keys.into_boxed_slice(),
                origin_actions: origin_actions.into_boxed_slice(),
                counter_free_match_at_end: false, // Not used for non-counting transitions.
                nb_counter_free_mae,
            }
        }
    }

    /// Compute counter-free no-break `is_match_at_end`.
    ///
    /// Filters `targets_per_origin` to only counter-free origins
    /// ([`Tier3Analysis::reachable_without_break`]), runs an epsilon closure
    /// on those targets, and returns `is_match_at_end` from the result.
    /// Returns `false` immediately when `mae` is `false` (the full no-break
    /// closure already has no match-at-end, so the filtered version can't
    /// either).
    ///
    /// Used by both the counting and non-counting branches of [`populate`]
    /// to avoid false positives from origins that entered the DFA state
    /// via previous counter breaks (Bug 13).
    fn counter_free_nb_mae(
        &mut self,
        memory: &mut DfaMemory,
        mae: bool,
        targets_per_origin: &[(StateIdx, Vec<StateIdx>)],
        regex: &Regex,
        analysis: &Tier3Analysis,
        byte: u8,
    ) -> bool {
        if !mae {
            return false;
        }
        let cf_targets: Vec<StateIdx> = targets_per_origin
            .iter()
            .filter(|(origin, _)| analysis.reachable_without_break[origin.idx()])
            .flat_map(|(_, ts)| ts.iter().copied())
            .collect();
        // Note: we intentionally do NOT bail out when cf_targets is empty.
        // The re-seeded `regex.start` is always counter-free (it represents
        // the unanchored match restart, not a counter break), so its epsilon
        // closure may contribute `$ → Match` even when no counter-free
        // consuming origins exist.  The previous guard checked
        // `reachable_without_break[regex.start.idx()]`, but that array only
        // marks *consuming* states; the start state (typically a Split) is
        // never marked, causing false negatives for patterns like
        // `([b-ed-ie-f].{4,43})?$` where the `?`-skip to `$` is the only
        // counter-free path to Match.
        let cr_cf = self.epsilon_closure(
            memory,
            cf_targets
                .iter()
                .copied()
                .chain(std::iter::once(regex.start)),
            regex,
            analysis,
            false,
            Some(byte),
            None,
            false,
        );
        cr_cf.is_match_at_end
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
        let bc = break_closure(&cinc_break_outs, states, can_reach_match);

        Some(Tier3OriginKind::Increment {
            advance_origins: advance_origins.into_boxed_slice(),
            min: min as u32,
            max: max as u32,
            continue_origins: continue_origins.into_boxed_slice(),
            break_is_match: bc.is_match,
            break_is_match_at_end: bc.is_match_at_end,
            break_deferred_asserts: bc.deferred_asserts,
            // Populated in a post-pass by compute_tier3_analysis after all
            // targets are known (break_consuming_tails needs the targets array).
            break_consuming_states: Box::new([]),
        })
    } else if advance_origins.is_empty() {
        None // Dead — no consuming states reachable.
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
/// Result of walking the CInc break-path epsilon closure.
struct BreakClosureResult {
    /// True if `Match` is directly reachable without going through any
    /// deferred assertion (\b, \B, etc.).
    is_match: bool,
    /// True if `$ → Match` is reachable without going through any
    /// deferred assertion.  At runtime, this fires unconditionally when
    /// the counter breaks with enough value.
    is_match_at_end: bool,
    /// Deferred assertion NFA state indices found on break paths.
    ///
    /// These are assertions (e.g. `\b`, `\B`) encountered while walking
    /// epsilon transitions from CInc break outputs.  They may lead to
    /// either `Match` directly (e.g. `\b → Match`) or to `$ → Match`
    /// (e.g. `\b → $ → Match`).
    ///
    /// At runtime, when the counter breaks with enough value, these are
    /// stored in `verified_deferred_asserts` and evaluated at end-of-input
    /// in `finish()` using `can_reach_match_at_end()`, which handles both
    /// `Match` and `$ → Match` paths.
    deferred_asserts: Box<[StateIdx]>,
}

fn break_closure(
    break_seeds: &[StateIdx],
    states: &[State],
    can_reach_match: &[bool],
) -> BreakClosureResult {
    let mut is_match = false;
    let mut is_match_at_end = false;
    let mut deferred_asserts = Vec::new();

    // Walk epsilon transitions from break seeds.
    //
    // Track whether we've passed through a deferred assertion to
    // correctly attribute `Match` and `$ → Match` discoveries:
    // - Pure paths (no deferred assertion): set is_match / is_match_at_end
    // - Paths through deferred assertions: record the assertion indices
    //   for runtime evaluation; don't set the static flags.
    let mut stack: Vec<(StateIdx, bool)> = break_seeds.iter().map(|&s| (s, false)).collect();

    let n = states.len();
    let mut visited_pure = vec![false; n];
    let mut visited_deferred = vec![false; n];

    while let Some((idx, through_deferred)) = stack.pop() {
        let i = idx.idx();
        let visited = if through_deferred {
            &mut visited_deferred
        } else {
            &mut visited_pure
        };
        if visited[i] {
            continue;
        }
        visited[i] = true;

        match states[idx] {
            State::Split { out, out1 } => {
                stack.push((out1, through_deferred));
                stack.push((out, through_deferred));
            }
            State::Assert { kind, out } => {
                if kind == AssertKind::End {
                    if !through_deferred && can_reach_match[out.idx()] {
                        // Pure $ → Match — safe to fire unconditionally.
                        is_match_at_end = true;
                    }
                    // $ → Match through a deferred assertion is handled
                    // by the deferred_asserts list + finish() evaluation.
                } else if can_reach_match[out.idx()] {
                    if !through_deferred {
                        // First deferred assertion on this path — record
                        // it as the entry point.  Subsequent assertions
                        // deeper in the chain (e.g. \B → \b → $) are
                        // evaluated dynamically by can_reach_match_at_end
                        // when this entry point is checked in finish().
                        deferred_asserts.push(idx);
                    }
                    // Follow through regardless, so we detect Match /
                    // $ → Match reachability from deeper in the chain.
                    stack.push((out, true));
                }
            }
            State::Match => {
                if !through_deferred {
                    is_match = true;
                }
                // Match through a deferred assertion: handled by the
                // deferred_asserts list + finish() evaluation.
            }
            // Do NOT follow through CounterInstance — downstream
            // counters have not accumulated their required count.
            State::CounterInstance { .. } => {}
            _ => {}
        }
    }

    deferred_asserts.sort_unstable_by_key(|s| s.0);
    deferred_asserts.dedup();
    BreakClosureResult {
        is_match,
        is_match_at_end,
        deferred_asserts: deferred_asserts.into_boxed_slice(),
    }
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
/// Two instance-tracking strategies are available, selected once at
/// construction by [`Tier3Analysis::all_counters_rangeable`]:
///
/// - **Range-compressed** ([`RangeCounters`]):  O(num_origins) per byte.
///   Used when every `CounterInstance` is epsilon-reachable from start,
///   guaranteeing contiguous value ranges (see proof on [`InstanceRange`]).
/// - **Per-instance** ([`InstanceCounters`]):  O(live_instances) per byte.
///   Fallback when some counter is gated behind a consuming prefix or
///   assertion that blocks continuous seed injection.
pub struct Tier3DfaMatcher<'a> {
    cache: &'a mut Tier3DfaCache,
    memory: &'a mut DfaMemory,
    regex: &'a Regex,
    analysis: &'a Tier3Analysis,
    current: DfaStateId,
    /// The no-break DFA state from the last transition.  Used in `finish()`
    /// to resolve deferred assertions at end-of-input.
    ///
    /// The with-break DFA state includes deferred assertions (e.g. `\b`,
    /// `\B`) from ALL CInc break paths — including counters that haven't
    /// reached their minimum.  `resolve_deferred_at_end` on the with-break
    /// state would evaluate those assertions and follow through to
    /// `$ → Match`, causing false positives.  The no-break state only
    /// includes deferred assertions reachable without this transition's
    /// counter breaks, which is the counter-free subset safe for EOI
    /// resolution.  Counter-dependent `$ → Match` through deferred
    /// assertions is handled by per-instance `break_is_match_at_end`.
    no_break_current: DfaStateId,
    /// Counter-free subset of the no-break DFA state's `is_match_at_end`.
    /// Only includes `$ → Match` from targets of `reachable_without_break`
    /// origins.  Used in `finish()` instead of the raw DFA state flag
    /// to avoid false positives from counter-dependent origins (Bug 13).
    last_nb_counter_free_mae: bool,
    /// `true` when range-compressed instance tracking is active.
    use_ranges: bool,
    /// Range-compressed instance storage — flat per-counter slots.
    /// Active when `use_ranges` is true.
    ranged_counters: RangeCounters,
    /// Scratch buffer for the next ranged step (double-buffered).
    next_ranged: RangeCounters,
    /// Per-instance flat storage — fixed-stride slots per counter.
    /// Active when `use_ranges` is false.
    inst_counters: InstanceCounters,
    /// Scratch buffer for the next per-instance step (double-buffered).
    next_instances: InstanceCounters,
    ever_matched: bool,
    match_at_end: bool,
    has_live_instances: bool,
    /// NFA consuming states currently active from a counter break path.
    /// These track the post-counter "tail" (e.g. a trailing `.` or
    /// literal before `$ → Match`).  Updated each step: each tail is
    /// advanced through its [`OriginAction`].  When a tail's action is
    /// `None` (dead) and `target_is_match_at_end` is true, `match_at_end` is set.
    ///
    /// This replaces the use of DFA-level `is_match_at_end` for counting
    /// transitions, which is overly optimistic because the DFA state
    /// includes post-break states before counters have actually broken.
    post_break_tails: Vec<StateIdx>,
    /// Scratch buffer for advancing `post_break_tails` (double-buffered).
    next_post_break_tails: Vec<StateIdx>,
    /// Deferred assertion NFA states from counter break paths that fired
    /// during the last `step_slow()`.  These are assertions (e.g. `\b`)
    /// on the path to `$ → Match` from a counter that actually broke with
    /// enough value.  Evaluated in `finish()` at end-of-input.
    verified_deferred_asserts: Vec<StateIdx>,
    prefilter: Prefilter,
}

// ---------------------------------------------------------------------------
// step_slow_impl! — generates step_slow_ranged / step_slow_instances
// ---------------------------------------------------------------------------

/// Generates a `step_slow_*` method for a specific [`CounterStorage`] backend.
///
/// Both the range-compressed and per-instance slow paths share identical
/// control flow: advance post-break tails, apply pre-seeds, iterate
/// counter entries, select the DFA successor, swap buffers, and apply
/// seeds.  Only the counter storage operations differ (range merging vs
/// individual instance tracking), which are abstracted through the
/// [`CounterStorage`] trait.
///
/// Parameters:
/// - `$method`:  generated method name (`step_slow_ranged` or `step_slow_instances`)
/// - `$current`: field name for the current-step counter buffer
/// - `$next`:    field name for the next-step scratch buffer
macro_rules! step_slow_impl {
    ($method:ident, $current:ident, $next:ident) => {
        #[inline(never)]
        fn $method(&mut self, slot: usize) {
            let t = &self.cache.transitions[slot];

            // Reset next buffer and match flags.
            self.$next.clear();
            self.match_at_end = false;
            self.verified_deferred_asserts.clear();

            // Advance existing post-break tails through this transition.
            // Each tail is a consuming NFA state from a previous counter
            // break.  If it consumed this byte and its target reaches
            // `$ → Match`, set match_at_end.
            self.next_post_break_tails.clear();
            for &pbo in &self.post_break_tails {
                let pos = t.origin_keys.iter().position(|&k| k == pbo);
                match pos.map(|i| &t.origin_actions[i]) {
                    Some(Some(Tier3OriginKind::Advance { new_origins })) => {
                        for &new_o in new_origins.iter() {
                            if !self.next_post_break_tails.contains(&new_o) {
                                self.next_post_break_tails.push(new_o);
                            }
                        }
                        if self.analysis.target_is_match_at_end[pbo.idx()] {
                            self.match_at_end = true;
                        }
                    }
                    Some(Some(Tier3OriginKind::Increment { .. })) => {
                        // Tail hit a CInc — stop tracking.  The counter
                        // instance machinery handles match detection from
                        // here.
                    }
                    Some(None) => {
                        // `None` (dead) target.  Check $ → Match via precomputed flag.
                        if self.analysis.target_is_match_at_end[pbo.idx()] {
                            self.match_at_end = true;
                        }
                    }
                    None => {
                        // Origin not in transition — byte not accepted.
                    }
                }
            }

            // Apply pre_seeds BEFORE counter increment.
            for &(counter, origin, value) in t.pre_seeds.iter() {
                self.$current.seed(counter.idx(), origin, value);
            }

            let mut any_can_break = false;
            let num_counters = self.$current.num_counters();
            debug_assert!(num_counters <= 64);
            let mut counter_broke: u64 = 0;

            #[allow(clippy::needless_range_loop)]
            for c_idx in 0..num_counters {
                for entry in self.$current.entries(c_idx) {
                    let origin = entry.origin;
                    let action = t
                        .origin_keys
                        .iter()
                        .position(|&k| k == origin)
                        .and_then(|i| t.origin_actions[i].as_ref());

                    match action {
                        Some(Tier3OriginKind::Advance { new_origins }) => {
                            for &new_o in new_origins.iter() {
                                self.$next.advance(c_idx, entry, new_o);
                            }
                        }
                        None => {}
                        Some(Tier3OriginKind::Increment {
                            advance_origins,
                            min,
                            max,
                            continue_origins,
                            break_is_match,
                            break_is_match_at_end,
                            break_deferred_asserts,
                            break_consuming_states,
                        }) => {
                            // Advance-or-increment: entry survives at
                            // advance_origins with the same values.
                            for &new_o in advance_origins.iter() {
                                self.$next.advance(c_idx, entry, new_o);
                            }

                            // Continue: incremented entry stays in the loop.
                            if self.$current.can_continue(entry, *max) {
                                for &new_o in continue_origins.iter() {
                                    self.$next.insert_continued(c_idx, entry, new_o, *max);
                                }
                            }

                            // Break: entry has reached the minimum threshold.
                            if self.$current.can_break(entry, *min) {
                                any_can_break = true;
                                counter_broke |= 1u64 << c_idx;
                                if *break_is_match {
                                    self.ever_matched = true;
                                }
                                if *break_is_match_at_end {
                                    self.match_at_end = true;
                                }
                                if !break_deferred_asserts.is_empty() {
                                    for &da in break_deferred_asserts.iter() {
                                        if !self.verified_deferred_asserts.contains(&da) {
                                            self.verified_deferred_asserts.push(da);
                                        }
                                    }
                                }
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
            self.no_break_current = t.no_break;
            self.last_nb_counter_free_mae = t.nb_counter_free_mae;
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
                // For counting transitions, propagate only no_break_is_match
                // and counter_free_match_at_end.  The full
                // no_break_is_match_at_end may include $ → Match paths from
                // post-break origins that haven't met their counter minimums.
                // Counter-dependent paths are handled by per-instance break
                // checks and the post_break_tails mechanism.
                if t.no_break_is_match {
                    self.ever_matched = true;
                }
                if t.counter_free_match_at_end {
                    self.match_at_end = true;
                }
            }

            // Swap buffers (O(1) — just swaps flat Vec pointers).
            std::mem::swap(&mut self.$current, &mut self.$next);
            std::mem::swap(&mut self.post_break_tails, &mut self.next_post_break_tails);

            // Seed new instances.
            for &(counter, origin, value) in t.seeds.iter() {
                self.$current.seed(counter.idx(), origin, value);
            }
            // Break seeds gated on the triggering counter.
            for &(trigger, counter, origin, value) in t.break_seeds.iter() {
                if counter_broke & (1u64 << trigger.idx()) != 0 {
                    self.$current.seed(counter.idx(), origin, value);
                }
            }

            // Update has_live_instances flag.
            self.has_live_instances = self.$current.any_live();
        }
    };
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
        let use_ranges = analysis.all_counters_rangeable;

        let mut ranged_counters = RangeCounters::new(nc, stride);
        let next_ranged = RangeCounters::new(nc, stride);
        let inst_stride = analysis.max_instance_stride;
        let mut inst_counters = InstanceCounters::new(nc, inst_stride);
        let next_instances = InstanceCounters::new(nc, inst_stride);

        for &(counter, origin, value) in cache.start_seeds.iter() {
            if use_ranges {
                ranged_counters.insert(counter.idx(), origin, value, value);
            } else {
                inst_counters.push(counter.idx(), Instance { value, origin });
            }
        }

        let ever_matched = cache.inner.start_is_match;
        let match_at_end = cache.inner.start_is_match_at_end;
        let has_live_instances = !cache.start_seeds.is_empty();

        Tier3DfaMatcher {
            current: cache.inner.start_id,
            no_break_current: cache.inner.start_id,
            last_nb_counter_free_mae: cache.inner.start_is_match_at_end,
            use_ranges,
            ever_matched,
            match_at_end,
            has_live_instances,
            post_break_tails: Vec::new(),
            next_post_break_tails: Vec::new(),
            verified_deferred_asserts: Vec::new(),
            cache,
            memory,
            regex,
            analysis,
            ranged_counters,
            next_ranged,
            inst_counters,
            next_instances,
            prefilter: regex.prefilter,
        }
    }

    step_slow_impl!(step_slow_ranged, ranged_counters, next_ranged);
    step_slow_impl!(step_slow_instances, inst_counters, next_instances);

    fn step_from_dead(&mut self, byte: u8) {
        if self.has_live_instances {
            if self.use_ranges {
                self.ranged_counters.clear();
            } else {
                self.inst_counters.clear();
            }
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
        self.no_break_current = trans.no_break;
        self.last_nb_counter_free_mae = trans.nb_counter_free_mae;

        if !trans.is_counting {
            if trans.no_break_is_match {
                self.ever_matched = true;
            }
            if trans.no_break_is_match_at_end {
                self.match_at_end = true;
            }
        }

        // Seed instances.
        if self.use_ranges {
            for &(counter, origin, value) in trans.seeds.iter() {
                self.ranged_counters
                    .insert(counter.idx(), origin, value, value);
            }
        } else {
            for &(counter, origin, value) in trans.seeds.iter() {
                self.inst_counters
                    .push(counter.idx(), Instance { value, origin });
            }
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
                && t.pre_seeds.is_empty()
                && t.seeds.is_empty()
                && !self.has_live_instances
                && self.post_break_tails.is_empty()
            {
                self.current = t.no_break;
                self.no_break_current = t.no_break;
                self.last_nb_counter_free_mae = t.nb_counter_free_mae;
                // Use the counter-free filtered flag instead of the raw
                // no_break_is_match_at_end, which may include `$ → Match`
                // from counter-dependent origins (Bug 13).
                self.match_at_end = t.nb_counter_free_mae;
                if t.no_break_is_match {
                    self.ever_matched = true;
                }
                continue;
            }

            if self.use_ranges {
                self.step_slow_ranged(slot);
            } else {
                self.step_slow_instances(slot);
            }
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
        //
        // We use `no_break_current` (the no-break DFA state from the last
        // transition) for deferred assertion resolution.  The with-break
        // state includes deferred assertions from ALL CInc break paths
        // (e.g. `\b` after counter 1's CInc break in `.{6,39}.{7,31}\b$`).
        // Those assertions fire here even when the counter never reached
        // its minimum, causing false positives.  The no-break state only
        // includes deferred assertions reachable without this transition's
        // counter breaks — the counter-free subset safe for EOI resolution.
        //
        // For counters that actually broke with enough value and had a
        // pure `$ → Match` break path, per-instance `break_is_match_at_end`
        // in step_slow already sets `match_at_end` (checked above).
        //
        // Counter-free match-at-end: use the transition's filtered flag
        // instead of the raw DFA state's `is_match_at_end`.  The raw flag
        // includes `$ → Match` from ALL targets in the no-break closure,
        // including targets from counter-dependent origins that entered
        // the DFA state via previous counter breaks.  The filtered flag
        // only includes `$ → Match` from targets of `reachable_without_break`
        // origins, which are genuinely counter-free.
        //
        // This catches cases like `(0{2,2}|1*)$` on "0" where the start
        // closure's `1* → $ → Match` path is counter-free, but avoids
        // false positives from patterns like `^c{2,12}.{6,6}(a?)?$` on
        // "ccca" where the `(a?)?$` suffix is counter-dependent (Bug 13).
        if self.last_nb_counter_free_mae {
            return true;
        }
        // Deferred assertions from the no-break state.
        if self.no_break_current != DfaStateId::DEAD {
            let state = &self.cache.inner.states[self.no_break_current.idx()];
            if state.resolve_deferred_at_end(self.regex) {
                return true;
            }
        }
        // Counter-dependent deferred assertions: evaluate assertions from
        // counter break paths that actually broke with enough value.  These
        // are NFA Assert state indices (e.g. \b, \B) on the path to
        // `$ → Match` from a counter whose break condition was met.
        //
        // We use the with-break DFA state (`self.current`) for prev_byte
        // context, since the current DFA state reflects the actual match
        // position (including break paths that fired).
        if !self.verified_deferred_asserts.is_empty() && self.current != DfaStateId::DEAD {
            let state = &self.cache.inner.states[self.current.idx()];
            let prev = state.prev_byte_representative();
            for &assert_idx in &self.verified_deferred_asserts {
                if let State::Assert { kind, out } = self.regex.states[assert_idx]
                    && kind.eval(false, true, prev, None) == AssertEval::Pass
                    && DfaState::can_reach_match_at_end(out, prev, self.regex)
                {
                    return true;
                }
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
