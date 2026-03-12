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

    /// Indexed by NFA state index.  True if consuming a byte at this state
    /// leads directly to `Match` through epsilon transitions (Split, Assert,
    /// CI — but NOT through CInc).  Used by the post-break tail tracker to
    /// detect direct matches (not just match-at-end) when a tail consumes
    /// a byte and its target has no further consuming states (Bug 25).
    pub(crate) target_is_match: Box<[bool]>,

    /// Per-consuming-state deferred assertions on the epsilon path from
    /// the byte-consumption target to further consuming states or Match.
    ///
    /// Indexed by NFA state index.  For each consuming state, contains the
    /// NFA state indices of non-End Assert states (e.g. `\b`, `\B`) that
    /// lie on the epsilon path from its target.  Empty for non-consuming
    /// states or when no assertions are on the path.
    ///
    /// Used by the post-break tail tracker (Bug 30): when a tail advances
    /// through a consuming state, these assertions are deposited into
    /// `verified_deferred_asserts` for proper counter-gated evaluation,
    /// rather than relying on the contaminated `no_break_current` DFA
    /// state's deferred asserts.
    pub(crate) target_deferred_asserts: Box<[Box<[StateIdx]>]>,

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
    Advance {
        new_origins: Box<[StateIdx]>,
        /// True if the epsilon closure ALSO reaches `$ → Match`
        /// (i.e. in addition to consuming states).  Used by the
        /// post-break tail tracker to set `match_at_end` when a
        /// tail advances through this path (Bug 37: byte-specific
        /// flag, replacing the static `target_is_match_at_end` which
        /// cannot handle ByteTable origins with per-byte targets).
        is_match_at_end: bool,
    },
    /// Epsilon closure from the target reached CInc.  Counter is
    /// incremented; min/max determine continue vs. break.
    Increment {
        /// The counter being incremented.
        counter: CounterIdx,
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
    /// Deferred assertion NFA state indices on the path from the trigger's
    /// CInc break output to this seed's CI.  These assertions must pass at
    /// the break position before the seed is applied (Bug 27).  Empty when
    /// the path has no assertions.
    pub(crate) deferred_asserts: Box<[StateIdx]>,
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

    let mut break_seeds_raw: Vec<(CounterIdx, CounterIdx, StateIdx, Vec<StateIdx>)> = Vec::new();
    for &(trigger, break_target) in &all_cinc_nodes {
        // Phase 1: walk epsilon transitions from break target (Split,
        // Assert, CI) — same as the original code.  Self-referential
        // seeds (counter == trigger) are allowed here because they
        // represent the outer loop re-entering the counter body
        // through epsilon transitions (e.g. `(a{2,3})+`).
        //
        // The stack carries accumulated deferred assert state indices
        // (Bug 27): non-End Assert states on the path from break_target
        // to a CI are gating conditions that must pass at break time.
        let mut ci_stack: Vec<(StateIdx, Vec<StateIdx>)> = vec![(break_target, Vec::new())];
        let mut ci_visited = vec![false; n];
        // Collect consuming states reachable via epsilon from the break
        // path — these are the "hop points" for phase 2.
        let mut break_consuming: Vec<StateIdx> = Vec::new();
        while let Some((idx, deferred)) = ci_stack.pop() {
            let i = idx.idx();
            if ci_visited[i] {
                continue;
            }
            ci_visited[i] = true;
            match states[idx] {
                State::CounterInstance { counter, out } => {
                    let ci_consuming = consuming_states_from(out, states);
                    for c in ci_consuming {
                        break_seeds_raw.push((trigger, counter, c, deferred.clone()));
                    }
                    ci_stack.push((out, deferred));
                }
                State::Split { out, out1 } => {
                    ci_stack.push((out1, deferred.clone()));
                    ci_stack.push((out, deferred));
                }
                State::Assert { kind, out } => {
                    let mut d = deferred;
                    // Non-End asserts are deferred — they need the next
                    // byte to evaluate.  End asserts are handled separately
                    // by the break closure (break_is_match_at_end).
                    if kind != AssertKind::End {
                        d.push(idx);
                    }
                    ci_stack.push((out, d));
                }
                State::Byte { .. }
                | State::ByteCI { .. }
                | State::ByteClass { .. }
                | State::ByteTable { .. } => {
                    break_consuming.push(idx);
                }
                _ => {}
            }
        }

        // Phase 2 was previously here: it followed consuming states'
        // targets to find CI nodes reachable after one byte consumption
        // ("multi-hop break seeds").  However, these seeds fire on the
        // same byte as the counter break, one step too early — the
        // intervening consuming state hasn't consumed its byte yet.
        // The tail mechanism (break_consuming_tails + post_break_tails)
        // now correctly handles these cases: the consuming state is
        // tracked as a tail, consumes its byte on the next step, and
        // seeds the downstream counter with proper byte timing.
        //
        // Bug 38: removing phase 2 fixes false positives on patterns
        // like `^e{4,5}e{4,5}ee{4,5}$` where the break path crosses
        // a consuming state before reaching the next counter's CI.
    }
    break_seeds_raw.sort_by_key(|e| (e.0.idx(), e.1.idx(), e.2 .0));
    break_seeds_raw.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1 && a.2 == b.2);

    let break_seeds: Vec<Tier3BreakSeed> = break_seeds_raw
        .into_iter()
        .map(|(trigger, counter, origin, deferred)| {
            let mut da = deferred;
            da.sort_unstable_by_key(|s| s.0);
            da.dedup();
            Tier3BreakSeed {
                trigger,
                counter,
                origin,
                deferred_asserts: da.into_boxed_slice(),
            }
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

    // -- Step 6b: compute per-consuming-state target_is_match -------------------
    // Similar to target_mae but checks for direct `Match` reachability (not
    // `$ → Match`).  Used by the post-break tail tracker to detect matches
    // when a tail consumes a byte and its target leads to Match without any
    // intervening `Assert(End)` (Bug 25).
    let mut target_match = vec![false; n];
    for (i, state) in states.iter().enumerate() {
        let target = match *state {
            State::Byte { out, .. } | State::ByteCI { out, .. } | State::ByteClass { out, .. } => {
                if out != StateIdx::NONE {
                    Some(out)
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(t) = target {
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
                    // Do NOT follow Assert states: target_is_match
                    // means Match is reachable unconditionally (no
                    // assertion gating).  Assert(End) → Match is
                    // already handled by target_is_match_at_end.
                    State::Assert { .. } => {}
                    State::CounterInstance { out, .. } => estack.push(out),
                    State::Match => {
                        target_match[i] = true;
                    }
                    _ => {}
                }
            }
        }
    }

    // -- Step 6c: compute per-consuming-state target_deferred_asserts -----------
    // For each consuming NFA state, walk epsilon transitions from its target
    // and collect the FIRST non-End Assert state indices on each reachable
    // path.  These are the "entry-point" assertions that
    // `resolve_verified_deferred_asserts` will evaluate at runtime; subsequent
    // chained assertions (e.g. `\b → \B`) are handled by the downstream
    // `can_reach_match_at_end` / `can_reach_match_mid` walk that follows
    // the assert's `out` edge.
    //
    // IMPORTANT: the walk does NOT follow through non-End Assert states
    // (Bug 31).  Following through would record every assert in a chain
    // independently, causing `resolve_verified_deferred_asserts` to evaluate
    // them as if they were on separate paths — a chain like `\b → \B`
    // (contradictory) would spuriously match because `\B` alone can pass.
    //
    // Used by the post-break tail tracker to deposit these into
    // `verified_deferred_asserts` when a tail advances through the state
    // (Bug 30).
    let mut target_da: Vec<Box<[StateIdx]>> = Vec::with_capacity(n);
    for state in states.iter() {
        let target = match *state {
            State::Byte { out, .. } | State::ByteCI { out, .. } | State::ByteClass { out, .. } => {
                if out != StateIdx::NONE {
                    Some(out)
                } else {
                    None
                }
            }
            _ => None,
        };
        let mut asserts = Vec::new();
        if let Some(t) = target {
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
                    State::Assert { kind, .. } => {
                        // Record non-End asserts but do NOT follow `out`.
                        // Subsequent chained asserts are evaluated by
                        // `can_reach_match_at_end` / `can_reach_match_mid`
                        // at runtime (they walk from the assert's `out`).
                        // End asserts are handled by target_is_match_at_end.
                        if kind != AssertKind::End {
                            asserts.push(eidx);
                        }
                        // Do NOT push `out` — stop the walk here for this
                        // path.  See Bug 31 comment above.
                    }
                    State::CounterInstance { out, .. } => estack.push(out),
                    _ => {}
                }
            }
        }
        // Deduplicate (shouldn't be needed but defensive).
        asserts.sort_unstable_by_key(|s| s.0);
        asserts.dedup_by_key(|s| s.0);
        target_da.push(asserts.into_boxed_slice());
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
        target_is_match: target_match.into_boxed_slice(),
        target_deferred_asserts: target_da.into_boxed_slice(),
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
    /// DFA successor when at least one instance can break (both closure).
    with_break: DfaStateId,
    with_break_is_match: bool,
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
    /// subset of the DFA state's `is_match_at_end` — safe to propagate
    /// for counting transitions, because the `$ → Match` path doesn't
    /// depend on any counter reaching its minimum.
    counter_free_match_at_end: bool,
    /// True if the no-break closure's `is_match_at_end` comes from
    /// counter-free paths (targets of `reachable_without_break` origins,
    /// or epsilon `$ → Match` paths from the start state).
    ///
    /// Used for both counting and non-counting transitions in `step_slow`
    /// and `step_from_dead` (Bug 23), as well as in `finish()`.  The raw
    /// DFA state's `is_match_at_end` may include `$ → Match` from
    /// counter-dependent origins that entered the DFA state via previous
    /// counter breaks (Bug 13) or from post-break tail NFA states that
    /// are only valid when all upstream counters have broken (Bug 23).
    nb_counter_free_mae: bool,
}

impl Transition {
    fn empty() -> Self {
        Self {
            no_break: DfaStateId::UNPOPULATED,
            no_break_is_match: false,
            with_break: DfaStateId::UNPOPULATED,
            with_break_is_match: false,
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
        // Map from resolved seed origin → post-consumption target.
        // Used to remap L>1 seeds to the correct body position (Bug 16).
        let mut resolved_body_targets: Vec<(StateIdx, StateIdx)> = Vec::new();
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
                //
                // Phase 1 consumes the resolved closure's NFA states
                // against `byte`.  Build a map from seed origin → target
                // for later remapping of resolved seeds (Bug 16).
                for &idx in cr.nfa_states.iter() {
                    if let Some(t) = consume_byte(idx, byte, regex) {
                        targets_per_origin.push((idx, vec![t]));
                        resolved_body_targets.push((idx, t));
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
            let (wb_m, _wb_mae) = self.match_flags(wb_id);

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
            //
            // Resolved seeds that correspond to a break-gated seed
            // (analysis.break_seeds) are excluded — they should only fire
            // when the triggering counter actually breaks, not
            // unconditionally on every transition from the with_break DFA
            // state.  The break_seeds mechanism handles them instead.
            let pre_seeds: Vec<(CounterIdx, StateIdx, u32)> = resolved_seeds
                .iter()
                .filter(|rs| {
                    origin_keys
                        .iter()
                        .position(|&k| k == rs.1)
                        .is_some_and(|pos| {
                            matches!(origin_actions[pos], Some(Tier3OriginKind::Increment { .. }))
                        })
                        // Exclude break-gated seeds UNLESS the origin is
                        // reachable_without_break — meaning it's also
                        // unconditionally reachable from the start state
                        // (Bug 17).
                        && !analysis.break_seeds.iter().any(|bs| {
                            bs.counter == rs.0
                                && bs.origin == rs.1
                                && !analysis.reachable_without_break[rs.1.idx()]
                        })
                })
                .cloned()
                .collect();

            // Unconditional seeds: seeds that are reachable from
            // counter-free origins only.  When the DFA state includes
            // counter-dependent NFA states (from a previous with_break
            // closure), the no-break closure `cr_nb` may produce seeds
            // from CI nodes that are structurally reachable but
            // semantically counter-dependent (Bug 15).
            //
            // To distinguish truly unconditional seeds from
            // counter-dependent ones, we run a separate counter-free
            // no-break closure from only `reachable_without_break`
            // origins (analogous to `counter_free_nb_mae` for
            // match-at-end).  Seeds appearing in that closure are
            // unconditional; seeds in `cr_nb` but NOT in the
            // counter-free closure are counter-dependent and left for
            // the break_seeds mechanism and the post-break tail
            // CInc handoff (which seeds the counter when a
            // post-break tail consumes a byte and reaches CInc).
            let counter_free_seeds: Box<[(CounterIdx, StateIdx)]> = {
                let cf_targets: Vec<StateIdx> = targets_per_origin
                    .iter()
                    .filter(|(origin, _)| analysis.reachable_without_break[origin.idx()])
                    .flat_map(|(_, ts)| ts.iter().copied())
                    .collect();
                let cr_cf_nb = self.epsilon_closure(
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
                    false, // follow_break=false
                );
                cr_cf_nb.seed_instances
            };
            let mut seeds: Vec<(CounterIdx, StateIdx, u32)> = cr_nb
                .seed_instances
                .iter()
                .filter(|si| counter_free_seeds.iter().any(|cf| cf == *si))
                .map(|&(c, s)| (c, s, 0u32))
                .collect();
            // Only merge resolved seeds that are NOT pre_seeds AND that
            // don't correspond to a break-gated seed.  Resolved seeds
            // from deferred assertions in with_break DFA states may come
            // from counter break paths (e.g. CInc-0 break → \B → CI-1);
            // treating them as unconditional would suppress the
            // corresponding break_seed and fire the seed on every
            // transition from the with_break state, even when the
            // triggering counter didn't actually break (Bug 14).
            //
            // For L>1 bodies: the resolved seed's origin is the first
            // body byte (consuming state from the resolved closure).
            // Phase 1 already consumed that byte, so the instance has
            // advanced one position into the body.  Remap the seed's
            // origin to the post-consumption TARGET so it appears in
            // the next transition's origin_keys.  Without remapping,
            // the instance would be at a stale origin and get dropped
            // by the counter loop (Bug 16).
            for s in &resolved_seeds {
                let is_pre = pre_seeds.iter().any(|p| p.0 == s.0 && p.1 == s.1);
                // A resolved seed is break-gated only when it appears
                // in analysis.break_seeds AND its origin is NOT
                // reachable_without_break.  If the origin IS reachable
                // without break, the seed is unconditionally reachable
                // even though it also appears on a break path (Bug 17).
                let is_break_gated = analysis.break_seeds.iter().any(|bs| {
                    bs.counter == s.0
                        && bs.origin == s.1
                        && !analysis.reachable_without_break[s.1.idx()]
                });
                if !is_pre && !is_break_gated {
                    // Remap origin if Phase 1 consumed this seed's
                    // origin byte (Bug 16).
                    let remapped_origin = resolved_body_targets
                        .iter()
                        .find(|&&(from, _)| from == s.1)
                        .map_or(s.1, |&(_, to)| to);
                    let rs = (s.0, remapped_origin, s.2);
                    if !seeds
                        .iter()
                        .any(|e| e.0 == rs.0 && e.1 == rs.1 && e.2 == rs.2)
                    {
                        seeds.push(rs);
                    }
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
                with_break: wb_id,
                with_break_is_match: wb_m || resolved_is_match,
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
            // Merge resolved seeds with origin remapping for L>1
            // bodies (Bug 16): Phase 1 consumed the first body byte,
            // so the seed should start at the post-consumption target.
            for s in &resolved_seeds {
                let remapped_origin = resolved_body_targets
                    .iter()
                    .find(|&&(from, _)| from == s.1)
                    .map_or(s.1, |&(_, to)| to);
                let rs = (s.0, remapped_origin, s.2);
                if !seeds
                    .iter()
                    .any(|e| e.0 == rs.0 && e.1 == rs.1 && e.2 == rs.2)
                {
                    seeds.push(rs);
                }
            }

            let nb_counter_free_mae =
                self.counter_free_nb_mae(memory, mae, &targets_per_origin, regex, analysis, byte);

            Transition {
                no_break: id,
                no_break_is_match: m || resolved_is_match,
                with_break: id,
                with_break_is_match: m || resolved_is_match,
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
        let _ = seeds; // Bug 32: previously skipped break_seeds that duplicated
        // unconditional seeds.  Now we keep all break_seeds because
        // the runtime seed filter (Bug 32) may suppress the
        // unconditional seed when the DFA state is contaminated.
        // Double-seeding is harmless: `seed()` deduplicates via
        // `contains()`.
        let mut result: Vec<(CounterIdx, CounterIdx, StateIdx, u32)> = Vec::new();
        for bs in analysis.break_seeds.iter() {
            let entry = (bs.trigger, bs.counter, bs.origin, 0u32);
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
    let mut found_cinc: Option<(CounterIdx, usize, usize)> = None;
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
                counter,
                min,
                max,
                out,
                out1,
            } => {
                found_cinc = Some((counter, min, max));
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

    // Bug 37: compute `$ → Match` reachability for the Advance case
    // with a separate epsilon walk that mirrors the static
    // target_is_match_at_end logic.  The walk follows Split and CI
    // but does NOT follow non-End Assert states (they may block the
    // path, e.g. `\B` before `$`).  Only Assert(End) with
    // can_reach_match sets the flag.
    let advance_is_match_at_end = {
        let mut mae = false;
        let mut estack = vec![target];
        let mut evisited = vec![false; states.len()];
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
                    if kind == AssertKind::End && can_reach_match[out.idx()] {
                        mae = true;
                    }
                    // Do NOT follow non-End Assert states: they may
                    // block the `$ → Match` path (e.g. `\B$` only
                    // matches when `\B` passes).
                }
                State::CounterInstance { out, .. } => estack.push(out),
                _ => {} // stop at consuming states, CInc, Match
            }
        }
        mae
    };

    if let Some((counter, min, max)) = found_cinc {
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
            counter,
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
        if advance_is_match_at_end {
            // Bug 39: no consuming states downstream but `$ → Match`
            // IS reachable.  Return an Advance with empty origins so
            // that post-break tails pick up the byte-specific
            // `is_match_at_end` flag.  Previously returned `None`,
            // which lost the match-at-end information for ByteTable
            // tails (the static `target_is_match_at_end` array skips
            // ByteTable states).
            Some(Tier3OriginKind::Advance {
                new_origins: Box::new([]),
                is_match_at_end: true,
            })
        } else {
            // Truly dead — no consuming states and no `$ → Match`.
            None
        }
    } else {
        Some(Tier3OriginKind::Advance {
            new_origins: advance_origins.into_boxed_slice(),
            is_match_at_end: advance_is_match_at_end,
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
            State::ByteTable { .. } => {
                // ByteTable maps multiple bytes to different targets.
                // Include it as a tail unconditionally — the runtime
                // tail tracking handles byte-specific matching via the
                // DFA transition's origin_keys/origin_actions.
                // Previously skipped ("conservatively skip"), causing
                // Bug 35: break paths through ByteTable states lost all
                // downstream tail tracking and match-at-end detection.
                result.push(idx);
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
    /// Pending break seeds whose deferred assertions (e.g. `\b`) could not
    /// be evaluated at break time because the next byte was unknown (Bug 28).
    /// Each entry stores `(trigger, counter, origin, value, prev_was_word)` —
    /// the triggering counter, the seed to apply, and the word-ness of the
    /// byte at the break position.  Resolved at the start of the next byte
    /// (in `chunk()`) or at end-of-input (in `finish()`).
    pending_break_seeds: Vec<(CounterIdx, CounterIdx, StateIdx, u32, bool)>,
    /// True when the current DFA state (`self.current`) was reached via a
    /// `with_break` transition and may contain NFA consuming states that
    /// entered through a counter break path.  When true, the precomputed
    /// `counter_free_match_at_end` and `nb_counter_free_mae` flags on
    /// transitions FROM this state are unreliable — they may include
    /// `$ → Match` paths from origins that are statically
    /// `reachable_without_break` but entered this DFA state via a counter
    /// break (Bug 19).
    current_has_break_extras: bool,
    /// The "clean" no-break DFA state chain: always computed from the
    /// previous clean_nb state, never from a contaminated state.
    /// Used as the reference for which NFA states are genuinely reachable
    /// without counter breaks (Bug 22).
    /// Updated each step: `clean_nb = no_break_successor(clean_nb, byte)`.
    clean_nb: DfaStateId,
    /// The `nb_counter_free_mae` from the clean chain's last transition.
    /// When contaminated, this replaces the old `clean_counter_free_mae(t)`
    /// which was incorrectly checking origins against the post-transition
    /// `clean_nb` state (Bug 24).  The clean chain's transition is from
    /// a guaranteed-clean DFA state, so its `nb_counter_free_mae` correctly
    /// indicates whether a counter-free `$ → Match` path exists.
    clean_nb_cf_mae: bool,
    /// The `no_break_is_match` from the clean chain's last transition.
    /// When contaminated, the main transition's `no_break_is_match` may
    /// include `Match` from break-gated origins (e.g. `Byte('f') → Match`
    /// entered via a counter break) that haven't met their counter minimum
    /// (Bug 25).  The clean chain's transition is from a guaranteed-clean
    /// DFA state, so its `no_break_is_match` only reflects genuinely
    /// counter-free `Match` paths.
    clean_nb_is_match: bool,
    /// Transition-table slot for the clean_nb chain's transition on the
    /// current byte.  Set when the current state is contaminated
    /// (`current_has_break_extras`) and used in `step_slow` to filter
    /// unconditional seeds on **counting** transitions: only seeds that
    /// also appear in the clean_nb transition are truly counter-free
    /// (Bug 32).  `None` when clean_nb is DEAD or the state is not
    /// contaminated.
    clean_nb_trans_slot: Option<usize>,
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
        fn $method(&mut self, slot: usize, byte: u8) {
            let t = &self.cache.transitions[slot];

            // Reset next buffer and match flags.
            self.$next.clear();
            self.match_at_end = false;
            self.verified_deferred_asserts.clear();
            self.pending_break_seeds.clear();

            // Advance existing post-break tails through this transition.
            // Each tail is a consuming NFA state from a previous counter
            // break.  If it consumed this byte and its target reaches
            // `$ → Match`, set match_at_end.
            //
            // Shared match-flag check for Advance and None branches
            // (Bug 29: ensures both branches check target_is_match and
            // target_is_match_at_end consistently).
            macro_rules! check_tail_match_flags {
                ($self:ident, $origin:expr) => {
                    if $self.analysis.target_is_match_at_end[$origin.idx()] {
                        $self.match_at_end = true;
                    }
                    if $self.analysis.target_is_match[$origin.idx()] {
                        $self.ever_matched = true;
                    }
                };
            }
            let mut any_can_break = false;
            let mut counter_broke: u64 = 0;

            self.next_post_break_tails.clear();
            for &pbo in &self.post_break_tails {
                let pos = t.origin_keys.iter().position(|&k| k == pbo);
                match pos.map(|i| &t.origin_actions[i]) {
                    Some(Some(Tier3OriginKind::Advance {
                        new_origins,
                        is_match_at_end,
                    })) => {
                        for &new_o in new_origins.iter() {
                            if !self.next_post_break_tails.contains(&new_o) {
                                self.next_post_break_tails.push(new_o);
                            }
                        }
                        // Bug 30: deposit target deferred asserts for this
                        // consuming state into verified_deferred_asserts.
                        // These are non-End Assert states on the epsilon
                        // path from pbo's target, which would otherwise
                        // only fire via the contaminated no_break_current
                        // DFA state.
                        for &da in self.analysis.target_deferred_asserts[pbo.idx()].iter() {
                            if !self.verified_deferred_asserts.contains(&da) {
                                self.verified_deferred_asserts.push(da);
                            }
                        }
                        // Bug 37: use the byte-specific is_match_at_end
                        // from the Advance action instead of the static
                        // target_is_match_at_end array.  ByteTable origins
                        // have per-byte targets, so the static array
                        // (which skipped ByteTable) was always false for
                        // ByteTable states.
                        //
                        // target_is_match uses the static array (skips
                        // ByteTable conservatively; a ByteTable tail whose
                        // byte-specific target reaches Match directly
                        // without Assert would need a similar byte-specific
                        // flag, but no such pattern has been found yet).
                        if *is_match_at_end {
                            self.match_at_end = true;
                        }
                        if self.analysis.target_is_match[pbo.idx()] {
                            self.ever_matched = true;
                        }
                    }
                    Some(Some(Tier3OriginKind::Increment {
                        counter,
                        advance_origins: _,
                        min,
                        max,
                        continue_origins,
                        break_is_match,
                        break_is_match_at_end,
                        break_deferred_asserts,
                        break_consuming_states,
                    })) => {
                        // Tail hit a CInc — hand off to the counter
                        // instance machinery.  The post-break tail
                        // consumed this byte, entering the CInc for the
                        // first time (value 0 → incremented to 1).
                        //
                        // Insert the continued instance directly into
                        // `$next` to avoid double-counting: the counter
                        // loop (below) processes `$current`, and we
                        // don't want this byte to be counted twice.
                        //
                        // This handoff is needed when counter-dependent
                        // unconditional seeds are filtered out (Bug 15
                        // fix): the post-break tail is the only way
                        // the downstream counter gets its first instance.
                        let pbt_value: u32 = 0;
                        // Check continue (value after increment < max).
                        if pbt_value + 1 < *max {
                            for &new_o in continue_origins.iter() {
                                self.$next.seed(counter.idx(), new_o, pbt_value + 1);
                            }
                        }
                        // Check break (value after increment >= min).
                        // Bug 36: the tail→CInc handoff IS a counter
                        // break — set any_can_break and counter_broke so
                        // the DFA selects with_break (which includes
                        // post-break NFA origins) and break_seeds fire.
                        // Without this, the DFA goes to the no_break
                        // state and subsequent tails are dropped because
                        // their origin isn't in the no_break closure.
                        if pbt_value + 1 >= *min {
                            any_can_break = true;
                            counter_broke |= 1u64 << counter.idx();
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
                    Some(None) => {
                        // `None` target — the tail consumed this byte
                        // but the post-consumption epsilon path has no
                        // consuming states or CInc (only asserts and/or
                        // Match).  Still need to deposit deferred asserts
                        // (Bug 34: `y\b$` after counter break was lost
                        // because analyze_target returned None for the
                        // assert-only path).
                        for &da in self.analysis.target_deferred_asserts[pbo.idx()].iter() {
                            if !self.verified_deferred_asserts.contains(&da) {
                                self.verified_deferred_asserts.push(da);
                            }
                        }
                        check_tail_match_flags!(self, pbo);
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

            // Note: any_can_break and counter_broke are declared ABOVE
            // the tail loop (Bug 36) so that tail→CInc handoffs that
            // produce a break can set them before the counter loop.
            let num_counters = self.$current.num_counters();
            debug_assert!(num_counters <= 64);

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
                        Some(Tier3OriginKind::Advance { new_origins, .. }) => {
                            for &new_o in new_origins.iter() {
                                self.$next.advance(c_idx, entry, new_o);
                            }
                        }
                        None => {}
                        Some(Tier3OriginKind::Increment {
                            counter: _,
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
            //
            // Contamination check (Bug 19/22): When the FROM state was
            // reached via a with_break transition and the pattern has ≥2
            // counters, origins from downstream counter break chains may
            // be in the DFA state before those downstream counters have
            // actually reached their minimums.
            self.no_break_current = t.no_break;
            let from_contaminated = self.current_has_break_extras && self.regex.num_counters > 1;
            self.last_nb_counter_free_mae = if from_contaminated {
                self.clean_nb_cf_mae
            } else {
                t.nb_counter_free_mae
            };
            if t.is_counting && any_can_break {
                self.current = t.with_break;
                self.current_has_break_extras = true;
            } else {
                self.current = t.no_break;
                // Propagate contamination: the no-break successor of a
                // contaminated state inherits NFA states that originally
                // entered via counter breaks (Bug 22).
                if !from_contaminated {
                    self.current_has_break_extras = false;
                }
            }
            if !t.is_counting {
                // For non-counting transitions, use counter-free mae just
                // like the fast path (Bug 23).  The raw
                // no_break_is_match_at_end may include $ → Match paths
                // from post-break tail NFA states that only become active
                // after counter breaks — but those counters may not have
                // met their minimums.
                let mae_before = self.match_at_end;
                let cf_mae = if from_contaminated {
                    self.clean_nb_cf_mae
                } else {
                    t.nb_counter_free_mae
                };
                if cf_mae {
                    self.match_at_end = true;
                }
                // Bug 23 guard: match_at_end must only become true via
                // cf_mae, never from the raw DFA state is_match_at_end.
                debug_assert!(
                    !self.match_at_end || mae_before || cf_mae,
                    "Bug 23 guard: non-counting transition set match_at_end \
                     without cf_mae (state={:?})",
                    self.current,
                );
                let m = if any_can_break {
                    t.with_break_is_match
                } else if from_contaminated {
                    // When contaminated and no counter broke, the
                    // no_break_is_match flag may include Match from
                    // break-gated origins that never reached their
                    // counter minimum (Bug 25).  Use the clean chain's
                    // flag instead.  Post-break tail matches are handled
                    // separately via target_is_match in the tail loop.
                    self.clean_nb_is_match
                } else {
                    t.no_break_is_match
                };
                if m {
                    self.ever_matched = true;
                }
            } else {
                // For counting transitions, propagate only no_break_is_match
                // and counter_free_match_at_end.  The full
                // no_break_is_match_at_end may include $ → Match paths from
                // post-break origins that haven't met their counter minimums.
                // Counter-dependent paths are handled by per-instance break
                // checks and the post_break_tails mechanism.
                //
                // When contaminated, the no_break_is_match flag may
                // include Match from break-gated origins that never
                // reached their counter minimum (Bug 25).  Use the
                // clean chain's flag instead.  Per-instance break
                // checks (above) already set ever_matched when an
                // actual counter breaks with break_is_match=true,
                // so no signal is lost.
                let m = if from_contaminated {
                    self.clean_nb_is_match
                } else {
                    t.no_break_is_match
                };
                if m {
                    self.ever_matched = true;
                }
                let cf_mae = if from_contaminated {
                    self.clean_nb_cf_mae
                } else {
                    t.counter_free_match_at_end
                };
                if cf_mae {
                    self.match_at_end = true;
                }
            }

            // Swap buffers (O(1) — just swaps flat Vec pointers).
            std::mem::swap(&mut self.$current, &mut self.$next);
            std::mem::swap(&mut self.post_break_tails, &mut self.next_post_break_tails);

            // Seed new instances (unconditional).
            //
            // Bug 32: on COUNTING transitions when the current DFA state
            // is contaminated (has break extras), the `counter_free_seeds`
            // computation in `populate()` may include seeds from origins
            // that are globally `reachable_without_break` but entered THIS
            // DFA state only via a counter break.  Gate these seeds on the
            // clean_nb chain's transition: only seeds that also appear in
            // the clean chain are truly counter-free.
            //
            // Non-counting transitions are NOT filtered: their seeds come
            // from the full probe closure (follow_break=true) and there
            // are no break_seeds to fall back on.  Filtering them would
            // kill legitimate downstream counter seeds (regression on
            // `^(a{1,17}b){2,3}$` / "abab").
            //
            // Suppressed seeds are still available as break_seeds (Bug 32
            // part 2: `compute_break_seeds` no longer deduplicates against
            // unconditional seeds on counting transitions, so the break
            // path fires when the triggering counter actually breaks).
            for &(counter, origin, value) in t.seeds.iter() {
                if t.is_counting {
                    if let Some(cn_slot) = self.clean_nb_trans_slot {
                        let cn_t = &self.cache.transitions[cn_slot];
                        if !cn_t.seeds.iter().any(|s| s.0 == counter && s.1 == origin) {
                            continue;
                        }
                    }
                }
                self.$current.seed(counter.idx(), origin, value);
            }
            // Break seeds gated on the triggering counter.
            // Bug 27: break seeds with deferred assertions (e.g. `\b`) on
            // the path from the trigger's CInc break to the seed's CI.
            // Bug 28: the assertion is at the position AFTER consuming
            // `byte` (between `byte` and the next input byte).  Since the
            // next byte is unknown at break time, the assertion must be
            // DEFERRED — stored in `pending_break_seeds` and evaluated at
            // the start of the next byte (or at end-of-input in `finish()`).
            for &(trigger, counter, origin, value) in t.break_seeds.iter() {
                if counter_broke & (1u64 << trigger.idx()) != 0 {
                    let has_deferred = self
                        .analysis
                        .break_seeds
                        .iter()
                        .find(|bs| {
                            bs.trigger == trigger
                                && bs.counter == counter
                                && bs.origin == origin
                        })
                        .is_some_and(|bs| !bs.deferred_asserts.is_empty());
                    if has_deferred {
                        // Defer: store the seed with the word-ness of the
                        // break position (= `byte`, the byte just consumed
                        // by the body) for later evaluation.
                        self.pending_break_seeds.push((
                            trigger,
                            counter,
                            origin,
                            value,
                            crate::is_word_byte(byte),
                        ));
                    } else {
                        // No assertions on break path — apply immediately.
                        self.$current.seed(counter.idx(), origin, value);
                    }
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
            pending_break_seeds: Vec::new(),
            // The start state may include NFA states from break paths
            // (the start closure is computed with follow_break=true), but
            // these are freshly seeded — not from a previous counter break.
            // Treat the start state as uncontaminated.
            current_has_break_extras: false,
            clean_nb: cache.inner.start_id,
            clean_nb_cf_mae: cache.inner.start_is_match_at_end,
            clean_nb_is_match: cache.inner.start_is_match,
            clean_nb_trans_slot: None,
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

    // `clean_counter_free_mae` was removed in Bug 24.  The method
    // compared origins against `clean_nb`'s *post-transition* NFA states,
    // which could be empty when the transition's only target was an
    // Assert(End) with no consuming successors — losing the mae signal
    // from legitimate counter-free origins.  Replaced by `clean_nb_cf_mae`,
    // the clean chain transition's `nb_counter_free_mae`, which is
    // computed against the *pre-transition* clean state.

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
        self.current_has_break_extras = false;
        self.clean_nb = trans.no_break;
        self.clean_nb_cf_mae = trans.nb_counter_free_mae;
        self.clean_nb_is_match = trans.no_break_is_match;

        if !trans.is_counting {
            if trans.no_break_is_match {
                self.ever_matched = true;
            }
            // Use counter-free mae, not raw no_break_is_match_at_end,
            // for the same reason as in step_slow (Bug 23).
            // From DEAD, no contamination possible.
            if trans.nb_counter_free_mae {
                self.match_at_end = true;
            }
            // Bug 23 guard: match_at_end was reset to false above, so
            // it can only be true here if nb_counter_free_mae is true.
            debug_assert!(
                !self.match_at_end || trans.nb_counter_free_mae,
                "Bug 23 guard: step_from_dead set match_at_end without \
                 nb_counter_free_mae (state={:?})",
                self.current,
            );
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

            // Resolve counter-break deferred asserts from the PREVIOUS
            // step (Bug 26).  These asserts (e.g. `\B → Match`) were
            // emitted when a counter broke but need the NEXT byte to
            // evaluate word-boundary conditions.  The DFA state's own
            // deferred_asserts are resolved during transition population
            // (resolve_deferred), but counter-break asserts are runtime-
            // only and live in verified_deferred_asserts.
            if self.resolve_verified_deferred_asserts(false, Some(b)) {
                self.ever_matched = true;
            }

            // Resolve pending break seeds from the PREVIOUS step (Bug 28).
            // These are break seeds whose deferred assertions (e.g. `\b`)
            // could not be evaluated at break time because the next byte
            // was unknown.  Now that we have the next byte (`b`), evaluate
            // the assertions and seed the counter if they pass.
            if !self.pending_break_seeds.is_empty() {
                for i in 0..self.pending_break_seeds.len() {
                    let (trigger, counter, origin, value, prev_was_word) =
                        self.pending_break_seeds[i];
                    let prev = if prev_was_word {
                        Some(b'a')
                    } else {
                        Some(b' ')
                    };
                    // Look up deferred asserts for this seed.
                    let pass = self
                        .analysis
                        .break_seeds
                        .iter()
                        .find(|bs| {
                            bs.trigger == trigger
                                && bs.counter == counter
                                && bs.origin == origin
                        })
                        .is_none_or(|bs| {
                            bs.deferred_asserts.iter().all(|&assert_idx| {
                                if let State::Assert { kind, .. } =
                                    self.regex.states[assert_idx]
                                {
                                    kind.eval(false, false, prev, Some(b))
                                        == AssertEval::Pass
                                } else {
                                    true
                                }
                            })
                        });
                    if pass {
                        if self.use_ranges {
                            self.ranged_counters.insert(
                                counter.idx(),
                                origin,
                                value,
                                value,
                            );
                        } else {
                            self.inst_counters
                                .seed(counter.idx(), origin, value);
                        }
                        self.has_live_instances = true;
                    }
                }
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

            // Advance the clean no-break chain (Bug 22) BEFORE borrowing
            // the main transition `t`, since populate() mutates the cache.
            // When contaminated, advance `clean_nb` independently from the
            // previous clean state.  When not contaminated, defer update
            // to after reading `t.no_break` (see below).
            //
            // Bug 32: save the clean_nb transition slot so step_slow can
            // filter unconditional seeds on counting transitions against
            // the clean chain's seeds.
            if self.current_has_break_extras && self.regex.num_counters > 1 {
                if self.clean_nb != DfaStateId::DEAD {
                    let cn_slot = self.clean_nb.idx() * stride + class;
                    if self.cache.transitions[cn_slot].no_break == DfaStateId::UNPOPULATED {
                        let cn_trans = self.cache.populate(
                            self.memory,
                            self.clean_nb,
                            b,
                            self.regex,
                            self.analysis,
                        );
                        self.cache.transitions[cn_slot] = cn_trans;
                    }
                    self.clean_nb_trans_slot = Some(cn_slot);
                    let cn_trans = &self.cache.transitions[cn_slot];
                    self.clean_nb_cf_mae = cn_trans.nb_counter_free_mae;
                    self.clean_nb_is_match = cn_trans.no_break_is_match;
                    self.clean_nb = cn_trans.no_break;
                } else {
                    self.clean_nb_cf_mae = false;
                    self.clean_nb_is_match = false;
                    self.clean_nb_trans_slot = None;
                }
            } else {
                self.clean_nb_trans_slot = None;
            }

            let t = &self.cache.transitions[slot];

            // When not contaminated, clean_nb simply tracks no_break.
            if !(self.current_has_break_extras && self.regex.num_counters > 1) {
                self.clean_nb = t.no_break;
                self.clean_nb_cf_mae = t.nb_counter_free_mae;
                self.clean_nb_is_match = t.no_break_is_match;
            }

            if !t.is_counting
                && t.pre_seeds.is_empty()
                && t.seeds.is_empty()
                && !self.has_live_instances
                && self.post_break_tails.is_empty()
            {
                let from_contaminated =
                    self.current_has_break_extras && self.regex.num_counters > 1;
                self.current = t.no_break;
                self.no_break_current = t.no_break;
                let cf_mae = if from_contaminated {
                    self.clean_nb_cf_mae
                } else {
                    t.nb_counter_free_mae
                };
                self.last_nb_counter_free_mae = cf_mae;
                self.match_at_end = cf_mae;
                // Propagate contamination through no-break successors
                // (Bug 22): the no-break closure of a contaminated state
                // inherits NFA states from counter breaks.
                if !from_contaminated {
                    self.current_has_break_extras = false;
                }
                // When contaminated, `t.no_break_is_match` may include
                // Match from break-gated origins (Bug 25).  Use the
                // clean chain's transition flag instead.
                let m = if from_contaminated {
                    self.clean_nb_is_match
                } else {
                    t.no_break_is_match
                };
                if m {
                    self.ever_matched = true;
                }
                continue;
            }

            if self.use_ranges {
                self.step_slow_ranged(slot, b);
            } else {
                self.step_slow_instances(slot, b);
            }
        }
    }

    /// Walk epsilon transitions from `start` mid-input (not at end-of-input),
    /// evaluating any assertions encountered with `prev` and `next`.  Returns
    /// Evaluate counter-break deferred assertions against a given byte context.
    ///
    /// Returns `true` if any deferred assertion passes and its downstream path
    /// reaches `Match`.  Used from two sites:
    ///
    /// - **Mid-input** (`chunk()`, `at_end=false`, `next=Some(byte)`): sets
    ///   `self.ever_matched` when the assertion passes and Match is reachable
    ///   through epsilon transitions with mid-input evaluation.
    /// - **End-of-input** (`finish()`, `at_end=true`, `next=None`): checks
    ///   whether the assertion passes at end-of-input and Match is reachable
    ///   through `$ → Match` paths.
    fn resolve_verified_deferred_asserts(&self, at_end: bool, next: Option<u8>) -> bool {
        if self.verified_deferred_asserts.is_empty() || self.current == DfaStateId::DEAD {
            return false;
        }
        let state = &self.cache.inner.states[self.current.idx()];
        let prev = state.prev_byte_representative();
        for &assert_idx in &self.verified_deferred_asserts {
            if let State::Assert { kind, out } = self.regex.states[assert_idx]
                && kind.eval(false, at_end, prev, next) == AssertEval::Pass
            {
                let reachable = if at_end {
                    DfaState::can_reach_match_at_end(out, prev, self.regex)
                } else {
                    Self::can_reach_match_mid(out, prev, next, self.regex)
                };
                if reachable {
                    return true;
                }
            }
        }
        false
    }

    /// true if `Match` is reachable with all assertions passing.
    ///
    /// Similar to [`DfaState::can_reach_match_at_end`] but uses `at_end=false`
    /// and `next=Some(byte)` for mid-input assertion evaluation.  This correctly
    /// rejects paths through `$` or `\B` that only pass at end-of-input.
    fn can_reach_match_mid(
        start: StateIdx,
        prev: Option<u8>,
        next: Option<u8>,
        regex: &Regex,
    ) -> bool {
        let states = &regex.states;
        let num_states = states.len();
        let mut visited = vec![false; num_states];
        let mut stack = vec![start];
        while let Some(idx) = stack.pop() {
            let i = idx.idx();
            if i >= num_states || visited[i] || !regex.state_can_reach_match[i] {
                continue;
            }
            visited[i] = true;
            match states[idx] {
                State::Match => return true,
                State::Assert { kind, out } => {
                    if kind.eval(false, false, prev, next) == AssertEval::Pass {
                        stack.push(out);
                    }
                }
                State::Split { out, out1 } => {
                    stack.push(out);
                    stack.push(out1);
                }
                State::CounterInstance { out, .. } => {
                    stack.push(out);
                }
                // Consuming states don't fire here — the assert resolution
                // is about reaching Match through epsilon transitions only.
                _ => {}
            }
        }
        false
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
        //
        // Bug 30: when contaminated (break_extras && num_counters > 1),
        // `no_break_current` may include deferred assertions from
        // break-gated origins (e.g. `\b` after `Byte('f')` which is only
        // reachable via c1's break path).  These spurious deferred asserts
        // fire at EOI even when the gating counter never reached its min.
        //
        // Fix: when contaminated, skip `no_break_current`'s deferred asserts
        // entirely.  Legitimate counter-break deferred asserts are deposited
        // into `verified_deferred_asserts` by the post-break tail tracker
        // (using `target_deferred_asserts`) and resolved separately above.
        // Non-contaminated deferred asserts from `clean_nb` are handled below.
        let from_contaminated = self.current_has_break_extras && self.regex.num_counters > 1;
        if self.no_break_current != DfaStateId::DEAD && !from_contaminated {
            let nb_state = &self.cache.inner.states[self.no_break_current.idx()];
            if nb_state.resolve_deferred_at_end(self.regex) {
                return true;
            }
        }
        // When contaminated, fall back to clean_nb for counter-free deferred
        // asserts (clean_nb excludes break-gated origins).
        if from_contaminated && self.clean_nb != DfaStateId::DEAD {
            let clean_state = &self.cache.inner.states[self.clean_nb.idx()];
            if clean_state.resolve_deferred_at_end(self.regex) {
                return true;
            }
        }
        // Pending break seeds at end-of-input (Bug 28): evaluate the
        // deferred assertions with `at_end=true`.  If the assertion passes
        // AND the seeded counter can immediately break (value >= min) with
        // a match-producing break path, this is a match.
        if !self.pending_break_seeds.is_empty() {
            for &(trigger, counter, origin, value, prev_was_word) in &self.pending_break_seeds {
                let prev = if prev_was_word {
                    Some(b'a')
                } else {
                    Some(b' ')
                };
                let pass = self
                    .analysis
                    .break_seeds
                    .iter()
                    .find(|bs| {
                        bs.trigger == trigger
                            && bs.counter == counter
                            && bs.origin == origin
                    })
                    .is_none_or(|bs| {
                        bs.deferred_asserts.iter().all(|&assert_idx| {
                            if let State::Assert { kind, .. } = self.regex.states[assert_idx] {
                                kind.eval(false, true, prev, None) == AssertEval::Pass
                            } else {
                                true
                            }
                        })
                    });
                if pass {
                    // The seed counter starts at `value` with no more input.
                    // Check if it can immediately break (value >= min) and if
                    // the break path reaches Match/$ → Match.
                    // Look up the counter's CInc target to find min/break info.
                    // The origin is a consuming state; its target (after byte
                    // consumption) holds the CInc increment action.
                    for target_action in self.analysis.targets.iter().flatten() {
                        if let Tier3OriginKind::Increment {
                            counter: tc,
                            min,
                            break_is_match_at_end,
                            break_is_match,
                            ..
                        } = target_action
                            && *tc == counter
                            && value >= *min
                            && (*break_is_match_at_end || *break_is_match)
                        {
                            return true;
                        }
                    }
                    // Even if the counter can't immediately break, the seed
                    // doesn't help at end-of-input (no more bytes to process).
                }
            }
        }
        // Counter-dependent deferred assertions: evaluate assertions from
        // counter break paths that actually broke with enough value.  These
        // are NFA Assert state indices (e.g. \b, \B) on the path to
        // `$ → Match` from a counter whose break condition was met.
        self.resolve_verified_deferred_asserts(true, None)
    }

    #[allow(dead_code)]
    pub fn ismatch(&self) -> bool {
        self.ever_matched
    }
}

impl fmt::Debug for Tier3DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Tier3DfaMatcher");
        s.field("current", &self.current);
        if self.current != DfaStateId::DEAD {
            let state = &self.cache.inner.states[self.current.idx()];
            s.field("nfa_states", &state.nfa_states);
            s.field("is_match", &state.is_match);
            s.field("is_match_at_end", &state.is_match_at_end);
        }
        s.field("no_break_current", &self.no_break_current)
            .field("ever_matched", &self.ever_matched)
            .field("match_at_end", &self.match_at_end)
            .field("has_live_instances", &self.has_live_instances)
            .field("current_has_break_extras", &self.current_has_break_extras)
            .field("clean_nb", &self.clean_nb)
            .field("clean_nb_cf_mae", &self.clean_nb_cf_mae)
            .field("clean_nb_is_match", &self.clean_nb_is_match)
            .field("post_break_tails_len", &self.post_break_tails.len())
            .field(
                "verified_deferred_asserts_len",
                &self.verified_deferred_asserts.len(),
            )
            .field(
                "pending_break_seeds_len",
                &self.pending_break_seeds.len(),
            )
            .field("use_ranges", &self.use_ranges);
        s.finish()
    }
}

impl fmt::Display for Tier3DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.current == DfaStateId::DEAD {
            write!(f, "DFA[T3] state=DEAD matched={}", self.ever_matched)?;
            if self.has_live_instances {
                write!(f, " live")?;
            }
            return Ok(());
        }
        let state = &self.cache.inner.states[self.current.idx()];
        write!(
            f,
            "DFA[T3] state={} nfa={{{}}} matched={}",
            self.current.0,
            state
                .nfa_states
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.ever_matched,
        )?;
        if state.is_match {
            write!(f, " is_match")?;
        }
        if self.match_at_end {
            write!(f, " mae")?;
        }
        if self.last_nb_counter_free_mae {
            write!(f, " cf_mae")?;
        }
        if self.current_has_break_extras {
            write!(f, " break_extras")?;
        }
        // Clean chain state (only shown when contaminated or divergent).
        if self.current_has_break_extras && self.regex.num_counters > 1 {
            if self.clean_nb == DfaStateId::DEAD {
                write!(f, " clean_nb=DEAD")?;
            } else {
                let clean_state = &self.cache.inner.states[self.clean_nb.idx()];
                write!(
                    f,
                    " clean_nb={{{}}}",
                    clean_state
                        .nfa_states
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                )?;
            }
            if self.clean_nb_is_match {
                write!(f, " clean_is_match")?;
            }
            if self.clean_nb_cf_mae {
                write!(f, " clean_cf_mae")?;
            }
        }
        // Counter summary.
        if self.has_live_instances {
            if self.use_ranges {
                let rc = &self.ranged_counters;
                for ci in 0..rc.num_counters() {
                    let entries = rc.entries(ci);
                    if entries.is_empty() {
                        continue;
                    }
                    let count = entries.len();
                    write!(f, "\n  c{ci}: {count} range(s) [")?;
                    for (j, entry) in entries.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(
                            f,
                            "origin:{}={}-{}",
                            entry.origin, entry.min_val, entry.max_val
                        )?;
                    }
                    write!(f, "]")?;
                }
            } else {
                let ic = &self.inst_counters;
                for ci in 0..ic.num_counters() {
                    let entries = ic.entries(ci);
                    if entries.is_empty() {
                        continue;
                    }
                    let count = entries.len();
                    write!(f, "\n  c{ci}: {count} inst [")?;
                    for (j, entry) in entries.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}@{}", entry.origin, entry.value)?;
                    }
                    write!(f, "]")?;
                }
            }
        }
        if !self.post_break_tails.is_empty() {
            write!(
                f,
                "\n  tails: [{}]",
                self.post_break_tails
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )?;
        }
        if !self.verified_deferred_asserts.is_empty() {
            write!(f, "\n  deferred: [")?;
            for (i, &da) in self.verified_deferred_asserts.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                if let State::Assert { kind, .. } = self.regex.states[da] {
                    write!(f, "{}@{}", kind.label(), da)?;
                } else {
                    write!(f, "?@{da}")?;
                }
            }
            write!(f, "]")?;
        }
        if !self.pending_break_seeds.is_empty() {
            write!(f, "\n  pending_seeds: [")?;
            for (i, &(trigger, counter, origin, value, prev_word)) in
                self.pending_break_seeds.iter().enumerate()
            {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(
                    f,
                    "c{}→c{}@{}(val={},pw={})",
                    trigger, counter, origin, value, prev_word
                )?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}
