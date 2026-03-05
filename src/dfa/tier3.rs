//! Tier 3: DFA with conditional transitions for non-nested counters.
//!
//! Unlike Tier 3 which unions both CInc paths (continue + break) into one
//! DFA successor and uses runtime counter-program trees, Tier 3 precomputes
//! **two DFA successor states** per counting transition: one that follows
//! only the continue path (NoBreak), and one that follows both paths
//! (WithBreak).  At runtime, the matcher evaluates counter values to pick
//! the correct successor.
//!
//! **Eligibility**: patterns with bounded repetitions where no counter is
//! nested inside another counter's body.  Body can be any length/structure.
//!
//! # Terminology: "origin"
//!
//! An **origin** is the NFA consuming state (Byte, ByteCI, ByteClass, or
//! ByteTable) where a counter instance is currently parked, waiting to
//! consume the next input byte.
//!
//! When an instance is first created at a CI (CounterInstance) node, its
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
    AssertKind, CounterIdx, Prefilter, Regex, State, StateIdx, byte_match_ci, dfa::DfaMemory,
    is_word_byte,
};

use super::DfaStateId;

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
    /// New counter instances from CI nodes in the successor closure.
    /// The third element is the initial counter value (0 for fresh seeds,
    /// 1 for seeds from resolved deferred assertions whose origin already
    /// consumed the resolving byte through a CInc increment).
    seeds: Box<[(CounterIdx, StateIdx, u32)]>,
    /// Parallel arrays: `origin_keys[i]` → `origin_actions[i]`.
    origin_keys: Box<[StateIdx]>,
    origin_actions: Box<[OriginAction]>,
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
            origin_keys: Box::new([]),
            origin_actions: Box::new([]),
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
    },
}

// ---------------------------------------------------------------------------
// Instance tracking
// ---------------------------------------------------------------------------

/// A single active counter instance.
#[derive(Clone, Debug)]
struct Instance {
    /// Number of completed iterations (0 when freshly seeded at CI).
    value: u32,
    /// The NFA consuming state this instance is waiting at.
    origin: StateIdx,
}

// ---------------------------------------------------------------------------
// Tier 3 DFA cache
// ---------------------------------------------------------------------------

/// Lazy DFA cache for Tier 2.
pub(crate) struct Tier3DfaCache {
    memory: DfaMemory,
    transitions: Vec<Transition>,
    closure_seeds: Vec<(CounterIdx, StateIdx)>,
    start_seeds: Box<[(CounterIdx, StateIdx, u32)]>,
}

impl fmt::Debug for Tier3DfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier3DfaCache")
            .field("num_states", &self.memory.states.len())
            .finish()
    }
}

impl Tier3DfaCache {
    pub(crate) fn new(num_nfa_states: usize) -> Self {
        Self {
            memory: DfaMemory::new(num_nfa_states),
            transitions: Vec::new(),
            closure_seeds: Vec::new(),
            start_seeds: Box::new([]),
        }
    }

    fn intern_state(
        &mut self,
        nfa_states: Box<[StateIdx]>,
        deferred_asserts: Box<[StateIdx]>,
        is_match: bool,
        is_match_at_end: bool,
        prev_was_word: bool,
    ) -> Option<DfaStateId> {
        let stride = self.memory.stride;
        self.memory.intern_state(
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
    fn epsilon_closure(
        &mut self,
        seeds: impl Iterator<Item = StateIdx>,
        regex: &Regex,
        at_start: bool,
        prev_byte: Option<u8>,
        next_byte: Option<u8>,
        follow_break: bool,
    ) -> ClosureResult {
        let mut encountered_cinc = false;

        self.closure_seeds.clear();

        let (is_match, is_match_at_end) = self.memory.epsilon_closure(
            seeds,
            regex,
            at_start,
            false,
            prev_byte,
            next_byte,
            |memory, counter, out| {
                memory.closure_stack.push(out);
                self.closure_seeds.push((counter, out));
            },
            |memory, _, out, out1, _, _| {
                encountered_cinc = true;
                memory.closure_stack.push(out);
                if follow_break {
                    memory.closure_stack.push(out1);
                }
            },
        );

        // Resolve CI seed pairs to (counter, consuming_state) pairs.
        let mut seed_instances: Vec<(CounterIdx, StateIdx)> = Vec::new();
        for &(counter, ci_out) in &self.closure_seeds {
            let consuming = consuming_states_from(ci_out, &regex.states);
            for c in consuming {
                seed_instances.push((counter, c));
            }
        }
        seed_instances.sort_by_key(|&(c, s)| (c.idx(), s.0));
        seed_instances.dedup();

        ClosureResult {
            nfa_states: self.memory.closure_result.as_slice().into(),
            deferred_asserts: self.memory.closure_deferred.as_slice().into(),
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
    fn populate(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> Transition {
        // Phase 1: collect targets — NFA states reached after consuming `byte`.
        let mut targets_per_origin: Vec<(StateIdx, Vec<StateIdx>)> = Vec::new();

        let mut resolved_seeds: Vec<(CounterIdx, StateIdx, u32)> = Vec::new();
        let mut resolved_cinc = false;
        let mut resolved_is_match = false;
        let mut resolved_is_match_at_end = false;

        if from != DfaStateId::DEAD {
            let from_state = &self.memory.states[from.idx()];

            // Resolve deferred assertions.
            let extra = from_state.resolve_deferred(byte, regex);
            if !extra.is_empty() {
                let resolved_prev = from_state.prev_byte_representative();
                let cr = self.epsilon_closure(
                    extra.into_iter(),
                    regex,
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
                resolved_is_match_at_end = cr.is_match_at_end;
                for &idx in cr.nfa_states.iter() {
                    if let Some(t) = consume_byte(idx, byte, regex) {
                        targets_per_origin.push((idx, vec![t]));
                    }
                }
            }

            // Phase 2: consuming states in `from` consume `byte`.
            let nfa_states = self.memory.states[from.idx()].nfa_states.clone();
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
            all_targets
                .iter()
                .copied()
                .chain(std::iter::once(regex.start)),
            regex,
            false,
            Some(byte),
            None,
            true, // follow_break=true
        );

        let is_counting = probe.encountered_cinc || resolved_cinc;

        // Build per-origin actions.
        let mut origin_keys = Vec::new();
        let mut origin_actions = Vec::new();
        for &(origin, ref targets) in &targets_per_origin {
            let action = self.compute_origin_action(targets, regex);
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

        let mut seed_instances: Vec<(CounterIdx, StateIdx, u32)> = probe
            .seed_instances
            .iter()
            .map(|&(c, s)| (c, s, 0u32))
            .collect();
        for s in &resolved_seeds {
            if !seed_instances
                .iter()
                .any(|e| e.0 == s.0 && e.1 == s.1 && e.2 == s.2)
            {
                seed_instances.push(*s);
            }
        }
        let seed_instances: Box<[(CounterIdx, StateIdx, u32)]> = seed_instances.into();

        // Compute DFA successors: no_break and with_break.
        if is_counting {
            // Two separate closures.
            let cr_nb = self.epsilon_closure(
                all_targets
                    .iter()
                    .copied()
                    .chain(std::iter::once(regex.start)),
                regex,
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

            // Fold resolved deferred assertion matches into both
            // successors' flags.  `resolved_is_match` applies
            // unconditionally (the DFA state that was transitioned FROM
            // already encodes the correct counter-aware path).
            Transition {
                no_break: nb_id,
                no_break_is_match: nb_m || resolved_is_match,
                no_break_is_match_at_end: nb_mae || resolved_is_match_at_end,
                with_break: wb_id,
                with_break_is_match: wb_m || resolved_is_match,
                with_break_is_match_at_end: wb_mae || resolved_is_match_at_end,
                is_counting: true,
                seeds: seed_instances,
                origin_keys: origin_keys.into_boxed_slice(),
                origin_actions: origin_actions.into_boxed_slice(),
            }
        } else {
            // Non-counting: both successors are the same.
            let id = self.intern_closure_result(&probe, byte);
            let (m, mae) = self.match_flags(id);

            Transition {
                no_break: id,
                no_break_is_match: m || resolved_is_match,
                no_break_is_match_at_end: mae || resolved_is_match_at_end,
                with_break: id,
                with_break_is_match: m || resolved_is_match,
                with_break_is_match_at_end: mae || resolved_is_match_at_end,
                is_counting: false,
                seeds: seed_instances,
                origin_keys: origin_keys.into_boxed_slice(),
                origin_actions: origin_actions.into_boxed_slice(),
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
            cr.nfa_states.clone(),
            cr.deferred_asserts.clone(),
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
            let s = &self.memory.states[id.idx()];
            (s.is_match, s.is_match_at_end)
        }
    }

    /// Compute the origin action for a specific set of targets.
    fn compute_origin_action(&mut self, targets: &[StateIdx], regex: &Regex) -> OriginAction {
        // Check if any target leads to CInc through epsilon transitions.
        let mut found_cinc: Option<(usize, usize)> = None;
        for &t in targets {
            if let Some(info) = find_cinc_through_epsilon(t, &regex.states) {
                found_cinc = Some(info);
                break;
            }
        }

        if let Some((min, max)) = found_cinc {
            // Check for non-CInc paths (e.g., `+` loop before CInc).
            let cr_no_cinc = epsilon_closure_stop_at_cinc(targets, &regex.states);

            let cr_continue = self.epsilon_closure(
                targets.iter().copied(),
                regex,
                false,
                None,
                None,
                false, // follow_break=false → continue only
            );
            let cr_break_only = self.epsilon_closure_break_only(
                targets,
                &regex.states,
                &regex.state_can_reach_match,
            );

            OriginAction::Increment {
                advance_origins: cr_no_cinc.into_boxed_slice(),
                min: min as u32,
                max: max as u32,
                continue_origins: cr_continue.nfa_states,
                break_is_match: cr_break_only.0,
                break_is_match_at_end: cr_break_only.1,
            }
        } else {
            // No CInc ��� just advance.
            let cr = self.epsilon_closure(targets.iter().copied(), regex, false, None, None, true);
            if cr.nfa_states.is_empty() {
                OriginAction::Dead
            } else {
                OriginAction::Advance {
                    new_origins: cr.nfa_states,
                }
            }
        }
    }

    /// Epsilon closure that only follows the CInc break path (out1), not
    /// the continue path (out).  Returns (is_match, is_match_at_end).
    fn epsilon_closure_break_only(
        &self,
        targets: &[StateIdx],
        states: &[State],
        can_reach_match: &[bool],
    ) -> (bool, bool) {
        let mut stack: Vec<StateIdx> = Vec::new();
        let mut visited = vec![false; states.len()];
        let mut is_match = false;
        let mut is_match_at_end = false;

        // First, walk from targets to find CInc nodes, then follow only out1.
        let mut init_stack: Vec<StateIdx> = targets.to_vec();
        let mut init_visited = vec![false; states.len()];
        while let Some(idx) = init_stack.pop() {
            let i = idx.idx();
            if init_visited[i] {
                continue;
            }
            init_visited[i] = true;
            match states[idx] {
                State::Split { out, out1 } => {
                    init_stack.push(out1);
                    init_stack.push(out);
                }
                State::Assert { out, .. } => init_stack.push(out),
                State::CounterInstance { out, .. } => init_stack.push(out),
                State::CounterIncrement { out1, .. } => {
                    // Only follow break path.
                    stack.push(out1);
                }
                _ => {}
            }
        }

        // Now do a standard epsilon closure from the break targets.
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
                    // Do NOT follow other assertions.  Deferred assertions
                    // (\b, EndLF, etc.) are resolved at DFA transition time
                    // via the with_break state's deferred_asserts.  Start/
                    // StartLF cannot pass in break context (past the start
                    // of input).
                }
                State::Match => {
                    is_match = true;
                }
                State::CounterInstance { out, .. } => stack.push(out),
                _ => {}
            }
        }

        (is_match, is_match_at_end)
    }

    // -----------------------------------------------------------------------
    // Cache management
    // -----------------------------------------------------------------------

    fn clear(&mut self, num_nfa_states: usize, stride: usize) {
        self.memory.clear(num_nfa_states, stride);
        self.transitions.clear();
        self.closure_seeds.clear();
        self.start_seeds = Box::new([]);
    }

    /// Prepare the cache for `regex`.
    pub(crate) fn prepare(&mut self, regex: &Regex) {
        let id = regex.id;
        if self.memory.regex_id == id && self.memory.start_id != DfaStateId::DEAD {
            return;
        }
        self.clear(regex.states.len(), regex.num_byte_classes);
        self.memory.regex_id = id;

        let cr = self.epsilon_closure(
            std::iter::once(regex.start),
            regex,
            true,
            None,
            None,
            true, // follow_break=true for start closure
        );
        self.memory.start_id = self
            .intern_state(
                cr.nfa_states,
                cr.deferred_asserts,
                cr.is_match,
                cr.is_match_at_end,
                false,
            )
            .expect("start state exceeds DFA_MAX_STATES");
        self.memory.start_is_match = self.memory.states[self.memory.start_id.idx()].is_match;
        self.memory.start_is_match_at_end =
            self.memory.states[self.memory.start_id.idx()].is_match_at_end;
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

/// Epsilon closure that stops at CInc (doesn't follow continue or break).
/// Returns consuming NFA states reachable without passing through CInc.
fn epsilon_closure_stop_at_cinc(targets: &[StateIdx], states: &[State]) -> Vec<StateIdx> {
    let mut result = Vec::new();
    let mut stack: Vec<StateIdx> = targets.to_vec();
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
            State::CounterIncrement { .. } => { /* stop */ }
            State::Match => {}
            State::Byte { .. }
            | State::ByteCI { .. }
            | State::ByteClass { .. }
            | State::ByteTable { .. } => {
                result.push(idx);
            }
        }
    }
    result.sort_unstable_by_key(|s| s.0);
    result.dedup();
    result
}

/// Check if `start` can reach a CInc through epsilon transitions.
/// Returns (min, max) if found.
fn find_cinc_through_epsilon(start: StateIdx, states: &[State]) -> Option<(usize, usize)> {
    let mut stack = vec![start];
    let mut visited = vec![false; states.len()];
    while let Some(idx) = stack.pop() {
        let i = idx.idx();
        if visited[i] {
            continue;
        }
        visited[i] = true;
        match states[idx] {
            State::CounterIncrement { min, max, .. } => {
                return Some((min, max));
            }
            State::Split { out, out1 } => {
                stack.push(out1);
                stack.push(out);
            }
            State::Assert { out, .. } => stack.push(out),
            State::CounterInstance { out, .. } => stack.push(out),
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tier 3 Matcher
// ---------------------------------------------------------------------------

/// Tier 3 DFA matcher.
pub struct Tier3DfaMatcher<'a> {
    cache: &'a mut Tier3DfaCache,
    regex: &'a Regex,
    current: DfaStateId,
    counters: Vec<Vec<Instance>>,
    next_instances: Vec<Vec<Instance>>,
    ever_matched: bool,
    match_at_end: bool,
    has_live_instances: bool,
    prefilter: Prefilter,
}

impl<'a> Tier3DfaMatcher<'a> {
    pub(crate) fn new(cache: &'a mut Tier3DfaCache, regex: &'a Regex) -> Self {
        let mut counters: Vec<Vec<Instance>> =
            (0..regex.num_counters).map(|_| Vec::new()).collect();
        let next_instances: Vec<Vec<Instance>> =
            (0..regex.num_counters).map(|_| Vec::new()).collect();

        for &(counter, origin, value) in cache.start_seeds.iter() {
            counters[counter.idx()].push(Instance { value, origin });
        }

        let ever_matched = cache.memory.start_is_match;
        let match_at_end = cache.memory.start_is_match_at_end;
        let has_live_instances = !cache.start_seeds.is_empty();

        Tier3DfaMatcher {
            current: cache.memory.start_id,
            ever_matched,
            match_at_end,
            has_live_instances,
            cache,
            regex,
            counters,
            next_instances,
            prefilter: regex.prefilter,
        }
    }

    /// Slow path: handles counting transitions and instance processing.
    #[inline(never)]
    fn step_slow(&mut self, slot: usize) {
        let t = &self.cache.transitions[slot];

        // Reset next_instances.
        for ni in &mut self.next_instances {
            ni.clear();
        }
        self.match_at_end = false;

        // Process each counter's instances against origin actions.
        let mut any_can_break = false;

        for c_idx in 0..self.counters.len() {
            for inst in &self.counters[c_idx] {
                let action = t
                    .origin_keys
                    .iter()
                    .position(|&k| k == inst.origin)
                    .map(|i| &t.origin_actions[i]);

                match action {
                    Some(OriginAction::Advance { new_origins }) => {
                        for &new_o in new_origins.iter() {
                            self.next_instances[c_idx].push(Instance {
                                value: inst.value,
                                origin: new_o,
                            });
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
                    }) => {
                        // Advance-or-increment: instance always survives
                        // at advance_origins with same value.
                        for &new_o in advance_origins.iter() {
                            self.next_instances[c_idx].push(Instance {
                                value: inst.value,
                                origin: new_o,
                            });
                        }
                        // CInc fires.
                        let new_val = inst.value + 1;
                        let do_continue = new_val < *max;
                        let do_break = new_val >= *min;

                        if do_continue {
                            for &new_o in continue_origins.iter() {
                                self.next_instances[c_idx].push(Instance {
                                    value: new_val,
                                    origin: new_o,
                                });
                            }
                        }
                        if do_break {
                            any_can_break = true;
                            if *break_is_match {
                                self.ever_matched = true;
                            }
                            if *break_is_match_at_end {
                                self.match_at_end = true;
                            }
                        }
                    }
                }
            }
        }

        // Select DFA successor.
        if t.is_counting && any_can_break {
            self.current = t.with_break;
            if !t.is_counting {
                // unreachable given the outer `if`, but kept for clarity
                if t.with_break_is_match {
                    self.ever_matched = true;
                }
                if t.with_break_is_match_at_end {
                    self.match_at_end = true;
                }
            }
        } else {
            self.current = t.no_break;
            if !t.is_counting {
                if t.no_break_is_match {
                    self.ever_matched = true;
                }
                if t.no_break_is_match_at_end {
                    self.match_at_end = true;
                }
            }
        }

        // Swap instance lists.
        std::mem::swap(&mut self.counters, &mut self.next_instances);

        // Seed new instances.
        for &(counter, origin, value) in t.seeds.iter() {
            let c_idx = counter.idx();
            let already = self.counters[c_idx]
                .iter()
                .any(|inst| inst.value == value && inst.origin == origin);
            if !already {
                self.counters[c_idx].push(Instance { value, origin });
            }
        }

        // Update has_live_instances flag.
        self.has_live_instances = self.counters.iter().any(|c| !c.is_empty());
    }

    fn step_from_dead(&mut self, byte: u8) {
        if self.has_live_instances {
            for c in &mut self.counters {
                c.clear();
            }
            self.has_live_instances = false;
        }
        self.match_at_end = false;

        let trans = self.cache.populate(DfaStateId::DEAD, byte, self.regex);

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
            self.counters[counter.idx()].push(Instance { value, origin });
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
        };

        let stride = self.cache.memory.stride;

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
                let trans = self.cache.populate(self.current, b, self.regex);
                self.cache.transitions[slot] = trans;
            }
            let t = &self.cache.transitions[slot];

            if !t.is_counting && t.seeds.is_empty() && !self.has_live_instances {
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
        if self.current != DfaStateId::DEAD {
            let state = &self.cache.memory.states[self.current.idx()];
            if state.is_match_at_end {
                return true;
            }
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
