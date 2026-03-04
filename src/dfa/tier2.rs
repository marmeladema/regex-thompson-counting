//! Tier 2: DFA with differential counters for fixed-length repetition bodies.
//!
//! Uses the Becchi differential-counter technique for O(1) per-character
//! counter management.  Supports the same deferred assertions as Tier 1.
//!
//! **Eligibility**: patterns with bounded repetitions where:
//! - No counter is nested inside another counter's body.
//! - Every counter body has a fixed byte-length (all paths through the body
//!   consume the same number of bytes).
//! - No CRLF assertions.
//! - No deferred assertions inside counter bodies.
//!
//! Counter-free patterns (Tier 1 eligible) are also Tier 2 eligible — the
//! differential counter logic is simply dormant.

use std::collections::{HashMap, VecDeque};
use std::fmt;

use crate::{
    AssertEval, AssertKind, CounterIdx, Prefilter, Regex, State, StateIdx, byte_match_ci,
    is_word_byte,
};

use super::{DfaState, DfaStateId};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of DFA states.
const DFA_MAX_STATES: usize = 2048;

// ---------------------------------------------------------------------------
// Differential counter
// ---------------------------------------------------------------------------

/// Differential counter for one phase of one counter.
///
/// All instances in the same phase increment in lockstep.  We store only
/// the oldest instance's absolute value and the count of active instances.
/// Deltas between consecutive instances are recorded in a queue so that
/// when the oldest is deallocated, the new oldest's value can be recovered
/// in O(1).
#[derive(Clone, Debug)]
struct DiffCounter {
    /// Absolute value of the oldest (highest-value) instance.
    oldest: u32,
    /// Number of active instances.
    count: u32,
    /// Sum of all deltas (oldest_value − youngest_value).
    total_delta: u32,
    /// Queue of deltas: front = gap between oldest and 2nd-oldest.
    deltas: VecDeque<u32>,
}

impl DiffCounter {
    fn new() -> Self {
        Self {
            oldest: 0,
            count: 0,
            total_delta: 0,
            deltas: VecDeque::new(),
        }
    }

    /// Youngest instance value, or 0 if no instances.
    #[inline]
    fn youngest(&self) -> u32 {
        debug_assert!(self.count > 0);
        self.oldest - self.total_delta
    }

    /// Increment all instances (only need to bump oldest).
    #[inline]
    fn increment_all(&mut self) {
        debug_assert!(self.count > 0);
        self.oldest += 1;
    }

    /// Deallocate the oldest instance.  Returns `true` if instances remain.
    #[inline]
    fn dealloc_oldest(&mut self) -> bool {
        debug_assert!(self.count > 0);
        self.count -= 1;
        if self.count > 0 {
            let d = self.deltas.pop_front().unwrap();
            // New oldest = old oldest − delta
            self.oldest -= d;
            self.total_delta -= d;
            true
        } else {
            self.total_delta = 0;
            self.deltas.clear();
            false
        }
    }

    /// Allocate a new instance with a given initial value (youngest).
    #[inline]
    fn alloc_new_with_value(&mut self, value: u32) {
        if self.count == 0 {
            self.oldest = value;
            self.count = 1;
            self.total_delta = 0;
        } else {
            let youngest_val = self.youngest();
            let gap = youngest_val - value;
            self.deltas.push_back(gap);
            self.total_delta += gap;
            self.count += 1;
        }
    }

    /// Allocate a new instance with value 0 (youngest).
    #[inline]
    fn alloc_new(&mut self) {
        self.alloc_new_with_value(0);
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reset to empty state.
    fn clear(&mut self) {
        self.oldest = 0;
        self.count = 0;
        self.total_delta = 0;
        self.deltas.clear();
    }
}

// ---------------------------------------------------------------------------
// Per-counter state
// ---------------------------------------------------------------------------

/// Runtime state for one counter using differential encoding.
///
/// For a counter body of fixed byte-length `L`, instances partition into
/// `L` phase groups based on `creation_position mod L`.  Within each
/// phase, all instances pass through CInc simultaneously (every `L` bytes).
///
/// For single-byte bodies (L=1), there is exactly one phase.
#[derive(Clone, Debug)]
struct CounterState {
    /// One differential counter per phase.  `phases.len() == body_length`.
    phases: Vec<DiffCounter>,
    /// Which phase fires CInc on the current byte (cycles 0..L-1).
    /// After CInc fires, this advances to `(active_phase + 1) % L`.
    active_phase: usize,
    /// Counter parameters.
    min: u32,
    max: u32,
}

impl CounterState {
    fn new(body_length: usize, min: u32, max: u32) -> Self {
        Self {
            phases: (0..body_length).map(|_| DiffCounter::new()).collect(),
            active_phase: 0,
            min,
            max,
        }
    }

    fn is_empty(&self) -> bool {
        self.phases.iter().all(|p| p.is_empty())
    }

    fn clear(&mut self) {
        for p in &mut self.phases {
            p.clear();
        }
        self.active_phase = 0;
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// Cached DFA transition for `(state, byte)`.
///
/// Non-counting transitions use `no_break` only.
/// Counting transitions select between `no_break` and `with_break`
/// based on the differential counter condition.
#[derive(Clone)]
struct Transition {
    /// DFA successor when no counter instance can break.
    no_break: DfaStateId,
    no_break_is_match: bool,
    no_break_is_match_at_end: bool,
    /// DFA successor when at least one instance can break.
    with_break: DfaStateId,
    with_break_is_match: bool,
    with_break_is_match_at_end: bool,
    /// True if this transition crosses at least one CInc node.
    is_counting: bool,
    /// Bitmask: bit `i` set means counter `i` fires CInc at this transition.
    counting_mask: u64,
    /// Whether new counter instances should be seeded at this transition.
    seeds: Box<[(CounterIdx, u32)]>,
    /// Bitmask: bit `i` set means counter `i` has NO body-progress NFA
    /// states in the DFA successor (only re-seeded first-byte states).
    /// Existing instances for that counter must be cleared before seeding.
    counter_reset: u64,
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
            counting_mask: 0,
            seeds: Box::new([]),
            counter_reset: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 2 DFA cache
// ---------------------------------------------------------------------------

/// Lazy DFA cache for Tier 2 (differential counters + deferred assertions).
pub(crate) struct Tier2DfaCache {
    states: Vec<DfaState>,
    state_map: HashMap<DfaState, DfaStateId>,
    transitions: Vec<Transition>,
    stride: usize,
    // Scratch space for closure.
    closure_stack: Vec<StateIdx>,
    closure_result: Vec<StateIdx>,
    closure_deferred: Vec<StateIdx>,
    closure_visited: Vec<bool>,
    closure_seeds: Vec<(CounterIdx, StateIdx)>,
    regex_id: u64,
    start_id: DfaStateId,
    start_is_match: bool,
    start_is_match_at_end: bool,
    start_seeds: Box<[(CounterIdx, u32)]>,
    /// Per-counter: set of NFA consuming state indices that are in the
    /// "interior" of the counter body (i.e., body positions 1..L-1 for
    /// a body of length L).  For L=1, this is empty.
    ///
    /// Used to detect when a DFA successor has in-progress body instances:
    /// if the successor contains any interior state for counter `c`, then
    /// counter `c`'s instances should survive.  Otherwise, old instances
    /// should be cleared (only re-seeded first-byte instances remain).
    counter_body_interior: Vec<u32>,
    counter_body_ranges: Vec<(usize, usize)>,
}

impl fmt::Debug for Tier2DfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier2DfaCache")
            .field("num_states", &self.states.len())
            .finish()
    }
}

impl Tier2DfaCache {
    pub(crate) fn new(num_nfa_states: usize) -> Self {
        Self {
            states: Vec::new(),
            state_map: HashMap::new(),
            transitions: Vec::new(),
            stride: 256,
            closure_stack: Vec::new(),
            closure_result: Vec::new(),
            closure_deferred: Vec::new(),
            closure_visited: vec![false; num_nfa_states],
            closure_seeds: Vec::new(),
            regex_id: 0,
            start_id: DfaStateId::DEAD,
            start_is_match: false,
            start_is_match_at_end: false,
            start_seeds: Box::new([]),
            counter_body_interior: Vec::new(),
            counter_body_ranges: Vec::new(),
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
        let state = DfaState {
            nfa_states,
            deferred_asserts,
            is_match,
            is_match_at_end,
            prev_was_word,
        };
        if let Some(&id) = self.state_map.get(&state) {
            return Some(id);
        }
        if self.states.len() >= DFA_MAX_STATES {
            return None;
        }
        let id = DfaStateId(self.states.len() as u32);
        self.state_map.insert(state.clone(), id);
        self.states.push(state);
        self.transitions
            .extend(std::iter::repeat_with(Transition::empty).take(self.stride));
        Some(id)
    }

    // -----------------------------------------------------------------------
    // Epsilon closure (reuses Tier 1's approach + counter awareness)
    // -----------------------------------------------------------------------

    /// Compute epsilon closure.
    ///
    /// `follow_break`: whether to follow the CInc break path (out1).
    /// The continue path (out) is always followed.
    fn epsilon_closure(
        &mut self,
        seeds: impl Iterator<Item = StateIdx>,
        states: &[State],
        at_start: bool,
        prev_byte: Option<u8>,
        next_byte: Option<u8>,
        follow_break: bool,
    ) -> ClosureResult {
        self.closure_stack.clear();
        self.closure_result.clear();
        self.closure_deferred.clear();
        self.closure_seeds.clear();
        for v in self.closure_visited.iter_mut() {
            *v = false;
        }

        let mut is_match = false;
        let mut is_match_at_end = false;
        let mut encountered_cinc = false;

        self.closure_stack.extend(seeds);
        while let Some(idx) = self.closure_stack.pop() {
            let i = idx.idx();
            if self.closure_visited[i] {
                continue;
            }
            self.closure_visited[i] = true;

            match states[idx] {
                State::Split { out, out1 } => {
                    self.closure_stack.push(out1);
                    self.closure_stack.push(out);
                }
                State::Assert { kind, out } => {
                    if kind == AssertKind::End {
                        if self.can_reach_match(out, states) {
                            is_match_at_end = true;
                        }
                        continue;
                    }
                    let result = kind.eval(at_start, false, prev_byte, next_byte);
                    match result {
                        AssertEval::Pass => self.closure_stack.push(out),
                        AssertEval::Fail => {}
                        AssertEval::Defer => self.closure_deferred.push(idx),
                    }
                }
                State::Match => {
                    is_match = true;
                }
                State::CounterInstance { counter, out } => {
                    self.closure_stack.push(out);
                    self.closure_seeds.push((counter, out));
                }
                State::CounterIncrement { out, out1, .. } => {
                    encountered_cinc = true;
                    self.closure_stack.push(out);
                    if follow_break {
                        self.closure_stack.push(out1);
                    }
                }
                State::Byte { .. }
                | State::ByteCI { .. }
                | State::ByteClass { .. }
                | State::ByteTable { .. } => {
                    self.closure_result.push(idx);
                }
            }
        }

        self.closure_result.sort_unstable_by_key(|s| s.0);
        self.closure_deferred.sort_unstable_by_key(|s| s.0);

        let mut seed_instances: Vec<(CounterIdx, StateIdx)> = Vec::new();
        for &(counter, ci_out) in &self.closure_seeds {
            let consuming = consuming_states_from(ci_out, states);
            for c in consuming {
                seed_instances.push((counter, c));
            }
        }
        seed_instances.sort_by_key(|&(c, s)| (c.idx(), s.0));
        seed_instances.dedup();

        ClosureResult {
            nfa_states: self.closure_result.as_slice().into(),
            deferred_asserts: self.closure_deferred.as_slice().into(),
            is_match,
            is_match_at_end,
            encountered_cinc,
            seed_instances: seed_instances.into_boxed_slice(),
        }
    }

    fn can_reach_match(&self, start: StateIdx, states: &[State]) -> bool {
        let mut stack = vec![start];
        let mut visited = vec![false; states.len()];
        while let Some(idx) = stack.pop() {
            let i = idx.idx();
            if visited[i] {
                continue;
            }
            visited[i] = true;
            match states[idx] {
                State::Match => return true,
                State::Split { out, out1 } => {
                    stack.push(out1);
                    stack.push(out);
                }
                State::Assert { out, .. } => stack.push(out),
                State::CounterInstance { out, .. } => stack.push(out),
                State::CounterIncrement { out, .. } => {
                    stack.push(out);
                }
                _ => {}
            }
        }
        false
    }

    fn resolve_deferred(&self, from_state: &DfaState, byte: u8, regex: &Regex) -> Vec<StateIdx> {
        let mut extra = Vec::new();
        if from_state.deferred_asserts.is_empty() {
            return extra;
        }
        let prev = from_state.prev_byte_representative();
        for &assert_idx in from_state.deferred_asserts.iter() {
            if let State::Assert { kind, out } = regex.states[assert_idx]
                && kind.eval(false, false, prev, Some(byte)) == AssertEval::Pass
            {
                extra.push(out);
            }
        }
        extra
    }

    fn resolve_deferred_at_end(&self, state: &DfaState, regex: &Regex) -> bool {
        if state.deferred_asserts.is_empty() {
            return false;
        }
        let prev = state.prev_byte_representative();
        for &assert_idx in state.deferred_asserts.iter() {
            if let State::Assert { kind, out } = regex.states[assert_idx]
                && kind.eval(false, true, prev, None) == AssertEval::Pass
                && self.can_reach_match(out, &regex.states)
            {
                return true;
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Transition computation
    // -----------------------------------------------------------------------

    fn populate(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> Transition {
        let mut targets: Vec<StateIdx> = Vec::new();

        let mut resolved_seeds: Vec<(CounterIdx, u32)> = Vec::new();
        let mut resolved_cinc = false;
        let mut resolved_is_match = false;
        let mut resolved_is_match_at_end = false;

        if from != DfaStateId::DEAD {
            let from_state = &self.states[from.idx()];

            // Phase 1: resolve deferred assertions.
            let extra = self.resolve_deferred(from_state, byte, regex);
            if !extra.is_empty() {
                let resolved_prev = from_state.prev_byte_representative();
                let cr = self.epsilon_closure(
                    extra.into_iter(),
                    &regex.states,
                    false,
                    resolved_prev,
                    Some(byte),
                    true,
                );
                resolved_seeds = cr.seed_instances.iter().map(|&(c, _s)| (c, 0u32)).collect();

                resolved_cinc = cr.encountered_cinc;
                resolved_is_match = cr.is_match;
                resolved_is_match_at_end = cr.is_match_at_end;
                for &idx in cr.nfa_states.iter() {
                    if let Some(t) = consume_byte(idx, byte, regex) {
                        targets.push(t);
                    }
                }
            }

            // Phase 2: consuming states consume `byte`.
            let nfa_states = self.states[from.idx()].nfa_states.clone();
            for &idx in nfa_states.iter() {
                if let Some(t) = consume_byte(idx, byte, regex) {
                    targets.push(t);
                }
            }
        }

        // Probe: full "both" closure to detect CInc and collect seeds.
        let probe = self.epsilon_closure(
            targets.iter().copied().chain(std::iter::once(regex.start)),
            &regex.states,
            false,
            Some(byte),
            None,
            true,
        );

        let is_counting = probe.encountered_cinc || resolved_cinc;

        // Find ALL counters that fire CInc at this transition.
        let counting_mask = if is_counting {
            find_cinc_counters(&targets, &regex.states)
        } else {
            0
        };

        // For resolved seeds on a counting transition with L=1 body,
        // the seed's body byte was consumed on this very transition
        // (resolved assert -> CI -> body(L=1) -> consume -> CInc).
        // Since seeding happens after CInc processing in step_slow,
        // the seed misses its first increment.  Set initial_value=1.
        if is_counting {
            for s in &mut resolved_seeds {
                let ci = s.0.idx();
                let (_, _, body_len) = regex.counter_info(ci);
                if body_len == 1 {
                    s.1 = 1;
                }
            }
        }

        // Compute seed list.  Resolved seeds take priority (higher
        // initial value) over probe seeds for the same counter.
        let mut seed_list: Vec<(CounterIdx, u32)> = probe
            .seed_instances
            .iter()
            .map(|&(c, _s)| (c, 0u32))
            .collect();
        for s in &resolved_seeds {
            // If there is already a probe seed for this counter with
            // a lower initial value, replace it.
            if let Some(existing) = seed_list.iter_mut().find(|e| e.0 == s.0) {
                existing.1 = existing.1.max(s.1);
            } else {
                seed_list.push(*s);
            }
        }
        seed_list.sort_by_key(|&(c, _)| c.idx());
        seed_list.dedup();

        // Compute DFA successors.
        if is_counting {
            let cr_nb = self.epsilon_closure(
                targets.iter().copied().chain(std::iter::once(regex.start)),
                &regex.states,
                false,
                Some(byte),
                None,
                false,
            );
            let nb_id = self.intern_closure_result(&cr_nb, byte);
            let wb_id = self.intern_closure_result(&probe, byte);
            let (nb_m, nb_mae) = self.match_flags(nb_id);
            let (wb_m, wb_mae) = self.match_flags(wb_id);

            // For counting transitions, compute counter_reset for all
            // counters EXCEPT those firing CInc (managed by increment logic).
            let counter_reset = self.compute_counter_reset(&probe.nfa_states, counting_mask);

            Transition {
                no_break: nb_id,
                no_break_is_match: nb_m || resolved_is_match,
                no_break_is_match_at_end: nb_mae || resolved_is_match_at_end,
                with_break: wb_id,
                with_break_is_match: wb_m || resolved_is_match,
                with_break_is_match_at_end: wb_mae || resolved_is_match_at_end,
                is_counting: true,
                counting_mask,
                seeds: seed_list.into_boxed_slice(),
                counter_reset,
            }
        } else {
            let id = self.intern_closure_result(&probe, byte);
            let (m, mae) = self.match_flags(id);

            // For non-counting transitions, compute counter_reset for
            // all counters.
            let counter_reset = self.compute_counter_reset(&probe.nfa_states, 0);

            Transition {
                no_break: id,
                no_break_is_match: m || resolved_is_match,
                no_break_is_match_at_end: mae || resolved_is_match_at_end,
                with_break: id,
                with_break_is_match: m || resolved_is_match,
                with_break_is_match_at_end: mae || resolved_is_match_at_end,
                is_counting: false,
                counting_mask: 0,
                seeds: seed_list.into_boxed_slice(),
                counter_reset,
            }
        }
    }

    /// Compute the `counter_reset` bitmask for a transition.
    ///
    /// For each counter NOT in `skip_mask`, check if the successor's NFA
    /// states contain any "body interior" states.  If they don't, the
    /// counter's instances should be cleared (the body was interrupted).
    fn compute_counter_reset(&self, successor_nfa_states: &[StateIdx], skip_mask: u64) -> u64 {
        let mut mask: u64 = 0;
        for (ci, &(start, end)) in self.counter_body_ranges.iter().enumerate() {
            let interior = &self.counter_body_interior[start..end];
            if (skip_mask >> ci) & 1 != 0 {
                continue;
            }
            // For L=1 counters, interior is empty, so body is never
            // "in progress" → always reset on non-CInc transitions.
            if interior.is_empty() {
                mask |= 1u64 << ci;
                continue;
            }
            // Check if any of the successor's NFA states are in the interior.
            let has_interior = successor_nfa_states
                .iter()
                .any(|s| interior.binary_search(&s.0).is_ok());
            if !has_interior {
                mask |= 1u64 << ci;
            }
        }
        mask
    }

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

    fn match_flags(&self, id: DfaStateId) -> (bool, bool) {
        if id == DfaStateId::DEAD {
            (false, false)
        } else {
            let s = &self.states[id.idx()];
            (s.is_match, s.is_match_at_end)
        }
    }

    fn clear(&mut self, num_nfa_states: usize, stride: usize) {
        self.states.clear();
        self.state_map.clear();
        self.transitions.clear();
        self.stride = stride;
        self.closure_deferred.clear();
        self.closure_seeds.clear();
        self.closure_visited.clear();
        self.closure_visited.resize(num_nfa_states, false);
        self.regex_id = 0;
        self.start_id = DfaStateId::DEAD;
        self.start_is_match = false;
        self.start_is_match_at_end = false;
        self.start_seeds = Box::new([]);
        self.counter_body_interior.clear();
        self.counter_body_ranges.clear();
    }

    pub(crate) fn prepare(&mut self, regex: &Regex) {
        let id = regex.id;
        if self.regex_id == id && self.start_id != DfaStateId::DEAD {
            return;
        }
        self.clear(regex.states.len(), regex.num_byte_classes);
        self.regex_id = id;

        // Precompute body interior consuming states for each counter.
        // For a counter with body length L, the interior states are all
        // consuming states reachable from the body entry that are NOT the
        // first consuming state (i.e., body positions 1..L-1).
        let (flat, ranges) = compute_body_interiors(regex);
        self.counter_body_interior = flat;
        self.counter_body_ranges = ranges;

        let cr = self.epsilon_closure(
            std::iter::once(regex.start),
            &regex.states,
            true,
            None,
            None,
            true,
        );
        self.start_id = self
            .intern_state(
                cr.nfa_states,
                cr.deferred_asserts,
                cr.is_match,
                cr.is_match_at_end,
                false,
            )
            .expect("start state exceeds DFA_MAX_STATES");
        self.start_is_match = self.states[self.start_id.idx()].is_match;
        self.start_is_match_at_end = self.states[self.start_id.idx()].is_match_at_end;
        self.start_seeds = cr.seed_instances.iter().map(|&(c, _s)| (c, 0u32)).collect();
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

/// For each counter, compute the set of NFA consuming-state indices that
/// are in the "interior" of the counter body (body positions 1..L-1).
///
/// For a body of length L, the "first" consuming states are those reachable
/// from CI.out through epsilon states.  The "interior" states are all other
/// consuming states reachable within the body BEFORE CInc.
///
/// For L=1, there are no interior states (only the first consuming state).
fn compute_body_interiors(regex: &Regex) -> (Vec<u32>, Vec<(usize, usize)>) {
    let states = &regex.states;
    let byte_tables = &regex.byte_tables;
    let num_counters = regex.num_counters;
    let mut per_counter: Vec<Vec<u32>> = vec![Vec::new(); num_counters];

    for s in states.iter() {
        if let State::CounterInstance { counter, out } = s {
            let ci = counter.idx();
            // Find "first" consuming states: reachable from CI.out through
            // epsilon states only.
            let first = consuming_states_from(*out, states);
            let first_set: std::collections::HashSet<u32> = first.iter().map(|s| s.0).collect();

            // Walk the entire body from CI.out, following consuming states
            // through their successors, to find all consuming states.
            let mut all_body_consuming: Vec<u32> = Vec::new();
            let mut stack: Vec<StateIdx> = vec![*out];
            let mut visited = vec![false; states.len()];
            while let Some(idx) = stack.pop() {
                let i = idx.idx();
                if visited[i] {
                    continue;
                }
                visited[i] = true;
                match states[idx] {
                    State::CounterIncrement { counter: c, .. } if c == *counter => {
                        // End of body for this counter — don't follow further.
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
                        all_body_consuming.push(idx.0);
                        // Follow through the successor to find more body states.
                        stack.push(out);
                    }
                    State::ByteTable { table } => {
                        all_body_consuming.push(idx.0);
                        for &succ in byte_tables[table.idx()].0.iter() {
                            if succ != StateIdx::NONE {
                                stack.push(succ);
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Interior = all body consuming states minus the first states.
            let mut interior: Vec<u32> = all_body_consuming
                .into_iter()
                .filter(|s| !first_set.contains(s))
                .collect();
            interior.sort_unstable();
            interior.dedup();
            per_counter[ci] = interior;
        }
    }

    // Flatten into a single Vec with (start, end) ranges.
    let mut flat: Vec<u32> = Vec::new();
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(num_counters);
    for v in per_counter {
        let start = flat.len();
        flat.extend(v);
        ranges.push((start, flat.len()));
    }
    (flat, ranges)
}

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

/// Find ALL counters that fire CInc reachable from `targets`, as a bitmask.
fn find_cinc_counters(targets: &[StateIdx], states: &[State]) -> u64 {
    let mut mask: u64 = 0;
    for &t in targets {
        let mut stack = vec![t];
        let mut visited = vec![false; states.len()];
        while let Some(idx) = stack.pop() {
            let i = idx.idx();
            if visited[i] {
                continue;
            }
            visited[i] = true;
            match states[idx] {
                State::CounterIncrement { counter, .. } => {
                    mask |= 1u64 << counter.idx();
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
    }
    mask
}

// ---------------------------------------------------------------------------
// Tier 2 Matcher
// ---------------------------------------------------------------------------

/// Tier 2 DFA matcher with differential counters.
pub struct Tier2DfaMatcher<'a> {
    cache: &'a mut Tier2DfaCache,
    regex: &'a Regex,
    current: DfaStateId,
    /// Per-counter differential state.
    counter_states: Vec<CounterState>,
    ever_matched: bool,
    match_at_end: bool,
    has_live_counters: bool,
    prefilter: Prefilter,
}

impl<'a> Tier2DfaMatcher<'a> {
    pub(crate) fn new(cache: &'a mut Tier2DfaCache, regex: &'a Regex) -> Self {
        let counter_states: Vec<CounterState> = (0..regex.num_counters)
            .map(|ci| {
                let (min, max, body_len) = regex.counter_info(ci);
                CounterState::new(body_len.max(1), min as u32, max as u32)
            })
            .collect();

        // Seed initial counter instances.
        // At the start (before any bytes), active_phase is 0.
        // Seed into the phase that fires CInc when the body completes
        // (after L bytes).  That's phase (0 + L - 1) % L = L - 1.
        let mut has_live = false;
        let mut cs = counter_states;
        for &(counter, _initial_value) in cache.start_seeds.iter() {
            let c_idx = counter.idx();
            let nph = cs[c_idx].phases.len();
            let target_phase = (nph - 1) % nph;
            cs[c_idx].phases[target_phase].alloc_new();
            has_live = true;
        }

        Tier2DfaMatcher {
            current: cache.start_id,
            ever_matched: cache.start_is_match,
            match_at_end: cache.start_is_match_at_end,
            has_live_counters: has_live,
            cache,
            regex,
            counter_states: cs,
            prefilter: regex.prefilter,
        }
    }

    /// Slot lookup using byte-class compression (stride < 256).
    #[inline(always)]
    fn ensure_transition(&mut self, byte: u8) -> usize {
        let class = self.regex.byte_classes[byte as usize] as usize;
        let slot = self.current.idx() * self.cache.stride + class;
        if self.cache.transitions[slot].no_break == DfaStateId::UNPOPULATED {
            let trans = self.cache.populate(self.current, byte, self.regex);
            self.cache.transitions[slot] = trans;
        }
        slot
    }

    /// Slot lookup using identity mapping (stride = 256, no byte-class indirection).
    #[inline(always)]
    fn ensure_transition_direct(&mut self, byte: u8) -> usize {
        let slot = self.current.idx() * 256 + byte as usize;
        if self.cache.transitions[slot].no_break == DfaStateId::UNPOPULATED {
            let trans = self.cache.populate(self.current, byte, self.regex);
            self.cache.transitions[slot] = trans;
        }
        slot
    }

    #[inline(always)]
    fn step(&mut self, byte: u8) {
        if self.current == DfaStateId::DEAD {
            self.step_from_dead(byte);
            return;
        }

        let slot = self.ensure_transition(byte);
        self.step_inner(slot);
    }

    #[inline(always)]
    fn step_direct(&mut self, byte: u8) {
        if self.current == DfaStateId::DEAD {
            self.step_from_dead(byte);
            return;
        }

        let slot = self.ensure_transition_direct(byte);
        self.step_inner(slot);
    }

    #[inline(always)]
    fn step_inner(&mut self, slot: usize) {
        let t = &self.cache.transitions[slot];

        // Fast path: non-counting, no seeds, no live counters.
        if !t.is_counting && t.seeds.is_empty() && !self.has_live_counters {
            self.current = t.no_break;
            self.match_at_end = t.no_break_is_match_at_end;
            if t.no_break_is_match {
                self.ever_matched = true;
            }
            return;
        }

        self.match_at_end = false;
        let mut any_can_break = false;

        if t.is_counting {
            // Process ALL counters that fire CInc on this transition.
            let mut mask = t.counting_mask;
            while mask != 0 {
                let c_idx = mask.trailing_zeros() as usize;
                mask &= mask - 1; // clear lowest set bit

                if c_idx < self.counter_states.len() {
                    let cs = &mut self.counter_states[c_idx];
                    let phase_idx = cs.active_phase;
                    let phase = &mut cs.phases[phase_idx];

                    if !phase.is_empty() {
                        // Increment all instances in this phase.
                        phase.increment_all();

                        // Check break condition: oldest >= min.
                        if phase.oldest >= cs.min {
                            any_can_break = true;

                            // Deallocate instances that must break (value >= max).
                            while !phase.is_empty() && phase.oldest >= cs.max {
                                phase.dealloc_oldest();
                            }
                        }
                    }
                }
            }

            // Record match from break path.
            if any_can_break {
                if t.with_break_is_match {
                    self.ever_matched = true;
                }
                if t.with_break_is_match_at_end {
                    self.match_at_end = true;
                }
            }
        }

        // Select DFA successor.
        if t.is_counting && any_can_break {
            self.current = t.with_break;
        } else {
            self.current = t.no_break;
            if t.no_break_is_match {
                self.ever_matched = true;
            }
            if t.no_break_is_match_at_end {
                self.match_at_end = true;
            }
        }

        // Apply counter_reset: clear instances for counters whose body
        // was interrupted (no interior body NFA states in the successor).
        if t.counter_reset != 0 {
            for (ci, cs) in self.counter_states.iter_mut().enumerate() {
                if (t.counter_reset >> ci) & 1 != 0 {
                    cs.clear();
                }
            }
        }

        // Advance active_phase for EVERY counter on EVERY byte.
        // This keeps the phase clock synchronized with byte position.
        for cs in &mut self.counter_states {
            let nph = cs.phases.len();
            cs.active_phase = (cs.active_phase + 1) % nph;
        }

        // Seed new counter instances.
        for &(counter, initial_value) in t.seeds.iter() {
            let c_idx = counter.idx();
            if c_idx < self.counter_states.len() {
                let cs = &mut self.counter_states[c_idx];
                // Seed into the phase that will fire CInc when this
                // instance completes its first body iteration.  The body
                // has L bytes.  CInc fires L-1 bytes from now (the phase
                // clock has already been advanced for this byte).
                // active_phase was just advanced, so it's one step ahead.
                // We need the phase that fires after L-1 MORE bytes:
                //   target_phase = (active_phase + L - 1) % L
                let nph = cs.phases.len();
                let target_phase = (cs.active_phase + nph - 1) % nph;
                cs.phases[target_phase].alloc_new_with_value(initial_value);
            }
        }

        // Update has_live_counters flag.
        self.has_live_counters = self.counter_states.iter().any(|cs| !cs.is_empty());
    }

    fn step_from_dead(&mut self, byte: u8) {
        if self.has_live_counters {
            for cs in &mut self.counter_states {
                cs.clear();
            }
            self.has_live_counters = false;
        }
        self.match_at_end = false;

        let trans = self.cache.populate(DfaStateId::DEAD, byte, self.regex);
        self.current = trans.no_break;

        if !trans.is_counting {
            if trans.no_break_is_match {
                self.ever_matched = true;
            }
            if trans.no_break_is_match_at_end {
                self.match_at_end = true;
            }
        }

        // Advance active_phase for every counter (every byte ticks the clock).
        for cs in &mut self.counter_states {
            let nph = cs.phases.len();
            cs.active_phase = (cs.active_phase + 1) % nph;
        }

        // Seed with phase-aware placement.
        for &(counter, initial_value) in trans.seeds.iter() {
            let c_idx = counter.idx();
            if c_idx < self.counter_states.len() {
                let cs = &mut self.counter_states[c_idx];
                let nph = cs.phases.len();
                let target_phase = (cs.active_phase + nph - 1) % nph;
                cs.phases[target_phase].alloc_new_with_value(initial_value);
            }
        }
        self.has_live_counters = self.counter_states.iter().any(|cs| !cs.is_empty());
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

        if self.cache.stride == 256 {
            for &b in input {
                if self.ever_matched {
                    return;
                }
                self.step_direct(b);
            }
        } else {
            for &b in input {
                if self.ever_matched {
                    return;
                }
                self.step(b);
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
        if self.current != DfaStateId::DEAD {
            let state = &self.cache.states[self.current.idx()];
            if state.is_match_at_end {
                return true;
            }
            if self.cache.resolve_deferred_at_end(state, self.regex) {
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

impl fmt::Debug for Tier2DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier2DfaMatcher")
            .field("current", &self.current)
            .field("ever_matched", &self.ever_matched)
            .finish()
    }
}
