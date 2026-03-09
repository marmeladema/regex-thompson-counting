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

use std::fmt;

use crate::{
    AssertEval, CounterIdx, Prefilter, Regex, State, StateIdx, byte_match_ci, dfa::DfaMemory,
    is_word_byte,
};

use super::{DfaState, DfaStateId};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Sentinel value for "no node" in the [`DeltaPool`] linked list.
const SENTINEL: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Delta pool (arena-backed linked list for counter deltas)
// ---------------------------------------------------------------------------

/// Arena-backed pool of linked-list nodes for counter deltas.
///
/// All counters share a single `DeltaPool`.  Each [`DiffCounter`] owns a
/// linked list (head/tail indices) whose nodes live in this pool.  Freed
/// nodes are threaded into an intrusive free list and recycled by
/// subsequent allocations.
///
/// Between matches, [`reset`](Self::reset) clears the backing vecs
/// (retaining heap capacity) so the pool's memory is reused across
/// matcher invocations without per-counter loops.
#[derive(Clone, Debug)]
struct DeltaPool {
    /// Node payload (delta value).
    values: Vec<u32>,
    /// `next[i]` = index of the successor node, or [`SENTINEL`].
    next: Vec<u32>,
    /// Head of the intrusive free list threaded through `next[]`,
    /// or [`SENTINEL`] if the free list is empty.
    free_head: u32,
}

impl DeltaPool {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            next: Vec::new(),
            free_head: SENTINEL,
        }
    }

    /// Allocate a node with the given value.  Reuses a freed slot if
    /// available, otherwise appends to the end of the arena.
    #[inline]
    fn alloc(&mut self, val: u32) -> u32 {
        if self.free_head != SENTINEL {
            let idx = self.free_head;
            self.free_head = self.next[idx as usize];
            self.values[idx as usize] = val;
            self.next[idx as usize] = SENTINEL;
            idx
        } else {
            let idx = self.values.len() as u32;
            self.values.push(val);
            self.next.push(SENTINEL);
            idx
        }
    }

    /// Return a single node to the free list for reuse.
    #[inline]
    fn free_node(&mut self, idx: u32) {
        self.next[idx as usize] = self.free_head;
        self.free_head = idx;
    }

    /// Return an entire linked-list chain `[head … tail]` to the free
    /// list in O(1).  `tail.next` must be [`SENTINEL`].
    #[inline]
    fn free_chain(&mut self, head: u32, tail: u32) {
        debug_assert_ne!(head, SENTINEL);
        debug_assert_ne!(tail, SENTINEL);
        self.next[tail as usize] = self.free_head;
        self.free_head = head;
    }

    /// Clear all nodes (retaining heap capacity).
    fn reset(&mut self) {
        self.values.clear();
        self.next.clear();
        self.free_head = SENTINEL;
    }
}

// ---------------------------------------------------------------------------
// Differential counter (per-phase, no heap)
// ---------------------------------------------------------------------------

/// Differential counter for one phase of one counter.
///
/// All instances in the same phase increment in lockstep.  We store only
/// the oldest instance's absolute value and the count of active instances.
/// Deltas between consecutive instances are recorded as a linked list in
/// the shared [`DeltaPool`] so that when the oldest is deallocated, the
/// new oldest's value can be recovered in O(1).
#[derive(Clone, Debug)]
struct DiffCounter {
    /// Absolute value of the oldest (highest-value) instance.
    oldest: u32,
    /// Number of active instances.
    count: u32,
    /// Sum of all deltas (oldest_value − youngest_value).
    total_delta: u32,
    /// Head of the delta linked list in the pool, or [`SENTINEL`].
    head: u32,
    /// Tail of the delta linked list in the pool, or [`SENTINEL`].
    tail: u32,
}

impl DiffCounter {
    fn new() -> Self {
        Self {
            oldest: 0,
            count: 0,
            total_delta: 0,
            head: SENTINEL,
            tail: SENTINEL,
        }
    }

    /// Youngest instance value.
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
    fn dealloc_oldest(&mut self, pool: &mut DeltaPool) -> bool {
        debug_assert!(self.count > 0);
        self.count -= 1;
        if self.count > 0 {
            let old_head = self.head;
            debug_assert_ne!(old_head, SENTINEL);
            let d = pool.values[old_head as usize];
            self.head = pool.next[old_head as usize];
            if self.head == SENTINEL {
                self.tail = SENTINEL;
            }
            pool.free_node(old_head);
            self.oldest -= d;
            self.total_delta -= d;
            true
        } else {
            if self.head != SENTINEL {
                pool.free_chain(self.head, self.tail);
            }
            self.total_delta = 0;
            self.head = SENTINEL;
            self.tail = SENTINEL;
            false
        }
    }

    /// Allocate a new instance with a given initial value (youngest).
    #[inline]
    fn alloc_new_with_value(&mut self, value: u32, pool: &mut DeltaPool) {
        if self.count == 0 {
            self.oldest = value;
            self.count = 1;
            self.total_delta = 0;
        } else {
            let youngest_val = self.youngest();
            let gap = youngest_val - value;
            let node = pool.alloc(gap);
            if self.tail != SENTINEL {
                pool.next[self.tail as usize] = node;
            } else {
                self.head = node;
            }
            self.tail = node;
            self.total_delta += gap;
            self.count += 1;
        }
    }

    /// Allocate a new instance with value 0 (youngest).
    #[inline]
    fn alloc_new(&mut self, pool: &mut DeltaPool) {
        self.alloc_new_with_value(0, pool);
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Return nodes to the pool and reset to empty state.
    #[inline]
    fn clear(&mut self, pool: &mut DeltaPool) {
        if self.head != SENTINEL {
            pool.free_chain(self.head, self.tail);
        }
        self.oldest = 0;
        self.count = 0;
        self.total_delta = 0;
        self.head = SENTINEL;
        self.tail = SENTINEL;
    }
}

// ---------------------------------------------------------------------------
// Per-counter metadata (no heap)
// ---------------------------------------------------------------------------

/// Per-counter metadata indexing into the flat phases array.
#[derive(Clone, Debug)]
struct CounterMeta {
    /// Start index in the flat `phases` array.
    phase_start: usize,
    /// Number of phases (= body_length).
    num_phases: usize,
    /// Which phase fires CInc on the current byte (cycles 0..L-1).
    active_phase: usize,
    /// Counter parameters.
    min: u32,
    max: u32,
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
    memory: DfaMemory,
    closure_seeds: Vec<(CounterIdx, StateIdx)>,
    transitions: Vec<Transition>,
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
    /// Shared arena for delta linked-list nodes across all counters.
    delta_pool: DeltaPool,
    /// Flat array of differential counters for all phases of all counters.
    /// Indexed via [`CounterMeta::phase_start`].
    phases: Vec<DiffCounter>,
    /// Per-counter metadata (phase range, active_phase, min, max).
    counter_meta: Vec<CounterMeta>,
    /// Total number of phase slots (sum of num_phases across counters).
    /// Computed once in `prepare()`, used by `ensure_phases()` for lazy resize.
    total_phases: usize,
}

impl fmt::Debug for Tier2DfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier2DfaCache")
            .field("num_states", &self.memory.states.len())
            .finish()
    }
}

impl Tier2DfaCache {
    pub(crate) fn new(num_nfa_states: usize) -> Self {
        Self {
            memory: DfaMemory::new(num_nfa_states),
            transitions: Vec::new(),
            closure_seeds: Vec::new(),
            start_seeds: Box::new([]),
            counter_body_interior: Vec::new(),
            counter_body_ranges: Vec::new(),
            delta_pool: DeltaPool::new(),
            phases: Vec::new(),
            counter_meta: Vec::new(),
            total_phases: 0,
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
    // Counter helper methods (split-borrow safe: these operate on `self`
    // directly, so the borrow checker can see field disjointness)
    // -----------------------------------------------------------------------

    /// Increment the active phase of counter `c_idx`.
    /// Returns `true` if any instance can break (oldest >= min).
    #[inline]
    fn counter_increment(&mut self, c_idx: usize) -> bool {
        let m = &self.counter_meta[c_idx];
        let phase_idx = m.phase_start + m.active_phase;
        let min = m.min;
        let max = m.max;
        // m borrow ends here (NLL: all fields copied to locals).

        let phase = &mut self.phases[phase_idx];
        if phase.is_empty() {
            return false;
        }
        phase.increment_all();
        let can_break = phase.oldest >= min;
        if can_break {
            while !phase.is_empty() && phase.oldest >= max {
                phase.dealloc_oldest(&mut self.delta_pool);
            }
        }
        can_break
    }

    /// Clear all phases of counter `c_idx`, returning nodes to the pool.
    #[inline]
    fn counter_reset_idx(&mut self, c_idx: usize) {
        let start = self.counter_meta[c_idx].phase_start;
        let end = start + self.counter_meta[c_idx].num_phases;
        for p in &mut self.phases[start..end] {
            p.clear(&mut self.delta_pool);
        }
    }

    /// Advance active_phase for every counter (called on every byte).
    #[inline]
    fn advance_all_phases(&mut self) {
        for m in &mut self.counter_meta {
            m.active_phase = (m.active_phase + 1) % m.num_phases;
        }
    }

    /// Seed a new counter instance into the appropriate phase.
    ///
    /// Targets the phase that will fire CInc when this instance completes
    /// its first body iteration.  The body has L bytes, and the phase
    /// clock has already been advanced for this byte, so:
    ///   `target_phase = (active_phase + L - 1) % L`
    #[inline]
    fn seed_counter(&mut self, c_idx: usize, initial_value: u32) {
        let m = &self.counter_meta[c_idx];
        let nph = m.num_phases;
        let target_phase = (m.active_phase + nph - 1) % nph;
        let phase_idx = m.phase_start + target_phase;
        // m borrow ends here.
        self.phases[phase_idx].alloc_new_with_value(initial_value, &mut self.delta_pool);
    }

    /// Check if any phase has live instances.
    #[inline]
    fn has_live_phases(&self) -> bool {
        self.phases.iter().any(|p| !p.is_empty())
    }

    /// Clear all counter instances and reset active phases.
    fn clear_all_counters(&mut self) {
        for ci in 0..self.counter_meta.len() {
            let start = self.counter_meta[ci].phase_start;
            let end = start + self.counter_meta[ci].num_phases;
            for p in &mut self.phases[start..end] {
                p.clear(&mut self.delta_pool);
            }
            self.counter_meta[ci].active_phase = 0;
        }
    }

    /// Reset all phases and the delta pool for a new match.
    /// O(1): just clears the pool and phases vec (capacity retained).
    /// Phases are lazily repopulated by [`ensure_phases`] on the first
    /// slow-path byte that actually needs counter operations.
    fn reset_for_new_match(&mut self) {
        self.delta_pool.reset();
        self.phases.clear();
    }

    /// Lazily populate the phases vec and reset active_phase for all counters.
    /// Called once per match on the first slow-path byte; subsequent calls
    /// short-circuit on the `is_empty` check.
    #[inline]
    fn ensure_phases(&mut self) {
        if !self.phases.is_empty() {
            return;
        }
        self.phases.resize(self.total_phases, DiffCounter::new());
        for m in &mut self.counter_meta {
            m.active_phase = 0;
        }
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
        regex: &Regex,
        at_start: bool,
        prev_byte: Option<u8>,
        next_byte: Option<u8>,
        follow_break: bool,
    ) -> ClosureResult {
        let mut encountered_cinc = false;
        let mut cinc_mask: u64 = 0;

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
            |memory, counter, out, out1, _, _| {
                encountered_cinc = true;
                cinc_mask |= 1u64 << counter.idx();
                memory.closure_stack.push(out);
                if follow_break {
                    memory.closure_stack.push(out1);
                }
            },
        );

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
            cinc_mask,
            seed_instances: seed_instances.into_boxed_slice(),
        }
    }

    // -----------------------------------------------------------------------
    // Transition computation
    // -----------------------------------------------------------------------

    fn populate(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> Transition {
        let mut targets: Vec<StateIdx> = Vec::new();

        let mut resolved_seeds: Vec<(CounterIdx, u32)> = Vec::new();
        let mut resolved_cinc = false;
        let mut resolved_cinc_mask: u64 = 0;
        let mut resolved_is_match = false;
        let mut resolved_is_match_at_end = false;
        // Match reachable through CInc break path in Phase 1 resolution.
        // Only valid when any_can_break is true (counter >= min).
        let mut resolved_break_match = false;

        if from != DfaStateId::DEAD {
            let from_state = &self.memory.states[from.idx()];

            // Phase 1: resolve deferred assertions.
            let extra = from_state.resolve_deferred(byte, regex);
            if !extra.is_empty() {
                let resolved_prev = from_state.prev_byte_representative();
                // Use follow_break=false: the CInc break path should NOT
                // contribute to resolved_is_match.  Match via counter break
                // is handled by the with_break DFA successor, gated by
                // the counter condition in step_inner.
                let cr = self.epsilon_closure(
                    extra.into_iter(),
                    regex,
                    false,
                    resolved_prev,
                    Some(byte),
                    false,
                );
                resolved_seeds = cr.seed_instances.iter().map(|&(c, _s)| (c, 0u32)).collect();

                resolved_cinc = cr.encountered_cinc;
                resolved_cinc_mask = cr.cinc_mask;
                resolved_is_match = cr.is_match;
                resolved_is_match_at_end = cr.is_match_at_end;

                // If CInc was reached via Phase 1 resolution, check if
                // the break path (CInc.out1) can reach Match.  This match
                // is conditional on the counter meeting min.
                if cr.encountered_cinc {
                    // Check if any counter reached via Phase 1 resolution
                    // has a break path that can reach Match.
                    let mut mask = cr.cinc_mask;
                    while mask != 0 {
                        let ci = mask.trailing_zeros() as usize;
                        mask &= mask - 1;
                        if regex.counter_break_can_match[ci] {
                            resolved_break_match = true;
                            break;
                        }
                    }
                }
                for &idx in cr.nfa_states.iter() {
                    if let Some(t) = consume_byte(idx, byte, regex) {
                        targets.push(t);
                    }
                }
            }

            // Phase 2: consuming states consume `byte`.
            let nfa_states = self.memory.states[from.idx()].nfa_states.clone();
            for &idx in nfa_states.iter() {
                if let Some(t) = consume_byte(idx, byte, regex) {
                    targets.push(t);
                }
            }
        }

        // Probe: full "both" closure to detect CInc and collect seeds.
        let probe = self.epsilon_closure(
            targets.iter().copied().chain(std::iter::once(regex.start)),
            regex,
            false,
            Some(byte),
            None,
            true,
        );

        let is_counting = probe.encountered_cinc || resolved_cinc;

        // Find ALL counters that fire CInc at this transition.
        // Include counters reached via resolved deferred assertions
        // (Phase 1) which are not reachable from targets alone.
        let counting_mask = if is_counting {
            find_cinc_counters(&targets, &regex.states) | resolved_cinc_mask
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
                regex,
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
                with_break_is_match: wb_m || resolved_is_match || resolved_break_match,
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
            let s = &self.memory.states[id.idx()];
            (s.is_match, s.is_match_at_end)
        }
    }

    fn clear(&mut self, num_nfa_states: usize, stride: usize) {
        self.memory.clear(num_nfa_states, stride);
        self.transitions.clear();
        self.closure_seeds.clear();
        self.start_seeds = Box::new([]);
        self.counter_body_interior.clear();
        self.counter_body_ranges.clear();
        self.delta_pool.reset();
        self.phases.clear();
        self.counter_meta.clear();
        self.total_phases = 0;
    }

    #[inline]
    pub(crate) fn prepare(&mut self, regex: &Regex) {
        let id = regex.id;
        if self.memory.regex_id == id && self.memory.start_id != DfaStateId::DEAD {
            return;
        }
        self.clear(regex.states.len(), regex.num_byte_classes);
        self.memory.regex_id = id;

        // Precompute body interior consuming states for each counter.
        // For a counter with body length L, the interior states are all
        // consuming states reachable from the body entry that are NOT the
        // first consuming state (i.e., body positions 1..L-1).
        let (flat, ranges) = compute_body_interiors(regex);
        self.counter_body_interior = flat;
        self.counter_body_ranges = ranges;

        let cr = self.epsilon_closure(std::iter::once(regex.start), regex, true, None, None, true);
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
        self.start_seeds = cr.seed_instances.iter().map(|&(c, _s)| (c, 0u32)).collect();

        // Initialize flat phases and counter metadata.
        // Allocated once per regex, reused across matcher invocations.
        self.phases.clear();
        self.counter_meta.clear();
        for ci in 0..regex.num_counters {
            let (min, max, body_len) = regex.counter_info(ci);
            let nph = body_len.max(1);
            let phase_start = self.phases.len();
            for _ in 0..nph {
                self.phases.push(DiffCounter::new());
            }
            self.counter_meta.push(CounterMeta {
                phase_start,
                num_phases: nph,
                active_phase: 0,
                min: min as u32,
                max: max as u32,
            });
        }
        self.total_phases = self.phases.len();
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
    /// Bitmask of counters that reached CInc during this closure.
    cinc_mask: u64,
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
            // Find "first" consuming states: reachable from CI.out
            // through epsilon states only.
            let first = consuming_states_from(*out, states);
            let first_set: std::collections::HashSet<u32> = first.iter().map(|s| s.0).collect();

            // Walk the entire body from CI.out, following consuming
            // states through their successors, to find all consuming
            // states.
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
    ever_matched: bool,
    match_at_end: bool,
    has_live_counters: bool,
    prefilter: Prefilter,
}

impl<'a> Tier2DfaMatcher<'a> {
    #[inline]
    pub(crate) fn new(cache: &'a mut Tier2DfaCache, regex: &'a Regex) -> Self {
        // O(1) reset: clears the delta pool and phases vec.
        // Phases are lazily repopulated on the first slow-path byte.
        cache.reset_for_new_match();

        // Seed initial counter instances (only for patterns where the
        // start state directly reaches a CounterInstance via epsilons).
        // For most patterns (e.g. AWS keys), start_seeds is empty and
        // this block is skipped entirely — no counter work at all.
        let mut has_live = false;
        if !cache.start_seeds.is_empty() {
            cache.ensure_phases();
            // Iterate start_seeds by index to avoid borrow conflict with
            // seed_counter (which borrows cache mutably).
            let num_start_seeds = cache.start_seeds.len();
            for si in 0..num_start_seeds {
                let (counter, _initial_value) = cache.start_seeds[si];
                let c_idx = counter.idx();
                let m = &cache.counter_meta[c_idx];
                let nph = m.num_phases;
                let target_phase = (nph - 1) % nph;
                let phase_idx = m.phase_start + target_phase;
                cache.phases[phase_idx].alloc_new(&mut cache.delta_pool);
            }
            has_live = true;
        }

        Tier2DfaMatcher {
            current: cache.memory.start_id,
            ever_matched: cache.memory.start_is_match,
            match_at_end: cache.memory.start_is_match_at_end,
            has_live_counters: has_live,
            cache,
            regex,
            prefilter: regex.prefilter,
        }
    }

    /// Slot lookup using byte-class compression (stride < 256).
    #[inline(always)]
    fn ensure_transition(&mut self, byte: u8) -> usize {
        let class = self.regex.byte_classes[byte as usize] as usize;
        let slot = self.current.idx() * self.cache.memory.stride + class;
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

        // Slow path: snapshot scalar transition fields into locals so the
        // borrow on self.cache.transitions is released before we call
        // helper methods (including ensure_phases) that borrow self.cache
        // mutably.
        let is_counting = t.is_counting;
        let counting_mask = t.counting_mask;
        let no_break = t.no_break;
        let no_break_is_match = t.no_break_is_match;
        let no_break_is_match_at_end = t.no_break_is_match_at_end;
        let with_break = t.with_break;
        let with_break_is_match = t.with_break_is_match;
        let with_break_is_match_at_end = t.with_break_is_match_at_end;
        let counter_reset = t.counter_reset;
        let num_seeds = t.seeds.len();
        // `t` borrow ends here (NLL: last use was t.seeds.len()).

        // Ensure counter phases are populated before any counter ops.
        // First call per match does the work; subsequent calls short-circuit.
        self.cache.ensure_phases();

        self.match_at_end = false;
        let mut any_can_break = false;

        if is_counting {
            // Process ALL counters that fire CInc on this transition.
            let mut mask = counting_mask;
            while mask != 0 {
                let c_idx = mask.trailing_zeros() as usize;
                mask &= mask - 1; // clear lowest set bit
                if c_idx < self.cache.counter_meta.len() && self.cache.counter_increment(c_idx) {
                    any_can_break = true;
                }
            }

            // Record match from break path.
            if any_can_break {
                if with_break_is_match {
                    self.ever_matched = true;
                }
                if with_break_is_match_at_end {
                    self.match_at_end = true;
                }
            }
        }

        // Select DFA successor.
        if is_counting && any_can_break {
            self.current = with_break;
        } else {
            self.current = no_break;
            if no_break_is_match {
                self.ever_matched = true;
            }
            if no_break_is_match_at_end {
                self.match_at_end = true;
            }
        }

        // Apply counter_reset: clear instances for counters whose body
        // was interrupted (no interior body NFA states in the successor).
        if counter_reset != 0 {
            let mut mask = counter_reset;
            while mask != 0 {
                let c_idx = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                if c_idx < self.cache.counter_meta.len() {
                    self.cache.counter_reset_idx(c_idx);
                }
            }
        }

        // Advance active_phase for EVERY counter on EVERY byte.
        // This keeps the phase clock synchronized with byte position.
        self.cache.advance_all_phases();

        // Seed new counter instances.
        // Access seeds by index: each index yields a Copy tuple, so the
        // temporary borrow on self.cache.transitions ends before we
        // call seed_counter.
        for si in 0..num_seeds {
            let (counter, initial_value) = self.cache.transitions[slot].seeds[si];
            let c_idx = counter.idx();
            if c_idx < self.cache.counter_meta.len() {
                self.cache.seed_counter(c_idx, initial_value);
            }
        }

        // Update has_live_counters flag.
        self.has_live_counters = self.cache.has_live_phases();
    }

    fn step_from_dead(&mut self, byte: u8) {
        if self.has_live_counters {
            self.cache.clear_all_counters();
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

        // Ensure counter phases are populated before counter ops.
        self.cache.ensure_phases();

        // Advance active_phase for every counter (every byte ticks the clock).
        self.cache.advance_all_phases();

        // Seed with phase-aware placement.
        // `trans` is an owned local — no borrow conflict with cache.
        for &(counter, initial_value) in trans.seeds.iter() {
            let c_idx = counter.idx();
            if c_idx < self.cache.counter_meta.len() {
                self.cache.seed_counter(c_idx, initial_value);
            }
        }
        self.has_live_counters = self.cache.has_live_phases();
    }

    #[inline(always)]
    pub fn chunk(&mut self, input: &[u8]) {
        if self.ever_matched {
            return;
        }

        let start_id = self.cache.memory.start_id;

        // Prefilter-aware scanning: when the DFA is in the start state and
        // a prefilter is available, use memchr to skip ahead to the next
        // candidate byte instead of stepping byte-by-byte.  This is
        // re-engaged every time the DFA returns to the start state, not
        // just once at chunk entry.
        match self.prefilter {
            Prefilter::None => self.chunk_no_prefilter(input),
            Prefilter::Memchr1(needle) => {
                self.chunk_prefilter(input, start_id, |hay| memchr::memchr(needle, hay));
            }
            Prefilter::Memchr2(b1, b2) => {
                self.chunk_prefilter(input, start_id, |hay| memchr::memchr2(b1, b2, hay));
            }
            Prefilter::Memchr3(b1, b2, b3) => {
                self.chunk_prefilter(input, start_id, |hay| memchr::memchr3(b1, b2, b3, hay));
            }
            Prefilter::Range(lo, hi) => {
                self.chunk_prefilter(input, start_id, |hay| {
                    crate::memrange::memrange(lo, hi, hay)
                });
            }
        }
    }

    #[inline(always)]
    fn chunk_no_prefilter(&mut self, input: &[u8]) {
        if self.cache.memory.stride == 256 {
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

    /// Prefilter-integrated scanning loop.
    ///
    /// When `current == start_id`, the DFA is in its start state where
    /// non-matching bytes loop back to start.  Instead of stepping through
    /// those bytes one by one, we use the `finder` (memchr) to skip ahead
    /// to the next candidate byte.  Once a candidate is found, we step
    /// through the DFA normally until it returns to the start state (or
    /// matches / reaches end of input).
    #[inline(always)]
    fn chunk_prefilter(
        &mut self,
        input: &[u8],
        start_id: DfaStateId,
        finder: impl Fn(&[u8]) -> Option<usize>,
    ) {
        let mut i = 0;
        while i < input.len() {
            if self.ever_matched {
                return;
            }
            // When at the start state, skip ahead with memchr.
            if self.current == start_id && !self.has_live_counters {
                if let Some(offset) = finder(&input[i..]) {
                    i += offset;
                } else {
                    return;
                }
            }
            if self.cache.memory.stride == 256 {
                self.step_direct(input[i]);
            } else {
                self.step(input[i]);
            }
            i += 1;
        }
    }

    #[inline]
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
            // Standard deferred resolution: handles assertions outside
            // counter bodies (deferred assert -> Match path).
            if state.resolve_deferred_at_end(self.regex) {
                return true;
            }
            // Tier 2 specific: handle deferred assertions inside counter
            // bodies.  When a deferred assertion resolves at end-of-input
            // and reaches CInc, we need to check whether incrementing the
            // counter allows a break to Match.
            if self.resolve_deferred_cinc_at_end(state) {
                return true;
            }
        }
        false
    }

    /// Check if resolving deferred assertions at end-of-input reaches a
    /// CInc whose counter, after one more increment, allows a break to
    /// Match.  This handles patterns like `(\w\b){1,3}` where the last
    /// `\b` is only resolved at end-of-input.
    fn resolve_deferred_cinc_at_end(&self, state: &DfaState) -> bool {
        if state.deferred_asserts.is_empty() {
            return false;
        }
        let prev = state.prev_byte_representative();
        let states = &self.regex.states;
        for &assert_idx in state.deferred_asserts.iter() {
            if let State::Assert { kind, out } = states[assert_idx]
                && kind.eval(false, true, prev, None) == AssertEval::Pass
            {
                // The assertion passes at end-of-input.  Walk epsilon states
                // from `out` looking for CInc.
                let mut stack = vec![out];
                let mut visited = vec![false; states.len()];
                while let Some(idx) = stack.pop() {
                    let i = idx.idx();
                    if visited[i] {
                        continue;
                    }
                    visited[i] = true;
                    match states[idx] {
                        State::CounterIncrement { counter, min, .. } => {
                            let ci = counter.idx();
                            if !self.regex.counter_break_can_match[ci] {
                                continue;
                            }
                            // Check if this counter has a live instance
                            // that would reach >= min after one increment.
                            if ci < self.cache.counter_meta.len() {
                                let m = &self.cache.counter_meta[ci];
                                for ph in 0..m.num_phases {
                                    let phase = &self.cache.phases[m.phase_start + ph];
                                    if !phase.is_empty() && phase.oldest + 1 >= min as u32 {
                                        return true;
                                    }
                                }
                            }
                            // min=0: always breakable even with no live instances.
                            if min == 0 {
                                return true;
                            }
                        }
                        State::Split { out, out1 } => {
                            stack.push(out1);
                            stack.push(out);
                        }
                        State::Assert { kind, out } => {
                            // Nested assertion (unlikely but handle it):
                            // evaluate at end-of-input.
                            if kind.eval(false, true, prev, None) == AssertEval::Pass {
                                stack.push(out);
                            }
                        }
                        State::CounterInstance { out, .. } => stack.push(out),
                        _ => {}
                    }
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

impl fmt::Debug for Tier2DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier2DfaMatcher")
            .field("current", &self.current)
            .field("ever_matched", &self.ever_matched)
            .finish()
    }
}
