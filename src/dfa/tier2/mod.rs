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

pub(crate) mod eligibility;
pub(crate) mod overlap;

use std::fmt;

use crate::{
    AssertEval, AssertKind, CounterIdx, Prefilter, Regex, State, StateIdx, byte_match_ci,
    is_word_byte,
};

use super::{DfaCache, DfaMemory, DfaStateId};

/// Maximum counters for tier 2: the `counting_mask` and `counter_reset`
/// fields in `Transition` are `u64` bitmasks.
pub(crate) const MAX_TIER2_COUNTERS: usize = 64;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Delta pool (arena-backed linked list for counter deltas)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Differential counter (per-phase, ring-buffer deltas)
// ---------------------------------------------------------------------------

/// Differential counter for one phase of one counter.
///
/// All instances in the same phase increment in lockstep.  We store only
/// the oldest instance's absolute value and the count of active instances.
/// Deltas between consecutive instances are stored in a power-of-two
/// ring buffer for cache-friendly FIFO access: dealloc pops from head,
/// alloc pushes to tail, and both ends wrap around the same contiguous
/// memory.
#[derive(Clone, Debug)]
struct DiffCounter {
    /// Absolute value of the oldest (highest-value) instance.
    oldest: u32,
    /// Number of active instances.
    count: u32,
    /// Sum of all deltas (oldest_value − youngest_value).
    total_delta: u32,
    /// Ring buffer of delta values between consecutive instances.
    /// Capacity is always a power of two.
    ring: Vec<u32>,
    /// Index of the oldest delta in `ring` (head of queue).
    head: u32,
    /// Bitmask for fast modular indexing: `ring.len() - 1`.
    mask: u32,
}

impl DiffCounter {
    fn new() -> Self {
        Self {
            oldest: 0,
            count: 0,
            total_delta: 0,
            ring: Vec::new(),
            head: 0,
            mask: 0,
        }
    }

    /// Pre-allocate the ring buffer.  `max_instances` is the counter's
    /// `max` value — capped to avoid huge allocations for unbounded
    /// repetitions.  The ring grows on demand if more capacity is needed.
    fn reserve(&mut self, max_instances: usize) {
        // Cap initial allocation: 4096 entries = 16 KB per phase.
        // Unbounded counters (max = usize::MAX) start small and grow.
        let cap = max_instances.clamp(1, 4096).next_power_of_two();
        if self.ring.len() < cap {
            self.ring.resize(cap, 0);
        }
        self.mask = (self.ring.len() as u32) - 1;
    }

    /// Double the ring buffer capacity, preserving existing contents
    /// in their correct logical order.
    fn grow(&mut self) {
        let old_cap = self.ring.len();
        let new_cap = (old_cap * 2).max(4);
        let old_mask = self.mask;
        let head = self.head;
        let n = self.count.saturating_sub(1) as usize; // number of deltas

        let mut new_ring = vec![0u32; new_cap];
        for (i, dst) in new_ring[..n].iter_mut().enumerate() {
            *dst = self.ring[((head.wrapping_add(i as u32)) & old_mask) as usize];
        }
        self.ring = new_ring;
        self.mask = (new_cap as u32) - 1;
        self.head = 0;
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
    fn dealloc_oldest(&mut self) -> bool {
        debug_assert!(self.count > 0);
        self.count -= 1;
        if self.count > 0 {
            let d = self.ring[(self.head & self.mask) as usize];
            self.head = self.head.wrapping_add(1);
            self.oldest -= d;
            self.total_delta -= d;
            true
        } else {
            self.total_delta = 0;
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
            // Number of deltas = count - 1 (before adding the new one).
            // If the ring is full, grow it.
            let num_deltas = (self.count - 1) as usize;
            if num_deltas >= self.ring.len() {
                self.grow();
            }
            let youngest_val = self.youngest();
            let gap = youngest_val - value;
            let tail = self.head.wrapping_add(self.count - 1);
            self.ring[(tail & self.mask) as usize] = gap;
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

    /// Reset to empty state (ring buffer capacity retained).
    #[inline]
    fn clear(&mut self) {
        self.oldest = 0;
        self.count = 0;
        self.total_delta = 0;
        self.head = 0;
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
    /// Seeds to apply BEFORE counter_increment.
    ///
    /// These come from Phase 1 deferred-assertion resolution on counting
    /// transitions where the resolved path consumed the L=1 counter body
    /// (initial_value > 0).  The seed must be visible to the increment
    /// so the counter can break on this very transition.
    pre_seeds: Box<[(CounterIdx, u32)]>,
    /// Seeds to apply AFTER counter_increment and counter_reset.
    ///
    /// Probe seeds (initial_value = 0) that start new counter instances
    /// for subsequent transitions.
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
            pre_seeds: Box::new([]),
            seeds: Box::new([]),
            counter_reset: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 2 analysis (precomputed at build time)
// ---------------------------------------------------------------------------

/// Precomputed NFA analysis for Tier 2 patterns.
///
/// Built once at regex compile time by [`compute_tier2_analysis`],
/// stored on the [`Regex`] struct, and shared by all matchers.
#[derive(Debug)]
pub(crate) struct Tier2Analysis {
    /// Flat array of NFA consuming-state indices that are in the "interior"
    /// of each counter body (body positions 1..L-1 for a body of length L).
    /// For L=1 counters, there are no interior states.
    ///
    /// Used to detect when a DFA successor has in-progress body instances:
    /// if the successor contains any interior state for counter `c`, then
    /// counter `c`'s instances should survive.  Otherwise, old instances
    /// should be cleared (only re-seeded first-byte instances remain).
    body_interior: Box<[u32]>,
    /// Per-counter `(start, end)` range into [`body_interior`](Self::body_interior).
    body_ranges: Box<[(usize, usize)]>,
    /// Flat array of NFA consuming-state indices reachable from each
    /// counter's break target (`CInc.out1`) via epsilon transitions.
    /// These are the states that need to consume the current byte when
    /// a deferred-assertion-gated counter breaks (Bug 18).
    ///
    /// Precomputed at build time — depends only on the NFA structure.
    break_consuming: Box<[StateIdx]>,
    /// Per-counter `(start, end)` range into [`break_consuming`](Self::break_consuming).
    break_consuming_ranges: Box<[(usize, usize)]>,
    /// Per-counter flag: `true` if `Match` is directly reachable (without
    /// passing through an end-of-input assertion) from the counter's break
    /// path (`CInc.out1`).  Precomputed at build time.
    break_direct_match: Box<[bool]>,
    /// Per-NFA-state bitmask of counter indices whose `CounterIncrement`
    /// is reachable via epsilon transitions.  Indexed by `StateIdx`.
    /// Replaces the runtime `find_cinc_counters` and `deferred_cinc_mask`
    /// graph walks.
    cinc_reachable: Box<[u64]>,
    /// Flat array of consuming NFA states reachable from each counter's
    /// `CounterInstance.out` via epsilon transitions.  Used to build
    /// seed instances at populate time without a graph walk.
    ci_seed_consuming: Box<[StateIdx]>,
    /// Per-counter `(start, end)` range into [`ci_seed_consuming`].
    ci_seed_consuming_ranges: Box<[(usize, usize)]>,
}

impl Tier2Analysis {
    /// Returns the sorted slice of interior NFA state indices for counter
    /// `ci`.  Empty for L=1 counters.
    pub(crate) fn interior(&self, ci: usize) -> &[u32] {
        let (start, end) = self.body_ranges[ci];
        &self.body_interior[start..end]
    }

    /// Returns `true` if `Match` is directly reachable (without an
    /// end-of-input assertion) from counter `ci`'s break path.
    pub(crate) fn break_has_direct_match(&self, ci: usize) -> bool {
        self.break_direct_match[ci]
    }

    /// Returns the bitmask of counter indices whose `CounterIncrement`
    /// is reachable via epsilon transitions from NFA state `idx`.
    pub(crate) fn cinc_reachable_from(&self, idx: StateIdx) -> u64 {
        self.cinc_reachable[idx.idx()]
    }

    /// Returns the consuming NFA states reachable from counter `ci`'s
    /// `CounterInstance.out` via epsilon transitions.
    pub(crate) fn seed_consuming(&self, ci: usize) -> &[StateIdx] {
        let (start, end) = self.ci_seed_consuming_ranges[ci];
        &self.ci_seed_consuming[start..end]
    }

    /// Returns the slice of consuming NFA states reachable from counter
    /// `ci`'s break target (`CInc.out1`) via epsilon transitions.
    pub(crate) fn break_consuming(&self, ci: usize) -> &[StateIdx] {
        let (start, end) = self.break_consuming_ranges[ci];
        &self.break_consuming[start..end]
    }
}

/// Precompute body interior data and break-path consuming states for all
/// counters in a Tier 2 pattern.
///
/// For each counter, identifies:
/// 1. **Interior states** — NFA consuming states in the "interior" of the
///    counter body (body positions 1..L-1 for a body of length L).  Used to
///    detect when a DFA successor has in-progress body instances.
/// 2. **Break consuming states** — NFA consuming states reachable from
///    `CInc.out1` (the break target) via epsilon transitions.  Used when
///    a deferred-assertion-gated counter breaks (Bug 18).
pub(crate) fn compute_tier2_analysis(
    states: &[State],
    byte_tables: &[crate::ByteMap],
    num_counters: usize,
) -> Tier2Analysis {
    let mut per_counter: Vec<Vec<u32>> = vec![Vec::new(); num_counters];
    let mut break_per_counter: Vec<Vec<StateIdx>> = vec![Vec::new(); num_counters];
    let mut break_direct: Vec<bool> = vec![false; num_counters];

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
                    State::Byte { out, out_exit, .. }
                    | State::ByteCI { out, out_exit, .. }
                    | State::ByteClass { out, out_exit, .. } => {
                        all_body_consuming.push(idx.0);
                        // Follow through the successor to find more body states.
                        stack.push(out);
                        if out_exit != StateIdx::NONE {
                            stack.push(out_exit);
                        }
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

        // Collect break-path consuming states and match reachability
        // for each CInc.
        if let State::CounterIncrement { counter, out1, .. } = *s {
            let ci = counter.idx();
            if ci < num_counters {
                break_per_counter[ci] = collect_break_consuming(out1, states);
                let (direct, _at_end) = break_path_match_kind(out1, states);
                if direct {
                    break_direct[ci] = true;
                }
            }
        }
    }

    // Flatten body interior into a single Vec with (start, end) ranges.
    let mut flat: Vec<u32> = Vec::new();
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(num_counters);
    for v in per_counter {
        let start = flat.len();
        flat.extend(v);
        ranges.push((start, flat.len()));
    }

    // Flatten break consuming into a single Vec with (start, end) ranges.
    let mut break_flat: Vec<StateIdx> = Vec::new();
    let mut break_ranges: Vec<(usize, usize)> = Vec::with_capacity(num_counters);
    for v in break_per_counter {
        let start = break_flat.len();
        break_flat.extend(v);
        break_ranges.push((start, break_flat.len()));
    }

    // Precompute per-NFA-state CInc reachability bitmask.
    // For each state, walk epsilon transitions and record which
    // CounterIncrement counters are reachable.
    let n = states.len();
    let mut cinc_reachable = vec![0u64; n];
    for (si, _) in states.iter().enumerate() {
        let mut mask = 0u64;
        let mut stack = vec![StateIdx(si as u32)];
        let mut visited = vec![false; n];
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
        cinc_reachable[si] = mask;
    }

    // Precompute per-counter seed consuming states: consuming NFA states
    // reachable from each CounterInstance.out via epsilon transitions.
    let mut seed_per_counter: Vec<Vec<StateIdx>> = vec![Vec::new(); num_counters];
    for s in states.iter() {
        if let State::CounterInstance { counter, out } = *s {
            let ci = counter.idx();
            if ci < num_counters && seed_per_counter[ci].is_empty() {
                seed_per_counter[ci] = consuming_states_from(out, states);
            }
        }
    }
    let mut seed_flat: Vec<StateIdx> = Vec::new();
    let mut seed_ranges: Vec<(usize, usize)> = Vec::with_capacity(num_counters);
    for v in seed_per_counter {
        let start = seed_flat.len();
        seed_flat.extend(v);
        seed_ranges.push((start, seed_flat.len()));
    }

    Tier2Analysis {
        body_interior: flat.into_boxed_slice(),
        body_ranges: ranges.into_boxed_slice(),
        break_consuming: break_flat.into_boxed_slice(),
        break_consuming_ranges: break_ranges.into_boxed_slice(),
        break_direct_match: break_direct.into_boxed_slice(),
        cinc_reachable: cinc_reachable.into_boxed_slice(),
        ci_seed_consuming: seed_flat.into_boxed_slice(),
        ci_seed_consuming_ranges: seed_ranges.into_boxed_slice(),
    }
}

/// Collect consuming NFA states reachable from `start` via epsilon
/// transitions (Split, Assert, CounterInstance).  Stops at consuming
/// states and CInc — does not recurse into nested counters.
fn collect_break_consuming(start: StateIdx, states: &[State]) -> Vec<StateIdx> {
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
                stack.push(out);
                stack.push(out1);
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

// ---------------------------------------------------------------------------
// Counter state (differential counter runtime)
// ---------------------------------------------------------------------------

/// Runtime state for all differential counters in a Tier 2 matcher.
///
/// Owns the per-phase [`DiffCounter`]s and per-counter [`CounterMeta`].
/// Extracted from [`Tier2DfaCache`] so that counter operations can be
/// called while other cache fields (e.g. `transitions`) are borrowed.
pub(crate) struct CounterState {
    /// Flat array of differential counters for all phases of all counters.
    /// Indexed via [`CounterMeta::phase_start`].
    phases: Vec<DiffCounter>,
    /// Per-counter metadata (phase range, active_phase, min, max).
    meta: Vec<CounterMeta>,
    /// Total number of phase slots (sum of num_phases across counters).
    /// Computed once in [`Tier2DfaCache::prepare`].
    total_phases: usize,
}

impl CounterState {
    fn new() -> Self {
        Self {
            phases: Vec::new(),
            meta: Vec::new(),
            total_phases: 0,
        }
    }

    /// Populate counter metadata from the regex's counter info.
    /// Called once per regex in [`Tier2DfaCache::prepare`].
    fn populate(&mut self, regex: &crate::Regex) {
        self.phases.clear();
        self.meta.clear();
        for ci in 0..regex.num_counters {
            let (min, max, body_len) = regex.counter_info(ci);
            let nph = body_len.max(1);
            let phase_start = self.phases.len();
            for _ in 0..nph {
                self.phases.push(DiffCounter::new());
            }
            self.meta.push(CounterMeta {
                phase_start,
                num_phases: nph,
                active_phase: 0,
                min: min as u32,
                max: max as u32,
            });
        }
        self.total_phases = self.phases.len();
    }

    /// Ensure phases are allocated (first match) or already present
    /// (subsequent matches — ring buffers survive via [`reset`]).
    #[inline]
    fn ensure_phases(&mut self) {
        if !self.phases.is_empty() {
            return;
        }
        self.phases.resize(self.total_phases, DiffCounter::new());
        for m in &mut self.meta {
            m.active_phase = 0;
            let max = m.max as usize;
            for p in &mut self.phases[m.phase_start..m.phase_start + m.num_phases] {
                p.reserve(max);
            }
        }
    }

    /// Reset counter state between matches.  Clears logical state of
    /// each phase but retains ring buffer allocations for reuse.
    fn reset(&mut self) {
        for p in &mut self.phases {
            p.clear();
        }
        for m in &mut self.meta {
            m.active_phase = 0;
        }
    }

    /// Clear everything including metadata (used when switching regex).
    fn clear(&mut self) {
        self.phases.clear();
        self.meta.clear();
        self.total_phases = 0;
    }

    /// Increment the active phase of counter `c_idx`.
    /// Returns `true` if any instance can break (oldest >= min).
    #[inline]
    fn increment(&mut self, c_idx: usize) -> bool {
        let m = &self.meta[c_idx];
        let phase_idx = m.phase_start + m.active_phase;
        let min = m.min;
        let max = m.max;

        let phase = &mut self.phases[phase_idx];
        if phase.is_empty() {
            return false;
        }
        phase.increment_all();
        let can_break = phase.oldest >= min;
        if can_break {
            while !phase.is_empty() && phase.oldest >= max {
                phase.dealloc_oldest();
            }
        }
        can_break
    }

    /// Clear all phases of counter `c_idx`.
    #[inline]
    fn reset_counter(&mut self, c_idx: usize) {
        let start = self.meta[c_idx].phase_start;
        let end = start + self.meta[c_idx].num_phases;
        for p in &mut self.phases[start..end] {
            p.clear();
        }
    }

    /// Advance active_phase for every counter (called on every byte).
    #[inline]
    fn advance_all_phases(&mut self) {
        for m in &mut self.meta {
            m.active_phase = (m.active_phase + 1) % m.num_phases;
        }
    }

    /// Seed a new counter instance into the appropriate phase.
    #[inline]
    fn seed(&mut self, c_idx: usize, initial_value: u32) {
        let m = &self.meta[c_idx];
        let nph = m.num_phases;
        let target_phase = (m.active_phase + nph - 1) % nph;
        let phase_idx = m.phase_start + target_phase;
        self.phases[phase_idx].alloc_new_with_value(initial_value);
    }

    /// Check if any phase has live instances.
    #[inline]
    fn has_live(&self) -> bool {
        self.phases.iter().any(|p| !p.is_empty())
    }
}

// ---------------------------------------------------------------------------
// Tier 2 DFA cache
// ---------------------------------------------------------------------------

/// Lazy DFA cache for Tier 2 (differential counters + deferred assertions).
pub(crate) struct Tier2DfaCache {
    inner: DfaCache,
    stride: usize,
    transitions: Vec<Transition>,
    start_seeds: Box<[(CounterIdx, u32)]>,
    /// Runtime state for all differential counters.
    counters: CounterState,
}

impl fmt::Debug for Tier2DfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier2DfaCache")
            .field("num_states", &self.inner.states.len())
            .finish()
    }
}

impl Tier2DfaCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: DfaCache::new(),
            stride: 256,
            transitions: Vec::new(),
            start_seeds: Box::new([]),
            counters: CounterState::new(),
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
    // Epsilon closure (reuses Tier 1's approach + counter awareness)
    // -----------------------------------------------------------------------

    /// Compute epsilon closure.
    ///
    /// `follow_break`: whether to follow the CInc break path (out1).
    /// The continue path (out) is always followed.
    #[allow(clippy::too_many_arguments)]
    fn epsilon_closure(
        &mut self,
        memory: &mut DfaMemory,
        seeds: impl Iterator<Item = StateIdx>,
        regex: &Regex,
        analysis: &Tier2Analysis,
        at_start: bool,
        prev_byte: Option<u8>,
        next_byte: Option<u8>,
        follow_break: bool,
    ) -> ClosureResult {
        let mut encountered_cinc = false;
        let mut cinc_mask: u64 = 0;

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
            |mem, counter, out, out1, _, _| {
                encountered_cinc = true;
                cinc_mask |= 1u64 << counter.idx();
                mem.closure_stack.push(out);
                if follow_break {
                    mem.closure_stack.push(out1);
                }
            },
        );

        let mut seed_instances: Vec<(CounterIdx, StateIdx)> = Vec::new();
        for &(counter, _ci_out) in &memory.closure_seeds {
            for &c in analysis.seed_consuming(counter.idx()) {
                seed_instances.push((counter, c));
            }
        }
        seed_instances.sort();
        seed_instances.dedup();

        ClosureResult {
            nfa_states: memory.closure_result.as_slice().into(),
            deferred_asserts: memory.closure_deferred.as_slice().into(),
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

    fn populate(
        &mut self,
        memory: &mut DfaMemory,
        from: DfaStateId,
        byte: u8,
        regex: &Regex,
        analysis: &Tier2Analysis,
    ) -> Transition {
        let mut targets: Vec<StateIdx> = Vec::new();

        let mut resolved_seeds: Vec<(CounterIdx, u32)> = Vec::new();
        let mut resolved_cinc = false;
        let mut resolved_cinc_mask: u64 = 0;
        let mut resolved_is_match = false;
        // Note: we do NOT track resolved_is_match_at_end here.  That flag
        // means "the deferred assertion passed mid-stream (with next=byte) and
        // $ → Match was reachable."  But the assertion was evaluated with
        // next=Some(byte) — at actual end-of-input, next is None and word-
        // boundary conditions may flip.  finish() correctly re-evaluates via
        // resolve_deferred_at_end() and resolve_deferred_cinc_at_end().
        // Match reachable through CInc break path in Phase 1 resolution.
        // Only valid when any_can_break is true (counter >= min).
        let mut resolved_break_match = false;
        // Whether Phase 1 resolution actually produced consuming targets
        // (the first counter body byte was consumed on this transition).
        // When false, resolved_seeds are speculative and should NOT be
        // used on non-counting transitions (the body byte doesn't match).
        let mut resolved_body_consumed = false;
        // Targets from consuming states on CInc break paths, consumed on
        // this transition.  These states are only reachable when the
        // deferred assertion passes AND the counter breaks, so they are
        // added exclusively to the `with_break` DFA successor (Bug 18).
        let mut resolved_break_targets: Vec<StateIdx> = Vec::new();

        if from != DfaStateId::DEAD {
            let from_state = &self.inner.states[from.idx()];
            // Phase 1: resolve deferred assertions.
            let extra = from_state.resolve_deferred(byte, regex);
            if !extra.is_empty() {
                let resolved_prev = from_state.prev_byte_representative();
                // Use follow_break=false: the CInc break path should NOT
                // contribute to resolved_is_match.  Match via counter break
                // is handled by the with_break DFA successor, gated by
                // the counter condition in step_inner.
                let cr = self.epsilon_closure(
                    memory,
                    extra.into_iter(),
                    regex,
                    analysis,
                    false,
                    resolved_prev,
                    Some(byte),
                    false,
                );
                resolved_seeds = cr.seed_instances.iter().map(|&(c, _s)| (c, 0u32)).collect();

                resolved_cinc = cr.encountered_cinc;
                resolved_cinc_mask = cr.cinc_mask;
                resolved_is_match = cr.is_match;
                // Discard cr.is_match_at_end — see comment above.

                // If CInc was reached via Phase 1 resolution, check if
                // the break path (CInc.out1) can reach Match directly
                // (without going through $ or end-line assertions).
                //
                // Phase 1 fires mid-stream (we have a next byte), so $
                // cannot pass at the break position.  Paths through $
                // are handled at actual end-of-input by
                // resolve_deferred_cinc_at_end() in finish().
                if cr.encountered_cinc {
                    let mut mask = cr.cinc_mask;
                    while mask != 0 {
                        let ci = mask.trailing_zeros() as usize;
                        mask &= mask - 1;

                        // Check direct (epsilon-only) match from break
                        // target — precomputed at analysis time.
                        if regex.counter_break_can_match[ci] && analysis.break_has_direct_match(ci)
                        {
                            resolved_break_match = true;
                        }

                        // Consume the current byte at precomputed
                        // break-path consuming states (Bug 18).
                        // These are only reachable when the counter
                        // breaks AND the deferred assertion passed.
                        for &bc in analysis.break_consuming(ci) {
                            if let Some((t, te)) = consume_byte(bc, byte, regex) {
                                resolved_break_targets.push(t);
                                if te != StateIdx::NONE {
                                    resolved_break_targets.push(te);
                                }
                            }
                        }
                    }
                }
                for &idx in cr.nfa_states.iter() {
                    if let Some((t, te)) = consume_byte(idx, byte, regex) {
                        targets.push(t);
                        if te != StateIdx::NONE {
                            targets.push(te);
                        }
                        resolved_body_consumed = true;
                    }
                }
            }

            // Phase 2: consuming states consume `byte`.
            let nfa_states = self.inner.states[from.idx()].nfa_states.clone();
            for &idx in nfa_states.iter() {
                if let Some((t, te)) = consume_byte(idx, byte, regex) {
                    targets.push(t);
                    if te != StateIdx::NONE {
                        targets.push(te);
                    }
                }
            }
        }

        // Probe: full "both" closure to detect CInc and collect seeds.
        let probe = self.epsilon_closure(
            memory,
            targets.iter().copied().chain(std::iter::once(regex.start)),
            regex,
            analysis,
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
            targets
                .iter()
                .fold(0u64, |acc, &t| acc | analysis.cinc_reachable_from(t))
                | resolved_cinc_mask
        } else {
            0
        };

        // Debug assertion for non-overlap-proven patterns: the disjoint-bytes
        // guarantee means counting_mask should have at most 1 bit set.
        // For overlap-proven patterns, multi-CInc is expected and was
        // validated by the binary-exactness proof at compile time.
        debug_assert!(
            regex.tier2_overlap_proven || counting_mask.count_ones() <= 1,
            "Tier 2 disjoint-body pattern has counting_mask with {} bits set \
             on DFA state {:?} byte {}: this violates the disjoint-bytes \
             invariant (the overlap proof was not used for this pattern)",
            counting_mask.count_ones(),
            from,
            byte,
        );

        // Build the post-seed list from probe seeds.
        let mut seed_list: Vec<(CounterIdx, u32)> = probe
            .seed_instances
            .iter()
            .map(|&(c, _s)| (c, 0u32))
            .collect();
        // Resolved seeds from deferred-assertion resolution are NOT
        // merged into seed_list.  They go to pre_seeds (built below)
        // for correct phase alignment — see the pre_seed comment.
        seed_list.sort_by_key(|&(c, _)| c);
        seed_list.dedup();

        // Compute DFA successors.
        if is_counting {
            let cr_nb = self.epsilon_closure(
                memory,
                targets.iter().copied().chain(std::iter::once(regex.start)),
                regex,
                analysis,
                false,
                Some(byte),
                None,
                false,
            );
            let nb_id = self.intern_closure_result(&cr_nb, byte);
            // When Phase 1 resolved a deferred assertion and reached
            // CInc, the break path's consuming states may have consumed
            // `byte`.  Those targets are only reachable when the counter
            // actually breaks, so they go into the with_break closure
            // exclusively (Bug 18).
            let wb_id = if resolved_break_targets.is_empty() {
                self.intern_closure_result(&probe, byte)
            } else {
                let wb_closure = self.epsilon_closure(
                    memory,
                    targets
                        .iter()
                        .copied()
                        .chain(resolved_break_targets.iter().copied())
                        .chain(std::iter::once(regex.start)),
                    regex,
                    analysis,
                    false,
                    Some(byte),
                    None,
                    true,
                );
                self.intern_closure_result(&wb_closure, byte)
            };
            let (nb_m, nb_mae) = self.match_flags(nb_id);
            let (wb_m, wb_mae) = self.match_flags(wb_id);

            // For counting transitions, compute counter_reset for all
            // counters EXCEPT those firing CInc (managed by increment logic).
            let counter_reset =
                self.compute_counter_reset(analysis, &probe.nfa_states, counting_mask);

            // Resolved seeds from deferred assertions must be pre_seeds
            // (applied BEFORE advance_all_phases) to get the correct
            // Tier 2 phase alignment.  Phase 1 already consumed one
            // body byte, so the instance must be placed one phase
            // ahead of what seed_counter's default targeting computes.
            // Pre-seed timing achieves this: active_phase has not yet
            // been advanced for this byte, so the target phase is
            // effectively shifted by -1 relative to a post-advance
            // seed.
            //
            // For L=1 bodies: the seed goes to pre_seeds with value 0;
            // the counter_increment on this transition provides the +1.
            // For L>1 bodies: the seed goes to pre_seeds with value 0;
            // the next CInc (after the remaining L-1 body bytes) will
            // provide the increment.
            //
            // Skip reset: the counter must NOT be cleared by
            // counter_reset, because the pre-seeded instance
            // represents valid work from the resolved assertion path.
            //
            // When the body was NOT consumed (e.g. `\B {7,12}` on `!`),
            // the resolved seeds are speculative and must be discarded
            // to avoid false positives (Bug 14).
            let mut resolved_body_mask: u64 = 0;
            let mut pre_seed_list: Vec<(CounterIdx, u32)> = Vec::new();
            if resolved_body_consumed {
                for &(counter, _val) in &resolved_seeds {
                    let ci = counter.idx();
                    resolved_body_mask |= 1u64 << ci;
                    pre_seed_list.push((counter, 0));
                }
            }
            let counter_reset = counter_reset & !resolved_body_mask;

            Transition {
                no_break: nb_id,
                no_break_is_match: nb_m || resolved_is_match,
                no_break_is_match_at_end: nb_mae,
                with_break: wb_id,
                with_break_is_match: wb_m || resolved_is_match || resolved_break_match,
                with_break_is_match_at_end: wb_mae,
                is_counting: true,
                counting_mask,
                pre_seeds: pre_seed_list.into_boxed_slice(),
                seeds: seed_list.into_boxed_slice(),
                counter_reset,
            }
        } else {
            let id = self.intern_closure_result(&probe, byte);
            let (m, mae) = self.match_flags(id);

            // For non-counting transitions, compute counter_reset for
            // all counters.
            let counter_reset = self.compute_counter_reset(analysis, &probe.nfa_states, 0);

            // Resolved seeds from deferred assertions go to pre_seeds
            // for correct phase alignment (same rationale as counting
            // transitions — see comment above).
            let mut resolved_body_mask: u64 = 0;
            let mut pre_seed_list: Vec<(CounterIdx, u32)> = Vec::new();
            if resolved_body_consumed {
                for &(counter, _val) in &resolved_seeds {
                    let ci = counter.idx();
                    resolved_body_mask |= 1u64 << ci;
                    pre_seed_list.push((counter, 0));
                }
            }
            let counter_reset = counter_reset & !resolved_body_mask;

            // Counters reachable from deferred assertions (for pre-reset
            // snapshot in step_inner).
            let dcm = probe
                .deferred_asserts
                .iter()
                .fold(0u64, |acc, &assert_idx| {
                    if let State::Assert { out, .. } = regex.states[assert_idx] {
                        acc | analysis.cinc_reachable_from(out)
                    } else {
                        acc
                    }
                });

            // For L=1 counter bodies with deferred assertions gating
            // CInc, the probe can't see through the deferred assertion
            // to find CI → body seeds.  If counter_reset will clear
            // the counter AND dcm shows CInc is behind a deferred
            // assertion, inject a seed so the counter is re-populated
            // after clearing.  This allows the counter to accumulate
            // across transitions where the assertion always defers.
            let needs_deferred_seed = dcm & counter_reset;
            if needs_deferred_seed != 0 {
                let mut mask = needs_deferred_seed;
                while mask != 0 {
                    let ci = mask.trailing_zeros() as usize;
                    mask &= mask - 1;
                    let counter = CounterIdx(ci as u8);
                    if !seed_list.iter().any(|s| s.0 == counter) {
                        seed_list.push((counter, 0));
                    }
                }
                seed_list.sort_by_key(|&(c, _)| c);
            }

            Transition {
                no_break: id,
                no_break_is_match: m || resolved_is_match,
                no_break_is_match_at_end: mae,
                with_break: id,
                with_break_is_match: m || resolved_is_match,
                with_break_is_match_at_end: mae,
                is_counting: false,
                counting_mask: 0,
                pre_seeds: pre_seed_list.into_boxed_slice(),
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
    fn compute_counter_reset(
        &self,
        analysis: &Tier2Analysis,
        successor_nfa_states: &[StateIdx],
        skip_mask: u64,
    ) -> u64 {
        let mut mask: u64 = 0;
        for ci in 0..analysis.body_ranges.len() {
            let interior = analysis.interior(ci);
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
            &cr.nfa_states,
            &cr.deferred_asserts,
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
            let s = &self.inner.states[id.idx()];
            (s.is_match, s.is_match_at_end)
        }
    }

    fn clear(&mut self, memory: &mut DfaMemory, num_nfa_states: usize, stride: usize) {
        self.inner.clear();
        self.stride = stride;
        memory.clear(num_nfa_states);
        self.transitions.clear();
        self.start_seeds = Box::new([]);
        self.counters.clear();
    }

    #[inline]
    pub(crate) fn prepare(
        &mut self,
        memory: &mut DfaMemory,
        regex: &Regex,
        analysis: &Tier2Analysis,
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
            true,
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
        self.start_seeds = cr.seed_instances.iter().map(|&(c, _s)| (c, 0u32)).collect();

        // Initialize counter state (phases + metadata).
        self.counters.populate(regex);
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

/// Consume `byte` at NFA state `idx`.  Returns `(out, out_exit)` where
/// `out_exit` is `StateIdx::NONE` when there is no exit branch.
fn consume_byte(idx: StateIdx, byte: u8, regex: &Regex) -> Option<(StateIdx, StateIdx)> {
    match regex.states[idx] {
        State::Byte {
            byte: b,
            out,
            out_exit,
        } if byte == b => Some((out, out_exit)),
        State::ByteCI {
            byte: b,
            out,
            out_exit,
        } if byte_match_ci(byte, b) => Some((out, out_exit)),
        State::ByteClass {
            class,
            out,
            out_exit,
        } if regex.classes[class][byte] => Some((out, out_exit)),
        State::ByteTable { table } => {
            let t = regex.byte_tables[table][byte];
            if t != StateIdx::NONE {
                Some((t, StateIdx::NONE))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Walk epsilon transitions from `start` and collect all reachable
/// consuming states.  Allocates its own scratch buffers.
fn consuming_states_from(start: StateIdx, states: &[State]) -> Vec<StateIdx> {
    let mut result = Vec::new();
    let mut stack = vec![start];
    let mut visited = vec![false; states.len()];
    consuming_states_walk(&mut stack, &mut visited, states, &mut result);
    result
}

/// Walk epsilon transitions from `start` and append reachable
/// consuming states to `result`, using `DfaMemory` scratch buffers
/// for the visited array and work stack.
///
/// Does NOT touch `memory.closure_result` — the caller may still
/// need it for the epsilon closure result built earlier.
/// Shared walk logic for `consuming_states_from`.
fn consuming_states_walk(
    stack: &mut Vec<StateIdx>,
    visited: &mut [bool],
    states: &[State],
    result: &mut Vec<StateIdx>,
) {
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
}

/// Walk from `start` through epsilon states on the CInc break path
/// (CInc.out1) and determine how `Match` is reachable:
///
/// Returns `(direct, at_end)`:
/// - `direct`: `Match` is reachable without going through `$` / end-line
///   assertions — this fires mid-stream.
/// - `at_end`: `Match` is reachable only through `$` / `(?m:$)` /
///   `(?Rm:$)` — this fires only at end-of-input.
///
/// Both can be true if there are multiple paths.
/// Walk from `start` through epsilon states on the CInc break path
/// (CInc.out1) and determine how `Match` is reachable:
///
/// Returns `(direct, at_end)`:
/// - `direct`: Match is reachable without passing through an end-like
///   assertion (`$`, `\Z`, `\z`).
/// - `at_end`: Match is reachable through an end-like assertion.
fn break_path_match_kind(start: StateIdx, states: &[State]) -> (bool, bool) {
    let mut direct = false;
    let mut at_end = false;
    let mut stack: Vec<(StateIdx, bool)> = vec![(start, false)];
    let mut visited = vec![[false; 2]; states.len()];
    while let Some((idx, through_end)) = stack.pop() {
        let i = idx.idx();
        let te = through_end as usize;
        if visited[i][te] {
            continue;
        }
        visited[i][te] = true;
        match states[idx] {
            State::Match => {
                if through_end {
                    at_end = true;
                } else {
                    direct = true;
                }
            }
            State::Split { out, out1 } => {
                stack.push((out, through_end));
                stack.push((out1, through_end));
            }
            State::Assert { kind, out } => {
                let is_end_like = matches!(
                    kind,
                    AssertKind::End | AssertKind::EndLF | AssertKind::EndCRLF
                );
                stack.push((out, through_end || is_end_like));
            }
            State::CounterInstance { out, .. } => {
                stack.push((out, through_end));
            }
            _ => {}
        }
    }
    (direct, at_end)
}

// ---------------------------------------------------------------------------
// Tier 2 Matcher
// ---------------------------------------------------------------------------

/// Tier 2 DFA matcher with differential counters.
pub struct Tier2DfaMatcher<'a> {
    cache: &'a mut Tier2DfaCache,
    memory: &'a mut DfaMemory,
    regex: &'a Regex,
    analysis: &'a Tier2Analysis,
    current: DfaStateId,
    ever_matched: bool,
    match_at_end: bool,
    has_live_counters: bool,
    prefilter: Prefilter,
}

impl<'a> Tier2DfaMatcher<'a> {
    #[inline]
    pub(crate) fn new(
        cache: &'a mut Tier2DfaCache,
        memory: &'a mut DfaMemory,
        regex: &'a Regex,
        analysis: &'a Tier2Analysis,
    ) -> Self {
        cache.counters.reset();
        cache.counters.ensure_phases();

        // Seed initial counter instances (only for patterns where the
        // start state directly reaches a CounterInstance via epsilons).
        // For most patterns (e.g. AWS keys), start_seeds is empty and
        // this block is skipped entirely — no counter work at all.
        let mut has_live = false;
        if !cache.start_seeds.is_empty() {
            // Iterate start_seeds by index to avoid borrow conflict with
            // seed_counter (which borrows cache mutably).
            let num_start_seeds = cache.start_seeds.len();
            for si in 0..num_start_seeds {
                let (counter, _initial_value) = cache.start_seeds[si];
                let c_idx = counter.idx();
                let m = &cache.counters.meta[c_idx];
                let nph = m.num_phases;
                let target_phase = (nph - 1) % nph;
                let phase_idx = m.phase_start + target_phase;
                cache.counters.phases[phase_idx].alloc_new();
            }
            has_live = true;
        }

        Tier2DfaMatcher {
            current: cache.inner.start_id,
            ever_matched: cache.inner.start_is_match,
            match_at_end: cache.inner.start_is_match_at_end,
            has_live_counters: has_live,
            cache,
            memory,
            regex,
            analysis,
            prefilter: regex.prefilter,
        }
    }

    /// Slot lookup using byte-class compression (stride < 256).
    #[inline(always)]
    fn ensure_transition(&mut self, byte: u8) -> usize {
        let class = self.regex.byte_classes[byte as usize] as usize;
        let slot = self.current.idx() * self.cache.stride + class;
        if self.cache.transitions[slot].no_break == DfaStateId::UNPOPULATED {
            let trans =
                self.cache
                    .populate(self.memory, self.current, byte, self.regex, self.analysis);
            self.cache.transitions[slot] = trans;
        }
        slot
    }

    /// Slot lookup using identity mapping (stride = 256, no byte-class indirection).
    #[inline(always)]
    fn ensure_transition_direct(&mut self, byte: u8) -> usize {
        let slot = self.current.idx() * 256 + byte as usize;
        if self.cache.transitions[slot].no_break == DfaStateId::UNPOPULATED {
            let trans =
                self.cache
                    .populate(self.memory, self.current, byte, self.regex, self.analysis);
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
        if !t.is_counting && t.pre_seeds.is_empty() && t.seeds.is_empty() && !self.has_live_counters
        {
            self.current = t.no_break;
            self.match_at_end = t.no_break_is_match_at_end;
            if t.no_break_is_match {
                self.ever_matched = true;
            }
            return;
        }

        self.match_at_end = false;
        let mut any_can_break = false;

        // Apply pre_seeds BEFORE counter_increment.
        // These are resolved seeds (initial_value > 0) from Phase 1
        // deferred-assertion resolution on L=1 counter bodies.  The seed
        // must be visible to the increment so the counter can break on
        // this very transition.
        for (counter, initial_value) in &t.pre_seeds {
            let c_idx = counter.idx();
            if c_idx < self.cache.counters.meta.len() {
                self.cache.counters.seed(c_idx, *initial_value);
            }
        }

        if t.is_counting {
            // Process ALL counters that fire CInc on this transition.
            let mut mask = t.counting_mask;
            while mask != 0 {
                let c_idx = mask.trailing_zeros() as usize;
                mask &= mask - 1; // clear lowest set bit
                if c_idx < self.cache.counters.meta.len() && self.cache.counters.increment(c_idx) {
                    any_can_break = true;
                }
            }
        }

        // Select DFA successor and record match flags.
        if any_can_break {
            self.current = t.with_break;
            if t.with_break_is_match {
                self.ever_matched = true;
            }
            if t.with_break_is_match_at_end {
                self.match_at_end = true;
            }
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
        let mut mask = t.counter_reset;
        while mask != 0 {
            let c_idx = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            if c_idx < self.cache.counters.meta.len() {
                self.cache.counters.reset_counter(c_idx);
            }
        }

        // Advance active_phase for EVERY counter on EVERY byte.
        // This keeps the phase clock synchronized with byte position.
        self.cache.counters.advance_all_phases();

        // Seed new counter instances.
        // Access seeds by index: each index yields a Copy tuple, so the
        // temporary borrow on self.cache.transitions ends before we
        // call seed_counter.
        for (counter, initial_value) in &t.seeds {
            let c_idx = counter.idx();
            if c_idx < self.cache.counters.meta.len() {
                self.cache.counters.seed(c_idx, *initial_value);
            }
        }

        // Update has_live_counters flag.
        self.has_live_counters = self.cache.counters.has_live();
    }

    fn step_from_dead(&mut self, byte: u8) {
        if self.has_live_counters {
            self.cache.counters.reset();
            self.has_live_counters = false;
        }
        self.match_at_end = false;

        let trans = self.cache.populate(
            self.memory,
            DfaStateId::DEAD,
            byte,
            self.regex,
            self.analysis,
        );
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
        self.cache.counters.ensure_phases();

        // Advance active_phase for every counter (every byte ticks the clock).
        self.cache.counters.advance_all_phases();

        // Seed with phase-aware placement.
        // `trans` is an owned local — no borrow conflict with cache.
        for &(counter, initial_value) in trans.seeds.iter() {
            let c_idx = counter.idx();
            if c_idx < self.cache.counters.meta.len() {
                self.cache.counters.seed(c_idx, initial_value);
            }
        }
        self.has_live_counters = self.cache.counters.has_live();
    }

    #[inline(always)]
    pub fn chunk(&mut self, input: &[u8]) {
        if self.ever_matched {
            return;
        }

        let start_id = self.cache.inner.start_id;

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
            if self.cache.stride == 256 {
                self.step_direct(input[i]);
            } else {
                self.step(input[i]);
            }
            i += 1;
        }
    }

    #[inline]
    pub fn finish(mut self) -> bool {
        if self.ever_matched {
            return true;
        }
        if self.match_at_end {
            return true;
        }
        if self.current != DfaStateId::DEAD {
            let state = &self.cache.inner.states[self.current.idx()];
            if state.is_match_at_end {
                return true;
            }
            // Standard deferred resolution: handles assertions outside
            // counter bodies (deferred assert -> Match path).
            let mut scratch = super::ReachScratch::new();
            if state.resolve_deferred_at_end(self.regex, &mut scratch) {
                return true;
            }
            // Tier 2 specific: handle deferred assertions inside counter
            // bodies.  When a deferred assertion resolves at end-of-input
            // and reaches CInc, we need to check whether incrementing the
            // counter allows a break to Match.
            // Extract what we need before borrowing self mutably.
            let deferred = state.deferred_asserts.clone();
            let prev = state.prev_byte_representative();
            if self.resolve_deferred_cinc_at_end(&deferred, prev) {
                return true;
            }
        }
        false
    }

    /// Check if resolving deferred assertions at end-of-input reaches a
    /// CInc whose counter, after one more increment, allows a break to
    /// Match.  This handles patterns like `(\w\b){1,3}` where the last
    /// `\b` is only resolved at end-of-input.
    fn resolve_deferred_cinc_at_end(&mut self, deferred: &[StateIdx], prev: Option<u8>) -> bool {
        if deferred.is_empty() {
            return false;
        }
        let states = &self.regex.states;

        for v in self.memory.closure_visited.iter_mut() {
            *v = false;
        }
        self.memory.closure_stack.clear();
        for &assert_idx in deferred {
            if let State::Assert { kind, out } = states[assert_idx]
                && kind.eval(false, true, prev, None) == AssertEval::Pass
            {
                self.memory.closure_stack.push(out);
            }
        }

        while let Some(idx) = self.memory.closure_stack.pop() {
            let i = idx.idx();
            if self.memory.closure_visited[i] {
                continue;
            }
            self.memory.closure_visited[i] = true;
            match states[idx] {
                State::CounterIncrement { counter, min, .. } => {
                    let ci = counter.idx();
                    if !self.regex.counter_break_can_match[ci] {
                        continue;
                    }
                    if ci < self.cache.counters.meta.len() {
                        let m = &self.cache.counters.meta[ci];
                        for ph in 0..m.num_phases {
                            let phase = &self.cache.counters.phases[m.phase_start + ph];
                            if !phase.is_empty() && phase.oldest + 1 >= min as u32 {
                                return true;
                            }
                        }
                    }
                }
                State::Split { out, out1 } => {
                    self.memory.closure_stack.push(out1);
                    self.memory.closure_stack.push(out);
                }
                State::Assert { kind, out } => {
                    if kind.eval(false, true, prev, None) == AssertEval::Pass {
                        self.memory.closure_stack.push(out);
                    }
                }
                State::CounterInstance { out, .. } => self.memory.closure_stack.push(out),
                _ => {}
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
        let mut s = f.debug_struct("Tier2DfaMatcher");
        s.field("current", &self.current);
        if self.current != DfaStateId::DEAD {
            let state = &self.cache.inner.states[self.current.idx()];
            s.field("nfa_states", &state.nfa_states);
            s.field("is_match", &state.is_match);
            s.field("is_match_at_end", &state.is_match_at_end);
        }
        s.field("ever_matched", &self.ever_matched)
            .field("match_at_end", &self.match_at_end)
            .field("has_live_counters", &self.has_live_counters);
        s.finish()
    }
}

impl fmt::Display for Tier2DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.current == DfaStateId::DEAD {
            write!(f, "DFA[T2] state=DEAD matched={}", self.ever_matched)?;
            if self.has_live_counters {
                write!(f, " live_counters")?;
            }
            return Ok(());
        }
        let state = &self.cache.inner.states[self.current.idx()];
        write!(
            f,
            "DFA[T2] state={} nfa={{{}}} matched={}",
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
        // Counter summary: show per-counter active phase info.
        if self.has_live_counters && !self.cache.counters.meta.is_empty() {
            write!(f, " counters=[")?;
            for (ci, meta) in self.cache.counters.meta.iter().enumerate() {
                if ci > 0 {
                    write!(f, ", ")?;
                }
                // Find the active phase's DiffCounter.
                let phase_idx = meta.phase_start + meta.active_phase;
                if phase_idx < self.cache.counters.phases.len() {
                    let dc = &self.cache.counters.phases[phase_idx];
                    if dc.count > 0 {
                        write!(
                            f,
                            "c{ci}: {} inst oldest={} youngest={}",
                            dc.count,
                            dc.oldest,
                            dc.youngest()
                        )?;
                    } else {
                        write!(f, "c{ci}: idle")?;
                    }
                } else {
                    write!(f, "c{ci}: idle")?;
                }
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MatcherMemory, Regex, RegexConfig};

    /// Helper: build a tier-2-eligible regex.  Disables unrolling so that
    /// even small bounded repeats use counters (tier 2), and raises the
    /// estimated-states limit for large patterns.
    fn build(pattern: &str) -> Regex {
        Regex::with_config(
            pattern,
            RegexConfig {
                max_unroll_states: 0,
                max_estimated_states: usize::MAX,
                ..Default::default()
            },
        )
        .unwrap()
    }

    /// Helper: run a single tier-2 match and return the result.
    fn match_tier2(mem: &mut MatcherMemory, re: &Regex, input: &[u8]) -> bool {
        let mut m = mem.matcher_for_tier(re, 2).unwrap();
        m.chunk(input);
        m.finish()
    }

    // ── Ring buffer reuse across matches ──

    #[test]
    fn test_prepare_ring_reuse_same_regex() {
        let re = build(".{1,100}a");
        let mut mem = MatcherMemory::default();

        // First match — allocates phases and ring buffers.
        assert!(match_tier2(&mut mem, &re, b"xxxxa"));

        // Access the cache to check ring capacity was retained.
        let cache = mem.tier2_cache.as_ref().unwrap();
        assert!(
            !cache.counters.phases.is_empty(),
            "phases should be retained after first match"
        );
        let cap_after_first = cache.counters.phases[0].ring.len();
        assert!(cap_after_first > 0, "ring should be allocated");

        // Second match — reuses ring buffers, no reallocation.
        assert!(match_tier2(&mut mem, &re, b"ya"));

        let cache = mem.tier2_cache.as_ref().unwrap();
        assert_eq!(
            cache.counters.phases[0].ring.len(),
            cap_after_first,
            "ring capacity should be unchanged across matches"
        );
    }

    #[test]
    fn test_prepare_ring_cleared_between_matches() {
        let re = build(".{1,100}a");
        let mut mem = MatcherMemory::default();

        // Match with many instances seeded.
        assert!(match_tier2(&mut mem, &re, b"xxxxxxxxxa"));

        // Counter state should be clean for next match — a short input
        // that only seeds 1 instance should still work.
        assert!(match_tier2(&mut mem, &re, b"xa"));

        // No match.
        assert!(!match_tier2(&mut mem, &re, b"xxx"));
    }

    #[test]
    fn test_prepare_different_regex_reallocates() {
        let re1 = build(".{1,100}a");
        let re2 = build("[a-z]{1,50}x");
        let mut mem = MatcherMemory::default();

        // Warm up with re1.
        assert!(match_tier2(&mut mem, &re1, b"xxxxa"));
        let phases_after_re1 = mem.tier2_cache.as_ref().unwrap().counters.phases.len();
        assert!(phases_after_re1 > 0);

        // Switch to re2 — should clear and reallocate.
        assert!(match_tier2(&mut mem, &re2, b"abcx"));

        // Phases may have a different count depending on re2's counters.
        let phases_after_re2 = mem.tier2_cache.as_ref().unwrap().counters.phases.len();
        assert!(phases_after_re2 > 0);

        // Switch back to re1 — clear again.
        assert!(match_tier2(&mut mem, &re1, b"za"));
    }

    // ── Ring buffer growth for unbounded counters ──

    #[test]
    fn test_prepare_ring_grows_for_large_input() {
        let re = build(".{1,100}a");
        let mut mem = MatcherMemory::default();

        // Short input.
        assert!(match_tier2(&mut mem, &re, b"xxxa"));

        // Long input — ring grows to accommodate many live instances.
        let mut long_input = vec![b'x'; 8000];
        long_input.push(b'a');
        assert!(match_tier2(&mut mem, &re, &long_input));
    }

    // ── Ring buffer growth for unbounded counters ──

    #[test]
    fn test_prepare_ring_grows_for_unbounded() {
        // a{2,}b — unbounded counter, ring starts at the initial cap
        // and must grow for long inputs.
        let re = build("a{2,}b");
        let mut mem = MatcherMemory::default();

        assert!(match_tier2(&mut mem, &re, b"aab"));
        assert!(!match_tier2(&mut mem, &re, b"ab"));

        let mut long_input = vec![b'a'; 8000];
        long_input.push(b'b');
        assert!(match_tier2(&mut mem, &re, &long_input));
    }

    // ── Correctness across reused matches ──

    #[test]
    fn test_prepare_no_stale_state() {
        let re = build(".{3,5}x");
        let mut mem = MatcherMemory::default();

        // Match: 4 dots then x.
        assert!(match_tier2(&mut mem, &re, b"aaaax"));

        // No match: only 2 dots (below min=3).
        assert!(!match_tier2(&mut mem, &re, b"aax"));

        // Match again: exactly 3.
        assert!(match_tier2(&mut mem, &re, b"bbbx"));

        // No match: x without enough prefix.
        assert!(!match_tier2(&mut mem, &re, b"x"));
    }

    #[test]
    fn test_prepare_no_stale_state_small_bound() {
        // a{2,5}b with unroll disabled → tier 2 counter.
        let re = build("a{2,5}b");
        let mut mem = MatcherMemory::default();

        assert!(match_tier2(&mut mem, &re, b"aab"));
        assert!(match_tier2(&mut mem, &re, b"aaaaab"));
        assert!(!match_tier2(&mut mem, &re, b"ab"));
        assert!(!match_tier2(&mut mem, &re, b"aaa"));
        // Reuse after no-match.
        assert!(match_tier2(&mut mem, &re, b"aaab"));
    }
}
