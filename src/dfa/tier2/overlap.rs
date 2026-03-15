//! Binary-exactness proof for Tier 2 overlap analysis.
//!
//! When counter body byte sets overlap, the strict disjointness check
//! rejects the pattern from Tier 2.  This module implements a more precise
//! structural proof: for every reachable DFA transition, it verifies that
//! Tier 2's binary `no_break` / `with_break` compression remains exact
//! across all possible break subsets.
//!
//! # Safety rule
//!
//! Tier 2 stores exactly two cached outcomes per transition (`no_break` and
//! `with_break`) plus shared per-transition fields (`counter_reset`, `seeds`).
//! The runtime selects between them using only `any_can_break`.
//!
//! For this to be exact, every reachable transition must satisfy:
//!
//! 1. **Branch-specific** fields (successor closure, is_match, is_match_at_end)
//!    collapse into exactly two classes: one for S=∅ and one for all S≠∅.
//! 2. **Shared** fields (counter_reset, seeds) are identical for ALL subsets S ⊆ M.
//!
//! If any transition violates this, the pattern is rejected from Tier 2.

use hashbrown::HashSet;

use crate::{ByteMap, CounterIdx, State, StateIdx};

/// Maximum DFA states the probe will explore before giving up.
const PROBE_STATE_LIMIT: usize = 10_000;

/// Maximum counters in a single CInc set M for subset enumeration.
/// |M| = k means 2^k subsets.  k>8 → 256+ subsets per transition.
const MAX_SUBSET_WIDTH: u8 = 8;

// ---------------------------------------------------------------------------
// Proof result types
// ---------------------------------------------------------------------------

/// Outcome of the Tier 2 binary-exactness proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tier2OverlapProof {
    /// Body byte sets are pairwise disjoint — no proof needed.
    DisjointFastPath,
    /// Proof succeeded: all reachable transitions are binary-exact.
    ProvenBinaryExact,
    /// Proof failed: at least one transition is not binary-exact.
    RejectedNonBinaryExact,
    /// Proof aborted: state-space or subset-space exceeded limits.
    RejectedStateLimit,
}

/// Statistics from the overlap probe.
#[derive(Debug)]
pub(crate) struct Tier2OverlapStats {
    pub(crate) proof: Tier2OverlapProof,
    pub(crate) max_reachable_counting_width: u8,
    pub(crate) max_subset_width_checked: u8,
    pub(crate) reachable_probe_states: usize,
}

// ---------------------------------------------------------------------------
// Transition summary types
// ---------------------------------------------------------------------------

/// Branch-specific summary (NFA successor set + match flags).
#[derive(Clone, PartialEq, Eq, Hash)]
struct BranchSummary {
    nfa_states: Vec<StateIdx>,
    deferred_asserts: Vec<StateIdx>,
    is_match: bool,
    is_match_at_end: bool,
}

/// Shared per-transition summary (must be identical across ALL subsets).
#[derive(Clone, PartialEq, Eq, Hash)]
struct SharedSummary {
    counter_reset: u64,
    seeds: Vec<(CounterIdx, u32)>,
}

// ---------------------------------------------------------------------------
// Epsilon closure result
// ---------------------------------------------------------------------------

struct SubsetClosure {
    nfa_states: Vec<StateIdx>,
    deferred_asserts: Vec<StateIdx>,
    is_match: bool,
    is_match_at_end: bool,
    seed_instances: Vec<(CounterIdx, StateIdx)>,
}

// ---------------------------------------------------------------------------
// Main probe
// ---------------------------------------------------------------------------

/// Run the binary-exactness proof.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tier2_overlap_probe(
    states: &[State],
    classes: &[crate::ByteClass],
    byte_tables: &[ByteMap],
    byte_classes: &[u8; 256],
    num_byte_classes: usize,
    start: StateIdx,
    analysis: &super::Tier2Analysis,
    num_counters: usize,
) -> Tier2OverlapStats {
    let mut visited: HashSet<Vec<StateIdx>> = HashSet::new();
    let mut work: Vec<Vec<StateIdx>> = Vec::new();
    let mut max_width: u8 = 0;
    let mut max_subset_checked: u8 = 0;
    let mut probe_states: usize = 0;

    // Start closure: consuming states from start, no break paths.
    let start_set = epsilon_consuming_closure(&[start], states, false);
    if !start_set.is_empty() {
        visited.insert(start_set.clone());
        work.push(start_set);
    }

    let class_reps = byte_class_representatives(byte_classes, num_byte_classes);

    while let Some(nfa_set) = work.pop() {
        probe_states += 1;
        if probe_states > PROBE_STATE_LIMIT {
            return stats(
                Tier2OverlapProof::RejectedStateLimit,
                max_width,
                max_subset_checked,
                probe_states,
            );
        }

        for &byte in &class_reps {
            // Step 1: compute targets (post-consumption states).
            let mut targets = consume_all(&nfa_set, byte, states, classes, byte_tables);
            // Add start re-entry for unanchored patterns.
            add_start_targets(&mut targets, start, byte, states, classes, byte_tables);
            if targets.is_empty() {
                continue;
            }

            // Step 2: find M — set of counters whose CInc is crossed.
            let cinc_mask = find_cinc_counters(&targets, states);
            let k = cinc_mask.count_ones() as u8;
            if k > max_width {
                max_width = k;
            }

            // k ≤ 1: trivially binary-exact.
            if k <= 1 {
                enqueue_successors(&targets, start, states, &mut visited, &mut work);
                continue;
            }

            if k > MAX_SUBSET_WIDTH {
                return stats(
                    Tier2OverlapProof::RejectedStateLimit,
                    max_width,
                    max_subset_checked,
                    probe_states,
                );
            }
            if k > max_subset_checked {
                max_subset_checked = k;
            }

            // Step 3: enumerate all 2^k subsets of M.
            let m_counters: Vec<usize> = (0..64).filter(|&i| (cinc_mask >> i) & 1 != 0).collect();
            let num_subsets = 1u64 << k;

            let mut empty_branch: Option<BranchSummary> = None;
            let mut nonempty_branch: Option<BranchSummary> = None;
            let mut first_shared: Option<SharedSummary> = None;
            let mut exact = true;

            for s_bits in 0..num_subsets {
                let break_subset = bits_to_mask(s_bits, &m_counters);
                let is_empty = break_subset == 0;

                let closure =
                    epsilon_closure_with_break_subset(&targets, states, start, break_subset);

                let branch = BranchSummary {
                    nfa_states: closure.nfa_states.clone(),
                    deferred_asserts: closure.deferred_asserts,
                    is_match: closure.is_match,
                    is_match_at_end: closure.is_match_at_end,
                };

                let cr =
                    compute_counter_reset(analysis, &closure.nfa_states, cinc_mask, num_counters);
                let seeds = compute_seeds(&closure.seed_instances);
                let shared = SharedSummary {
                    counter_reset: cr,
                    seeds,
                };

                // Check branch-specific: S=∅ vs S≠∅.
                if is_empty {
                    empty_branch = Some(branch);
                } else {
                    match &nonempty_branch {
                        None => nonempty_branch = Some(branch),
                        Some(prev) if *prev != branch => {
                            exact = false;
                            break;
                        }
                        _ => {}
                    }
                }

                // Check shared: must be identical for ALL subsets.
                match &first_shared {
                    None => first_shared = Some(shared),
                    Some(prev) if *prev != shared => {
                        exact = false;
                        break;
                    }
                    _ => {}
                }
            }

            if !exact {
                return stats(
                    Tier2OverlapProof::RejectedNonBinaryExact,
                    max_width,
                    max_subset_checked,
                    probe_states,
                );
            }

            // Enqueue both successor sets.
            if let Some(ref b) = empty_branch {
                enqueue_nfa_set(&b.nfa_states, &mut visited, &mut work);
            }
            if let Some(ref b) = nonempty_branch {
                enqueue_nfa_set(&b.nfa_states, &mut visited, &mut work);
            }
        }
    }

    stats(
        Tier2OverlapProof::ProvenBinaryExact,
        max_width,
        max_subset_checked,
        probe_states,
    )
}

fn stats(proof: Tier2OverlapProof, w: u8, s: u8, n: usize) -> Tier2OverlapStats {
    Tier2OverlapStats {
        proof,
        max_reachable_counting_width: w,
        max_subset_width_checked: s,
        reachable_probe_states: n,
    }
}

// ---------------------------------------------------------------------------
// Subset-controlled epsilon closure
// ---------------------------------------------------------------------------

/// Epsilon closure with a specific break subset.
///
/// - `CounterIncrement.out` (continue) is always followed.
/// - `CounterIncrement.out1` (break) is followed only when that counter
///   is in `break_subset`.
/// - Also includes the start state's consuming successors (re-entry).
fn epsilon_closure_with_break_subset(
    targets: &[StateIdx],
    states: &[State],
    start: StateIdx,
    break_subset: u64,
) -> SubsetClosure {
    let n = states.len();
    let mut visited = vec![false; n];
    let mut stack: Vec<StateIdx> = targets.to_vec();
    // Also add start re-entry.
    stack.push(start);

    let mut nfa_states: Vec<StateIdx> = Vec::new();
    let mut deferred_asserts: Vec<StateIdx> = Vec::new();
    let mut is_match = false;
    let mut is_match_at_end = false;
    let mut seed_instances: Vec<(CounterIdx, StateIdx)> = Vec::new();

    while let Some(idx) = stack.pop() {
        let i = idx.idx();
        if i >= n || visited[i] {
            continue;
        }
        visited[i] = true;
        match states[i] {
            State::Split { out, out1 } => {
                stack.push(out);
                stack.push(out1);
            }
            State::Assert { kind, out } => {
                if kind == crate::AssertKind::End {
                    is_match_at_end = true;
                } else {
                    deferred_asserts.push(idx);
                    stack.push(out);
                }
            }
            State::CounterInstance { counter, out } => {
                seed_instances.push((counter, out));
                stack.push(out);
            }
            State::CounterIncrement {
                counter, out, out1, ..
            } => {
                // Continue: always.
                stack.push(out);
                // Break: only if counter is in break_subset.
                if (break_subset >> counter.idx()) & 1 != 0 {
                    stack.push(out1);
                }
            }
            State::Byte { .. }
            | State::ByteCI { .. }
            | State::ByteClass { .. }
            | State::ByteTable { .. } => {
                nfa_states.push(idx);
            }
            State::Match => {
                is_match = true;
            }
        }
    }

    nfa_states.sort();
    nfa_states.dedup();
    deferred_asserts.sort();
    deferred_asserts.dedup();
    seed_instances.sort();
    seed_instances.dedup();

    SubsetClosure {
        nfa_states,
        deferred_asserts,
        is_match,
        is_match_at_end,
        seed_instances,
    }
}

// ---------------------------------------------------------------------------
// Simple epsilon closure (no subset control)
// ---------------------------------------------------------------------------

fn epsilon_consuming_closure(
    seeds: &[StateIdx],
    states: &[State],
    follow_break: bool,
) -> Vec<StateIdx> {
    let n = states.len();
    let mut visited = vec![false; n];
    let mut stack: Vec<StateIdx> = seeds.to_vec();
    let mut result: Vec<StateIdx> = Vec::new();

    while let Some(idx) = stack.pop() {
        let i = idx.idx();
        if i >= n || visited[i] {
            continue;
        }
        visited[i] = true;
        match states[i] {
            State::Split { out, out1 } => {
                stack.push(out);
                stack.push(out1);
            }
            State::Assert { out, .. } => stack.push(out),
            State::CounterInstance { out, .. } => stack.push(out),
            State::CounterIncrement { out, out1, .. } => {
                stack.push(out);
                if follow_break {
                    stack.push(out1);
                }
            }
            State::Byte { .. }
            | State::ByteCI { .. }
            | State::ByteClass { .. }
            | State::ByteTable { .. } => {
                result.push(idx);
            }
            State::Match => {}
        }
    }

    result.sort();
    result.dedup();
    result
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn consume_all(
    nfa_set: &[StateIdx],
    byte: u8,
    states: &[State],
    classes: &[crate::ByteClass],
    byte_tables: &[ByteMap],
) -> Vec<StateIdx> {
    let mut targets = Vec::new();
    for &s in nfa_set {
        if let Some(t) = consume_byte_at(s, byte, states, classes, byte_tables) {
            targets.push(t);
        }
    }
    targets
}

fn add_start_targets(
    targets: &mut Vec<StateIdx>,
    start: StateIdx,
    byte: u8,
    states: &[State],
    classes: &[crate::ByteClass],
    byte_tables: &[ByteMap],
) {
    let start_consuming = epsilon_consuming_closure(&[start], states, false);
    for &s in &start_consuming {
        if let Some(t) = consume_byte_at(s, byte, states, classes, byte_tables)
            && !targets.contains(&t)
        {
            targets.push(t);
        }
    }
}

fn consume_byte_at(
    idx: StateIdx,
    byte: u8,
    states: &[State],
    classes: &[crate::ByteClass],
    byte_tables: &[ByteMap],
) -> Option<StateIdx> {
    match states[idx.idx()] {
        State::Byte { byte: b, out } if byte == b => Some(out),
        State::ByteCI { byte: b, out } if crate::byte_match_ci(byte, b) => Some(out),
        State::ByteClass { class, out } if classes[class.idx()][byte] => Some(out),
        State::ByteTable { table } => {
            let t = byte_tables[table][byte];
            if t != StateIdx::NONE {
                Some(t)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn find_cinc_counters(targets: &[StateIdx], states: &[State]) -> u64 {
    let n = states.len();
    let mut mask: u64 = 0;
    let mut visited = vec![false; n];
    for &t in targets {
        visited.fill(false);
        let mut stack = vec![t];
        while let Some(idx) = stack.pop() {
            let i = idx.idx();
            if i >= n || visited[i] {
                continue;
            }
            visited[i] = true;
            match states[i] {
                State::CounterIncrement { counter, .. } => {
                    mask |= 1u64 << counter.idx();
                }
                State::Split { out, out1 } => {
                    stack.push(out);
                    stack.push(out1);
                }
                State::Assert { out, .. } => stack.push(out),
                State::CounterInstance { out, .. } => stack.push(out),
                _ => {}
            }
        }
    }
    mask
}

fn compute_counter_reset(
    analysis: &super::Tier2Analysis,
    successor_nfa_states: &[StateIdx],
    skip_mask: u64,
    num_counters: usize,
) -> u64 {
    let mut mask: u64 = 0;
    for ci in 0..num_counters {
        if (skip_mask >> ci) & 1 != 0 {
            continue;
        }
        let interior = analysis.interior(ci);
        if interior.is_empty() {
            mask |= 1u64 << ci;
            continue;
        }
        let has_interior = successor_nfa_states
            .iter()
            .any(|s| interior.binary_search(&s.0).is_ok());
        if !has_interior {
            mask |= 1u64 << ci;
        }
    }
    mask
}

fn compute_seeds(seed_instances: &[(CounterIdx, StateIdx)]) -> Vec<(CounterIdx, u32)> {
    let mut seeds: Vec<(CounterIdx, u32)> =
        seed_instances.iter().map(|&(c, _)| (c, 0u32)).collect();
    seeds.sort();
    seeds.dedup();
    seeds
}

fn bits_to_mask(s_bits: u64, m_counters: &[usize]) -> u64 {
    let mut mask: u64 = 0;
    for (bit_pos, &c_idx) in m_counters.iter().enumerate() {
        if (s_bits >> bit_pos) & 1 != 0 {
            mask |= 1u64 << c_idx;
        }
    }
    mask
}

fn enqueue_successors(
    targets: &[StateIdx],
    start: StateIdx,
    states: &[State],
    visited: &mut HashSet<Vec<StateIdx>>,
    work: &mut Vec<Vec<StateIdx>>,
) {
    // No-break successor.
    let nb = epsilon_closure_with_break_subset(targets, states, start, 0);
    enqueue_nfa_set(&nb.nfa_states, visited, work);
    // Broad successor (all breaks).
    let wb = epsilon_consuming_closure(
        &{
            let mut s = targets.to_vec();
            s.push(start);
            s
        },
        states,
        true,
    );
    enqueue_nfa_set(&wb, visited, work);
}

fn enqueue_nfa_set(
    nfa_states: &[StateIdx],
    visited: &mut HashSet<Vec<StateIdx>>,
    work: &mut Vec<Vec<StateIdx>>,
) {
    if !nfa_states.is_empty() {
        let key = nfa_states.to_vec();
        if !visited.contains(&key) {
            visited.insert(key.clone());
            work.push(key);
        }
    }
}

fn byte_class_representatives(byte_classes: &[u8; 256], num_byte_classes: usize) -> Vec<u8> {
    let mut seen = vec![false; num_byte_classes];
    let mut reps = Vec::new();
    for b in 0..=255u8 {
        let class = byte_classes[b as usize] as usize;
        if !seen[class] {
            seen[class] = true;
            reps.push(b);
        }
    }
    reps
}
