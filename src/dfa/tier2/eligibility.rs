//! Tier 2 eligibility analysis.
//!
//! Determines whether a non-nested counter pattern qualifies for Tier 2's
//! differential-counter DFA.  The analysis is purely compile-time and does
//! not affect the Tier 2 runtime.
//!
//! # Requirements for Tier 2 eligibility
//!
//! All of the following must hold (given `non_nested_eligible == true`):
//!
//! 1. Every counter body has a **fixed byte length** > 0.
//! 2. No counter with body length > 1 contains deferred assertions.
//! 3. If any counter body contains deferred assertions, there is exactly
//!    one counter (multi-counter + deferred is unsupported).
//! 4. The number of counters is ≤ [`MAX_TIER2_COUNTERS`](super::MAX_TIER2_COUNTERS) (64).
//! 5. Counter body byte sets are **pairwise disjoint** (so at most one
//!    counter fires `CInc` on any DFA transition).
//!
//! Requirement 5 will be relaxed in a later patch via a reachability probe
//! that proves `counting_mask.count_ones() <= 1` for all reachable
//! transitions even when raw byte sets overlap.

use crate::{AssertKind, ByteClass, ByteMap, CounterIdx, State, StateIdx};

// ---------------------------------------------------------------------------
// Eligibility result
// ---------------------------------------------------------------------------

/// Summary of Tier 2 eligibility analysis.
///
/// Returned by [`compute_tier2_eligibility()`] and consumed by
/// `RegexBuilder::build()` to decide `tier2_eligible`.
#[derive(Debug)]
pub(crate) struct Tier2Eligibility {
    /// Per-counter `(min, max, body_byte_length)`.
    /// `body_byte_length` is 0 if the body has variable length.
    pub(crate) counter_info: Box<[(usize, usize, usize)]>,
    /// True when the byte sets of different counter bodies are pairwise
    /// disjoint.
    pub(crate) disjoint_bytes: bool,
    /// True when every counter body has a fixed byte length > 0.
    pub(crate) all_fixed_length: bool,
    /// True when a counter with body_length > 1 contains deferred
    /// assertions in its body.
    pub(crate) has_deferred_in_long_body: bool,
    /// True when deferred assertions exist in any counter body AND
    /// there is more than one counter.
    pub(crate) has_deferred_in_multi_counter_body: bool,
    /// True when any two counters have identical body byte sets.
    /// This causes Tier 2's runtime to overcount (both counters advance
    /// simultaneously and `counter_reset` never fires).
    pub(crate) has_identical_body_bytes: bool,
}

impl Tier2Eligibility {
    /// Returns true when the pattern passes all Tier 2 requirements
    /// except the byte-overlap check.
    pub(crate) fn base_eligible(&self) -> bool {
        self.all_fixed_length
            && !self.has_deferred_in_long_body
            && !self.has_deferred_in_multi_counter_body
            && !self.counter_info.is_empty()
            && self.counter_info.len() <= super::MAX_TIER2_COUNTERS
    }

    /// Returns true under the strict (disjoint bytes) rule.
    pub(crate) fn is_eligible(&self) -> bool {
        self.base_eligible() && self.disjoint_bytes
    }

    /// Returns true under the relaxed rule: either disjoint bytes
    /// (fast path) or a proven binary-exact overlap proof.
    pub(crate) fn is_eligible_with_proof(&self, proof: super::overlap::Tier2OverlapProof) -> bool {
        self.base_eligible()
            && !self.has_identical_body_bytes
            && matches!(
                proof,
                super::overlap::Tier2OverlapProof::DisjointFastPath
                    | super::overlap::Tier2OverlapProof::ProvenBinaryExact
            )
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compute Tier 2 eligibility for a pattern with non-nested counters.
///
/// `has_deferred_in_counter_body` must be pre-computed by the caller
/// (it is shared with other tier checks).
pub(crate) fn compute_tier2_eligibility(
    states: &[State],
    classes: &indexmap::set::IndexSet<ByteClass>,
    byte_tables: &[ByteMap],
    counters_len: usize,
    has_deferred_in_counter_body: bool,
) -> Tier2Eligibility {
    let counter_info = build_counter_info(states, byte_tables, counters_len);
    let disjoint_bytes = counter_bodies_have_disjoint_bytes(states, classes, byte_tables);

    let all_fixed_length =
        !counter_info.is_empty() && counter_info.iter().all(|&(_, _, body_len)| body_len > 0);

    let has_deferred_in_long_body = has_deferred_in_counter_body
        && states.iter().any(|s| {
            if let State::CounterInstance { counter, out } = s {
                let ci = counter.idx();
                let body_len = counter_info.get(ci).map_or(0, |info| info.2);
                body_len > 1 && body_has_deferred(*out, *counter, states, byte_tables)
            } else {
                false
            }
        });

    let has_deferred_in_multi_counter_body = has_deferred_in_counter_body && counter_info.len() > 1;

    let has_identical_body_bytes =
        counter_bodies_have_identical_bytes(states, classes, byte_tables);

    Tier2Eligibility {
        counter_info,
        disjoint_bytes,
        all_fixed_length,
        has_deferred_in_long_body,
        has_deferred_in_multi_counter_body,
        has_identical_body_bytes,
    }
}

// ---------------------------------------------------------------------------
// Counter info
// ---------------------------------------------------------------------------

/// Build per-counter `(min, max, body_byte_length)`.
fn build_counter_info(
    states: &[State],
    byte_tables: &[ByteMap],
    counters_len: usize,
) -> Box<[(usize, usize, usize)]> {
    let mut info = vec![(0usize, 0usize, 0usize); counters_len];
    // Collect min/max from CInc states.
    for s in states {
        if let State::CounterIncrement {
            counter, min, max, ..
        } = s
        {
            info[counter.idx()] = (*min, *max, 0);
        }
    }
    // Compute body lengths from CI states.
    for s in states {
        if let State::CounterInstance { counter, out } = s {
            let ci = counter.idx();
            let body_len = counter_body_length(*out, *counter, states, byte_tables).unwrap_or(0);
            info[ci].2 = body_len;
        }
    }
    info.into_boxed_slice()
}

// ---------------------------------------------------------------------------
// Byte-disjointness check
// ---------------------------------------------------------------------------

/// Check that the byte sets matched by different counter bodies are
/// pairwise disjoint.  When this holds, at most one counter fires `CInc`
/// on any DFA transition, so the binary `with_break` / `no_break` split
/// is correct for multiple counters.
pub(crate) fn counter_bodies_have_disjoint_bytes(
    states: &[State],
    classes: &indexmap::set::IndexSet<ByteClass>,
    byte_tables: &[ByteMap],
) -> bool {
    let counter_bytes = collect_counter_byte_sets(states, classes, byte_tables);
    for i in 0..counter_bytes.len() {
        for j in (i + 1)..counter_bytes.len() {
            if counter_bytes[i].0 == counter_bytes[j].0 {
                debug_assert!(false, "duplicate counter pair in byte overlap check");
                continue;
            }
            for b in 0..256 {
                if counter_bytes[i].1[b] && counter_bytes[j].1[b] {
                    return false;
                }
            }
        }
    }
    true
}

/// Check if any two counters have identical body byte sets.
///
/// When two counters have identical body bytes, both advance on every byte,
/// and `counter_reset` never fires for either (both have body-interior
/// progress).  This causes Tier 2's differential counter model to overcount.
pub(crate) fn counter_bodies_have_identical_bytes(
    states: &[State],
    classes: &indexmap::set::IndexSet<ByteClass>,
    byte_tables: &[ByteMap],
) -> bool {
    let counter_bytes = collect_counter_byte_sets(states, classes, byte_tables);
    for i in 0..counter_bytes.len() {
        for j in (i + 1)..counter_bytes.len() {
            if counter_bytes[i].0 == counter_bytes[j].0 {
                continue;
            }
            if counter_bytes[i].1 == counter_bytes[j].1 {
                return true;
            }
        }
    }
    false
}

/// Collect per-counter byte sets (shared by disjoint and identical checks).
fn collect_counter_byte_sets(
    states: &[State],
    classes: &indexmap::set::IndexSet<ByteClass>,
    byte_tables: &[ByteMap],
) -> Vec<(CounterIdx, [bool; 256])> {
    let mut counter_bytes: Vec<(CounterIdx, [bool; 256])> = Vec::new();
    for s in states {
        if let State::CounterInstance { counter, out } = s {
            let mut bytes = [false; 256];
            let mut stack = vec![*out];
            let mut visited = vec![false; states.len()];
            while let Some(idx) = stack.pop() {
                let i = idx.idx();
                if visited[i] {
                    continue;
                }
                visited[i] = true;
                match states[idx] {
                    State::CounterIncrement { counter: c, .. } if c == *counter => {}
                    State::Split { out, out1 } => {
                        stack.push(out1);
                        stack.push(out);
                    }
                    State::Assert { out, .. } | State::CounterInstance { out, .. } => {
                        stack.push(out);
                    }
                    State::Byte { byte, out, .. } => {
                        bytes[byte as usize] = true;
                        stack.push(out);
                    }
                    State::ByteCI { byte, out, .. } => {
                        bytes[byte as usize] = true;
                        bytes[(byte ^ 0x20) as usize] = true;
                        stack.push(out);
                    }
                    State::ByteClass { class, out, .. } => {
                        let table = &classes[class.idx()];
                        for b in 0..=255u8 {
                            if table[b] {
                                bytes[b as usize] = true;
                            }
                        }
                        stack.push(out);
                    }
                    State::ByteTable { table } => {
                        let map = &byte_tables[table.idx()];
                        for b in 0..=255u8 {
                            if map[b] != StateIdx::NONE {
                                bytes[b as usize] = true;
                                stack.push(map[b]);
                            }
                        }
                    }
                    _ => {}
                }
            }
            counter_bytes.push((*counter, bytes));
        }
    }
    counter_bytes
}

// ---------------------------------------------------------------------------
// Body-length computation
// ---------------------------------------------------------------------------

/// Compute the fixed byte-length of a counter body.
///
/// Returns `Some(len)` if all paths through the body consume exactly
/// `len` bytes, `None` if variable-length.
pub(crate) fn counter_body_length(
    ci_out: StateIdx,
    own_counter: CounterIdx,
    states: &[State],
    byte_tables: &[ByteMap],
) -> Option<usize> {
    let mut result: Option<usize> = None;
    let mut stack: Vec<(StateIdx, usize)> = vec![(ci_out, 0)];
    let mut visited: Vec<Option<usize>> = vec![None; states.len()];
    let mut in_stack = vec![false; states.len()];
    while let Some((idx, depth)) = stack.pop() {
        let i = idx.idx();
        in_stack[i] = false;
        match states[idx] {
            State::CounterIncrement { counter, .. } if counter == own_counter => match result {
                None => result = Some(depth),
                Some(prev) if prev != depth => return None,
                _ => {}
            },
            State::Split { out, out1 } => {
                for succ in [out, out1] {
                    let si = succ.idx();
                    if !in_stack[si] {
                        match visited[si] {
                            Some(d) if d == depth => {}
                            Some(_) => return None,
                            None => {
                                visited[si] = Some(depth);
                                in_stack[si] = true;
                                stack.push((succ, depth));
                            }
                        }
                    }
                }
            }
            State::Assert { out, .. } | State::CounterInstance { out, .. } => {
                let si = out.idx();
                if !in_stack[si] {
                    match visited[si] {
                        Some(d) if d == depth => {}
                        Some(_) => return None,
                        None => {
                            visited[si] = Some(depth);
                            in_stack[si] = true;
                            stack.push((out, depth));
                        }
                    }
                }
            }
            State::Byte { out, .. } | State::ByteCI { out, .. } | State::ByteClass { out, .. } => {
                let si = out.idx();
                let nd = depth + 1;
                if !in_stack[si] {
                    match visited[si] {
                        Some(d) if d == nd => {}
                        Some(_) => return None,
                        None => {
                            visited[si] = Some(nd);
                            in_stack[si] = true;
                            stack.push((out, nd));
                        }
                    }
                }
            }
            State::ByteTable { table } => {
                let nd = depth + 1;
                for &succ in &byte_tables[table.idx()].0 {
                    if succ != StateIdx::NONE {
                        let si = succ.idx();
                        if !in_stack[si] {
                            match visited[si] {
                                Some(d) if d == nd => {}
                                Some(_) => return None,
                                None => {
                                    visited[si] = Some(nd);
                                    in_stack[si] = true;
                                    stack.push((succ, nd));
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                debug_assert!(
                    !matches!(states[idx], State::Match),
                    "Match state reachable in counter body walk from state {}",
                    idx.idx()
                );
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Deferred-assertion-in-body check
// ---------------------------------------------------------------------------

/// Returns true if any deferred assertion kind is reachable within the
/// counter body from `start` to `CInc(own_counter)`.
pub(crate) fn body_has_deferred(
    start: StateIdx,
    own_counter: CounterIdx,
    states: &[State],
    byte_tables: &[ByteMap],
) -> bool {
    let mut stack = vec![start];
    let mut visited = vec![false; states.len()];
    while let Some(idx) = stack.pop() {
        let i = idx.idx();
        if visited[i] {
            continue;
        }
        visited[i] = true;
        match states[idx] {
            State::CounterIncrement { counter, .. } if counter == own_counter => {
                continue;
            }
            State::Assert { kind, out } => {
                if matches!(
                    kind,
                    AssertKind::EndLF
                        | AssertKind::EndCRLF
                        | AssertKind::StartCRLF
                        | AssertKind::WordAscii
                        | AssertKind::WordAsciiNegate
                        | AssertKind::WordStartAscii
                        | AssertKind::WordEndAscii
                ) {
                    return true;
                }
                stack.push(out);
            }
            State::Split { out, out1 } => {
                stack.push(out1);
                stack.push(out);
            }
            State::CounterInstance { out, .. } => stack.push(out),
            State::Byte { out, .. } | State::ByteCI { out, .. } | State::ByteClass { out, .. } => {
                stack.push(out)
            }
            State::ByteTable { table } => {
                for &succ in &byte_tables[table.idx()].0 {
                    if succ != StateIdx::NONE {
                        stack.push(succ);
                    }
                }
            }
            _ => {}
        }
    }
    false
}
