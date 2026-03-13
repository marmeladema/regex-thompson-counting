//! Lazy DFA (Tier 1, Tier 2, Tier 3, and Tier 4).
//!
//! **Tier 1** (counter-free patterns): standard lazy subset construction.
//! DFA-eligible patterns may include deferred assertions (`\b`, `\B`,
//! `EndLF`) whose evaluation needs the *next* byte.  These are parked
//! in the DFA state and resolved at the start of `populate()` when the
//! next input byte is known.
//!
//! Only `StartCRLF` and `EndCRLF` remain DFA-ineligible.
//!
//! **Tier 2** (non-nested, fixed-length-body repetitions): Becchi-style
//! differential counters.  O(1) per byte for patterns where every counter
//! body has a fixed byte length.
//!
//! **Tier 3** (non-nested counted repetitions): DFA + conditional transitions.
//! Precomputes separate DFA successor states per counter condition instead
//! of using runtime counter programs.  Each counter tracked independently.
//!
//! **Tier 4** (nested counted repetitions): DFA + explicit counter contexts.
//! The DFA state (set of NFA consuming states) is separated from the
//! counter state (a set of `CounterCtx` values).  DFA transitions are
//! cached normally; each transition also stores a compiled *counter
//! program* that describes how to update counter values.  On a cache
//! hit the program is replayed against each active counter context.

mod tier1;
mod tier2;
mod tier3;
pub(crate) mod tier3_effects;
mod tier4;

use indexmap::Equivalent;
use indexmap::IndexSet;

pub(crate) use tier1::{DfaMatcher, Tier1DfaCache};
pub(crate) use tier2::{Tier2Analysis, Tier2DfaCache, Tier2DfaMatcher, compute_tier2_analysis};
pub(crate) use tier3::{
    Tier3Analysis, Tier3DfaCache, Tier3DfaMatcher, Tier3OriginKind, compute_tier3_analysis,
};
pub(crate) use tier4::{Tier4DfaCache, Tier4DfaMatcher};

use crate::{AssertEval, AssertKind, CounterIdx, Regex, State, StateIdx};

/// Maximum number of DFA states before the flat transition table stops
/// growing.  2048 states × stride entries × 4 bytes.
const DFA_MAX_STATES: usize = 2048;

// ---------------------------------------------------------------------------
// DFA state table (shared by all tiers)
// ---------------------------------------------------------------------------

/// Index into the DFA state table ([`DfaCache::states`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DfaStateId(u32);

impl DfaStateId {
    /// Sentinel: the "dead" state (no NFA states, no match possible).
    pub(crate) const DEAD: Self = Self(u32::MAX);
    /// Sentinel for unpopulated flat-table slots.
    pub(super) const UNPOPULATED: Self = Self(u32::MAX - 1);

    #[inline]
    pub(super) fn idx(self) -> usize {
        self.0 as usize
    }
}

/// A DFA state: a sorted, deduplicated set of NFA consuming/assert state
/// indices, plus flags derived from the epsilon closure.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct DfaState {
    /// Sorted NFA state indices (consuming states only: Byte, ByteCI,
    /// ByteClass, ByteTable).  Assert states are resolved during closure
    /// computation;
    /// Match states set the `is_match` / `is_match_at_end` flags.
    pub(super) nfa_states: Box<[StateIdx]>,
    /// NFA Assert state indices whose evaluation was deferred because the
    /// next byte was unknown.  These are `WordAscii`, `WordAsciiNegate`,
    /// or `EndLF` states.  Resolved in `populate()` when the next byte
    /// arrives.
    pub(super) deferred_asserts: Box<[StateIdx]>,
    /// True if `Match` is directly reachable (no pending `$` gate).
    pub(super) is_match: bool,
    /// True if `Match` is reachable through an `Assert(End)` gate.
    /// Only fires when `at_end = true` (in `finish()`).
    pub(super) is_match_at_end: bool,
    /// Whether the previous byte (the byte that *entered* this state) was
    /// an ASCII word character.  Only meaningful when `deferred_asserts`
    /// is non-empty; always `false` otherwise, to avoid unnecessary state
    /// splitting.
    pub(super) prev_was_word: bool,
}

impl DfaState {
    /// Return a representative previous byte for assertion resolution.
    ///
    /// Deferred assertions that depend on the previous byte are exclusively
    /// from the word-boundary family (`\b`, `\B`, `\b{start}`, `\b{end}`),
    /// which only inspect `is_word_byte(prev)`.  `StartLF` never defers
    /// (it resolves immediately since `prev` is always known), and `EndLF`
    /// does not depend on `prev` at all.  So the bool captures all needed
    /// information without unnecessary DFA state splitting.
    #[inline]
    pub(super) fn prev_byte_representative(&self) -> Option<u8> {
        if self.prev_was_word {
            Some(b'a')
        } else {
            Some(b' ')
        }
    }

    /// Resolve deferred assertions from `from_state` given that `byte` is the
    /// next input byte.  Returns extra NFA targets (the `out` states of
    /// passing assertions) that should be followed through epsilon closure.
    ///
    /// `prev_was_word` is the word-ness of the byte that *entered* `from_state`.
    #[inline]
    fn resolve_deferred(&self, byte: u8, regex: &Regex) -> Vec<StateIdx> {
        let mut extra = Vec::new();
        if self.deferred_asserts.is_empty() {
            return extra;
        }
        let prev = self.prev_byte_representative();
        for &assert_idx in self.deferred_asserts.iter() {
            if let State::Assert { kind, out } = regex.states[assert_idx]
                && kind.eval(false, false, prev, Some(byte)) == AssertEval::Pass
            {
                extra.push(out);
            }
        }
        extra
    }

    /// Resolve deferred assertions at end-of-input.  Returns true if any
    /// deferred assertion passes and `Match` is reachable from its `out`.
    #[inline]
    fn resolve_deferred_at_end(&self, regex: &Regex) -> bool {
        if self.deferred_asserts.is_empty() {
            return false;
        }
        let prev = self.prev_byte_representative();
        for &assert_idx in self.deferred_asserts.iter() {
            if let State::Assert { kind, out } = regex.states[assert_idx]
                && kind.eval(false, true, prev, None) == AssertEval::Pass
                && Self::can_reach_match_at_end(out, prev, regex)
            {
                return true;
            }
        }
        false
    }

    /// Follow epsilon transitions from `start` at end-of-input, evaluating
    /// any assertions encountered.  Returns true if `Match` is reachable
    /// with all assertions passing.
    ///
    /// Unlike the static `state_can_reach_match` precomputation, this
    /// correctly handles adjacent assertions like `\b\B` which are
    /// statically reachable but dynamically impossible at the same position.
    pub(super) fn can_reach_match_at_end(start: StateIdx, prev: Option<u8>, regex: &Regex) -> bool {
        let states = &regex.states;
        let num_states = states.len();
        let mut visited = vec![false; num_states];
        let mut stack = vec![start];
        while let Some(idx) = stack.pop() {
            let i = idx.idx();
            // Quick static check: if Match is unreachable from this state
            // at all, skip it.
            if i >= num_states || visited[i] || !regex.state_can_reach_match[i] {
                continue;
            }
            visited[i] = true;
            match states[idx] {
                State::Match => return true,
                State::Assert { kind, out } => {
                    if kind.eval(false, true, prev, None) == AssertEval::Pass {
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
                // Consuming states cannot fire at end-of-input.
                _ => {}
            }
        }
        false
    }
}

/// Borrowed view of [`DfaState`] for zero-allocation [`IndexSet`] lookups.
///
/// `DfaStateRef` mirrors `DfaState` but holds `&[StateIdx]` slices instead
/// of `Box<[StateIdx]>`.  Because `Box<[T]>` and `&[T]` both delegate to
/// `[T]::hash()`, the derived `Hash` impls produce identical hashes —
/// making this safe to use with [`Equivalent`] for probing.
#[derive(Hash)]
pub(super) struct DfaStateRef<'a> {
    nfa_states: &'a [StateIdx],
    deferred_asserts: &'a [StateIdx],
    is_match: bool,
    is_match_at_end: bool,
    prev_was_word: bool,
}

impl Equivalent<DfaState> for DfaStateRef<'_> {
    fn equivalent(&self, key: &DfaState) -> bool {
        self.nfa_states == &*key.nfa_states
            && self.deferred_asserts == &*key.deferred_asserts
            && self.is_match == key.is_match
            && self.is_match_at_end == key.is_match_at_end
            && self.prev_was_word == key.prev_was_word
    }
}

/// Scratch space for epsilon-closure computation, shared across DFA
/// tiers 1–3.
///
/// All fields are temporary buffers that are cleared at the start of each
/// [`epsilon_closure`](Self::epsilon_closure) call.  The struct carries no
/// semantically meaningful state between calls — it exists purely to avoid
/// repeated heap allocation.
#[derive(Debug, Default)]
pub(crate) struct DfaMemory {
    // Scratch space for closure.
    pub(super) closure_stack: Vec<StateIdx>,
    pub(super) closure_result: Vec<StateIdx>,
    pub(super) closure_deferred: Vec<StateIdx>,
    pub(super) closure_visited: Vec<bool>,
    pub(super) closure_seeds: Vec<(CounterIdx, StateIdx)>,
}

impl DfaMemory {
    /// Reset the scratch buffers for reuse with a new regex.
    #[inline]
    pub(super) fn clear(&mut self, num_nfa_states: usize) {
        self.closure_deferred.clear();
        self.closure_visited.clear();
        self.closure_visited.resize(num_nfa_states, false);
        self.closure_seeds.clear();
    }

    /// Compute the epsilon closure from a set of NFA seed states.
    ///
    /// Follows `Split` and evaluable assertions (`Start`, `StartLF`).
    /// `End` (`$`) is handled specially: when not at end-of-input it sets
    /// `is_match_at_end` instead of following `out`.  Assertions that may
    /// need the next byte (`WordAscii`, `WordAsciiNegate`, `EndLF`) can
    /// return `Defer`, in which case the Assert state index is collected
    /// in `closure_deferred`.
    ///
    /// Returns `(is_match, is_match_at_end)`.  Consuming states are left
    /// in `self.closure_result`; deferred asserts in `self.closure_deferred`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn epsilon_closure(
        &mut self,
        seeds: impl Iterator<Item = StateIdx>,
        regex: &Regex,
        at_start: bool,
        at_end: bool,
        prev_byte: Option<u8>,
        next_byte: Option<u8>,
        mut counter_instance: impl FnMut(&mut Self, CounterIdx, StateIdx),
        mut counter_increment: impl FnMut(&mut Self, CounterIdx, StateIdx, StateIdx, usize, usize),
    ) -> (bool, bool) {
        self.closure_stack.clear();
        self.closure_result.clear();
        self.closure_deferred.clear();
        self.closure_seeds.clear();

        for v in self.closure_visited.iter_mut() {
            *v = false;
        }

        let mut is_match = false;
        let mut is_match_at_end = false;

        self.closure_stack.extend(seeds);
        while let Some(idx) = self.closure_stack.pop() {
            let i = idx.idx();
            if self.closure_visited[i] {
                continue;
            }
            self.closure_visited[i] = true;

            match regex.states[idx] {
                State::Split { out, out1 } => {
                    self.closure_stack.push(out1);
                    self.closure_stack.push(out);
                }
                State::Assert { kind, out } => {
                    // Special handling for $ (End): it fails when not at
                    // end-of-input, but we still need to record that a
                    // match is possible if we reach end-of-input later.
                    if kind == AssertKind::End {
                        if at_end {
                            self.closure_stack.push(out);
                        } else if regex.state_can_reach_match[out.idx()] {
                            is_match_at_end = true;
                        }
                        continue;
                    }
                    let result = kind.eval(at_start, at_end, prev_byte, next_byte);
                    match result {
                        AssertEval::Pass => {
                            self.closure_stack.push(out);
                        }
                        AssertEval::Fail => {
                            // Assertion failed — skip.
                        }
                        AssertEval::Defer => {
                            // Park this assert for deferred resolution.
                            self.closure_deferred.push(idx);
                        }
                    }
                }
                State::Match => {
                    is_match = true;
                }
                State::Byte { .. }
                | State::ByteCI { .. }
                | State::ByteClass { .. }
                | State::ByteTable { .. } => {
                    self.closure_result.push(idx);
                }
                State::CounterInstance { counter, out } => {
                    counter_instance(self, counter, out);
                }
                State::CounterIncrement {
                    counter,
                    out,
                    out1,
                    min,
                    max,
                } => {
                    counter_increment(self, counter, out, out1, min, max);
                }
            }
        }

        self.closure_result.sort_unstable_by_key(|s| s.0);
        self.closure_deferred.sort_unstable_by_key(|s| s.0);

        (is_match, is_match_at_end)
    }
}

struct DfaCache {
    states: IndexSet<DfaState, ahash::RandomState>,
    regex_id: u64,
    start_id: DfaStateId,
    start_is_match: bool,
    start_is_match_at_end: bool,
}

impl DfaCache {
    fn new() -> Self {
        Self {
            states: IndexSet::default(),
            regex_id: 0,
            start_id: DfaStateId::DEAD,
            start_is_match: false,
            start_is_match_at_end: false,
        }
    }

    /// Look up or insert a DFA state for the given sorted NFA state set.
    /// Returns `None` if the state cap ([`DFA_MAX_STATES`]) has been reached
    /// and the state is not already interned.
    ///
    /// Takes borrowed slices so that cache-hit probes (the common case)
    /// are zero-allocation.  A heap-allocated [`DfaState`] is only created
    /// on a cache miss when actual insertion is needed.
    #[inline]
    fn intern_state(
        &mut self,
        nfa_states: &[StateIdx],
        deferred_asserts: &[StateIdx],
        is_match: bool,
        is_match_at_end: bool,
        prev_was_word: bool,
        mut transition: impl FnMut(),
    ) -> Option<DfaStateId> {
        let probe = DfaStateRef {
            nfa_states,
            deferred_asserts,
            is_match,
            is_match_at_end,
            prev_was_word,
        };
        if let Some(idx) = self.states.get_index_of(&probe) {
            return Some(DfaStateId(idx as u32));
        }
        if self.states.len() >= DFA_MAX_STATES {
            return None;
        }
        let state = DfaState {
            nfa_states: nfa_states.into(),
            deferred_asserts: deferred_asserts.into(),
            is_match,
            is_match_at_end,
            prev_was_word,
        };
        let (idx, _) = self.states.insert_full(state);
        transition();
        Some(DfaStateId(idx as u32))
    }

    /// Reset the cache for reuse with a new regex.
    #[inline]
    fn clear(&mut self) {
        self.states.clear();
        self.regex_id = 0;
        self.start_id = DfaStateId::DEAD;
        self.start_is_match = false;
        self.start_is_match_at_end = false;
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{DefaultHasher, Hash, Hasher};

    use super::*;

    /// Compute the 64-bit hash of a value using [`DefaultHasher`].
    fn hash_of(val: &impl Hash) -> u64 {
        let mut h = DefaultHasher::new();
        val.hash(&mut h);
        h.finish()
    }

    /// Build a [`DfaState`] from the given fields (shorthand for tests).
    fn make_state(
        nfa: &[u32],
        deferred: &[u32],
        is_match: bool,
        is_match_at_end: bool,
        prev_was_word: bool,
    ) -> DfaState {
        DfaState {
            nfa_states: nfa.iter().map(|&v| StateIdx(v)).collect(),
            deferred_asserts: deferred.iter().map(|&v| StateIdx(v)).collect(),
            is_match,
            is_match_at_end,
            prev_was_word,
        }
    }

    /// Build a [`DfaStateRef`] from the given slices and flags.
    fn make_ref<'a>(
        nfa: &'a [StateIdx],
        deferred: &'a [StateIdx],
        is_match: bool,
        is_match_at_end: bool,
        prev_was_word: bool,
    ) -> DfaStateRef<'a> {
        DfaStateRef {
            nfa_states: nfa,
            deferred_asserts: deferred,
            is_match,
            is_match_at_end,
            prev_was_word,
        }
    }

    // -----------------------------------------------------------------------
    // Hash consistency: DfaState and DfaStateRef must hash identically
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_ref_hash_empty() {
        let state = make_state(&[], &[], false, false, false);
        let r = make_ref(&[], &[], false, false, false);
        assert_eq!(hash_of(&state), hash_of(&r));
    }

    #[test]
    fn test_state_ref_hash_with_nfa_states() {
        let state = make_state(&[1, 5, 42], &[], true, false, false);
        let nfa: Vec<StateIdx> = [1, 5, 42].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &[], true, false, false);
        assert_eq!(hash_of(&state), hash_of(&r));
    }

    #[test]
    fn test_state_ref_hash_with_deferred() {
        let state = make_state(&[3, 7], &[10, 20], false, true, true);
        let nfa: Vec<StateIdx> = [3, 7].iter().map(|&v| StateIdx(v)).collect();
        let deferred: Vec<StateIdx> = [10, 20].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &deferred, false, true, true);
        assert_eq!(hash_of(&state), hash_of(&r));
    }

    #[test]
    fn test_state_ref_hash_all_flags() {
        // Exhaustively check all 8 flag combinations.
        let nfa_raw = &[2, 4, 6];
        let def_raw = &[8];
        for m in [false, true] {
            for mae in [false, true] {
                for pw in [false, true] {
                    let state = make_state(nfa_raw, def_raw, m, mae, pw);
                    let nfa: Vec<StateIdx> = nfa_raw.iter().map(|&v| StateIdx(v)).collect();
                    let deferred: Vec<StateIdx> = def_raw.iter().map(|&v| StateIdx(v)).collect();
                    let r = make_ref(&nfa, &deferred, m, mae, pw);
                    assert_eq!(
                        hash_of(&state),
                        hash_of(&r),
                        "hash mismatch for flags ({m}, {mae}, {pw})"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Equivalent: DfaStateRef must correctly identify equal/unequal states
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_ref_equivalent_match() {
        let state = make_state(&[1, 2, 3], &[10], true, false, true);
        let nfa: Vec<StateIdx> = [1, 2, 3].iter().map(|&v| StateIdx(v)).collect();
        let deferred: Vec<StateIdx> = [10].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &deferred, true, false, true);
        assert!(r.equivalent(&state));
    }

    #[test]
    fn test_state_ref_equivalent_empty() {
        let state = make_state(&[], &[], false, false, false);
        let r = make_ref(&[], &[], false, false, false);
        assert!(r.equivalent(&state));
    }

    #[test]
    fn test_state_ref_not_equivalent_nfa_differs() {
        let state = make_state(&[1, 2], &[], false, false, false);
        let nfa: Vec<StateIdx> = [1, 3].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &[], false, false, false);
        assert!(!r.equivalent(&state));
    }

    #[test]
    fn test_state_ref_not_equivalent_deferred_differs() {
        let state = make_state(&[1], &[10], false, false, false);
        let nfa: Vec<StateIdx> = [1].iter().map(|&v| StateIdx(v)).collect();
        let deferred: Vec<StateIdx> = [11].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &deferred, false, false, false);
        assert!(!r.equivalent(&state));
    }

    #[test]
    fn test_state_ref_not_equivalent_is_match_differs() {
        let state = make_state(&[1], &[], true, false, false);
        let nfa: Vec<StateIdx> = [1].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &[], false, false, false);
        assert!(!r.equivalent(&state));
    }

    #[test]
    fn test_state_ref_not_equivalent_is_match_at_end_differs() {
        let state = make_state(&[1], &[], false, true, false);
        let nfa: Vec<StateIdx> = [1].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &[], false, false, false);
        assert!(!r.equivalent(&state));
    }

    #[test]
    fn test_state_ref_not_equivalent_prev_was_word_differs() {
        let state = make_state(&[1], &[], false, false, true);
        let nfa: Vec<StateIdx> = [1].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &[], false, false, false);
        assert!(!r.equivalent(&state));
    }

    #[test]
    fn test_state_ref_not_equivalent_length_differs() {
        let state = make_state(&[1, 2, 3], &[], false, false, false);
        let nfa: Vec<StateIdx> = [1, 2].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &[], false, false, false);
        assert!(!r.equivalent(&state));
    }

    // -----------------------------------------------------------------------
    // IndexSet round-trip: insert a DfaState, probe with DfaStateRef
    // -----------------------------------------------------------------------

    #[test]
    fn test_indexset_probe_with_ref() {
        let mut set: IndexSet<DfaState, ahash::RandomState> = IndexSet::default();
        let state = make_state(&[5, 10, 15], &[20], true, false, false);
        let (idx, inserted) = set.insert_full(state);
        assert!(inserted);
        assert_eq!(idx, 0);

        // Probe with a DfaStateRef — should find the same index.
        let nfa: Vec<StateIdx> = [5, 10, 15].iter().map(|&v| StateIdx(v)).collect();
        let deferred: Vec<StateIdx> = [20].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &deferred, true, false, false);
        assert_eq!(set.get_index_of(&r), Some(0));
    }

    #[test]
    fn test_indexset_probe_miss() {
        let mut set: IndexSet<DfaState, ahash::RandomState> = IndexSet::default();
        set.insert_full(make_state(&[1, 2], &[], false, false, false));

        // Different nfa_states — should not find.
        let nfa: Vec<StateIdx> = [1, 3].iter().map(|&v| StateIdx(v)).collect();
        let r = make_ref(&nfa, &[], false, false, false);
        assert_eq!(set.get_index_of(&r), None);
    }

    #[test]
    fn test_indexset_multiple_states() {
        let mut set: IndexSet<DfaState, ahash::RandomState> = IndexSet::default();
        set.insert_full(make_state(&[1], &[], false, false, false));
        set.insert_full(make_state(&[2], &[], false, false, false));
        set.insert_full(make_state(&[1], &[], true, false, false));

        let nfa1: Vec<StateIdx> = [1].iter().map(|&v| StateIdx(v)).collect();
        let nfa2: Vec<StateIdx> = [2].iter().map(|&v| StateIdx(v)).collect();

        // Each ref finds its corresponding state at the correct index.
        assert_eq!(
            set.get_index_of(&make_ref(&nfa1, &[], false, false, false)),
            Some(0)
        );
        assert_eq!(
            set.get_index_of(&make_ref(&nfa2, &[], false, false, false)),
            Some(1)
        );
        assert_eq!(
            set.get_index_of(&make_ref(&nfa1, &[], true, false, false)),
            Some(2)
        );

        // Non-existent combination.
        assert_eq!(
            set.get_index_of(&make_ref(&nfa2, &[], true, false, false)),
            None
        );
    }
}
