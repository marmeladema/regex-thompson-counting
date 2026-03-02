//! Tier 1: Lazy DFA for counter-free patterns.
//!
//! Standard lazy subset construction with support for deferred
//! assertions (`\b`, `\B`, `EndLF`).

use std::collections::HashMap;
use std::fmt;

use crate::{
    AssertEval, AssertKind, Prefilter, Regex, State, StateIdx, byte_match_ci, is_word_byte,
};

use super::{DfaState, DfaStateId};

// ---------------------------------------------------------------------------
// DFA cache (Tier 1)
// ---------------------------------------------------------------------------

/// Maximum number of DFA states before the flat transition table stops
/// growing.  2048 states × stride entries × 4 bytes.
const DFA_MAX_STATES: usize = 2048;

/// Lazy DFA cache: append-only state table + flat transition table.
///
/// The transition table is a flat `Vec<DfaStateId>` indexed by
/// `state.0 * stride + byte_class`, where `stride` is the number
/// of byte equivalence classes ([`Regex::num_byte_classes`]).
/// Unpopulated slots contain `DfaStateId::UNPOPULATED`.  When the
/// state count reaches [`DFA_MAX_STATES`], new states are no longer
/// interned — the transition falls back to recomputing without caching.
///
/// The cache is **persisted across `matcher()` calls** for the same
/// `Regex`.  A unique regex ID detects when a different regex is used
/// and clears the cache.
pub(crate) struct DfaCache {
    /// Append-only table of DFA states.  Index = `DfaStateId`.
    states: Vec<DfaState>,
    /// Reverse lookup: canonical `DfaState → DfaStateId`.
    state_map: HashMap<DfaState, DfaStateId>,
    /// Flat transition table: `transitions[state.0 * stride + byte_class]`.
    /// Unpopulated slots contain `DfaStateId::UNPOPULATED`.
    transitions: Vec<DfaStateId>,
    /// Number of byte equivalence classes — the stride of each DFA state
    /// row in the transition table.  Copied from [`Regex::num_byte_classes`]
    /// during [`prepare()`].
    stride: usize,
    /// Scratch space for epsilon closure (avoids allocation per populate).
    closure_stack: Vec<StateIdx>,
    /// Scratch space for collecting NFA consuming states during closure.
    closure_result: Vec<StateIdx>,
    /// Scratch space for collecting deferred assert states during closure.
    closure_deferred: Vec<StateIdx>,
    /// Scratch visited set for epsilon closure.
    closure_visited: Vec<bool>,
    /// Identity of the regex this cache was built for (see `Regex::id`).
    regex_id: u64,
    /// Cached start state (at_start=true, position 0).
    start_id: DfaStateId,
    /// Whether the start state is itself a match.
    start_is_match: bool,
}

impl fmt::Debug for DfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DfaCache")
            .field("num_states", &self.states.len())
            .finish()
    }
}

impl DfaCache {
    pub(crate) fn new(num_nfa_states: usize) -> Self {
        Self {
            states: Vec::new(),
            state_map: HashMap::new(),
            transitions: Vec::new(),
            stride: 256, // default; overwritten by prepare()
            closure_stack: Vec::new(),
            closure_result: Vec::new(),
            closure_deferred: Vec::new(),
            closure_visited: vec![false; num_nfa_states],
            regex_id: 0,
            start_id: DfaStateId::DEAD,
            start_is_match: false,
        }
    }

    /// Look up or insert a DFA state for the given sorted NFA state set.
    /// Returns `None` if the state cap ([`DFA_MAX_STATES`]) has been reached
    /// and the state is not already interned.
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
        // Extend the flat transition table with `stride` UNPOPULATED slots.
        self.transitions
            .extend(std::iter::repeat_n(DfaStateId::UNPOPULATED, self.stride));
        Some(id)
    }

    /// Compute the epsilon closure from a set of NFA seed states.
    ///
    /// Follows `Split` and evaluable assertions (`Start`, `End`, `StartLF`).
    /// Assertions that need the next byte (`WordAscii`, `WordAsciiNegate`,
    /// `EndLF`) are evaluated with `next=None`; if they return `Defer`, the
    /// Assert state index is collected in `closure_deferred`.
    ///
    /// Returns `(consuming_states, deferred_asserts, is_match, is_match_at_end)`.
    fn epsilon_closure(
        &mut self,
        seeds: impl Iterator<Item = StateIdx>,
        states: &[State],
        at_start: bool,
        at_end: bool,
        prev_byte: Option<u8>,
    ) -> (Box<[StateIdx]>, Box<[StateIdx]>, bool, bool) {
        self.closure_stack.clear();
        self.closure_result.clear();
        self.closure_deferred.clear();
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

            match states[idx] {
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
                        } else if self.can_reach_match(out, states) {
                            is_match_at_end = true;
                        }
                        continue;
                    }
                    let result = kind.eval(at_start, at_end, prev_byte, None);
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
                State::CounterInstance { .. } | State::CounterIncrement { .. } => {
                    debug_assert!(false, "counter state in DFA-eligible pattern");
                }
            }
        }

        self.closure_result.sort_unstable_by_key(|s| s.0);
        self.closure_result.dedup();
        self.closure_deferred.sort_unstable_by_key(|s| s.0);
        self.closure_deferred.dedup();

        let nfa_states: Box<[StateIdx]> = self.closure_result.as_slice().into();
        let deferred: Box<[StateIdx]> = self.closure_deferred.as_slice().into();
        (nfa_states, deferred, is_match, is_match_at_end)
    }

    /// Check if `Match` is reachable from `idx` through epsilon transitions.
    /// Follows Split, all Assert kinds (optimistically), to determine if
    /// a match is *possible* (used for `is_match_at_end` via `$`).
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
                State::Assert { out, .. } => {
                    // Follow all assertions optimistically — if there is
                    // *any* path through assertions to Match, we consider
                    // a match possible.
                    stack.push(out);
                }
                _ => {}
            }
        }
        false
    }

    /// Resolve deferred assertions from `from_state` given that `byte` is the
    /// next input byte.  Returns extra NFA targets (the `out` states of
    /// passing assertions) that should be followed through epsilon closure.
    ///
    /// `prev_was_word` is the word-ness of the byte that *entered* `from_state`.
    fn resolve_deferred(&self, from_state: &DfaState, byte: u8, regex: &Regex) -> Vec<StateIdx> {
        let mut extra_targets = Vec::new();
        if from_state.deferred_asserts.is_empty() {
            return extra_targets;
        }
        // Reconstruct prev byte for assertion evaluation.
        // We only need word-ness, so use synthetic bytes.
        let prev = if from_state.prev_was_word {
            Some(b'a')
        } else {
            Some(b' ')
        };
        for &assert_idx in from_state.deferred_asserts.iter() {
            if let State::Assert { kind, out } = regex.states[assert_idx] {
                let result = kind.eval(false, false, prev, Some(byte));
                if result == AssertEval::Pass {
                    extra_targets.push(out);
                }
            }
        }
        extra_targets
    }

    /// Resolve deferred assertions at end-of-input.  Returns true if any
    /// deferred assertion passes and `Match` is reachable from its `out`.
    fn resolve_deferred_at_end(&self, state: &DfaState, regex: &Regex) -> bool {
        if state.deferred_asserts.is_empty() {
            return false;
        }
        let prev = if state.prev_was_word {
            Some(b'a')
        } else {
            Some(b' ')
        };
        for &assert_idx in state.deferred_asserts.iter() {
            if let State::Assert { kind, out } = regex.states[assert_idx] {
                let result = kind.eval(false, true, prev, None);
                if result == AssertEval::Pass && self.can_reach_match(out, &regex.states) {
                    return true;
                }
            }
        }
        false
    }

    /// Compute the DFA transition for `(from_state, byte)`.
    #[inline(always)]
    fn transition(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> DfaStateId {
        if from == DfaStateId::DEAD {
            return self.transition_from_dead(byte, regex);
        }
        let class = regex.byte_classes[byte as usize] as usize;
        let slot = from.idx() * self.stride + class;
        let cached = self.transitions[slot];
        if cached != DfaStateId::UNPOPULATED {
            return cached;
        }
        let to = self.populate(from, byte, regex);
        self.transitions[slot] = to;
        to
    }

    /// Transition from the DEAD state (re-seed from start).
    /// Separated to keep the hot path of `transition()` small.
    #[inline(never)]
    fn transition_from_dead(&mut self, byte: u8, regex: &Regex) -> DfaStateId {
        self.populate(DfaStateId::DEAD, byte, regex)
    }

    /// On cache miss: compute the next DFA state.
    ///
    /// Phase 1: Resolve deferred assertions from `from` using `byte` as next.
    ///          Passing assertions produce extra epsilon-closure targets.
    /// Phase 2: All consuming NFA states in `from` attempt to consume `byte`.
    /// Phase 3: Epsilon closure of (consumed targets + resolved targets + seed).
    fn populate(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> DfaStateId {
        let mut targets: Vec<StateIdx> = Vec::new();

        if from != DfaStateId::DEAD {
            let from_state = &self.states[from.idx()];

            // Phase 1: resolve deferred assertions.
            let extra = self.resolve_deferred(from_state, byte, regex);
            // The extra targets are `out` states of passing assertions.
            // These need to go through epsilon closure, where they may
            // reach consuming states that then also consume `byte`.
            // But wait — the assert's `out` leads to epsilon states
            // (Split, more asserts, consuming states).  The consuming
            // states reached from `out` are NEW consuming states that
            // were blocked behind the deferred assertion.  They need to
            // consume `byte` too.  So we do a mini epsilon closure of
            // the `extra` targets to find their consuming states, then
            // let those consuming states consume `byte`.
            if !extra.is_empty() {
                // Mini epsilon closure: find consuming states reachable
                // from the resolved assertion outputs.
                let (resolved_consumers, _resolved_deferred, resolved_match, resolved_match_at_end) =
                    self.epsilon_closure(
                        extra.into_iter(),
                        &regex.states,
                        false,
                        false,
                        Some(byte), // prev_byte = byte for the assertion context
                    );
                // These resolved consumers can now consume `byte`.
                // NOTE: We must re-read from_state after epsilon_closure
                // because it borrows &mut self.
                let from_state = &self.states[from.idx()];
                for &idx in resolved_consumers.iter() {
                    let target = match regex.states[idx] {
                        State::Byte { byte: b2, out } if byte == b2 => Some(out),
                        State::ByteCI { byte: b2, out } if byte_match_ci(byte, b2) => Some(out),
                        State::ByteClass { class, out } if regex.classes[class][byte] => Some(out),
                        State::ByteTable { table } => {
                            let t = regex.byte_tables[table][byte];
                            if t != StateIdx::NONE { Some(t) } else { None }
                        }
                        _ => None,
                    };
                    if let Some(t) = target {
                        targets.push(t);
                    }
                }
                // If the resolved closure itself found a match or
                // match_at_end, we need to handle it.  But `resolved_match`
                // means Match was directly in the closure — which means
                // matching without consuming `byte`.  For unanchored search,
                // that would mean the pattern matched at the position before
                // this byte.  However, the DFA uses `is_match` on the FROM
                // state to detect this — so we should not set it on the TO
                // state.  The deferred assert resolving to a match means
                // the match was at the *previous* step.  We handle this
                // through `is_match_at_end` logic instead.
                // Actually — if the resolved closure hits Match directly,
                // that means the pattern matches here (the deferred assert
                // passed with `next=byte` and Match follows).  This match
                // occurs at the current position, so we should propagate it.
                // We'll merge these flags into the final closure result below.
                let _ = (resolved_match, resolved_match_at_end);
                // Actually we need to track these — see below after Phase 2.
                // For now, also note any deferred asserts from resolved closure
                // go into the TO state, but since we just resolved with
                // `next=Some(byte)`, there shouldn't be new deferred asserts
                // (prev_byte is known, next is known → no Defer).
                // _resolved_deferred should be empty.

                // Phase 2: original consuming states consume `byte`.
                let nfa_states = from_state.nfa_states.clone();
                for &idx in nfa_states.iter() {
                    let target = match regex.states[idx] {
                        State::Byte { byte: b2, out } if byte == b2 => Some(out),
                        State::ByteCI { byte: b2, out } if byte_match_ci(byte, b2) => Some(out),
                        State::ByteClass { class, out } if regex.classes[class][byte] => Some(out),
                        State::ByteTable { table } => {
                            let t = regex.byte_tables[table][byte];
                            if t != StateIdx::NONE { Some(t) } else { None }
                        }
                        _ => None,
                    };
                    if let Some(t) = target {
                        targets.push(t);
                    }
                }

                // Phase 3: epsilon closure of all targets + re-seed.
                if targets.is_empty() {
                    // No consuming state produced a target.  Re-seed.
                    let (nfa_set, deferred, mut m, mut mae) = self.epsilon_closure(
                        std::iter::once(regex.start),
                        &regex.states,
                        false,
                        false,
                        Some(byte),
                    );
                    m |= resolved_match;
                    mae |= resolved_match_at_end;
                    if nfa_set.is_empty() && deferred.is_empty() && !m && !mae {
                        return DfaStateId::DEAD;
                    }
                    let pw = if deferred.is_empty() {
                        false
                    } else {
                        is_word_byte(byte)
                    };
                    return self
                        .intern_state(nfa_set, deferred, m, mae, pw)
                        .unwrap_or(DfaStateId::DEAD);
                }

                let seeds = targets.into_iter().chain(std::iter::once(regex.start));
                let (nfa_set, deferred, mut m, mut mae) =
                    self.epsilon_closure(seeds, &regex.states, false, false, Some(byte));
                m |= resolved_match;
                mae |= resolved_match_at_end;
                if nfa_set.is_empty() && deferred.is_empty() && !m && !mae {
                    return DfaStateId::DEAD;
                }
                let pw = if deferred.is_empty() {
                    false
                } else {
                    is_word_byte(byte)
                };
                return self
                    .intern_state(nfa_set, deferred, m, mae, pw)
                    .unwrap_or(DfaStateId::DEAD);
            }

            // No deferred assertions to resolve — fast path (Phase 2 only).
            let nfa_states = from_state.nfa_states.clone();
            for &idx in nfa_states.iter() {
                let target = match regex.states[idx] {
                    State::Byte { byte: b2, out } if byte == b2 => Some(out),
                    State::ByteCI { byte: b2, out } if byte_match_ci(byte, b2) => Some(out),
                    State::ByteClass { class, out } if regex.classes[class][byte] => Some(out),
                    State::ByteTable { table } => {
                        let t = regex.byte_tables[table][byte];
                        if t != StateIdx::NONE { Some(t) } else { None }
                    }
                    _ => None,
                };
                if let Some(t) = target {
                    targets.push(t);
                }
            }
        }

        if targets.is_empty() {
            let (nfa_set, deferred, is_match, is_match_at_end) = self.epsilon_closure(
                std::iter::once(regex.start),
                &regex.states,
                false,
                false,
                Some(byte),
            );
            if nfa_set.is_empty() && deferred.is_empty() && !is_match && !is_match_at_end {
                return DfaStateId::DEAD;
            }
            let pw = if deferred.is_empty() {
                false
            } else {
                is_word_byte(byte)
            };
            return self
                .intern_state(nfa_set, deferred, is_match, is_match_at_end, pw)
                .unwrap_or(DfaStateId::DEAD);
        }

        let seeds = targets.into_iter().chain(std::iter::once(regex.start));
        let (nfa_set, deferred, is_match, is_match_at_end) =
            self.epsilon_closure(seeds, &regex.states, false, false, Some(byte));

        if nfa_set.is_empty() && deferred.is_empty() && !is_match && !is_match_at_end {
            return DfaStateId::DEAD;
        }

        let pw = if deferred.is_empty() {
            false
        } else {
            is_word_byte(byte)
        };
        self.intern_state(nfa_set, deferred, is_match, is_match_at_end, pw)
            .unwrap_or(DfaStateId::DEAD)
    }

    /// Reset the cache for reuse with a new regex.
    fn clear(&mut self, num_nfa_states: usize, stride: usize) {
        self.states.clear();
        self.state_map.clear();
        self.transitions.clear();
        self.stride = stride;
        self.closure_deferred.clear();
        self.closure_visited.clear();
        self.closure_visited.resize(num_nfa_states, false);
        self.regex_id = 0;
        self.start_id = DfaStateId::DEAD;
        self.start_is_match = false;
    }

    /// Number of interned DFA states (for diagnostics / testing).
    #[allow(dead_code)]
    pub(crate) fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Prepare the cache for use with `regex`.
    pub(crate) fn prepare(&mut self, regex: &Regex) {
        let id = regex.id;
        if self.regex_id == id && self.start_id != DfaStateId::DEAD {
            return;
        }
        self.clear(regex.states.len(), regex.num_byte_classes);
        self.regex_id = id;

        // Start state: at_start=true, prev_byte=None, at_end=false.
        // Deferred assertions at start (e.g., `\b` at pos 0) get
        // prev=None → prev_was_word=false.
        let (nfa_set, deferred, is_match, is_match_at_end) = self.epsilon_closure(
            std::iter::once(regex.start),
            &regex.states,
            true,
            false,
            None,
        );
        let pw = false; // At start, prev is always non-word.
        self.start_id = self
            .intern_state(nfa_set, deferred, is_match, is_match_at_end, pw)
            .expect("start state exceeds DFA_MAX_STATES");
        self.start_is_match = self.states[self.start_id.idx()].is_match;
    }
}

// ---------------------------------------------------------------------------
// Tier 1 DFA matcher (counter-free)
// ---------------------------------------------------------------------------

/// Lazy DFA matcher for Tier 1 (counter-free) patterns.
pub struct DfaMatcher<'a> {
    cache: &'a mut DfaCache,
    regex: &'a Regex,
    current: DfaStateId,
    ever_matched: bool,
    prefilter: Prefilter,
}

impl<'a> DfaMatcher<'a> {
    pub(crate) fn new(cache: &'a mut DfaCache, regex: &'a Regex) -> Self {
        DfaMatcher {
            current: cache.start_id,
            ever_matched: cache.start_is_match,
            cache,
            regex,
            prefilter: regex.prefilter,
        }
    }

    #[inline(always)]
    pub fn step(&mut self, byte: u8) {
        self.current = self.cache.transition(self.current, byte, self.regex);
        if self.current != DfaStateId::DEAD && self.cache.states[self.current.idx()].is_match {
            self.ever_matched = true;
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

        for &b in input {
            if self.ever_matched {
                return;
            }
            self.step(b);
        }
    }

    pub fn finish(self) -> bool {
        if self.ever_matched {
            return true;
        }
        if self.current != DfaStateId::DEAD {
            let state = &self.cache.states[self.current.idx()];
            if state.is_match_at_end {
                return true;
            }
            // Check deferred assertions at end-of-input.
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

impl fmt::Debug for DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DfaMatcher")
            .field("current", &self.current)
            .field("ever_matched", &self.ever_matched)
            .finish()
    }
}
