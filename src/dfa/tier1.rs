//! Tier 1: Lazy DFA for counter-free patterns.
//!
//! Standard lazy subset construction with support for deferred
//! assertions (`\b`, `\B`, `EndLF`).

use std::fmt;

use crate::{Prefilter, Regex, State, StateIdx, byte_match_ci, is_word_byte};

use super::{DfaCache, DfaMemory, DfaStateId};

// ---------------------------------------------------------------------------
// DFA cache (Tier 1)
// ---------------------------------------------------------------------------

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
pub(crate) struct Tier1DfaCache {
    inner: DfaCache,
    stride: usize,
    /// Flat transition table: `transitions[state.0 * stride + byte_class]`.
    /// Unpopulated slots contain `DfaStateId::UNPOPULATED`.
    transitions: Vec<DfaStateId>,
}

impl fmt::Debug for Tier1DfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier1DfaCache")
            .field("num_states", &self.inner.states.len())
            .finish()
    }
}

impl Tier1DfaCache {
    pub(crate) fn new() -> Self {
        Self {
            inner: DfaCache::new(),
            stride: 256,
            transitions: Vec::new(),
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
        let stride = self.stride;
        self.inner.intern_state(
            nfa_states,
            deferred_asserts,
            is_match,
            is_match_at_end,
            prev_was_word,
            || {
                self.transitions
                    .extend(std::iter::repeat_n(DfaStateId::UNPOPULATED, stride));
            },
        )
    }

    /// Compute the epsilon closure from a set of NFA seed states.
    ///
    /// Follows `Split` and evaluable assertions (`Start`, `End`, `StartLF`).
    /// Assertions that need the next byte (`WordAscii`, `WordAsciiNegate`,
    /// `EndLF`) are evaluated with `next=None`; if they return `Defer`, the
    /// Assert state index is collected in `closure_deferred`.
    ///
    /// Returns `(consuming_states, deferred_asserts, is_match, is_match_at_end)`.
    #[allow(clippy::too_many_arguments)]
    fn epsilon_closure(
        &mut self,
        memory: &mut DfaMemory,
        seeds: impl Iterator<Item = StateIdx>,
        regex: &Regex,
        at_start: bool,
        at_end: bool,
        prev_byte: Option<u8>,
        next_byte: Option<u8>,
    ) -> (Box<[StateIdx]>, Box<[StateIdx]>, bool, bool) {
        let (is_match, is_match_at_end) = memory.epsilon_closure(
            seeds,
            regex,
            at_start,
            at_end,
            prev_byte,
            next_byte,
            |_, _, _| unreachable!(),
            |_, _, _, _, _, _| unreachable!(),
        );

        let nfa_states: Box<[StateIdx]> = memory.closure_result.as_slice().into();
        let deferred: Box<[StateIdx]> = memory.closure_deferred.as_slice().into();
        (nfa_states, deferred, is_match, is_match_at_end)
    }

    /// Compute the DFA transition for `(from_state, byte)`, using byte-class
    /// compression (stride < 256).
    #[inline(always)]
    fn transition(
        &mut self,
        memory: &mut DfaMemory,
        from: DfaStateId,
        byte: u8,
        regex: &Regex,
    ) -> DfaStateId {
        if from == DfaStateId::DEAD {
            return self.transition_from_dead(memory, byte, regex);
        }
        let class = regex.byte_classes[byte as usize] as usize;
        let slot = from.idx() * self.stride + class;
        let cached = self.transitions[slot];
        if cached != DfaStateId::UNPOPULATED {
            return cached;
        }
        let to = self.populate(memory, from, byte, regex);
        self.transitions[slot] = to;
        to
    }

    /// Compute the DFA transition for `(from_state, byte)`, using the
    /// identity mapping (stride=256, no byte-class indirection).
    #[inline(always)]
    fn transition_direct(
        &mut self,
        memory: &mut DfaMemory,
        from: DfaStateId,
        byte: u8,
        regex: &Regex,
    ) -> DfaStateId {
        if from == DfaStateId::DEAD {
            return self.transition_from_dead(memory, byte, regex);
        }
        let slot = from.idx() * 256 + byte as usize;
        let cached = self.transitions[slot];
        if cached != DfaStateId::UNPOPULATED {
            return cached;
        }
        let to = self.populate(memory, from, byte, regex);
        self.transitions[slot] = to;
        to
    }

    /// Transition from the DEAD state (re-seed from start).
    /// Separated to keep the hot path of `transition()` small.
    #[inline(never)]
    fn transition_from_dead(
        &mut self,
        memory: &mut DfaMemory,
        byte: u8,
        regex: &Regex,
    ) -> DfaStateId {
        self.populate(memory, DfaStateId::DEAD, byte, regex)
    }

    /// On cache miss: compute the next DFA state.
    ///
    /// Phase 1: Resolve deferred assertions from `from` using `byte` as next.
    ///          Passing assertions produce extra epsilon-closure targets.
    /// Phase 2: All consuming NFA states in `from` attempt to consume `byte`.
    /// Phase 3: Epsilon closure of (consumed targets + resolved targets + seed).
    fn populate(
        &mut self,
        memory: &mut DfaMemory,
        from: DfaStateId,
        byte: u8,
        regex: &Regex,
    ) -> DfaStateId {
        let mut targets: Vec<StateIdx> = Vec::new();

        if from != DfaStateId::DEAD {
            let from_state = &self.inner.states[from.idx()];

            // Phase 1: resolve deferred assertions.
            let extra = from_state.resolve_deferred(byte, regex);
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
                let resolved_prev = from_state.prev_byte_representative();
                let (resolved_consumers, _resolved_deferred, resolved_match, resolved_match_at_end) =
                    self.epsilon_closure(
                        memory,
                        extra.into_iter(),
                        regex,
                        false,
                        false,
                        resolved_prev,
                        Some(byte), // next_byte: assertions in chain can see the current byte
                    );
                // These resolved consumers can now consume `byte`.
                // NOTE: We must re-read from_state after epsilon_closure
                // because it borrows &mut self.
                let from_state = &self.inner.states[from.idx()];
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
                        memory,
                        std::iter::once(regex.start),
                        regex,
                        false,
                        false,
                        Some(byte),
                        None,
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
                    self.epsilon_closure(memory, seeds, regex, false, false, Some(byte), None);
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
                memory,
                std::iter::once(regex.start),
                regex,
                false,
                false,
                Some(byte),
                None,
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
            self.epsilon_closure(memory, seeds, regex, false, false, Some(byte), None);

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
    fn clear(&mut self, memory: &mut DfaMemory, num_nfa_states: usize, stride: usize) {
        self.inner.clear();
        self.stride = stride;
        memory.clear(num_nfa_states);
        self.transitions.clear();
    }

    /// Prepare the cache for use with `regex`.
    #[inline]
    pub(crate) fn prepare(&mut self, memory: &mut DfaMemory, regex: &Regex) {
        let id = regex.id;
        if self.inner.regex_id == id && self.inner.start_id != DfaStateId::DEAD {
            return;
        }
        self.clear(memory, regex.states.len(), regex.num_byte_classes);
        self.inner.regex_id = id;

        // Start state: at_start=true, prev_byte=None, at_end=false.
        // Deferred assertions at start (e.g., `\b` at pos 0) get
        // prev=None → prev_was_word=false.
        let (nfa_set, deferred, is_match, is_match_at_end) = self.epsilon_closure(
            memory,
            std::iter::once(regex.start),
            regex,
            true,
            false,
            None,
            None,
        );
        let pw = false; // At start, prev is always non-word.
        self.inner.start_id = self
            .intern_state(nfa_set, deferred, is_match, is_match_at_end, pw)
            .expect("start state exceeds DFA_MAX_STATES");
        self.inner.start_is_match = self.inner.states[self.inner.start_id.idx()].is_match;
    }
}

// ---------------------------------------------------------------------------
// Tier 1 DFA matcher (counter-free)
// ---------------------------------------------------------------------------

/// Lazy DFA matcher for Tier 1 (counter-free) patterns.
pub struct DfaMatcher<'a> {
    cache: &'a mut Tier1DfaCache,
    memory: &'a mut DfaMemory,
    regex: &'a Regex,
    current: DfaStateId,
    ever_matched: bool,
    prefilter: Prefilter,
}

impl<'a> DfaMatcher<'a> {
    #[inline]
    pub(crate) fn new(
        cache: &'a mut Tier1DfaCache,
        memory: &'a mut DfaMemory,
        regex: &'a Regex,
    ) -> Self {
        DfaMatcher {
            current: cache.inner.start_id,
            ever_matched: cache.inner.start_is_match,
            cache,
            memory,
            regex,
            prefilter: regex.prefilter,
        }
    }

    #[inline(always)]
    pub fn step(&mut self, byte: u8) {
        if self.cache.stride == 256 {
            self.current =
                self.cache
                    .transition_direct(self.memory, self.current, byte, self.regex);
        } else {
            self.current = self
                .cache
                .transition(self.memory, self.current, byte, self.regex);
        }
        if self.current != DfaStateId::DEAD && self.cache.inner.states[self.current.idx()].is_match
        {
            self.ever_matched = true;
        }
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
                self.current =
                    self.cache
                        .transition_direct(self.memory, self.current, b, self.regex);
                if self.current != DfaStateId::DEAD
                    && self.cache.inner.states[self.current.idx()].is_match
                {
                    self.ever_matched = true;
                }
            }
        } else {
            for &b in input {
                if self.ever_matched {
                    return;
                }
                self.current = self
                    .cache
                    .transition(self.memory, self.current, b, self.regex);
                if self.current != DfaStateId::DEAD
                    && self.cache.inner.states[self.current.idx()].is_match
                {
                    self.ever_matched = true;
                }
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
            if self.current == start_id {
                if let Some(offset) = finder(&input[i..]) {
                    i += offset;
                } else {
                    return;
                }
            }
            if self.cache.stride == 256 {
                self.current =
                    self.cache
                        .transition_direct(self.memory, self.current, input[i], self.regex);
            } else {
                self.current =
                    self.cache
                        .transition(self.memory, self.current, input[i], self.regex);
            }
            if self.current != DfaStateId::DEAD
                && self.cache.inner.states[self.current.idx()].is_match
            {
                self.ever_matched = true;
            }
            i += 1;
        }
    }

    #[inline]
    pub fn finish(self) -> bool {
        if self.ever_matched {
            return true;
        }
        if self.current != DfaStateId::DEAD {
            let state = &self.cache.inner.states[self.current.idx()];
            if state.is_match_at_end {
                return true;
            }

            // Check deferred assertions at end-of-input.
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

impl fmt::Debug for DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DfaMatcher")
            .field("current", &self.current)
            .field("ever_matched", &self.ever_matched)
            .finish()
    }
}
