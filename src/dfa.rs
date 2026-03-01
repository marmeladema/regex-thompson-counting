//! Lazy DFA (Tier 1: counter-free patterns).
//!
//! Standard lazy subset construction over the NFA.  DFA-eligible patterns
//! have no bounded repetition counters and no complex assertions (`\b`,
//! `\B`, `EndLF`, `EndCRLF`, `StartCRLF`).  Simple assertions (`^`, `$`,
//! `StartLF`) are handled natively.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;

use clru::CLruCache;

use crate::{AssertKind, Regex, State, StateIdx};

// ---------------------------------------------------------------------------
// DFA state table
// ---------------------------------------------------------------------------

/// Index into the DFA state table ([`DfaCache::states`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DfaStateId(u32);

impl DfaStateId {
    /// Sentinel: the "dead" state (no NFA states, no match possible).
    const DEAD: Self = Self(u32::MAX);

    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}

/// A DFA state: a sorted, deduplicated set of NFA consuming/assert state
/// indices, plus flags derived from the epsilon closure.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DfaState {
    /// Sorted NFA state indices (consuming states only: Byte, ByteClass,
    /// ByteTable).  Assert states are resolved during closure computation;
    /// Match states set the `is_match` / `is_match_at_end` flags.
    nfa_states: Box<[StateIdx]>,
    /// True if `Match` is directly reachable (no pending `$` gate).
    is_match: bool,
    /// True if `Match` is reachable through an `Assert(End)` gate.
    /// Only fires when `at_end = true` (in `finish()`).
    is_match_at_end: bool,
}

// ---------------------------------------------------------------------------
// DFA cache
// ---------------------------------------------------------------------------

/// Default DFA transition cache capacity (number of (state, byte) entries).
const DFA_CACHE_CAPACITY: usize = 16_384;

/// Lazy DFA cache: append-only state table + LRU transition cache.
///
/// The state table never evicts entries (DFA states are canonical and
/// referenced by `DfaStateId`).  The transition cache uses LRU eviction
/// via `CLruCache`; evicting a transition is always safe — it just
/// causes a cache miss that triggers re-population.
///
/// The cache is **persisted across `matcher()` calls** for the same
/// `Regex`.  A pointer-based identity check (`regex_id`) detects when
/// a different regex is used and clears the cache.
pub(crate) struct DfaCache {
    /// Append-only table of DFA states.  Index = `DfaStateId`.
    states: Vec<DfaState>,
    /// Reverse lookup: canonical `DfaState → DfaStateId`.
    ///
    /// The key includes the match flags because the same set of NFA
    /// consuming states can have different match reachability depending
    /// on how the epsilon closure was reached (e.g. whether a `$` gate
    /// was traversed).
    state_map: HashMap<DfaState, DfaStateId>,
    /// LRU transition cache: `(from_state, byte) -> to_state`.
    transitions: CLruCache<(DfaStateId, u8), DfaStateId>,
    /// Scratch space for epsilon closure (avoids allocation per populate).
    closure_stack: Vec<StateIdx>,
    /// Scratch space for collecting NFA states during closure.
    closure_result: Vec<StateIdx>,
    /// Scratch visited set for epsilon closure.
    closure_visited: Vec<bool>,
    /// Identity of the regex this cache was built for (see `Regex::id`).
    /// If a different regex is used, the cache is cleared.
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
            transitions: CLruCache::new(NonZeroUsize::new(DFA_CACHE_CAPACITY).unwrap()),
            closure_stack: Vec::new(),
            closure_result: Vec::new(),
            closure_visited: vec![false; num_nfa_states],
            regex_id: 0,
            start_id: DfaStateId::DEAD,
            start_is_match: false,
        }
    }

    /// Look up or insert a DFA state for the given sorted NFA state set.
    /// Returns the `DfaStateId`.
    fn intern_state(
        &mut self,
        nfa_states: Box<[StateIdx]>,
        is_match: bool,
        is_match_at_end: bool,
    ) -> DfaStateId {
        // Move nfa_states into a DfaState for the lookup -- get() only
        // borrows, so no clone is needed on the hot (hit) path.
        // On a miss we clone the state for the map key.
        let state = DfaState {
            nfa_states,
            is_match,
            is_match_at_end,
        };
        if let Some(&id) = self.state_map.get(&state) {
            return id;
        }
        let id = DfaStateId(self.states.len() as u32);
        self.state_map.insert(state.clone(), id);
        self.states.push(state);
        id
    }

    /// Compute the epsilon closure from a set of NFA seed states.
    ///
    /// Follows `Split` and `Assert(Start)`/`Assert(End)` states,
    /// collecting consuming leaves (Byte, ByteClass, ByteTable).
    ///
    /// - `Assert(Start)`: followed only when `at_start` is true.
    /// - `Assert(End)`: never followed during normal closure
    ///   (`at_end` is false).  Instead, a secondary check determines
    ///   if `Match` is reachable through it (stored as `is_match_at_end`).
    /// - `Assert(StartLF)`: followed when `prev_byte == Some(b'\n')`
    ///   or `at_start`.
    ///
    /// Counter states (`CounterInstance`, `CounterIncrement`) are
    /// never encountered for DFA-eligible patterns.
    fn epsilon_closure(
        &mut self,
        seeds: impl Iterator<Item = StateIdx>,
        states: &[State],
        at_start: bool,
        prev_byte: Option<u8>,
    ) -> (Box<[StateIdx]>, bool, bool) {
        // Reset scratch space.
        self.closure_stack.clear();
        self.closure_result.clear();
        // Reset only the visited entries we touched last time (faster than
        // clearing the whole vec for small NFA).
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
                    match kind {
                        AssertKind::Start => {
                            if at_start {
                                self.closure_stack.push(out);
                            }
                        }
                        AssertKind::End => {
                            // Don't follow during normal operation.
                            // Check if Match is reachable through this
                            // for the finish() path.
                            if self.can_reach_match(out, states) {
                                is_match_at_end = true;
                            }
                        }
                        AssertKind::StartLF => {
                            if at_start || prev_byte == Some(b'\n') {
                                self.closure_stack.push(out);
                            }
                        }
                        // Complex assertions should not appear in DFA-eligible
                        // patterns (filtered at compile time).
                        _ => {
                            debug_assert!(
                                false,
                                "complex assertion in DFA-eligible pattern: {:?}",
                                kind
                            );
                        }
                    }
                }
                State::Match => {
                    is_match = true;
                }
                // Consuming states: include in result.
                State::Byte { .. } | State::ByteClass { .. } | State::ByteTable { .. } => {
                    self.closure_result.push(idx);
                }
                // Counter states: should never appear in DFA-eligible patterns.
                State::CounterInstance { .. } | State::CounterIncrement { .. } => {
                    debug_assert!(false, "counter state in DFA-eligible pattern");
                }
            }
        }

        // Sort and deduplicate for canonical representation.
        self.closure_result.sort_unstable_by_key(|s| s.0);
        self.closure_result.dedup();

        let nfa_states: Box<[StateIdx]> = self.closure_result.as_slice().into();
        (nfa_states, is_match, is_match_at_end)
    }

    /// Check if `Match` is reachable from `idx` through epsilon
    /// transitions only (Split, Assert(End) with at_end=true).
    /// Used to compute `is_match_at_end`.
    fn can_reach_match(&self, start: StateIdx, states: &[State]) -> bool {
        // Small inline DFS.  We cannot reuse closure_stack/closure_visited
        // because we are called *during* epsilon_closure.
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
                State::Assert {
                    kind: AssertKind::End,
                    out,
                } => {
                    // Nested `$$`: follow through.
                    stack.push(out);
                }
                State::Assert {
                    kind: AssertKind::Start,
                    out,
                } => {
                    // `$^` — start after end.  Follow (unusual but correct).
                    // at_start is false here so this won't match in practice,
                    // but for reachability analysis we follow it.
                    stack.push(out);
                }
                State::Assert {
                    kind: AssertKind::StartLF,
                    out,
                } => {
                    stack.push(out);
                }
                _ => {} // Consuming states block the path.
            }
        }
        false
    }

    /// Compute the DFA transition for `(from_state, byte)`.
    ///
    /// On cache hit, returns the cached target state.  On cache miss,
    /// computes the epsilon closure of all NFA states reachable by
    /// consuming `byte` from `from_state`, interns the result, and
    /// caches the transition.
    fn transition(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> DfaStateId {
        // Cache lookup.
        if let Some(&to) = self.transitions.get(&(from, byte)) {
            return to;
        }

        // Cache miss: compute the transition.
        let to = self.populate(from, byte, regex);

        // Cache the result (LRU eviction handles capacity).
        let _ = self.transitions.put((from, byte), to);

        to
    }

    /// On cache miss: compute the next DFA state by advancing all NFA
    /// states in `from` by `byte`, then epsilon-closing the results.
    fn populate(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> DfaStateId {
        // Even DEAD states need re-seeding for unanchored search.
        let mut targets: Vec<StateIdx> = Vec::new();

        if from != DfaStateId::DEAD {
            // Collect NFA targets by consuming `byte` from each NFA state.
            // We must clone the nfa_states slice because `self` is borrowed
            // mutably by epsilon_closure.
            let nfa_states = self.states[from.idx()].nfa_states.clone();

            for &idx in nfa_states.iter() {
                let target = match regex.states[idx] {
                    State::Byte { byte: b2, out } if byte == b2 => Some(out),
                    State::ByteClass { class, out } if regex.classes[class][byte] => Some(out),
                    State::ByteTable { table } => {
                        let t = regex.byte_tables[table][byte];
                        if t != StateIdx::NONE {
                            Some(t)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(t) = target {
                    targets.push(t);
                }
            }
        }

        if targets.is_empty() {
            // No NFA state consumed this byte.  But we still need to
            // account for re-seeding (unanchored search).
            // Re-seed: epsilon closure from regex.start with at_start=false.
            let (nfa_set, is_match, is_match_at_end) = self.epsilon_closure(
                std::iter::once(regex.start),
                &regex.states,
                false,
                Some(byte),
            );
            if nfa_set.is_empty() && !is_match && !is_match_at_end {
                return DfaStateId::DEAD;
            }
            return self.intern_state(nfa_set, is_match, is_match_at_end);
        }

        // Epsilon closure of the targets + re-seed from start.
        // Re-seed is lower priority (appended after targets).
        let seeds = targets.into_iter().chain(std::iter::once(regex.start));
        let (nfa_set, is_match, is_match_at_end) =
            self.epsilon_closure(seeds, &regex.states, false, Some(byte));

        if nfa_set.is_empty() && !is_match && !is_match_at_end {
            return DfaStateId::DEAD;
        }

        self.intern_state(nfa_set, is_match, is_match_at_end)
    }

    /// Reset the cache for reuse with a new regex (keeps allocated memory).
    fn clear(&mut self, num_nfa_states: usize) {
        self.states.clear();
        self.state_map.clear();
        self.transitions.clear();
        self.closure_visited.clear();
        self.closure_visited.resize(num_nfa_states, false);
        self.regex_id = 0;
        self.start_id = DfaStateId::DEAD;
        self.start_is_match = false;
    }

    /// Prepare the cache for use with `regex`.
    ///
    /// If the cache already holds data for the same regex (identified by
    /// pointer identity of its states slice), this is a no-op and the
    /// previously interned DFA states and transitions are reused.
    ///
    /// If the regex is different, the cache is cleared and the start
    /// state is computed fresh.
    pub(crate) fn prepare(&mut self, regex: &Regex) {
        let id = regex.id;
        if self.regex_id == id && self.start_id != DfaStateId::DEAD {
            // Cache is valid for this regex — reuse everything.
            return;
        }

        // Different regex (or first use): clear and compute start state.
        self.clear(regex.states.len());
        self.regex_id = id;

        let (nfa_set, is_match, is_match_at_end) = self.epsilon_closure(
            std::iter::once(regex.start),
            &regex.states,
            true, // at_start = true (position 0, ^ passes)
            None, // no prev byte
        );
        self.start_id = self.intern_state(nfa_set, is_match, is_match_at_end);
        self.start_is_match = self.states[self.start_id.idx()].is_match;
    }
}

// ---------------------------------------------------------------------------
// DFA matcher
// ---------------------------------------------------------------------------

/// Lazy DFA matcher for Tier 1 (counter-free) patterns.
///
/// Wraps a `DfaCache` and drives the search loop.  The DFA is built
/// lazily: transitions are computed on first encounter and cached.
pub struct DfaMatcher<'a> {
    cache: &'a mut DfaCache,
    regex: &'a Regex,
    /// Current DFA state.
    current: DfaStateId,
    /// Whether a match has been found.
    ever_matched: bool,
}

impl<'a> DfaMatcher<'a> {
    /// Create a new DFA matcher.
    ///
    /// Assumes `cache.prepare(regex)` has already been called.
    pub(crate) fn new(cache: &'a mut DfaCache, regex: &'a Regex) -> Self {
        DfaMatcher {
            current: cache.start_id,
            ever_matched: cache.start_is_match,
            cache,
            regex,
        }
    }

    /// Advance the DFA by one input byte.
    #[inline(always)]
    pub fn step(&mut self, byte: u8) {
        self.current = self.cache.transition(self.current, byte, self.regex);
        if self.current != DfaStateId::DEAD && self.cache.states[self.current.idx()].is_match {
            self.ever_matched = true;
        }
    }

    /// Feed a byte slice, stop early on match.
    pub fn chunk(&mut self, input: &[u8]) {
        for &b in input {
            if self.ever_matched {
                return;
            }
            self.step(b);
        }
    }

    /// Signal end-of-input and return match result.
    pub fn finish(self) -> bool {
        if self.ever_matched {
            return true;
        }
        // Check if the current DFA state can match at end-of-input
        // (`$` assertion fires).
        if self.current != DfaStateId::DEAD {
            return self.cache.states[self.current.idx()].is_match_at_end;
        }
        false
    }

    /// Check whether a match has been found so far.
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
