//! Lazy DFA (Tier 1 and Tier 3).
//!
//! **Tier 1** (counter-free patterns): standard lazy subset construction.
//! DFA-eligible patterns have no bounded repetition counters and no complex
//! assertions (`\b`, `\B`, `EndLF`, `EndCRLF`, `StartCRLF`).  Simple
//! assertions (`^`, `$`, `StartLF`) are handled natively.
//!
//! **Tier 3** (counted repetitions): DFA + explicit counter contexts.
//! The DFA state (set of NFA consuming states) is separated from the
//! counter state (a set of `CounterCtx` values).  DFA transitions are
//! cached normally; each transition also stores a compiled *counter
//! program* that describes how to update counter values.  On a cache
//! hit the program is replayed against each active counter context.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;

use clru::CLruCache;

use crate::{AssertKind, CounterCtx, CounterIdx, CounterPool, Regex, State, StateIdx};

// ---------------------------------------------------------------------------
// DFA state table (shared by Tier 1 and Tier 3)
// ---------------------------------------------------------------------------

/// Index into the DFA state table ([`DfaCache::states`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DfaStateId(u32);

impl DfaStateId {
    /// Sentinel: the "dead" state (no NFA states, no match possible).
    pub(crate) const DEAD: Self = Self(u32::MAX);

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
// DFA cache (Tier 1)
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
/// `Regex`.  A unique regex ID detects when a different regex is used
/// and clears the cache.
pub(crate) struct DfaCache {
    /// Append-only table of DFA states.  Index = `DfaStateId`.
    states: Vec<DfaState>,
    /// Reverse lookup: canonical `DfaState → DfaStateId`.
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
    fn intern_state(
        &mut self,
        nfa_states: Box<[StateIdx]>,
        is_match: bool,
        is_match_at_end: bool,
    ) -> DfaStateId {
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
    /// Follows `Split` and `Assert(Start)`/`Assert(End)`/`Assert(StartLF)`.
    /// Counter states cause a debug panic (Tier 1 only).
    fn epsilon_closure(
        &mut self,
        seeds: impl Iterator<Item = StateIdx>,
        states: &[State],
        at_start: bool,
        prev_byte: Option<u8>,
    ) -> (Box<[StateIdx]>, bool, bool) {
        self.closure_stack.clear();
        self.closure_result.clear();
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
                State::Assert { kind, out } => match kind {
                    AssertKind::Start => {
                        if at_start {
                            self.closure_stack.push(out);
                        }
                    }
                    AssertKind::End => {
                        if self.can_reach_match(out, states) {
                            is_match_at_end = true;
                        }
                    }
                    AssertKind::StartLF => {
                        if at_start || prev_byte == Some(b'\n') {
                            self.closure_stack.push(out);
                        }
                    }
                    _ => {
                        debug_assert!(
                            false,
                            "complex assertion in DFA-eligible pattern: {:?}",
                            kind
                        );
                    }
                },
                State::Match => {
                    is_match = true;
                }
                State::Byte { .. } | State::ByteClass { .. } | State::ByteTable { .. } => {
                    self.closure_result.push(idx);
                }
                State::CounterInstance { .. } | State::CounterIncrement { .. } => {
                    debug_assert!(false, "counter state in DFA-eligible pattern");
                }
            }
        }

        self.closure_result.sort_unstable_by_key(|s| s.0);
        self.closure_result.dedup();

        let nfa_states: Box<[StateIdx]> = self.closure_result.as_slice().into();
        (nfa_states, is_match, is_match_at_end)
    }

    /// Check if `Match` is reachable from `idx` through epsilon transitions.
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
                State::Assert {
                    kind: AssertKind::End | AssertKind::Start | AssertKind::StartLF,
                    out,
                } => {
                    stack.push(out);
                }
                _ => {}
            }
        }
        false
    }

    /// Compute the DFA transition for `(from_state, byte)`.
    fn transition(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> DfaStateId {
        if let Some(&to) = self.transitions.get(&(from, byte)) {
            return to;
        }
        let to = self.populate(from, byte, regex);
        let _ = self.transitions.put((from, byte), to);
        to
    }

    /// On cache miss: compute the next DFA state.
    fn populate(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> DfaStateId {
        let mut targets: Vec<StateIdx> = Vec::new();

        if from != DfaStateId::DEAD {
            let nfa_states = self.states[from.idx()].nfa_states.clone();
            for &idx in nfa_states.iter() {
                let target = match regex.states[idx] {
                    State::Byte { byte: b2, out } if byte == b2 => Some(out),
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

        let seeds = targets.into_iter().chain(std::iter::once(regex.start));
        let (nfa_set, is_match, is_match_at_end) =
            self.epsilon_closure(seeds, &regex.states, false, Some(byte));

        if nfa_set.is_empty() && !is_match && !is_match_at_end {
            return DfaStateId::DEAD;
        }

        self.intern_state(nfa_set, is_match, is_match_at_end)
    }

    /// Reset the cache for reuse with a new regex.
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
    pub(crate) fn prepare(&mut self, regex: &Regex) {
        let id = regex.id;
        if self.regex_id == id && self.start_id != DfaStateId::DEAD {
            return;
        }
        self.clear(regex.states.len());
        self.regex_id = id;

        let (nfa_set, is_match, is_match_at_end) =
            self.epsilon_closure(std::iter::once(regex.start), &regex.states, true, None);
        self.start_id = self.intern_state(nfa_set, is_match, is_match_at_end);
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
}

impl<'a> DfaMatcher<'a> {
    pub(crate) fn new(cache: &'a mut DfaCache, regex: &'a Regex) -> Self {
        DfaMatcher {
            current: cache.start_id,
            ever_matched: cache.start_is_match,
            cache,
            regex,
        }
    }

    #[inline(always)]
    pub fn step(&mut self, byte: u8) {
        self.current = self.cache.transition(self.current, byte, self.regex);
        if self.current != DfaStateId::DEAD && self.cache.states[self.current.idx()].is_match {
            self.ever_matched = true;
        }
    }

    pub fn chunk(&mut self, input: &[u8]) {
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
            return self.cache.states[self.current.idx()].is_match_at_end;
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

// ===========================================================================
// Tier 3: Counting DFA — DFA + explicit counter contexts
// ===========================================================================

// ---------------------------------------------------------------------------
// Counter programs
// ---------------------------------------------------------------------------

/// Index into [`CountingDfaCache::programs`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CounterProgramIdx(u32);

/// A single node in the counter program tree.
///
/// Programs are compiled once per DFA transition (on cache miss) by
/// tracing the NFA epsilon closure and recording counter operations
/// instead of executing them.  On cache hits the compiled program is
/// replayed against each active counter context.
#[derive(Debug)]
enum CounterOp {
    /// Initialize counter `counter` to 0 (entering a repetition body).
    /// Then continue with `then`.
    Init {
        counter: CounterIdx,
        then: Vec<CounterOp>,
    },
    /// Increment counter `counter`.  Branch based on the new value:
    /// - `on_continue` is taken when `new_value < max` (stay in loop).
    /// - `on_break` is taken when `new_value >= min` (may exit loop).
    ///
    /// Both branches can fire when `min <= new_value < max`.
    Increment {
        counter: CounterIdx,
        min: usize,
        max: usize,
        on_continue: Vec<CounterOp>,
        on_break: Vec<CounterOp>,
    },
    /// Deactivate counter `counter` (exiting a repetition).
    /// Then continue with `then`.
    Remove {
        counter: CounterIdx,
        then: Vec<CounterOp>,
    },
    /// Emit the current context as a surviving thread.
    /// `origin` is the NFA consuming state the context will be "at".
    EmitContinue { origin: StateIdx },
    /// Signal that a match was found.
    EmitMatch,
    /// Signal that a match is reachable at end-of-input (`$` gate).
    EmitMatchAtEnd,
}

/// Outcome of applying a counter program to one context.
enum ProgramResult {
    /// The context survives with updated counter values at the given
    /// NFA consuming state.
    Continue(CounterCtx, StateIdx),
    /// A match was found (directly reachable).
    Match,
    /// A match is reachable at end-of-input (`$`).
    MatchAtEnd,
}

/// Execute a counter program against a single context.
///
/// The program tree is walked depth-first.  At each `Increment` node the
/// counter value determines which branches fire.  `EmitContinue` clones
/// the (possibly modified) context into the output.  `EmitMatch` records
/// that a match was found.
///
/// `ctx` is borrowed; the function clones it as needed.  When only one
/// branch of an `Increment` fires, the clone is avoided by reusing the
/// parent context directly.
fn execute_program(
    ops: &[CounterOp],
    ctx: &CounterCtx,
    pool: &mut CounterPool,
    results: &mut Vec<ProgramResult>,
) {
    for op in ops {
        match op {
            CounterOp::EmitContinue { origin } => {
                results.push(ProgramResult::Continue(ctx.clone(pool), *origin));
            }
            CounterOp::EmitMatch => {
                results.push(ProgramResult::Match);
            }
            CounterOp::EmitMatchAtEnd => {
                results.push(ProgramResult::MatchAtEnd);
            }
            CounterOp::Init { counter, then } => {
                let mut child = ctx.clone(pool);
                child.set(*counter, 0, pool);
                execute_program(then, &child, pool, results);
                pool.free(child.into_range());
            }
            CounterOp::Remove { counter, then } => {
                let mut child = ctx.clone(pool);
                child.remove(*counter, pool);
                execute_program(then, &child, pool, results);
                pool.free(child.into_range());
            }
            CounterOp::Increment {
                counter,
                min,
                max,
                on_continue,
                on_break,
            } => {
                // Skip this Increment if the counter is not active in
                // this context.  This happens when a DFA state merges
                // NFA states from different counter loops — a context
                // in loop A shouldn't process Increment ops from loop B.
                let cur = match ctx.get(*counter, pool) {
                    Some(v) => v,
                    None => continue,
                };
                let new_val = cur + 1;
                let do_continue = new_val < *max;
                let do_break = new_val >= *min;
                if do_continue && do_break {
                    // Both branches fire — need two clones.
                    let mut child_c = ctx.clone(pool);
                    child_c.set(*counter, new_val, pool);
                    execute_program(on_continue, &child_c, pool, results);
                    pool.free(child_c.into_range());
                    let mut child_b = ctx.clone(pool);
                    child_b.set(*counter, new_val, pool);
                    execute_program(on_break, &child_b, pool, results);
                    pool.free(child_b.into_range());
                } else if do_continue {
                    // Only continue — fast-path when sub-program is a
                    // single EmitContinue: avoid the intermediate clone by
                    // building the result context directly.
                    if let [CounterOp::EmitContinue { origin }] = on_continue.as_slice() {
                        let mut result = ctx.clone(pool);
                        result.set(*counter, new_val, pool);
                        results.push(ProgramResult::Continue(result, *origin));
                    } else {
                        let mut child = ctx.clone(pool);
                        child.set(*counter, new_val, pool);
                        execute_program(on_continue, &child, pool, results);
                        pool.free(child.into_range());
                    }
                } else if do_break {
                    // Only break — same fast-path for single EmitContinue.
                    if let [CounterOp::EmitContinue { origin }] = on_break.as_slice() {
                        let mut result = ctx.clone(pool);
                        result.set(*counter, new_val, pool);
                        results.push(ProgramResult::Continue(result, *origin));
                    } else if let [CounterOp::EmitMatch] = on_break.as_slice() {
                        results.push(ProgramResult::Match);
                    } else if let [CounterOp::EmitMatchAtEnd] = on_break.as_slice() {
                        results.push(ProgramResult::MatchAtEnd);
                    } else {
                        let mut child = ctx.clone(pool);
                        child.set(*counter, new_val, pool);
                        execute_program(on_break, &child, pool, results);
                        pool.free(child.into_range());
                    }
                }
                // Neither fires: context dies silently.
            }
        }
    }
}

/// Fast-path execution for the common case of a single `Increment` where
/// only one branch fires and that branch is a single leaf op.
///
/// Mutates `ctx` in-place (avoiding allocation) and returns the single
/// result.  Returns `None` if the program doesn't match this shape, in
/// which case the caller falls back to `execute_program`.
#[inline]
fn try_execute_inplace(
    prog: &[CounterOp],
    ctx: &mut CounterCtx,
    pool: &mut CounterPool,
) -> Option<ProgramResult> {
    if let [
        CounterOp::Increment {
            counter,
            min,
            max,
            on_continue,
            on_break,
        },
    ] = prog
    {
        let cur = ctx.get(*counter, pool)?;
        let new_val = cur + 1;
        let do_continue = new_val < *max;
        let do_break = new_val >= *min;
        if do_continue && !do_break {
            // Only continue fires.
            if let [CounterOp::EmitContinue { origin }] = on_continue.as_slice() {
                ctx.set(*counter, new_val, pool);
                return Some(ProgramResult::Continue(CounterCtx::new(), *origin));
                // ^ dummy ctx — caller will use the mutated `ctx` directly
            }
        } else if do_break && !do_continue {
            // Only break fires.
            if let [
                CounterOp::Remove {
                    then: break_then, ..
                },
            ] = on_break.as_slice()
            {
                if let [CounterOp::EmitMatch] = break_then.as_slice() {
                    return Some(ProgramResult::Match);
                }
                if let [CounterOp::EmitMatchAtEnd] = break_then.as_slice() {
                    return Some(ProgramResult::MatchAtEnd);
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Counting DFA cache (Tier 3)
// ---------------------------------------------------------------------------

/// Index into [`CountingDfaCache::origin_program_tables`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OriginTableIdx(u32);

impl OriginTableIdx {
    /// Sentinel value indicating an unpopulated transition slot.
    const NONE: Self = Self(u32::MAX);
}

/// Cached transition for the counting DFA.  In addition to the next DFA
/// state, stores per-origin counter programs and a seed program.
#[derive(Clone, Copy, Debug)]
struct CountingTransition {
    next: DfaStateId,
    /// Index into origin_program_tables: for each NFA consuming state in
    /// the `from` DFA state, the counter program to apply to contexts at
    /// that origin.
    origin_table: OriginTableIdx,
    /// Counter program for the re-seed context (from regex.start).
    seed_program: CounterProgramIdx,
}

impl CountingTransition {
    /// Sentinel value for an unpopulated transition slot.
    const UNPOPULATED: Self = Self {
        next: DfaStateId::DEAD,
        origin_table: OriginTableIdx::NONE,
        seed_program: CounterProgramIdx(u32::MAX),
    };

    /// True if this slot has not been populated yet.
    #[inline]
    fn is_unpopulated(self) -> bool {
        self.origin_table.0 == u32::MAX
    }
}

/// Lazy DFA cache for Tier 3 (patterns with counters).
///
/// Shares the `DfaState` representation with Tier 1 — the DFA state is
/// the set of NFA consuming states, ignoring counter values.  Counter
/// values live in a separate side-channel of `CounterCtx` entries.
pub(crate) struct CountingDfaCache {
    /// Append-only table of DFA states.
    states: Vec<DfaState>,
    /// Reverse lookup: canonical `DfaState → DfaStateId`.
    state_map: HashMap<DfaState, DfaStateId>,
    /// Flat transition table: indexed by `state.0 * 256 + byte`.
    /// Unpopulated slots have `origin_table == OriginTableIdx::NONE`.
    transitions: Vec<CountingTransition>,
    /// Compiled counter programs (append-only).
    programs: Vec<Vec<CounterOp>>,
    /// Per-origin program tables (append-only).
    /// Each entry maps NFA consuming state → counter program index.
    origin_program_tables: Vec<Vec<(StateIdx, CounterProgramIdx)>>,
    /// Precomputed seed origin tables for pure-emit seed programs.
    /// Indexed by `CounterProgramIdx` (same as `programs`).  If the
    /// seed program at index `i` consists only of `EmitContinue` ops,
    /// `seed_emit_origins[i]` is `Some(origins)`.  Otherwise `None`.
    seed_emit_origins: Vec<Option<Box<[StateIdx]>>>,
    /// Scratch visited set for epsilon closure.
    closure_visited: Vec<bool>,
    /// Regex identity for cache reuse.
    regex_id: u64,
    /// Start DFA state (at_start=true, position 0).
    start_id: DfaStateId,
    /// Start counter program (applied to the initial seed context).
    start_program: CounterProgramIdx,
    /// Whether the start state can match directly.
    start_is_match: bool,
    /// Whether the start state can match at end-of-input.
    start_is_match_at_end: bool,
}

impl fmt::Debug for CountingDfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingDfaCache")
            .field("num_states", &self.states.len())
            .field("num_programs", &self.programs.len())
            .finish()
    }
}

impl CountingDfaCache {
    pub(crate) fn new(num_nfa_states: usize) -> Self {
        Self {
            states: Vec::new(),
            state_map: HashMap::new(),
            transitions: Vec::new(),
            programs: Vec::new(),
            origin_program_tables: Vec::new(),
            seed_emit_origins: Vec::new(),
            closure_visited: vec![false; num_nfa_states],
            regex_id: 0,
            start_id: DfaStateId::DEAD,
            start_program: CounterProgramIdx(0),
            start_is_match: false,
            start_is_match_at_end: false,
        }
    }

    /// Look up or insert a DFA state.
    fn intern_state(
        &mut self,
        nfa_states: Box<[StateIdx]>,
        is_match: bool,
        is_match_at_end: bool,
    ) -> DfaStateId {
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
        self.transitions
            .resize(self.states.len() * 256, CountingTransition::UNPOPULATED);
        id
    }

    /// Store a counter program and return its index.
    ///
    /// Also precomputes whether the program is pure `EmitContinue` ops
    /// (useful for fast-path seed execution).
    fn intern_program(&mut self, ops: Vec<CounterOp>) -> CounterProgramIdx {
        let idx = CounterProgramIdx(self.programs.len() as u32);
        // Check if the program is pure EmitContinue (no counter ops).
        let pure_origins: Option<Box<[StateIdx]>> = {
            let mut origins = Vec::new();
            let mut pure = true;
            for op in &ops {
                match op {
                    CounterOp::EmitContinue { origin } => origins.push(*origin),
                    _ => {
                        pure = false;
                        break;
                    }
                }
            }
            if pure {
                Some(origins.into_boxed_slice())
            } else {
                None
            }
        };
        self.seed_emit_origins.push(pure_origins);
        self.programs.push(ops);
        idx
    }

    /// Store a per-origin program table and return its index.
    fn intern_origin_table(&mut self, table: Vec<(StateIdx, CounterProgramIdx)>) -> OriginTableIdx {
        let idx = OriginTableIdx(self.origin_program_tables.len() as u32);
        self.origin_program_tables.push(table);
        idx
    }

    /// Epsilon closure that records counter operations into a program tree.
    ///
    /// Unlike Tier 1's `epsilon_closure` which uses an iterative stack,
    /// this uses a recursive DFS because the counter program structure
    /// mirrors the recursion tree (Init → body → Increment with branches).
    ///
    /// Returns: `(sorted NFA consuming states, is_match, is_match_at_end, program ops)`.
    fn epsilon_closure_with_program(
        &mut self,
        seeds: impl Iterator<Item = StateIdx>,
        states: &[State],
        at_start: bool,
        prev_byte: Option<u8>,
    ) -> (Box<[StateIdx]>, bool, bool, Vec<CounterOp>) {
        // Reset visited.
        for v in self.closure_visited.iter_mut() {
            *v = false;
        }

        let mut nfa_result: Vec<StateIdx> = Vec::new();
        let mut is_match = false;
        let mut is_match_at_end = false;
        let mut ops = Vec::new();

        let seeds: Vec<StateIdx> = seeds.collect();
        for seed in seeds {
            Self::trace_epsilon(
                seed,
                states,
                at_start,
                prev_byte,
                &mut self.closure_visited,
                &mut nfa_result,
                &mut is_match,
                &mut is_match_at_end,
                &mut ops,
            );
        }

        nfa_result.sort_unstable_by_key(|s| s.0);
        nfa_result.dedup();

        let nfa_states: Box<[StateIdx]> = nfa_result.into_boxed_slice();
        (nfa_states, is_match, is_match_at_end, ops)
    }

    /// Recursive trace through epsilon states, recording counter ops.
    ///
    /// Uses **stack-based cycle detection**: a state is marked visited on
    /// entry and unmarked on exit.  This prevents infinite loops on
    /// epsilon-only cycles (e.g. `(a?){2}` where `CInc.continue → Split
    /// → CInc`) while allowing sibling branches of the counter program
    /// tree to re-traverse shared states.  The `nfa_result` vec may
    /// contain duplicates; the caller deduplicates it.
    #[allow(clippy::too_many_arguments)]
    fn trace_epsilon(
        idx: StateIdx,
        states: &[State],
        at_start: bool,
        prev_byte: Option<u8>,
        on_stack: &mut [bool],
        nfa_result: &mut Vec<StateIdx>,
        is_match: &mut bool,
        is_match_at_end: &mut bool,
        ops: &mut Vec<CounterOp>,
    ) {
        let i = idx.idx();
        if on_stack[i] {
            return;
        }
        on_stack[i] = true;

        match states[idx] {
            State::Split { out, out1 } => {
                Self::trace_epsilon(
                    out,
                    states,
                    at_start,
                    prev_byte,
                    on_stack,
                    nfa_result,
                    is_match,
                    is_match_at_end,
                    ops,
                );
                Self::trace_epsilon(
                    out1,
                    states,
                    at_start,
                    prev_byte,
                    on_stack,
                    nfa_result,
                    is_match,
                    is_match_at_end,
                    ops,
                );
            }
            State::Assert { kind, out } => match kind {
                AssertKind::Start => {
                    if at_start {
                        Self::trace_epsilon(
                            out,
                            states,
                            at_start,
                            prev_byte,
                            on_stack,
                            nfa_result,
                            is_match,
                            is_match_at_end,
                            ops,
                        );
                    }
                }
                AssertKind::End => {
                    // Check if Match is reachable through this `$` gate.
                    let mut sub_match = false;
                    let mut sub_match_at_end = false;
                    let mut sub_ops = Vec::new();
                    Self::trace_epsilon_for_end(
                        out,
                        states,
                        &mut sub_match,
                        &mut sub_match_at_end,
                        &mut sub_ops,
                    );
                    if sub_match {
                        *is_match_at_end = true;
                        // Wrap the sub-program ops so they emit
                        // MatchAtEnd instead of Match.
                        rewrite_match_to_match_at_end(&mut sub_ops);
                        ops.extend(sub_ops);
                    }
                }
                AssertKind::StartLF => {
                    if at_start || prev_byte == Some(b'\n') {
                        Self::trace_epsilon(
                            out,
                            states,
                            at_start,
                            prev_byte,
                            on_stack,
                            nfa_result,
                            is_match,
                            is_match_at_end,
                            ops,
                        );
                    }
                }
                _ => {
                    debug_assert!(
                        false,
                        "complex assertion in counting-DFA pattern: {:?}",
                        kind
                    );
                }
            },
            State::CounterInstance { counter, out } => {
                let mut sub_ops = Vec::new();
                Self::trace_epsilon(
                    out,
                    states,
                    at_start,
                    prev_byte,
                    on_stack,
                    nfa_result,
                    is_match,
                    is_match_at_end,
                    &mut sub_ops,
                );
                ops.push(CounterOp::Init {
                    counter,
                    then: sub_ops,
                });
            }
            State::CounterIncrement {
                counter,
                out,
                out1,
                min,
                max,
            } => {
                // Continue path (re-enter body).
                let mut cont_ops = Vec::new();
                Self::trace_epsilon(
                    out,
                    states,
                    at_start,
                    prev_byte,
                    on_stack,
                    nfa_result,
                    is_match,
                    is_match_at_end,
                    &mut cont_ops,
                );
                // Break path (exit repetition).
                // Wrap in Remove(counter) so that the counter is
                // deactivated when the context exits this loop.  This
                // prevents stale counter values from triggering
                // Increment ops from other positions in the DFA state.
                let mut raw_break_ops = Vec::new();
                Self::trace_epsilon(
                    out1,
                    states,
                    at_start,
                    prev_byte,
                    on_stack,
                    nfa_result,
                    is_match,
                    is_match_at_end,
                    &mut raw_break_ops,
                );
                let break_ops = vec![CounterOp::Remove {
                    counter,
                    then: raw_break_ops,
                }];
                ops.push(CounterOp::Increment {
                    counter,
                    min,
                    max,
                    on_continue: cont_ops,
                    on_break: break_ops,
                });
            }
            State::Match => {
                *is_match = true;
                ops.push(CounterOp::EmitMatch);
            }
            State::Byte { .. } | State::ByteClass { .. } | State::ByteTable { .. } => {
                nfa_result.push(idx);
                ops.push(CounterOp::EmitContinue { origin: idx });
            }
        }

        on_stack[i] = false;
    }

    /// Trace epsilon transitions for the `$` (End) path.
    ///
    /// This is a separate function because the `$` gate's sub-graph may
    /// contain counter operations that should only fire at end-of-input.
    /// We don't use the main `visited` set because these states may also
    /// appear in the main closure.
    #[allow(clippy::only_used_in_recursion)]
    fn trace_epsilon_for_end(
        idx: StateIdx,
        states: &[State],
        is_match: &mut bool,
        is_match_at_end: &mut bool,
        ops: &mut Vec<CounterOp>,
    ) {
        match states[idx] {
            State::Match => {
                *is_match = true;
                ops.push(CounterOp::EmitMatch);
            }
            State::Split { out, out1 } => {
                Self::trace_epsilon_for_end(out, states, is_match, is_match_at_end, ops);
                Self::trace_epsilon_for_end(out1, states, is_match, is_match_at_end, ops);
            }
            State::Assert {
                kind: AssertKind::End | AssertKind::Start | AssertKind::StartLF,
                out,
            } => {
                Self::trace_epsilon_for_end(out, states, is_match, is_match_at_end, ops);
            }
            State::CounterInstance { counter, out } => {
                let mut sub_ops = Vec::new();
                Self::trace_epsilon_for_end(out, states, is_match, is_match_at_end, &mut sub_ops);
                ops.push(CounterOp::Init {
                    counter,
                    then: sub_ops,
                });
            }
            State::CounterIncrement {
                counter,
                out,
                out1,
                min,
                max,
            } => {
                let mut cont_ops = Vec::new();
                Self::trace_epsilon_for_end(out, states, is_match, is_match_at_end, &mut cont_ops);
                let mut raw_break_ops = Vec::new();
                Self::trace_epsilon_for_end(
                    out1,
                    states,
                    is_match,
                    is_match_at_end,
                    &mut raw_break_ops,
                );
                let break_ops = vec![CounterOp::Remove {
                    counter,
                    then: raw_break_ops,
                }];
                ops.push(CounterOp::Increment {
                    counter,
                    min,
                    max,
                    on_continue: cont_ops,
                    on_break: break_ops,
                });
            }
            _ => {} // Consuming states block the path at end-of-input.
        }
    }

    /// Compute a counting transition for `(from_state, byte)`.
    #[inline]
    fn transition(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> CountingTransition {
        if from == DfaStateId::DEAD {
            let t = self.populate(from, byte, regex);
            return t;
        }
        let slot = from.0 as usize * 256 + byte as usize;
        let t = self.transitions[slot];
        if !t.is_unpopulated() {
            return t;
        }
        let t = self.populate(from, byte, regex);
        self.transitions[slot] = t;
        t
    }

    /// On cache miss: compute the next DFA state and per-origin counter
    /// programs.
    ///
    /// For each NFA consuming state in the `from` DFA state, we compute
    /// the NFA target when consuming `byte`, then run a separate epsilon
    /// closure to get the counter program for that origin.  This ensures
    /// that contexts at different NFA positions receive only their own
    /// counter ops.
    fn populate(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> CountingTransition {
        // Collect (origin_nfa_state, nfa_target) pairs.
        let mut origin_targets: Vec<(StateIdx, StateIdx)> = Vec::new();

        if from != DfaStateId::DEAD {
            let nfa_states = self.states[from.idx()].nfa_states.clone();
            for &idx in nfa_states.iter() {
                let target = match regex.states[idx] {
                    State::Byte { byte: b2, out } if byte == b2 => Some(out),
                    State::ByteClass { class, out } if regex.classes[class][byte] => Some(out),
                    State::ByteTable { table } => {
                        let t = regex.byte_tables[table][byte];
                        if t != StateIdx::NONE { Some(t) } else { None }
                    }
                    _ => None,
                };
                if let Some(t) = target {
                    origin_targets.push((idx, t));
                }
            }
        }

        let mut merged: Vec<StateIdx> = Vec::new();
        let mut is_match = false;
        let mut is_match_at_end = false;
        let mut origin_table: Vec<(StateIdx, CounterProgramIdx)> = Vec::new();

        // Per-origin closures.
        for (origin, target) in &origin_targets {
            let (nfa, m, mae, ops) = self.epsilon_closure_with_program(
                std::iter::once(*target),
                &regex.states,
                false,
                Some(byte),
            );
            is_match = is_match || m;
            is_match_at_end = is_match_at_end || mae;
            merged.extend_from_slice(&nfa);
            let prog_idx = self.intern_program(ops);
            origin_table.push((*origin, prog_idx));
        }

        // Seed closure (from regex.start, at_start=false for re-seed).
        let (seed_nfa, seed_match, seed_match_at_end, seed_ops) = self
            .epsilon_closure_with_program(
                std::iter::once(regex.start),
                &regex.states,
                false,
                Some(byte),
            );
        let seed_program = self.intern_program(seed_ops);
        is_match = is_match || seed_match;
        is_match_at_end = is_match_at_end || seed_match_at_end;
        merged.extend_from_slice(&seed_nfa);

        // Dedup merged NFA states.
        merged.sort_unstable_by_key(|s| s.0);
        merged.dedup();

        let origin_table_idx = self.intern_origin_table(origin_table);

        if merged.is_empty() && !is_match && !is_match_at_end {
            return CountingTransition {
                next: DfaStateId::DEAD,
                origin_table: origin_table_idx,
                seed_program,
            };
        }
        let next = self.intern_state(merged.into_boxed_slice(), is_match, is_match_at_end);
        CountingTransition {
            next,
            origin_table: origin_table_idx,
            seed_program,
        }
    }

    /// Reset the cache.
    fn clear(&mut self, num_nfa_states: usize) {
        self.states.clear();
        self.state_map.clear();
        self.transitions.clear();
        self.programs.clear();
        self.origin_program_tables.clear();
        self.seed_emit_origins.clear();
        self.closure_visited.clear();
        self.closure_visited.resize(num_nfa_states, false);
        self.regex_id = 0;
        self.start_id = DfaStateId::DEAD;
        self.start_program = CounterProgramIdx(0);
        self.start_is_match = false;
        self.start_is_match_at_end = false;
    }

    /// Prepare the cache for `regex`.
    pub(crate) fn prepare(&mut self, regex: &Regex) {
        let id = regex.id;
        if self.regex_id == id && self.start_id != DfaStateId::DEAD {
            return;
        }
        self.clear(regex.states.len());
        self.regex_id = id;

        let (nfa_set, is_match, is_match_at_end, ops) = self.epsilon_closure_with_program(
            std::iter::once(regex.start),
            &regex.states,
            true,
            None,
        );
        self.start_id = self.intern_state(nfa_set, is_match, is_match_at_end);
        self.start_program = self.intern_program(ops);
        self.start_is_match = is_match;
        self.start_is_match_at_end = is_match_at_end;
    }
}

/// Rewrite `EmitMatch` to `EmitMatchAtEnd` in a program tree.
fn rewrite_match_to_match_at_end(ops: &mut [CounterOp]) {
    for op in ops.iter_mut() {
        match op {
            CounterOp::EmitMatch => {
                *op = CounterOp::EmitMatchAtEnd;
            }
            CounterOp::Init { then, .. } | CounterOp::Remove { then, .. } => {
                rewrite_match_to_match_at_end(then);
            }
            CounterOp::Increment {
                on_continue,
                on_break,
                ..
            } => {
                rewrite_match_to_match_at_end(on_continue);
                rewrite_match_to_match_at_end(on_break);
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 3 matcher
// ---------------------------------------------------------------------------

/// Counting DFA matcher for patterns with bounded repetitions.
///
/// The DFA handles state transitions (cacheable); counter values live
/// in a side-channel of `CounterCtx` entries updated by compiled
/// counter programs.  Each context also tracks its NFA "origin"
/// (the consuming state it is waiting at) so that only the relevant
/// counter program is applied to it.
pub struct CountingDfaMatcher<'a> {
    cache: &'a mut CountingDfaCache,
    regex: &'a Regex,
    pool: &'a mut CounterPool,
    /// Current DFA state.
    current: DfaStateId,
    /// Active counter contexts, each paired with its NFA origin.
    /// Does NOT include implicit "seed" contexts — those are tracked
    /// separately via `seed_origins` to avoid redundant alloc/free.
    contexts: Vec<(CounterCtx, StateIdx)>,
    /// Scratch space for new contexts during step (avoids realloc).
    next_contexts: Vec<(CounterCtx, StateIdx)>,
    /// Scratch space for program execution results (avoids per-step alloc).
    results: Vec<ProgramResult>,
    /// Whether a match has been found.
    ever_matched: bool,
    /// Whether a match-at-end has been found (current step only).
    match_at_end: bool,
    /// When `Some`, the given set of NFA origins has implicit empty
    /// contexts that are not stored in `contexts`.  This avoids
    /// allocating and freeing empty contexts every step for unanchored
    /// patterns where the seed program is pure `EmitContinue`.
    /// The index points into `cache.seed_emit_origins`.
    seed_origins_idx: Option<CounterProgramIdx>,
}

impl<'a> CountingDfaMatcher<'a> {
    pub(crate) fn new(
        cache: &'a mut CountingDfaCache,
        regex: &'a Regex,
        pool: &'a mut CounterPool,
    ) -> Self {
        pool.clear();
        pool.num_counters = regex.num_counters;

        // Apply the start program to a fresh context to get initial contexts.
        let mut contexts = Vec::new();
        let mut ever_matched = false;
        let mut match_at_end = false;
        let mut seed_origins_idx = None;

        let start_idx = cache.start_program;
        if cache.seed_emit_origins[start_idx.0 as usize].is_some() {
            // Pure-emit start program: track seed origins implicitly.
            seed_origins_idx = Some(start_idx);
        } else {
            let fresh = CounterCtx::new();
            let start_prog = &cache.programs[start_idx.0 as usize];
            let mut results = Vec::new();
            execute_program(start_prog, &fresh, pool, &mut results);
            for r in results {
                match r {
                    ProgramResult::Continue(ctx, origin) => contexts.push((ctx, origin)),
                    ProgramResult::Match => ever_matched = true,
                    ProgramResult::MatchAtEnd => match_at_end = true,
                }
            }
            // Dedup initial contexts.
            dedup_contexts(&mut contexts, pool);
        }

        if cache.start_is_match {
            ever_matched = true;
        }
        if cache.start_is_match_at_end {
            match_at_end = true;
        }

        CountingDfaMatcher {
            current: cache.start_id,
            ever_matched,
            match_at_end,
            seed_origins_idx,
            cache,
            regex,
            pool,
            contexts,
            next_contexts: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Advance by one byte.
    #[inline]
    pub fn step(&mut self, byte: u8) {
        let trans = self.cache.transition(self.current, byte, self.regex);
        self.current = trans.next;

        let origin_table = &self.cache.origin_program_tables[trans.origin_table.0 as usize];

        // ---- Ultra-fast path: seed-only + empty origin table ----
        // When we only have implicit seed contexts and no NFA consuming
        // state consumed this byte, the step is a no-op: DFA state
        // advances, seed resets, done.  This is the overwhelmingly
        // common case for unanchored patterns on non-matching bytes.
        if self.seed_origins_idx.is_some() && self.contexts.is_empty() && origin_table.is_empty() {
            let seed_idx = trans.seed_program;
            if self.cache.seed_emit_origins[seed_idx.0 as usize].is_some() {
                self.seed_origins_idx = Some(seed_idx);
                self.match_at_end = false;
                return;
            }
        }

        // Reset match_at_end: it must reflect only the CURRENT step,
        // not be accumulated across steps.  The `$` assertion is only
        // meaningful at the actual end-of-input, evaluated in finish().
        self.match_at_end = false;

        self.next_contexts.clear();

        // --- Phase 1: Process implicit seed contexts ---
        // When seed_origins_idx is set, we have implicit empty contexts
        // at those origins.  Check if any of them survive this transition.
        if let Some(seed_prog_idx) = self.seed_origins_idx
            && !origin_table.is_empty()
        {
            let origins = self.cache.seed_emit_origins[seed_prog_idx.0 as usize]
                .as_ref()
                .unwrap();
            for &seed_origin in origins.iter() {
                // Look up the program for this seed origin.
                let prog_idx = origin_table
                    .iter()
                    .find(|(o, _)| *o == seed_origin)
                    .map(|(_, idx)| *idx);
                if let Some(prog_idx) = prog_idx {
                    let prog = &self.cache.programs[prog_idx.0 as usize];
                    // Seed contexts are empty — execute the program
                    // with a fresh context.
                    let fresh = CounterCtx::new();
                    self.results.clear();
                    execute_program(prog, &fresh, self.pool, &mut self.results);
                    for r in self.results.drain(..) {
                        match r {
                            ProgramResult::Continue(new_ctx, new_origin) => {
                                self.next_contexts.push((new_ctx, new_origin));
                            }
                            ProgramResult::Match => {
                                self.ever_matched = true;
                            }
                            ProgramResult::MatchAtEnd => {
                                self.match_at_end = true;
                            }
                        }
                    }
                }
                // If no program for this seed origin, it dies (no-op).
            }
            // If origin_table is empty, all implicit seed contexts die (no-op).
        }

        // --- Phase 2: Process explicit (non-seed) contexts ---
        if origin_table.is_empty() {
            // All contexts die — free them in bulk.
            for (ctx, _) in self.contexts.drain(..) {
                self.pool.free(ctx.into_range());
            }
        } else {
            for (mut ctx, ctx_origin) in self.contexts.drain(..) {
                // Find the program for this context's origin NFA state.
                let prog_idx = if origin_table.len() == 1 {
                    let (origin, idx) = origin_table[0];
                    if origin == ctx_origin {
                        Some(idx)
                    } else {
                        None
                    }
                } else {
                    origin_table
                        .iter()
                        .find(|(origin, _)| *origin == ctx_origin)
                        .map(|(_, idx)| *idx)
                };
                if let Some(prog_idx) = prog_idx {
                    let prog = &self.cache.programs[prog_idx.0 as usize];
                    // Fast path: single Increment with one active branch.
                    if let Some(result) = try_execute_inplace(prog, &mut ctx, self.pool) {
                        match result {
                            ProgramResult::Continue(_, new_origin) => {
                                self.next_contexts.push((ctx, new_origin));
                                continue;
                            }
                            ProgramResult::Match => {
                                self.ever_matched = true;
                            }
                            ProgramResult::MatchAtEnd => {
                                self.match_at_end = true;
                            }
                        }
                    } else {
                        self.results.clear();
                        execute_program(prog, &ctx, self.pool, &mut self.results);
                        for r in self.results.drain(..) {
                            match r {
                                ProgramResult::Continue(new_ctx, new_origin) => {
                                    self.next_contexts.push((new_ctx, new_origin));
                                }
                                ProgramResult::Match => {
                                    self.ever_matched = true;
                                }
                                ProgramResult::MatchAtEnd => {
                                    self.match_at_end = true;
                                }
                            }
                        }
                    }
                }
                self.pool.free(ctx.into_range());
            }
        }

        // --- Phase 3: Compute new seed ---
        let seed_idx = trans.seed_program;
        if self.cache.seed_emit_origins[seed_idx.0 as usize].is_some() {
            // Pure-emit seed: track implicitly.
            self.seed_origins_idx = Some(seed_idx);
        } else {
            self.seed_origins_idx = None;
            let seed_prog = &self.cache.programs[seed_idx.0 as usize];
            if !seed_prog.is_empty() {
                let fresh = CounterCtx::new();
                self.results.clear();
                execute_program(seed_prog, &fresh, self.pool, &mut self.results);
                for r in self.results.drain(..) {
                    match r {
                        ProgramResult::Continue(new_ctx, new_origin) => {
                            self.next_contexts.push((new_ctx, new_origin));
                        }
                        ProgramResult::Match => {
                            self.ever_matched = true;
                        }
                        ProgramResult::MatchAtEnd => {
                            self.match_at_end = true;
                        }
                    }
                }
            }
        }

        // Dedup explicit contexts (skip when ≤1 or no explicit contexts).
        if self.next_contexts.len() > 1 {
            dedup_contexts(&mut self.next_contexts, self.pool);
        }

        std::mem::swap(&mut self.contexts, &mut self.next_contexts);
    }

    /// Feed a byte slice.
    pub fn chunk(&mut self, input: &[u8]) {
        for &b in input {
            if self.ever_matched {
                return;
            }
            self.step(b);
        }
    }

    /// Signal end-of-input and return match result.
    pub fn finish(mut self) -> bool {
        if self.ever_matched {
            // Free remaining contexts.
            for (ctx, _) in self.contexts.drain(..) {
                self.pool.free(ctx.into_range());
            }
            return true;
        }

        // Check if any active context can match at end-of-input.
        if self.match_at_end {
            for (ctx, _) in self.contexts.drain(..) {
                self.pool.free(ctx.into_range());
            }
            return true;
        }

        for (ctx, _) in self.contexts.drain(..) {
            self.pool.free(ctx.into_range());
        }
        false
    }

    /// Check whether a match has been found so far.
    #[allow(dead_code)]
    pub fn ismatch(&self) -> bool {
        self.ever_matched
    }
}

impl fmt::Debug for CountingDfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingDfaMatcher")
            .field("current", &self.current)
            .field("num_contexts", &self.contexts.len())
            .field("ever_matched", &self.ever_matched)
            .finish()
    }
}

/// Deduplicate counter contexts by (origin, value).  O(n²) linear scan.
fn dedup_contexts(contexts: &mut Vec<(CounterCtx, StateIdx)>, pool: &mut CounterPool) {
    let mut i = 0;
    while i < contexts.len() {
        let mut dup = false;
        for j in 0..i {
            if contexts[i].1 == contexts[j].1 && pool.ctx_eq(&contexts[i].0, &contexts[j].0) {
                dup = true;
                break;
            }
        }
        if dup {
            let (removed, _) = contexts.swap_remove(i);
            pool.free(removed.into_range());
            // Don't increment i — the swapped-in element needs checking.
        } else {
            i += 1;
        }
    }
}
