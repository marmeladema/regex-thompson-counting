//! Tier 4: Tier 4: Counting DFA for patterns with bounded repetitions.
//!
//! The DFA handles state transitions (cacheable); counter values live
//! in a side-channel of `CounterCtx` entries updated by compiled
//! counter programs.  Each context also tracks its NFA "origin"
//! (the consuming state it is waiting at) so that only the relevant
//! counter program is applied to it.

use indexmap::IndexSet;
use std::fmt;

use crate::{
    AssertKind, CounterCtx, CounterIdx, CounterPool, CtxDedupTable, Regex, State, StateIdx,
    byte_match_ci,
};

use super::{DfaState, DfaStateId, DfaStateRef};

/// Index into [`Tier4DfaCache::programs`].
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

// ---------------------------------------------------------------------------
// Iterative trace frames
// ---------------------------------------------------------------------------

/// Work item for the iterative epsilon closure in
/// `epsilon_closure_with_program`.  Processed LIFO.
///
/// See [`Tier4DfaCache::epsilon_closure_with_program`] for an
/// explanation of the stack-based mark/unmark cycle detection.
enum TraceFrame {
    /// Process NFA state `idx`: mark it visited, inspect the NFA state
    /// type, and push child frames.
    Visit(StateIdx),
    /// Unmark `idx` from `closure_visited`.  Pushed *before* children
    /// so it fires *after* them (LIFO).  This is the "exit" half of
    /// stack-based cycle detection: once a subtree completes, sibling
    /// branches may revisit the state to build their own program nodes.
    Unmark(usize),
    /// Pop the ops stack and wrap the sub-program in `Init`.
    FinishInit { counter: CounterIdx },
    /// Finish the continue subtree of a `CounterIncrement`: pop continue
    /// ops, push them as a sentinel, push a new level for the break path.
    FinishIncrementContinue,
    /// Finish the break subtree: pop break ops and continue ops, build
    /// the `Increment` node on the parent level.
    FinishIncrementBreak {
        counter: CounterIdx,
        min: usize,
        max: usize,
    },
}

/// Work item for the iterative `$`-gate epsilon trace
/// (`trace_end_iterative`).  Same structure as `TraceFrame` but without
/// `closure_result` or assertion context (simpler sub-graph).
enum TraceEndFrame {
    Visit(StateIdx),
    Unmark(usize),
    FinishInit {
        counter: CounterIdx,
    },
    FinishIncrementContinue,
    FinishIncrementBreak {
        counter: CounterIdx,
        min: usize,
        max: usize,
    },
}

// ---------------------------------------------------------------------------
// Program execution
// ---------------------------------------------------------------------------

/// Outcome of applying a counter program to one context.
enum ProgramResult {
    /// The context survives with updated counter values at the given
    /// NFA consuming state.
    Continue(StateIdx, CounterCtx),
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
                results.push(ProgramResult::Continue(*origin, ctx.clone(pool)));
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
                        results.push(ProgramResult::Continue(*origin, result));
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
                        results.push(ProgramResult::Continue(*origin, result));
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
                return Some(ProgramResult::Continue(*origin, CounterCtx::new()));
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
// Counting DFA cache (Tier 4)
// ---------------------------------------------------------------------------

/// Index into [`Tier4DfaCache::origin_program_tables`].
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

/// Lazy DFA cache for Tier 4 (patterns with counters).
///
/// Shares the `DfaState` representation with Tier 1 — the DFA state is
/// the set of NFA consuming states, ignoring counter values.  Counter
/// values live in a separate side-channel of `CounterCtx` entries.
pub(crate) struct Tier4DfaCache {
    /// Append-only, deduplicated table of DFA states.  Acts as both the
    /// state table (index → state) and the reverse lookup (state → index).
    states: IndexSet<DfaState, ahash::RandomState>,
    /// Flat transition table: indexed by `state.0 * stride + byte_class`.
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
    /// Scratch visited set for epsilon closure (stack-based mark/unmark).
    closure_visited: Vec<bool>,
    /// Scratch work stack for the main epsilon closure.
    closure_work: Vec<TraceFrame>,
    /// Scratch flat buffer for counter ops being built.  Levels are
    /// delimited by indices in `closure_ops_levels`.
    closure_ops: Vec<CounterOp>,
    /// Stack of start-indices into `closure_ops`, one per nesting level.
    closure_ops_levels: Vec<usize>,
    /// Scratch buffer for NFA consuming states found during closure.
    closure_result: Vec<StateIdx>,
    /// Scratch visited set for the `$`-gate sub-closure (separate from
    /// the main `closure_visited` because `$`-gate states may overlap).
    closure_end_visited: Vec<bool>,
    /// Scratch work stack for the `$`-gate sub-closure.
    closure_end_work: Vec<TraceEndFrame>,
    /// Scratch flat buffer for `$`-gate counter ops.
    closure_end_ops: Vec<CounterOp>,
    /// Level indices for `closure_end_ops`.
    closure_end_levels: Vec<usize>,
    /// Number of byte equivalence classes — the stride of each DFA state
    /// row in the transition table.  Copied from [`Regex::num_byte_classes`]
    /// during [`prepare()`].
    stride: usize,
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

impl fmt::Debug for Tier4DfaCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tier4DfaCache")
            .field("num_states", &self.states.len())
            .field("num_programs", &self.programs.len())
            .finish()
    }
}

impl Tier4DfaCache {
    pub(crate) fn new(num_nfa_states: usize) -> Self {
        Self {
            states: IndexSet::default(),
            transitions: Vec::new(),
            programs: Vec::new(),
            origin_program_tables: Vec::new(),
            seed_emit_origins: Vec::new(),
            closure_visited: vec![false; num_nfa_states],
            closure_work: Vec::new(),
            closure_ops: Vec::new(),
            closure_ops_levels: Vec::new(),
            closure_result: Vec::new(),
            closure_end_visited: vec![false; num_nfa_states],
            closure_end_work: Vec::new(),
            closure_end_ops: Vec::new(),
            closure_end_levels: Vec::new(),
            stride: 256,
            regex_id: 0,
            start_id: DfaStateId::DEAD,
            start_program: CounterProgramIdx(0),
            start_is_match: false,
            start_is_match_at_end: false,
        }
    }

    /// Look up or insert a DFA state.
    ///
    /// Takes a borrowed slice so that cache-hit probes (the common case)
    /// are zero-allocation.  A heap-allocated [`DfaState`] is only created
    /// on a cache miss when actual insertion is needed.
    fn intern_state(
        &mut self,
        nfa_states: &[StateIdx],
        is_match: bool,
        is_match_at_end: bool,
    ) -> DfaStateId {
        let probe = DfaStateRef {
            nfa_states,
            deferred_asserts: &[],
            is_match,
            is_match_at_end,
            prev_was_word: false,
        };
        if let Some(idx) = self.states.get_index_of(&probe) {
            return DfaStateId(idx as u32);
        }
        let state = DfaState {
            nfa_states: nfa_states.into(),
            deferred_asserts: Box::new([]),
            is_match,
            is_match_at_end,
            prev_was_word: false,
        };
        let (idx, _) = self.states.insert_full(state);
        self.transitions.resize(
            self.states.len() * self.stride,
            CountingTransition::UNPOPULATED,
        );
        DfaStateId(idx as u32)
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
    /// Uses an iterative work stack with an explicit frame stack for
    /// building the nested `CounterOp` tree.  This avoids unbounded
    /// recursion depth on adversarial patterns (the engine accepts
    /// untrusted patterns and inputs).
    ///
    /// # Cycle detection: stack-based mark/unmark
    ///
    /// Unlike tiers 1–3 which permanently mark each state as visited,
    /// tier 4 uses **stack-based** marking: a state is marked on entry
    /// and unmarked (via `Unmark` frame) when its subtree completes.
    ///
    /// This is necessary because the counter program tree requires
    /// sibling branches to independently traverse shared NFA states.
    /// For example, in `(a?){2}` the NFA has a cycle `CInc → Split →
    /// CInc`.  The `on_continue` subtree of the outer `Increment` must
    /// re-enter the `CInc` node to build the inner `Increment` — with
    /// permanent marking, this second visit would be skipped and the
    /// program tree would be incomplete.
    ///
    /// Stack-based marking prevents infinite loops (a state on the
    /// current DFS path is skipped) while allowing sibling branches to
    /// revisit it after the first branch completes.
    ///
    /// Leaf states (Match, Byte, etc.) and dead-end assertions unmark
    /// immediately since they have no children.  Branching states
    /// (Split, CounterInstance, CounterIncrement) push `Unmark` before
    /// their children so it fires after all children complete (LIFO).
    ///
    /// # Scratch buffer reuse
    ///
    /// All scratch buffers (`closure_work`, `closure_ops`,
    /// `closure_ops_levels`, `closure_result`) live on `Tier4DfaCache`
    /// and are reused across calls.  The ops buffer is flat: nesting
    /// levels are tracked by start-indices in `closure_ops_levels`.
    /// When a level completes, ops are drained from that index onward.
    ///
    /// Variable naming follows the conventions in `DfaMemory`:
    /// - `closure_visited` — per-state visited flag
    /// - `closure_result` — sorted NFA consuming states
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

        self.closure_result.clear();
        let mut is_match = false;
        let mut is_match_at_end = false;

        // Flat ops buffer with level tracking.  `closure_ops_levels`
        // stores the start index of each nesting level in `closure_ops`.
        // Popping a level drains ops from that index onward.
        self.closure_ops.clear();
        self.closure_ops_levels.clear();
        self.closure_ops_levels.push(0); // root level starts at 0

        // Reusable work stack.
        self.closure_work.clear();
        self.closure_work.extend(seeds.map(TraceFrame::Visit));
        self.closure_work.reverse(); // so first seed is processed first (LIFO)

        while let Some(frame) = self.closure_work.pop() {
            match frame {
                TraceFrame::Visit(idx) => {
                    let i = idx.idx();
                    if self.closure_visited[i] {
                        continue;
                    }
                    self.closure_visited[i] = true;

                    match states[idx] {
                        State::Split { out, out1 } => {
                            self.closure_work.push(TraceFrame::Unmark(i));
                            self.closure_work.push(TraceFrame::Visit(out1));
                            self.closure_work.push(TraceFrame::Visit(out));
                        }
                        State::Assert { kind, out } => match kind {
                            AssertKind::Start => {
                                if at_start {
                                    self.closure_work.push(TraceFrame::Unmark(i));
                                    self.closure_work.push(TraceFrame::Visit(out));
                                } else {
                                    self.closure_visited[i] = false;
                                }
                            }
                            AssertKind::End => {
                                // `$` gate: trace separately for
                                // match-at-end paths.
                                let sub_ops = self.trace_end_iterative(out, states);
                                if !sub_ops.is_empty() {
                                    is_match_at_end = true;
                                    self.closure_ops.extend(sub_ops);
                                }
                                self.closure_visited[i] = false;
                            }
                            AssertKind::StartLF => {
                                if at_start || prev_byte == Some(b'\n') {
                                    self.closure_work.push(TraceFrame::Unmark(i));
                                    self.closure_work.push(TraceFrame::Visit(out));
                                } else {
                                    self.closure_visited[i] = false;
                                }
                            }
                            _ => {
                                debug_assert!(
                                    false,
                                    "unsupported assertion in tier 4 pattern: {:?}",
                                    kind
                                );
                                self.closure_visited[i] = false;
                            }
                        },
                        State::CounterInstance { counter, out } => {
                            self.closure_work.push(TraceFrame::Unmark(i));
                            self.closure_work.push(TraceFrame::FinishInit { counter });
                            // Push a new level: ops from here onward
                            // belong to the Init body.
                            self.closure_ops_levels.push(self.closure_ops.len());
                            self.closure_work.push(TraceFrame::Visit(out));
                        }
                        State::CounterIncrement {
                            counter,
                            out,
                            out1,
                            min,
                            max,
                        } => {
                            // Two subtrees: continue (out) then break (out1).
                            // Push in reverse order for correct LIFO processing.
                            self.closure_work.push(TraceFrame::Unmark(i));
                            self.closure_work.push(TraceFrame::FinishIncrementBreak {
                                counter,
                                min,
                                max,
                            });
                            self.closure_work.push(TraceFrame::Visit(out1));
                            self.closure_work.push(TraceFrame::FinishIncrementContinue);
                            // New level for continue path.
                            self.closure_ops_levels.push(self.closure_ops.len());
                            self.closure_work.push(TraceFrame::Visit(out));
                        }
                        State::Match => {
                            is_match = true;
                            self.closure_ops.push(CounterOp::EmitMatch);
                            self.closure_visited[i] = false;
                        }
                        State::Byte { .. }
                        | State::ByteCI { .. }
                        | State::Wildcard { .. }
                        | State::ByteClassStatic { .. }
                        | State::ByteClassCustom { .. }
                        | State::ByteTable { .. } => {
                            self.closure_result.push(idx);
                            self.closure_ops
                                .push(CounterOp::EmitContinue { origin: idx });
                            self.closure_visited[i] = false;
                        }
                    }
                }
                TraceFrame::Unmark(i) => {
                    self.closure_visited[i] = false;
                }
                TraceFrame::FinishInit { counter } => {
                    let start = self.closure_ops_levels.pop().unwrap();
                    let sub_ops = self.closure_ops.drain(start..).collect();
                    self.closure_ops.push(CounterOp::Init {
                        counter,
                        then: sub_ops,
                    });
                }
                TraceFrame::FinishIncrementContinue => {
                    // Drain the continue ops, stash them as a temporary
                    // Init carrier on the flat buffer, then start a new
                    // level for the break path.
                    let start = self.closure_ops_levels.pop().unwrap();
                    let cont_ops: Vec<CounterOp> = self.closure_ops.drain(start..).collect();
                    // Stash continue ops inside a sentinel Init node.
                    self.closure_ops.push(CounterOp::Init {
                        counter: CounterIdx(u8::MAX),
                        then: cont_ops,
                    });
                    // New level for break path.
                    self.closure_ops_levels.push(self.closure_ops.len());
                }
                TraceFrame::FinishIncrementBreak { counter, min, max } => {
                    // Pop break ops.
                    let start = self.closure_ops_levels.pop().unwrap();
                    let raw_break_ops: Vec<CounterOp> = self.closure_ops.drain(start..).collect();
                    // Pop the stashed continue ops (sentinel Init node).
                    let cont_sentinel = self.closure_ops.pop().unwrap();
                    let cont_ops = match cont_sentinel {
                        CounterOp::Init { then, .. } => then,
                        _ => unreachable!("expected stashed continue ops"),
                    };
                    let break_ops = vec![CounterOp::Remove {
                        counter,
                        then: raw_break_ops,
                    }];
                    self.closure_ops.push(CounterOp::Increment {
                        counter,
                        min,
                        max,
                        on_continue: cont_ops,
                        on_break: break_ops,
                    });
                }
            }
        }

        debug_assert_eq!(
            self.closure_ops_levels.len(),
            1,
            "closure_ops_levels should have exactly one level (root)"
        );

        self.closure_result.sort_unstable();
        self.closure_result.dedup();

        let ops = self.closure_ops.drain(..).collect();
        let result = self
            .closure_result
            .drain(..)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        (result, is_match, is_match_at_end, ops)
    }

    /// Iterative epsilon trace for the `$` (End) path.
    ///
    /// This is separate from the main closure because the `$` gate's
    /// sub-graph may contain counter operations that should only fire
    /// at end-of-input.  Uses its own scratch buffers (`closure_end_*`)
    /// because these states may also appear in the main closure, and
    /// this method is called *during* `epsilon_closure_with_program`.
    ///
    /// Uses the same stack-based mark/unmark cycle detection as the
    /// main closure — see [`epsilon_closure_with_program`] for details.
    ///
    /// Returns the sub-program with `EmitMatch` rewritten to
    /// `EmitMatchAtEnd`.  Returns an empty vec if no Match is reachable.
    fn trace_end_iterative(&mut self, start: StateIdx, states: &[State]) -> Vec<CounterOp> {
        for v in self.closure_end_visited.iter_mut() {
            *v = false;
        }
        let mut is_match = false;

        self.closure_end_ops.clear();
        self.closure_end_levels.clear();
        self.closure_end_levels.push(0);

        self.closure_end_work.clear();
        self.closure_end_work.push(TraceEndFrame::Visit(start));

        while let Some(frame) = self.closure_end_work.pop() {
            match frame {
                TraceEndFrame::Visit(idx) => {
                    let i = idx.idx();
                    if self.closure_end_visited[i] {
                        continue;
                    }
                    self.closure_end_visited[i] = true;

                    match states[idx] {
                        State::Match => {
                            is_match = true;
                            self.closure_end_ops.push(CounterOp::EmitMatch);
                            self.closure_end_visited[i] = false;
                        }
                        State::Split { out, out1 } => {
                            self.closure_end_work.push(TraceEndFrame::Unmark(i));
                            self.closure_end_work.push(TraceEndFrame::Visit(out1));
                            self.closure_end_work.push(TraceEndFrame::Visit(out));
                        }
                        State::Assert { kind, out } => {
                            // Inside a `$` gate, `$` itself is trivially
                            // true.  `^` / `(?m:^)` cannot pass here.
                            if kind == AssertKind::End {
                                self.closure_end_work.push(TraceEndFrame::Unmark(i));
                                self.closure_end_work.push(TraceEndFrame::Visit(out));
                            } else {
                                self.closure_end_visited[i] = false;
                            }
                        }
                        State::CounterInstance { counter, out } => {
                            self.closure_end_work.push(TraceEndFrame::Unmark(i));
                            self.closure_end_work
                                .push(TraceEndFrame::FinishInit { counter });
                            self.closure_end_levels.push(self.closure_end_ops.len());
                            self.closure_end_work.push(TraceEndFrame::Visit(out));
                        }
                        State::CounterIncrement {
                            counter,
                            out,
                            out1,
                            min,
                            max,
                        } => {
                            self.closure_end_work.push(TraceEndFrame::Unmark(i));
                            self.closure_end_work
                                .push(TraceEndFrame::FinishIncrementBreak { counter, min, max });
                            self.closure_end_work.push(TraceEndFrame::Visit(out1));
                            self.closure_end_work
                                .push(TraceEndFrame::FinishIncrementContinue);
                            self.closure_end_levels.push(self.closure_end_ops.len());
                            self.closure_end_work.push(TraceEndFrame::Visit(out));
                        }
                        _ => {
                            // Consuming states block at end-of-input.
                            self.closure_end_visited[i] = false;
                        }
                    }
                }
                TraceEndFrame::Unmark(i) => {
                    self.closure_end_visited[i] = false;
                }
                TraceEndFrame::FinishInit { counter } => {
                    let start = self.closure_end_levels.pop().unwrap();
                    let sub_ops = self.closure_end_ops.drain(start..).collect();
                    self.closure_end_ops.push(CounterOp::Init {
                        counter,
                        then: sub_ops,
                    });
                }
                TraceEndFrame::FinishIncrementContinue => {
                    let start = self.closure_end_levels.pop().unwrap();
                    let cont_ops: Vec<CounterOp> = self.closure_end_ops.drain(start..).collect();
                    self.closure_end_ops.push(CounterOp::Init {
                        counter: CounterIdx(u8::MAX),
                        then: cont_ops,
                    });
                    self.closure_end_levels.push(self.closure_end_ops.len());
                }
                TraceEndFrame::FinishIncrementBreak { counter, min, max } => {
                    let start = self.closure_end_levels.pop().unwrap();
                    let raw_break_ops: Vec<CounterOp> =
                        self.closure_end_ops.drain(start..).collect();
                    let cont_sentinel = self.closure_end_ops.pop().unwrap();
                    let cont_ops = match cont_sentinel {
                        CounterOp::Init { then, .. } => then,
                        _ => unreachable!("expected stashed continue ops"),
                    };
                    let break_ops = vec![CounterOp::Remove {
                        counter,
                        then: raw_break_ops,
                    }];
                    self.closure_end_ops.push(CounterOp::Increment {
                        counter,
                        min,
                        max,
                        on_continue: cont_ops,
                        on_break: break_ops,
                    });
                }
            }
        }

        if !is_match {
            return Vec::new();
        }

        debug_assert_eq!(self.closure_end_levels.len(), 1);
        let mut ops: Vec<CounterOp> = self.closure_end_ops.drain(..).collect();
        rewrite_match_to_match_at_end(&mut ops);
        ops
    }

    /// Compute a counting transition for `(from_state, byte)`, using
    /// byte-class compression (stride < 256).
    #[inline(always)]
    fn transition(&mut self, from: DfaStateId, byte: u8, regex: &Regex) -> CountingTransition {
        if from == DfaStateId::DEAD {
            let t = self.populate(from, byte, regex);
            return t;
        }
        let slot = from.0 as usize * self.stride + regex.byte_classes[byte as usize] as usize;
        let t = self.transitions[slot];
        if !t.is_unpopulated() {
            return t;
        }
        let t = self.populate(from, byte, regex);
        self.transitions[slot] = t;
        t
    }

    /// Compute a counting transition for `(from_state, byte)`, using the
    /// identity mapping (stride=256, no byte-class indirection).
    #[inline(always)]
    fn transition_direct(
        &mut self,
        from: DfaStateId,
        byte: u8,
        regex: &Regex,
    ) -> CountingTransition {
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
                match regex.states[idx] {
                    State::Byte {
                        byte: b2,
                        out,
                        out_exit,
                    } if byte == b2 => {
                        origin_targets.push((idx, out));
                        if out_exit != StateIdx::NONE {
                            origin_targets.push((idx, out_exit));
                        }
                    }
                    State::ByteCI {
                        byte: b2,
                        out,
                        out_exit,
                    } if byte_match_ci(byte, b2) => {
                        origin_targets.push((idx, out));
                        if out_exit != StateIdx::NONE {
                            origin_targets.push((idx, out_exit));
                        }
                    }
                    State::Wildcard { out, out_exit } => {
                        origin_targets.push((idx, out));
                        if out_exit != StateIdx::NONE {
                            origin_targets.push((idx, out_exit));
                        }
                    }
                    State::ByteClassStatic {
                        table,
                        out,
                        out_exit,
                    } if table[byte as usize] => {
                        origin_targets.push((idx, out));
                        if out_exit != StateIdx::NONE {
                            origin_targets.push((idx, out_exit));
                        }
                    }
                    State::ByteClassCustom {
                        class,
                        out,
                        out_exit,
                    } if regex.classes[class.idx()].contains(byte) => {
                        origin_targets.push((idx, out));
                        if out_exit != StateIdx::NONE {
                            origin_targets.push((idx, out_exit));
                        }
                    }
                    State::ByteTable { table } => {
                        let t = regex.byte_tables[table][byte];
                        if t != StateIdx::NONE {
                            origin_targets.push((idx, t));
                        }
                    }
                    _ => {}
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
        merged.sort_unstable();
        merged.dedup();

        let origin_table_idx = self.intern_origin_table(origin_table);

        if merged.is_empty() && !is_match && !is_match_at_end {
            return CountingTransition {
                next: DfaStateId::DEAD,
                origin_table: origin_table_idx,
                seed_program,
            };
        }
        let next = self.intern_state(&merged, is_match, is_match_at_end);
        CountingTransition {
            next,
            origin_table: origin_table_idx,
            seed_program,
        }
    }

    /// Reset the cache.
    fn clear(&mut self, num_nfa_states: usize, stride: usize) {
        self.states.clear();
        self.transitions.clear();
        self.programs.clear();
        self.origin_program_tables.clear();
        self.seed_emit_origins.clear();
        self.closure_visited.clear();
        self.closure_visited.resize(num_nfa_states, false);
        self.closure_end_visited.clear();
        self.closure_end_visited.resize(num_nfa_states, false);
        // Scratch vecs are just cleared, not resized — they keep their
        // heap allocation for reuse.
        self.closure_work.clear();
        self.closure_ops.clear();
        self.closure_ops_levels.clear();
        self.closure_result.clear();
        self.closure_end_work.clear();
        self.closure_end_ops.clear();
        self.closure_end_levels.clear();
        self.stride = stride;
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
        self.clear(regex.states.len(), regex.num_byte_classes);
        self.regex_id = id;

        let (nfa_set, is_match, is_match_at_end, ops) = self.epsilon_closure_with_program(
            std::iter::once(regex.start),
            &regex.states,
            true,
            None,
        );
        self.start_id = self.intern_state(&nfa_set, is_match, is_match_at_end);
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
// Tier 4 matcher
// ---------------------------------------------------------------------------

/// Tier 4 DFA matcher for patterns with bounded repetitions.
///
/// The DFA handles state transitions (cacheable); counter values live
/// in a side-channel of `CounterCtx` entries updated by compiled
/// counter programs.  Each context also tracks its NFA "origin"
/// (the consuming state it is waiting at) so that only the relevant
/// counter program is applied to it.
pub struct Tier4DfaMatcher<'a> {
    cache: &'a mut Tier4DfaCache,
    regex: &'a Regex,
    pool: &'a mut CounterPool,
    /// Current DFA state.
    current: DfaStateId,
    /// Active counter contexts, each paired with its NFA origin.
    /// Does NOT include implicit "seed" contexts — those are tracked
    /// separately via `seed_origins` to avoid redundant alloc/free.
    contexts: Vec<(StateIdx, CounterCtx)>,
    /// Scratch space for new contexts during step (avoids realloc).
    next_contexts: Vec<(StateIdx, CounterCtx)>,
    /// Scratch space for program execution results (avoids per-step alloc).
    results: Vec<ProgramResult>,
    /// Scratch table for hash-based context dedup.  Indices point into
    /// `contexts` / `next_contexts` during [`dedup_contexts`] calls.
    /// Cleared at the start of each dedup pass; retains capacity across
    /// steps.
    dedup_seen: CtxDedupTable,
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

impl<'a> Tier4DfaMatcher<'a> {
    pub(crate) fn new(
        cache: &'a mut Tier4DfaCache,
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
                    ProgramResult::Continue(origin, ctx) => contexts.push((origin, ctx)),
                    ProgramResult::Match => ever_matched = true,
                    ProgramResult::MatchAtEnd => match_at_end = true,
                }
            }
            // Dedup initial contexts.
            let mut dedup_seen = CtxDedupTable::new();
            dedup_contexts(&mut contexts, pool, &mut dedup_seen);
        }

        if cache.start_is_match {
            ever_matched = true;
        }
        if cache.start_is_match_at_end {
            match_at_end = true;
        }

        Tier4DfaMatcher {
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
            dedup_seen: CtxDedupTable::new(),
        }
    }

    /// Advance by one byte.
    #[inline]
    fn step(&mut self, byte: u8) {
        let trans = if self.cache.stride == 256 {
            self.cache.transition_direct(self.current, byte, self.regex)
        } else {
            self.cache.transition(self.current, byte, self.regex)
        };
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
                // Look up ALL programs for this seed origin (may be >1
                // with dense out_exit encoding).
                for &(o, prog_idx) in origin_table.iter() {
                    if o != seed_origin {
                        continue;
                    }
                    let prog = &self.cache.programs[prog_idx.0 as usize];
                    // Seed contexts are empty — execute the program
                    // with a fresh context.
                    let fresh = CounterCtx::new();
                    self.results.clear();
                    execute_program(prog, &fresh, self.pool, &mut self.results);
                    for r in self.results.drain(..) {
                        match r {
                            ProgramResult::Continue(new_origin, new_ctx) => {
                                self.next_contexts.push((new_origin, new_ctx));
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
            for (_, ctx) in self.contexts.drain(..) {
                self.pool.free(ctx.into_range());
            }
        } else {
            for (ctx_origin, mut ctx) in self.contexts.drain(..) {
                // Find the program for this context's origin NFA state.
                // With dense out_exit encoding, an origin may have
                // multiple programs; execute all of them.
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
                // Execute additional programs for this origin (second+
                // targets from out_exit), cloning the context for each.
                if prog_idx.is_some() {
                    for &(o, extra_prog_idx) in origin_table.iter() {
                        if o != ctx_origin {
                            continue;
                        }
                        if Some(extra_prog_idx) == prog_idx {
                            continue; // skip the primary — handled below
                        }
                        let cloned_ctx = ctx.clone(self.pool);
                        let prog = &self.cache.programs[extra_prog_idx.0 as usize];
                        self.results.clear();
                        execute_program(prog, &cloned_ctx, self.pool, &mut self.results);
                        for r in self.results.drain(..) {
                            match r {
                                ProgramResult::Continue(new_origin, new_ctx) => {
                                    self.next_contexts.push((new_origin, new_ctx));
                                }
                                ProgramResult::Match => {
                                    self.ever_matched = true;
                                }
                                ProgramResult::MatchAtEnd => {
                                    self.match_at_end = true;
                                }
                            }
                        }
                        self.pool.free(cloned_ctx.into_range());
                    }
                }
                if let Some(prog_idx) = prog_idx {
                    let prog = &self.cache.programs[prog_idx.0 as usize];
                    // Fast path: single Increment with one active branch.
                    if let Some(result) = try_execute_inplace(prog, &mut ctx, self.pool) {
                        match result {
                            ProgramResult::Continue(new_origin, _) => {
                                self.next_contexts.push((new_origin, ctx));
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
                                ProgramResult::Continue(new_origin, new_ctx) => {
                                    self.next_contexts.push((new_origin, new_ctx));
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
                        ProgramResult::Continue(new_origin, new_ctx) => {
                            self.next_contexts.push((new_origin, new_ctx));
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
            dedup_contexts(&mut self.next_contexts, self.pool, &mut self.dedup_seen);
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
            for (_, ctx) in self.contexts.drain(..) {
                self.pool.free(ctx.into_range());
            }
            return true;
        }

        // Check if any active context can match at end-of-input.
        if self.match_at_end {
            for (_, ctx) in self.contexts.drain(..) {
                self.pool.free(ctx.into_range());
            }
            return true;
        }

        for (_, ctx) in self.contexts.drain(..) {
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

impl fmt::Debug for Tier4DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Tier4DfaMatcher");
        s.field("current", &self.current);
        if self.current != DfaStateId::DEAD {
            let st = &self.cache.states[self.current.idx()];
            s.field("nfa_states", &st.nfa_states);
            s.field("is_match", &st.is_match);
            s.field("is_match_at_end", &st.is_match_at_end);
        }
        s.field("num_contexts", &self.contexts.len());
        s.field("match_at_end", &self.match_at_end);
        s.field("seed_origins_idx", &self.seed_origins_idx);
        s.field("ever_matched", &self.ever_matched);
        s.finish()
    }
}

impl fmt::Display for Tier4DfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.current == DfaStateId::DEAD {
            return write!(f, "DFA[T4] state=DEAD matched={}", self.ever_matched);
        }
        let state = &self.cache.states[self.current.idx()];
        write!(
            f,
            "DFA[T4] state={} nfa={{{}}} matched={}",
            self.current.0,
            state
                .nfa_states
                .iter()
                .map(|s: &StateIdx| s.to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.ever_matched,
        )?;
        if state.is_match {
            write!(f, " is_match")?;
        }
        if state.is_match_at_end {
            write!(f, " mae")?;
        }
        write!(f, " contexts={}", self.contexts.len())?;
        // Summarise non-empty contexts.
        if !self.contexts.is_empty() {
            let non_empty = self.contexts.iter().filter(|(_, c)| !c.is_empty()).count();
            if non_empty > 0 {
                write!(f, " ({non_empty} with active counters)")?;
            }
        }
        if self.seed_origins_idx.is_some() {
            write!(f, " +seed")?;
        }
        Ok(())
    }
}

/// Threshold below which the O(n²) linear scan is faster than the
/// hash-based approach (avoids hash table overhead for small context lists).
const DEDUP_HASH_THRESHOLD: usize = 32;

/// Deduplicate counter contexts by (origin, counter values).
///
/// For small lists (≤ [`DEDUP_HASH_THRESHOLD`]), uses a direct O(n²)
/// pairwise scan.  For larger lists, uses [`CtxDedupTable`] for O(n)
/// expected time.  On hash collision, falls back to full
/// [`CounterPool::ctx_eq`] comparison for correctness.
///
/// The `seen` table is passed in to avoid per-call allocation; it is
/// cleared on entry and retains capacity across calls.
fn dedup_contexts(
    contexts: &mut Vec<(StateIdx, CounterCtx)>,
    pool: &mut CounterPool,
    seen: &mut CtxDedupTable,
) {
    if contexts.len() <= DEDUP_HASH_THRESHOLD {
        dedup_contexts_linear(contexts, pool);
    } else {
        dedup_contexts_hash(contexts, pool, seen);
    }
}

/// O(n²) pairwise dedup for small context lists.
fn dedup_contexts_linear(contexts: &mut Vec<(StateIdx, CounterCtx)>, pool: &mut CounterPool) {
    let mut i = 0;
    while i < contexts.len() {
        let mut dup = false;
        for j in 0..i {
            if contexts[i].0 == contexts[j].0 && pool.ctx_eq(&contexts[i].1, &contexts[j].1) {
                dup = true;
                break;
            }
        }
        if dup {
            let (_, removed) = contexts.swap_remove(i);
            pool.free(removed.into_range());
        } else {
            i += 1;
        }
    }
}

/// Hash-based dedup for large context lists.
fn dedup_contexts_hash(
    contexts: &mut Vec<(StateIdx, CounterCtx)>,
    pool: &mut CounterPool,
    seen: &mut CtxDedupTable,
) {
    seen.clear();

    let mut i = 0;
    while i < contexts.len() {
        let hash = pool.ctx_hash(&contexts[i].1, contexts[i].0);
        let dup = seen.find(hash, |j| {
            contexts[i].0 == contexts[j].0 && pool.ctx_eq(&contexts[i].1, &contexts[j].1)
        });
        if dup {
            let (_, removed) = contexts.swap_remove(i);
            pool.free(removed.into_range());
        } else {
            seen.insert(hash, i);
            i += 1;
        }
    }
}
