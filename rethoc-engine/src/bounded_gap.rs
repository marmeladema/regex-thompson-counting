//! Bounded-gap engine types and HIR analysis helpers.
//!
//! This module contains:
//!
//! - **Plan types**: [`BoundedGapPlan`], [`GapPlan`], [`AnchorPlan`],
//!   [`AnchorProgram`], and supporting enums that define the compile-time
//!   representation of a bounded-gap chain.
//! - **Analysis helpers**: functions that operate on `regex-syntax` HIR
//!   to detect whether a pattern can be compiled as a bounded-gap chain
//!   and extract the structural parameters needed for plan construction.

use regex_syntax::hir::{Hir, HirKind};

use crate::classes::{ByteClass, ByteClassBits};
use crate::prefilter::Prefilter;
use crate::{ByteMap, RegexConfig, State, StateIdx};

// ---------------------------------------------------------------------------
// Analysis Types
// ---------------------------------------------------------------------------

/// Classification of a gap body predicate.
///
/// Version 1 gap predicates are always single-byte.  `Any` is kept
/// separate to enable a meaningful runtime fast path where bad-byte
/// tracking is skipped entirely for `.{0,K}` / `[\s\S]{0,K}` style gaps.
///
/// Used both during HIR analysis ([`classify_gap_body`]) and as the
/// predicate stored in [`GapPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GapPredicate {
    /// Matches any single byte (wildcard).
    Any,
    /// Matches a specific set of bytes, stored as a 256-bit inline set.
    /// Negation is already normalized at compile time.
    ByteClass(ByteClassBits),
}

/// Fixed-length information for an anchor fragment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedLengthInfo {
    /// The exact byte length consumed by this anchor.
    pub(crate) len: u16,
}

// ---------------------------------------------------------------------------
// Plan Types
// ---------------------------------------------------------------------------

/// Compiled plan for a bounded-gap chain.
///
/// Represents a pattern of the form `Anchor0 (Gap0 Anchor1)* GapTail?`
/// where each gap is a bounded repetition of a single-byte predicate
/// and each anchor is a separately compiled fixed-length regex fragment.
///
/// Invariant: `anchors.len() == interior_gaps.len() + 1`.
#[derive(Debug)]
pub(crate) struct BoundedGapPlan {
    /// The anchor programs, in chain order.
    pub(crate) anchors: Box<[AnchorPlan]>,
    /// Interior gap constraints between consecutive anchors.
    pub(crate) interior_gaps: Box<[GapPlan]>,
    /// Optional terminal gap after the final anchor.
    pub(crate) tail_gap: Option<GapPlan>,
    /// Whole-pattern start anchoring (`^`).
    pub(crate) start_anchor: StartAnchorKind,
    /// Whole-pattern end anchoring (`$`).
    pub(crate) end_anchor: EndAnchorKind,
}

/// A gap constraint between two anchors (or after the final anchor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GapPlan {
    /// Minimum gap length in bytes.
    pub(crate) min_gap: u32,
    /// Maximum gap length in bytes.
    pub(crate) max_gap: u32,
    /// Per-byte predicate for bytes in the gap.
    pub(crate) predicate: GapPredicate,
}

/// Compiled anchor within a bounded-gap chain.
#[derive(Debug)]
pub(crate) struct AnchorPlan {
    /// The compiled execution program for this anchor.
    pub(crate) program: AnchorProgram,
    /// Length information (fixed in Version 1).
    pub(crate) length_info: AnchorLengthInfo,
}

/// Compiled execution data for an anchor submatcher.
///
/// This is deliberately **not** a [`Regex`](crate::Regex).  It stores
/// only the data needed for event-producing anchor runners, without
/// nested specialisation probes, public diagnostics, or top-level
/// matcher dispatch.
#[derive(Debug)]
pub(crate) struct AnchorProgram {
    /// Tier-specific execution data.
    pub(crate) engine: AnchorEngine,
    /// Optional prefilter for skipping non-candidate bytes.
    pub(crate) prefilter: Prefilter,
}

/// Tier-specific anchor execution data.
///
/// Each variant stores only the compiled state that the corresponding
/// tier's anchor scanner needs to emit end-position events.  The exact
/// fields will be refined during Phase 3 (anchor runner implementation).
#[derive(Debug)]
pub(crate) enum AnchorEngine {
    /// NFA simulation (Tier 0).
    Tier0(AnchorNfaProgram),
    /// Lazy DFA (Tier 1).
    Tier1(AnchorDfaProgram),
    /// Differential-counter DFA (Tier 2).
    Tier2(AnchorTier2Program),
}

/// Anchor execution data for Tier 0 (NFA simulation).
///
/// Contains a self-contained NFA compiled from the anchor HIR fragment,
/// including its own state array, byte-class tables, and precomputed
/// start closure.  Counter-free in Version 1.
#[derive(Debug)]
pub(crate) struct AnchorNfaProgram {
    /// NFA state array for this anchor.
    pub(crate) states: Box<[State]>,
    /// Byte-class lookup tables for [`State::ByteClassCustom`].
    pub(crate) classes: Box<[ByteClassBits]>,
    /// Byte dispatch tables for [`State::ByteTable`].
    pub(crate) byte_tables: Box<[ByteMap]>,
    /// NFA start state index.
    pub(crate) start: StateIdx,
    /// Precomputed consuming leaves from the start state.
    pub(crate) start_closure: Box<[StateIdx]>,
    /// Whether the empty string matches (start closure reaches Match).
    pub(crate) start_closure_matches: bool,
}

/// Anchor execution data for Tier 1 (lazy DFA).
///
/// Placeholder — will be populated in a future phase.
#[derive(Debug)]
pub(crate) struct AnchorDfaProgram {
    pub(crate) _placeholder: (),
}

/// Anchor execution data for Tier 2 (differential-counter DFA).
///
/// Placeholder — will be populated in a future phase.
#[derive(Debug)]
pub(crate) struct AnchorTier2Program {
    pub(crate) _placeholder: (),
}

/// Anchor length information.
///
/// Version 1 supports only fixed-length anchors.  Phase 2 will add
/// `ExactSmallSet` and `ExactSparse` variants for bounded
/// variable-length anchors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AnchorLengthInfo {
    /// Anchor consumes exactly this many bytes.
    Fixed(u16),
}

/// Whole-pattern start anchoring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StartAnchorKind {
    /// No start anchoring — the chain can begin at any input position.
    #[default]
    None,
    /// `^` — the first anchor must start at position 0.
    StartOfInput,
}

/// Whole-pattern end anchoring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum EndAnchorKind {
    /// No end anchoring — the chain can end at any input position.
    #[default]
    None,
    /// `$` — the match must end at the final input position.
    EndOfInput,
}

/// Controls how a pattern is compiled.
///
/// Anchors are compiled in `Anchor` mode, which disables bounded-gap
/// detection and other top-level-only specialisations.  This prevents
/// recursive specialisation trees and keeps the ownership graph flat.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompileMode {
    /// Normal top-level compilation (specialisation probes enabled).
    TopLevel,
    /// Anchor sub-compilation (specialisation probes disabled).
    Anchor,
}

// ---------------------------------------------------------------------------
// Plan memory accounting
// ---------------------------------------------------------------------------

impl BoundedGapPlan {
    /// Estimated heap size of the plan (excluding the inline `Option` on
    /// `Regex`).  Used by [`Regex::memory_size`](crate::Regex::memory_size).
    pub(crate) fn heap_size(&self) -> usize {
        let anchors = self.anchors.len() * std::mem::size_of::<AnchorPlan>();
        let gaps = self.interior_gaps.len() * std::mem::size_of::<GapPlan>();
        let tail = if self.tail_gap.is_some() {
            std::mem::size_of::<GapPlan>()
        } else {
            0
        };
        anchors + gaps + tail
    }
}

// ---------------------------------------------------------------------------
// Anchor Compilation
// ---------------------------------------------------------------------------

/// Compile anchor HIR pieces into an [`AnchorProgram`].
///
/// Reconstructs a single HIR from the given pieces (concatenating if
/// necessary), compiles it through the standard [`RegexBuilder`] pipeline,
/// and extracts the NFA data.  The temporary [`Regex`] is consumed.
///
/// Returns `None` if compilation fails (e.g. unsupported constructs).
fn compile_anchor_program(pieces: &[&Hir], config: &RegexConfig) -> Option<AnchorProgram> {
    let anchor_hir = if pieces.len() == 1 {
        pieces[0].clone()
    } else {
        Hir::concat(pieces.iter().map(|h| (*h).clone()).collect())
    };
    let mut builder = crate::RegexBuilder::with_config(config.clone());
    match builder.build(&anchor_hir) {
        Ok(regex) => Some(regex.into_anchor_program()),
        Err(e) => {
            // Anchor compilation failed — fall back to normal engine.
            debug_assert!(false, "anchor compilation failed: {e}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Anchor Runner
// ---------------------------------------------------------------------------

/// Convenience wrapper for standalone anchor runner usage (tests).
///
/// Delegates NFA stepping to [`RunnerScratch`] while providing a
/// simpler API with owned scratch.
pub(crate) struct AnchorRunner<'a> {
    program: &'a AnchorNfaProgram,
    scratch: RunnerScratch,
}

impl<'a> AnchorRunner<'a> {
    /// Create a new runner from an [`AnchorPlan`].
    pub(crate) fn new(plan: &'a AnchorPlan) -> Self {
        let program = match &plan.program.engine {
            AnchorEngine::Tier0(p) => p,
            _ => unimplemented!("only Tier 0 anchor runners in Version 1"),
        };
        let mut scratch = RunnerScratch::new(program.states.len());
        scratch.seed_start(program);
        AnchorRunner { program, scratch }
    }

    /// Reset the runner for a new match (same anchor program).
    pub(crate) fn reset(&mut self) {
        self.scratch.reset();
        self.scratch.seed_start(self.program);
    }

    /// Process a chunk of input, emitting end-position events.
    pub(crate) fn scan_chunk(&mut self, input: &[u8], base_pos: u64, emit: &mut dyn FnMut(u64)) {
        for (offset, &byte) in input.iter().enumerate() {
            if self.scratch.step(self.program, byte) {
                emit(base_pos + offset as u64);
            }
        }
    }

    /// Signal end-of-input (no-op for V1 assertion-free anchors).
    pub(crate) fn finish_scan(&mut self, _emit: &mut dyn FnMut(u64)) {}

    /// Process one byte (for direct testing).
    fn step(&mut self, byte: u8) -> bool {
        self.scratch.step(self.program, byte)
    }
}

// ---------------------------------------------------------------------------
// Bounded-Gap Matcher
// ---------------------------------------------------------------------------

use std::collections::VecDeque;

/// Per-interior-gap stage queues.
#[derive(Debug)]
struct GapStageState {
    /// Accepted end positions from the preceding anchor, sorted ascending.
    accepted_prev_ends: VecDeque<u64>,
    /// Positions where the gap predicate failed, sorted ascending.
    bad_positions: VecDeque<u64>,
}

/// Terminal gap queues.
#[derive(Debug)]
struct TailGapState {
    /// Accepted end positions from the final anchor, sorted ascending.
    accepted_final_anchor_ends: VecDeque<u64>,
    /// Positions where the terminal gap predicate failed, sorted ascending.
    bad_positions: VecDeque<u64>,
}

/// Per-anchor NFA scratch state (reusable across matches).
#[derive(Debug)]
struct RunnerScratch {
    lastlist: Vec<usize>,
    listid: usize,
    clist: Vec<StateIdx>,
    nlist: Vec<StateIdx>,
    addstack: Vec<StateIdx>,
}

impl RunnerScratch {
    fn new(num_states: usize) -> Self {
        RunnerScratch {
            lastlist: vec![0; num_states],
            listid: 1,
            clist: Vec::with_capacity(num_states),
            nlist: Vec::with_capacity(num_states),
            addstack: Vec::with_capacity(num_states),
        }
    }

    /// Clear state for a new match, retaining allocations.
    fn reset(&mut self) {
        self.clist.clear();
        self.nlist.clear();
        self.addstack.clear();
        self.listid = 1;
        for slot in self.lastlist.iter_mut() {
            *slot = 0;
        }
    }

    /// Seed clist from the program's start closure.
    fn seed_start(&mut self, program: &AnchorNfaProgram) {
        if !program.start_closure.is_empty() {
            for &s in &*program.start_closure {
                self.clist.push(s);
                self.lastlist[s.idx()] = self.listid;
            }
        } else {
            self.addstack.push(program.start);
            self.drain_addstack_into_clist(program);
        }
    }

    /// Process one input byte against the anchor NFA.
    /// Returns `true` if the anchor matched ending at this byte.
    fn step(&mut self, program: &AnchorNfaProgram, byte: u8) -> bool {
        self.nlist.clear();
        self.listid += 1;
        let mut matched = false;

        // Phase 1: consume byte from each consuming state in clist.
        for i in 0..self.clist.len() {
            let state_idx = self.clist[i];
            let state = &program.states[state_idx.idx()];
            match *state {
                State::Byte {
                    byte: b,
                    out,
                    out_exit,
                } => {
                    if b == byte {
                        self.addstack.push(out);
                        if out_exit != StateIdx::NONE {
                            self.addstack.push(out_exit);
                        }
                    }
                }
                State::ByteCI {
                    byte: lower,
                    out,
                    out_exit,
                } => {
                    if byte == lower || byte == (lower ^ 0x20) {
                        self.addstack.push(out);
                        if out_exit != StateIdx::NONE {
                            self.addstack.push(out_exit);
                        }
                    }
                }
                State::Wildcard { out, out_exit } => {
                    self.addstack.push(out);
                    if out_exit != StateIdx::NONE {
                        self.addstack.push(out_exit);
                    }
                }
                State::ByteClassStatic {
                    table,
                    out,
                    out_exit,
                } => {
                    if table[byte as usize] {
                        self.addstack.push(out);
                        if out_exit != StateIdx::NONE {
                            self.addstack.push(out_exit);
                        }
                    }
                }
                State::ByteClassCustom {
                    class,
                    out,
                    out_exit,
                } => {
                    if program.classes[class.idx()].contains(byte) {
                        self.addstack.push(out);
                        if out_exit != StateIdx::NONE {
                            self.addstack.push(out_exit);
                        }
                    }
                }
                State::ByteTable { table } => {
                    let target = program.byte_tables[table.idx()][byte];
                    if target != StateIdx::NONE {
                        self.addstack.push(target);
                    }
                }
                _ => {}
            }
        }

        // Phase 2: drain addstack — follow epsilon chains into nlist.
        while let Some(idx) = self.addstack.pop() {
            if idx == StateIdx::NONE {
                continue;
            }
            let i = idx.idx();
            if self.lastlist[i] == self.listid {
                continue;
            }
            self.lastlist[i] = self.listid;
            match program.states[i] {
                State::Split { out, out1 } => {
                    self.addstack.push(out1);
                    self.addstack.push(out);
                }
                State::Match => {
                    matched = true;
                }
                _ => {
                    self.nlist.push(idx);
                }
            }
        }

        // Phase 3: re-seed from start closure.
        if !program.start_closure.is_empty() {
            for &s in &*program.start_closure {
                if self.lastlist[s.idx()] != self.listid {
                    self.lastlist[s.idx()] = self.listid;
                    self.nlist.push(s);
                }
            }
        } else {
            self.addstack.push(program.start);
            self.drain_addstack_into_nlist(program);
        }

        std::mem::swap(&mut self.clist, &mut self.nlist);
        matched
    }

    fn drain_addstack_into_clist(&mut self, program: &AnchorNfaProgram) {
        while let Some(idx) = self.addstack.pop() {
            if idx == StateIdx::NONE {
                continue;
            }
            let i = idx.idx();
            if self.lastlist[i] == self.listid {
                continue;
            }
            self.lastlist[i] = self.listid;
            match program.states[i] {
                State::Split { out, out1 } => {
                    self.addstack.push(out1);
                    self.addstack.push(out);
                }
                State::Match => {}
                _ => {
                    self.clist.push(idx);
                }
            }
        }
    }

    fn drain_addstack_into_nlist(&mut self, program: &AnchorNfaProgram) {
        while let Some(idx) = self.addstack.pop() {
            if idx == StateIdx::NONE {
                continue;
            }
            let i = idx.idx();
            if self.lastlist[i] == self.listid {
                continue;
            }
            self.lastlist[i] = self.listid;
            match program.states[i] {
                State::Split { out, out1 } => {
                    self.addstack.push(out1);
                    self.addstack.push(out);
                }
                State::Match => {}
                _ => {
                    self.nlist.push(idx);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Bounded-Gap Cache
// ---------------------------------------------------------------------------

/// Reusable scratch for the bounded-gap matcher.
///
/// Stored on [`MatcherMemory`](crate::MatcherMemory) and reused across
/// matches, analogous to `Tier1DfaCache` / `Tier2DfaCache` etc.
/// Allocations (Vecs, VecDeques) retain capacity across matches.
#[derive(Debug)]
pub(crate) struct BoundedGapCache {
    runner_scratch: Vec<RunnerScratch>,
    gap_stages: Vec<GapStageState>,
    tail_state: Option<TailGapState>,
    anchor_active: Vec<bool>,
}

impl BoundedGapCache {
    pub(crate) fn new() -> Self {
        BoundedGapCache {
            runner_scratch: Vec::new(),
            gap_stages: Vec::new(),
            tail_state: None,
            anchor_active: Vec::new(),
        }
    }

    /// Prepare the cache for a new match with the given plan.
    /// Resizes scratch to match the plan dimensions and clears
    /// all queues, retaining heap allocations.
    pub(crate) fn prepare(&mut self, plan: &BoundedGapPlan) {
        let n = plan.anchors.len();

        // Resize runner scratch.
        self.runner_scratch.truncate(n);
        for (i, anchor) in plan.anchors.iter().enumerate() {
            let num_states = match &anchor.program.engine {
                AnchorEngine::Tier0(p) => p.states.len(),
                _ => 16, // placeholder
            };
            if i < self.runner_scratch.len() {
                let scratch = &mut self.runner_scratch[i];
                scratch.reset();
                scratch.lastlist.resize(num_states, 0);
            } else {
                self.runner_scratch.push(RunnerScratch::new(num_states));
            }
        }

        // Seed start closures.
        for (i, anchor) in plan.anchors.iter().enumerate() {
            let program = match &anchor.program.engine {
                AnchorEngine::Tier0(p) => p,
                _ => continue,
            };
            self.runner_scratch[i].seed_start(program);
        }

        // Resize gap stage queues.
        self.gap_stages.truncate(plan.interior_gaps.len());
        for stage in &mut self.gap_stages {
            stage.accepted_prev_ends.clear();
            stage.bad_positions.clear();
        }
        while self.gap_stages.len() < plan.interior_gaps.len() {
            self.gap_stages.push(GapStageState {
                accepted_prev_ends: VecDeque::new(),
                bad_positions: VecDeque::new(),
            });
        }

        // Tail state.
        if plan.tail_gap.is_some() {
            if let Some(ref mut ts) = self.tail_state {
                ts.accepted_final_anchor_ends.clear();
                ts.bad_positions.clear();
            } else {
                self.tail_state = Some(TailGapState {
                    accepted_final_anchor_ends: VecDeque::new(),
                    bad_positions: VecDeque::new(),
                });
            }
        } else {
            self.tail_state = None;
        }

        // Anchor activation.
        self.anchor_active.clear();
        self.anchor_active.resize(n, false);
        self.anchor_active[0] = true;
    }
}

/// Runtime matcher for bounded-gap chains.
///
/// Coordinates N anchor runners with queue-based gap validation to
/// determine whether an `Anchor (Gap Anchor)* Gap?` chain matches
/// anywhere in the input.  Produces an existence-only result (no
/// match positions or captures).
///
/// Borrows all reusable scratch from a [`BoundedGapCache`] stored on
/// [`MatcherMemory`](crate::MatcherMemory).
pub struct BoundedGapMatcher<'a> {
    /// The compiled plan.
    plan: &'a BoundedGapPlan,
    /// Reusable scratch (borrowed from MatcherMemory).
    cache: &'a mut BoundedGapCache,
    /// Current absolute byte position.
    position: u64,
    /// Existence result: set once a full chain match is confirmed.
    matched: bool,
    /// Per-position flag: the chain is fully satisfied at the current
    /// byte.  When `end_anchor == EndOfInput`, this is a candidate
    /// that becomes `matched` only in `finish()`.
    matched_here: bool,
}

impl<'a> BoundedGapMatcher<'a> {
    /// Create a matcher from a plan and a prepared cache.
    pub(crate) fn new(plan: &'a BoundedGapPlan, cache: &'a mut BoundedGapCache) -> Self {
        cache.prepare(plan);
        BoundedGapMatcher {
            plan,
            cache,
            position: 0,
            matched: false,
            matched_here: false,
        }
    }

    /// Process a chunk of input.
    pub(crate) fn chunk(&mut self, input: &[u8]) {
        match self.plan.anchors[0].program.prefilter {
            Prefilter::None => self.chunk_no_prefilter(input),
            Prefilter::Memchr1(b1) => {
                self.chunk_prefilter(input, |hay| memchr::memchr(b1, hay));
            }
            Prefilter::Memchr2(b1, b2) => {
                self.chunk_prefilter(input, |hay| memchr::memchr2(b1, b2, hay));
            }
            Prefilter::Memchr3(b1, b2, b3) => {
                self.chunk_prefilter(input, |hay| memchr::memchr3(b1, b2, b3, hay));
            }
            Prefilter::Memclass(t) => {
                self.chunk_prefilter(input, |hay| crate::memclass::memclass(&t, hay));
            }
            Prefilter::Range(lo, hi) => {
                self.chunk_prefilter(input, |hay| crate::memrange::memrange(lo, hi, hay));
            }
        }
    }

    /// No prefilter: process every byte.
    fn chunk_no_prefilter(&mut self, input: &[u8]) {
        for &byte in input {
            if self.matched {
                return;
            }
            self.step(byte);
        }
    }

    /// Prefilter path: when the matcher is quiescent (all gap stage
    /// queues empty, only anchor[0] active), use the prefilter to skip
    /// non-candidate bytes at SIMD speed.
    fn chunk_prefilter(&mut self, input: &[u8], finder: impl Fn(&[u8]) -> Option<usize>) {
        let mut i = 0;
        while i < input.len() {
            if self.matched {
                return;
            }
            if self.is_quiescent() {
                if let Some(offset) = finder(&input[i..]) {
                    // Advance position past the skipped bytes.
                    self.position += offset as u64;
                    i += offset;
                } else {
                    // No candidate in remaining input.
                    self.position += (input.len() - i) as u64;
                    return;
                }
            }
            self.step(input[i]);
            i += 1;
        }
    }

    /// Check whether the matcher is in quiescent state: all gap stage
    /// queues are empty, only anchor[0] is active, and anchor[0]'s NFA
    /// is at start configuration.  In this state, anchor[0]'s prefilter
    /// can safely skip non-candidate bytes.
    fn is_quiescent(&self) -> bool {
        // Anchor[0]'s NFA must be at start config (no partial match
        // in progress).
        let program = match &self.plan.anchors[0].program.engine {
            AnchorEngine::Tier0(p) => p,
            _ => return false,
        };
        let scratch = &self.cache.runner_scratch[0];
        if scratch.clist.len() != program.start_closure.len() {
            return false;
        }
        // Any gap stage has accepted ends → not quiescent.
        for stage in &self.cache.gap_stages {
            if !stage.accepted_prev_ends.is_empty() {
                return false;
            }
        }
        // Tail state has accepted ends → not quiescent.
        if let Some(ref ts) = self.cache.tail_state
            && !ts.accepted_final_anchor_ends.is_empty()
        {
            return false;
        }
        true
    }

    /// Signal end-of-input.  Resolves `EndOfInput` candidates.
    pub(crate) fn finish(self) -> bool {
        if self.matched {
            return true;
        }
        if self.plan.end_anchor == EndAnchorKind::EndOfInput {
            return self.matched_here;
        }
        false
    }

    /// Return whether a match has been found.
    pub(crate) fn ismatch(&self) -> bool {
        self.matched
    }

    /// Process one input byte.
    fn step(&mut self, byte: u8) {
        let p = self.position;
        self.position += 1;
        self.matched_here = false;

        // -- Step 1: Feed byte to active anchor runners. -----------------
        let n = self.plan.anchors.len();
        let mut anchor_matched = [false; 16]; // V1 chains are small
        debug_assert!(n <= 16, "chain too long for fixed buffer");
        #[allow(clippy::needless_range_loop)] // indexes into 4 parallel arrays
        for i in 0..n {
            if self.cache.anchor_active[i] {
                let program = match &self.plan.anchors[i].program.engine {
                    AnchorEngine::Tier0(p) => p,
                    _ => unimplemented!("only Tier 0 anchor runners in V1"),
                };
                anchor_matched[i] = self.cache.runner_scratch[i].step(program, byte);
            }
        }

        // -- Step 2: Update bad-byte queues. -----------------------------
        for (i, gap) in self.plan.interior_gaps.iter().enumerate() {
            if let GapPredicate::ByteClass(bits) = gap.predicate
                && !bits.contains(byte)
            {
                self.cache.gap_stages[i].bad_positions.push_back(p);
            }
        }
        // Terminal gap: current byte IS part of the gap (ordering rule).
        if let Some(ref mut tail_state) = self.cache.tail_state
            && let Some(tail_gap) = &self.plan.tail_gap
            && let GapPredicate::ByteClass(bits) = tail_gap.predicate
            && !bits.contains(byte)
        {
            tail_state.bad_positions.push_back(p);
        }

        // -- Step 3: Process anchor matches left-to-right. ---------------
        self.process_anchor_events(p, &anchor_matched[..n]);

        // -- Step 5: Terminal-gap success checks. ------------------------
        if self.plan.tail_gap.is_some() {
            self.check_terminal_gap(p);
        }

        // -- Step 4: Expire dead queue entries. --------------------------
        self.expire_queues(p);

        // -- Step 6: Update matched flag. --------------------------------
        if self.matched_here && self.plan.end_anchor == EndAnchorKind::None {
            self.matched = true;
        }
    }

    /// Process anchor match events through the chain.
    fn process_anchor_events(&mut self, p: u64, anchor_matched: &[bool]) {
        let n = self.plan.anchors.len();

        // Anchor[0] match.
        if anchor_matched[0] {
            // StartOfInput filter: accept only if the anchor starts at 0.
            let anchor_len = self.anchor_length(0);
            let start_ok = if self.plan.start_anchor == StartAnchorKind::StartOfInput {
                p + 1 == anchor_len
            } else {
                true
            };

            if start_ok {
                if !self.cache.gap_stages.is_empty() {
                    // Push to first interior gap stage.
                    self.cache.gap_stages[0].accepted_prev_ends.push_back(p);
                } else if self.cache.tail_state.is_some() {
                    // Single anchor + tail gap.
                    self.cache
                        .tail_state
                        .as_mut()
                        .unwrap()
                        .accepted_final_anchor_ends
                        .push_back(p);
                }
                // Activate anchor[1] if it exists.
                if n > 1 && !self.cache.anchor_active[1] {
                    self.cache.anchor_active[1] = true;
                }
            }
        }

        // Subsequent anchors.
        for i in 0..self.plan.interior_gaps.len() {
            let anchor_idx = i + 1;
            if !anchor_matched[anchor_idx] {
                continue;
            }
            let anchor_len = self.anchor_length(anchor_idx);
            let s = p + 1 - anchor_len; // start position of this anchor

            // Validate the preceding gap stage.
            let valid = Self::validate_interior_gap(
                &mut self.cache.gap_stages[i],
                &self.plan.interior_gaps[i],
                s,
            );

            if valid {
                let next_stage = i + 1;
                if next_stage < self.cache.gap_stages.len() {
                    // Push to next interior gap stage.
                    self.cache.gap_stages[next_stage]
                        .accepted_prev_ends
                        .push_back(p);
                } else if self.cache.tail_state.is_some() {
                    // Final anchor before tail gap.
                    self.cache
                        .tail_state
                        .as_mut()
                        .unwrap()
                        .accepted_final_anchor_ends
                        .push_back(p);
                } else {
                    // Final anchor, no tail gap → chain complete.
                    self.matched_here = true;
                }
                // Activate the next anchor if it exists.
                let next_anchor = anchor_idx + 1;
                if next_anchor < n && !self.cache.anchor_active[next_anchor] {
                    self.cache.anchor_active[next_anchor] = true;
                }
            }
        }
    }

    /// Validate an interior gap stage: check if any accepted prior end
    /// satisfies the gap constraints for a next-anchor start at `s`.
    ///
    /// Uses the destructive-pop strategy: expired and bad-byte-invalidated
    /// entries are permanently removed (correct due to monotonicity).
    fn validate_interior_gap(stage: &mut GapStageState, gap: &GapPlan, s: u64) -> bool {
        // 1. Pop expired entries (gap too large for max_gap).
        let min_valid_e = s.saturating_sub(1 + gap.max_gap as u64);
        while let Some(&front) = stage.accepted_prev_ends.front() {
            if front < min_valid_e {
                stage.accepted_prev_ends.pop_front();
            } else {
                break;
            }
        }

        // 2. Find most recent bad position before s.
        let bad_before_s = stage.bad_positions.iter().rev().find(|&&b| b < s).copied();

        // 3. Pop entries invalidated by bad bytes.
        if let Some(bad) = bad_before_s {
            while let Some(&front) = stage.accepted_prev_ends.front() {
                if front < bad {
                    stage.accepted_prev_ends.pop_front();
                } else {
                    break;
                }
            }
        }

        // 4. Check if front satisfies min_gap.
        if let Some(&e) = stage.accepted_prev_ends.front() {
            if let Some(gap_len) = s.checked_sub(e + 1) {
                gap_len >= gap.min_gap as u64
            } else {
                false // s <= e, anchors overlap
            }
        } else {
            false // no valid entries
        }
    }

    /// Check terminal gap validity at position `p`.
    fn check_terminal_gap(&mut self, p: u64) {
        let tail_gap = match &self.plan.tail_gap {
            Some(g) => g,
            None => return,
        };
        let tail_state = match &mut self.cache.tail_state {
            Some(s) => s,
            None => return,
        };

        // Expire entries where tail_len > max_gap.
        while let Some(&e) = tail_state.accepted_final_anchor_ends.front() {
            if p.saturating_sub(e) > tail_gap.max_gap as u64 {
                tail_state.accepted_final_anchor_ends.pop_front();
            } else {
                break;
            }
        }

        // Expire bad positions older than the oldest accepted end.
        if let Some(&oldest_e) = tail_state.accepted_final_anchor_ends.front() {
            while let Some(&front_b) = tail_state.bad_positions.front() {
                if front_b <= oldest_e {
                    tail_state.bad_positions.pop_front();
                } else {
                    break;
                }
            }
        }

        // Pop entries invalidated by bad bytes.
        // After expiry, front of bad_positions (if any) is the earliest
        // bad byte > oldest_e.  Pop accepted ends where e < front_b.
        if let Some(&front_b) = tail_state.bad_positions.front() {
            while let Some(&e) = tail_state.accepted_final_anchor_ends.front() {
                if e < front_b {
                    tail_state.accepted_final_anchor_ends.pop_front();
                } else {
                    break;
                }
            }
        }

        // Check if the front entry satisfies min_gap.
        if let Some(&e) = tail_state.accepted_final_anchor_ends.front() {
            let tail_len = p - e;
            if tail_len >= tail_gap.min_gap as u64 {
                self.matched_here = true;
            }
        }
    }

    /// Expire stale entries from interior gap stage queues.
    fn expire_queues(&mut self, p: u64) {
        for (i, gap) in self.plan.interior_gaps.iter().enumerate() {
            // Compute expiry threshold before borrowing the stage mutably.
            let next_anchor_len = match self.plan.anchors[i + 1].length_info {
                AnchorLengthInfo::Fixed(len) => len as u64,
            };
            let stage = &mut self.cache.gap_stages[i];

            // Expire accepted ends that are too old.
            // An entry `e` is stale when no future anchor start can
            // form a gap within max_gap: e < p - max_gap - max_anchor_len.
            let expiry = p
                .saturating_sub(gap.max_gap as u64)
                .saturating_sub(next_anchor_len);
            while let Some(&front) = stage.accepted_prev_ends.front() {
                if front < expiry {
                    stage.accepted_prev_ends.pop_front();
                } else {
                    break;
                }
            }

            // Expire bad positions older than the oldest accepted end.
            if let Some(&oldest_e) = stage.accepted_prev_ends.front() {
                while let Some(&front_b) = stage.bad_positions.front() {
                    if front_b <= oldest_e {
                        stage.bad_positions.pop_front();
                    } else {
                        break;
                    }
                }
            } else {
                // No accepted ends → all bad positions are irrelevant.
                stage.bad_positions.clear();
            }
        }
    }

    /// Get the fixed byte length of anchor `i`.
    fn anchor_length(&self, i: usize) -> u64 {
        match self.plan.anchors[i].length_info {
            AnchorLengthInfo::Fixed(len) => len as u64,
        }
    }
}

impl std::fmt::Debug for BoundedGapMatcher<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoundedGapMatcher")
            .field("position", &self.position)
            .field("matched", &self.matched)
            .field("matched_here", &self.matched_here)
            .field("anchors", &self.plan.anchors.len())
            .field("interior_gaps", &self.plan.interior_gaps.len())
            .field("tail_gap", &self.plan.tail_gap.is_some())
            .finish()
    }
}

impl std::fmt::Display for BoundedGapMatcher<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BoundedGap(pos={}, matched={}, here={})",
            self.position, self.matched, self.matched_here
        )
    }
}

// ---------------------------------------------------------------------------
// Compile-Time Specialisation Probe
// ---------------------------------------------------------------------------

/// Attempt to detect a Version 1 bounded-gap chain in the given HIR.
///
/// Returns `Some(plan)` if the pattern matches the chain shape
/// `Anchor (Gap Anchor)* Gap?` with Version 1 restrictions:
///
/// - Fixed-length, assertion-free anchors
/// - One-byte finite gap predicates
/// - No leading gap
/// - No internal assertions (whole-pattern `^`/`$` are hoisted)
///
/// Returns `None` for any pattern that does not qualify — the caller
/// should continue with normal compilation.
pub(crate) fn try_build_bounded_gap_plan(
    hir: &Hir,
    config: &RegexConfig,
) -> Option<BoundedGapPlan> {
    // Step 1: Flatten top-level concat.
    let pieces = flatten_top_level_concat(hir);
    if pieces.is_empty() {
        return None;
    }

    // Step 2: Hoist whole-pattern ^ / $.
    let mut start_anchor = StartAnchorKind::None;
    let mut end_anchor = EndAnchorKind::None;
    let mut start = 0;
    let mut end = pieces.len();

    // Leading ^
    if let Some(HirKind::Look(look)) = pieces.first().map(|h| h.kind()) {
        match look {
            regex_syntax::hir::Look::Start => {
                start_anchor = StartAnchorKind::StartOfInput;
                start = 1;
            }
            // StartLF / StartCRLF → reject in V1.
            regex_syntax::hir::Look::StartLF | regex_syntax::hir::Look::StartCRLF => {
                return None;
            }
            _ => {}
        }
    }
    // Trailing $
    if end > start
        && let Some(HirKind::Look(look)) = pieces.get(end - 1).map(|h| h.kind())
    {
        match look {
            regex_syntax::hir::Look::End => {
                end_anchor = EndAnchorKind::EndOfInput;
                end -= 1;
            }
            // EndLF / EndCRLF → reject in V1.
            regex_syntax::hir::Look::EndLF | regex_syntax::hir::Look::EndCRLF => {
                return None;
            }
            _ => {}
        }
    }

    let pieces = &pieces[start..end];
    if pieces.is_empty() {
        return None;
    }

    // Step 3: Partition into alternating anchor/gap segments.
    //
    // Each piece that classify_gap() recognises is a gap.  Everything
    // else is anchor material.  Consecutive anchor-material pieces
    // concatenate into a single anchor.
    let mut anchor_pieces: Vec<Vec<&Hir>> = Vec::new();
    let mut gaps: Vec<GapPlan> = Vec::new();
    let mut current_anchor: Vec<&Hir> = Vec::new();
    let mut last_was_gap = false;

    for piece in pieces {
        if let Some((min, max, pred)) = classify_gap(piece) {
            // Normalise {0,0} → skip (merge adjacent anchors).
            if min == 0 && max == 0 {
                continue;
            }
            if current_anchor.is_empty() && anchor_pieces.is_empty() {
                // Leading gap → reject in V1.
                return None;
            }
            if last_was_gap {
                // Adjacent gaps — the HIR should already have same-body
                // repetitions merged by optimize(hir, merge_repetitions=true)
                // before the probe runs.  If we still see adjacent gaps,
                // they have different predicates and can't be merged.
                return None;
            }
            // Close the current anchor.
            anchor_pieces.push(std::mem::take(&mut current_anchor));
            gaps.push(GapPlan {
                min_gap: min,
                max_gap: max,
                predicate: pred,
            });
            last_was_gap = true;
        } else {
            // Anchor material.
            if let HirKind::Empty = piece.kind() {
                // Skip empty nodes that survived optimisation.
                continue;
            }
            current_anchor.push(piece);
            last_was_gap = false;
        }
    }

    // Close trailing anchor (if any).
    if !current_anchor.is_empty() {
        anchor_pieces.push(std::mem::take(&mut current_anchor));
    }

    // Determine chain shape.
    let (interior_gaps, tail_gap) = if anchor_pieces.len() == gaps.len() + 1 {
        // Anchor-terminated: A (G A)*
        (gaps, None)
    } else if anchor_pieces.len() == gaps.len() && !gaps.is_empty() {
        // Tail-gap: A (G A)* G
        let tail = gaps.pop().unwrap();
        (gaps, Some(tail))
    } else {
        // Invalid shape.
        return None;
    };

    // Must have at least one gap (otherwise no point in the specialisation).
    if interior_gaps.is_empty() && tail_gap.is_none() {
        return None;
    }

    // Step 4: Validate Version 1 anchor restrictions and build anchor plans.
    let mut anchor_plans: Vec<AnchorPlan> = Vec::with_capacity(anchor_pieces.len());
    for pieces in &anchor_pieces {
        // Compute aggregate fixed length.
        let mut total_len: u16 = 0;
        for piece in pieces {
            let fl = compute_fixed_length(piece)?;
            total_len = total_len.checked_add(fl.len)?;
        }
        if total_len == 0 {
            return None; // Empty anchor.
        }

        // All pieces must be assertion-free.
        for piece in pieces {
            if !is_assertion_free(piece) {
                return None;
            }
        }

        // Compile the anchor fragment into a real NFA program.
        let program = compile_anchor_program(pieces, config)?;

        anchor_plans.push(AnchorPlan {
            program,
            length_info: AnchorLengthInfo::Fixed(total_len),
        });
    }

    Some(BoundedGapPlan {
        anchors: anchor_plans.into_boxed_slice(),
        interior_gaps: interior_gaps.into_boxed_slice(),
        tail_gap,
        start_anchor,
        end_anchor,
    })
}

// ---------------------------------------------------------------------------
// HIR Flattening
// ---------------------------------------------------------------------------

/// Strip non-semantic HIR wrappers (captures, singleton concats).
///
/// Peels away `Capture`, single-child `Concat`, and single-child
/// `Alternation` to expose the structurally relevant node underneath.
fn strip_wrappers(hir: &Hir) -> &Hir {
    match hir.kind() {
        HirKind::Capture(cap) => strip_wrappers(&cap.sub),
        HirKind::Concat(children) if children.len() == 1 => strip_wrappers(&children[0]),
        HirKind::Alternation(children) if children.len() == 1 => strip_wrappers(&children[0]),
        _ => hir,
    }
}

/// Flatten a top-level concat HIR into a linear sequence of pieces.
///
/// Non-semantic wrappers (captures, singleton concats) are stripped
/// from the outermost level.  The result is a flat view suitable for
/// partitioning into alternating anchor/gap segments.
///
/// If the outermost node (after stripping) is not a `Concat`, a
/// single-element vector is returned.
pub(crate) fn flatten_top_level_concat(hir: &Hir) -> Vec<&Hir> {
    let stripped = strip_wrappers(hir);
    match stripped.kind() {
        HirKind::Concat(children) => children.iter().collect(),
        _ => vec![stripped],
    }
}

// ---------------------------------------------------------------------------
// Gap Body Recognition
// ---------------------------------------------------------------------------

/// Recognize whether an HIR fragment is a Version 1 gap body: a single
/// byte-consuming predicate atom.
///
/// Returns `Some(GapPredicate)` if the body qualifies, `None` otherwise.
///
/// Qualifying bodies:
/// - `.` (wildcard)
/// - `[\s\S]`, `[^]` (all-bytes class)
/// - `[^/]`, `\s`, `[A-Za-z0-9_]` (specific byte class)
/// - Single-byte literal like `a`
///
/// Non-qualifying bodies:
/// - Multi-byte literals (`ab`)
/// - Bodies containing assertions (`\b`)
/// - Alternations, repetitions, or other compound structures
pub(crate) fn classify_gap_body(hir: &Hir) -> Option<GapPredicate> {
    match hir.kind() {
        // Single-byte literal → byte class with one byte set.
        HirKind::Literal(lit) if lit.0.len() == 1 => {
            let mut table = ByteClass::NONE;
            table.0[lit.0[0] as usize] = true;
            Some(GapPredicate::ByteClass(table.to_bits()))
        }
        // Byte or Unicode class → check for wildcard or specific class.
        HirKind::Class(class) => {
            let table = ByteClass::from_hir_class(class)?;
            if table.is_all() {
                Some(GapPredicate::Any)
            } else {
                Some(GapPredicate::ByteClass(table.to_bits()))
            }
        }
        // Capture is just a wrapper — recurse.
        HirKind::Capture(cap) => classify_gap_body(&cap.sub),
        // Anything else is not a valid gap body.
        _ => None,
    }
}

/// Recognize whether a top-level HIR piece is a Version 1 gap: a
/// bounded repetition whose body is a single-byte predicate.
///
/// Returns `Some((min, max, class))` if the piece is a valid gap,
/// `None` otherwise.
///
/// Requires:
/// - `Repetition` with finite `max`
/// - Body qualifies under [`classify_gap_body`]
pub(crate) fn classify_gap(hir: &Hir) -> Option<(u32, u32, GapPredicate)> {
    let hir = strip_wrappers(hir);
    match hir.kind() {
        HirKind::Repetition(rep) => {
            let max = rep.max?; // None = unbounded → not a gap
            let body_class = classify_gap_body(&rep.sub)?;
            Some((rep.min, max, body_class))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Fixed-Length Detection
// ---------------------------------------------------------------------------

/// Compute whether an HIR fragment has exact fixed byte length.
///
/// Uses the `minimum_len` / `maximum_len` properties computed by
/// `regex-syntax`.  Returns `None` if the fragment has variable length,
/// is unbounded, matches nothing, or has length 0 (empty).
///
/// Note: assertions (`Look` nodes) have zero length and do not affect
/// the byte count.  A fragment like `\bfoo` has fixed length 3 (from
/// the 3-byte literal) — the assertion-free check is separate.
pub(crate) fn compute_fixed_length(hir: &Hir) -> Option<FixedLengthInfo> {
    let props = hir.properties();
    let min = props.minimum_len()?; // None if matches nothing
    let max = props.maximum_len()?; // None if unbounded
    if min == max && min > 0 {
        let len = u16::try_from(min).ok()?; // reject if > 65535
        Some(FixedLengthInfo { len })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Assertion-Free Check
// ---------------------------------------------------------------------------

/// Check whether an HIR fragment is free of all assertions (`Look` nodes).
///
/// Version 1 anchors must be assertion-free.  Top-level `^` / `$` are
/// handled separately by hoisting into plan flags before individual
/// anchor fragments are checked.
pub(crate) fn is_assertion_free(hir: &Hir) -> bool {
    hir.properties().look_set().is_empty()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_hir;

    /// Helper: parse a pattern to HIR using the same settings as the engine
    /// (without repetition merging, matching `parse_hir` behavior).
    fn hir(pattern: &str) -> Hir {
        parse_hir(pattern).expect("test pattern should parse")
    }

    /// Helper: parse a pattern to HIR with repetition merging enabled,
    /// matching what the bounded-gap probe sees in `build()`.
    fn hir_merged(pattern: &str) -> Hir {
        let h = parse_hir(pattern).expect("test pattern should parse");
        crate::hir_optimize::optimize(h, true)
    }

    // -- flatten_top_level_concat -------------------------------------------

    #[test]
    fn test_flatten_simple_concat() {
        // foo.{0,10}bar → 3 pieces: Literal("foo"), Repetition, Literal("bar")
        let h = hir(r"foo.{0,10}bar");
        let pieces = flatten_top_level_concat(&h);
        assert_eq!(pieces.len(), 3, "expected 3 concat pieces");
    }

    #[test]
    fn test_flatten_wrapped_in_capture() {
        // (foo.{0,10}bar) → same 3 pieces after stripping capture
        let h = hir(r"(foo.{0,10}bar)");
        let pieces = flatten_top_level_concat(&h);
        assert_eq!(pieces.len(), 3, "capture wrapper should be stripped");
    }

    #[test]
    fn test_flatten_single_literal() {
        // "foo" → 1 piece (the whole literal, or 3 single-byte literals)
        let h = hir(r"foo");
        let pieces = flatten_top_level_concat(&h);
        // After HIR optimization, "foo" is a single Literal node with 3 bytes.
        assert_eq!(pieces.len(), 1, "single literal should be one piece");
    }

    #[test]
    fn test_flatten_alternation_at_top() {
        // "foo|bar" → 1 piece (the alternation itself)
        let h = hir(r"foo|bar");
        let pieces = flatten_top_level_concat(&h);
        assert_eq!(pieces.len(), 1, "alternation should be one piece");
    }

    #[test]
    fn test_flatten_multi_gap_chain() {
        // foo.{0,10}bar.{0,20}baz → 5 pieces
        let h = hir(r"foo.{0,10}bar.{0,20}baz");
        let pieces = flatten_top_level_concat(&h);
        assert_eq!(
            pieces.len(),
            5,
            "expected 5 concat pieces for 3-anchor chain"
        );
    }

    // -- classify_gap_body --------------------------------------------------

    #[test]
    fn test_gap_body_wildcard() {
        // "." with dot_matches_new_line(true) → Any
        let h = hir(r".");
        assert_eq!(classify_gap_body(&h), Some(GapPredicate::Any));
    }

    #[test]
    fn test_gap_body_all_bytes_class() {
        // [\s\S] should match all bytes → Any
        let h = hir(r"[\s\S]");
        assert_eq!(classify_gap_body(&h), Some(GapPredicate::Any));
    }

    #[test]
    fn test_gap_body_negated_slash() {
        // [^/] → ByteClass with '/' excluded
        let h = hir(r"[^/]");
        match classify_gap_body(&h) {
            Some(GapPredicate::ByteClass(bits)) => {
                assert!(!bits.contains(b'/'), "'/' should not be in [^/] class");
                assert!(bits.contains(b'a'), "'a' should be in [^/] class");
            }
            other => panic!("expected ByteClass, got {:?}", other),
        }
    }

    #[test]
    fn test_gap_body_whitespace() {
        // \s → ByteClass with whitespace bytes
        let h = hir(r"\s");
        match classify_gap_body(&h) {
            Some(GapPredicate::ByteClass(bits)) => {
                assert!(bits.contains(b' '), "space should be in \\s class");
                assert!(bits.contains(b'\t'), "tab should be in \\s class");
                assert!(!bits.contains(b'a'), "'a' should not be in \\s class");
            }
            other => panic!("expected ByteClass, got {:?}", other),
        }
    }

    #[test]
    fn test_gap_body_single_byte_literal() {
        // "a" → ByteClass with just 'a'
        let h = hir(r"a");
        match classify_gap_body(&h) {
            Some(GapPredicate::ByteClass(bits)) => {
                assert!(bits.contains(b'a'), "'a' should be set");
                assert!(!bits.contains(b'b'), "'b' should not be set");
            }
            other => panic!("expected ByteClass, got {:?}", other),
        }
    }

    #[test]
    fn test_gap_body_multi_byte_literal_rejected() {
        // "ab" → None (not a single atom)
        let h = hir(r"ab");
        assert_eq!(classify_gap_body(&h), None);
    }

    #[test]
    fn test_gap_body_assertion_rejected() {
        // \b → None
        let h = hir(r"\b");
        assert_eq!(classify_gap_body(&h), None);
    }

    #[test]
    fn test_gap_body_simple_alternation_lowered_to_class() {
        // a|b → regex-syntax normalizes to [ab] → ByteClass
        let h = hir(r"a|b");
        match classify_gap_body(&h) {
            Some(GapPredicate::ByteClass(bits)) => {
                assert!(bits.contains(b'a'));
                assert!(bits.contains(b'b'));
                assert!(!bits.contains(b'c'));
            }
            other => panic!("expected ByteClass for a|b, got {:?}", other),
        }
    }

    #[test]
    fn test_gap_body_multi_byte_alternation_rejected() {
        // (ab|cd) → alternation of multi-byte literals, not a single atom
        let h = hir(r"ab|cd");
        assert_eq!(classify_gap_body(&h), None);
    }

    #[test]
    fn test_gap_body_word_class() {
        // \w → ByteClass
        let h = hir(r"\w");
        match classify_gap_body(&h) {
            Some(GapPredicate::ByteClass(bits)) => {
                assert!(bits.contains(b'a'));
                assert!(bits.contains(b'Z'));
                assert!(bits.contains(b'0'));
                assert!(bits.contains(b'_'));
                assert!(!bits.contains(b' '));
            }
            other => panic!("expected ByteClass, got {:?}", other),
        }
    }

    // -- classify_gap -------------------------------------------------------

    #[test]
    fn test_classify_gap_bounded_wildcard() {
        // .{0,100} → (0, 100, Any)
        let h = hir(r".{0,100}");
        assert_eq!(classify_gap(&h), Some((0, 100, GapPredicate::Any)));
    }

    #[test]
    fn test_classify_gap_exact_count() {
        // .{254} → (254, 254, Any)
        let h = hir(r".{254}");
        assert_eq!(classify_gap(&h), Some((254, 254, GapPredicate::Any)));
    }

    #[test]
    fn test_classify_gap_constrained() {
        // [^/]{0,20} → (0, 20, ByteClass(...))
        let h = hir(r"[^/]{0,20}");
        match classify_gap(&h) {
            Some((0, 20, GapPredicate::ByteClass(bits))) => {
                assert!(!bits.contains(b'/'));
                assert!(bits.contains(b'a'));
            }
            other => panic!("expected constrained gap, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_gap_unbounded_rejected() {
        // .* → None (unbounded)
        let h = hir(r".*");
        assert_eq!(classify_gap(&h), None);
    }

    #[test]
    fn test_classify_gap_multi_byte_body_rejected() {
        // (ab){0,10} → None (not a single-byte body)
        let h = hir(r"(ab){0,10}");
        assert_eq!(classify_gap(&h), None);
    }

    #[test]
    fn test_classify_gap_zero_length_optimized_away() {
        // .{0,0} → the HIR optimizer collapses this to Empty, so
        // classify_gap correctly returns None.  The {0,0} normalization
        // step in the detection pipeline handles this at the chain level.
        let h = hir(r".{0,0}");
        assert_eq!(classify_gap(&h), None);
    }

    // -- compute_fixed_length -----------------------------------------------

    #[test]
    fn test_fixed_length_literal() {
        let h = hir(r"foo");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 3 }));
    }

    #[test]
    fn test_fixed_length_alternation_same_lengths() {
        // (abc|def) → fixed length 3
        let h = hir(r"abc|def");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 3 }));
    }

    #[test]
    fn test_fixed_length_alternation_different_lengths() {
        // (abc|de) → None
        let h = hir(r"abc|de");
        assert_eq!(compute_fixed_length(&h), None);
    }

    #[test]
    fn test_fixed_length_exact_repetition() {
        // a{3} → fixed length 3
        let h = hir(r"a{3}");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 3 }));
    }

    #[test]
    fn test_fixed_length_variable_repetition() {
        // a{2,4} → None
        let h = hir(r"a{2,4}");
        assert_eq!(compute_fixed_length(&h), None);
    }

    #[test]
    fn test_fixed_length_unbounded() {
        // a+ → None
        let h = hir(r"a+");
        assert_eq!(compute_fixed_length(&h), None);
    }

    #[test]
    fn test_fixed_length_empty() {
        // Empty pattern → None (length 0 is rejected as non-empty is required)
        let h = hir(r"");
        assert_eq!(compute_fixed_length(&h), None);
    }

    #[test]
    fn test_fixed_length_complex_anchor() {
        // foo\d{4} → fixed length 7
        let h = hir(r"foo\d{4}");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 7 }));
    }

    #[test]
    fn test_fixed_length_case_insensitive() {
        // (?i)file → fixed length 4
        let h = hir(r"(?i)file");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 4 }));
    }

    #[test]
    fn test_fixed_length_class_repetition() {
        // [A-Z]{2}admin → fixed length 7
        let h = hir(r"[A-Z]{2}admin");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 7 }));
    }

    #[test]
    fn test_fixed_length_with_assertion() {
        // \bfoo → fixed length 3 (assertion is zero-width)
        // Note: this has fixed length but is NOT assertion-free.
        let h = hir(r"\bfoo");
        assert_eq!(compute_fixed_length(&h), Some(FixedLengthInfo { len: 3 }));
    }

    // -- is_assertion_free --------------------------------------------------

    #[test]
    fn test_assertion_free_literal() {
        let h = hir(r"foo");
        assert!(is_assertion_free(&h));
    }

    #[test]
    fn test_assertion_free_alternation() {
        let h = hir(r"GET|POST");
        assert!(is_assertion_free(&h));
    }

    #[test]
    fn test_assertion_free_class_repetition() {
        let h = hir(r"foo\d{4}");
        assert!(is_assertion_free(&h));
    }

    #[test]
    fn test_assertion_free_case_insensitive() {
        let h = hir(r"(?i)file");
        assert!(is_assertion_free(&h));
    }

    #[test]
    fn test_not_assertion_free_word_boundary() {
        let h = hir(r"\bfoo");
        assert!(!is_assertion_free(&h));
    }

    #[test]
    fn test_not_assertion_free_start_anchor() {
        let h = hir(r"^foo");
        assert!(!is_assertion_free(&h));
    }

    #[test]
    fn test_not_assertion_free_end_anchor() {
        let h = hir(r"foo$");
        assert!(!is_assertion_free(&h));
    }

    #[test]
    fn test_not_assertion_free_word_boundary_negated() {
        let h = hir(r"foo\B");
        assert!(!is_assertion_free(&h));
    }

    /// Default config for probe tests.
    fn default_config() -> RegexConfig {
        RegexConfig::default()
    }

    /// Helper: run the probe with default config.
    fn probe(h: &Hir) -> Option<BoundedGapPlan> {
        try_build_bounded_gap_plan(h, &default_config())
    }

    /// Helper: run the probe with merged HIR and default config.
    fn probe_merged(h: &Hir) -> Option<BoundedGapPlan> {
        let merged = crate::hir_optimize::optimize(h.clone(), true);
        try_build_bounded_gap_plan(&merged, &default_config())
    }

    // -- try_build_bounded_gap_plan: recognition ----------------------------

    #[test]
    fn test_probe_simple_two_anchor() {
        // foo.{0,10}bar → 2 anchors, 1 interior gap, no tail gap
        let h = hir(r"foo.{0,10}bar");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.anchors.len(), 2);
        assert_eq!(plan.interior_gaps.len(), 1);
        assert!(plan.tail_gap.is_none());
        assert_eq!(plan.interior_gaps[0].min_gap, 0);
        assert_eq!(plan.interior_gaps[0].max_gap, 10);
        assert_eq!(plan.interior_gaps[0].predicate, GapPredicate::Any);
        assert_eq!(plan.anchors[0].length_info, AnchorLengthInfo::Fixed(3));
        assert_eq!(plan.anchors[1].length_info, AnchorLengthInfo::Fixed(3));
    }

    #[test]
    fn test_probe_three_anchor_chain() {
        // foo.{0,10}bar.{0,20}baz → 3 anchors, 2 interior gaps
        let h = hir(r"foo.{0,10}bar.{0,20}baz");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.anchors.len(), 3);
        assert_eq!(plan.interior_gaps.len(), 2);
        assert!(plan.tail_gap.is_none());
        assert_eq!(plan.interior_gaps[1].max_gap, 20);
    }

    #[test]
    fn test_probe_trailing_gap() {
        // CWS.{254} → 1 anchor, 0 interior gaps, 1 tail gap
        let h = hir(r"CWS.{254}");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.anchors.len(), 1);
        assert_eq!(plan.interior_gaps.len(), 0);
        let tail = plan.tail_gap.as_ref().expect("should have tail gap");
        assert_eq!(tail.min_gap, 254);
        assert_eq!(tail.max_gap, 254);
        assert_eq!(tail.predicate, GapPredicate::Any);
        assert_eq!(plan.anchors[0].length_info, AnchorLengthInfo::Fixed(3));
    }

    #[test]
    fn test_probe_constrained_trailing_gap() {
        // token[^/]{0,20} → 1 anchor, 0 interior gaps, 1 constrained tail gap
        let h = hir(r"token[^/]{0,20}");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.anchors.len(), 1);
        assert_eq!(plan.interior_gaps.len(), 0);
        let tail = plan.tail_gap.as_ref().expect("should have tail gap");
        assert_eq!(tail.min_gap, 0);
        assert_eq!(tail.max_gap, 20);
        match tail.predicate {
            GapPredicate::ByteClass(bits) => {
                assert!(!bits.contains(b'/'));
                assert!(bits.contains(b'a'));
            }
            _ => panic!("expected constrained predicate"),
        }
    }

    #[test]
    fn test_probe_hoisted_start_anchor() {
        // ^foo.{0,10}bar → StartOfInput hoisted
        let h = hir(r"^foo.{0,10}bar");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.start_anchor, StartAnchorKind::StartOfInput);
        assert_eq!(plan.end_anchor, EndAnchorKind::None);
        assert_eq!(plan.anchors.len(), 2);
    }

    #[test]
    fn test_probe_hoisted_end_anchor() {
        // foo.{0,10}bar$ → EndOfInput hoisted
        let h = hir(r"foo.{0,10}bar$");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.start_anchor, StartAnchorKind::None);
        assert_eq!(plan.end_anchor, EndAnchorKind::EndOfInput);
    }

    #[test]
    fn test_probe_both_anchors_hoisted() {
        // ^foo.{0,10}bar$ → both hoisted
        let h = hir(r"^foo.{0,10}bar$");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.start_anchor, StartAnchorKind::StartOfInput);
        assert_eq!(plan.end_anchor, EndAnchorKind::EndOfInput);
    }

    #[test]
    fn test_probe_exact_gap() {
        // foo.{5}bar → exact gap of 5
        let h = hir(r"foo.{5}bar");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.interior_gaps[0].min_gap, 5);
        assert_eq!(plan.interior_gaps[0].max_gap, 5);
    }

    #[test]
    fn test_probe_case_insensitive_anchor() {
        // (?i)file.{0,10}path → case-insensitive anchors
        let h = hir(r"(?i)file.{0,10}path");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.anchors[0].length_info, AnchorLengthInfo::Fixed(4));
        assert_eq!(plan.anchors[1].length_info, AnchorLengthInfo::Fixed(4));
    }

    // -- try_build_bounded_gap_plan: rejection ------------------------------

    #[test]
    fn test_probe_reject_no_gap() {
        // "foobar" → no gaps, specialisation does not apply
        let h = hir(r"foobar");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_reject_leading_gap() {
        // .{0,20}foo → leading gap, rejected
        let h = hir(r".{0,20}foo");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_reject_unbounded_gap() {
        // foo.*bar → unbounded gap
        let h = hir(r"foo.*bar");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_reject_variable_length_anchor() {
        // foo.{0,10}ba+r → "ba+r" contains unbounded a+ which is
        // anchor material with variable length → rejected
        let h = hir(r"foo.{0,10}ba+r");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_reject_assertion_in_anchor() {
        // \bfoo.{0,10}bar → \b in anchor
        let h = hir(r"\bfoo.{0,10}bar");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_reject_startlf() {
        // (?m:^)foo.{0,10}bar → StartLF, rejected in V1
        let h = hir(r"(?m:^)foo.{0,10}bar");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_reject_endlf() {
        // foo.{0,10}bar(?m:$) → EndLF, rejected in V1
        let h = hir(r"foo.{0,10}bar(?m:$)");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_reject_alternation_whole_chain() {
        // (foo|bar).{0,10}baz → top level is alternation, not concat
        // Actually this IS a concat: (foo|bar) then .{0,10} then baz
        // The first anchor (foo|bar) has fixed length 3, no assertions → OK
        let h = hir(r"(foo|bar).{0,10}baz");
        let plan = probe(&h);
        // This should actually be recognised if (foo|bar) has fixed length 3
        assert!(plan.is_some());
    }

    #[test]
    fn test_probe_reject_top_level_alternation() {
        // foo|bar — alternation at top level, not a concat chain
        let h = hir(r"foo|bar");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_reject_multi_byte_gap_body() {
        // foo(ab){0,10}bar → gap body is multi-byte
        let h = hir(r"foo(ab){0,10}bar");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_adjacent_same_predicate_gaps_merged() {
        // .{0,10}.{0,20} → optimize(merge=true) merges to .{0,30}
        let h = hir_merged(r"foo.{0,10}.{0,20}bar");
        let plan = probe(&h).expect("should recognise merged gaps");
        assert_eq!(plan.interior_gaps.len(), 1);
        assert_eq!(plan.interior_gaps[0].min_gap, 0);
        assert_eq!(plan.interior_gaps[0].max_gap, 30);
        assert_eq!(plan.interior_gaps[0].predicate, GapPredicate::Any);
    }

    #[test]
    fn test_probe_adjacent_different_predicate_gaps_rejected() {
        // foo.{0,10}[^/]{0,20}bar → different predicates, can't merge
        let h = hir_merged(r"foo.{0,10}[^/]{0,20}bar");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_trailing_adjacent_gaps_merged() {
        // foo.{0,10}.{0,20} → optimize(merge=true) merges to .{0,30}
        let h = hir_merged(r"foo.{0,10}.{0,20}");
        let plan = probe(&h).expect("should recognise merged trailing gaps");
        assert_eq!(plan.anchors.len(), 1);
        assert_eq!(plan.interior_gaps.len(), 0);
        let tail = plan.tail_gap.as_ref().expect("should have merged tail gap");
        assert_eq!(tail.min_gap, 0);
        assert_eq!(tail.max_gap, 30);
    }

    #[test]
    fn test_probe_three_adjacent_gaps_merged() {
        // foo.{0,10}.{0,20}.{0,30}bar → optimize(merge=true) merges to .{0,60}
        let h = hir_merged(r"foo.{0,10}.{0,20}.{0,30}bar");
        let plan = probe(&h).expect("should recognise three merged gaps");
        assert_eq!(plan.interior_gaps.len(), 1);
        assert_eq!(plan.interior_gaps[0].max_gap, 60);
    }

    #[test]
    fn test_probe_zero_zero_gap_normalised_away() {
        // foo.{0,0}bar.{0,10}baz → {0,0} gap merged, becomes foo+bar anchor
        // The HIR optimizer should already collapse .{0,0} to Empty,
        // so foo and bar concatenate into one literal.
        let h = hir(r"foo.{0,0}bar.{0,10}baz");
        let plan = probe(&h).expect("should recognise after normalisation");
        // foobar is one anchor (6 bytes), then gap, then baz
        assert_eq!(plan.anchors.len(), 2);
        assert_eq!(plan.anchors[0].length_info, AnchorLengthInfo::Fixed(6));
        assert_eq!(plan.interior_gaps.len(), 1);
    }

    #[test]
    fn test_probe_trailing_zero_gap_dropped() {
        // foo.{0,10}bar.{0,0} → trailing {0,0} dropped by HIR optimizer
        let h = hir(r"foo.{0,10}bar.{0,0}");
        let plan = probe(&h).expect("should recognise");
        // No tail gap since {0,0} was optimised away
        assert!(plan.tail_gap.is_none());
        assert_eq!(plan.anchors.len(), 2);
    }

    #[test]
    fn test_probe_all_gaps_normalised_away_rejects() {
        // foo.{0,0}bar → becomes "foobar", no gaps → rejected
        let h = hir(r"foo.{0,0}bar");
        assert!(probe(&h).is_none());
    }

    #[test]
    fn test_probe_end_anchor_with_trailing_gap() {
        // foo.{0,3}$ → anchor "foo", tail gap {0,3}, EndOfInput
        let h = hir(r"foo.{0,3}$");
        let plan = probe(&h).expect("should recognise");
        assert_eq!(plan.anchors.len(), 1);
        assert_eq!(plan.end_anchor, EndAnchorKind::EndOfInput);
        let tail = plan.tail_gap.as_ref().expect("should have tail gap");
        assert_eq!(tail.min_gap, 0);
        assert_eq!(tail.max_gap, 3);
    }

    // -- AnchorRunner -------------------------------------------------------

    /// Helper: compile an anchor plan for a simple pattern.
    fn anchor_plan(pattern: &str) -> AnchorPlan {
        let h = hir(pattern);
        let pieces = vec![&h];
        let program = compile_anchor_program(&pieces, &default_config())
            .unwrap_or_else(|| panic!("anchor should compile for pattern: {pattern}"));
        AnchorPlan {
            program,
            length_info: AnchorLengthInfo::Fixed(compute_fixed_length(&h).unwrap().len),
        }
    }

    /// Collect all events emitted by scanning the input.
    fn scan_events(plan: &AnchorPlan, input: &[u8]) -> Vec<u64> {
        let mut runner = AnchorRunner::new(plan);
        let mut events = Vec::new();
        runner.scan_chunk(input, 0, &mut |pos| events.push(pos));
        runner.finish_scan(&mut |pos| events.push(pos));
        events
    }

    #[test]
    fn test_runner_literal_match() {
        let plan = anchor_plan("foo");
        // "xxfooxx" → match ending at position 4 (0-indexed: f=2, o=3, o=4)
        let events = scan_events(&plan, b"xxfooxx");
        assert_eq!(events, vec![4]);
    }

    #[test]
    fn test_runner_literal_multiple_matches() {
        let plan = anchor_plan("ab");
        // "ababab" → matches ending at 1, 3, 5
        let events = scan_events(&plan, b"ababab");
        assert_eq!(events, vec![1, 3, 5]);
    }

    #[test]
    fn test_runner_literal_no_match() {
        let plan = anchor_plan("xyz");
        let events = scan_events(&plan, b"abc");
        assert!(events.is_empty());
    }

    #[test]
    fn test_runner_overlapping_matches() {
        let plan = anchor_plan("aa");
        // "aaa" → "aa" ends at position 1, then again at position 2
        let events = scan_events(&plan, b"aaa");
        assert_eq!(events, vec![1, 2]);
    }

    #[test]
    fn test_runner_class_match() {
        let plan = anchor_plan(r"\d\d");
        // "a12b34c" → "12" ends at 2, "34" ends at 5
        let events = scan_events(&plan, b"a12b34c");
        assert_eq!(events, vec![2, 5]);
    }

    #[test]
    fn test_runner_case_insensitive() {
        let plan = anchor_plan(r"(?i)file");
        // "FILE_file" → matches ending at 3, 8
        let events = scan_events(&plan, b"FILE_file");
        assert_eq!(events, vec![3, 8]);
    }

    #[test]
    fn test_runner_alternation_anchor() {
        let plan = anchor_plan(r"GET|PUT");
        // "GET_PUT" → GET ends at 2, PUT ends at 6
        let events = scan_events(&plan, b"GET_PUT");
        assert_eq!(events, vec![2, 6]);
    }

    #[test]
    fn test_runner_cross_chunk() {
        let plan = anchor_plan("foo");
        let mut runner = AnchorRunner::new(&plan);
        let mut events = Vec::new();
        // "fo" in chunk 0, "obar" in chunk 1 → "foo" ends at position 2
        runner.scan_chunk(b"fo", 0, &mut |pos| events.push(pos));
        runner.scan_chunk(b"obar", 2, &mut |pos| events.push(pos));
        runner.finish_scan(&mut |pos| events.push(pos));
        assert_eq!(events, vec![2]);
    }

    #[test]
    fn test_runner_cross_chunk_split_inside_match() {
        let plan = anchor_plan("abc");
        let mut runner = AnchorRunner::new(&plan);
        let mut events = Vec::new();
        // "a" then "b" then "cxx"
        runner.scan_chunk(b"a", 0, &mut |pos| events.push(pos));
        runner.scan_chunk(b"b", 1, &mut |pos| events.push(pos));
        runner.scan_chunk(b"cxx", 2, &mut |pos| events.push(pos));
        runner.finish_scan(&mut |pos| events.push(pos));
        assert_eq!(events, vec![2]);
    }

    #[test]
    fn test_runner_events_monotone() {
        let plan = anchor_plan(r"a|b");
        let events = scan_events(&plan, b"abba");
        // "a" ends at 0, "b" ends at 1, "b" ends at 2, "a" ends at 3
        assert_eq!(events, vec![0, 1, 2, 3]);
        // Verify monotone ordering.
        for w in events.windows(2) {
            assert!(w[0] <= w[1], "events must be monotone");
        }
    }

    #[test]
    fn test_runner_reset() {
        let plan = anchor_plan("ab");
        let mut runner = AnchorRunner::new(&plan);

        // First match.
        let mut events = Vec::new();
        runner.scan_chunk(b"xabx", 0, &mut |pos| events.push(pos));
        assert_eq!(events, vec![2]);

        // Reset and match again.
        runner.reset();
        events.clear();
        runner.scan_chunk(b"yaby", 0, &mut |pos| events.push(pos));
        assert_eq!(events, vec![2]);
    }

    // -- BoundedGapMatcher --------------------------------------------------

    /// Helper: build a plan and run the matcher, returning match result.
    fn gap_match(pattern: &str, input: &[u8]) -> bool {
        let h = hir(pattern);
        let plan = probe(&h)
            .unwrap_or_else(|| panic!("pattern should be recognized as bounded-gap: {pattern}"));
        let mut cache = BoundedGapCache::new();
        let mut m = BoundedGapMatcher::new(&plan, &mut cache);
        m.chunk(input);
        m.finish()
    }

    /// Helper: build a plan from merged HIR and run the matcher.
    fn gap_match_merged(pattern: &str, input: &[u8]) -> bool {
        let h = hir_merged(pattern);
        let plan = probe(&h)
            .unwrap_or_else(|| panic!("pattern should be recognized as bounded-gap: {pattern}"));
        let mut cache = BoundedGapCache::new();
        let mut m = BoundedGapMatcher::new(&plan, &mut cache);
        m.chunk(input);
        m.finish()
    }

    #[test]
    fn test_matcher_simple_two_anchor() {
        // abc.{0,10}xyz
        assert!(gap_match(r"abc.{0,10}xyz", b"abc___xyz"));
        assert!(gap_match(r"abc.{0,10}xyz", b"abcxyz")); // gap=0
        assert!(!gap_match(r"abc.{0,10}xyz", b"abc"));
        assert!(!gap_match(r"abc.{0,10}xyz", b"xyz"));
        assert!(!gap_match(r"abc.{0,10}xyz", b"abc____________xyz")); // gap=12 > 10
    }

    #[test]
    fn test_matcher_exact_gap() {
        // ab.{3}cd → gap must be exactly 3
        assert!(gap_match(r"ab.{3}cd", b"ab123cd"));
        assert!(!gap_match(r"ab.{3}cd", b"ab12cd")); // gap=2
        assert!(!gap_match(r"ab.{3}cd", b"ab1234cd")); // gap=4
    }

    #[test]
    fn test_matcher_trailing_gap() {
        // abc.{0,5} → match once "abc" found (min_gap=0)
        assert!(gap_match(r"abc.{0,5}", b"abc"));
        assert!(gap_match(r"abc.{0,5}", b"abcXXX"));
        assert!(!gap_match(r"abc.{0,5}", b"xyz"));
    }

    #[test]
    fn test_matcher_trailing_gap_min_nonzero() {
        // abc.{3,5} → need at least 3 bytes after "abc"
        assert!(!gap_match(r"abc.{3,5}", b"abc"));
        assert!(!gap_match(r"abc.{3,5}", b"abcXX"));
        assert!(gap_match(r"abc.{3,5}", b"abcXXX"));
        assert!(gap_match(r"abc.{3,5}", b"abcXXXXX"));
        assert!(gap_match(r"abc.{3,5}", b"abcXXXXXXXX")); // gap=8 > 5, but min=3 satisfied at some point
    }

    #[test]
    fn test_matcher_constrained_gap() {
        // ab[^/]{0,10}cd → gap bytes must not be '/'
        assert!(gap_match(r"ab[^/]{0,10}cd", b"ab___cd"));
        assert!(!gap_match(r"ab[^/]{0,10}cd", b"ab/cd")); // '/' in gap
        assert!(gap_match(r"ab[^/]{0,10}cd", b"abcd")); // gap=0
    }

    #[test]
    fn test_matcher_constrained_trailing_gap() {
        // ab[^/]{0,10} → trailing gap rejects '/'
        assert!(gap_match(r"ab[^/]{0,10}", b"ab"));
        assert!(gap_match(r"ab[^/]{0,10}", b"abxyz"));
        // Once a '/' is seen, the tail gap is invalidated for entries
        // before the '/', but "ab" at a later position could still work.
        assert!(gap_match(r"ab[^/]{0,10}", b"ab/ab"));
    }

    #[test]
    fn test_matcher_three_anchor_chain() {
        // ab.{0,5}cd.{0,5}ef
        assert!(gap_match(r"ab.{0,5}cd.{0,5}ef", b"abcdef"));
        assert!(gap_match(r"ab.{0,5}cd.{0,5}ef", b"ab__cd__ef"));
        assert!(!gap_match(r"ab.{0,5}cd.{0,5}ef", b"abef"));
    }

    #[test]
    fn test_matcher_start_anchor() {
        // ^abc.{0,10}xyz → must start at position 0
        assert!(gap_match(r"^abc.{0,10}xyz", b"abc___xyz"));
        assert!(!gap_match(r"^abc.{0,10}xyz", b"_abc___xyz"));
    }

    #[test]
    fn test_matcher_end_anchor() {
        // abc.{0,10}xyz$ → must end at end-of-input
        assert!(gap_match(r"abc.{0,10}xyz$", b"abc___xyz"));
        assert!(!gap_match(r"abc.{0,10}xyz$", b"abc___xyz_"));
    }

    #[test]
    fn test_matcher_both_anchors() {
        // ^abc.{0,10}xyz$ → anchored both ends
        assert!(gap_match(r"^abc.{0,10}xyz$", b"abc___xyz"));
        assert!(!gap_match(r"^abc.{0,10}xyz$", b"_abc___xyz"));
        assert!(!gap_match(r"^abc.{0,10}xyz$", b"abc___xyz_"));
    }

    #[test]
    fn test_matcher_end_anchor_trailing_gap() {
        // abc.{0,5}$ → must end at EOI
        assert!(gap_match(r"abc.{0,5}$", b"abc"));
        assert!(gap_match(r"abc.{0,5}$", b"abcXXX"));
        assert!(!gap_match(r"abc.{0,5}$", b"abc______")); // 6 > 5
        assert!(gap_match(r"abc.{0,5}$", b"__abc__"));
    }

    #[test]
    fn test_matcher_multi_chunk() {
        let h = hir(r"abc.{0,10}xyz");
        let plan = probe(&h).expect("should recognise");
        let mut cache = BoundedGapCache::new();
        let mut m = BoundedGapMatcher::new(&plan, &mut cache);
        m.chunk(b"ab");
        m.chunk(b"c__");
        m.chunk(b"_xy");
        m.chunk(b"z");
        assert!(m.finish());
    }

    #[test]
    fn test_matcher_overlapping_matches() {
        // a.{0,3}a → "aaa" has overlapping matches
        assert!(gap_match(r"a.{0,3}a", b"aa")); // gap=0
        assert!(gap_match(r"a.{0,3}a", b"a___a")); // gap=3
        assert!(!gap_match(r"a.{0,3}a", b"a____a")); // gap=4 > 3
    }

    #[test]
    fn test_matcher_merged_gaps() {
        // abc.{0,10}.{0,20}xyz → gaps merge to .{0,30}
        assert!(gap_match_merged(
            r"abc.{0,10}.{0,20}xyz",
            b"abc______________________________xyz"
        )); // gap=30
    }

    #[test]
    fn test_matcher_early_exit() {
        // Once matched, further input doesn't change result.
        let h = hir(r"ab.{0,5}cd");
        let plan = probe(&h).expect("should recognise");
        let mut cache = BoundedGapCache::new();
        let mut m = BoundedGapMatcher::new(&plan, &mut cache);
        m.chunk(b"abcd"); // match immediately
        assert!(m.ismatch());
        m.chunk(b"xxxxxxxxxxxxxxxx"); // should be skipped
        assert!(m.finish());
    }

    // -- Prefilter integration tests ----------------------------------------

    #[test]
    fn test_prefilter_skips_non_candidate_bytes() {
        // "foo" has prefilter Memchr1('f').  Large input with match
        // near the end — the prefilter should skip most of the input.
        let input: Vec<u8> = {
            let mut v = vec![b'x'; 10_000];
            v.extend_from_slice(b"foo___bar");
            v
        };
        assert!(gap_match(r"foo.{0,10}bar", &input));
    }

    #[test]
    fn test_prefilter_no_match_in_large_input() {
        // No 'f' anywhere → prefilter skips entire input.
        let input = vec![b'x'; 10_000];
        assert!(!gap_match(r"foo.{0,10}bar", &input));
    }

    #[test]
    fn test_prefilter_position_tracking_across_skip() {
        // Verify gap length is computed correctly when prefilter skips
        // bytes.  Anchor "ab" prefilter skips to 'a'.
        // Input: 1000 x's, then "ab___cd".  Gap must be ≤5.
        let input: Vec<u8> = {
            let mut v = vec![b'x'; 1000];
            v.extend_from_slice(b"ab___cd");
            v
        };
        assert!(gap_match(r"ab.{0,5}cd", &input));
    }

    #[test]
    fn test_prefilter_gap_too_large_after_skip() {
        // After prefilter skip, gap exceeds max.
        // "ab" at position 1000, "cd" needs to be within 5 bytes.
        // Put "cd" 10 bytes after "ab" → gap=8 > 5 → no match.
        let input: Vec<u8> = {
            let mut v = vec![b'x'; 1000];
            v.extend_from_slice(b"ab");
            v.extend_from_slice(b"xxxxxxxx");
            v.extend_from_slice(b"cd");
            v
        };
        assert!(!gap_match(r"ab.{0,5}cd", &input));
    }

    #[test]
    fn test_prefilter_multiple_candidates() {
        // Multiple 'f' bytes — prefilter finds first, fails match,
        // re-enters quiescent, skips to next 'f', matches.
        assert!(gap_match(r"foo.{0,5}bar", b"xxxfxxxxxxxxxxfoo__barxxx"));
    }

    #[test]
    fn test_prefilter_with_constrained_gap() {
        // ab[^/]{0,10}cd — prefilter for 'a', gap rejects '/'.
        let input: Vec<u8> = {
            let mut v = vec![b'x'; 500];
            v.extend_from_slice(b"ab___cd");
            v
        };
        assert!(gap_match(r"ab[^/]{0,10}cd", &input));
    }

    #[test]
    fn test_prefilter_with_trailing_gap() {
        // foo.{3,10} with prefilter for 'f'.
        let input: Vec<u8> = {
            let mut v = vec![b'x'; 500];
            v.extend_from_slice(b"fooXXX");
            v
        };
        assert!(gap_match(r"foo.{3,10}", &input));
    }

    #[test]
    fn test_prefilter_multi_chunk_with_skip() {
        // Prefilter skip across chunk boundaries.
        let h = hir(r"foo.{0,5}bar");
        let plan = probe(&h).expect("should recognise");
        let mut cache = BoundedGapCache::new();
        let mut m = BoundedGapMatcher::new(&plan, &mut cache);
        // First chunk: no 'f' — entirely skipped by prefilter.
        m.chunk(&[b'x'; 5000]);
        // Second chunk: contains the match.
        m.chunk(b"foo__bar");
        assert!(m.finish());
    }

    #[test]
    fn test_prefilter_quiescence_after_failed_anchor() {
        // After anchor[0] partially matches then fails, the matcher
        // should return to quiescent and resume prefilter skipping.
        // "fo" starts anchor[0] but "fx" fails → back to quiescent.
        assert!(gap_match(r"foo.{0,5}bar", b"fxxxxxxxxxxxxxfoo_barxxx"));
    }

    #[test]
    fn test_prefilter_end_anchor_with_skip() {
        // ^foo.{0,5}bar$ with large padding before — ^foo requires
        // start at 0, so prefilter doesn't help (first byte must be 'f').
        assert!(!gap_match(r"^foo.{0,5}bar$", b"xxxfoo_bar"));
        assert!(gap_match(r"^foo.{0,5}bar$", b"foo_bar"));
    }
}
