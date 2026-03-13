//! Thompson NFA with per-thread counting constraints.
//!
//! Based on Russ Cox's article <https://swtch.com/~rsc/regexp/regexp1.html>
//! (Thompson NFA construction and simulation) with additional support for
//! bounded repetitions (`{min,max}`) via per-thread counter contexts.
//!
//! # Architecture
//!
//! The pipeline is:
//!
//! ```text
//! regex_syntax::hir::Hir  ──hir2postfix──>  postfix HIR  ──next_fragment──>  NFA states
//! ```
//!
//! ## Counting constraints
//!
//! A repetition `body{min,max}` is lowered to a **single-copy** NFA:
//!
//! ```text
//! CounterInstance(c) ── body ── CounterIncrement(c, min, max)
//!                         ^                │
//!                         └── continue ────┘
//!                                  break ──> (next)
//! ```
//!
//! `CounterInstance` is an epsilon state that adds `(counter, 0)` to the
//! thread's counter context.  The body runs, then `CounterIncrement`
//! increments the counter in the context.  If `value < max`, the
//! continue path loops back to the body.  If `value >= min`, the break
//! path exits the repetition (removing the counter from context).
//!
//! ## Per-thread counter contexts
//!
//! Each thread carries a [`CounterCtx`] — a fixed-length vector indexed
//! by counter, where each slot holds the counter's current value or
//! `COUNTER_INACTIVE`.  This replaces the Becchi multi-instance
//! differential counter representation, eliminating the need for:
//!
//! - Two-copy body duplication (body₁/body₂ with counter index remapping)
//! - DeltaPool (arena-backed linked-list for counter deltas)
//! - `body_counter_map` (compile-time BFS for stale-instance detection)
//! - `single_byte_body` flags and `anchored_start` bypass
//! - `CounterGeneration` stamps for epsilon-closure re-entry detection
//!
//! ## Deduplication
//!
//! Two-tier dedup prevents duplicate work in the epsilon closure:
//!
//! - **Empty context** (threads outside all repetitions): fast O(1)
//!   dedup via `lastlist[state] == listid`.
//! - **Non-empty context**: linear scan of a `Vec<(StateIdx, CounterCtx)>`
//!   with value comparison through the [`CounterPool`], cleared per step.
//!
//! ## Complexity
//!
//! - **Anchored patterns** (`^...$`): only 1 counter instance per
//!   repetition (no re-seeding).  O(|states|) per step — identical to
//!   the delta approach.
//! - **Unanchored patterns**: O(|states| × max) per step.  The
//!   `max_repetition` compile-time cap (default 1000) bounds this for
//!   untrusted patterns.

use std::fmt;
use std::io::Write;
use std::ops::{Index, IndexMut, Range};
use std::sync::atomic::{AtomicU64, Ordering};

use regex_syntax::hir::{self, HirKind};

use ahash::HashMap;

/// Global counter for assigning unique IDs to compiled regexes.
static NEXT_REGEX_ID: AtomicU64 = AtomicU64::new(1);

mod dfa;
mod dump;
pub mod fuzz_gen;
mod info;
mod memrange;

use dfa::{
    DfaMatcher, DfaMemory, Tier1DfaCache, Tier2Analysis, Tier2DfaCache, Tier2DfaMatcher,
    Tier3Analysis, Tier3DfaCache, Tier3DfaMatcher, Tier4DfaCache, Tier4DfaMatcher,
    compute_tier2_analysis, compute_tier3_analysis,
};
pub use dump::DumpRegex;
pub use info::{
    CounterInfo, ExecutionInfo, MemoryInfo, NfaStateBreakdown, RegexInfo, StartClosureInfo,
};

// Exposed for benchmarking SIMD vs scalar memrange throughput.
#[doc(hidden)]
pub use memrange::memrange as memrange_simd;
#[doc(hidden)]
pub use memrange::memrange_scalar;

/// Re-export so users do not need a direct `regex-syntax` dependency.
pub use regex_syntax::hir::Hir;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// An error returned when the HIR contains constructs we don't support.
#[derive(Debug)]
pub enum Error {
    /// A Unicode character class that cannot be lowered to single bytes
    /// (i.e. contains codepoints above U+00FF).
    UnsupportedClass(hir::Class),
    /// A look-around assertion other than `^` (Start) or `$` (End) was
    /// encountered (e.g. `\b`, `\B`).
    UnsupportedLook(hir::Look),
    /// A bounded repetition `{n,m}` where `m` exceeds the configured
    /// `max_repetition` limit.  Contains `(actual_max, limit)`.
    RepetitionTooLarge(usize, usize),
    /// The pattern requires more than 256 counters (the maximum supported
    /// by the `CounterIdx(u8)` representation).
    TooManyCounters,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedClass(class) => {
                write!(f, "unsupported character class: {:?}", class)
            }
            Self::UnsupportedLook(look) => {
                write!(f, "unsupported look-around assertion: {:?}", look)
            }
            Self::RepetitionTooLarge(max, limit) => {
                write!(f, "repetition max {} exceeds limit {}", max, limit)
            }
            Self::TooManyCounters => {
                write!(f, "pattern requires more than {} counters", MAX_COUNTERS)
            }
        }
    }
}

impl std::error::Error for Error {}

/// A 256-entry boolean lookup table indicating which byte values belong
/// to a character class.  `class[b]` is `true` when byte `b` matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ByteClass([bool; 256]);

impl ByteClass {
    /// A class that matches every byte value (`[true; 256]` — equivalent to `.`).
    #[allow(dead_code)]
    const ALL: Self = Self([true; 256]);

    /// A class that matches no byte value.
    const NONE: Self = Self([false; 256]);
}

/// `class[byte]` — test whether a byte matches this class.
impl Index<u8> for ByteClass {
    type Output = bool;

    #[inline]
    fn index(&self, byte: u8) -> &bool {
        &self.0[byte as usize]
    }
}

/// Index into the byte-class lookup tables ([`Regex::classes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ClassIdx(usize);

impl ClassIdx {
    #[inline]
    fn idx(self) -> usize {
        self.0
    }
}

/// `classes[class_idx]` — typed access to byte-class lookup tables.
impl Index<ClassIdx> for [ByteClass] {
    type Output = ByteClass;

    #[inline]
    fn index(&self, idx: ClassIdx) -> &ByteClass {
        &self[idx.idx()]
    }
}

/// Index into the byte-dispatch tables ([`Regex::byte_tables`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ByteTableIdx(usize);

impl ByteTableIdx {
    #[inline]
    pub(crate) fn idx(self) -> usize {
        self.0
    }
}

/// A 256-entry dispatch table mapping each byte value to a target
/// [`StateIdx`], or [`StateIdx::NONE`] for "no transition".
#[derive(Clone, Copy, Debug)]
pub(crate) struct ByteMap([StateIdx; 256]);

impl ByteMap {
    /// A table with no transitions (all entries are [`StateIdx::NONE`]).
    const EMPTY: Self = Self([StateIdx::NONE; 256]);
}

/// `map[byte]` — look up the target state for a given byte value.
impl Index<u8> for ByteMap {
    type Output = StateIdx;

    #[inline]
    fn index(&self, byte: u8) -> &StateIdx {
        &self.0[byte as usize]
    }
}

/// `byte_tables[table_idx]` — typed access to byte-dispatch tables.
impl Index<ByteTableIdx> for [ByteMap] {
    type Output = ByteMap;

    #[inline]
    fn index(&self, idx: ByteTableIdx) -> &ByteMap {
        &self[idx.idx()]
    }
}

// ---------------------------------------------------------------------------
// Zero-width assertions
// ---------------------------------------------------------------------------

/// The kind of zero-width assertion.
///
/// Each variant knows how to evaluate itself given the surrounding
/// context (position flags and neighbouring bytes).  See [`eval`].
///
/// [`eval`]: AssertKind::eval
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AssertKind {
    /// `^`  — start of input only.
    Start,
    /// `$`  — end of input only.
    End,
    /// `(?m:^)` — start of any line (LF-terminated).
    StartLF,
    /// `(?m:$)` — end of any line (LF-terminated).
    EndLF,
    /// `(?Rm:^)` — start of any line (CRLF-aware).
    ///
    /// Matches at the start of input, or immediately after `\n`, or
    /// immediately after `\r` **unless** a `\n` follows (i.e. not
    /// between `\r` and `\n`).
    StartCRLF,
    /// `(?Rm:$)` — end of any line (CRLF-aware).
    ///
    /// Matches at the end of input, or immediately before `\r`, or
    /// immediately before `\n` **unless** `\r` precedes (i.e. not
    /// between `\r` and `\n`).
    EndCRLF,
    /// `\b` — ASCII word boundary.
    ///
    /// Matches at a position where the previous byte and next byte
    /// differ in "word-ness" (one is `[0-9A-Za-z_]` and the other is
    /// not, or one side is start/end of input).
    WordAscii,
    /// `\B` — ASCII non-word boundary.
    ///
    /// Matches at a position where both sides are word characters or
    /// both sides are non-word characters.
    WordAsciiNegate,
    /// `\b{start}` — ASCII word-start boundary.
    ///
    /// Matches where the previous byte is NOT a word character (or
    /// start-of-input) and the next byte IS a word character.
    WordStartAscii,
    /// `\b{end}` — ASCII word-end boundary.
    ///
    /// Matches where the previous byte IS a word character and the
    /// next byte is NOT a word character (or end-of-input).
    WordEndAscii,
}

/// Result of evaluating an assertion at a given position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AssertEval {
    /// Assertion is satisfied — follow the `out` transition.
    Pass,
    /// Assertion is not satisfied — skip.
    Fail,
    /// Cannot determine yet (the upcoming byte is unknown).
    /// The state is parked in the list for deferred resolution in
    /// [`Matcher::step`] or [`Matcher::finish`].
    Defer,
}

/// Returns `true` if `b` is an ASCII word byte: `[0-9A-Za-z_]`.
#[inline]
pub(crate) fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Case-insensitive ASCII byte match.
///
/// `target` **must** be a lowercase ASCII letter (`b'a'..=b'z'`).
/// Returns `true` if `input` is the same letter in either case.
///
/// Uses the bit-5 trick: for ASCII letters, lowercase and uppercase
/// differ only in bit 5 (`0x20`), so `input | 0x20` folds to lowercase.
#[inline]
pub(crate) fn byte_match_ci(input: u8, target: u8) -> bool {
    debug_assert!(target.is_ascii_lowercase());
    input | 0x20 == target
}

/// Prefilter for skipping bytes that can never start a match.
///
/// Derived from [`Regex::start_closure`]: the set of consuming NFA states
/// reachable from the start state determines which input bytes can
/// possibly begin a match.  When the DFA is in the DEAD state, `memchr`
/// is used to jump directly to the next candidate byte, skipping over
/// large runs of irrelevant input at SIMD speed.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum Prefilter {
    /// No prefilter — all bytes are potentially interesting (e.g. dot-star,
    /// empty pattern, or too many distinct start bytes).
    #[default]
    None,
    /// Exactly one byte can start a match.
    Memchr1(u8),
    /// Exactly two distinct bytes can start a match.
    Memchr2(u8, u8),
    /// Exactly three distinct bytes can start a match.
    Memchr3(u8, u8, u8),
    /// All start bytes fall in a contiguous range `[lo, hi]` (inclusive).
    /// Used when there are more than 3 distinct start bytes but they fit
    /// within a range of at most 16 values.
    Range(u8, u8),
}

impl AssertKind {
    /// Evaluate this assertion.
    ///
    /// * `at_start` — true when at position 0.
    /// * `at_end`   — true when at end-of-input.
    /// * `prev`     — the byte before the current position, or `None`
    ///   at the very beginning.
    /// * `next`     — the byte at the current position (the one about
    ///   to be consumed), or `None` when unknown or at end-of-input.
    #[inline]
    fn eval(self, at_start: bool, at_end: bool, prev: Option<u8>, next: Option<u8>) -> AssertEval {
        use AssertEval::*;
        match self {
            AssertKind::Start => {
                if at_start {
                    Pass
                } else {
                    Fail
                }
            }
            AssertKind::End => {
                if at_end {
                    Pass
                } else {
                    Fail
                }
            }
            AssertKind::StartLF => {
                if at_start || prev == Some(b'\n') {
                    Pass
                } else {
                    Fail
                }
            }
            AssertKind::EndLF => {
                if at_end {
                    return Pass;
                }
                match next {
                    Some(b'\n') => Pass,
                    Some(_) => Fail,
                    None => Defer,
                }
            }
            AssertKind::StartCRLF => {
                if at_start || prev == Some(b'\n') {
                    return Pass;
                }
                match prev {
                    Some(b'\r') => {
                        // After \r: line start unless \n follows (\r\n is
                        // a single line terminator).
                        if at_end {
                            return Pass;
                        }
                        match next {
                            Some(b'\n') => Fail,
                            Some(_) => Pass,
                            None => Defer,
                        }
                    }
                    _ => Fail,
                }
            }
            AssertKind::EndCRLF => {
                if at_end {
                    return Pass;
                }
                match next {
                    Some(b'\r') => Pass,
                    Some(b'\n') => {
                        // Before \n: line end unless \r precedes (\r\n is
                        // a single line terminator; the end was before \r).
                        if prev == Some(b'\r') { Fail } else { Pass }
                    }
                    Some(_) => Fail,
                    None => Defer,
                }
            }
            AssertKind::WordAscii => {
                let prev_w = prev.is_some_and(is_word_byte);
                match next {
                    Some(b) => {
                        if prev_w != is_word_byte(b) {
                            Pass
                        } else {
                            Fail
                        }
                    }
                    None => {
                        if at_end {
                            // At end-of-input: boundary iff prev is a word char.
                            if prev_w { Pass } else { Fail }
                        } else {
                            Defer
                        }
                    }
                }
            }
            AssertKind::WordAsciiNegate => {
                let prev_w = prev.is_some_and(is_word_byte);
                match next {
                    Some(b) => {
                        if prev_w == is_word_byte(b) {
                            Pass
                        } else {
                            Fail
                        }
                    }
                    None => {
                        if at_end {
                            // At end-of-input: non-boundary iff prev is NOT a word char.
                            if prev_w { Fail } else { Pass }
                        } else {
                            Defer
                        }
                    }
                }
            }
            AssertKind::WordStartAscii => {
                // prev must NOT be word, next must be word.
                let prev_w = prev.is_some_and(is_word_byte);
                if prev_w {
                    return Fail;
                }
                match next {
                    Some(b) => {
                        if is_word_byte(b) {
                            Pass
                        } else {
                            Fail
                        }
                    }
                    None => {
                        if at_end {
                            Fail
                        } else {
                            Defer
                        }
                    }
                }
            }
            AssertKind::WordEndAscii => {
                // prev must be word, next must NOT be word.
                let prev_w = prev.is_some_and(is_word_byte);
                if !prev_w {
                    return Fail;
                }
                match next {
                    Some(b) => {
                        if is_word_byte(b) {
                            Fail
                        } else {
                            Pass
                        }
                    }
                    None => {
                        if at_end {
                            Pass
                        } else {
                            Defer
                        }
                    }
                }
            }
        }
    }

    /// Dot-graph label for this assertion kind.
    fn label(self) -> &'static str {
        match self {
            AssertKind::Start => "^",
            AssertKind::End => "$",
            AssertKind::StartLF => "^LF",
            AssertKind::EndLF => "$LF",
            AssertKind::StartCRLF => "^CRLF",
            AssertKind::EndCRLF => "$CRLF",
            AssertKind::WordAscii => "\\b",
            AssertKind::WordAsciiNegate => "\\B",
            AssertKind::WordStartAscii => "\\b{start}",
            AssertKind::WordEndAscii => "\\b{end}",
        }
    }
}

// ---------------------------------------------------------------------------
// NFA states
// ---------------------------------------------------------------------------

/// A single NFA state.
///
/// Epsilon states (`Split`, `CounterInstance`, `CounterIncrement`,
/// `Assert`) are followed during [`Matcher::addstate`].
/// Byte-consuming states (`Byte`, `ByteClass`) are stepped over in
/// [`Matcher::step`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum State {
    /// Epsilon fork: follow both `out` and `out1`.
    Split { out: StateIdx, out1: StateIdx },

    /// Allocate (or push) a new instance on counter `counter`, then
    /// follow `out`.
    CounterInstance { counter: CounterIdx, out: StateIdx },

    /// Increment counter `counter`.
    ///
    /// - **Continue** (`out`): re-enter the repetition body (taken when
    ///   the counter has not yet reached `max`, or when there are
    ///   multiple instances and the oldest has not yet reached `max`).
    /// - **Break** (`out1`): exit the repetition (taken when any instance
    ///   value falls in `[min, max]`).
    CounterIncrement {
        counter: CounterIdx,
        out: StateIdx,
        out1: StateIdx,
        min: usize,
        max: usize,
    },

    /// Match a literal byte, then follow `out`.
    Byte { byte: u8, out: StateIdx },

    /// Case-insensitive ASCII byte match, then follow `out`.
    ///
    /// `byte` is a **lowercase** ASCII letter (`b'a'..=b'z'`).
    /// Matches both `byte` and `byte ^ 0x20` (the uppercase variant)
    /// via [`byte_match_ci`].
    ///
    /// Emitted instead of [`ByteClass`] when the class contains exactly
    /// one ASCII letter pair (e.g. `[cC]` under `(?i)`).  Avoids the
    /// 256-byte class table lookup — match is a single `OR` + `CMP`.
    ByteCI { byte: u8, out: StateIdx },

    /// Match any byte in the class (lookup table), then follow `out`.
    ///
    /// `class` is an index into [`Regex::classes`], a side-table of
    /// [`ByteClass`] lookup tables — one per possible byte value.
    /// A full-range table ([`ByteClass::ALL`]) is equivalent to the old
    /// `Wildcard` state.
    ByteClass { class: ClassIdx, out: StateIdx },

    /// Byte dispatch table: for input byte `b`, follow
    /// `byte_tables[table][b]` if the target is not [`StateIdx::NONE`].
    ///
    /// Replaces a chain of `Split` + `Byte` states when an alternation's
    /// branches all start with distinct literal bytes.  `table` is an
    /// index into [`Regex::byte_tables`].
    ByteTable { table: ByteTableIdx },

    /// Zero-width assertion (see [`AssertKind`] for the full catalogue).
    ///
    /// Evaluated in [`Matcher::addstate`] via [`AssertKind::eval`].
    /// When the result is [`AssertEval::Pass`], the `out` transition
    /// is followed.  When [`AssertEval::Defer`], the state is parked
    /// in the list and resolved by a pre-consumption pass in
    /// [`Matcher::step`] or by [`Matcher::finish`].
    Assert { kind: AssertKind, out: StateIdx },

    /// Accepting state.
    Match,
}

impl State {
    /// Return the "dangling out" pointer used by [`RegexBuilder::patch`]
    /// and [`RegexBuilder::append`] to thread fragment lists.
    fn next(&self) -> StateIdx {
        match self {
            State::Byte { out, .. }
            | State::ByteCI { out, .. }
            | State::ByteClass { out, .. }
            | State::CounterInstance { out, .. }
            | State::Assert { out, .. } => *out,
            State::Split { out1, .. } | State::CounterIncrement { out1, .. } => *out1,
            _ => unreachable!(),
        }
    }

    /// Overwrite the "dangling out" pointer.
    fn append(&mut self, next: StateIdx) {
        match self {
            State::Byte { out, .. }
            | State::ByteCI { out, .. }
            | State::ByteClass { out, .. }
            | State::CounterInstance { out, .. }
            | State::Assert { out, .. } => *out = next,
            State::Split { out1, .. } | State::CounterIncrement { out1, .. } => *out1 = next,
            _ => unreachable!(),
        }
    }
}

/// Index into the NFA state array ([`Regex::states`]).
///
/// [`StateIdx::NONE`] is used both as a "dangling/unpatched" marker
/// during construction and as "no transition" in byte-table entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StateIdx(pub(crate) u32);

impl StateIdx {
    /// Sentinel value for unpatched `out` pointers during construction
    /// and for "no transition" entries in byte-dispatch tables.
    const NONE: Self = Self(u32::MAX);

    /// Return the raw index as `usize`.  Panics on `NONE` in debug builds.
    #[inline]
    fn idx(self) -> usize {
        debug_assert!(self != Self::NONE, "StateIdx::NONE used as index");
        self.0 as usize
    }

    /// Return the raw index as `usize` **without** asserting against `NONE`.
    /// Use only where `NONE` is a valid/expected value (e.g. bounds checks).
    #[inline]
    fn raw(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for StateIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// `states[state_idx]` — typed access to the NFA state array.
impl Index<StateIdx> for [State] {
    type Output = State;

    #[inline]
    fn index(&self, idx: StateIdx) -> &State {
        &self[idx.idx()]
    }
}

impl IndexMut<StateIdx> for [State] {
    #[inline]
    fn index_mut(&mut self, idx: StateIdx) -> &mut State {
        &mut self[idx.idx()]
    }
}

/// Bounds-checked mutable access by [`StateIdx`].
trait StateSliceExt {
    /// Returns `None` for [`StateIdx::NONE`] or any out-of-range index.
    fn get_mut_state(&mut self, idx: StateIdx) -> Option<&mut State>;
}

impl StateSliceExt for [State] {
    #[inline]
    fn get_mut_state(&mut self, idx: StateIdx) -> Option<&mut State> {
        self.get_mut(idx.raw())
    }
}

// ---------------------------------------------------------------------------
// NFA fragment (used during construction)
// ---------------------------------------------------------------------------

/// A partially-built NFA fragment with a `start` state and a dangling
/// `out` pointer that will be patched to the next fragment's start.
#[derive(Debug)]
struct Fragment {
    start: StateIdx,
    out: StateIdx,
}

impl Fragment {
    fn new(start: StateIdx, out: StateIdx) -> Self {
        Self { start, out }
    }
}

// ---------------------------------------------------------------------------
// Postfix HIR nodes
// ---------------------------------------------------------------------------

/// A postfix HIR instruction consumed by [`RegexBuilder::next_fragment`]
/// to emit NFA states.
#[derive(Clone, Copy, Debug)]
enum RegexHirNode {
    Alternate,
    Catenate,
    Byte(u8),
    /// Case-insensitive ASCII letter.  `byte` is lowercase (`b'a'..=b'z'`).
    ByteCI(u8),
    RepeatZeroOne,
    RepeatZeroPlus,
    RepeatOnePlus,
    /// Index into [`RegexBuilder::classes`].
    ByteClass(ClassIdx),
    /// Single-copy counter loop: pops the body fragment and wires
    /// CI → body → CInc with a break exit.
    CounterLoop {
        counter: CounterIdx,
        min: usize,
        max: usize,
    },
    Assert(AssertKind),
}

// ---------------------------------------------------------------------------
// Compiled regex
// ---------------------------------------------------------------------------

pub(crate) struct StateList(Box<[State]>);

impl fmt::Debug for StateList {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_map().entries(self.0.iter().enumerate()).finish()
    }
}

impl std::ops::Deref for StateList {
    type Target = [State];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A compiled NFA ready for matching.
#[derive(Debug)]
pub struct Regex {
    /// Unique identity for this compiled regex, used by [] to
    /// detect when its cached DFA states are stale (built for a different
    /// regex).  Assigned from a global atomic counter at build time.
    pub(crate) id: u64,
    pub(crate) states: StateList,
    pub(crate) start: StateIdx,
    /// Number of counter variables allocated during compilation.
    pub(crate) num_counters: usize,
    /// Byte-class lookup tables referenced by [`State::ByteClass::class`].
    pub(crate) classes: Box<[ByteClass]>,
    /// Byte dispatch tables referenced by [`State::ByteTable::table`].
    /// Each entry maps a byte value to a target state index, or
    /// [`StateIdx::NONE`] for "no transition".
    pub(crate) byte_tables: Box<[ByteMap]>,
    /// Precomputed epsilon closure of the start state (consuming leaves only).
    ///
    /// If the start state's epsilon closure contains only `Split` transitions
    /// leading to consuming states (`Byte`, `ByteClass`, `ByteTable`) or
    /// `Match`, the consuming leaves are cached here.  `Match` states are
    /// recorded separately in [`start_closure_matches`] and excluded from
    /// this array so the re-seed loop needs no per-state type check.
    ///
    /// Empty if the closure contains `CounterInstance`, `Assert`, or
    /// `CounterIncrement` states (which require context manipulation);
    /// the runtime falls back to full epsilon-closure via `addstate()`.
    start_closure: Box<[StateIdx]>,
    /// `true` if `Match` is reachable from the start state through
    /// the precomputed start closure (i.e. the pattern can match the
    /// empty string without consuming any input).  Used to initialise
    /// `Matcher::ever_matched` so that `chunk()` can short-circuit
    /// immediately.
    start_closure_matches: bool,
    /// `true` when this pattern can be matched using the lazy DFA
    /// (Tier 1: no counters, no CRLF assertion types like
    /// EndCRLF/StartCRLF).  Deferred assertions (`\b`, `\B`, `EndLF`)
    /// are handled natively by Tier 1.
    dfa_eligible: bool,
    /// `true` when this pattern has non-nested counters with fixed-length
    /// bodies and can use the Tier 2 differential-counter DFA.
    tier2_eligible: bool,
    /// Precomputed NFA analysis for Tier 2 (body interior data for
    /// differential counters).  `None` when the pattern is not tier 2
    /// eligible.
    pub(crate) tier2_analysis: Option<Tier2Analysis>,
    /// `true` when this pattern has non-nested counters and can use the
    /// Tier 3 conditional-transition DFA.
    tier3_eligible: bool,
    /// Precomputed NFA analysis for Tier 3 (per-target actions, CI seed
    /// origins, break-path seeds).  `None` when the pattern is not tier 3
    /// eligible.
    pub(crate) tier3_analysis: Option<Tier3Analysis>,
    /// `true` when this pattern has counters but can use the counting DFA
    /// (Tier 4: no complex assertions).
    tier4_eligible: bool,
    /// Per-counter info: `(min, max, body_byte_length)`.
    /// `body_byte_length` is 0 if the body has variable length.
    /// Empty if the pattern has no counters.
    counter_info: Box<[(usize, usize, usize)]>,
    /// Byte equivalence classes for DFA transition table compression.
    ///
    /// Maps each input byte (0..255) to a small equivalence class index.
    /// Two bytes are in the same class when they produce identical
    /// transitions from *every* DFA state (i.e. they match exactly the
    /// same set of NFA consuming states, with the same targets, and have
    /// the same word-ness for deferred assertion resolution).
    ///
    /// The DFA transition table is indexed by
    /// `state * num_byte_classes + byte_classes[byte]` instead of
    /// `state * 256 + byte`, giving a ~6-7× compression for typical
    /// case-insensitive patterns.
    ///
    /// **Tradeoff**: the indirection adds ~2-3 instructions per byte in the
    /// DFA hot loop (an extra array load + a real multiply instead of a
    /// shift for the slot computation).  For simple patterns with few DFA
    /// states the full stride=256 table fits comfortably in L1/L2 cache,
    /// so the compression is not needed and the indirection is pure
    /// overhead.  For complex patterns (many NFA states → many DFA states)
    /// the table compression keeps the working set small enough to stay in
    /// cache.
    ///
    /// We therefore only enable byte-class compression when the NFA has more
    /// than [`BYTE_CLASSES_NFA_THRESHOLD`] states.  Below that threshold,
    /// `byte_classes` is the identity mapping and `num_byte_classes` is 256.
    pub(crate) byte_classes: [u8; 256],
    /// Number of distinct byte equivalence classes (the "stride" of
    /// each DFA state row in the flat transition table).  Either 256
    /// (identity, no compression) or the actual class count when byte-class
    /// compression is enabled.
    pub(crate) num_byte_classes: usize,
    /// Prefilter for skipping non-starting bytes in the DFA hot loop.
    pub(crate) prefilter: Prefilter,
    /// Precomputed reachability: `state_can_reach_match[i]` is `true` iff
    /// the `Match` state is reachable from NFA state `i` through epsilon
    /// transitions (Split, Assert — optimistically, CounterInstance,
    /// CounterIncrement continue path).  Used by DFA tiers 1–3 to avoid
    /// a per-call DFS in `epsilon_closure` and `resolve_deferred_at_end`.
    pub(crate) state_can_reach_match: Box<[bool]>,
    /// Precomputed per-counter flag: `counter_break_can_match[ci]` is `true`
    /// iff the break path (`out1`) of the `CounterIncrement` state for
    /// counter `ci` can reach the `Match` state through epsilon transitions.
    /// Avoids scanning all NFA states at DFA populate time.
    pub(crate) counter_break_can_match: Box<[bool]>,
}
impl Regex {
    /// Return the total memory footprint (in bytes) of this compiled
    /// regex, including both inline and heap-allocated data.
    ///
    /// This accounts for:
    /// - The `Regex` struct itself (inline fields).
    /// - The `states` boxed slice (header + per-state inline size).
    /// - The `classes` boxed slice (byte-class lookup tables).
    /// - The `byte_tables` boxed slice.
    pub fn memory_size(&self) -> usize {
        let inline = std::mem::size_of::<Self>();
        let states_alloc = self.states.len() * std::mem::size_of::<State>();
        let classes_alloc = self.classes.len() * std::mem::size_of::<ByteClass>();
        let byte_tables_alloc = self.byte_tables.len() * std::mem::size_of::<ByteMap>();
        let reach_match_alloc = self.state_can_reach_match.len() * std::mem::size_of::<bool>();
        let break_match_alloc = self.counter_break_can_match.len() * std::mem::size_of::<bool>();
        inline
            + states_alloc
            + classes_alloc
            + byte_tables_alloc
            + reach_match_alloc
            + break_match_alloc
    }
    /// Return per-counter info: `(min, max, body_byte_length)`.
    ///
    /// `body_byte_length` is the fixed number of bytes consumed per counter
    /// iteration (used by the Tier 2 differential-counter DFA).  It is 0
    /// for variable-length bodies.
    pub fn counter_info(&self, counter_idx: usize) -> (usize, usize, usize) {
        self.counter_info[counter_idx]
    }
    /// Return the minimum DFA tier that can handle this regex.
    ///
    /// - `0` = NFA only (e.g. CRLF assertions, zero-width counter bodies)
    /// - `1` = Tier 1+ (counter-free, deferred assertions supported)
    /// - `2` = Tier 2+ (non-nested fixed-length-body counters, differential counters)
    /// - `3` = Tier 3+ (non-nested counters, deferred assertions supported)
    /// - `4` = Tier 4 (nested counters, no deferred assertions)
    pub fn min_tier(&self) -> u8 {
        if self.dfa_eligible {
            1
        } else if self.tier2_eligible {
            2
        } else if self.tier3_eligible {
            3
        } else if self.tier4_eligible {
            4
        } else {
            0
        }
    }
    /// Print diagnostic information about this compiled regex.
    /// Return a [`RegexInfo`] snapshot of compiled regex diagnostics.
    pub fn info(&self) -> RegexInfo {
        // -- NFA state breakdown --
        let mut n_split = 0usize;
        let mut n_byte = 0usize;
        let mut n_byte_ci = 0usize;
        let mut n_byte_class = 0usize;
        let mut n_byte_table = 0usize;
        let mut n_assert = 0usize;
        let mut n_counter_instance = 0usize;
        let mut n_counter_increment = 0usize;
        let mut n_match = 0usize;
        for s in self.states.iter() {
            match s {
                State::Split { .. } => n_split += 1,
                State::Byte { .. } => n_byte += 1,
                State::ByteCI { .. } => n_byte_ci += 1,
                State::ByteClass { .. } => n_byte_class += 1,
                State::ByteTable { .. } => n_byte_table += 1,
                State::Assert { .. } => n_assert += 1,
                State::CounterInstance { .. } => n_counter_instance += 1,
                State::CounterIncrement { .. } => n_counter_increment += 1,
                State::Match => n_match += 1,
            }
        }

        // -- Counters --
        let mut counters = Vec::new();
        for s in self.states.iter() {
            if let State::CounterIncrement {
                counter, min, max, ..
            } = s
            {
                counters.push(CounterInfo {
                    index: counter.idx(),
                    min: *min,
                    max: *max,
                });
            }
        }

        // -- Assertions --
        let mut assert_kinds = std::collections::BTreeSet::new();
        for s in self.states.iter() {
            if let State::Assert { kind, .. } = s {
                assert_kinds.insert(format!("{:?}", kind));
            }
        }

        // -- Deferred assertions --
        let n_deferred = self
            .states
            .iter()
            .filter(|s| {
                matches!(s,
                    State::Assert { kind, .. } if matches!(kind,
                        AssertKind::WordAscii
                        | AssertKind::WordAsciiNegate
                        | AssertKind::WordStartAscii
                        | AssertKind::WordEndAscii
                        | AssertKind::EndLF
                    )
                )
            })
            .count();

        // -- Execution tier --
        let execution_tier = if self.dfa_eligible {
            "Tier 1: Lazy DFA (flat table, no counters, deferred assertions)"
        } else if self.tier2_eligible {
            "Tier 2: Differential-counter DFA (fixed-length bodies, O(1) counters)"
        } else if self.tier3_eligible {
            if self
                .tier3_analysis
                .as_ref()
                .is_some_and(|a| a.all_counters_rangeable)
            {
                "Tier 3: Conditional DFA (non-nested counters, range-compressed)"
            } else {
                "Tier 3: Conditional DFA (non-nested counters, per-instance)"
            }
        } else if self.tier4_eligible {
            "Tier 4: Counting DFA (flat table + counter programs)"
        } else {
            "NFA simulator (general case)"
        };

        // -- Prefilter --
        let prefilter_desc = match &self.prefilter {
            Prefilter::None => "none".to_string(),
            Prefilter::Memchr1(b) => format!("memchr1({:?})", *b as char),
            Prefilter::Memchr2(a, b) => format!("memchr2({:?}, {:?})", *a as char, *b as char),
            Prefilter::Memchr3(a, b, c) => {
                format!(
                    "memchr3({:?}, {:?}, {:?})",
                    *a as char, *b as char, *c as char
                )
            }
            Prefilter::Range(lo, hi) => format!("range(0x{:02X}..=0x{:02X})", lo, hi),
        };

        RegexInfo {
            memory: MemoryInfo {
                total: self.memory_size(),
                states: self.states.len() * std::mem::size_of::<State>(),
                classes: self.classes.len() * std::mem::size_of::<ByteClass>(),
                byte_tables: self.byte_tables.len() * std::mem::size_of::<ByteMap>(),
                num_states: self.states.len(),
                state_size: std::mem::size_of::<State>(),
                num_classes: self.classes.len(),
                class_size: std::mem::size_of::<ByteClass>(),
                num_byte_tables: self.byte_tables.len(),
                byte_table_size: std::mem::size_of::<ByteMap>(),
            },
            nfa_states: NfaStateBreakdown {
                split: n_split,
                byte_: n_byte,
                byte_ci: n_byte_ci,
                byte_class: n_byte_class,
                byte_table: n_byte_table,
                assert: n_assert,
                counter_instance: n_counter_instance,
                counter_increment: n_counter_increment,
                match_: n_match,
            },
            counters,
            assert_kinds: assert_kinds.into_iter().collect(),
            deferred_assertions: n_deferred,
            byte_classes: self.num_byte_classes,
            execution: ExecutionInfo {
                tier: self.min_tier(),
                tier_name: execution_tier.to_string(),
            },
            start_closure: StartClosureInfo {
                len: self.start_closure.len(),
                matches_empty: self.start_closure_matches,
            },
            prefilter: prefilter_desc,
        }
    }

    /// Emit a Graphviz DOT representation of the NFA.
    pub fn to_dot(&self, mut buffer: impl Write) {
        let mut visited = vec![false; self.states.len()];
        writeln!(buffer, "digraph graphname {{").unwrap();
        writeln!(buffer, "\trankdir=LR;").unwrap();
        writeln!(&mut buffer, "\t{} [shape=box];", self.start).unwrap();
        let mut stack = vec![self.start];
        while let Some(s) = stack.pop() {
            let i = s.idx();
            if !visited[i] {
                writeln!(buffer, "\t// [{}] {:?}", s, self.states[s]).unwrap();
                self.write_dot_state(s, &mut buffer, &mut stack);
                visited[i] = true;
            }
        }
        writeln!(buffer, "}}").unwrap();
    }

    fn write_dot_state(&self, idx: StateIdx, buffer: &mut impl Write, stack: &mut Vec<StateIdx>) {
        match self.states[idx] {
            State::Split { out, out1 } => {
                self.write_dot_state(out, buffer, stack);
                self.write_dot_state(out1, buffer, stack);
            }
            State::CounterInstance { counter, out } => {
                stack.push(out);
                writeln!(buffer, "\t{} -> {} [label=\"CI-{}\"];", idx, out, counter).unwrap();
            }
            State::CounterIncrement {
                counter,
                out,
                out1,
                min,
                max,
            } => {
                stack.push(out);
                writeln!(
                    buffer,
                    "\t{} -> {} [label=\"cont-{}{{{},{}}}\"];",
                    idx, out, counter, min, max
                )
                .unwrap();
                stack.push(out1);
                writeln!(
                    buffer,
                    "\t{} -> {} [label=\"break-{}{{{},{}}}\"];",
                    idx, out1, counter, min, max
                )
                .unwrap();
            }
            State::Byte { byte: b, out } => {
                stack.push(out);
                writeln!(buffer, "\t{} -> {} [label=\"{}\"];", idx, out, b as char).unwrap();
            }
            State::ByteCI { byte: b, out } => {
                stack.push(out);
                writeln!(
                    buffer,
                    "\t{} -> {} [label=\"(?i){}\"];",
                    idx, out, b as char
                )
                .unwrap();
            }
            State::ByteClass { class, out } => {
                stack.push(out);
                // Summarise the class for the label.
                let table = &self.classes[class];
                let count = table.0.iter().filter(|&&b| b).count();
                if count == 256 {
                    writeln!(buffer, "\t{} -> {} [label=\".\"];", idx, out).unwrap();
                } else {
                    writeln!(buffer, "\t{} -> {} [label=\"[{}B]\"];", idx, out, count).unwrap();
                }
            }
            State::Assert { kind, out } => {
                stack.push(out);
                writeln!(buffer, "\t{} -> {} [label=\"{}\"];", idx, out, kind.label()).unwrap();
            }
            State::ByteTable { table } => {
                let t = &self.byte_tables[table];
                for (b, &target) in t.0.iter().enumerate() {
                    if target != StateIdx::NONE {
                        stack.push(target);
                        writeln!(
                            buffer,
                            "\t{} -> {} [label=\"{}\"];",
                            idx, target, b as u8 as char
                        )
                        .unwrap();
                    }
                }
            }
            State::Match => {
                writeln!(buffer, "\t{} [peripheries=2];", idx).unwrap();
            }
        }
    }

    /// Return a human-readable dump of the compiled NFA states, counters,
    /// and analysis data.
    ///
    /// When `dfa` is `true`, the dump also includes tier-specific DFA
    /// analysis (Tier 2 body interior, Tier 3 origin actions, break seeds,
    /// reachability flags, etc.).
    ///
    /// The returned value implements [`Display`](fmt::Display), so it can
    /// be used directly with `print!` or `write!`.
    pub fn dump(&self, dfa: bool) -> DumpRegex<'_> {
        DumpRegex::new(self, dfa)
    }
}

// ---------------------------------------------------------------------------
// NFA builder (regex-syntax HIR -> postfix -> NFA)
// ---------------------------------------------------------------------------

/// Builds a compiled [`Regex`] from a [`regex_syntax::hir::Hir`].
///
/// The pipeline is:
/// 1. [`hir2postfix`](Self::hir2postfix) — recursively lowers the
///    `regex-syntax` HIR into a postfix sequence of [`RegexHirNode`]s.
/// 2. [`next_fragment`](Self::next_fragment) — consumes postfix nodes one
///    at a time, emitting NFA [`State`]s and wiring [`Fragment`]s together.
/// 3. [`build`](Self::build) — drives the pipeline and patches the final
///    fragment to the `Match` state.
use indexmap::IndexSet;

#[derive(Debug)]
pub struct RegexBuilder {
    postfix: Vec<RegexHirNode>,
    states: Vec<State>,
    frags: Vec<Fragment>,
    counters: Vec<usize>,
    /// Deduplicated byte-class lookup tables; indices are stored in
    /// [`RegexHirNode::ByteClass`] and [`State::ByteClass`].
    classes: IndexSet<ByteClass>,
    /// Byte dispatch tables created by the post-construction
    /// [`optimize_byte_tables`](Self::optimize_byte_tables) pass.
    byte_tables: Vec<ByteMap>,
    /// Maximum allowed `max` value for bounded repetitions (e.g.
    /// `a{1,1000}`).  Patterns exceeding this limit are rejected at
    /// compile time.  Default: 1000.
    max_repetition: usize,
    /// Maximum NFA states that may be produced by unrolling a repetition
    /// (fixed or non-fixed) of a simple body into concatenated copies.
    /// Set to 0 to disable unrolling entirely.  Default: 32.
    max_unroll_states: usize,
}

impl Default for RegexBuilder {
    fn default() -> Self {
        Self {
            postfix: Vec::new(),
            states: Vec::new(),
            frags: Vec::new(),
            counters: Vec::new(),
            classes: IndexSet::new(),
            byte_tables: Vec::new(),
            max_repetition: 1000,
            max_unroll_states: DEFAULT_MAX_UNROLL_STATES,
        }
    }
}
impl RegexBuilder {
    /// Set the maximum allowed `max` value for bounded repetitions
    /// (e.g. `a{1,1000}`).  Patterns exceeding this limit are rejected
    /// at compile time.  Default: 1000.
    pub fn max_repetition(&mut self, limit: usize) -> &mut Self {
        self.max_repetition = limit;
        self
    }

    /// Set the maximum NFA states that may be produced by unrolling a
    /// repetition of a simple body into concatenated copies.  Set to 0
    /// to disable unrolling entirely.  Default: 32.
    pub fn max_unroll_states(&mut self, limit: usize) -> &mut Self {
        self.max_unroll_states = limit;
        self
    }

    /// Allocate a fresh counter index.
    fn next_counter(&mut self) -> Result<CounterIdx, Error> {
        let counter = self.counters.len();
        if counter >= MAX_COUNTERS {
            return Err(Error::TooManyCounters);
        }
        self.counters.push(counter);
        Ok(CounterIdx(counter as u8))
    }

    /// Return the index of `table` in `self.classes`, inserting it if it
    /// is not already present.  Identical tables are deduplicated so that
    /// patterns like `\d{3,5}` (which unroll to multiple ByteClass states)
    /// share a single lookup table.
    fn intern_class(&mut self, table: ByteClass) -> ClassIdx {
        let (idx, _) = self.classes.insert_full(table);
        ClassIdx(idx)
    }

    /// Detect a case-insensitive ASCII letter pair in a byte-class.
    ///
    /// Returns `Some(lowercase_byte)` if the class has exactly two
    /// single-byte ranges that are the upper/lowercase pair of the same
    /// ASCII letter (e.g. `[Cc]`).
    fn detect_ci_letter_bytes(ranges: &[hir::ClassBytesRange]) -> Option<u8> {
        if ranges.len() != 2 {
            return None;
        }
        let (a_lo, a_hi) = (ranges[0].start(), ranges[0].end());
        let (b_lo, b_hi) = (ranges[1].start(), ranges[1].end());
        // Both ranges must be single bytes.
        if a_lo != a_hi || b_lo != b_hi {
            return None;
        }
        // One must be uppercase, the other lowercase of the same letter.
        let (lo, hi) = if a_lo < b_lo {
            (a_lo, b_lo)
        } else {
            (b_lo, a_lo)
        };
        if hi.is_ascii_lowercase() && lo.is_ascii_uppercase() && lo | 0x20 == hi {
            Some(hi) // lowercase
        } else {
            None
        }
    }

    /// Detect a case-insensitive ASCII letter pair in a Unicode class.
    ///
    /// Same logic as [`detect_ci_letter_bytes`] but for `ClassUnicodeRange`.
    fn detect_ci_letter_unicode(ranges: &[hir::ClassUnicodeRange]) -> Option<u8> {
        if ranges.len() != 2 {
            return None;
        }
        let (a_lo, a_hi) = (ranges[0].start() as u32, ranges[0].end() as u32);
        let (b_lo, b_hi) = (ranges[1].start() as u32, ranges[1].end() as u32);
        // Both ranges must be single bytes in ASCII.
        if a_lo != a_hi || b_lo != b_hi || a_lo > 0x7F || b_lo > 0x7F {
            return None;
        }
        let (lo, hi) = if a_lo < b_lo {
            (a_lo as u8, b_lo as u8)
        } else {
            (b_lo as u8, a_lo as u8)
        };
        if hi.is_ascii_lowercase() && lo.is_ascii_uppercase() && lo | 0x20 == hi {
            Some(hi) // lowercase
        } else {
            None
        }
    }

    /// Compute byte equivalence classes for DFA transition table compression.
    ///
    /// Two bytes are in the same class when, for every NFA consuming state,
    /// they produce the same match outcome (both match to the same target,
    /// or both fail).  Additionally, when deferred assertions are present,
    /// bytes with different word-ness are always in different classes.
    ///
    /// The purpose is to reduce the DFA transition table stride from 256 to
    /// the number of classes, shrinking each state row and allowing more
    /// states to fit in cache.  This does **not** speed up per-byte
    /// transitions — it adds ~2-3 instructions of indirection overhead per
    /// byte.  The win comes from keeping the table small enough to stay in
    /// L1/L2 cache when the DFA has many states.
    ///
    /// Returns `(mapping, num_classes)` where `mapping[byte]` is the class
    /// index for that byte and `num_classes` is the total count.
    fn compute_byte_classes(
        states: &[State],
        classes: &[ByteClass],
        byte_tables: &[ByteMap],
        has_deferred_assert: bool,
    ) -> ([u8; 256], usize) {
        // For each byte, build a signature: a compact key that captures
        // its behavior across all consuming states.  Bytes with identical
        // signatures are placed in the same equivalence class.
        //
        // Signature element per consuming state:
        //   0            → byte does not match this state
        //   1 + out.idx  → byte matches, transitions to `out`
        //   (for ByteTable: 1 + table_target.idx, or 0 if NONE)

        // Collect consuming state indices.
        let consuming: Vec<usize> = states
            .iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                State::Byte { .. }
                | State::ByteCI { .. }
                | State::ByteClass { .. }
                | State::ByteTable { .. } => Some(i),
                _ => None,
            })
            .collect();

        // Build signature for each byte.
        // To keep things efficient, we hash signatures on the fly.
        let mut class_map: HashMap<Vec<u32>, u8> = HashMap::default();
        let mut mapping = [0u8; 256];
        let mut next_class: u8 = 0;

        for b in 0..=255u8 {
            let mut sig: Vec<u32> = Vec::with_capacity(consuming.len() + 1);

            // First element: word-ness (when deferred assertions are present).
            if has_deferred_assert {
                sig.push(if is_word_byte(b) { 1 } else { 0 });
            }

            for &si in &consuming {
                let action = match states[si] {
                    State::Byte { byte, out } => {
                        if b == byte {
                            1 + out.0
                        } else {
                            0
                        }
                    }
                    State::ByteCI { byte, out } => {
                        if byte_match_ci(b, byte) {
                            1 + out.0
                        } else {
                            0
                        }
                    }
                    State::ByteClass { class, out } => {
                        if classes[class][b] {
                            1 + out.0
                        } else {
                            0
                        }
                    }
                    State::ByteTable { table } => {
                        let t = byte_tables[table][b];
                        if t != StateIdx::NONE { 1 + t.0 } else { 0 }
                    }
                    _ => 0,
                };
                sig.push(action);
            }

            if let Some(&class) = class_map.get(&sig) {
                mapping[b as usize] = class;
            } else {
                let class = next_class;
                class_map.insert(sig, class);
                mapping[b as usize] = class;
                // Safety: we cannot have more than 256 distinct classes.
                next_class = next_class
                    .checked_add(1)
                    .expect("more than 256 byte classes");
            }
        }

        (mapping, next_class as usize)
    }

    /// Recursively lower a `regex-syntax` HIR node into a postfix sequence
    /// appended to `self.postfix`.
    ///
    /// Bounded repetitions are lowered to a single `CounterLoop` node
    /// that wires CI → body → CInc in a single-copy loop.  Nested
    /// repetitions share the same body copy (each gets its own counter
    /// index, no remapping needed).
    /// Estimate the number of NFA states a single copy of `hir` would
    /// produce.  Returns `Some(n)` for bodies we can unroll, `None` for
    /// bodies that are too complex or would require special handling
    /// (e.g. nested non-fixed repetitions that would need their own
    /// counter).  Fixed inner repetitions are estimated recursively.
    fn estimate_nfa_states(hir: &Hir) -> Option<usize> {
        match hir.kind() {
            // Single-byte literal → 1 Byte state.
            HirKind::Literal(lit) if lit.0.len() == 1 => Some(1),
            // Multi-byte literal → N Byte states.
            HirKind::Literal(lit) => Some(lit.0.len()),
            // Byte or Unicode class → 1 ByteClass / ByteCI state.
            HirKind::Class(_) => Some(1),
            // Assertion → 1 Assert state.
            HirKind::Look(_) => Some(1),
            // Wildcard (.) → 1 state.
            HirKind::Empty => Some(0),
            // Capture is just a wrapper.
            HirKind::Capture(cap) => Self::estimate_nfa_states(&cap.sub),
            // Concatenation: sum of children.
            HirKind::Concat(children) => {
                let mut total = 0;
                for child in children {
                    total += Self::estimate_nfa_states(child)?;
                }
                Some(total)
            }
            // Alternation: sum of children + (N-1) Split states.
            HirKind::Alternation(children) => {
                let mut total = 0;
                let mut count = 0;
                for child in children {
                    total += Self::estimate_nfa_states(child)?;
                    count += 1;
                }
                if count > 1 {
                    total += count - 1; // Split states
                }
                Some(total)
            }
            HirKind::Repetition(rep) => {
                let inner = Self::estimate_nfa_states(&rep.sub)?;
                let min = rep.min as usize;
                let max = rep.max.map_or(usize::MAX, |m| m as usize);
                if min == 0 && max == 1 {
                    // `?` → inner + 1 Split
                    Some(inner + 1)
                } else if min == 0 && max == usize::MAX {
                    // `*` → inner + 1 Split
                    Some(inner + 1)
                } else if min == 1 && max == usize::MAX {
                    // `+` → inner + 1 Split
                    Some(inner + 1)
                } else if min == max {
                    // Fixed: can be unrolled — N copies.
                    Some(min * inner)
                } else {
                    // Non-fixed bounded/unbounded: will use a counter
                    // (CI + body + CInc) or be unrolled.  Estimate the
                    // counter path cost: inner + 2 (CI + CInc) + 1 (Split
                    // for the loop).  If min==0, add 1 for the outer `?`.
                    let counter_cost = inner + 3 + if min == 0 { 1 } else { 0 };
                    Some(counter_cost)
                }
            }
        }
    }

    /// Try to unroll `X{min,max}` (where `min >= 1`) into concatenated
    /// copies, eliminating the counter.  Returns `Ok(true)` if unrolling
    /// was performed.
    ///
    /// Three cases:
    /// - **Fixed** (`min == max`): emit `min` copies concatenated.
    ///   Cost: `min * body_nfa` states.
    /// - **Bounded non-fixed** (`min < max < ∞`): emit `min` mandatory
    ///   copies, then `(max - min)` nested optional copies.
    ///   Cost: `max * body_nfa + (max - min)` states (Split per `?`).
    /// - **Unbounded** (`min >= 2`, `max == ∞`): emit `(min - 1)` copies
    ///   then `X+`.  Cost: `min * body_nfa + 1` states (Split for `+`).
    fn try_unroll(
        &mut self,
        min: usize,
        max: usize,
        body_nfa: usize,
        limit: usize,
        sub: &Hir,
    ) -> Result<bool, Error> {
        if body_nfa == 0 || limit == 0 {
            return Ok(false);
        }

        if min == max {
            // Fixed: X{N} → X·X·…·X  (N copies)
            let cost = min * body_nfa;
            if cost > limit {
                return Ok(false);
            }
            self.emit_n_copies(min, sub)?;
            return Ok(true);
        }

        if max != usize::MAX {
            // Bounded non-fixed: X{min,max}
            //   → X·…·X · (X·(X·(…·X?)?)?)?
            //   min mandatory + (max-min) nested optional
            //
            // Postfix sequence (max body copies total):
            //   X×max  ?  (Cat ?)×(optional-1)  Cat×min
            let optional = max - min;
            let cost = max * body_nfa + optional;
            if cost > limit {
                return Ok(false);
            }
            // Emit all max body copies.
            for _ in 0..max {
                self.hir2postfix(sub)?;
            }
            // Innermost optional: wrap last body in ?
            self.postfix.push(RegexHirNode::RepeatZeroOne);
            // Each remaining optional layer: Cat with previous body, then ?
            for _ in 1..optional {
                self.postfix.push(RegexHirNode::Catenate);
                self.postfix.push(RegexHirNode::RepeatZeroOne);
            }
            // Concatenate each mandatory body with the growing tail.
            for _ in 0..min {
                self.postfix.push(RegexHirNode::Catenate);
            }
            return Ok(true);
        }

        // Unbounded: X{min,∞} where min >= 2 → X·…·X·X+
        //   (min-1) copies then X+.  Cost: min*body_nfa + 1
        let cost = min * body_nfa + 1;
        if cost > limit {
            return Ok(false);
        }
        self.emit_n_copies(min - 1, sub)?;
        self.hir2postfix(sub)?;
        self.postfix.push(RegexHirNode::RepeatOnePlus);
        self.postfix.push(RegexHirNode::Catenate);
        Ok(true)
    }

    /// Emit `n` concatenated copies of `sub` in postfix form.
    /// Produces a single fragment on the stack: X·X·…·X.
    fn emit_n_copies(&mut self, n: usize, sub: &Hir) -> Result<(), Error> {
        for i in 0..n {
            self.hir2postfix(sub)?;
            if i > 0 {
                self.postfix.push(RegexHirNode::Catenate);
            }
        }
        Ok(())
    }

    fn hir2postfix(&mut self, hir: &Hir) -> Result<(), Error> {
        match hir.kind() {
            HirKind::Empty => {
                // Empty matches the empty string.  We still need a fragment
                // on the stack, so emit a Wildcard wrapped in ZeroOne (?).
                // Actually, simpler: just don't push anything if this is
                // inside a Concat.  But as a standalone node we need a
                // fragment.  Use a zero-width match: a split that always
                // takes the skip path.  Easiest: emit nothing and let the
                // caller handle it via Concat/Alternation.  But for
                // standalone Empty, we need a fragment.  Emit a ZeroOne
                // around a Wildcard — no, that would match a character.
                // The correct approach: emit no fragment.  But build()
                // expects exactly one fragment.  Let's handle this by
                // checking: if postfix is empty after hir2postfix, treat
                // it as matching the empty string (which is what the
                // start->Match NFA does).  Actually, the simplest thing:
                // don't emit anything.  If Empty appears in a Concat, the
                // Concat logic handles it.  If standalone, postfix is empty
                // and build() will see an empty fragment stack — we handle
                // that specially.
                Ok(())
            }
            HirKind::Literal(lit) => {
                let bytes = &lit.0;
                for (idx, &b) in bytes.iter().enumerate() {
                    self.postfix.push(RegexHirNode::Byte(b));
                    if idx > 0 {
                        self.postfix.push(RegexHirNode::Catenate);
                    }
                }
                Ok(())
            }
            HirKind::Class(hir::Class::Bytes(class)) => {
                // Detect case-insensitive ASCII letter pair: exactly two
                // single-byte ranges like [C-C][c-c].
                if let Some(lower) = Self::detect_ci_letter_bytes(class.ranges()) {
                    self.postfix.push(RegexHirNode::ByteCI(lower));
                    return Ok(());
                }
                let mut table = ByteClass::NONE;
                for range in class.ranges() {
                    for b in range.start()..=range.end() {
                        table.0[b as usize] = true;
                    }
                }
                let idx = self.intern_class(table);
                self.postfix.push(RegexHirNode::ByteClass(idx));
                Ok(())
            }
            HirKind::Class(hir::Class::Unicode(class)) => {
                // regex-syntax may produce Unicode classes for ASCII-only
                // patterns like `(a|b)` �� `[ab]`.  If all ranges fit in a
                // single byte (0x00..=0xFF), lower them to a ByteClass;
                // otherwise reject.
                let ranges = class.ranges();
                let all_single_byte = ranges
                    .iter()
                    .all(|r| (r.start() as u32) <= 0xFF && (r.end() as u32) <= 0xFF);
                if !all_single_byte {
                    return Err(Error::UnsupportedClass(hir::Class::Unicode(class.clone())));
                }
                // Detect case-insensitive ASCII letter pair: exactly two
                // single-char ranges like [C-C][c-c].
                if let Some(lower) = Self::detect_ci_letter_unicode(ranges) {
                    self.postfix.push(RegexHirNode::ByteCI(lower));
                    return Ok(());
                }
                let mut table = ByteClass::NONE;
                for range in ranges {
                    for b in (range.start() as u8)..=(range.end() as u8) {
                        table.0[b as usize] = true;
                    }
                }
                let idx = self.intern_class(table);
                self.postfix.push(RegexHirNode::ByteClass(idx));
                Ok(())
            }
            HirKind::Look(look) => {
                let kind = match look {
                    hir::Look::Start => AssertKind::Start,
                    hir::Look::End => AssertKind::End,
                    hir::Look::StartLF => AssertKind::StartLF,
                    hir::Look::EndLF => AssertKind::EndLF,
                    hir::Look::StartCRLF => AssertKind::StartCRLF,
                    hir::Look::EndCRLF => AssertKind::EndCRLF,
                    hir::Look::WordAscii => AssertKind::WordAscii,
                    hir::Look::WordAsciiNegate => AssertKind::WordAsciiNegate,
                    hir::Look::WordStartAscii => AssertKind::WordStartAscii,
                    hir::Look::WordEndAscii => AssertKind::WordEndAscii,
                    _ => return Err(Error::UnsupportedLook(*look)),
                };
                self.postfix.push(RegexHirNode::Assert(kind));
                Ok(())
            }
            HirKind::Capture(cap) => self.hir2postfix(&cap.sub),
            HirKind::Concat(children) => {
                let mut count = 0;
                for child in children {
                    let before = self.postfix.len();
                    self.hir2postfix(child)?;
                    // Only emit Catenate if the child actually produced
                    // output (Empty produces nothing).
                    if self.postfix.len() > before {
                        count += 1;
                        if count > 1 {
                            self.postfix.push(RegexHirNode::Catenate);
                        }
                    }
                }
                Ok(())
            }
            HirKind::Alternation(children) => {
                let mut count = 0;
                let mut has_empty = false;
                for child in children {
                    let before = self.postfix.len();
                    self.hir2postfix(child)?;
                    if self.postfix.len() > before {
                        count += 1;
                        if count > 1 {
                            self.postfix.push(RegexHirNode::Alternate);
                        }
                    } else {
                        // Child produced nothing (Empty).  Record it so we
                        // can wrap the whole alternation in `?` below.
                        has_empty = true;
                    }
                }
                // An empty alternative means the pattern can match the
                // empty string at this position, i.e. the whole alternation
                // is optional.
                if has_empty && count > 0 {
                    self.postfix.push(RegexHirNode::RepeatZeroOne);
                }
                Ok(())
            }
            HirKind::Repetition(rep) => {
                let min = rep.min as usize;
                let max = rep.max.map_or(usize::MAX, |m| m as usize);
                assert!(min <= max);

                // {0,0} matches exactly zero times — semantically Empty.
                // Don't emit anything; callers (Concat, Alternation, build)
                // already handle missing fragments.
                if max == 0 {
                    return Ok(());
                }

                // Reject repetitions exceeding the compile-time cap.
                if max != usize::MAX && max > self.max_repetition {
                    return Err(Error::RepetitionTooLarge(max, self.max_repetition));
                }

                // Special-case common quantifiers to avoid counter overhead.
                if min == 0 && max == 1 {
                    // `?`
                    let before = self.postfix.len();
                    self.hir2postfix(&rep.sub)?;
                    if self.postfix.len() > before {
                        self.postfix.push(RegexHirNode::RepeatZeroOne);
                    }
                    return Ok(());
                }
                if min == 0 && max == usize::MAX {
                    // `*`
                    let before = self.postfix.len();
                    self.hir2postfix(&rep.sub)?;
                    if self.postfix.len() > before {
                        self.postfix.push(RegexHirNode::RepeatZeroPlus);
                    }
                    return Ok(());
                }
                if min == 1 && max == usize::MAX {
                    // `+`
                    let before = self.postfix.len();
                    self.hir2postfix(&rep.sub)?;
                    if self.postfix.len() > before {
                        self.postfix.push(RegexHirNode::RepeatOnePlus);
                    }
                    return Ok(());
                }

                // Try to unroll simple bodies to eliminate the counter.
                let body_nfa = Self::estimate_nfa_states(&rep.sub).unwrap_or(0);
                let limit = self.max_unroll_states;

                if min > 0 && self.try_unroll(min, max, body_nfa, limit, &rep.sub)? {
                    // Successfully unrolled — no counter needed.
                } else if min == 0
                    && max != usize::MAX
                    && self.try_unroll(1, max, body_nfa, limit.saturating_sub(1), &rep.sub)?
                {
                    // {0,max}: unrolled inner {1,max}, wrap in `?`.
                    // (limit-1 because the outer `?` adds 1 Split state.)
                    self.postfix.push(RegexHirNode::RepeatZeroOne);
                } else if min > 0 {
                    let counter = self.next_counter()?;
                    let before = self.postfix.len();
                    self.hir2postfix(&rep.sub)?;
                    if self.postfix.len() > before {
                        self.postfix
                            .push(RegexHirNode::CounterLoop { counter, min, max });
                    }
                    // else: body is empty — repetition of nothing is nothing.
                } else {
                    // {0,max}: lower to (body{1,max})? — the `?` wrapping
                    // provides the zero-match path.
                    let counter = self.next_counter()?;
                    let before = self.postfix.len();
                    self.hir2postfix(&rep.sub)?;
                    if self.postfix.len() > before {
                        self.postfix.push(RegexHirNode::CounterLoop {
                            counter,
                            min: 1,
                            max,
                        });
                        self.postfix.push(RegexHirNode::RepeatZeroOne);
                    }
                    // else: body is empty — repetition of nothing is nothing.
                }
                Ok(())
            }
        }
    }

    /// Push a new NFA state and return its index.
    fn state(&mut self, state: State) -> StateIdx {
        let idx = StateIdx(self.states.len() as u32);
        self.states.push(state);
        idx
    }

    /// Walk the linked list of dangling `out` pointers starting at `list`
    /// and patch each one to point to `idx`.
    fn patch(&mut self, mut list: StateIdx, idx: StateIdx) {
        while let Some(state) = self.states.get_mut_state(list) {
            list = match state {
                State::Byte { out, .. }
                | State::ByteCI { out, .. }
                | State::ByteClass { out, .. }
                | State::CounterInstance { out, .. }
                | State::Assert { out, .. } => {
                    let next = *out;
                    *out = idx;
                    next
                }
                State::Split { out1, .. } | State::CounterIncrement { out1, .. } => {
                    let next = *out1;
                    *out1 = idx;
                    next
                }
                _ => panic!("patch: unexpected state {:?}", state),
            };
        }
    }

    /// Append `list2` to the end of the dangling-pointer chain starting at
    /// `list1`.
    fn append(&mut self, list1: StateIdx, list2: StateIdx) -> StateIdx {
        let len = self.states.len();
        let mut s = &mut self.states.as_mut_slice()[list1];
        let mut next = s.next();
        while next.raw() < len {
            s = &mut self.states.as_mut_slice()[next];
            next = s.next();
        }
        s.append(list2);
        list1
    }

    /// Consume one postfix HIR node and return the corresponding NFA
    /// fragment.
    #[inline]
    fn next_fragment(&mut self, node: RegexHirNode) -> Fragment {
        match node {
            RegexHirNode::Catenate => {
                let e2 = self.frags.pop().unwrap();
                let e1 = self.frags.pop().unwrap();
                self.patch(e1.out, e2.start);
                Fragment::new(e1.start, e2.out)
            }
            RegexHirNode::Alternate => {
                let e2 = self.frags.pop().unwrap();
                let e1 = self.frags.pop().unwrap();
                let s = self.state(State::Split {
                    out: e1.start,
                    out1: e2.start,
                });
                Fragment::new(s, self.append(e1.out, e2.out))
            }
            RegexHirNode::RepeatZeroOne => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::Split {
                    out: e.start,
                    out1: StateIdx::NONE,
                });
                Fragment::new(s, self.append(e.out, s))
            }
            RegexHirNode::RepeatZeroPlus => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::Split {
                    out: e.start,
                    out1: StateIdx::NONE,
                });
                self.patch(e.out, s);
                Fragment::new(s, s)
            }
            RegexHirNode::RepeatOnePlus => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::Split {
                    out: e.start,
                    out1: StateIdx::NONE,
                });
                self.patch(e.out, s);
                Fragment::new(e.start, s)
            }
            RegexHirNode::CounterLoop { counter, min, max } => {
                // Single-copy NFA:
                //   CI → body.start → ... → body.end → CInc
                //          ↑                            | (continue)
                //          └────────────────────────────┘
                //                                       | (break)
                //                                       ↓ [exit]
                let body = self.frags.pop().unwrap();
                let cinc = self.state(State::CounterIncrement {
                    out: body.start,      // continue → body start
                    out1: StateIdx::NONE, // break (dangling exit)
                    min,
                    max,
                    counter,
                });
                self.patch(body.out, cinc); // body end → CInc
                let ci = self.state(State::CounterInstance {
                    counter,
                    out: body.start, // CI → body start
                });
                Fragment::new(ci, cinc) // entry=CI, exit=CInc.out1
            }
            RegexHirNode::ByteClass(class) => {
                let idx = self.state(State::ByteClass {
                    class,
                    out: StateIdx::NONE,
                });
                Fragment::new(idx, idx)
            }
            RegexHirNode::ByteCI(byte) => {
                let idx = self.state(State::ByteCI {
                    byte,
                    out: StateIdx::NONE,
                });
                Fragment::new(idx, idx)
            }
            RegexHirNode::Byte(byte) => {
                let idx = self.state(State::Byte {
                    byte,
                    out: StateIdx::NONE,
                });
                Fragment::new(idx, idx)
            }
            RegexHirNode::Assert(kind) => {
                let idx = self.state(State::Assert {
                    kind,
                    out: StateIdx::NONE,
                });
                Fragment::new(idx, idx)
            }
        }
    }

    /// Minimum NFA state count to enable byte equivalence class compression.
    ///
    /// Below this threshold the DFA uses stride=256 (identity mapping) to
    /// avoid the ~2-3 insn/byte indirection overhead.  Above it, byte
    /// classes compress the transition table so more DFA states fit in cache.
    const BYTE_CLASSES_NFA_THRESHOLD: usize = 128;

    /// Compile a `regex-syntax` HIR into a ready-to-match [`Regex`].
    pub fn build(&mut self, hir: &Hir) -> Result<Regex, Error> {
        self.states.clear();
        self.frags.clear();
        self.postfix.clear();
        self.counters.clear();
        self.classes.clear();
        self.byte_tables.clear();
        self.hir2postfix(hir)?;

        let mut postfix = std::mem::take(&mut self.postfix);
        for node in postfix.drain(..) {
            let frag = self.next_fragment(node);
            self.frags.push(frag);
        }
        self.postfix = postfix;

        // Handle the empty-regex case (e.g. HirKind::Empty): no fragments
        // were produced, so we just create a Match state directly.
        let start = if let Some(e) = self.frags.pop() {
            assert!(self.frags.is_empty());
            let s = self.state(State::Match);
            self.patch(e.out, s);
            e.start
        } else {
            self.state(State::Match)
        };

        self.optimize_byte_tables();
        let (start_closure, start_closure_matches) = self.compute_start_closure(start);

        // DFA eligibility: no counters and no CRLF assertion types.
        let has_counters = !self.counters.is_empty();
        // Tier 1 DFA handles deferred assertions (WordAscii, WordAsciiNegate,
        // EndLF) natively.  Only StartCRLF and EndCRLF remain ineligible.
        let has_crlf_assert = self.states.iter().any(|s| {
            matches!(s,
                State::Assert { kind, .. } if matches!(kind,
                    AssertKind::EndCRLF
                    | AssertKind::StartCRLF
                )
            )
        });
        let dfa_eligible = !has_counters && !has_crlf_assert;

        // Tier 4 counting DFA does NOT handle deferred assertions yet.
        let has_deferred_assert = self.states.iter().any(|s| {
            matches!(s,
                State::Assert { kind, .. } if matches!(kind,
                    AssertKind::EndLF
                    | AssertKind::EndCRLF
                    | AssertKind::StartCRLF
                    | AssertKind::WordAscii
                    | AssertKind::WordAsciiNegate
                    | AssertKind::WordStartAscii
                    | AssertKind::WordEndAscii
                )
            )
        });

        // The counting DFA cannot handle zero-width counter bodies
        // (where the body can match empty, creating epsilon-only loops
        // through CounterIncrement).  Detect this by checking if any
        // CounterInstance can reach its corresponding CounterIncrement
        // via epsilon transitions only.
        let has_zero_width_counter_body = has_counters && {
            let states = &self.states;
            fn can_reach_cinc_via_epsilon(
                idx: StateIdx,
                counter: CounterIdx,
                states: &[State],
                visited: &mut Vec<bool>,
            ) -> bool {
                let i = idx.idx();
                if visited[i] {
                    return false;
                }
                visited[i] = true;
                match states[idx] {
                    State::CounterIncrement { counter: c, .. } if c == counter => true,
                    State::Split { out, out1 } => {
                        can_reach_cinc_via_epsilon(out, counter, states, visited)
                            || can_reach_cinc_via_epsilon(out1, counter, states, visited)
                    }
                    State::Assert { out, .. } => {
                        can_reach_cinc_via_epsilon(out, counter, states, visited)
                    }
                    State::CounterInstance { out, .. } => {
                        can_reach_cinc_via_epsilon(out, counter, states, visited)
                    }
                    // Consuming states or Match block the path.
                    _ => false,
                }
            }

            states.iter().any(|s| {
                if let State::CounterInstance { counter, out } = s {
                    let mut visited = vec![false; states.len()];
                    can_reach_cinc_via_epsilon(*out, *counter, states, &mut visited)
                } else {
                    false
                }
            })
        };
        let tier4_eligible = has_counters && !has_deferred_assert && !has_zero_width_counter_body;

        // Check for deferred assertions inside a counter body.  Tier 3
        // cannot handle these at all.  Tier 2 can handle them for L=1
        // bodies (the single-phase model correctly defers the CInc to
        // the next byte via Phase 1 resolution + cinc_mask), but NOT
        // for L>1 bodies (counter_reset can prematurely clear instances
        // when the NFA is between the last consuming state and CInc).
        fn body_has_deferred(
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
                    // Follow through consuming states -- deferred
                    // assertions may appear after a byte match within
                    // the counter body (e.g. `(?m:a$){2,3}`).
                    State::Byte { out, .. }
                    | State::ByteCI { out, .. }
                    | State::ByteClass { out, .. } => stack.push(out),
                    State::ByteTable { table } => {
                        for &succ in byte_tables[table.idx()].0.iter() {
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

        let has_deferred_in_counter_body = has_counters && {
            let states = &self.states;
            let byte_tables = &self.byte_tables;
            states.iter().any(|s| {
                if let State::CounterInstance { counter, out } = s {
                    body_has_deferred(*out, *counter, states, byte_tables)
                } else {
                    false
                }
            })
        };

        // Non-nested counter eligibility: shared base for tier 2 and tier 3.
        // Does NOT include the deferred-in-counter-body check (tier 2 handles
        // deferred assertions inside counter bodies via Phase 1 resolution;
        // tier 3 does not).
        let non_nested_eligible =
            has_counters && !has_crlf_assert && !has_zero_width_counter_body && {
                let states = &self.states;
                fn has_nesting(states: &[State]) -> bool {
                    for s in states.iter() {
                        if let State::CounterInstance { counter, out } = s {
                            // Walk from CI.out to CInc(counter) through epsilon
                            // states.  If we encounter another counter's CI or
                            // CInc on the way, it's nested.
                            if body_crosses_other_counter(*out, *counter, states) {
                                return true;
                            }
                        }
                    }
                    false
                }
                fn body_crosses_other_counter(
                    start: StateIdx,
                    own_counter: CounterIdx,
                    states: &[State],
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
                            State::CounterInstance { counter, out } => {
                                if counter != own_counter {
                                    return true; // nested: another CI in body
                                }
                                stack.push(out);
                            }
                            State::CounterIncrement {
                                counter,
                                out: _,
                                out1: _,
                                ..
                            } => {
                                if counter == own_counter {
                                    // Reached our own CInc — don't follow further
                                    // (we only check the body, not past the break).
                                    continue;
                                }
                                return true; // nested: another CInc in body
                            }
                            State::Split { out, out1 } => {
                                stack.push(out1);
                                stack.push(out);
                            }
                            State::Assert { out, .. } => {
                                stack.push(out);
                            }
                            // Follow consuming states to detect nesting
                            // behind byte-consuming instructions.
                            State::Byte { out, .. }
                            | State::ByteCI { out, .. }
                            | State::ByteClass { out, .. } => {
                                stack.push(out);
                            }
                            // ByteTable dispatches via a table — skip it.
                            // Nested counters behind ByteTable are extremely
                            // unlikely in practice.
                            State::ByteTable { .. } | State::Match => {}
                        }
                    }
                    false
                }
                // Check for counter self-loops: CInc.out1 (break) can
                // reach CI for the same counter.  This happens with
                // patterns like `(.{6,9})+` where the `+` wraps a counted
                // repetition.  After break, CI is re-entered, creating
                // overlapping instances for the same counter from different
                // repetition rounds.  The differential counter model
                // (tier 2) and per-instance tracking (tier 3) cannot
                // handle this correctly — instances accumulate and
                // spuriously satisfy the break condition.
                fn has_self_loop(states: &[State]) -> bool {
                    for s in states.iter() {
                        if let State::CounterIncrement { counter, out1, .. } = *s {
                            // Walk from break path looking for CI of same counter.
                            let mut stack = vec![out1];
                            let mut visited = vec![false; states.len()];
                            while let Some(idx) = stack.pop() {
                                let i = idx.idx();
                                if i >= states.len() || visited[i] {
                                    continue;
                                }
                                visited[i] = true;
                                match states[idx] {
                                    State::CounterInstance { counter: c, .. } if c == counter => {
                                        return true;
                                    }
                                    State::Split { out, out1 } => {
                                        stack.push(out1);
                                        stack.push(out);
                                    }
                                    State::Assert { out, .. } => stack.push(out),
                                    State::CounterInstance { out, .. } => {
                                        stack.push(out);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                    false
                }
                !has_nesting(states) && !has_self_loop(states)
            };

        // Tier 3 eligibility: non-nested counters WITHOUT deferred assertions
        // inside counter bodies.  Tier 3 cannot handle deferred assertions in
        // counter bodies because it lacks the Phase 1 deferred resolution that
        // tier 2 has.
        //
        // Also limited to MAX_TIER2_COUNTERS (64) counters: tier 3 uses a
        // `u64` bitmask (`counter_broke`) to gate break-seeds, so counter
        // indices ≥ 64 would alias lower indices and corrupt seeding.
        // TODO: support >64 counters by replacing the `u64` bitmask with a
        // wider representation (e.g. `[u64; 4]` for 256 counters), or by
        // lowering MAX_COUNTERS to 64 globally.
        let tier3_eligible = non_nested_eligible
            && !has_deferred_in_counter_body
            && self.counters.len() <= MAX_TIER2_COUNTERS;

        // Compute per-counter info: (min, max, body_byte_length).
        // body_byte_length is the fixed number of bytes consumed per iteration,
        // or 0 if the body has variable length.
        #[allow(clippy::type_complexity)]
        let (counter_info, disjoint_bytes): (Box<[(usize, usize, usize)]>, bool) = if has_counters {
            let states = &self.states;
            let byte_tables = &self.byte_tables;

            /// Check that the byte sets matched by different counter bodies
            /// are pairwise disjoint.  When this holds, at most one counter
            /// fires CInc on any DFA transition, so the binary
            /// with_break/no_break split is correct for multiple counters.
            fn counter_bodies_have_disjoint_bytes(
                states: &[State],
                classes: &indexmap::set::IndexSet<ByteClass>,
                byte_tables: &[ByteMap],
            ) -> bool {
                // Collect per-counter byte sets (as [bool; 256]).
                let mut counter_bytes: Vec<(CounterIdx, [bool; 256])> = Vec::new();
                for s in states.iter() {
                    if let State::CounterInstance { counter, out } = s {
                        let mut bytes = [false; 256];
                        // Walk the body from CI.out to find all consuming
                        // states, stopping at CInc for the same counter.
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
                                State::Byte { byte, out } => {
                                    bytes[byte as usize] = true;
                                    stack.push(out);
                                }
                                State::ByteCI { byte, out } => {
                                    bytes[byte as usize] = true;
                                    bytes[(byte ^ 0x20) as usize] = true;
                                    stack.push(out);
                                }
                                State::ByteClass { class, out } => {
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
                // Pairwise disjointness check.
                for i in 0..counter_bytes.len() {
                    for j in (i + 1)..counter_bytes.len() {
                        if counter_bytes[i].0 == counter_bytes[j].0 {
                            continue; // same counter (shouldn't happen)
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

            /// Compute the fixed byte-length of a counter body.
            /// Returns `Some(len)` if all paths through the body consume
            /// exactly `len` bytes, `None` if variable-length.
            fn counter_body_length(
                ci_out: StateIdx,
                own_counter: CounterIdx,
                states: &[State],
                byte_tables: &[ByteMap],
            ) -> Option<usize> {
                // DFS from ci_out (the body entry).  Each path counts
                // consuming states until it reaches CInc(own_counter).
                // All paths must agree on the count.
                let mut result: Option<usize> = None;
                let mut stack: Vec<(StateIdx, usize)> = vec![(ci_out, 0)];
                let mut visited: Vec<Option<usize>> = vec![None; states.len()];
                // Use a separate "in_stack" to avoid infinite loops on
                // cycles that don't pass through CInc.
                let mut in_stack = vec![false; states.len()];
                while let Some((idx, depth)) = stack.pop() {
                    let i = idx.idx();
                    in_stack[i] = false;
                    match states[idx] {
                        State::CounterIncrement { counter, .. } if counter == own_counter => {
                            // Reached our CInc: this path consumed `depth` bytes.
                            match result {
                                None => result = Some(depth),
                                Some(prev) if prev != depth => return None,
                                _ => {}
                            }
                        }
                        State::Split { out, out1 } => {
                            for succ in [out, out1] {
                                let si = succ.idx();
                                if !in_stack[si] {
                                    match visited[si] {
                                        Some(d) if d == depth => {} // already explored at same depth
                                        Some(_) => return None,     // different depth = variable
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
                        State::Byte { out, .. }
                        | State::ByteCI { out, .. }
                        | State::ByteClass { out, .. } => {
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
                            for &succ in byte_tables[table.idx()].0.iter() {
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
                        // Match or other terminal: shouldn't happen in a
                        // well-formed counter body, but treat as dead end.
                        _ => {}
                    }
                }
                result
            }

            let num_counters = self.counters.len();
            let mut info = vec![(0usize, 0usize, 0usize); num_counters];
            // First, collect min/max from CInc states.
            for s in states.iter() {
                if let State::CounterIncrement {
                    counter, min, max, ..
                } = s
                {
                    info[counter.idx()] = (*min, *max, 0);
                }
            }
            // Then, compute body lengths from CI states.
            for s in states.iter() {
                if let State::CounterInstance { counter, out } = s {
                    let ci = counter.idx();
                    let body_len =
                        counter_body_length(*out, *counter, states, byte_tables).unwrap_or(0);
                    info[ci].2 = body_len;
                }
            }
            let disjoint_bytes =
                counter_bodies_have_disjoint_bytes(states, &self.classes, byte_tables);

            (info.into_boxed_slice(), disjoint_bytes)
        } else {
            (Box::new([]), true)
        };

        // Tier 2 eligibility: non-nested counters with fixed-length bodies
        // and pairwise-disjoint byte sets.
        //
        // Uses non_nested_eligible (not tier3_eligible) because tier 2
        // supports deferred assertions inside L=1 counter bodies via its
        // Phase 1 deferred resolution + cinc_mask mechanism.  For L>1
        // bodies, deferred assertions break the phase-clock model (the
        // counter_reset logic can prematurely clear instances when the
        // NFA is between the last consuming state and CInc due to a
        // deferred assertion), so they remain ineligible.
        //
        // Additional requirements beyond non-nested:
        //  - every counter body must have a fixed byte-length (> 0),
        //  - the byte sets of different counter bodies must not overlap,
        //  - at most MAX_TIER2_COUNTERS counters (u64 bitmask limit), and
        //  - if a body contains a deferred assertion, body_len must be 1.
        //
        // The disjoint-bytes condition guarantees that at most one counter
        // fires CInc per DFA transition, so the binary with_break/no_break
        // DFA split remains correct with multiple counters.
        let has_deferred_in_long_body = has_deferred_in_counter_body && {
            // Check if any counter with a deferred assertion in its body
            // has body_length > 1.  L=1 bodies are safe because the single
            // phase handles the 1-byte deferral correctly.
            let states = &self.states;
            let byte_tables = &self.byte_tables;
            states.iter().any(|s| {
                if let State::CounterInstance { counter, out } = s {
                    let ci = counter.idx();
                    let body_len = counter_info.get(ci).map_or(0, |info| info.2);
                    body_len > 1 && body_has_deferred(*out, *counter, states, byte_tables)
                } else {
                    false
                }
            })
        };
        // Multi-counter + deferred-in-body: the probe closure cannot
        // see past deferred assertions to follow CInc break paths, so
        // the with_break DFA successor misses the next counter's entry
        // states.  Single-counter is fine (break leads to Match, handled
        // by resolved_break_match).
        let has_deferred_in_multi_counter_body =
            has_deferred_in_counter_body && counter_info.len() > 1;

        let tier2_eligible = non_nested_eligible
            && !has_deferred_in_long_body
            && !has_deferred_in_multi_counter_body
            && !counter_info.is_empty()
            && counter_info.len() <= MAX_TIER2_COUNTERS
            && counter_info.iter().all(|&(_, _, body_len)| body_len > 0)
            && disjoint_bytes;

        // Compute byte equivalence classes before moving data out.
        let classes_slice: Vec<ByteClass> = self.classes.iter().copied().collect();
        let byte_tables_slice: Vec<ByteMap> = self.byte_tables.to_vec();
        // Deferred assertions need word-ness splitting.
        let has_any_deferred = self.states.iter().any(|s| {
            matches!(
                s,
                State::Assert { kind, .. } if matches!(
                    kind,
                    AssertKind::WordAscii
                    | AssertKind::WordAsciiNegate
                    | AssertKind::EndLF
                )
            )
        });
        let (byte_classes, num_byte_classes) =
            if self.states.len() >= Self::BYTE_CLASSES_NFA_THRESHOLD {
                Self::compute_byte_classes(
                    self.states.as_slice(),
                    &classes_slice,
                    &byte_tables_slice,
                    has_any_deferred,
                )
            } else {
                // Identity mapping: stride=256, no indirection overhead.
                let mut identity = [0u8; 256];
                for (i, slot) in identity.iter_mut().enumerate() {
                    *slot = i as u8;
                }
                (identity, 256)
            };

        let prefilter = Self::compute_prefilter(
            &start_closure,
            start_closure_matches,
            self.states.as_slice(),
            &classes_slice,
            &byte_tables_slice,
        );

        // Precompute can-reach-match for every NFA state.
        let state_can_reach_match = Self::compute_can_reach_match(self.states.as_slice());

        // Precompute per-counter: can the break path (out1) reach Match?
        let counter_break_can_match = Self::compute_counter_break_can_match(
            self.states.as_slice(),
            &state_can_reach_match,
            self.counters.len(),
        );

        // Precompute Tier 2 analysis if eligible.
        let tier2_analysis = if tier2_eligible {
            Some(compute_tier2_analysis(
                self.states.as_slice(),
                &byte_tables_slice,
                self.counters.len(),
            ))
        } else {
            None
        };

        // Precompute Tier 3 analysis if eligible.
        let tier3_analysis = if tier3_eligible {
            Some(compute_tier3_analysis(
                self.states.as_slice(),
                &byte_tables_slice,
                &state_can_reach_match,
                start,
            ))
        } else {
            None
        };

        Ok(Regex {
            id: NEXT_REGEX_ID.fetch_add(1, Ordering::Relaxed),
            states: StateList(self.states.to_vec().into_boxed_slice()),
            start,
            num_counters: self.counters.len(),
            classes: classes_slice.into_boxed_slice(),
            byte_tables: byte_tables_slice.into_boxed_slice(),
            start_closure,
            start_closure_matches,
            dfa_eligible,
            tier2_eligible,
            tier2_analysis,
            tier3_eligible,
            tier3_analysis,
            tier4_eligible,
            counter_info,
            byte_classes,
            num_byte_classes,
            prefilter,
            state_can_reach_match,
            counter_break_can_match,
        })
    }
    // -----------------------------------------------------------------------
    // Post-construction optimisation: collapse Split+Byte chains into
    // ByteTable dispatch tables.
    // -----------------------------------------------------------------------

    /// Try to collect all `Byte` leaf states reachable from `idx` through
    /// a pure `Split` chain.  Returns `None` if any leaf is not a `Byte`
    /// state (e.g. `ByteClass`, `CounterInstance`, etc.) or if two leaves
    /// share the same byte value.
    fn collect_byte_leaves(&self, idx: StateIdx, out: &mut Vec<(u8, StateIdx)>) -> bool {
        match self.states.as_slice()[idx] {
            State::Byte { byte, out: target } => {
                // Check for duplicate byte values.
                if out.iter().any(|&(b, _)| b == byte) {
                    return false;
                }
                out.push((byte, target));
                true
            }
            State::Split {
                out: left,
                out1: right,
            } => self.collect_byte_leaves(left, out) && self.collect_byte_leaves(right, out),
            _ => false,
        }
    }

    /// Scan all states for Split chains whose leaves are all `Byte` states
    /// with distinct byte values.  Replace the root Split with a
    /// [`State::ByteTable`] and mark interior states as dead (`Match`
    /// sentinels — they become unreachable).
    fn optimize_byte_tables(&mut self) {
        let mut leaves = Vec::new();
        // Process in reverse order so that outer Splits (higher indices,
        // created later by the left-fold in hir2postfix) are processed
        // first, maximising the number of alternatives collapsed into a
        // single ByteTable.
        for raw in (0..self.states.len()).rev() {
            let idx = StateIdx(raw as u32);
            if !matches!(self.states.as_slice()[idx], State::Split { .. }) {
                continue;
            }
            leaves.clear();
            if !self.collect_byte_leaves(idx, &mut leaves) {
                continue;
            }
            // Need at least 3 alternatives to justify the 2 KiB table.
            // Two-way Splits (common from `+`, `?`, `*` loops) are not
            // worth optimising — the Split+2×Byte overhead is tiny.
            if leaves.len() < 3 {
                continue;
            }
            // Build the dispatch table.
            let mut table = ByteMap::EMPTY;
            for &(byte, target) in &leaves {
                table.0[byte as usize] = target;
            }
            let table_idx = ByteTableIdx(self.byte_tables.len());
            self.byte_tables.push(table);
            self.states.as_mut_slice()[idx] = State::ByteTable { table: table_idx };
        }
    }

    // -----------------------------------------------------------------------
    // Pre-computed start closure: cache consuming-state leaves reachable
    // from the start state through pure Split chains.
    // -----------------------------------------------------------------------

    /// Walk the epsilon closure of `start` through `Split` states only.
    /// If every leaf is a consuming state (`Byte`, `ByteClass`, `ByteTable`)
    /// or `Match`, return the collected list.  Returns `None` if any
    /// `CounterInstance`, `CounterIncrement`, or `Assert` is encountered,
    /// since those require context manipulation during traversal.
    /// Precompute `can_reach_match` for every NFA state.
    ///
    /// `result[i]` is `true` iff the `Match` state is reachable from state `i`
    /// through epsilon transitions: `Split`, `Assert` (optimistically — any
    /// assertion is assumed passable), `CounterInstance`, and the continue
    /// path of `CounterIncrement`.  Consuming states (`Byte`, `ByteClass`,
    /// `ByteTable`, `ByteCI`) block the walk.
    fn compute_can_reach_match(states: &[State]) -> Box<[bool]> {
        let n = states.len();
        let mut result = vec![false; n];

        // Build reverse-epsilon graph: for each edge u→v in the forward
        // epsilon graph, store v→u in `rev_adj`.
        let mut rev_adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (i, s) in states.iter().enumerate() {
            match *s {
                State::Match => {
                    result[i] = true;
                }
                State::Split { out, out1 } => {
                    rev_adj[out.idx()].push(i as u32);
                    rev_adj[out1.idx()].push(i as u32);
                }
                State::Assert { out, .. } | State::CounterInstance { out, .. } => {
                    rev_adj[out.idx()].push(i as u32);
                }
                State::CounterIncrement { out, .. } => {
                    // Follow continue path only (same as tier 2/3 logic).
                    rev_adj[out.idx()].push(i as u32);
                }
                _ => {}
            }
        }

        // BFS backwards from all Match states.
        let mut queue: Vec<u32> = (0..n as u32).filter(|&i| result[i as usize]).collect();
        while let Some(v) = queue.pop() {
            for &u in &rev_adj[v as usize] {
                if !result[u as usize] {
                    result[u as usize] = true;
                    queue.push(u);
                }
            }
        }

        result.into_boxed_slice()
    }

    /// For each counter index, check whether its `CounterIncrement` break
    /// path (`out1`) can reach the `Match` state.  Returns a `Box<[bool]>`
    /// of length `num_counters`.
    fn compute_counter_break_can_match(
        states: &[State],
        state_can_reach_match: &[bool],
        num_counters: usize,
    ) -> Box<[bool]> {
        let mut result = vec![false; num_counters];
        for s in states {
            if let State::CounterIncrement { counter, out1, .. } = *s {
                let ci = counter.idx();
                if ci < num_counters && state_can_reach_match[out1.idx()] {
                    result[ci] = true;
                }
            }
        }
        result.into_boxed_slice()
    }

    /// Derive a [`Prefilter`] from the precomputed start closure.
    ///
    /// Examines which bytes each consuming state in `start_closure` can accept
    /// and picks the tightest `memchr` variant (1, 2, or 3 bytes).  Falls back
    /// to `Prefilter::None` when the set is too large or `start_closure` is
    /// empty (runtime will use full epsilon closure).
    fn compute_prefilter(
        start_closure: &[StateIdx],
        start_closure_matches: bool,
        states: &[State],
        classes: &[ByteClass],
        byte_tables: &[ByteMap],
    ) -> Prefilter {
        // If the pattern can match empty, every byte is "interesting"
        // (we already matched, so prefilter doesn't help).
        if start_closure_matches {
            return Prefilter::None;
        }
        // Empty closure means runtime falls back to full epsilon-closure;
        // we can't prefilter.
        if start_closure.is_empty() {
            return Prefilter::None;
        }
        // Collect the set of bytes that ANY start-closure state accepts.
        let mut start_bytes = [false; 256];
        for &idx in start_closure {
            match states[idx] {
                State::Byte { byte, .. } => {
                    start_bytes[byte as usize] = true;
                }
                State::ByteCI { byte, .. } => {
                    // byte is lowercase; also mark uppercase
                    start_bytes[byte as usize] = true;
                    start_bytes[(byte ^ 0x20) as usize] = true;
                }
                State::ByteClass { class, .. } => {
                    for b in 0..=255u8 {
                        if classes[class.idx()][b] {
                            start_bytes[b as usize] = true;
                        }
                    }
                }
                State::ByteTable { table, .. } => {
                    for b in 0..=255u8 {
                        if byte_tables[table.idx()].0[b as usize] != StateIdx::NONE {
                            start_bytes[b as usize] = true;
                        }
                    }
                }
                _ => return Prefilter::None, // unexpected state type
            }
        }

        let bytes: Vec<u8> = (0..=255u8).filter(|&b| start_bytes[b as usize]).collect();
        match bytes.len() {
            0 => Prefilter::None, // shouldn't happen, but be safe
            1 => Prefilter::Memchr1(bytes[0]),
            2 => Prefilter::Memchr2(bytes[0], bytes[1]),
            3 => Prefilter::Memchr3(bytes[0], bytes[1], bytes[2]),
            _ => {
                // Check if all start bytes fit in a range of at most 16 values.
                let lo = bytes[0]; // bytes is sorted (from 0..=255 filter)
                let hi = bytes[bytes.len() - 1];
                if hi - lo < 16 {
                    Prefilter::Range(lo, hi)
                } else {
                    Prefilter::None
                }
            }
        }
    }

    fn compute_start_closure(&self, start: StateIdx) -> (Box<[StateIdx]>, bool) {
        let mut leaves = Vec::new();
        let mut has_match = false;
        let mut stack = vec![start];
        let mut visited = vec![false; self.states.len()];
        while let Some(idx) = stack.pop() {
            let i = idx.idx();
            if visited[i] {
                continue;
            }
            visited[i] = true;
            match self.states.as_slice()[idx] {
                State::Split { out, out1 } => {
                    stack.push(out);
                    stack.push(out1);
                }
                State::Byte { .. }
                | State::ByteCI { .. }
                | State::ByteClass { .. }
                | State::ByteTable { .. } => {
                    leaves.push(idx);
                }
                State::Match => {
                    // Match is reachable from start — the pattern can
                    // match without consuming input.  Record the fact
                    // but do NOT include Match in the closure array;
                    // the caller stores it as start_closure_matches.
                    has_match = true;
                }
                // Counter or assertion states require context work;
                // fall back to full epsilon-closure at runtime.
                State::CounterInstance { .. }
                | State::CounterIncrement { .. }
                | State::Assert { .. } => {
                    return (Box::default(), false);
                }
            }
        }
        (leaves.into_boxed_slice(), has_match)
    }

    // -----------------------------------------------------------------------
}

// ---------------------------------------------------------------------------
// Counter context (per-thread counter values)
// ---------------------------------------------------------------------------

/// Index into the counter variable array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CounterIdx(u8);

impl CounterIdx {
    #[inline]
    fn idx(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for CounterIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Maximum number of counters supported (limited by `CounterIdx(u8)`).
const MAX_COUNTERS: usize = 256;

/// Maximum counters for tier 2: the `counting_mask` and `counter_reset`
/// fields in `Transition` are `u64` bitmasks.
const MAX_TIER2_COUNTERS: usize = 64;

/// Default maximum NFA states produced by unrolling a repetition.
/// Used by [`RegexBuilder::default()`].
const DEFAULT_MAX_UNROLL_STATES: usize = 32;

/// Sentinel value: this counter slot is inactive (thread is not inside
/// this counter's repetition body).
const COUNTER_INACTIVE: usize = usize::MAX;

/// Arena allocator for counter context slots.
///
/// All [`CounterCtx`] slot data lives here in a flat `Vec<usize>`.
/// Each context holds a `Range<usize>` into this arena.  "Cloning" a
/// context copies only the lightweight range + active count; the
/// actual slot data is shared (copy-on-write via [`allocate_clone`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct CounterPool {
    arena: Vec<usize>,
    free: Vec<Range<usize>>,
    pub(crate) num_counters: usize,
}

impl CounterPool {
    /// Allocate a fresh slot (all counters inactive).
    pub(crate) fn allocate(&mut self) -> Range<usize> {
        debug_assert!(self.num_counters > 0);
        if let Some(range) = self.free.pop() {
            debug_assert_eq!(range.len(), self.num_counters);
            // Re-initialize to inactive.
            self.arena[range.clone()].fill(COUNTER_INACTIVE);
            return range;
        }
        let start = self.arena.len();
        self.arena
            .resize(start + self.num_counters, COUNTER_INACTIVE);
        start..self.arena.len()
    }

    /// Allocate a new slot that is a copy of `src`.
    pub(crate) fn allocate_clone(&mut self, src: &Range<usize>) -> Range<usize> {
        debug_assert_eq!(src.len(), self.num_counters);
        let dst = self.allocate();
        self.arena.copy_within(src.start..src.end, dst.start);
        dst
    }

    /// Return a slot to the free list for reuse.
    #[inline]
    pub(crate) fn free(&mut self, range: Range<usize>) {
        if !range.is_empty() {
            debug_assert_eq!(range.len(), self.num_counters);
            self.free.push(range);
        }
    }

    /// Read the slot values for a context range.
    #[inline]
    fn slots(&self, range: &Range<usize>) -> &[usize] {
        &self.arena[range.start..range.end]
    }

    /// Read the counter slots for a context.
    #[inline]
    pub(crate) fn slots_of(&self, ctx: &CounterCtx) -> &[usize] {
        self.slots(&ctx.range)
    }

    /// Compare two contexts by value through the pool.
    #[inline]
    pub(crate) fn ctx_eq(&self, a: &CounterCtx, b: &CounterCtx) -> bool {
        a.active == b.active && self.slots(&a.range) == self.slots(&b.range)
    }

    /// Compute a content hash of `(state, active_count, counter_slots)`.
    ///
    /// Used by [`CtxDedupTable`] for O(1) amortized dedup of
    /// `(StateIdx, CounterCtx)` pairs.
    #[inline]
    pub(crate) fn ctx_hash(&self, ctx: &CounterCtx, state: StateIdx) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = ahash::AHasher::default();
        state.0.hash(&mut h);
        ctx.active_count().hash(&mut h);
        self.slots_of(ctx).hash(&mut h);
        h.finish()
    }

    /// Reset the pool for a new match (keeps allocated memory).
    pub(crate) fn clear(&mut self) {
        self.arena.clear();
        self.free.clear();
    }
}

/// Per-thread counter context.
///
/// A lightweight handle into a [`CounterPool`].  The `range` field
/// indexes a contiguous slice of counter values in the pool's arena.
/// An empty range (`0..0`) means the context has never entered a
/// counted repetition (lazy allocation).
///
/// For threads outside all counted repetitions (the common case), the
/// context is empty (`is_empty() == true`) and dedup falls back to
/// the fast `lastlist` path.
#[derive(Debug)]
pub(crate) struct CounterCtx {
    range: Range<usize>,
    /// Number of active (non-`COUNTER_INACTIVE`) slots.  Maintained by
    /// `set` / `remove` so that `is_empty` is O(1).
    active: usize,
}

impl CounterCtx {
    /// Create an empty context (all counters inactive, no allocation).
    pub(crate) fn new() -> Self {
        Self {
            range: 0..0,
            active: 0,
        }
    }

    /// True when no counter is active (all slots are `COUNTER_INACTIVE`).
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.active == 0
    }

    /// Number of active (non-inactive) counter slots.
    #[inline]
    pub(crate) fn active_count(&self) -> usize {
        self.active
    }

    /// Get the value of counter `idx`, or `None` if inactive.
    #[inline]
    pub(crate) fn get(&self, idx: CounterIdx, pool: &CounterPool) -> Option<usize> {
        if idx.idx() >= self.range.len() {
            return None;
        }
        let v = pool.arena[self.range.start + idx.idx()];
        if v == COUNTER_INACTIVE { None } else { Some(v) }
    }

    /// Set the value of counter `idx`.  Lazily allocates a pool slot
    /// on first use.
    #[inline]
    pub(crate) fn set(&mut self, idx: CounterIdx, value: usize, pool: &mut CounterPool) {
        debug_assert!(value != COUNTER_INACTIVE);
        if self.range.is_empty() {
            self.range = pool.allocate();
        }
        let slot = &mut pool.arena[self.range.start + idx.idx()];
        if *slot == COUNTER_INACTIVE {
            self.active += 1;
        }
        *slot = value;
    }

    /// Deactivate counter `idx`.  The counter must currently be active.
    #[inline]
    pub(crate) fn remove(&mut self, idx: CounterIdx, pool: &mut CounterPool) {
        let slot = &mut pool.arena[self.range.start + idx.idx()];
        debug_assert!(*slot != COUNTER_INACTIVE, "remove on inactive counter");
        *slot = COUNTER_INACTIVE;
        self.active -= 1;
    }

    /// Create an independent copy of this context with its own pool
    /// slot, so mutations don't affect the original.
    pub(crate) fn clone(&self, pool: &mut CounterPool) -> Self {
        if self.range.is_empty() {
            return Self::new();
        }
        Self {
            range: pool.allocate_clone(&self.range),
            active: self.active,
        }
    }

    /// Consume this context and return its pool range for freeing.
    pub(crate) fn into_range(self) -> std::ops::Range<usize> {
        self.range
    }
}

// ---------------------------------------------------------------------------
// Context dedup table
// ---------------------------------------------------------------------------

/// Hash table for O(1) amortized dedup of `(StateIdx, CounterCtx)` pairs.
///
/// Wraps a flat [`hashbrown::HashTable`] storing `(hash, index)` pairs
/// where *index* refers to an element in an external backing
/// `Vec<(StateIdx, CounterCtx)>` owned by the caller.  The table is
/// purely an index structure — it does not own the contexts themselves.
///
/// Used by:
/// - **NFA** ([`NfaMatcher`]): indices point into `ctx_visited`.
/// - **Tier 4** ([`Tier4DfaMatcher`](crate::dfa::tier4::Tier4DfaMatcher)):
///   indices point into the `contexts` vec during dedup.
///
/// # Invariants
///
/// 1. **Hash consistency** — each entry's stored hash must have been
///    computed by [`CounterPool::ctx_hash`] over the same
///    `(StateIdx, CounterCtx)` that the index refers to.
/// 2. **Insert-after-find** — [`insert`](Self::insert) must only be
///    called after [`find`](Self::find) returned `false` for the same
///    hash (i.e. the entry is known to be unique).
/// 3. **Lockstep clearing** — the table **must** be
///    [`clear`](Self::clear)ed whenever the backing store is drained or
///    invalidated, so that stale indices are never probed.  In the NFA
///    this is enforced by [`NfaMatcher::free_ctx_visited`]; in Tier 4
///    by calling `clear()` at the top of [`dedup_contexts_hash`].
/// 4. **Index validity** — stored indices must remain valid for the
///    lifetime of the lookup.  In Tier 4's `swap_remove` dedup pattern,
///    this holds because only indices `< i` are stored and `swap_remove`
///    only moves elements from positions `≥ i`.
#[derive(Debug, Default)]
pub(crate) struct CtxDedupTable {
    table: hashbrown::HashTable<(u64, usize)>,
}

impl CtxDedupTable {
    /// Create an empty dedup table.
    pub(crate) fn new() -> Self {
        Self {
            table: hashbrown::HashTable::new(),
        }
    }

    /// Remove all entries, retaining allocated capacity for reuse.
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.table.clear();
    }

    /// Probe the table for an entry matching `hash` whose backing-store
    /// index satisfies `eq`.
    ///
    /// `eq` receives the stored index and should return `true` only when
    /// the indexed element is equal to the query element (typically via
    /// [`CounterPool::ctx_eq`]).
    #[inline]
    pub(crate) fn find(&self, hash: u64, mut eq: impl FnMut(usize) -> bool) -> bool {
        self.table
            .find(hash, |&(h, idx)| h == hash && eq(idx))
            .is_some()
    }

    /// Insert a new unique entry.
    ///
    /// # Safety contract (debug-only)
    ///
    /// The caller must have verified via [`find`](Self::find) that no
    /// matching entry exists.  Violating this adds a duplicate, which
    /// wastes space but does not cause unsoundness.
    #[inline]
    pub(crate) fn insert(&mut self, hash: u64, index: usize) {
        self.table.insert_unique(hash, (hash, index), |&(h, _)| h);
    }
}

// ---------------------------------------------------------------------------
// Matcher (NFA simulation)

/// Reusable memory for [`Matcher`].  Create once, call
/// [`matcher`](Self::matcher) for each regex to match.
#[derive(Debug, Default)]
pub struct MatcherMemory {
    /// Per-state: the `listid` when the state was last added.  Used for
    /// O(1) deduplication of empty-context threads in `addstate`.
    lastlist: Vec<usize>,
    /// Current and next state lists (swapped each step).
    /// Each entry carries the state index and its counter context.
    clist: Vec<(StateIdx, CounterCtx)>,
    nlist: Vec<(StateIdx, CounterCtx)>,
    /// Explicit work stack used by [`Matcher::addstate`] to avoid
    /// recursive epsilon-closure traversal.
    addstack: Vec<AddStateOp>,
    /// Backing store for context-aware dedup of threads with non-empty
    /// counter contexts.  Deep-cloned snapshots of `(state, ctx)` are
    /// appended here during epsilon closure so that later visits to the
    /// same `(state, counter_values)` can be detected and pruned.
    ///
    /// **Must be cleared in lockstep with [`ctx_visited_map`](Self::ctx_visited_map).**
    /// See [`NfaMatcher::free_ctx_visited`].
    ctx_visited: Vec<(StateIdx, CounterCtx)>,
    /// Hash-based index into [`ctx_visited`](Self::ctx_visited) for O(1)
    /// amortized dedup.  Retains capacity across steps.
    ///
    /// **Must be cleared in lockstep with [`ctx_visited`](Self::ctx_visited).**
    /// See [`NfaMatcher::free_ctx_visited`].
    ctx_visited_map: CtxDedupTable,
    counter_pool: CounterPool,
    /// Shared scratch space for epsilon-closure computation (DFA tiers 1–3).
    dfa_memory: DfaMemory,
    /// Lazy DFA cache (allocated on first use with a DFA-eligible regex).
    dfa_cache: Option<Tier1DfaCache>,
    /// Tier 2 DFA cache (non-nested fixed-length-body counters, differential counters).
    tier2_cache: Option<Tier2DfaCache>,
    /// Tier 3 DFA cache (non-nested counters, conditional transitions).
    tier3_cache: Option<Tier3DfaCache>,
    /// Counting DFA cache (Tier 4: patterns with counters).
    tier4_cache: Option<Tier4DfaCache>,
    /// Counter pool for the counting DFA (separate from NFA's pool).
    counting_pool: CounterPool,
}

impl MatcherMemory {
    /// Create a matcher for the given regex.
    ///
    /// Returns an [`AnyMatcher`] that transparently dispatches to the
    /// lazy DFA (for DFA-eligible patterns) or the NFA simulator.
    #[inline]
    pub fn matcher<'a>(&'a mut self, regex: &'a Regex) -> AnyMatcher<'a> {
        if regex.dfa_eligible {
            // Tier 1: pure DFA (no counters, simple assertions).
            let cache = self.dfa_cache.get_or_insert_with(Tier1DfaCache::new);
            cache.prepare(&mut self.dfa_memory, regex);
            let dfa = DfaMatcher::new(cache, &mut self.dfa_memory, regex);
            AnyMatcher::Dfa(dfa)
        } else if regex.tier2_eligible {
            // Tier 2: DFA + differential counters (non-nested, fixed-length bodies).
            let analysis = regex
                .tier2_analysis
                .as_ref()
                .expect("tier2_analysis must be present for tier 2 patterns");
            let cache = self.tier2_cache.get_or_insert_with(Tier2DfaCache::new);
            cache.prepare(&mut self.dfa_memory, regex);
            let dfa = Tier2DfaMatcher::new(cache, &mut self.dfa_memory, regex, analysis);
            AnyMatcher::Tier2Dfa(dfa)
        } else if regex.tier3_eligible {
            // Tier 3: DFA + conditional transitions (non-nested counters).
            let analysis = regex
                .tier3_analysis
                .as_ref()
                .expect("tier3_analysis must be present for tier 3 patterns");
            let cache = self.tier3_cache.get_or_insert_with(Tier3DfaCache::new);
            cache.prepare(&mut self.dfa_memory, regex, analysis);
            let dfa = Tier3DfaMatcher::new(cache, &mut self.dfa_memory, regex, analysis);
            AnyMatcher::Tier3Dfa(dfa)
        } else if regex.tier4_eligible {
            // Tier 4: DFA + explicit counter contexts.
            let cache = self
                .tier4_cache
                .get_or_insert_with(|| Tier4DfaCache::new(regex.states.len()));
            cache.prepare(regex);
            let dfa = Tier4DfaMatcher::new(cache, regex, &mut self.counting_pool);
            AnyMatcher::Tier4Dfa(dfa)
        } else {
            let nfa = self.nfa_matcher(regex);
            AnyMatcher::Nfa(nfa)
        }
    }

    /// Create a matcher forced to a specific execution tier.
    ///
    /// Tier 0 = NFA simulator (always available), tiers 1–4 = DFA tiers.
    /// Returns `Err` with a message if the pattern is not eligible for the
    /// requested tier.
    pub fn matcher_for_tier<'a>(
        &'a mut self,
        regex: &'a Regex,
        tier: u8,
    ) -> Result<AnyMatcher<'a>, String> {
        match tier {
            0 => {
                let nfa = self.nfa_matcher(regex);
                Ok(AnyMatcher::Nfa(nfa))
            }
            1 => {
                if !regex.dfa_eligible {
                    return Err("pattern is not eligible for tier 1 (pure DFA)".into());
                }
                let cache = self.dfa_cache.get_or_insert_with(Tier1DfaCache::new);
                cache.prepare(&mut self.dfa_memory, regex);
                let dfa = DfaMatcher::new(cache, &mut self.dfa_memory, regex);
                Ok(AnyMatcher::Dfa(dfa))
            }
            2 => {
                if !regex.tier2_eligible {
                    return Err(
                        "pattern is not eligible for tier 2 (differential-counter DFA)".into(),
                    );
                }
                let analysis = regex
                    .tier2_analysis
                    .as_ref()
                    .expect("tier2_analysis must be present for tier 2 patterns");
                let cache = self.tier2_cache.get_or_insert_with(Tier2DfaCache::new);
                cache.prepare(&mut self.dfa_memory, regex);
                let dfa = Tier2DfaMatcher::new(cache, &mut self.dfa_memory, regex, analysis);
                Ok(AnyMatcher::Tier2Dfa(dfa))
            }
            3 => {
                if !regex.tier3_eligible {
                    return Err(
                        "pattern is not eligible for tier 3 (conditional-transition DFA)".into(),
                    );
                }
                let analysis = regex
                    .tier3_analysis
                    .as_ref()
                    .expect("tier3_analysis must be present for tier 3 patterns");
                let cache = self.tier3_cache.get_or_insert_with(Tier3DfaCache::new);
                cache.prepare(&mut self.dfa_memory, regex, analysis);
                let dfa = Tier3DfaMatcher::new(cache, &mut self.dfa_memory, regex, analysis);
                Ok(AnyMatcher::Tier3Dfa(dfa))
            }
            4 => {
                if !regex.tier4_eligible {
                    return Err("pattern is not eligible for tier 4 (counter-program DFA)".into());
                }
                let cache = self
                    .tier4_cache
                    .get_or_insert_with(|| Tier4DfaCache::new(regex.states.len()));
                cache.prepare(regex);
                let dfa = Tier4DfaMatcher::new(cache, regex, &mut self.counting_pool);
                Ok(AnyMatcher::Tier4Dfa(dfa))
            }
            _ => Err(format!("invalid tier {tier}: must be 0..4")),
        }
    }

    /// Create an NFA matcher (always available, used as fallback).
    fn nfa_matcher<'a>(&'a mut self, regex: &'a Regex) -> NfaMatcher<'a> {
        self.lastlist.clear();
        self.lastlist.resize(regex.states.len(), usize::MAX);
        self.clist.clear();
        self.nlist.clear();
        self.addstack.clear();
        self.ctx_visited.clear();
        self.ctx_visited_map.clear();
        self.counter_pool.clear();
        self.counter_pool.num_counters = regex.num_counters;

        let mut m = NfaMatcher {
            states: &regex.states,
            classes: &regex.classes,
            byte_tables: &regex.byte_tables,
            lastlist: &mut self.lastlist,
            listid: 0,
            clist: &mut self.clist,
            nlist: &mut self.nlist,
            addstack: &mut self.addstack,
            ctx_visited: &mut self.ctx_visited,
            ctx_visited_map: &mut self.ctx_visited_map,
            start: regex.start,
            start_closure: &regex.start_closure,
            at_start: true,
            at_end: false,
            prev_byte: None,
            ever_matched: regex.start_closure_matches,
            has_assert: false,
            nlist_has_assert: false,
            counter_pool: &mut self.counter_pool,
            prefilter: regex.prefilter,
            at_start_state: false,
        };

        m.startlist(m.start);
        // Initially at start state — clist is exactly the start closure.
        // If the start closure has deferred asserts, disable re-engagement
        // because the prefilter doesn't account for bytes behind asserts.
        m.at_start_state = !m.has_assert;
        m
    }
}

/// A matcher that dispatches to either the lazy DFA or the NFA simulator.
#[allow(clippy::large_enum_variant)]
pub enum AnyMatcher<'a> {
    /// Lazy DFA path (Tier 1: counter-free, simple-assertion patterns).
    Dfa(DfaMatcher<'a>),
    /// Tier 2 DFA path (non-nested fixed-length-body counters, differential counters).
    Tier2Dfa(Tier2DfaMatcher<'a>),
    /// Tier 3 DFA path (non-nested counters, conditional transitions).
    Tier3Dfa(Tier3DfaMatcher<'a>),
    /// Counting DFA path (Tier 4: DFA + explicit counter contexts).
    Tier4Dfa(Tier4DfaMatcher<'a>),
    /// NFA simulator path (general case).
    Nfa(NfaMatcher<'a>),
}

impl<'a> AnyMatcher<'a> {
    /// Feed an entire byte slice through the matcher.
    #[inline]
    pub fn chunk(&mut self, input: &[u8]) {
        match self {
            Self::Dfa(d) => d.chunk(input),
            Self::Tier2Dfa(d) => d.chunk(input),
            Self::Tier3Dfa(d) => d.chunk(input),
            Self::Tier4Dfa(d) => d.chunk(input),
            Self::Nfa(n) => n.chunk(input),
        }
    }

    /// Signal end-of-input and return the final match result.
    #[inline]
    pub fn finish(self) -> bool {
        match self {
            Self::Dfa(d) => d.finish(),
            Self::Tier2Dfa(d) => d.finish(),
            Self::Tier3Dfa(d) => d.finish(),
            Self::Tier4Dfa(d) => d.finish(),
            Self::Nfa(n) => n.finish(),
        }
    }

    /// Check whether the matcher has reached an accepting state so far.
    #[allow(dead_code)]
    pub fn ismatch(&self) -> bool {
        match self {
            Self::Dfa(d) => d.ismatch(),
            Self::Tier2Dfa(d) => d.ismatch(),
            Self::Tier3Dfa(d) => d.ismatch(),
            Self::Tier4Dfa(d) => d.ismatch(),
            Self::Nfa(n) => n.ismatch(),
        }
    }
}

impl<'a> fmt::Debug for AnyMatcher<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dfa(d) => f.debug_tuple("AnyMatcher::Dfa").field(d).finish(),
            Self::Tier2Dfa(d) => f.debug_tuple("AnyMatcher::Tier2Dfa").field(d).finish(),
            Self::Tier3Dfa(d) => f.debug_tuple("AnyMatcher::Tier3Dfa").field(d).finish(),
            Self::Tier4Dfa(d) => f.debug_tuple("AnyMatcher::Tier4Dfa").field(d).finish(),
            Self::Nfa(n) => f.debug_tuple("AnyMatcher::Nfa").field(n).finish(),
        }
    }
}

impl<'a> fmt::Display for AnyMatcher<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dfa(d) => fmt::Display::fmt(d, f),
            Self::Tier2Dfa(d) => fmt::Display::fmt(d, f),
            Self::Tier3Dfa(d) => fmt::Display::fmt(d, f),
            Self::Tier4Dfa(d) => fmt::Display::fmt(d, f),
            Self::Nfa(n) => fmt::Display::fmt(n, f),
        }
    }
}

/// Runs a Thompson NFA simulation with per-thread counter contexts.
#[derive(Debug)]
pub struct NfaMatcher<'a> {
    states: &'a [State],
    /// Byte-class lookup tables referenced by [`State::ByteClass::class`].
    classes: &'a [ByteClass],
    /// Byte dispatch tables referenced by [`State::ByteTable::table`].
    byte_tables: &'a [ByteMap],
    /// Per-state deduplication stamp (compared against `listid`).
    /// Used for empty-context threads only.
    lastlist: &'a mut [usize],
    /// Monotonically increasing step ID.
    listid: usize,
    /// Current active state list.
    clist: &'a mut Vec<(StateIdx, CounterCtx)>,
    /// Next active state list (built during a step).
    nlist: &'a mut Vec<(StateIdx, CounterCtx)>,
    /// Explicit work stack for iterative epsilon-closure traversal.
    addstack: &'a mut Vec<AddStateOp>,
    /// Backing store for context-aware dedup (see [`MatcherMemory::ctx_visited`]).
    /// **Cleared in lockstep with `ctx_visited_map`** via [`free_ctx_visited`](Self::free_ctx_visited).
    ctx_visited: &'a mut Vec<(StateIdx, CounterCtx)>,
    /// Hash-based index into `ctx_visited` for O(1) amortized dedup.
    /// **Cleared in lockstep with `ctx_visited`** via [`free_ctx_visited`](Self::free_ctx_visited).
    ctx_visited_map: &'a mut CtxDedupTable,

    /// The NFA start state index.
    start: StateIdx,
    /// Precomputed start closure (see [`Regex::start_closure`]).
    start_closure: &'a [StateIdx],
    /// `true` until the first [`step`](Self::step) call.
    at_start: bool,
    /// Set to `true` by [`finish`](Self::finish).
    at_end: bool,
    /// The last byte consumed by [`step`](Self::step).
    prev_byte: Option<u8>,
    /// Tracks whether a `Match` state was ever reached.
    ever_matched: bool,
    /// Whether `clist` contains any `Assert` states that need deferred
    /// resolution in the pre-consumption phase of [`step`](Self::step).
    /// Maintained by `drain_addstack` and swapped with `nlist_has_assert`
    /// at the end of each step.
    has_assert: bool,
    /// Whether `nlist` (being built) contains any `Assert` states.
    nlist_has_assert: bool,
    counter_pool: &'a mut CounterPool,
    prefilter: Prefilter,
    /// True when `clist` is in its initial configuration (only start closure
    /// entries, no in-progress matches, no live deferred assertions).
    /// Safe to re-engage memchr prefilter when true.
    at_start_state: bool,
}

/// Internal operations for iterative [`Matcher::addstate`] traversal.
#[derive(Debug)]
enum AddStateOp {
    /// Visit (and epsilon-expand) a state with the given context.
    Visit(StateIdx, CounterCtx),
    /// Push this state + context to `nlist` after its epsilon successors
    /// are handled.  Only used for consuming and Assert states.
    PostPush(StateIdx, CounterCtx),
}

impl fmt::Display for NfaMatcher<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NFA threads={} nfa={{{}}} matched={}",
            self.clist.len(),
            self.clist
                .iter()
                .map(|(s, _)| s.to_string())
                .collect::<Vec<_>>()
                .join(","),
            self.ever_matched,
        )?;
        if self.at_start {
            write!(f, " at_start")?;
        }
        if self.has_assert {
            write!(f, " has_assert")?;
        }
        // Show non-empty counter contexts.
        let with_ctx = self.clist.iter().filter(|(_, c)| !c.is_empty()).count();
        if with_ctx > 0 {
            write!(f, " ({with_ctx} with counters)")?;
        }
        Ok(())
    }
}

impl<'a> NfaMatcher<'a> {
    /// Drain `ctx_visited` and return all arena slots to the pool.
    ///
    /// Also clears `ctx_visited_map` to maintain the lockstep invariant
    /// (see [`CtxDedupTable`]).
    #[inline]
    fn free_ctx_visited(&mut self) {
        for (_, ctx) in self.ctx_visited.drain(..) {
            self.counter_pool.free(ctx.range);
        }
        self.ctx_visited_map.clear();
    }

    /// Compute the initial state list by following all epsilon transitions
    /// from `start`.
    #[inline]
    fn startlist(&mut self, start: StateIdx) {
        if !self.start_closure.is_empty() {
            // Fast path: directly insert precomputed consuming leaves.
            // Match states are excluded from the closure array;
            // ever_matched was initialised from start_closure_matches.
            for &idx in self.start_closure {
                self.nlist.push((idx, CounterCtx::new()));
                self.lastlist[idx.idx()] = self.listid;
            }
        } else {
            self.addstate(start, CounterCtx::new(), None);
        }
        std::mem::swap(self.clist, self.nlist);
        self.has_assert = self.nlist_has_assert;
        self.nlist_has_assert = false;
        self.listid += 1;
    }

    /// Follow epsilon transitions from state `idx` with counter context
    /// `ctx`, adding all reachable states to `nlist`.
    ///
    /// `next_byte`: when known (e.g. during pre-consumption assertion
    /// resolution), passing `Some(b)` lets chained deferred assertions
    /// (like `\b` → `EndLF`) resolve immediately instead of being
    /// deferred again and lost.
    ///
    /// Dedup strategy:
    /// - Empty context: fast path via `lastlist`/`listid` (O(1) per state).
    /// - Non-empty context: hash-based lookup via `ctx_visited_map` with
    ///   full value comparison on collision through the counter pool.
    #[inline]
    fn addstate(&mut self, idx: StateIdx, ctx: CounterCtx, next_byte: Option<u8>) {
        self.addstack.clear();
        self.addstack.push(AddStateOp::Visit(idx, ctx));
        self.drain_addstack(next_byte);
    }

    /// Process all operations on the work stack until empty.
    fn drain_addstack(&mut self, next_byte: Option<u8>) {
        while let Some(op) = self.addstack.pop() {
            match op {
                AddStateOp::Visit(idx, ctx) => {
                    // --- Dedup ---
                    let ctx = if ctx.is_empty() {
                        let i = idx.idx();
                        if self.lastlist[i] == self.listid {
                            self.counter_pool.free(ctx.range);
                            continue;
                        }
                        self.lastlist[i] = self.listid;
                        ctx
                    } else {
                        // Hash-based lookup with full value comparison on
                        // collision.  O(1) amortized instead of O(K²).
                        let hash = self.counter_pool.ctx_hash(&ctx, idx);
                        let already_seen = self.ctx_visited_map.find(hash, |j| {
                            let (s, c) = &self.ctx_visited[j];
                            *s == idx && self.counter_pool.ctx_eq(c, &ctx)
                        });
                        if already_seen {
                            self.counter_pool.free(ctx.range);
                            continue;
                        }
                        // Record for future dedup.  Deep-clone so the dedup
                        // entry is independent of later mutations to ctx.
                        let record_idx = self.ctx_visited.len();
                        self.ctx_visited.push((idx, ctx.clone(self.counter_pool)));
                        self.ctx_visited_map.insert(hash, record_idx);
                        ctx
                    };

                    match self.states[idx] {
                        State::Split { out, out1 } => {
                            // No PostPush for epsilon states.
                            // Deep-clone for out1 so each branch gets its own
                            // pool slot and mutations are independent.
                            let ctx1 = ctx.clone(self.counter_pool);
                            self.addstack.push(AddStateOp::Visit(out1, ctx1));
                            self.addstack.push(AddStateOp::Visit(out, ctx));
                        }

                        State::Assert { kind, out } => {
                            // Always PostPush — the assertion must be in
                            // nlist for deferred resolution in step()/finish().
                            self.nlist_has_assert = true;
                            let post_ctx = ctx.clone(self.counter_pool);
                            self.addstack.push(AddStateOp::PostPush(idx, post_ctx));
                            // Evaluate: use next_byte when available (during
                            // pre-consumption resolution) so chained deferred
                            // assertions like \b → EndLF can resolve.
                            if kind.eval(self.at_start, self.at_end, self.prev_byte, next_byte)
                                == AssertEval::Pass
                            {
                                self.addstack.push(AddStateOp::Visit(out, ctx));
                            } else {
                                self.counter_pool.free(ctx.range);
                            }
                        }

                        State::CounterInstance { counter, out } => {
                            // Enter the counted repetition: set counter = 0.
                            // Deep-clone so we get our own mutable pool slot.
                            let mut new_ctx = ctx.clone(self.counter_pool);
                            self.counter_pool.free(ctx.range);
                            new_ctx.set(counter, 0, self.counter_pool);
                            self.addstack.push(AddStateOp::Visit(out, new_ctx));
                        }

                        State::CounterIncrement {
                            counter,
                            out,
                            out1,
                            min,
                            max,
                        } => {
                            let value = ctx
                                .get(counter, self.counter_pool)
                                .expect("counter must be active at CInc");
                            let new_value = value + 1;
                            let take_continue = new_value < max;
                            let take_break = new_value >= min;

                            match (take_continue, take_break) {
                                (true, true) => {
                                    // Both paths need independent slots.
                                    let mut break_ctx = ctx.clone(self.counter_pool);
                                    break_ctx.remove(counter, self.counter_pool);
                                    self.addstack.push(AddStateOp::Visit(out1, break_ctx));
                                    // Mutate the original for the continue path.
                                    let mut ctx = ctx;
                                    ctx.set(counter, new_value, self.counter_pool);
                                    self.addstack.push(AddStateOp::Visit(out, ctx));
                                }
                                (true, false) => {
                                    // Continue only: mutate in place.
                                    let mut ctx = ctx;
                                    ctx.set(counter, new_value, self.counter_pool);
                                    self.addstack.push(AddStateOp::Visit(out, ctx));
                                }
                                (false, true) => {
                                    // Break only: mutate in place.
                                    let mut ctx = ctx;
                                    ctx.remove(counter, self.counter_pool);
                                    self.addstack.push(AddStateOp::Visit(out1, ctx));
                                }
                                (false, false) => {
                                    self.counter_pool.free(ctx.range);
                                }
                            }
                        }

                        State::Match => {
                            self.counter_pool.free(ctx.range);
                            self.ever_matched = true;
                        }

                        // Consuming states: record in nlist for step().
                        State::Byte { .. }
                        | State::ByteCI { .. }
                        | State::ByteClass { .. }
                        | State::ByteTable { .. } => {
                            self.addstack.push(AddStateOp::PostPush(idx, ctx));
                        }
                    }
                }

                AddStateOp::PostPush(idx, ctx) => {
                    self.nlist.push((idx, ctx));
                }
            }
        }
    }

    /// Advance the simulation by one input byte.
    #[inline(always)]
    fn step(&mut self, b: u8) {
        self.at_start_state = false;

        // --- Pre-consumption: resolve deferred assertions ---
        let mut had_deferred_resolution = false;
        if self.has_assert {
            let mut any_expanded = false;
            let clist_len = self.clist.len();
            for i in 0..clist_len {
                let (idx, ref ctx) = self.clist[i];
                if let State::Assert { kind, out } = self.states[idx]
                    && kind.eval(self.at_start, self.at_end, self.prev_byte, None)
                        == AssertEval::Defer
                    && kind.eval(self.at_start, self.at_end, self.prev_byte, Some(b))
                        == AssertEval::Pass
                {
                    if !any_expanded {
                        self.listid += 1;
                        // Inline free_ctx_visited (can't call &mut self
                        // method while self.clist[i] is borrowed).
                        // Must clear both ctx_visited and ctx_visited_map
                        // in lockstep — see CtxDedupTable invariant #3.
                        for (_, c) in self.ctx_visited.drain(..) {
                            self.counter_pool.free(c.range);
                        }
                        self.ctx_visited_map.clear();
                        debug_assert!(
                            self.nlist.is_empty(),
                            "nlist must be empty before assert expansion"
                        );
                        any_expanded = true;
                    }
                    // Deep-clone so the context owns its own arena slot.
                    // A shallow clone() would alias the clist entry's range,
                    // causing use-after-free when addstate frees dead contexts.
                    let ctx_clone = ctx.clone(self.counter_pool);
                    // Pass Some(b) so chained deferred assertions (e.g.
                    // \b → EndLF) can resolve immediately.
                    self.addstate(out, ctx_clone, Some(b));
                }
            }
            had_deferred_resolution = any_expanded;
            if any_expanded {
                self.clist.append(self.nlist);
                self.listid += 1;
                self.free_ctx_visited();
            }
        }

        self.at_start = false;
        self.prev_byte = Some(b);

        debug_assert!(
            self.nlist.is_empty(),
            "nlist must be empty before consumption"
        );
        self.free_ctx_visited();
        let mut clist = std::mem::take(self.clist);
        let mut any_consumed = false;

        // Fused pass: push Visit ops for matching consuming states
        // (back-to-front for LIFO ordering) + re-seed.
        // pop() yields owned values back-to-front — no clones needed.
        self.addstack.clear();

        if self.start_closure.is_empty() {
            // Slow path: push re-seed as lowest-priority item (processed
            // last by the LIFO addstack, so it appears after all
            // continuing threads in nlist).
            self.addstack
                .push(AddStateOp::Visit(self.start, CounterCtx::new()));
        }

        while let Some((idx, ctx)) = clist.pop() {
            let target = match self.states[idx] {
                State::Byte { byte: b2, out } if b == b2 => out,
                State::ByteCI { byte: b2, out } if byte_match_ci(b, b2) => out,
                State::ByteClass { class, out } if self.classes[class][b] => out,
                State::ByteTable { table } => {
                    let t = self.byte_tables[table][b];
                    if t == StateIdx::NONE {
                        self.counter_pool.free(ctx.range);
                        continue;
                    }
                    t
                }
                _ => {
                    self.counter_pool.free(ctx.range);
                    continue;
                }
            };
            self.addstack.push(AddStateOp::Visit(target, ctx));
            any_consumed = true;
        }

        self.drain_addstack(None);

        // Fast-path re-seed: directly insert precomputed start closure
        // leaves into nlist.  Done after drain_addstack so continuing
        // threads have higher priority.  lastlist dedup ensures no
        // duplicates if a continuing thread already reached one of
        // these states.
        if !self.start_closure.is_empty() {
            // Match states are excluded from the closure array;
            // ever_matched was initialised from start_closure_matches.
            for &idx in self.start_closure {
                let i = idx.idx();
                if self.lastlist[i] != self.listid {
                    self.lastlist[i] = self.listid;
                    self.nlist.push((idx, CounterCtx::new()));
                }
            }
        }

        // clist is empty after drain but retains its capacity for reuse.
        *self.clist = std::mem::replace(self.nlist, clist);
        self.has_assert = self.nlist_has_assert;
        self.nlist_has_assert = false;
        self.listid += 1;
        self.free_ctx_visited();

        // Re-engagement: if no consuming state matched and no deferred
        // assertion resolved, clist is just the re-seeded start closure.
        // The prefilter can safely skip to the next candidate byte.
        self.at_start_state = !any_consumed && !had_deferred_resolution && !self.has_assert;
    }

    /// Feed an entire byte slice through the matcher, one byte at a time.
    ///
    /// Stops early if a match has already been found, since further input
    /// cannot change the outcome (our API only reports existence, not
    /// positions or counts).
    pub fn chunk(&mut self, input: &[u8]) {
        if self.ever_matched {
            return;
        }
        match self.prefilter {
            Prefilter::None => self.chunk_no_prefilter(input),
            Prefilter::Memchr1(needle) => {
                self.chunk_prefilter(input, |hay| memchr::memchr(needle, hay));
            }
            Prefilter::Memchr2(b1, b2) => {
                self.chunk_prefilter(input, |hay| memchr::memchr2(b1, b2, hay));
            }
            Prefilter::Memchr3(b1, b2, b3) => {
                self.chunk_prefilter(input, |hay| memchr::memchr3(b1, b2, b3, hay));
            }
            Prefilter::Range(lo, hi) => {
                self.chunk_prefilter(input, |hay| crate::memrange::memrange(lo, hi, hay));
            }
        }
    }

    /// Fast path: no prefilter, process every byte.
    fn chunk_no_prefilter(&mut self, input: &[u8]) {
        for &b in input {
            if self.ever_matched {
                return;
            }
            self.step(b);
        }
    }

    /// Prefilter path: when `at_start_state` is true, use memchr to skip
    /// to the next candidate byte.  Falls back to byte-at-a-time when a
    /// partial match is in progress.
    fn chunk_prefilter(&mut self, input: &[u8], finder: impl Fn(&[u8]) -> Option<usize>) {
        let mut i = 0;
        while i < input.len() {
            if self.ever_matched {
                return;
            }
            // When the NFA is in its start configuration (only re-seeded
            // start closure threads, no live deferred assertions), use
            // memchr to skip non-candidate bytes.
            if self.at_start_state {
                if let Some(offset) = finder(&input[i..]) {
                    i += offset;
                } else {
                    return;
                }
            }
            self.step(input[i]);
            i += 1;
        }
    }

    /// Check whether the matcher has reached an accepting state so far.
    pub fn ismatch(&self) -> bool {
        self.ever_matched
    }

    /// Signal end-of-input and return the final match result.
    ///
    /// Allows `$` / `(?m:$)` assertions to fire.
    ///
    /// Consumes the matcher, since no further input can be fed after
    /// end-of-input has been signalled.
    pub fn finish(mut self) -> bool {
        if self.ever_matched {
            // Free all remaining contexts before returning.
            for (_, ctx) in self.clist.drain(..) {
                self.counter_pool.free(ctx.range);
            }
            self.free_ctx_visited();
            return true;
        }

        self.at_end = true;

        debug_assert!(
            self.nlist.is_empty(),
            "nlist must be empty before finish expansion"
        );

        if !self.clist.is_empty() {
            self.listid += 1;
            self.free_ctx_visited();

            // Take clist out so addstate can borrow &mut self.
            let clist = std::mem::take(self.clist);
            for (idx, ctx) in clist.into_iter() {
                if let State::Assert { kind, out } = self.states[idx]
                    && kind.eval(self.at_start, true, self.prev_byte, None) == AssertEval::Pass
                {
                    self.addstate(out, ctx, None);
                } else {
                    self.counter_pool.free(ctx.range);
                }
            }
        }

        self.free_ctx_visited();

        self.ever_matched
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    // -----------------------------------------------------------------------
    // CounterCtx + CounterPool unit tests
    // -----------------------------------------------------------------------

    fn test_pool(num_counters: usize) -> CounterPool {
        CounterPool {
            arena: Vec::new(),
            free: Vec::new(),
            num_counters,
        }
    }

    #[test]
    fn test_counter_ctx_empty() {
        let ctx = CounterCtx::new();
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_counter_ctx_all_inactive() {
        let pool = test_pool(3);
        let ctx = CounterCtx::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.get(CounterIdx(0), &pool), None);
        assert_eq!(ctx.get(CounterIdx(1), &pool), None);
        assert_eq!(ctx.get(CounterIdx(2), &pool), None);
    }

    #[test]
    fn test_counter_ctx_set_get() {
        let mut pool = test_pool(2);
        let mut ctx = CounterCtx::new();
        ctx.set(CounterIdx(0), 5, &mut pool);
        assert!(!ctx.is_empty());
        assert_eq!(ctx.get(CounterIdx(0), &pool), Some(5));
        assert_eq!(ctx.get(CounterIdx(1), &pool), None);
    }

    #[test]
    fn test_counter_ctx_remove() {
        let mut pool = test_pool(2);
        let mut ctx = CounterCtx::new();
        ctx.set(CounterIdx(0), 5, &mut pool);
        ctx.set(CounterIdx(1), 3, &mut pool);
        ctx.remove(CounterIdx(0), &mut pool);
        assert_eq!(ctx.get(CounterIdx(0), &pool), None);
        assert_eq!(ctx.get(CounterIdx(1), &pool), Some(3));
        assert!(!ctx.is_empty());
        ctx.remove(CounterIdx(1), &mut pool);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_counter_ctx_set_mutates() {
        let mut pool = test_pool(2);
        let mut ctx = CounterCtx::new();
        assert!(ctx.is_empty());
        ctx.set(CounterIdx(1), 7, &mut pool);
        assert!(!ctx.is_empty());
        assert_eq!(ctx.get(CounterIdx(1), &pool), Some(7));
    }

    #[test]
    fn test_counter_ctx_remove_mutates() {
        let mut pool = test_pool(2);
        let mut ctx = CounterCtx::new();
        ctx.set(CounterIdx(0), 10, &mut pool);
        ctx.set(CounterIdx(1), 20, &mut pool);
        ctx.remove(CounterIdx(0), &mut pool);
        assert_eq!(ctx.get(CounterIdx(0), &pool), None);
        assert_eq!(ctx.get(CounterIdx(1), &pool), Some(20));
    }

    #[test]
    fn test_counter_ctx_pool_eq() {
        let mut pool = test_pool(2);
        let mut a = CounterCtx::new();
        a.set(CounterIdx(0), 3, &mut pool);
        let mut b = CounterCtx::new();
        b.set(CounterIdx(0), 3, &mut pool);
        let mut c = CounterCtx::new();
        c.set(CounterIdx(0), 4, &mut pool);
        assert!(pool.ctx_eq(&a, &b));
        assert!(!pool.ctx_eq(&a, &c));
    }

    #[test]
    fn test_counter_ctx_clone() {
        let mut pool = test_pool(2);
        let mut ctx = CounterCtx::new();
        ctx.set(CounterIdx(0), 5, &mut pool);
        ctx.set(CounterIdx(1), 10, &mut pool);
        let mut cloned = ctx.clone(&mut pool);
        // Values are identical.
        assert!(pool.ctx_eq(&ctx, &cloned));
        // Mutating the clone doesn't affect the original.
        cloned.set(CounterIdx(0), 99, &mut pool);
        assert_eq!(ctx.get(CounterIdx(0), &pool), Some(5));
        assert_eq!(cloned.get(CounterIdx(0), &pool), Some(99));
    }

    #[test]
    fn test_counter_pool_free_reuse() {
        let mut pool = test_pool(2);
        let mut ctx = CounterCtx::new();
        ctx.set(CounterIdx(0), 42, &mut pool);
        let range = ctx.range.clone();
        // Free the slot.
        pool.free(ctx.range);
        // Next allocation should reuse it, with values re-initialized.
        let mut ctx2 = CounterCtx::new();
        ctx2.set(CounterIdx(0), 7, &mut pool);
        assert_eq!(ctx2.range, range);
        assert_eq!(ctx2.get(CounterIdx(0), &pool), Some(7));
        assert_eq!(ctx2.get(CounterIdx(1), &pool), None);
    }

    #[test]
    fn test_counter_ctx_eq_both_empty() {
        let pool = test_pool(2);
        let a = CounterCtx::new();
        let b = CounterCtx::new();
        assert!(pool.ctx_eq(&a, &b));
    }

    #[test]
    fn test_counter_ctx_eq_active_fast_reject() {
        let mut pool = test_pool(2);
        let mut a = CounterCtx::new();
        a.set(CounterIdx(0), 3, &mut pool);
        // b has one counter active, different from a's two-slot layout
        let mut b = CounterCtx::new();
        b.set(CounterIdx(0), 3, &mut pool);
        b.set(CounterIdx(1), 1, &mut pool);
        // Same slot 0 value but different active counts → not equal.
        assert!(!pool.ctx_eq(&a, &b));
    }

    #[test]
    fn test_counter_ctx_clone_empty() {
        let mut pool = test_pool(2);
        let ctx = CounterCtx::new();
        let cloned = ctx.clone(&mut pool);
        assert!(cloned.is_empty());
        assert!(cloned.range.is_empty());
        // Pool arena should be untouched — no allocation for empty ctx.
        assert!(pool.arena.is_empty());
    }

    #[test]
    fn test_counter_ctx_clone_independence() {
        let mut pool = test_pool(2);
        let mut src = CounterCtx::new();
        src.set(CounterIdx(0), 10, &mut pool);
        src.set(CounterIdx(1), 20, &mut pool);
        let mut c1 = src.clone(&mut pool);
        let mut c2 = src.clone(&mut pool);
        // All three start equal.
        assert!(pool.ctx_eq(&src, &c1));
        assert!(pool.ctx_eq(&src, &c2));
        // Mutate each independently.
        c1.set(CounterIdx(0), 99, &mut pool);
        c2.set(CounterIdx(1), 77, &mut pool);
        // Original unchanged.
        assert_eq!(src.get(CounterIdx(0), &pool), Some(10));
        assert_eq!(src.get(CounterIdx(1), &pool), Some(20));
        // Clones are independent.
        assert_eq!(c1.get(CounterIdx(0), &pool), Some(99));
        assert_eq!(c1.get(CounterIdx(1), &pool), Some(20));
        assert_eq!(c2.get(CounterIdx(0), &pool), Some(10));
        assert_eq!(c2.get(CounterIdx(1), &pool), Some(77));
    }

    #[test]
    fn test_counter_ctx_set_overwrites_active() {
        let mut pool = test_pool(2);
        let mut ctx = CounterCtx::new();
        ctx.set(CounterIdx(0), 5, &mut pool);
        assert_eq!(ctx.active, 1);
        // Overwrite with a new value — active count should stay 1.
        ctx.set(CounterIdx(0), 10, &mut pool);
        assert_eq!(ctx.active, 1);
        assert_eq!(ctx.get(CounterIdx(0), &pool), Some(10));
    }

    // -----------------------------------------------------------------------
    // ByteTable optimisation tests
    // -----------------------------------------------------------------------

    /// Helper: count how many states in the NFA are `ByteTable`.
    fn count_byte_tables(regex: &Regex) -> usize {
        regex
            .states
            .iter()
            .filter(|s| matches!(s, State::ByteTable { .. }))
            .count()
    }

    #[test]
    fn test_byte_table_not_created_for_two_way_split() {
        // a+b+ has a 2-way Split (loop vs exit) — below threshold.
        let re = build_regex_unchecked("^a+b+$");
        assert_eq!(count_byte_tables(&re), 0);
        // a?b similarly.
        let re = build_regex_unchecked("^a?b$");
        assert_eq!(count_byte_tables(&re), 0);
        // 2-way alternation of multi-byte literals.
        let re = build_regex_unchecked("^(ab|cd)$");
        assert_eq!(count_byte_tables(&re), 0);
    }

    #[test]
    fn test_byte_table_created_for_three_way_alternation() {
        // (ab|cd|ef) — 3 branches with distinct first bytes.
        let re = build_regex_unchecked("^(ab|cd|ef)$");
        assert!(count_byte_tables(&re) > 0);
        assert_eq!(re.byte_tables.len(), count_byte_tables(&re));
    }

    #[test]
    fn test_byte_table_three_way_matching() {
        let re = build_regex_unchecked("^(ab|cd|ef)$");
        let mut mem = MatcherMemory::default();
        // Positive cases.
        for input in &[b"ab" as &[u8], b"cd", b"ef"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
        // Negative cases.
        for input in &[b"ac" as &[u8], b"cb", b"a", b"abc", b""] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(!m.finish(), "expected no match for {:?}", input);
        }
    }

    #[test]
    fn test_byte_table_four_way_alternation() {
        let re = build_regex_unchecked("^(foo|bar|baz|qux)$");
        // 'f' and 'q' are unique, 'b' appears in both bar and baz so
        // the Split(bar_branch, baz_branch) can't be collapsed (duplicate
        // byte 'b'→'a'→...).  However the outer 3-way Split tree with
        // first bytes {f, b, q} has 'b' pointing to two branches — that
        // should fail collect_byte_leaves (duplicate 'b').  Let's verify.
        //
        // Actually: foo|bar|baz|qux has first bytes [f, b, b, q] — 'b'
        // duplicated, so the outermost Split chain can't be fully collapsed.
        // But inner sub-chains might be.  Let's just verify matching works.
        let mut mem = MatcherMemory::default();
        for input in &[b"foo" as &[u8], b"bar", b"baz", b"qux"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
        for input in &[b"fox" as &[u8], b"bat", b"qu", b""] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(!m.finish(), "expected no match for {:?}", input);
        }
    }

    #[test]
    fn test_byte_table_five_way_distinct_first_bytes() {
        // 5 branches all with distinct first bytes — should create a
        // ByteTable.
        let re = build_regex_unchecked("^(ab|cd|ef|gh|ij)$");
        assert!(count_byte_tables(&re) > 0);
        let mut mem = MatcherMemory::default();
        for input in &[b"ab" as &[u8], b"cd", b"ef", b"gh", b"ij"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
        for input in &[b"ac" as &[u8], b"eg", b""] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(!m.finish(), "expected no match for {:?}", input);
        }
    }

    #[test]
    fn test_byte_table_unanchored() {
        // Unanchored 3-way alternation.
        let re = build_regex_unchecked("(ab|cd|ef)");
        let mut mem = MatcherMemory::default();
        for input in &[b"xxab" as &[u8], b"cdyy", b"xxefyy", b"ab"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
        for input in &[b"ac" as &[u8], b"xyz"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(!m.finish(), "expected no match for {:?}", input);
        }
    }

    #[test]
    fn test_byte_table_inside_repetition() {
        // 3-way alternation inside a counted repetition.
        let re = build_regex_unchecked("^(ab|cd|ef){2,3}$");
        let mut mem = MatcherMemory::default();
        for input in &[b"abcd" as &[u8], b"efab", b"ababab", b"cdefab", b"abcdef"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
        for input in &[b"ab" as &[u8], b"abcdefgh", b"abcde"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(!m.finish(), "expected no match for {:?}", input);
        }
    }

    #[test]
    fn test_byte_table_inside_one_plus() {
        // 3-way alternation inside +.
        let re = build_regex_unchecked("^(ab|cd|ef)+$");
        let mut mem = MatcherMemory::default();
        for input in &[b"ab" as &[u8], b"abcd", b"ababababef", b"efcdab"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
        for input in &[b"" as &[u8], b"a", b"abc"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(!m.finish(), "expected no match for {:?}", input);
        }
    }

    #[test]
    fn test_byte_table_no_optimization_on_duplicate_first_bytes() {
        // (ab|ac|ad) — all start with 'a', can't collapse.
        let re = build_regex_unchecked("^(ab|ac|ad)$");
        assert_eq!(count_byte_tables(&re), 0);
        // Still matches correctly.
        let mut mem = MatcherMemory::default();
        for input in &[b"ab" as &[u8], b"ac", b"ad"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
    }

    #[test]
    fn test_byte_table_no_optimization_on_byte_class_leaf() {
        // ([a-c]x|dy|ez) — leaves are ByteClass, not Byte, for [a-c].
        let re = build_regex_unchecked("^([a-c]x|dy|ez)$");
        // The Split should not be optimised since [a-c] is a ByteClass leaf.
        assert_eq!(count_byte_tables(&re), 0);
        let mut mem = MatcherMemory::default();
        for input in &[b"ax" as &[u8], b"bx", b"cx", b"dy", b"ez"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
    }

    // -- ByteTable: cross-validation with regex crate ------------------------

    /// Cross-validate ByteTable patterns against the `regex` crate using
    /// both `chunk()` and byte-at-a-time `step()`.  This catches any
    /// semantic divergence the optimization might introduce.
    #[test]
    fn test_byte_table_cross_validate_three_way() {
        let p = "^(ab|cd|ef)$";
        let re = build_regex_unchecked(p);
        assert!(count_byte_tables(&re) > 0);
        for input in &[
            "ab", "cd", "ef", // positives
            "ac", "cb", "a", "abc", "", "af", "eb", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

    #[test]
    fn test_byte_table_cross_validate_five_way() {
        let p = "^(ab|cd|ef|gh|ij)$";
        let re = build_regex_unchecked(p);
        assert!(count_byte_tables(&re) > 0);
        for input in &[
            "ab", "cd", "ef", "gh", "ij", // positives
            "ac", "eg", "ih", "gj", "", "a", "abcd", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }
    // -- ByteTable: streaming / multi-chunk ----------------------------------

    /// Feed a ByteTable pattern across multiple chunks.  The ByteTable
    /// dispatch happens in `step()` which is driven per-byte, so this
    /// verifies correctness when a match straddles chunk boundaries.
    #[test]
    fn test_byte_table_multi_chunk() {
        let re = build_regex_unchecked("^(ab|cd|ef)$");
        assert!(count_byte_tables(&re) > 0);
        let mut mem = MatcherMemory::default();

        // Split "cd" across two chunks: "c" then "d".
        let mut m = mem.matcher(&re);
        m.chunk(b"c");
        m.chunk(b"d");
        assert!(m.finish(), "expected match for 'cd' split across chunks");

        // Split "ef" as "e" + "f".
        let mut m = mem.matcher(&re);
        m.chunk(b"e");
        m.chunk(b"f");
        assert!(m.finish(), "expected match for 'ef' split across chunks");

        // Non-match across chunks: "a" + "c" → "ac".
        let mut m = mem.matcher(&re);
        m.chunk(b"a");
        m.chunk(b"c");
        assert!(
            !m.finish(),
            "expected no match for 'ac' split across chunks"
        );
    }

    /// Multi-chunk with a longer pattern that goes through ByteTable
    /// multiple times (one-plus loop).
    #[test]
    fn test_byte_table_multi_chunk_looping() {
        let re = build_regex_unchecked("^(ab|cd|ef)+$");
        let mut mem = MatcherMemory::default();

        // "abcdef" fed one byte at a time.
        let mut m = mem.matcher(&re);
        for &b in b"abcdef" {
            m.chunk(&[b]);
        }
        assert!(m.finish(), "expected match for 'abcdef' byte-at-a-time");

        // "abcdef" fed in odd-sized chunks: "abc", "de", "f".
        let mut m = mem.matcher(&re);
        m.chunk(b"abc");
        m.chunk(b"de");
        m.chunk(b"f");
        assert!(
            m.finish(),
            "expected match for 'abcdef' in chunks [abc,de,f]"
        );

        // Non-match split across chunks.
        let mut m = mem.matcher(&re);
        m.chunk(b"ab");
        m.chunk(b"c"); // 'c' starts but 'd' never comes
        assert!(!m.finish(), "expected no match for 'abc'");
    }

    // -- ByteTable + counters ------------------------------------------------
    // -- ByteTable: mixed-length branches ------------------------------------

    /// Alternation branches of different lengths with distinct first bytes.
    #[test]
    fn test_byte_table_mixed_length_branches() {
        let p = "^(a|bc|def)$";
        let re = build_regex_unchecked(p);
        // First bytes are 'a', 'b', 'd' — all distinct → should optimise.
        assert!(count_byte_tables(&re) > 0);
        for input in &[
            "a", "bc", "def", // positives
            "b", "d", "de", "ab", "bcd", "", "abc", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }
    // -- ByteTable: large alternation ----------------------------------------

    /// 10-way alternation — well above threshold.
    #[test]
    fn test_byte_table_ten_way_alternation() {
        let p = "^(ax|by|cz|dw|ev|fu|gt|hs|ir|jq)$";
        let re = build_regex_unchecked(p);
        assert!(count_byte_tables(&re) > 0);
        for input in &[
            "ax", "by", "cz", "dw", "ev", "fu", "gt", "hs", "ir", "jq", // positives
            "ab", "bx", "az", "xx", "", "axx", "jqq", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

    /// 26-way alternation — one branch per lowercase letter.
    #[test]
    fn test_byte_table_twenty_six_way() {
        let p = "^(a1|b2|c3|d4|e5|f6|g7|h8|i9|j0|kA|lB|mC|nD|oE|pF|qG|rH|sI|tJ|uK|vL|wM|xN|yO|zP)$";
        let re = build_regex_unchecked(p);
        assert!(count_byte_tables(&re) > 0);
        for input in &["a1", "m3", "zP", "j0", "uK"] {
            assert_matches_regex_crate(p, &re, input);
        }
        for input in &["a2", "b1", "A1", "", "a1b2"] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

    // -- ByteTable: nested / multiple tables ---------------------------------

    /// Two independent ByteTable-eligible alternations in sequence.
    #[test]
    fn test_byte_table_two_independent_tables() {
        let p = "^(ab|cd|ef)(gh|ij|kl)$";
        let re = build_regex_unchecked(p);
        // Both groups should produce a ByteTable.
        assert!(count_byte_tables(&re) >= 2);
        for input in &[
            "abgh", "cdij", "efkl", "abkl", "efgh", // positives
            "abab", "ghij", "ab", "abg", "", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

    /// ByteTable alternation nested inside another alternation that also
    /// qualifies.  e.g. `((ax|by|cz)|(dw|ev|fu))`.
    #[test]
    fn test_byte_table_nested_alternations() {
        let p = "^((ax|by|cz)|(dw|ev|fu))$";
        let re = build_regex_unchecked(p);
        // Inner groups have 3 branches each, distinct first bytes.
        assert!(count_byte_tables(&re) > 0);
        for input in &[
            "ax", "by", "cz", "dw", "ev", "fu", // positives
            "ab", "aw", "dx", "", "axby", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

    // -- ByteTable: surrounding literal context ------------------------------
    // -- ByteTable: single-char branches → ByteClass (no optimisation) -------

    /// `(a|b|c)` is collapsed to `[abc]` by regex-syntax, so no ByteTable.
    #[test]
    fn test_byte_table_single_char_alt_becomes_byte_class() {
        let re = build_regex_unchecked("^(a|b|c)$");
        assert_eq!(
            count_byte_tables(&re),
            0,
            "single-char alt should be ByteClass, not ByteTable"
        );
        // Still matches correctly.
        let mut mem = MatcherMemory::default();
        for input in &[b"a" as &[u8], b"b", b"c"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(m.finish(), "expected match for {:?}", input);
        }
        for input in &[b"d" as &[u8], b"", b"ab"] {
            let mut m = mem.matcher(&re);
            m.chunk(input);
            assert!(!m.finish(), "expected no match for {:?}", input);
        }
    }

    /// `(a|b|c|d|e|f|g)` — even many single-char branches become one
    /// ByteClass, not a ByteTable.
    #[test]
    fn test_byte_table_many_single_char_alt_still_byte_class() {
        let re = build_regex_unchecked("^(a|b|c|d|e|f|g)$");
        assert_eq!(count_byte_tables(&re), 0);
    }

    // -- ByteTable: with wildcard / complex suffix ---------------------------
    // -- ByteTable: memory_size accounts for tables --------------------------

    /// Verify memory_size() grows by ~1024 bytes per ByteTable entry.
    #[test]
    fn test_byte_table_memory_size_accounts_for_tables() {
        let re_no_bt = build_regex_unchecked("^(ab|cd)$"); // 2-way, no table
        let re_bt = build_regex_unchecked("^(ab|cd|ef)$"); // 3-way, 1 table
        assert_eq!(count_byte_tables(&re_no_bt), 0);
        assert!(count_byte_tables(&re_bt) > 0);
        // The ByteTable version should be at least 1024 bytes larger
        // (one ByteMap = 256 × size_of::<StateIdx>() = 256 × 4 = 1024).
        let size_diff = re_bt.memory_size() as i64 - re_no_bt.memory_size() as i64;
        assert!(
            size_diff >= 1024,
            "ByteTable regex should be ≥1024 bytes larger, got diff={}",
            size_diff
        );
    }

    // -----------------------------------------------------------------------
    // Regex matching tests
    // -----------------------------------------------------------------------

    /// Parse a pattern in full byte mode (no UTF-8 validity requirement).
    /// Parses `pattern` into HIR in byte mode: Unicode disabled, dot
    /// matches any byte (including newline).  Equivalent to prepending
    /// `(?s-u)` but configured via the builder API instead.
    fn parse_hir_bytes(pattern: &str) -> Hir {
        use regex_syntax::ast::parse::ParserBuilder;
        use regex_syntax::hir::translate::TranslatorBuilder;

        let ast = ParserBuilder::new()
            .build()
            .parse(pattern)
            .expect("regex-syntax AST parse should succeed");
        TranslatorBuilder::new()
            .unicode(false)
            .utf8(false)
            .dot_matches_new_line(true)
            .build()
            .translate(pattern, &ast)
            .expect("regex-syntax HIR translation should succeed")
    }

    /// Assert that a compiled [`Regex`]'s memory footprint equals
    /// `expected_bytes`.  Placed at the end of tests so that matching
    /// failures are surfaced before size mismatches.
    fn assert_memory_size(pattern: &str, regex: &Regex, expected_bytes: usize) {
        let actual = regex.memory_size();
        assert_eq!(
            actual, expected_bytes,
            "memory_size mismatch for pattern `{pattern}`: actual={actual}, expected={expected_bytes}"
        );
    }
    /// Assert that our NFA matcher and the `regex` crate agree on whether
    /// `input` matches the given pattern.
    ///
    /// The pattern is expected to include its own `^` and `$` anchors
    /// where needed.  The `regex` crate is used in byte mode
    /// (`regex::bytes::Regex`) so that `.` matches any byte, consistent
    /// with our engine.
    ///
    /// Two independent matcher runs are exercised using the same
    /// [`MatcherMemory`]:
    /// 1. **chunk** — feeds the entire input at once via [`Matcher::chunk`].
    /// 2. **step-by-step** — feeds bytes one at a time via [`Matcher::step`].
    ///
    /// Both paths call [`Matcher::finish`] to signal end-of-input and
    /// obtain the final match result (which also evaluates `$`).
    ///
    /// Both results are compared against the `regex` crate oracle.
    fn assert_matches_regex_crate(pattern: &str, regex: &Regex, input: &str) {
        let full = format!("(?s-u){}", pattern);
        let re = regex::bytes::Regex::new(&full).expect("regex crate should parse pattern");
        let expected = re.is_match(input.as_bytes());

        // Path 1: feed the whole input via chunk().
        let mut memory = MatcherMemory::default();
        let mut matcher = memory.matcher(regex);
        matcher.chunk(input.as_bytes());
        let actual_chunk = matcher.finish();

        assert_eq!(
            actual_chunk, expected,
            "chunk mismatch for pattern `{}` on input {:?}: ours={}, regex crate={}",
            pattern, input, actual_chunk, expected
        );

        // Path 2: feed bytes one at a time via step().
        // Re-use the same MatcherMemory — matcher() resets all state.
        let mut matcher = memory.matcher(regex);
        for &b in input.as_bytes() {
            matcher.chunk(&[b]);
        }
        let actual_step = matcher.finish();

        assert_eq!(
            actual_step, expected,
            "step-by-step mismatch for pattern `{}` on input {:?}: ours={}, regex crate={}",
            pattern, input, actual_step, expected
        );
    }
    /// Build a compiled [`Regex`] from a pattern string *without*
    /// asserting a specific memory size.  Used by dedup tests that
    /// compare sizes relatively rather than absolutely.
    fn build_regex_unchecked(pattern: &str) -> Regex {
        build_regex_with_unroll(pattern, DEFAULT_MAX_UNROLL_STATES)
    }

    /// Build a compiled [`Regex`] with a specific unroll limit.
    fn build_regex_with_unroll(pattern: &str, max_unroll_states: usize) -> Regex {
        let hir = parse_hir_bytes(pattern);
        let mut builder = RegexBuilder::default();
        builder.max_unroll_states(max_unroll_states);
        builder
            .build(&hir)
            .expect("our builder should accept the HIR")
    }
    // -----------------------------------------------------------------------
    // Data-driven test infrastructure
    // -----------------------------------------------------------------------

    /// Compute the minimum DFA tier that can handle this regex.
    /// 0 = NFA only, 1 = Tier 1+, 2 = Tier 2+, 3 = Tier 3+, 4 = Tier 4 only.
    fn compute_min_tier(re: &Regex) -> u8 {
        if re.dfa_eligible {
            1
        } else if re.tier2_eligible {
            2
        } else if re.tier3_eligible {
            3
        } else if re.tier4_eligible {
            4
        } else {
            0
        }
    }

    /// Test a pattern+input via the NFA simulator (full-chunk + byte-at-a-time).
    fn test_nfa(pattern: &str, re: &Regex, input: &str, expected: bool, unroll: usize) {
        let mut memory = MatcherMemory::default();
        let mut m = memory.nfa_matcher(re);
        m.chunk(input.as_bytes());
        let actual = m.finish();
        assert_eq!(
            actual, expected,
            "NFA chunk mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
        let mut m = memory.nfa_matcher(re);
        for &b in input.as_bytes() {
            m.chunk(&[b]);
        }
        let actual = m.finish();
        assert_eq!(
            actual, expected,
            "NFA single-byte mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
    }

    /// Test a pattern+input via Tier 1 DFA (full-chunk + byte-at-a-time).
    fn test_tier1(pattern: &str, re: &Regex, input: &str, expected: bool, unroll: usize) {
        use crate::dfa::{DfaMatcher, DfaMemory, Tier1DfaCache};
        let mut memory = DfaMemory::default();
        let mut cache = Tier1DfaCache::new();
        cache.prepare(&mut memory, re);
        let mut d = DfaMatcher::new(&mut cache, &mut memory, re);
        d.chunk(input.as_bytes());
        let actual = d.finish();
        assert_eq!(
            actual, expected,
            "Tier1 chunk mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
        let mut d = DfaMatcher::new(&mut cache, &mut memory, re);
        for &b in input.as_bytes() {
            d.chunk(&[b]);
        }
        let actual = d.finish();
        assert_eq!(
            actual, expected,
            "Tier1 single-byte mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
    }

    /// Test a pattern+input via Tier 2 DFA (full-chunk + byte-at-a-time).
    fn test_tier2(pattern: &str, re: &Regex, input: &str, expected: bool, unroll: usize) {
        use crate::dfa::{DfaMemory, Tier2DfaCache, Tier2DfaMatcher};
        let analysis = re
            .tier2_analysis
            .as_ref()
            .expect("tier2_analysis must be present for tier 2 test");
        let mut memory = DfaMemory::default();
        let mut cache = Tier2DfaCache::new();
        cache.prepare(&mut memory, re);
        let mut d = Tier2DfaMatcher::new(&mut cache, &mut memory, re, analysis);
        d.chunk(input.as_bytes());
        let actual = d.finish();
        assert_eq!(
            actual, expected,
            "Tier2 chunk mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
        let mut d = Tier2DfaMatcher::new(&mut cache, &mut memory, re, analysis);
        for &b in input.as_bytes() {
            d.chunk(&[b]);
        }
        let actual = d.finish();
        assert_eq!(
            actual, expected,
            "Tier2 single-byte mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
    }

    /// Test a pattern+input via Tier 3 DFA (full-chunk + byte-at-a-time).
    fn test_tier3(pattern: &str, re: &Regex, input: &str, expected: bool, unroll: usize) {
        use crate::dfa::{DfaMemory, Tier3DfaCache, Tier3DfaMatcher};
        let analysis = re
            .tier3_analysis
            .as_ref()
            .expect("tier3_analysis must be present for tier 3 test");
        let mut memory = DfaMemory::default();
        let mut cache = Tier3DfaCache::new();
        cache.prepare(&mut memory, re, analysis);
        let mut d = Tier3DfaMatcher::new(&mut cache, &mut memory, re, analysis);
        d.chunk(input.as_bytes());
        let actual = d.finish();
        assert_eq!(
            actual, expected,
            "Tier3 chunk mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
        let mut d = Tier3DfaMatcher::new(&mut cache, &mut memory, re, analysis);
        for &b in input.as_bytes() {
            d.chunk(&[b]);
        }
        let actual = d.finish();
        assert_eq!(
            actual, expected,
            "Tier3 single-byte mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
    }

    /// Test a pattern+input via Tier 4 DFA (full-chunk + byte-at-a-time).
    fn test_tier4(pattern: &str, re: &Regex, input: &str, expected: bool, unroll: usize) {
        use crate::dfa::{Tier4DfaCache, Tier4DfaMatcher};
        let mut cache = Tier4DfaCache::new(re.states.len());
        cache.prepare(re);
        let mut pool = CounterPool {
            arena: Vec::new(),
            free: Vec::new(),
            num_counters: re.num_counters,
        };
        let mut d = Tier4DfaMatcher::new(&mut cache, re, &mut pool);
        d.chunk(input.as_bytes());
        let actual = d.finish();
        assert_eq!(
            actual, expected,
            "Tier4 chunk mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
        pool.clear();
        pool.num_counters = re.num_counters;
        let mut d = Tier4DfaMatcher::new(&mut cache, re, &mut pool);
        for &b in input.as_bytes() {
            d.chunk(&[b]);
        }
        let actual = d.finish();
        assert_eq!(
            actual, expected,
            "Tier4 single-byte mismatch for `{}` on {:?} (unroll={}): got={}, expected={}",
            pattern, input, unroll, actual, expected
        );
    }

    /// Test a single compiled regex against the oracle on all inputs,
    /// exercising NFA and every eligible DFA tier.
    fn test_all_tiers(
        pattern: &str,
        re: &Regex,
        oracle: &regex::bytes::Regex,
        inputs: &[(&str, bool)],
        unroll: usize,
    ) {
        for &(input, expected_hint) in inputs {
            let expected = oracle.is_match(input.as_bytes());

            // Sanity-check the hint against the oracle.
            assert_eq!(
                expected, expected_hint,
                "Oracle/hint mismatch for `{}` on {:?} (unroll={}): oracle={}, hint={}",
                pattern, input, unroll, expected, expected_hint
            );

            // Always test NFA.
            test_nfa(pattern, re, input, expected, unroll);

            // Test each DFA tier the regex is actually eligible for.
            if re.dfa_eligible {
                test_tier1(pattern, re, input, expected, unroll);
            }
            if re.tier2_eligible {
                test_tier2(pattern, re, input, expected, unroll);
            }
            if re.tier3_eligible {
                test_tier3(pattern, re, input, expected, unroll);
            }
            if re.tier4_eligible {
                test_tier4(pattern, re, input, expected, unroll);
            }
        }
    }

    /// Central test runner for data-driven match tests.
    ///
    /// For each input, queries the regex crate oracle for the expected
    /// result, then tests NFA and all compatible DFA tiers.
    ///
    /// Two builds are exercised:
    /// 1. With the specified `unroll_limit` — verifies `min_tier` and memory.
    /// 2. With `unroll_limit=0` (disabled) — ensures counter-based DFA tiers
    ///    are still tested even when unrolling would promote the pattern.
    fn run_match_test(
        pattern: &str,
        memory: usize,
        min_tier: u8,
        unroll_limit: usize,
        inputs: &[(&str, bool)],
    ) {
        let re = build_regex_with_unroll(pattern, unroll_limit);

        // Verify declared min_tier matches the compiled regex.
        let actual_tier = compute_min_tier(&re);
        assert_eq!(
            actual_tier, min_tier,
            "min_tier mismatch for `{}` (unroll={}): declared={}, actual={}",
            pattern, unroll_limit, min_tier, actual_tier
        );

        // Oracle: regex crate is the source of truth.
        let full = format!("(?s-u){}", pattern);
        let oracle = regex::bytes::Regex::new(&full).expect("regex crate should parse pattern");

        // Run with the configured unroll limit.
        test_all_tiers(pattern, &re, &oracle, inputs, unroll_limit);

        // Run again with unrolling disabled to exercise counter-based tiers.
        if unroll_limit > 0 {
            let re_no_unroll = build_regex_with_unroll(pattern, 0);
            test_all_tiers(pattern, &re_no_unroll, &oracle, inputs, 0);
        }

        // Assert memory size matches the compiled regex.
        assert_memory_size(pattern, &re, memory);

        // NFA-guided input generation: walk the NFA with counter-boundary
        // schedules to produce structurally meaningful inputs.  The regex
        // crate oracle determines expected results — no manual hints needed.
        let generated = generate_nfa_inputs(&re, 1000);
        test_generated(pattern, &re, &oracle, &generated, unroll_limit);
        if unroll_limit > 0 {
            let re_nu = build_regex_with_unroll(pattern, 0);
            let generated_nu = generate_nfa_inputs(&re_nu, 1000);
            test_generated(pattern, &re_nu, &oracle, &generated_nu, 0);
        }
    }

    // -------------------------------------------------------------------
    // NFA-walk input generator
    // -------------------------------------------------------------------

    /// Maximum byte-length of a generated input.  Keeps runtime bounded
    /// for patterns with large counter ranges or unbounded repetitions.
    const GEN_MAX_INPUT_LEN: usize = 200;

    /// Generate test inputs by walking the NFA with different counter
    /// schedules.
    ///
    /// For each counter, boundary values `{min-1, min, max, max+1}` are
    /// tried (where applicable).  The cross-product of boundary values
    /// across all counters forms the set of schedules, capped at
    /// `max_inputs`.  Each schedule drives a deterministic NFA walk that
    /// emits a byte string.
    ///
    /// The generated inputs exercise counter-boundary edge cases that
    /// hand-picked inputs might miss: off-by-one at min, off-by-one at
    /// max, and combinatorial interactions between multiple counters.
    fn generate_nfa_inputs(regex: &Regex, max_inputs: usize) -> Vec<Vec<u8>> {
        // Build boundary values for each counter.
        let boundary_values: Vec<Vec<usize>> = regex
            .counter_info
            .iter()
            .map(|&(min, max, _)| {
                let mut vals = Vec::new();
                if min > 0 {
                    vals.push(min - 1);
                }
                vals.push(min);
                if max > min {
                    vals.push(max);
                }
                if max < usize::MAX - 1 {
                    vals.push(max + 1);
                }
                vals.sort();
                vals.dedup();
                vals
            })
            .collect();

        let schedules = if boundary_values.is_empty() {
            // No counters — single empty schedule.
            vec![vec![]]
        } else {
            gen_cross_product(&boundary_values, max_inputs)
        };

        let mut inputs: Vec<Vec<u8>> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for schedule in &schedules {
            if let Some(input) = gen_walk_nfa(regex, schedule) {
                if seen.insert(input.clone()) {
                    inputs.push(input);
                }
            }
            if inputs.len() >= max_inputs {
                break;
            }
        }
        inputs
    }

    /// Cartesian product of boundary-value lists, capped at `max`.
    fn gen_cross_product(values: &[Vec<usize>], max: usize) -> Vec<Vec<usize>> {
        let mut result = vec![vec![]];
        for vals in values {
            let mut next = Vec::new();
            for existing in &result {
                for &v in vals {
                    let mut combined = existing.clone();
                    combined.push(v);
                    next.push(combined);
                    if next.len() >= max {
                        return next;
                    }
                }
            }
            result = next;
        }
        result
    }

    /// Walk the NFA from `start` to `Match` using the given counter
    /// schedule (target iteration count per counter).  Returns the byte
    /// string consumed along the path, or `None` if no path to `Match`
    /// exists with this schedule.
    fn gen_walk_nfa(regex: &Regex, schedule: &[usize]) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        let mut counter_counts = vec![0usize; regex.num_counters];
        let mut steps = 0u32;
        if gen_walk_rec(
            regex,
            regex.start,
            schedule,
            &mut counter_counts,
            &mut out,
            GEN_EPSILON_BUDGET,
            &mut steps,
        ) {
            Some(out)
        } else {
            None
        }
    }

    /// Recursive NFA walk.  Returns `true` if `Match` is reachable from
    /// `idx` under the given counter schedule.  Bytes consumed along the
    /// path are appended to `out`.
    ///
    /// `epsilon_budget` limits consecutive epsilon (non-consuming) steps.
    /// Each consuming state resets it.  This prevents infinite loops
    /// through zero-width repetitions like `(a{0,2})+`.
    const GEN_EPSILON_BUDGET: usize = 200;

    /// Global step budget per NFA walk.  Limits total work (including
    /// backtracking) to prevent combinatorial explosion on patterns with
    /// deep nesting or many Split nodes.
    const GEN_MAX_STEPS: u32 = 50_000;

    fn gen_walk_rec(
        regex: &Regex,
        idx: StateIdx,
        schedule: &[usize],
        counter_counts: &mut [usize],
        out: &mut Vec<u8>,
        epsilon_budget: usize,
        steps: &mut u32,
    ) -> bool {
        *steps += 1;
        if *steps > GEN_MAX_STEPS || epsilon_budget == 0 || idx == StateIdx::NONE {
            return false;
        }
        if idx.idx() >= regex.states.len() {
            return false;
        }

        let eb = epsilon_budget - 1; // decremented for epsilon states
        match regex.states[idx] {
            State::Match => true,

            State::Split { out: o1, out1: o2 } => {
                // Try first branch; backtrack to second on failure.
                let saved_len = out.len();
                let saved_counts: Vec<usize> = counter_counts.to_vec();
                if gen_walk_rec(regex, o1, schedule, counter_counts, out, eb, steps) {
                    return true;
                }
                out.truncate(saved_len);
                counter_counts.copy_from_slice(&saved_counts);
                gen_walk_rec(regex, o2, schedule, counter_counts, out, eb, steps)
            }

            // Consuming states reset the epsilon budget.
            State::Byte { byte, out: next } => {
                if out.len() >= GEN_MAX_INPUT_LEN {
                    return false;
                }
                out.push(byte);
                if gen_walk_rec(
                    regex, next, schedule, counter_counts, out, GEN_EPSILON_BUDGET, steps,
                ) {
                    true
                } else {
                    out.pop();
                    false
                }
            }

            State::ByteCI { byte, out: next } => {
                if out.len() >= GEN_MAX_INPUT_LEN {
                    return false;
                }
                out.push(byte);
                if gen_walk_rec(
                    regex, next, schedule, counter_counts, out, GEN_EPSILON_BUDGET, steps,
                ) {
                    true
                } else {
                    out.pop();
                    false
                }
            }

            State::ByteClass { class, out: next } => {
                if out.len() >= GEN_MAX_INPUT_LEN {
                    return false;
                }
                let bc = &regex.classes[class.idx()];
                // Prefer 'a' (word char), then '0', ' ', then first match.
                let byte = [b'a', b'0', b' ']
                    .into_iter()
                    .find(|&b| bc[b])
                    .or_else(|| (0..=255u8).find(|&b| bc[b]));
                let Some(byte) = byte else { return false };
                out.push(byte);
                if gen_walk_rec(
                    regex, next, schedule, counter_counts, out, GEN_EPSILON_BUDGET, steps,
                ) {
                    true
                } else {
                    out.pop();
                    false
                }
            }

            State::ByteTable { table } => {
                if out.len() >= GEN_MAX_INPUT_LEN {
                    return false;
                }
                let bt = &regex.byte_tables[table.idx()];
                // Try preferred bytes first, then scan.
                let candidates = [b'a', b'0', b' '];
                for &byte in &candidates {
                    let target = bt[byte];
                    if target != StateIdx::NONE {
                        out.push(byte);
                        let saved_counts: Vec<usize> = counter_counts.to_vec();
                        if gen_walk_rec(
                            regex, target, schedule, counter_counts, out,
                            GEN_EPSILON_BUDGET, steps,
                        ) {
                            return true;
                        }
                        out.pop();
                        counter_counts.copy_from_slice(&saved_counts);
                    }
                }
                for byte in 0..=255u8 {
                    if candidates.contains(&byte) {
                        continue;
                    }
                    let target = bt[byte];
                    if target != StateIdx::NONE {
                        out.push(byte);
                        let saved_counts: Vec<usize> = counter_counts.to_vec();
                        if gen_walk_rec(
                            regex, target, schedule, counter_counts, out,
                            GEN_EPSILON_BUDGET, steps,
                        ) {
                            return true;
                        }
                        out.pop();
                        counter_counts.copy_from_slice(&saved_counts);
                    }
                }
                false
            }

            // Epsilon states decrement the budget.
            State::Assert { out: next, .. } => {
                gen_walk_rec(regex, next, schedule, counter_counts, out, eb, steps)
            }

            State::CounterInstance { counter, out: next } => {
                counter_counts[counter.idx()] = 0;
                gen_walk_rec(regex, next, schedule, counter_counts, out, eb, steps)
            }

            State::CounterIncrement {
                counter,
                out: cont,
                out1: break_out,
                ..
            } => {
                let c = counter.idx();
                let target = schedule.get(c).copied().unwrap_or(1);
                counter_counts[c] += 1;
                if counter_counts[c] < target {
                    gen_walk_rec(regex, cont, schedule, counter_counts, out, eb, steps)
                } else {
                    gen_walk_rec(regex, break_out, schedule, counter_counts, out, eb, steps)
                }
            }
        }
    }

    /// Test NFA-generated inputs against the oracle on all eligible tiers.
    fn test_generated(
        pattern: &str,
        re: &Regex,
        oracle: &regex::bytes::Regex,
        inputs: &[Vec<u8>],
        unroll: usize,
    ) {
        for input_bytes in inputs {
            let input = match std::str::from_utf8(input_bytes) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let expected = oracle.is_match(input.as_bytes());
            test_nfa(pattern, re, input, expected, unroll);
            if re.dfa_eligible {
                test_tier1(pattern, re, input, expected, unroll);
            }
            if re.tier2_eligible {
                test_tier2(pattern, re, input, expected, unroll);
            }
            if re.tier3_eligible {
                test_tier3(pattern, re, input, expected, unroll);
            }
            if re.tier4_eligible {
                test_tier4(pattern, re, input, expected, unroll);
            }
        }
    }

    /// Resolve unroll_limit: if specified use it, otherwise use the default.
    macro_rules! unroll_limit {
        () => {
            DEFAULT_MAX_UNROLL_STATES
        };
        ($val:literal) => {
            $val
        };
    }

    /// Generate one `#[test]` function per entry in the table.
    ///
    /// Each entry specifies `pattern`, `memory` (0 to skip), `min_tier`,
    /// an optional `unroll_limit` (defaults to `DEFAULT_MAX_UNROLL_STATES`),
    /// and a list of `(input, expected)` pairs.
    macro_rules! match_tests {
        ($(
            $name:ident {
                pattern: $pattern:literal,
                memory: $memory:literal,
                min_tier: $tier:literal,
                $(unroll_limit: $unroll:literal,)?
                inputs: [$(($input:literal, $expected:literal)),* $(,)?],
            }
        )*) => {
            $(
                #[test]
                fn $name() {
                    run_match_test(
                        $pattern,
                        $memory,
                        $tier,
                        unroll_limit!($($unroll)?),
                        &[$(($input, $expected)),*],
                    );
                }
            )*
        };
    }

    match_tests! {
        test_counting {
            pattern: "^.*a.{3}bc$",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("aybzbc", true),
                ("axaybzbc", true),
                ("a123bc", true),
                ("za999bc", true),
                ("abc", false),
                ("a12bc", false),
                ("a123bd", false),
                ("a123xc", false),
                ("", false),
                ("bc", false),
                ("a123b", false),
                ("a123", false),
            ],
        }
        test_range {
            pattern: "^(a|bc){1,2}$",
            memory: 1004,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("bc", true),
                ("", false),
                ("x", false),
                ("b", false),
                ("c", false),
                ("ab", false),
                ("ca", false),
                ("bca", true),
            ],
        }
        test_nested_counting {
            pattern: "^((a|bc){1,2}){2,3}$",
            memory: 1631,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                ("bc", false),
                ("x", false),
                ("b", false),
                ("c", false),
                ("aax", false),
                ("aaaaaa", true),
                ("abcbc", true),
                ("bcbca", true),
            ],
        }
        test_aaaaa {
            pattern: "^(a|a?){2,3}$",
            memory: 1136,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", false),
                ("aaaaa", false),
                ("b", false),
                ("ab", false),
                ("ba", false),
                ("aab", false),
            ],
        }
        test_one_plus_basic {
            pattern: "^a+$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("", false),
                ("b", false),
                ("ab", false),
                ("ba", false),
                ("aab", false),
            ],
        }
        test_one_plus_wildcard {
            pattern: "^.+$",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", true),
                ("ab", true),
                ("abc", true),
            ],
        }
        test_one_plus_catenation {
            pattern: "^a+b+$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                ("b", false),
                ("ab", true),
                ("aab", true),
                ("abb", true),
                ("aabb", true),
                ("ba", false),
            ],
        }
        test_one_plus_group {
            pattern: "^(ab)+$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                ("ab", true),
                ("abab", true),
                ("ababab", true),
                ("aba", false),
            ],
        }
        test_one_plus_alternate {
            pattern: "^(a|b)+$",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("b", true),
                ("ab", true),
                ("ba", true),
                ("aab", true),
                ("bba", true),
                ("abab", true),
                ("", false),
                ("c", false),
                ("ac", false),
                ("ca", false),
                ("abc", false),
            ],
        }
        test_one_plus_with_counting {
            pattern: "^.*a.{3}b+c$",
            memory: 1260,
            min_tier: 1,
            inputs: [
                ("a123bc", true),
                ("a123bbc", true),
                ("a123bbbc", true),
                ("xa123bc", true),
                ("xxxa999bc", true),
                ("a123c", false),
                ("a12bc", false),
                ("a123bd", false),
                ("a123bx", false),
                ("", false),
                ("x123bc", false),
                ("a123b", false),
            ],
        }
        test_repetition_inside_one_plus {
            pattern: "^(a{2,3})+$",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", true),
                ("aaaaaa", true),
                ("aaaaaaa", true),
                ("aaaaaaaa", true),
                ("aaaaaaaaa", true),
                ("b", false),
            ],
        }
        test_range_alternation_inside_one_plus {
            pattern: "^((a|bc){1,2})+$",
            memory: 1037,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", true),
                ("bc", true),
                ("b", false),
                ("c", false),
            ],
        }
        test_one_plus_inside_repetition {
            pattern: "^(a+){2,3}$",
            memory: 938,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", true),
                ("aaaaaa", true),
                ("aaaaaaa", true),
                ("b", false),
            ],
        }
        test_one_plus_alternation_inside_repetition {
            pattern: "^((a|b)+){2,4}$",
            memory: 1293,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                ("b", false),
                ("c", false),
            ],
        }
        test_mixed_plus_and_repetition_inside_one_plus {
            pattern: "^(a+b{2,3})+$",
            memory: 938,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                ("ab", false),
                ("abb", true),
                ("abbb", true),
                ("abbbb", false),
                ("aabb", true),
                ("aabbb", true),
                ("abbabb", true),
                ("abbaabb", true),
                ("abbabbbabb", true),
                ("aabbaabbb", true),
                ("aabbbaabbb", true),
            ],
        }
        test_min_zero_basic {
            pattern: "^a{0,2}$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", false),
                ("b", false),
            ],
        }
        test_min_zero_max_one {
            pattern: "^a{0,1}$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", false),
                ("b", false),
            ],
        }
        test_min_zero_alternation {
            pattern: "^(a|bc){0,3}$",
            memory: 1202,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("bc", true),
                ("b", false),
            ],
        }
        test_min_zero_unbounded {
            pattern: "^a{0,}$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("b", false),
                ("ab", false),
                ("ba", false),
                ("aab", false),
                ("baa", false),
            ],
        }
        test_min_zero_unbounded_group {
            pattern: "^(ab){0,}$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", false),
                ("ab", true),
                ("abab", true),
                ("ababab", true),
                ("aba", false),
            ],
        }
        test_min_zero_inside_one_plus {
            pattern: "^x(a{0,2})+y$",
            memory: 938,
            min_tier: 1,
            inputs: [
                ("xy", true),
                ("xay", true),
                ("xaay", true),
                ("xaaay", true),
                ("xaaaay", true),
                ("x", false),
                ("y", false),
                ("", false),
                ("xby", false),
                ("xaby", false),
                ("ay", false),
                ("xa", false),
                ("aay", false),
            ],
        }
        test_min_zero_inside_repetition {
            pattern: "^(a{0,2}){2,3}$",
            memory: 1136,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", true),
                ("aaaaaa", true),
                ("aaaaaaa", false),
                ("b", false),
            ],
        }
        test_one_plus_inside_min_zero_repetition {
            pattern: "^(a+){0,3}$",
            memory: 1004,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", true),
                ("aaaaaa", true),
                ("b", false),
            ],
        }
        test_min_zero_wildcard {
            pattern: "^.{0,3}$",
            memory: 1161,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("ab", true),
                ("abc", true),
                ("abcd", false),
            ],
        }
        test_none_min_repetition {
            pattern: "^a{0,3}$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", false),
                ("b", false),
                ("ab", false),
                ("aab", false),
            ],
        }
        test_literal_single {
            pattern: "^a$",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("", false),
                ("b", false),
                ("aa", false),
                ("ab", false),
                ("ba", false),
            ],
        }
        test_literal_multi {
            pattern: "^abc$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("", false),
                ("a", false),
                ("ab", false),
                ("abd", false),
                ("abcd", false),
                ("xabc", false),
                ("abcx", false),
                ("xabcx", false),
                ("cba", false),
                ("bac", false),
            ],
        }
        test_dot_single {
            pattern: "^.$",
            memory: 996,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("z", true),
                ("0", true),
                (" ", true),
                ("", false),
                ("ab", false),
                ("abc", false),
            ],
        }
        test_alternation_bare {
            pattern: "^(a|bc)$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("bc", true),
                ("", false),
                ("b", false),
                ("c", false),
                ("ab", false),
                ("abc", false),
                ("x", false),
                ("bca", false),
            ],
        }
        test_alternation_three_way {
            pattern: "^(a|b|c)$",
            memory: 996,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("b", true),
                ("c", true),
                ("", false),
                ("d", false),
                ("ab", false),
                ("abc", false),
            ],
        }
        test_question_mark_single {
            pattern: "^a?$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", false),
                ("b", false),
                ("ab", false),
            ],
        }
        test_question_mark_group {
            pattern: "^(ab)?$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("", true),
                ("ab", true),
                ("a", false),
                ("b", false),
                ("ba", false),
                ("abab", false),
                ("abc", false),
            ],
        }
        test_question_mark_prefix {
            pattern: "^a?b$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("b", true),
                ("ab", true),
                ("", false),
                ("a", false),
                ("aab", false),
                ("bb", false),
                ("cb", false),
                ("abc", false),
            ],
        }
        test_star_single {
            pattern: "^a*$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("b", false),
                ("ab", false),
                ("ba", false),
                ("aab", false),
                ("baa", false),
            ],
        }
        test_star_group {
            pattern: "^(ab)*$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("", true),
                ("ab", true),
                ("abab", true),
                ("ababab", true),
                ("a", false),
                ("b", false),
                ("ba", false),
                ("aba", false),
                ("abba", false),
                ("abc", false),
            ],
        }
        test_star_then_literal {
            pattern: "^a*b$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("b", true),
                ("ab", true),
                ("aab", true),
                ("aaab", true),
                ("", false),
                ("a", false),
                ("aa", false),
                ("bb", false),
                ("ba", false),
                ("abc", false),
                ("aabb", false),
            ],
        }
        test_min_n_unbounded {
            pattern: "^a{2,}$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", true),
                ("", false),
                ("a", false),
                ("b", false),
                ("ab", false),
                ("aab", false),
                ("baa", false),
            ],
        }
        test_min_n_unbounded_group {
            pattern: "^(ab){2,}$",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("abab", true),
                ("ababab", true),
                ("abababab", true),
                ("", false),
                ("ab", false),
                ("a", false),
                ("aba", false),
                ("abba", false),
                ("ababc", false),
                ("xabab", false),
            ],
        }
        test_bounded_range {
            pattern: "^a{3,5}$",
            memory: 938,
            min_tier: 1,
            inputs: [
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", true),
                ("", false),
                ("a", false),
                ("aa", false),
                ("aaaaaa", false),
                ("b", false),
                ("aaab", false),
                ("baaa", false),
            ],
        }
        // ── Non-fixed unrolling edge-case tests ──────────────────────────
        test_unroll_byte_class_bounded {
            pattern: "[0-9a-f]{1,4}",
            memory: 1128,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("0f", true),
                ("abcd", true),
                ("", false),
                ("g", false),
                ("xyz", false),
            ],
        }
        test_unroll_byte_class_unbounded {
            pattern: "[0-9a-f]{2,}",
            memory: 996,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("0123456789", true),
                ("", false),
                ("a", false),
                ("g", false),
            ],
        }
        test_unroll_at_budget_limit {
            pattern: "^a{1,16}$",
            memory: 1730,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("aaaa", true),
                ("aaaaaaaaaaaaaaaa", true),
                ("", false),
                ("aaaaaaaaaaaaaaaaa", false),
                ("b", false),
            ],
        }
        test_unroll_over_budget {
            pattern: "^a{1,17}$",
            memory: 807,
            min_tier: 2,
            inputs: [
                ("a", true),
                ("aaaa", true),
                ("aaaaaaaaaaaaaaaaa", true),
                ("", false),
                ("aaaaaaaaaaaaaaaaaa", false),
                ("b", false),
            ],
        }
        test_unroll_zero_min_bounded {
            pattern: "^a{0,4}$",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", false),
                ("b", false),
            ],
        }
        // ── Complex-body unrolling tests (estimate_nfa_states) ────────────
        test_unroll_multi_byte_fixed {
            pattern: "^(ab){3}$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("ababab", true),
                ("", false),
                ("ab", false),
                ("abab", false),
                ("abababab", false),
                ("aaa", false),
            ],
        }
        test_unroll_multi_byte_bounded {
            pattern: "^(ab){2,4}$",
            memory: 1037,
            min_tier: 1,
            inputs: [
                ("abab", true),
                ("ababab", true),
                ("abababab", true),
                ("", false),
                ("ab", false),
                ("ababababab", false),
            ],
        }
        test_unroll_alternation_body {
            pattern: "^(a|bc){2,3}$",
            memory: 1136,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("abc", true),
                ("bca", true),
                ("bcbc", true),
                ("aaa", true),
                ("bcbcbc", true),
                ("", false),
                ("a", false),
                ("bc", false),
                ("aaaa", false),
            ],
        }
        test_unroll_nested_fixed_flattened {
            pattern: "^((a{2}){3}){2}$",
            memory: 1103,
            min_tier: 1,
            inputs: [
                ("aaaaaaaaaaaa", true),
                ("aaaaaaaaaaa", false),
                ("aaaaaaaaaaaaa", false),
                ("", false),
            ],
        }
        test_unroll_ipv4_octets {
            pattern: r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}",
            memory: 1656,
            min_tier: 1,
            inputs: [
                ("192.168.1.1", true),
                ("10.0.0.1", true),
                ("0.0.0.0", true),
                ("999.999.999.999", true),
                ("1.2.3", false),
                ("1.2.3.", false),
                ("abc", false),
            ],
        }
        test_unroll_concat_with_inner_rep {
            pattern: "(a+b){2,3}",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("abab", true),
                ("aabab", true),
                ("ababab", true),
                ("abaabb", true),
                ("ab", false),
                ("a", false),
                ("bb", false),
            ],
        }
        test_exact_repetition {
            pattern: "^a{3,3}$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("aaa", true),
                ("", false),
                ("a", false),
                ("aa", false),
                ("aaaa", false),
                ("aaaaa", false),
                ("b", false),
                ("bbb", false),
            ],
        }
        test_wildcard_fixed_unrolled {
            pattern: "^a.{3}b$",
            memory: 1128,
            min_tier: 1,
            inputs: [
                ("a123b", true),
                ("a   b", true),
                ("abcdb", true),
                ("ab", false),
                ("a12b", false),
                ("a1234b", false),
                ("", false),
            ],
        }
        test_byte_class_range {
            pattern: "^[a-c]$",
            memory: 996,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", true),
                ("b", true),
                ("c", true),
                ("d", false),
                ("ab", false),
            ],
        }
        test_byte_class_one_plus {
            pattern: "^[a-c]+$",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", true),
                ("abc", true),
                ("cba", true),
                ("abcd", false),
                ("d", false),
            ],
        }
        test_byte_class_counted {
            pattern: "^[a-c]{2,3}$",
            memory: 1095,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                ("ab", true),
                ("abc", true),
                ("abca", false),
                ("cc", true),
                ("dd", false),
            ],
        }
        test_byte_class_disjoint {
            pattern: "^[ax]$",
            memory: 996,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", true),
                ("x", true),
                ("b", false),
                ("ax", false),
            ],
        }
        test_byte_class_multi_range {
            pattern: "^[a-cx-z]+$",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", true),
                ("x", true),
                ("axbycz", true),
                ("d", false),
                ("w", false),
                ("abcxyz", true),
            ],
        }
        test_byte_class_with_wildcard {
            pattern: "^[a-c].*[x-z]$",
            memory: 1607,
            min_tier: 1,
            inputs: [
                ("ax", true),
                ("a123z", true),
                ("bx", true),
                ("dx", false),
                ("", false),
                ("a", false),
            ],
        }
        test_digit {
            pattern: r#"^\d$"#,
            memory: 996,
            min_tier: 1,
            inputs: [
                ("0", true),
                ("5", true),
                ("9", true),
                ("", false),
                ("a", false),
                ("z", false),
                (" ", false),
                ("00", false),
                ("12", false),
            ],
        }
        test_digit_plus {
            pattern: r#"^\d+$"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("0", true),
                ("42", true),
                ("999", true),
                ("0123456789", true),
                ("", false),
                ("a", false),
                ("12a", false),
                ("a12", false),
                ("1 2", false),
            ],
        }
        test_digit_counted {
            pattern: r#"^\d{3,5}$"#,
            memory: 1194,
            min_tier: 1,
            inputs: [
                ("123", true),
                ("1234", true),
                ("12345", true),
                ("", false),
                ("1", false),
                ("12", false),
                ("123456", false),
                ("abc", false),
                ("12a", false),
            ],
        }
        test_non_digit {
            pattern: r#"^\D$"#,
            memory: 996,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("z", true),
                (" ", true),
                ("!", true),
                ("", false),
                ("0", false),
                ("5", false),
                ("9", false),
                ("aa", false),
            ],
        }
        test_non_digit_plus {
            pattern: r#"^\D+$"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("hello world", true),
                ("!@#", true),
                ("", false),
                ("0", false),
                ("abc1", false),
                ("1abc", false),
            ],
        }
        test_space {
            pattern: r#"^\s$"#,
            memory: 996,
            min_tier: 1,
            inputs: [
                (" ", true),
                ("\t", true),
                ("\n", true),
                ("\r", true),
                ("", false),
                ("a", false),
                ("0", false),
                ("  ", false),
            ],
        }
        test_space_plus {
            pattern: r#"^\s+$"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                (" ", true),
                ("   ", true),
                (" \t\n\r", true),
                ("", false),
                ("a", false),
                (" a", false),
                ("a ", false),
            ],
        }
        test_non_space {
            pattern: r#"^\S$"#,
            memory: 996,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("0", true),
                ("!", true),
                ("", false),
                (" ", false),
                ("\t", false),
                ("\n", false),
                ("aa", false),
            ],
        }
        test_non_space_plus {
            pattern: r#"^\S+$"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("123", true),
                ("a1!", true),
                ("", false),
                (" ", false),
                ("a b", false),
                (" abc", false),
            ],
        }
        test_word {
            pattern: r#"^\w$"#,
            memory: 996,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("Z", true),
                ("0", true),
                ("9", true),
                ("_", true),
                ("", false),
                (" ", false),
                ("!", false),
                ("-", false),
                ("ab", false),
            ],
        }
        test_word_plus {
            pattern: r#"^\w+$"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("hello", true),
                ("foo_bar", true),
                ("x123", true),
                ("___", true),
                ("", false),
                (" ", false),
                ("hello world", false),
                ("foo-bar", false),
            ],
        }
        test_word_counted {
            pattern: r#"^\w{2,4}$"#,
            memory: 1161,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("abc", true),
                ("a1_Z", true),
                ("", false),
                ("a", false),
                ("abcde", false),
                ("!!", false),
                ("a b", false),
            ],
        }
        test_non_word {
            pattern: r#"^\W$"#,
            memory: 996,
            min_tier: 1,
            inputs: [
                (" ", true),
                ("!", true),
                ("-", true),
                (".", true),
                ("", false),
                ("a", false),
                ("0", false),
                ("_", false),
                ("  ", false),
            ],
        }
        test_non_word_plus {
            pattern: r#"^\W+$"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                (" ", true),
                ("!@#", true),
                (" - ", true),
                ("", false),
                ("a", false),
                ("0", false),
                (" a ", false),
                ("!a!", false),
            ],
        }
        test_predefined_mixed {
            pattern: r#"^\d+\s+\w+$"#,
            memory: 1673,
            min_tier: 1,
            inputs: [
                ("42 hello", true),
                ("0\tfoo", true),
                ("123  x", true),
                ("7 _", true),
                ("", false),
                ("42", false),
                ("42 ", false),
                (" hello", false),
                ("hello 42", false),
                ("42hello", false),
            ],
        }
        test_anchor_start_only {
            pattern: "^abc",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abcdef", true),
                ("abc123", true),
                ("", false),
                ("xabc", false),
                ("ab", false),
                ("abd", false),
                ("zabc", false),
            ],
        }
        test_anchor_end_only {
            pattern: "abc$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("xabc", true),
                ("123abc", true),
                ("xxabc", true),
                ("", false),
                ("abcx", false),
                ("ab", false),
                ("abcabd", false),
            ],
        }
        test_unanchored_literal {
            pattern: "abc",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("xabc", true),
                ("abcx", true),
                ("xabcx", true),
                ("xxabcxx", true),
                ("", false),
                ("ab", false),
                ("abd", false),
                ("axbc", false),
                ("bca", false),
            ],
        }
        test_anchor_start_wildcard {
            pattern: "^a.b",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("axb", true),
                ("axbyyy", true),
                ("a1b", true),
                ("ab", false),
                ("yaxb", false),
                ("", false),
            ],
        }
        test_anchor_end_wildcard {
            pattern: "a.b$",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("axb", true),
                ("yyyaxb", true),
                ("a1b", true),
                ("ab", false),
                ("axby", false),
                ("", false),
            ],
        }
        test_both_anchors_quantifiers {
            pattern: "^a+b+$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("aab", true),
                ("abb", true),
                ("aabb", true),
                ("", false),
                ("a", false),
                ("b", false),
                ("ba", false),
                ("xab", false),
                ("abx", false),
            ],
        }
        test_unanchored_quantifiers {
            pattern: "a+b+",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("aab", true),
                ("abb", true),
                ("xab", true),
                ("xaabb", true),
                ("abx", true),
                ("xabx", true),
                ("", false),
                ("a", false),
                ("b", false),
                ("ba", false),
                ("xyz", false),
            ],
        }
        test_anchor_start_one_plus {
            pattern: "^a+",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("aa", true),
                ("aab", true),
                ("abc", true),
                ("", false),
                ("b", false),
                ("ba", false),
                ("baa", false),
            ],
        }
        test_anchor_end_one_plus {
            pattern: "a+$",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("aa", true),
                ("ba", true),
                ("baa", true),
                ("xxa", true),
                ("", false),
                ("b", false),
                ("ab", false),
                ("aab", false),
            ],
        }
        test_anchors_empty {
            pattern: "^$",
            memory: 707,
            min_tier: 1,
            inputs: [("", true), ("a", false), ("ab", false)],
        }
        test_anchor_start_bare {
            pattern: "^",
            memory: 674,
            min_tier: 1,
            inputs: [("", true), ("a", true), ("abc", true)],
        }
        test_anchor_end_bare {
            pattern: "$",
            memory: 674,
            min_tier: 1,
            inputs: [("", true), ("a", true), ("abc", true)],
        }
        test_unanchored_alternation {
            pattern: "(a|b)",
            memory: 930,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("b", true),
                ("xa", true),
                ("bx", true),
                ("xax", true),
                ("", false),
                ("c", false),
                ("xyz", false),
            ],
        }
        test_unanchored_star_literal {
            pattern: "a*b",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("b", true),
                ("ab", true),
                ("aab", true),
                ("xb", true),
                ("xab", true),
                ("bx", true),
                ("", false),
                ("a", false),
                ("aa", false),
                ("xyz", false),
            ],
        }
        test_anchor_in_alternation_start {
            pattern: "a|^b",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("xa", true),
                ("ax", true),
                ("xax", true),
                ("b", true),
                ("bx", true),
                ("bxx", true),
                ("ba", true),
                ("ab", true),
                ("", false),
                ("c", false),
                ("xb", false),
                ("xbx", false),
                ("xyz", false),
            ],
        }
        test_anchor_in_alternation_end {
            pattern: "a$|b",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("xa", true),
                ("xxa", true),
                ("b", true),
                ("xb", true),
                ("bx", true),
                ("xbx", true),
                ("ba", true),
                ("ab", true),
                ("", false),
                ("c", false),
                ("ax", false),
                ("xax", false),
                ("xyz", false),
            ],
        }
        test_multiline_start_basic {
            pattern: "(?m)^abc",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abc\ndef", true),
                ("xxx\nabc", true),
                ("xxx\nabc\nyyy", true),
                ("\nabc", true),
                ("xabc", false),
                ("", false),
                ("\n", false),
            ],
        }
        test_multiline_end_basic {
            pattern: "(?m)abc$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abc\ndef", true),
                ("xxx\nabc", true),
                ("xxxabc\nyyy", true),
                ("abc\n", true),
                ("abcx", false),
                ("", false),
                ("\n", false),
            ],
        }
        test_multiline_both_anchors {
            pattern: "(?m)^abc$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abc\ndef", true),
                ("def\nabc", true),
                ("xxx\nabc\nyyy", true),
                ("\nabc\n", true),
                ("abc\n", true),
                ("\nabc", true),
                ("xabc", false),
                ("abcx", false),
                ("xabcx", false),
                ("", false),
                ("\n", false),
                ("\n\n", false),
            ],
        }
        test_multiline_multi_lines {
            pattern: "(?m)^abc$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc\ndef\nghi", true),
                ("def\nabc\nghi", true),
                ("def\nghi\nabc", true),
                ("def\nghi\njkl", false),
            ],
        }
        test_multiline_catenation_across_newline {
            pattern: r#"(?m)abc$\n^def"#,
            memory: 938,
            min_tier: 1,
            inputs: [
                ("abc\ndef", true),
                ("xxx\nabc\ndef\nyyy", true),
                ("abc\nxef", false),
                ("abcdef", false),
            ],
        }
        test_multiline_with_dot_plus {
            pattern: "(?m)^.+$",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abc\ndef", true),
                ("\nabc", true),
                ("abc\n", true),
                ("", false),
                ("\n", true),
                ("\n\n", true),
            ],
        }
        test_multiline_with_counting {
            pattern: r#"(?m)^\d{2,4}$"#,
            memory: 1161,
            min_tier: 1,
            inputs: [
                ("12", true),
                ("123", true),
                ("1234", true),
                ("1", false),
                ("12345", false),
                ("xx\n12\nyy", true),
                ("xx\n12345\nyy", false),
                ("xx\n1\nyy", false),
                ("12\n1234\n12345", true),
            ],
        }
        test_multiline_edge_cases_empty {
            pattern: "(?m)^$",
            memory: 707,
            min_tier: 1,
            inputs: [
                ("", true),
                ("\n", true),
                ("\n\n", true),
                ("abc", false),
                ("abc\n", true),
                ("\nabc", true),
                ("abc\n\ndef", true),
            ],
        }
        test_multiline_start_only_empty_lines {
            pattern: "(?m)^",
            memory: 674,
            min_tier: 1,
            inputs: [("", true), ("a", true), ("\n", true)],
        }
        test_multiline_end_only {
            pattern: "(?m)$",
            memory: 674,
            min_tier: 1,
            inputs: [("", true), ("a", true), ("\n", true)],
        }
        test_multiline_alternation {
            pattern: "(?m)^(abc|def)$",
            memory: 938,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("def", true),
                ("abc\ndef", true),
                ("ghi\nabc\njkl", true),
                ("ghi\njkl", false),
                ("abcdef", false),
            ],
        }
        test_multiline_mixed_with_nonmultiline {
            pattern: "^abc(?m:$)",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abc\ndef", true),
                ("xabc", false),
                ("\nabc", false),
            ],
        }
        test_crlf_start_basic {
            pattern: "(?Rm)^abc",
            memory: 773,
            min_tier: 0,
            inputs: [
                ("abc", true),
                ("\nabc", true),
                ("\rabc", true),
                ("\r\nabc", true),
                ("xxx\nabc", true),
                ("xxx\rabc", true),
                ("xxx\r\nabc", true),
                ("xabc", false),
            ],
        }
        test_crlf_end_basic {
            pattern: "(?Rm)abc$",
            memory: 773,
            min_tier: 0,
            inputs: [
                ("abc", true),
                ("abc\n", true),
                ("abc\r", true),
                ("abc\r\n", true),
                ("abc\nxxx", true),
                ("abc\rxxx", true),
                ("abc\r\nxxx", true),
                ("abcx", false),
            ],
        }
        test_crlf_both_anchors {
            pattern: "(?Rm)^abc$",
            memory: 806,
            min_tier: 0,
            inputs: [
                ("abc", true),
                ("abc\r\n", true),
                ("\r\nabc", true),
                ("\r\nabc\r\n", true),
                ("xxx\r\nabc\r\nyyy", true),
                ("abc\n", true),
                ("\nabc\n", true),
                ("abc\r", true),
                ("\rabc\r", true),
                ("xabc", false),
                ("abcx", false),
            ],
        }
        test_crlf_empty_lines {
            pattern: "(?Rm)^$",
            memory: 707,
            min_tier: 0,
            inputs: [
                ("", true),
                ("\n", true),
                ("\r\n", true),
                ("\r", true),
                ("\r\n\r\n", true),
                ("abc\r\n\r\ndef", true),
                ("abc", false),
            ],
        }
        test_crlf_multiline_vs_lf_1 {
            pattern: "(?Rm)^abc$",
            memory: 806,
            min_tier: 0,
            inputs: [("\r\nabc\r\n", true)],
        }
        test_crlf_multiline_vs_lf_2 {
            pattern: "(?m)^abc$",
            memory: 806,
            min_tier: 1,
            inputs: [("\r\nabc\r\n", false)],
        }
        test_crlf_with_dot_plus {
            pattern: "(?Rm)^.+$",
            memory: 1029,
            min_tier: 0,
            inputs: [
                ("abc", true),
                ("abc\r\ndef", true),
                ("\r\nabc", true),
                ("abc\r\n", true),
                ("", false),
                ("\r\n", true),
                ("\r\n\r\n", true),
            ],
        }
        test_crlf_with_counting {
            pattern: r#"(?Rm)^\d{2,4}$"#,
            memory: 1161,
            min_tier: 0,
            inputs: [
                ("12", true),
                ("1234", true),
                ("12345", false),
                ("xx\r\n12\r\nyy", true),
                ("xx\r\n1\r\nyy", false),
            ],
        }
        test_crlf_bare_cr_as_line_terminator {
            pattern: "(?Rm)^abc$",
            memory: 806,
            min_tier: 0,
            inputs: [("xxx\rabc\ryyy", true), ("xxx\rabc", true), ("abc\ryyy", true)],
        }
        test_crlf_end_before_cr {
            pattern: "(?Rm)abc$",
            memory: 773,
            min_tier: 0,
            inputs: [("abc\r", true), ("abc\r\n", true), ("abc\rxxx", true)],
        }
        test_crlf_start_after_lf_not_crlf_middle {
            pattern: "(?Rm)^x",
            memory: 707,
            min_tier: 0,
            inputs: [("\nx", true), ("\r\nx", true), ("\rx", true)],
        }
        test_crlf_mixed_terminators {
            pattern: "(?Rm)^abc$",
            memory: 806,
            min_tier: 0,
            inputs: [
                ("xxx\nabc\ryyy", true),
                ("xxx\rabc\nyyy", true),
                ("xxx\r\nabc\r\nyyy", true),
                ("xxx\nabc\r\nyyy", true),
                ("xxx\r\nabc\nyyy", true),
            ],
        }
        test_crlf_end_only {
            pattern: "(?Rm)$",
            memory: 674,
            min_tier: 0,
            inputs: [
                ("", true),
                ("a", true),
                ("\r\n", true),
                ("\r", true),
                ("\n", true),
            ],
        }
        test_crlf_start_only {
            pattern: "(?Rm)^",
            memory: 674,
            min_tier: 0,
            inputs: [
                ("", true),
                ("a", true),
                ("\r\n", true),
                ("\r", true),
                ("\n", true),
            ],
        }
        test_word_boundary_literal {
            pattern: r#"\bfoo\b"#,
            memory: 806,
            min_tier: 1,
            inputs: [
                ("foo", true),
                (" foo ", true),
                ("foo bar", true),
                ("bar foo", true),
                ("bar foo baz", true),
                ("(foo)", true),
                ("foobar", false),
                ("barfoo", false),
                ("xfooy", false),
                ("", false),
            ],
        }
        test_word_boundary_start_end_1 {
            pattern: r#"\bx"#,
            memory: 707,
            min_tier: 1,
            inputs: [("x", true), (" x", true), ("ax", false)],
        }
        test_word_boundary_start_end_2 {
            pattern: r#"x\b"#,
            memory: 707,
            min_tier: 1,
            inputs: [("x", true), ("x ", true), ("xa", false)],
        }
        test_word_boundary_quantifiers {
            pattern: r#"\b\w+\b"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("hello", true),
                ("hello world", true),
                ("  hello  ", true),
                ("", false),
                ("   ", false),
                ("a", true),
            ],
        }
        test_word_boundary_counter {
            pattern: r#"\b\w{3,5}\b"#,
            memory: 1194,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abcde", true),
                ("ab", false),
                ("abcdef", false),
                (" abc ", true),
                (" ab ", false),
                ("hello world", true),
            ],
        }
        test_non_word_boundary {
            pattern: r#"\Bfoo\B"#,
            memory: 806,
            min_tier: 1,
            inputs: [
                ("xfooy", true),
                ("afoobar", true),
                ("foo", false),
                (" foo ", false),
                ("xfoo", false),
                ("fooy", false),
            ],
        }
        test_word_boundary_digits_underscore_1 {
            pattern: r#"\b\d+\b"#,
            memory: 1029,
            min_tier: 1,
            inputs: [("123", true), (" 456 ", true), ("abc123def", false)],
        }
        test_word_boundary_digits_underscore_2 {
            pattern: r#"\b_test_\b"#,
            memory: 905,
            min_tier: 1,
            inputs: [("_test_", true), (" _test_ ", true), ("x_test_y", false)],
        }
        test_word_boundary_bare {
            pattern: r#"\b"#,
            memory: 674,
            min_tier: 1,
            inputs: [("a", true), (" ", false), ("", false)],
        }
        test_non_word_boundary_bare {
            pattern: r#"\B"#,
            memory: 674,
            min_tier: 1,
            inputs: [("", true), (" ", true), ("a", false)],
        }
        // Regression: `\B$` on word-char inputs.  At end-of-input after a
        // word char, `\B` should fail (prev=word, next=EOI=non-word →
        // boundary, so non-boundary assertion fails).
        test_word_boundary_b_neg_end_of_word {
            pattern: r#"\B$"#,
            memory: 707,
            min_tier: 1,
            inputs: [
                ("zzz", false),
                ("a", false),
                ("abc", false),
                ("z", false),
                ("", true),
                ("  ", true),
                (" ", true),
            ],
        }
        // Regression: `^\B$` can only match the empty string at a position
        // that is simultaneously start-of-input, end-of-input, and a
        // non-word-boundary.  Empty input: prev=SOI(non-word),
        // next=EOI(non-word) → same class → \B passes.  Non-empty inputs
        // cannot satisfy both ^ and $ at the same position.
        test_word_boundary_b_neg_anchored {
            pattern: r#"^\B$"#,
            memory: 740,
            min_tier: 1,
            inputs: [
                ("", true),
                ("\x00", false),
                ("a", false),
                (" ", false),
                ("\n", false),
            ],
        }
        // Regression: `^\b$` can only match the empty string at a position
        // that is both start/end-of-input and a word boundary.  At
        // SOI=EOI, prev=non-word, next=non-word → same class → \b fails.
        // So `^\b$` never matches any input.
        test_word_boundary_b_pos_anchored {
            pattern: r#"^\b$"#,
            memory: 740,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                (" ", false),
                ("ab", false),
                ("\x00", false),
            ],
        }
        test_word_boundary_mixed {
            pattern: r#"\bfoo\B"#,
            memory: 806,
            min_tier: 1,
            inputs: [
                ("foobar", true),
                (" foobar", true),
                ("foo", false),
                ("foo ", false),
                ("xfoobar", false),
            ],
        }
        test_word_boundary_alternation {
            pattern: r#"\b(cat|dog)\b"#,
            memory: 938,
            min_tier: 1,
            inputs: [
                ("cat", true),
                ("dog", true),
                ("the cat sat", true),
                ("hotdog", false),
                ("concatenate", false),
            ],
        }
        test_word_boundary_special_chars {
            pattern: r#"\btest\b"#,
            memory: 839,
            min_tier: 1,
            inputs: [
                ("\0test\0", true),
                (".test.", true),
                ("\ttest\n", true),
                ("/test/", true),
                ("hello test bye", true),
                ("test", true),
                ("testing", false),
            ],
        }
        // ---------------------------------------------------------------
        // Regression: \B at end-of-input in Tier 2/3 DFA transitions.
        //
        // The deferred assertion resolution (Phase 1) ran with
        // next=Some(byte) and set resolved_is_match_at_end.  This was
        // merged into the transition's is_match_at_end, but at actual
        // end-of-input next=None flips the word-boundary condition,
        // causing a false positive.  Fixed by discarding
        // resolved_is_match_at_end (same fix as Tier 1, commit 66eb9b7).
        // ---------------------------------------------------------------

        // Tier 2: \B$ after word chars.  \B at end-of-input after a word
        // char fails because prev=word, next=EOI(non-word) is a boundary.
        test_tier2_b_neg_eoi_counter {
            pattern: r#"^a{3,34}b{1,6}\B$"#,
            memory: 1203,
            min_tier: 2,
            inputs: [
                ("aaabb", false),
                ("aaabbbbb", false),
                ("aaa", false),
                ("", false),
            ],
        }
        // Tier 2: optional wrapper makes empty string valid but \B$
        // should still reject non-empty word-char inputs.
        test_tier2_b_neg_eoi_optional {
            pattern: r#"^(a{3,34}b{1,6}\B)?$"#,
            memory: 1236,
            min_tier: 2,
            inputs: [
                ("", true),
                ("aaabb", false),
                ("aaabbbbb", false),
                ("aaa", false),
            ],
        }
        // Tier 2: counted rep + \b\B (impossible assertion) at EOI.
        test_tier2_b_pos_b_neg_eoi {
            pattern: r#"^a?b{6,30}\b\B$"#,
            memory: 939,
            min_tier: 2,
            inputs: [
                ("abbbbbbb", false),
                ("bbbbbb", false),
                ("bbbbbbb", false),
                ("", false),
            ],
        }
        // Tier 3: variable-length body counter + \B$ at EOI.  The body
        // `ab?` is variable-length, making this Tier 3 eligible.  \B at
        // end-of-input after word chars should fail.
        test_tier3_b_neg_eoi_counter {
            pattern: r#"^(ab?){3,34}\B$"#,
            memory: 906,
            min_tier: 3,
            inputs: [
                ("aaa", false),
                ("abab", false),
                ("ababababab", false),
                ("", false),
            ],
        }
        // ---------------------------------------------------------------
        // Regression: deferred assertions on counter break paths.
        //
        // The with-break DFA state includes deferred assertions from ALL
        // CInc break paths, including counters that haven't reached their
        // minimum.  resolve_deferred_at_end on the with-break state would
        // evaluate those assertions at EOI, causing false positives.
        //
        // Fix: use no_break_current for counter-free deferred assertions,
        // and verified_deferred_asserts for break-path deferred assertions
        // from counters that actually broke with enough value.
        // ---------------------------------------------------------------

        // Two counters + trailing \b: min total = 6+7 = 13.
        // 12 chars: NO MATCH (too short).
        // 13 chars: MATCH (\b passes at word→end).
        test_tier3_multi_counter_deferred_break {
            pattern: r#"^.{6,39}.{7,31}\b$"#,
            memory: 1196,
            min_tier: 3,
            inputs: [
                ("aaaaaaaaaaaa", false),
                ("aaaaaaaaaaaaa", true),
                ("aaaaaaaaaaaaaaa", true),
            ],
        }
        // Same pattern with \B instead: \B fails at word→end.
        test_tier3_multi_counter_deferred_break_neg {
            pattern: r#"^.{6,39}.{7,31}\B$"#,
            memory: 1196,
            min_tier: 3,
            inputs: [
                ("aaaaaaaaaaaaa", false),
                ("aaaaaaaaaaaaaaa", false),
            ],
        }
        // ---------------------------------------------------------------
        // Bug 8: Tier 3 no_break DFA state is_match_at_end not checked
        // in finish().
        //
        // Pattern `(0{2,2}|1*)$` on "0": the 1* branch can match zero
        // times at EOI (1*$), but after processing "0" (which activates
        // the counter but doesn't reach min=2), Tier 3's finish() didn't
        // check is_match_at_end on the no_break DFA state.  The per-
        // transition counter_free_match_at_end only looks at byte-
        // consuming origins, missing epsilon paths (like 1* → $ → Match)
        // in the target state's start closure.  Fixed by checking
        // no_break_current.is_match_at_end in finish().
        // ---------------------------------------------------------------
        test_tier3_counter_alt_epsilon_mae {
            pattern: r#"(0{2,2}|1*)$"#,
            memory: 839,
            min_tier: 1,
            inputs: [
                ("0", true),
                ("00", true),
                ("1", true),
                ("10", true),
                ("x", true),
                ("", true),
                ("000", true),
            ],
        }

        // ---------------------------------------------------------------
        // Regression: adjacent \b\B assertions in resolve_deferred_at_end.
        //
        // resolve_deferred_at_end checked each deferred assertion
        // independently, using state_can_reach_match for the
        // downstream path.  But state_can_reach_match is static and
        // doesn't evaluate subsequent assertions.  When \b passes at
        // end-of-input but \B (its successor) fails, the static check
        // still reported reachability.  Fixed by walking assertions
        // dynamically in can_reach_match_at_end.
        // ---------------------------------------------------------------

        // \b\B is impossible: \b and \B at the same position is always
        // false.  On empty input, \b fails (SOI→EOI: same class=non-word).
        test_b_pos_b_neg_adjacent_simple {
            pattern: r#"^a?\b\B$"#,
            memory: 839,
            min_tier: 1,
            inputs: [
                ("", false),
                ("a", false),
                (" ", false),
            ],
        }
        // Optional wrapper: outer ? makes empty valid (the inner group
        // takes the zero-repetition path, bypassing \b\B entirely).
        test_b_pos_b_neg_adjacent_optional {
            pattern: r#"^(a?\b\B)?$"#,
            memory: 872,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", false),
                (" ", false),
                ("aa", false),
            ],
        }

        test_unanchored_counter_simple_1 {
            pattern: r#"\w{3,5}"#,
            memory: 1128,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abcde", true),
                ("ab", false),
                ("a b c", false),
                (" abc ", true),
                ("x y z", false),
                ("", false),
                ("a", false),
                ("abcdef", true),
                ("ab cde fg", true),
            ],
        }
        test_unanchored_counter_simple_2 {
            pattern: "a{3}",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("aaa", true),
                ("aa", false),
                ("xaaax", true),
                ("a a a", false),
                ("", false),
                ("aaaa", true),
                ("baaab", true),
            ],
        }
        test_unanchored_counter_simple_3 {
            pattern: "[0-9]{4}",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("1234", true),
                ("123", false),
                ("a1234b", true),
                ("1 2 3 4", false),
                ("12345", true),
                ("", false),
            ],
        }
        test_unanchored_counter_alternation_body {
            pattern: "(a|bc){2,4}",
            memory: 1235,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("abc", true),
                ("bca", true),
                ("bcbc", true),
                ("a", false),
                ("bc", false),
                ("", false),
                ("xaay", true),
                ("xbcay", true),
                ("a bc a", false),
                ("abcbca", true),
                ("abcbcbc", true),
            ],
        }
        test_unanchored_counter_multi_byte_body {
            pattern: "(ab){2,3}",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("abab", true),
                ("ababab", true),
                ("ab", false),
                ("", false),
                ("xababx", true),
                ("ab ab ab", false),
                ("aab", false),
                ("abb", false),
                ("abababab", true),
            ],
        }
        test_unanchored_counter_nested {
            pattern: "((a|b){1,2}){2,3}",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("aabb", true),
                ("ababab", true),
                ("a", false),
                ("", false),
                ("xaby", true),
                ("a b", false),
                ("abba", true),
            ],
        }
        test_unanchored_counter_min_zero {
            pattern: "a{0,3}",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("b", true),
                ("xaaax", true),
            ],
        }
        test_unanchored_counter_unbounded {
            pattern: "a{2,}",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aaa", true),
                ("a", false),
                ("", false),
                ("xaay", true),
                ("a a", false),
                ("baaaab", true),
            ],
        }
        test_unanchored_counter_wildcard_body {
            pattern: ".{3,5}",
            memory: 1128,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abcde", true),
                ("ab", false),
                ("a", false),
                ("", false),
                ("abcdef", true),
            ],
        }
        test_partial_anchor_start_counter_1 {
            pattern: "^a{2,3}",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aaa", true),
                ("a", false),
                ("aaax", true),
                ("", false),
                ("xaa", false),
            ],
        }
        test_partial_anchor_start_counter_2 {
            pattern: r#"^\d{2,4}"#,
            memory: 1128,
            min_tier: 1,
            inputs: [
                ("12", true),
                ("1234", true),
                ("1", false),
                ("12345", true),
                ("", false),
                ("a12", false),
            ],
        }
        test_partial_anchor_end_counter_1 {
            pattern: "a{2,3}$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aaa", true),
                ("a", false),
                ("xaa", true),
                ("", false),
                ("aax", false),
            ],
        }
        test_partial_anchor_end_counter_2 {
            pattern: r#"\d{2,4}$"#,
            memory: 1128,
            min_tier: 1,
            inputs: [
                ("12", true),
                ("1234", true),
                ("1", false),
                ("x1234", true),
                ("", false),
                ("12x", false),
            ],
        }
        test_unanchored_byte_class_1 {
            pattern: r#"\d+"#,
            memory: 963,
            min_tier: 1,
            inputs: [
                ("123", true),
                ("abc", false),
                ("a1b2c3", true),
                ("", false),
            ],
        }
        test_unanchored_byte_class_2 {
            pattern: r#"\w+"#,
            memory: 963,
            min_tier: 1,
            inputs: [
                ("hello", true),
                (" ", false),
                ("a b c", true),
                ("", false),
            ],
        }
        test_unanchored_byte_class_3 {
            pattern: "[a-c]+",
            memory: 963,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abcabc", true),
                ("xyz", false),
                ("xabcy", true),
                ("", false),
            ],
        }
        test_unanchored_byte_table_counter {
            pattern: "(ab|cd|ef){2,3}",
            memory: 4538,
            min_tier: 1,
            inputs: [
                ("abcd", true),
                ("abcdef", true),
                ("ab", false),
                ("", false),
                ("xabcdy", true),
                ("ab cd ef", false),
                ("ababab", true),
                ("efef", true),
                ("abefcd", true),
            ],
        }
        test_unanchored_question_mark {
            pattern: "a?b",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("b", true),
                ("xaby", true),
                ("xby", true),
                ("a", false),
                ("", false),
                ("aab", true),
            ],
        }
        test_counter_exact_one {
            pattern: "a{1,1}",
            memory: 674,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("", false),
                ("xax", true),
                ("aa", true),
                ("b", false),
            ],
        }
        test_multiline_alternation_counter {
            pattern: "(?m:^)(a|bc){2,3}(?m:$)",
            memory: 1136,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("abc", true),
                ("bcbc", true),
                ("a", false),
                ("", false),
                ("xxx\naa\nyyy", true),
                ("xxx\nabc\nyyy", true),
                ("xxx\na\nyyy", false),
            ],
        }
        test_empty_input_unanchored_counter {
            pattern: "a{0,3}",
            memory: 839,
            min_tier: 1,
            inputs: [("", true)],
        }
        test_unanchored_byte_class_counter {
            pattern: r#"\d{2,4}"#,
            memory: 1095,
            min_tier: 1,
            inputs: [
                ("12", true),
                ("1234", true),
                ("1", false),
                ("", false),
                ("a12b", true),
                ("a1b2c", false),
                ("12345", true),
                ("a1234b", true),
            ],
        }
        test_counter_body_with_assertion_1 {
            pattern: "(?m:^a){2,3}",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("a\na", false),
                ("a\na\na", false),
                ("a", false),
                ("", false),
                ("a\nb\na", false),
                ("ba\na", false),
                ("a\na\na\na", false),
            ],
        }
        test_counter_body_with_assertion_2 {
            pattern: "(?m:a$){2,3}",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("a\na", false),
                ("a\na\na", false),
                ("a", false),
                ("", false),
                ("a\nb\na", false),
                ("xa\nxa", false),
            ],
        }
        test_counter_body_with_assertion_3 {
            pattern: "(?m:^a$){2,3}",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("a\na", false),
                ("a\na\na", false),
                ("a", false),
                ("", false),
                ("a\nab\na", false),
                ("x\na\na\nx", false),
            ],
        }
        // Deferred assertion in L=1 counter body (promoted to tier 2)
        test_counter_body_word_boundary_end_1 {
            pattern: r"(\w\b){1,3}",
            memory: 1161,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("ab", true),
                ("a ", true),
                ("a b", true),
                ("abc", true),
                ("", false),
                (" ", false),
            ],
        }
        test_counter_body_word_boundary_end_2 {
            pattern: r"(\w\b){2,4}",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("a", false),
                ("ab", false),
                ("a b", false),
                ("a b c", false),
                ("", false),
            ],
        }
        test_counter_body_word_boundary_end_3 {
            pattern: r"(\w\b){1,2}",
            memory: 1062,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("ab", true),
                ("a ", true),
                ("a b", true),
                ("", false),
            ],
        }
        test_counter_body_dot_boundary {
            pattern: r"(.\b){2,4}",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("a b", true),
                ("a b c", true),
                ("abcd", false),
                ("ab", false),
                ("a", false),
                ("", false),
                ("a  b", true),
            ],
        }
        test_counter_body_boundary_at_start {
            pattern: r"(\ba){2,4}",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("aa", false),
                ("a a", false),
                ("a", false),
                ("", false),
            ],
        }
        // Bug 9: deferred \B inside counter body with L=1.  The deferred
        // assertion gates the path to CInc; counter_reset clears the counter
        // because the "interior" set is empty.  Seed injection re-populates
        // the counter.  The break path goes through $ → Match, which must
        // only fire at actual EOI (not mid-stream via match_at_end).
        test_counter_body_non_word_boundary_1 {
            pattern: r"^(.\B){1,2}$",
            memory: 1128,
            min_tier: 1,
            inputs: [
                (" ", true),    // 1 non-word: \B passes at EOI
                ("  ", true),   // 2 non-words: \B passes between them and at EOI
                ("ab", false),  // 2 word chars: \B passes between a/b, but fails at b/EOI
                ("a", false),   // 1 word char: \B fails at a/EOI (word→non-word)
                ("", false),
            ],
        }
        test_counter_body_non_word_boundary_2 {
            pattern: r"^(.\B){2,3}$",
            memory: 1194,
            min_tier: 1,
            inputs: [
                ("  ", true),
                ("   ", true),
                ("ab", false),
                ("abc", false),
                (" ", false),   // too short (min=2)
                ("", false),
            ],
        }
        test_counter_body_non_word_boundary_3 {
            pattern: r"^(.\B){1,12}$",
            memory: 1096,
            min_tier: 2,
            inputs: [
                (" ", true),
                ("    ", true),
                ("ab", false),
                ("", false),
            ],
        }
        // Multi-counter with deferred assertion in body → NFA simulator
        // (break path through deferred assert can't be followed in probe closure)
        test_multi_counter_deferred_body {
            pattern: r"(a\B){2,3}(b\B){2,3}",
            memory: 1103,
            min_tier: 1,
            inputs: [
                ("aabbc", true),
                ("aabb", false),
                ("aaabbbx", true),
                ("aabbx", true),
                ("abx", false),
                ("", false),
            ],
        }
        // Bug 10: deferred assertion BEFORE counter body with L=1.
        // When \b resolves via Phase 1, the resolved path reaches CI → body
        // → CInc on the same transition.  The counter must be pre-seeded
        // before increment so the counter is non-empty when CInc fires.
        test_deferred_assert_before_counter_1 {
            pattern: r"\b.{1,2}",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("ab", true),
                ("abc", true),   // unanchored, 3 chars still match (prefix)
                (" ", false),    // \b fails: non-word→non-word
                ("", false),
            ],
        }
        test_deferred_assert_before_counter_2 {
            pattern: r"^(\b.{1,34}?)?$",
            memory: 1129,
            min_tier: 2,
            inputs: [
                ("a", true),
                ("", true),     // optional group matches empty
                ("abcdef", true),
                (" ", false),   // \b fails: non-word start
            ],
        }
        test_deferred_assert_before_counter_3 {
            pattern: r"^(\b((.{1,34}|a?))?)?$",
            memory: 1261,
            min_tier: 2,
            inputs: [
                ("x", true),
                ("a", true),
                ("", true),
            ],
        }
        test_deferred_assert_before_counter_4 {
            pattern: r"^(\B.{1,34}?)?$",
            memory: 1129,
            min_tier: 2,
            inputs: [
                ("\x00", true), // \B: non-word→non-word
                ("", true),     // optional matches empty
                ("a", false),   // \B fails: non-word→word
            ],
        }
        // Bug 11 (fuzz): {0,0} repetition nested in another repetition
        // caused a fragment-stack underflow panic during compilation.
        // The inner {0,0} produces Empty (no postfix output), but the
        // outer ? / * pushed RepeatZeroOne expecting a fragment.
        test_zero_zero_rep_nested {
            pattern: r"^(.{0,0}?)?((a?a?)?(a?a?)?)?$",
            memory: 1070,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaaa", true),
                ("b", false),
            ],
        }
        // Bug 12 (fuzz): Tier 3 false positive on contradictory adjacent
        // assertions \B\b after counter break.  break_closure() recorded
        // BOTH assertions as independent entries in deferred_asserts, so
        // \b was evaluated without \B gating it.  Fix: only record the
        // first (entry-point) assertion; deeper ones are handled by
        // can_reach_match_at_end().
        test_contradictory_adjacent_asserts {
            pattern: r"^.{10,31}\B\b$",
            memory: 1129,
            min_tier: 2,
            inputs: [
                ("yyyyyyyyyy0aaaaaaaaaaaaaaaaa", false), // \B\b always fails
                ("aaaaaaaaaa", false),
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false), // 31 chars
                ("", false),
            ],
        }
        // Bug 14: Tier 2 false positive — Phase 1 deferred resolution
        // created resolved_seeds on non-body-byte transitions.  Pattern
        // `\B {7,12}`: when processing `!` (non-space), Phase 1 resolved
        // \B (non-word→non-word) and walked to CI, creating a seed for
        // counter 0.  But `!` doesn't match the body ` `, so the seed
        // was spurious.  Merged into seed_list with value 1, causing an
        // off-by-one in the counter.
        test_tier2_resolved_seed_non_body_byte {
            pattern: r"\B {7,12}",
            memory: 1235,
            min_tier: 1,
            inputs: [
                (" !      ", false),   // 1 space + ! + 6 spaces: need 7 contiguous from \B
                (" !       ", true),   // 1 space + ! + 7 spaces: 7 from pos 2
                ("       ", true),     // 7 spaces: \B at start (non-word→non-word)
                ("        ", true),    // 8 spaces
                (" !     ", false),    // 1 space + ! + 5 spaces: too few
                ("a       ", false),   // word char then spaces: \B fails at start
            ],
        }
        // Deferred assertion patterns (migrated from standalone tests)
        test_non_word_boundary_inside_word {
            pattern: r#"\Boo\B"#,
            memory: 773,
            min_tier: 1,
            inputs: [
                ("foobar", true),
                ("oo", false),
                (" oo ", false),
                ("xoox", true),
            ],
        }
        test_endlf_before_newline {
            pattern: r#"(?m)foo$"#,
            memory: 773,
            min_tier: 1,
            inputs: [
                ("foo", true),
                ("foo\nbar", true),
                ("bar\nfoo", true),
                ("bar\nfoo\nbaz", true),
                ("foobar", false),
                ("barfoo\n", true),
            ],
        }
        test_endlf_at_eof {
            pattern: r#"(?m)bar$"#,
            memory: 773,
            min_tier: 1,
            inputs: [
                ("bar", true),
                ("foobar", true),
                ("foo\nbar", true),
                ("bar\nfoo", true),
            ],
        }
        test_word_boundary_with_endlf {
            pattern: r#"(?m)\bfoo\b$"#,
            memory: 839,
            min_tier: 1,
            inputs: [
                ("foo", true),
                ("foo\nbar", true),
                ("bar\nfoo", true),
                ("bar\nfoo\nbaz", true),
                ("barfoo", false),
                ("foobar", false),
            ],
        }
        test_word_boundary_sql_keywords {
            pattern: r#"\b(?:select|insert|update|delete)\b"#,
            memory: 3646,
            min_tier: 1,
            inputs: [
                ("select", true),
                ("run select now", true),
                ("selected", false),
                ("preselect", false),
                ("delete from", true),
                ("undelete", false),
            ],
        }
        // Word-start and word-end boundary assertions
        test_word_start_basic {
            pattern: r#"\b{start}\w+\b{end}"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("hello", true),
                ("hello world", true),
                ("  hello  ", true),
                ("", false),
                ("   ", false),
                ("a", true),
            ],
        }
        test_word_start_only {
            pattern: r#"\b{start}foo"#,
            memory: 773,
            min_tier: 1,
            inputs: [
                ("foo", true),
                ("foobar", true),
                (" foo", true),
                ("xfoo", false),
                ("", false),
            ],
        }
        test_word_end_only {
            pattern: r#"foo\b{end}"#,
            memory: 773,
            min_tier: 1,
            inputs: [
                ("foo", true),
                ("barfoo", true),
                ("foo ", true),
                ("foobar", false),
                ("", false),
            ],
        }
        test_word_start_alternation {
            pattern: r#"\b{start}(?:cat|dog)\b{end}"#,
            memory: 938,
            min_tier: 1,
            inputs: [
                ("cat", true),
                ("dog", true),
                ("the cat sat", true),
                ("hotdog", false),
                ("concatenate", false),
            ],
        }
        test_word_start_with_counter {
            pattern: r#"\b{start}\w{3,5}\b{end}"#,
            memory: 1194,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abcde", true),
                ("ab", false),
                ("abcdef", false),
                (" abc ", true),
                (" ab ", false),
                ("hello world", true),
            ],
        }
        test_word_end_at_eof {
            pattern: r#"test\b{end}"#,
            memory: 806,
            min_tier: 1,
            inputs: [
                ("test", true),
                ("a test", true),
                ("testing", false),
                ("test!", true),
            ],
        }
        test_word_start_at_start {
            pattern: r#"\b{start}test"#,
            memory: 806,
            min_tier: 1,
            inputs: [
                ("test", true),
                ("test case", true),
                ("attest", false),
                ("!test", true),
            ],
        }
        test_multiline_abc {
            pattern: "(?m:^)abc(?m:$)",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("xxx\nabc\nyyy", true),
                ("abd", false),
                ("xabc", false),
                ("abcx", false),
                ("xx\nabcx", false),
            ],
        }
        test_unanchored_counter_a35 {
            pattern: "a{3,5}",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", true),
                ("aa", false),
                ("a", false),
                ("", false),
                ("xaaax", true),
            ],
        }
        // Nested counter patterns (migrated from survey_nested_counter_bugs)
        test_nested_a2_x3 {
            pattern: "(a{2}){3}",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("aaaaa", false),
                ("aaaaaa", true),
                ("aaaaaaa", true),
            ],
        }
        test_nested_a3_x2 {
            pattern: "(a{3}){2}",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("aaaa", false),
                ("aaaaa", false),
                ("aaaaaa", true),
                ("aaaaaaa", true),
            ],
        }
        test_nested_ab_x2 {
            pattern: "(ab){2}",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("aba", false),
                ("abab", true),
                ("xababx", true),
                ("ab", false),
            ],
        }
        test_nested_ab_x3 {
            pattern: "(ab){3}",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("ababa", false),
                ("ababab", true),
                ("xabababx", true),
                ("abab", false),
            ],
        }
        test_nested_a23_x2 {
            pattern: "(a{2,3}){2}",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("aaa", false),
                ("aaaa", true),
                ("aaaaa", true),
                ("aaaaaa", true),
                ("aaaaaaa", true),
            ],
        }
        test_nested_a2_x2_x2 {
            pattern: "((a{2}){2}){2}",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("aaaaaaa", false),
                ("aaaaaaaa", true),
                ("aaaaaaaaa", true),
            ],
        }
        test_nested_a2_x23 {
            pattern: "(a{2}){2,3}",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("aaa", false),
                ("aaaa", true),
                ("aaaaa", true),
                ("aaaaaa", true),
                ("aaaaaaa", true),
            ],
        }
        test_nested_dotdot_x2 {
            pattern: "(..){2}",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("aaa", false),
                ("aaaa", true),
                ("abcde", true),
                ("ab", false),
            ],
        }
        test_nested_abc_x2 {
            pattern: "(abc){2}",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("abcab", false),
                ("abcabc", true),
                ("xabcabcx", true),
                ("abc", false),
            ],
        }
        test_non_nested_a4 {
            pattern: "a{4}",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("aaa", false),
                ("aaaa", true),
                ("aaaaa", true),
                ("aa", false),
            ],
        }
        test_non_nested_a2 {
            pattern: "a{2}",
            memory: 707,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("a", false),
                ("xaax", true),
                ("", false),
            ],
        }
        test_nested_a12_x2 {
            pattern: "^(a{1,2}){2}$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("a", false),
                ("aaaaa", false),
            ],
        }
        test_byte_table_cross_validate_unanchored {
            pattern: "(ab|cd|ef)",
            memory: 1929,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("cd", true),
                ("ef", true),
                ("xxab", true),
                ("cdyy", true),
                ("xxefyy", true),
                ("ac", false),
                ("xyz", false),
                ("", false),
                ("a", false),
                ("f", false),
            ],
        }
        test_byte_table_with_bounded_repetition_cross_validate {
            pattern: "^(ab|cd|ef){2,4}$",
            memory: 2062,
            min_tier: 2,
            inputs: [
                ("abcd", true),
                ("efab", true),
                ("cdef", true),
                ("ababab", true),
                ("abcdef", true),
                ("abcdefab", true),
                ("ab", false),
                ("ef", false),
                ("", false),
                ("abcdefabcd", false),
                ("abcdefef", true),
            ],
        }
        test_byte_table_counted_with_suffix {
            pattern: "^(ab|cd|ef){1,3}x$",
            memory: 6718,
            min_tier: 1,
            inputs: [
                ("abx", true),
                ("cdx", true),
                ("efx", true),
                ("abcdx", true),
                ("efabx", true),
                ("abcdefx", true),
                ("x", false),
                ("ab", false),
                ("abcdefabx", false),
                ("", false),
                ("abcdefxx", false),
            ],
        }
        test_byte_table_mixed_length_unanchored {
            pattern: "(a|bc|def)",
            memory: 1929,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("bc", true),
                ("def", true),
                ("xxa", true),
                ("xxbcyy", true),
                ("xxdefyy", true),
                ("xx", false),
                ("", false),
                ("bd", false),
            ],
        }
        test_byte_table_with_prefix_and_suffix {
            pattern: "^xx(ab|cd|ef)yy$",
            memory: 2127,
            min_tier: 1,
            inputs: [
                ("xxabyy", true),
                ("xxcdyy", true),
                ("xxefyy", true),
                ("xxabyyz", false),
                ("xabyy", false),
                ("xxaby", false),
                ("xxyy", false),
                ("abyy", false),
                ("", false),
            ],
        }
        test_byte_table_after_wildcard {
            pattern: "^..(ab|cd|ef)$",
            memory: 2317,
            min_tier: 1,
            inputs: [
                ("xxab", true),
                ("zzcd", true),
                ("qqef", true),
                ("\0\x7fab", true),
                ("xab", false),
                ("xxxab", false),
                ("xxac", false),
                ("", false),
            ],
        }
        test_byte_table_followed_by_wildcard {
            pattern: "^(ab|cd|ef).*x$",
            memory: 2350,
            min_tier: 1,
            inputs: [
                ("abx", true),
                ("cdx", true),
                ("efx", true),
                ("ab123x", true),
                ("cdxxxxxx", true),
                ("abX", false),
                ("cd", false),
                ("ef123", false),
                ("x", false),
                ("", false),
            ],
        }
        test_byte_table_sandwiched_by_wildcards {
            pattern: "^.*(ab|cd|ef).*$",
            memory: 2383,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("cd", true),
                ("ef", true),
                ("xxxab", true),
                ("cdyyy", true),
                ("xxxefyyy", true),
                ("xxabyycdzzef", true),
                ("", false),
                ("x", false),
                ("ac", false),
            ],
        }
        test_step_fused_multi_counter_same_byte {
            pattern: "a{2}.*a{3}",
            memory: 1128,
            min_tier: 1,
            inputs: [
                ("aaaaa", true),
                ("aabaa", false),
                ("aaxaaa", true),
                ("aa aaa", true),
                ("a", false),
                ("aaaa", false),
                ("aaa", false),
                ("", false),
            ],
        }
        test_step_fused_alternation_overlap {
            pattern: "(a|a){2,3}",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aaa", true),
                ("a", false),
                ("aaaa", true),
                ("", false),
                ("xaax", true),
                ("xaaax", true),
            ],
        }
        test_step_fused_byteclass_and_literal {
            pattern: "[a-z]{2}a{2}",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("xyaa", true),
                ("aaaa", true),
                ("aa", false),
                ("aaa", false),
                ("abaa", true),
                ("a", false),
                ("", false),
                ("xyzaa", true),
                (" aaaa ", true),
            ],
        }
        test_step_fused_reseed_with_counter {
            pattern: "(a{2}){2}",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("aaaa", true),
                ("aaa", false),
                ("aaaaa", true),
                ("aa", false),
                ("a", false),
                ("", false),
                ("xaaaax", true),
                ("aa aa", false),
                ("aaxaa", false),
            ],
        }
        test_ci_literal_single {
            pattern: "^(?i)a$",
            memory: 740,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("A", true),
                ("", false),
                ("b", false),
                ("B", false),
                ("aa", false),
                ("AA", false),
                ("aA", false),
                ("1", false),
            ],
        }
        test_ci_literal_multi {
            pattern: "^(?i)abc$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("ABC", true),
                ("Abc", true),
                ("aBc", true),
                ("abC", true),
                ("ABc", true),
                ("AbC", true),
                ("aBC", true),
                ("", false),
                ("ab", false),
                ("AB", false),
                ("abcd", false),
                ("ABCD", false),
                ("xabc", false),
                ("abd", false),
                ("cba", false),
                ("CBA", false),
            ],
        }
        test_ci_one_plus {
            pattern: "^(?i)a+$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("A", true),
                ("aa", true),
                ("AA", true),
                ("aAaA", true),
                ("AaAa", true),
                ("aaAAAAaa", true),
                ("", false),
                ("b", false),
                ("ab", false),
                ("ba", false),
                ("aab", false),
                ("bAA", false),
            ],
        }
        test_ci_star {
            pattern: "^(?i)a*$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("A", true),
                ("aA", true),
                ("AaA", true),
                ("b", false),
                ("ab", false),
                ("Ba", false),
            ],
        }
        test_ci_question {
            pattern: "^(?i)a?$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("A", true),
                ("b", false),
                ("aa", false),
                ("AA", false),
                ("aA", false),
            ],
        }
        test_ci_counted {
            pattern: "^(?i)a{2,4}$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("AA", true),
                ("aA", true),
                ("Aa", true),
                ("aaa", true),
                ("AAA", true),
                ("aAa", true),
                ("aaaa", true),
                ("AAAA", true),
                ("aAaA", true),
                ("AaAa", true),
                ("", false),
                ("a", false),
                ("A", false),
                ("aaaaa", false),
                ("AAAAA", false),
                ("b", false),
                ("bb", false),
                ("aab", false),
                ("baa", false),
            ],
        }
        test_ci_counted_min_zero {
            pattern: "^(?i)a{0,3}$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("A", true),
                ("aA", true),
                ("AaA", true),
                ("aAa", true),
                ("aaaa", false),
                ("AAAA", false),
                ("b", false),
                ("aab", false),
            ],
        }
        test_ci_alternation {
            pattern: "^(?i)(a|bc)$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("A", true),
                ("bc", true),
                ("BC", true),
                ("Bc", true),
                ("bC", true),
                ("", false),
                ("b", false),
                ("B", false),
                ("c", false),
                ("C", false),
                ("ab", false),
                ("abc", false),
                ("cb", false),
                ("CB", false),
            ],
        }
        test_ci_alternation_three_way {
            pattern: "^(?i)(abc|def|ghi)$",
            memory: 1070,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("ABC", true),
                ("AbC", true),
                ("def", true),
                ("DEF", true),
                ("DeF", true),
                ("ghi", true),
                ("GHI", true),
                ("gHi", true),
                ("", false),
                ("ab", false),
                ("abcd", false),
                ("abcdef", false),
                ("xyz", false),
                ("aef", false),
                ("abi", false),
            ],
        }
        test_ci_mixed_letter_nonletter {
            pattern: "^(?i)hello world$",
            memory: 1070,
            min_tier: 1,
            inputs: [
                ("hello world", true),
                ("HELLO WORLD", true),
                ("Hello World", true),
                ("hElLo WoRlD", true),
                ("", false),
                ("hello", false),
                ("helloworld", false),
                ("hello  world", false),
                ("hello world!", false),
                ("xhello world", false),
            ],
        }
        test_ci_with_digits {
            pattern: "^(?i)a1b2c$",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("a1b2c", true),
                ("A1B2C", true),
                ("A1b2C", true),
                ("a1B2c", true),
                ("a2b2c", false),
                ("a1b1c", false),
                ("a1b2", false),
                ("1b2c", false),
                ("", false),
            ],
        }
        test_ci_byte_class {
            pattern: "^(?i)[abc]$",
            memory: 996,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("A", true),
                ("b", true),
                ("B", true),
                ("c", true),
                ("C", true),
                ("", false),
                ("d", false),
                ("D", false),
                ("z", false),
                ("Z", false),
                ("ab", false),
                ("1", false),
            ],
        }
        test_ci_byte_class_full_alpha {
            pattern: "^(?i)[a-z]$",
            memory: 996,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("A", true),
                ("m", true),
                ("M", true),
                ("z", true),
                ("Z", true),
                ("", false),
                ("0", false),
                ("9", false),
                ("!", false),
                ("ab", false),
            ],
        }
        test_ci_byte_class_alpha_digit {
            pattern: "^(?i)[a-z0-9]+$",
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("abc123", true),
                ("ABC123", true),
                ("AbC789", true),
                ("0", true),
                ("Z", true),
                ("", false),
                ("!", false),
                ("abc!", false),
                (" ", false),
            ],
        }
        test_ci_word_boundary {
            pattern: r"(?i)\bfoo\b",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("foo", true),
                ("FOO", true),
                ("Foo", true),
                ("fOo", true),
                (" foo ", true),
                (" FOO ", true),
                ("!foo!", true),
                ("x foo y", true),
                ("x FOO y", true),
                ("foobar", false),
                ("FOOBAR", false),
                ("barfoo1", false),
                ("afoo", false),
                ("foob", false),
                ("", false),
                ("bar", false),
                ("BAR", false),
                ("fo", false),
                ("oo", false),
            ],
        }
        test_ci_unanchored {
            pattern: "(?i)select",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("SELECT", true),
                ("select", true),
                ("Select", true),
                ("sElEcT", true),
                ("xxx SELECT yyy", true),
                ("beforeSELECTafter", true),
                ("123select456", true),
                ("", false),
                ("selec", false),
                ("elect", false),
                ("SELCT", false),
                ("slect", false),
            ],
        }
        test_ci_counted_group {
            pattern: "^(?i)(ab){2,3}$",
            memory: 938,
            min_tier: 1,
            inputs: [
                ("abab", true),
                ("ABAB", true),
                ("AbAb", true),
                ("aBaB", true),
                ("ababab", true),
                ("ABABAB", true),
                ("AbAbAb", true),
                ("aBAbab", true),
                ("ab", false),
                ("AB", false),
                ("abababab", false),
                ("ABABABAB", false),
                ("", false),
                ("ba", false),
                ("aabb", false),
            ],
        }
        test_ci_catenation_one_plus {
            pattern: "^(?i)a+b+c+$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("ABC", true),
                ("aaBBcc", true),
                ("AaaBbCcc", true),
                ("aaaBBBccc", true),
                ("", false),
                ("a", false),
                ("ab", false),
                ("bc", false),
                ("ac", false),
                ("cba", false),
                ("CBA", false),
                ("abca", false),
                ("aabbc1", false),
            ],
        }
        test_ci_wildcard {
            pattern: "^(?i).*foo.*$",
            memory: 1194,
            min_tier: 1,
            inputs: [
                ("foo", true),
                ("FOO", true),
                ("Foo", true),
                ("xxxfooyyy", true),
                ("XXXFOOYYY", true),
                ("123Foo456", true),
                ("foobar", true),
                ("barFOO", true),
                ("", false),
                ("fo", false),
                ("oo", false),
                ("bar", false),
                ("fxoo", false),
            ],
        }
        test_ci_digit_then_hex {
            pattern: r#"^(?i)\d+[a-f]+$"#,
            memory: 1351,
            min_tier: 1,
            inputs: [
                ("1a", true),
                ("1A", true),
                ("99ff", true),
                ("99FF", true),
                ("123abcDEF", true),
                ("0fF", true),
                ("", false),
                ("a1", false),
                ("123", false),
                ("abc", false),
                ("12g", false),
                ("12G", false),
            ],
        }
        test_ci_word_class {
            pattern: r#"^(?i)\w+$"#,
            memory: 1029,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("ABC", true),
                ("Hello123", true),
                ("_underscore", true),
                ("MiXeD_CaSe_123", true),
                ("", false),
                (" ", false),
                ("hello world", false),
                ("!", false),
            ],
        }
        test_ci_waf_sql_keywords {
            pattern: r"(?i)\b(?:select|insert|update|delete)\b",
            memory: 1598,
            min_tier: 1,
            inputs: [
                ("SELECT", true),
                ("select", true),
                ("Select", true),
                ("INSERT", true),
                ("insert", true),
                ("Insert", true),
                ("UPDATE", true),
                ("update", true),
                ("UpDaTe", true),
                ("DELETE", true),
                ("delete", true),
                ("DeLeTe", true),
                ("please SELECT * from", true),
                ("do INSERT into", true),
                ("run UPDATE set", true),
                ("run delete from t", true),
                ("selected", false),
                ("SELECTED", false),
                ("inserts", false),
                ("updated", false),
                ("deletes", false),
                ("preselect", false),
                ("", false),
                ("hello world", false),
                ("CREATE TABLE", false),
            ],
        }
        test_ci_function_call {
            pattern: r#"(?i)\bfoo\("#,
            memory: 806,
            min_tier: 1,
            inputs: [
                ("foo(", true),
                ("FOO(", true),
                ("Foo(", true),
                ("x foo(", true),
                ("x FOO( y", true),
                ("!Foo(1)", true),
                ("foo", false),
                ("FOO", false),
                ("barfoo(", false),
                ("afoo(", false),
                ("", false),
                ("bar(", false),
            ],
        }
        test_ci_counted_alternation {
            pattern: "^(?i)(a|b){1,2}$",
            memory: 1062,
            min_tier: 1,
            inputs: [
                ("a", true),
                ("A", true),
                ("b", true),
                ("B", true),
                ("aa", true),
                ("AA", true),
                ("ab", true),
                ("AB", true),
                ("aB", true),
                ("Ab", true),
                ("ba", true),
                ("BA", true),
                ("bA", true),
                ("Ba", true),
                ("bb", true),
                ("BB", true),
                ("", false),
                ("aab", false),
                ("AAB", false),
                ("c", false),
                ("C", false),
            ],
        }
        test_ci_dot_in_middle {
            pattern: "^(?i)x.y$",
            memory: 1062,
            min_tier: 1,
            inputs: [
                ("xay", true),
                ("XAY", true),
                ("xAy", true),
                ("X1Y", true),
                ("x y", true),
                ("X!Y", true),
                ("", false),
                ("xy", false),
                ("XY", false),
                ("xaby", false),
                ("axy", false),
            ],
        }
        test_ci_negated_class {
            pattern: "^(?i)[^a-z]$",
            memory: 996,
            min_tier: 1,
            inputs: [
                ("0", true),
                ("9", true),
                ("!", true),
                (" ", true),
                ("@", true),
                ("a", false),
                ("A", false),
                ("z", false),
                ("Z", false),
                ("m", false),
                ("M", false),
                ("", false),
                ("12", false),
            ],
        }
        test_ci_counted_wildcard {
            pattern: "^(?i).{2,4}$",
            memory: 1161,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("AB", true),
                ("12", true),
                ("abc", true),
                ("abcd", true),
                ("!@#", true),
                ("", false),
                ("a", false),
                ("abcde", false),
            ],
        }
        test_ci_repeated_alternation {
            pattern: "^(?i)(foo|bar)+$",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("foo", true),
                ("FOO", true),
                ("bar", true),
                ("BAR", true),
                ("foobar", true),
                ("FOOBAR", true),
                ("FooBar", true),
                ("barfoo", true),
                ("BARFOO", true),
                ("foofoofoo", true),
                ("FOOBARFOO", true),
                ("barBARbar", true),
                ("", false),
                ("fo", false),
                ("ba", false),
                ("baz", false),
                ("foobaz", false),
                ("foobarx", false),
            ],
        }
        test_ci_non_word_class {
            pattern: r#"^(?i)a\Wb$"#,
            memory: 1062,
            min_tier: 1,
            inputs: [
                ("a b", true),
                ("A B", true),
                ("a!b", true),
                ("A!B", true),
                ("a.B", true),
                ("A-b", true),
                ("aab", false),
                ("a1b", false),
                ("a_b", false),
                ("ab", false),
                ("", false),
                ("c d", false),
            ],
        }
        test_partial_ci_middle {
            pattern: "^a(?i:b)c$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("aBc", true),
                ("Abc", false),
                ("abC", false),
                ("ABC", false),
                ("aBC", false),
                ("AbC", false),
                ("", false),
                ("ac", false),
                ("axc", false),
                ("abbc", false),
                ("abcx", false),
            ],
        }
        test_partial_ci_prefix {
            pattern: "^(?i:a)b$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("Ab", true),
                ("aB", false),
                ("AB", false),
                ("", false),
                ("b", false),
                ("cb", false),
                ("abc", false),
            ],
        }
        test_partial_ci_suffix {
            pattern: "^a(?i:b)$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("aB", true),
                ("Ab", false),
                ("AB", false),
                ("", false),
                ("a", false),
                ("ac", false),
            ],
        }
        test_partial_ci_toggle_on_off {
            pattern: "^(?i)a(?-i)b$",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("Ab", true),
                ("aB", false),
                ("AB", false),
                ("", false),
                ("a", false),
                ("b", false),
                ("abc", false),
            ],
        }
        test_partial_ci_toggle_off_then_on {
            pattern: "^a(?-i)b(?i)c$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("abC", true),
                ("Abc", false),
                ("aBc", false),
                ("ABC", false),
                ("ABc", false),
                ("", false),
                ("ab", false),
                ("abx", false),
            ],
        }
        test_partial_ci_quantified_group {
            pattern: "^a(?i:b)+c$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("aBc", true),
                ("abbc", true),
                ("aBBc", true),
                ("aBbBc", true),
                ("abBbc", true),
                ("Abc", false),
                ("abC", false),
                ("ABC", false),
                ("ac", false),
                ("axc", false),
                ("", false),
            ],
        }
        test_partial_ci_star_group {
            pattern: "^a(?i:b)*c$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("ac", true),
                ("abc", true),
                ("aBc", true),
                ("aBBBc", true),
                ("Ac", false),
                ("aC", false),
                ("ABc", false),
                ("", false),
                ("a", false),
                ("axc", false),
            ],
        }
        test_partial_ci_counted_group {
            pattern: "^a(?i:b){2,4}c$",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("abbc", true),
                ("aBBc", true),
                ("aBbc", true),
                ("abbbc", true),
                ("aBBBc", true),
                ("abBBc", true),
                ("abbbbc", true),
                ("aBBBBc", true),
                ("abc", false),
                ("aBc", false),
                ("abbbbbc", false),
                ("aBBBBBc", false),
                ("ABBc", false),
                ("abbC", false),
                ("axxc", false),
                ("", false),
            ],
        }
        test_partial_ci_multi_letter_group {
            pattern: "^a(?i:bc)d$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("abcd", true),
                ("aBCd", true),
                ("aBcd", true),
                ("abCd", true),
                ("Abcd", false),
                ("abcD", false),
                ("ABCD", false),
                ("", false),
                ("abc", false),
                ("abxd", false),
                ("axcd", false),
            ],
        }
        test_partial_ci_adjacent_groups {
            pattern: "^(?i:a)(?i:b)c$",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("Abc", true),
                ("aBc", true),
                ("ABc", true),
                ("abC", false),
                ("ABC", false),
                ("AbC", false),
                ("", false),
                ("ab", false),
                ("xbc", false),
            ],
        }
        test_partial_ci_alternation_then_literal {
            pattern: "^(?i:a|bc)d$",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("ad", true),
                ("Ad", true),
                ("bcd", true),
                ("BCd", true),
                ("Bcd", true),
                ("bCd", true),
                ("aD", false),
                ("AD", false),
                ("bcD", false),
                ("BCD", false),
                ("", false),
                ("d", false),
                ("bd", false),
                ("cd", false),
            ],
        }
        test_partial_ci_word_in_middle {
            pattern: "^x(?i:foo)y$",
            memory: 872,
            min_tier: 1,
            inputs: [
                ("xfooy", true),
                ("xFOOy", true),
                ("xFooy", true),
                ("xfOoy", true),
                ("xfoOy", true),
                ("Xfooy", false),
                ("xfooY", false),
                ("XFOOY", false),
                ("xbary", false),
                ("xfoy", false),
                ("", false),
            ],
        }
        test_partial_ci_three_segments {
            pattern: "^(?i)abc(?-i)def(?i)ghi$",
            memory: 1004,
            min_tier: 1,
            inputs: [
                ("abcdefghi", true),
                ("ABCdefGHI", true),
                ("AbcdefGhi", true),
                ("aBcdefgHi", true),
                ("ABCdefghi", true),
                ("abcdefGHI", true),
                ("abcDEFghi", false),
                ("ABCDEFghi", false),
                ("abcDefghi", false),
                ("abcdEfghi", false),
                ("abcdeF ghi", false),
                ("ABCDEFGHI", false),
                ("", false),
                ("abcdef", false),
                ("defghi", false),
            ],
        }
        test_partial_ci_waf_keyword_prefix {
            pattern: r#"(?i:select)\s+\w+"#,
            memory: 1483,
            min_tier: 1,
            inputs: [
                ("select foo", true),
                ("SELECT foo", true),
                ("Select foo", true),
                ("sElEcT bar", true),
                ("xxx SELECT users yyy", true),
                ("SELECT  a", true),
                ("selectfoo", false),
                ("SELECT", false),
                ("SELECT ", false),
                ("", false),
                ("insert foo", false),
            ],
        }
        test_partial_ci_optional_group {
            pattern: "^a(?i:b)?c$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("ac", true),
                ("abc", true),
                ("aBc", true),
                ("Ac", false),
                ("aC", false),
                ("Abc", false),
                ("abC", false),
                ("axc", false),
                ("abbc", false),
                ("", false),
            ],
        }
        test_partial_ci_unanchored_key_value {
            pattern: "x(?i:key)=",
            memory: 806,
            min_tier: 1,
            inputs: [
                ("xkey=val", true),
                ("xKEY=val", true),
                ("xKey=123", true),
                ("pre xkEy=v post", true),
                ("Xkey=val", false),
                ("xkey:val", false),
                ("", false),
                ("xkey", false),
                ("ykey=val", false),
            ],
        }
        test_partial_ci_counted_group_then_literal {
            pattern: "^(?i:ab){2,3}c$",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("ababc", true),
                ("ABABc", true),
                ("AbAbc", true),
                ("aBaBc", true),
                ("abababc", true),
                ("ABABABc", true),
                ("AbAbAbc", true),
                ("ababC", false),
                ("ABABC", false),
                ("abc", false),
                ("ababababc", false),
                ("", false),
                ("xyzc", false),
            ],
        }
        test_partial_ci_class_in_group {
            pattern: "^a(?i:[b-d])e$",
            memory: 1062,
            min_tier: 1,
            inputs: [
                ("abe", true),
                ("aBe", true),
                ("ace", true),
                ("aCe", true),
                ("ade", true),
                ("aDe", true),
                ("Abe", false),
                ("abE", false),
                ("ABE", false),
                ("aee", false),
                ("axe", false),
                ("aae", false),
                ("", false),
            ],
        }
        test_partial_ci_four_toggles {
            pattern: "^(?i)a(?-i)b(?i)c(?-i)d$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("abcd", true),
                ("AbCd", true),
                ("Abcd", true),
                ("abCd", true),
                ("aBcd", false),
                ("abcD", false),
                ("aBcD", false),
                ("ABCD", false),
                ("", false),
                ("abc", false),
                ("abcde", false),
            ],
        }
        test_partial_ci_interleaved_digits {
            pattern: "^(?i:a)1(?i:b)2$",
            memory: 839,
            min_tier: 1,
            inputs: [
                ("a1b2", true),
                ("A1b2", true),
                ("a1B2", true),
                ("A1B2", true),
                ("a2b2", false),
                ("a1b1", false),
                ("A2B2", false),
                ("", false),
                ("a1b", false),
                ("1b2", false),
            ],
        }

        // -----------------------------------------------------------------
        // Multi-counter tier 2 tests: non-nested counters with disjoint
        // character classes should use tier 2 (differential counters).
        // -----------------------------------------------------------------

        test_multi_counter_two_fixed {
            pattern: "^[A-Z]{4}[0-9]{16}$",
            memory: 1879,
            min_tier: 1,
            inputs: [
                ("ABCD1234567890123456", true),
                ("ABCD12345678901234567", false),
                ("ABCD123456789012345", false),
                ("ABC1234567890123456", false),
                ("ABCDE1234567890123456", false),
                ("abcd1234567890123456", false),
                ("", false),
            ],
        }
        test_multi_counter_two_ranges {
            pattern: "^a{3,5}b{2,4}$",
            memory: 1136,
            min_tier: 1,
            inputs: [
                ("aaabb", true),
                ("aaaabb", true),
                ("aaaaabb", true),
                ("aaabbb", true),
                ("aaabbbb", true),
                ("aaaabbbb", true),
                ("aaaaabbb", true),
                ("aaaaabbbb", true),
                ("aab", false),
                ("aaab", false),
                ("aaaaaabb", false),
                ("aaabbbbb", false),
                ("", false),
                ("ab", false),
                ("ba", false),
            ],
        }
        test_multi_counter_three_fixed {
            pattern: "^a{2}b{3}c{2}$",
            memory: 938,
            min_tier: 1,
            inputs: [
                ("aabbbcc", true),
                ("abbbcc", false),
                ("aaabbcc", false),
                ("aabbcc", false),
                ("aabbbbcc", false),
                ("aabbbc", false),
                ("aabbbccc", false),
                ("", false),
            ],
        }
        test_multi_counter_three_ranges {
            pattern: "^x{1,3}y{2,4}z{1,2}$",
            memory: 1169,
            min_tier: 1,
            inputs: [
                ("xyyz", true),
                ("xxyyz", true),
                ("xxxyyz", true),
                ("xyyyz", true),
                ("xyyyyz", true),
                ("xxyyyyzz", true),
                ("xxxyyyyzz", true),
                ("yz", false),
                ("xyz", false),
                ("xxxxyyyz", false),
                ("xxyyyyyzz", false),
                ("xxxyyzzzz", false),
                ("", false),
            ],
        }
        test_multi_counter_unanchored {
            pattern: "[A-Z]{3}[0-9]{3}",
            memory: 1351,
            min_tier: 1,
            inputs: [
                ("ABC123", true),
                ("xxABC123xx", true),
                ("AB123", false),
                ("ABC12", false),
                ("abc123", false),
                ("", false),
            ],
        }
        test_multi_counter_with_literal_prefix {
            pattern: "^ID-[A-Z]{4}-[0-9]{6}$",
            memory: 1681,
            min_tier: 1,
            inputs: [
                ("ID-ABCD-123456", true),
                ("ID-ABCD-12345", false),
                ("ID-ABC-123456", false),
                ("ID-ABCDE-123456", false),
                ("ID-ABCD-1234567", false),
                ("ID-abcd-123456", false),
                ("IDABCD123456", false),
                ("", false),
            ],
        }
        // Overlapping character classes: must NOT be tier 2.
        test_multi_counter_overlapping_falls_to_tier3 {
            pattern: r"^\w{3}\d{2}$",
            memory: 1384,
            min_tier: 1,
            inputs: [
                ("abc12", true),
                ("a1b23", true),
                ("12345", true),
                ("ab1", false),
                ("abcd1", false),
                ("", false),
            ],
        }

        // --- Alternation with empty branches (common-prefix factoring) ---

        // Regression: `(?i)(groups|group)` panicked because regex-syntax
        // factors it into `group(s|ε)`, producing an Alternation with an
        // Empty child.
        test_alternation_empty_branch_ci {
            pattern: r"(?i)\[(groups|group)\]",
            memory: 938,
            min_tier: 1,
            inputs: [
                ("[groups]", true),
                ("[group]", true),
                ("[GROUP]", true),
                ("[GROUPS]", true),
                ("[grape]", false),
                ("hello", false),
            ],
        }
        // Explicit `(a|)` — one branch is empty.
        test_alternation_explicit_empty {
            pattern: "(a|)",
            memory: 707,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("b", true),
            ],
        }
        // Empty branch first: `(|a)`.
        test_alternation_empty_first {
            pattern: "(|a)",
            memory: 707,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("b", true),
            ],
        }
        // Three-way alternation with empty branch: `(a|b|)`.
        test_alternation_three_way_empty {
            pattern: "(a|b|)",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("b", true),
                ("c", true),
            ],
        }
        // Anchored common-prefix factoring: `^(abc|ab)$` → `^ab(c?)$`.
        test_alternation_common_prefix_anchored {
            pattern: "^(abc|ab)$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("abc", true),
                ("ab", true),
                ("a", false),
                ("abcd", false),
            ],
        }
        // Case-insensitive common-prefix: `(?i)(abc|ab)`.
        test_alternation_common_prefix_ci {
            pattern: "(?i)(abc|ab)",
            memory: 773,
            min_tier: 1,
            inputs: [
                ("ab", true),
                ("abc", true),
                ("AB", true),
                ("ABC", true),
                ("x", false),
            ],
        }
        // Common-prefix with counter body: `^(a{2,3}b|a{2,3})$`.
        test_alternation_common_prefix_counter {
            pattern: "^(a{2,3}b|a{2,3})$",
            memory: 1037,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aaa", true),
                ("aab", true),
                ("aaab", true),
                ("a", false),
                ("aaaab", false),
            ],
        }
        // ---------------------------------------------------------------
        // Tier 3 regression: optional elements in counter bodies.
        //
        // These patterns have variable-length bodies (due to `?`) and are
        // not tier 2 eligible, so they exercise tier 3's per-instance
        // tracking.  The bug was that `continue_origins` included states
        // reachable without crossing CInc, causing phantom counter
        // increments when the optional element's byte overlapped with the
        // next iteration's prefix.
        // ---------------------------------------------------------------

        // Optional at end of body: `(aaa?){2}` — the original bug case.
        test_tier3_optional_end {
            pattern: "^(aaa?){2}$",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("aaaa", true),
                ("aaaaa", true),
                ("aaaaaa", true),
                ("aaa", false),
                ("aa", false),
                ("a", false),
                ("", false),
            ],
        }
        // Optional in 2-char body: `(ab?){2}`.
        test_tier3_optional_short_body {
            pattern: "^(ab?){2}$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aba", true),
                ("aab", true),
                ("abab", true),
                ("ab", false),
                ("a", false),
                ("abba", false),
                ("", false),
            ],
        }
        // Optional at start of body: `(a?b){2}`.
        test_tier3_optional_start {
            pattern: "^(a?b){2}$",
            memory: 905,
            min_tier: 1,
            inputs: [
                ("bb", true),
                ("abb", true),
                ("bab", true),
                ("abab", true),
                ("ab", false),
                ("b", false),
                ("ba", false),
                ("", false),
            ],
        }
        // Optional in middle of 3-char body: `(ab?c){2}`.
        test_tier3_optional_middle {
            pattern: "^(ab?c){2}$",
            memory: 971,
            min_tier: 1,
            inputs: [
                ("acac", true),
                ("abcac", true),
                ("acabc", true),
                ("abcabc", true),
                ("ac", false),
                ("abc", false),
                ("abcab", false),
                ("", false),
            ],
        }
        // Higher count: `(ab?){3}`.
        test_tier3_optional_higher_count {
            pattern: "^(ab?){3}$",
            memory: 1004,
            min_tier: 1,
            inputs: [
                ("aaa", true),
                ("aaba", true),
                ("aabab", true),
                ("ababab", true),
                ("abab", false),
                ("aa", false),
                ("abba", false),
                ("", false),
            ],
        }
        // Range count with optional: `(aab?){2,4}`.
        test_tier3_optional_range {
            pattern: "^(aab?){2,4}$",
            memory: 1301,
            min_tier: 1,
            inputs: [
                ("aaaa", true),
                ("aabaa", true),
                ("aabaab", true),
                ("aabaabaab", true),
                ("aabaabaabaa", true),
                ("aabaabaabaabaab", false),
                ("aa", false),
                ("aab", false),
                ("", false),
            ],
        }
        // Range count, optional at start: `(a?b){3,5}`.
        test_tier3_optional_range_start {
            pattern: "^(a?b){3,5}$",
            memory: 1268,
            min_tier: 1,
            inputs: [
                ("bbb", true),
                ("abbb", true),
                ("abbab", true),
                ("ababab", true),
                ("abababab", true),
                ("bb", false),
                ("bbbbbb", false),
                ("", false),
            ],
        }
        // Multi-char body with optional: `(abc?d){2}`.
        test_tier3_optional_multichar {
            pattern: "^(abc?d){2}$",
            memory: 1037,
            min_tier: 1,
            inputs: [
                ("abdabd", true),
                ("abcdabd", true),
                ("abdabcd", true),
                ("abcdabcd", true),
                ("abd", false),
                ("abcd", false),
                ("abcabc", false),
                ("", false),
            ],
        }

        // ── Adjacent same-byte sequential counters (pre-existing bug) ───
        // Two counters on the same byte 'a' without a separator — tests
        // the break_seeds fix in tier 3.

        test_adjacent_same_byte_counters {
            pattern: "^a{2,50}a{3,70}$",
            memory: 907,
            min_tier: 3,
            inputs: [
                ("aaaaa", true),
                ("aaaaaa", true),
                ("aaaaaaaaaa", true),
                ("aaaa", false),
                ("aaa", false),
                ("aa", false),
                ("a", false),
                ("", false),
            ],
        }

        // ── Inner-unrolling-inside-counter-body tests ───────────────────
        // These patterns previously required tier 4 (nested counters) but
        // now the inner repetition unrolls, leaving a single counter → tier 3.

        // Simple: inner {1,3} unrolls (cost 5 ≤ 32), body becomes variable-length → tier 3.
        test_inner_unroll_simple {
            pattern: "^(a{1,3}){2,4}$",
            memory: 1433,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("aaaaaa", true),
                ("aaaaaaaaaa", true),
                ("aaaaaaaaaaaa", true),
                ("a", false),
                ("aaaaaaaaaaaaa", false),
                ("", false),
            ],
        }
        // Inner {1,2} with multi-byte body inside counter.
        test_inner_unroll_multibyte_body {
            pattern: "^(ab{1,2}){2,3}$",
            memory: 1136,
            min_tier: 1,
            inputs: [
                ("abab", true),
                ("abbab", true),
                ("ababb", true),
                ("abbabb", true),
                ("abbabbabb", true),
                ("ab", false),
                ("abbabbabbabb", false),
                ("", false),
            ],
        }
        // Inner {0,3} inside counter — zero-min inner unrolls too.
        test_inner_unroll_zero_min {
            pattern: "^(a{0,3}){2,4}$",
            memory: 1565,
            min_tier: 1,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", true),
                ("aaa", true),
                ("aaaaaa", true),
                ("aaaaaaaaaaaa", true),
                ("aaaaaaaaaaaaa", false),
            ],
        }
        // Two inner repetitions in the same counter body.
        test_inner_unroll_two_reps {
            pattern: "^(a{1,2}b{1,2}){2,3}$",
            memory: 1334,
            min_tier: 1,
            inputs: [
                ("abab", true),
                ("aabbab", true),
                ("ababb", true),
                ("aabbaabb", true),
                ("aabbabbaabb", true),
                ("aabbaabbaabb", true),
                ("ab", false),
                ("", false),
            ],
        }
        // Case-insensitive inner unrolling.
        test_inner_unroll_case_insensitive {
            pattern: "^(?i)(a{1,3}){2,4}$",
            memory: 1433,
            min_tier: 1,
            inputs: [
                ("aA", true),
                ("AaA", true),
                ("aaAA", true),
                ("AaAaAaAaAaAa", true),
                ("a", false),
                ("AaAaAaAaAaAaA", false),
                ("", false),
            ],
        }
        // Inner over budget: a{1,17} costs 33 > 32, stays as counter.
        // Outer {2,3} unrolls to 3 sequential counters → tier 3.
        // Uses (a{1,17}b) body so the 3 counters match different
        // byte sequences, avoiding a tier 3 limitation with same-byte
        // sequential counters.
        test_inner_unroll_over_budget {
            pattern: "^(a{1,17}b){2,3}$",
            memory: 1139,
            min_tier: 3,
            inputs: [
                ("abab", true),
                ("aaaaabab", true),
                ("aaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaab", true),
                ("aaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaab", true),
                ("ab", false),
                ("aaaaaaaaaaaaaaaaaaaab", false),
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false),
                ("", false),
            ],
        }
        // The motivating pattern from the rebar benchmarks.
        test_inner_unroll_ip_pattern {
            pattern: r"(?i)(?:(?:[0-9]{1,3}\.){3}[0-9]{1,3}|(?:[0-9a-f]{1,4}::?){2,7}[0-9a-f]{1,4}):$",
            memory: 2639,
            min_tier: 3,
            inputs: [
                ("192.168.1.1:", true),
                ("10.0.0.1:", true),
                ("255.255.255.255:", true),
                ("a:b:c:", true),
                ("fe80:0:1:", true),
                ("192.168.1:", false),
                ("192.168.1.1", false),
                ("", false),
            ],
        }
        // Anchored inner {1,4} inside {3} (fixed outer).
        test_inner_unroll_fixed_outer {
            pattern: "^(a{1,4}){3}$",
            memory: 1400,
            min_tier: 1,
            inputs: [
                ("aaa", true),
                ("aaaa", true),
                ("aaaaaaaaaaaa", true),
                ("aa", false),
                ("aaaaaaaaaaaaa", false),
                ("", false),
            ],
        }
        // Inner {2,3} inside {0,2} — zero-min outer with inner unrolling.
        test_inner_unroll_zero_min_outer {
            pattern: "^(a{2,3}){0,2}$",
            memory: 1037,
            min_tier: 1,
            inputs: [
                ("", true),
                ("aa", true),
                ("aaa", true),
                ("aaaa", true),
                ("aaaaa", true),
                ("aaaaaa", true),
                ("a", false),
                ("aaaaaaa", false),
            ],
        }
        // Dot-based inner repetition inside counter.
        test_inner_unroll_dot_body {
            pattern: "^(.{1,3}x){2,3}$",
            memory: 1590,
            min_tier: 1,
            inputs: [
                ("axbx", true),
                ("abxcdx", true),
                ("abcxdexfgx", true),
                ("abcde", false),
                ("x", false),
                ("", false),
            ],
        }

        // ── Variable-length-body range compression ──────────────────────
        //
        // Tier 3 uses range-compressed instance tracking (InstanceRange)
        // for all counters, including those with variable-length bodies.
        // These tests exercise the multi-origin range-merge path to verify
        // that RangeCounters::insert()'s min/max extension never introduces phantom
        // values.
        //
        // The original implementation only enabled range compression for
        // fixed-length bodies (body_byte_length > 0), where each counter
        // has exactly one consuming-state origin.  Variable-length bodies
        // have multiple origins (one per consuming state in the unrolled
        // body), but the contiguity invariant still holds:
        //
        //  - Seeds always inject value 0 at the body entry origin.
        //  - Advance moves (value, origin_k) → (value, origin_{k+1}),
        //    preserving contiguity.
        //  - Continue (CInc) moves (value, origin_last) →
        //    (value+1, origin_0), where the +1 shift always overlaps or
        //    extends the range already present at origin_0 (which starts
        //    from 0 due to seeding).
        //  - Dead actions kill an entire range at an origin but never
        //    split it, so surviving ranges remain contiguous.
        //
        // Therefore RangeCounters::insert()'s min/max merge is correct: it never
        // bridges a gap, because gaps cannot form.

        // Alternation body: `(ab|a)` has 2 consuming states (body lengths
        // 1 and 2), producing 2 origins.  Tests selective byte matching
        // where the 'a' origin dies on non-'a' bytes.
        test_tier3_varlen_alternation_body {
            pattern: "(ab|a){1,100}c",
            memory: 873,
            min_tier: 3,
            inputs: [
                ("ac", true),
                ("abc", true),
                ("aababc", true),
                ("aac", true),
                ("ababababababababababababc", true),
                ("c", false),
                ("bc", false),
                ("xc", false),
            ],
        }
        // Dot-range body: `(.{1,5})` has 5 consuming states → 5 origins.
        // All bytes match at every origin, so ranges grow uniformly.
        test_tier3_varlen_dot_range_body {
            pattern: "(.{1,5}){1,100}z",
            memory: 1294,
            min_tier: 3,
            inputs: [
                ("az", true),
                ("abcz", true),
                ("abcdez", true),
                ("abcdefz", true),
                ("z", false),
            ],
        }
        // Mixed-selectivity body: `(a.|b)` has 2 origins — origin 0
        // (`a` or `b`) is selective, origin 1 (`.` after `a`) matches
        // any byte.  The 'b' branch bypasses origin 1 entirely.
        test_tier3_varlen_mixed_selectivity {
            pattern: "(a.|b){1,50}c",
            memory: 1129,
            min_tier: 3,
            inputs: [
                ("axc", true),
                ("bc", true),
                ("axbaxbc", true),
                ("bbbbbbc", true),
                ("ababababc", true),
                ("c", false),
            ],
        }
        // Large variable body with higher count — stress test for
        // range merge at 10 origins, 1000 max iterations.
        test_tier3_varlen_large_body_high_count {
            pattern: ".{0,1000}(.{1,10}){0,1000}c",
            memory: 1790,
            min_tier: 3,
            inputs: [
                ("c", true),
                ("xc", true),
                ("xxxxxxxxxxc", true),
                ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxc", true),
            ],
        }
        // Anchored variable body: ensures end-anchor interacts correctly
        // with multi-origin range tracking.
        test_tier3_varlen_anchored {
            pattern: "^(ab|a){2,5}$",
            memory: 1466,
            min_tier: 1,
            inputs: [
                ("aa", true),
                ("aba", true),
                ("aab", true),
                ("abab", true),
                ("aabab", true),
                ("ababab", true),
                ("a", false),
                ("ab", false),
                ("", false),
            ],
        }
        // Two sequential variable-length counters: each tracked
        // independently with its own set of origin ranges.
        test_tier3_varlen_two_counters {
            pattern: "(ab|a){1,50}(xy|x){1,50}z",
            memory: 1072,
            min_tier: 3,
            inputs: [
                ("axz", true),
                ("abxyz", true),
                ("aaxxz", true),
                ("ababxyxyz", true),
                ("z", false),
                ("az", false),
                ("xz", false),
            ],
        }
        // Variable-length body behind a consuming prefix: the `+` loop
        // inside `a+{9,28}` makes `advance_origins` and `continue_origins`
        // overlap at the same NFA state.  Per-instance tracking must dedup
        // `advance` and `insert_continued` calls to prevent exponential
        // instance growth that silently drops high-value entries when the
        // flat buffer fills up.  (Bug: InstanceCounters advance/continue
        // lacked dedup, causing the counter to never reach its minimum.)
        test_tier3_varlen_plus_body_dedup {
            pattern: ".a+{9,28}",
            memory: 1063,
            min_tier: 3,
            inputs: [
                ("aaaaaaaaaa", true),      // 10 a's: . + a+{9} (each a+ = 1 a)
                ("baaaaaaaaaa", true),     // b + 10 a's
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", true), // 48 a's
                ("ba", false),
                ("baa", false),
                ("baaaaaaaa", false),      // b + 8 a's: only 8 iterations, need 9
            ],
        }

        // ── break_consuming_states fix regression tests ──────────────────
        //
        // These exercise the post-break tail tracker in tier 3, which
        // detects `$ → Match` from consuming states on the CInc break
        // path.  Before the fix, the break_consuming_states field was
        // always empty because step 2b tried to extract the consuming
        // state's `out` from the target state itself (which may be a CInc
        // or other epsilon node, not a consuming state).

        // Single counter with trailing literal: the break path from CInc
        // leads to `b`, then `$ → Match`.  Only the post-break tail
        // tracker detects this when counter 0 breaks.
        test_tier3_post_break_tail_literal {
            pattern: "^a.{3}b$",
            memory: 1128,
            min_tier: 1,
            inputs: [
                ("a123b", true),
                ("a___b", true),
                ("axxxb", true),
                ("a12b", false),
                ("a1234b", false),
                ("a123c", false),
                ("", false),
            ],
        }
        // Single counter with trailing wildcard: break leads to `.` then
        // `$ → Match`.
        test_tier3_post_break_tail_wildcard {
            pattern: "^.{3,5}.$",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("abcd", true),
                ("abcde", true),
                ("abcdef", true),
                ("abc", false),
                ("abcdefg", false),
                ("", false),
            ],
        }
        // Counter followed by two literals: break tail must advance
        // through the two-hop path `b → c → $ → Match`.
        test_tier3_post_break_tail_two_hop {
            pattern: "^.{2,4}bc$",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("aabc", true),
                ("aaabc", true),
                ("aaaabc", true),
                ("abc", false),
                ("aaaaabc", false),
                ("aabb", false),
                ("", false),
            ],
        }

        // ── counter_free_match_at_end Advance-action fix ─────────────────
        //
        // The counter break path's target may have BOTH consuming states
        // (action = Advance) AND an epsilon path to `$ → Match`.  Before
        // the fix, `counter_free_match_at_end` only checked Dead actions
        // (targets with no consuming states), missing the Advance case.
        // Input "zzz": `.{0,2}` consumes 2, `.` consumes 1, `a?` skips,
        // reaching `$` — but tier 3 incorrectly reported NO MATCH because
        // the Advance-action origin's `target_is_match_at_end` was ignored.
        test_tier3_counter_free_mae_advance {
            pattern: "^(.{0,2}.a?)?$",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("", true),
                ("z", true),
                ("zz", true),
                ("zzz", true),
                ("zzza", true),
                ("za", true),
                ("zza", true),
                ("a", true),
                ("aa", true),
                ("zzzz", false),
                ("zzzzz", false),
            ],
        }
        // Regression: counter_free_match_at_end and no_break_current's
        // is_match_at_end in Tier 3 were over-broad — they included
        // `$ → Match` paths from origins/targets only reachable via counter
        // breaks (e.g. the suffix `(a?)?$` in `^c{2,12}.{6,6}(a?)?$` is
        // only reachable after both counters break).  Fix: filter by
        // `reachable_without_break` (transitive NFA walk excluding CInc
        // break paths) and track `nb_counter_free_mae` separately.
        test_tier3_counter_free_mae_multi_counter {
            pattern: "^c{2,12}.{6,6}(a?)?$",
            memory: 1986,
            min_tier: 1,
            inputs: [
                ("ccaaaaaa", true),
                ("ccaaaaaaa", true),
                ("ccccccccaaaaaaa", true),
                ("ccccccaaaaaa", true),
                ("ccccccaa", true),
                ("ccca", false),
                ("cca", false),
                ("cccccc", false),
                ("c", false),
                ("", false),
            ],
        }
        test_tier3_counter_free_mae_two_counters_optional {
            pattern: "^.{0,2}.{4,4}a?$",
            memory: 1293,
            min_tier: 1,
            inputs: [
                ("aaaa", true),
                ("aaaaaa", true),
                ("aaaaa", true),
                ("aaaaab", true),
                ("aa", false),
                ("a", false),
                ("", false),
            ],
        }
        // Regression: counter_free_nb_mae early-return guard used
        // `reachable_without_break[regex.start.idx()]` to decide whether
        // the re-seeded start state could contribute a counter-free
        // `$ → Match`.  But reachable_without_break only marks *consuming*
        // states; the start state (a Split) was never marked, so the guard
        // bailed out and returned false.  For patterns like `(X.{n,m})?$`
        // where the `?`-skip directly reaches `$`, the match_at_end flag
        // was overwritten to false on every byte, causing false negatives.
        test_tier3_counter_free_mae_optional_skip {
            pattern: "([b-ed-ie-f].{4,43})?$",
            memory: 1352,
            min_tier: 2,
            inputs: [
                ("zzz", true),       // optional group skipped, $ matches at end
                ("", true),          // empty input, $ matches
                ("b12345", true),    // group matches: b + 5 chars ≥ 4
                ("d1234", true),     // group matches: d + 4 chars = 4
                ("a", true),         // 'a' not in class, group skipped, $ matches
                ("bbb", true),       // too short for body, group skipped, $ matches
            ],
        }
        // Regression: post_break_tails Advance action did not check
        // target_is_match_at_end.  When a post-break tail's target has
        // BOTH consuming states (producing Advance) AND `$ → Match`, the
        // match_at_end flag was never set.  E.g. `^.{7,23}.a?$` on 8 a's:
        // counter breaks at 7, `.` consumes byte 8, target has `a?$` which
        // includes both `Byte('a')` (Advance) and `$ → Match`.  Without
        // the fix, finish() missed the `$ → Match` path.
        test_tier3_post_break_advance_mae {
            pattern: "^.{7,23}.a?$",
            memory: 1162,
            min_tier: 2,
            inputs: [
                ("aaaaaaaa", true),    // len=8: min+1, match via $ after .
                ("aaaaaaaaa", true),   // len=9
                ("aaaaaaa", false),    // len=7: too short (need 7+1=8 min)
                ("aaaaaaaaaaaaaaaaaaaaaaaa", true),   // len=23: max-1
                ("aaaaaaaaaaaaaaaaaaaaaaaaa", true),  // len=24: max
                ("aaaaaaaaaaaaaaaaaaaaaaaaaa", false), // len=26: too long
            ],
        }
        // Regression: counter self-loop — `(.{6,9})+` wraps a counted
        // repetition in a `+` quantifier.  With unroll_limit=0, the CInc
        // break path reaches CI for the same counter, creating overlapping
        // instances that spuriously satisfy the break condition.  The fix
        // rejects such patterns from tier 2/3 (demoted to tier 4).
        test_counter_self_loop_possessive_plus {
            pattern: "^.{6,9}++$",
            memory: 1425,
            min_tier: 1,
            inputs: [
                ("aaaaaa", true),       // len=6: one rep of 6
                ("aaaaaaaaa", true),    // len=9: one rep of 9
                ("aaaaaaaaaaaa", true), // len=12: two reps of 6
                ("aaaaaaaaaa", false),  // len=10: no valid split
                ("aaaaaaaaaaa", false), // len=11: no valid split
                ("aaaaa", false),       // len=5: too short
            ],
        }
        // Same self-loop via {6,9}+ parsed as possessive (one-or-more).
        test_counter_self_loop_possessive {
            pattern: "^.{6,9}+$",
            memory: 1392,
            min_tier: 1,
            inputs: [
                ("aaaaaa", true),       // len=6
                ("aaaaaaaaa", true),    // len=9
                ("aaaaaaaaaaaa", true), // len=12: 6+6
                ("aaaaaaaaaaaaaaa", true), // len=15: 6+9 or 7+8
                ("aaaaaaaaaa", false),  // len=10: no valid split
                ("aaaaaaaaaaa", false), // len=11: no valid split
                ("aaaaa", false),       // len=5: too short
            ],
        }
        // Same self-loop via * wrapping counted repetition.
        test_counter_self_loop_star {
            pattern: "^.{6,9}*$",
            memory: 1392,
            min_tier: 1,
            inputs: [
                ("", true),             // len=0: zero reps
                ("aaaaaa", true),       // len=6
                ("aaaaaaaaa", true),    // len=9
                ("aaaaaaaaaaaa", true), // len=12: 6+6
                ("aaaaaaaaaa", false),  // len=10: no valid split
                ("aaaaa", false),       // len=5: too short
            ],
        }
        // Same self-loop via +* wrapping counted repetition.
        test_counter_self_loop_plus_star {
            pattern: "^.{6,9}+*$",
            memory: 1425,
            min_tier: 1,
            inputs: [
                ("", true),             // len=0: * allows zero reps
                ("aaaaaa", true),       // len=6
                ("aaaaaaaaaaaa", true), // len=12: 6+6
                ("aaaaaaaaaa", false),  // len=10: no valid split
                ("aaaaa", false),       // len=5: too short
            ],
        }
        // Same self-loop, unanchored.  Since unanchored, any input
        // with >= 6 chars matches (a 6-9 substring always exists).
        test_counter_self_loop_unanchored {
            pattern: ".{6,9}+",
            memory: 1326,
            min_tier: 1,
            inputs: [
                ("aaaaaa", true),       // len=6: exact match
                ("aaaaaaaaaa", true),   // len=10: substring of 6-9 exists
                ("aaaaaaaaaaaa", true), // len=12
                ("aaaaa", false),       // len=5: too short
            ],
        }
        // Regression: counter-free seed filtering in tier 3 must NOT remove
        // unconditional seeds whose downstream counter is only reachable via
        // a CInc break path, when the post-break tail mechanism is the only
        // way to seed that counter.  The Bug 15 fix (counter-free closure
        // filtering) removes counter-dependent seeds from the unconditional
        // list, but the post-break tail CInc handoff compensates: when a
        // post-break tail consumes a byte and hits a CInc, it seeds the
        // downstream counter directly into the next buffer.
        //
        // This test ensures the handoff works: counter 1 (`x{6,50}`) is
        // only reachable after counter 0 (`0{8,31}`) breaks.  The post-break
        // tail [6] (Byte 'x') hits CInc-1 on the first 'x' and seeds
        // counter 1 at value 1.
        test_tier3_seed_filtering_revert {
            pattern: "^(0{8,31}(0+x{6,50}1y?)?(a?a?)?)?$",
            memory: 1303,
            min_tier: 2,
            inputs: [
                ("", true),
                ("00000000", true),
                ("0000000000000000000000000000000", true),
                ("00000000aa", true),
                ("0000000000000000000000000000000000000xxxxxx1", true),
                ("0000000000000000000000000000000000000xxxxxx1y", true),
                ("0000000", false),
                ("00000000000000000000000000000000", false),
                ("xxxx", false),
                ("aaa", false),
            ],
        }
        // Regression (Bug 15): counter-dependent unconditional seeds in
        // tier 3 caused false positives when the seed's CI was only
        // reachable from counter-dependent origins but the seed origin's
        // byte class overlapped with counter-free origins.  Pattern
        // `^(.{6,39}.0?.{8,8}a?)?$` with 50 'a's: counter 1 (`.{8,8}`)
        // was seeded unconditionally on every byte, allowing it to
        // break at position 49 even though no valid NFA path exists
        // (max total = 39 + 1 + 8 + 1 = 49 < 50).
        //
        // Fix: filter unconditional seeds to only those in the
        // counter-free no-break closure (seeds from
        // `reachable_without_break` origins only).  Counter-dependent
        // seeds are handled by break_seeds + post-break tail CInc
        // handoff.
        test_tier3_counter_dep_seed_false_positive {
            pattern: "^(.{6,39}.0?.{8,8}a?)?$",
            memory: 1525,
            min_tier: 2,
            inputs: [
                ("", true),
                ("aaaaaaaaaaaaaaaa", true),   // 16: 6+1+8+1=16
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", true), // 49: 39+1+8+1
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false), // 50: exceeds max
                ("aaaaaaaaaaaaaaa", true),    // 15: 6+1+8+0=15
                ("aaaaaaaaaaaaaa", false),    // 14: too short (min is 6+1+8=15)
                ("aaaaaaa0aaaaaaaa", true),   // 16: 6+1(.)+1(0)+8=16
                ("aaa", false),
            ],
        }
        // Regression (Bug 16): deferred assertions (`\b`, `\B`) before
        // counter bodies with L>1 caused false negatives in Tier 2 and
        // Tier 3.  Phase 1 of `populate()` resolves the assertion and
        // consumes the first body byte, but the resulting seed was
        // misaligned:
        //
        // - Tier 2 (differential phases): the seed was placed as a
        //   post-advance_all_phases seed, off by one phase.  Fix: use
        //   pre_seeds (applied before advance) for correct alignment.
        //
        // - Tier 3 (per-instance): the seed's origin was the consumed
        //   NFA state, which no longer appears in subsequent transitions'
        //   origin_keys.  Fix: remap the seed's origin to the post-
        //   consumption target.
        //
        // L=1 bodies were already handled correctly (pre_seeds in
        // counting, value bump in non-counting).  This test covers L=2
        // (Tier 2), L=6 (Tier 2), and L=6..16 (Tier 3).
        test_deferred_assert_before_counter_l2_body {
            pattern: r"^...\B(ee){6,40}$",
            memory: 1228,
            min_tier: 2,
            inputs: [
                ("aaaeeeeeeeeeeee", true),     // 3 + 12 = 15 (6 iterations of ee)
                ("aaaeeeeeeeeeeeeeeeeeeee", true), // 3 + 20 = 23 (10 iterations)
                ("aaaeeeeeeeee", false),        // 3 + 8 = 11 (4 iterations, below min)
                ("aaae", false),                // too short
                ("   eeeeeeeeeeee", false),     // \B fails: space→e is word boundary
                ("aaaeeeeeeeeeeeee", false),    // 3 + 13: odd e count, can't fill (ee) body
            ],
        }
        test_deferred_assert_before_counter_l6_body {
            pattern: r"^...\Be{6,6}{6,40}$",
            memory: 1360,
            min_tier: 2,
            inputs: [
                ("aaaeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", true), // 3+36=39 (6 iters)
                ("aaaeeeeeeeeeeeeeeeee", false), // 3+16: not multiple of body_len*min
                ("aaa", false),
                ("   eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", false), // \B fails
            ],
        }
        test_deferred_assert_before_counter_varlen_body {
            pattern: r"^...\Be{6,16}{6,40}$",
            memory: 2020,
            min_tier: 3,
            inputs: [
                ("aaaeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", true), // 3+36=39
                ("aaaaaaaaaaaaaaaaaaaaa", false), // 21: all 'a', body is 'e'-only
                ("aaa", false),
                ("   eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee", false), // \B fails
            ],
        }
        test_deferred_assert_before_counter_two_counters {
            pattern: r"^.{3,5}\Be{6,16}{6,40}c{6,33}$",
            memory: 2252,
            min_tier: 3,
            inputs: [
                // 3 a's + 36 e's + 6 c's = 45: min scenario (3 prefix, 6*6 e body, 6 c's)
                ("aaaeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecccccc", true),
                // 5 a's + 36 e's + 6 c's = 47
                ("aaaaaeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeecccccc", true),
                ("aaa", false),
                ("aaaeeeeeecccccc", false), // not enough e iterations
            ],
        }
        // Regression (Bug 17): `\b` deferred assertion at the start state
        // gates a Split → CI-0 (optional counter body) / CI-1 (required
        // counter body).  CI-1's seed at its first body byte appears in
        // `analysis.break_seeds` because CInc-0's break path also reaches
        // CI-1.  Before the fix, the Bug 14 filter incorrectly excluded
        // this seed as break-gated even though the origin was
        // reachable_without_break (i.e., unconditionally reachable from
        // start).  The fix checks reachable_without_break before
        // classifying resolved seeds as break-gated.
        test_deferred_assert_rwb_break_seed {
            pattern: r"^\b(b{2,8})?.{9,9}$",
            memory: 1788,
            min_tier: 1,
            inputs: [
                ("aaaaaaaaa", true),       // \b satisfied, (b{2,8})? zero iters, .{9,9} matches
                ("bbaaaaaaaaa", true),      // \b, bb matches b{2,8}, then 9 a's
                ("bbbbbbbbaaaaaaaaa", true), // \b, 8 b's, then 9 a's
                ("bbbbbbbbb", true),        // \b, 2 b's from counter, then 7+2? Let NFA decide
                ("aaaaaaaab", true),        // 9 chars, .{9,9} matches
                ("aaaaaaaa", false),        // only 8 chars
                ("aaaaaaaaaa", false),      // 10 chars, too many
                (" aaaaaaaaa", false),      // \b fails at pos 0: prev=None + space = no boundary; also 10 chars
            ],
        }
        test_deferred_assert_rwb_break_seed_star {
            pattern: r"^\b(b{2,8}b)*.{9,9}$",
            memory: 1821,
            min_tier: 1,
            inputs: [
                ("aaaaaaaaa", true),        // zero iters of (b{2,8}b)*, .{9,9}
                ("bbbaaaaaaa", false),       // 3 b's + 8 a's = 11, too many for .{9,9}
                ("12345678a", true),         // .{9,9}
                ("12345678", false),         // 8 chars
            ],
        }
        // Regression (Bug 18): deferred assertion (\B) inside an L=1
        // counter body gates the path to CInc.  When the counter breaks,
        // the break path's consuming states need to consume the current
        // byte and enter the with_break DFA successor.  Before the fix,
        // the code only checked break_path_match_kind (epsilon-to-Match)
        // and never collected consuming states from the break path.
        // The with_break DFA successor was missing those states, causing
        // a false negative.
        test_deferred_assert_inside_counter_break_consuming {
            pattern: r"^(a\B){4,4}a$",
            memory: 1004,
            min_tier: 1,
            inputs: [
                ("aaaaa", true),       // 4 × a\B + a$
                ("aaaa", false),        // only 4 chars, need 5
                ("aaaaaa", false),      // 6 chars, too many
                ("baaaa", false),       // first byte 'b' doesn't match body Byte('a')
            ],
        }
        test_deferred_assert_inside_counter_break_consuming_varlen {
            pattern: r"^(a\B){2,5}a$",
            memory: 1169,
            min_tier: 1,
            inputs: [
                ("aaa", true),         // 2 × a\B + a$
                ("aaaa", true),        // 3 × a\B + a$
                ("aaaaa", true),       // 4 × a\B + a$
                ("aaaaaa", true),      // 5 × a\B + a$
                ("aa", false),         // too few
                ("aaaaaaa", false),    // too many
            ],
        }
        // Bug 19: Tier 3 false positive from counter_free_match_at_end on
        // with_break DFA states.  When counter 0 breaks, the with_break
        // closure follows CInc-0 break → CI-1 → ... → CInc-1 break → the
        // optional suffix's consuming state (Byte 'a').  This state is
        // statically `reachable_without_break` (via the initial Split that
        // skips the optional counter group), so `counter_free_match_at_end`
        // fires.  But the suffix state entered the DFA state via a counter
        // break, not via the counter-free path.  The fix tracks whether the
        // current DFA state was reached via a with_break transition and
        // suppresses `counter_free_match_at_end` in that case.
        //
        // Minimal reproducer: 2 chained counters needing 20+ chars with a
        // counter-free suffix `a?$`, tested on 12 chars (too short for the
        // counters to complete).
        test_tier3_with_break_counter_free_mae_false_positive {
            pattern: r"^(.{10,40}{2,2})?a?$",
            memory: 1262,
            min_tier: 3,
            inputs: [
                ("", true),             // optional group skips, a? skips, $ matches
                ("a", true),            // optional group skips, a? matches 'a', $ matches
                ("aa", false),          // optional group can't complete, suffix too short
                ("aaaaaaaaa", false),   // 9 chars — counter 0 can't break (needs 10)
                ("aaaaaaaaaaaa", false), // 12 chars — counter 0 breaks, counter 1 can't complete
                ("aaaaaaaaaaaaaaaaaaa", false), // 19 chars — still short of 20
                ("aaaaaaaaaaaaaaaaaaaa", true), // 20 chars — both counters complete exactly
                ("aaaaaaaaaaaaaaaaaaaaa", true), // 21 chars — counters + suffix 'a'
            ],
        }
        // Same pattern shape as Bug 19 but with 8 chained counters (the
        // original fuzz artifact).
        test_tier3_with_break_counter_free_mae_8_counters {
            pattern: r"^(.{7,40}{8,8})?a?$",
            memory: 1862,
            min_tier: 3,
            inputs: [
                ("", true),
                ("a", true),
                ("aa", false),
                // 44 chars — need 56 minimum (8 × 7)
                ("aad0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false),
                // 55 chars — still 1 short of 56
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false),
                // 56 chars — exactly 8 × 7, counters complete
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", true),
            ],
        }
        // Bug 20 regression test: The Bug 19 fix must NOT suppress
        // legitimate counter-free match-at-end paths.  In `^(.{0,2}|d+)$`,
        // the `d+` loop exits to `$ → Match` through a Split — this path
        // is genuinely counter-free (no CInc involved).  When the `.{0,2}`
        // counter breaks, the with_break DFA state includes both the `d+`
        // loop state (Byte 'd') and the counter's break-path states.  The
        // `d+` origin is also in the no-break DFA state (entered via the
        // initial Split), so `clean_counter_free_mae()` correctly preserves
        // it while the Bug 19 fix blocks origins that entered only via
        // counter breaks.
        test_tier3_clean_counter_free_mae_preserves_legit {
            pattern: r"^(.{0,2}|d+)$",
            memory: 1194,
            min_tier: 1,
            inputs: [
                ("", true),        // optional .{0,2} matches empty via 0-count
                ("d", true),       // d+ matches, or .{0,2} matches 1 char
                ("dd", true),      // d+ matches, or .{0,2} matches 2 chars
                ("ddd", true),     // d+ matches 3 'd's → $ → Match
                ("dddddd", true),  // d+ matches 6 'd's
                ("a", true),       // .{0,2} matches 1 char
                ("ab", true),      // .{0,2} matches 2 chars
                ("abc", false),    // too long for .{0,2}, not all 'd' for d+
                ("dda", false),    // d+ fails (non-d), .{0,2} only covers 2
            ],
        }
        // Bug 21 regression test: With a single counter (.{0,36} → {1,36}),
        // the counter's break path leads into an unrolled c chain → suffix.
        // When the counter breaks, the with_break DFA state includes suffix
        // origins (Byte 'e', Byte 'a' variants).  These origins are NOT in
        // the no-break DFA state, but with a single counter their presence
        // is valid — the counter DID break.  The Bug 19 fix must not
        // suppress `counter_free_mae` for single-counter patterns, or this
        // gives a false negative.
        test_tier3_single_counter_break_path_mae {
            pattern: r"^.{0,36}c{2,12}e?((a?a?)?(a?a?)?)?$",
            memory: 2251,
            min_tier: 2,
            inputs: [
                ("cc", true),       // .{0,36} = "", c{2,12} = "cc"
                ("ccc", true),      // .{0,36} = "", c{2,12} = "ccc"
                ("cce", true),      // .{0,36} = "", c{2,12} = "cc", e? = "e"
                ("c", false),       // c{2,12} needs at least 2
                ("e", false),       // no 'c' prefix
                ("cccccccccccccc", true),  // 14 c's: .{0,36} eats 2, c{2,12} eats 12
                ("acccc", true),    // .{0,36} = "a", c{2,12} = "cccc"
                ("accea", true),    // .{0,36} = "a", c{2,12} = "cc", e? = "e", suffix = "a"
            ],
        }

        // Bug 22: Tier 3 false positive from contaminated no-break chain.
        // In `^(.{7,25}.{7,25})?((a?a?)?(a?a?)?)?$`, the two chained
        // `.{7,25}` counters need at least 14 chars combined.  The suffix
        // `((a?a?)?(a?a?)?)?` only accepts 'a' chars.  On input
        // "dcya cbcyaa" (11 chars, includes non-'a'), neither counter
        // completes — the answer should be NO MATCH.
        //
        // Root cause: When counter 0 breaks, the with_break DFA state
        // includes the suffix's consuming states (Byte 'a').  These
        // propagate into subsequent no-break DFA states because the
        // no-break closure from the contaminated state inherits them.
        // The `clean_counter_free_mae` check (Bug 19) used
        // `self.no_break_current` as the "clean" reference, but that state
        // was ITSELF contaminated.
        //
        // Fix: Maintain a separate `clean_nb` DFA state chain that always
        // advances from the previous clean state, never from a contaminated
        // one.  `clean_counter_free_mae` uses `clean_nb` to verify that
        // an origin is genuinely reachable without counter breaks.
        test_tier3_contaminated_no_break_chain_false_positive {
            pattern: r"^(.{7,25}.{7,25})?((a?a?)?(a?a?)?)?$",
            memory: 1559,
            min_tier: 3,
            inputs: [
                ("", true),              // optional group skips, suffix skips, $ matches
                ("a", true),             // optional group skips, suffix matches "a"
                ("aa", true),            // optional group skips, suffix matches "aa"
                ("aaaa", true),          // optional group skips, suffix matches "aaaa"
                ("d", false),            // suffix only accepts 'a', counter needs 14+
                ("dcya cbcyaa", false),  // Bug 22 reproducer: 11 chars, non-'a' chars
                ("aaaaaaaaaaaaaaa", true), // 15 chars: counters complete (7+7), suffix "a"
                ("aaaaaaaaaaaaaa", true),  // 14 chars: counters complete exactly (7+7)
                ("aaaaaaaaaaaaa", false),  // 13 chars: too short for 7+7
            ],
        }

        // Bug 23: Tier 3 false positive from non-counting transition
        // propagating raw `no_break_is_match_at_end` without counter-free
        // filtering.
        //
        // Pattern `c{2,12}c{2,12}f$` with input "cccccccdcccf": the 'd' at
        // position 7 breaks all counter threads, leaving only 3 c's (not
        // enough for c{2,12}c{2,12} which needs ≥4).  NFA correctly says
        // NO MATCH.
        //
        // Root cause: DFA state 2 (nfa={0,3,6}) includes NFA state 6
        // (Byte 'f'), a post-break tail only reachable after both counters
        // break.  The transition from state 2 on 'f' is classified as
        // non-counting (probe closure doesn't encounter CInc).  The
        // non-counting branch in step_slow directly propagated
        // no_break_is_match_at_end=true (from state 6 → Assert(End) →
        // Match), without filtering through nb_counter_free_mae.
        //
        // Fix: use nb_counter_free_mae (or clean_counter_free_mae when
        // contaminated) for non-counting transitions in step_slow and
        // step_from_dead, matching the fast-path logic.
        test_tier3_non_counting_mae_false_positive {
            pattern: r"c{2,12}c{2,12}f$",
            memory: 2159,
            min_tier: 1,
            inputs: [
                ("ccf", false),               // only 2 c's, need ≥4 (2+2)
                ("cccf", false),              // only 3 c's, need ≥4 (2+2)
                ("ccccf", true),              // c{2,12}="cc", c{2,12}="cc", f
                ("cccccccdcccf", false),       // Bug 23 reproducer: 'd' breaks chain
                ("cccccccccccf", true),        // 10 c's + f: c{2,12}="cc...", c{2,12}="cc...", f
                ("ccccccccccccf", true),       // 11 c's + f
                ("cf", false),                // too few c's
                ("f", false),                 // no c's
                ("cccccccccccccccccccccccccf", true), // 24 c's (12+12) + f
            ],
        }

        // Bug 24: When contaminated (break_extras && num_counters > 1),
        // `clean_counter_free_mae` compared origins against `clean_nb`'s
        // post-transition NFA states.  When the transition's only target
        // was Assert(End) → Match (no consuming successors), `clean_nb`
        // had empty NFA states, losing the mae signal from legitimate
        // counter-free origins like `.{8,8}` → state 8.
        //
        // Pattern: ^(.{8,8}|f{5,26}|f{5,26})$ — the two f{5,26}
        // alternatives create 2 counters (triggering contamination).
        // Input "fffffabc" (8 chars): 5 f's match both f{5,26} alts
        // (counters reach min), then "abc" continues the .{8,8} path.
        // The .{8,8} alternative should match at exactly 8 chars, but
        // the contaminated cf_mae check failed because origin 8 wasn't
        // in the post-transition clean_nb state.
        test_tier3_contaminated_cf_mae_empty_nfa {
            pattern: r"^(.{8,8}|f{5,26}|f{5,26})$",
            memory: 1493,
            min_tier: 3,
            inputs: [
                ("fffffabc", true),            // Bug 24: .{8,8} via counter-free path
                ("fffffXXX", true),            // Same: 5 f's + 3 non-f's = 8 chars
                ("ffffffxx", true),            // 6 f's + 2 non-f's = 8 chars
                ("fffffffx", true),            // 7 f's + 1 non-f = 8 chars
                ("ffffffff", true),            // 8 f's: matches both .{8,8} and f{5,26}
                ("abcdefgh", true),            // 8 non-f chars: only .{8,8}
                ("fffff", true),               // 5 f's: f{5,26} matches
                ("ffffffffffffffffffffffffffffff", false), // 30 f's > 26: f{5,26} fails, .{8,8} fails
                ("ffff", false),               // 4 f's: too short for both
                ("ffffabcd", true),            // 4 f's + 4 non-f = 8 chars: .{8,8}
                ("fffffab", false),            // 7 chars: too short for .{8,8}
                ("", false),
            ],
        }

        // Regression (defense-in-depth): `break_closure()` used to follow
        // through `CounterInstance` nodes, which meant that for sequential
        // multi-counter patterns like `.{1,2}.{4,4}$`, counter A's
        // `break_is_match_at_end` flag would be set to true because the
        // break closure reached `$ → Match` through counter B's CI → body
        // → CInc → break → `$` path — even though counter B had not yet
        // accumulated any iterations.
        //
        // The fix stops `break_closure` at `CounterInstance` nodes, so
        // `break_is_match_at_end` only reflects paths reachable without
        // entering another counter.  The downstream counter's match paths
        // are handled by `break_seeds` and `post_break_tails` at runtime.
        test_tier3_break_closure_stops_at_ci {
            pattern: "^.{1,2}.{4,4}$",
            memory: 1194,
            min_tier: 1,
            inputs: [
                ("aaaaa", true),
                ("aaaaaa", true),
                ("a", false),
                ("aa", false),
                ("aaa", false),
                ("aaaa", false),
                ("aaaaaaa", false),
            ],
        }

        // Tier 3 range-compression contiguity bug: for ^-anchored patterns
        // with variable-length bodies like (a|aaa), seed injection stops
        // after byte 0 (the ^ assert blocks re-seeding).  Without the
        // "anchored at 0" property, values at a body-entry origin can be
        // non-contiguous (e.g. {1, 3} after 3 bytes — only same-parity
        // counts are reachable).  RangeCounters::insert merges these via
        // min/max into [1, 3], introducing phantom value 2.  The phantom
        // can grow into the break range and trigger a false match.
        //
        // Valid input lengths for {4,4}: s+l=4, s+3l=k ⟹ k=4+2l, so
        // k ∈ {4, 6, 8, 10, …}.  Odd-length 'a' runs (5, 7) are invalid.
        test_tier3_range_contiguity_anchored {
            pattern: r"^(a|aaa){4,4}b",
            memory: 1367,
            min_tier: 1,
            inputs: [
                ("aaaab", true),
                ("aaaaab", false),
                ("aaaaaab", true),
                ("aaaaaaab", false),
            ],
        }

        // Same contiguity bug for unanchored patterns with a consuming
        // prefix before the counter.  The unanchored loop re-seeds the
        // 'c' state every byte, but seeds only enter the counter body
        // when 'c' is actually consumed — not every step.  Between 'c'
        // positions the counter runs without fresh value-0 seeds, so
        // the same parity-gap phantoms appear.
        test_tier3_range_contiguity_prefix {
            pattern: r"c(a|aaa){4,4}b",
            memory: 1367,
            min_tier: 1,
            inputs: [
                ("caaaab", true),
                ("caaaaab", false),
                ("caaaaaab", true),
                ("caaaaaaab", false),
            ],
        }

        // Regression: the `a+` self-loop causes `advance_origins` and
        // `continue_origins` to contain the same NFA state.  Without
        // dedup in `InstanceCounters::advance` / `insert_continued`,
        // entries doubled every byte (1→2→4→8→…), overflowed the
        // fixed-stride buffer, and silently dropped high-value instances
        // — producing false negatives on long inputs.
        test_tier3_instance_dedup_self_loop {
            pattern: r".a+{9,28}",
            memory: 1063,
            min_tier: 3,
            inputs: [
                ("aaaaaaaaa", false),
                ("aaaaaaaaaa", true),
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", true),
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", true),
            ],
        }

        // Regression: `counter_free_nb_mae` had an early-return guard
        // that checked `reachable_without_break[start]`, but the start
        // state (a Split) is never consuming and thus never appears in
        // `reachable_without_break`.  This blocked the counter-free
        // `$ → Match` path for optional groups, producing false negatives
        // when the optional group didn't match but `$` should have.
        // The `?` makes every input match via the empty-to-`$` path.
        test_tier3_counter_free_mae_optional_group {
            pattern: r"([b-e].+{4,43})?$",
            memory: 1385,
            min_tier: 3,
            inputs: [
                ("zzz", true),
                ("", true),
                ("baaa", true),
                ("xyz123", true),
            ],
        }

        // Regression: false positive on anchored optional group with \B
        // assertion inside the body.  Resolved seeds from break-path
        // deferred assertions in with_break DFA states were treated as
        // unconditional, suppressing the corresponding break-gated seed
        // and causing the seed to fire on every transition — even when
        // the triggering counter didn't break (Bug 14).
        test_tier3_optional_anchored_false_positive {
            pattern: r"^(a?.{2,2}\Bx{2,2})?$",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("", true),
                ("xxxx", true),
                ("xxxxxx", false),
                ("xxxxxxxx", false),
                ("xxxxxxxxxx", false),
                ("xxxxx", false),
            ],
        }

        // Bug 25: Tier 3 false positive from contaminated no_break_is_match.
        //
        // Pattern `^.{0,46}x{10,35}f` has two counters: c0 (`.{0,46}`) and
        // c1 (`x{10,35}`).  The DFA state includes `Byte('f')` (state 8)
        // via c1's break closure (break_extras).  When byte 'f' arrives,
        // state 8 consumes it → Match, setting `no_break_is_match` on the
        // transition.  But c1 never reached its minimum (10): the instance
        // dies because 'f' ≠ 'x'.  The contaminated `no_break_is_match`
        // flag must be replaced by the clean chain's flag.
        test_tier3_contaminated_no_break_is_match {
            pattern: r"^.{0,46}x{10,35}f",
            memory: 1196,
            min_tier: 3,
            inputs: [
                ("xxxxxxxxxf", false),          // 9 x's — c1 max count 9 < min 10
                ("xxxxxxxxxxf", true),           // 10 x's — c1 reaches min
                ("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxf", true), // 35 x's — . absorbs extras
                ("f", false),                    // no x's at all
                ("xxxxxxxxxx", false),           // 10 x's but no trailing 'f'
            ],
        }

        // Bug 26: Tier 3 false negative — counter-break deferred asserts
        // not resolved mid-input.
        //
        // Pattern `.{2,38}c{4,10}\B` has a `\B` assertion gated behind
        // c1's break path.  When c1 breaks with count >= 4, the deferred
        // assert `\B` is emitted but was only evaluated at end-of-input.
        // At end-of-input, `\B` fails (prev='c' word, next=end non-word).
        // Mid-input with a following word char (e.g. 'a'), `\B` should
        // pass (prev='c' word, next='a' word → non-boundary).
        test_tier3_counter_break_deferred_assert_mid_input {
            pattern: r".{2,38}c{4,10}\B",
            memory: 1558,
            min_tier: 2,
            inputs: [
                ("cccccca", true),              // 6c + a: c1 breaks, \B(c,a) passes
                ("cccccc1", true),              // 6c + digit: \B(c,1) passes
                ("cccc ", false),               // 4c + space: \B(c, ) fails (boundary)
                ("ccccc", false),               // 5c at EOI: \B(c,end) fails
                ("ccc", false),                 // 3c: c0 min=2 ok but c1 min=4 not reached
            ],
        }

        // Bug 27: Tier 3 false positive — break seeds unconditionally
        // seeded through deferred assertion.
        //
        // Pattern `^.{0,13}\b.{2,2}$` has a `\b` between c0's break and
        // c1.  The break seed `trigger:c0 → seed c1 at origin:6` was
        // applied unconditionally when c0 broke, ignoring the `\b` gate.
        // For input "abc", the `\b` between positions should fail (both
        // sides are word chars) but c1 was seeded anyway, leading to a
        // false match_at_end.
        test_tier3_break_seed_deferred_assert_gate {
            pattern: r"^.{0,13}\b.{2,2}$",
            memory: 1920,
            min_tier: 1,
            inputs: [
                ("abc", false),                 // all word chars — no \b boundary
                ("abcd", false),                // all word chars — no \b boundary
                (" ab", true),                  // \b between ' ' and 'a'
                ("a b", true),                  // \b between 'a' and ' '
                ("ab ", false),                 // \b(a,b)=no, \b(b, )=yes but .{2,2}$ fails
                ("a  ", true),                  // \b between 'a' and ' '
            ],
        }

        // Bug 28: break-seed deferred assertion evaluated at wrong position.
        // The `\b` on the break path from c0 to c1 was evaluated with
        // `prev=byte, next=byte` (always false for `\b`).  The assertion
        // must be deferred to the next byte: `prev=byte, next=next_byte`.
        // Without the fix, `\b` never passes in break seeds, and c1 is
        // never seeded through that path.
        test_tier3_pending_break_seed {
            pattern: r"^f{3,4}\b.{5,10}a?$",
            memory: 1722,
            min_tier: 1,
            inputs: [
                ("fff abcde", true),            // \b between 'f' and ' '
                ("fff abcd", true),             // \b between 'f' and ' '
                ("ffffffff", false),            // no \b (all word chars)
                ("fff     ", true),             // \b between 'f' and ' '
                ("fffabcde", false),            // no \b (f/a both word chars)
                ("fff", false),                 // too short for c1
                ("ffff bcde", true),            // 4 f's, \b between 'f' and ' '
            ],
        }

        // Bug 28 variant: `\B` (non-word-boundary) on break path.
        // The break seed for c1 passes through `\B`.  Same deferred
        // evaluation applies — `\B(byte, byte)` is always true, so
        // without the fix, `\B`-gated seeds fire incorrectly.
        test_tier3_pending_break_seed_non_word_boundary {
            pattern: r"^(a?.{2,2}\Bx{2,2})?$",
            memory: 1227,
            min_tier: 1,
            inputs: [
                ("xxxx", true),                 // \B between x and x (same word class)
                ("aa  ", false),                // \B between ' ' and ' ' would pass, but no 'x'
                ("", true),                     // empty matches via outer `?`
                ("aaxx", true),                 // a, then .{2,2}=ax, \B(x,x), xx
            ],
        }

        // Bug 29: post-break tail Advance branch missing target_is_match check.
        // The tail at state 8 (Byte('e')) consumes 'e' and advances to {9}
        // (new_origins), but the epsilon closure of its target includes
        // Match (state 11 via Split at 10).  The Advance branch checked
        // target_is_match_at_end but not target_is_match, so the direct
        // match was missed — producing a false negative.
        test_tier3_post_break_tail_advance_match {
            pattern: r"^c{3,5}c{1,5}cee?",
            memory: 1334,
            min_tier: 1,
            inputs: [
                ("ccccce", true),               // ccc + c + c + e (+ empty e?)
                ("cccccce", true),              // ccc + cc + c + e
                ("ccccccee", true),             // ccc + ccc + c + ee
                ("cccccee", true),              // ccc + cc + c + ee
                ("cccce", false),               // only 4 c's: ccc + c = min, no c left for literal
                ("ccce", false),                // 3 c's + e: c{3,5}=ccc, no c left for c{1,5}
            ],
        }

        // -- Bug 32 regression: contaminated counter_free_seeds false positive --
        // Pattern `^((a{8,13}d*)?.?a?.{6,6}a?)?$` has two counters:
        // c0={8,13} and c1={6,6}.  When c0 breaks, the with_break DFA state
        // includes origins {4,7,9} from the break chain.  Origins 7,9 are
        // globally `reachable_without_break` (reachable from `^` via the
        // optional skip), so the `counter_free_seeds` computation in
        // `populate()` treats them as unconditional — seeding c1 on every
        // byte from the contaminated state.  This causes c1 instances to
        // roll indefinitely even after c0 dies (position 13+), producing
        // a false match_at_end.
        //
        // Fix: at runtime, on counting transitions from a contaminated
        // state, filter unconditional seeds against the clean_nb chain's
        // transition seeds.  Only seeds also present in the uncontaminated
        // chain are truly counter-free.  Non-counting transitions are NOT
        // filtered because they have no break_seeds fallback.
        test_tier3_contaminated_counter_free_seeds {
            pattern: r"^((a{8,13}d*)?.?a?.{6,6}a?)?$",
            memory: 2085,
            min_tier: 1,
            inputs: [
                ("aaaaaaaaa", true),                                     // 9 'a's: matches (a{8}+.?=a → 9)
                ("aaaaaaaaaa", false),                                   // 10 'a's: no valid partition
                ("a]]]]]", true),                                       // 6 chars: skip c0, .?a?.{6,6}a? can match
                ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false), // 50 'a's: false positive in Bug 32
                ("aaaaaaaaaaaaaa", true),                                // 14 'a's: a{8}+.?a?.{6,6}a?
                ("aaaaaaaaaaaaaaaaaaaa", true),                          // 20 'a's: a{13}+.?.{6,6}
                ("aaaaaaaaaaaaaaaaaaaaa", true),                         // 21 'a's: a{13}+.?a?.{6,6}a?
            ],
        }

        // -- Bug 33 regression: OOM in can_reach_match epsilon walk --
        // Patterns containing `(\b|\B)*` create epsilon cycles in the NFA:
        // the `*`'s Split loops back through the alternation, and since
        // either `\b` or `\B` always passes, the walk never terminates
        // without a visited set.  Both `can_reach_match_at_end` (mod.rs)
        // and `can_reach_match_mid` (tier3.rs) were missing visited sets,
        // causing OOM during deferred assertion resolution.
        //
        // Fix: add a `visited` bitvec to both epsilon walk functions.
        test_epsilon_cycle_assert_oom {
            pattern: r"^a{2,5}(\b|\B)*b$",
            memory: 1136,
            min_tier: 1,
            inputs: [
                ("aab", true),
                ("aaab", true),
                ("aaaab", true),
                ("aaaaab", true),
                ("ab", false),       // too few a's
                ("aaaaaab", false),   // too many a's
                ("aa", false),        // no trailing b
                ("", false),
            ],
        }

        // -- Bug 35 regression: ByteTable skipped in break_consuming_tails --
        // Pattern `^(c{2,48}c?y?b1)?((a?a?)?(a?a?)?)?$` has a counter c0
        // for `c{2,48}`.  When c0 breaks, the break path goes to state 5
        // (ByteTable for `c?y?b1`).  break_consuming_tails was skipping
        // ByteTable states entirely, so state 5 never became a
        // post_break_tail.  The downstream consuming chain (y→b→1) was
        // never tracked, and target_is_match_at_end[9] (for state 9, the
        // '1' byte leading to `$→Match`) was never checked → false negative.
        //
        // Fix: include ByteTable states in break_consuming_tails.
        test_tier3_byte_table_break_tail {
            pattern: r"^(c{2,48}c?y?b1)?((a?a?)?(a?a?)?)?$",
            memory: 2425,
            min_tier: 2,
            inputs: [
                ("cccccccccccccccccccccccccccccccccccccccccyb1", true), // 41 c's + yb1: Bug 35 crash case
                ("ccyb1", true),           // 2 c's + yb1
                ("ccb1", true),            // 2 c's + b1 (skip optional y)
                ("ccyb1aa", true),         // 2 c's + yb1 + aa
                ("cb1", false),            // 1 c (< min 2)
                ("", true),               // empty matches via outer ?
                ("cccccccccccccccccccccccccccccccccccccccccccccccccccyb1", false), // 51 c's (> max 48+1)
            ],
        }

        // -- Bug 34 regression: deferred asserts lost on None target in post_break_tails --
        // Pattern `^1{2,30}c{9,48}y\b$` has two counters.  When c1 breaks,
        // the break path goes to state 7 (Byte('y')), which is tracked as
        // a post_break_tail.  When 'y' is consumed, state 7's target is
        // state 8 (Assert(\b)), which leads only to Assert($) → Match.
        // analyze_target returns None for state 8 (no consuming states),
        // so the action is None.  The post_break_tails code's Some(None)
        // arm was not depositing target_deferred_asserts, losing the \b
        // assertion → false negative at EOI.
        //
        // Fix: deposit target_deferred_asserts in the Some(None) arm.
        test_tier3_tail_none_target_deferred_asserts {
            pattern: r"^1{2,30}c{9,48}y\b$",
            memory: 973,
            min_tier: 2,
            inputs: [
                ("1111111cccccccccccccy", true),    // 7 ones + 13 c's + y: Bug 34 crash case
                ("11ccccccccccy", true),             // 2 ones + 10 c's + y
                ("1111111ccccccccccy", true),        // 7 ones + 10 c's + y
                ("1ccccccccccy", false),             // 1 one (< min 2)
                ("11ccccccccy", false),              // 2 ones + 8 c's (< min 9)
                ("", false),
                ("1111111cccccccccccccy ", false),   // trailing space breaks \b
            ],
        }

        // -- Bug 31 regression: chained assertions in target_deferred_asserts --
        // Pattern has `.\b\B` after counter break.  State 4's target epsilon
        // path goes through \b@5 then \B@6.  If target_deferred_asserts
        // deposits both independently, \B passes alone at EOI (non-word →
        // non-word), giving a false positive even though \b fails.
        //
        // Fix: target_deferred_asserts records only the FIRST assert on each
        // path (here: \b@5).  `resolve_verified_deferred_asserts` evaluates
        // \b@5 first, and only calls `can_reach_match_at_end(\b.out)` if it
        // passes, which properly handles the downstream \B chain.
        test_tier3_chained_assert_target_deferred {
            pattern: r"^.{7,7}(.\b\Ba?(a?a?)?)?$",
            memory: 1557,
            min_tier: 1,
            inputs: [
                ("cy1aacc", true),              // 7 chars, optional group empty, $ matches
                ("cy1aacc ", false),             // 8 chars: .=space, \b fails (space→EOI both non-word)
                ("abcdefg", true),              // exactly 7, optional group empty
                ("abcdefgh", false),            // 8 chars: .=h, \b fails (h→EOI: word→non-word passes)
                                                // but \B requires same category → fails.  \b\B always false.
            ],
        }

        // -- Bug 36 regression: tail→CInc handoff break not setting any_can_break --
        // Pattern `^(.{1,3}.{0,0} )+$` has one counter c0 {1,3} for
        // `.{1,3}`.  The `+` loop re-enters via Split → CI(c0) → body.
        // After matching "ax " (c0 breaks at 2, ' ' consumed via
        // post_break_tail [4]), the tail chain Advance → [1] creates a
        // new tail at origin 1.  On the next 'a', this tail hits CInc
        // and the handoff produces a break (value 0→1 ≥ min 1), adding
        // break_consuming_states [4].  But `any_can_break` was only set
        // by the counter entry loop (which had 0 entries), so the DFA
        // selected `no_break` (nfa={1}), dropping origin 4 from the
        // state.  At the next ' ', tail [4]'s origin was not in
        // origin_keys → dropped → false negative.
        //
        // Fix: set `any_can_break` and `counter_broke` in the tail→CInc
        // Increment arm when value >= min, so the DFA selects `with_break`.
        test_tier3_tail_cinc_handoff_break {
            pattern: r"^(.{1,3}.{0,0} )+$",
            memory: 1194,
            min_tier: 1,
            inputs: [
                ("ax a ", true),            // Bug 36 minimal crash case: 2-char body + space, 1-char body + space
                ("ax a ca f c0 c 1 c ", true), // Bug 36 original fuzz input (len=19)
                ("a ", true),               // single iteration: 1 char + space
                ("ab ", true),              // single iteration: 2 chars + space
                ("abc ", true),             // single iteration: 3 chars + space (max)
                ("a b ", true),             // two iterations: 1+space, 1+space
                ("abc ab a ", true),        // three iterations with different body lengths
                ("abcd ", false),           // 4 chars before space exceeds max 3
                ("", false),               // empty: + requires ≥1 iteration
                (" ", false),              // space only: body needs ≥1 char
            ],
        }

        // -- Bug 37 regression: ByteTable tail target_is_match_at_end skipped --
        // Pattern `^0{3,43}((a?c)*x(a?a?)?(a?a?)?)?$` has a ByteTable at
        // state 7 with entries 'a'→6, 'c'→7, 'x'→13.  After c0 breaks,
        // state 7 appears as a post_break_tail.  Consuming 'x' through
        // state 7 leads to state 13 (Split → optional a? chains → $ → Match).
        // The static target_is_match_at_end array skipped ByteTable states
        // (per-byte targets made a single static flag unsound), so
        // target_is_match_at_end[7] was false.  The check_tail_match_flags
        // macro never set match_at_end → false negative on "000x".
        //
        // Fix: store is_match_at_end on the Advance action variant (computed
        // per-byte-target in analyze_target), and use it in the Advance arm
        // of tail processing instead of the static array.
        test_tier3_byte_table_tail_mae {
            pattern: r"^0{3,43}((a?c)*x(a?a?)?(a?a?)?)?$",
            memory: 2359,
            min_tier: 2,
            inputs: [
                ("000x", true),                // Bug 37 crash case: 3 zeros + x
                ("000", true),                 // 3 zeros, outer ? skips group, $ matches
                ("000acx", true),              // 3 zeros + ac loop + x
                ("000xa", true),               // 3 zeros + x + a
                ("000xaa", true),              // 3 zeros + x + aa
                ("000xaaaa", true),            // 3 zeros + x + aaaa (both optional groups)
                ("00x", false),                // 2 zeros (< min 3)
                ("000b", false),               // 3 zeros + b (not in ByteTable)
                ("000xaaaaa", false),          // too many a's for optional groups
                ("", false),                   // empty
            ],
        }

        // -- Bug 38 regression: phase-2 break_seeds fire one byte too early --
        // Pattern `^e{4,5}e{4,5}ee{4,5}$` requires 13-16 e's.  When c1
        // breaks at value 4, the break path crosses a consuming state
        // (literal 'e' at state 7) before reaching c2's CounterInstance.
        // Phase-2 break_seeds (which follow consuming states' targets to
        // find downstream CIs) seeded c2 immediately at the break byte,
        // bypassing state 7's byte consumption.  This gave c2 one extra
        // counting step, producing a false positive on 12 e's.
        //
        // Fix: remove phase-2 break_seeds entirely.  The tail mechanism
        // (break_consuming_tails + post_break_tails) handles multi-hop
        // break paths with correct byte timing.
        test_tier3_phase2_break_seed_false_positive {
            pattern: r"^e{4,5}e{4,5}ee{4,5}$",
            memory: 1334,
            min_tier: 1,
            inputs: [
                ("eeeeeeeeeeee", false),       // 12 e's: Bug 38 crash case (< min 13)
                ("eeeeeeeeeeeee", true),        // 13 e's: 4+4+1+4 = minimum
                ("eeeeeeeeeeeeee", true),       // 14 e's
                ("eeeeeeeeeeeeeee", true),      // 15 e's
                ("eeeeeeeeeeeeeeee", true),     // 16 e's: 5+5+1+5 = maximum
                ("eeeeeeeeeeeeeeeee", false),   // 17 e's: > max 16
                ("", false),                    // empty
            ],
        }

        // -- Bug 39 regression: dead target loses match_at_end for ByteTable tails --
        // Pattern `^(.{6,36}a{0,4}e*c)?$` has a ByteTable at state 11
        // (unrolled `a{0,4}` → entries 'a', 'c', 'e').  After c0 breaks,
        // state 11 is a post_break_tail.  Consuming 'c' → state 16
        // (Assert(End) → Match) should set match_at_end.  But
        // analyze_target(16) returned None (no consuming states) and the
        // tail processing's Some(None) branch used the static
        // target_is_match_at_end array — always false for ByteTable.
        //
        // Fix: return Advance { new_origins: [], is_match_at_end: true }
        // from analyze_target when the epsilon walk reaches $ → Match
        // even with no consuming states downstream.
        test_tier3_dead_target_byte_table_mae {
            pattern: r"^(.{6,36}a{0,4}e*c)?$",
            memory: 5555,
            min_tier: 2,
            inputs: [
                ("xxxcbd ccac", true),          // Bug 39 crash case
                ("xxxxxxc", true),              // 6 wildcards + c (minimum)
                ("xxxxxxac", true),             // 6 wildcards + a + c
                ("xxxxxxaaac", true),           // 6 wildcards + aaa + c
                ("xxxxxxaaaaec", true),         // 6 wildcards + aaaa + e + c
                ("xxxxx", false),               // 5 chars, no c (< min 6 wildcards)
                ("xxxxxxx", false),             // 7 wildcards but no c
                ("", true),                     // empty (outer ? skips group, $ matches)
            ],
        }

        // -- Bug 40 regression: assertion-gated counter seeding false positive --
        // The {3,3} outer repetition is unrolled into three counter copies.
        // Each has structure: ByteClass → Split(optional '1') → Assert(\B)
        // → CI(counter) → body → CInc.  The with_break DFA state has \B
        // as a deferred assertion; when resolved, it produces seeds for
        // downstream counters (c1, c2).  These resolved seeds were merged
        // into the transition's unconditional seed list, causing c1/c2
        // to be seeded on every transition from the with_break state —
        // even when the triggering counter (c0) hadn't actually broken.
        //
        // Fix: treat resolved seeds whose origin is not
        // reachable_without_break as break-gated (suppress from
        // unconditional seeds).  The tail mechanism (post_break_tails +
        // tail-to-counter handoff) handles the actual seeding with
        // correct byte timing.
        test_tier3_assert_gated_counter_seeding {
            pattern: r"^((.1?\B.{3,28}){3,3}|(a?a?)?)$",
            memory: 1857,
            min_tier: 3,
            inputs: [
                ("aaaaaaaaaa", false),            // Bug 40 crash: 10 a's, too short for 3 reps
                ("aaaaaaaaaaaa", true),            // 12 a's: 3 reps × (1 prefix + 3 body) = 12
                ("aaaaaaaaaaaaa", true),           // 13 a's: extra body bytes absorbed
                ("a1aa1aa1aaa", false),            // has '1' but \B fails at word boundaries
                ("", true),                        // empty: outer ? skips, $ matches
                ("a", true),                       // a?a? matches 'a'
                ("aa", true),                      // a?a? matches 'aa'
                ("aaa", false),                    // a?a? max 2 chars, {3,3} needs \B-gated body
            ],
        }

        // -- Bug 30 regression: contaminated no_break_current deferred asserts --
        // When c0 (.{0,44}) breaks, the DFA state inherits state 8 (Byte 'f')
        // from c1's break path.  State 8→9 (\b) ends up as a deferred assert
        // in no_break_current.  For "a11f" (c1 count=2 < min=3), the \b fires
        // spuriously at EOI via no_break_current → false positive.
        //
        // Fix: when contaminated, skip no_break_current deferred asserts
        // entirely; legitimate deferred asserts from post-break tails are
        // deposited via target_deferred_asserts into verified_deferred_asserts.
        test_tier3_contaminated_deferred_assert {
            pattern: r"^.{0,44}1{3,38}f\ba?$",
            memory: 1328,
            min_tier: 3,
            inputs: [
                ("111f", true),                 // c1=3 ≥ min, break, 'f' consumed, \b passes at EOI
                ("1111f", true),                // c1=4 ≥ min
                ("a111f", true),                // c0=1 ('a'), c1=3, break, 'f', \b at EOI
                ("a11f", false),                // c1=2 < min=3, no break → no match
                ("11f", false),                 // c1=2 < min=3
                ("1f", false),                  // c1=1 < min=3
                ("f", false),                   // c1=0 < min=3
                ("111fa", false),               // \b between f(word) and a(word) fails
            ],
        }

        // Bug 41: stale verified_deferred_asserts surviving the fast path.
        // When a counter breaks and deposits a deferred `\b` assert, the
        // assert is resolved on the NEXT byte.  If the assert fails
        // (e.g. c→z = word→word, no boundary), the entry must be cleared.
        // Previously, the inline fast path (no live counters, no seeds)
        // did not clear verified_deferred_asserts, allowing stale entries
        // to persist until EOI where they would incorrectly pass.
        test_tier3_stale_deferred_assert_fast_path {
            pattern: r"\bc{2,12}\b",
            memory: 1433,
            min_tier: 1,
            inputs: [
                ("cc", true),                      // 2 c's, \b at both ends
                ("cccccccccccc", true),             // 12 c's (max), \b at EOI
                ("ccccccccccccz", false),           // 12 c's + z: no \b after c's (word→word)
                ("cccccccccccczzz", false),         // 12 c's + zzz: same, stale deferred \b at EOI
                ("ccccccccccccc", false),           // 13 c's: exceeds max, no match
                ("cccccccccccc ", true),            // 12 c's + space: \b passes (word→non-word)
                ("ccc", true),                      // 3 c's, boundary at both ends
                ("c", false),                       // 1 c: below min
                (" cccc ", true),                   // spaces around: \b passes at both boundaries
                (" ccccccccccccz ", false),         // 12 c's + z after space: no \b at c/z boundary
            ],
        }

        // ---------------------------------------------------------------
        // Coverage expansion: lock down Tier 3 code paths that were
        // previously untested or only partially tested.  These tests
        // exist to catch regressions when fixing future bugs.
        // ---------------------------------------------------------------

        // PATH 14: finish() pending break seeds at EOI.
        // When c0 breaks on the last input byte, the break seed for c1
        // has a deferred \b that can't be evaluated yet (no next byte).
        // It goes into pending_break_seeds and is resolved in finish().
        // With unroll_limit=0, .{0,3} stays as a counter.  c1 has min=0,
        // so value=0 >= min → immediate break → break_is_match_at_end.
        test_tier3_cov_pending_break_seed_eoi {
            pattern: r"^c{3,5}\b.{0,3}$",
            memory: 1425,
            min_tier: 1,
            inputs: [
                ("ccc", true),           // c0=3, \b pending at EOI, passes (word→end), c1 seed value=0 ≥ min=0
                ("cccc", true),          // c0=4
                ("ccccc", true),         // c0=5 (max)
                ("cc", false),           // c0=2 < min=3
                ("cccccc", false),       // c0=5 then extra 'c': c0 max exceeded
                ("ccc ", true),          // c0=3, \b(c, ' ') passes mid-input, c1 seeded, ' ' consumed
                ("ccc  ", true),         // c0=3, \b mid, c1 counts 2 spaces
                ("ccc   ", true),        // c0=3, \b mid, c1 counts 3 spaces
                ("ccc    ", false),      // c0=3, \b mid, c1 max=3 exceeded
                ("ccca", false),         // \b(c,a) fails (both word)
            ],
        }

        // PATH 2a: tail→CInc handoff where value+1 < min (continue only).
        // c0 (.{2,5}) breaks, tail 'b' consumed, tail hits c1's CInc with
        // value 0→1 < min 3.  Instance continues counting (no immediate break).
        test_tier3_cov_tail_cinc_handoff_continue {
            pattern: r"^.{2,5}bc{3,6}$",
            memory: 1557,
            min_tier: 1,
            inputs: [
                ("aabccc", true),        // c0=2, tail 'b', handoff to c1, c1 counts to 3
                ("aabcccccc", true),      // c0=2, c1=6 (max)
                ("aabcc", false),         // c1=2 < min=3
                ("aab", false),           // no bytes for c1
                ("aabccccccc", false),    // c1=7 > max=6
                ("aaaaabccc", true),      // c0=5, tail 'b', handoff c1=3
                ("aaaaabcccccc", true),   // c0=5, c1=6
                ("abccc", false),         // c0=1 < min=2
            ],
        }

        // PATH 2b: tail→CInc handoff with break_deferred_asserts.
        // c0 (a{3,5}) breaks, tail 'b' consumed, tail hits c1's CInc.
        // c1 (.{1,4}) has min=1, so the handoff (value 0→1) immediately
        // breaks.  The break path has \b as a deferred assert.
        test_tier3_cov_tail_cinc_handoff_break_deferred {
            pattern: r"^a{3,5}b.{1,4}\b$",
            memory: 1491,
            min_tier: 1,
            inputs: [
                ("aaabx", true),         // c0=3, tail 'b', handoff c1=1, break, \b at EOI passes
                ("aaabxy", true),        // c1=2, \b at EOI
                ("aaabxyz", true),       // c1=3
                ("aaabxyzw", true),      // c1=4 (max)
                ("aaab", false),         // no byte for c1 body
                ("aab", false),          // c0=2 < min=3
                ("aaaaabx", true),       // c0=5 (max), tail 'b', handoff c1=1
                ("aaabxyzwq", false),    // c1=5 > max=4
            ],
        }

        // PATH 2c: multi-hop tail chain (tail→CInc→break→new tails).
        // c0 (a{3,5}) breaks, tail 'b' consumed, tail hits c1's CInc
        // (.{1,3}).  c1 has min=1, so the handoff immediately breaks.
        // The break path has consuming states 'c','d'.  These become
        // new post-break tails that must advance through 'c' then 'd'.
        test_tier3_cov_multi_hop_tail_chain {
            pattern: r"^a{3,5}b.{1,3}cd$",
            memory: 1458,
            min_tier: 1,
            inputs: [
                ("aaabxcd", true),       // c0=3, tail 'b', c1 handoff, c1=1 breaks, tails 'c','d'
                ("aaabxycd", true),      // c1=2
                ("aaabxyzcd", true),     // c1=3 (max)
                ("aaabcd", false),       // no byte for c1 body (c1 min=1 requires at least 1)
                ("aaab", false),         // no bytes after tail 'b'
                ("aaabxyzwcd", false),   // c1=4 > max=3
                ("aab", false),          // c0=2 < min=3
            ],
        }

        // PATH 1a: Advance tail with target_deferred_asserts resolved
        // mid-input.  c0,c1 (.{2,5} each) break, tail 'c' consumed,
        // Advance deposits \b, resolved on NEXT byte (' ') mid-input.
        test_tier3_cov_advance_tail_deferred_mid_input {
            pattern: r"^.{2,5}.{2,5}c\b x",
            memory: 1590,
            min_tier: 1,
            inputs: [
                ("aaaac x", true),       // c0=2,c1=2, tail 'c', \b(c,' ') passes, ' ' consumed, 'x' consumed
                ("aaaaac x", true),      // c0+c1 split across 5 chars
                ("aaaaaac x", true),     // c0=3,c1=3
                ("aaaacax", false),      // \b(c,a) fails (both word)
                ("aac x", false),        // total=2+2+3=7 min, "aac x" is 5 chars: too short
            ],
        }

        // PATH 5a: counter break \b deferred assert resolved mid-input.
        // c0 (c{4,10}) breaks, \b deposited in verified_deferred_asserts,
        // resolved on the NEXT byte (' ').
        test_tier3_cov_counter_break_wb_mid_input {
            pattern: r"^c{4,10}\b x",
            memory: 1301,
            min_tier: 1,
            inputs: [
                ("cccc x", true),        // c0=4, \b(c,' ') passes mid-input, ' ' and 'x' consumed
                ("ccccc x", true),       // c0=5
                ("cccccccccc x", true),  // c0=10 (max)
                ("ccccx", false),        // \b(c,x) fails (both word)
                ("ccc x", false),        // c0=3 < min=4
                ("ccccccccccc x", false), // c0=11 > max=10
            ],
        }

        // PATH 3b: None target with deferred asserts that FAIL.
        // c0,c1 break, tail 'y' consumed, None target deposits \B,
        // \B at EOI fails (word→end = boundary, \B requires non-boundary).
        test_tier3_cov_none_target_deferred_fail {
            pattern: r"^.{2,30}c{9,48}y\B$",
            memory: 1229,
            min_tier: 3,
            inputs: [
                ("aaccccccccccy", false),    // c0=2, c1=10, tail 'y', \B at EOI fails
                ("aacccccccccccccy", false),  // c0=2, c1=13
                ("aacccccccccy", false),      // c0=2, c1=9 (min)
                ("aaccccccccy", false),       // c1=8 < min=9
            ],
        }

        // PATH 13a: contaminated clean_nb deferred assert that PASSES.
        // The x+ alternation provides a counter-free path to \b$.
        // When both counters break (contaminated state), clean_nb has
        // the x+ path's deferred \b.  clean_nb.resolve_deferred_at_end
        // should return true.
        test_tier3_cov_contaminated_clean_nb_deferred_pass {
            pattern: r"^(.{3,10}.{3,10}|x+)\b$",
            memory: 2217,
            min_tier: 1,
            inputs: [
                ("xxx", true),           // x+ path, \b at EOI, counter-free
                ("aaaaaa", true),         // counter path 3+3=6, contaminated, clean_nb \b passes
                ("aaaaaaaaaa", true),     // counter path 5+5=10
                ("aaa", false),           // too short for both counters (min 3+3=6)
                ("aa", false),            // even shorter
                ("", false),             // empty: x+ needs at least 1
            ],
        }

        // PATH 6a: contamination correctly resolves to a TRUE match.
        // 2 counters → contaminated state.  Inputs exercise the case
        // where contamination is present and the match is legitimate.
        test_tier3_cov_contamination_true_match {
            pattern: r"^.{3,5}.{3,5}a?$",
            memory: 1491,
            min_tier: 1,
            inputs: [
                ("aaaaaa", true),        // 3+3=6, both complete, a? skips
                ("aaaaaaa", true),       // 3+4 or 4+3
                ("aaaaaaaa", true),      // 4+4
                ("aaaaaaaaa", true),     // 4+5 or 5+4
                ("aaaaaaaaaa", true),    // 5+5
                ("aaaaa", false),        // 5 chars: can't split into 3+3 minimum
                ("aaa", false),          // too short
            ],
        }

        // PATH 15a: verified_deferred_asserts at EOI that FAILS.
        // Counter breaks, tail deposits \B into verified_deferred_asserts.
        // At EOI, \B(word, end) fails — word→end is a boundary.
        test_tier3_cov_verified_deferred_fail_eoi {
            pattern: r"^.{3,5}b\B$",
            memory: 1260,
            min_tier: 1,
            inputs: [
                ("aaab", false),         // \B at word(b)→EOI fails
                ("aaaab", false),        // c0=4
                ("aaaaab", false),       // c0=5 (max)
                ("aab", false),          // c0=2 < min=3
            ],
        }

        // PATH 16a: inline fast path with contamination.
        // After both counters complete, subsequent bytes (or EOI) go
        // through the fast path while the state is contaminated.
        // Tests that cf_mae uses clean_nb_cf_mae correctly.
        test_tier3_cov_fast_path_contaminated {
            pattern: r"^(.{3,5}.{3,5})?$",
            memory: 1458,
            min_tier: 1,
            inputs: [
                ("aaaaaa", true),        // 3+3, ? matches, $
                ("aaaaaaa", true),       // 3+4
                ("aaaaaaaaaa", true),    // 5+5
                ("", true),              // ? skips entirely
                ("aaaaa", false),        // 5 chars: can't split into 3+3
                ("aaaaaaaaaaa", false),  // 11 chars: 5+5=10 max + 1 extra
            ],
        }
    }

    /// Tier 2 encodes counter identity in `u64` bitmasks, so patterns with
    /// more than 64 counters must fall through to tier 3.
    #[test]
    fn test_tier2_counter_limit_64() {
        // Helper: build "^<ch1>{2}<ch2>{2}...$" for `n` disjoint single-byte
        // counters.  Uses a-z, A-Z, 0-9, then hex-escaped byte singletons
        // so we never need to worry about regex metacharacter escaping.
        fn build_n_counter_pattern(n: usize) -> String {
            // 26 + 26 + 10 = 62 plain chars, then hex-escaped singletons.
            let plain: Vec<char> = ('a'..='z').chain('A'..='Z').chain('0'..='9').collect();
            let mut pat = String::from("^");
            for i in 0..n {
                if i < plain.len() {
                    pat.push(plain[i]);
                } else {
                    // Use a byte value that doesn't collide with the plain set.
                    // 0x80..0xFF are all > 127, so disjoint from ASCII letters/digits.
                    let byte_val = 0x80 + (i - plain.len());
                    assert!(byte_val <= 0xFF, "ran out of disjoint bytes");
                    pat.push_str(&format!("[\\x{:02x}]", byte_val));
                }
                pat.push_str("{2,3}");
            }
            pat.push('$');
            pat
        }

        let re64 = build_regex_with_unroll(&build_n_counter_pattern(64), 0);
        assert!(
            re64.tier2_eligible,
            "64-counter pattern should be tier 2 eligible"
        );

        let re65 = build_regex_with_unroll(&build_n_counter_pattern(65), 0);
        assert!(
            !re65.tier2_eligible,
            "65-counter pattern should NOT be tier 2 eligible"
        );
        assert!(
            !re65.tier3_eligible,
            "65-counter pattern should NOT be tier 3 eligible (u64 bitmask limit)"
        );
        assert!(
            re65.tier4_eligible,
            "65-counter pattern should fall to tier 4"
        );
    }

    /// `\d\d` — two identical predefined classes share one lookup table.
    ///
    /// Without dedup this would allocate two 256-byte tables; with dedup
    /// the memory is the same as `\d` plus one extra `ByteClass` state.
    #[test]
    fn test_dedup_same_class() {
        let one = build_regex_unchecked(r"^\d$");
        let two = build_regex_unchecked(r"^\d\d$");
        // The second \`\d\` adds one ByteClass state and one Catenate
        // join — so the difference is exactly one State plus one bool
        // in state_can_reach_match.
        let state_size = std::mem::size_of::<State>();
        let reach_entry = std::mem::size_of::<bool>();
        assert_eq!(
            two.memory_size() - one.memory_size(),
            state_size + reach_entry,
            "second \\d should add one state, no extra class table",
        );
        assert_eq!(one.classes.len(), 1);
        assert_eq!(two.classes.len(), 1);
    }

    /// `[0-9]` and `\d` produce the same 256-byte lookup table.  When
    /// concatenated as `[0-9]\d`, only one table should be stored.
    #[test]
    fn test_dedup_different_representation() {
        let digit_only = build_regex_unchecked(r"^\d\d$");
        let mixed = build_regex_unchecked(r"^[0-9]\d$");
        assert_eq!(
            digit_only.memory_size(),
            mixed.memory_size(),
            "[0-9]\\d should be the same size as \\d\\d (same table, deduped)",
        );
        assert_eq!(mixed.classes.len(), 1);
        // Also verify correctness.
        assert_matches_regex_crate(r"^[0-9]\d$", &mixed, "42");
        assert_matches_regex_crate(r"^[0-9]\d$", &mixed, "00");
        assert_matches_regex_crate(r"^[0-9]\d$", &mixed, "");
        assert_matches_regex_crate(r"^[0-9]\d$", &mixed, "a1");
        assert_matches_regex_crate(r"^[0-9]\d$", &mixed, "1a");
        assert_matches_regex_crate(r"^[0-9]\d$", &mixed, "1");
        assert_matches_regex_crate(r"^[0-9]\d$", &mixed, "123");
    }

    /// `\w` and `[0-9A-Za-z_]` should produce the same table and dedup.
    #[test]
    fn test_dedup_word_explicit_range() {
        let shorthand = build_regex_unchecked(r"^\w\w$");
        let explicit = build_regex_unchecked(r"^[0-9A-Za-z_]\w$");
        assert_eq!(
            shorthand.memory_size(),
            explicit.memory_size(),
            "[0-9A-Za-z_]\\w should dedup to same table as \\w\\w",
        );
        assert_eq!(explicit.classes.len(), 1);
        assert_matches_regex_crate(r"^[0-9A-Za-z_]\w$", &explicit, "aZ");
        assert_matches_regex_crate(r"^[0-9A-Za-z_]\w$", &explicit, "_0");
        assert_matches_regex_crate(r"^[0-9A-Za-z_]\w$", &explicit, "!a");
    }

    /// `\s` and `[\t\n\x0B\x0C\r ]` should produce the same table.
    #[test]
    fn test_dedup_space_explicit_range() {
        let shorthand = build_regex_unchecked(r"^\s\s$");
        let explicit = build_regex_unchecked(r"^[\t\n\x0B\x0C\r ]\s$");
        assert_eq!(
            shorthand.memory_size(),
            explicit.memory_size(),
            "explicit whitespace class should dedup with \\s",
        );
        assert_eq!(explicit.classes.len(), 1);
        assert_matches_regex_crate(r"^[\t\n\x0B\x0C\r ]\s$", &explicit, " \t");
        assert_matches_regex_crate(r"^[\t\n\x0B\x0C\r ]\s$", &explicit, "a ");
    }

    /// `.*.*` — two wildcards produce the same `[true; 256]` table and
    /// should be deduplicated to a single class.
    #[test]
    fn test_dedup_wildcard() {
        let one_wild = build_regex_unchecked("^.$");
        let two_wild = build_regex_unchecked("^..$");
        let state_size = std::mem::size_of::<State>();
        let reach_entry = std::mem::size_of::<bool>();
        assert_eq!(
            two_wild.memory_size() - one_wild.memory_size(),
            state_size + reach_entry,
            "second `.` should add one state, no extra class table",
        );
        assert_eq!(one_wild.classes.len(), 1);
        assert_eq!(two_wild.classes.len(), 1);
    }

    /// `\d\D` — complementary classes are *not* the same table, so both
    /// must be stored.  This is a negative dedup test.
    #[test]
    fn test_no_dedup_complementary() {
        let same = build_regex_unchecked(r"^\d\d$");
        let comp = build_regex_unchecked(r"^\d\D$");
        // \d\D has two distinct tables; \d\d has one.
        let class_size = std::mem::size_of::<ByteClass>();
        assert_eq!(
            comp.memory_size() - same.memory_size(),
            class_size,
            "\\d\\D should have one more class table than \\d\\d",
        );
        assert_eq!(same.classes.len(), 1);
        assert_eq!(comp.classes.len(), 2);
    }

    /// `\d{3,5}` — counted repetition unrolls multiple ByteClass states
    /// that all refer to the same class.  Only one table is stored.
    #[test]
    fn test_dedup_counted_repetition() {
        let single = build_regex_unchecked(r"^\d$");
        let counted = build_regex_unchecked(r"^\d{3,5}$");
        // The counted version has more states (counter machinery) but
        // should still have exactly one class table, same as the single.
        assert_eq!(
            single.classes.len(),
            1,
            "\\d should have exactly 1 class table",
        );
        assert_eq!(
            counted.classes.len(),
            1,
            "\\d{{3,5}} should still have exactly 1 class table (deduped)",
        );
    }

    // -- Partial anchor tests -----------------------------------------------
    // -------------------------------------------------------------------
    // Multiline assertions: (?m:^) = StartLF, (?m:$) = EndLF
    // -------------------------------------------------------------------
    // -------------------------------------------------------------------
    // CRLF multiline tests  ((?Rm) → StartCRLF / EndCRLF)
    // -------------------------------------------------------------------
    // ===================================================================
    // Word boundary assertions: \b (WordAscii) and \B (WordAsciiNegate)
    // ===================================================================
    // ===================================================================
    // Deferred assertion DFA tests: verify \b, \B, EndLF work via Tier 1 DFA
    // ===================================================================

    /// Verify that patterns with \b are DFA-eligible (no counters).
    #[test]
    fn test_word_boundary_dfa_eligible() {
        let re = build_regex_unchecked(r"\bfoo\b");
        assert!(re.dfa_eligible, "\\bfoo\\b should be DFA-eligible");
        assert!(!re.tier4_eligible);
    }

    /// Verify that patterns with \B are DFA-eligible.
    #[test]
    fn test_non_word_boundary_dfa_eligible() {
        let re = build_regex_unchecked(r"\Bfoo\B");
        assert!(re.dfa_eligible, "\\Bfoo\\B should be DFA-eligible");
    }

    /// Verify that patterns with EndLF (multiline $) are DFA-eligible.
    #[test]
    fn test_endlf_dfa_eligible() {
        let re = build_regex_unchecked(r"(?m)foo$");
        assert!(re.dfa_eligible, "(?m)foo$ should be DFA-eligible");
    }

    /// \b at start of input — word follows non-word (input boundary).
    #[test]
    fn test_word_boundary_dfa_at_start() {
        let p = r"\bfoo";
        let re = build_regex_unchecked(p);
        assert!(re.dfa_eligible);
        assert_matches_regex_crate(p, &re, "foo");
        assert_matches_regex_crate(p, &re, "foobar");
        assert_matches_regex_crate(p, &re, "afoo");
        assert_matches_regex_crate(p, &re, " foo");
    }

    /// \b at end of input — word at end.
    #[test]
    fn test_word_boundary_dfa_at_end() {
        let p = r"foo\b";
        let re = build_regex_unchecked(p);
        assert!(re.dfa_eligible);
        assert_matches_regex_crate(p, &re, "foo");
        assert_matches_regex_crate(p, &re, "foo ");
        assert_matches_regex_crate(p, &re, "foo.");
        assert_matches_regex_crate(p, &re, "foob");
        assert_matches_regex_crate(p, &re, "barfoo");
    }

    /// Deep epsilon-closure chain should not depend on call stack depth.
    #[test]
    fn test_addstate_deep_epsilon_chain() {
        const N: usize = 30_000;

        let mut pattern = String::with_capacity(2 + 2 * N);
        pattern.push('^');
        for _ in 0..N {
            pattern.push_str("a?");
        }
        pattern.push('$');

        let re = build_regex_unchecked(&pattern);
        let mut mem = MatcherMemory::default();

        // All pieces are optional, so empty input matches.
        let m = mem.matcher(&re);
        assert!(m.finish());

        // A short input should also match.
        let mut m = mem.matcher(&re);
        m.chunk(b"aaa");
        assert!(m.finish());
    }

    #[test]
    fn test_start_closure_computed() {
        // Pure alternation of literals: closure should be non-empty
        let re = build_regex_unchecked("(a|b|c)");
        assert!(
            !re.start_closure.is_empty(),
            "pure alt should have start_closure"
        );

        // Pattern starting with counter: closure should be empty
        // (disable unrolling so the counter is preserved)
        let re = build_regex_with_unroll("a{2,3}", 0);
        assert!(
            re.start_closure.is_empty(),
            "counter at start should disable start_closure"
        );

        // Pattern starting with assertion: closure should be empty
        let re = build_regex_unchecked("^abc");
        assert!(
            re.start_closure.is_empty(),
            "assertion at start should disable start_closure"
        );

        // Simple literal: closure should have a single Byte leaf
        let re = build_regex_unchecked("abc");
        assert!(
            !re.start_closure.is_empty(),
            "simple literal should have start_closure"
        );
        assert_eq!(re.start_closure.len(), 1, "literal has 1 start leaf");

        // aws-keys-like pattern: should have 4 leaves
        let re = build_regex_unchecked("(?:ASIA|AKIA|AROA|AIDA)");
        assert!(
            !re.start_closure.is_empty(),
            "aws-keys alt should have start_closure"
        );
        assert_eq!(re.start_closure.len(), 4, "4 branches = 4 leaves");
    }

    // -----------------------------------------------------------------------
    // Case-insensitive ((?i)) tests
    // -----------------------------------------------------------------------
    /// Verify ByteCI state is actually emitted for simple (?i) patterns.
    #[test]
    fn test_ci_byteci_state_emitted() {
        let re = build_regex_unchecked("^(?i)a$");
        let has_byteci = re.states.iter().any(|s| matches!(s, State::ByteCI { .. }));
        assert!(has_byteci, "expected ByteCI state for (?i)a");

        // Non-CI pattern should NOT have ByteCI
        let re2 = build_regex_unchecked("^a$");
        let has_byteci2 = re2.states.iter().any(|s| matches!(s, State::ByteCI { .. }));
        assert!(!has_byteci2, "plain \"a\" should not use ByteCI");
    }

    /// Byte equivalence classes are only enabled for complex patterns
    /// (>= BYTE_CLASSES_NFA_THRESHOLD NFA states).  A simple (?i)abc
    /// stays at stride=256 to avoid per-byte indirection overhead.
    /// A sufficiently large CI pattern gets compression.
    #[test]
    fn test_ci_byte_classes_compress() {
        // Small pattern: should use identity mapping (stride=256).
        let small = build_regex_unchecked("^(?i)abc$");
        assert_eq!(
            small.num_byte_classes, 256,
            "small pattern should use identity mapping, got {} classes",
            small.num_byte_classes
        );

        // Large pattern: use a WAF-style keyword alternation that exceeds
        // the NFA threshold.  This should get byte class compression.
        let large = build_regex_unchecked(
            "(?i)\\b(?:select|insert|update|delete|drop|alter|create|grant|\
             revoke|union|where|having|order|group|limit|offset|from|into|\
             table|index|exec|execute|declare|set|cast|convert|char|concat|\
             substring|ascii|benchmark|sleep|waitfor|delay|load_file|outfile)\\b",
        );
        assert!(
            large.num_byte_classes < 256,
            "large pattern should use byte class compression, got {} classes",
            large.num_byte_classes
        );
        // Verify a and A map to the same class in the compressed pattern
        assert_eq!(
            large.byte_classes[b'a' as usize], large.byte_classes[b'A' as usize],
            "a and A should be in the same byte equivalence class"
        );
        assert_eq!(
            large.byte_classes[b'b' as usize], large.byte_classes[b'B' as usize],
            "b and B should be in the same byte equivalence class"
        );
    }
    // -----------------------------------------------------------------------
    // Partial case-insensitive tests (mixed CI / case-sensitive regions)
    // -----------------------------------------------------------------------
    /// Verify structural: partial CI emits ByteCI only for CI letters.
    #[test]
    fn test_partial_ci_byteci_selective() {
        // a(?i:b)c ��� only b should be ByteCI, a and c should be Byte
        let re = build_regex_unchecked("^a(?i:b)c$");
        let byte_count = re
            .states
            .iter()
            .filter(|s| matches!(s, State::Byte { .. }))
            .count();
        let byteci_count = re
            .states
            .iter()
            .filter(|s| matches!(s, State::ByteCI { .. }))
            .count();
        assert_eq!(byte_count, 2, "a and c should be plain Byte states");
        assert_eq!(byteci_count, 1, "only b should be ByteCI");
    }

    /// `(?i)abc(?-i)def` — verify byte equivalence classes differ by region.
    /// Letters in the CI region should share classes with their upper counterparts;
    /// letters in the non-CI region should not.
    ///
    /// Uses a large enough pattern to trigger byte class compression, by
    /// padding with additional alternations.
    #[test]
    fn test_partial_ci_byte_classes_selective() {
        // Build a pattern large enough to exceed BYTE_CLASSES_NFA_THRESHOLD
        // while preserving the CI/non-CI split we want to test.
        let re = build_regex_unchecked(
            "(?i)(?:abc|ghi|jkl|mno|pqr|stu|vwx|yz0|123|456|789|\
             aaa|bbb|ccc|ddd|eee|fff|ggg|hhh|iii|jjj|kkk|lll|mmm|\
             nnn|ooo|ppp|qqq|rrr|sss|ttt|uuu|vvv|www|xxx|yyy|zzz)(?-i)def",
        );
        assert!(
            re.num_byte_classes < 256,
            "pattern should use byte class compression, got {} classes",
            re.num_byte_classes
        );
        // In the CI region: a/A should share a class
        assert_eq!(
            re.byte_classes[b'a' as usize], re.byte_classes[b'A' as usize],
            "a and A should be in the same byte equivalence class (CI region)"
        );
        // In the non-CI region: d/D should be in different classes because
        // d matches a consuming state that D does not.
        assert_ne!(
            re.byte_classes[b'd' as usize], re.byte_classes[b'D' as usize],
            "d and D should be in different byte equivalence classes (non-CI region)"
        );
    }

    #[test]
    fn test_too_many_counters() {
        // Build a pattern with 257 independent counted repetitions,
        // which requires 257 counters and should exceed the u8 limit.
        // Disable unrolling so each `a{2,3}` creates a counter.
        let mut pattern = String::from("^");
        for _ in 0..257 {
            pattern.push_str("a{2,3}");
        }
        pattern.push('$');
        let mut builder = RegexBuilder::default();
        builder.max_unroll_states(0);
        let result = builder.build(&regex_syntax::parse(&pattern).unwrap());
        assert!(
            matches!(result, Err(Error::TooManyCounters)),
            "expected TooManyCounters error, got {result:?}"
        );
    }

    #[test]
    fn test_256_counters_ok() {
        // 256 counters should be fine (indices 0..255 fit in u8).
        // Disable unrolling so each `a{2,3}` creates a counter.
        let mut pattern = String::from("^");
        for _ in 0..256 {
            pattern.push_str("a{2,3}");
        }
        pattern.push('$');
        let mut builder = RegexBuilder::default();
        builder.max_unroll_states(0);
        let result = builder.build(&regex_syntax::parse(&pattern).unwrap());
        assert!(
            result.is_ok(),
            "256 counters should succeed, got {result:?}"
        );
    }

    /// Regression test: tier 3 false positive with adjacent same-byte
    /// sequential counters.  `^a{2,50}a{3,70}$` has two counters on byte
    /// 'a' (both too large to unroll), and the minimum total is 2+3=5.
    /// Before the fix, the break-seed mechanism would prematurely seed the
    /// second counter, producing false positives for short inputs.
    #[test]
    fn test_tier3_adjacent_counter_false_positive() {
        let pattern = "^a{2,50}a{3,70}$";
        let re = build_regex_unchecked(pattern);
        let oracle = regex::bytes::Regex::new(&format!("(?s-u){}", pattern)).unwrap();
        assert!(re.tier3_eligible, "pattern should be tier 3 eligible");
        // Sweep lengths around the min (5) and max (120) boundaries.
        for n in 0..130 {
            let input: String = "a".repeat(n);
            let expected = oracle.is_match(input.as_bytes());
            test_nfa(pattern, &re, &input, expected, 0);
            test_tier3(pattern, &re, &input, expected, 0);
        }
    }

    /// Regression test: inner-unroll producing adjacent same-byte counters.
    /// `^(a{2,18}){2,3}$` unrolls the outer {2,3} into 2-3 sequential
    /// a{2,18} counters — all on the same byte 'a'.
    #[test]
    fn test_tier3_unrolled_adjacent_counter_false_positive() {
        let pattern = "^(a{2,18}){2,3}$";
        let re = build_regex_with_unroll(pattern, DEFAULT_MAX_UNROLL_STATES);
        let oracle = regex::bytes::Regex::new(&format!("(?s-u){}", pattern)).unwrap();
        // Lengths around the min (4) and max (54) boundaries.
        for n in 0..70 {
            let input: String = "a".repeat(n);
            let expected = oracle.is_match(input.as_bytes());
            test_nfa(pattern, &re, &input, expected, DEFAULT_MAX_UNROLL_STATES);
            if re.tier3_eligible {
                test_tier3(pattern, &re, &input, expected, DEFAULT_MAX_UNROLL_STATES);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Proptest: fuzz-style property-based tests
    // -----------------------------------------------------------------------

    /// Shared helper: compile a pattern with both engines and test many
    /// inputs, comparing the oracle (`regex` crate) against every eligible
    /// rethoc tier.
    fn fuzz_oracle_and_tiers(pattern: &str, inputs: &[Vec<u8>]) {
        // Parse with regex-syntax in byte mode.
        use regex_syntax::ast::parse::ParserBuilder;
        use regex_syntax::hir::translate::TranslatorBuilder;

        let ast = match ParserBuilder::new().build().parse(pattern) {
            Ok(a) => a,
            Err(_) => return, // unparseable — skip
        };
        let hir = match TranslatorBuilder::new()
            .unicode(false)
            .utf8(false)
            .dot_matches_new_line(true)
            .build()
            .translate(pattern, &ast)
        {
            Ok(h) => h,
            Err(_) => return,
        };

        // Compile with rethoc.
        let mut builder = RegexBuilder::default();
        let re = match builder.build(&hir) {
            Ok(r) => r,
            Err(_) => return, // unsupported construct — skip
        };

        // Compile with the regex crate oracle.
        let full = format!("(?s-u){}", pattern);
        let oracle = match regex::bytes::Regex::new(&full) {
            Ok(o) => o,
            Err(_) => return,
        };

        let mut memory = MatcherMemory::default();

        for input in inputs {
            let expected = oracle.is_match(input);

            // Test the default (highest-eligible) tier via chunk().
            let mut matcher = memory.matcher(&re);
            matcher.chunk(input);
            let actual = matcher.finish();
            assert_eq!(
                actual,
                expected,
                "oracle mismatch (default tier) for pattern `{}` on input {:?} (len={}): \
                 ours={}, oracle={}",
                pattern,
                &input[..input.len().min(80)],
                input.len(),
                actual,
                expected
            );

            // Test NFA (always available).
            let mut m = memory.nfa_matcher(&re);
            m.chunk(input);
            let nfa_result = m.finish();
            assert_eq!(
                nfa_result,
                expected,
                "NFA mismatch for pattern `{}` on input {:?} (len={}): \
                 nfa={}, oracle={}",
                pattern,
                &input[..input.len().min(80)],
                input.len(),
                nfa_result,
                expected
            );

            // Test each eligible DFA tier.
            for tier in 1..=4u8 {
                if let Ok(mut m) = memory.matcher_for_tier(&re, tier) {
                    m.chunk(input);
                    let tier_result = m.finish();
                    assert_eq!(
                        tier_result,
                        expected,
                        "Tier {} mismatch for pattern `{}` on input {:?} (len={}): \
                         tier{}={}, oracle={}",
                        tier,
                        pattern,
                        &input[..input.len().min(80)],
                        input.len(),
                        tier,
                        tier_result,
                        expected
                    );
                }
            }
        }

        // Second pass: recompile with unrolling disabled to exercise
        // counter-based code paths.
        builder.max_unroll_states(0);
        if let Ok(re_no_unroll) = builder.build(&hir) {
            for input in inputs {
                let expected = oracle.is_match(input);

                let mut matcher = memory.matcher(&re_no_unroll);
                matcher.chunk(input);
                let actual = matcher.finish();
                assert_eq!(
                    actual,
                    expected,
                    "oracle mismatch (no-unroll) for pattern `{}` on input {:?} (len={}): \
                     ours={}, oracle={}",
                    pattern,
                    &input[..input.len().min(80)],
                    input.len(),
                    actual,
                    expected
                );

                for tier in 0..=4u8 {
                    if let Ok(mut m) = memory.matcher_for_tier(&re_no_unroll, tier) {
                        m.chunk(input);
                        let tier_result = m.finish();
                        assert_eq!(
                            tier_result,
                            expected,
                            "Tier {} mismatch (no-unroll) for pattern `{}` on input {:?} \
                             (len={}): tier{}={}, oracle={}",
                            tier,
                            pattern,
                            &input[..input.len().min(80)],
                            input.len(),
                            tier,
                            tier_result,
                            expected
                        );
                    }
                }
            }
        }
    }

    /// Default number of proptest cases per fuzz test.  Each case generates
    /// one pattern and ~50-80 targeted inputs, so 100 cases ≈ 5000-8000
    /// individual match comparisons.  For deeper exploration use `cargo-fuzz`.
    ///
    /// Run these tests explicitly with:
    /// ```sh
    /// cargo test test_fuzz -- --ignored
    /// ```
    const FUZZ_PROPTEST_CASES: u32 = 100;

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(FUZZ_PROPTEST_CASES))]

        /// Oracle differential: generated pattern × many pattern-aware inputs.
        ///
        /// For each generated pattern, produces targeted inputs from the
        /// pattern AST (positive candidates, near-miss mutations, edge cases)
        /// plus random fuzzer-supplied bytes.  Compiles with both rethoc and
        /// the `regex` crate, then asserts all tiers agree with the oracle.
        #[test]
        #[ignore]
        fn test_fuzz_oracle(
            pattern_seed in proptest::collection::vec(proptest::num::u8::ANY, 0..64),
            extra_bytes in proptest::collection::vec(proptest::num::u8::ANY, 0..128),
        ) {
            use crate::fuzz_gen::{FuzzRng, generate_pattern, generate_inputs};

            let (pattern, ast) = generate_pattern(&mut FuzzRng::new(&pattern_seed));
            let mut inputs = generate_inputs(&mut FuzzRng::new(&extra_bytes), &ast);

            // Also add the raw extra bytes as one more input.
            inputs.push(extra_bytes);

            fuzz_oracle_and_tiers(&pattern, &inputs);
        }

        /// Cross-tier differential: NFA is the oracle, all eligible DFA
        /// tiers must agree.  Does not require the `regex` crate — useful
        /// for finding tier-specific bugs independently.
        #[test]
        #[ignore]
        fn test_fuzz_differential_tiers(
            pattern_seed in proptest::collection::vec(proptest::num::u8::ANY, 0..64),
            extra_bytes in proptest::collection::vec(proptest::num::u8::ANY, 0..128),
        ) {
            use crate::fuzz_gen::{FuzzRng, generate_pattern, generate_inputs};

            let (pattern, ast) = generate_pattern(&mut FuzzRng::new(&pattern_seed));
            let inputs = generate_inputs(&mut FuzzRng::new(&extra_bytes), &ast);

            // Parse and compile.
            let hir = match parse_hir_bytes_fallible(&pattern) {
                Some(h) => h,
                None => return Ok(()),
            };
            let mut builder = RegexBuilder::default();
            let re = match builder.build(&hir) {
                Ok(r) => r,
                Err(_) => return Ok(()),
            };

            let mut memory = MatcherMemory::default();

            for input in &inputs {
                // NFA result is the oracle.
                let mut m = memory.nfa_matcher(&re);
                m.chunk(input);
                let nfa_result = m.finish();

                for tier in 1..=4u8 {
                    if let Ok(mut m) = memory.matcher_for_tier(&re, tier) {
                        m.chunk(input);
                        let tier_result = m.finish();
                        assert_eq!(
                            tier_result, nfa_result,
                            "Tier {tier} disagrees with NFA for pattern `{pattern}` \
                             on input {:?} (len={}): tier{tier}={tier_result}, nfa={nfa_result}",
                            &input[..input.len().min(80)], input.len()
                        );
                    }
                }
            }
        }
    }

    /// Fallible HIR parse in byte mode — returns `None` if the pattern
    /// is rejected by `regex-syntax`.
    fn parse_hir_bytes_fallible(pattern: &str) -> Option<Hir> {
        use regex_syntax::ast::parse::ParserBuilder;
        use regex_syntax::hir::translate::TranslatorBuilder;

        let ast = ParserBuilder::new().build().parse(pattern).ok()?;
        TranslatorBuilder::new()
            .unicode(false)
            .utf8(false)
            .dot_matches_new_line(true)
            .build()
            .translate(pattern, &ast)
            .ok()
    }

}
