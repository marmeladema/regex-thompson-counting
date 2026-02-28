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
//! - **Non-empty context**: `HashSet<(StateIdx, CounterCtx)>` cleared
//!   per step.
//!
//! ## Complexity
//!
//! - **Anchored patterns** (`^...$`): only 1 counter instance per
//!   repetition (no re-seeding).  O(|states|) per step — identical to
//!   the delta approach.
//! - **Unanchored patterns**: O(|states| × max) per step.  The
//!   `max_repetition` compile-time cap (default 1000) bounds this for
//!   untrusted patterns.

use std::collections::HashSet;
use std::fmt;
use std::io::Write;
use std::ops::{Index, IndexMut};

use regex_syntax::hir::{self, HirKind};

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
        }
    }
}

impl std::error::Error for Error {}

/// A 256-entry boolean lookup table indicating which byte values belong
/// to a character class.  `class[b]` is `true` when byte `b` matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ByteClass([bool; 256]);

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
struct ClassIdx(usize);

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
struct ByteTableIdx(usize);

impl ByteTableIdx {
    #[inline]
    fn idx(self) -> usize {
        self.0
    }
}

/// A 256-entry dispatch table mapping each byte value to a target
/// [`StateIdx`], or [`StateIdx::NONE`] for "no transition".
#[derive(Clone, Copy, Debug)]
struct ByteMap([StateIdx; 256]);

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
enum AssertKind {
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
}

/// Result of evaluating an assertion at a given position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AssertEval {
    /// Assertion is satisfied — follow the `out` transition.
    Pass,
    /// Assertion is not satisfied — skip.
    Fail,
    /// Cannot determine yet (the upcoming byte is unknown).
    /// The state is parked in the list for deferred resolution in
    /// [`Matcher::step`] or [`Matcher::finish`].
    Defer,
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
enum State {
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
struct StateIdx(u32);

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

struct StateList(Box<[State]>);

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
    states: StateList,
    start: StateIdx,
    /// Number of counter variables allocated during compilation.
    num_counters: usize,
    /// Byte-class lookup tables referenced by [`State::ByteClass::class`].
    classes: Box<[ByteClass]>,
    /// Byte dispatch tables referenced by [`State::ByteTable::table`].
    /// Each entry maps a byte value to a target state index, or
    /// [`StateIdx::NONE`] for "no transition".
    byte_tables: Box<[ByteMap]>,
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
        inline + states_alloc + classes_alloc + byte_tables_alloc
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
    pub max_repetition: usize,
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
        }
    }
}
impl RegexBuilder {
    /// Allocate a fresh counter index.
    fn next_counter(&mut self) -> CounterIdx {
        let counter = self.counters.len();
        self.counters.push(counter);
        CounterIdx(counter)
    }

    /// Return the index of `table` in `self.classes`, inserting it if it
    /// is not already present.  Identical tables are deduplicated so that
    /// patterns like `\d{3,5}` (which unroll to multiple ByteClass states)
    /// share a single lookup table.
    fn intern_class(&mut self, table: ByteClass) -> ClassIdx {
        let (idx, _) = self.classes.insert_full(table);
        ClassIdx(idx)
    }

    /// Recursively lower a `regex-syntax` HIR node into a postfix sequence
    /// appended to `self.postfix`.
    ///
    /// Bounded repetitions are lowered to a single `CounterLoop` node
    /// that wires CI → body → CInc in a single-copy loop.  Nested
    /// repetitions share the same body copy (each gets its own counter
    /// index, no remapping needed).
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
                // patterns like `(a|b)` → `[ab]`.  If all ranges fit in a
                // single byte (0x00..=0xFF), lower them to a ByteClass;
                // otherwise reject.
                let ranges = class.ranges();
                let all_single_byte = ranges
                    .iter()
                    .all(|r| (r.start() as u32) <= 0xFF && (r.end() as u32) <= 0xFF);
                if !all_single_byte {
                    return Err(Error::UnsupportedClass(hir::Class::Unicode(class.clone())));
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
                for (idx, child) in children.iter().enumerate() {
                    self.hir2postfix(child)?;
                    if idx > 0 {
                        self.postfix.push(RegexHirNode::Alternate);
                    }
                }
                Ok(())
            }
            HirKind::Repetition(rep) => {
                let min = rep.min as usize;
                let max = rep.max.map_or(usize::MAX, |m| m as usize);
                assert!(min <= max);

                // Reject repetitions exceeding the compile-time cap.
                if max != usize::MAX && max > self.max_repetition {
                    return Err(Error::RepetitionTooLarge(max, self.max_repetition));
                }

                // Special-case common quantifiers to avoid counter overhead.
                if min == 0 && max == 1 {
                    // `?`
                    self.hir2postfix(&rep.sub)?;
                    self.postfix.push(RegexHirNode::RepeatZeroOne);
                    return Ok(());
                }
                if min == 0 && max == usize::MAX {
                    // `*`
                    self.hir2postfix(&rep.sub)?;
                    self.postfix.push(RegexHirNode::RepeatZeroPlus);
                    return Ok(());
                }
                if min == 1 && max == usize::MAX {
                    // `+`
                    self.hir2postfix(&rep.sub)?;
                    self.postfix.push(RegexHirNode::RepeatOnePlus);
                    return Ok(());
                }

                if min > 0 {
                    let counter = self.next_counter();
                    self.hir2postfix(&rep.sub)?;
                    self.postfix
                        .push(RegexHirNode::CounterLoop { counter, min, max });
                } else {
                    // {0,max}: lower to (body{1,max})? — the `?` wrapping
                    // provides the zero-match path.
                    let counter = self.next_counter();
                    self.hir2postfix(&rep.sub)?;
                    self.postfix.push(RegexHirNode::CounterLoop {
                        counter,
                        min: 1,
                        max,
                    });
                    self.postfix.push(RegexHirNode::RepeatZeroOne);
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

        Ok(Regex {
            states: StateList(self.states.to_vec().into_boxed_slice()),
            start,
            num_counters: self.counters.len(),
            classes: self
                .classes
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            byte_tables: self.byte_tables.to_vec().into_boxed_slice(),
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
}

// ---------------------------------------------------------------------------
// Counter context (per-thread counter values)
// ---------------------------------------------------------------------------

/// Index into the counter variable array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct CounterIdx(usize);

impl CounterIdx {
    #[inline]
    fn idx(self) -> usize {
        self.0
    }
}

impl fmt::Display for CounterIdx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Sentinel value: this counter slot is inactive (thread is not inside
/// this counter's repetition body).
const COUNTER_INACTIVE: usize = usize::MAX;

/// Per-thread counter context.
///
/// A fixed-length vector indexed by [`CounterIdx`]: `values[i]` holds
/// the iteration count for counter `i`, or [`COUNTER_INACTIVE`] when
/// the thread is not inside counter `i`'s repetition body.
///
/// For threads outside all counted repetitions (the common case), the
/// context is empty (`0.is_empty() == true`) and dedup falls back to
/// the fast `lastlist` path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CounterCtx(Box<[usize]>);

impl CounterCtx {
    /// Create a context with all counters inactive.
    fn new(num_counters: usize) -> Self {
        if num_counters == 0 {
            Self(Box::new([]))
        } else {
            Self(vec![COUNTER_INACTIVE; num_counters].into_boxed_slice())
        }
    }

    /// True when no counter is active (all slots are `COUNTER_INACTIVE`).
    #[inline]
    fn is_empty(&self) -> bool {
        self.0.iter().all(|&v| v == COUNTER_INACTIVE)
    }

    /// Get the value of counter `idx`, or `None` if inactive.
    #[inline]
    fn get(&self, idx: CounterIdx) -> Option<usize> {
        let v = self.0[idx.idx()];
        if v == COUNTER_INACTIVE { None } else { Some(v) }
    }

    /// Set the value of counter `idx`.
    #[inline]
    fn set(&mut self, idx: CounterIdx, value: usize) {
        debug_assert!(value != COUNTER_INACTIVE);
        self.0[idx.idx()] = value;
    }

    /// Deactivate counter `idx`.
    #[inline]
    fn remove(&mut self, idx: CounterIdx) {
        self.0[idx.idx()] = COUNTER_INACTIVE;
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
    /// Context-aware dedup for threads with non-empty counter contexts.
    /// Cleared at each step.
    ctx_visited: HashSet<(StateIdx, CounterCtx)>,
}

impl MatcherMemory {
    pub fn matcher<'a>(&'a mut self, regex: &'a Regex) -> Matcher<'a> {
        self.lastlist.clear();
        self.lastlist.resize(regex.states.len(), usize::MAX);
        self.clist.clear();
        self.nlist.clear();
        self.addstack.clear();
        self.ctx_visited.clear();

        let empty_ctx = CounterCtx::new(regex.num_counters);

        let mut m = Matcher {
            states: &regex.states,
            classes: &regex.classes,
            byte_tables: &regex.byte_tables,
            lastlist: &mut self.lastlist,
            listid: 0,
            clist: &mut self.clist,
            nlist: &mut self.nlist,
            addstack: &mut self.addstack,
            ctx_visited: &mut self.ctx_visited,
            start: regex.start,
            at_start: true,
            at_end: false,
            prev_byte: None,
            ever_matched: false,
            empty_ctx,
        };

        m.startlist(m.start);
        m
    }
}

/// Runs a Thompson NFA simulation with per-thread counter contexts.
#[derive(Debug)]
pub struct Matcher<'a> {
    states: &'a [State],
    /// Byte-class lookup tables referenced by [`State::ByteClass::class`].
    classes: &'a [ByteClass],
    /// Byte dispatch tables referenced by [`State::ByteTable::table`].
    byte_tables: &'a [ByteMap],
    /// Number of counter variables in the compiled regex.
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
    /// Context-aware dedup for non-empty counter contexts.
    ctx_visited: &'a mut HashSet<(StateIdx, CounterCtx)>,

    /// The NFA start state index.
    start: StateIdx,
    /// `true` until the first [`step`](Self::step) call.
    at_start: bool,
    /// Set to `true` by [`finish`](Self::finish).
    at_end: bool,
    /// The last byte consumed by [`step`](Self::step).
    prev_byte: Option<u8>,
    /// Tracks whether a `Match` state was ever reached.
    ever_matched: bool,
    /// Pre-allocated empty context (all counters inactive).
    empty_ctx: CounterCtx,
}

/// Internal operations for iterative [`Matcher::addstate`] traversal.
#[derive(Clone, Debug)]
enum AddStateOp {
    /// Visit (and epsilon-expand) a state with the given context.
    Visit(StateIdx, CounterCtx),
    /// Push this state + context to `nlist` after its epsilon successors
    /// are handled.  Only used for consuming and Assert states.
    PostPush(StateIdx, CounterCtx),
}

impl<'a> Matcher<'a> {
    /// Compute the initial state list by following all epsilon transitions
    /// from `start`.
    #[inline]
    fn startlist(&mut self, start: StateIdx) {
        self.addstate(start, self.empty_ctx.clone());
        std::mem::swap(self.clist, self.nlist);
        self.listid += 1;
    }

    /// Follow epsilon transitions from state `idx` with counter context
    /// `ctx`, adding all reachable states to `nlist`.
    ///
    /// Dedup strategy:
    /// - Empty context: fast path via `lastlist`/`listid` (O(1) per state).
    /// - Non-empty context: `ctx_visited` HashSet keyed on `(state, ctx)`.
    #[inline]
    fn addstate(&mut self, idx: StateIdx, ctx: CounterCtx) {
        self.addstack.clear();
        self.addstack.push(AddStateOp::Visit(idx, ctx));
        self.drain_addstack();
    }

    /// Process all operations on the work stack until empty.
    fn drain_addstack(&mut self) {
        while let Some(op) = self.addstack.pop() {
            match op {
                AddStateOp::Visit(idx, ctx) => {
                    // --- Dedup ---
                    if ctx.is_empty() {
                        let i = idx.idx();
                        if self.lastlist[i] == self.listid {
                            continue;
                        }
                        self.lastlist[i] = self.listid;
                    } else if !self.ctx_visited.insert((idx, ctx.clone())) {
                        continue;
                    }

                    match self.states[idx] {
                        State::Split { out, out1 } => {
                            // No PostPush for epsilon states.
                            self.addstack.push(AddStateOp::Visit(out1, ctx.clone()));
                            self.addstack.push(AddStateOp::Visit(out, ctx));
                        }

                        State::Assert { kind, out } => {
                            // PostPush so the Assert is in nlist for
                            // deferred resolution in step()/finish().
                            self.addstack.push(AddStateOp::PostPush(idx, ctx.clone()));
                            if kind.eval(self.at_start, self.at_end, self.prev_byte, None)
                                == AssertEval::Pass
                            {
                                self.addstack.push(AddStateOp::Visit(out, ctx));
                            }
                        }

                        State::CounterInstance { counter, out } => {
                            // Enter the counted repetition: set counter = 0.
                            // We own ctx — mutate in place, no clone needed.
                            let mut ctx = ctx;
                            ctx.set(counter, 0);
                            self.addstack.push(AddStateOp::Visit(out, ctx));
                        }

                        State::CounterIncrement {
                            counter,
                            out,
                            out1,
                            min,
                            max,
                        } => {
                            let value = ctx.get(counter).expect("counter must be active at CInc");
                            let new_value = value + 1;
                            let take_continue = new_value < max;
                            let take_break = new_value >= min;

                            match (take_continue, take_break) {
                                (true, true) => {
                                    // Both paths: clone once for break, mutate for continue.
                                    let mut break_ctx = ctx.clone();
                                    break_ctx.remove(counter);
                                    self.addstack.push(AddStateOp::Visit(out1, break_ctx));
                                    let mut ctx = ctx;
                                    ctx.set(counter, new_value);
                                    self.addstack.push(AddStateOp::Visit(out, ctx));
                                }
                                (true, false) => {
                                    // Continue only: mutate in place.
                                    let mut ctx = ctx;
                                    ctx.set(counter, new_value);
                                    self.addstack.push(AddStateOp::Visit(out, ctx));
                                }
                                (false, true) => {
                                    // Break only: mutate in place.
                                    let mut ctx = ctx;
                                    ctx.remove(counter);
                                    self.addstack.push(AddStateOp::Visit(out1, ctx));
                                }
                                (false, false) => {}
                            }
                        }

                        State::Match => {
                            self.ever_matched = true;
                        }

                        // Consuming states: record in nlist for step().
                        State::Byte { .. } | State::ByteClass { .. } | State::ByteTable { .. } => {
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
    pub fn step(&mut self, b: u8) {
        // --- Pre-consumption: resolve deferred assertions ---
        {
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
                        self.ctx_visited.clear();
                        self.nlist.clear();
                        any_expanded = true;
                    }
                    self.addstate(out, ctx.clone());
                }
            }
            if any_expanded {
                self.clist.append(self.nlist);
                self.listid += 1;
                self.ctx_visited.clear();
            }
        }

        self.at_start = false;
        self.prev_byte = Some(b);

        self.nlist.clear();
        self.ctx_visited.clear();
        let clist = std::mem::take(self.clist);

        // Fused pass: push Visit ops for matching consuming states
        // (in reverse for LIFO ordering) + re-seed.
        self.addstack.clear();
        self.addstack
            .push(AddStateOp::Visit(self.start, self.empty_ctx.clone()));

        for (idx, ctx) in clist.iter().rev() {
            let target = match self.states[*idx] {
                State::Byte { byte: b2, out } if b == b2 => out,
                State::ByteClass { class, out } if self.classes[class][b] => out,
                State::ByteTable { table } => {
                    let t = self.byte_tables[table][b];
                    if t == StateIdx::NONE {
                        continue;
                    }
                    t
                }
                _ => continue,
            };
            self.addstack.push(AddStateOp::Visit(target, ctx.clone()));
        }

        self.drain_addstack();

        *self.clist = std::mem::replace(self.nlist, clist);
        self.listid += 1;
        self.ctx_visited.clear();
    }

    /// Feed an entire byte slice through the matcher, one byte at a time.
    pub fn chunk(&mut self, input: &[u8]) {
        for &b in input {
            self.step(b);
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
            return true;
        }

        self.at_end = true;

        let clist_len = self.clist.len();
        if clist_len > 0 {
            self.listid += 1;
            self.ctx_visited.clear();

            self.nlist.clear();
            for i in 0..clist_len {
                let (idx, ref ctx) = self.clist[i];
                if let State::Assert { kind, out } = self.states[idx]
                    && kind.eval(self.at_start, true, self.prev_byte, None) == AssertEval::Pass
                {
                    self.addstate(out, ctx.clone());
                }
            }

            self.clist.append(self.nlist);
        }

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
    // CounterCtx unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_counter_ctx_empty() {
        let ctx = CounterCtx::new(0);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_counter_ctx_all_inactive() {
        let ctx = CounterCtx::new(3);
        assert!(ctx.is_empty());
        assert_eq!(ctx.get(CounterIdx(0)), None);
        assert_eq!(ctx.get(CounterIdx(1)), None);
        assert_eq!(ctx.get(CounterIdx(2)), None);
    }

    #[test]
    fn test_counter_ctx_set_get() {
        let mut ctx = CounterCtx::new(2);
        ctx.set(CounterIdx(0), 5);
        assert!(!ctx.is_empty());
        assert_eq!(ctx.get(CounterIdx(0)), Some(5));
        assert_eq!(ctx.get(CounterIdx(1)), None);
    }

    #[test]
    fn test_counter_ctx_remove() {
        let mut ctx = CounterCtx::new(2);
        ctx.set(CounterIdx(0), 5);
        ctx.set(CounterIdx(1), 3);
        ctx.remove(CounterIdx(0));
        assert_eq!(ctx.get(CounterIdx(0)), None);
        assert_eq!(ctx.get(CounterIdx(1)), Some(3));
        assert!(!ctx.is_empty());
        ctx.remove(CounterIdx(1));
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_counter_ctx_set_mutates() {
        let mut ctx = CounterCtx::new(2);
        assert!(ctx.is_empty());
        ctx.set(CounterIdx(1), 7);
        assert!(!ctx.is_empty());
        assert_eq!(ctx.get(CounterIdx(1)), Some(7));
    }

    #[test]
    fn test_counter_ctx_remove_mutates() {
        let mut ctx = CounterCtx::new(2);
        ctx.set(CounterIdx(0), 10);
        ctx.set(CounterIdx(1), 20);
        ctx.remove(CounterIdx(0));
        assert_eq!(ctx.get(CounterIdx(0)), None);
        assert_eq!(ctx.get(CounterIdx(1)), Some(20));
    }

    #[test]
    fn test_counter_ctx_equality_and_hash() {
        use HashSet;
        let mut a = CounterCtx::new(2);
        a.set(CounterIdx(0), 3);
        let mut b = CounterCtx::new(2);
        b.set(CounterIdx(0), 3);
        let mut c = CounterCtx::new(2);
        c.set(CounterIdx(0), 4);
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut set = HashSet::new();
        assert!(set.insert(a.clone()));
        assert!(!set.insert(b)); // duplicate
        assert!(set.insert(c));
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

    #[test]
    fn test_byte_table_cross_validate_unanchored() {
        let p = "(ab|cd|ef)";
        let re = build_regex_unchecked(p);
        for input in &[
            "ab", "cd", "ef", "xxab", "cdyy", "xxefyy", // positives
            "ac", "xyz", "", "a", "f", // negatives
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
            m.step(b);
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

    /// ByteTable inside a bounded repetition with real counting.
    #[test]
    fn test_byte_table_with_bounded_repetition_cross_validate() {
        let p = "^(ab|cd|ef){2,4}$";
        let re = build_regex_unchecked(p);
        for input in &[
            "abcd",
            "efab",
            "cdef",
            "ababab",
            "abcdef",
            "abcdefab", // positives (2,3,4 reps)
            "ab",
            "ef",
            "",
            "abcdefabcd",
            "abcdefef", // negatives (1, 0, 5 reps)
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

    /// ByteTable inside a counted repetition with a non-trivial suffix.
    #[test]
    fn test_byte_table_counted_with_suffix() {
        let p = "^(ab|cd|ef){1,3}x$";
        let re = build_regex_unchecked(p);
        for input in &[
            "abx",
            "cdx",
            "efx", // 1 rep + suffix
            "abcdx",
            "efabx",   // 2 reps + suffix
            "abcdefx", // 3 reps + suffix
            "x",
            "ab",
            "abcdefabx",
            "",
            "abcdefxx", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

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

    /// Mixed-length unanchored — ByteTable in a substring context.
    #[test]
    fn test_byte_table_mixed_length_unanchored() {
        let p = "(a|bc|def)";
        let re = build_regex_unchecked(p);
        for input in &[
            "a", "bc", "def", "xxa", "xxbcyy", "xxdefyy", // positives
            "xx", "", "bd", // negatives
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

    /// Literal prefix and suffix around a ByteTable alternation.
    #[test]
    fn test_byte_table_with_prefix_and_suffix() {
        let p = "^xx(ab|cd|ef)yy$";
        let re = build_regex_unchecked(p);
        for input in &[
            "xxabyy", "xxcdyy", "xxefyy", // positives
            "xxabyyz", "xabyy", "xxaby", "xxyy", "abyy", "", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

    /// ByteTable preceded by a wildcard.
    #[test]
    fn test_byte_table_after_wildcard() {
        let p = "^..(ab|cd|ef)$";
        let re = build_regex_unchecked(p);
        for input in &[
            "xxab",
            "zzcd",
            "qqef",
            "\x00\x7fab", // positives
            "xab",
            "xxxab",
            "xxac",
            "", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

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

    /// ByteTable followed by `.*` wildcard and a literal.
    #[test]
    fn test_byte_table_followed_by_wildcard() {
        let p = "^(ab|cd|ef).*x$";
        let re = build_regex_unchecked(p);
        for input in &[
            "abx", "cdx", "efx", "ab123x", "cdxxxxxx", // positives
            "abX", "cd", "ef123", "x", "", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

    /// ByteTable inside a complex pattern with wildcards on both sides.
    #[test]
    fn test_byte_table_sandwiched_by_wildcards() {
        let p = "^.*(ab|cd|ef).*$";
        let re = build_regex_unchecked(p);
        for input in &[
            "ab",
            "cd",
            "ef",
            "xxxab",
            "cdyyy",
            "xxxefyyy",
            "xxabyycdzzef", // positives
            "",
            "x",
            "ac", // negatives
        ] {
            assert_matches_regex_crate(p, &re, input);
        }
    }

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
            matcher.step(b);
        }
        let actual_step = matcher.finish();

        assert_eq!(
            actual_step, expected,
            "step-by-step mismatch for pattern `{}` on input {:?}: ours={}, regex crate={}",
            pattern, input, actual_step, expected
        );
    }

    /// `.*a.{3}bc` — counting constraint on a wildcard.
    #[test]
    fn test_counting() {
        let p = "^.*a.{3}bc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aybzbc");
        assert_matches_regex_crate(p, &re, "axaybzbc");
        assert_matches_regex_crate(p, &re, "a123bc");
        assert_matches_regex_crate(p, &re, "za999bc");
        // negatives: too few wildcard chars
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "a12bc");
        // negatives: wrong trailing literal
        assert_matches_regex_crate(p, &re, "a123bd");
        assert_matches_regex_crate(p, &re, "a123xc");
        // negatives: empty, no 'a' at all, missing suffix
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "bc");
        assert_matches_regex_crate(p, &re, "a123b");
        assert_matches_regex_crate(p, &re, "a123");
        assert_memory_size(p, &re, 760);
    }

    /// `(a|bc){1,2}` — flat range repetition with all combos up to 3.
    #[test]
    fn test_range() {
        use itertools::Itertools;

        let p = "^(a|bc){1,2}$";
        let re = build_regex_unchecked(p);

        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "bc");
        // negatives: empty, wrong chars, partial match
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "x");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ca");
        assert_matches_regex_crate(p, &re, "bca");

        // Two repetitions (all combos)
        for v in std::iter::repeat_n(["a", "bc"], 2)
            .map(|a| a.into_iter())
            .multi_cartesian_product()
        {
            let input = v.into_iter().collect::<String>();
            assert_matches_regex_crate(p, &re, &input);
        }

        // Three repetitions — should not match (max is 2)
        for v in std::iter::repeat_n(["a", "bc"], 3)
            .map(|a| a.into_iter())
            .multi_cartesian_product()
        {
            let input = v.into_iter().collect::<String>();
            assert_matches_regex_crate(p, &re, &input);
        }
        assert_memory_size(p, &re, 424);
    }

    /// `((a|bc){1,2}){2,3}` — nested counting constraints.
    #[test]
    fn test_nested_counting() {
        use itertools::Itertools;

        let p = "^((a|bc){1,2}){2,3}$";
        let re = build_regex_unchecked(p);

        // negatives: below outer min, wrong chars
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "bc");
        assert_matches_regex_crate(p, &re, "x");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "aax");

        for i in 2..=6 {
            for v in std::iter::repeat_n(["a", "bc"], i)
                .map(|a| a.into_iter())
                .multi_cartesian_product()
            {
                let input = v.into_iter().collect::<String>();
                assert_matches_regex_crate(p, &re, &input);
            }
        }

        for v in std::iter::repeat_n(["a", "bc"], 7)
            .map(|a| a.into_iter())
            .multi_cartesian_product()
        {
            let input = v.into_iter().collect::<String>();
            assert_matches_regex_crate(p, &re, &input);
        }
        assert_memory_size(p, &re, 504);
    }

    /// `(a|a?){2,3}` — epsilon-matchable body (the `a?` branch can match
    /// empty).  Exercises the epsilon-body detection logic in `addstate`.
    #[test]
    fn test_aaaaa() {
        let p = "^(a|a?){2,3}$";
        let re = build_regex_unchecked(p);

        // positives (epsilon branches make all lengths 0..=3 matchable)
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        // negatives: too many a's
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        // negatives: wrong characters
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "aab");
        assert_memory_size(p, &re, 424);
    }

    /// `a+` — basic one-or-more repetition.
    #[test]
    fn test_one_plus_basic() {
        let p = "^a+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "aab");
        assert_memory_size(p, &re, 264);
    }

    /// `.+` — one-or-more wildcard.
    #[test]
    fn test_one_plus_wildcard() {
        let p = "^.+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_memory_size(p, &re, 520);
    }

    /// `a+b+` — consecutive one-or-more repetitions.
    #[test]
    fn test_one_plus_catenation() {
        let p = "^a+b+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "abb");
        assert_matches_regex_crate(p, &re, "aabb");
        assert_matches_regex_crate(p, &re, "ba");
        assert_memory_size(p, &re, 344);
    }

    /// `(ab)+` — one-or-more of a multi-byte sequence.
    #[test]
    fn test_one_plus_group() {
        let p = "^(ab)+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "ababab");
        assert_matches_regex_crate(p, &re, "aba");
        assert_memory_size(p, &re, 304);
    }

    /// `(a|b)+` — one-or-more alternation.
    #[test]
    fn test_one_plus_alternate() {
        let p = "^(a|b)+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "bba");
        assert_matches_regex_crate(p, &re, "abab");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "ac");
        assert_matches_regex_crate(p, &re, "ca");
        assert_matches_regex_crate(p, &re, "abc");
        assert_memory_size(p, &re, 520);
    }

    /// `.*a.{3}b+c` — one-or-more mixed with counting constraints.
    #[test]
    fn test_one_plus_with_counting() {
        let p = "^.*a.{3}b+c$";
        let re = build_regex_unchecked(p);
        // positives
        assert_matches_regex_crate(p, &re, "a123bc");
        assert_matches_regex_crate(p, &re, "a123bbc");
        assert_matches_regex_crate(p, &re, "a123bbbc");
        assert_matches_regex_crate(p, &re, "xa123bc");
        assert_matches_regex_crate(p, &re, "xxxa999bc");
        // negatives: missing b+ section
        assert_matches_regex_crate(p, &re, "a123c");
        // negatives: too few wildcard chars
        assert_matches_regex_crate(p, &re, "a12bc");
        // negatives: wrong trailing literal
        assert_matches_regex_crate(p, &re, "a123bd");
        assert_matches_regex_crate(p, &re, "a123bx");
        // negatives: empty, no 'a', missing 'c'
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "x123bc");
        assert_matches_regex_crate(p, &re, "a123b");
        assert_memory_size(p, &re, 800);
    }

    /// `(a{2,3})+` — inner repetition, outer one-or-more.
    /// The body of `+` is itself a counted repetition.
    #[test]
    fn test_repetition_inside_one_plus() {
        let p = "^(a{2,3})+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaaaaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_memory_size(p, &re, 344);
    }

    /// `((a|bc){1,2})+` — inner range repetition of alternation, outer `+`.
    #[test]
    fn test_range_alternation_inside_one_plus() {
        use itertools::Itertools;

        let p = "^((a|bc){1,2})+$";
        let re = build_regex_unchecked(p);

        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "bc");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");

        // 2 through 6 atoms — exercises multiple iterations of the outer `+`
        for i in 2..=6 {
            for v in std::iter::repeat_n(["a", "bc"], i)
                .map(|a| a.into_iter())
                .multi_cartesian_product()
            {
                let input = v.into_iter().collect::<String>();
                assert_matches_regex_crate(p, &re, &input);
            }
        }
        assert_memory_size(p, &re, 464);
    }

    /// `(a+){2,3}` — inner one-or-more, outer counted repetition.
    #[test]
    fn test_one_plus_inside_repetition() {
        let p = "^(a+){2,3}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_memory_size(p, &re, 344);
    }

    /// `((a|b)+){2,4}` — inner `+` of alternation, outer counted repetition.
    #[test]
    fn test_one_plus_alternation_inside_repetition() {
        use itertools::Itertools;

        let p = "^((a|b)+){2,4}$";
        let re = build_regex_unchecked(p);

        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");

        for i in 2..=8 {
            for v in std::iter::repeat_n(["a", "b"], i)
                .map(|a| a.into_iter())
                .multi_cartesian_product()
            {
                let input = v.into_iter().collect::<String>();
                assert_matches_regex_crate(p, &re, &input);
            }
        }
        assert_memory_size(p, &re, 600);
    }

    /// `(a+b{2,3})+` — inner `+` and inner repetition side-by-side,
    /// wrapped in outer `+`.
    #[test]
    fn test_mixed_plus_and_repetition_inside_one_plus() {
        let p = "^(a+b{2,3})+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abb");
        assert_matches_regex_crate(p, &re, "abbb");
        assert_matches_regex_crate(p, &re, "abbbb");
        assert_matches_regex_crate(p, &re, "aabb");
        assert_matches_regex_crate(p, &re, "aabbb");
        assert_matches_regex_crate(p, &re, "abbabb");
        assert_matches_regex_crate(p, &re, "abbaabb");
        assert_matches_regex_crate(p, &re, "abbabbbabb");
        assert_matches_regex_crate(p, &re, "aabbaabbb");
        assert_matches_regex_crate(p, &re, "aabbbaabbb");
        assert_memory_size(p, &re, 424);
    }

    // -- min=0 repetition tests ---------------------------------------------

    /// `a{0,2}` — zero to two occurrences of a single byte.
    #[test]
    fn test_min_zero_basic() {
        let p = "^a{0,2}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_memory_size(p, &re, 344);
    }

    /// `a{0,1}` — equivalent to `a?`.
    #[test]
    fn test_min_zero_max_one() {
        let p = "^a{0,1}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "b");
        assert_memory_size(p, &re, 264);
    }

    /// `(a|bc){0,3}` — zero to three of an alternation.
    #[test]
    fn test_min_zero_alternation() {
        use itertools::Itertools;

        let p = "^(a|bc){0,3}$";
        let re = build_regex_unchecked(p);

        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "bc");
        assert_matches_regex_crate(p, &re, "b");

        for i in 2..=4 {
            for v in std::iter::repeat_n(["a", "bc"], i)
                .map(|a| a.into_iter())
                .multi_cartesian_product()
            {
                let input = v.into_iter().collect::<String>();
                assert_matches_regex_crate(p, &re, &input);
            }
        }
        assert_memory_size(p, &re, 464);
    }

    /// `a{0,}` — zero or more, lowered to `a*` (no counter overhead).
    #[test]
    fn test_min_zero_unbounded() {
        let p = "^a{0,}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        // negatives
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "baa");
        assert_memory_size(p, &re, 264);
    }

    /// `(ab){0,}` — zero or more of a group, lowered to `(ab)*`.
    #[test]
    fn test_min_zero_unbounded_group() {
        let p = "^(ab){0,}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "ababab");
        assert_matches_regex_crate(p, &re, "aba");
        assert_memory_size(p, &re, 304);
    }

    /// `x(a{0,2})+y` — min=0 repetition nested inside `+`.
    #[test]
    fn test_min_zero_inside_one_plus() {
        let p = "^x(a{0,2})+y$";
        let re = build_regex_unchecked(p);
        // positives
        assert_matches_regex_crate(p, &re, "xy");
        assert_matches_regex_crate(p, &re, "xay");
        assert_matches_regex_crate(p, &re, "xaay");
        assert_matches_regex_crate(p, &re, "xaaay");
        assert_matches_regex_crate(p, &re, "xaaaay");
        // negatives: missing suffix/prefix
        assert_matches_regex_crate(p, &re, "x");
        assert_matches_regex_crate(p, &re, "y");
        assert_matches_regex_crate(p, &re, "");
        // negatives: wrong characters in middle
        assert_matches_regex_crate(p, &re, "xby");
        assert_matches_regex_crate(p, &re, "xaby");
        // negatives: wrong delimiters
        assert_matches_regex_crate(p, &re, "ay");
        assert_matches_regex_crate(p, &re, "xa");
        assert_matches_regex_crate(p, &re, "aay");
        assert_memory_size(p, &re, 464);
    }

    /// `(a{0,2}){2,3}` — min=0 inner, counted outer.
    #[test]
    fn test_min_zero_inside_repetition() {
        let p = "^(a{0,2}){2,3}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_memory_size(p, &re, 424);
    }

    /// `(a+){0,3}` — `+` inside a min=0 counted repetition.
    #[test]
    fn test_one_plus_inside_min_zero_repetition() {
        let p = "^(a+){0,3}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_memory_size(p, &re, 384);
    }

    /// `.{0,3}` — min=0 repetition on wildcard.
    #[test]
    fn test_min_zero_wildcard() {
        let p = "^.{0,3}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abcd");
        assert_memory_size(p, &re, 600);
    }

    /// `a{0,3}` — same as `a{0,3}` (the old test used `min: None`).
    #[test]
    fn test_none_min_repetition() {
        let p = "^a{0,3}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        // negatives: above max
        assert_matches_regex_crate(p, &re, "aaaa");
        // negatives: wrong characters
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_memory_size(p, &re, 344);
    }

    // -- Standalone primitive tests ------------------------------------------

    /// `a` — single literal byte.
    #[test]
    fn test_literal_single() {
        let p = "^a$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ba");
        assert_memory_size(p, &re, 224);
    }

    /// `abc` — multi-byte literal concatenation.
    #[test]
    fn test_literal_multi() {
        let p = "^abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abd");
        assert_matches_regex_crate(p, &re, "abcd");
        assert_matches_regex_crate(p, &re, "xabc");
        assert_matches_regex_crate(p, &re, "abcx");
        assert_matches_regex_crate(p, &re, "xabcx");
        assert_matches_regex_crate(p, &re, "cba");
        assert_matches_regex_crate(p, &re, "bac");
        assert_memory_size(p, &re, 304);
    }

    /// `.` — bare wildcard (matches exactly one byte).
    #[test]
    fn test_dot_single() {
        let p = "^.$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "z");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, " ");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_memory_size(p, &re, 480);
    }

    /// `a|bc` — bare alternation (no repetition).
    #[test]
    fn test_alternation_bare() {
        let p = "^(a|bc)$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "bc");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "x");
        assert_matches_regex_crate(p, &re, "bca");
        assert_memory_size(p, &re, 344);
    }

    /// `a|b|c` — three-way alternation.
    #[test]
    fn test_alternation_three_way() {
        let p = "^(a|b|c)$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "d");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_memory_size(p, &re, 480);
    }

    /// `a?` — standalone zero-or-one.
    #[test]
    fn test_question_mark_single() {
        let p = "^a?$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        // negatives
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_memory_size(p, &re, 264);
    }

    /// `(ab)?` — zero-or-one of a group.
    #[test]
    fn test_question_mark_group() {
        let p = "^(ab)?$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "ab");
        // negatives
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_memory_size(p, &re, 304);
    }

    /// `a?b` — optional prefix followed by a literal.
    #[test]
    fn test_question_mark_prefix() {
        let p = "^a?b$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "bb");
        assert_matches_regex_crate(p, &re, "cb");
        assert_matches_regex_crate(p, &re, "abc");
        assert_memory_size(p, &re, 304);
    }

    /// `a*` — standalone zero-or-more.
    #[test]
    fn test_star_single() {
        let p = "^a*$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        // negatives
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "baa");
        assert_memory_size(p, &re, 264);
    }

    /// `(ab)*` — zero-or-more of a group.
    #[test]
    fn test_star_group() {
        let p = "^(ab)*$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "ababab");
        // negatives
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "aba");
        assert_matches_regex_crate(p, &re, "abba");
        assert_matches_regex_crate(p, &re, "abc");
        assert_memory_size(p, &re, 304);
    }

    /// `a*b` — star followed by a literal.
    #[test]
    fn test_star_then_literal() {
        let p = "^a*b$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "aaab");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "bb");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "aabb");
        assert_memory_size(p, &re, 304);
    }

    /// `a{2,}` — unbounded min with n>0.
    #[test]
    fn test_min_n_unbounded() {
        let p = "^a{2,}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "baa");
        assert_memory_size(p, &re, 304);
    }

    /// `(ab){2,}` — unbounded min of a group.
    #[test]
    fn test_min_n_unbounded_group() {
        let p = "^(ab){2,}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "ababab");
        assert_matches_regex_crate(p, &re, "abababab");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aba");
        assert_matches_regex_crate(p, &re, "abba");
        assert_matches_regex_crate(p, &re, "ababc");
        assert_matches_regex_crate(p, &re, "xabab");
        assert_memory_size(p, &re, 344);
    }

    /// `a{3,5}` — bounded min>0 range.
    #[test]
    fn test_bounded_range() {
        let p = "^a{3,5}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaaaaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "aaab");
        assert_matches_regex_crate(p, &re, "baaa");
        assert_memory_size(p, &re, 304);
    }

    /// `a{3,3}` — exact repetition.
    #[test]
    fn test_exact_repetition() {
        let p = "^a{3,3}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aaa");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "bbb");
        assert_memory_size(p, &re, 304);
    }

    // -- Byte class tests ---------------------------------------------------

    /// `[a-c]` — a small contiguous byte range (Class::Bytes).
    #[test]
    fn test_byte_class_range() {
        let p = "^[a-c]$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "d");
        assert_matches_regex_crate(p, &re, "ab");
        assert_memory_size(p, &re, 480);
    }

    /// `[a-c]+` — one-or-more of a byte class.
    #[test]
    fn test_byte_class_one_plus() {
        let p = "^[a-c]+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "cba");
        assert_matches_regex_crate(p, &re, "abcd");
        assert_matches_regex_crate(p, &re, "d");
        assert_memory_size(p, &re, 520);
    }

    /// `[a-c]{2,3}` — counted repetition of a byte class.
    #[test]
    fn test_byte_class_counted() {
        let p = "^[a-c]{2,3}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abca");
        assert_matches_regex_crate(p, &re, "cc");
        assert_matches_regex_crate(p, &re, "dd");
        assert_memory_size(p, &re, 560);
    }

    /// `[ax]` — disjoint single bytes (multi-range Class::Bytes).
    #[test]
    fn test_byte_class_disjoint() {
        let p = "^[ax]$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "x");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ax");
        assert_memory_size(p, &re, 480);
    }

    /// `[a-cx-z]+` — multiple disjoint ranges in a byte class.
    #[test]
    fn test_byte_class_multi_range() {
        let p = "^[a-cx-z]+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "x");
        assert_matches_regex_crate(p, &re, "axbycz");
        assert_matches_regex_crate(p, &re, "d");
        assert_matches_regex_crate(p, &re, "w");
        assert_matches_regex_crate(p, &re, "abcxyz");
        assert_memory_size(p, &re, 520);
    }

    /// `[a-c].*[x-z]` — byte classes mixed with wildcard.
    #[test]
    fn test_byte_class_with_wildcard() {
        let p = "^[a-c].*[x-z]$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "ax");
        assert_matches_regex_crate(p, &re, "a123z");
        assert_matches_regex_crate(p, &re, "bx");
        assert_matches_regex_crate(p, &re, "dx");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_memory_size(p, &re, 1112);
    }

    // -- Predefined character class tests -----------------------------------

    /// `\d` — matches a single ASCII digit.
    #[test]
    fn test_digit() {
        let p = r"^\d$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "5");
        assert_matches_regex_crate(p, &re, "9");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "z");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "00");
        assert_matches_regex_crate(p, &re, "12");
        assert_memory_size(p, &re, 480);
    }

    /// `\d+` — one-or-more digits.
    #[test]
    fn test_digit_plus() {
        let p = r"^\d+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "42");
        assert_matches_regex_crate(p, &re, "999");
        assert_matches_regex_crate(p, &re, "0123456789");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "12a");
        assert_matches_regex_crate(p, &re, "a12");
        assert_matches_regex_crate(p, &re, "1 2");
        assert_memory_size(p, &re, 520);
    }

    /// `\d{3,5}` — counted digit repetition.
    #[test]
    fn test_digit_counted() {
        let p = r"^\d{3,5}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "123");
        assert_matches_regex_crate(p, &re, "1234");
        assert_matches_regex_crate(p, &re, "12345");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "1");
        assert_matches_regex_crate(p, &re, "12");
        assert_matches_regex_crate(p, &re, "123456");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "12a");
        assert_memory_size(p, &re, 560);
    }

    /// `\D` — matches a single non-digit byte.
    #[test]
    fn test_non_digit() {
        let p = r"^\D$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "z");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "!");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "5");
        assert_matches_regex_crate(p, &re, "9");
        assert_matches_regex_crate(p, &re, "aa");
        assert_memory_size(p, &re, 480);
    }

    /// `\D+` — one-or-more non-digits.
    #[test]
    fn test_non_digit_plus() {
        let p = r"^\D+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "hello world");
        assert_matches_regex_crate(p, &re, "!@#");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "abc1");
        assert_matches_regex_crate(p, &re, "1abc");
        assert_memory_size(p, &re, 520);
    }

    /// `\s` — matches a single ASCII whitespace byte.
    #[test]
    fn test_space() {
        let p = r"^\s$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "\t");
        assert_matches_regex_crate(p, &re, "\n");
        assert_matches_regex_crate(p, &re, "\r");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "  ");
        assert_memory_size(p, &re, 480);
    }

    /// `\s+` — one-or-more whitespace.
    #[test]
    fn test_space_plus() {
        let p = r"^\s+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "   ");
        assert_matches_regex_crate(p, &re, " \t\n\r");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, " a");
        assert_matches_regex_crate(p, &re, "a ");
        assert_memory_size(p, &re, 520);
    }

    /// `\S` — matches a single non-whitespace byte.
    #[test]
    fn test_non_space() {
        let p = r"^\S$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "!");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "\t");
        assert_matches_regex_crate(p, &re, "\n");
        assert_matches_regex_crate(p, &re, "aa");
        assert_memory_size(p, &re, 480);
    }

    /// `\S+` — one-or-more non-whitespace.
    #[test]
    fn test_non_space_plus() {
        let p = r"^\S+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "123");
        assert_matches_regex_crate(p, &re, "a1!");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "a b");
        assert_matches_regex_crate(p, &re, " abc");
        assert_memory_size(p, &re, 520);
    }

    /// `\w` — matches a single ASCII word byte (`[0-9A-Za-z_]`).
    #[test]
    fn test_word() {
        let p = r"^\w$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "Z");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "9");
        assert_matches_regex_crate(p, &re, "_");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "!");
        assert_matches_regex_crate(p, &re, "-");
        assert_matches_regex_crate(p, &re, "ab");
        assert_memory_size(p, &re, 480);
    }

    /// `\w+` — one-or-more word bytes.
    #[test]
    fn test_word_plus() {
        let p = r"^\w+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "hello");
        assert_matches_regex_crate(p, &re, "foo_bar");
        assert_matches_regex_crate(p, &re, "x123");
        assert_matches_regex_crate(p, &re, "___");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "hello world");
        assert_matches_regex_crate(p, &re, "foo-bar");
        assert_memory_size(p, &re, 520);
    }

    /// `\w{2,4}` — counted word repetition.
    #[test]
    fn test_word_counted() {
        let p = r"^\w{2,4}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "a1_Z");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "abcde");
        assert_matches_regex_crate(p, &re, "!!");
        assert_matches_regex_crate(p, &re, "a b");
        assert_memory_size(p, &re, 560);
    }

    /// `\W` — matches a single non-word byte.
    #[test]
    fn test_non_word() {
        let p = r"^\W$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "!");
        assert_matches_regex_crate(p, &re, "-");
        assert_matches_regex_crate(p, &re, ".");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "_");
        assert_matches_regex_crate(p, &re, "  ");
        assert_memory_size(p, &re, 480);
    }

    /// `\W+` — one-or-more non-word bytes.
    #[test]
    fn test_non_word_plus() {
        let p = r"^\W+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "!@#");
        assert_matches_regex_crate(p, &re, " - ");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, " a ");
        assert_matches_regex_crate(p, &re, "!a!");
        assert_memory_size(p, &re, 520);
    }

    /// `\d+\s+\w+` — mixed predefined classes in concatenation.
    #[test]
    fn test_predefined_mixed() {
        let p = r"^\d+\s+\w+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "42 hello");
        assert_matches_regex_crate(p, &re, "0\tfoo");
        assert_matches_regex_crate(p, &re, "123  x");
        assert_matches_regex_crate(p, &re, "7 _");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "42");
        assert_matches_regex_crate(p, &re, "42 ");
        assert_matches_regex_crate(p, &re, " hello");
        assert_matches_regex_crate(p, &re, "hello 42");
        assert_matches_regex_crate(p, &re, "42hello");
        assert_memory_size(p, &re, 1192);
    }

    // -- Byte-class deduplication tests --------------------------------------

    /// Build a compiled [`Regex`] from a pattern string *without*
    /// asserting a specific memory size.  Used by dedup tests that
    /// compare sizes relatively rather than absolutely.
    fn build_regex_unchecked(pattern: &str) -> Regex {
        let hir = parse_hir_bytes(pattern);
        let mut builder = RegexBuilder::default();
        builder
            .build(&hir)
            .expect("our builder should accept the HIR")
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
        // join — so the difference is exactly one State.
        let state_size = std::mem::size_of::<State>();
        assert_eq!(
            two.memory_size() - one.memory_size(),
            state_size,
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
        assert_eq!(
            two_wild.memory_size() - one_wild.memory_size(),
            state_size,
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

    /// `^abc` — start-anchored only.  Matches strings that start with
    /// "abc", regardless of what follows.
    #[test]
    fn test_anchor_start_only() {
        let p = "^abc";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abcdef");
        assert_matches_regex_crate(p, &re, "abc123");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xabc");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abd");
        assert_matches_regex_crate(p, &re, "zabc");
    }

    /// `abc$` — end-anchored only.  Matches strings that end with "abc",
    /// regardless of what precedes.
    #[test]
    fn test_anchor_end_only() {
        let p = "abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "xabc");
        assert_matches_regex_crate(p, &re, "123abc");
        assert_matches_regex_crate(p, &re, "xxabc");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "abcx");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abcabd");
    }

    /// `abc` — fully unanchored.  Matches any string containing "abc".
    #[test]
    fn test_unanchored_literal() {
        let p = "abc";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "xabc");
        assert_matches_regex_crate(p, &re, "abcx");
        assert_matches_regex_crate(p, &re, "xabcx");
        assert_matches_regex_crate(p, &re, "xxabcxx");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abd");
        assert_matches_regex_crate(p, &re, "axbc");
        assert_matches_regex_crate(p, &re, "bca");
    }

    /// `^a.b` — start-anchored with wildcard.
    #[test]
    fn test_anchor_start_wildcard() {
        let p = "^a.b";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "axb");
        assert_matches_regex_crate(p, &re, "axbyyy");
        assert_matches_regex_crate(p, &re, "a1b");
        // negatives
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "yaxb");
        assert_matches_regex_crate(p, &re, "");
    }

    /// `a.b$` — end-anchored with wildcard.
    #[test]
    fn test_anchor_end_wildcard() {
        let p = "a.b$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "axb");
        assert_matches_regex_crate(p, &re, "yyyaxb");
        assert_matches_regex_crate(p, &re, "a1b");
        // negatives
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "axby");
        assert_matches_regex_crate(p, &re, "");
    }

    /// `^a+b+$` — anchored with quantifiers (non-counter).
    #[test]
    fn test_both_anchors_quantifiers() {
        let p = "^a+b+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "abb");
        assert_matches_regex_crate(p, &re, "aabb");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "xab");
        assert_matches_regex_crate(p, &re, "abx");
    }

    /// `a+b+` — unanchored with quantifiers (non-counter).
    #[test]
    fn test_unanchored_quantifiers() {
        let p = "a+b+";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "abb");
        assert_matches_regex_crate(p, &re, "xab");
        assert_matches_regex_crate(p, &re, "xaabb");
        assert_matches_regex_crate(p, &re, "abx");
        assert_matches_regex_crate(p, &re, "xabx");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "xyz");
    }

    /// `^a+` — start-anchored with one-or-more.
    #[test]
    fn test_anchor_start_one_plus() {
        let p = "^a+";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "abc");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "baa");
    }

    /// `a+$` — end-anchored with one-or-more.
    #[test]
    fn test_anchor_end_one_plus() {
        let p = "a+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "baa");
        assert_matches_regex_crate(p, &re, "xxa");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
    }

    /// `^$` — empty string, both anchors.
    #[test]
    fn test_anchors_empty() {
        let p = "^$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        // negatives
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
    }

    /// `^` — start anchor only, matches any string.
    #[test]
    fn test_anchor_start_bare() {
        let p = "^";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "abc");
    }

    /// `$` — end anchor only, matches any string.
    #[test]
    fn test_anchor_end_bare() {
        let p = "$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "abc");
    }

    /// `(a|b)` — unanchored alternation, matches any string containing
    /// 'a' or 'b'.
    #[test]
    fn test_unanchored_alternation() {
        let p = "(a|b)";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "xa");
        assert_matches_regex_crate(p, &re, "bx");
        assert_matches_regex_crate(p, &re, "xax");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "xyz");
    }

    /// `a*b` — unanchored star-then-literal.
    #[test]
    fn test_unanchored_star_literal() {
        let p = "a*b";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "xb");
        assert_matches_regex_crate(p, &re, "xab");
        assert_matches_regex_crate(p, &re, "bx");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "xyz");
    }

    /// `a|^b` — anchor inside an alternation arm.  Matches any string
    /// containing `a` (unanchored), OR any string starting with `b`.
    #[test]
    fn test_anchor_in_alternation_start() {
        let p = "a|^b";
        let re = build_regex_unchecked(p);
        // positives via the `a` arm (unanchored)
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "xa");
        assert_matches_regex_crate(p, &re, "ax");
        assert_matches_regex_crate(p, &re, "xax");
        // positives via the `^b` arm
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "bx");
        assert_matches_regex_crate(p, &re, "bxx");
        // positives via both arms
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "ab");
        // negatives: no `a` and doesn't start with `b`
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "xb");
        assert_matches_regex_crate(p, &re, "xbx");
        assert_matches_regex_crate(p, &re, "xyz");
    }

    /// `a$|b` — anchor inside an alternation arm.  Matches any string
    /// ending with `a`, OR any string containing `b` (unanchored).
    #[test]
    fn test_anchor_in_alternation_end() {
        let p = "a$|b";
        let re = build_regex_unchecked(p);
        // positives via the `a$` arm
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "xa");
        assert_matches_regex_crate(p, &re, "xxa");
        // positives via the `b` arm (unanchored)
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "xb");
        assert_matches_regex_crate(p, &re, "bx");
        assert_matches_regex_crate(p, &re, "xbx");
        // positives via both arms
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "ab");
        // negatives: doesn't end with `a` and no `b`
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "ax");
        assert_matches_regex_crate(p, &re, "xax");
        assert_matches_regex_crate(p, &re, "xyz");
    }

    // -------------------------------------------------------------------
    // Multiline assertions: (?m:^) = StartLF, (?m:$) = EndLF
    // -------------------------------------------------------------------

    #[test]
    fn test_multiline_start_basic() {
        let p = "(?m)^abc";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abc\ndef");
        assert_matches_regex_crate(p, &re, "xxx\nabc");
        assert_matches_regex_crate(p, &re, "xxx\nabc\nyyy");
        assert_matches_regex_crate(p, &re, "\nabc");
        assert_matches_regex_crate(p, &re, "xabc"); // no match: not at line start
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "\n");
    }

    #[test]
    fn test_multiline_end_basic() {
        let p = "(?m)abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abc\ndef");
        assert_matches_regex_crate(p, &re, "xxx\nabc");
        assert_matches_regex_crate(p, &re, "xxxabc\nyyy");
        assert_matches_regex_crate(p, &re, "abc\n");
        assert_matches_regex_crate(p, &re, "abcx"); // no match: not at line end
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "\n");
    }

    #[test]
    fn test_multiline_both_anchors() {
        let p = "(?m)^abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abc\ndef");
        assert_matches_regex_crate(p, &re, "def\nabc");
        assert_matches_regex_crate(p, &re, "xxx\nabc\nyyy");
        assert_matches_regex_crate(p, &re, "\nabc\n");
        assert_matches_regex_crate(p, &re, "abc\n");
        assert_matches_regex_crate(p, &re, "\nabc");
        assert_matches_regex_crate(p, &re, "xabc"); // no match
        assert_matches_regex_crate(p, &re, "abcx"); // no match
        assert_matches_regex_crate(p, &re, "xabcx"); // no match
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "\n");
        assert_matches_regex_crate(p, &re, "\n\n");
    }

    #[test]
    fn test_multiline_multi_lines() {
        // Pattern that matches "abc" on its own line somewhere in the input.
        let p = "(?m)^abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc\ndef\nghi");
        assert_matches_regex_crate(p, &re, "def\nabc\nghi");
        assert_matches_regex_crate(p, &re, "def\nghi\nabc");
        assert_matches_regex_crate(p, &re, "def\nghi\njkl"); // no match
    }

    #[test]
    fn test_multiline_catenation_across_newline() {
        // Pattern: line ending with abc, newline, line starting with def
        let p = r"(?m)abc$\n^def";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc\ndef");
        assert_matches_regex_crate(p, &re, "xxx\nabc\ndef\nyyy");
        assert_matches_regex_crate(p, &re, "abc\nxef"); // no match
        assert_matches_regex_crate(p, &re, "abcdef"); // no match
    }

    #[test]
    fn test_multiline_with_dot_plus() {
        let p = "(?m)^.+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abc\ndef");
        assert_matches_regex_crate(p, &re, "\nabc");
        assert_matches_regex_crate(p, &re, "abc\n");
        assert_matches_regex_crate(p, &re, ""); // no match: .+ needs >=1 char
        assert_matches_regex_crate(p, &re, "\n"); // no match: each line is empty
        assert_matches_regex_crate(p, &re, "\n\n");
    }

    #[test]
    fn test_multiline_with_counting() {
        let p = r"(?m)^\d{2,4}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "12");
        assert_matches_regex_crate(p, &re, "123");
        assert_matches_regex_crate(p, &re, "1234");
        assert_matches_regex_crate(p, &re, "1"); // too short
        assert_matches_regex_crate(p, &re, "12345"); // too long
        assert_matches_regex_crate(p, &re, "xx\n12\nyy");
        assert_matches_regex_crate(p, &re, "xx\n12345\nyy"); // no match on that line
        assert_matches_regex_crate(p, &re, "xx\n1\nyy"); // too short
        assert_matches_regex_crate(p, &re, "12\n1234\n12345");
    }

    #[test]
    fn test_multiline_edge_cases_empty() {
        // (?m)^$ matches any empty line (including empty input).
        let p = "(?m)^$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "\n");
        assert_matches_regex_crate(p, &re, "\n\n");
        assert_matches_regex_crate(p, &re, "abc"); // no empty line
        assert_matches_regex_crate(p, &re, "abc\n"); // empty line after trailing \n
        assert_matches_regex_crate(p, &re, "\nabc"); // empty line before \n
        assert_matches_regex_crate(p, &re, "abc\n\ndef"); // empty line between
    }

    #[test]
    fn test_multiline_start_only_empty_lines() {
        let p = "(?m)^";
        let re = build_regex_unchecked(p);
        // ^ in multiline always matches (every input has at least position 0).
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "\n");
    }

    #[test]
    fn test_multiline_end_only() {
        let p = "(?m)$";
        let re = build_regex_unchecked(p);
        // $ in multiline always matches.
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "\n");
    }

    #[test]
    fn test_multiline_alternation() {
        let p = "(?m)^(abc|def)$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "def");
        assert_matches_regex_crate(p, &re, "abc\ndef");
        assert_matches_regex_crate(p, &re, "ghi\nabc\njkl");
        assert_matches_regex_crate(p, &re, "ghi\njkl"); // no match
        assert_matches_regex_crate(p, &re, "abcdef"); // no match
    }

    #[test]
    fn test_multiline_mixed_with_nonmultiline() {
        // Non-multiline ^ and multiline $ in the same pattern.
        // regex-syntax: ^ without (?m) = Look::Start, $ with (?m) = Look::EndLF
        let p = "^abc(?m:$)";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abc\ndef");
        assert_matches_regex_crate(p, &re, "xabc"); // no match: ^ is not multiline
        assert_matches_regex_crate(p, &re, "\nabc"); // no match: ^ is not multiline
    }

    // -------------------------------------------------------------------
    // CRLF multiline tests  ((?Rm) → StartCRLF / EndCRLF)
    // -------------------------------------------------------------------

    #[test]
    fn test_crlf_start_basic() {
        // (?Rm:^) matches at start of input, after \n, and after bare \r
        // (but NOT between \r and \n — \r\n is a single line terminator).
        let p = r"(?Rm)^abc";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "\nabc");
        assert_matches_regex_crate(p, &re, "\rabc");
        assert_matches_regex_crate(p, &re, "\r\nabc");
        assert_matches_regex_crate(p, &re, "xxx\nabc");
        assert_matches_regex_crate(p, &re, "xxx\rabc");
        assert_matches_regex_crate(p, &re, "xxx\r\nabc");
        assert_matches_regex_crate(p, &re, "xabc"); // no match
    }

    #[test]
    fn test_crlf_end_basic() {
        // (?Rm:$) matches at end of input, before \r, and before \n
        // (but NOT between \r and \n).
        let p = r"(?Rm)abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abc\n");
        assert_matches_regex_crate(p, &re, "abc\r");
        assert_matches_regex_crate(p, &re, "abc\r\n");
        assert_matches_regex_crate(p, &re, "abc\nxxx");
        assert_matches_regex_crate(p, &re, "abc\rxxx");
        assert_matches_regex_crate(p, &re, "abc\r\nxxx");
        assert_matches_regex_crate(p, &re, "abcx"); // no match
    }

    #[test]
    fn test_crlf_both_anchors() {
        let p = r"(?Rm)^abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abc\r\n");
        assert_matches_regex_crate(p, &re, "\r\nabc");
        assert_matches_regex_crate(p, &re, "\r\nabc\r\n");
        assert_matches_regex_crate(p, &re, "xxx\r\nabc\r\nyyy");
        assert_matches_regex_crate(p, &re, "abc\n");
        assert_matches_regex_crate(p, &re, "\nabc\n");
        assert_matches_regex_crate(p, &re, "abc\r");
        assert_matches_regex_crate(p, &re, "\rabc\r");
        assert_matches_regex_crate(p, &re, "xabc"); // no match
        assert_matches_regex_crate(p, &re, "abcx"); // no match
    }

    #[test]
    fn test_crlf_empty_lines() {
        // (?Rm)^$ matches any empty line, including empty input and
        // positions between \r\n line terminators.
        let p = r"(?Rm)^$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "\n");
        assert_matches_regex_crate(p, &re, "\r\n");
        assert_matches_regex_crate(p, &re, "\r");
        assert_matches_regex_crate(p, &re, "\r\n\r\n");
        assert_matches_regex_crate(p, &re, "abc\r\n\r\ndef");
        assert_matches_regex_crate(p, &re, "abc"); // no empty line
    }

    #[test]
    fn test_crlf_multiline_vs_lf() {
        // Demonstrate difference between LF-mode and CRLF-mode.
        // In CRLF mode, \r\n is a single terminator, so ^ after \r
        // does NOT match between \r and \n.
        let p_crlf = r"(?Rm)^abc$";
        let re_crlf = build_regex_unchecked(p_crlf);

        // \r\nabc\r\n — should match in CRLF mode
        assert_matches_regex_crate(p_crlf, &re_crlf, "\r\nabc\r\n");

        let p_lf = r"(?m)^abc$";
        let re_lf = build_regex_unchecked(p_lf);

        // In LF mode, \r is NOT a line terminator:
        // \r\nabc\r\n — the "line" is "\rabc\r" which doesn't match "abc"
        assert_matches_regex_crate(p_lf, &re_lf, "\r\nabc\r\n");
    }

    #[test]
    fn test_crlf_with_dot_plus() {
        let p = r"(?Rm)^.+$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abc\r\ndef");
        assert_matches_regex_crate(p, &re, "\r\nabc");
        assert_matches_regex_crate(p, &re, "abc\r\n");
        assert_matches_regex_crate(p, &re, ""); // no match
        assert_matches_regex_crate(p, &re, "\r\n"); // no match: lines are empty
        assert_matches_regex_crate(p, &re, "\r\n\r\n");
    }

    #[test]
    fn test_crlf_with_counting() {
        let p = r"(?Rm)^\d{2,4}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "12");
        assert_matches_regex_crate(p, &re, "1234");
        assert_matches_regex_crate(p, &re, "12345"); // too long
        assert_matches_regex_crate(p, &re, "xx\r\n12\r\nyy");
        assert_matches_regex_crate(p, &re, "xx\r\n1\r\nyy"); // too short
    }

    #[test]
    fn test_crlf_bare_cr_as_line_terminator() {
        // Bare \r (without following \n) acts as a line terminator in
        // CRLF mode.
        let p = r"(?Rm)^abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "xxx\rabc\ryyy");
        assert_matches_regex_crate(p, &re, "xxx\rabc");
        assert_matches_regex_crate(p, &re, "abc\ryyy");
    }

    #[test]
    fn test_crlf_end_before_cr() {
        // EndCRLF fires before \r.
        let p = r"(?Rm)abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc\r");
        assert_matches_regex_crate(p, &re, "abc\r\n");
        assert_matches_regex_crate(p, &re, "abc\rxxx");
    }

    #[test]
    fn test_crlf_start_after_lf_not_crlf_middle() {
        // StartCRLF after \n: matches. Between \r and \n: does NOT match.
        let p = r"(?Rm)^x";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "\nx"); // match: after \n
        assert_matches_regex_crate(p, &re, "\r\nx"); // match: after \r\n
        assert_matches_regex_crate(p, &re, "\rx"); // match: after bare \r
    }

    #[test]
    fn test_crlf_mixed_terminators() {
        // Input with mixed \n, \r, and \r\n terminators.
        let p = r"(?Rm)^abc$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "xxx\nabc\ryyy");
        assert_matches_regex_crate(p, &re, "xxx\rabc\nyyy");
        assert_matches_regex_crate(p, &re, "xxx\r\nabc\r\nyyy");
        assert_matches_regex_crate(p, &re, "xxx\nabc\r\nyyy");
        assert_matches_regex_crate(p, &re, "xxx\r\nabc\nyyy");
    }

    #[test]
    fn test_crlf_end_only() {
        let p = r"(?Rm)$";
        let re = build_regex_unchecked(p);
        // EndCRLF always matches (every input has end-of-input).
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "\r\n");
        assert_matches_regex_crate(p, &re, "\r");
        assert_matches_regex_crate(p, &re, "\n");
    }

    #[test]
    fn test_crlf_start_only() {
        let p = r"(?Rm)^";
        let re = build_regex_unchecked(p);
        // StartCRLF always matches (every input has position 0).
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "\r\n");
        assert_matches_regex_crate(p, &re, "\r");
        assert_matches_regex_crate(p, &re, "\n");
    }

    // ===================================================================
    // Stale Becchi counter fix: regression + coverage tests
    //
    // These tests exercise unanchored, partially-anchored, and
    // multi-chunk scenarios that were previously untested and would
    // have triggered phantom counter accumulation before the fix.
    // ===================================================================

    /// Simple unanchored counters: `\w{3,5}`, `a{3}`, `[0-9]{4}`.
    /// Before the fix, gap-separated inputs would accumulate phantom
    /// counts across non-matching positions.
    #[test]
    fn test_unanchored_counter_simple() {
        // \w{3,5}
        let p = r"\w{3,5}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abcde");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "a b c");
        assert_matches_regex_crate(p, &re, " abc ");
        assert_matches_regex_crate(p, &re, "x y z");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "abcdef");
        assert_matches_regex_crate(p, &re, "ab cde fg");

        // a{3} — exact count
        let p = "a{3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "xaaax");
        assert_matches_regex_crate(p, &re, "a a a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "baaab");

        // [0-9]{4} — digit run
        let p = "[0-9]{4}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "1234");
        assert_matches_regex_crate(p, &re, "123");
        assert_matches_regex_crate(p, &re, "a1234b");
        assert_matches_regex_crate(p, &re, "1 2 3 4");
        assert_matches_regex_crate(p, &re, "12345");
        assert_matches_regex_crate(p, &re, "");
    }

    /// Unanchored counter with alternation body: `(a|bc){2,4}`.
    #[test]
    fn test_unanchored_counter_alternation_body() {
        let p = "(a|bc){2,4}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "bca");
        assert_matches_regex_crate(p, &re, "bcbc");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "bc");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xaay");
        assert_matches_regex_crate(p, &re, "xbcay");
        assert_matches_regex_crate(p, &re, "a bc a");
        assert_matches_regex_crate(p, &re, "abcbca");
        assert_matches_regex_crate(p, &re, "abcbcbc");
    }

    /// Unanchored counter with multi-byte body: `(ab){2,3}`.
    #[test]
    fn test_unanchored_counter_multi_byte_body() {
        let p = "(ab){2,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "ababab");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xababx");
        assert_matches_regex_crate(p, &re, "ab ab ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "abb");
        assert_matches_regex_crate(p, &re, "abababab");
    }

    /// Unanchored nested counting: `((a|b){1,2}){2,3}`.
    #[test]
    fn test_unanchored_counter_nested() {
        let p = "((a|b){1,2}){2,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aabb");
        assert_matches_regex_crate(p, &re, "ababab");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xaby");
        assert_matches_regex_crate(p, &re, "a b");
        assert_matches_regex_crate(p, &re, "abba");
    }

    /// Unanchored counter with min=0: `a{0,3}`.
    #[test]
    fn test_unanchored_counter_min_zero() {
        let p = "a{0,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "xaaax");
    }

    /// Unanchored unbounded counter: `a{2,}`.
    #[test]
    fn test_unanchored_counter_unbounded() {
        let p = "a{2,}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xaay");
        assert_matches_regex_crate(p, &re, "a a");
        assert_matches_regex_crate(p, &re, "baaaab");
    }

    /// Unanchored counter with wildcard body: `.{3,5}`.
    #[test]
    fn test_unanchored_counter_wildcard_body() {
        let p = ".{3,5}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abcde");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "abcdef");
    }

    /// Start-anchored counter: `^a{2,3}`, `^\d{2,4}`.
    #[test]
    fn test_partial_anchor_start_counter() {
        let p = r"^a{2,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aaax");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xaa");

        let p = r"^\d{2,4}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "12");
        assert_matches_regex_crate(p, &re, "1234");
        assert_matches_regex_crate(p, &re, "1");
        assert_matches_regex_crate(p, &re, "12345");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a12");
    }

    /// End-anchored counter: `a{2,3}$`, `\d{2,4}$`.
    #[test]
    fn test_partial_anchor_end_counter() {
        let p = r"a{2,3}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "xaa");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "aax");

        let p = r"\d{2,4}$";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "12");
        assert_matches_regex_crate(p, &re, "1234");
        assert_matches_regex_crate(p, &re, "1");
        assert_matches_regex_crate(p, &re, "x1234");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "12x");
    }

    /// Unanchored byte-class patterns: `\d+`, `\w+`, `[a-c]+`.
    #[test]
    fn test_unanchored_byte_class() {
        let p = r"\d+";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "123");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "a1b2c3");
        assert_matches_regex_crate(p, &re, "");

        let p = r"\w+";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "hello");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "a b c");
        assert_matches_regex_crate(p, &re, "");

        let p = "[a-c]+";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abcabc");
        assert_matches_regex_crate(p, &re, "xyz");
        assert_matches_regex_crate(p, &re, "xabcy");
        assert_matches_regex_crate(p, &re, "");
    }

    /// Unanchored ByteTable with counter: `(ab|cd|ef){2,3}`.
    #[test]
    fn test_unanchored_byte_table_counter() {
        let p = "(ab|cd|ef){2,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "abcd");
        assert_matches_regex_crate(p, &re, "abcdef");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xabcdy");
        assert_matches_regex_crate(p, &re, "ab cd ef");
        assert_matches_regex_crate(p, &re, "ababab");
        assert_matches_regex_crate(p, &re, "efef");
        assert_matches_regex_crate(p, &re, "abefcd");
    }

    /// Counter pattern split across chunk boundaries.
    #[test]
    fn test_chunk_boundary_counter() {
        let p = r"a{3,5}";
        let re = build_regex_unchecked(p);
        let mut mem = MatcherMemory::default();

        // "aaa" in one chunk
        let mut m = mem.matcher(&re);
        m.chunk(b"aaa");
        assert!(m.finish(), "aaa should match a{{3,5}}");

        // "aaa" split as "a" + "aa"
        let mut m = mem.matcher(&re);
        m.chunk(b"a");
        m.chunk(b"aa");
        assert!(m.finish(), "a+aa should match a{{3,5}}");

        // "aaa" split as "aa" + "a"
        let mut m = mem.matcher(&re);
        m.chunk(b"aa");
        m.chunk(b"a");
        assert!(m.finish(), "aa+a should match a{{3,5}}");

        // "aaaaa" split as "aa" + "aaa"
        let mut m = mem.matcher(&re);
        m.chunk(b"aa");
        m.chunk(b"aaa");
        assert!(m.finish(), "aa+aaa should match a{{3,5}}");

        // "aa" — too short
        let mut m = mem.matcher(&re);
        m.chunk(b"a");
        m.chunk(b"a");
        assert!(!m.finish(), "a+a should not match a{{3,5}}");

        // Multi-byte body across chunks: (ab){2,3}
        let p = "(ab){2,3}";
        let re = build_regex_unchecked(p);

        let mut m = mem.matcher(&re);
        m.chunk(b"ab");
        m.chunk(b"ab");
        assert!(m.finish(), "ab+ab should match (ab){{2,3}}");

        let mut m = mem.matcher(&re);
        m.chunk(b"a");
        m.chunk(b"bab");
        assert!(m.finish(), "a+bab should match (ab){{2,3}}");

        let mut m = mem.matcher(&re);
        m.chunk(b"aba");
        m.chunk(b"b");
        assert!(m.finish(), "aba+b should match (ab){{2,3}}");
    }

    /// Multiline assertions across chunk boundaries.
    #[test]
    fn test_chunk_boundary_multiline() {
        let p = r"(?m:^)abc(?m:$)";
        let re = build_regex_unchecked(p);
        let mut mem = MatcherMemory::default();

        // Single chunk
        let mut m = mem.matcher(&re);
        m.chunk(b"abc");
        assert!(m.finish(), "abc should match ^abc$");

        // Split across \n boundary
        let mut m = mem.matcher(&re);
        m.chunk(b"xxx\n");
        m.chunk(b"abc\nyyy");
        assert!(m.finish(), "xxx\\nabc\\nyyy should match ^abc$ multiline");

        // abc split across chunks
        let mut m = mem.matcher(&re);
        m.chunk(b"ab");
        m.chunk(b"c");
        assert!(m.finish(), "ab+c should match ^abc$");

        // No match
        let mut m = mem.matcher(&re);
        m.chunk(b"ab");
        m.chunk(b"d");
        assert!(!m.finish(), "ab+d should not match ^abc$");
    }

    /// Unanchored `?` quantifier: `a?b`.
    #[test]
    fn test_unanchored_question_mark() {
        let p = "a?b";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "xaby");
        assert_matches_regex_crate(p, &re, "xby");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "aab");
    }

    /// Degenerate counter `{1,1}`: `a{1,1}`.
    #[test]
    fn test_counter_exact_one() {
        let p = "a{1,1}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xax");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "b");
    }

    /// Multiline + alternation + counter: `(?m:^)(a|bc){2,3}(?m:$)`.
    #[test]
    fn test_multiline_alternation_counter() {
        let p = r"(?m:^)(a|bc){2,3}(?m:$)";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "bcbc");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xxx\naa\nyyy");
        assert_matches_regex_crate(p, &re, "xxx\nabc\nyyy");
        assert_matches_regex_crate(p, &re, "xxx\na\nyyy");
    }

    /// Empty input with unanchored counter `a{0,3}`.
    #[test]
    fn test_empty_input_unanchored_counter() {
        let p = "a{0,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "");
    }

    /// Unanchored byte-class + counter: `\d{2,4}`.
    #[test]
    fn test_unanchored_byte_class_counter() {
        let p = r"\d{2,4}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "12");
        assert_matches_regex_crate(p, &re, "1234");
        assert_matches_regex_crate(p, &re, "1");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a12b");
        assert_matches_regex_crate(p, &re, "a1b2c");
        assert_matches_regex_crate(p, &re, "12345");
        assert_matches_regex_crate(p, &re, "a1234b");
    }

    /// Counter body containing assertions.
    ///
    /// The counted group includes a multiline start-of-line anchor,
    /// so the assertion must be evaluated inside the loop body on
    /// each iteration.  This exercises the interaction between the
    /// pre-consumption assertion expansion and the counter machinery.
    #[test]
    fn test_counter_body_with_assertion() {
        // Each iteration must start at the beginning of a line.
        // `(?m:^a){2,3}` — "a" at start of line, repeated 2–3 times.
        let p = r"(?m:^a){2,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a\na");
        assert_matches_regex_crate(p, &re, "a\na\na");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a\nb\na");
        assert_matches_regex_crate(p, &re, "ba\na");
        assert_matches_regex_crate(p, &re, "a\na\na\na");

        // End-of-line assertion inside counted body.
        // `(?m:a$){2,3}` — "a" at end of line, repeated 2–3 times.
        let p = r"(?m:a$){2,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a\na");
        assert_matches_regex_crate(p, &re, "a\na\na");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a\nb\na");
        assert_matches_regex_crate(p, &re, "xa\nxa");

        // Both anchors inside counted body.
        // `(?m:^a$){2,3}` — full-line "a", repeated.
        let p = r"(?m:^a$){2,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "a\na");
        assert_matches_regex_crate(p, &re, "a\na\na");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a\nab\na");
        assert_matches_regex_crate(p, &re, "x\na\na\nx");
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

    // ===================================================================
    // Batched step() coverage: tests that exercise the fused pre-mark +
    // consumption loop where multiple clist entries match the same byte
    // in a single step, interacting with counters.
    // ===================================================================

    /// Two counters whose bodies share the same consuming byte class.
    /// When the input byte matches, both counters' body_alive_gen must
    /// be stamped before any epsilon closure runs — a single fused pass
    /// must not interleave marking and closure expansion.
    #[test]
    fn test_step_fused_multi_counter_same_byte() {
        // `a{2}.*a{3}`: two counters both counting 'a', separated by `.*`
        let p = "a{2}.*a{3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aabaa");
        assert_matches_regex_crate(p, &re, "aaxaaa");
        assert_matches_regex_crate(p, &re, "aa aaa");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "");
    }

    /// Alternation creates multiple consuming states at different
    /// positions in clist that all match the same byte.  Verifies that
    /// the reverse-iteration + LIFO ordering preserves priority.
    #[test]
    fn test_step_fused_alternation_overlap() {
        // `(a|a){2,3}` — both alt branches produce consuming states
        // for 'a'; counter must still count correctly.
        let p = "(a|a){2,3}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xaax");
        assert_matches_regex_crate(p, &re, "xaaax");
    }

    /// ByteClass + exact byte consuming states coexisting in clist,
    /// both matching the same input byte.  Exercises the fused loop
    /// handling different State variants in one pass.
    #[test]
    fn test_step_fused_byteclass_and_literal() {
        // `[a-z]{2}a{2}`: [a-z] as ByteClass and 'a' as Byte both
        // match 'a', and both are wrapped in counters.
        let p = r"[a-z]{2}a{2}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "xyaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "abaa");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xyzaa");
        assert_matches_regex_crate(p, &re, " aaaa ");
    }

    /// Unanchored nested counters where the outer body's consuming
    /// states and the re-seeded start state all compete in the same
    /// step.  The re-seed (pushed first on the stack, drained last)
    /// must not interfere with the ongoing counter threads.
    #[test]
    fn test_step_fused_reseed_with_counter() {
        let p = r"(a{2}){2}";
        let re = build_regex_unchecked(p);
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "xaaaax");
        assert_matches_regex_crate(p, &re, "aa aa");
        assert_matches_regex_crate(p, &re, "aaxaa");
    }

    #[test]
    fn survey_nested_counter_bugs() {
        let cases: &[(&str, &str)] = &[
            ("(a{2}){2}", "aaa"),
            ("(a{2}){2}", "aaaa"),
            ("(a{2}){2}", "aaaaa"),
            ("(a{2}){3}", "aaaaa"),
            ("(a{2}){3}", "aaaaaa"),
            ("(a{3}){2}", "aaaa"),
            ("(a{3}){2}", "aaaaa"),
            ("(a{3}){2}", "aaaaaa"),
            ("(ab){2}", "aba"),
            ("(ab){2}", "abab"),
            ("(ab){3}", "ababa"),
            ("(ab){3}", "ababab"),
            ("(a{2,3}){2}", "aaa"),
            ("(a{2,3}){2}", "aaaa"),
            ("(a{2,3}){2}", "aaaaa"),
            ("(a{2,3}){2}", "aaaaaa"),
            ("((a{2}){2}){2}", "aaaaaaa"),
            ("((a{2}){2}){2}", "aaaaaaaa"),
            ("(a{2}){2,3}", "aaa"),
            ("(a{2}){2,3}", "aaaa"),
            ("(a{2}){2,3}", "aaaaa"),
            ("(a{2}){2,3}", "aaaaaa"),
            // multi-byte body (non-nested sub-counter)
            ("(ab){2}", "aba"),
            ("(ab){2}", "abab"),
            ("(ab){3}", "ababab"),
            ("(..){2}", "aaa"),
            ("(..){2}", "aaaa"),
            ("(abc){2}", "abcab"),
            ("(abc){2}", "abcabc"),
            // non-nested controls
            ("a{4}", "aaa"),
            ("a{4}", "aaaa"),
            ("a{2}", "aa"),
            // regression cases (suppression must NOT apply)
            ("^(a|a?){2,3}$", "aaa"),
            ("^(a{0,2}){2,3}$", "aaaaa"),
            ("^((a|bc){1,2}){2,3}$", "aaaaaa"),
            // variable-inner-length cases
            ("^(a{1,2}){2}$", "aa"),
            ("^(a{1,2}){2}$", "aaa"),
            ("^(a{1,2}){2}$", "aaaa"),
        ];
        let mut failures = Vec::new();
        for &(pat, input) in cases {
            let re = build_regex_unchecked(pat);
            let regex_re = regex::Regex::new(pat).unwrap();
            let mut mem = MatcherMemory::default();
            let mut m = mem.matcher(&re);
            m.chunk(input.as_bytes());
            let ours = m.finish();
            let theirs = regex_re.is_match(input);
            if ours != theirs {
                failures.push((pat, input, ours, theirs));
            }
        }
        if !failures.is_empty() {
            let mut msg = String::from("Mismatches found:\n");
            for (pat, input, ours, theirs) in &failures {
                msg.push_str(&format!(
                    "  pattern={pat:25} input={input:12} ours={ours:<5} regex={theirs}\n"
                ));
            }
            panic!("{msg}");
        }
    }
}
