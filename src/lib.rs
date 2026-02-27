//! Thompson NFA with multi-instance counting constraints.
//!
//! Based on Russ Cox's article <https://swtch.com/~rsc/regexp/regexp1.html>
//! (Thompson NFA construction and simulation) with additional support for
//! bounded repetitions (`{min,max}`) via the counting-FA model described in
//! Becchi & Crowley, "Extending Finite Automata to Efficiently Match
//! Perl-Compatible Regular Expressions" (CoNEXT 2008)
//! <https://www.arl.wustl.edu/~pcrowley/a25-becchi.pdf>.
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
//! A repetition `body{min,max}` is lowered to a **two-copy** NFA:
//!
//! ```text
//! body₁ ── CounterInstance(c) ── body₂ ── CounterIncrement(c, min, max)
//!                                  ^                    │
//!                                  └── continue ────────┘
//!                                            break ──> (next)
//! ```
//!
//! `body₁` is the mandatory initial match.  `CounterInstance` allocates a
//! new counter instance (or creates the counter from scratch).  The
//! `body₂` + `CounterIncrement` loop runs zero or more additional times.
//! `CounterIncrement` increments all active instances; when the oldest
//! instance's value falls in `[min, max]`, the break path is followed.
//! When it reaches `max`, the oldest instance is de-allocated.
//!
//! Counters use a **differential representation** (from the Becchi paper)
//! so that increment and condition-evaluation require O(1) work regardless
//! of how many instances are active.
//!
//! ### Why two copies are required
//!
//! It may seem wasteful to duplicate the body NFA (O(2·b) states per
//! repetition, O(b·2^d) for d levels of nesting).  A tempting
//! optimisation is to share a single body copy in a loop:
//!
//! ```text
//! CounterInstance(c) ── body ── CounterIncrement(c, min, max)
//!                         ^                │
//!                         └── continue ────┘
//!                                  break ──> (next)
//! ```
//!
//! This **does not work** when multiple counter instances overlap — that
//! is, when the NFA can enter the repetition from multiple match threads
//! simultaneously (e.g. `^.*a.{3}bc$` on "axaybzbc", where `.*`
//! produces two threads entering `.{3}` via different `a` positions).
//!
//! The root cause is the Thompson NFA's state deduplication.  During
//! `step()`, each consuming state (Byte, ByteClass) appears at most once
//! in the current state list (`clist`).  With a shared body, both the
//! *loop re-entry* from `CounterIncrement` (for an existing counter
//! instance) and the *initial entry* from `CounterInstance` (for a newly
//! pushed instance) feed into the **same** body state.  NFA
//! deduplication merges them into a single entry, so the body is only
//! processed once per step.
//!
//! The problem manifests during a step where **both** `CounterInstance`
//! (push) and `CounterIncrement` (increment) fire for the same counter:
//!
//!  1. Consuming state A matches byte `b`, triggering
//!     `addstate(CounterInstance)` which **pushes** a new counter
//!     instance (delta=0) and follows `out` to the shared `body`.
//!     The body is a consuming state, so it is recorded in `nlist`.
//!
//!  2. Consuming state B (the shared body, from an older instance's loop)
//!     also matches byte `b`, triggering `addstate(CounterIncrement)`
//!     which **increments** all counter instances — including the one
//!     just pushed in (1) that has not yet consumed a body byte.
//!
//! The new instance receives an "unearned" increment because the
//! `CounterIncrement` from the *old* instance's body match blindly
//! advances all instances.  This causes an **off-by-one** in the new
//! instance's counter value: it reaches `max` one step too early,
//! breaking the `break` condition and producing false negatives.
//!
//! Concrete example — `^.*a.{3}bc$` on "axaybzbc":
//!
//! - Position 0: `a` matches → `CounterInstance` pushes instance A,
//!   body enters `nlist`.
//! - Position 1: `x` consumed by body → `CounterIncrement` (A.value=1).
//! - Position 2: `a` matches again → `CounterInstance` pushes instance B
//!   (B.delta=0).  **In the same step**, the shared body (from A's loop)
//!   matches `a` → `CounterIncrement` fires, incrementing **both** A
//!   (value=2) and B (delta=0→1).  But B has not consumed a body byte
//!   yet — this increment is wrong.
//! - The off-by-one propagates: B reaches max=3 at position 4 instead
//!   of position 5, causing the break to `Byte(b)` one position too
//!   early.  The trailing `bc` then fails to match "zbc".
//!
//! With **two copies**, the problem disappears: `body₁` (state X) and
//! `body₂` (state Y) are distinct NFA states, so they coexist in `clist`
//! independently.  `body₁`'s match triggers `CounterInstance` (push) then
//! `CounterIncrement` (increment), while `body₂`'s match triggers only
//! `CounterIncrement` — but the `incremented` flag prevents double-
//! processing in the same epsilon closure.  The push and increment are
//! thus properly synchronized: each body match contributes exactly one
//! increment to the instances it belongs to.
//!
//! Attempts to fix the shared-body approach (e.g. a `pushed_this_pass`
//! flag to suppress the spurious delta increment) break the Becchi
//! differential representation's invariant (`value == delta` when only
//! one instance remains), causing assertion failures in `pop()`.
//!
//! The two-copy design is therefore **necessary for correctness** with
//! the Becchi multi-instance counter representation, and is used for all
//! bounded repetitions.
//!
//! ## Nested repetitions
//!
//! When a repetition body itself contains a repetition, the body copy in
//! the outer counting loop gets **remapped** counter indices so that each
//! copy of the inner repetition operates its own independent counter.
//!
//! A subtle interaction arises when the outer counter is exhausted
//! (`None`) by one NFA path in the same simulation step, and a second
//! path (arriving via an inner `CounterInstance`) legitimately needs to
//! restart the outer counter.  We distinguish this from the
//! epsilon-body case (where no `CounterInstance` fired and the `None`
//! counter should stay dead) using a **generation counter** (`ci_gen`)
//! that increments every time any `CounterInstance` fires.  Re-entry at
//! a `CounterIncrement` with a `None` counter is allowed only if `ci_gen`
//! has advanced since the state's first visit in the current step.

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
// NFA states
// ---------------------------------------------------------------------------

/// A single NFA state.
///
/// Epsilon states (`Split`, `CounterInstance`, `CounterIncrement`,
/// `AssertStart`, `AssertEnd`) are followed during
/// [`Matcher::addstate`].  Byte-consuming states (`Byte`, `ByteClass`)
/// are stepped over in [`Matcher::step`].
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

    /// Zero-width assertion: matches only at the start of input.
    /// Followed in `addstate` only when `at_start` is true.
    AssertStart { out: StateIdx },

    /// Zero-width assertion: matches only at the end of input.
    /// Followed in `addstate` only when `at_end` is true.
    AssertEnd { out: StateIdx },

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
            | State::AssertStart { out }
            | State::AssertEnd { out } => *out,
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
            | State::AssertStart { out }
            | State::AssertEnd { out } => *out = next,
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
    CounterInstance {
        counter: CounterIdx,
    },
    CounterIncrement {
        counter: CounterIdx,
        min: usize,
        max: usize,
    },
    AssertStart,
    AssertEnd,
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
    /// One slot per counter variable allocated during compilation.
    counters: Box<[usize]>,
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
    /// - The `counters` boxed slice.
    pub fn memory_size(&self) -> usize {
        let inline = std::mem::size_of::<Self>();
        let states_alloc = self.states.len() * std::mem::size_of::<State>();
        let classes_alloc = self.classes.len() * std::mem::size_of::<ByteClass>();
        let counters_alloc = self.counters.len() * std::mem::size_of::<usize>();
        let byte_tables_alloc = self.byte_tables.len() * std::mem::size_of::<ByteMap>();
        inline + states_alloc + classes_alloc + counters_alloc + byte_tables_alloc
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
            State::AssertStart { out } => {
                stack.push(out);
                writeln!(buffer, "\t{} -> {} [label=\"^\"];", idx, out).unwrap();
            }
            State::AssertEnd { out } => {
                stack.push(out);
                writeln!(buffer, "\t{} -> {} [label=\"$\"];", idx, out).unwrap();
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

#[derive(Debug, Default)]
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
    /// For bounded repetitions, the body is emitted twice: once before
    /// `CounterInstance` (the mandatory first match) and once inside the
    /// counting loop (before `CounterIncrement`).  The second copy has all
    /// inner counter indices **remapped** so that nested repetitions get
    /// independent counters.
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
            HirKind::Look(hir::Look::Start) => {
                self.postfix.push(RegexHirNode::AssertStart);
                Ok(())
            }
            HirKind::Look(hir::Look::End) => {
                self.postfix.push(RegexHirNode::AssertEnd);
                Ok(())
            }
            HirKind::Look(look) => Err(Error::UnsupportedLook(*look)),
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

                    // Two-copy lowering: body₁ · CounterInstance · body₂ · CounterIncrement
                    //
                    // The body is emitted twice (body₁ and body₂).  This is
                    // necessary for correct multi-instance counter tracking;
                    // sharing a single body copy causes an off-by-one when
                    // overlapping instances exist.  See the module-level
                    // "Why two copies are required" section for details.
                    let start = self.postfix.len();
                    self.hir2postfix(&rep.sub)?;
                    let end = self.postfix.len();

                    self.postfix.push(RegexHirNode::CounterInstance { counter });

                    // Copy the body HIR for the counting loop, remapping
                    // counter indices for nested repetitions.
                    self.emit_remapped_body(start, end);

                    self.postfix
                        .push(RegexHirNode::CounterIncrement { counter, min, max });
                    self.postfix.push(RegexHirNode::Catenate);
                } else {
                    // {0,max}: lower to (body{1,max})? — the `?` wrapping
                    // provides the zero-match path without polluting the
                    // body with an epsilon alternative.
                    let counter = self.next_counter();

                    let start = self.postfix.len();
                    self.hir2postfix(&rep.sub)?;
                    let end = self.postfix.len();

                    self.postfix.push(RegexHirNode::CounterInstance { counter });

                    self.emit_remapped_body(start, end);

                    self.postfix.push(RegexHirNode::CounterIncrement {
                        counter,
                        min: 1,
                        max,
                    });
                    self.postfix.push(RegexHirNode::Catenate);

                    // Wrap in `?` to provide the zero-match path.
                    self.postfix.push(RegexHirNode::RepeatZeroOne);
                }
                Ok(())
            }
        }
    }

    /// Copy the body HIR slice `postfix[start..end]` into postfix,
    /// remapping any counter indices so that each copy of a nested
    /// repetition gets its own independent counter.
    fn emit_remapped_body(&mut self, start: usize, end: usize) {
        let body = self.postfix[start..end].to_vec();
        let mut counter_map = std::collections::HashMap::new();
        for hir_node in body {
            let remapped = match hir_node {
                RegexHirNode::CounterInstance { counter: c } => {
                    let new_c = *counter_map.entry(c).or_insert_with(|| self.next_counter());
                    RegexHirNode::CounterInstance { counter: new_c }
                }
                RegexHirNode::CounterIncrement {
                    counter: c,
                    min: mn,
                    max: mx,
                } => {
                    let new_c = *counter_map.entry(c).or_insert_with(|| self.next_counter());
                    RegexHirNode::CounterIncrement {
                        counter: new_c,
                        min: mn,
                        max: mx,
                    }
                }
                other => other,
            };
            self.postfix.push(remapped);
        }
    }

    // -- Low-level NFA construction helpers ----------------------------------

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
                | State::AssertStart { out }
                | State::AssertEnd { out } => {
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
            RegexHirNode::CounterInstance { counter } => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::CounterInstance {
                    counter,
                    out: StateIdx::NONE,
                });
                self.patch(e.out, s);
                Fragment::new(e.start, s)
            }
            RegexHirNode::CounterIncrement { counter, min, max } => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::CounterIncrement {
                    out: e.start,
                    out1: StateIdx::NONE,
                    min,
                    max,
                    counter,
                });
                self.patch(e.out, s);
                Fragment::new(s, s)
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
            RegexHirNode::AssertStart => {
                let idx = self.state(State::AssertStart {
                    out: StateIdx::NONE,
                });
                Fragment::new(idx, idx)
            }
            RegexHirNode::AssertEnd => {
                let idx = self.state(State::AssertEnd {
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
            counters: self.counters.to_vec().into_boxed_slice(),
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
}

// ---------------------------------------------------------------------------
// Multi-instance counter (Becchi differential representation)
// ---------------------------------------------------------------------------

/// A multi-instance counter using the differential representation from
/// the Becchi paper.
///
/// Multiple overlapping occurrences of a counted sub-expression can be
/// tracked simultaneously.  All instances are incremented in parallel,
/// and the oldest instance is always the one with the highest `value`.
///
/// The *deltas* between consecutive instances are stored as a linked
/// list in a shared [`DeltaPool`] (held in [`MatcherMemory`]).  The
/// counter only stores `head`/`tail` indices into that pool, so it
/// remains a small `Copy`-able struct with no per-counter heap
/// allocation.  The pool retains its capacity across `matcher()` calls.
///
/// # Fields
///
/// - `value` — the oldest (highest) instance's value.
/// - `delta` — how much the newest instance has accumulated since the
///   last [`push`](Self::push).
/// - `head` / `tail` — indices into `DeltaPool` forming a singly-linked
///   FIFO of deltas for intermediate instances.  Both are [`SENTINEL`]
///   when the list is empty.
/// - `incremented` — set by [`incr`](Self::incr), cleared at each
///   simulation step; used to prevent a `CounterIncrement` state from
///   being processed twice in the same epsilon closure.
#[derive(Clone, Copy, Debug)]
struct Counter {
    incremented: bool,
    value: usize,
    delta: usize,
    head: usize,
    tail: usize,
}

impl Default for Counter {
    fn default() -> Self {
        Self {
            incremented: false,
            value: 0,
            delta: 0,
            head: SENTINEL,
            tail: SENTINEL,
        }
    }
}

impl Counter {
    /// Push (allocate) a new instance.  The current `delta` is saved
    /// as a new node in `pool` and reset to zero.
    fn push(&mut self, pool: &mut DeltaPool) {
        assert!(self.delta > 0);
        let node = pool.alloc(self.delta);
        if self.tail != SENTINEL {
            pool.next[self.tail] = node;
        } else {
            self.head = node;
        }
        self.tail = node;
        self.delta = 0;
    }

    /// Increment all instances by 1.
    fn incr(&mut self) {
        self.value += 1;
        self.delta += 1;
        self.incremented = true;
    }

    /// De-allocate the oldest instance.  Returns `true` if the counter
    /// has no instances left (should be set to `None`).
    fn pop(&mut self, pool: &mut DeltaPool) -> bool {
        assert!(self.value > 0);
        if self.head != SENTINEL {
            let old_head = self.head;
            let delta = pool.values[old_head];
            self.head = pool.next[old_head];
            if self.head == SENTINEL {
                self.tail = SENTINEL;
            }
            pool.free_node(old_head);
            assert!(delta < self.value);
            self.value -= delta;
            false
        } else {
            assert_eq!(self.value, self.delta);
            true
        }
    }

    /// Returns `true` when the delta linked list is empty (exactly one
    /// active instance).
    fn is_single(&self) -> bool {
        self.head == SENTINEL
    }

    /// Check whether any active instance's value falls in `[min, max]`.
    /// Walks from oldest (highest value) to newest.
    fn any_instance_in_range(&self, pool: &DeltaPool, min: usize, max: usize) -> bool {
        let mut val = self.value;
        if val >= min && val <= max {
            return true;
        }
        let mut node = self.head;
        while node != SENTINEL {
            val -= pool.values[node];
            if val >= min && val <= max {
                return true;
            }
            node = pool.next[node];
        }
        false
    }
}

/// Index into the counter variable array ([`Regex::counters`]).
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

/// `counters[counter_idx]` — typed access to the counter variable array.
impl Index<CounterIdx> for [Option<Counter>] {
    type Output = Option<Counter>;

    #[inline]
    fn index(&self, idx: CounterIdx) -> &Option<Counter> {
        &self[idx.idx()]
    }
}

impl IndexMut<CounterIdx> for [Option<Counter>] {
    #[inline]
    fn index_mut(&mut self, idx: CounterIdx) -> &mut Option<Counter> {
        &mut self[idx.idx()]
    }
}

/// Sentinel value meaning "no node" (end of linked list / empty).
const SENTINEL: usize = usize::MAX;

/// Arena-backed pool of linked-list nodes for counter deltas.
///
/// All counters share a single `DeltaPool`.  Each counter owns a
/// linked list (head/tail indices stored in [`Counter`]) whose nodes
/// live in this pool.  Freed nodes are pushed onto a free-list stack
/// and recycled by subsequent allocations.
///
/// Between matches, [`reset`](Self::reset) clears all three backing
/// vecs (retaining heap capacity), so the pool's memory is reused
/// across [`MatcherMemory::matcher`] calls without per-counter loops.
#[derive(Clone, Debug, Default)]
struct DeltaPool {
    /// Node payload (delta value).
    values: Vec<usize>,
    /// `next[i]` = index of the successor node, or [`SENTINEL`].
    next: Vec<usize>,
    /// Stack of freed node indices available for reuse.
    free: Vec<usize>,
}

impl DeltaPool {
    /// Allocate a node with the given value.  Reuses a freed slot if
    /// available, otherwise appends to the end of the arena.
    fn alloc(&mut self, val: usize) -> usize {
        if let Some(idx) = self.free.pop() {
            self.values[idx] = val;
            self.next[idx] = SENTINEL;
            idx
        } else {
            let idx = self.values.len();
            self.values.push(val);
            self.next.push(SENTINEL);
            idx
        }
    }

    /// Return a node to the free list for reuse.
    fn free_node(&mut self, idx: usize) {
        self.free.push(idx);
    }

    /// Clear all nodes (retaining heap capacity).
    fn reset(&mut self) {
        self.values.clear();
        self.next.clear();
        self.free.clear();
    }
}

// ---------------------------------------------------------------------------
// Matcher (NFA simulation)
// ---------------------------------------------------------------------------

/// Reusable memory for [`Matcher`].  Create once, call
/// [`matcher`](Self::matcher) for each regex to match.
#[derive(Debug, Default)]
pub struct MatcherMemory {
    /// Per-state: the `listid` when the state was last added.  Used for
    /// O(1) deduplication in `addstate`.
    lastlist: Vec<usize>,
    /// One slot per counter variable.
    counters: Vec<Option<Counter>>,
    /// Shared arena for counter delta linked lists.  Retains heap
    /// capacity across `matcher()` calls.
    delta_pool: DeltaPool,
    /// Current and next state lists (swapped each step).
    clist: Vec<StateIdx>,
    nlist: Vec<StateIdx>,
    /// Per-state snapshot of `ci_gen` at first visit in the current step.
    /// See the module-level doc comment for the motivation.
    ci_gen_at_visit: Vec<usize>,
}

impl MatcherMemory {
    pub fn matcher<'a>(&'a mut self, regex: &'a Regex) -> Matcher<'a> {
        self.lastlist.clear();
        self.lastlist.resize(regex.states.len(), usize::MAX);
        self.counters.clear();
        self.counters.resize(regex.counters.len(), None);
        self.delta_pool.reset();
        self.clist.clear();
        self.nlist.clear();
        self.ci_gen_at_visit.clear();
        self.ci_gen_at_visit.resize(regex.states.len(), 0);

        let mut m = Matcher {
            counters: &mut self.counters,
            delta_pool: &mut self.delta_pool,
            states: &regex.states,
            classes: &regex.classes,
            byte_tables: &regex.byte_tables,
            lastlist: &mut self.lastlist,
            listid: 0,
            clist: &mut self.clist,
            nlist: &mut self.nlist,
            ci_gen: 0,
            ci_gen_at_visit: &mut self.ci_gen_at_visit,
            start: regex.start,
            at_start: true,
            at_end: false,
            ever_matched: false,
        };

        m.startlist(m.start);
        // Check if the initial epsilon closure already reached Match
        // (e.g. for patterns like `^$`, `^`, `a?`, etc.).
        m.ever_matched = m.clist_has_match();
        m
    }
}

/// Runs a Thompson NFA simulation with counting-constraint support.
#[derive(Debug)]
pub struct Matcher<'a> {
    counters: &'a mut [Option<Counter>],
    /// Shared arena for counter delta linked lists.
    delta_pool: &'a mut DeltaPool,
    states: &'a [State],
    /// Byte-class lookup tables referenced by [`State::ByteClass::class`].
    classes: &'a [ByteClass],
    /// Byte dispatch tables referenced by [`State::ByteTable::table`].
    byte_tables: &'a [ByteMap],
    /// Per-state deduplication stamp (compared against `listid`).
    lastlist: &'a mut [usize],
    /// Monotonically increasing step ID.
    listid: usize,
    /// Current active state list.
    clist: &'a mut Vec<StateIdx>,
    /// Next active state list (built during a step).
    nlist: &'a mut Vec<StateIdx>,
    /// Monotonically increasing generation counter; incremented every
    /// time [`addcounter`](Self::addcounter) is called.  Used together
    /// with `ci_gen_at_visit` to detect whether a `CounterInstance`
    /// fired between the first visit and a re-entry at a
    /// `CounterIncrement` state.
    ci_gen: usize,
    /// Per-state snapshot of `ci_gen` recorded on first visit.
    ci_gen_at_visit: &'a mut [usize],

    /// The NFA start state index, used for re-seeding at each step
    /// (unanchored matching).
    start: StateIdx,
    /// `true` until the first [`step`](Self::step) call.  Controls
    /// whether `AssertStart` states are followed in `addstate`.
    at_start: bool,
    /// Set to `true` by [`finish`](Self::finish); controls whether
    /// `AssertEnd` states are followed in `addstate`.
    at_end: bool,
    /// Tracks whether a `Match` state was ever reached in `clist`
    /// during the simulation (before `finish`).
    ever_matched: bool,
}

impl<'a> Matcher<'a> {
    /// Compute the initial state list by following all epsilon transitions
    /// from `start`.
    #[inline]
    fn startlist(&mut self, start: StateIdx) {
        self.addstate(start);
        std::mem::swap(self.clist, self.nlist);
        self.listid += 1;
    }

    /// Allocate (or push) a counter instance.  If the counter is `None`
    /// (not yet created or previously exhausted), a fresh default counter
    /// is created.  Otherwise a new instance is pushed onto the existing
    /// counter.
    ///
    /// Increments `ci_gen` so that downstream `CounterIncrement` re-entry
    /// checks can detect that a new instance was allocated.
    fn addcounter(&mut self, idx: CounterIdx) {
        if let Some(counter) = self.counters[idx].as_mut() {
            counter.push(self.delta_pool);
        } else {
            self.counters[idx] = Some(Counter::default());
        }
        self.ci_gen += 1;
    }

    /// Increment all instances of counter `idx`.
    fn inccounter(&mut self, idx: CounterIdx) {
        self.counters[idx].as_mut().unwrap().incr();
    }

    /// De-allocate the oldest instance of counter `idx`.  If no instances
    /// remain, the counter is set to `None`.
    fn delcounter(&mut self, idx: CounterIdx) {
        if self.counters[idx].as_mut().unwrap().pop(self.delta_pool) {
            self.counters[idx] = None;
        }
    }

    /// Returns `true` if a `CounterIncrement` for `counter` should be
    /// allowed to proceed.
    ///
    /// The counter is processable when:
    /// - It is `Some` and has not yet been incremented in this epsilon
    ///   closure pass (`!incremented`), OR
    /// - It is `None` (exhausted) and a `CounterInstance` fired since
    ///   this state's first visit in the current step (`ci_gen` advanced).
    fn counter_is_processable(&self, counter: CounterIdx, state_idx: StateIdx) -> bool {
        self.counters[counter]
            .as_ref()
            .map_or(self.ci_gen > self.ci_gen_at_visit[state_idx.idx()], |c| {
                !c.incremented
            })
    }

    /// Recursively follow epsilon transitions from state `idx`, adding
    /// all reachable states to `nlist`.
    ///
    /// This is the heart of the Thompson NFA simulation.  The
    /// `lastlist`/`listid` mechanism provides O(1) deduplication so each
    /// state is visited at most once per step (with a controlled exception
    /// for `CounterIncrement` re-entry — see below).
    ///
    /// ## CounterIncrement re-entry
    ///
    /// Normally each state is visited at most once.  However, a
    /// `CounterIncrement` state may need to be re-visited when a new
    /// counter instance arrives via a different path in the same epsilon
    /// closure.  Re-entry is allowed when
    /// [`counter_is_processable`](Self::counter_is_processable) returns
    /// `true`.
    ///
    /// ## Epsilon-body detection
    ///
    /// If the repetition body can match the empty string (e.g. `(a?)`),
    /// the continue path's epsilon closure will loop back to this same
    /// `CounterIncrement` state.  We detect this by temporarily clearing
    /// our `lastlist` mark before following the continue path: if the
    /// epsilon closure re-marks us, the body is epsilon-matchable.  In
    /// that case, the break condition is relaxed: since epsilon matches
    /// can advance the counter for free, any value in `[min, max]` is
    /// reachable and we always allow the break.
    #[inline]
    fn addstate(&mut self, idx: StateIdx) {
        let i = idx.idx();
        if self.lastlist[i] == self.listid {
            let should_reenter = match &self.states[idx] {
                State::CounterIncrement { counter, .. } => {
                    self.counter_is_processable(*counter, idx)
                }
                _ => false,
            };
            if !should_reenter {
                return;
            }
        }

        // Record ci_gen only on first visit so re-entry compares against
        // the original snapshot.
        if self.lastlist[i] != self.listid {
            self.ci_gen_at_visit[i] = self.ci_gen;
        }
        self.lastlist[i] = self.listid;

        match &self.states[idx] {
            State::Split { out, out1 } => {
                let (out, out1) = (*out, *out1);
                self.addstate(out);
                self.addstate(out1);
            }

            State::AssertStart { out } => {
                if self.at_start {
                    let out = *out;
                    self.addstate(out);
                }
            }

            State::AssertEnd { out } => {
                if self.at_end {
                    let out = *out;
                    self.addstate(out);
                }
            }

            State::CounterInstance { out, counter } => {
                let (out, counter) = (*out, *counter);
                self.addcounter(counter);
                self.addstate(out);
            }

            State::CounterIncrement {
                out,
                out1,
                counter,
                min,
                max,
            } if self.counter_is_processable(*counter, idx) => {
                let (out, out1, counter, min, max) = (*out, *out1, *counter, *min, *max);
                // Re-create the counter if it was exhausted (None).  This
                // only happens when ci_gen advanced (i.e. a CounterInstance
                // for an inner counter fired since our first visit).
                if self.counters[counter].is_none() {
                    self.counters[counter] = Some(Counter::default());
                }

                self.inccounter(counter);
                let value = self.counters[counter].as_ref().unwrap().value;
                debug_assert!(value > 0 && value <= max);
                let is_single = self.counters[counter].as_ref().unwrap().is_single();

                // -- Continue path --
                // Follow the body again unless the single remaining
                // instance has reached max.
                let should_continue = value != max || !is_single;
                let mut is_epsilon_body = false;
                if should_continue {
                    // Temporarily clear our mark to detect epsilon-body
                    // loops.  The `incremented` flag (set by inccounter)
                    // prevents the recursive call from re-processing this
                    // state — it just sets lastlist[idx] and falls through.
                    self.lastlist[i] = self.listid.wrapping_sub(1);
                    self.addstate(out);
                    is_epsilon_body = self.lastlist[i] == self.listid;
                    self.lastlist[i] = self.listid;
                }

                // -- Break condition --
                // For epsilon bodies the counter can freely advance to any
                // value in [value, max] via empty matches, so the break
                // condition is always satisfiable when min <= max (which
                // is an invariant).  For normal bodies, check all current
                // counter instances against [min, max].
                let stop = if is_epsilon_body {
                    min <= max
                } else {
                    self.counters[counter]
                        .as_ref()
                        .unwrap()
                        .any_instance_in_range(self.delta_pool, min, max)
                };

                // De-allocate the oldest instance if it reached max.
                if value == max {
                    self.delcounter(counter);
                }

                // Follow the break path if any instance satisfies the
                // counting constraint.
                if stop {
                    self.addstate(out1);
                }
            }

            // Byte, ByteClass, Match, or CounterIncrement whose guard
            // failed — just record the state for step() to inspect.
            _ => {}
        }

        self.nlist.push(idx);
    }

    /// Advance the simulation by one input byte.
    ///
    /// For each state in `clist`, if the byte matches (`Byte` or
    /// `ByteClass`), follow the `out` pointer through `addstate` to build
    /// the next `nlist`.
    ///
    /// After processing `clist`, the start state is re-seeded into `nlist`
    /// to support unanchored matching (when there is no leading `^`, the
    /// start state is a consuming state that will try matching from every
    /// position; when `^` is present, the start state is `AssertStart`
    /// and `at_start=false` prevents it from being followed).
    pub fn step(&mut self, b: u8) {
        // `at_start` is cleared before processing so that `^` cannot
        // match at any position other than the very beginning.
        self.at_start = false;

        self.nlist.clear();
        let clist = std::mem::take(self.clist);

        // Reset the per-step `incremented` flag on every active counter
        // so that CounterIncrement states can be processed in the new
        // epsilon closure.
        for counter in self.counters.iter_mut().filter_map(|c| c.as_mut()) {
            counter.incremented = false;
        }

        for &idx in &clist {
            match self.states[idx] {
                State::Byte { byte: b2, out } if b == b2 => self.addstate(out),
                State::ByteClass { class, out } if self.classes[class][b] => self.addstate(out),
                State::ByteTable { table } => {
                    let target = self.byte_tables[table][b];
                    if target != StateIdx::NONE {
                        self.addstate(target);
                    }
                }
                _ => {}
            }
        }

        // Re-seed the start state so the pattern can match starting at
        // the next position.  For `^`-anchored patterns this is harmless:
        // the start state is `AssertStart`, and `at_start` is false, so
        // `addstate` will record it but not follow `out`.
        let start = self.start;
        self.addstate(start);

        *self.clist = std::mem::replace(self.nlist, clist);
        self.listid += 1;

        // Track whether we ever reach Match during stepping.
        if !self.ever_matched {
            self.ever_matched = self.clist_has_match();
        }
    }

    /// Feed an entire byte slice through the matcher, one byte at a time.
    pub fn chunk(&mut self, input: &[u8]) {
        for &b in input {
            self.step(b);
        }
    }

    /// Check whether `clist` currently contains a `Match` state.
    fn clist_has_match(&self) -> bool {
        self.clist
            .iter()
            .any(|&idx| matches!(self.states[idx], State::Match))
    }

    /// Check whether the matcher has reached an accepting state so far.
    ///
    /// This returns `true` if a `Match` state is currently in `clist`
    /// *or* was reached during any previous step.  It does **not**
    /// signal end-of-input, so `$` assertions are not evaluated — use
    /// [`finish`](Self::finish) for that.
    pub fn ismatch(&self) -> bool {
        self.ever_matched || self.clist_has_match()
    }

    /// Signal end-of-input and return the final match result.
    ///
    /// This allows `$` (AssertEnd) states to fire: `at_end` is set to
    /// `true` and any `AssertEnd` states currently in `clist` have their
    /// `out` pointers followed through `addstate`.
    ///
    /// Only `AssertEnd` states are re-expanded — other states (counters,
    /// splits, consuming states) were already fully epsilon-expanded when
    /// they entered `clist` and must not be re-processed.
    ///
    /// Consumes the matcher, since no further input can be fed after
    /// end-of-input has been signalled.
    pub fn finish(mut self) -> bool {
        if self.ever_matched {
            return true;
        }

        self.at_end = true;

        // Walk clist by index to avoid holding a borrow across addstate().
        // addstate() only writes to nlist, never clist, so this is safe.
        let clist_len = self.clist.len();
        if clist_len > 0 {
            // Bump listid so addstate can visit new states.
            self.listid += 1;

            // Reset counter incremented flags for the new epsilon closure.
            for counter in self.counters.iter_mut().filter_map(|c| c.as_mut()) {
                counter.incremented = false;
            }

            // Follow the AssertEnd targets, building new states into nlist.
            self.nlist.clear();
            for i in 0..clist_len {
                let idx = self.clist[i];
                if let State::AssertEnd { out } = self.states[idx] {
                    self.addstate(out);
                }
            }

            // Append the newly discovered states to clist.
            self.clist.append(self.nlist);
        }

        // ever_matched is known false here (early return above).
        self.clist_has_match()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // DeltaPool unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_delta_pool_alloc_sequential() {
        let mut pool = DeltaPool::default();
        let a = pool.alloc(10);
        let b = pool.alloc(20);
        let c = pool.alloc(30);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(pool.values[a], 10);
        assert_eq!(pool.values[b], 20);
        assert_eq!(pool.values[c], 30);
        assert_eq!(pool.next[a], SENTINEL);
        assert_eq!(pool.next[b], SENTINEL);
        assert_eq!(pool.next[c], SENTINEL);
    }

    #[test]
    fn test_delta_pool_free_and_reuse() {
        let mut pool = DeltaPool::default();
        let a = pool.alloc(10);
        let b = pool.alloc(20);
        pool.free_node(a);
        // Next alloc should reuse slot `a`.
        let c = pool.alloc(99);
        assert_eq!(c, a);
        assert_eq!(pool.values[c], 99);
        assert_eq!(pool.next[c], SENTINEL);
        // `b` is unaffected.
        assert_eq!(pool.values[b], 20);
    }

    #[test]
    fn test_delta_pool_free_lifo_order() {
        let mut pool = DeltaPool::default();
        let a = pool.alloc(1);
        let b = pool.alloc(2);
        let c = pool.alloc(3);
        pool.free_node(a);
        pool.free_node(b);
        pool.free_node(c);
        // Free stack is LIFO: c, b, a.
        assert_eq!(pool.alloc(10), c);
        assert_eq!(pool.alloc(20), b);
        assert_eq!(pool.alloc(30), a);
    }

    #[test]
    fn test_delta_pool_reset_retains_capacity() {
        let mut pool = DeltaPool::default();
        for i in 0..100 {
            pool.alloc(i);
        }
        let cap_values = pool.values.capacity();
        let cap_next = pool.next.capacity();
        pool.reset();
        assert!(pool.values.is_empty());
        assert!(pool.next.is_empty());
        assert!(pool.free.is_empty());
        assert_eq!(pool.values.capacity(), cap_values);
        assert_eq!(pool.next.capacity(), cap_next);
    }

    #[test]
    fn test_delta_pool_alloc_after_reset() {
        let mut pool = DeltaPool::default();
        pool.alloc(42);
        pool.alloc(43);
        pool.reset();
        // After reset, free list is empty, vecs are empty — fresh allocs
        // start from index 0 again.
        let a = pool.alloc(99);
        assert_eq!(a, 0);
        assert_eq!(pool.values[a], 99);
    }

    #[test]
    fn test_delta_pool_mixed_alloc_free_alloc() {
        let mut pool = DeltaPool::default();
        let a = pool.alloc(1);
        let b = pool.alloc(2);
        let c = pool.alloc(3);
        // Free middle node.
        pool.free_node(b);
        // Alloc reuses `b`.
        let d = pool.alloc(4);
        assert_eq!(d, b);
        assert_eq!(pool.values[d], 4);
        // Alloc new — extends the arena.
        let e = pool.alloc(5);
        assert_eq!(e, 3);
        // Original a, c untouched.
        assert_eq!(pool.values[a], 1);
        assert_eq!(pool.values[c], 3);
    }

    // -----------------------------------------------------------------------
    // Counter + DeltaPool integration tests
    // -----------------------------------------------------------------------

    /// Helper: collect all instance values from a counter (oldest to
    /// newest) by walking the linked list.
    fn collect_instance_values(c: &Counter, pool: &DeltaPool) -> Vec<usize> {
        let mut vals = vec![c.value];
        let mut val = c.value;
        let mut node = c.head;
        while node != SENTINEL {
            val -= pool.values[node];
            vals.push(val);
            node = pool.next[node];
        }
        vals
    }

    #[test]
    fn test_counter_default_is_single() {
        let c = Counter::default();
        assert!(c.is_single());
        assert_eq!(c.value, 0);
        assert_eq!(c.delta, 0);
        assert!(!c.incremented);
        assert_eq!(c.head, SENTINEL);
        assert_eq!(c.tail, SENTINEL);
    }

    #[test]
    fn test_counter_incr_single_instance() {
        let mut c = Counter::default();
        c.incr();
        assert_eq!(c.value, 1);
        assert_eq!(c.delta, 1);
        assert!(c.incremented);
        assert!(c.is_single());
        c.incr();
        assert_eq!(c.value, 2);
        assert_eq!(c.delta, 2);
    }

    #[test]
    fn test_counter_push_creates_linked_list() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        // Instance 1: increment 3 times.
        c.incr();
        c.incr();
        c.incr();
        assert_eq!(c.value, 3);
        assert_eq!(c.delta, 3);
        // Push: saves delta=3 to pool, resets delta to 0.
        c.push(&mut pool);
        assert_eq!(c.value, 3);
        assert_eq!(c.delta, 0);
        assert!(!c.is_single());
        // Instance 2: increment 2 times.
        c.incr();
        c.incr();
        assert_eq!(c.value, 5);
        assert_eq!(c.delta, 2);
        // Values: oldest=5 (value), newest=5-3=2.
        assert_eq!(collect_instance_values(&c, &pool), vec![5, 2]);
    }

    #[test]
    fn test_counter_push_multiple() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        // Instance 1: 3 increments.
        for _ in 0..3 {
            c.incr();
        }
        c.push(&mut pool);
        // Instance 2: 2 increments.
        for _ in 0..2 {
            c.incr();
        }
        c.push(&mut pool);
        // Instance 3: 1 increment.
        c.incr();
        assert_eq!(c.value, 6);
        assert_eq!(c.delta, 1);
        // Instances: 6, 6-3=3, 3-2=1
        assert_eq!(collect_instance_values(&c, &pool), vec![6, 3, 1]);
    }

    #[test]
    fn test_counter_pop_oldest() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        // Two instances: delta 3, then delta 2.
        for _ in 0..3 {
            c.incr();
        }
        c.push(&mut pool);
        for _ in 0..2 {
            c.incr();
        }
        assert_eq!(c.value, 5);
        // Instances: 5, 2.
        // Pop oldest (value=5), remaining value becomes 5-3=2.
        let exhausted = c.pop(&mut pool);
        assert!(!exhausted);
        assert_eq!(c.value, 2);
        assert!(c.is_single());
        assert_eq!(collect_instance_values(&c, &pool), vec![2]);
    }

    #[test]
    fn test_counter_pop_last_instance_returns_true() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        c.incr();
        c.incr();
        // Single instance, no deltas in list.
        assert!(c.is_single());
        let exhausted = c.pop(&mut pool);
        assert!(exhausted);
    }

    #[test]
    fn test_counter_push_pop_cycle() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        // Push 3 instances, then pop them all.
        for _ in 0..2 {
            c.incr();
        }
        c.push(&mut pool); // delta=2
        for _ in 0..3 {
            c.incr();
        }
        c.push(&mut pool); // delta=3
        c.incr(); // delta=1
        // value=6, instances: 6, 6-2=4, 4-3=1
        assert_eq!(collect_instance_values(&c, &pool), vec![6, 4, 1]);

        // Pop oldest (6).
        assert!(!c.pop(&mut pool));
        assert_eq!(c.value, 4);
        assert_eq!(collect_instance_values(&c, &pool), vec![4, 1]);

        // Pop oldest (4).
        assert!(!c.pop(&mut pool));
        assert_eq!(c.value, 1);
        assert!(c.is_single());

        // Pop last.
        assert!(c.pop(&mut pool));
    }

    #[test]
    fn test_counter_pop_recycles_pool_nodes() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        // Create 2 instances, pop one, verify pool free list.
        c.incr();
        c.push(&mut pool);
        c.incr();
        assert_eq!(pool.free.len(), 0);
        c.pop(&mut pool);
        assert_eq!(pool.free.len(), 1);
    }

    #[test]
    fn test_counter_any_instance_in_range_single() {
        let pool = DeltaPool::default();
        let mut c = Counter::default();
        for _ in 0..5 {
            c.incr();
        }
        assert!(c.any_instance_in_range(&pool, 3, 7));
        assert!(c.any_instance_in_range(&pool, 5, 5));
        assert!(!c.any_instance_in_range(&pool, 6, 10));
        assert!(!c.any_instance_in_range(&pool, 1, 4));
    }

    #[test]
    fn test_counter_any_instance_in_range_multiple() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        // Instance 1: 3 increments.
        for _ in 0..3 {
            c.incr();
        }
        c.push(&mut pool);
        // Instance 2: 2 increments.
        for _ in 0..2 {
            c.incr();
        }
        // Instances: 5, 2.
        // Check range that only matches oldest.
        assert!(c.any_instance_in_range(&pool, 5, 5));
        // Check range that only matches newest.
        assert!(c.any_instance_in_range(&pool, 2, 2));
        // Check range that matches both.
        assert!(c.any_instance_in_range(&pool, 1, 10));
        // Check range that matches neither.
        assert!(!c.any_instance_in_range(&pool, 3, 4));
    }

    #[test]
    fn test_counter_any_instance_in_range_three_instances() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        for _ in 0..4 {
            c.incr();
        }
        c.push(&mut pool); // delta=4
        for _ in 0..3 {
            c.incr();
        }
        c.push(&mut pool); // delta=3
        for _ in 0..2 {
            c.incr();
        }
        // value=9, instances: 9, 5, 2
        assert_eq!(collect_instance_values(&c, &pool), vec![9, 5, 2]);
        assert!(c.any_instance_in_range(&pool, 9, 9));
        assert!(c.any_instance_in_range(&pool, 5, 5));
        assert!(c.any_instance_in_range(&pool, 2, 2));
        assert!(!c.any_instance_in_range(&pool, 3, 4));
        assert!(!c.any_instance_in_range(&pool, 6, 8));
    }

    #[test]
    fn test_counter_is_single_transitions() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        assert!(c.is_single());
        c.incr(); // value=1, delta=1
        assert!(c.is_single());
        c.push(&mut pool); // saves delta=1
        assert!(!c.is_single());
        c.incr(); // value=2, delta=1
        c.push(&mut pool); // saves delta=1
        assert!(!c.is_single());
        c.incr(); // value=3, delta=1
        // Instances: 3, 2, 1.  Pop oldest (3) → value=2.
        c.pop(&mut pool);
        assert!(!c.is_single());
        // Pop oldest (2) → value=1.
        c.pop(&mut pool);
        assert!(c.is_single());
    }

    #[test]
    fn test_two_counters_share_pool() {
        let mut pool = DeltaPool::default();
        let mut c1 = Counter::default();
        let mut c2 = Counter::default();
        // Counter 1: push delta=2.
        c1.incr();
        c1.incr();
        c1.push(&mut pool);
        // Counter 2: push delta=5.
        for _ in 0..5 {
            c2.incr();
        }
        c2.push(&mut pool);
        // Continue incrementing both.
        c1.incr();
        c2.incr();
        // c1 instances: 3, 1.  c2 instances: 6, 1.
        assert_eq!(collect_instance_values(&c1, &pool), vec![3, 1]);
        assert_eq!(collect_instance_values(&c2, &pool), vec![6, 1]);
        // Pop from c1 ��� should not affect c2.
        c1.pop(&mut pool);
        assert_eq!(collect_instance_values(&c1, &pool), vec![1]);
        assert_eq!(collect_instance_values(&c2, &pool), vec![6, 1]);
    }

    #[test]
    fn test_pool_node_reuse_across_counters() {
        let mut pool = DeltaPool::default();
        let mut c1 = Counter::default();
        let mut c2 = Counter::default();
        // c1 allocates and frees a node.
        c1.incr();
        c1.push(&mut pool);
        c1.incr();
        let node_used = c1.head;
        c1.pop(&mut pool);
        assert_eq!(pool.free.len(), 1);
        assert_eq!(pool.free[0], node_used);
        // c2 allocates — should reuse the freed node.
        c2.incr();
        c2.push(&mut pool);
        assert_eq!(c2.head, node_used);
        assert_eq!(pool.free.len(), 0);
    }

    #[test]
    fn test_counter_push_pop_interleaved_with_incr() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        // Simulate a realistic pattern: push, incr everything, pop, push.
        c.incr();
        c.push(&mut pool);
        c.incr();
        // Instances: 2, 1.
        assert_eq!(collect_instance_values(&c, &pool), vec![2, 1]);
        c.push(&mut pool);
        c.incr();
        // Instances: 3, 2, 1.
        assert_eq!(collect_instance_values(&c, &pool), vec![3, 2, 1]);
        // Pop the oldest (3).
        c.pop(&mut pool);
        // Instances: 2, 1.
        assert_eq!(collect_instance_values(&c, &pool), vec![2, 1]);
        // Push again.
        c.push(&mut pool);
        c.incr();
        // Instances: 3, 2, 1.
        assert_eq!(collect_instance_values(&c, &pool), vec![3, 2, 1]);
    }

    #[test]
    fn test_pool_reset_then_counter_from_scratch() {
        let mut pool = DeltaPool::default();
        let mut c = Counter::default();
        c.incr();
        c.push(&mut pool);
        c.incr();
        pool.reset();
        // After reset, start fresh.
        let mut c2 = Counter::default();
        c2.incr();
        c2.incr();
        c2.push(&mut pool);
        c2.incr();
        assert_eq!(collect_instance_values(&c2, &pool), vec![3, 1]);
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
    /// Uses `ParserBuilder` with `utf8(false)` so that `.` in `(?s-u)`
    /// mode produces a `Class::Bytes` covering all 256 byte values.
    fn parse_hir_bytes(pattern: &str) -> Hir {
        use regex_syntax::ast::parse::ParserBuilder;
        use regex_syntax::hir::translate::TranslatorBuilder;

        let full = format!("(?s-u){}", pattern);
        let ast = ParserBuilder::new()
            .build()
            .parse(&full)
            .expect("regex-syntax AST parse should succeed");
        TranslatorBuilder::new()
            .utf8(false)
            .build()
            .translate(&full, &ast)
            .expect("regex-syntax HIR translation should succeed")
    }

    /// Build a compiled [`Regex`] from a pattern string and assert that
    /// its memory footprint equals `expected_bytes`.
    fn build_regex(pattern: &str, expected_bytes: usize) -> Regex {
        let hir = parse_hir_bytes(pattern);
        let mut builder = RegexBuilder::default();
        let regex = builder
            .build(&hir)
            .expect("our builder should accept the HIR");
        let actual = regex.memory_size();
        assert_eq!(
            actual, expected_bytes,
            "memory_size mismatch for pattern `{}`: actual={}, expected={}",
            pattern, actual, expected_bytes,
        );
        regex
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
        let re = build_regex(p, 816);
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
    }

    /// `(a|bc){1,2}` — flat range repetition with all combos up to 3.
    #[test]
    fn test_range() {
        use itertools::Itertools;

        let p = "^(a|bc){1,2}$";
        let re = build_regex(p, 600);

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
    }

    /// `((a|bc){1,2}){2,3}` — nested counting constraints.
    #[test]
    fn test_nested_counting() {
        use itertools::Itertools;

        let p = "^((a|bc){1,2}){2,3}$";
        let re = build_regex(p, 1096);

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
    }

    /// `(a|a?){2,3}` — epsilon-matchable body (the `a?` branch can match
    /// empty).  Exercises the epsilon-body detection logic in `addstate`.
    #[test]
    fn test_aaaaa() {
        let p = "^(a|a?){2,3}$";
        let re = build_regex(p, 600);

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
    }

    /// `a+` — basic one-or-more repetition.
    #[test]
    fn test_one_plus_basic() {
        let p = "^a+$";
        let re = build_regex(p, 272);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "aab");
    }

    /// `.+` — one-or-more wildcard.
    #[test]
    fn test_one_plus_wildcard() {
        let p = "^.+$";
        let re = build_regex(p, 528);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
    }

    /// `a+b+` — consecutive one-or-more repetitions.
    #[test]
    fn test_one_plus_catenation() {
        let p = "^a+b+$";
        let re = build_regex(p, 352);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "abb");
        assert_matches_regex_crate(p, &re, "aabb");
        assert_matches_regex_crate(p, &re, "ba");
    }

    /// `(ab)+` — one-or-more of a multi-byte sequence.
    #[test]
    fn test_one_plus_group() {
        let p = "^(ab)+$";
        let re = build_regex(p, 312);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "ababab");
        assert_matches_regex_crate(p, &re, "aba");
    }

    /// `(a|b)+` — one-or-more alternation.
    #[test]
    fn test_one_plus_alternate() {
        let p = "^(a|b)+$";
        let re = build_regex(p, 528);
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
    }

    /// `.*a.{3}b+c` — one-or-more mixed with counting constraints.
    #[test]
    fn test_one_plus_with_counting() {
        let p = "^.*a.{3}b+c$";
        let re = build_regex(p, 856);
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
    }

    /// `(a{2,3})+` — inner repetition, outer one-or-more.
    /// The body of `+` is itself a counted repetition.
    #[test]
    fn test_repetition_inside_one_plus() {
        let p = "^(a{2,3})+$";
        let re = build_regex(p, 400);
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
    }

    /// `((a|bc){1,2})+` — inner range repetition of alternation, outer `+`.
    #[test]
    fn test_range_alternation_inside_one_plus() {
        use itertools::Itertools;

        let p = "^((a|bc){1,2})+$";
        let re = build_regex(p, 640);

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
    }

    /// `(a+){2,3}` — inner one-or-more, outer counted repetition.
    #[test]
    fn test_one_plus_inside_repetition() {
        let p = "^(a+){2,3}$";
        let re = build_regex(p, 440);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaaa");
        assert_matches_regex_crate(p, &re, "b");
    }

    /// `((a|b)+){2,4}` — inner `+` of alternation, outer counted repetition.
    #[test]
    fn test_one_plus_alternation_inside_repetition() {
        use itertools::Itertools;

        let p = "^((a|b)+){2,4}$";
        let re = build_regex(p, 696);

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
    }

    /// `(a+b{2,3})+` — inner `+` and inner repetition side-by-side,
    /// wrapped in outer `+`.
    #[test]
    fn test_mixed_plus_and_repetition_inside_one_plus() {
        let p = "^(a+b{2,3})+$";
        let re = build_regex(p, 480);
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
    }

    // -- min=0 repetition tests ---------------------------------------------

    /// `a{0,2}` — zero to two occurrences of a single byte.
    #[test]
    fn test_min_zero_basic() {
        let p = "^a{0,2}$";
        let re = build_regex(p, 400);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "b");
    }

    /// `a{0,1}` — equivalent to `a?`.
    #[test]
    fn test_min_zero_max_one() {
        let p = "^a{0,1}$";
        let re = build_regex(p, 272);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "b");
    }

    /// `(a|bc){0,3}` — zero to three of an alternation.
    #[test]
    fn test_min_zero_alternation() {
        use itertools::Itertools;

        let p = "^(a|bc){0,3}$";
        let re = build_regex(p, 640);

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
    }

    /// `a{0,}` — zero or more, lowered to `a*` (no counter overhead).
    #[test]
    fn test_min_zero_unbounded() {
        let p = "^a{0,}$";
        let re = build_regex(p, 272);
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
    }

    /// `(ab){0,}` — zero or more of a group, lowered to `(ab)*`.
    #[test]
    fn test_min_zero_unbounded_group() {
        let p = "^(ab){0,}$";
        let re = build_regex(p, 312);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "ababab");
        assert_matches_regex_crate(p, &re, "aba");
    }

    /// `x(a{0,2})+y` — min=0 repetition nested inside `+`.
    #[test]
    fn test_min_zero_inside_one_plus() {
        let p = "^x(a{0,2})+y$";
        let re = build_regex(p, 520);
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
    }

    /// `(a{0,2}){2,3}` — min=0 inner, counted outer.
    #[test]
    fn test_min_zero_inside_repetition() {
        let p = "^(a{0,2}){2,3}$";
        let re = build_regex(p, 696);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaaa");
        assert_matches_regex_crate(p, &re, "b");
    }

    /// `(a+){0,3}` — `+` inside a min=0 counted repetition.
    #[test]
    fn test_one_plus_inside_min_zero_repetition() {
        let p = "^(a+){0,3}$";
        let re = build_regex(p, 480);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "aaaaaa");
        assert_matches_regex_crate(p, &re, "b");
    }

    /// `.{0,3}` — min=0 repetition on wildcard.
    #[test]
    fn test_min_zero_wildcard() {
        let p = "^.{0,3}$";
        let re = build_regex(p, 656);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abcd");
    }

    /// `a{0,3}` — same as `a{0,3}` (the old test used `min: None`).
    #[test]
    fn test_none_min_repetition() {
        let p = "^a{0,3}$";
        let re = build_regex(p, 400);
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
    }

    // -- Standalone primitive tests ------------------------------------------

    /// `a` — single literal byte.
    #[test]
    fn test_literal_single() {
        let p = "^a$";
        let re = build_regex(p, 232);
        assert_matches_regex_crate(p, &re, "a");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "ba");
    }

    /// `abc` — multi-byte literal concatenation.
    #[test]
    fn test_literal_multi() {
        let p = "^abc$";
        let re = build_regex(p, 312);
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
    }

    /// `.` — bare wildcard (matches exactly one byte).
    #[test]
    fn test_dot_single() {
        let p = "^.$";
        let re = build_regex(p, 488);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "z");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, " ");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
    }

    /// `a|bc` — bare alternation (no repetition).
    #[test]
    fn test_alternation_bare() {
        let p = "^(a|bc)$";
        let re = build_regex(p, 352);
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
    }

    /// `a|b|c` — three-way alternation.
    #[test]
    fn test_alternation_three_way() {
        let p = "^(a|b|c)$";
        let re = build_regex(p, 488);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "d");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
    }

    /// `a?` — standalone zero-or-one.
    #[test]
    fn test_question_mark_single() {
        let p = "^a?$";
        let re = build_regex(p, 272);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        // negatives
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
    }

    /// `(ab)?` — zero-or-one of a group.
    #[test]
    fn test_question_mark_group() {
        let p = "^(ab)?$";
        let re = build_regex(p, 312);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "ab");
        // negatives
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ba");
        assert_matches_regex_crate(p, &re, "abab");
        assert_matches_regex_crate(p, &re, "abc");
    }

    /// `a?b` — optional prefix followed by a literal.
    #[test]
    fn test_question_mark_prefix() {
        let p = "^a?b$";
        let re = build_regex(p, 312);
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ab");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aab");
        assert_matches_regex_crate(p, &re, "bb");
        assert_matches_regex_crate(p, &re, "cb");
        assert_matches_regex_crate(p, &re, "abc");
    }

    /// `a*` — standalone zero-or-more.
    #[test]
    fn test_star_single() {
        let p = "^a*$";
        let re = build_regex(p, 272);
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
    }

    /// `(ab)*` — zero-or-more of a group.
    #[test]
    fn test_star_group() {
        let p = "^(ab)*$";
        let re = build_regex(p, 312);
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
    }

    /// `a*b` — star followed by a literal.
    #[test]
    fn test_star_then_literal() {
        let p = "^a*b$";
        let re = build_regex(p, 312);
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
    }

    /// `a{2,}` — unbounded min with n>0.
    #[test]
    fn test_min_n_unbounded() {
        let p = "^a{2,}$";
        let re = build_regex(p, 360);
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
    }

    /// `(ab){2,}` — unbounded min of a group.
    #[test]
    fn test_min_n_unbounded_group() {
        let p = "^(ab){2,}$";
        let re = build_regex(p, 440);
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
    }

    /// `a{3,5}` — bounded min>0 range.
    #[test]
    fn test_bounded_range() {
        let p = "^a{3,5}$";
        let re = build_regex(p, 360);
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
    }

    /// `a{3,3}` — exact repetition.
    #[test]
    fn test_exact_repetition() {
        let p = "^a{3,3}$";
        let re = build_regex(p, 360);
        assert_matches_regex_crate(p, &re, "aaa");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "aa");
        assert_matches_regex_crate(p, &re, "aaaa");
        assert_matches_regex_crate(p, &re, "aaaaa");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "bbb");
    }

    // -- Byte class tests ---------------------------------------------------

    /// `[a-c]` — a small contiguous byte range (Class::Bytes).
    #[test]
    fn test_byte_class_range() {
        let p = "^[a-c]$";
        let re = build_regex(p, 488);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "c");
        assert_matches_regex_crate(p, &re, "d");
        assert_matches_regex_crate(p, &re, "ab");
    }

    /// `[a-c]+` — one-or-more of a byte class.
    #[test]
    fn test_byte_class_one_plus() {
        let p = "^[a-c]+$";
        let re = build_regex(p, 528);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "cba");
        assert_matches_regex_crate(p, &re, "abcd");
        assert_matches_regex_crate(p, &re, "d");
    }

    /// `[a-c]{2,3}` — counted repetition of a byte class.
    #[test]
    fn test_byte_class_counted() {
        let p = "^[a-c]{2,3}$";
        let re = build_regex(p, 616);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "abca");
        assert_matches_regex_crate(p, &re, "cc");
        assert_matches_regex_crate(p, &re, "dd");
    }

    /// `[ax]` — disjoint single bytes (multi-range Class::Bytes).
    #[test]
    fn test_byte_class_disjoint() {
        let p = "^[ax]$";
        let re = build_regex(p, 488);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "x");
        assert_matches_regex_crate(p, &re, "b");
        assert_matches_regex_crate(p, &re, "ax");
    }

    /// `[a-cx-z]+` — multiple disjoint ranges in a byte class.
    #[test]
    fn test_byte_class_multi_range() {
        let p = "^[a-cx-z]+$";
        let re = build_regex(p, 528);
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "x");
        assert_matches_regex_crate(p, &re, "axbycz");
        assert_matches_regex_crate(p, &re, "d");
        assert_matches_regex_crate(p, &re, "w");
        assert_matches_regex_crate(p, &re, "abcxyz");
    }

    /// `[a-c].*[x-z]` — byte classes mixed with wildcard.
    #[test]
    fn test_byte_class_with_wildcard() {
        let p = "^[a-c].*[x-z]$";
        let re = build_regex(p, 1120);
        assert_matches_regex_crate(p, &re, "ax");
        assert_matches_regex_crate(p, &re, "a123z");
        assert_matches_regex_crate(p, &re, "bx");
        assert_matches_regex_crate(p, &re, "dx");
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
    }

    // -- Predefined character class tests -----------------------------------

    /// `\d` — matches a single ASCII digit.
    #[test]
    fn test_digit() {
        let p = r"^\d$";
        let re = build_regex(p, 488);
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
    }

    /// `\d+` — one-or-more digits.
    #[test]
    fn test_digit_plus() {
        let p = r"^\d+$";
        let re = build_regex(p, 528);
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
    }

    /// `\d{3,5}` — counted digit repetition.
    #[test]
    fn test_digit_counted() {
        let p = r"^\d{3,5}$";
        let re = build_regex(p, 616);
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
    }

    /// `\D` — matches a single non-digit byte.
    #[test]
    fn test_non_digit() {
        let p = r"^\D$";
        let re = build_regex(p, 488);
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
    }

    /// `\D+` — one-or-more non-digits.
    #[test]
    fn test_non_digit_plus() {
        let p = r"^\D+$";
        let re = build_regex(p, 528);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "hello world");
        assert_matches_regex_crate(p, &re, "!@#");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "abc1");
        assert_matches_regex_crate(p, &re, "1abc");
    }

    /// `\s` — matches a single ASCII whitespace byte.
    #[test]
    fn test_space() {
        let p = r"^\s$";
        let re = build_regex(p, 488);
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "\t");
        assert_matches_regex_crate(p, &re, "\n");
        assert_matches_regex_crate(p, &re, "\r");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "  ");
    }

    /// `\s+` — one-or-more whitespace.
    #[test]
    fn test_space_plus() {
        let p = r"^\s+$";
        let re = build_regex(p, 528);
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "   ");
        assert_matches_regex_crate(p, &re, " \t\n\r");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, " a");
        assert_matches_regex_crate(p, &re, "a ");
    }

    /// `\S` — matches a single non-whitespace byte.
    #[test]
    fn test_non_space() {
        let p = r"^\S$";
        let re = build_regex(p, 488);
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, "!");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "\t");
        assert_matches_regex_crate(p, &re, "\n");
        assert_matches_regex_crate(p, &re, "aa");
    }

    /// `\S+` — one-or-more non-whitespace.
    #[test]
    fn test_non_space_plus() {
        let p = r"^\S+$";
        let re = build_regex(p, 528);
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "123");
        assert_matches_regex_crate(p, &re, "a1!");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "a b");
        assert_matches_regex_crate(p, &re, " abc");
    }

    /// `\w` — matches a single ASCII word byte (`[0-9A-Za-z_]`).
    #[test]
    fn test_word() {
        let p = r"^\w$";
        let re = build_regex(p, 488);
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
    }

    /// `\w+` — one-or-more word bytes.
    #[test]
    fn test_word_plus() {
        let p = r"^\w+$";
        let re = build_regex(p, 528);
        assert_matches_regex_crate(p, &re, "hello");
        assert_matches_regex_crate(p, &re, "foo_bar");
        assert_matches_regex_crate(p, &re, "x123");
        assert_matches_regex_crate(p, &re, "___");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "hello world");
        assert_matches_regex_crate(p, &re, "foo-bar");
    }

    /// `\w{2,4}` — counted word repetition.
    #[test]
    fn test_word_counted() {
        let p = r"^\w{2,4}$";
        let re = build_regex(p, 616);
        assert_matches_regex_crate(p, &re, "ab");
        assert_matches_regex_crate(p, &re, "abc");
        assert_matches_regex_crate(p, &re, "a1_Z");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "abcde");
        assert_matches_regex_crate(p, &re, "!!");
        assert_matches_regex_crate(p, &re, "a b");
    }

    /// `\W` — matches a single non-word byte.
    #[test]
    fn test_non_word() {
        let p = r"^\W$";
        let re = build_regex(p, 488);
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
    }

    /// `\W+` — one-or-more non-word bytes.
    #[test]
    fn test_non_word_plus() {
        let p = r"^\W+$";
        let re = build_regex(p, 528);
        assert_matches_regex_crate(p, &re, " ");
        assert_matches_regex_crate(p, &re, "!@#");
        assert_matches_regex_crate(p, &re, " - ");
        // negatives
        assert_matches_regex_crate(p, &re, "");
        assert_matches_regex_crate(p, &re, "a");
        assert_matches_regex_crate(p, &re, "0");
        assert_matches_regex_crate(p, &re, " a ");
        assert_matches_regex_crate(p, &re, "!a!");
    }

    /// `\d+\s+\w+` — mixed predefined classes in concatenation.
    #[test]
    fn test_predefined_mixed() {
        let p = r"^\d+\s+\w+$";
        let re = build_regex(p, 1200);
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
        // The second `\d` adds one ByteClass state but no extra class
        // table — so the difference is exactly one state.
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
}
