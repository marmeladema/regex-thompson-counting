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
//! A repetition `body{min,max}` is lowered to:
//!
//! ```text
//! body ── CounterInstance(c) ── body_copy ── CounterIncrement(c, min, max)
//!                                  ^                    │
//!                                  └── continue ────────┘
//!                                            break ──> (next)
//! ```
//!
//! The first `body` is the mandatory initial match.  `CounterInstance`
//! allocates a new counter instance (or creates the counter from scratch).
//! The `body_copy` + `CounterIncrement` loop runs zero or more additional
//! times.  `CounterIncrement` increments all active instances; when the
//! oldest instance's value falls in `[min, max]`, the break path is
//! followed.  When it reaches `max`, the oldest instance is de-allocated.
//!
//! Counters use a **differential representation** (from the Becchi paper)
//! so that increment and condition-evaluation require O(1) work regardless
//! of how many instances are active.
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

use std::collections::VecDeque;
use std::fmt;
use std::io::Write;

use regex_syntax::hir::{self, Hir, HirKind};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// An error returned when the HIR contains constructs we don't support.
#[derive(Debug)]
enum Error {
    /// A character class (e.g. `\w`, `[a-z]`) was encountered.
    UnsupportedClass(hir::Class),
    /// A look-around assertion (e.g. `^`, `$`, `\b`) was encountered.
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

// ---------------------------------------------------------------------------
// NFA states
// ---------------------------------------------------------------------------

/// A single NFA state.
///
/// Epsilon states (`Split`, `CounterInstance`, `CounterIncrement`)
/// are followed during [`Matcher::addstate`].  Byte-consuming states
/// (`Byte`, `Wildcard`) are stepped over in [`Matcher::step`].
#[derive(Clone, Copy, Debug)]
enum State {
    /// Epsilon fork: follow both `out` and `out1`.
    Split { out: usize, out1: usize },

    /// Allocate (or push) a new instance on counter `counter`, then
    /// follow `out`.
    CounterInstance { counter: usize, out: usize },

    /// Increment counter `counter`.
    ///
    /// - **Continue** (`out`): re-enter the repetition body (taken when
    ///   the counter has not yet reached `max`, or when there are
    ///   multiple instances and the oldest has not yet reached `max`).
    /// - **Break** (`out1`): exit the repetition (taken when any instance
    ///   value falls in `[min, max]`).
    CounterIncrement {
        counter: usize,
        out: usize,
        out1: usize,
        min: usize,
        max: usize,
    },

    /// Match a literal byte, then follow `out`.
    Byte { byte: u8, out: usize },

    /// Match any byte, then follow `out`.
    Wildcard { out: usize },

    /// Accepting state.
    Match,
}

impl State {
    /// Return the "dangling out" pointer used by [`RegexBuilder::patch`]
    /// and [`RegexBuilder::append`] to thread fragment lists.
    fn next(self) -> usize {
        match self {
            State::Byte { out, .. }
            | State::Wildcard { out, .. }
            | State::CounterInstance { out, .. } => out,
            State::Split { out1, .. } | State::CounterIncrement { out1, .. } => out1,
            _ => unreachable!(),
        }
    }

    /// Overwrite the "dangling out" pointer.
    fn append(&mut self, next: usize) {
        match self {
            State::Byte { out, .. }
            | State::Wildcard { out, .. }
            | State::CounterInstance { out, .. } => *out = next,
            State::Split { out1, .. } | State::CounterIncrement { out1, .. } => *out1 = next,
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// NFA fragment (used during construction)
// ---------------------------------------------------------------------------

/// A partially-built NFA fragment with a `start` state and a dangling
/// `out` pointer that will be patched to the next fragment's start.
#[derive(Debug)]
struct Fragment {
    start: usize,
    out: usize,
}

impl Fragment {
    fn new(start: usize, out: usize) -> Self {
        Self { start, out }
    }
}

// ---------------------------------------------------------------------------
// Postfix HIR nodes
// ---------------------------------------------------------------------------

/// A postfix HIR instruction consumed by [`RegexBuilder::next_fragment`]
/// to emit NFA states.
#[derive(Copy, Clone, Debug)]
enum RegexHirNode {
    Alternate,
    Catenate,
    Byte(u8),
    RepeatZeroOne,
    RepeatZeroPlus,
    RepeatOnePlus,
    Wildcard,
    CounterInstance {
        counter: usize,
    },
    CounterIncrement {
        counter: usize,
        min: usize,
        max: usize,
    },
}

// ---------------------------------------------------------------------------
// Compiled regex
// ---------------------------------------------------------------------------

struct StateList(Box<[State]>);

impl fmt::Debug for StateList {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_map()
            .entries(self.0.iter().enumerate().map(|(idx, state)| (idx, state)))
            .finish()
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
struct Regex {
    states: StateList,
    start: usize,
    /// One slot per counter variable allocated during compilation.
    counters: Box<[usize]>,
}

impl Regex {
    /// Emit a Graphviz DOT representation of the NFA.
    #[allow(dead_code)]
    fn to_dot(&self, mut buffer: impl Write) {
        let mut visited = vec![false; self.states.len()];
        writeln!(buffer, "digraph graphname {{").unwrap();
        writeln!(buffer, "\trankdir=LR;").unwrap();
        writeln!(&mut buffer, "\t{} [shape=box];", self.start).unwrap();
        let mut stack = vec![self.start];
        while let Some(idx) = stack.pop() {
            if !visited[idx] {
                writeln!(buffer, "\t// [{}] {:?}", idx, self.states[idx]).unwrap();
                self.write_dot_state(idx, &mut buffer, &mut stack);
                visited[idx] = true;
            }
        }
        writeln!(buffer, "}}").unwrap();
    }

    fn write_dot_state(&self, idx: usize, buffer: &mut impl Write, stack: &mut Vec<usize>) {
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
            State::Wildcard { out } => {
                stack.push(out);
                writeln!(buffer, "\t{} -> {} [label=\".\"];", idx, out).unwrap();
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

/// Sentinel used for yet-unpatched `out` pointers in NFA states.
const DANGLING: usize = usize::MAX;

/// Builds a compiled [`Regex`] from a [`regex_syntax::hir::Hir`].
///
/// The pipeline is:
/// 1. [`hir2postfix`](Self::hir2postfix) — recursively lowers the
///    `regex-syntax` HIR into a postfix sequence of [`RegexHirNode`]s.
/// 2. [`next_fragment`](Self::next_fragment) — consumes postfix nodes one
///    at a time, emitting NFA [`State`]s and wiring [`Fragment`]s together.
/// 3. [`build`](Self::build) — drives the pipeline and patches the final
///    fragment to the `Match` state.
#[derive(Debug, Default)]
struct RegexBuilder {
    postfix: Vec<RegexHirNode>,
    states: Vec<State>,
    frags: Vec<Fragment>,
    counters: Vec<usize>,
}

impl RegexBuilder {
    /// Allocate a fresh counter index.
    fn next_counter(&mut self) -> usize {
        let counter = self.counters.len();
        self.counters.push(counter);
        counter
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
                let ranges = class.ranges();
                // A single range covering all bytes (0x00..=0xFF) is our
                // Wildcard.
                if ranges.len() == 1 && ranges[0].start() == 0x00 && ranges[0].end() == 0xFF {
                    self.postfix.push(RegexHirNode::Wildcard);
                    return Ok(());
                }
                // Lower an arbitrary byte class to alternations of Byte
                // nodes.  Each contiguous range `[lo..=hi]` expands to
                // individual bytes joined by Alternate.
                let mut total = 0usize;
                for range in ranges {
                    for b in range.start()..=range.end() {
                        self.postfix.push(RegexHirNode::Byte(b));
                        total += 1;
                        if total > 1 {
                            self.postfix.push(RegexHirNode::Alternate);
                        }
                    }
                }
                Ok(())
            }
            HirKind::Class(hir::Class::Unicode(class)) => {
                // regex-syntax may produce Unicode classes for ASCII-only
                // patterns like `(a|b)` → `[ab]`.  If all ranges fit in a
                // single byte (0x00..=0xFF), lower them to byte
                // alternations; otherwise reject.
                let ranges = class.ranges();
                let all_single_byte = ranges
                    .iter()
                    .all(|r| (r.start() as u32) <= 0xFF && (r.end() as u32) <= 0xFF);
                if !all_single_byte {
                    return Err(Error::UnsupportedClass(hir::Class::Unicode(class.clone())));
                }
                let mut total = 0usize;
                for range in ranges {
                    for b in (range.start() as u8)..=(range.end() as u8) {
                        self.postfix.push(RegexHirNode::Byte(b));
                        total += 1;
                        if total > 1 {
                            self.postfix.push(RegexHirNode::Alternate);
                        }
                    }
                }
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

                    // Emit the body once (mandatory initial match).
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
    fn state(&mut self, state: State) -> usize {
        let idx = self.states.len();
        self.states.push(state);
        idx
    }

    /// Walk the linked list of dangling `out` pointers starting at `list`
    /// and patch each one to point to `idx`.
    fn patch(&mut self, mut list: usize, idx: usize) {
        while let Some(state) = self.states.get_mut(list) {
            list = match state {
                State::Byte { out, .. }
                | State::Wildcard { out, .. }
                | State::CounterInstance { out, .. } => {
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
    fn append(&mut self, list1: usize, list2: usize) -> usize {
        let len = self.states.len();
        let mut s = &mut self.states[list1];
        let mut next = s.next();
        while next < len {
            s = &mut self.states[next];
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
                    out1: DANGLING,
                });
                Fragment::new(s, self.append(e.out, s))
            }
            RegexHirNode::RepeatZeroPlus => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::Split {
                    out: e.start,
                    out1: DANGLING,
                });
                self.patch(e.out, s);
                Fragment::new(s, s)
            }
            RegexHirNode::RepeatOnePlus => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::Split {
                    out: e.start,
                    out1: DANGLING,
                });
                self.patch(e.out, s);
                Fragment::new(e.start, s)
            }
            RegexHirNode::CounterInstance { counter } => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::CounterInstance {
                    counter,
                    out: DANGLING,
                });
                self.patch(e.out, s);
                Fragment::new(e.start, s)
            }
            RegexHirNode::CounterIncrement { counter, min, max } => {
                let e = self.frags.pop().unwrap();
                let s = self.state(State::CounterIncrement {
                    out: e.start,
                    out1: DANGLING,
                    min,
                    max,
                    counter,
                });
                self.patch(e.out, s);
                Fragment::new(s, s)
            }
            RegexHirNode::Wildcard => {
                let idx = self.state(State::Wildcard { out: DANGLING });
                Fragment::new(idx, idx)
            }
            RegexHirNode::Byte(byte) => {
                let idx = self.state(State::Byte {
                    byte,
                    out: DANGLING,
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

        Ok(Regex {
            states: StateList(self.states.to_vec().into_boxed_slice()),
            start,
            counters: self.counters.to_vec().into_boxed_slice(),
        })
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
/// # Fields
///
/// - `value` — the oldest (highest) instance's value.
/// - `delta` — how much the newest instance has accumulated since the
///   last [`push`](Self::push).
/// - `deltas` — FIFO of deltas for intermediate instances (oldest at
///   front).
/// - `incremented` — set by [`incr`](Self::incr), cleared at each
///   simulation step; used to prevent a `CounterIncrement` state from
///   being processed twice in the same epsilon closure.
#[derive(Clone, Debug)]
struct Counter {
    incremented: bool,
    value: usize,
    delta: usize,
    deltas: VecDeque<usize>,
}

impl Default for Counter {
    fn default() -> Self {
        Self {
            incremented: false,
            value: 0,
            delta: 0,
            deltas: VecDeque::default(),
        }
    }
}

impl Counter {
    /// Push (allocate) a new instance.  The current `delta` is saved and
    /// reset to zero.
    fn push(&mut self) {
        assert!(self.delta > 0);
        self.deltas.push_back(self.delta);
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
    fn pop(&mut self) -> bool {
        assert!(self.value > 0);
        if let Some(delta) = self.deltas.pop_front() {
            assert!(delta < self.value);
            self.value -= delta;
            false
        } else {
            assert_eq!(self.value, self.delta);
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Matcher (NFA simulation)
// ---------------------------------------------------------------------------

/// Reusable memory for [`Matcher`].  Create once, call
/// [`matcher`](Self::matcher) for each regex to match.
#[derive(Debug, Default)]
struct MatcherMemory {
    /// Per-state: the `listid` when the state was last added.  Used for
    /// O(1) deduplication in `addstate`.
    lastlist: Vec<usize>,
    /// One slot per counter variable.
    counters: Vec<Option<Counter>>,
    /// Current and next state lists (swapped each step).
    clist: Vec<usize>,
    nlist: Vec<usize>,
    /// Per-state snapshot of `ci_gen` at first visit in the current step.
    /// See the module-level doc comment for the motivation.
    ci_gen_at_visit: Vec<usize>,
}

impl MatcherMemory {
    fn matcher<'a>(&'a mut self, regex: &'a Regex) -> Matcher<'a> {
        self.lastlist.clear();
        self.lastlist.resize(regex.states.len(), usize::MAX);
        self.counters.clear();
        self.counters.resize(regex.counters.len(), None);
        self.clist.clear();
        self.nlist.clear();
        self.ci_gen_at_visit.clear();
        self.ci_gen_at_visit.resize(regex.states.len(), 0);

        let mut m = Matcher {
            counters: &mut self.counters,
            states: &regex.states,
            lastlist: &mut self.lastlist,
            listid: 0,
            clist: &mut self.clist,
            nlist: &mut self.nlist,
            ci_gen: 0,
            ci_gen_at_visit: &mut self.ci_gen_at_visit,
        };

        m.startlist(regex.start);
        m
    }
}

/// Runs a Thompson NFA simulation with counting-constraint support.
#[derive(Debug)]
struct Matcher<'a> {
    counters: &'a mut [Option<Counter>],
    states: &'a [State],
    /// Per-state deduplication stamp (compared against `listid`).
    lastlist: &'a mut [usize],
    /// Monotonically increasing step ID.
    listid: usize,
    /// Current active state list.
    clist: &'a mut Vec<usize>,
    /// Next active state list (built during a step).
    nlist: &'a mut Vec<usize>,
    /// Monotonically increasing generation counter; incremented every
    /// time [`addcounter`](Self::addcounter) is called.  Used together
    /// with `ci_gen_at_visit` to detect whether a `CounterInstance`
    /// fired between the first visit and a re-entry at a
    /// `CounterIncrement` state.
    ci_gen: usize,
    /// Per-state snapshot of `ci_gen` recorded on first visit.
    ci_gen_at_visit: &'a mut [usize],
}

impl<'a> Matcher<'a> {
    /// Compute the initial state list by following all epsilon transitions
    /// from `start`.
    #[inline]
    fn startlist(&mut self, start: usize) {
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
    fn addcounter(&mut self, idx: usize) {
        if let Some(counter) = self.counters[idx].as_mut() {
            counter.push();
        } else {
            self.counters[idx] = Some(Counter::default());
        }
        self.ci_gen += 1;
    }

    /// Increment all instances of counter `idx`.
    fn inccounter(&mut self, idx: usize) {
        self.counters[idx].as_mut().unwrap().incr();
    }

    /// De-allocate the oldest instance of counter `idx`.  If no instances
    /// remain, the counter is set to `None`.
    fn delcounter(&mut self, idx: usize) {
        if self.counters[idx].as_mut().unwrap().pop() {
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
    fn counter_is_processable(&self, counter: usize, state_idx: usize) -> bool {
        self.counters[counter]
            .as_ref()
            .map_or(self.ci_gen > self.ci_gen_at_visit[state_idx], |c| {
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
    fn addstate(&mut self, idx: usize) {
        if self.lastlist[idx] == self.listid {
            let should_reenter = match self.states[idx] {
                State::CounterIncrement { counter, .. } => {
                    self.counter_is_processable(counter, idx)
                }
                _ => false,
            };
            if !should_reenter {
                return;
            }
        }

        // Record ci_gen only on first visit so re-entry compares against
        // the original snapshot.
        if self.lastlist[idx] != self.listid {
            self.ci_gen_at_visit[idx] = self.ci_gen;
        }
        self.lastlist[idx] = self.listid;

        match self.states[idx] {
            State::Split { out, out1 } => {
                self.addstate(out);
                self.addstate(out1);
            }

            State::CounterInstance { out, counter } => {
                self.addcounter(counter);
                self.addstate(out);
            }

            State::CounterIncrement {
                out,
                out1,
                counter,
                min,
                max,
            } if self.counter_is_processable(counter, idx) => {
                // Re-create the counter if it was exhausted (None).  This
                // only happens when ci_gen advanced (i.e. a CounterInstance
                // for an inner counter fired since our first visit).
                if self.counters[counter].is_none() {
                    self.counters[counter] = Some(Counter::default());
                }

                self.inccounter(counter);
                let value = self.counters[counter].as_ref().unwrap().value;
                debug_assert!(value > 0 && value <= max);
                let is_single = self.counters[counter].as_ref().unwrap().deltas.is_empty();

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
                    self.lastlist[idx] = self.listid.wrapping_sub(1);
                    self.addstate(out);
                    is_epsilon_body = self.lastlist[idx] == self.listid;
                    self.lastlist[idx] = self.listid;
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
                    let cnt = self.counters[counter].as_ref().unwrap();
                    let mut val = cnt.value;
                    let mut ok = val >= min && val <= max;
                    for delta in &cnt.deltas {
                        val -= delta;
                        ok = ok || (val >= min && val <= max);
                    }
                    ok
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

            // Byte, Wildcard, Match, or CounterIncrement whose guard
            // failed — just record the state for step() to inspect.
            _ => {}
        }

        self.nlist.push(idx);
    }

    /// Advance the simulation by one input byte.
    ///
    /// For each state in `clist`, if the byte matches (`Byte` or
    /// `Wildcard`), follow the `out` pointer through `addstate` to build
    /// the next `nlist`.
    fn step(&mut self, b: u8) {
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
                State::Wildcard { out } => self.addstate(out),
                _ => {}
            }
        }

        *self.clist = std::mem::replace(self.nlist, clist);
        self.listid += 1;
    }

    /// Feed an entire byte slice through the matcher, one byte at a time.
    fn chunk(&mut self, input: &[u8]) {
        for &b in input {
            self.step(b);
        }
    }

    /// Check whether the current state list contains a `Match` state.
    pub fn ismatch(&self) -> bool {
        self.clist
            .iter()
            .any(|&idx| matches!(self.states[idx], State::Match))
    }
}

// ---------------------------------------------------------------------------
// Entry point (unused — this crate is test-driven)
// ---------------------------------------------------------------------------

fn main() {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Assert that our NFA matcher and the `regex` crate agree on whether
    /// `input` matches the given pattern (anchored at both ends).
    ///
    /// The `regex` crate is used in byte mode (`regex::bytes::Regex`) so
    /// that `.` matches any byte, consistent with our engine.
    fn assert_matches_regex_crate(pattern: &str, input: &str) {
        let anchored = format!("^(?s-u){}$", pattern);
        let re = regex::bytes::Regex::new(&anchored).expect("regex crate should parse pattern");
        let expected = re.is_match(input.as_bytes());

        let hir = parse_hir_bytes(pattern);
        let mut builder = RegexBuilder::default();
        let regex = builder
            .build(&hir)
            .expect("our builder should accept the HIR");
        let mut memory = MatcherMemory::default();
        let mut matcher = memory.matcher(&regex);
        matcher.chunk(input.as_bytes());
        let actual = matcher.ismatch();

        assert_eq!(
            actual, expected,
            "mismatch for pattern `{}` on input {:?}: ours={}, regex crate={}",
            pattern, input, actual, expected
        );
    }

    /// `.*a.{3}bc` — counting constraint on a wildcard.
    #[test]
    fn test_counting() {
        let p = ".*a.{3}bc";
        assert_matches_regex_crate(p, "aybzbc");
        assert_matches_regex_crate(p, "axaybzbc");
        assert_matches_regex_crate(p, "abc");
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a123bc");
    }

    /// `(a|bc){1,2}` — flat range repetition with all combos up to 3.
    #[test]
    fn test_range() {
        use itertools::Itertools;

        let p = "(a|bc){1,2}";

        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "bc");

        // Two repetitions (all combos)
        for v in std::iter::repeat(["a", "bc"])
            .take(2)
            .map(|a| a.into_iter())
            .multi_cartesian_product()
        {
            let input = v.into_iter().collect::<String>();
            assert_matches_regex_crate(p, &input);
        }

        // Three repetitions — should not match (max is 2)
        for v in std::iter::repeat(["a", "bc"])
            .take(3)
            .map(|a| a.into_iter())
            .multi_cartesian_product()
        {
            let input = v.into_iter().collect::<String>();
            assert_matches_regex_crate(p, &input);
        }
    }

    /// `((a|bc){1,2}){2,3}` — nested counting constraints.
    #[test]
    fn test_nested_counting() {
        use itertools::Itertools;

        let p = "((a|bc){1,2}){2,3}";

        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "bc");

        for i in 2..=6 {
            for v in std::iter::repeat(["a", "bc"])
                .take(i)
                .map(|a| a.into_iter())
                .multi_cartesian_product()
            {
                let input = v.into_iter().collect::<String>();
                assert_matches_regex_crate(p, &input);
            }
        }

        for v in std::iter::repeat(["a", "bc"])
            .take(7)
            .map(|a| a.into_iter())
            .multi_cartesian_product()
        {
            let input = v.into_iter().collect::<String>();
            assert_matches_regex_crate(p, &input);
        }
    }

    /// `(a|a?){2,3}` — epsilon-matchable body (the `a?` branch can match
    /// empty).  Exercises the epsilon-body detection logic in `addstate`.
    #[test]
    fn test_aaaaa() {
        let p = "(a|a?){2,3}";

        assert_matches_regex_crate(p, "aaaaa");
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
    }

    /// `a+` — basic one-or-more repetition.
    #[test]
    fn test_one_plus_basic() {
        let p = "a+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
        assert_matches_regex_crate(p, "b");
    }

    /// `.+` — one-or-more wildcard.
    #[test]
    fn test_one_plus_wildcard() {
        let p = ".+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "ab");
        assert_matches_regex_crate(p, "abc");
    }

    /// `a+b+` — consecutive one-or-more repetitions.
    #[test]
    fn test_one_plus_catenation() {
        let p = "a+b+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "b");
        assert_matches_regex_crate(p, "ab");
        assert_matches_regex_crate(p, "aab");
        assert_matches_regex_crate(p, "abb");
        assert_matches_regex_crate(p, "aabb");
        assert_matches_regex_crate(p, "ba");
    }

    /// `(ab)+` — one-or-more of a multi-byte sequence.
    #[test]
    fn test_one_plus_group() {
        let p = "(ab)+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "ab");
        assert_matches_regex_crate(p, "abab");
        assert_matches_regex_crate(p, "ababab");
        assert_matches_regex_crate(p, "aba");
    }

    /// `(a|b)+` — one-or-more alternation.
    #[test]
    fn test_one_plus_alternate() {
        let p = "(a|b)+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "b");
        assert_matches_regex_crate(p, "ab");
        assert_matches_regex_crate(p, "ba");
        assert_matches_regex_crate(p, "aab");
        assert_matches_regex_crate(p, "c");
    }

    /// `.*a.{3}b+c` — one-or-more mixed with counting constraints.
    #[test]
    fn test_one_plus_with_counting() {
        let p = ".*a.{3}b+c";
        assert_matches_regex_crate(p, "a123bc");
        assert_matches_regex_crate(p, "a123bbc");
        assert_matches_regex_crate(p, "a123bbbc");
        assert_matches_regex_crate(p, "a123c");
        assert_matches_regex_crate(p, "xa123bc");
        assert_matches_regex_crate(p, "");
    }

    /// `(a{2,3})+` — inner repetition, outer one-or-more.
    /// The body of `+` is itself a counted repetition.
    #[test]
    fn test_repetition_inside_one_plus() {
        let p = "(a{2,3})+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
        assert_matches_regex_crate(p, "aaaa");
        assert_matches_regex_crate(p, "aaaaa");
        assert_matches_regex_crate(p, "aaaaaa");
        assert_matches_regex_crate(p, "aaaaaaa");
        assert_matches_regex_crate(p, "aaaaaaaa");
        assert_matches_regex_crate(p, "aaaaaaaaa");
        assert_matches_regex_crate(p, "b");
    }

    /// `((a|bc){1,2})+` — inner range repetition of alternation, outer `+`.
    #[test]
    fn test_range_alternation_inside_one_plus() {
        use itertools::Itertools;

        let p = "((a|bc){1,2})+";

        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "bc");
        assert_matches_regex_crate(p, "b");
        assert_matches_regex_crate(p, "c");

        // 2 through 6 atoms — exercises multiple iterations of the outer `+`
        for i in 2..=6 {
            for v in std::iter::repeat(["a", "bc"])
                .take(i)
                .map(|a| a.into_iter())
                .multi_cartesian_product()
            {
                let input = v.into_iter().collect::<String>();
                assert_matches_regex_crate(p, &input);
            }
        }
    }

    /// `(a+){2,3}` — inner one-or-more, outer counted repetition.
    #[test]
    fn test_one_plus_inside_repetition() {
        let p = "(a+){2,3}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
        assert_matches_regex_crate(p, "aaaa");
        assert_matches_regex_crate(p, "aaaaa");
        assert_matches_regex_crate(p, "aaaaaa");
        assert_matches_regex_crate(p, "aaaaaaa");
        assert_matches_regex_crate(p, "b");
    }

    /// `((a|b)+){2,4}` — inner `+` of alternation, outer counted repetition.
    #[test]
    fn test_one_plus_alternation_inside_repetition() {
        use itertools::Itertools;

        let p = "((a|b)+){2,4}";

        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "b");
        assert_matches_regex_crate(p, "c");

        for i in 2..=8 {
            for v in std::iter::repeat(["a", "b"])
                .take(i)
                .map(|a| a.into_iter())
                .multi_cartesian_product()
            {
                let input = v.into_iter().collect::<String>();
                assert_matches_regex_crate(p, &input);
            }
        }
    }

    /// `(a+b{2,3})+` — inner `+` and inner repetition side-by-side,
    /// wrapped in outer `+`.
    #[test]
    fn test_mixed_plus_and_repetition_inside_one_plus() {
        let p = "(a+b{2,3})+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "ab");
        assert_matches_regex_crate(p, "abb");
        assert_matches_regex_crate(p, "abbb");
        assert_matches_regex_crate(p, "abbbb");
        assert_matches_regex_crate(p, "aabb");
        assert_matches_regex_crate(p, "aabbb");
        assert_matches_regex_crate(p, "abbabb");
        assert_matches_regex_crate(p, "abbaabb");
        assert_matches_regex_crate(p, "abbabbbabb");
        assert_matches_regex_crate(p, "aabbaabbb");
        assert_matches_regex_crate(p, "aabbbaabbb");
    }

    // -- min=0 repetition tests ---------------------------------------------

    /// `a{0,2}` — zero to two occurrences of a single byte.
    #[test]
    fn test_min_zero_basic() {
        let p = "a{0,2}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
        assert_matches_regex_crate(p, "b");
    }

    /// `a{0,1}` — equivalent to `a?`.
    #[test]
    fn test_min_zero_max_one() {
        let p = "a{0,1}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "b");
    }

    /// `(a|bc){0,3}` — zero to three of an alternation.
    #[test]
    fn test_min_zero_alternation() {
        use itertools::Itertools;

        let p = "(a|bc){0,3}";

        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "bc");
        assert_matches_regex_crate(p, "b");

        for i in 2..=4 {
            for v in std::iter::repeat(["a", "bc"])
                .take(i)
                .map(|a| a.into_iter())
                .multi_cartesian_product()
            {
                let input = v.into_iter().collect::<String>();
                assert_matches_regex_crate(p, &input);
            }
        }
    }

    /// `a{0,}` — zero or more, lowered to `a*` (no counter overhead).
    #[test]
    fn test_min_zero_unbounded() {
        let p = "a{0,}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
        assert_matches_regex_crate(p, "aaaa");
        assert_matches_regex_crate(p, "b");
    }

    /// `(ab){0,}` — zero or more of a group, lowered to `(ab)*`.
    #[test]
    fn test_min_zero_unbounded_group() {
        let p = "(ab){0,}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "ab");
        assert_matches_regex_crate(p, "abab");
        assert_matches_regex_crate(p, "ababab");
        assert_matches_regex_crate(p, "aba");
    }

    /// `x(a{0,2})+y` — min=0 repetition nested inside `+`.
    #[test]
    fn test_min_zero_inside_one_plus() {
        let p = "x(a{0,2})+y";
        assert_matches_regex_crate(p, "xy");
        assert_matches_regex_crate(p, "xay");
        assert_matches_regex_crate(p, "xaay");
        assert_matches_regex_crate(p, "xaaay");
        assert_matches_regex_crate(p, "xaaaay");
        assert_matches_regex_crate(p, "x");
        assert_matches_regex_crate(p, "y");
        assert_matches_regex_crate(p, "");
    }

    /// `(a{0,2}){2,3}` — min=0 inner, counted outer.
    #[test]
    fn test_min_zero_inside_repetition() {
        let p = "(a{0,2}){2,3}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
        assert_matches_regex_crate(p, "aaaa");
        assert_matches_regex_crate(p, "aaaaa");
        assert_matches_regex_crate(p, "aaaaaa");
        assert_matches_regex_crate(p, "aaaaaaa");
        assert_matches_regex_crate(p, "b");
    }

    /// `(a+){0,3}` — `+` inside a min=0 counted repetition.
    #[test]
    fn test_one_plus_inside_min_zero_repetition() {
        let p = "(a+){0,3}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
        assert_matches_regex_crate(p, "aaaa");
        assert_matches_regex_crate(p, "aaaaa");
        assert_matches_regex_crate(p, "aaaaaa");
        assert_matches_regex_crate(p, "b");
    }

    /// `.{0,3}` — min=0 repetition on wildcard.
    #[test]
    fn test_min_zero_wildcard() {
        let p = ".{0,3}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "ab");
        assert_matches_regex_crate(p, "abc");
        assert_matches_regex_crate(p, "abcd");
    }

    /// `a{0,3}` — same as `a{0,3}` (the old test used `min: None`).
    #[test]
    fn test_none_min_repetition() {
        let p = "a{0,3}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "aa");
        assert_matches_regex_crate(p, "aaa");
        assert_matches_regex_crate(p, "aaaa");
    }

    // -- Byte class tests ---------------------------------------------------

    /// `[a-c]` — a small contiguous byte range (Class::Bytes).
    #[test]
    fn test_byte_class_range() {
        let p = "[a-c]";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "b");
        assert_matches_regex_crate(p, "c");
        assert_matches_regex_crate(p, "d");
        assert_matches_regex_crate(p, "ab");
    }

    /// `[a-c]+` — one-or-more of a byte class.
    #[test]
    fn test_byte_class_one_plus() {
        let p = "[a-c]+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "abc");
        assert_matches_regex_crate(p, "cba");
        assert_matches_regex_crate(p, "abcd");
        assert_matches_regex_crate(p, "d");
    }

    /// `[a-c]{2,3}` — counted repetition of a byte class.
    #[test]
    fn test_byte_class_counted() {
        let p = "[a-c]{2,3}";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "ab");
        assert_matches_regex_crate(p, "abc");
        assert_matches_regex_crate(p, "abca");
        assert_matches_regex_crate(p, "cc");
        assert_matches_regex_crate(p, "dd");
    }

    /// `[ax]` — disjoint single bytes (multi-range Class::Bytes).
    #[test]
    fn test_byte_class_disjoint() {
        let p = "[ax]";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "x");
        assert_matches_regex_crate(p, "b");
        assert_matches_regex_crate(p, "ax");
    }

    /// `[a-cx-z]+` — multiple disjoint ranges in a byte class.
    #[test]
    fn test_byte_class_multi_range() {
        let p = "[a-cx-z]+";
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
        assert_matches_regex_crate(p, "x");
        assert_matches_regex_crate(p, "axbycz");
        assert_matches_regex_crate(p, "d");
        assert_matches_regex_crate(p, "w");
        assert_matches_regex_crate(p, "abcxyz");
    }

    /// `[a-c].*[x-z]` — byte classes mixed with wildcard.
    #[test]
    fn test_byte_class_with_wildcard() {
        let p = "[a-c].*[x-z]";
        assert_matches_regex_crate(p, "ax");
        assert_matches_regex_crate(p, "a123z");
        assert_matches_regex_crate(p, "bx");
        assert_matches_regex_crate(p, "dx");
        assert_matches_regex_crate(p, "");
        assert_matches_regex_crate(p, "a");
    }
}
