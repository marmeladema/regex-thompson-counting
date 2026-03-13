//! Typed effect model for Tier 3 transition consequences.
//!
//! This module defines a structured representation for the nonlocal effects
//! of Tier 3 DFA transitions: seeds, tails, deferred assertions, and match
//! signals.  The types here replace several ad hoc side channels that were
//! previously scattered across [`Tier3OriginKind`], [`Transition`], and the
//! runtime pending vectors in [`Tier3DfaMatcher`].
//!
//! # Design
//!
//! Every semantic consequence of a transition is represented as an explicit
//! effect with three dimensions:
//!
//! - **payload** ([`EffectAtom`]): what happens (seed a counter, add a tail,
//!   signal a match, etc.)
//! - **guard** ([`EffectGuard`]): under what condition the effect is valid
//!   (unconditional, counter-break-gated, assertion-chain-gated, or both)
//! - **timing** ([`EffectTiming`]): when the effect becomes actionable (now,
//!   next byte boundary, or end-of-input only)
//!
//! Local counter stepping (`Advance` vs `Increment`) is kept separate in
//! [`TargetStep`] because the counter storage backends answer local
//! questions about individual entries.  The effect system handles the
//! *nonlocal* and *guarded* consequences.
//!
//! # Provenance
//!
//! [`BreakMask`] reserves space for break provenance on guards and pending
//! effects.  In the first implementation `0` means counter-free.  This
//! field exists so that Proposal 3-style provenance can be adopted later
//! without restructuring the effect types.
//!
//! [`super::Tier3OriginKind`]: super::Tier3OriginKind
//! [`Transition`]: super::Tier3DfaCache
//! [`Tier3DfaMatcher`]: super::Tier3DfaMatcher

use std::fmt;

use crate::{AssertEval, AssertKind, CounterIdx, Regex, State, StateIdx};

// ---------------------------------------------------------------------------
// Break mask (provenance-ready)
// ---------------------------------------------------------------------------

/// Bitmask identifying which counter breaks were required to reach a fact.
///
/// - `0` means the fact is counter-free (unconditional).
/// - Bit `i` set means counter `i` must have broken for this fact to be
///   valid.
///
/// The current Tier 3 is capped to 64 counters by bitmask width elsewhere
/// (`MAX_TIER2_COUNTERS`), so `u64` is sufficient.
pub(crate) type BreakMask = u64;

// ---------------------------------------------------------------------------
// Effect timing
// ---------------------------------------------------------------------------

/// When an effect becomes actionable relative to the current byte boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum EffectTiming {
    /// Apply immediately during the current step.
    Now,
    /// Defer until the next byte boundary (when the next byte is known).
    NextByte,
    /// Defer until end-of-input processing.
    EndOnly,
}

impl fmt::Display for EffectTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Now => write!(f, "now"),
            Self::NextByte => write!(f, "next_byte"),
            Self::EndOnly => write!(f, "end_only"),
        }
    }
}

// ---------------------------------------------------------------------------
// Assertion chain arena
// ---------------------------------------------------------------------------

/// Compact ID referencing an assertion chain in [`AssertChainArena`].
///
/// An assertion chain is an ordered sequence of NFA assertion states
/// (e.g. `\b` → `\B`) that must all pass *in order* for the guarded
/// effect to be valid.  Chains must not be flattened into independent
/// assertions — the order and completeness matter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AssertChainId(pub(crate) u32);

impl AssertChainId {
    /// Sentinel: no assertion gating.
    pub(crate) const NONE: Self = Self(u32::MAX);

    /// Return the raw index as `usize`.
    pub(crate) fn idx(self) -> usize {
        debug_assert!(self != Self::NONE, "AssertChainId::NONE used as index");
        self.0 as usize
    }
}

impl fmt::Display for AssertChainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::NONE {
            write!(f, "NONE")
        } else {
            write!(f, "chain#{}", self.0)
        }
    }
}

/// Arena of interned assertion chains.
///
/// Each chain is a `Box<[StateIdx]>` of ordered NFA assertion state indices.
/// Duplicate chains are interned so that structurally identical assertion
/// paths share a single [`AssertChainId`].
#[derive(Clone, Debug, Default)]
pub(crate) struct AssertChainArena {
    /// The chains themselves, indexed by [`AssertChainId`].
    chains: Vec<Box<[StateIdx]>>,
}

impl AssertChainArena {
    /// Create an empty arena.
    pub(crate) fn new() -> Self {
        Self { chains: Vec::new() }
    }

    /// Intern a chain of assertion state indices.
    ///
    /// If an identical chain already exists, returns its existing ID.
    /// If `asserts` is empty, returns [`AssertChainId::NONE`].
    pub(crate) fn intern(&mut self, asserts: &[StateIdx]) -> AssertChainId {
        if asserts.is_empty() {
            return AssertChainId::NONE;
        }
        // Linear scan for dedup — chain count is very small in practice.
        for (i, existing) in self.chains.iter().enumerate() {
            if existing.as_ref() == asserts {
                return AssertChainId(i as u32);
            }
        }
        let id = AssertChainId(self.chains.len() as u32);
        self.chains.push(asserts.into());
        id
    }

    /// Look up the assertion states for a chain ID.
    ///
    /// Returns an empty slice for [`AssertChainId::NONE`].
    pub(crate) fn get(&self, id: AssertChainId) -> &[StateIdx] {
        if id == AssertChainId::NONE {
            &[]
        } else {
            &self.chains[id.idx()]
        }
    }

    /// Number of interned chains (not counting NONE).
    pub(crate) fn len(&self) -> usize {
        self.chains.len()
    }
}

// ---------------------------------------------------------------------------
// Effect guard
// ---------------------------------------------------------------------------

/// Condition under which an effect is valid.
///
/// A guard is a conjunction of two orthogonal conditions:
///
/// - **`required_breaks`**: which counter breaks must have occurred.
///   `0` means counter-free (always valid with respect to breaks).
/// - **`assert_chain`**: an ordered sequence of NFA assertions that must
///   all pass at the evaluation boundary.  [`AssertChainId::NONE`] means
///   no assertion gating.
///
/// Both conditions must hold for the guard to pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EffectGuard {
    /// Which counter breaks are required.  `0` = counter-free.
    pub(crate) required_breaks: BreakMask,
    /// Assertion chain that must pass.  `NONE` = no assertions.
    pub(crate) assert_chain: AssertChainId,
}

impl EffectGuard {
    /// Unconditional guard: counter-free, no assertions.
    pub(crate) const ALWAYS: Self = Self {
        required_breaks: 0,
        assert_chain: AssertChainId::NONE,
    };

    /// Whether this guard is unconditional.
    pub(crate) fn is_always(&self) -> bool {
        self.required_breaks == 0 && self.assert_chain == AssertChainId::NONE
    }

    /// Whether this guard requires at least one counter break.
    pub(crate) fn is_break_gated(&self) -> bool {
        self.required_breaks != 0
    }

    /// Whether this guard requires an assertion chain.
    pub(crate) fn is_assert_gated(&self) -> bool {
        self.assert_chain != AssertChainId::NONE
    }
}

impl fmt::Display for EffectGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_always() {
            return write!(f, "always");
        }
        let mut parts = Vec::new();
        if self.required_breaks != 0 {
            parts.push(format!("breaks=0x{:x}", self.required_breaks));
        }
        if self.assert_chain != AssertChainId::NONE {
            parts.push(format!("asserts={}", self.assert_chain));
        }
        write!(f, "{}", parts.join("+"))
    }
}

// ---------------------------------------------------------------------------
// Effect atoms
// ---------------------------------------------------------------------------

/// A single atomic effect produced by a transition.
///
/// Atoms are intentionally small — they describe *what happens*, not
/// *when* or *under what condition*.  Timing and guards are attached at
/// the [`GuardedEffect`] or [`CompiledTargetEffects`] level.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EffectAtom {
    /// Seed a new counter instance.
    AddSeed {
        /// Which counter to seed.
        counter: CounterIdx,
        /// The consuming origin state for the new instance.
        origin: StateIdx,
        /// Initial counter value for the new instance.
        value: u32,
    },
    /// Add a post-break consuming tail state.
    AddTail {
        /// The consuming state to inject into `post_break_tails`.
        origin: StateIdx,
    },
    /// Signal an immediate match (`ever_matched = true`).
    Match,
    /// Signal a match-at-end (`match_at_end = true`, resolved at EOI or
    /// via `$ → Match` deferred assertion path).
    MatchAtEnd,
}

impl fmt::Display for EffectAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AddSeed {
                counter,
                origin,
                value,
            } => write!(f, "AddSeed(c{counter}, origin:{origin}, val={value})"),
            Self::AddTail { origin } => write!(f, "AddTail(origin:{origin})"),
            Self::Match => write!(f, "Match"),
            Self::MatchAtEnd => write!(f, "MatchAtEnd"),
        }
    }
}

// ---------------------------------------------------------------------------
// Guarded effect
// ---------------------------------------------------------------------------

/// A bundle of effect atoms sharing the same timing and guard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GuardedEffect {
    /// When the effect becomes actionable.
    pub(crate) timing: EffectTiming,
    /// Under what condition the effect is valid.
    pub(crate) guard: EffectGuard,
    /// The atoms to apply when the guard passes at the given timing.
    pub(crate) atoms: Box<[EffectAtom]>,
}

impl fmt::Display for GuardedEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}|{}]", self.timing, self.guard)?;
        for (i, atom) in self.atoms.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            } else {
                write!(f, " ")?;
            }
            write!(f, "{atom}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Local target step
// ---------------------------------------------------------------------------

/// Description of the local counter motion for a target state.
///
/// This replaces only the structural part of [`Tier3OriginKind`] that
/// describes how an active entry moves through the NFA.  The nonlocal
/// consequences (seeds, tails, matches) are in [`CompiledTargetEffects`].
///
/// [`Tier3OriginKind`]: super::Tier3OriginKind
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum TargetStep {
    /// No counter motion — the target is dead or unreachable.
    None,
    /// The entry advances without incrementing: epsilon closure from the
    /// target did not reach `CInc`.  The entry keeps its counter value
    /// and moves to the listed consuming origins.
    Advance {
        /// Consuming NFA states reachable from the target via epsilon
        /// transitions (before any `CInc`).
        new_origins: Box<[StateIdx]>,
    },
    /// The entry increments a counter: epsilon closure reached `CInc`.
    Increment {
        /// Which counter is incremented.
        counter: CounterIdx,
        /// Consuming states reachable from the target before `CInc`
        /// (the "advance" path for entries that skip the increment).
        advance_origins: Box<[StateIdx]>,
        /// Minimum counter value for this counter to break.
        min: u32,
        /// Maximum counter value for this counter (continue ceiling).
        max: u32,
        /// Consuming states reachable via the continue path after `CInc`.
        continue_origins: Box<[StateIdx]>,
    },
}

impl fmt::Display for TargetStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Advance { new_origins } => {
                write!(f, "Advance → [")?;
                for (i, o) in new_origins.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{o}")?;
                }
                write!(f, "]")
            }
            Self::Increment {
                counter,
                advance_origins,
                min,
                max,
                continue_origins,
            } => {
                write!(f, "Increment(c{counter}, {{{min},{max}}})")?;
                write!(f, " adv=[")?;
                for (i, o) in advance_origins.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{o}")?;
                }
                write!(f, "] cont=[")?;
                for (i, o) in continue_origins.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{o}")?;
                }
                write!(f, "]")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compiled target effects
// ---------------------------------------------------------------------------

/// Complete effect description for one post-consumption NFA target.
///
/// Replaces the old "action + scattered booleans/lists" representation.
///
/// # Semantics
///
/// - **`step`**: local counter motion (Advance / Increment / None).
/// - **`immediate`**: unconditional `Now` effects from the target (e.g. a
///   direct `Match` or counter-free `MatchAtEnd`).
/// - **`on_break`**: `Now` effects valid only when an incrementing instance
///   actually breaks (e.g. break-path `Match`, `MatchAtEnd`, tails, seeds).
/// - **`guarded`**: effects with assertion-chain guards and/or deferred
///   timing (`NextByte`, `EndOnly`) that cannot be resolved immediately.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CompiledTargetEffects {
    /// Local counter motion for this target.
    pub(crate) step: TargetStep,
    /// Unconditional immediate effects (counter-free, no assertions).
    pub(crate) immediate: Box<[EffectAtom]>,
    /// Immediate effects gated on a counter break occurring.
    pub(crate) on_break: Box<[EffectAtom]>,
    /// Deferred or assertion-gated effects.
    pub(crate) guarded: Box<[GuardedEffect]>,
}

impl fmt::Display for CompiledTargetEffects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "step={}", self.step)?;
        if !self.immediate.is_empty() {
            write!(f, " immediate=[")?;
            for (i, a) in self.immediate.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{a}")?;
            }
            write!(f, "]")?;
        }
        if !self.on_break.is_empty() {
            write!(f, " on_break=[")?;
            for (i, a) in self.on_break.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{a}")?;
            }
            write!(f, "]")?;
        }
        if !self.guarded.is_empty() {
            write!(f, " guarded=[")?;
            for (i, g) in self.guarded.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{g}")?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pending effect (runtime)
// ---------------------------------------------------------------------------

/// A deferred effect waiting to be evaluated at the next byte boundary
/// or at end-of-input.
///
/// `prev_was_word` captures the word-boundary context at the time the
/// effect was scheduled, because deferred assertion evaluation needs the
/// boundary context from the *scheduling* byte, not the *evaluation* byte.
#[derive(Clone, Debug)]
pub(crate) struct PendingEffect {
    /// When this effect should be evaluated.
    pub(crate) timing: EffectTiming,
    /// Condition for this effect to be valid.
    pub(crate) guard: EffectGuard,
    /// The atoms to apply when the guard passes.
    pub(crate) atoms: Box<[EffectAtom]>,
    /// Word-boundary context at the time the effect was scheduled.
    pub(crate) prev_was_word: bool,
}

impl fmt::Display for PendingEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}|{}|pw={}]",
            self.timing, self.guard, self.prev_was_word
        )?;
        for (i, atom) in self.atoms.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            } else {
                write!(f, " ")?;
            }
            write!(f, "{atom}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Effect resolution results
// ---------------------------------------------------------------------------

/// Actions produced by resolving pending effects at a boundary.
///
/// The caller (e.g. `chunk()` or `finish()`) reads these and applies
/// them to the matcher state.  This decouples the effect evaluation
/// from the matcher's internal state layout.
#[derive(Debug, Default)]
pub(crate) struct ResolvedActions {
    /// Seeds to apply: `(counter, origin, value)`.
    pub(crate) seeds: Vec<(CounterIdx, StateIdx, u32)>,
    /// Tail states to inject into `post_break_tails`.
    pub(crate) tails: Vec<StateIdx>,
    /// Whether an immediate match was signaled.
    pub(crate) set_match: bool,
    /// Whether a match-at-end was signaled.
    pub(crate) set_match_at_end: bool,
}

impl ResolvedActions {
    /// Whether any action was produced.
    pub(crate) fn is_empty(&self) -> bool {
        self.seeds.is_empty() && self.tails.is_empty() && !self.set_match && !self.set_match_at_end
    }
}

// ---------------------------------------------------------------------------
// Effect resolution: evaluate pending effects at a boundary
// ---------------------------------------------------------------------------

/// Evaluate an assertion chain against the current boundary context.
///
/// Returns `true` if all assertions in the chain pass.  An empty chain
/// (i.e. `AssertChainId::NONE`) always passes.
///
/// # Arguments
///
/// - `chain_id`: the assertion chain to evaluate.
/// - `arena`: the arena containing interned chains.
/// - `at_end`: whether we are at end-of-input.
/// - `prev`: the byte *before* the boundary (or `None` at start-of-input).
/// - `next`: the byte *after* the boundary (or `None` at end-of-input).
/// - `regex`: the compiled regex (for NFA state access and downstream
///   reachability checks).
pub(crate) fn eval_assert_chain(
    chain_id: AssertChainId,
    arena: &AssertChainArena,
    at_end: bool,
    prev: Option<u8>,
    next: Option<u8>,
    regex: &Regex,
) -> bool {
    if chain_id == AssertChainId::NONE {
        return true;
    }
    let chain = arena.get(chain_id);
    for &assert_idx in chain {
        let State::Assert { kind, out } = regex.states.0[assert_idx] else {
            // Not an Assert state — should not happen, but be defensive.
            return false;
        };
        match kind.eval(false, at_end, prev, next) {
            AssertEval::Pass => {
                // Check if the downstream path from this assertion's `out`
                // can still reach a match.  This handles chained assertions
                // (e.g. `\b → \B → $ → Match`).
                if at_end {
                    if !super::DfaState::can_reach_match_at_end(out, prev, regex) {
                        return false;
                    }
                } else {
                    // Mid-input: check if can_reach_match_mid from the
                    // assertion's out.  We use the Tier3DfaMatcher static
                    // method for this.
                    if !super::Tier3DfaMatcher::can_reach_match_mid(out, prev, next, regex) {
                        return false;
                    }
                }
            }
            AssertEval::Fail => return false,
            AssertEval::Defer => {
                // The assertion needs the next byte — can't resolve yet.
                // This shouldn't happen for NextByte effects (we have the
                // next byte), but for EndOnly it means we treat it as
                // failure (conservative).
                return false;
            }
        }
    }
    true
}

/// Resolve a list of pending effects against the current boundary.
///
/// Evaluates each pending effect's guard (break mask + assertion chain)
/// and, if the guard passes, collects the resulting actions.
///
/// # Arguments
///
/// - `pending`: the pending effects to evaluate (drained by the caller).
/// - `timing_filter`: only evaluate effects with this timing.
/// - `arena`: assertion chain arena.
/// - `at_end`: whether we are at end-of-input.
/// - `next`: the byte after the boundary (for mid-input resolution).
/// - `regex`: the compiled regex.
///
/// Effects whose guards fail are silently dropped (they are consumed).
pub(crate) fn resolve_pending(
    pending: &[PendingEffect],
    timing_filter: EffectTiming,
    arena: &AssertChainArena,
    at_end: bool,
    next: Option<u8>,
    regex: &Regex,
) -> ResolvedActions {
    let mut actions = ResolvedActions::default();

    for pe in pending {
        if pe.timing != timing_filter {
            continue;
        }

        // Evaluate the assertion chain with the captured boundary context.
        // `prev` is reconstructed from `prev_was_word`:
        // - `Some(b'a')` if prev_was_word (word character representative)
        // - `Some(b' ')` if not (non-word character representative)
        let prev = if pe.prev_was_word {
            Some(b'a')
        } else {
            Some(b' ')
        };

        if !eval_assert_chain(pe.guard.assert_chain, arena, at_end, prev, next, regex) {
            continue;
        }

        // Guard passed — apply atoms.
        for atom in pe.atoms.iter() {
            match atom {
                EffectAtom::AddSeed {
                    counter,
                    origin,
                    value,
                } => {
                    actions.seeds.push((*counter, *origin, *value));
                }
                EffectAtom::AddTail { origin } => {
                    if !actions.tails.contains(origin) {
                        actions.tails.push(*origin);
                    }
                }
                EffectAtom::Match => {
                    actions.set_match = true;
                }
                EffectAtom::MatchAtEnd => {
                    actions.set_match_at_end = true;
                }
            }
        }
    }

    actions
}

// ---------------------------------------------------------------------------
// Debug assertions
// ---------------------------------------------------------------------------

/// Validate that no effect atom references a non-consuming origin when it
/// should reference a consuming one.
///
/// This is a debug-only check intended to be called after effect compilation.
/// The `is_consuming` closure should return `true` for states that are
/// consuming (Byte, ByteTable, etc.).
#[cfg(debug_assertions)]
pub(crate) fn debug_assert_origins_consuming(
    effects: &CompiledTargetEffects,
    is_consuming: impl Fn(StateIdx) -> bool,
) {
    // Check immediate atoms.
    for atom in effects.immediate.iter() {
        assert_atom_origins_consuming(atom, &is_consuming);
    }
    // Check on_break atoms.
    for atom in effects.on_break.iter() {
        assert_atom_origins_consuming(atom, &is_consuming);
    }
    // Check guarded atoms.
    for ge in effects.guarded.iter() {
        for atom in ge.atoms.iter() {
            assert_atom_origins_consuming(atom, &is_consuming);
        }
    }
}

#[cfg(debug_assertions)]
fn assert_atom_origins_consuming(atom: &EffectAtom, is_consuming: &impl Fn(StateIdx) -> bool) {
    match atom {
        EffectAtom::AddSeed { origin, .. } => {
            debug_assert!(
                is_consuming(*origin),
                "AddSeed origin {} is not a consuming state",
                origin
            );
        }
        EffectAtom::AddTail { origin } => {
            debug_assert!(
                is_consuming(*origin),
                "AddTail origin {} is not a consuming state",
                origin
            );
        }
        EffectAtom::Match | EffectAtom::MatchAtEnd => {}
    }
}

// ---------------------------------------------------------------------------
// Shadow compilation: legacy → effects
// ---------------------------------------------------------------------------

/// Compile [`CompiledTargetEffects`] for a single target from its legacy
/// [`Tier3OriginKind`] and the global break seed table.
///
/// This is the "shadow compiler" for Phase 2: it reads the already-computed
/// legacy analysis and translates it into the effect representation.  The
/// result must be semantically equivalent to the legacy data.
///
/// # Arguments
///
/// - `target_idx`: NFA state index of the post-consumption target.
/// - `origin_kind`: the legacy structural action for this target.
/// - `break_seeds`: global break seed table from `Tier3Analysis`.
/// - `arena`: assertion chain arena (mutably borrowed for interning).
///
/// [`Tier3OriginKind`]: super::Tier3OriginKind
pub(crate) fn compile_target_effects(
    _target_idx: StateIdx,
    origin_kind: &super::Tier3OriginKind,
    break_seeds: &[super::Tier3BreakSeed],
    arena: &mut AssertChainArena,
) -> CompiledTargetEffects {
    match origin_kind {
        super::Tier3OriginKind::Advance {
            new_origins,
            is_match_at_end,
            is_match,
        } => {
            // Advance: no counter increment involved.
            let step = TargetStep::Advance {
                new_origins: new_origins.clone(),
            };

            // Immediate effects: direct match and/or match-at-end.
            let mut immediate = Vec::new();
            if *is_match {
                immediate.push(EffectAtom::Match);
            }
            if *is_match_at_end {
                immediate.push(EffectAtom::MatchAtEnd);
            }

            CompiledTargetEffects {
                step,
                immediate: immediate.into_boxed_slice(),
                on_break: Box::new([]),
                guarded: Box::new([]),
            }
        }

        super::Tier3OriginKind::Increment {
            counter,
            advance_origins,
            min,
            max,
            continue_origins,
            break_is_match,
            break_is_match_at_end,
            break_deferred_asserts,
            break_consuming_states: _,
            break_consuming_pure,
            break_consuming_deferred,
        } => {
            let step = TargetStep::Increment {
                counter: *counter,
                advance_origins: advance_origins.clone(),
                min: *min as u32,
                max: *max as u32,
                continue_origins: continue_origins.clone(),
            };

            // --- on_break: immediate effects gated on counter break ---
            let mut on_break = Vec::new();

            // Break-path direct match.
            if *break_is_match {
                on_break.push(EffectAtom::Match);
            }

            // Break-path match-at-end (only when no deferred asserts gate it).
            if *break_is_match_at_end && break_deferred_asserts.is_empty() {
                on_break.push(EffectAtom::MatchAtEnd);
            }

            // Pure tails: consuming states reachable from break path
            // WITHOUT deferred assertions — deposited immediately on break.
            for &tail in break_consuming_pure.iter() {
                on_break.push(EffectAtom::AddTail { origin: tail });
            }

            // Break seeds triggered by THIS counter's break.
            for bs in break_seeds.iter() {
                if bs.trigger == *counter && bs.deferred_asserts.is_empty() {
                    on_break.push(EffectAtom::AddSeed {
                        counter: bs.counter,
                        origin: bs.origin,
                        value: 0,
                    });
                }
            }

            // --- guarded: deferred / assertion-gated effects ---
            let mut guarded = Vec::new();

            // Break-path deferred assertions.
            //
            // When break_deferred_asserts is non-empty, the break path
            // includes assertions (e.g. `\b`, `\B`) that gate access to
            // Match or $ → Match.  The runtime stores these in
            // `verified_deferred_asserts` and evaluates them via
            // `can_reach_match_at_end()` at the next byte boundary and
            // at end-of-input.
            //
            // The effect representation captures this as a guarded
            // MatchAtEnd — the assertion chain determines whether the
            // match fires.  This covers three legacy cases:
            // 1. break_is_match_at_end=true with deferred asserts
            //    (e.g. `\b → $ → Match`)
            // 2. break_is_match=false, break_is_match_at_end=false,
            //    but deferred asserts lead to Match (e.g. `\b → Match`)
            // 3. Both Match and $ → Match behind the same chain
            if !break_deferred_asserts.is_empty() {
                let chain_id = arena.intern(break_deferred_asserts);
                guarded.push(GuardedEffect {
                    timing: EffectTiming::NextByte,
                    guard: EffectGuard {
                        required_breaks: 1u64 << counter.idx(),
                        assert_chain: chain_id,
                    },
                    atoms: vec![EffectAtom::MatchAtEnd].into_boxed_slice(),
                });
                // Also an EndOnly variant for finish().
                guarded.push(GuardedEffect {
                    timing: EffectTiming::EndOnly,
                    guard: EffectGuard {
                        required_breaks: 1u64 << counter.idx(),
                        assert_chain: chain_id,
                    },
                    atoms: vec![EffectAtom::MatchAtEnd].into_boxed_slice(),
                });
            }

            // Per-tail deferred assertions: each tail has its own chain.
            for &(tail, ref per_tail_asserts) in break_consuming_deferred.iter() {
                if per_tail_asserts.is_empty() {
                    // Pure tail — already handled above.
                    continue;
                }
                let chain_id = arena.intern(per_tail_asserts);
                guarded.push(GuardedEffect {
                    timing: EffectTiming::NextByte,
                    guard: EffectGuard {
                        required_breaks: 1u64 << counter.idx(),
                        assert_chain: chain_id,
                    },
                    atoms: vec![EffectAtom::AddTail { origin: tail }].into_boxed_slice(),
                });
            }

            // Break seeds with deferred assertions.
            for bs in break_seeds.iter() {
                if bs.trigger == *counter && !bs.deferred_asserts.is_empty() {
                    let chain_id = arena.intern(&bs.deferred_asserts);
                    guarded.push(GuardedEffect {
                        timing: EffectTiming::NextByte,
                        guard: EffectGuard {
                            required_breaks: 1u64 << counter.idx(),
                            assert_chain: chain_id,
                        },
                        atoms: vec![EffectAtom::AddSeed {
                            counter: bs.counter,
                            origin: bs.origin,
                            value: 0,
                        }]
                        .into_boxed_slice(),
                    });
                }
            }

            CompiledTargetEffects {
                step,
                immediate: Box::new([]),
                on_break: on_break.into_boxed_slice(),
                guarded: guarded.into_boxed_slice(),
            }
        }
    }
}

/// Shadow-compile all target effects for a [`Tier3Analysis`].
///
/// Iterates over all targets in the analysis and compiles a
/// `CompiledTargetEffects` for each non-`None` target.  Returns the
/// per-target effect array and the populated assertion chain arena.
pub(crate) fn compile_all_target_effects(
    analysis: &super::Tier3Analysis,
) -> (Box<[Option<CompiledTargetEffects>]>, AssertChainArena) {
    let mut arena = AssertChainArena::new();
    let effects: Vec<Option<CompiledTargetEffects>> = analysis
        .targets
        .iter()
        .enumerate()
        .map(|(i, target)| {
            target.as_ref().map(|kind| {
                compile_target_effects(StateIdx(i as u32), kind, &analysis.break_seeds, &mut arena)
            })
        })
        .collect();
    (effects.into_boxed_slice(), arena)
}

// ---------------------------------------------------------------------------
// Shadow lowering: effects → legacy categories (Phase 3)
// ---------------------------------------------------------------------------

/// Legacy facts extracted from a [`CompiledTargetEffects`].
///
/// This is a test-only / debug-only structure that "lowers" the typed
/// effects back into the same categories the legacy analysis uses.
/// Comparing these against the original legacy data proves that the
/// effect compiler captured the same information.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LoweredLegacyFacts {
    // -- From Advance targets --
    /// True if there is an immediate `EffectAtom::Match`.
    pub(crate) is_match: bool,
    /// True if there is an immediate `EffectAtom::MatchAtEnd`.
    pub(crate) is_match_at_end: bool,

    // -- From Increment targets --
    /// True if on_break contains `EffectAtom::Match`.
    pub(crate) break_is_match: bool,
    /// True if on_break contains `EffectAtom::MatchAtEnd`.
    pub(crate) break_is_match_at_end: bool,
    /// Pure tails from on_break `EffectAtom::AddTail`.
    pub(crate) break_consuming_pure: Vec<StateIdx>,
    /// Break seeds (ungated) from on_break `EffectAtom::AddSeed`.
    pub(crate) break_seeds_ungated: Vec<(CounterIdx, StateIdx, u32)>,
    /// Whether there are deferred assertion chains on break path.
    pub(crate) has_break_deferred_asserts: bool,
    /// Per-tail deferred assertion entries from guarded `AddTail` effects.
    pub(crate) break_consuming_deferred: Vec<(StateIdx, AssertChainId)>,
    /// Break seeds gated by deferred assertions.
    pub(crate) break_seeds_gated: Vec<(CounterIdx, StateIdx, u32, AssertChainId)>,
}

/// Lower a [`CompiledTargetEffects`] back into legacy-compatible facts.
pub(crate) fn lower_to_legacy(effects: &CompiledTargetEffects) -> LoweredLegacyFacts {
    let mut facts = LoweredLegacyFacts {
        is_match: false,
        is_match_at_end: false,
        break_is_match: false,
        break_is_match_at_end: false,
        break_consuming_pure: Vec::new(),
        break_seeds_ungated: Vec::new(),
        has_break_deferred_asserts: false,
        break_consuming_deferred: Vec::new(),
        break_seeds_gated: Vec::new(),
    };

    // Immediate effects (counter-free).
    for atom in effects.immediate.iter() {
        match atom {
            EffectAtom::Match => facts.is_match = true,
            EffectAtom::MatchAtEnd => facts.is_match_at_end = true,
            _ => {}
        }
    }

    // On-break effects (break-gated, no assertion chain).
    for atom in effects.on_break.iter() {
        match atom {
            EffectAtom::Match => facts.break_is_match = true,
            EffectAtom::MatchAtEnd => facts.break_is_match_at_end = true,
            EffectAtom::AddTail { origin } => {
                facts.break_consuming_pure.push(*origin);
            }
            EffectAtom::AddSeed {
                counter,
                origin,
                value,
            } => {
                facts.break_seeds_ungated.push((*counter, *origin, *value));
            }
        }
    }

    // Guarded effects.
    for ge in effects.guarded.iter() {
        if ge.guard.is_assert_gated() {
            facts.has_break_deferred_asserts = true;
        }

        for atom in ge.atoms.iter() {
            match atom {
                EffectAtom::MatchAtEnd if ge.guard.is_assert_gated() => {
                    // Break-path match-at-end behind deferred assertions.
                    // Already captured by has_break_deferred_asserts.
                }
                EffectAtom::AddTail { origin } if ge.guard.is_assert_gated() => {
                    facts
                        .break_consuming_deferred
                        .push((*origin, ge.guard.assert_chain));
                }
                EffectAtom::AddSeed {
                    counter,
                    origin,
                    value,
                } if ge.guard.is_assert_gated() => {
                    facts.break_seeds_gated.push((
                        *counter,
                        *origin,
                        *value,
                        ge.guard.assert_chain,
                    ));
                }
                _ => {}
            }
        }
    }

    // Sort for stable comparison.
    facts.break_consuming_pure.sort_unstable_by_key(|s| s.0);
    facts
        .break_seeds_ungated
        .sort_unstable_by_key(|s| (s.0.idx(), s.1 .0, s.2));
    facts
        .break_consuming_deferred
        .sort_unstable_by_key(|s| s.0 .0);
    facts
        .break_seeds_gated
        .sort_unstable_by_key(|s| (s.0.idx(), s.1 .0, s.2));

    facts
}

/// Validate that shadow-compiled effects agree with legacy analysis for
/// every target.
///
/// This is the core Phase 3 invariant check.  Call it from tests after
/// `compute_tier3_analysis()` populates both legacy and effect data.
///
/// Panics with a descriptive message if any disagreement is found.
pub(crate) fn validate_effects_vs_legacy(analysis: &super::Tier3Analysis) {
    for (i, (legacy, effect)) in analysis
        .targets
        .iter()
        .zip(analysis.target_effects.iter())
        .enumerate()
    {
        match (legacy, effect) {
            (None, None) => {}
            (Some(_), None) => {
                panic!("state {i}: legacy has target but effect is None");
            }
            (None, Some(_)) => {
                panic!("state {i}: effect has target but legacy is None");
            }
            (Some(kind), Some(eff)) => {
                validate_one_target(i, kind, eff, analysis);
            }
        }
    }
}

/// Validate one target's effects against its legacy `Tier3OriginKind`.
fn validate_one_target(
    state_idx: usize,
    kind: &super::Tier3OriginKind,
    eff: &CompiledTargetEffects,
    analysis: &super::Tier3Analysis,
) {
    let lowered = lower_to_legacy(eff);

    match kind {
        super::Tier3OriginKind::Advance {
            new_origins,
            is_match_at_end,
            is_match,
        } => {
            // Step must be Advance with same origins.
            match &eff.step {
                TargetStep::Advance {
                    new_origins: eff_origins,
                } => {
                    assert_eq!(
                        new_origins.as_ref(),
                        eff_origins.as_ref(),
                        "state {state_idx}: Advance new_origins mismatch"
                    );
                }
                other => panic!("state {state_idx}: expected Advance step, got {other}"),
            }

            assert_eq!(
                *is_match, lowered.is_match,
                "state {state_idx}: Advance is_match mismatch"
            );
            assert_eq!(
                *is_match_at_end, lowered.is_match_at_end,
                "state {state_idx}: Advance is_match_at_end mismatch"
            );
        }

        super::Tier3OriginKind::Increment {
            counter,
            advance_origins,
            min,
            max,
            continue_origins,
            break_is_match,
            break_is_match_at_end: _,
            break_deferred_asserts,
            break_consuming_states: _,
            break_consuming_pure,
            break_consuming_deferred,
        } => {
            // Step must be Increment with matching fields.
            match &eff.step {
                TargetStep::Increment {
                    counter: eff_counter,
                    advance_origins: eff_adv,
                    min: eff_min,
                    max: eff_max,
                    continue_origins: eff_cont,
                } => {
                    assert_eq!(
                        counter, eff_counter,
                        "state {state_idx}: Increment counter mismatch"
                    );
                    assert_eq!(
                        advance_origins.as_ref(),
                        eff_adv.as_ref(),
                        "state {state_idx}: Increment advance_origins mismatch"
                    );
                    assert_eq!(
                        *min as u32, *eff_min,
                        "state {state_idx}: Increment min mismatch"
                    );
                    assert_eq!(
                        *max as u32, *eff_max,
                        "state {state_idx}: Increment max mismatch"
                    );
                    assert_eq!(
                        continue_origins.as_ref(),
                        eff_cont.as_ref(),
                        "state {state_idx}: Increment continue_origins mismatch"
                    );
                }
                other => panic!("state {state_idx}: expected Increment step, got {other}"),
            }

            // Break-path direct match.
            assert_eq!(
                *break_is_match, lowered.break_is_match,
                "state {state_idx}: break_is_match mismatch"
            );

            // Break deferred asserts presence.
            assert_eq!(
                !break_deferred_asserts.is_empty(),
                lowered.has_break_deferred_asserts,
                "state {state_idx}: break_deferred_asserts presence mismatch"
            );

            // Pure tails.
            let mut expected_pure: Vec<StateIdx> = break_consuming_pure.to_vec();
            expected_pure.sort_unstable_by_key(|s| s.0);
            assert_eq!(
                expected_pure, lowered.break_consuming_pure,
                "state {state_idx}: break_consuming_pure mismatch"
            );

            // Per-tail deferred assertions: check that each deferred tail
            // appears in the lowered data with an assertion chain.
            let deferred_only: Vec<StateIdx> = break_consuming_deferred
                .iter()
                .filter(|(_, asserts)| !asserts.is_empty())
                .map(|(s, _)| *s)
                .collect();
            let mut lowered_deferred_tails: Vec<StateIdx> = lowered
                .break_consuming_deferred
                .iter()
                .map(|(s, _)| *s)
                .collect();
            lowered_deferred_tails.sort_unstable_by_key(|s| s.0);
            let mut expected_deferred: Vec<StateIdx> = deferred_only;
            expected_deferred.sort_unstable_by_key(|s| s.0);
            assert_eq!(
                expected_deferred, lowered_deferred_tails,
                "state {state_idx}: break_consuming_deferred tails mismatch"
            );

            // Break seeds: check ungated seeds match legacy break_seeds
            // for this counter.
            let expected_ungated: Vec<(CounterIdx, StateIdx, u32)> = analysis
                .break_seeds
                .iter()
                .filter(|bs| bs.trigger == *counter && bs.deferred_asserts.is_empty())
                .map(|bs| (bs.counter, bs.origin, 0u32))
                .collect();
            assert_eq!(
                expected_ungated, lowered.break_seeds_ungated,
                "state {state_idx}: ungated break seeds mismatch"
            );

            // Gated seeds: check that the right number exist.
            let expected_gated_count = analysis
                .break_seeds
                .iter()
                .filter(|bs| bs.trigger == *counter && !bs.deferred_asserts.is_empty())
                .count();
            assert_eq!(
                expected_gated_count,
                lowered.break_seeds_gated.len(),
                "state {state_idx}: gated break seeds count mismatch"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_chain_arena_empty_returns_none() {
        let mut arena = AssertChainArena::new();
        let id = arena.intern(&[]);
        assert_eq!(id, AssertChainId::NONE);
        assert_eq!(arena.get(id), &[]);
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_assert_chain_arena_intern_dedup() {
        let mut arena = AssertChainArena::new();
        let s4 = StateIdx(4);
        let s7 = StateIdx(7);

        let id1 = arena.intern(&[s4, s7]);
        let id2 = arena.intern(&[s4, s7]);
        assert_eq!(id1, id2, "identical chains must share the same ID");

        let id3 = arena.intern(&[s7, s4]);
        assert_ne!(id1, id3, "different order → different chain");

        let id4 = arena.intern(&[s4]);
        assert_ne!(id1, id4, "different length → different chain");

        assert_eq!(arena.len(), 3);
    }

    #[test]
    fn test_assert_chain_arena_get() {
        let mut arena = AssertChainArena::new();
        let s4 = StateIdx(4);
        let s7 = StateIdx(7);

        let id = arena.intern(&[s4, s7]);
        assert_eq!(arena.get(id), &[s4, s7]);
    }

    #[test]
    fn test_effect_guard_always() {
        let g = EffectGuard::ALWAYS;
        assert!(g.is_always());
        assert!(!g.is_break_gated());
        assert!(!g.is_assert_gated());
    }

    #[test]
    fn test_effect_guard_break_gated() {
        let g = EffectGuard {
            required_breaks: 1,
            assert_chain: AssertChainId::NONE,
        };
        assert!(!g.is_always());
        assert!(g.is_break_gated());
        assert!(!g.is_assert_gated());
    }

    #[test]
    fn test_effect_guard_assert_gated() {
        let g = EffectGuard {
            required_breaks: 0,
            assert_chain: AssertChainId(0),
        };
        assert!(!g.is_always());
        assert!(!g.is_break_gated());
        assert!(g.is_assert_gated());
    }

    #[test]
    fn test_effect_guard_both_gated() {
        let g = EffectGuard {
            required_breaks: 3,
            assert_chain: AssertChainId(1),
        };
        assert!(!g.is_always());
        assert!(g.is_break_gated());
        assert!(g.is_assert_gated());
    }

    #[test]
    fn test_effect_timing_display() {
        assert_eq!(format!("{}", EffectTiming::Now), "now");
        assert_eq!(format!("{}", EffectTiming::NextByte), "next_byte");
        assert_eq!(format!("{}", EffectTiming::EndOnly), "end_only");
    }

    #[test]
    fn test_effect_atom_display() {
        let seed = EffectAtom::AddSeed {
            counter: CounterIdx(0),
            origin: StateIdx(5),
            value: 1,
        };
        assert_eq!(format!("{seed}"), "AddSeed(c0, origin:5, val=1)");

        let tail = EffectAtom::AddTail {
            origin: StateIdx(3),
        };
        assert_eq!(format!("{tail}"), "AddTail(origin:3)");

        assert_eq!(format!("{}", EffectAtom::Match), "Match");
        assert_eq!(format!("{}", EffectAtom::MatchAtEnd), "MatchAtEnd");
    }

    #[test]
    fn test_guarded_effect_display() {
        let ge = GuardedEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard {
                required_breaks: 1,
                assert_chain: AssertChainId(0),
            },
            atoms: vec![EffectAtom::Match].into_boxed_slice(),
        };
        assert_eq!(
            format!("{ge}"),
            "[next_byte|breaks=0x1+asserts=chain#0] Match"
        );
    }

    #[test]
    fn test_compiled_target_effects_display() {
        let cte = CompiledTargetEffects {
            step: TargetStep::Advance {
                new_origins: vec![StateIdx(1), StateIdx(2)].into_boxed_slice(),
            },
            immediate: vec![EffectAtom::MatchAtEnd].into_boxed_slice(),
            on_break: vec![].into_boxed_slice(),
            guarded: vec![].into_boxed_slice(),
        };
        let s = format!("{cte}");
        assert!(s.contains("Advance → [1, 2]"));
        assert!(s.contains("MatchAtEnd"));
    }

    #[test]
    fn test_pending_effect_display() {
        let pe = PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard::ALWAYS,
            atoms: vec![EffectAtom::AddTail {
                origin: StateIdx(3),
            }]
            .into_boxed_slice(),
            prev_was_word: true,
        };
        let s = format!("{pe}");
        assert!(s.contains("next_byte"));
        assert!(s.contains("always"));
        assert!(s.contains("pw=true"));
        assert!(s.contains("AddTail(origin:3)"));
    }

    #[test]
    fn test_target_step_display() {
        assert_eq!(format!("{}", TargetStep::None), "None");

        let adv = TargetStep::Advance {
            new_origins: vec![StateIdx(1)].into_boxed_slice(),
        };
        assert_eq!(format!("{adv}"), "Advance → [1]");

        let inc = TargetStep::Increment {
            counter: CounterIdx(0),
            advance_origins: vec![StateIdx(1)].into_boxed_slice(),
            min: 2,
            max: 5,
            continue_origins: vec![StateIdx(3)].into_boxed_slice(),
        };
        let s = format!("{inc}");
        assert!(s.contains("Increment(c0, {2,5})"));
        assert!(s.contains("adv=[1]"));
        assert!(s.contains("cont=[3]"));
    }

    // -----------------------------------------------------------------------
    // Phase 3: Shadow lowering validation tests
    // -----------------------------------------------------------------------

    /// Helper: compile a pattern with unroll_limit=0 and validate that
    /// the shadow-compiled effects agree with the legacy analysis.
    fn validate_pattern(pattern: &str) {
        use crate::RegexBuilder;
        let hir = regex_syntax::ParserBuilder::new()
            .utf8(false)
            .unicode(false)
            .build()
            .parse(pattern)
            .unwrap_or_else(|e| panic!("failed to parse '{pattern}': {e}"));
        let mut builder = RegexBuilder::default();
        builder.max_unroll_states(0);
        let regex = builder
            .build(&hir)
            .unwrap_or_else(|e| panic!("failed to compile '{pattern}': {e}"));
        if let Some(ref analysis) = regex.tier3_analysis {
            validate_effects_vs_legacy(analysis);
        }
        // If no tier3_analysis, the pattern isn't tier-3-eligible — skip.
    }

    #[test]
    fn test_shadow_validate_simple_counter() {
        validate_pattern("a{2,4}");
    }

    #[test]
    fn test_shadow_validate_counter_with_end_anchor() {
        validate_pattern("a{2,4}$");
    }

    #[test]
    fn test_shadow_validate_word_boundary_counter() {
        validate_pattern(r"\b\w{2,4}\b");
    }

    #[test]
    fn test_shadow_validate_word_boundary_class() {
        validate_pattern(r"\b[a-z]{2,3}\b");
    }

    #[test]
    fn test_shadow_validate_alternation_counter() {
        validate_pattern("(a|bb){2,3}c");
    }

    #[test]
    fn test_shadow_validate_class_counter_with_tail() {
        validate_pattern("[ab]{2,3}c");
    }

    #[test]
    fn test_shadow_validate_dot_counter() {
        validate_pattern(".{2,5}x");
    }

    #[test]
    fn test_shadow_validate_multi_byte_body() {
        validate_pattern("(ab){2,4}");
    }

    #[test]
    fn test_shadow_validate_counter_with_prefix() {
        validate_pattern("x.{2,3}y");
    }

    #[test]
    fn test_shadow_validate_complex_body() {
        validate_pattern("([a-z][0-9]){2,3}!");
    }
}
