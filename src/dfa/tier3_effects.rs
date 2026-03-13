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
//! - **timing** ([`EffectTiming`]): when the effect becomes actionable
//!   (next byte boundary or end-of-input only)
//!
//! Local counter stepping (`Advance` vs `Increment`) is kept separate in
//! [`TargetStep`] because the counter storage backends answer local
//! questions about individual entries.  The effect system handles the
//! *nonlocal* and *guarded* consequences.
//!
//! # Provenance (future)
//!
//! [`BreakMask`] is a type alias reserved for future break-provenance
//! tracking.  It is NOT currently used at runtime.
//!
//! An earlier version stored `required_breaks: BreakMask` on [`EffectGuard`]
//! to record which counter breaks were necessary to reach a given effect.
//! That field was removed because:
//!
//! 1. It was **never checked** during effect resolution — `resolve_pending()`
//!    evaluated only the assertion chain, not the break mask.
//! 2. All deposit sites already guard effect creation behind an explicit
//!    break-condition check (`can_break(entry, min)` or `value >= min`), so
//!    the mask was always trivially satisfied by construction.
//! 3. Carrying an unenforced guard field creates a false sense of safety and
//!    risks semantic drift (Bug 51 was an instance of this broader pattern).
//!
//! If Proposal 3 (provenance-aware state) is later adopted, reintroduce
//! `required_breaks` on `EffectGuard` with the following enforcement pattern:
//!
//! ```text
//! // In resolve_pending():
//! fn resolve_pending(
//!     effects: &[PendingEffect],
//!     satisfied_breaks: BreakMask,   // ← new parameter
//!     ...
//! ) {
//!     for pe in effects {
//!         if (pe.guard.required_breaks & satisfied_breaks)
//!             != pe.guard.required_breaks
//!         {
//!             continue; // break condition not met — skip this effect
//!         }
//!         // ... evaluate assert_chain as today ...
//!     }
//! }
//!
//! // At call sites: accumulate a break mask during step_slow by
//! // OR-ing in `1u64 << counter.idx()` whenever a counter breaks,
//! // then pass it to resolve_pending() on the next byte.
//! ```
//!
//! [`super::Tier3OriginKind`]: super::Tier3OriginKind
//! [`Transition`]: super::Tier3DfaCache
//! [`Tier3DfaMatcher`]: super::Tier3DfaMatcher

use std::fmt;

use crate::{AssertEval, CounterIdx, Regex, State, StateIdx};

// ---------------------------------------------------------------------------
// Break mask (reserved for future provenance tracking)
// ---------------------------------------------------------------------------

/// Bitmask identifying which counter breaks were required to reach a fact.
///
/// - `0` means the fact is counter-free (unconditional).
/// - Bit `i` set means counter `i` must have broken for this fact to be
///   valid.
///
/// **Currently unused at runtime.**  This type alias is retained so that
/// Proposal 3-style provenance can be adopted later without restructuring
/// the effect types.  See the module-level documentation for the full
/// reintroduction pattern.
///
/// The current Tier 3 is capped to 64 counters by bitmask width elsewhere
/// (`MAX_TIER2_COUNTERS`), so `u64` is sufficient.
#[allow(dead_code)]
pub(crate) type BreakMask = u64;

// ---------------------------------------------------------------------------
// Effect timing
// ---------------------------------------------------------------------------

/// When an effect becomes actionable relative to the current byte boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum EffectTiming {
    /// Defer until the next byte boundary (when the next byte is known).
    NextByte,
    /// Defer until end-of-input processing.
    EndOnly,
}

impl fmt::Display for EffectTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
/// Currently the only runtime condition is an assertion chain.
/// [`AssertChainId::NONE`] means no assertion gating (unconditional).
///
/// # Provenance (future)
///
/// An earlier version included a `required_breaks: BreakMask` field
/// to record which counter breaks were necessary for the effect to be
/// valid.  That field was removed because it was **never enforced** at
/// resolution time — all deposit sites already ensured the break had
/// occurred before creating the effect, making the field redundant.
///
/// If Proposal 3 (provenance-aware state) is adopted, reintroduce
/// `required_breaks` here and enforce it in [`resolve_pending()`] by
/// passing a `satisfied_breaks` mask.  See the module-level doc for
/// the full pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EffectGuard {
    /// Assertion chain that must pass.  `NONE` = no assertions.
    pub(crate) assert_chain: AssertChainId,
}

impl EffectGuard {
    /// A guard that is always satisfied (no assertions, no break requirement).
    ///
    /// Currently unused — deposit sites construct `EffectGuard` inline.
    /// Patch 7 (deposit helpers) will route through this constant.
    #[allow(dead_code)]
    pub(crate) const ALWAYS: Self = Self {
        assert_chain: AssertChainId::NONE,
    };

    /// Whether this guard is unconditional (no assertions).
    pub(crate) fn is_always(&self) -> bool {
        self.assert_chain == AssertChainId::NONE
    }
}

impl fmt::Display for EffectGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_always() {
            return write!(f, "always");
        }
        write!(f, "asserts={}", self.assert_chain)
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

// ---------------------------------------------------------------------------
// Effect resolution: evaluate pending effects at a boundary
// ---------------------------------------------------------------------------

/// Evaluate an assertion chain against the current boundary context.
///
/// Returns `true` if all assertions in the chain pass.  An empty chain
/// (i.e. `AssertChainId::NONE`) always passes.
///
/// When `check_reachability` is `true`, also verifies that the downstream
/// path from each assertion's `out` state can reach a match — this is
/// needed for `Match`/`MatchAtEnd` effects where the assertion guards the
/// match signal itself.  When `false`, only the assertion evaluation is
/// performed — this is correct for `AddSeed`/`AddTail` effects where
/// the assertion gates a non-match action and downstream reachability is
/// irrelevant.
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
/// - `check_reachability`: whether to verify downstream match reachability
///   after each assertion passes.
pub(crate) fn eval_assert_chain(
    chain_id: AssertChainId,
    arena: &AssertChainArena,
    at_end: bool,
    prev: Option<u8>,
    next: Option<u8>,
    regex: &Regex,
    check_reachability: bool,
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
                if check_reachability {
                    // Check if the downstream path from this assertion's
                    // `out` can still reach a match.  This handles chained
                    // assertions (e.g. `\b → \B → $ → Match`).
                    if at_end {
                        if !super::DfaState::can_reach_match_at_end(out, prev, regex) {
                            return false;
                        }
                    } else {
                        // Mid-input: check if can_reach_match_mid from the
                        // assertion's out.
                        if !super::Tier3DfaMatcher::can_reach_match_mid(out, prev, next, regex) {
                            return false;
                        }
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

        // Check downstream reachability only when the effect contains
        // Match or MatchAtEnd atoms — for AddSeed/AddTail the assertion
        // is just a gate on a non-match action.
        let has_match_atoms = pe
            .atoms
            .iter()
            .any(|a| matches!(a, EffectAtom::Match | EffectAtom::MatchAtEnd));
        if !eval_assert_chain(
            pe.guard.assert_chain,
            arena,
            at_end,
            prev,
            next,
            regex,
            has_match_atoms,
        ) {
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
// Effect compilation: Tier3OriginKind → CompiledTargetEffects
// ---------------------------------------------------------------------------

/// Compile [`CompiledTargetEffects`] for a single target from its
/// [`Tier3OriginKind`] and the global break seed table.
///
/// Reads the structural analysis computed by `compute_tier3_analysis()` and
/// translates each target's origin kind into the typed effect representation
/// used at runtime by `resolve_pending()`.
///
/// # Arguments
///
/// - `target_idx`: NFA state index of the post-consumption target.
/// - `origin_kind`: the structural action for this target.
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
            break_deferred_chain_ids: _,
        } => {
            let step = TargetStep::Increment {
                counter: *counter,
                advance_origins: advance_origins.clone(),
                min: *min,
                max: *max,
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
            // Match or $ → Match.  At runtime, these are emitted as
            // PendingEffect entries with Match atoms and evaluated via
            // `eval_assert_chain` with downstream reachability checks at
            // the next byte boundary and at end-of-input.
            //
            // Bug 51: break_deferred_asserts contains independent
            // assertion entry points from different NFA paths (OR
            // semantics: any one passing = match).  Each must be
            // interned as a separate 1-element chain so they are
            // evaluated independently.  Previously they were interned
            // as a single chain (AND semantics), causing false negatives
            // when paths had contradictory assertions like `\B` and `\b`.
            if !break_deferred_asserts.is_empty() {
                for assert_state in break_deferred_asserts.iter() {
                    let chain_id = arena.intern(&[*assert_state]);
                    guarded.push(GuardedEffect {
                        timing: EffectTiming::NextByte,
                        guard: EffectGuard {
                            assert_chain: chain_id,
                        },
                        atoms: vec![EffectAtom::MatchAtEnd].into_boxed_slice(),
                    });
                    // Also an EndOnly variant for finish().
                    guarded.push(GuardedEffect {
                        timing: EffectTiming::EndOnly,
                        guard: EffectGuard {
                            assert_chain: chain_id,
                        },
                        atoms: vec![EffectAtom::MatchAtEnd].into_boxed_slice(),
                    });
                }
            }

            // Per-tail deferred assertions: each tail has its own chain.
            for &(tail, ref per_tail_asserts, _) in break_consuming_deferred.iter() {
                if per_tail_asserts.is_empty() {
                    // Pure tail — already handled above.
                    continue;
                }
                let chain_id = arena.intern(per_tail_asserts);
                guarded.push(GuardedEffect {
                    timing: EffectTiming::NextByte,
                    guard: EffectGuard {
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

/// Compile all target effects for a [`Tier3Analysis`].
///
/// Iterates over all targets in the analysis and compiles a
/// `CompiledTargetEffects` for each non-`None` target.  Also interns
/// each individual assert from `target_deferred_asserts` as a 1-element
/// chain so that deposit sites can emit [`PendingEffect`] entries with
/// proper chain guards.
///
/// Returns `(per-target effects, per-state chain IDs for target deferred
/// asserts, populated assertion chain arena)`.
#[allow(clippy::type_complexity)]
#[cfg_attr(not(debug_assertions), allow(unused_variables))]
pub(crate) fn compile_all_target_effects(
    analysis: &super::Tier3Analysis,
    states: &[crate::State],
) -> (
    Box<[Option<CompiledTargetEffects>]>,
    Box<[Box<[AssertChainId]>]>,
    AssertChainArena,
) {
    let mut arena = AssertChainArena::new();
    let effects: Vec<Option<CompiledTargetEffects>> = analysis
        .targets
        .iter()
        .enumerate()
        .map(|(i, target)| {
            target.as_ref().map(|kind| {
                let eff = compile_target_effects(
                    StateIdx(i as u32),
                    kind,
                    &analysis.break_seeds,
                    &mut arena,
                );
                #[cfg(debug_assertions)]
                debug_assert_origins_consuming(&eff, |s| {
                    matches!(
                        states[s.idx()],
                        crate::State::Byte { .. }
                            | crate::State::ByteCI { .. }
                            | crate::State::ByteClass { .. }
                            | crate::State::ByteTable { .. }
                    )
                });
                eff
            })
        })
        .collect();

    // Intern each individual assert from target_deferred_asserts as a
    // 1-element chain.  This allows deposit sites to create PendingEffect
    // entries with the chain as a guard for Match atoms.
    let target_chain_ids: Vec<Box<[AssertChainId]>> = analysis
        .target_deferred_asserts
        .iter()
        .map(|asserts| {
            asserts
                .iter()
                .map(|&assert_idx| arena.intern(&[assert_idx]))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
        .collect();

    (
        effects.into_boxed_slice(),
        target_chain_ids.into_boxed_slice(),
        arena,
    )
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
    fn test_effect_guard_is_always() {
        assert!(EffectGuard::ALWAYS.is_always());

        let assert_gated = EffectGuard {
            assert_chain: AssertChainId(0),
        };
        assert!(!assert_gated.is_always());
    }

    #[test]
    fn test_effect_timing_display() {
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
                assert_chain: AssertChainId(0),
            },
            atoms: vec![EffectAtom::Match].into_boxed_slice(),
        };
        assert_eq!(format!("{ge}"), "[next_byte|asserts=chain#0] Match");
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
}
