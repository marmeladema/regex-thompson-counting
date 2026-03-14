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
///
/// Only `NextByte` exists.  An `EndOnly` variant is unnecessary because:
///
/// - **`MatchAtEnd` atoms** in `on_break`/`immediate` handle `$ → Match`
///   paths directly by setting `self.match_at_end = true` during the step
///   loop, without going through the pending queue.
/// - **Pending effects from the last byte** are resolved at EOI by
///   `finish()` calling `resolve_pending(NextByte, at_end=true, next=None)`.
///   Assertion chains that include `$` naturally pass at EOI, and the
///   reachability check uses `can_reach_match_at_end`.
/// - Effects deposited on byte N−1 that fail mid-input resolution on byte N
///   are correctly dropped — their `MatchAtEnd` semantics are already
///   captured by the `match_at_end` flag from on_break atoms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum EffectTiming {
    /// Defer until the next byte boundary (when the next byte is known).
    /// Also used for end-of-input resolution (with `at_end=true` and
    /// `next=None`).
    NextByte,
}

impl fmt::Display for EffectTiming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NextByte => write!(f, "next_byte"),
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
// Break effects (interned per-counter break-path data)
// ---------------------------------------------------------------------------

/// Compact ID referencing a [`BreakEffects`] entry in
/// [`Tier3Analysis::break_effects`](super::Tier3Analysis::break_effects).
///
/// All `Tier3OriginKind::Increment` entries for the same counter share
/// the same break-path structure (because Tier 3 has exactly one CInc
/// per counter), so break effects are interned per-counter and referenced
/// by this ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BreakEffectsId(pub(crate) u16);

impl BreakEffectsId {
    /// Sentinel: not yet assigned (used during `analyze_target()`; must be
    /// overwritten before the analysis is consumed at runtime).
    pub(crate) const NONE: Self = Self(u16::MAX);

    /// Return the raw index as `usize`.
    pub(crate) fn idx(self) -> usize {
        debug_assert!(self != Self::NONE, "BreakEffectsId::NONE used as index");
        self.0 as usize
    }
}

impl fmt::Display for BreakEffectsId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::NONE {
            write!(f, "NONE")
        } else {
            write!(f, "be{}", self.0)
        }
    }
}

/// A consuming "tail" state on a counter break path gated by zero or
/// more deferred assertions.
///
/// Each entry represents a single consuming NFA state reachable from a
/// CInc break output.  If the path to the tail passes through non-End
/// assertion states (`\b`, `\B`, etc.), those are recorded as deferred
/// assertions that must pass at runtime before the tail is deposited.
///
/// - `origin`: the consuming NFA state (byte/byte-class/byte-table).
/// - `assert_states`: deferred assertion NFA state indices on the
///   epsilon path from the CInc break output to `origin`.
/// - `chain_id`: interned [`AssertChainId`] for runtime evaluation.
///   Initialized to [`AssertChainId::NONE`] during analysis and
///   populated when the assertion chain arena is built.
///
/// **Semantics: AND per tail.**  All assertions in `assert_states`
/// must pass for this tail to be deposited.  Different `DeferredTail`
/// entries are independent (OR across entries).
#[derive(Clone, Debug)]
pub(crate) struct DeferredTail {
    /// The consuming NFA state on the break path.
    pub(crate) origin: StateIdx,
    /// Deferred assertion NFA state indices gating this tail.
    pub(crate) assert_states: Box<[StateIdx]>,
    /// Interned assertion chain for runtime evaluation.
    pub(crate) chain_id: AssertChainId,
}

/// Pre-interned break-path effect data for a counter's CInc break.
///
/// Extracted from `Tier3OriginKind::Increment`'s break-* fields and
/// stored once on [`Tier3Analysis`](super::Tier3Analysis).  **Build-time
/// only** after Patch 8E — consumed by [`compile_target_effects()`] to
/// produce `on_break`/`guarded` atoms in [`CompiledTargetEffects`], but
/// not read at runtime.  Also used by `dump.rs`.
///
/// # Field semantics
///
/// - `break_deferred_chain_ids`: **OR semantics** — each chain is an
///   independent assertion; any one passing suffices for a match.
/// - `break_consuming_pure`: tails with no assertion gates (deposited
///   immediately as `AddTail` in `on_break`).
/// - `break_consuming_deferred`: tails with per-tail assertion gates
///   (**AND per tail**, OR across tuples; compiled into `guarded`).
/// - `break_consuming_states`: union of pure + deferred tails.
#[derive(Clone, Debug)]
pub(crate) struct BreakEffects {
    /// Per-entry assertion chain IDs for break-deferred asserts (OR
    /// semantics, parallel to the original `break_deferred_asserts`).
    pub(crate) break_deferred_chain_ids: Box<[AssertChainId]>,
    /// Consuming states reachable from the break path WITHOUT passing
    /// through any deferred assertion.
    pub(crate) break_consuming_pure: Box<[StateIdx]>,
    /// Per-tail deferred assertions.  Each tail is gated on its own
    /// assertion chain (AND per tail, OR across entries).
    pub(crate) break_consuming_deferred: Box<[DeferredTail]>,
    /// All consuming states on the break path (union of pure + deferred).
    pub(crate) break_consuming_states: Box<[StateIdx]>,
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
    /// Convenience constant for ungated effects (no assertion chain).
    /// Used in tests only.
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
    ///
    /// Used for deferred assertion paths that gate access to a Match
    /// state: when the assertion chain passes and downstream Match is
    /// reachable, this atom fires.  This covers both break-deferred
    /// assertions (counter break → `\b` → Match) and per-tail deferred
    /// assertions (tail target → `\b` → Match).
    Match,
    /// Signal a match-at-end (`match_at_end = true`).
    ///
    /// Reserved for contexts where the match is only valid at end-of-input
    /// — specifically, counter-free `$ → Match` paths where the pattern
    /// requires the end-of-string anchor.  **Not** used for deferred
    /// assertion paths that may resolve mid-input; those use [`Match`].
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
    /// The atom to apply when the guard passes at the given timing.
    ///
    /// Each `GuardedEffect` carries exactly one atom.  Multi-atom effects
    /// are represented as multiple `GuardedEffect` entries in the parent
    /// [`CompiledTargetEffects::guarded`] slice.
    pub(crate) atom: EffectAtom,
}

impl fmt::Display for GuardedEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}|{}] {}", self.timing, self.guard, self.atom)
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
/// - **`guarded`**: effects with assertion-chain guards and deferred
///   `NextByte` timing that cannot be resolved immediately.
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
// Compiled per-origin effects
// ---------------------------------------------------------------------------

/// Per-consuming-state compiled origin effects.
///
/// Captures match flags and deferred assertion chain IDs that depend on the
/// epsilon path from a consuming state's byte-consumption edge.  Indexed by
/// the **origin** consuming state, NOT the post-consumption target — this is
/// distinct from [`CompiledTargetEffects`] which is keyed by target.
///
/// Used at runtime for:
/// - Dead-target fallback: when `target_effects[target.idx()]` is `None`,
///   the origin's match flags determine whether `$ → Match` or direct
///   `Match` is reachable from the consumption point.
/// - Deferred assertion deposition: when a tail advances through a live
///   target, its per-origin deferred assertions are deposited as
///   [`PendingEffect`] entries with `Match` atoms.
#[derive(Clone, Debug)]
pub(crate) struct CompiledOriginEffects {
    /// True if consuming a byte at this origin's state leads to
    /// `$ → Match` through epsilon transitions.
    pub(crate) is_match_at_end: bool,
    /// True if consuming a byte at this origin's state leads directly
    /// to `Match` through epsilon transitions (not through CInc).
    pub(crate) is_match: bool,
    /// Interned assertion chain IDs for deferred assertions on the
    /// epsilon path from this origin's target to downstream consuming
    /// states or `Match`.
    ///
    /// **Semantics: AlternativeChains (OR).**  Each chain ID is an
    /// independent 1-element assertion chain.  At runtime, each is
    /// emitted as its own `PendingEffect` with `EffectAtom::Match`;
    /// any one chain passing suffices for the match to fire.
    pub(crate) deferred_chain_ids: Box<[AssertChainId]>,
}

impl fmt::Display for CompiledOriginEffects {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut flags = Vec::new();
        if self.is_match {
            flags.push("match");
        }
        if self.is_match_at_end {
            flags.push("mae");
        }
        if flags.is_empty() && self.deferred_chain_ids.is_empty() {
            return write!(f, "(none)");
        }
        if !flags.is_empty() {
            write!(f, "{}", flags.join("+"))?;
        }
        if !self.deferred_chain_ids.is_empty() {
            if !flags.is_empty() {
                write!(f, " ")?;
            }
            write!(f, "deferred=[")?;
            for (i, &cid) in self.deferred_chain_ids.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{cid}")?;
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
    /// The atom to apply when the guard passes.
    ///
    /// Each `PendingEffect` carries exactly one atom.  The deposition
    /// helpers create one `PendingEffect` per atom, so multi-atom
    /// effects are represented as multiple entries in the queue.
    pub(crate) atom: EffectAtom,
    /// Word-boundary context at the time the effect was scheduled.
    pub(crate) prev_was_word: bool,
}

impl fmt::Display for PendingEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}|{}|pw={}] {}",
            self.timing, self.guard, self.prev_was_word, self.atom
        )
    }
}

// ---------------------------------------------------------------------------
// Effect deposition helpers
// ---------------------------------------------------------------------------
//
// These free functions replace the ~11 open-coded `PendingEffect`
// construction sites scattered across `tier3.rs`.  Each helper captures
// one semantic pattern of effect deposition.  Callers pass whichever
// queue is appropriate (`pending_effects_current` for first-order deposits
// during `step_slow`, `pending_effects_next` for second-order deposits
// during effect resolution).
//
// By centralising the construction here, future semantic changes (e.g.
// adjusting OR/AND guard semantics) require a single-site fix instead of
// a multi-site audit.

/// Deposit target-deferred match effects.
///
/// For each assertion chain in `chain_ids`, pushes a `PendingEffect` with
/// a single [`EffectAtom::Match`] atom.  These represent non-End assertion
/// states on the epsilon path from a target, which would otherwise only
/// fire via the contaminated no_break_current DFA state.
///
/// **Used for:** Bug 30 / Bug 34 target deferred asserts in both the
/// `Advance` and `Some(None)` arms of post-break tail processing and
/// effect resolution.
pub(crate) fn enqueue_target_deferred_match(
    queue: &mut Vec<PendingEffect>,
    chain_ids: &[AssertChainId],
    prev_was_word: bool,
) {
    for &chain_id in chain_ids {
        queue.push(PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard {
                assert_chain: chain_id,
            },
            atom: EffectAtom::Match,
            prev_was_word,
        });
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
/// - `scratch`: reusable buffers for epsilon-walk reachability checks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn eval_assert_chain(
    chain_id: AssertChainId,
    arena: &AssertChainArena,
    at_end: bool,
    prev: Option<u8>,
    next: Option<u8>,
    regex: &Regex,
    check_reachability: bool,
    scratch: &mut super::ReachScratch,
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
                        if !super::DfaState::can_reach_match_at_end(out, prev, regex, scratch) {
                            return false;
                        }
                    } else {
                        // Mid-input: check if can_reach_match_mid from the
                        // assertion's out.
                        if !super::Tier3DfaMatcher::can_reach_match_mid(
                            out, prev, next, regex, scratch,
                        ) {
                            return false;
                        }
                    }
                }
            }
            AssertEval::Fail => return false,
            AssertEval::Defer => {
                // The assertion needs the next byte — can't resolve yet.
                // For mid-input NextByte effects this shouldn't happen
                // (we have the next byte).  For EOI resolution (at_end=true,
                // next=None), treat as failure (conservative).
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
    scratch: &mut super::ReachScratch,
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

        // Check downstream reachability only when the effect carries a
        // Match or MatchAtEnd atom — for AddSeed/AddTail the assertion
        // is just a gate on a non-match action.
        let has_match_atom = matches!(pe.atom, EffectAtom::Match | EffectAtom::MatchAtEnd);
        if !eval_assert_chain(
            pe.guard.assert_chain,
            arena,
            at_end,
            prev,
            next,
            regex,
            has_match_atom,
            scratch,
        ) {
            continue;
        }

        // Guard passed — apply atom.
        match pe.atom {
            EffectAtom::AddSeed {
                counter,
                origin,
                value,
            } => {
                actions.seeds.push((counter, origin, value));
            }
            EffectAtom::AddTail { origin } => {
                if !actions.tails.contains(&origin) {
                    actions.tails.push(origin);
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
        assert_atom_origins_consuming(&ge.atom, &is_consuming);
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
    break_effects: &[BreakEffects],
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
            break_effects_id,
        } => {
            let step = TargetStep::Increment {
                counter: *counter,
                advance_origins: advance_origins.clone(),
                min: *min,
                max: *max,
                continue_origins: continue_origins.clone(),
            };

            // Look up the pre-interned break effects for this counter.
            let be = &break_effects[break_effects_id.idx()];

            // --- on_break: immediate effects gated on counter break ---
            let mut on_break = Vec::new();

            // Break-path direct match.
            if *break_is_match {
                on_break.push(EffectAtom::Match);
            }

            // Break-path match-at-end.
            //
            // `break_is_match_at_end` is ONLY true when `break_closure()`
            // found a pure (assertion-free) `$ → Match` path from the CInc
            // break output.  The `has_deferred` flag means OTHER branches
            // of the break epsilon closure have deferred assertions — it
            // does NOT mean the `$ → Match` sub-path is gated.  A CInc
            // break can have both a pure `$ → Match` branch and a deferred
            // `\b → consuming-state` branch via Split.
            //
            // Previous code gated this on `!be.has_deferred`, which
            // incorrectly suppressed `MatchAtEnd` for patterns like
            // `^.{2,5}(\b|$)` where the `$` branch is pure but the
            // sibling `\b` branch causes `has_deferred = true`.
            if *break_is_match_at_end {
                on_break.push(EffectAtom::MatchAtEnd);
            }

            // Pure tails: consuming states reachable from break path
            // WITHOUT deferred assertions — deposited immediately on break.
            for &tail in be.break_consuming_pure.iter() {
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
            // When the break path has deferred assertions (e.g. `\b`,
            // `\B`), they gate access to Match or $ → Match.  At
            // runtime, these are emitted as PendingEffect entries with
            // Match atoms and evaluated via `eval_assert_chain` with
            // downstream reachability checks at the next byte boundary
            // and at end-of-input.
            //
            // We emit `EffectAtom::Match` (not `MatchAtEnd`) to align
            // with the runtime, which sets `ever_matched` when the
            // deferred assertion passes and downstream Match reachability
            // exists.  `MatchAtEnd` is reserved for contexts where the
            // match signal is only valid at end-of-input (e.g. counter-free
            // `$ → Match` paths), not for assertion-gated break paths that
            // may resolve mid-input.
            //
            // Bug 51: break_deferred_chain_ids contains independent
            // assertion chains from different NFA paths (OR semantics:
            // any one passing = match).  Each is a separate 1-element
            // chain so they are evaluated independently.  Previously
            // they were interned as a single chain (AND semantics),
            // causing false negatives when paths had contradictory
            // assertions like `\B` and `\b`.
            for &chain_id in be.break_deferred_chain_ids.iter() {
                guarded.push(GuardedEffect {
                    timing: EffectTiming::NextByte,
                    guard: EffectGuard {
                        assert_chain: chain_id,
                    },
                    atom: EffectAtom::Match,
                });
            }

            // Per-tail deferred assertions: each tail has its own
            // pre-interned chain.  Skip pure tails (chain_id == NONE).
            for dt in be.break_consuming_deferred.iter() {
                if dt.chain_id == AssertChainId::NONE {
                    // Pure tail — already handled above.
                    continue;
                }
                guarded.push(GuardedEffect {
                    timing: EffectTiming::NextByte,
                    guard: EffectGuard {
                        assert_chain: dt.chain_id,
                    },
                    atom: EffectAtom::AddTail { origin: dt.origin },
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
                        atom: EffectAtom::AddSeed {
                            counter: bs.counter,
                            origin: bs.origin,
                            value: 0,
                        },
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
/// The `arena` is passed in pre-populated with break-effect chains
/// (from [`BreakEffects`] construction).  Additional chains (target
/// deferred asserts, break-seed deferred asserts) are interned into
/// the same arena during compilation.
///
/// Returns `(per-target effects, per-state chain IDs for target deferred
/// asserts)`.  The caller retains ownership of the arena.
#[cfg_attr(not(debug_assertions), allow(unused_variables))]
pub(crate) fn compile_all_target_effects(
    analysis: &super::Tier3Analysis,
    states: &[crate::State],
    arena: &mut AssertChainArena,
) -> (
    Box<[Option<CompiledTargetEffects>]>,
    Box<[CompiledOriginEffects]>,
) {
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
                    &analysis.break_effects,
                    arena,
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

    // Build per-origin compiled effects: consolidate match flags and
    // deferred assertion chain IDs into a single structure per consuming
    // state.  Each individual assert from target_deferred_asserts is
    // interned as a 1-element chain so it can be used as a guard in
    // PendingEffect entries.
    let origin_effects: Vec<CompiledOriginEffects> = (0..states.len())
        .map(|i| {
            let chain_ids: Box<[AssertChainId]> = analysis
                .target_deferred_asserts
                .get(i)
                .map(|asserts| {
                    asserts
                        .iter()
                        .map(|&assert_idx| arena.intern(&[assert_idx]))
                        .collect::<Vec<_>>()
                        .into_boxed_slice()
                })
                .unwrap_or_default();
            CompiledOriginEffects {
                is_match_at_end: analysis
                    .target_is_match_at_end
                    .get(i)
                    .copied()
                    .unwrap_or(false),
                is_match: analysis.target_is_match.get(i).copied().unwrap_or(false),
                deferred_chain_ids: chain_ids,
            }
        })
        .collect();

    (
        effects.into_boxed_slice(),
        origin_effects.into_boxed_slice(),
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
            atom: EffectAtom::Match,
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
            atom: EffectAtom::AddTail {
                origin: StateIdx(3),
            },
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

    // -----------------------------------------------------------------------
    // Semantic tests for resolve_pending() and eval_assert_chain()
    // -----------------------------------------------------------------------

    /// Create a fresh [`ReachScratch`] (test helper).
    fn scratch() -> super::super::ReachScratch {
        super::super::ReachScratch::new()
    }

    /// Build a compiled `Regex` from a pattern string (test helper).
    fn build_regex(pattern: &str) -> crate::Regex {
        use regex_syntax::ast::parse::ParserBuilder;
        use regex_syntax::hir::translate::TranslatorBuilder;

        let ast = ParserBuilder::new()
            .build()
            .parse(pattern)
            .expect("regex-syntax AST parse should succeed");
        let hir = TranslatorBuilder::new()
            .unicode(false)
            .utf8(false)
            .dot_matches_new_line(true)
            .build()
            .translate(pattern, &ast)
            .expect("regex-syntax HIR translation should succeed");
        crate::RegexBuilder::default()
            .build(&hir)
            .expect("builder should accept the HIR")
    }

    /// Find the first NFA state index with `State::Assert { kind, .. }`
    /// matching the given `AssertKind`.
    fn find_assert_state(regex: &crate::Regex, target_kind: crate::AssertKind) -> StateIdx {
        for (i, state) in regex.states.0.iter().enumerate() {
            if let crate::State::Assert { kind, .. } = state {
                if *kind == target_kind {
                    return StateIdx(i as u32);
                }
            }
        }
        panic!("no Assert({target_kind:?}) state found in regex");
    }

    #[test]
    fn test_resolve_pending_unconditional_match() {
        let regex = build_regex("a");
        let arena = AssertChainArena::new();
        let effects = vec![PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard::ALWAYS,
            atom: EffectAtom::Match,
            prev_was_word: false,
        }];
        // Should resolve regardless of boundary context.
        let actions = resolve_pending(
            &effects,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b'x'),
            &regex,
            &mut scratch(),
        );
        assert!(actions.set_match, "unconditional Match should resolve");
        assert!(!actions.set_match_at_end);
        assert!(actions.seeds.is_empty());
        assert!(actions.tails.is_empty());
    }

    #[test]
    fn test_resolve_pending_assert_chain_passes() {
        // Pattern `a\b$`: after consuming `a`, the \b assert leads to
        // `$ → Match`.  Use AddTail to test the assertion guard without
        // the downstream match reachability check (which is skipped for
        // non-match atoms).
        let regex = build_regex(r"a\b");
        let wb_state = find_assert_state(&regex, crate::AssertKind::WordAscii);
        let mut arena = AssertChainArena::new();
        let chain_id = arena.intern(&[wb_state]);
        let effects = vec![PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard {
                assert_chain: chain_id,
            },
            atom: EffectAtom::AddTail {
                origin: StateIdx(0),
            },
            // prev byte is word
            prev_was_word: true,
        }];
        // next byte is non-word → boundary exists → \b passes
        let actions = resolve_pending(
            &effects,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b' '),
            &regex,
            &mut scratch(),
        );
        assert!(
            !actions.tails.is_empty(),
            "\\b should pass at word→non-word boundary"
        );
    }

    #[test]
    fn test_resolve_pending_assert_chain_fails() {
        // Same pattern `a\b`, but boundary condition NOT met: word→word.
        let regex = build_regex(r"a\b");
        let wb_state = find_assert_state(&regex, crate::AssertKind::WordAscii);
        let mut arena = AssertChainArena::new();
        let chain_id = arena.intern(&[wb_state]);
        let effects = vec![PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard {
                assert_chain: chain_id,
            },
            atom: EffectAtom::AddTail {
                origin: StateIdx(0),
            },
            // prev byte is word
            prev_was_word: true,
        }];
        // next byte is also word → no boundary → \b fails
        let actions = resolve_pending(
            &effects,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b'a'),
            &regex,
            &mut scratch(),
        );
        assert!(
            actions.tails.is_empty(),
            "\\b should fail at word→word (no boundary)"
        );
    }

    #[test]
    fn test_resolve_pending_contradictory_or_alternatives() {
        // Two pending effects for the same action — one guarded by \b,
        // one guarded by \B.  At ANY boundary, exactly one should resolve
        // (OR semantics via multiple effects).
        // Use AddTail atoms to avoid downstream reachability checks.
        let regex = build_regex(r"a\b|a\B");
        let wb = find_assert_state(&regex, crate::AssertKind::WordAscii);
        let nwb = find_assert_state(&regex, crate::AssertKind::WordAsciiNegate);

        let mut arena = AssertChainArena::new();
        let wb_chain = arena.intern(&[wb]);
        let nwb_chain = arena.intern(&[nwb]);

        let make_effects = || {
            vec![
                PendingEffect {
                    timing: EffectTiming::NextByte,
                    guard: EffectGuard {
                        assert_chain: wb_chain,
                    },
                    atom: EffectAtom::AddTail {
                        origin: StateIdx(0),
                    },
                    prev_was_word: true,
                },
                PendingEffect {
                    timing: EffectTiming::NextByte,
                    guard: EffectGuard {
                        assert_chain: nwb_chain,
                    },
                    atom: EffectAtom::AddTail {
                        origin: StateIdx(1),
                    },
                    prev_was_word: true,
                },
            ]
        };

        // Word → non-word: \b passes, \B fails — one tail should resolve.
        let effects1 = make_effects();
        let actions = resolve_pending(
            &effects1,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b' '),
            &regex,
            &mut scratch(),
        );
        assert!(
            !actions.tails.is_empty(),
            "OR alternatives: \\b should pass at word→non-word"
        );
        assert!(
            actions.tails.contains(&StateIdx(0)),
            "\\b tail should be present"
        );

        // Word → word: \b fails, \B passes — other tail should resolve.
        let effects2 = make_effects();
        let actions2 = resolve_pending(
            &effects2,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b'a'),
            &regex,
            &mut scratch(),
        );
        assert!(
            !actions2.tails.is_empty(),
            "OR alternatives: \\B should pass at word→word"
        );
        assert!(
            actions2.tails.contains(&StateIdx(1)),
            "\\B tail should be present"
        );
    }

    #[test]
    fn test_resolve_pending_conjunctive_chain() {
        // A single chain of [\b, \B] (AND semantics).
        // \b and \B are contradictory — AND can never pass.
        // Use AddTail to avoid downstream reachability checks.
        let regex = build_regex(r"a\b|a\B");
        let wb = find_assert_state(&regex, crate::AssertKind::WordAscii);
        let nwb = find_assert_state(&regex, crate::AssertKind::WordAsciiNegate);

        let mut arena = AssertChainArena::new();
        // Intern as a single chain with both (AND semantics).
        let and_chain = arena.intern(&[wb, nwb]);

        let effects = vec![PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard {
                assert_chain: and_chain,
            },
            atom: EffectAtom::AddTail {
                origin: StateIdx(0),
            },
            prev_was_word: true,
        }];

        // Word → non-word: \b passes but \B fails → AND fails.
        let actions = resolve_pending(
            &effects,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b' '),
            &regex,
            &mut scratch(),
        );
        assert!(
            actions.tails.is_empty(),
            "AND chain of \\b+\\B should never pass"
        );

        // Word → word: \b fails → AND fails immediately.
        let actions2 = resolve_pending(
            &effects,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b'a'),
            &regex,
            &mut scratch(),
        );
        assert!(
            actions2.tails.is_empty(),
            "AND chain of \\b+\\B should never pass (2)"
        );
    }

    #[test]
    fn test_resolve_pending_add_tail() {
        let regex = build_regex("a");
        let arena = AssertChainArena::new();
        let effects = vec![PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard::ALWAYS,
            atom: EffectAtom::AddTail {
                origin: StateIdx(5),
            },
            prev_was_word: false,
        }];
        let actions = resolve_pending(
            &effects,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b'x'),
            &regex,
            &mut scratch(),
        );
        assert_eq!(actions.tails, vec![StateIdx(5)]);
        assert!(!actions.set_match);
    }

    #[test]
    fn test_resolve_pending_add_seed() {
        let regex = build_regex("a");
        let arena = AssertChainArena::new();
        let effects = vec![PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard::ALWAYS,
            atom: EffectAtom::AddSeed {
                counter: CounterIdx(0),
                origin: StateIdx(3),
                value: 1,
            },
            prev_was_word: false,
        }];
        let actions = resolve_pending(
            &effects,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b'x'),
            &regex,
            &mut scratch(),
        );
        assert_eq!(actions.seeds.len(), 1);
        assert_eq!(actions.seeds[0], (CounterIdx(0), StateIdx(3), 1));
        assert!(!actions.set_match);
    }

    #[test]
    fn test_resolve_pending_match_at_end_atom() {
        // MatchAtEnd atom sets set_match_at_end, not set_match.
        let regex = build_regex("a");
        let arena = AssertChainArena::new();
        let effects = vec![PendingEffect {
            timing: EffectTiming::NextByte,
            guard: EffectGuard::ALWAYS,
            atom: EffectAtom::MatchAtEnd,
            prev_was_word: false,
        }];
        let actions = resolve_pending(
            &effects,
            EffectTiming::NextByte,
            &arena,
            false,
            Some(b'x'),
            &regex,
            &mut scratch(),
        );
        assert!(!actions.set_match, "MatchAtEnd should not set set_match");
        assert!(
            actions.set_match_at_end,
            "MatchAtEnd should set set_match_at_end"
        );
    }
}
