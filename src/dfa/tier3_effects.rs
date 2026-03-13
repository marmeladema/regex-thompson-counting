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

use crate::{CounterIdx, StateIdx};

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
}
