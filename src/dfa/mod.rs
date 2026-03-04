//! Lazy DFA (Tier 1, Tier 2, Tier 3, and Tier 4).
//!
//! **Tier 1** (counter-free patterns): standard lazy subset construction.
//! DFA-eligible patterns may include deferred assertions (`\b`, `\B`,
//! `EndLF`) whose evaluation needs the *next* byte.  These are parked
//! in the DFA state and resolved at the start of `populate()` when the
//! next input byte is known.
//!
//! Only `StartCRLF` and `EndCRLF` remain DFA-ineligible.
//!
//! **Tier 3** (non-nested counted repetitions): DFA + conditional transitions.
//! Precomputes separate DFA successor states per counter condition instead
//! of using runtime counter programs.  Each counter tracked independently.
//!
//! **Tier 4** (nested counted repetitions): DFA + explicit counter contexts.
//! The DFA state (set of NFA consuming states) is separated from the
//! counter state (a set of `CounterCtx` values).  DFA transitions are
//! cached normally; each transition also stores a compiled *counter
//! program* that describes how to update counter values.  On a cache
//! hit the program is replayed against each active counter context.

mod tier1;
mod tier2;
mod tier3;
mod tier4;

pub(crate) use tier1::{DfaCache, DfaMatcher};
pub(crate) use tier2::{Tier2DfaCache, Tier2DfaMatcher};
pub(crate) use tier3::{Tier3DfaCache, Tier3DfaMatcher};
pub(crate) use tier4::{Tier4DfaCache, Tier4DfaMatcher};

use crate::StateIdx;

// ---------------------------------------------------------------------------
// DFA state table (shared by Tier 1 and Tier 3)
// ---------------------------------------------------------------------------

/// Index into the DFA state table ([`DfaCache::states`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DfaStateId(u32);

impl DfaStateId {
    /// Sentinel: the "dead" state (no NFA states, no match possible).
    pub(crate) const DEAD: Self = Self(u32::MAX);
    /// Sentinel for unpopulated flat-table slots.
    pub(super) const UNPOPULATED: Self = Self(u32::MAX - 1);

    #[inline]
    pub(super) fn idx(self) -> usize {
        self.0 as usize
    }
}

/// A DFA state: a sorted, deduplicated set of NFA consuming/assert state
/// indices, plus flags derived from the epsilon closure.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct DfaState {
    /// Sorted NFA state indices (consuming states only: Byte, ByteClass,
    /// ByteTable).  Assert states are resolved during closure computation;
    /// Match states set the `is_match` / `is_match_at_end` flags.
    pub(super) nfa_states: Box<[StateIdx]>,
    /// NFA Assert state indices whose evaluation was deferred because the
    /// next byte was unknown.  These are `WordAscii`, `WordAsciiNegate`,
    /// or `EndLF` states.  Resolved in `populate()` when the next byte
    /// arrives.
    pub(super) deferred_asserts: Box<[StateIdx]>,
    /// True if `Match` is directly reachable (no pending `$` gate).
    pub(super) is_match: bool,
    /// True if `Match` is reachable through an `Assert(End)` gate.
    /// Only fires when `at_end = true` (in `finish()`).
    pub(super) is_match_at_end: bool,
    /// Whether the previous byte (the byte that *entered* this state) was
    /// an ASCII word character.  Only meaningful when `deferred_asserts`
    /// is non-empty; always `false` otherwise, to avoid unnecessary state
    /// splitting.
    pub(super) prev_was_word: bool,
}
