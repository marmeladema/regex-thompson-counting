//! Human-readable dump of compiled NFA states, counters, and DFA analysis.
//!
//! The primary entry point is [`Regex::dump`], which returns a [`DumpRegex`]
//! wrapper implementing [`Display`](std::fmt::Display).  All formatting is
//! done through `Display` impls on thin wrapper types that borrow the
//! underlying data.

use std::fmt;

use crate::dfa::Tier3OriginKind;
use crate::dfa::tier3_effects::{AssertChainArena, CompiledOriginEffects, CompiledTargetEffects};
use crate::{AssertKind, ByteClass, ByteMap, Regex, State, StateIdx};

// ---------------------------------------------------------------------------
// Byte formatting helpers
// ---------------------------------------------------------------------------

/// Format a byte as a human-readable character literal or hex escape.
///
/// Printable ASCII (0x20..=0x7E) is shown as `'c'`, with special escapes
/// for common control characters.  Everything else uses `\xNN` hex notation.
fn format_byte(b: u8, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match b {
        b'\\' => write!(f, "'\\\\'"),
        b'\'' => write!(f, "'\\''"),
        b'\n' => write!(f, "'\\n'"),
        b'\r' => write!(f, "'\\r'"),
        b'\t' => write!(f, "'\\t'"),
        0x20..=0x7E => write!(f, "'{}'", b as char),
        _ => write!(f, "\\x{b:02x}"),
    }
}

/// Format a [`ByteClass`] as a compact set of byte ranges.
///
/// Groups consecutive matching bytes into `lo-hi` ranges (or single
/// values when `lo == hi`).  Uses the same printable/hex notation as
/// [`format_byte`].
fn format_byte_class(class: &ByteClass, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[")?;

    let mut first = true;
    let mut i: usize = 0;
    while i < 256 {
        if !class.0[i] {
            i += 1;
            continue;
        }
        let lo = i as u8;
        while i < 256 && class.0[i] {
            i += 1;
        }
        let hi = (i - 1) as u8;

        if !first {
            write!(f, ", ")?;
        }
        first = false;

        if lo == hi {
            format_byte(lo, f)?;
        } else {
            format_byte(lo, f)?;
            write!(f, "-")?;
            format_byte(hi, f)?;
        }
    }
    write!(f, "]")
}

// ---------------------------------------------------------------------------
// DumpState — one NFA state per line
// ---------------------------------------------------------------------------

/// Human-readable display wrapper for a single NFA [`State`].
///
/// Formats one NFA state as a compact one-liner:
/// ```text
///   0: Assert(Start) → 7
///   1: ByteClass(cls:0) → 2
///   2: CInc(c0, {7,25}) → cont:1 | break:6
/// ```
struct DumpState<'a> {
    state: &'a State,
    byte_tables: &'a [ByteMap],
}

impl fmt::Display for DumpState<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.state {
            State::Split { out, out1 } => {
                write!(f, "Split → {} | {}", out, out1)
            }
            State::CounterInstance { counter, out } => {
                write!(f, "CI(c{counter}) → {out}")
            }
            State::CounterIncrement {
                counter,
                out,
                out1,
                min,
                max,
            } => {
                write!(
                    f,
                    "CInc(c{counter}, {{{min},{max}}}) → cont:{out} | break:{out1}",
                )
            }
            State::Byte { byte, out } => {
                write!(f, "Byte(")?;
                format_byte(byte, f)?;
                write!(f, ") → {out}")
            }
            State::ByteCI { byte, out } => {
                write!(f, "ByteCI(")?;
                format_byte(byte, f)?;
                write!(f, ") → {out}")
            }
            State::ByteClass { class, out } => {
                write!(f, "ByteClass(cls:{}) → {out}", class.idx())
            }
            State::ByteTable { table } => {
                let map = &self.byte_tables[table.idx()];
                let entries: Vec<(u8, StateIdx)> = (0..=255u8)
                    .filter_map(|b| {
                        let target = map[b];
                        if target != StateIdx::NONE {
                            Some((b, target))
                        } else {
                            None
                        }
                    })
                    .collect();
                write!(f, "ByteTable(tbl:{}) ", table.idx())?;
                for (i, &(b, target)) in entries.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    format_byte(b, f)?;
                    write!(f, "→{target}")?;
                }
                write!(f, " [{} entries]", entries.len())
            }
            State::Assert { kind, out } => {
                let name = match kind {
                    AssertKind::Start => "Start",
                    AssertKind::End => "End",
                    AssertKind::StartLF => "StartLF",
                    AssertKind::EndLF => "EndLF",
                    AssertKind::StartCRLF => "StartCRLF",
                    AssertKind::EndCRLF => "EndCRLF",
                    AssertKind::WordAscii => "WordAscii",
                    AssertKind::WordAsciiNegate => "WordAsciiNegate",
                    AssertKind::WordStartAscii => "WordStartAscii",
                    AssertKind::WordEndAscii => "WordEndAscii",
                };
                write!(f, "Assert({name}) → {out}")
            }
            State::Match => write!(f, "Match"),
        }
    }
}

// ---------------------------------------------------------------------------
// Effect dump helpers
// ---------------------------------------------------------------------------

/// Format compiled target effects for dump output.
///
/// Displays the assertion chain arena contents followed by each non-`None`
/// target effect entry.  Used by the Tier 3 DFA analysis dump section.
fn fmt_target_effects(
    f: &mut fmt::Formatter<'_>,
    target_effects: &[Option<CompiledTargetEffects>],
    arena: &AssertChainArena,
    states: &[State],
) -> fmt::Result {
    // Assertion chain arena.
    if arena.len() > 0 {
        writeln!(f, "  assert_chains:")?;
        for i in 0..arena.len() {
            let chain = arena.get(crate::dfa::tier3_effects::AssertChainId(i as u32));
            let labels: Vec<String> = chain
                .iter()
                .map(|&s| {
                    if let State::Assert { kind, .. } = states[s] {
                        format!("{}@{}", kind.label(), s)
                    } else {
                        format!("?@{s}")
                    }
                })
                .collect();
            writeln!(f, "    chain#{i}: [{}]", labels.join(", "))?;
        }
    } else {
        writeln!(f, "  assert_chains: (none)")?;
    }

    // Per-target effects.
    let entries: Vec<(usize, &CompiledTargetEffects)> = target_effects
        .iter()
        .enumerate()
        .filter_map(|(i, e)| e.as_ref().map(|eff| (i, eff)))
        .collect();
    if !entries.is_empty() {
        writeln!(f, "  target_effects:")?;
        for (i, eff) in entries {
            writeln!(f, "    state {i}: {eff}")?;
        }
    } else {
        writeln!(f, "  target_effects: (none)")?;
    }
    Ok(())
}

fn fmt_origin_effects(
    f: &mut fmt::Formatter<'_>,
    origin_effects: &[CompiledOriginEffects],
    states: &[State],
) -> fmt::Result {
    // Only show origins that have non-empty effects.
    let entries: Vec<(usize, &CompiledOriginEffects)> = origin_effects
        .iter()
        .enumerate()
        .filter(|(_, oe)| !oe.immediate.is_empty() || !oe.guarded.is_empty())
        .collect();
    if !entries.is_empty() {
        writeln!(f, "  origin_effects:")?;
        for (i, oe) in entries {
            let label = match states.get(i) {
                Some(State::Byte { byte, .. }) => format!("Byte('{}')", *byte as char),
                Some(State::ByteClass { .. }) => "ByteClass".to_string(),
                Some(State::ByteTable { .. }) => "ByteTable".to_string(),
                Some(State::ByteCI { .. }) => "ByteCI".to_string(),
                _ => "?".to_string(),
            };
            writeln!(f, "    state {i} ({label}): {oe}")?;
        }
    } else {
        writeln!(f, "  origin_effects: (none)")?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// DumpRegex — full dump of a compiled Regex
// ---------------------------------------------------------------------------

/// Human-readable dump of a compiled [`Regex`].
///
/// Created by [`Regex::dump`].  Implements [`Display`] so it can be
/// used directly with `print!("{}", regex.dump(dfa))`.
///
/// When `dfa` is `true`, appends tier-specific analysis after the NFA
/// and counter sections.
pub struct DumpRegex<'a> {
    regex: &'a Regex,
    dfa: bool,
}

impl<'a> DumpRegex<'a> {
    /// Create a new dump wrapper.
    pub(crate) fn new(regex: &'a Regex, dfa: bool) -> Self {
        Self { regex, dfa }
    }
}

impl fmt::Display for DumpRegex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.regex;
        let states = &r.states.0;
        let width = if states.len() <= 1 {
            1
        } else {
            (states.len() - 1).ilog10() as usize + 1
        };

        // -- NFA States -------------------------------------------------------
        writeln!(f, "NFA States ({}):", states.len())?;
        for (i, state) in states.iter().enumerate() {
            let ds = DumpState {
                state,
                byte_tables: &r.byte_tables,
            };
            writeln!(f, "  {i:>width$}: {ds}")?;
        }
        writeln!(f)?;

        // -- Byte Classes -----------------------------------------------------
        if !r.classes.is_empty() {
            writeln!(f, "Byte Classes ({}):", r.classes.len())?;
            for (i, class) in r.classes.iter().enumerate() {
                write!(f, "  cls:{i}: ")?;
                format_byte_class(class, f)?;
                writeln!(f)?;
            }
            writeln!(f)?;
        }

        // -- Byte Tables (summary) --------------------------------------------
        if !r.byte_tables.is_empty() {
            writeln!(f, "Byte Tables ({}):", r.byte_tables.len())?;
            for (i, map) in r.byte_tables.iter().enumerate() {
                let count = (0..=255u8).filter(|&b| map[b] != StateIdx::NONE).count();
                writeln!(f, "  tbl:{i}: {count} entries")?;
            }
            writeln!(f)?;
        }

        // -- Counters ---------------------------------------------------------
        if r.num_counters > 0 {
            writeln!(f, "Counters ({}):", r.num_counters)?;
            for (i, &(min, max, body_len)) in r.counter_info.iter().enumerate() {
                if body_len == 0 {
                    writeln!(f, "  c{i}: {{{min},{max}}}, body_len=variable")?;
                } else {
                    writeln!(f, "  c{i}: {{{min},{max}}}, body_len={body_len}")?;
                }
            }
            writeln!(f)?;
        }

        // -- Basic Info -------------------------------------------------------
        writeln!(f, "Basic Info:")?;
        writeln!(f, "  start: {}", r.start)?;
        writeln!(
            f,
            "  start_closure: [{}]",
            r.start_closure
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        writeln!(f, "  matches_empty: {}", r.start_closure_matches)?;
        writeln!(f, "  prefilter: {:?}", r.prefilter)?;
        writeln!(
            f,
            "  byte_classes: {} classes (compression: {})",
            r.num_byte_classes,
            if r.num_byte_classes < 256 {
                "on"
            } else {
                "off"
            }
        )?;
        writeln!(f)?;

        // -- Tier Eligibility -------------------------------------------------
        writeln!(f, "Tier Eligibility:")?;
        writeln!(f, "  tier1 (lazy DFA):              {}", r.dfa_eligible)?;
        writeln!(f, "  tier2 (differential counters): {}", r.tier2_eligible)?;
        writeln!(f, "  tier3 (conditional trans.):     {}", r.tier3_eligible)?;
        writeln!(f, "  tier4 (counter programs):       {}", r.tier4_eligible)?;
        writeln!(f)?;

        // -- Auxiliary Arrays --------------------------------------------------
        writeln!(f, "Auxiliary Arrays:")?;
        write!(f, "  state_can_reach_match: [")?;
        for (i, &v) in r.state_can_reach_match.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if v {
                write!(f, "{i}")?;
            } else {
                write!(f, ".")?;
            }
        }
        writeln!(f, "]")?;
        if !r.counter_break_can_match.is_empty() {
            write!(f, "  counter_break_can_match: [")?;
            for (i, &v) in r.counter_break_can_match.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "c{i}={v}")?;
            }
            writeln!(f, "]")?;
        }

        // -- DFA Analysis (optional) ------------------------------------------
        if self.dfa {
            self.fmt_dfa_analysis(f)?;
        }

        Ok(())
    }
}

impl DumpRegex<'_> {
    /// Format tier-specific DFA analysis sections.
    fn fmt_dfa_analysis(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.regex;

        // -- Tier 2 Analysis --------------------------------------------------
        if let Some(ref t2) = r.tier2_analysis {
            writeln!(f)?;
            writeln!(f, "Tier 2 Analysis:")?;
            for ci in 0..r.num_counters {
                let interior = t2.interior(ci);
                if interior.is_empty() {
                    writeln!(f, "  c{ci}: body_interior: (none — L=1)")?;
                } else {
                    writeln!(
                        f,
                        "  c{ci}: body_interior: [{}]",
                        interior
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?;
                }
                let bc = t2.break_consuming(ci);
                if !bc.is_empty() {
                    writeln!(
                        f,
                        "  c{ci}: break_consuming: [{}]",
                        bc.iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?;
                }
            }
        }

        // -- Tier 3 Analysis --------------------------------------------------
        if let Some(ref t3) = r.tier3_analysis {
            writeln!(f)?;
            writeln!(f, "Tier 3 Analysis:")?;
            writeln!(f, "  all_counters_rangeable: {}", t3.all_counters_rangeable)?;
            writeln!(f, "  max_body_origins: {}", t3.max_body_origins)?;
            writeln!(f, "  max_instance_stride: {}", t3.max_instance_stride)?;

            // reachable_without_break
            let rwb_indices: Vec<usize> = t3
                .reachable_without_break
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                .collect();
            writeln!(
                f,
                "  reachable_without_break: [{}]",
                rwb_indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;

            // target_is_match
            let tim_indices: Vec<usize> = t3
                .target_is_match
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                .collect();
            writeln!(
                f,
                "  target_is_match: [{}]",
                tim_indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;

            // target_is_match_at_end
            let mae_indices: Vec<usize> = t3
                .target_is_match_at_end
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                .collect();
            writeln!(
                f,
                "  target_is_match_at_end: [{}]",
                mae_indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;

            // target_deferred_asserts (non-empty entries only)
            let tda_entries: Vec<(usize, &[StateIdx])> = t3
                .target_deferred_asserts
                .iter()
                .enumerate()
                .filter(|(_, da)| !da.is_empty())
                .map(|(i, da)| (i, da.as_ref()))
                .collect();
            if !tda_entries.is_empty() {
                writeln!(f, "  target_deferred_asserts:")?;
                for (i, asserts) in tda_entries {
                    let labels: Vec<String> = asserts
                        .iter()
                        .map(|&da| {
                            if let State::Assert { kind, .. } = r.states.0[da] {
                                format!("{}@{}", kind.label(), da)
                            } else {
                                format!("?@{da}")
                            }
                        })
                        .collect();
                    writeln!(f, "    state {i}: [{}]", labels.join(", "))?;
                }
            } else {
                writeln!(f, "  target_deferred_asserts: (none)")?;
            }

            // break_seeds (always shown, even when empty)
            if t3.break_seeds.is_empty() {
                writeln!(f, "  break_seeds: (none)")?;
            } else {
                writeln!(f, "  break_seeds:")?;
                for bs in t3.break_seeds.iter() {
                    let gates_str = if bs.deferred_asserts.is_empty() {
                        String::new()
                    } else {
                        let gates: Vec<String> = bs
                            .deferred_asserts
                            .iter()
                            .map(|&da| {
                                if let State::Assert { kind, .. } = r.states.0[da] {
                                    format!("{}@{}", kind.label(), da)
                                } else {
                                    format!("?@{da}")
                                }
                            })
                            .collect();
                        format!(" (gated by {})", gates.join(", "))
                    };
                    writeln!(
                        f,
                        "    trigger:c{} → seed c{} at origin:{}{}",
                        bs.trigger, bs.counter, bs.origin, gates_str
                    )?;
                }
            }

            // ci_origins (non-empty entries only)
            let non_empty: Vec<(usize, &[StateIdx])> = t3
                .ci_origins
                .iter()
                .enumerate()
                .filter(|(_, origins)| !origins.is_empty())
                .map(|(i, origins)| (i, origins.as_ref()))
                .collect();
            if !non_empty.is_empty() {
                writeln!(f, "  ci_origins:")?;
                for (i, origins) in non_empty {
                    writeln!(
                        f,
                        "    state {i}: [{}]",
                        origins
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )?;
                }
            }

            // targets (non-None entries only)
            let target_entries: Vec<(usize, &Tier3OriginKind)> = t3
                .targets
                .iter()
                .enumerate()
                .filter_map(|(i, t)| t.as_ref().map(|kind| (i, kind)))
                .collect();
            if !target_entries.is_empty() {
                writeln!(f, "  targets:")?;
                for (i, kind) in target_entries {
                    match kind {
                        Tier3OriginKind::Advance {
                            new_origins,
                            is_match_at_end,
                            is_match,
                        } => {
                            writeln!(
                                f,
                                "    state {i}: Advance → [{}] mae={is_match_at_end} im={is_match}",
                                new_origins
                                    .iter()
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )?;
                        }
                        Tier3OriginKind::Increment {
                            counter,
                            advance_origins,
                            min,
                            max,
                            continue_origins,
                            break_is_match,
                            break_is_match_at_end,
                            break_effects_id,
                        } => {
                            writeln!(
                                f,
                                "    state {i}: Increment(c{counter}, {{{min},{max}}}, {break_effects_id})",
                            )?;
                            writeln!(
                                f,
                                "      advance_origins: [{}]",
                                advance_origins
                                    .iter()
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )?;
                            writeln!(
                                f,
                                "      continue_origins: [{}]",
                                continue_origins
                                    .iter()
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            )?;
                            writeln!(
                                f,
                                "      break_is_match: {break_is_match}, \
                                 break_is_match_at_end: {break_is_match_at_end}",
                            )?;
                            // Break-path details come from the BreakEffects table.
                            let be = &t3.break_effects[break_effects_id.idx()];
                            let chain_str: String = be
                                .break_deferred_chain_ids
                                .iter()
                                .map(|c| c.to_string())
                                .collect::<Vec<_>>()
                                .join(", ");
                            writeln!(f, "      break_deferred_chains: [{chain_str}]")?;
                            let bcs_str: String = be
                                .break_consuming_states
                                .iter()
                                .map(|s| s.to_string())
                                .collect::<Vec<_>>()
                                .join(", ");
                            writeln!(f, "      break_consuming_states: [{bcs_str}]")?;
                            if !be.break_consuming_pure.is_empty() {
                                let bcp_str: String = be
                                    .break_consuming_pure
                                    .iter()
                                    .map(|s| s.to_string())
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                writeln!(f, "      break_consuming_pure: [{bcp_str}]")?;
                            }
                            if !be.break_consuming_deferred.is_empty() {
                                let bcd_str: String = be
                                    .break_consuming_deferred
                                    .iter()
                                    .map(|dt| {
                                        let a_str: String = dt
                                            .assert_states
                                            .iter()
                                            .map(|a| a.to_string())
                                            .collect::<Vec<_>>()
                                            .join(",");
                                        format!("{}(?@[{a_str}])", dt.origin)
                                    })
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                writeln!(f, "      break_consuming_deferred: [{bcd_str}]")?;
                            }
                        }
                    }
                }
            }

            // Typed effects compiled from target analysis.
            writeln!(f)?;
            writeln!(f, "  --- Typed Effects ---")?;
            fmt_target_effects(f, &t3.target_effects, &t3.assert_chain_arena, &r.states.0)?;
            fmt_origin_effects(f, &t3.origin_effects, &r.states.0)?;
        }

        Ok(())
    }
}
