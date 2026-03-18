//! Diagnostic types for inspecting compiled regex internals.
//!
//! The main entry point is [`RegexInfo`], returned by [`Regex::info`](crate::Regex::info).
//! It provides a structured snapshot of memory layout, NFA state counts,
//! counter parameters, execution tier, and prefilter strategy.
//!
//! All types implement [`Serialize`](serde::Serialize) for JSON output and
//! [`Display`](std::fmt::Display) (on `RegexInfo`) for human-readable text.

use std::fmt;

use serde::Serialize;

// ---------------------------------------------------------------------------
// Sub-types
// ---------------------------------------------------------------------------

/// Information about a single bounded-repetition counter.
///
/// Each counter corresponds to a `{min,max}` repetition in the pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CounterInfo {
    /// Counter slot index (0-based).  May differ from the position in
    /// the array for patterns with nested counters.
    pub index: usize,
    /// Minimum number of iterations required for this repetition.
    pub min: usize,
    /// Maximum number of iterations allowed for this repetition.
    pub max: usize,
    /// Fixed number of bytes consumed per iteration of the counter
    /// body, or 0 for variable-length bodies.
    pub body_byte_length: usize,
}

/// Memory layout breakdown of a compiled [`Regex`](crate::Regex).
///
/// All sizes are in bytes.  The `total` field is the sum of all
/// sub-allocations (struct inline size + heap arrays).
#[derive(Debug, Clone, Serialize)]
pub struct MemoryInfo {
    /// Total memory footprint in bytes (inline + heap).
    pub total: usize,
    /// Bytes consumed by the NFA state array.
    pub states: usize,
    /// Bytes consumed by byte-class lookup tables.
    pub classes: usize,
    /// Bytes consumed by 256-entry byte-map tables.
    pub byte_tables: usize,
    /// Number of NFA states.
    pub num_states: usize,
    /// Size of one NFA state in bytes.
    pub state_size: usize,
    /// Number of byte-class lookup tables.
    pub num_classes: usize,
    /// Size of one byte-class table in bytes.
    pub class_size: usize,
    /// Number of 256-entry byte-map tables.
    pub num_byte_tables: usize,
    /// Size of one byte-map table in bytes.
    pub byte_table_size: usize,
}

/// Breakdown of NFA states by kind.
///
/// The total across all fields equals [`MemoryInfo::num_states`].
#[derive(Debug, Clone, Serialize)]
pub struct NfaStateBreakdown {
    /// Number of epsilon split states.
    pub split: usize,
    /// Number of single-byte match states (case-sensitive).
    #[serde(rename = "byte")]
    pub byte_: usize,
    /// Number of case-insensitive single-byte match states.
    pub byte_ci: usize,
    /// Number of byte-class match states.
    pub byte_class: usize,
    /// Number of byte-table match states (256-entry lookup).
    pub byte_table: usize,
    /// Number of assertion states (anchors, word boundaries, etc.).
    pub assert: usize,
    /// Number of counter-instance (initialization) states.
    pub counter_instance: usize,
    /// Number of counter-increment (loop/break) states.
    pub counter_increment: usize,
    /// Number of match (accept) states.
    #[serde(rename = "match")]
    pub match_: usize,
}

/// Execution tier selected for this pattern.
///
/// Higher tiers handle more complex patterns but may be slower.
/// A tier-1 pattern has no counters and uses a plain lazy DFA.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionInfo {
    /// Numeric tier: 0 = NFA only, 1–4 = DFA tiers.
    ///
    /// When a specialisation is active, this reports the tier that
    /// would have been selected without the specialisation.
    pub tier: u8,
    /// Human-readable description of the selected tier.
    pub tier_name: String,
    /// Optional specialisation override (e.g. `"BoundedGap"`).
    ///
    /// When present, the pattern is executed by a specialised engine
    /// rather than the generic tier indicated by [`tier`](Self::tier).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub specialization: Option<String>,
}

/// Start-closure diagnostics.
///
/// The start closure is the set of byte-consuming states reachable
/// from the start state without consuming any input.  A non-empty
/// start closure enables fast-path re-seeding in the matcher.
#[derive(Debug, Clone, Serialize)]
pub struct StartClosureInfo {
    /// Number of consuming states in the start closure.
    pub len: usize,
    /// Whether the empty string matches (start closure reaches Match).
    pub matches_empty: bool,
}

// ---------------------------------------------------------------------------
// Top-level info type
// ---------------------------------------------------------------------------

/// Diagnostic snapshot of a compiled [`Regex`](crate::Regex).
///
/// Returned by [`Regex::info`](crate::Regex::info).  Implements
/// [`Display`](fmt::Display) for human-readable text output and
/// [`Serialize`] for JSON output.
///
/// # Sections
///
/// | Field              | Description                                       |
/// |--------------------|---------------------------------------------------|
/// | `memory`           | Allocation sizes and layout                       |
/// | `nfa_states`       | Per-kind NFA state counts                         |
/// | `counters`         | Bounded-repetition counter details                |
/// | `assert_kinds`     | Assertion kinds present in the pattern             |
/// | `deferred_assertions` | Assertions resolved at DFA transition time     |
/// | `byte_classes`     | DFA alphabet stride (equivalence classes)         |
/// | `execution`        | Which DFA tier was selected                       |
/// | `start_closure`    | Reachable start states                            |
/// | `prefilter`        | Literal acceleration strategy                     |
#[derive(Debug, Clone, Serialize)]
pub struct RegexInfo {
    /// Memory layout breakdown.
    pub memory: MemoryInfo,
    /// NFA state counts by kind.
    pub nfa_states: NfaStateBreakdown,
    /// Bounded-repetition counters (empty if pattern is counter-free).
    pub counters: Vec<CounterInfo>,
    /// Distinct assertion kinds used (e.g. `"StartLine"`, `"WordAscii"`).
    pub assert_kinds: Vec<String>,
    /// Number of deferred assertions resolved at DFA transition time.
    pub deferred_assertions: usize,
    /// Number of byte equivalence classes (DFA alphabet stride).
    /// 256 means equivalence classes are disabled.
    pub byte_classes: usize,
    /// Execution tier information.
    pub execution: ExecutionInfo,
    /// Start-closure diagnostics.
    pub start_closure: StartClosureInfo,
    /// Prefilter description (e.g. `"memchr1('a')"` or `"none"`).
    pub prefilter: String,
}

// ---------------------------------------------------------------------------
// Display impl (text output)
// ---------------------------------------------------------------------------

impl fmt::Display for RegexInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Memory
        writeln!(f, "Memory")?;
        writeln!(f, "  total:       {} bytes", self.memory.total)?;
        writeln!(
            f,
            "  states:      {} bytes ({} states \u{00d7} {} bytes/state)",
            self.memory.states, self.memory.num_states, self.memory.state_size
        )?;
        writeln!(
            f,
            "  classes:     {} bytes ({} tables \u{00d7} {} bytes/table)",
            self.memory.classes, self.memory.num_classes, self.memory.class_size
        )?;
        writeln!(
            f,
            "  byte_tables: {} bytes ({} tables \u{00d7} {} bytes/table)",
            self.memory.byte_tables, self.memory.num_byte_tables, self.memory.byte_table_size
        )?;

        // NFA state breakdown
        writeln!(f)?;
        writeln!(f, "NFA states: {}", self.memory.num_states)?;
        writeln!(f, "  Split:            {}", self.nfa_states.split)?;
        writeln!(f, "  Byte:             {}", self.nfa_states.byte_)?;
        writeln!(f, "  ByteCI:           {}", self.nfa_states.byte_ci)?;
        writeln!(f, "  ByteClass:        {}", self.nfa_states.byte_class)?;
        writeln!(f, "  ByteTable:        {}", self.nfa_states.byte_table)?;
        writeln!(f, "  Assert:           {}", self.nfa_states.assert)?;
        writeln!(
            f,
            "  CounterInstance:  {}",
            self.nfa_states.counter_instance
        )?;
        writeln!(
            f,
            "  CounterIncrement: {}",
            self.nfa_states.counter_increment
        )?;
        writeln!(f, "  Match:            {}", self.nfa_states.match_)?;

        // Counters
        if !self.counters.is_empty() {
            writeln!(f)?;
            writeln!(f, "Counters: {}", self.counters.len())?;
            for c in &self.counters {
                if c.body_byte_length > 0 {
                    writeln!(
                        f,
                        "  counter[{}]: {{{},{}}}, body_len={}",
                        c.index, c.min, c.max, c.body_byte_length
                    )?;
                } else {
                    writeln!(
                        f,
                        "  counter[{}]: {{{},{}}}, body_len=variable",
                        c.index, c.min, c.max
                    )?;
                }
            }
        }

        // Assertions
        if !self.assert_kinds.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "Assertion kinds: {} (across {} state{})",
                self.assert_kinds.len(),
                self.nfa_states.assert,
                if self.nfa_states.assert == 1 { "" } else { "s" }
            )?;
            for k in &self.assert_kinds {
                writeln!(f, "  {k}")?;
            }
        }

        // Deferred assertions
        if self.deferred_assertions > 0 {
            writeln!(f)?;
            writeln!(
                f,
                "Deferred assertions: {} (resolved at DFA transition time)",
                self.deferred_assertions
            )?;
        }

        // Byte equivalence classes
        writeln!(f)?;
        if self.byte_classes < 256 {
            writeln!(
                f,
                "Byte equivalence classes: {} (DFA stride, vs 256 raw)",
                self.byte_classes
            )?;
        } else {
            writeln!(f, "Byte equivalence classes: disabled (stride=256)")?;
        }

        // Execution tier
        writeln!(f)?;
        if let Some(spec) = &self.execution.specialization {
            writeln!(f, "Execution: {} specialization", spec)?;
            writeln!(f, "  fallback: {}", self.execution.tier_name)?;
        } else {
            writeln!(f, "Execution: {}", self.execution.tier_name)?;
        }

        // Start closure
        writeln!(f)?;
        writeln!(
            f,
            "Start closure: {} consuming states (precomputed: {})",
            self.start_closure.len,
            self.start_closure.len > 0
        )?;
        writeln!(
            f,
            "Start matches empty: {}",
            self.start_closure.matches_empty
        )?;
        write!(f, "Prefilter: {}", self.prefilter)?;
        Ok(())
    }
}
