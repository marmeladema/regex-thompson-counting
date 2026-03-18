use rethoc_engine::{MatcherMemory, Regex, RegexConfig};

use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::process;

fn parse_pattern(pattern: &str, config: RegexConfig) -> Regex {
    Regex::with_config(pattern, config).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        process::exit(1);
    })
}

/// Output format for the `info` and `dump` commands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    Text,
    Json,
    /// Rust `{:#?}` Debug output.
    Debug,
}

fn print_usage() {
    eprintln!(
        "\
Usage: rethoc [OPTIONS] <COMMAND>

Commands:
  info  <pattern>                Print diagnostic information about the compiled regex
  dot   <pattern>                Output DOT (Graphviz) representation of the NFA
  match <pattern> <input>...     Match pattern against one or more inputs
  grep  <pattern> [file]         Print matching lines from file (or stdin)
  dump  <pattern>                Dump compiled NFA states, counters, and analysis

Options:
  --format <text|json|debug>  Output format (default: text; debug uses Rust {{:#?}})
  --chunk-size <N>     Feed input in chunks of N bytes (default: entire input at once)
  --tier <0|1|2|3|4>   Force a specific execution tier (0=NFA, 1-4=DFA tiers)
  --bounded-gap        Force the bounded-gap engine (error if not eligible)
  --unroll-limit <N>   Max NFA states for repetition unrolling (0=disable, default: 32)
  --max-states <N>     Max estimated fully-unrolled states (default: 2048)
  --max-repetition <N> Max bounded repetition count (default: 1000)
  --optimize-hir <true|false>  Enable/disable HIR optimisations (default: true)
  --dfa                Include tier-specific DFA analysis in dump output
  --debug              Print matcher state after each step
  -q, --quiet          Suppress output (grep: exit code only)
  -h, --help           Print this help message"
    );
}

enum Command {
    Info {
        pattern: String,
        config: RegexConfig,
        format: Format,
    },
    Dot {
        pattern: String,
        config: RegexConfig,
    },
    Match {
        pattern: String,
        inputs: Vec<String>,
        chunk_size: Option<usize>,
        tier: Option<u8>,
        force_bounded_gap: bool,
        config: RegexConfig,
        debug: bool,
    },
    Grep {
        pattern: String,
        file: Option<String>,
        config: RegexConfig,
        quiet: bool,
    },
    Dump {
        pattern: String,
        config: RegexConfig,
        format: Format,
        dfa: bool,
    },
}

fn parse_args() -> Command {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        process::exit(1);
    }

    let mut chunk_size: Option<usize> = None;
    let mut tier: Option<u8> = None;
    let mut force_bounded_gap = false;
    let mut unroll_limit: Option<usize> = None;
    let mut max_estimated_states: Option<usize> = None;
    let mut max_repetition: Option<usize> = None;
    let mut format = Format::Text;
    let mut debug = false;
    let mut dfa = false;
    let mut optimize_hir_flag: Option<bool> = None;
    let mut quiet = false;
    let mut positional = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                process::exit(0);
            }
            "--chunk-size" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --chunk-size requires a value");
                    process::exit(1);
                }
                chunk_size = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --chunk-size must be a positive integer");
                    process::exit(1);
                }));
                if chunk_size == Some(0) {
                    eprintln!("error: --chunk-size must be > 0");
                    process::exit(1);
                }
            }
            "--tier" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --tier requires a value (0-4)");
                    process::exit(1);
                }
                let t = args[i].parse::<u8>().unwrap_or_else(|_| {
                    eprintln!("error: --tier must be 0, 1, 2, 3, or 4");
                    process::exit(1);
                });
                if t > 4 {
                    eprintln!("error: --tier must be 0, 1, 2, 3, or 4");
                    process::exit(1);
                }
                tier = Some(t);
            }
            "--bounded-gap" => {
                force_bounded_gap = true;
            }
            "--unroll-limit" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --unroll-limit requires a value");
                    process::exit(1);
                }
                unroll_limit = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --unroll-limit must be a non-negative integer");
                    process::exit(1);
                }));
            }
            "--max-states" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --max-states requires a value");
                    process::exit(1);
                }
                max_estimated_states = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --max-states must be a non-negative integer");
                    process::exit(1);
                }));
            }
            "--max-repetition" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --max-repetition requires a value");
                    process::exit(1);
                }
                max_repetition = Some(args[i].parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("error: --max-repetition must be a non-negative integer");
                    process::exit(1);
                }));
            }
            "--optimize-hir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --optimize-hir requires a value (true or false)");
                    process::exit(1);
                }
                optimize_hir_flag = Some(match args[i].as_str() {
                    "true" => true,
                    "false" => false,
                    other => {
                        eprintln!("error: --optimize-hir must be 'true' or 'false', got '{other}'");
                        process::exit(1);
                    }
                });
            }
            "-q" | "--quiet" => {
                quiet = true;
            }
            "--format" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --format requires a value (text, json, or debug)");
                    process::exit(1);
                }
                format = match args[i].as_str() {
                    "text" => Format::Text,
                    "json" => Format::Json,
                    "debug" => Format::Debug,
                    other => {
                        eprintln!(
                            "error: unknown format: {other} (expected 'text', 'json', or 'debug')"
                        );
                        process::exit(1);
                    }
                };
            }
            "--dfa" => {
                dfa = true;
            }
            "--debug" => {
                debug = true;
            }
            other if other.starts_with('-') => {
                eprintln!("error: unknown option: {other}");
                print_usage();
                process::exit(1);
            }
            _ => {
                positional.push(args[i].clone());
            }
        }
        i += 1;
    }

    if positional.is_empty() {
        print_usage();
        process::exit(1);
    }

    // Build config from parsed flags.
    let mut config = RegexConfig::default();
    if let Some(u) = unroll_limit {
        config.max_unroll_states = u;
    }
    if let Some(m) = max_estimated_states {
        config.max_estimated_states = m;
    }
    if let Some(r) = max_repetition {
        config.max_repetition = r;
    }
    if let Some(opt) = optimize_hir_flag {
        config.optimize_hir = opt;
    }

    match positional[0].as_str() {
        "info" => {
            if positional.len() != 2 {
                eprintln!("error: 'info' command takes exactly one pattern argument");
                process::exit(1);
            }
            Command::Info {
                pattern: positional[1].clone(),
                config,
                format,
            }
        }
        "dot" => {
            if positional.len() != 2 {
                eprintln!("error: 'dot' command takes exactly one pattern argument");
                process::exit(1);
            }
            Command::Dot {
                pattern: positional[1].clone(),
                config,
            }
        }
        "match" => {
            if positional.len() < 3 {
                eprintln!("error: 'match' command requires a pattern and at least one input");
                process::exit(1);
            }
            Command::Match {
                pattern: positional[1].clone(),
                inputs: positional[2..].to_vec(),
                chunk_size,
                tier,
                force_bounded_gap,
                config,
                debug,
            }
        }
        "grep" => {
            if positional.len() < 2 || positional.len() > 3 {
                eprintln!("error: 'grep' command takes a pattern and an optional file");
                process::exit(1);
            }
            Command::Grep {
                pattern: positional[1].clone(),
                file: positional.get(2).cloned(),
                config,
                quiet,
            }
        }
        "dump" => {
            if positional.len() != 2 {
                eprintln!("error: 'dump' command takes exactly one pattern argument");
                process::exit(1);
            }
            Command::Dump {
                pattern: positional[1].clone(),
                config,
                format,
                dfa,
            }
        }
        other => {
            eprintln!("error: unknown command: {other}");
            print_usage();
            process::exit(1);
        }
    }
}

fn run_info(pattern: &str, config: RegexConfig, format: Format) {
    let regex = parse_pattern(pattern, config);
    let info = regex.info();
    match format {
        Format::Text => {
            println!("Pattern: {pattern}");
            println!();
            println!("{info}");
        }
        Format::Json => {
            println!(
                "{}",
                serde_json::to_string(&info).expect("RegexInfo should serialize to JSON")
            );
        }
        Format::Debug => {
            println!("{:#?}", regex);
        }
    }
}

fn run_dot(pattern: &str, config: RegexConfig) {
    let regex = parse_pattern(pattern, config);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    regex.to_dot(&mut out);
    out.flush().unwrap();
}

fn run_dump(pattern: &str, config: RegexConfig, format: Format, dfa: bool) {
    let regex = parse_pattern(pattern, config);
    match format {
        Format::Text => {
            print!("{}", regex.dump(dfa));
        }
        Format::Debug => {
            println!("{:#?}", regex);
        }
        Format::Json => {
            eprintln!("error: JSON format is not supported for 'dump' (use 'info' instead)");
            process::exit(1);
        }
    }
}

fn run_match(
    pattern: &str,
    inputs: &[String],
    chunk_size: Option<usize>,
    tier: Option<u8>,
    force_bounded_gap: bool,
    config: RegexConfig,
    debug: bool,
) {
    let regex = parse_pattern(pattern, config);
    let mut memory = MatcherMemory::default();

    eprintln!("pattern: {pattern}");
    eprintln!("memory_size: {} bytes", regex.memory_size());
    if let Some(cs) = chunk_size {
        eprintln!("chunk_size: {cs}");
    }
    if tier.is_some() && force_bounded_gap {
        eprintln!("error: --tier and --bounded-gap are mutually exclusive");
        process::exit(1);
    }
    if let Some(t) = tier {
        eprintln!("tier: {t} (forced)");
    }
    if force_bounded_gap {
        eprintln!("engine: bounded-gap (forced)");
    }
    eprintln!();

    let mut any_failed = false;
    for input in inputs {
        let bytes = input.as_bytes();
        let mut matcher = if force_bounded_gap {
            memory.bounded_gap_matcher(&regex).unwrap_or_else(|| {
                eprintln!("error: pattern is not eligible for bounded-gap engine");
                process::exit(1);
            })
        } else {
            match tier {
                Some(t) => memory.matcher_for_tier(&regex, t).unwrap_or_else(|e| {
                    eprintln!("error: {e}");
                    process::exit(1);
                }),
                None => memory.matcher(&regex),
            }
        };

        if debug {
            eprintln!("--- input: {:?} ---", input);
            eprintln!("[init] {:#?}", matcher);
        }

        match chunk_size {
            None => {
                // Feed entire input at once.
                matcher.chunk(bytes);
                if debug {
                    eprintln!("[after chunk({:?})] {}", input, matcher);
                }
            }
            Some(cs) => {
                // Feed in chunks of cs bytes.
                for (i, chunk) in bytes.chunks(cs).enumerate() {
                    matcher.chunk(chunk);
                    if debug {
                        let chunk_str = String::from_utf8_lossy(chunk);
                        eprintln!(
                            "[after chunk #{} {:?} (bytes {}..{})] {}",
                            i,
                            chunk_str,
                            i * cs,
                            i * cs + chunk.len(),
                            matcher
                        );
                    }
                }
            }
        }

        let matched = matcher.finish();

        if matched {
            println!("  \x1b[32mMATCH\x1b[0m  {:?}", input);
        } else {
            println!("  \x1b[31mNO MATCH\x1b[0m  {:?}", input);
            any_failed = true;
        }
    }

    if any_failed {
        process::exit(1);
    }
}

fn run_grep(pattern: &str, file: Option<&str>, config: RegexConfig, quiet: bool) {
    let regex = parse_pattern(pattern, config);
    let mut memory = MatcherMemory::default();

    let reader: Box<dyn BufRead> = match file {
        Some(path) => {
            let f = File::open(path).unwrap_or_else(|e| {
                eprintln!("error: cannot open {path}: {e}");
                process::exit(1);
            });
            Box::new(BufReader::new(f))
        }
        None => Box::new(BufReader::new(io::stdin())),
    };

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut any_matched = false;

    for line_result in reader.lines() {
        let line = line_result.unwrap_or_else(|e| {
            eprintln!("error: reading input: {e}");
            process::exit(1);
        });
        let mut matcher = memory.matcher(&regex);
        matcher.chunk(line.as_bytes());
        if matcher.finish() {
            if !quiet {
                writeln!(out, "{}", line).unwrap();
            }
            any_matched = true;
            if quiet {
                break;
            }
        }
    }

    if !any_matched {
        process::exit(1);
    }
}

fn main() {
    match parse_args() {
        Command::Info {
            pattern,
            config,
            format,
        } => run_info(&pattern, config, format),
        Command::Dot { pattern, config } => run_dot(&pattern, config),
        Command::Match {
            pattern,
            inputs,
            chunk_size,
            tier,
            force_bounded_gap,
            config,
            debug,
        } => run_match(
            &pattern,
            &inputs,
            chunk_size,
            tier,
            force_bounded_gap,
            config,
            debug,
        ),
        Command::Grep {
            pattern,
            file,
            config,
            quiet,
        } => run_grep(&pattern, file.as_deref(), config, quiet),
        Command::Dump {
            pattern,
            config,
            format,
            dfa,
        } => run_dump(&pattern, config, format, dfa),
    }
}
