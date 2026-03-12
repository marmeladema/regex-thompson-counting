use regex_syntax::ast::parse::ParserBuilder;
use regex_syntax::hir::translate::TranslatorBuilder;

use regex_thompson_counting::{MatcherMemory, Regex, RegexBuilder};

use std::io::{self, Write};
use std::process;

fn parse_pattern(pattern: &str, unroll_limit: Option<usize>) -> Regex {
    let ast = ParserBuilder::new()
        .build()
        .parse(pattern)
        .unwrap_or_else(|e| {
            eprintln!("error: failed to parse pattern: {e}");
            process::exit(1);
        });
    let hir = TranslatorBuilder::new()
        .unicode(false)
        .utf8(false)
        .dot_matches_new_line(true)
        .build()
        .translate(pattern, &ast)
        .unwrap_or_else(|e| {
            eprintln!("error: failed to translate pattern: {e}");
            process::exit(1);
        });
    let mut builder = RegexBuilder::default();
    if let Some(limit) = unroll_limit {
        builder.max_unroll_states(limit);
    }
    builder.build(&hir).unwrap_or_else(|e| {
        eprintln!("error: failed to compile pattern: {e}");
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
  dump  <pattern>                Dump compiled NFA states, counters, and analysis

Options:
  --format <text|json|debug>  Output format (default: text; debug uses Rust {{:#?}})
  --chunk-size <N>     Feed input in chunks of N bytes (default: entire input at once)
  --tier <0|1|2|3|4>   Force a specific execution tier (0=NFA, 1-4=DFA tiers)
  --unroll-limit <N>   Max NFA states for repetition unrolling (0=disable, default: 32)
  --dfa                Include tier-specific DFA analysis in dump output
  --debug              Print matcher state after each step
  -h, --help           Print this help message"
    );
}

enum Command {
    Info {
        pattern: String,
        unroll_limit: Option<usize>,
        format: Format,
    },
    Dot {
        pattern: String,
        unroll_limit: Option<usize>,
    },
    Match {
        pattern: String,
        inputs: Vec<String>,
        chunk_size: Option<usize>,
        tier: Option<u8>,
        unroll_limit: Option<usize>,
        debug: bool,
    },
    Dump {
        pattern: String,
        unroll_limit: Option<usize>,
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
    let mut unroll_limit: Option<usize> = None;
    let mut format = Format::Text;
    let mut debug = false;
    let mut dfa = false;
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

    match positional[0].as_str() {
        "info" => {
            if positional.len() != 2 {
                eprintln!("error: 'info' command takes exactly one pattern argument");
                process::exit(1);
            }
            Command::Info {
                pattern: positional[1].clone(),
                unroll_limit,
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
                unroll_limit,
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
                unroll_limit,
                debug,
            }
        }
        "dump" => {
            if positional.len() != 2 {
                eprintln!("error: 'dump' command takes exactly one pattern argument");
                process::exit(1);
            }
            Command::Dump {
                pattern: positional[1].clone(),
                unroll_limit,
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

fn run_info(pattern: &str, unroll_limit: Option<usize>, format: Format) {
    let regex = parse_pattern(pattern, unroll_limit);
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

fn run_dot(pattern: &str, unroll_limit: Option<usize>) {
    let regex = parse_pattern(pattern, unroll_limit);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    regex.to_dot(&mut out);
    out.flush().unwrap();
}

fn run_dump(pattern: &str, unroll_limit: Option<usize>, format: Format, dfa: bool) {
    let regex = parse_pattern(pattern, unroll_limit);
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
    unroll_limit: Option<usize>,
    debug: bool,
) {
    let regex = parse_pattern(pattern, unroll_limit);
    let mut memory = MatcherMemory::default();

    eprintln!("pattern: {pattern}");
    eprintln!("memory_size: {} bytes", regex.memory_size());
    if let Some(cs) = chunk_size {
        eprintln!("chunk_size: {cs}");
    }
    if let Some(t) = tier {
        eprintln!("tier: {t} (forced)");
    }
    eprintln!();

    let mut any_failed = false;
    for input in inputs {
        let bytes = input.as_bytes();
        let mut matcher = match tier {
            Some(t) => memory.matcher_for_tier(&regex, t).unwrap_or_else(|e| {
                eprintln!("error: {e}");
                process::exit(1);
            }),
            None => memory.matcher(&regex),
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
                    eprintln!("[after chunk({:?})] {:#?}", input, matcher);
                }
            }
            Some(cs) => {
                // Feed in chunks of cs bytes.
                for (i, chunk) in bytes.chunks(cs).enumerate() {
                    matcher.chunk(chunk);
                    if debug {
                        let chunk_str = String::from_utf8_lossy(chunk);
                        eprintln!(
                            "[after chunk #{} {:?} (bytes {}..{})] {:#?}",
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

fn main() {
    match parse_args() {
        Command::Info {
            pattern,
            unroll_limit,
            format,
        } => run_info(&pattern, unroll_limit, format),
        Command::Dot {
            pattern,
            unroll_limit,
        } => run_dot(&pattern, unroll_limit),
        Command::Match {
            pattern,
            inputs,
            chunk_size,
            tier,
            unroll_limit,
            debug,
        } => run_match(&pattern, &inputs, chunk_size, tier, unroll_limit, debug),
        Command::Dump {
            pattern,
            unroll_limit,
            format,
            dfa,
        } => run_dump(&pattern, unroll_limit, format, dfa),
    }
}
