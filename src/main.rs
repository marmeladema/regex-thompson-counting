use regex_syntax::ast::parse::ParserBuilder;
use regex_syntax::hir::translate::TranslatorBuilder;

use regex_thompson_counting::{MatcherMemory, Regex, RegexBuilder};

use std::io::{self, Write};
use std::process;

fn parse_pattern(pattern: &str) -> Regex {
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
    builder.build(&hir).unwrap_or_else(|e| {
        eprintln!("error: failed to compile pattern: {e}");
        process::exit(1);
    })
}

fn print_usage() {
    eprintln!(
        "\
Usage: rethoc [OPTIONS] <COMMAND>

Commands:
  dot   <pattern>                Output DOT (Graphviz) representation of the NFA
  match <pattern> <input>...     Match pattern against one or more inputs

Options:
  --chunk-size <N>   Feed input in chunks of N bytes (default: entire input at once)
  --debug            Print matcher state after each step
  -h, --help         Print this help message"
    );
}

enum Command {
    Dot {
        pattern: String,
    },
    Match {
        pattern: String,
        inputs: Vec<String>,
        chunk_size: Option<usize>,
        debug: bool,
    },
}

fn parse_args() -> Command {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        process::exit(1);
    }

    let mut chunk_size: Option<usize> = None;
    let mut debug = false;
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
        "dot" => {
            if positional.len() != 2 {
                eprintln!("error: 'dot' command takes exactly one pattern argument");
                process::exit(1);
            }
            Command::Dot {
                pattern: positional[1].clone(),
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
                debug,
            }
        }
        other => {
            eprintln!("error: unknown command: {other}");
            print_usage();
            process::exit(1);
        }
    }
}

fn run_dot(pattern: &str) {
    let regex = parse_pattern(pattern);
    let stdout = io::stdout();
    let mut out = stdout.lock();
    regex.to_dot(&mut out);
    out.flush().unwrap();
}

fn run_match(pattern: &str, inputs: &[String], chunk_size: Option<usize>, debug: bool) {
    let regex = parse_pattern(pattern);
    let mut memory = MatcherMemory::default();

    eprintln!("pattern: {pattern}");
    eprintln!("memory_size: {} bytes", regex.memory_size());
    if let Some(cs) = chunk_size {
        eprintln!("chunk_size: {cs}");
    }
    eprintln!();

    let mut any_failed = false;
    for input in inputs {
        let bytes = input.as_bytes();
        let mut matcher = memory.matcher(&regex);

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
        Command::Dot { pattern } => run_dot(&pattern),
        Command::Match {
            pattern,
            inputs,
            chunk_size,
            debug,
        } => run_match(&pattern, &inputs, chunk_size, debug),
    }
}
