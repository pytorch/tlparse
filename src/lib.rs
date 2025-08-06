use anyhow::{anyhow, bail};
use chrono::Datelike;
use fxhash::{FxHashMap, FxHashSet};
use md5::{Digest, Md5};
use std::ffi::{OsStr, OsString};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use regex::Regex;
use std::cell::RefCell;
use std::fs::{self, File};
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::time::Instant;
use tinytemplate::TinyTemplate;

use crate::parsers::default_parsers;
use crate::parsers::ParserOutput;
use crate::parsers::StructuredLogParser;
use crate::templates::*;
use crate::types::*;
pub mod parsers;
mod templates;
mod types;

pub use types::{
    DivergenceGroup, GraphAnalysis, GraphRuntime, RankMetaData, RuntimeAnalysis, RuntimeRankDetail,
};

#[derive(Debug)]
enum ParserResult {
    NoPayload,
    PayloadFilename(String),
}

pub struct ParseConfig {
    pub strict: bool,
    pub strict_compile_id: bool,
    pub custom_parsers: Vec<Box<dyn crate::parsers::StructuredLogParser>>,
    pub custom_header_html: String,
    pub verbose: bool,
    pub plain_text: bool,
    pub export: bool,
    pub inductor_provenance: bool,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            strict: false,
            strict_compile_id: false,
            custom_parsers: Vec::default(),
            custom_header_html: String::default(),
            verbose: false,
            plain_text: false,
            export: false,
            inductor_provenance: false,
        }
    }
}

fn maybe_remove_convert_frame_suffixes(frames: &mut Vec<FrameSummary>) {
    let all_target_frames = [
        [
            ("torch/_dynamo/convert_frame.py", "catch_errors"),
            ("torch/_dynamo/convert_frame.py", "_convert_frame"),
            ("torch/_dynamo/convert_frame.py", "_convert_frame_assert"),
        ],
        [
            ("torch/_dynamo/convert_frame.py", "__call__"),
            ("torch/_dynamo/convert_frame.py", "__call__"),
            ("torch/_dynamo/convert_frame.py", "__call__"),
        ],
    ];

    let len = frames.len();
    for target_frames in all_target_frames {
        if len >= target_frames.len() {
            let suffix = &frames[len - target_frames.len()..];
            if suffix
                .iter()
                .zip(target_frames.iter())
                .all(|(frame, target)| {
                    simplify_filename(unintern_str(frame.filename).as_ref()) == target.0
                        && frame.name == target.1
                })
            {
                frames.truncate(len - target_frames.len());
            }
        }
    }
}

fn add_unique_suffix(raw_filename: PathBuf, output_count: i32) -> PathBuf {
    if let Some(stem) = raw_filename.file_stem() {
        let mut r = OsString::new();
        r.push(stem);
        r.push(OsStr::new("_"));
        r.push(output_count.to_string());
        if let Some(e) = raw_filename.extension() {
            r.push(OsStr::new("."));
            r.push(e);
        };
        raw_filename.with_file_name(r)
    } else {
        raw_filename
    }
}

fn add_file_output(
    filename: PathBuf,
    content: String,
    output: &mut ParseOutput,
    compile_directory: &mut Vec<OutputFile>,
    output_count: &mut i32,
) {
    output.push((filename.clone(), content));
    let filename_str = filename.to_string_lossy().to_string();
    let suffix = if filename_str.contains("cache_miss") {
        "❌".to_string()
    } else if filename_str.contains("cache_hit") {
        "✅".to_string()
    } else if filename_str.contains("cache_bypass") {
        "❓".to_string()
    } else {
        "".to_string()
    };
    compile_directory.push(OutputFile {
        url: filename_str.clone(),
        name: filename_str,
        number: *output_count,
        suffix: suffix,
    });
    *output_count += 1;
}

fn run_parser<'t>(
    lineno: usize,
    parser: &Box<dyn StructuredLogParser + 't>,
    e: &Envelope,
    payload: &str,
    output_count: &mut i32,
    output: &mut ParseOutput,
    compile_directory: &mut Vec<OutputFile>,
    multi: &MultiProgress,
    stats: &mut Stats,
) -> ParserResult {
    let mut payload_filename = ParserResult::NoPayload;
    if let Some(md) = parser.get_metadata(&e) {
        let results = parser.parse(lineno, md, e.rank, &e.compile_id, &payload);
        match results {
            Ok(results) => {
                for parser_result in results {
                    match parser_result {
                        ParserOutput::File(raw_filename, out) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            add_file_output(filename, out, output, compile_directory, output_count);
                        }
                        ParserOutput::GlobalFile(filename, out) => {
                            add_file_output(filename, out, output, compile_directory, output_count);
                        }
                        ParserOutput::PayloadFile(raw_filename) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            payload_filename = ParserResult::PayloadFilename(
                                filename.to_string_lossy().to_string(),
                            );
                            add_file_output(
                                filename,
                                payload.to_string(),
                                output,
                                compile_directory,
                                output_count,
                            );
                        }
                        ParserOutput::PayloadReformatFile(raw_filename, formatter) => {
                            let filename = add_unique_suffix(raw_filename, *output_count);
                            match formatter(payload) {
                                Ok(formatted_content) => {
                                    payload_filename = ParserResult::PayloadFilename(
                                        filename.to_string_lossy().to_string(),
                                    );
                                    add_file_output(
                                        filename,
                                        formatted_content,
                                        output,
                                        compile_directory,
                                        output_count,
                                    );
                                }
                                Err(err) => {
                                    multi.suspend(|| {
                                        eprintln!(
                                            "Failed to format payload for {}: {}",
                                            filename.to_string_lossy(),
                                            err
                                        )
                                    });
                                    stats.fail_parser += 1;
                                }
                            }
                        }
                        ParserOutput::Link(name, url) => {
                            compile_directory.push(OutputFile {
                                url: url,
                                name: name,
                                number: *output_count,
                                suffix: "".to_string(),
                            });
                            *output_count += 1;
                        }
                    }
                }
            }
            Err(err) => match parser.name() {
                "dynamo_guards" => {
                    multi.suspend(|| eprintln!("Failed to parse guards json: {}", err));
                    stats.fail_dynamo_guards_json += 1;
                }
                name => {
                    multi.suspend(|| eprintln!("Parser {name} failed: {err}"));
                    stats.fail_parser += 1;
                }
            },
        }
    }
    payload_filename
}

fn directory_to_json(
    directory: &FxIndexMap<Option<CompileId>, Vec<OutputFile>>,
) -> serde_json::Value {
    let mut json_map = serde_json::Map::new();

    for (compile_id, output_files) in directory {
        let key = compile_id
            .as_ref()
            .map_or_else(|| "unknown".to_string(), |cid| cid.to_string());

        let artifacts: Vec<serde_json::Value> = output_files
            .iter()
            .map(|file| {
                serde_json::json!({
                    "url": file.url,
                    // Strip away any leading directory names, that will just be in the url path anyway
                    "name": file.name.split('/').last().unwrap_or(&file.name),
                    "number": file.number,
                    "suffix": file.suffix
                })
            })
            .collect();

        json_map.insert(key, serde_json::json!({"artifacts": artifacts}));
    }
    serde_json::Value::Object(json_map)
}

fn handle_guard(
    failure_type: &str,
    reason: &str,
    lineno: usize,
    e: &Envelope,
    payload: &str,
    output_count: &mut i32,
    output: &mut Vec<(PathBuf, String)>,
    compile_directory: &mut Vec<OutputFile>,
    multi: &MultiProgress,
    stats: &mut Stats,
    tt: &TinyTemplate,
    sym_expr_info_index: &RefCell<SymExprInfoIndex>,
    export_failures: &mut Vec<ExportFailure>,
) {
    let sym_expr_info_index_borrowed = sym_expr_info_index.borrow();
    let parser: Box<dyn StructuredLogParser> =
        Box::new(crate::parsers::PropagateRealTensorsParser {
            tt,
            sym_expr_info_index: &sym_expr_info_index_borrowed,
        });
    let _ = run_parser(
        lineno,
        &parser,
        e,
        payload,
        output_count,
        output,
        compile_directory,
        multi,
        stats,
    );

    let filename = format!(
        "symbolic_guard_information_{}.html",
        (*output_count - 1).to_string()
    );
    let compile_id_dir: PathBuf = e
        .compile_id
        .as_ref()
        .map_or(format!("unknown_{lineno}"), |cid| cid.as_directory_name())
        .into();
    let additional_info = format!(
        "Please click <a href='{}/{}'>here</a> for more information.",
        compile_id_dir.display(),
        filename,
    );

    export_failures.push(ExportFailure {
        failure_type: failure_type.to_string(),
        reason: reason.to_string(),
        additional_info,
    });
}

pub fn parse_path(path: &PathBuf, config: &ParseConfig) -> anyhow::Result<ParseOutput> {
    let strict = config.strict;
    if !path.is_file() {
        bail!("{} is not a file", path.display())
    }
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    // TODO: abstract out this spinner to not be part of the library
    // Instead, add a callback trait for CLIs to implement
    let multi = MultiProgress::new();
    let pb = multi.add(ProgressBar::new(file_size));
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} [{bytes_per_sec}] ({eta})")?
        .progress_chars("#>-"));
    let spinner = multi.add(ProgressBar::new_spinner());

    let reader = io::BufReader::new(file);

    let re_glog = Regex::new(concat!(
        r"(?<level>[VIWEC])(?<month>\d{2})(?<day>\d{2}) ",
        r"(?<hour>\d{2}):(?<minute>\d{2}):(?<second>\d{2}).(?<millisecond>\d{6}) ",
        r"(?<thread>\d+)",
        r"(?<pathname>[^:]+):(?<line>\d+)\] ",
        r"(?<payload>.)"
    ))?;

    // Helper functions to reduce repetitive serde_json::Value creation
    let make_string_value = |caps: &regex::Captures, name: &str| -> serde_json::Value {
        serde_json::Value::String(caps.name(name).unwrap().as_str().to_string())
    };

    let make_number_value = |caps: &regex::Captures, name: &str| -> serde_json::Value {
        let parsed: u64 = caps.name(name).unwrap().as_str().parse().unwrap();
        serde_json::Value::Number(serde_json::Number::from(parsed))
    };

    // Helper function to format timestamp as ISO-8601
    let format_timestamp = |caps: &regex::Captures| -> String {
        let month: u32 = caps.name("month").unwrap().as_str().parse().unwrap();
        let day: u32 = caps.name("day").unwrap().as_str().parse().unwrap();
        let hour: u32 = caps.name("hour").unwrap().as_str().parse().unwrap();
        let minute: u32 = caps.name("minute").unwrap().as_str().parse().unwrap();
        let second: u32 = caps.name("second").unwrap().as_str().parse().unwrap();
        let microsecond: u32 = caps.name("millisecond").unwrap().as_str().parse().unwrap();

        // Assume current year since glog doesn't include year
        let year = chrono::Utc::now().year();

        // Format as ISO-8601 with microsecond precision
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
            year, month, day, hour, minute, second, microsecond
        )
    };

    let mut stack_trie = StackTrieNode::default();
    let mut unknown_stack_trie = StackTrieNode::default();

    let mut stats = Stats::default();
    let _mod_count: FxHashMap<String, i32> = FxHashMap::default();

    let mut bytes_read: u64 = 0;

    // Some stuff for profiling
    let mut fastest_time = std::time::Duration::MAX;
    let mut slowest_time = std::time::Duration::ZERO;

    let mut expected_rank: Option<Option<u32>> = None;

    // Each entry is a compile id => (link, rendered name, output number)
    // For files, link and rendered name are the same
    // For links, you can specify a custom name for the link
    let mut directory: FxIndexMap<Option<CompileId>, Vec<OutputFile>> = FxIndexMap::default();

    let mut metrics_index: CompilationMetricsIndex = FxIndexMap::default();
    let stack_index: RefCell<StackIndex> = RefCell::new(FxHashMap::default());

    let symbolic_shape_specialization_index: RefCell<SymbolicShapeSpecializationIndex> =
        RefCell::new(FxHashMap::default());
    let guard_added_fast_index: RefCell<GuardAddedFastIndex> = RefCell::new(FxHashMap::default());
    let sym_expr_info_index: RefCell<SymExprInfoIndex> = RefCell::new(FxHashMap::default());

    // Store results in an output ParseOutput
    let mut output: ParseOutput = Vec::new();

    // Store raw.jsonl content (without payloads)
    let mut shortraw_content = String::new();

    let mut tt: TinyTemplate = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    if config.export {
        tt.add_template("index.html", TEMPLATE_EXPORT_INDEX)?;
        tt.add_template(
            "symbolic_guard_information.html",
            TEMPLATE_SYMBOLIC_GUARD_INFO,
        )?;
    } else {
        tt.add_template("index.html", TEMPLATE_INDEX)?;
        tt.add_template("failures_and_restarts.html", TEMPLATE_FAILURES_AND_RESTARTS)?;
        tt.add_template("dynamo_guards.html", TEMPLATE_DYNAMO_GUARDS)?;
        tt.add_template("compilation_metrics.html", TEMPLATE_COMPILATION_METRICS)?;
        tt.add_template(
            "bwd_compilation_metrics.html",
            TEMPLATE_BWD_COMPILATION_METRICS,
        )?;
        tt.add_template(
            "aot_autograd_backward_compilation_metrics.html",
            TEMPLATE_AOT_AUTOGRAD_BACKWARD_COMPILATION_METRICS,
        )?;
    }
    tt.add_template("provenance_tracking.html", TEMPLATE_PROVENANCE_TRACKING)?;

    let mut unknown_fields: FxHashSet<String> = FxHashSet::default();

    let mut output_count = 0;

    let mut breaks = RestartsAndFailuresContext {
        css: TEMPLATE_FAILURES_CSS,
        failures: Vec::new(),
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
    };

    let mut export_failures: Vec<ExportFailure> = Vec::new();

    // NB: Sometimes, the log output we get from Logarithm stutters with a blank line.
    // Filter them out, they're never valid (a blank line in payload will still be \t)
    let mut iter = reader
        .lines()
        .enumerate()
        .filter_map(|(i, l)| match l {
            // 1-indexed line numbers please
            Ok(l) if !l.is_empty() => Some((i + 1, l)),
            _ => None,
        })
        .peekable();

    let default_parsers = default_parsers(&tt, config);
    let mut all_parsers: Vec<&Box<dyn StructuredLogParser>> = default_parsers.iter().collect();
    let mut chromium_events: Vec<serde_json::Value> = Vec::new();
    all_parsers.extend(config.custom_parsers.iter());

    while let Some((lineno, line)) = iter.next() {
        bytes_read += line.len() as u64;
        pb.set_position(bytes_read);
        spinner.set_message(format!("{}", stats));
        //spinner.set_message(format!("{:?} {:?}", slowest_time, fastest_time));
        let start = Instant::now();

        let Some(caps) = re_glog.captures(&line) else {
            multi.suspend(|| eprintln!("Failed to parse glog prefix on line {}", lineno));
            stats.fail_glog += 1;
            continue;
        };

        let end = start.elapsed();
        if end < fastest_time {
            fastest_time = end;
        }
        if end > slowest_time {
            slowest_time = end;
        }
        let payload = &line[caps.name("payload").unwrap().start()..];
        let original_json_envelope = payload; // Store the original JSON envelope

        // Helper function to safely insert keys and detect conflicts
        let try_insert = |obj: &mut serde_json::Map<String, serde_json::Value>,
                          key: &str,
                          value: serde_json::Value,
                          multi: &MultiProgress,
                          stats: &mut Stats|
         -> bool {
            if obj.contains_key(key) {
                multi.suspend(|| {
                    eprintln!("Key conflict: '{}' already exists in JSON payload, skipping raw.jsonl JSONL conversion", key);
                });
                stats.fail_key_conflict += 1;
                false
            } else {
                obj.insert(key.to_string(), value);
                true
            }
        };

        // Create cleanup lambda to handle raw.jsonl writing as JSONL
        let write_to_shortraw = |shortraw_content: &mut String,
                                 payload_filename: Option<String>,
                                 multi: &MultiProgress,
                                 stats: &mut Stats| {
            match serde_json::from_str::<serde_json::Value>(original_json_envelope) {
                Ok(mut json_value) => {
                    if let Some(obj) = json_value.as_object_mut() {
                        // Try to add all log fields, abort on any conflict
                        let success = try_insert(
                            obj,
                            "timestamp",
                            serde_json::Value::String(format_timestamp(&caps)),
                            multi,
                            stats,
                        ) && try_insert(
                            obj,
                            "thread",
                            make_number_value(&caps, "thread"),
                            multi,
                            stats,
                        ) && try_insert(
                            obj,
                            "pathname",
                            make_string_value(&caps, "pathname"),
                            multi,
                            stats,
                        ) && try_insert(
                            obj,
                            "lineno",
                            make_number_value(&caps, "line"),
                            multi,
                            stats,
                        );

                        // Try to add payload filename if provided
                        let success = if let Some(payload_file) = payload_filename {
                            success
                                && try_insert(
                                    obj,
                                    "payload_filename",
                                    serde_json::Value::String(payload_file),
                                    multi,
                                    stats,
                                )
                        } else {
                            success
                        };

                        if !success {
                            // Drop line due to key conflict - don't write anything to maintain JSONL format
                            return;
                        }

                        // Output as JSONL
                        match serde_json::to_string(&json_value) {
                            Ok(jsonl_line) => {
                                shortraw_content.push_str(&jsonl_line);
                                shortraw_content.push('\n');
                            }
                            Err(e) => {
                                multi.suspend(|| {
                                    eprintln!("Failed to serialize JSON for raw.jsonl: {}", e);
                                });
                                stats.fail_json_serialization += 1;
                                // Drop line to maintain JSONL format - don't write anything
                            }
                        }
                    } else {
                        // Not a JSON object, drop line to maintain JSONL format
                        multi.suspend(|| {
                            eprintln!(
                                "JSON payload is not an object, dropping line from raw.jsonl"
                            );
                        });
                        stats.fail_json += 1;
                    }
                }
                Err(e) => {
                    // JSON parsing failed, drop line to maintain JSONL format
                    multi.suspend(|| {
                        eprintln!("Failed to parse JSON envelope for raw.jsonl: {}", e);
                    });
                    stats.fail_json += 1;
                }
            }
        };

        let e = match serde_json::from_str::<Envelope>(payload) {
            Ok(r) => r,
            Err(err) => {
                multi.suspend(|| {
                    eprintln!("Failed to parse metadata JSON: {}\n{:?}", payload, err);
                });
                stats.fail_json += 1;
                write_to_shortraw(&mut shortraw_content, None, &multi, &mut stats);
                continue;
            }
        };

        stats.unknown += e._other.len() as u64;

        for k in e._other.keys() {
            unknown_fields.insert(k.clone());
            if config.verbose {
                multi.suspend(|| eprintln!("Unknown field {}", k))
            }
        }

        if let Some((s, i)) = e.str {
            let mut intern_table = INTERN_TABLE.lock().unwrap();
            intern_table.insert(i, s);
            continue;
        };

        let mut payload = String::new();
        if let Some(ref expect) = e.has_payload {
            let mut first = true;
            while let Some((_payload_lineno, payload_line)) =
                iter.next_if(|(_, l)| l.starts_with('\t'))
            {
                // Careful! Distinguish between missing EOL and not
                if !first {
                    payload.push('\n');
                }
                first = false;
                payload.push_str(&payload_line[1..]);
            }
            let mut hasher = Md5::new();
            hasher.update(&payload);
            let hash = hasher.finalize();
            let mut expect_buf = [0u8; 16];
            if base16ct::lower::decode(expect, &mut expect_buf).is_ok() {
                if expect_buf != hash[..] {
                    // TODO: error log
                    stats.fail_payload_md5 += 1;
                }
            } else {
                stats.fail_payload_md5 += 1;
            }
        }

        match expected_rank {
            Some(rank) => {
                if rank != e.rank {
                    stats.other_rank += 1;
                    write_to_shortraw(&mut shortraw_content, None, &multi, &mut stats);
                    continue;
                }
            }
            None => {
                // Allow logs with no rank and then some rank to be processed
                // Logs with no rank may be initialized before distributed rank is set
                if e.rank.is_some() {
                    multi.suspend(|| {
                        eprintln!("Detected rank: {:?}", e.rank);
                    });
                    expected_rank = Some(e.rank);
                }
            }
        };

        stats.ok += 1;

        // Some runtime compile ids don't have attempts. Collapse these entries into
        // attempt 0 for now.
        let mut compile_id_entry = e.compile_id.clone();
        if let Some(ref mut entry) = compile_id_entry {
            if entry.frame_compile_id.is_some() && entry.attempt.is_none() {
                entry.attempt = Some(0);
            }
        }

        // TODO: output should be able to generate this without explicitly creating
        let compile_directory = directory.entry(compile_id_entry).or_default();

        let mut parser_payload_filename = ParserResult::NoPayload;
        for parser in &all_parsers {
            let result = run_parser(
                lineno,
                parser,
                &e,
                &payload,
                &mut output_count,
                &mut output,
                compile_directory,
                &multi,
                &mut stats,
            );
            // Take the last PayloadFilename entry as per the requirement
            if matches!(result, ParserResult::PayloadFilename(_)) {
                parser_payload_filename = result;
            }
        }

        if let Some(ref m) = e.compilation_metrics {
            let copied_directory = compile_directory.clone();
            let compile_id_dir: PathBuf = e
                .compile_id
                .as_ref()
                .map_or(format!("unknown_{lineno}"), |cid| cid.as_directory_name())
                .into();
            let parser: Box<dyn StructuredLogParser> =
                Box::new(crate::parsers::CompilationMetricsParser {
                    tt: &tt,
                    stack_index: &stack_index,
                    symbolic_shape_specialization_index: &symbolic_shape_specialization_index,
                    guard_added_fast_index: &guard_added_fast_index,
                    output_files: &copied_directory,
                    compile_id_dir: &compile_id_dir,
                });
            let result = run_parser(
                lineno,
                &parser,
                &e,
                &payload,
                &mut output_count,
                &mut output,
                compile_directory,
                &multi,
                &mut stats,
            );
            // Take the last PayloadFilename entry as per the requirement
            if matches!(result, ParserResult::PayloadFilename(_)) {
                parser_payload_filename = result;
            }

            // compilation metrics is always the last output, since it just ran
            let metrics_filename = format!(
                "compilation_metrics_{}.html",
                (output_count - 1).to_string(),
            );
            let id = e.compile_id.clone().map_or("(unknown) ".to_string(), |c| {
                format!(
                    "<a href='{}/{}'>{cid}</a> ",
                    compile_id_dir.display(),
                    metrics_filename,
                    cid = c,
                )
            });
            if let Some(rr) = m.restart_reasons.as_ref() {
                for restart in rr {
                    breaks.failures.push((
                        id.clone(),
                        format!("{}", FailureReason::Restart(restart.clone())),
                    ));
                }
            }
            if let Some(f) = m.fail_type.as_ref() {
                let reason = m
                    .fail_reason
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("Fail reason not found"))?;
                let user_frame_filename = m
                    .fail_user_frame_filename
                    .clone()
                    .unwrap_or(String::from("N/A"));
                let user_frame_lineno = m.fail_user_frame_lineno.unwrap_or(0);
                let failure_reason = FailureReason::Failure((
                    f.clone(),
                    reason.clone(),
                    user_frame_filename.clone(),
                    user_frame_lineno.clone(),
                ));
                breaks
                    .failures
                    .push((id.clone(), format!("{failure_reason}")));
            }
            let mut cid = e.compile_id.clone();
            if let Some(c) = cid.as_mut() {
                if let Some(_frame_id) = c.frame_compile_id {
                    // data migration for old logs that don't have attempt
                    c.attempt = Some(0);
                }
            }
            metrics_index.entry(cid).or_default().push(m.clone());
        }

        if config.export {
            if let Some(ref guard) = e.guard_added {
                if guard.prefix.as_deref() != Some("eval") {
                    write_to_shortraw(&mut shortraw_content, None, &multi, &mut stats);
                    continue;
                }
                let failure_type = "Guard Evaluated";

                let reason = format!(
                    "When exporting, the following guard was evaluated <code>{}</code>. This
                    might've resulted in a constraint violation error.",
                    guard.expr.clone().unwrap(),
                );

                handle_guard(
                    failure_type,
                    &reason,
                    lineno,
                    &e,
                    &payload,
                    &mut output_count,
                    &mut output,
                    compile_directory,
                    &multi,
                    &mut stats,
                    &tt,
                    &sym_expr_info_index,
                    &mut export_failures,
                );
            }

            if let Some(ref guard) = e.propagate_real_tensors_provenance {
                let failure_type = "Data Dependent Error";

                let reason = format!(
                    "When exporting, we were unable to figure out if the
                    expression <code>{}</code> always holds.<br> As a result, it
                    was specialized to evaluate to <code>{}</code>, and asserts
                    were inserted into the graph.",
                    guard.expr.clone().unwrap(),
                    guard.result.clone().unwrap()
                );

                handle_guard(
                    failure_type,
                    &reason,
                    lineno,
                    &e,
                    &payload,
                    &mut output_count,
                    &mut output,
                    compile_directory,
                    &multi,
                    &mut stats,
                    &tt,
                    &sym_expr_info_index,
                    &mut export_failures,
                );
            }

            if let Some(fake_kernel) = e.missing_fake_kernel {
                let failure_type = "Missing Fake Kernel";

                let reason = format!(
                    "<code>torch.ops.{}</code> is missing a fake kernel implementation",
                    fake_kernel.op.unwrap()
                );

                let additional_info = "Please refer to <a href='https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz'>this doc</a> for more detailed instructions on how to write a fake kernel.";

                export_failures.push(ExportFailure {
                    failure_type: failure_type.to_string(),
                    reason: reason,
                    additional_info: additional_info.to_string(),
                });
            }

            if let Some(fake_kernel) = e.mismatched_fake_kernel {
                let failure_type = "Mismatched Fake Kernel";

                let reason = format!(
                    "<code>torch.ops.{}</code> has a fake kernel implementation,
                    but it has incorrect behavior, based on the real kernel.<br>
                    The reason for the mismatch is: {}",
                    fake_kernel.op.unwrap(),
                    fake_kernel.reason.unwrap(),
                );

                let additional_info = "Please refer to <a href='https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz'>this doc</a> for more detailed instructions on how to write a fake kernel.";

                export_failures.push(ExportFailure {
                    failure_type: failure_type.to_string(),
                    reason: reason,
                    additional_info: additional_info.to_string(),
                });
            }

            if let Some(sym_expr_info) = e.expression_created {
                sym_expr_info_index
                    .borrow_mut()
                    .insert(sym_expr_info.result_id.unwrap(), sym_expr_info);
            }

            if let Some(unbacked_symbol) = e.create_unbacked_symbol {
                sym_expr_info_index.borrow_mut().insert(
                    unbacked_symbol.node_id.unwrap(),
                    SymExprInfoMetadata {
                        result: unbacked_symbol.symbol.clone(),
                        result_id: unbacked_symbol.node_id.clone(),
                        user_stack: unbacked_symbol.user_stack.clone(),
                        stack: unbacked_symbol.stack.clone(),
                        ..Default::default()
                    },
                );
            }
        }

        if let Some(stack) = e.stack {
            unknown_stack_trie.insert(stack.clone(), None);
        }

        if let Some(_) = e.chromium_event {
            chromium_events.push(serde_json::from_str(&payload)?);
        }

        if let Some(specialization) = e.symbolic_shape_specialization {
            symbolic_shape_specialization_index
                .borrow_mut()
                .entry(e.compile_id.clone())
                .or_default()
                .push(specialization);
        }
        if let Some(guard_added_fast) = e.guard_added_fast {
            guard_added_fast_index
                .borrow_mut()
                .entry(e.compile_id.clone())
                .or_default()
                .push(guard_added_fast)
        }

        if let Some(m) = e.dynamo_start {
            if let Some(mut stack) = m.stack {
                maybe_remove_convert_frame_suffixes(&mut stack);
                stack_index
                    .borrow_mut()
                    .insert(e.compile_id.clone(), stack.clone());
                stack_trie.insert(stack, e.compile_id.clone());
            };
        };

        // Handle payload file writing and determine final payload filename, but skip chromium events
        let final_payload_filename = match parser_payload_filename {
            ParserResult::PayloadFilename(filename) => Some(filename),
            ParserResult::NoPayload => {
                if let Some(ref expect) = e.has_payload {
                    // Only write payload file if no parser generated PayloadFile/PayloadReformatFile output and not a chromium event
                    if !payload.is_empty() && e.chromium_event.is_none() {
                        let hash_str = expect;
                        let payload_path = PathBuf::from(format!("payloads/{}.txt", hash_str));
                        output.push((payload_path, payload.clone()));
                        Some(format!("payloads/{}.txt", hash_str))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        };

        // Write to raw.jsonl with optional payload filename, but skip chromium events
        if e.chromium_event.is_none() {
            write_to_shortraw(
                &mut shortraw_content,
                final_payload_filename,
                &multi,
                &mut stats,
            );
        }
    }

    if config.export {
        let num_failures = export_failures.len();

        let exported_program_url = directory
            .values()
            .flatten()
            .find(|output_file| output_file.url.contains("exported_program"))
            .map(|output_file| output_file.url.clone());

        let index_context = ExportIndexContext {
            css: EXPORT_CSS,
            javascript: JAVASCRIPT,
            custom_header_html: config.custom_header_html.clone(),
            directory: directory
                .drain(..)
                .map(|(x, y)| (x.map_or("(unknown)".to_string(), |e| e.to_string()), y))
                .collect(),
            failures: export_failures,
            num_failures: num_failures,
            success: num_failures == 0,
            exported_program_url: exported_program_url.unwrap_or("".to_string()),
            qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        };

        output.push((
            PathBuf::from("index.html"),
            tt.render("index.html", &index_context)?,
        ));

        return Ok(output);
    }

    output.push((
        PathBuf::from("failures_and_restarts.html"),
        tt.render("failures_and_restarts.html", &breaks)?,
    ));
    pb.finish_with_message("done");
    spinner.finish();

    output.push((
        PathBuf::from("chromium_events.json"),
        serde_json::to_string_pretty(&chromium_events).unwrap(),
    ));

    eprintln!("{}", stats);
    if unknown_fields.len() > 0 {
        eprintln!(
            "Unknown fields: {:?} (consider updating tlparse to render these)",
            unknown_fields
        );
    }

    let has_unknown_compile_id = directory.contains_key(&None);

    let directory_names: Vec<String> = directory
        .iter()
        .map(|(x, _)| {
            x.as_ref()
                .map_or("(unknown)".to_string(), |e| e.as_directory_name())
        })
        .collect();
    output.push((
        PathBuf::from("compile_directory.json"),
        serde_json::to_string_pretty(&directory_to_json(&directory))?,
    ));
    let index_context = IndexContext {
        css: CSS,
        javascript: JAVASCRIPT,
        custom_header_html: config.custom_header_html.clone(),
        directory: directory
            .drain(..)
            .map(|(x, y)| (x.map_or("(unknown)".to_string(), |e| e.to_string()), y))
            .collect(),
        stack_trie_html: stack_trie
            .fmt(Some(&metrics_index), "Stack", false)
            .unwrap(),
        unknown_stack_trie_html: unknown_stack_trie
            .fmt(Some(&metrics_index), "Stack", false)
            .unwrap(),
        has_unknown_stack_trie: !unknown_stack_trie.is_empty(),
        num_breaks: breaks.failures.len(),
        has_chromium_events: !chromium_events.is_empty(),
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        has_inductor_provenance: config.inductor_provenance,
        directory_names: directory_names.clone(),
    };
    output.push((
        PathBuf::from("index.html"),
        tt.render("index.html", &index_context)?,
    ));

    output.push((PathBuf::from("raw.log"), fs::read_to_string(path)?));

    // Create string table from INTERN_TABLE as an array with nulls for missing indices
    let intern_table = INTERN_TABLE.lock().unwrap();
    let max_index = intern_table.keys().max().copied().unwrap_or(0) as usize;
    let mut string_table: Vec<Option<String>> = vec![None; max_index + 1];
    for (&index, value) in intern_table.iter() {
        string_table[index as usize] = Some(value.clone());
    }
    drop(intern_table); // Release the lock early

    // Serialize string table as JSON object
    let string_table_json = serde_json::json!({
        "string_table": string_table
    });
    let string_table_line = serde_json::to_string(&string_table_json)?;

    // Prepend string table to raw.jsonl content
    let mut final_shortraw_content =
        String::with_capacity(string_table_line.len() + 1 + shortraw_content.len());
    final_shortraw_content.push_str(&string_table_line);
    final_shortraw_content.push('\n');
    final_shortraw_content.push_str(&shortraw_content);

    output.push((PathBuf::from("raw.jsonl"), final_shortraw_content));

    // other_rank is included here because you should only have logs from one rank when
    // configured properly
    if strict
        && (stats.fail_glog
            + stats.fail_json
            + stats.fail_payload_md5
            + stats.other_rank
            + stats.fail_dynamo_guards_json
            + stats.fail_parser
            > 0)
    {
        // Report something went wrong
        return Err(anyhow!("Something went wrong"));
    }

    if config.strict_compile_id && has_unknown_compile_id {
        return Err(anyhow!("Some log entries did not have compile id"));
    }

    if config.inductor_provenance {
        // Helper function to get file content for a specific directory name
        fn get_file_content(
            output: &[(PathBuf, String)],
            filename_patterns: &[&str],
            directory_name: &str,
        ) -> String {
            // Try each pattern in order and return the first match found
            for pattern in filename_patterns {
                if let Some((_, content)) = output.iter().rev().find(|(path, _)| {
                    path.to_string_lossy()
                        .contains(&format!("{}/{}", directory_name, pattern))
                }) {
                    return content.clone();
                }
            }
            String::default()
        }

        // Generate HTML for each directory name
        for directory_name in &directory_names {
            let pre_grad_graph_content = get_file_content(
                &output,
                &["before_pre_grad_graph", "inductor_pre_grad_graph"],
                directory_name,
            );
            let post_grad_graph_content = get_file_content(
                &output,
                &["after_post_grad_graph", "inductor_post_grad_graph"],
                directory_name,
            );
            let output_code_content =
                get_file_content(&output, &["inductor_output_code"], directory_name);
            let aot_code_content =
                get_file_content(&output, &["inductor_aot_wrapper_code"], directory_name);
            let node_mappings_content = get_file_content(
                &output,
                &["inductor_provenance_tracking_node_mappings"],
                directory_name,
            );

            output.push((
                PathBuf::from(format!("provenance_tracking_{}.html", directory_name)),
                tt.render(
                    "provenance_tracking.html",
                    &ProvenanceContext {
                        css: PROVENANCE_CSS,
                        js: PROVENANCE_JS,
                        pre_grad_graph_content,
                        post_grad_graph_content,
                        output_code_content,
                        aot_code_content,
                        node_mappings_content,
                    },
                )?,
            ));
        }
    }

    Ok(output)
}

pub fn read_chromium_events_with_pid(
    path: &std::path::Path,
    rank_num: u32,
) -> anyhow::Result<Vec<serde_json::Value>> {
    use std::fs;

    if !path.exists() {
        return Ok(Vec::new());
    }

    let file_content = fs::read_to_string(path)?;

    match serde_json::from_str::<Vec<serde_json::Value>>(&file_content) {
        Ok(mut events) => {
            for event in &mut events {
                if let Some(obj) = event.as_object_mut() {
                    obj.insert("pid".to_string(), serde_json::json!(rank_num));
                }
            }
            Ok(events)
        }
        Err(_) => Ok(Vec::new()),
    }
}

pub fn generate_multi_rank_html(
    out_path: &PathBuf,
    sorted_ranks: Vec<String>,
    cfg: &ParseConfig,
    has_chromium_events: bool,
    show_desync_warning: bool,
    cache_divergence_groups: Vec<DivergenceGroup>,
    collective_divergence_groups: Vec<DivergenceGroup>,
    compile_id_divergence: bool,
    has_cache_divergence: bool,
    has_collective_divergence: bool,
    runtime_analysis: Option<RuntimeAnalysis>,
) -> anyhow::Result<(PathBuf, String)> {
    // Create the TinyTemplate instance for rendering the landing page.
    let mut tt = TinyTemplate::new();
    tt.add_formatter("format_unescaped", tinytemplate::format_unescaped);
    tt.add_template("multi_rank_index.html", TEMPLATE_MULTI_RANK_INDEX)?;

    let ctx = MultiRankContext {
        css: CSS,
        custom_header_html: &cfg.custom_header_html,
        num_ranks: sorted_ranks.len(),
        ranks: sorted_ranks,
        qps: TEMPLATE_QUERY_PARAM_SCRIPT,
        has_chromium_events,
        show_desync_warning,
        cache_divergence_groups,
        collective_divergence_groups,
        compile_id_divergence,
        has_cache_divergence,
        has_collective_divergence,
        runtime_analysis,
    };
    let html = tt.render("multi_rank_index.html", &ctx)?;
    let landing_page_path = out_path.join("index.html");

    Ok((landing_page_path, html))
}

fn prepare_and_validate_graphs(
    runtime_estimations: &[GraphRuntime],
) -> Option<(
    std::collections::HashMap<u32, Vec<(&str, f64)>>,
    Vec<u32>,
    usize,
)> {
    use std::collections::HashMap;

    let rank_graphs: HashMap<u32, Vec<(&str, f64)>> = runtime_estimations
        .iter()
        .map(|gr| {
            (
                gr.rank,
                &gr.graph,
                gr.ops.iter().map(|op| op.estimated_runtime_ns).sum::<f64>(),
            )
        })
        .fold(HashMap::new(), |mut acc, (rank, graph, runtime)| {
            acc.entry(rank).or_default().push((graph, runtime));
            acc
        });

    let max_graphs = rank_graphs.values().map(|graphs| graphs.len()).max()?;
    let min_graphs = rank_graphs.values().map(|graphs| graphs.len()).min()?;

    if max_graphs != min_graphs {
        return None; // Different number of graphs across ranks
    }

    let mut ranks: Vec<_> = rank_graphs.keys().copied().collect();
    ranks.sort_unstable();

    Some((rank_graphs, ranks, max_graphs))
}

fn compare_graph_runtimes(
    rank_graphs: std::collections::HashMap<u32, Vec<(&str, f64)>>,
    ranks: Vec<u32>,
    max_graphs: usize,
) -> Vec<GraphAnalysis> {
    (0..max_graphs)
        .filter_map(|index| {
            let runtimes: Vec<_> = ranks
                .iter()
                .map(|&rank| {
                    rank_graphs
                        .get(&rank)
                        .and_then(|g| g.get(index))
                        .map(|(graph_id, runtime)| (rank, *graph_id, *runtime))
                })
                .collect::<Option<Vec<_>>>()?;

            let (min_runtime, max_runtime, fastest_rank, slowest_rank) = runtimes.iter().fold(
                (f64::INFINITY, f64::NEG_INFINITY, 0_u32, 0_u32),
                |(min_rt, max_rt, fast_rank, slow_rank), &(rank, _, rt)| {
                    let (new_min_rt, new_fast) = if rt <= min_rt {
                        (rt, rank)
                    } else {
                        (min_rt, fast_rank)
                    };
                    let (new_max_rt, new_slow) = if rt >= max_rt {
                        (rt, rank)
                    } else {
                        (max_rt, slow_rank)
                    };
                    (new_min_rt, new_max_rt, new_fast, new_slow)
                },
            );

            let delta_ns = max_runtime - min_runtime;
            let graph_id = runtimes[0].1.to_string();

            Some(GraphAnalysis {
                graph_index: index,
                delta_ns: delta_ns.round(),
                delta_ms: (delta_ns / 1e6 * 1000.0).round() / 1000.0,
                rank_details: vec![
                    RuntimeRankDetail {
                        rank: fastest_rank,
                        graph_id: graph_id.clone(),
                        runtime_ms: (min_runtime / 1e6 * 1000.0).round() / 1000.0,
                    },
                    RuntimeRankDetail {
                        rank: slowest_rank,
                        graph_id,
                        runtime_ms: (max_runtime / 1e6 * 1000.0).round() / 1000.0,
                    },
                ],
            })
        })
        .collect()
}

pub fn analyze_graph_runtime_deltas(
    runtime_estimations: &[GraphRuntime],
) -> Option<RuntimeAnalysis> {
    let Some((rank_graphs, ranks, max_graphs)) = prepare_and_validate_graphs(runtime_estimations)
    else {
        return Some(RuntimeAnalysis {
            graphs: vec![],
            has_mismatched_graph_counts: true,
        });
    };

    let mut graphs = compare_graph_runtimes(rank_graphs, ranks, max_graphs);
    graphs.sort_by(|a, b| a.rank_details[0].graph_id.cmp(&b.rank_details[0].graph_id));

    Some(RuntimeAnalysis {
        graphs,
        has_mismatched_graph_counts: false,
    })
}
