use clap::Parser;

use anyhow::{bail, Context};
use std::fs;
use std::path::PathBuf;

use fxhash::{FxHashMap, FxHashSet};
use tlparse::{
    analyze_graph_runtime_deltas, generate_multi_rank_html, parse_path,
    read_chromium_events_with_pid, ArtifactFlags, Diagnostics, DivergenceFlags, DivergenceGroup,
    ParseConfig, RankMetaData,
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    path: PathBuf,
    /// Parse most recent log
    #[arg(long)]
    latest: bool,
    /// Output directory, defaults to `tl_out`
    #[arg(short, default_value = "tl_out")]
    out: PathBuf,
    /// Delete out directory if it already exists
    #[arg(long)]
    overwrite: bool,
    /// Return non-zero exit code if unrecognized log lines are found.  Mostly useful for unit
    /// testing.
    #[arg(long)]
    strict: bool,
    /// Return non-zero exit code if some log lines do not have associated compile id.  Used for
    /// unit testing
    #[arg(long)]
    strict_compile_id: bool,
    /// Don't open browser at the end
    #[arg(long)]
    no_browser: bool,
    /// Some custom HTML to append to the top of report
    #[arg(long, default_value = "")]
    custom_header_html: String,
    /// Be more chatty
    #[arg(short, long)]
    verbose: bool,
    /// Some parsers will write output as rendered html for prettier viewing.
    /// Enabiling this option will enforce output as plain text for easier diffing
    #[arg(short, long)]
    plain_text: bool,
    /// For export specific logs
    #[arg(short, long)]
    export: bool,
    /// For inductor provenance tracking highlighter
    #[arg(short, long)]
    inductor_provenance: bool,
    /// Parse all ranks and create a unified multi-rank report
    #[arg(long)]
    all_ranks_html: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Early validation of incompatible flags
    if cli.all_ranks_html && cli.latest {
        bail!("--latest cannot be used with --all-ranks-html");
    }

    let path = if cli.latest {
        let input_path = cli.path;
        // Path should be a directory
        if !input_path.is_dir() {
            bail!(
                "Input path {} is not a directory (required when using --latest)",
                input_path.display()
            );
        }

        let last_modified_file = std::fs::read_dir(&input_path)
            .with_context(|| format!("Couldn't access directory {}", input_path.display()))?
            .flatten()
            .filter(|f| f.metadata().unwrap().is_file())
            .max_by_key(|x| x.metadata().unwrap().modified().unwrap());

        let Some(last_modified_file) = last_modified_file else {
            bail!("No files found in directory {}", input_path.display());
        };
        last_modified_file.path()
    } else {
        cli.path
    };

    let config = ParseConfig {
        strict: cli.strict,
        strict_compile_id: cli.strict_compile_id,
        custom_parsers: Vec::new(),
        custom_header_html: cli.custom_header_html,
        verbose: cli.verbose,
        plain_text: cli.plain_text,
        export: cli.export,
        inductor_provenance: cli.inductor_provenance,
    };

    if cli.all_ranks_html {
        handle_all_ranks(&config, path, cli.out, cli.overwrite, !cli.no_browser)?;
    } else {
        handle_one_rank(
            &config,
            path,
            cli.latest,
            cli.out,
            !cli.no_browser,
            cli.overwrite,
        )?;
    }
    Ok(())
}

/// Create the output directory
fn setup_output_directory(out_path: &PathBuf, overwrite: bool) -> anyhow::Result<()> {
    if out_path.exists() {
        if !overwrite {
            bail!(
                "Directory {} already exists; pass --overwrite to replace it or use -o OUTDIR",
                out_path.display()
            );
        }
        fs::remove_dir_all(&out_path)?;
    }
    fs::create_dir_all(&out_path)?;
    Ok(())
}

/// Parse a log file and write the rendered artefacts into `output_dir`.
fn parse_and_write_output(
    config: &ParseConfig,
    log_path: &PathBuf,
    output_dir: &PathBuf,
) -> anyhow::Result<PathBuf> {
    let output = parse_path(log_path, config)?;

    for (filename, content) in output {
        let out_path = output_dir.join(&filename);
        if let Some(dir) = out_path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(out_path, content)?;
    }
    Ok(output_dir.join("index.html"))
}

fn handle_one_rank(
    cfg: &ParseConfig,
    input_path: PathBuf,
    latest: bool,
    out_dir: PathBuf,
    open_browser: bool,
    overwrite: bool,
) -> anyhow::Result<()> {
    // Resolve which log file we should parse
    let log_path = if latest {
        if !input_path.is_dir() {
            bail!(
                "Input path {} is not a directory (required with --latest)",
                input_path.display()
            );
        }
        std::fs::read_dir(input_path)?
            .flatten()
            .filter(|e| e.metadata().ok().map_or(false, |m| m.is_file()))
            .max_by_key(|e| e.metadata().unwrap().modified().unwrap())
            .map(|e| e.path())
            .context("No files found in directory for --latest")?
    } else {
        input_path.clone()
    };

    setup_output_directory(&out_dir, overwrite)?;
    let main_output_file = parse_and_write_output(cfg, &log_path, &out_dir)?;

    if open_browser {
        opener::open(&main_output_file)?;
    }
    Ok(())
}

fn handle_all_ranks(
    cfg: &ParseConfig,
    path: PathBuf,
    out_path: PathBuf,
    overwrite: bool,
    open_browser: bool,
) -> anyhow::Result<()> {
    let input_dir = path;
    if !input_dir.is_dir() {
        bail!(
            "Input path {} must be a directory when using --all-ranks-html",
            input_dir.display()
        );
    }

    setup_output_directory(&out_path, overwrite)?;

    // Discover rank log files
    let rank_logs: Vec<_> = std::fs::read_dir(&input_dir)?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_file() {
                return None;
            }
            let filename = path.file_name()?.to_str()?;
            filename
                .strip_prefix("dedicated_log_torch_trace_rank_")?
                .strip_suffix(".log")?
                .split('_')
                .next()?
                .parse::<u32>()
                .ok()
                .map(|rank_num| (path.clone(), rank_num))
        })
        .collect();

    if rank_logs.is_empty() {
        bail!(
            "No rank log files found in directory {}",
            input_dir.display()
        );
    }

    // Extract rank numbers, sort numerically, then convert to strings for HTML generation
    let mut rank_nums: Vec<u32> = rank_logs.iter().map(|(_, rank)| *rank).collect();
    rank_nums.sort_unstable();
    let sorted_ranks: Vec<String> = rank_nums.iter().map(|r| r.to_string()).collect();
    let mut all_chromium_events: Vec<serde_json::Value> = Vec::new();
    let mut rank_metadata: Vec<RankMetaData> = Vec::new();

    for (log_path, rank_num) in rank_logs {
        let subdir = out_path.join(format!("rank_{rank_num}"));
        println!("Processing rank {rank_num} → {}", subdir.display());
        let chromium_events_path = subdir.join("chromium_events.json");
        let compile_dir_json = subdir.join("compile_directory.json");

        handle_one_rank(cfg, log_path, false, subdir, false, overwrite)?;

        // extract compile IDs and cache sequence from compile_directory.json
        let mut compile_ids: FxHashSet<String> = FxHashSet::default();
        let content = fs::read_to_string(&compile_dir_json)?;
        let mut artifact_entries: Vec<(u64, String)> = Vec::new();

        if let Ok(serde_json::Value::Object(map)) =
            serde_json::from_str::<serde_json::Value>(&content)
        {
            for (key, val) in map.iter() {
                if key != "unknown" && !key.starts_with("unknown_") {
                    compile_ids.insert(key.clone());
                }
                if let Some(arr) = val.get("artifacts").and_then(|v| v.as_array()) {
                    for art in arr {
                        let suffix = art.get("suffix").and_then(|s| s.as_str()).unwrap_or("");
                        if suffix.is_empty() {
                            continue;
                        }
                        if let Some(num) = art.get("number").and_then(|n| n.as_u64()) {
                            artifact_entries.push((num, suffix.to_string()));
                        }
                    }
                }
            }
        }

        artifact_entries.sort_by_key(|(n, _)| *n);
        let cache_sequence: String = artifact_entries.into_iter().map(|(_, s)| s).collect();

        rank_metadata.push(RankMetaData {
            rank: rank_num,
            compile_ids,
            cache_sequence,
        });

        // collect chromium events for each rank
        if chromium_events_path.exists() {
            let events = read_chromium_events_with_pid(&chromium_events_path, rank_num)?;
            all_chromium_events.extend(events);
        }
    }

    // Determine if there is any divergence in compile IDs across ranks
    let compile_id_divergence = if let Some(first) = rank_metadata.first() {
        rank_metadata
            .iter()
            .any(|md| md.compile_ids != first.compile_ids)
    } else {
        false
    };

    // Group ranks by their cache hit/miss sequence
    let cache_seq_groups: FxHashMap<String, Vec<u32>> =
        rank_metadata
            .into_iter()
            .fold(FxHashMap::default(), |mut acc, md| {
                acc.entry(md.cache_sequence).or_default().push(md.rank);
                acc
            });

    // Build groups describing cache hit/miss patterns per rank
    let cache_divergence_groups: Vec<DivergenceGroup> = if cache_seq_groups.len() > 1 {
        cache_seq_groups
            .iter()
            .map(|(seq, ranks_vec)| {
                let mut sorted_ranks = ranks_vec.clone();
                sorted_ranks.sort_unstable();
                DivergenceGroup {
                    sequence: seq.clone(),
                    ranks: sorted_ranks
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // combine chromium events from all ranks
    if !all_chromium_events.is_empty() {
        let combined_chromium_path = out_path.join("chromium_events.json");
        let combined_events_json = serde_json::to_string_pretty(&all_chromium_events)?;
        fs::write(combined_chromium_path, combined_events_json)?;
    }

    // Process runtime estimations from all ranks
    let runtime_estimations = tlparse::parsers::read_runtime_estimations(&out_path, &rank_nums)?;
    if !runtime_estimations.is_empty() {
        let runtime_path = out_path.join("runtime_estimations.json");
        fs::write(
            &runtime_path,
            serde_json::to_string_pretty(&runtime_estimations)?,
        )?;
        println!("Runtime estimations: {}", runtime_path.display());

        // Generate runtime trace events in a single pass
        let mut runtime_events: Vec<serde_json::Value> = Vec::new();
        let mut pid_set: FxHashSet<u32> = FxHashSet::default();
        let mut thread_names: FxHashMap<(u32, u32), String> = FxHashMap::default();

        // Concise, deterministic 32-bit TID from (rank, graph)
        let calc_tid = |rank: u32, graph: &str| -> u32 {
            use std::hash::{Hash, Hasher};
            let mut h = fxhash::FxHasher::default();
            (rank, graph).hash(&mut h);
            (h.finish() & 0xFFFF_FFFF) as u32
        };

        for gr in &runtime_estimations {
            pid_set.insert(gr.rank);
            let tid = calc_tid(gr.rank, &gr.graph);
            thread_names
                .entry((gr.rank, tid))
                .or_insert_with(|| gr.graph.clone());

            let mut time_offset_us: u64 = 0;
            for op in &gr.ops {
                let dur_us = (op.estimated_runtime_ns / 1000.0).ceil().max(1.0) as u64;
                runtime_events.push(serde_json::json!({
                    "name": op.name,
                    "ph": "X",
                    "ts": time_offset_us,
                    "dur": dur_us,
                    "pid": gr.rank,
                    "tid": tid,
                    "cat": "runtime",
                    "args": {
                        "graph": gr.graph,
                        "rank": gr.rank,
                        "runtime_ns": op.estimated_runtime_ns as u64
                    }
                }));
                time_offset_us += dur_us;
            }
        }

        let mut all_events: Vec<serde_json::Value> = runtime_events;

        // Emit process (rank) metadata in ascending pid order
        let mut pids: Vec<u32> = pid_set.into_iter().collect();
        pids.sort_unstable();
        for pid in pids.into_iter() {
            all_events.extend([
                serde_json::json!({
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "args": {"name": format!("Rank {}", pid)}
                }),
                serde_json::json!({
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "args": {"sort_index": pid as i64}
                }),
            ]);
        }

        // Emit thread names sorted by graph name within each pid
        let mut threads_by_pid: FxHashMap<u32, Vec<(u32, String)>> = FxHashMap::default();
        for ((pid, tid), graph_name) in thread_names.into_iter() {
            threads_by_pid
                .entry(pid)
                .or_default()
                .push((tid, graph_name));
        }
        let mut pids_for_threads: Vec<u32> = threads_by_pid.keys().copied().collect();
        pids_for_threads.sort_unstable();
        for pid in pids_for_threads {
            let entries = threads_by_pid.remove(&pid).unwrap_or_default();
            for (idx, (tid, graph_name)) in entries.into_iter().enumerate() {
                all_events.extend([
                    serde_json::json!({
                        "name": "thread_name",
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "args": {"name": format!("graph {}", graph_name)}
                    }),
                    serde_json::json!({
                        "name": "thread_sort_index",
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "args": {"sort_index": idx as i64}
                    }),
                ]);
            }
        }

        fs::write(
            out_path.join("chromium_trace_with_runtime.json"),
            serde_json::to_string_pretty(&all_events)?,
        )?;
    }

    // Analyze graph runtime deltas across ranks
    let runtime_analysis = if !runtime_estimations.is_empty() {
        analyze_graph_runtime_deltas(&runtime_estimations)
    } else {
        None
    };

    // Process collective schedules from all ranks
    let collective_schedules = tlparse::parsers::read_collective_schedules(&out_path, &rank_nums)?;
    if !collective_schedules.is_empty() {
        let schedules_path = out_path.join("collective_schedules.json");
        fs::write(
            &schedules_path,
            serde_json::to_string_pretty(&collective_schedules)?,
        )?;
        println!("Collective schedules: {}", schedules_path.display());
    }

    // Process tensor meta fingerprints from all ranks
    let tensor_meta = tlparse::parsers::read_tensor_meta_fingerprints(&out_path, &rank_nums)?;
    let mut tensor_meta_groups: FxHashMap<String, Vec<u32>> = FxHashMap::default();
    if !tensor_meta.is_empty() {
        use std::collections::HashMap;
        // rank -> sorted list of (graph_id, fingerprint)
        let mut by_rank: HashMap<u32, Vec<(String, String)>> = HashMap::new();
        for tm in &tensor_meta {
            by_rank
                .entry(tm.rank)
                .or_default()
                .push((tm.graph.clone(), tm.fingerprint.clone()));
        }
        for (&rank, entries) in &mut by_rank {
            // sort by graph id to make cross-rank concatenation consistent
            let mut entries_sorted = entries.clone();
            entries_sorted.sort_by(|a, b| a.0.cmp(&b.0));
            let signature = entries_sorted
                .into_iter()
                .map(|(_, fp)| fp)
                .collect::<Vec<_>>()
                .join(",");
            tensor_meta_groups.entry(signature).or_default().push(rank);
        }
    }

    let tensor_meta_divergence_groups: Vec<DivergenceGroup> = if tensor_meta_groups.len() > 1 {
        tensor_meta_groups
            .iter()
            .map(|(seq, ranks_vec)| {
                let mut sorted_ranks = ranks_vec.clone();
                sorted_ranks.sort_unstable();
                DivergenceGroup {
                    sequence: seq.clone(),
                    ranks: sorted_ranks
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Group ranks by their collective op sequence
    let mut collective_seq_groups: FxHashMap<String, Vec<u32>> = FxHashMap::default();
    if !collective_schedules.is_empty() {
        for &rank in &rank_nums {
            let ops_concat: String = collective_schedules
                .iter()
                .filter(|s| s.rank == rank)
                .flat_map(|s| s.ops.clone())
                .collect::<Vec<_>>()
                .join(",");
            collective_seq_groups
                .entry(ops_concat)
                .or_default()
                .push(rank);
        }
    }

    let collective_divergence_groups: Vec<DivergenceGroup> = if collective_seq_groups.len() > 1 {
        collective_seq_groups
            .iter()
            .map(|(seq, ranks_vec)| {
                let mut sorted_ranks = ranks_vec.clone();
                sorted_ranks.sort_unstable();
                DivergenceGroup {
                    sequence: seq.clone(),
                    ranks: sorted_ranks
                        .iter()
                        .map(|r| r.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    println!(
        "Multi-rank report generated under {}\nIndividual pages: rank_*/index.html",
        out_path.display()
    );

    let diagnostics = Diagnostics {
        divergence: DivergenceFlags {
            cache: cache_seq_groups.len() > 1,
            collective: collective_seq_groups.len() > 1,
            tensor_meta: tensor_meta_groups.len() > 1,
        },
        artifacts: ArtifactFlags {
            runtime_trace: !runtime_estimations.is_empty(),
        },
        analysis: runtime_analysis,
        cache_groups: cache_divergence_groups.clone(),
        collective_groups: collective_divergence_groups.clone(),
        tensor_meta_groups: tensor_meta_divergence_groups.clone(),
    };

    let (landing_page_path, landing_html) = generate_multi_rank_html(
        &out_path,
        sorted_ranks,
        cfg,
        !all_chromium_events.is_empty(),
        compile_id_divergence
            || diagnostics.divergence.cache
            || diagnostics.divergence.collective
            || diagnostics.divergence.tensor_meta,
        compile_id_divergence,
        diagnostics,
    )?;
    fs::write(&landing_page_path, landing_html)?;
    if open_browser {
        opener::open(&landing_page_path)?;
    }

    Ok(())
}
