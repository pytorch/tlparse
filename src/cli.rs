use clap::Parser;

use anyhow::{bail, Context};
use std::fs;
use std::path::PathBuf;

use tlparse::{parse_path, ParseConfig};

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
    /// Parse all ranks and generate a single unified page
    #[arg(long)]
    all_ranks: bool,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.all_ranks {
        return handle_all_ranks(&cli);
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

    let out_path = cli.out;

    if out_path.exists() {
        if !cli.overwrite {
            bail!(
                "Directory {} already exists, use -o OUTDIR to write to another location or pass --overwrite to overwrite the old contents",
                out_path.display()
            );
        }
        fs::remove_dir_all(&out_path)?;
    }
    fs::create_dir(&out_path)?;

    let config = ParseConfig {
        strict: cli.strict,
        strict_compile_id: cli.strict_compile_id,
        custom_parsers: Vec::new(),
        custom_header_html: cli.custom_header_html.clone(),
        verbose: cli.verbose,
        plain_text: cli.plain_text,
        export: cli.export,
        inductor_provenance: cli.inductor_provenance,
        all_ranks: cli.all_ranks,
    };

    let output = parse_path(&path, config)?;

    for (filename, path) in output {
        let out_file = out_path.join(filename);
        if let Some(dir) = out_file.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(out_file, path)?;
    }

    if !cli.no_browser {
        opener::open(out_path.join("index.html"))?;
    }
    Ok(())
}

// handle_all_ranks function with placeholder landing page
fn handle_all_ranks(cli: &Cli) -> anyhow::Result<()> {
    let input_path = &cli.path;

    if !input_path.is_dir() {
        bail!(
            "Input path {} must be a directory when using --all-ranks",
            input_path.display()
        );
    }

    let out_path = &cli.out;

    if out_path.exists() {
        if !cli.overwrite {
            bail!(
                "Directory {} already exists, use -o OUTDIR to write to another location or pass --overwrite to overwrite the old contents",
                out_path.display()
            );
        }
        fs::remove_dir_all(&out_path)?;
    }
    fs::create_dir(&out_path)?;

    // Find all rank log files in the directory
    let rank_files: Vec<_> = std::fs::read_dir(input_path)
        .with_context(|| format!("Couldn't access directory {}", input_path.display()))?
        .flatten()
        .filter(|entry| {
            let path = entry.path();
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.contains("rank_") && name.ends_with(".log"))
                    .unwrap_or(false)
        })
        .collect();

    if rank_files.is_empty() {
        bail!(
            "No rank log files found in directory {}",
            input_path.display()
        );
    }

    let mut rank_links = Vec::new();

    // Process each rank file
    for rank_file in rank_files {
        let rank_path = rank_file.path();
        let rank_name = rank_path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");

        // Extract rank number from filename
        let rank_num = if let Some(pos) = rank_name.find("rank_") {
            let after_rank = &rank_name[pos + 5..];
            after_rank
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
        } else {
            "unknown".to_string()
        };

        println!(
            "Processing rank {} from file: {}",
            rank_num,
            rank_path.display()
        );

        // Create subdirectory for this rank
        let rank_out_dir = out_path.join(format!("rank_{rank_num}"));
        fs::create_dir(&rank_out_dir)?;

        let config = ParseConfig {
            strict: cli.strict,
            strict_compile_id: cli.strict_compile_id,
            custom_parsers: Vec::new(),
            custom_header_html: cli.custom_header_html.clone(),
            verbose: cli.verbose,
            plain_text: cli.plain_text,
            export: cli.export,
            inductor_provenance: cli.inductor_provenance,
            all_ranks: false,
        };

        let output = parse_path(&rank_path, config)?;

        // Write output files to rank subdirectory
        for (filename, content) in output {
            let out_file = rank_out_dir.join(filename);
            if let Some(dir) = out_file.parent() {
                fs::create_dir_all(dir)?;
            }
            fs::write(out_file, content)?;
        }

        // Add link to this rank's page
        rank_links.push((rank_num.clone(), format!("rank_{rank_num}/index.html")));
    }

    // Sort rank links by rank number
    rank_links.sort_by(|a, b| {
        let a_num: i32 = a.0.parse().unwrap_or(999);
        let b_num: i32 = b.0.parse().unwrap_or(999);
        a_num.cmp(&b_num)
    });

    // Core logic complete - no HTML generation yet
    // TODO - Add landing page HTML generation using template system

    println!(
        "Generated multi-rank report with {} ranks",
        rank_links.len()
    );
    println!("Individual rank reports available in:");
    for (rank_num, _) in &rank_links {
        println!("  - rank_{}/index.html", rank_num);
    }

    // No browser opening since no landing page yet
    // TODO - Generate landing page and open browser

    Ok(())
}
