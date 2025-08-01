use core::hash::BuildHasherDefault;
use fxhash::{FxHashMap, FxHashSet, FxHasher};
use html_escape::encode_text;
use indexmap::IndexMap;
use regex::Regex;
use serde_json::Value;

use std::fmt::{self, Display, Write};
use std::path::PathBuf;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

// Main function returns a list of files to save
pub type ParseOutput = Vec<(PathBuf, String)>;
pub type CompilationMetricsIndex = FxIndexMap<Option<CompileId>, Vec<CompilationMetricsMetadata>>;
pub type StackIndex = FxHashMap<Option<CompileId>, StackSummary>; // NB: attempt is always 0 here
pub type SymbolicShapeSpecializationIndex =
    FxHashMap<Option<CompileId>, Vec<SymbolicShapeSpecializationMetadata>>;
pub type GuardAddedFastIndex = FxHashMap<Option<CompileId>, Vec<GuardAddedFastMetadata>>;
pub type SymExprInfoIndex = FxHashMap<u64, SymExprInfoMetadata>;

pub type FxIndexMap<K, V> = IndexMap<K, V, BuildHasherDefault<FxHasher>>;

/// Per-rank metadata collected during multi-rank aggregation.
#[derive(Debug)]
pub struct RankMetaData {
    pub rank: u32,
    pub compile_ids: FxHashSet<String>,
    pub cache_sequence: String,
}

/// Grouping of ranks that share the same cache hit/miss sequence.
#[derive(Debug, Serialize)]
pub struct CacheDivergenceGroup {
    pub sequence: String,
    pub ranks: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CollectiveSchedule {
    pub rank: u32,
    pub graph: String,
    pub ops: Vec<String>,
}

pub fn extract_eval_with_key_id(filename: &str) -> Option<u64> {
    let re = Regex::new(r"<eval_with_key>\.([0-9]+)").unwrap();
    re.captures(filename)
        .and_then(|caps| caps.get(1))
        .and_then(|m| m.as_str().parse::<u64>().ok())
}

pub static INTERN_TABLE: Lazy<Mutex<FxHashMap<u32, String>>> =
    Lazy::new(|| Mutex::new(FxHashMap::default()));

#[derive(Default)]
pub struct StackTrieNode {
    terminal: Vec<Option<CompileId>>,
    // Ordered map so that when we print we roughly print in chronological order
    children: FxIndexMap<FrameSummary, StackTrieNode>,
}

impl StackTrieNode {
    pub fn insert(&mut self, mut stack: StackSummary, compile_id: Option<CompileId>) {
        let mut cur = self;
        for frame in stack.drain(..) {
            cur = cur.children.entry(frame).or_default();
        }
        cur.terminal.push(compile_id);
    }

    pub fn insert_no_terminal(&mut self, mut stack: StackSummary) {
        let mut cur = self;
        for frame in stack.drain(..) {
            cur = cur.children.entry(frame).or_default();
        }
    }

    pub fn is_empty(&self) -> bool {
        return self.children.is_empty() && self.terminal.is_empty();
    }

    pub fn fmt(
        &self,
        metrics_index: Option<&CompilationMetricsIndex>,
        caption: &str,
        open: bool,
    ) -> Result<String, fmt::Error> {
        let mut f = String::new();
        write!(f, "<details{}>", if open { " open" } else { "" })?;
        write!(f, "<summary>{}</summary>", caption)?;
        write!(f, "<div class='stack-trie'>")?;
        write!(f, "<ul>")?;
        self.fmt_inner(&mut f, metrics_index)?;
        write!(f, "</ul>")?;
        write!(f, "</div>")?;
        write!(f, "</details>")?;
        Ok(f)
    }

    pub fn fmt_inner(
        &self,
        f: &mut String,
        mb_metrics_index: Option<&CompilationMetricsIndex>,
    ) -> fmt::Result {
        for (frame, node) in self.children.iter() {
            let mut star = String::new();
            for t in &node.terminal {
                if let Some(c) = t {
                    let ok_class = mb_metrics_index.map_or("status-missing", |metrics_index| {
                        metrics_index.get(t).map_or("status-missing", |m| {
                            if m.iter().any(|n| n.fail_type.is_some()) {
                                "status-error"
                            } else if m.iter().any(|n| n.graph_op_count.unwrap_or(0) == 0) {
                                "status-empty"
                            } else if m.iter().any(|n| {
                                !n.restart_reasons.as_ref().map_or(false, |o| o.is_empty())
                            }) {
                                "status-break"
                            } else {
                                "status-ok"
                            }
                        })
                    });
                    write!(
                        star,
                        "<a href='#{cid}' class='{ok_class}'>{cid}</a> ",
                        cid = c,
                        ok_class = ok_class
                    )?;
                } else {
                    write!(star, "(unknown) ")?;
                }
            }

            if self.children.len() > 1 {
                // If the node has multiple children, increase the indent and print a hyphen
                writeln!(
                    f,
                    "<li><span onclick='toggleList(this)' class='marker'></span>{star}",
                    star = star
                )?;
                writeln!(f, "{}<ul>", frame)?;
                node.fmt_inner(f, mb_metrics_index)?;
                write!(f, "</ul></li>")?;
            } else {
                // If the node has only one child, don't increase the indent and don't print a hyphen
                writeln!(f, "<li>{star}{}</li>", frame, star = star)?;
                node.fmt_inner(f, mb_metrics_index)?;
            }
        }
        Ok(())
    }
}

#[derive(Eq, PartialEq, Hash, Deserialize, Serialize, Debug, Clone)]
pub struct CompileId {
    pub compiled_autograd_id: Option<u32>,
    pub frame_id: Option<u32>,
    pub frame_compile_id: Option<u32>,
    pub attempt: Option<u32>,
}

impl fmt::Display for CompileId {
    // NOTE: If you want to elide an id e.g. attempt, compiled_autograd_id, you need to ensure
    // the representation remains unique. One way is to use a unique prefix.

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        if let Some(compiled_autograd_id) = self.compiled_autograd_id {
            write!(f, "!{}/", compiled_autograd_id)?;
        }
        let frame_id = self.frame_id.map_or("-".to_string(), |v| v.to_string());
        let frame_compile_id = self
            .frame_compile_id
            .map_or("-".to_string(), |v| v.to_string());
        write!(f, "{}/{}", frame_id, frame_compile_id)?;
        if let Some(attempt) = self.attempt {
            if attempt != 0 {
                write!(f, "_{}", attempt)?;
            }
        }
        write!(f, "]")
    }
}

impl CompileId {
    pub fn as_directory_name(&self) -> String {
        let compiled_autograd_id_str = self
            .compiled_autograd_id
            .map_or("-".to_string(), |v| v.to_string());
        let frame_id_str = self.frame_id.map_or("-".to_string(), |v| v.to_string());
        let frame_compile_id_str = self
            .frame_compile_id
            .map_or("-".to_string(), |v| v.to_string());
        let attempt_str = self.attempt.map_or("-".to_string(), |v| v.to_string());

        format!("{compiled_autograd_id_str}_{frame_id_str}_{frame_compile_id_str}_{attempt_str}")
    }
}

#[derive(Default, Debug)]
pub struct Stats {
    pub ok: u64,
    pub other_rank: u64,
    pub fail_glog: u64,
    pub fail_json: u64,
    pub fail_payload_md5: u64,
    pub fail_dynamo_guards_json: u64,
    pub fail_parser: u64,
    pub fail_key_conflict: u64,
    pub fail_json_serialization: u64,
    pub unknown: u64,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut fields = Vec::new();

        if self.ok > 0 {
            fields.push(format!("ok: {}", self.ok));
        }
        if self.other_rank > 0 {
            fields.push(format!("other_rank: {}", self.other_rank));
        }
        if self.fail_glog > 0 {
            fields.push(format!("fail_glog: {}", self.fail_glog));
        }
        if self.fail_json > 0 {
            fields.push(format!("fail_json: {}", self.fail_json));
        }
        if self.fail_payload_md5 > 0 {
            fields.push(format!("fail_payload_md5: {}", self.fail_payload_md5));
        }
        if self.fail_dynamo_guards_json > 0 {
            fields.push(format!(
                "fail_dynamo_guards_json: {}",
                self.fail_dynamo_guards_json
            ));
        }
        if self.fail_parser > 0 {
            fields.push(format!("fail_parser: {}", self.fail_parser));
        }
        if self.fail_key_conflict > 0 {
            fields.push(format!("fail_key_conflict: {}", self.fail_key_conflict));
        }
        if self.fail_json_serialization > 0 {
            fields.push(format!(
                "fail_json_serialization: {}",
                self.fail_json_serialization
            ));
        }
        if self.unknown > 0 {
            fields.push(format!("unknown: {}", self.unknown));
        }

        if fields.is_empty() {
            write!(f, "Stats {{ }}")
        } else {
            write!(f, "Stats {{ {} }}", fields.join(", "))
        }
    }
}

#[derive(Debug, Hash, Eq, PartialEq, Deserialize, Serialize, Clone)]
pub struct FrameSummary {
    pub filename: u32,
    pub line: i32,
    pub name: String,
    pub loc: Option<String>,
    pub uninterned_filename: Option<String>,
}

pub fn simplify_filename<'a>(filename: &'a str) -> &'a str {
    let parts: Vec<&'a str> = filename.split("#link-tree/").collect();
    if parts.len() > 1 {
        return parts[1];
    }
    let re = Regex::new(r"[^/]+-seed-nspid[^/]+/").unwrap();
    if let Some(captures) = re.captures(filename) {
        if let Some(capture) = captures.get(0) {
            return &filename[capture.end()..];
        }
    }
    return filename;
}

pub fn unintern_str(interned_str: u32) -> String {
    let intern_table = INTERN_TABLE.lock().unwrap();
    let filename = intern_table
        .get(&interned_str)
        .map_or("(unknown)", |s| s.as_str());
    return filename.to_string();
}

impl fmt::Display for FrameSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let intern_table = INTERN_TABLE.lock().unwrap();
        let filename = if let Some(f) = &self.uninterned_filename {
            f.as_str()
        } else {
            intern_table
                .get(&self.filename)
                .map_or("(unknown)", |s| s.as_str())
        };
        if let Some(fx_id) = extract_eval_with_key_id(filename) {
            write!(
                f,
                "<a href='dump_file/eval_with_key_{fx_id}.html#L{line}'>{filename}:{line}</a> in {name}",
                fx_id = fx_id,
                filename = encode_text(simplify_filename(filename)),
                line = self.line,
                name = encode_text(&self.name)
            )?;
        } else {
            write!(
                f,
                "{}:{} in {}<br>&nbsp;&nbsp;&nbsp;&nbsp;{}",
                encode_text(simplify_filename(filename)),
                self.line,
                encode_text(&self.name),
                encode_text(&self.loc.clone().unwrap_or("".to_string()))
            )?;
        }
        Ok(())
    }
}

pub type StackSummary = Vec<FrameSummary>;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SymInt {
    Int(i64),
    Symbol(String),
}

impl Default for SymInt {
    fn default() -> Self {
        SymInt::Int(0)
    }
}

fn default_layout() -> String {
    "torch.strided".to_string()
}

#[derive(Debug, Deserialize)]
pub struct OptimizeDdpSplitChildMetadata {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct EmptyMetadata {}

#[derive(Debug, Deserialize)]
pub struct GraphDumpMetadata {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct DynamoOutputGraphMetadata {
    _sizes: Option<FxHashMap<String, Vec<SymInt>>>,
}

#[derive(Debug, Deserialize)]
pub struct DynamoStartMetadata {
    pub stack: Option<StackSummary>,
}

#[derive(Debug, Deserialize)]
pub struct InductorOutputCodeMetadata {
    pub filename: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
pub struct LinkMetadata {
    pub name: String,
    pub url: String,
}

#[derive(Debug, Deserialize)]
pub struct ArtifactMetadata {
    pub name: String,
    pub encoding: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CompilationMetricsMetadata {
    // Other information like frame_key are already in envelope
    pub co_name: Option<String>,
    pub co_filename: Option<String>,
    pub co_firstlineno: Option<i32>,
    pub cache_size: Option<u64>,
    pub accumulated_cache_size: Option<u64>,
    pub guard_count: Option<u64>,
    pub shape_env_guard_count: Option<u64>,
    pub graph_op_count: Option<u64>,
    pub graph_node_count: Option<u64>,
    pub graph_input_count: Option<u64>,
    pub start_time: Option<f64>,
    pub entire_frame_compile_time_s: Option<f64>,
    pub backend_compile_time_s: Option<f64>,
    pub inductor_compile_time_s: Option<f64>,
    pub code_gen_time_s: Option<f64>,
    pub fail_type: Option<String>,
    pub fail_reason: Option<String>,
    pub fail_user_frame_filename: Option<String>,
    pub fail_user_frame_lineno: Option<u32>,
    pub non_compliant_ops: Option<Vec<String>>,
    pub compliant_custom_ops: Option<Vec<String>>,
    pub restart_reasons: Option<Vec<String>>,
    pub dynamo_time_before_restart_s: Option<f64>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BwdCompilationMetricsMetadata {
    pub inductor_compile_time_s: Option<f64>,
    pub code_gen_time_s: Option<f64>,
    pub fail_type: Option<String>,
    pub fail_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AOTAutogradBackwardCompilationMetricsMetadata {
    pub start_time: Option<f64>,
    pub elapsed_time: Option<f64>, // technically redundant with envelope
    pub fail_type: Option<String>,
    pub fail_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SymbolicShapeSpecializationMetadata {
    pub symbol: Option<String>,
    pub sources: Option<Vec<String>>,
    pub value: Option<String>,
    pub reason: Option<String>,
    pub stack: Option<StackSummary>,
    pub user_stack: Option<StackSummary>,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct FrameLocals {
    pub locals: Option<FxHashMap<String, Option<String>>>,
    pub symbols: Option<FxHashMap<String, Option<String>>>,
}
impl Display for FrameLocals {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(locals) = &self.locals {
            write!(f, "Locals:<pre>\n")?;
            for (name, value) in locals {
                match value {
                    Some(v) => write!(f, "    {}: {}\n", name, v),
                    None => Ok(()),
                }?
            }
            write!(f, "</pre>")?;
        }
        if let Some(symbols) = &self.symbols {
            write!(f, "Symbols:<pre>\n")?;
            for (name, value) in symbols {
                match value {
                    Some(v) => write!(f, "    {}: {}\n", name, v),
                    None => Ok(()),
                }?
            }
            write!(f, "</pre>")?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SymbolicShapePropagateRealTensorMetadata {
    pub expr: Option<String>,
    pub result: Option<String>,
    pub user_stack: Option<StackSummary>,
    pub stack: Option<StackSummary>,
    pub expr_node_id: Option<u64>,
    pub symbol_to_sources: Option<FxHashMap<String, String>>,
    pub frame_locals: Option<FrameLocals>,
    pub prefix: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct UnbackedSymbolMetadata {
    pub symbol: Option<String>,
    pub node_id: Option<u64>,
    pub user_stack: Option<StackSummary>,
    pub stack: Option<StackSummary>,
    pub vr: Option<String>,
}

#[derive(Default, Debug, Deserialize, Serialize)]
pub struct SymExprInfoMetadata {
    pub method: Option<String>,
    pub result: Option<String>,
    pub result_id: Option<u64>,
    pub arguments: Option<Vec<String>>,
    pub argument_ids: Option<Vec<u64>>,
    pub user_stack: Option<StackSummary>,
    pub stack: Option<StackSummary>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FakeKernelMetadata {
    pub op: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct BwdCompilationMetricsContext<'e> {
    pub m: &'e BwdCompilationMetricsMetadata,
    pub css: &'static str,
    pub compile_id: String,
    pub qps: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AOTAutogradBackwardCompilationMetricsContext<'e> {
    pub m: &'e AOTAutogradBackwardCompilationMetricsMetadata,
    pub css: &'static str,
    pub compile_id: String,
    pub qps: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct OutputFile {
    pub url: String,
    pub name: String,
    pub number: i32,
    pub suffix: String,
}

#[derive(Debug, Serialize)]
pub struct CompilationMetricsContext<'e> {
    pub m: &'e CompilationMetricsMetadata,
    pub css: &'static str,
    pub compile_id: String,
    pub stack_html: String,
    pub symbolic_shape_specializations: Vec<SymbolicShapeSpecializationContext>,
    pub guards_added_fast: Vec<GuardAddedFastContext>,
    pub output_files: &'e Vec<OutputFile>,
    pub compile_id_dir: &'e PathBuf,
    pub mini_stack_html: String,
    pub qps: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SymbolicGuardContext {
    pub css: &'static str,
    pub expr: String,
    pub user_stack_html: String,
    pub framework_stack_html: String,
    pub locals_html: String,
    pub sym_expr_trie_html: String,
}

#[derive(Debug, Serialize)]
pub struct GuardsAddedFastContext {
    pub guards: Vec<GuardAddedFastContext>,
}

#[derive(Debug, Serialize)]
pub enum FailureReason {
    Failure((String, String, String, u32)), // (failure type, failure reason, user frame filename, user frame lineno)
    Restart(String),                        // restart reason
}
impl Display for FailureReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FailureReason::Failure((
                failure_type,
                failure_reason,
                user_frame_filename,
                user_frame_lineno,
            )) => {
                let failure_type = encode_text(failure_type);
                let failure_reason = encode_text(failure_reason);
                let user_frame_filename = encode_text(user_frame_filename);
                write!(
                    f,
                    "<td><pre>{failure_type}</pre></td>
                           <td><pre>{failure_reason}</pre></td>
                           <td><pre>{user_frame_filename}:{user_frame_lineno}</pre></td>
                          "
                )
            }
            FailureReason::Restart(restart_reason) => write!(
                f,
                r#"<td> RestartAnalysis </td><td><pre>{restart_reason}</pre></td><td>Not availble for restarts(yet)!</td>"#
            ),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ExportFailure {
    pub failure_type: String,
    pub reason: String,
    pub additional_info: String,
}
impl Display for ExportFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "<td>{0}</td>
            <td><pre>{1}</pre></td>
            <td><pre>{2}</pre></td>
            ",
            self.failure_type, self.reason, self.additional_info
        )
    }
}

#[derive(Debug, Serialize)]
pub struct RestartsAndFailuresContext {
    // Serialized versions of (CompileId, FailureReason)
    pub failures: Vec<(String, String)>,
    pub css: &'static str,
    pub qps: &'static str,
}

#[derive(Debug)]
pub enum Metadata<'e> {
    Empty(&'e EmptyMetadata),
    Link(&'e LinkMetadata),
    GraphDump(&'e GraphDumpMetadata),
    DynamoOutputGraph(&'e DynamoOutputGraphMetadata),
    #[allow(dead_code)]
    DynamoStart(&'e DynamoStartMetadata),
    InductorOutputCode(&'e InductorOutputCodeMetadata),
    OptimizeDdpSplitChild(&'e OptimizeDdpSplitChildMetadata),
    CompilationMetrics(&'e CompilationMetricsMetadata),
    AOTAutogradBackwardCompilationMetrics(&'e AOTAutogradBackwardCompilationMetricsMetadata),
    BwdCompilationMetrics(&'e BwdCompilationMetricsMetadata),
    Artifact(&'e ArtifactMetadata),
    DumpFile(&'e DumpFileMetadata),
    GuardAddedFast(&'e GuardAddedFastMetadata),
    SymbolicShapePropagateRealTensor(&'e SymbolicShapePropagateRealTensorMetadata),
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DumpFileMetadata {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GuardAddedFastMetadata {
    pub expr: Option<String>,
    pub stack: Option<StackSummary>,
    pub user_stack: Option<StackSummary>,
}

#[derive(Debug, Deserialize)]
pub struct Envelope {
    pub rank: Option<u32>,
    #[serde(flatten)]
    pub compile_id: Option<CompileId>,
    #[serde(default)]
    pub has_payload: Option<String>,
    pub stack: Option<StackSummary>,
    // externally tagged union, one field per log type we recognize
    pub dynamo_start: Option<DynamoStartMetadata>,
    pub str: Option<(String, u32)>,
    pub dynamo_output_graph: Option<DynamoOutputGraphMetadata>,
    pub optimize_ddp_split_graph: Option<EmptyMetadata>,
    pub optimize_ddp_split_child: Option<OptimizeDdpSplitChildMetadata>,
    pub compiled_autograd_graph: Option<EmptyMetadata>,
    pub dynamo_guards: Option<EmptyMetadata>,
    pub aot_forward_graph: Option<EmptyMetadata>,
    pub aot_backward_graph: Option<EmptyMetadata>,
    pub aot_inference_graph: Option<EmptyMetadata>,
    pub aot_joint_graph: Option<EmptyMetadata>,
    pub inductor_pre_grad_graph: Option<EmptyMetadata>,
    pub inductor_post_grad_graph: Option<EmptyMetadata>,
    pub dynamo_cpp_guards_str: Option<EmptyMetadata>,
    pub inductor_output_code: Option<InductorOutputCodeMetadata>,
    pub compilation_metrics: Option<CompilationMetricsMetadata>,
    pub bwd_compilation_metrics: Option<BwdCompilationMetricsMetadata>,
    pub aot_autograd_backward_compilation_metrics:
        Option<AOTAutogradBackwardCompilationMetricsMetadata>,
    pub graph_dump: Option<GraphDumpMetadata>,
    pub link: Option<LinkMetadata>,
    pub symbolic_shape_specialization: Option<SymbolicShapeSpecializationMetadata>,
    pub propagate_real_tensors_provenance: Option<SymbolicShapePropagateRealTensorMetadata>,
    pub guard_added: Option<SymbolicShapePropagateRealTensorMetadata>,
    pub create_unbacked_symbol: Option<UnbackedSymbolMetadata>,
    pub expression_created: Option<SymExprInfoMetadata>,
    pub missing_fake_kernel: Option<FakeKernelMetadata>,
    pub mismatched_fake_kernel: Option<FakeKernelMetadata>,
    pub artifact: Option<ArtifactMetadata>,
    pub describe_storage: Option<StorageDesc>,
    pub describe_tensor: Option<TensorDesc>,
    pub describe_source: Option<SourceDesc>,
    pub dump_file: Option<DumpFileMetadata>,
    pub chromium_event: Option<EmptyMetadata>,
    pub guard_added_fast: Option<GuardAddedFastMetadata>,
    pub exported_program: Option<EmptyMetadata>,
    #[serde(flatten)]
    pub _other: FxHashMap<String, Value>,
}

type MetaTensorId = u64;
type MetaStorageId = u64;

#[derive(Debug, Deserialize, Serialize)]
pub struct TensorDesc {
    id: MetaTensorId,
    describer_id: u64,
    ndim: u64,
    dtype: String,
    device: String,
    size: Vec<SymInt>,
    dynamo_dynamic_indices: Option<Vec<u64>>,
    // TODO: Make layout an enum
    #[serde(default = "default_layout")]
    layout: String,
    #[serde(default)]
    is_inference: bool,
    #[serde(default)]
    is_leaf: bool,
    #[serde(default)]
    requires_grad: bool,
    #[serde(default)]
    is_sparse: bool,
    #[serde(default)]
    is_mkldnn: bool,
    #[serde(default)]
    is_functorch_wrapped: bool,
    #[serde(default)]
    is_batchedtensor: bool,
    #[serde(default)]
    is_legacy_batchedtensor: bool,
    #[serde(default)]
    is_gradtrackingtensor: bool,
    #[serde(default)]
    is_view: bool,
    #[serde(default)]
    is_nested: bool,
    #[serde(default)]
    is_traceable_wrapper_subclass: bool,
    #[serde(default)]
    is_functional: bool,
    #[serde(default)]
    is_conj: bool,
    #[serde(default)]
    is_neg: bool,
    #[serde(default)]
    is_parameter: bool,
    stride: Option<Vec<SymInt>>,
    #[serde(default)]
    storage_offset: SymInt,
    storage: Option<MetaStorageId>,
    sparse_dim: Option<u64>,
    dense_dim: Option<u64>,
    is_coalesced: Option<bool>,
    crow_indices: Option<MetaTensorId>,
    col_indices: Option<MetaTensorId>,
    ccol_indices: Option<MetaTensorId>,
    row_indices: Option<MetaTensorId>,
    values: Option<MetaTensorId>,
    unwrapped: Option<MetaTensorId>,
    bdim: Option<u64>,
    base: Option<MetaTensorId>,
    attrs: Option<FxHashMap<String, MetaTensorId>>,
    creation_meta: Option<String>,
    grad: Option<MetaTensorId>,
    #[serde(flatten)]
    pub _other: FxHashMap<String, Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StorageDesc {
    id: MetaStorageId,
    describer_id: u64,
    size: u64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SourceDesc {
    describer_id: u64,
    id: MetaTensorId,
    source: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DynamoGuard {
    pub code: String,
    pub stack: Option<StackSummary>,
    pub user_stack: Option<StackSummary>,
}

#[derive(Debug, Serialize)]
pub struct DynamoGuardsContext {
    pub guards: Vec<DynamoGuard>,
    pub qps: &'static str,
}

#[derive(Debug, Serialize)]
pub struct IndexContext {
    pub css: &'static str,
    pub javascript: &'static str,
    pub directory: Vec<(String, Vec<OutputFile>)>,
    pub stack_trie_html: String,
    pub unknown_stack_trie_html: String,
    pub has_unknown_stack_trie: bool,
    pub num_breaks: usize,
    pub custom_header_html: String,
    pub has_chromium_events: bool,
    pub qps: &'static str,
    pub has_inductor_provenance: bool,
    pub directory_names: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ExportIndexContext {
    pub css: &'static str,
    pub javascript: &'static str,
    pub directory: Vec<(String, Vec<OutputFile>)>,
    pub failures: Vec<ExportFailure>,
    pub custom_header_html: String,
    pub num_failures: usize,
    pub success: bool,
    pub exported_program_url: String,
    pub qps: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SymbolicShapeSpecializationContext {
    pub symbol: String,
    pub sources: Vec<String>,
    pub value: String,
    pub user_stack_html: String,
    pub stack_html: String,
}

#[derive(Debug, Serialize)]
pub struct GuardAddedFastContext {
    pub expr: String,
    pub user_stack_html: String,
    pub stack_html: String,
}

#[derive(Serialize)]
pub struct ProvenanceContext<'a> {
    pub css: &'a str,
    pub js: &'a str,
    pub pre_grad_graph_content: String,
    pub post_grad_graph_content: String,
    pub output_code_content: String,
    pub aot_code_content: String,
    pub node_mappings_content: String,
}

#[derive(Serialize)]
pub struct MultiRankContext<'a> {
    pub css: &'a str,
    pub custom_header_html: &'a str,
    pub num_ranks: usize,
    pub ranks: Vec<String>,
    pub qps: &'a str,
    pub has_chromium_events: bool,
    pub show_desync_warning: bool,
    pub divergence_groups: Vec<CacheDivergenceGroup>,
}
