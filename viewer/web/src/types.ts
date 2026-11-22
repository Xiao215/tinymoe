// Response shapes of the viewer API (viewer/server/app.py) and the snapshot schema
// written by tinymoe/eval/snapshots.py + evaluator.py. Every field that a run may not
// have produced yet is optional, and panels must render with it missing.

export interface StageStatus {
  name: string;
  status: string;
}

export interface RunSummary {
  id: string;
  name: string;
  created: string | null;
  stages: StageStatus[];
  /** stages of the pipeline that made the run (absent for older runs) */
  planned?: string[] | null;
  mtime: number;
}

export interface Span {
  name: string;
  status: string;
  start: number;
  end: number;
}

export type MetricRecord = {
  stage: string;
  step: number;
  gstep: number;
  time: number;
} & Record<string, number | string | null>;

export interface MetricsResponse {
  records: MetricRecord[];
  offset: number;
  timeline: Span[];
}

export interface SnapshotEntry {
  stage: string;
  step: number;
  gstep: number;
  files: string[];
  arrays: string[];
}

export interface GenerationRecord {
  id: string;
  kind: string;
  prompt: string;
  completion: string;
  answer?: number;
  correct?: boolean;
  format_ok?: boolean;
}

export interface GenerationsSnapshot {
  stage: string;
  step: number;
  kind: string;
  generations: GenerationRecord[];
}

export interface ModelInfo {
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  d_model: number;
  moe_layers: number[];
  n_experts: number;
  top_k: number;
  max_seq_len: number;
}

export interface LayerStats {
  residual_norm: number[];
  attn_entropy: number[][];
  attn_out_norm: number[];
  ffn_out_norm: number[];
}

export interface LensPosition {
  pos: number;
  token: string;
  target: string;
  layers: [string, number][][];
}

export interface MoESnapshot {
  layers: number[];
  load: number[][] | null;
  bias: number[][];
  score_mean: number[][];
  score_hist: number[][];
  coactivation: number[][][];
  specialization: { categories: string[]; counts: number[][][] };
  assignments: Record<string, Record<string, number[][]>>;
  expert_out_norm: number[][];
}

export interface InternalsSnapshot {
  stage: string;
  step: number;
  model: ModelInfo;
  probes: { id: string; tokens: string[] }[];
  layers: Record<string, LayerStats>;
  logit_lens: Record<string, LensPosition[]>;
  weights: { attn: number[]; ffn: number[] };
  embeddings: { tokens: string[]; xy: [number, number][] };
  moe?: MoESnapshot;
}

export interface ArrayResponse {
  key: string;
  shape: number[];
  data: unknown;
}

export interface ManifestStage {
  name: string;
  status: string;
  started?: string;
  finished?: string;
  init_from?: string | null;
  checkpoint?: string | null;
  summary?: Record<string, number | null>;
  config?: Record<string, unknown>;
}

export interface Manifest {
  run_id: string;
  name: string;
  created?: string;
  git?: Record<string, unknown> | null;
  hardware?: Record<string, unknown>;
  runtime?: Record<string, unknown>;
  stages: ManifestStage[];
  pipeline?: string;
  pipeline_stages?: string[];
  pipeline_args?: string[];
}

// ------------------------------------------------------------- run control

export interface PipelineInfo {
  name: string;
  file: string;
  stages: string[];
}

export interface ControlInfo {
  enabled: boolean;
  pipelines: PipelineInfo[];
  busy: string | null;
}

export type PauseWhen = "now" | "after_stage";

export interface ControlState {
  enabled: boolean;
  running: boolean;
  starting: boolean;
  pid: number | null;
  started: string | null;
  pending: PauseWhen | null;
  pipeline: string | null;
  pipeline_stages: string[];
  busy: string | null;
  log: string[];
}

// ------------------------------------------------------------ architecture

export interface ParamInfo {
  name: string;
  shape: number[];
  numel: number;
}

export interface ModelConfigJson {
  vocab_size: number;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  head_dim: number | null;
  ffn_hidden: number;
  max_seq_len: number;
  rope_base: number;
  rope_scaling: { type: string; factor: number };
  qk_norm: boolean;
  tie_embeddings: boolean;
  sliding_window: number | null;
  global_every: number;
  moe: {
    enabled: boolean;
    n_experts: number;
    top_k: number;
    n_shared_experts: number;
    expert_hidden: number;
    n_dense_layers: number;
    score_func: string;
    balance: string;
  };
  mtp: { depth: number; loss_weight: number };
}

export interface Architecture {
  stage: string;
  which: "model" | "student";
  config: ModelConfigJson;
  counts: Record<string, number>;
  params: ParamInfo[];
  buffers: ParamInfo[];
  seq_len: number | null;
  tokens_per_step: number | null;
}

export interface RolloutExample {
  bucket: string;
  question: string;
  answer: number;
  completion: string;
  reward: number;
  advantage: number;
  correct: boolean;
  format_ok: boolean;
  length: number;
}

export interface RolloutsSnapshot {
  iteration: number;
  rollouts: RolloutExample[];
}

export interface Column {
  key: string;
  label: string;
}

export interface DistillRow {
  mode: string;
  init: string;
  data: string;
  objective: string;
  steps: [number, number];
  metrics: Record<string, number>;
  checkpoint: string | null;
}

export interface DistillComparison {
  teacher: Record<string, number>;
  rows: DistillRow[];
  columns: Column[];
  markdown: string;
}

export interface Report {
  run_id: string;
  columns: Column[];
  buckets: string[];
  rows: { stage: string; status: string | null; metrics: Record<string, number | null> }[];
  distill: DistillComparison | null;
}
