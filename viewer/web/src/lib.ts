import type { LineSeriesOption } from "echarts";
import type { MetricRecord, Span } from "./types";
import { stageColor } from "./theme";

const NON_METRIC = new Set(["stage", "step", "gstep", "time"]);

/** All numeric metric keys present in the records, sorted. */
export function metricKeys(records: MetricRecord[]): string[] {
  const keys = new Set<string>();
  for (const r of records) {
    for (const [k, v] of Object.entries(r)) {
      if (!NON_METRIC.has(k) && typeof v === "number") keys.add(k);
    }
  }
  return [...keys].sort();
}

export interface MetricGroup {
  title: string;
  keys: string[];
  log?: boolean;
}

/** Default dashboard layout; groups whose keys are absent are dropped by the caller. */
export function defaultGroups(keys: string[]): MetricGroup[] {
  const has = (k: string) => keys.includes(k);
  const prefix = (p: string) => keys.filter((k) => k.startsWith(p));
  const groups: MetricGroup[] = [
    { title: "Training loss (per data source)", keys: ["loss", ...prefix("loss/")].filter(has) },
    { title: "Validation loss", keys: prefix("val/loss/").concat(prefix("val_long/loss/")) },
    { title: "Math pass@1 per difficulty", keys: prefix("eval/math/pass1") },
    { title: "Math pass@k / format", keys: keys.filter((k) => /^eval\/math\/(pass\d+$|format)/.test(k)) },
    { title: "Response / think length", keys: ["eval/math/resp_len", "eval/math/think_len"].filter(has) },
    { title: "Story instruction following", keys: prefix("eval/story/") },
    { title: "Learning rate", keys: ["lr"].filter(has) },
    { title: "Gradient norm", keys: ["grad_norm", ...prefix("grad_norm/")].filter(has), log: true },
    { title: "Update / weight ratio", keys: prefix("update_ratio/"), log: true },
    { title: "Throughput", keys: keys.filter((k) => k.endsWith("_per_s")) },
    { title: "MFU / memory", keys: ["mfu", "mem_gb"].filter(has) },
    { title: "MoE balance", keys: prefix("moe/") },
    { title: "Auxiliary losses", keys: ["aux_loss", "mtp_loss", "z_loss"].filter(has) },
    { title: "RL reward", keys: prefix("rl/reward").concat(prefix("rl/pass")) },
    { title: "RL policy", keys: prefix("rl/").filter((k) => !k.startsWith("rl/reward") && !k.startsWith("rl/pass")) },
    { title: "Preference (DPO)", keys: prefix("dpo/") },
    { title: "Distillation", keys: prefix("distill/") },
  ];
  const used = new Set(groups.flatMap((g) => g.keys));
  const rest = keys.filter((k) => !used.has(k) && !["tokens", "examples", "step_time", "skipped", "lr_mult", "loss_scale"].includes(k));
  if (rest.length) groups.push({ title: "Other", keys: rest });
  return groups.filter((g) => g.keys.length > 0);
}

export function series(records: MetricRecord[], key: string): [number, number][] {
  const out: [number, number][] = [];
  for (const r of records) {
    const v = r[key];
    if (typeof v === "number") out.push([r.gstep, v]);
  }
  return out;
}

type AreaEdge = { name?: string; xAxis: number; itemStyle?: { color: string; opacity: number } };

/** Shaded stage regions + labels for any chart whose x axis is the global step. */
export function stageAreas(timeline: Span[]): LineSeriesOption["markArea"] {
  return {
    silent: true,
    label: { show: true, position: "insideTop" as const, fontSize: 10, color: "#888" },
    data: timeline
      .filter((s) => s.end > s.start)
      .map((s): [AreaEdge, AreaEdge] => [
        { name: s.name, xAxis: s.start, itemStyle: { color: stageColor(s.name), opacity: 0.07 } },
        { xAxis: s.end },
      ]),
  };
}

export interface ThinkParts {
  before: string;
  think: string | null;
  after: string;
}

/** Split a completion into text before, inside and after the `<think>` block. */
export function splitThink(text: string): ThinkParts {
  const open = text.indexOf("<think>");
  if (open < 0) return { before: text, think: null, after: "" };
  const close = text.indexOf("</think>", open);
  if (close < 0) return { before: text.slice(0, open), think: text.slice(open + 7), after: "" };
  return {
    before: text.slice(0, open),
    think: text.slice(open + 7, close),
    after: text.slice(close + 8),
  };
}

export type DiffOp = { op: "same" | "add" | "del"; text: string };

/** Word-level LCS diff (inputs are short completions, so O(n*m) is fine). */
export function wordDiff(a: string, b: string): DiffOp[] {
  const x = a.split(/(\s+)/);
  const y = b.split(/(\s+)/);
  const n = x.length;
  const m = y.length;
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i]![j] = x[i] === y[j] ? dp[i + 1]![j + 1]! + 1 : Math.max(dp[i + 1]![j]!, dp[i]![j + 1]!);
    }
  }
  const out: DiffOp[] = [];
  const push = (op: DiffOp["op"], text: string) => {
    const last = out[out.length - 1];
    if (last && last.op === op) last.text += text;
    else out.push({ op, text });
  };
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (x[i] === y[j]) {
      push("same", x[i]!);
      i++;
      j++;
    } else if (dp[i + 1]![j]! >= dp[i]![j + 1]!) {
      push("del", x[i++]!);
    } else {
      push("add", y[j++]!);
    }
  }
  while (i < n) push("del", x[i++]!);
  while (j < m) push("add", y[j++]!);
  return out;
}

export const fmt = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || Number.isNaN(v) ? "–" : Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(digits);

/** Sequential colour for t in [0, 1] (perceptually even enough for small heat cells). */
export function heatColor(t: number): string {
  const x = Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), 1);
  return `hsl(${220 - 180 * x}, 70%, ${92 - 50 * x}%)`;
}

/** Categorical palette for experts / token groups. */
export const PALETTE = [
  "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
  "#9c755f", "#bab0ac", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf",
];

export const paletteColor = (i: number) => PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length]!;

/** Token class for colouring (mirrors tinymoe/eval/snapshots.py token_category). */
export function tokenClass(tok: string): string {
  const s = tok.trim();
  if (/^<\|.*\|>$|^<\/?think>$/.test(s)) return "special";
  if (!s) return "space";
  if (/^\d+$/.test(s)) return "digit";
  if (["+", "-", "*", "=", "/", "(", ")"].includes(s)) return "operator";
  if (/^\p{L}+$/u.test(s)) return "word";
  return "punct";
}

export const maxOf = (xs: number[]): number => xs.reduce((m, x) => (x > m ? x : m), -Infinity);
