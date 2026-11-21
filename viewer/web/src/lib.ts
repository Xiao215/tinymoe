import type { LineSeriesOption } from "echarts";
import type { Manifest, MetricRecord, Span } from "./types";
import { categorical, chartInk, sequential, stageColor } from "./theme";

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
  /** sidebar section */
  section: string;
  hint?: string;
}

/** Default dashboard layout, path-style titles; groups whose keys are absent are dropped. */
export function defaultGroups(keys: string[]): MetricGroup[] {
  const has = (k: string) => keys.includes(k);
  const prefix = (p: string) => keys.filter((k) => k.startsWith(p));
  const digits = [...new Set(prefix("eval/math/pass1/").map((k) => /\/d(\d+)s\d+$/.exec(k)?.[1]).filter((d): d is string => !!d))].sort();
  const groups: MetricGroup[] = [
    { section: "train", title: "train/loss", keys: ["loss", ...prefix("loss/")].filter(has), hint: "per data source" },
    { section: "train", title: "train/aux_losses", keys: ["aux_loss", "mtp_loss", "z_loss"].filter(has), log: true },
    { section: "eval", title: "val/loss", keys: prefix("val/loss/") },
    { section: "eval", title: "val_long/loss", keys: prefix("val_long/loss/"), hint: "long context" },
    { section: "eval", title: "eval/math/pass1", keys: ["eval/math/pass1", "eval/math/pass4", "eval/math/format"].filter(has) },
    ...digits.map((d) => ({
      section: "eval",
      title: `eval/math/pass1/d${d}`,
      keys: prefix(`eval/math/pass1/d${d}s`),
      hint: `${d}-digit operands, by number of operations`,
    })),
    { section: "eval", title: "eval/math/length", keys: ["eval/math/resp_len", "eval/math/think_len"].filter(has), hint: "tokens" },
    { section: "eval", title: "eval/story", keys: prefix("eval/story/").filter((k) => k !== "eval/story/words") },
    { section: "optim", title: "optim/lr", keys: ["lr"].filter(has) },
    { section: "optim", title: "optim/grad_norm", keys: ["grad_norm", ...prefix("grad_norm/")].filter(has), log: true, hint: "per param group" },
    { section: "optim", title: "optim/update_ratio", keys: prefix("update_ratio/"), log: true, hint: "|Δw| / |w|" },
    { section: "optim", title: "optim/loss_scale", keys: ["loss_scale"].filter(has), log: true, hint: "fp16" },
    { section: "system", title: "system/throughput", keys: keys.filter((k) => k.endsWith("_per_s")) },
    { section: "system", title: "system/mfu", keys: ["mfu"].filter(has) },
    { section: "system", title: "system/mem_gb", keys: ["mem_gb"].filter(has) },
    { section: "system", title: "system/step_time", keys: ["step_time"].filter(has), hint: "seconds" },
    { section: "moe", title: "moe/load_max", keys: ["moe/load_max"].filter(has), hint: "1.0 = balanced" },
    { section: "moe", title: "moe/load_entropy", keys: ["moe/load_entropy"].filter(has) },
    { section: "moe", title: "moe/router", keys: prefix("moe/").filter((k) => !["moe/load_max", "moe/load_entropy"].includes(k)) },
    { section: "rl", title: "rl/reward", keys: ["rl/reward", "rl/pass", "rl/format"].filter(has) },
    { section: "rl", title: "rl/pass", keys: prefix("rl/pass/") },
    { section: "rl", title: "rl/policy", keys: ["rl/kl", "rl/entropy", "rl/clip_frac"].filter(has) },
    { section: "rl", title: "rl/length", keys: ["rl/resp_len", "rl/think_len"].filter(has) },
    {
      section: "rl",
      title: "rl/other",
      keys: prefix("rl/").filter((k) => !/^rl\/(reward|pass|format|kl|entropy|clip_frac|resp_len|think_len)/.test(k)),
    },
    { section: "dpo", title: "dpo/accuracy", keys: prefix("dpo/accuracy").concat(prefix("dpo/val_accuracy")) },
    { section: "dpo", title: "dpo/reward", keys: ["dpo/margin", "dpo/reward_chosen", "dpo/reward_rejected"].filter(has) },
    { section: "dpo", title: "dpo/logp", keys: prefix("dpo/logp") },
    { section: "distill", title: "distill/kd", keys: prefix("kd/").filter((k) => k !== "kd/tokens") },
    { section: "distill", title: "distill/pass1", keys: keys.filter((k) => /^distill\/[^/]+\/eval\/math\/pass1$/.test(k)) },
  ];
  const used = new Set(groups.flatMap((g) => g.keys));
  const skip = ["tokens", "examples", "skipped", "lr_mult", "pairs", "sequences", "kd/tokens", "eval/story/words"];
  const rest = keys.filter((k) => !used.has(k) && !skip.includes(k) && !/^distill\/[^/]+\/(eval|val)\//.test(k));
  for (const k of rest) groups.push({ section: "other", title: k, keys: [k] });
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

/** Faint stage regions + labels for any chart whose x axis is the global step. */
export function stageAreas(timeline: Span[]): LineSeriesOption["markArea"] {
  return {
    silent: true,
    label: { show: true, position: "insideTop" as const, fontSize: 10, color: chartInk().muted, fontFamily: MONO },
    data: timeline
      .filter((s) => s.end > s.start)
      .map((s): [AreaEdge, AreaEdge] => [
        { name: s.name, xAxis: s.start, itemStyle: { color: stageColor(s.name), opacity: 0.045 } },
        { xAxis: s.end },
      ]),
  };
}

export const MONO = 'ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace';

/** Stage boundaries as hairline rules with a small stage label (x = global step or ms). */
export function stageLines(bounds: { name: string; x: number }[]): LineSeriesOption["markLine"] {
  const ink = chartInk();
  return {
    silent: true,
    symbol: "none",
    animation: false,
    lineStyle: { color: ink.axis, type: "solid", width: 1 },
    label: { show: true, position: "end", formatter: "{b}", color: ink.muted, fontSize: 9.5, fontFamily: MONO, distance: 3 },
    data: bounds.map((b) => ({ name: b.name, xAxis: b.x })),
  };
}

/** TensorBoard-style debiased EMA; `alpha` in [0, 1), 0 = raw. */
export function ema(points: [number, number][], alpha: number): [number, number][] {
  if (alpha <= 0) return points;
  let acc = 0;
  let w = 0;
  return points.map(([x, y]) => {
    acc = alpha * acc + (1 - alpha) * y;
    w = alpha * w + (1 - alpha);
    return [x, acc / w];
  });
}

/** Compact number for KPIs: 14.0M, 2.36k, 0.0123, 1.43. */
export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e4) return `${(v / 1e3).toFixed(1)}k`;
  if (a >= 100) return v.toFixed(0);
  if (a === 0) return "0";
  if (a < 1e-3) return v.toExponential(1);
  return Number(v.toPrecision(digits)).toString();
}

export function fmtDur(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) return "–";
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, "0")}m`;
}

export const fmtClock = (ms: number): string =>
  new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });

/** "val/loss/stories" -> ["val/loss/", "stories"] for muted-prefix titles. */
export function splitKey(key: string): [string, string] {
  const i = key.lastIndexOf("/");
  return i < 0 ? ["", key] : [key.slice(0, i + 1), key.slice(i + 1)];
}

/** Direction in which a metric improves (for coloring deltas); null = neutral. */
export function polarity(key: string): 1 | -1 | null {
  if (/(^|\/)(loss|ppl|aux_loss|mtp_loss|z_loss|dead_experts|load_max|kl)($|\/)/.test(key) || /loss/.test(key)) return -1;
  if (/pass\d|format|reward|accuracy|load_entropy|tokens_per_s|mfu|length_ok|mention/.test(key)) return 1;
  return null;
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

/** Sequential colour for t in [0, 1]: one blue ramp, light to dark. */
export const heatColor = sequential;

/** Categorical colour by index (validated order; identities past 8 repeat the cycle). */
export const paletteColor = categorical;

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

/** Wall-clock start (ms) of every stage that has one; record time t is seconds since it. */
export function stageStarts(manifest: Manifest | null): Record<string, number> {
  const out: Record<string, number> = {};
  for (const s of manifest?.stages ?? []) {
    const t = s.started ? Date.parse(s.started) : NaN;
    if (Number.isFinite(t)) out[s.name] = t;
  }
  return out;
}

/** Latest record carrying `key` (optionally within one stage). */
export function lastWith(records: MetricRecord[], key: string, stage?: string): MetricRecord | null {
  for (let i = records.length - 1; i >= 0; i--) {
    const r = records[i]!;
    if (typeof r[key] === "number" && (stage === undefined || r.stage === stage)) return r;
  }
  return null;
}

export const median = (xs: number[]): number => {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
};
