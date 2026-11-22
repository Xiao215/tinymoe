import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useControl, type RunData } from "../api";
import { RunControl, runPosition } from "../components/RunControl";
import { Delta, MetricChart } from "../components/MetricChart";
import { Empty } from "../components/Picker";
import { fmtClock, fmtDur, fmtNum, lastWith, median, metricKeys, stageStarts } from "../lib";
import { sequential, sequentialInk, STAGE_ORDER, stageColor } from "../theme";
import type { ManifestStage, MetricRecord } from "../types";

interface Props {
  run: string;
  data: RunData;
  onOpen: (panel: string) => void;
}

const WHAT: Record<string, string> = {
  pretrain: "next-token prediction on stories + synthetic math",
  midtrain: "anneal on high-quality math, then long-context extension",
  sft: "chat format and <think> traces",
  dpo: "preferences mined with the verifier",
  grpo: "RL on verifiable math rewards",
  distill: "teach a smaller student, four ways",
};

/** The stage-specific number that says whether the stage did its job. */
const OUTCOME: Record<string, { key: string; label: string }[]> = {
  pretrain: [{ key: "val/loss/stories", label: "val loss" }],
  midtrain: [{ key: "val_long/loss/stories", label: "long-ctx loss" }],
  sft: [{ key: "eval/story/length_ok", label: "story ok" }],
  dpo: [{ key: "dpo/val_accuracy", label: "pref. acc" }],
  grpo: [{ key: "rl/reward", label: "reward" }],
  distill: [{ key: "distill/teacher/eval/math/pass1", label: "teacher p@1" }],
};

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

function cfgNum(stage: ManifestStage | undefined, ...path: string[]): number | null {
  let cur: unknown = stage?.config;
  for (const p of path) cur = cur && typeof cur === "object" ? (cur as Record<string, unknown>)[p] : undefined;
  return num(cur);
}

function useNow(ms = 15000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return now;
}

/** A stage's value for `key`: its final summary, else the latest logged value in it. */
function stageValue(stage: ManifestStage, records: MetricRecord[], key: string): number | null {
  const s = num(stage.summary?.[key]);
  if (s !== null) return s;
  const r = lastWith(records, key, stage.name);
  return r ? (r[key] as number) : null;
}

const isTrain = (r: MetricRecord) => typeof r.loss === "number" || typeof r.step_time === "number";
const isEval = (r: MetricRecord) => Object.keys(r).some((k) => k.startsWith("val/") || k.startsWith("eval/"));

/** Seconds per step over the last `n` logged training steps (includes eval/ckpt overhead). */
function pace(records: MetricRecord[], stage: string, n = 100): { sps: number; over: number } | null {
  const tr = records.filter((r) => r.stage === stage && isTrain(r));
  if (tr.length < 2) return null;
  const last = tr[tr.length - 1]!;
  const first = tr.find((r) => r.step >= last.step - n) ?? tr[0]!;
  if (last.step === first.step) return null;
  return { sps: (last.time - first.time) / (last.step - first.step), over: last.step - first.step };
}

// ---------------------------------------------------------------- KPIs

interface Kpi {
  label: string;
  value: string;
  unit?: string;
  k?: string;
  d?: number | null;
  note: ReactNode;
}

function kpis(records: MetricRecord[], stage: ManifestStage, memCap: number | null): Kpi[] {
  const out: Kpi[] = [];
  const st = stage.name;
  const inStage = records.filter((r) => r.stage === st);
  const train = inStage.filter(isTrain);
  const evals = records.filter(isEval);
  const lastEval = evals[evals.length - 1];
  const prevEval = evals[evals.length - 2];

  const lossRec = lastWith(inStage, "loss");
  if (lossRec) {
    const first = inStage.find((r) => typeof r.loss === "number")!;
    // smooth the headline a little: median of the last 5 logged values
    const tail = train.filter((r) => typeof r.loss === "number").slice(-5).map((r) => r.loss as number);
    out.push({ label: "train loss", value: fmtNum(median(tail), 4), k: "loss", d: median(tail) - (first.loss as number), note: `since step ${first.step}` });
  }
  if (lastEval) {
    for (const src of ["stories", "math"]) {
      const key = `val/loss/${src}`;
      const v = num(lastEval[key]);
      if (v === null) continue;
      const pv = prevEval ? num(prevEval[key]) : null;
      out.push({ label: `val loss · ${src}`, value: fmtNum(v, 4), k: key, d: pv === null ? null : v - pv, note: pv === null ? `first eval, step ${lastEval.step}` : "vs previous eval" });
    }
    const p1 = num(lastEval["eval/math/pass1"]);
    if (p1 !== null) {
      const pv = prevEval ? num(prevEval["eval/math/pass1"]) : null;
      out.push({ label: "math pass@1", value: (100 * p1).toFixed(1), unit: "%", k: "eval/math/pass1", d: pv === null ? null : 100 * (p1 - pv), note: pv === null ? `greedy, step ${lastEval.step}` : "pts vs previous eval" });
    }
  }
  const rate = ["tokens_per_s", "sequences_per_s", "pairs_per_s"].find((k) => train.some((r) => typeof r[k] === "number"));
  if (rate) {
    const tail = train.filter((r) => typeof r[rate] === "number").slice(-10).map((r) => r[rate] as number);
    out.push({ label: rate.replace("_per_s", "/s").replace("_", " "), value: fmtNum(median(tail)), note: "median, last 10 logs" });
  }
  const tok = lastWith(inStage, "tokens");
  if (tok) out.push({ label: "tokens this stage", value: fmtNum(tok.tokens as number), note: `${fmtNum(cfgNum(stage, "train", "tokens_per_step"))} per step` });
  const lm = lastWith(inStage, "moe/load_max");
  if (lm) {
    const dead = num(lm["moe/dead_experts"]) ?? 0;
    out.push({
      label: "moe max load",
      value: `${fmtNum(lm["moe/load_max"] as number, 3)}`,
      unit: "×",
      note: dead > 0 ? <span className="bad">⚠ {dead} dead expert{dead > 1 ? "s" : ""}</span> : "1.0 = balanced · 0 dead",
    });
  }
  const rw = lastWith(inStage, "rl/reward");
  if (rw) out.push({ label: "rl reward", value: fmtNum(rw["rl/reward"] as number, 3), note: `kl ${fmtNum(num(rw["rl/kl"]), 2)}` });
  const da = lastWith(inStage, "dpo/accuracy");
  if (da) out.push({ label: "dpo accuracy", value: fmtNum(da["dpo/accuracy"] as number, 3), note: `margin ${fmtNum(num(da["dpo/margin"]), 3)}` });
  const mem = lastWith(inStage, "mem_gb");
  if (mem) {
    const v = mem.mem_gb as number;
    const near = memCap !== null && v > 0.9 * memCap;
    out.push({ label: "device memory", value: v.toFixed(2), unit: "GB", note: memCap ? <span className={near ? "warn" : ""}>{near ? "⚠ " : ""}cap {memCap} GB</span> : "peak allocated" });
  }
  return out;
}

// ---------------------------------------------------------------- events

interface Event {
  ms: number;
  tone: "good" | "bad" | "warn" | "info";
  icon: string;
  body: ReactNode;
}

function events(records: MetricRecord[], stages: ManifestStage[], starts: Record<string, number>, created: number | null, runtime: Record<string, unknown> | undefined): Event[] {
  const out: Event[] = [];
  const wall = (r: MetricRecord) => (starts[r.stage] ?? NaN) + r.time * 1000;
  if (created !== null) {
    const rt = runtime ?? {};
    const bits = [rt.device, rt.precision, rt.loss_scaling ? "loss scaling" : null, rt.compile ? "compiled" : null].filter(Boolean).join(" · ");
    out.push({ ms: created, tone: "info", icon: "○", body: <>run created{bits && <> · <span className="num">{bits}</span></>}</> });
  }
  for (const s of stages) {
    const t0 = s.started ? Date.parse(s.started) : NaN;
    const t1 = s.finished ? Date.parse(s.finished) : NaN;
    if (Number.isFinite(t0)) out.push({ ms: t0, tone: "info", icon: "▸", body: <><b>{s.name}</b> started{s.init_from ? <> from <span className="num">{s.init_from.split("/").slice(-2).join("/")}</span></> : null}</> });
    if (Number.isFinite(t1)) {
      const p1 = num(s.summary?.["eval/math/pass1"]);
      out.push({
        ms: t1,
        tone: s.status === "failed" ? "bad" : "good",
        icon: s.status === "failed" ? "✕" : "✓",
        body: <><b>{s.name}</b> {s.status === "failed" ? "failed" : "finished"} in <span className="num">{fmtDur((t1 - t0) / 1000)}</span>{p1 !== null && <> · pass@1 <span className="num">{(100 * p1).toFixed(1)}%</span></>}</>,
      });
    }
  }
  // evals, with the change since the previous eval
  let prev: MetricRecord | null = null;
  for (const r of records) {
    if (!isEval(r) || (typeof r["val/loss/stories"] !== "number" && typeof r["eval/math/pass1"] !== "number")) continue;
    const p1 = num(r["eval/math/pass1"]);
    const vl = num(r["val/loss/stories"]);
    const pp1 = prev ? num(prev["eval/math/pass1"]) : null;
    const pvl = prev ? num(prev["val/loss/stories"]) : null;
    out.push({
      ms: wall(r),
      tone: "info",
      icon: "◆",
      body: (
        <>
          eval <span className="num">{r.stage} {r.step}</span>
          {vl !== null && <> · val loss <span className="num">{vl.toFixed(3)}</span>{pvl !== null && <> <Delta k="val/loss" d={vl - pvl} /></>}</>}
          {p1 !== null && <> · pass@1 <span className="num">{(100 * p1).toFixed(1)}%</span>{pp1 !== null && <> <Delta k="pass1" d={100 * (p1 - pp1)} /></>}</>}
        </>
      ),
    });
    prev = r;
  }
  // anomalies: stalls, fp16 overflow skips, dead experts, loss spikes
  for (const st of stages) {
    const tr = records.filter((r) => r.stage === st.name && isTrain(r));
    const evalSteps = records.filter((r) => r.stage === st.name && isEval(r) && !isTrain(r)).map((r) => r.step);
    const per: number[] = [];
    for (let i = 1; i < tr.length; i++) per.push((tr[i]!.time - tr[i - 1]!.time) / Math.max(1, tr[i]!.step - tr[i - 1]!.step));
    const med = median(per);
    let deadSeen = false;
    for (let i = 1; i < tr.length; i++) {
      const a = tr[i - 1]!;
      const b = tr[i]!;
      const dt = b.time - a.time;
      const hasEval = evalSteps.some((s) => s >= a.step && s <= b.step);
      const ratio = per[i - 1]! / med;
      if (!hasEval && dt > 120 && ratio > 4)
        out.push({
          ms: wall(b),
          tone: "warn",
          icon: "⚠",
          body: <>slow: <span className="num">{st.name} {a.step}→{b.step}</span> took <span className="num">{fmtDur(dt)}</span>, {ratio.toFixed(0)}× the stage median. Another app using the GPU, or swapping?</>,
        });
      if (num(b.skipped) && (b.skipped as number) > 0)
        out.push({ ms: wall(b), tone: "warn", icon: "⚠", body: <>fp16 overflow at <span className="num">{st.name} {b.step}</span>: step skipped, loss scale <span className="num">{fmtNum(num(b.loss_scale))}</span></> });
      const dead = num(b["moe/dead_experts"]) ?? 0;
      if (dead > 0 && !deadSeen) {
        deadSeen = true;
        out.push({ ms: wall(b), tone: "bad", icon: "✕", body: <><span className="num">{dead}</span> dead expert{dead > 1 ? "s" : ""} at <span className="num">{st.name} {b.step}</span> (no tokens routed)</> });
      }
      // distillation restarts losses per mode, so a jump there is expected
      if (st.name !== "distill" && i >= 6 && typeof b.loss === "number") {
        const base = median(tr.slice(i - 6, i).map((r) => r.loss as number).filter(Number.isFinite));
        if (b.loss > base * 1.5 && b.loss - base > 0.3)
          out.push({ ms: wall(b), tone: "warn", icon: "⚠", body: <>loss spike at <span className="num">{st.name} {b.step}</span>: <span className="num">{fmtNum(b.loss, 3)}</span> vs <span className="num">{fmtNum(base, 3)}</span> before</> });
      }
    }
  }
  return out.filter((e) => Number.isFinite(e.ms)).sort((a, b) => b.ms - a.ms);
}

// ---------------------------------------------------------------- pieces

function StageCell({ name, stage, records, prev }: { name: string; stage?: ManifestStage; records: MetricRecord[]; prev?: ManifestStage }) {
  const status = stage?.status ?? "pending";
  const st = records.filter((r) => r.stage === name);
  const step = st.length ? Math.max(...st.map((r) => r.step)) : 0;
  const max = cfgNum(stage, "train", "max_steps");
  const t0 = stage?.started ? Date.parse(stage.started) : NaN;
  const t1 = stage?.finished ? Date.parse(stage.finished) : NaN;
  const lastT = st.length ? st[st.length - 1]!.time : null;
  const dur = Number.isFinite(t1) ? (t1 - t0) / 1000 : lastT;
  const frac = status === "done" ? 1 : max ? Math.min(1, step / max) : 0;
  const p1 = stage ? stageValue(stage, records, "eval/math/pass1") : null;
  const pp1 = prev ? stageValue(prev, records, "eval/math/pass1") : null;
  return (
    <div className={`stage-cell ${status}`} style={{ ["--c" as string]: stageColor(name) }}>
      <header>
        <h4>{name}</h4>
        <span className={`status ${status}`}>
          <i />
          {status}
        </span>
      </header>
      <div className="what">{WHAT[name] ?? ""}</div>
      {stage && (
        <dl>
          <dt>steps</dt>
          <dd>
            {fmtNum(step)}
            {max ? <span className="muted"> / {fmtNum(max)}</span> : null}
          </dd>
          <dt>time</dt>
          <dd>{fmtDur(dur)}</dd>
          <dt>pass@1</dt>
          <dd>
            {p1 === null ? "–" : `${(100 * p1).toFixed(1)}%`}
            {p1 !== null && pp1 !== null && (
              <>
                {" "}
                <Delta k="pass1" d={100 * (p1 - pp1)} />
              </>
            )}
          </dd>
          {(OUTCOME[name] ?? []).map((o) => {
            const v = stageValue(stage, records, o.key);
            return v === null ? null : (
              <span key={o.key} style={{ display: "contents" }}>
                <dt>{o.label}</dt>
                <dd>{fmtNum(v, 3)}</dd>
              </span>
            );
          })}
        </dl>
      )}
      <div className="stage-bar">
        <span style={{ width: `${100 * frac}%` }} />
      </div>
    </div>
  );
}

/** pass@1 at the latest eval as a digits x operations grid, with the change since the first eval. */
function BucketGrid({ records }: { records: MetricRecord[] }) {
  const evals = records.filter((r) => Object.keys(r).some((k) => /^eval\/math\/pass1\/d\d+s\d+$/.test(k)));
  const last = evals[evals.length - 1];
  if (!last) return <Empty>No math eval yet. It runs every eval interval.</Empty>;
  const first = evals[0]!;
  const cells = Object.keys(last)
    .map((k) => /^eval\/math\/pass1\/d(\d+)s(\d+)$/.exec(k))
    .filter((m): m is RegExpExecArray => m !== null)
    .map((m) => ({ d: Number(m[1]), s: Number(m[2]), key: m[0] }));
  const ds = [...new Set(cells.map((c) => c.d))].sort((a, b) => a - b);
  const ss = [...new Set(cells.map((c) => c.s))].sort((a, b) => a - b);
  return (
    <div className="card">
      <h3>
        math pass@1 by difficulty{" "}
        <span className="muted" style={{ fontWeight: 400 }}>
          · eval at {last.stage} {last.step}
          {first !== last ? `, Δ since ${first.stage} ${first.step}` : ""}
        </span>
      </h3>
      <div className="bucket-grid" style={{ gridTemplateColumns: `64px repeat(${ss.length}, minmax(0, 1fr))` }}>
        <div className="h row" />
        {ss.map((s) => (
          <div key={s} className="h">
            {s} op{s > 1 ? "s" : ""}
          </div>
        ))}
        {ds.map((d) => (
          <span key={d} style={{ display: "contents" }}>
            <div className="h row">
              {d} digit{d > 1 ? "s" : ""}
            </div>
            {ss.map((s) => {
              const key = `eval/math/pass1/d${d}s${s}`;
              const v = num(last[key]);
              const v0 = first !== last ? num(first[key]) : null;
              return (
                <div key={s} className="c" style={{ background: v === null ? "var(--surface-2)" : sequential(v), color: v === null ? "var(--muted)" : sequentialInk(v) }} title={key}>
                  {v === null ? "–" : `${(100 * v).toFixed(0)}%`}
                  {v !== null && v0 !== null && (
                    <small>
                      {v - v0 >= 0 ? "+" : "−"}
                      {Math.abs(100 * (v - v0)).toFixed(0)}
                    </small>
                  )}
                </div>
              );
            })}
          </span>
        ))}
      </div>
      <div className="legend" style={{ alignItems: "center" }}>
        0%
        <span style={{ display: "inline-block", width: 120, height: 6, borderRadius: 3, background: `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map(sequential).join(",")})` }} />
        100% · operand digits × chained operations
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- page

/** Landing page: where the run is, what each stage bought, what to look at next. */
export function Overview({ run, data, onOpen }: Props) {
  const { manifest, records, timeline } = data;
  const { state: ctl, refresh } = useControl(run);
  const now = useNow();
  const starts = useMemo(() => stageStarts(manifest), [manifest]);
  const keys = useMemo(() => metricKeys(records), [records]);
  const stages = manifest?.stages ?? [];
  const current = stages.find((s) => s.status === "running") ?? stages[stages.length - 1];
  const created = manifest?.created ? Date.parse(manifest.created) : null;
  const memCap = cfgNum(current, "hardware", "memory_budget_gb");
  const strip = useMemo(() => (current ? kpis(records, current, memCap) : []), [records, current, memCap]);
  const feed = useMemo(() => events(records, stages, starts, created, manifest?.runtime), [records, stages, starts, created, manifest?.runtime]);

  if (!manifest) return <Empty>Loading run…</Empty>;
  if (!current)
    return (
      <div>
        <RunControl run={run} manifest={manifest} ctl={ctl} refresh={refresh} />
        <Empty>No stage has started yet.</Empty>
      </div>
    );

  const position = runPosition(manifest, ctl);
  const running = position.status === "running" && current.status === "running";
  const cur = records.filter((r) => r.stage === current.name);
  const step = cur.length ? Math.max(...cur.map((r) => r.step)) : 0;
  const max = cfgNum(current, "train", "max_steps");
  const p = pace(records, current.name);
  const eta = running && p && max ? Math.max(0, (max - step) * p.sps) : null;
  const lastFinished = Math.max(...stages.map((s) => (s.finished ? Date.parse(s.finished) : 0)));
  const elapsed = created === null ? null : ((running ? now : lastFinished || now) - created) / 1000;
  const m = (current.config?.model ?? {}) as Record<string, unknown>;
  const moe = (m.moe ?? {}) as Record<string, unknown>;
  const arch = m.n_layers ? `${m.n_layers}L · d${m.d_model}${moe.enabled ? ` · ${moe.n_experts}e top-${moe.top_k}` : " · dense"}` : null;
  const rt = (manifest.runtime ?? {}) as Record<string, unknown>;
  const git = manifest.git as Record<string, unknown> | null | undefined;
  const has = (k: string) => keys.includes(k);
  const pre = (p0: string) => keys.filter((k) => k.startsWith(p0));
  const lastRec = records[records.length - 1];
  const lastAgo = lastRec && starts[lastRec.stage] !== undefined ? (now - (starts[lastRec.stage]! + lastRec.time * 1000)) / 1000 : null;

  const charts: { title: string; keys: string[]; log?: boolean; smoothing?: number; hint?: string }[] = [
    { title: "train/loss", keys: ["loss", ...pre("loss/")].filter(has), smoothing: 0.6, hint: "by data source" },
    { title: "val/loss", keys: pre("val/loss/") },
    { title: "eval/math/pass1", keys: ["eval/math/pass1", "eval/math/pass4", "eval/math/format"].filter(has), hint: "greedy · pass@4 · format" },
    ...(pre("rl/reward").length ? [{ title: "rl/reward", keys: ["rl/reward", "rl/pass"].filter(has), smoothing: 0.5 }] : []),
    ...(has("rl/kl") ? [{ title: "rl/kl", keys: ["rl/kl"], smoothing: 0.5 }] : []),
    ...(has("dpo/margin") ? [{ title: "dpo/margin", keys: ["dpo/margin"], smoothing: 0.5 }] : []),
    { title: "moe/load_max", keys: ["moe/load_max"].filter(has), smoothing: 0.6, hint: "1.0 = perfectly balanced" },
    { title: "throughput/tokens_per_s", keys: ["tokens_per_s"].filter(has), smoothing: 0.6 },
    { title: "optim/lr", keys: ["lr"].filter(has) },
    { title: "optim/grad_norm", keys: ["grad_norm"].filter(has), log: true, smoothing: 0.6 },
  ].filter((c) => c.keys.length);

  return (
    <div>
      <div className="run-head">
        <div>
          <div className="run-title">
            <h1>{manifest.run_id}</h1>
            <span className={`status ${position.status}`}>
              <i />
              {position.label}
            </span>
          </div>
          <div className="run-sub">
            <span className="dot" style={{ background: stageColor(current.name) }} /> {current.name} · step <span className="num">{fmtNum(step)}</span>
            {max && running ? (
              <>
                {" "}
                / <span className="num">{fmtNum(max)}</span> <span className="muted">({((100 * step) / max).toFixed(0)}%)</span>
              </>
            ) : null}
            {!running && elapsed !== null && (
              <>
                {" "}
                · {stages.length} stage{stages.length > 1 ? "s" : ""} in <span className="num">{fmtDur(elapsed)}</span>
              </>
            )}
            {running && p && (
              <>
                {" "}
                · <span className="num">{p.sps.toFixed(1)} s/step</span> <span className="muted">over the last {p.over} steps</span>
              </>
            )}
            {eta !== null && (
              <>
                {" "}
                · stage ETA <span className="num">{fmtDur(eta)}</span> <span className="muted">(~{fmtClock(now + eta * 1000)})</span>
              </>
            )}
            {running && lastAgo !== null && lastAgo > 300 && <span className="warn"> · ⚠ no new metrics for {fmtDur(lastAgo)}</span>}
          </div>
        </div>
        <div className="run-facts">
          <div className="meta">
            <label>elapsed</label>
            <span>{fmtDur(elapsed)}</span>
          </div>
          <div className="meta">
            <label>started</label>
            <span>{created ? new Date(created).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false }) : "–"}</span>
          </div>
          <div className="meta">
            <label>device</label>
            <span>
              {String(rt.device ?? "–")} · {String(rt.precision ?? "–")}
            </span>
          </div>
          {arch && (
            <div className="meta">
              <label>model</label>
              <span>{arch}</span>
            </div>
          )}
          {git?.commit ? (
            <div className="meta">
              <label>commit</label>
              <span>
                {String(git.commit).slice(0, 7)}
                {git.dirty ? "*" : ""}
              </span>
            </div>
          ) : null}
        </div>
      </div>
      <RunControl run={run} manifest={manifest} ctl={ctl} refresh={refresh} />
      {max && running ? (
        <div className="progress" title={`${current.name}: ${step} of ${max} steps`}>
          <span style={{ width: `${Math.min(100, (100 * step) / max)}%`, background: stageColor(current.name) }} />
        </div>
      ) : null}

      {strip.length > 0 && (
        <div className="kpis">
          {strip.map((k) => (
            <div className="kpi" key={k.label}>
              <span className="label">{k.label}</span>
              <div className="value">
                {k.value}
                {k.unit && <small>{k.unit}</small>}
              </div>
              <div className="delta">
                {k.k && k.d !== undefined && k.d !== null ? (
                  <>
                    <Delta k={k.k} d={k.d} />{" "}
                  </>
                ) : null}
                {k.note}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="section-head">
        <h2>Lifecycle</h2>
        <span className="muted">each stage starts from the previous stage's final checkpoint · pass@1 change vs the previous stage</span>
      </div>
      <div className="lifecycle">
        {STAGE_ORDER.map((name) => {
          const i = stages.findIndex((s) => s.name === name);
          return <StageCell key={name} name={name} stage={stages[i]} prev={i > 0 ? stages[i - 1] : undefined} records={records} />;
        })}
      </div>

      <div className="ov-body">
        <div>
          <div className="section-head">
            <h2>Key signals</h2>
            <span className="muted">global step · stage boundaries as rules · shift+scroll to zoom</span>
            <span className="grow" />
            <button onClick={() => onOpen("metrics")}>all {keys.length} metrics →</button>
          </div>
          <div className="grid3">
            {charts.map((c) => (
              <MetricChart key={c.title} title={c.title} keys={c.keys} records={records} timeline={timeline} log={c.log} smoothing={c.smoothing ?? 0} hint={c.hint} />
            ))}
          </div>
          <div className="section-head">
            <h2>What the model can do</h2>
            <span className="grow" />
            <button onClick={() => onOpen("generations")}>generations →</button>
          </div>
          <BucketGrid records={records} />
        </div>
        <div>
          <div className="section-head">
            <h2>Events</h2>
            <span className="muted">{feed.length} · newest first</span>
          </div>
          <div className="card" style={{ padding: "4px 14px" }}>
            <ul className="feed">
              {feed.slice(0, 120).map((e, i) => (
                <li key={i}>
                  <span className={`ic ${e.tone}`}>{e.icon}</span>
                  <time dateTime={new Date(e.ms).toISOString()} title={new Date(e.ms).toLocaleString()}>
                    {fmtClock(e.ms)}
                  </time>
                  <span className="txt">{e.body}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
