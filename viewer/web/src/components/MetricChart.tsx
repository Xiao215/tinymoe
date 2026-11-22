import type { EChartsOption, LineSeriesOption, ScatterSeriesOption } from "echarts";
import { useMemo, useState } from "react";
import { Chart } from "./Chart";
import { ema, fmtClock, fmtNum, MONO, polarity, splitKey, stageLines } from "../lib";
import { categorical, chartInk } from "../theme";
import type { MetricRecord, Span } from "../types";

export type XMode = "step" | "time";

export interface MetricChartProps {
  title: string;
  keys: string[];
  records: MetricRecord[];
  timeline: Span[];
  x?: XMode;
  /** stage name -> wall-clock ms of its start (needed for x = "time") */
  starts?: Record<string, number>;
  log?: boolean;
  smoothing?: number;
  compare?: Record<string, MetricRecord[]>;
  height?: number;
  hint?: string;
}

const MAX_SLOTS = 8;

/** Shared prefix up to a "/" boundary, so legend entries can drop it. */
function commonPrefix(keys: string[]): string {
  if (keys.length < 2) return keys.length === 1 ? splitKey(keys[0]!)[0] : "";
  let p = keys[0]!;
  for (const k of keys) while (!k.startsWith(p)) p = p.slice(0, -1);
  return p.slice(0, p.lastIndexOf("/") + 1);
}

interface Line {
  key: string;
  label: string;
  color: string;
  overflow: boolean;
  pts: [number, number][];
  /** what is drawn as the main line (smoothed when smoothing applies) */
  main: [number, number][];
  smooth: boolean;
  latest: number | null;
  delta: number | null;
}

const wallX = (r: MetricRecord, x: XMode, starts: Record<string, number>): number | null => {
  if (x === "step") return r.gstep;
  const t0 = starts[r.stage];
  return t0 === undefined ? null : t0 + r.time * 1000;
};

function points(records: MetricRecord[], key: string, x: XMode, starts: Record<string, number>): { pts: [number, number][]; stages: string[] } {
  const pts: [number, number][] = [];
  const stages: string[] = [];
  for (const r of records) {
    const v = r[key];
    if (typeof v !== "number" || !Number.isFinite(v)) continue;
    const xv = wallX(r, x, starts);
    if (xv === null) continue;
    pts.push([xv, v]);
    stages.push(r.stage);
  }
  return { pts, stages };
}

export function Delta({ k, d }: { k: string; d: number | null }) {
  if (d === null || !Number.isFinite(d) || d === 0) return <span className="d muted">{d === 0 ? "±0" : ""}</span>;
  const pol = polarity(k);
  const cls = pol === null ? "muted" : Math.sign(d) === pol ? "good" : "bad";
  return (
    <span className={`d ${cls}`}>
      {d > 0 ? "▲" : "▼"}
      {fmtNum(Math.abs(d), 2)}
    </span>
  );
}

/** One metric small multiple: mono title with a muted prefix, a legend that carries each
 * series' latest value and its change over the current stage, end dots, stage rules. */
export function MetricChart({
  title,
  keys,
  records,
  timeline,
  x = "step",
  starts = {},
  log = false,
  smoothing = 0,
  compare = {},
  height = 190,
  hint,
}: MetricChartProps) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const prefix = commonPrefix(keys);
  const lines: Line[] = useMemo(
    () =>
      keys.map((key, i) => {
        const { pts, stages } = points(records, key, x, starts);
        const smooth = smoothing > 0 && pts.length > 8;
        const main = smooth ? ema(pts, smoothing) : pts;
        const n = main.length;
        const latest = n ? main[n - 1]![1] : null;
        // change over the stage that logged the latest value
        const first = n ? stages.indexOf(stages[n - 1]!) : -1;
        const delta = latest !== null && first >= 0 && first < n - 1 ? latest - pts[first]![1] : null;
        return {
          key,
          label: key.slice(prefix.length) || key,
          color: i < MAX_SLOTS ? categorical(i) : chartInk().muted,
          overflow: i >= MAX_SLOTS,
          pts,
          main,
          smooth,
          latest,
          delta,
        };
      }),
    [keys, records, x, starts, prefix, smoothing],
  );
  // one x domain for every chart of a run, so small multiples line up
  const domain = useMemo((): [number, number] | null => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const r of records) {
      const xv = wallX(r, x, starts);
      if (xv === null) continue;
      if (xv < lo) lo = xv;
      if (xv > hi) hi = xv;
    }
    return Number.isFinite(lo) && hi > lo ? [lo, hi] : null;
  }, [records, x, starts]);

  const option = useMemo((): EChartsOption => {
    const ink = chartInk();
    const visible = lines.filter((l) => !hidden.has(l.key) && l.pts.length);
    const series: (LineSeriesOption | ScatterSeriesOption)[] = [];
    const sparse = visible.every((l) => l.pts.length < 24);
    const labelEnds = visible.length <= 4;
    const bounds =
      x === "time"
        ? Object.entries(starts)
            .filter(([n]) => timeline.some((s) => s.name === n))
            .map(([name, ms]) => ({ name, x: ms }))
        : timeline.filter((s) => s.end > s.start).map((s) => ({ name: s.name, x: s.start }));
    const lo = domain ? domain[0] : Math.min(...visible.map((l) => l.pts[0]![0]));
    const hi = domain ? domain[1] : Math.max(...visible.map((l) => l.pts[l.pts.length - 1]![0]));
    // label a rule only when there is room for it before the next one
    const inView = bounds
      .filter((b) => b.x > lo && b.x < hi)
      .map((b, i, arr) => {
        const next = arr[i + 1]?.x ?? hi;
        return (next - b.x) / (hi - lo) < 0.09 ? { ...b, name: "" } : b;
      });
    visible.forEach((l, i) => {
      const smooth = l.smooth;
      const dash = l.overflow ? "dashed" : "solid";
      if (smooth) {
        series.push({
          name: `${l.label} (raw)`,
          type: "line",
          data: l.pts,
          showSymbol: false,
          silent: true,
          lineStyle: { width: 1, color: l.color, opacity: 0.22, type: dash },
          itemStyle: { color: l.color },
          tooltip: { show: false },
          z: 1,
        });
      }
      const main = l.main;
      series.push({
        name: l.label,
        type: "line",
        data: main,
        showSymbol: sparse,
        symbolSize: 6,
        sampling: sparse ? undefined : "lttb",
        connectNulls: true,
        lineStyle: { width: 2, color: l.color, type: dash },
        itemStyle: { color: l.color, borderColor: ink.surface, borderWidth: 1.5 },
        emphasis: { disabled: true },
        z: 3,
        ...(i === 0 && inView.length ? { markLine: stageLines(inView) } : {}),
      });
      const end = main[main.length - 1]!;
      series.push({
        name: `${l.label} (end)`,
        type: "scatter",
        data: [end],
        symbolSize: 9,
        silent: true,
        tooltip: { show: false },
        itemStyle: { color: l.color, borderColor: ink.surface, borderWidth: 2, opacity: 1 },
        label: {
          show: labelEnds && !l.overflow,
          position: "right",
          distance: 4,
          formatter: () => fmtNum(end[1]),
          color: ink.ink2,
          fontFamily: MONO,
          fontSize: 10,
        },
        z: 4,
      });
    });
    for (const [run, recs] of Object.entries(compare)) {
      if (x === "time") break; // other runs have their own clocks
      visible.forEach((l) => {
        const { pts } = points(recs, l.key, "step", {});
        if (pts.length)
          series.push({
            name: `${l.label} [${run}]`,
            type: "line",
            data: smoothing > 0 && pts.length > 8 ? ema(pts, smoothing) : pts,
            showSymbol: false,
            lineStyle: { width: 1.5, type: [4, 3], color: l.color, opacity: 0.75 },
            itemStyle: { color: l.color },
            z: 2,
          });
      });
    }
    const allPositive = visible.every((l) => l.pts.every((p) => p[1] > 0));
    const stageAt = (gx: number) => timeline.find((s) => gx >= s.start && gx <= s.end)?.name ?? "";
    return {
      grid: { left: 44, right: labelEnds ? 46 : 14, top: inView.length ? 20 : 10, bottom: 22 },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line" },
        formatter: (raw: unknown) => {
          const ps = (Array.isArray(raw) ? raw : [raw]) as { seriesName: string; color: string; value: [number, number] }[];
          const shown = ps.filter((p) => !p.seriesName.endsWith("(raw)") && !p.seriesName.endsWith("(end)"));
          if (!shown.length) return "";
          const xv = shown[0]!.value[0];
          const head = x === "time" ? fmtClock(xv) : `step ${fmtNum(xv)} <span style="color:${ink.muted}">${stageAt(xv)}</span>`;
          const rows = shown
            .slice(0, 14)
            .map(
              (p) =>
                `<div style="display:flex;gap:10px;justify-content:space-between"><span><span style="display:inline-block;width:8px;height:2px;background:${p.color};vertical-align:3px;margin-right:6px"></span>${p.seriesName}</span><b style="font-weight:500">${fmtNum(p.value[1], 4)}</b></div>`,
            )
            .join("");
          return `<div style="color:${ink.muted};margin-bottom:3px">${head}</div>${rows}`;
        },
      },
      dataZoom: [{ type: "inside", zoomOnMouseWheel: "shift", moveOnMouseWheel: false, moveOnMouseMove: true }],
      xAxis:
        x === "time"
          ? { type: "time", min: lo, max: hi, axisLabel: { formatter: (v: number) => fmtClock(v), hideOverlap: true }, splitLine: { show: false } }
          : { type: "value", min: lo, max: hi, axisLabel: { formatter: (v: number) => fmtNum(v), hideOverlap: true }, splitLine: { show: false } },
      yAxis:
        log && allPositive
          ? { type: "log", axisLabel: { formatter: (v: number) => fmtNum(v) } }
          : { type: "value", scale: true, splitNumber: 4, axisLabel: { formatter: (v: number) => fmtNum(v) } },
      series,
    };
  }, [lines, hidden, smoothing, x, starts, timeline, compare, log, domain]);

  const [pre, last] = splitKey(title);
  const hasData = lines.some((l) => l.pts.length);
  const toggle = (k: string) =>
    setHidden((h) => {
      const n = new Set(h);
      if (n.has(k)) n.delete(k);
      else n.add(k);
      return n;
    });
  return (
    <section className="mchart">
      <div className="mchart-head">
        <div className="mchart-title" title={title}>
          <span className="pre">{pre}</span>
          {last}
          {hint && <span className="hint">{hint}</span>}
        </div>
        {keys.length > 0 && (
          <div className={`mlegend${lines.length > 4 ? " wrap" : ""}`}>
            {lines.map((l) => (
              <button
                key={l.key}
                className={hidden.has(l.key) ? "off" : ""}
                onClick={() => toggle(l.key)}
                title={`${l.key}: click to ${hidden.has(l.key) ? "show" : "hide"}`}
              >
                <span className="sw" style={{ background: l.color, ...(l.overflow ? { opacity: 0.6 } : {}) }} />
                {lines.length > 1 && <span>{l.label}</span>}
                {lines.length <= 4 && <span className="v">{fmtNum(l.latest, 4)}</span>}
                {lines.length <= 4 && <Delta k={l.key} d={l.delta} />}
              </button>
            ))}
          </div>
        )}
      </div>
      {hasData ? (
        <Chart option={option} height={height} animate={false} />
      ) : (
        <div className="muted" style={{ height, display: "grid", placeItems: "center", fontSize: 12 }}>
          no data yet
        </div>
      )}
    </section>
  );
}

