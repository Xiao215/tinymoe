import type * as echarts from "echarts";
import { useMemo } from "react";
import { useArtifact, type RunData } from "../api";
import { Chart } from "../components/Chart";
import { Empty } from "../components/Picker";
import { fmt } from "../lib";
import { stageColor } from "../theme";
import type { Column, DistillComparison, Report as ReportData } from "../types";

interface Props {
  run: string;
  data: RunData;
}

type Row = { stage: string; status: string | null; metrics: Record<string, number | null> };

/** Columns when report.json is absent (live run): whichever eval keys the summaries have. */
function fallbackColumns(rows: Row[]): Column[] {
  const keys = new Set(rows.flatMap((r) => Object.keys(r.metrics)));
  return [...keys]
    .filter((k) => /^(val\/ppl\/|eval\/math\/(pass\d+|format|think_len)$|eval\/story\/)/.test(k))
    .sort()
    .map((k) => ({ key: k, label: k }));
}

function Table({ columns, rows }: { columns: Column[]; rows: Row[] }) {
  return (
    <table className="report">
      <thead>
        <tr>
          <th>stage</th>
          {columns.map((c) => (
            <th key={c.key}>{c.label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.stage}>
            <td>
              <span className="dot" style={{ background: stageColor(r.stage) }} /> {r.stage}
              {r.status && r.status !== "done" ? <span className="muted"> ({r.status})</span> : null}
            </td>
            {columns.map((c) => (
              <td key={c.key}>{fmt(r.metrics[c.key] ?? null)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Distill({ table }: { table: DistillComparison }) {
  return (
    <table className="report">
      <thead>
        <tr>
          {["model", "init", "data", "objective", "steps", ...table.columns.map((c) => c.label)].map((h) => (
            <th key={h}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>teacher</td>
          <td>–</td>
          <td>–</td>
          <td>–</td>
          <td>–</td>
          {table.columns.map((c) => (
            <td key={c.key}>{fmt(table.teacher[c.key])}</td>
          ))}
        </tr>
        {table.rows.map((r) => (
          <tr key={r.mode}>
            <td>{r.mode}</td>
            <td>{r.init}</td>
            <td>{r.data}</td>
            <td>{r.objective}</td>
            <td>
              {r.steps[0]}–{r.steps[1]}
            </td>
            {table.columns.map((c) => (
              <td key={c.key}>{fmt(r.metrics[c.key])}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** Stage comparison: the eval summary of every finished stage, pass@1 per difficulty, the
 * distillation ladder, and the context-extension / RL-calibration side results. */
export function Report({ run, data }: Props) {
  const report = useArtifact<ReportData>(run, "report.json", data);
  const distill = useArtifact<DistillComparison>(run, "distill_comparison.json", data);
  const ctx = useArtifact<Record<string, Record<string, number>>>(run, "context_extension.json", data);
  const calib = useArtifact<{ pass_rates: Record<string, number>; band: [number, number]; selected: string[] }>(run, "calibration.json", data);

  const rows: Row[] = useMemo(
    () =>
      (data.manifest?.stages ?? []).map((s) => ({
        stage: s.name,
        status: s.status,
        metrics: (s.summary ?? {}) as Record<string, number | null>,
      })),
    [data.manifest],
  );
  const columns = report.data?.columns ?? fallbackColumns(rows);
  const buckets = useMemo(
    () => [...new Set(rows.flatMap((r) => Object.keys(r.metrics).filter((k) => k.startsWith("eval/math/pass1/")).map((k) => k.slice(16))))].sort(),
    [rows],
  );
  const bucketOption = useMemo(
    (): echarts.EChartsOption => ({
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 40, right: 10, top: 40, bottom: 40 },
      xAxis: { type: "category", data: buckets, axisLabel: { rotate: buckets.length > 6 ? 30 : 0 } },
      yAxis: { type: "value", min: 0, max: 1, name: "pass@1" },
      series: rows
        .filter((r) => buckets.some((b) => typeof r.metrics[`eval/math/pass1/${b}`] === "number"))
        .map((r) => ({
          name: r.stage,
          type: "bar",
          itemStyle: { color: stageColor(r.stage) },
          data: buckets.map((b) => r.metrics[`eval/math/pass1/${b}`] ?? null),
        })),
    }),
    [rows, buckets],
  );

  if (!rows.length) return <Empty>No stages recorded yet.</Empty>;
  return (
    <div>
      <div className="card">
        <h3>eval summary per stage{report.data ? "" : " (live: from stage summaries)"}</h3>
        <Table columns={columns} rows={rows} />
      </div>
      {buckets.length > 0 && (
        <div className="card">
          <h3>math pass@1 per difficulty bucket</h3>
          <Chart option={bucketOption} height={300} />
        </div>
      )}
      {distill.data && (
        <div className="card">
          <h3>distillation: what the student imitates (same step budget per mode)</h3>
          <Distill table={distill.data} />
        </div>
      )}
      <div className="grid">
        {ctx.data && (
          <div className="card">
            <h3>context extension (long-context validation loss)</h3>
            <table className="report">
              <tbody>
                {Object.entries(ctx.data).map(([phase, m]) => (
                  <tr key={phase}>
                    <td>{phase}</td>
                    {Object.entries(m).map(([k, v]) => (
                      <td key={k} title={k}>
                        {k.replace(/^val_long\//, "")}: {fmt(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {calib.data && (
          <div className="card">
            <h3>
              RL calibration: pass@1 per bucket, band {calib.data.band[0]}–{calib.data.band[1]}
            </h3>
            <table className="report">
              <tbody>
                {Object.entries(calib.data.pass_rates).map(([b, r]) => (
                  <tr key={b}>
                    <td>{b}</td>
                    <td>{fmt(r)}</td>
                    <td>{calib.data!.selected.includes(b) ? <span className="badge ok">selected</span> : null}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
