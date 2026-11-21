import type * as echarts from "echarts";
import { useMemo, useState } from "react";
import { useSnapshot, type RunData } from "../api";
import { Chart } from "../components/Chart";
import { Empty } from "../components/Picker";
import { fmt } from "../lib";
import type { RolloutsSnapshot } from "../types";
import { Completion } from "./Generations";

interface Props {
  run: string;
  data: RunData;
}

/** Sampled RL rollouts per difficulty bucket with verifier reward and group advantage. */
export function Rollouts({ run, data }: Props) {
  const entries = useMemo(() => data.snapshots.filter((s) => s.files.includes("rollouts")), [data.snapshots]);
  const [idx, setIdx] = useState<number | null>(null);
  const cur = Math.min(idx ?? entries.length - 1, entries.length - 1);
  const entry = entries[cur] ?? null;
  const { data: snap } = useSnapshot<RolloutsSnapshot>(run, entry, "rollouts", data.version);
  const byBucket = useMemo(() => {
    const out = new Map<string, RolloutsSnapshot["rollouts"]>();
    for (const r of snap?.rollouts ?? []) out.set(r.bucket, [...(out.get(r.bucket) ?? []), r]);
    return out;
  }, [snap]);
  const option = useMemo((): echarts.EChartsOption => {
    const buckets = [...byBucket.keys()];
    const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 40, right: 10, top: 40, bottom: 30 },
      xAxis: { type: "category", data: buckets },
      yAxis: { type: "value" },
      series: [
        { name: "mean reward", type: "bar", data: buckets.map((b) => mean((byBucket.get(b) ?? []).map((r) => r.reward))) },
        { name: "correct", type: "bar", data: buckets.map((b) => mean((byBucket.get(b) ?? []).map((r) => Number(r.correct)))) },
      ],
    };
  }, [byBucket]);

  if (!entries.length) return <Empty>No RL rollouts yet (the grpo stage saves them every snapshot.interval iterations).</Empty>;
  return (
    <div>
      <div className="toolbar">
        <input type="range" min={0} max={entries.length - 1} value={cur} onChange={(e) => setIdx(Number(e.target.value))} aria-label="rollout snapshot" />
        <span>
          {entry?.stage} step {entry?.step}
          {snap ? ` · iteration ${snap.iteration}` : ""}
        </span>
      </div>
      {!snap ? (
        <Empty>loading…</Empty>
      ) : (
        <>
          <div className="card">
            <h3>this rollout's samples per bucket (2 shown per bucket)</h3>
            <Chart option={option} height={220} />
          </div>
          {[...byBucket.entries()].map(([bucket, rows]) => (
            <div key={bucket} className="card">
              <h3>{bucket}</h3>
              <table className="gen-table">
                <thead>
                  <tr>
                    <th>question</th>
                    <th>reward</th>
                    <th>completion</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, k) => (
                    <tr key={k}>
                      <td>
                        {r.question}
                        <div className="muted">answer {r.answer}</div>
                      </td>
                      <td>
                        <span className={`badge ${r.correct ? "ok" : "bad"}`}>{r.correct ? "correct" : r.format_ok ? "wrong" : "bad format"}</span>
                        <div>r = {fmt(r.reward, 2)}</div>
                        <div className={r.advantage >= 0 ? "pos" : "neg"}>A = {fmt(r.advantage, 2)}</div>
                        <div className="muted">{r.length} tok</div>
                      </td>
                      <td>
                        <Completion text={r.completion} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </>
      )}
    </div>
  );
}
