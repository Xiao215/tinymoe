import type * as echarts from "echarts";
import { useMemo, useState } from "react";
import { useSnapshot, useSnapshotSeries, type RunData } from "../api";
import { Chart } from "../components/Chart";
import { Heatmap } from "../components/Heatmap";
import { Empty, Picker, snapLabel } from "../components/Picker";
import { paletteColor, series, stageAreas } from "../lib";
import type { InternalsSnapshot, MoESnapshot, SnapshotEntry } from "../types";

interface Props {
  run: string;
  data: RunData;
  snapshot: SnapshotEntry | null;
}

const show = (t: string) => t.replace(/\n/g, "⏎") || "∅";

/** Per-layer expert share: counted load since the previous snapshot when available,
 * otherwise the routing of the specialisation probe text. */
function shares(moe: MoESnapshot): number[][] {
  if (moe.load) return moe.load;
  return moe.specialization.counts.map((perCat) => {
    const tot = perCat.reduce((acc, row) => acc.map((v, e) => v + (row[e] ?? 0)), new Array<number>(perCat[0]?.length ?? 0).fill(0));
    const sum = tot.reduce((a, b) => a + b, 0) || 1;
    return tot.map((v) => v / sum);
  });
}

/** Expert load (layer x expert, and over time), tokens coloured by expert, specialisation
 * by token type, co-activation, dead-expert alerts and RL routing-replay mismatch. */
export function MoE({ run, data, snapshot }: Props) {
  const { data: snap } = useSnapshot<InternalsSnapshot>(run, snapshot, "data", data.version);
  const internals = useMemo(() => data.snapshots.filter((s) => s.files.includes("data")), [data.snapshots]);
  const history = useSnapshotSeries<InternalsSnapshot>(run, internals, "data", data.version);
  const [li, setLi] = useState(0);
  const [probe, setProbe] = useState<string | null>(null);

  const metricsOption = useMemo((): echarts.EChartsOption => {
    const keys = ["moe/load_max", "moe/load_entropy", "moe/dead_experts", "rl/routing_mismatch"];
    return {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 30 },
      xAxis: { type: "value", name: "global step" },
      yAxis: { type: "value" },
      series: keys.map((k, i) => ({
        name: k,
        type: "line",
        showSymbol: false,
        data: series(data.records, k),
        ...(i === 0 ? { markArea: stageAreas(data.timeline) } : {}),
      })),
    };
  }, [data.records, data.timeline]);

  if (!snapshot) return <Empty>No internals snapshots yet.</Empty>;
  if (!snap) return <Empty>loading {snapLabel(snapshot)}…</Empty>;
  const moe = snap.moe;
  if (!moe) return <Empty>This model is dense: no experts to explore ({snapLabel(snapshot)}).</Empty>;
  const nE = snap.model.n_experts;
  const experts = Array.from({ length: nE }, (_, e) => `e${e}`);
  const i = Math.min(li, moe.layers.length - 1);
  const layer = moe.layers[i] ?? 0;
  const share = shares(moe);
  const dead = share.flatMap((row, k) => row.map((v, e) => [k, e, v] as const)).filter(([, , v]) => v === 0);
  const pid = probe ?? snap.probes[0]?.id ?? "";
  const tokens = snap.probes.find((p) => p.id === pid)?.tokens ?? [];
  const assign = moe.assignments[pid]?.[String(layer)] ?? [];
  const over = history
    .filter((h) => h.data.moe)
    .map((h) => ({ label: `${h.entry.stage}@${h.entry.step}`, row: shares(h.data.moe!)[i] ?? [] }));
  const spec = moe.specialization;
  const specOption: echarts.EChartsOption = {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 40, right: 10, top: 40, bottom: 30 },
    xAxis: { type: "category", data: experts },
    yAxis: { type: "value", name: "share of expert's tokens" },
    series: spec.categories.map((c, ci) => ({
      name: c,
      type: "bar",
      stack: "s",
      data: experts.map((_, e) => {
        const col = (spec.counts[i] ?? []).map((row) => row[e] ?? 0);
        const tot = col.reduce((a, b) => a + b, 0) || 1;
        return (col[ci] ?? 0) / tot;
      }),
      itemStyle: { color: paletteColor(ci) },
    })),
  };

  return (
    <div>
      <div className="toolbar">
        <span className="muted">{snapLabel(snapshot)} · {nE} experts, top-{snap.model.top_k}</span>
        <Picker label="MoE layer" value={i} options={moe.layers.map((l, k) => ({ value: k, label: `L${l}` }))} onChange={setLi} />
        <Picker label="probe" value={pid} options={snap.probes.map((p) => ({ value: p.id, label: p.id }))} onChange={setProbe} />
        {dead.length > 0 ? (
          <span className="badge bad" title={dead.map(([k, e]) => `L${moe.layers[k]} e${e}`).join(", ")}>
            {dead.length} dead expert{dead.length > 1 ? "s" : ""}
          </span>
        ) : (
          <span className="badge ok">no dead experts</span>
        )}
      </div>
      <div className="grid">
        <div className="card">
          <h3>expert load share, layer × expert {moe.load ? "(counted since last snapshot)" : "(probe routing)"}</h3>
          <Heatmap values={share} rows={moe.layers.map((l) => `L${l}`)} cols={experts} min={0} xName="expert" yName="layer" onCell={(r) => setLi(r)} />
        </div>
        <div className="card">
          <h3>load over time, layer L{layer}</h3>
          {over.length ? <Heatmap values={over.map((o) => o.row)} rows={over.map((o) => o.label)} cols={experts} min={0} xName="expert" yName="snapshot" /> : <Empty>loading history…</Empty>}
        </div>
        <div className="card">
          <h3>tokens coloured by first-choice expert, layer L{layer}</h3>
          <div className="token-strip">
            {tokens.map((t, k) => {
              const ex = assign[k] ?? [];
              return (
                <span key={k} className="tok" style={{ borderColor: paletteColor(ex[0] ?? -1), background: `${paletteColor(ex[0] ?? -1)}33` }} title={`experts ${ex.join(", ")}`}>
                  {show(t)}
                </span>
              );
            })}
          </div>
          <div className="legend">
            {experts.map((e, k) => (
              <span key={e}>
                <span className="dot" style={{ background: paletteColor(k) }} /> {e}{" "}
              </span>
            ))}
          </div>
        </div>
        <div className="card">
          <h3>specialisation: token types routed to each expert, layer L{layer}</h3>
          <Chart option={specOption} height={260} />
        </div>
        <div className="card">
          <h3>expert co-activation (same token), layer L{layer}</h3>
          <Heatmap values={moe.coactivation[i] ?? []} rows={experts} cols={experts} min={0} digits={0} xName="expert" yName="expert" />
        </div>
        <div className="card">
          <h3>router bias (aux-loss-free balancing), layer L{layer}</h3>
          <Chart
            height={220}
            option={{
              grid: { left: 50, right: 10, top: 10, bottom: 30 },
              tooltip: { trigger: "axis" },
              xAxis: { type: "category", data: experts },
              yAxis: { type: "value" },
              series: [{ type: "bar", data: moe.bias[i] ?? [], itemStyle: { color: "#b07aa1" } }],
            }}
          />
        </div>
        <div className="card">
          <h3>balance over the pipeline (and RL routing-replay mismatch)</h3>
          <Chart option={metricsOption} height={260} />
        </div>
      </div>
    </div>
  );
}
