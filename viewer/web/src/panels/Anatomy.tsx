import type * as echarts from "echarts";
import { useMemo, useState } from "react";
import { useSnapshot } from "../api";
import { Chart } from "../components/Chart";
import { Empty, Picker, snapLabel } from "../components/Picker";
import { fmt, heatColor, maxOf } from "../lib";
import type { InternalsSnapshot, SnapshotEntry } from "../types";

interface Props {
  run: string;
  snapshot: SnapshotEntry | null;
  version: number;
}

function bars(title: string, labels: string[], values: number[], color: string): echarts.EChartsOption {
  return {
    title: { text: title, textStyle: { fontSize: 12 } },
    tooltip: { trigger: "axis" },
    grid: { left: 50, right: 10, top: 30, bottom: 30 },
    xAxis: { type: "category", data: labels, axisLabel: { fontSize: 10 } },
    yAxis: { type: "value", axisLabel: { fontSize: 10 } },
    series: [{ type: "bar", data: values, itemStyle: { color } }],
  };
}

/** The architecture as a layer stack (from the snapshot's model config), coloured by the
 * selected snapshot's internals; click a layer for its heads, experts and norms. */
export function Anatomy({ run, snapshot, version }: Props) {
  const { data: snap, error } = useSnapshot<InternalsSnapshot>(run, snapshot, "data", version);
  const [probe, setProbe] = useState<string | null>(null);
  const [layer, setLayer] = useState(0);
  const pid = probe ?? snap?.probes[0]?.id ?? null;
  const stats = pid && snap ? snap.layers[pid] : undefined;
  const scales = useMemo(() => {
    if (!stats) return null;
    return {
      entropy: maxOf(stats.attn_entropy.flat()),
      resid: maxOf(stats.residual_norm),
    };
  }, [stats]);

  if (!snapshot) return <Empty>No internals snapshots yet (snapshot.interval steps into a stage).</Empty>;
  if (error) return <p className="error">{error}</p>;
  if (!snap || !stats || !scales || !pid) return <Empty>loading {snapLabel(snapshot)}…</Empty>;
  const m = snap.model;
  const moeIdx = (l: number) => (snap.moe ? snap.moe.layers.indexOf(l) : -1);
  const load = (l: number): number[] | null => {
    const i = moeIdx(l);
    return i >= 0 && snap.moe?.load ? (snap.moe.load[i] ?? null) : null; // per MoE layer
  };
  const sel = Math.min(layer, m.n_layers - 1);
  const selLoad = load(sel);
  const mi = moeIdx(sel);

  return (
    <div className="anatomy">
      <div className="toolbar">
        <span className="muted">{snapLabel(snapshot)}</span>
        <Picker label="probe" value={pid} options={snap.probes.map((p) => ({ value: p.id, label: p.id }))} onChange={setProbe} />
        <span className="muted">
          d_model {m.d_model} · {m.n_heads} heads ({m.n_kv_heads} kv) ·{" "}
          {m.n_experts ? `${m.n_experts} experts top-${m.top_k} in ${m.moe_layers.length} layers` : "dense"}
        </span>
      </div>
      <div className="anatomy-grid">
        <div className="stack card">
          <div className="stack-row head">lm head ← final norm</div>
          {Array.from({ length: m.n_layers }, (_, i) => m.n_layers - 1 - i).map((l) => {
            const ent = stats.attn_entropy[l] ?? [];
            const ld = load(l);
            const resid = stats.residual_norm[l + 1] ?? 0;
            return (
              <button key={l} className={`stack-row ${l === sel ? "active" : ""}`} onClick={() => setLayer(l)}>
                <span className="stack-label">L{l}</span>
                <span className="cells" title="attention heads, coloured by entropy">
                  {ent.map((e, h) => (
                    <span key={h} className="cell" style={{ background: heatColor(e / scales.entropy) }} />
                  ))}
                </span>
                <span className="cells" title={moeIdx(l) >= 0 ? "experts, coloured by load" : "dense FFN"}>
                  {moeIdx(l) >= 0 ? (
                    (ld ?? new Array<number>(m.n_experts).fill(NaN)).map((v, e) => (
                      <span
                        key={e}
                        className={`cell expert ${v === 0 ? "dead" : ""}`}
                        style={{ background: Number.isNaN(v) ? "var(--border)" : heatColor(v * m.n_experts / 2) }}
                      />
                    ))
                  ) : (
                    <span className="dense">FFN</span>
                  )}
                </span>
                <span className="resid" title={`residual norm ${fmt(resid)}`}>
                  <span style={{ width: `${(100 * resid) / scales.resid}%` }} />
                </span>
              </button>
            );
          })}
          <div className="stack-row head">token embedding</div>
        </div>
        <div>
          <div className="card">
            <h3>
              Layer {sel} {mi >= 0 ? "(MoE)" : "(dense)"}
            </h3>
            <table className="report">
              <tbody>
                <tr><td>residual norm in → out</td><td>{fmt(stats.residual_norm[sel])} → {fmt(stats.residual_norm[sel + 1])}</td></tr>
                <tr><td>attention output norm</td><td>{fmt(stats.attn_out_norm[sel])}</td></tr>
                <tr><td>FFN output norm</td><td>{fmt(stats.ffn_out_norm[sel])}</td></tr>
                <tr><td>weight norm attn / ffn</td><td>{fmt(snap.weights.attn[sel])} / {fmt(snap.weights.ffn[sel])}</td></tr>
              </tbody>
            </table>
          </div>
          <div className="card">
            <Chart
              height={200}
              option={bars("attention entropy per head (nats)", (stats.attn_entropy[sel] ?? []).map((_, h) => `h${h}`), stats.attn_entropy[sel] ?? [], "#4e79a7")}
            />
          </div>
          {mi >= 0 && snap.moe && (
            <div className="card">
              <Chart
                height={200}
                option={bars(
                  selLoad ? "expert load share since last snapshot" : "router bias (no load counted yet)",
                  Array.from({ length: m.n_experts }, (_, e) => `e${e}`),
                  selLoad ?? snap.moe.bias[mi] ?? [],
                  "#f28e2b",
                )}
              />
              <Chart
                height={200}
                option={bars("expert output norm", Array.from({ length: m.n_experts }, (_, e) => `e${e}`), snap.moe.expert_out_norm[mi] ?? [], "#59a14f")}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
