import { useState } from "react";
import { api, useFetch, useSnapshot } from "../api";
import { Heatmap } from "../components/Heatmap";
import { Empty, Picker, snapLabel } from "../components/Picker";
import type { ArrayResponse, InternalsSnapshot, SnapshotEntry } from "../types";

interface Props {
  run: string;
  snapshot: SnapshotEntry | null;
  version: number;
}

const show = (t: string) => t.replace(/\n/g, "⏎").replace(/ /g, "·") || "∅";

/** Token-to-token arcs for one query position: arc opacity = attention weight. */
function Arcs({ tokens, weights, query, onQuery }: { tokens: string[]; weights: number[]; query: number; onQuery: (q: number) => void }) {
  const step = 46;
  const w = Math.max(tokens.length * step, 200);
  const base = 130;
  const x = (i: number) => i * step + step / 2;
  return (
    <svg className="arcs" width={w} height={base + 40} role="img" aria-label="attention arcs">
      {weights.map((a, k) => {
        if (k > query || a < 0.02) return null;
        const r = Math.abs(x(query) - x(k)) / 2;
        const path = `M ${x(k)} ${base} A ${r} ${Math.min(r, base - 10)} 0 0 1 ${x(query)} ${base}`;
        return <path key={k} d={path} fill="none" stroke="var(--accent)" strokeWidth={1 + 5 * a} opacity={Math.min(0.15 + a, 1)} />;
      })}
      {tokens.map((t, i) => (
        <text
          key={i}
          x={x(i)}
          y={base + 20}
          textAnchor="middle"
          className={`arc-token ${i === query ? "active" : ""}`}
          onClick={() => onQuery(i)}
        >
          {show(t).slice(0, 6)}
        </text>
      ))}
    </svg>
  );
}

/** Attention maps per layer and head for the probe prompts (downsampled to the first
 * ``snapshot.attention_max_tokens`` tokens), as a heatmap and as arcs from one query. */
export function Attention({ run, snapshot, version }: Props) {
  const { data: snap } = useSnapshot<InternalsSnapshot>(run, snapshot, "data", version);
  const [probe, setProbe] = useState<string | null>(null);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [query, setQuery] = useState<number | null>(null);
  const pid = probe ?? snap?.probes[0]?.id ?? null;
  const key = snapshot && pid && snapshot.arrays.includes("data") ? `${run}/${snapshot.stage}/${snapshot.step}/${pid}/${version}` : null;
  const arr = useFetch<ArrayResponse>(key, () => api.array<ArrayResponse>(run, snapshot!.stage, snapshot!.step, "data", `attn_${pid}`));

  if (!snapshot) return <Empty>No internals snapshots yet.</Empty>;
  if (!snap || !pid) return <Empty>loading {snapLabel(snapshot)}…</Empty>;
  if (arr.error) return <p className="error">{arr.error}</p>;
  const maps = arr.data?.data as number[][][][] | undefined;
  if (!maps) return <Empty>loading attention maps…</Empty>;
  const [nl = 0, nh = 0, t = 0] = arr.data?.shape ?? [];
  const l = Math.min(layer, nl - 1);
  const h = Math.min(head, nh - 1);
  const tokens = (snap.probes.find((p) => p.id === pid)?.tokens ?? []).slice(0, t);
  const labels = tokens.map((tk, i) => `${i}:${show(tk)}`);
  const mat = maps[l]?.[h] ?? [];
  const q = Math.min(query ?? t - 1, t - 1);
  const entropy = snap.layers[pid]?.attn_entropy[l]?.[h];

  return (
    <div>
      <div className="toolbar">
        <span className="muted">{snapLabel(snapshot)}</span>
        <Picker label="probe" value={pid} options={snap.probes.map((p) => ({ value: p.id, label: p.id }))} onChange={setProbe} />
        <Picker label="layer" value={l} options={Array.from({ length: nl }, (_, i) => ({ value: i, label: `L${i}` }))} onChange={setLayer} />
        <Picker label="head" value={h} options={Array.from({ length: nh }, (_, i) => ({ value: i, label: `h${i}` }))} onChange={setHead} />
        {entropy !== undefined && <span className="muted">mean entropy {entropy.toFixed(2)} nats</span>}
      </div>
      <div className="grid">
        <div className="card">
          <h3>query (row) attends to key (column) — click a row for its arcs</h3>
          <Heatmap values={mat} rows={labels} cols={labels} min={0} max={1} xName="key" yName="query" onCell={(r) => setQuery(r)} height={Math.max(300, 16 * t + 120)} />
        </div>
        <div className="card scroll-x">
          <h3>arcs from query {q} ({show(tokens[q] ?? "")})</h3>
          <Arcs tokens={tokens} weights={mat[q] ?? []} query={q} onQuery={setQuery} />
          <h3>all heads, layer {l}: weight on each key from query {q}</h3>
          <Heatmap
            values={Array.from({ length: nh }, (_, hh) => maps[l]?.[hh]?.[q] ?? [])}
            rows={Array.from({ length: nh }, (_, hh) => `h${hh}`)}
            cols={labels}
            min={0}
            max={1}
            xName="key"
            yName="head"
            onCell={(r) => setHead(r)}
          />
        </div>
      </div>
    </div>
  );
}
