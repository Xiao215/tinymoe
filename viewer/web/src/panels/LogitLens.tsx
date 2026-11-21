import { useMemo, useState } from "react";
import { useSnapshot, useSnapshotSeries, type RunData } from "../api";
import { Heatmap } from "../components/Heatmap";
import { Empty, Picker, snapLabel } from "../components/Picker";
import { heatColor } from "../lib";
import { sequentialInk } from "../theme";
import type { InternalsSnapshot, LensPosition, SnapshotEntry } from "../types";

interface Props {
  run: string;
  data: RunData;
  snapshot: SnapshotEntry | null;
}

const show = (t: string) => t.replace(/\n/g, "⏎").replace(/ /g, "·") || "∅";

/** p(target) at each layer: from the recorded top-k, so 0 means "not in the top-k". */
const targetProbs = (lp: LensPosition) => lp.layers.map((top) => top.find(([t]) => t === lp.target)?.[1] ?? 0);

/** Logit lens (nostalgebraist, 2020): every layer's residual stream decoded through the
 * final norm + head. Rows are layers (0 = embeddings); later layers should sharpen onto
 * the target, and over training the "right answer" appears at ever earlier layers. */
export function LogitLens({ run, data, snapshot }: Props) {
  const { data: snap } = useSnapshot<InternalsSnapshot>(run, snapshot, "data", data.version);
  const internals = useMemo(() => data.snapshots.filter((s) => s.files.includes("data")), [data.snapshots]);
  const history = useSnapshotSeries<InternalsSnapshot>(run, internals, "data", data.version);
  const [probe, setProbe] = useState<string | null>(null);
  const [posIdx, setPosIdx] = useState(0);

  if (!snapshot) return <Empty>No internals snapshots yet.</Empty>;
  if (!snap) return <Empty>loading {snapLabel(snapshot)}…</Empty>;
  const pid = probe ?? Object.keys(snap.logit_lens)[0] ?? "";
  const positions = snap.logit_lens[pid] ?? [];
  if (!positions.length) return <Empty>No logit-lens positions recorded for this probe.</Empty>;
  const pi = Math.min(posIdx, positions.length - 1);
  const lp = positions[pi]!;
  const layers = lp.layers.map((_, l) => (l === 0 ? "emb" : `L${l - 1}`));
  const over = history
    .map((h) => ({ label: `${h.entry.stage}@${h.entry.step}`, lens: h.data.logit_lens[pid]?.find((p) => p.pos === lp.pos) }))
    .filter((o): o is { label: string; lens: LensPosition } => o.lens !== undefined);

  return (
    <div>
      <div className="toolbar">
        <span className="muted">{snapLabel(snapshot)}</span>
        <Picker label="probe" value={pid} options={Object.keys(snap.logit_lens).map((p) => ({ value: p, label: p }))} onChange={setProbe} />
        <Picker
          label="position"
          value={pi}
          options={positions.map((p, k) => ({ value: k, label: `${p.pos}: ${show(p.token)} → ${show(p.target)}` }))}
          onChange={setPosIdx}
        />
      </div>
      <div className="grid">
        <div className="card">
          <h3>
            top-k after "{show(lp.token)}" (target <b>{show(lp.target)}</b>)
          </h3>
          <table className="lens">
            <tbody>
              {lp.layers
                .map((top, l) => ({ top, l }))
                .reverse()
                .map(({ top, l }) => (
                  <tr key={l}>
                    <th>{layers[l]}</th>
                    {top.map(([tok, p], k) => (
                      <td key={k} className={tok === lp.target ? "target" : ""} style={{ background: heatColor(p), color: sequentialInk(p) }} title={p.toFixed(4)}>
                        {show(tok)} <small>{(100 * p).toFixed(0)}%</small>
                      </td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h3>p(target) per layer over training (0 = outside the recorded top-k)</h3>
          {over.length ? (
            <Heatmap values={over.map((o) => targetProbs(o.lens))} rows={over.map((o) => o.label)} cols={layers} min={0} max={1} xName="layer" yName="snapshot" />
          ) : (
            <Empty>loading history…</Empty>
          )}
        </div>
      </div>
    </div>
  );
}
