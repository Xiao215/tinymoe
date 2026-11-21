import type * as echarts from "echarts";
import { useEffect, useMemo, useState } from "react";
import { useSnapshotSeries, type RunData } from "../api";
import { Chart } from "../components/Chart";
import { Empty } from "../components/Picker";
import { paletteColor, tokenClass } from "../lib";
import type { InternalsSnapshot, SnapshotEntry } from "../types";

interface Props {
  run: string;
  data: RunData;
  snapshot: SnapshotEntry | null;
}

const CLASSES = ["digit", "operator", "word", "punct", "special", "space"];

/** Token-embedding PCA (digits, operators, common words) per snapshot, animated: watch the
 * digits line up in order as the model learns arithmetic. Distillation snapshots show the
 * student's own embedding table (its width is shown next to the step). */
export function Embeddings({ run, data, snapshot }: Props) {
  const internals = useMemo(() => data.snapshots.filter((s) => s.files.includes("data")), [data.snapshots]);
  const history = useSnapshotSeries<InternalsSnapshot>(run, internals, "data", data.version);
  const frames = useMemo(() => history.filter((h) => h.data.embeddings.xy.length > 0), [history]);
  const [idx, setIdx] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);

  const synced = snapshot ? frames.findIndex((f) => f.entry.stage === snapshot.stage && f.entry.step === snapshot.step) : -1;
  const cur = Math.min(idx ?? (synced >= 0 ? synced : frames.length - 1), frames.length - 1);

  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const t = window.setInterval(() => setIdx((i) => ((i ?? 0) + 1) % frames.length), 700);
    return () => window.clearInterval(t);
  }, [playing, frames.length]);

  const frame = frames[cur];
  const option = useMemo((): echarts.EChartsOption | null => {
    if (!frame) return null;
    const { tokens, xy } = frame.data.embeddings;
    const all = xy.flat();
    const lim = Number((Math.max(...all.map(Math.abs), 1e-6) * 1.1).toPrecision(2));
    const tick = { formatter: (v: number) => v.toFixed(1) };
    return {
      animationDurationUpdate: 600,
      tooltip: { formatter: (p: unknown) => String((p as { data: { name: string } }).data.name) },
      legend: { top: 0 },
      grid: { left: 40, right: 20, top: 40, bottom: 30 },
      xAxis: { type: "value", min: -lim, max: lim, name: "PC1", axisLabel: tick },
      yAxis: { type: "value", min: -lim, max: lim, name: "PC2", axisLabel: tick },
      series: CLASSES.map((c, ci) => ({
        name: c,
        type: "scatter",
        symbolSize: 9,
        itemStyle: { color: paletteColor(ci) },
        label: { show: true, formatter: "{b}", position: "right", fontSize: 10 },
        data: tokens
          .map((t, k) => ({ name: t.replace(/\n/g, "⏎") || "∅", value: xy[k] ?? [0, 0], cls: tokenClass(t) }))
          .filter((d) => d.cls === c)
          .map(({ name, value }) => ({ name, value })),
      })),
    };
  }, [frame]);

  if (!internals.length) return <Empty>No internals snapshots yet.</Empty>;
  if (!frame || !option) return <Empty>loading embeddings…</Empty>;
  return (
    <div className="card">
      <div className="toolbar">
        <button onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ pause" : "▶ play"}</button>
        <input
          type="range"
          min={0}
          max={frames.length - 1}
          value={cur}
          onChange={(e) => {
            setPlaying(false);
            setIdx(Number(e.target.value));
          }}
          aria-label="snapshot"
        />
        <span>
          {frame.entry.stage} step {frame.entry.step} · d_model {frame.data.model.d_model}
        </span>
      </div>
      <Chart option={option} height={520} />
    </div>
  );
}
