import type { EChartsOption, LineSeriesOption } from "echarts";
import { useMemo, useState } from "react";
import { Chart } from "../components/Chart";
import { defaultGroups, metricKeys, series, stageAreas, type MetricGroup } from "../lib";
import type { MetricRecord, Span } from "../types";

interface Props {
  records: MetricRecord[];
  timeline: Span[];
  compare: Record<string, MetricRecord[]>;
}

function groupOption(
  group: MetricGroup,
  records: MetricRecord[],
  timeline: Span[],
  compare: Record<string, MetricRecord[]>,
): EChartsOption {
  const lines: LineSeriesOption[] = group.keys.map((key, i) => ({
    name: key,
    type: "line",
    showSymbol: false,
    sampling: "lttb",
    connectNulls: true,
    data: series(records, key),
    ...(i === 0 ? { markArea: stageAreas(timeline) } : {}),
  }));
  for (const [run, recs] of Object.entries(compare)) {
    for (const key of group.keys) {
      const data = series(recs, key);
      if (data.length) {
        lines.push({
          name: `${key} [${run}]`,
          type: "line",
          showSymbol: false,
          lineStyle: { type: "dashed" },
          data,
        });
      }
    }
  }
  const sparse = lines.every((l) => (l.data as unknown[]).length < 30);
  for (const l of lines) l.showSymbol = sparse;
  return {
    animation: false,
    tooltip: { trigger: "axis", valueFormatter: (v) => (typeof v === "number" ? v.toPrecision(4) : String(v)) },
    legend: { type: "scroll", top: 0, textStyle: { fontSize: 10 } },
    grid: { left: 50, right: 16, top: 34, bottom: 40 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 14, bottom: 6 }],
    xAxis: { type: "value", name: "global step", nameLocation: "middle", nameGap: 22, min: "dataMin", max: "dataMax" },
    yAxis: { type: group.log ? "log" : "value", scale: true },
    series: lines,
  };
}

/** Every scalar across the whole pipeline, grouped; other runs overlay as dashed lines. */
export function Scalars({ records, timeline, compare }: Props) {
  const [filter, setFilter] = useState("");
  const keys = useMemo(() => metricKeys(records), [records]);
  const groups = useMemo(() => {
    const all = defaultGroups(keys);
    if (!filter) return all;
    const f = filter.toLowerCase();
    return all
      .map((g) => ({ ...g, keys: g.keys.filter((k) => k.toLowerCase().includes(f) || g.title.toLowerCase().includes(f)) }))
      .filter((g) => g.keys.length);
  }, [keys, filter]);

  if (!records.length) return <p className="muted">No metrics logged yet.</p>;
  return (
    <div>
      <div className="toolbar">
        <input placeholder="filter metrics (e.g. loss, moe, pass1)" value={filter} onChange={(e) => setFilter(e.target.value)} />
        <span className="muted">
          {keys.length} metrics · {records.length} records
        </span>
      </div>
      <div className="grid">
        {groups.map((g) => (
          <section className="card" key={g.title}>
            <h3>{g.title}</h3>
            <Chart option={groupOption(g, records, timeline, compare)} />
          </section>
        ))}
      </div>
    </div>
  );
}
