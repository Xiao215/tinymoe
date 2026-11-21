import type * as echarts from "echarts";
import { useMemo } from "react";
import { Chart } from "./Chart";

interface Props {
  /** values[row][col]; rows are drawn top to bottom */
  values: number[][];
  rows: string[];
  cols: string[];
  height?: number;
  min?: number;
  max?: number;
  xName?: string;
  yName?: string;
  onCell?: (row: number, col: number) => void;
  digits?: number;
}

/** Matrix heatmap with labelled axes (attention maps, expert load, lens probabilities). */
export function Heatmap({ values, rows, cols, height, min, max, xName, yName, onCell, digits = 3 }: Props) {
  const option = useMemo((): echarts.EChartsOption => {
    const cells: [number, number, number][] = [];
    let lo = Infinity;
    let hi = -Infinity;
    values.forEach((row, r) =>
      row.forEach((v, c) => {
        cells.push([c, rows.length - 1 - r, v]);
        if (Number.isFinite(v)) {
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
      }),
    );
    return {
      tooltip: {
        formatter: (p: unknown) => {
          const d = (p as { data: [number, number, number] }).data;
          const row = rows[rows.length - 1 - d[1]] ?? "";
          return `${yName ?? "row"} ${row}<br/>${xName ?? "col"} ${cols[d[0]] ?? ""}<br/><b>${d[2].toFixed(digits)}</b>`;
        },
      },
      grid: { left: 70, right: 20, top: 10, bottom: 70 },
      xAxis: { type: "category", data: cols, name: xName, axisLabel: { rotate: cols.length > 12 ? 60 : 0, fontSize: 10 } },
      yAxis: { type: "category", data: [...rows].reverse(), name: yName, axisLabel: { fontSize: 10 } },
      visualMap: {
        min: min ?? (Number.isFinite(lo) ? lo : 0),
        max: max ?? (Number.isFinite(hi) && hi > lo ? hi : lo + 1),
        calculable: true,
        precision: Math.min(digits, 2),
        orient: "horizontal",
        left: "center",
        bottom: 0,
        itemHeight: 120,
        inRange: { color: ["#eef5fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"] },
        itemWidth: 10,
      },
      series: [{ type: "heatmap", data: cells, progressive: 0, itemStyle: { borderWidth: 1, borderColor: "transparent" } }],
    };
  }, [values, rows, cols, min, max, xName, yName, digits]);
  const onClick = useMemo(
    () =>
      onCell
        ? (p: echarts.ECElementEvent) => {
            const d = p.data as [number, number, number] | undefined;
            if (d) onCell(rows.length - 1 - d[1], d[0]);
          }
        : undefined,
    [onCell, rows.length],
  );
  return <Chart option={option} height={height ?? Math.max(220, 26 * rows.length + 90)} onClick={onClick} />;
}
