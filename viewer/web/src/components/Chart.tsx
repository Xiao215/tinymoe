import * as echarts from "echarts";
import { useEffect, useRef } from "react";
import { CATEGORICAL, useTheme } from "../theme";

interface Props {
  option: echarts.EChartsOption;
  height?: number;
  onClick?: (params: echarts.ECElementEvent) => void;
}

const MONO = 'ui-monospace, "SF Mono", "JetBrains Mono", Menlo, monospace';
const SANS = 'system-ui, -apple-system, "Segoe UI", sans-serif';

/** One chart look for every panel: hairline grid, recessive mono axes, 2px lines,
 * flat tooltip. Light and dark are separate selections, not an inversion. */
function makeTheme(dark: boolean) {
  const c = dark
    ? { surface: "#1a1a19", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835", border: "rgba(255,255,255,0.12)" }
    : { surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7", border: "rgba(11,11,11,0.12)" };
  const axis = {
    axisLine: { show: true, lineStyle: { color: c.axis } },
    axisTick: { show: false },
    axisLabel: { color: c.muted, fontFamily: MONO, fontSize: 10 },
    splitLine: { show: true, lineStyle: { color: c.grid, width: 1 } },
    nameTextStyle: { color: c.muted, fontFamily: MONO, fontSize: 10 },
  };
  return {
    color: dark ? CATEGORICAL.dark : CATEGORICAL.light,
    backgroundColor: "transparent",
    textStyle: { fontFamily: SANS, color: c.ink2 },
    title: { textStyle: { color: c.ink, fontSize: 12, fontWeight: 600 } },
    legend: { textStyle: { color: c.ink2, fontFamily: MONO, fontSize: 10 }, icon: "roundRect", itemWidth: 10, itemHeight: 3, pageTextStyle: { color: c.muted } },
    tooltip: {
      backgroundColor: c.surface,
      borderColor: c.border,
      borderWidth: 1,
      padding: [6, 9],
      textStyle: { color: c.ink, fontFamily: MONO, fontSize: 11 },
      extraCssText: "box-shadow:none;border-radius:6px;",
      axisPointer: { lineStyle: { color: c.axis, width: 1 }, crossStyle: { color: c.axis }, label: { backgroundColor: c.ink2 } },
    },
    categoryAxis: { ...axis, splitLine: { show: false } },
    valueAxis: { ...axis, axisLine: { show: false } },
    logAxis: { ...axis, axisLine: { show: false } },
    timeAxis: axis,
    line: { lineStyle: { width: 2 }, symbolSize: 6, symbol: "circle", smooth: false },
    bar: { itemStyle: { borderRadius: [3, 3, 0, 0] } },
    scatter: { symbolSize: 7 },
    dataZoom: {
      borderColor: "transparent",
      backgroundColor: "transparent",
      fillerColor: dark ? "rgba(255,255,255,0.06)" : "rgba(11,11,11,0.05)",
      handleStyle: { color: c.surface, borderColor: c.axis },
      dataBackground: { lineStyle: { color: c.axis }, areaStyle: { color: c.grid } },
      textStyle: { color: c.muted, fontFamily: MONO, fontSize: 10 },
    },
    visualMap: { textStyle: { color: c.muted, fontFamily: MONO, fontSize: 10 } },
    markLine: { lineStyle: { color: c.axis } },
  };
}

echarts.registerTheme("tinymoe-light", makeTheme(false));
echarts.registerTheme("tinymoe-dark", makeTheme(true));

/** Thin ECharts wrapper: one instance per mount, re-themed and resized automatically. */
export function Chart({ option, height = 260, onClick }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const { dark } = useTheme();

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, dark ? "tinymoe-dark" : "tinymoe-light", { renderer: "canvas" });
    chart.current = c;
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
  }, [dark]);

  useEffect(() => {
    chart.current?.setOption({ animation: false, backgroundColor: "transparent", ...option }, { notMerge: true });
  }, [option, dark]);

  useEffect(() => {
    const c = chart.current;
    if (!c || !onClick) return;
    c.on("click", onClick);
    return () => {
      c.off("click", onClick);
    };
  }, [onClick, dark]);

  return <div ref={el} style={{ width: "100%", height }} />;
}
