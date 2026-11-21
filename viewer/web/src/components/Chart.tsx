import * as echarts from "echarts";
import { useEffect, useRef } from "react";
import { useTheme } from "../theme";

interface Props {
  option: echarts.EChartsOption;
  height?: number;
  onClick?: (params: echarts.ECElementEvent) => void;
}

/** Thin ECharts wrapper: one instance per mount, re-themed and resized automatically. */
export function Chart({ option, height = 260, onClick }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const { dark } = useTheme();

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, dark ? "dark" : undefined, { renderer: "canvas" });
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
    chart.current?.setOption({ backgroundColor: "transparent", ...option }, { notMerge: true });
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
