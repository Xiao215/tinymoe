import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

interface Theme {
  dark: boolean;
  toggle: () => void;
}

const ThemeContext = createContext<Theme>({ dark: false, toggle: () => undefined });

const isDark = () => document.documentElement.dataset.theme === "dark";

export function ThemeProvider({ children }: { children: ReactNode }) {
  const prefers = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  const stored = localStorage.getItem("tinymoe-theme");
  const [dark, setDark] = useState(stored ? stored === "dark" : prefers);
  // set synchronously so colour helpers called during this render see the right mode
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  useEffect(() => {
    localStorage.setItem("tinymoe-theme", dark ? "dark" : "light");
  }, [dark]);
  return <ThemeContext.Provider value={{ dark, toggle: () => setDark((d) => !d) }}>{children}</ThemeContext.Provider>;
}

export const useTheme = () => useContext(ThemeContext);

/** Lifecycle order: stage colours follow it (categorical slots 1-6 of the validated
 * palette, so adjacent stages always clear the colour-vision gates). */
export const STAGE_ORDER = ["pretrain", "midtrain", "sft", "dpo", "grpo", "distill"] as const;

/** Categorical palette, fixed order, light and dark steps selected separately. */
export const CATEGORICAL = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
};

export const categorical = (i: number): string => {
  const pal = isDark() ? CATEGORICAL.dark : CATEGORICAL.light;
  return pal[((i % pal.length) + pal.length) % pal.length]!;
};

export function stageColor(name: string): string {
  const i = (STAGE_ORDER as readonly string[]).indexOf(name);
  return i < 0 ? (isDark() ? "#898781" : "#898781") : categorical(i);
}

/** Chrome colours for charts (ECharts cannot read CSS variables). */
export function chartInk() {
  return isDark()
    ? { surface: "#1a1a19", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835", good: "#0ca30c", bad: "#e66767" }
    : { surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7", good: "#006300", bad: "#d03b3b" };
}

/** Sequential blue ramp (steps 100-700 of the palette's blue), t in [0, 1]. */
const BLUE = ["#eef5fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];

function mix(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return `#${pa.map((v, i) => Math.round(v + (pb[i]! - v) * t).toString(16).padStart(2, "0")).join("")}`;
}

export function sequential(t: number): string {
  const x = Math.min(Math.max(Number.isFinite(t) ? t : 0, 0), 1) * (BLUE.length - 1);
  const i = Math.min(Math.floor(x), BLUE.length - 2);
  return mix(BLUE[i]!, BLUE[i + 1]!, x - i);
}

/** Readable ink on a sequential cell. */
export const sequentialInk = (t: number): string => (t > 0.5 ? "#ffffff" : "#0b0b0b");
