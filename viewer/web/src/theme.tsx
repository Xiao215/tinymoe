import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

interface Theme {
  dark: boolean;
  toggle: () => void;
}

const ThemeContext = createContext<Theme>({ dark: false, toggle: () => undefined });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const prefers = window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
  const stored = localStorage.getItem("tinylab-theme");
  const [dark, setDark] = useState(stored ? stored === "dark" : prefers);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    localStorage.setItem("tinylab-theme", dark ? "dark" : "light");
  }, [dark]);
  return (
    <ThemeContext.Provider value={{ dark, toggle: () => setDark((d) => !d) }}>
      {children}
    </ThemeContext.Provider>
  );
}

export const useTheme = () => useContext(ThemeContext);

/** Stage colours: stable by pipeline position, readable in both themes. */
export const STAGE_COLORS: Record<string, string> = {
  pretrain: "#4e79a7",
  midtrain: "#59a14f",
  sft: "#f28e2b",
  dpo: "#e15759",
  grpo: "#b07aa1",
  distill: "#76b7b2",
};

export const stageColor = (name: string) => STAGE_COLORS[name] ?? "#9c755f";
