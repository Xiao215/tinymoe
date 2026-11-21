import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api, useCompare, useFetch, useRun, type RunData } from "./api";
import { RunPicker } from "./components/RunPicker";
import { Timeline } from "./components/Timeline";
import { Anatomy } from "./panels/Anatomy";
import { Attention } from "./panels/Attention";
import { Embeddings } from "./panels/Embeddings";
import { Generations } from "./panels/Generations";
import { LogitLens } from "./panels/LogitLens";
import { MoE } from "./panels/MoE";
import { Report } from "./panels/Report";
import { Rollouts } from "./panels/Rollouts";
import { Overview } from "./panels/Overview";
import { Scalars } from "./panels/Scalars";
import { stageStarts } from "./lib";
import { useTheme } from "./theme";
import type { SnapshotEntry } from "./types";

export interface PanelProps {
  run: string;
  data: RunData;
  /** internals snapshot chosen with the timeline scrubber (null = latest) */
  snapshot: SnapshotEntry | null;
  compare: string[];
  open: (panel: string) => void;
}

interface PanelDef {
  id: string;
  title: string;
  render: (p: PanelProps) => ReactNode;
  /** driven by the snapshot scrubber */
  snap?: boolean;
}

function ScalarsPanel({ data, compare }: PanelProps) {
  const others = useCompare(compare);
  const starts = useMemo(() => stageStarts(data.manifest), [data.manifest]);
  return <Scalars records={data.records} timeline={data.timeline} compare={others} starts={starts} />;
}

export const PANELS: PanelDef[] = [
  { id: "overview", title: "Overview", render: (p) => <Overview run={p.run} data={p.data} onOpen={p.open} /> },
  { id: "metrics", title: "Metrics", render: (p) => <ScalarsPanel {...p} /> },
  { id: "anatomy", title: "Anatomy", snap: true, render: (p) => <Anatomy run={p.run} snapshot={p.snapshot} version={p.data.version} /> },
  { id: "attention", title: "Attention", snap: true, render: (p) => <Attention run={p.run} snapshot={p.snapshot} version={p.data.version} /> },
  { id: "moe", title: "MoE", snap: true, render: (p) => <MoE run={p.run} data={p.data} snapshot={p.snapshot} /> },
  { id: "lens", title: "Logit lens", snap: true, render: (p) => <LogitLens run={p.run} data={p.data} snapshot={p.snapshot} /> },
  { id: "embeddings", title: "Embeddings", snap: true, render: (p) => <Embeddings run={p.run} data={p.data} snapshot={p.snapshot} /> },
  {
    id: "generations",
    title: "Generations",
    render: (p) => <Generations run={p.run} snapshots={p.data.snapshots} version={p.data.version} />,
  },
  { id: "rollouts", title: "RL rollouts", render: (p) => <Rollouts run={p.run} data={p.data} /> },
  { id: "report", title: "Report", render: (p) => <Report run={p.run} data={p.data} /> },
];

const ALIASES: Record<string, string> = { scalars: "metrics" };

function useHashState(): [string, (v: string) => void] {
  const [hash, setHash] = useState(() => window.location.hash.slice(1));
  useEffect(() => {
    const on = () => setHash(window.location.hash.slice(1));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [hash, (v) => (window.location.hash = v)];
}

export function App() {
  const { dark, toggle } = useTheme();
  const runs = useFetch("runs", api.runs);
  const [hash, setHash] = useHashState();
  const [hashRun, rawPanel] = hash.split("/");
  const hashPanel = ALIASES[rawPanel ?? ""] ?? rawPanel;
  const run = hashRun || runs.data?.[0]?.id || null;
  const panel = PANELS.find((p) => p.id === hashPanel) ?? PANELS[0]!;
  const [compare, setCompare] = useState<string[]>([]);
  const data = useRun(run);
  const internals = useMemo(() => data.snapshots.filter((s) => s.files.includes("data")), [data.snapshots]);
  const [selected, setSelected] = useState<number | null>(null);
  useEffect(() => setSelected(null), [run]);
  const snapshot = selected !== null ? (internals[selected] ?? null) : (internals[internals.length - 1] ?? null);
  const open = (id: string) => run && setHash(`${run}/${id}`);
  const current = data.manifest?.stages.find((s) => s.status === "running") ?? data.manifest?.stages[data.manifest.stages.length - 1];

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-row">
          <div className="brand">
            <b>TinyMoE</b>
            <span>lab</span>
          </div>
          {run && (
            <nav className="tabs">
              {PANELS.map((p) => (
                <button key={p.id} className={p.id === panel.id ? "active" : ""} onClick={() => open(p.id)}>
                  {p.title}
                </button>
              ))}
            </nav>
          )}
          <div className="topbar-meta">
            <RunPicker
              runs={runs.data ?? []}
              run={run}
              compare={compare}
              onOpen={(id) => {
                setCompare((c) => c.filter((x) => x !== id));
                setHash(`${id}/${panel.id}`);
              }}
              onToggleCompare={(id) => setCompare((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c, id]))}
            />
            {current && (
              <div className="meta">
                <label>stage</label>
                <span>{current.name}</span>
              </div>
            )}
            <span className={`live${data.live ? " on" : ""}`} title={data.live ? "receiving live updates" : "not connected"}>
              <i />
              {data.live ? "live" : "offline"}
            </span>
            <button className="icon-btn" onClick={toggle} aria-label={dark ? "switch to light theme" : "switch to dark theme"} title={dark ? "light theme" : "dark theme"}>
              {dark ? "☀" : "☾"}
            </button>
          </div>
        </div>
      </header>
      <main key={dark ? "dark" : "light"}>
        {runs.error && <p className="error">{runs.error}</p>}
        {data.error && <p className="error">{data.error}</p>}
        {!run && runs.data && <p className="muted empty">No runs found. Start one with `tinymoe pipeline configs/pipeline/smoke.yaml`.</p>}
        {run && (
          <>
            {panel.snap && <Timeline timeline={data.timeline} snapshots={internals} selected={selected} onSelect={setSelected} />}
            {panel.render({ run, data, snapshot, compare, open })}
          </>
        )}
      </main>
    </div>
  );
}
