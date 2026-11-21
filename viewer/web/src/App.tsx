import { useEffect, useMemo, useState, type ReactNode } from "react";
import { api, useCompare, useFetch, useRun, type RunData } from "./api";
import { Timeline } from "./components/Timeline";
import { Anatomy } from "./panels/Anatomy";
import { Attention } from "./panels/Attention";
import { Embeddings } from "./panels/Embeddings";
import { Generations } from "./panels/Generations";
import { LogitLens } from "./panels/LogitLens";
import { MoE } from "./panels/MoE";
import { Report } from "./panels/Report";
import { Rollouts } from "./panels/Rollouts";
import { Scalars } from "./panels/Scalars";
import { useTheme } from "./theme";
import type { SnapshotEntry } from "./types";

export interface PanelProps {
  run: string;
  data: RunData;
  /** internals snapshot chosen with the timeline scrubber (null = latest) */
  snapshot: SnapshotEntry | null;
  compare: string[];
}

interface PanelDef {
  id: string;
  title: string;
  render: (p: PanelProps) => ReactNode;
}

function ScalarsPanel({ data, compare }: PanelProps) {
  const others = useCompare(compare);
  return <Scalars records={data.records} timeline={data.timeline} compare={others} />;
}

export const PANELS: PanelDef[] = [
  { id: "scalars", title: "Scalars", render: (p) => <ScalarsPanel {...p} /> },
  { id: "anatomy", title: "Anatomy", render: (p) => <Anatomy run={p.run} snapshot={p.snapshot} version={p.data.version} /> },
  { id: "attention", title: "Attention", render: (p) => <Attention run={p.run} snapshot={p.snapshot} version={p.data.version} /> },
  { id: "moe", title: "MoE", render: (p) => <MoE run={p.run} data={p.data} snapshot={p.snapshot} /> },
  { id: "lens", title: "Logit lens", render: (p) => <LogitLens run={p.run} data={p.data} snapshot={p.snapshot} /> },
  { id: "embeddings", title: "Embeddings", render: (p) => <Embeddings run={p.run} data={p.data} snapshot={p.snapshot} /> },
  {
    id: "generations",
    title: "Generations",
    render: (p) => <Generations run={p.run} snapshots={p.data.snapshots} version={p.data.version} />,
  },
  { id: "rollouts", title: "RL rollouts", render: (p) => <Rollouts run={p.run} data={p.data} /> },
  { id: "report", title: "Report", render: (p) => <Report run={p.run} data={p.data} /> },
];

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
  const [hashRun, hashPanel] = hash.split("/");
  const run = hashRun || runs.data?.[0]?.id || null;
  const panel = PANELS.find((p) => p.id === hashPanel) ?? PANELS[0]!;
  const [compare, setCompare] = useState<string[]>([]);
  const data = useRun(run);
  const internals = useMemo(() => data.snapshots.filter((s) => s.files.includes("data")), [data.snapshots]);
  const [selected, setSelected] = useState<number | null>(null);
  useEffect(() => setSelected(null), [run]);
  const snapshot = selected !== null ? (internals[selected] ?? null) : (internals[internals.length - 1] ?? null);

  return (
    <div className="app">
      <header>
        <h1>TinyMoE</h1>
        <select value={run ?? ""} onChange={(e) => setHash(`${e.target.value}/${panel.id}`)} aria-label="run">
          {(runs.data ?? []).map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} ({r.stages.map((s) => s.name).join(" → ") || "no stages"})
            </option>
          ))}
        </select>
        <details className="compare">
          <summary>compare ({compare.length})</summary>
          {(runs.data ?? [])
            .filter((r) => r.id !== run)
            .map((r) => (
              <label key={r.id}>
                <input
                  type="checkbox"
                  checked={compare.includes(r.id)}
                  onChange={() => setCompare((c) => (c.includes(r.id) ? c.filter((x) => x !== r.id) : [...c, r.id]))}
                />
                {r.id}
              </label>
            ))}
        </details>
        <span className={`live ${data.live ? "on" : ""}`} title="live updates">
          {data.live ? "● live" : "○ offline"}
        </span>
        <button onClick={toggle} aria-label="toggle theme">
          {dark ? "☀ light" : "☾ dark"}
        </button>
      </header>
      {runs.error && <p className="error">{runs.error}</p>}
      {data.error && <p className="error">{data.error}</p>}
      {!run && runs.data && <p className="muted">No runs found. Start one with `tinymoe pipeline configs/pipeline/smoke.yaml`.</p>}
      {run && (
        <>
          <Timeline timeline={data.timeline} snapshots={internals} selected={selected} onSelect={setSelected} />
          <nav className="tabs">
            {PANELS.map((p) => (
              <button key={p.id} className={p.id === panel.id ? "active" : ""} onClick={() => setHash(`${run}/${p.id}`)}>
                {p.title}
              </button>
            ))}
          </nav>
          <main>{panel.render({ run, data, snapshot, compare })}</main>
        </>
      )}
    </div>
  );
}
