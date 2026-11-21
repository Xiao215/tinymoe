import { useMemo, useState } from "react";
import { MetricChart, type XMode } from "../components/MetricChart";
import { defaultGroups, metricKeys, type MetricGroup } from "../lib";
import type { MetricRecord, Span } from "../types";

interface Props {
  records: MetricRecord[];
  timeline: Span[];
  compare: Record<string, MetricRecord[]>;
  starts: Record<string, number>;
}

type YMode = "auto" | "linear" | "log";

/** "loss" (substring, case-insensitive) or "/^val\/.*math$/" (regex). Bad regex = no filter. */
function matcher(filter: string): (k: string) => boolean {
  const f = filter.trim();
  if (!f) return () => true;
  const re = /^\/(.+)\/([a-z]*)$/.exec(f);
  if (re) {
    try {
      const r = new RegExp(re[1]!, re[2] || "i");
      return (k) => r.test(k);
    } catch {
      return () => true;
    }
  }
  const lower = f.toLowerCase();
  return (k) => k.toLowerCase().includes(lower);
}

function Seg<T extends string>({ value, options, onChange, label }: { value: T; options: T[]; onChange: (v: T) => void; label: string }) {
  return (
    <span className="ctl">
      {label}
      <span className="seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button key={o} className={o === value ? "on" : ""} onClick={() => onChange(o)}>
            {o}
          </button>
        ))}
      </span>
    </span>
  );
}

/** Every scalar across the whole pipeline as small multiples, grouped in a sidebar tree;
 * other runs overlay as dashed lines. */
export function Scalars({ records, timeline, compare, starts }: Props) {
  const [filter, setFilter] = useState("");
  const [section, setSection] = useState<string | null>(null);
  const [x, setX] = useState<XMode>("step");
  const [y, setY] = useState<YMode>("auto");
  const [smoothing, setSmoothing] = useState(0.5);
  const keys = useMemo(() => metricKeys(records), [records]);
  const all = useMemo(() => defaultGroups(keys), [keys]);
  const groups = useMemo(() => {
    const ok = matcher(filter);
    return all
      .map((g): MetricGroup => (ok(g.title) ? g : { ...g, keys: g.keys.filter(ok) }))
      .filter((g) => g.keys.length);
  }, [all, filter]);
  const sections = useMemo(() => {
    const m = new Map<string, { groups: number; keys: number }>();
    for (const g of groups) {
      const e = m.get(g.section) ?? { groups: 0, keys: 0 };
      e.groups += 1;
      e.keys += g.keys.length;
      m.set(g.section, e);
    }
    return [...m.entries()];
  }, [groups]);
  const shown = section ? groups.filter((g) => g.section === section) : groups;
  const canTime = Object.keys(starts).length > 0;

  if (!records.length) return <p className="muted empty">No metrics logged yet.</p>;
  return (
    <div className="metrics-layout">
      <aside className="tree">
        <input placeholder="filter (substring or /regex/)" value={filter} onChange={(e) => setFilter(e.target.value)} aria-label="filter metrics" />
        <ul>
          <li>
            <button className={section === null ? "on" : ""} onClick={() => setSection(null)}>
              all <span className="num">{groups.reduce((a, g) => a + g.keys.length, 0)}</span>
            </button>
          </li>
          {sections.map(([name, c]) => (
            <li key={name}>
              <button className={section === name ? "on" : ""} onClick={() => setSection(section === name ? null : name)}>
                {name}/ <span className="num">{c.keys}</span>
              </button>
            </li>
          ))}
        </ul>
        <p className="muted" style={{ fontSize: 11, marginTop: 14 }}>
          {keys.length} metrics · {records.length} records
          <br />
          click a legend entry to hide it · shift+scroll zooms
        </p>
      </aside>
      <div>
        <div className="toolbar">
          <Seg label="x" value={x} options={canTime ? ["step", "time"] : ["step"]} onChange={setX} />
          <Seg label="y" value={y} options={["auto", "linear", "log"]} onChange={setY} />
          <label className="ctl">
            smoothing
            <input type="range" min={0} max={0.95} step={0.05} value={smoothing} onChange={(e) => setSmoothing(Number(e.target.value))} />
            <span className="num">{smoothing.toFixed(2)}</span>
          </label>
          {Object.keys(compare).length > 0 && <span className="ctl">overlay: {Object.keys(compare).join(", ")} (dashed)</span>}
        </div>
        <div className="chips">
          {shown.map((g) => (
            <button
              key={g.title}
              className="chip"
              onClick={() => document.getElementById(`metric-${g.title}`)?.scrollIntoView({ behavior: "smooth", block: "center" })}
            >
              {g.title} <b>{g.keys.length}</b>
            </button>
          ))}
        </div>
        <div className="grid3">
          {shown.map((g) => (
            <div key={g.title} id={`metric-${g.title}`}>
              <MetricChart
                title={g.title}
                hint={g.hint}
                keys={g.keys}
                records={records}
                timeline={timeline}
                x={x}
                starts={starts}
                log={y === "log" || (y === "auto" && !!g.log)}
                smoothing={smoothing}
                compare={compare}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
