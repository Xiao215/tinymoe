import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { splitThink, wordDiff } from "../lib";
import { stageColor } from "../theme";
import type { GenerationRecord, GenerationsSnapshot, SnapshotEntry } from "../types";

interface Props {
  run: string;
  snapshots: SnapshotEntry[];
  version: number;
}

type Loaded = { entry: SnapshotEntry; snap: GenerationsSnapshot };

export function Completion({ text }: { text: string }) {
  const parts = splitThink(text);
  return (
    <span className="completion">
      {parts.before}
      {parts.think !== null && (
        <details className="think">
          <summary>&lt;think&gt; ({parts.think.split(/\s+/).filter(Boolean).length} words)</summary>
          <span>{parts.think}</span>
        </details>
      )}
      {parts.after}
    </span>
  );
}

function Verdict({ g }: { g: GenerationRecord }) {
  if (g.answer === undefined) return null;
  return (
    <span className={`badge ${g.correct ? "ok" : "bad"}`} title={`expected ${g.answer}`}>
      {g.correct ? "correct" : g.format_ok ? "wrong" : "bad format"}
    </span>
  );
}

/** The same probe prompts answered at every eval of every stage, plus a diff view. */
export function Generations({ run, snapshots, version }: Props) {
  const entries = useMemo(() => snapshots.filter((s) => s.files.includes("generations")), [snapshots]);
  const [loaded, setLoaded] = useState<Loaded[]>([]);
  const [probe, setProbe] = useState<string | null>(null);
  const [picked, setPicked] = useState<number[]>([]);

  useEffect(() => {
    let live = true;
    Promise.all(
      entries.map((e) =>
        api.snapshot<GenerationsSnapshot>(run, e.stage, e.step, "generations").then((snap) => ({ entry: e, snap })),
      ),
    )
      .then((all) => live && setLoaded(all))
      .catch(() => live && setLoaded([]));
    return () => {
      live = false;
    };
  }, [run, entries, version]);

  const probes = useMemo(() => {
    const seen = new Map<string, GenerationRecord>();
    for (const l of loaded) for (const g of l.snap.generations) if (!seen.has(g.id)) seen.set(g.id, g);
    return [...seen.values()];
  }, [loaded]);
  const current = probe ?? probes[0]?.id ?? null;
  const rows = loaded
    .map((l) => ({ ...l, gen: l.snap.generations.find((g) => g.id === current) }))
    .filter((r): r is Loaded & { gen: GenerationRecord } => r.gen !== undefined);

  if (!entries.length) return <p className="muted">No evaluation generations yet.</p>;
  const prompt = probes.find((p) => p.id === current);
  const [first, second] = [...picked].sort((x, y) => x - y);
  const a = first !== undefined ? rows[first] : undefined;
  const b = second !== undefined ? rows[second] : undefined;
  return (
    <div>
      <div className="toolbar">
        {probes.map((p) => (
          <button key={p.id} className={p.id === current ? "active" : ""} onClick={() => { setProbe(p.id); setPicked([]); }}>
            {p.id}
          </button>
        ))}
      </div>
      {prompt && (
        <p className="prompt">
          <b>{prompt.kind}</b> prompt: <code>{prompt.prompt}</code>
          {prompt.answer !== undefined && <span className="muted"> (answer {prompt.answer})</span>}
        </p>
      )}
      <table className="gen-table">
        <thead>
          <tr>
            <th>stage</th>
            <th>step</th>
            <th>completion</th>
            <th>verifier</th>
            <th title="tick two rows">diff</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.entry.stage}-${r.entry.step}`}>
              <td>
                <span className="dot" style={{ background: stageColor(r.entry.stage) }} /> {r.entry.stage}
              </td>
              <td>{r.entry.step}</td>
              <td>
                <Completion text={r.gen.completion} />
              </td>
              <td>
                <Verdict g={r.gen} />
              </td>
              <td>
                <input
                  type="checkbox"
                  checked={picked.includes(i)}
                  onChange={() =>
                    setPicked((p) => (p.includes(i) ? p.filter((x) => x !== i) : [...p, i].slice(-2)))
                  }
                  aria-label="select for diff"
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {a && b && (
        <section className="card">
          <h3>
            Diff: {a.entry.stage}@{a.entry.step} → {b.entry.stage}@{b.entry.step}
          </h3>
          <p className="diff">
            {wordDiff(a.gen.completion, b.gen.completion).map((d, i) => (
              <span key={i} className={`diff-${d.op}`}>
                {d.text}
              </span>
            ))}
          </p>
        </section>
      )}
    </div>
  );
}
