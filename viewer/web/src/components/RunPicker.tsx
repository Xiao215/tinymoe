import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { api } from "../api";
import { Chevron, Select, usePopover } from "./Select";
import { STAGE_ORDER, stageColor } from "../theme";
import type { ControlInfo, RunSummary } from "../types";

interface Props {
  runs: RunSummary[];
  run: string | null;
  compare: string[];
  onOpen: (id: string) => void;
  onToggleCompare: (id: string) => void;
  control: ControlInfo | null;
}

const runStatus = (r: RunSummary): string => {
  if (r.stages.some((s) => s.status === "failed")) return "failed";
  if (r.stages.some((s) => s.status === "running")) return "running";
  if (r.stages.some((s) => s.status === "paused")) return "paused";
  const done = new Set(r.stages.filter((s) => s.status === "done").map((s) => s.name));
  if (r.planned?.some((s) => !done.has(s))) return "stopped";
  return r.stages.length ? "done" : "pending";
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Start a pipeline as a new run; opens it once its manifest exists. */
function NewRun({ control, onOpen }: { control: ControlInfo; onOpen: (id: string) => void }) {
  const [file, setFile] = useState(control.pipelines[0]?.file ?? "");
  const [until, setUntil] = useState("");
  const [state, setState] = useState<string | null>(null);
  const p = control.pipelines.find((x) => x.file === file);
  if (!p) return null;
  const start = async () => {
    setState("starting…");
    try {
      const { run_id } = await api.start(file, until || null);
      for (let i = 0; i < 120; i++) {
        if ((await api.runs()).some((r) => r.id === run_id)) {
          setState(null);
          onOpen(run_id);
          return;
        }
        await sleep(1000);
      }
      setState(`${run_id} started but has not written a manifest yet`);
    } catch (e) {
      setState(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <div className="rp-new" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
      <span className="label">new run</span>
      <Select
        value={file}
        options={control.pipelines.map((x) => ({ value: x.file, label: x.name, hint: `${x.stages.length} stages` }))}
        onChange={(v) => {
          setFile(v);
          setUntil("");
        }}
      />
      <Select
        label="until"
        value={until}
        options={[{ value: "", label: "the end" }, ...p.stages.slice(0, -1).map((s) => ({ value: s, label: `after ${s}` }))]}
        onChange={setUntil}
      />
      <button className="btn primary" disabled={!!control.busy || state === "starting…"} onClick={start} title={control.busy ? `${control.busy} is running` : `tinymoe pipeline configs/pipeline/${file}`}>
        Start
      </button>
      {control.busy && <span className="muted">{control.busy} is running; pause it first</span>}
      {state && <span className={state === "starting…" ? "muted" : "ctl-err"}>{state}</span>}
    </div>
  );
}

const when = (iso: string | null): string => {
  const t = iso ? Date.parse(iso) : NaN;
  if (!Number.isFinite(t)) return "";
  return new Date(t).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false });
};

/** Six slots in lifecycle order: filled = done, striped = running, hollow = not run. */
function StageStrip({ r }: { r: RunSummary }) {
  return (
    <span className="rp-strip" aria-hidden>
      {STAGE_ORDER.map((name) => {
        const s = r.stages.find((x) => x.name === name);
        return <i key={name} className={s ? s.status : "none"} style={{ ["--c" as string]: stageColor(name) }} title={`${name}: ${s?.status ?? "not run"}`} />;
      })}
    </span>
  );
}

/** Run switcher: a popover listing runs with their lifecycle at a glance, plus an
 * overlay toggle per run (other runs draw dashed on the metric charts). */
export function RunPicker({ runs, run, compare, onOpen, onToggleCompare, control }: Props) {
  const root = useRef<HTMLDivElement>(null);
  const [open, setOpen] = usePopover(root);
  const [active, setActive] = useState(0);
  const current = runs.find((r) => r.id === run);

  useEffect(() => {
    if (open) setActive(Math.max(0, runs.findIndex((r) => r.id === run)));
  }, [open, runs, run]);

  const onKey = (e: KeyboardEvent) => {
    if (!open) {
      if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }
    if (e.key === "Escape") setOpen(false);
    else if (e.key === "ArrowDown") setActive((i) => Math.min(runs.length - 1, i + 1));
    else if (e.key === "ArrowUp") setActive((i) => Math.max(0, i - 1));
    else if (e.key === "Enter") {
      const r = runs[active];
      if (r) {
        onOpen(r.id);
        setOpen(false);
      }
    } else return;
    e.preventDefault();
  };

  return (
    <div className="rp" ref={root} onKeyDown={onKey}>
      <button className={`rp-trigger${open ? " open" : ""}`} onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open}>
        <span className="label">run</span>
        <span className="rp-current">
          {current && <span className={`status ${runStatus(current)}`}><i /></span>}
          <span className="mono">{run ?? "–"}</span>
          {compare.length > 0 && <span className="rp-count">+{compare.length}</span>}
          <Chevron />
        </span>
      </button>
      {open && (
        <div className="pop rp-pop" role="listbox" aria-label="runs">
          <div className="rp-head">
            <span className="label">runs</span>
            <span className="label">overlay</span>
          </div>
          {runs.map((r, i) => {
            const st = runStatus(r);
            const isCur = r.id === run;
            const on = compare.includes(r.id);
            return (
              <div
                key={r.id}
                role="option"
                aria-selected={isCur}
                className={`rp-row${isCur ? " current" : ""}${i === active ? " active" : ""}`}
                onPointerEnter={() => setActive(i)}
                onClick={() => {
                  onOpen(r.id);
                  setOpen(false);
                }}
              >
                <span className={`status ${st}`} title={st}>
                  <i />
                </span>
                <span className="rp-main">
                  <span className="mono rp-id">{r.id}</span>
                  <span className="rp-sub">
                    <StageStrip r={r} />
                    <span>
                      {st === "running"
                        ? `running ${r.stages.find((s) => s.status === "running")?.name}`
                        : `${r.stages.filter((s) => s.status === "done").length}/${r.planned?.length ?? STAGE_ORDER.length} stages${st === "paused" || st === "stopped" ? ` · ${st}` : ""}`}
                    </span>
                    <span className="rp-dot">·</span>
                    <span>{when(r.created)}</span>
                  </span>
                </span>
                {isCur ? (
                  <span className="rp-tag">viewing</span>
                ) : (
                  <button
                    className={`rp-overlay${on ? " on" : ""}`}
                    title={on ? "remove dashed overlay" : "overlay on metric charts (dashed)"}
                    aria-pressed={on}
                    onClick={(e) => {
                      e.stopPropagation();
                      onToggleCompare(r.id);
                    }}
                  >
                    {on ? "✓" : "+"}
                  </button>
                )}
              </div>
            );
          })}
          {!runs.length && <div className="rp-empty muted">no runs yet</div>}
          {control?.enabled && (
            <NewRun
              control={control}
              onOpen={(id) => {
                onOpen(id);
                setOpen(false);
              }}
            />
          )}
        </div>
      )}
    </div>
  );
}
