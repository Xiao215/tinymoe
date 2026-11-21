import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { STAGE_ORDER, stageColor } from "../theme";
import type { RunSummary } from "../types";

interface Props {
  runs: RunSummary[];
  run: string | null;
  compare: string[];
  onOpen: (id: string) => void;
  onToggleCompare: (id: string) => void;
}

const runStatus = (r: RunSummary): string => {
  if (r.stages.some((s) => s.status === "failed")) return "failed";
  if (r.stages.some((s) => s.status === "running")) return "running";
  return r.stages.length ? "done" : "pending";
};

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
export function RunPicker({ runs, run, compare, onOpen, onToggleCompare }: Props) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const current = runs.find((r) => r.id === run);

  useEffect(() => {
    if (!open) return;
    setActive(Math.max(0, runs.findIndex((r) => r.id === run)));
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
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
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
            <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth="1.3" />
          </svg>
        </span>
      </button>
      {open && (
        <div className="rp-pop" role="listbox" aria-label="runs">
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
                onMouseEnter={() => setActive(i)}
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
                      {st === "running" ? `running ${r.stages.find((s) => s.status === "running")?.name}` : `${r.stages.length}/${STAGE_ORDER.length} stages`}
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
        </div>
      )}
    </div>
  );
}
