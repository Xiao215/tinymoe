import { useState } from "react";
import { api } from "../api";
import { stageColor } from "../theme";
import type { ControlState, Manifest, PauseWhen } from "../types";
import { Select } from "./Select";

/** Where a run stands with respect to its pipeline, from the manifest plus whether a
 * pipeline process is alive. */
export interface RunPosition {
  status: "running" | "starting" | "paused" | "stopped" | "interrupted" | "done" | "failed";
  label: string;
  /** the first planned stage that has not finished */
  next: string | null;
  planned: string[];
}

export function runPosition(manifest: Manifest, ctl: ControlState | null): RunPosition {
  const stages = manifest.stages;
  const planned = ctl?.pipeline_stages.length ? ctl.pipeline_stages : (manifest.pipeline_stages ?? stages.map((s) => s.name));
  const done = new Set(stages.filter((s) => s.status === "done").map((s) => s.name));
  const next = planned.find((s) => !done.has(s)) ?? null;
  const cur = stages.find((s) => s.status === "running") ?? stages[stages.length - 1];
  const alive = ctl?.running ?? null;
  if (ctl?.starting) return { status: "starting", label: "starting pipeline", next, planned };
  if (cur?.status === "failed") return { status: "failed", label: `${cur.name} failed`, next, planned };
  if (cur?.status === "running" && alive !== false) return { status: "running", label: `training ${cur.name}`, next, planned };
  if (alive) return { status: "running", label: next ? `starting ${next}` : "finishing", next, planned };
  if (cur?.status === "running") return { status: "interrupted", label: `${cur.name} interrupted`, next, planned };
  if (cur?.status === "paused") return { status: "paused", label: `paused in ${cur.name}`, next, planned };
  if (next) return { status: "stopped", label: `stopped before ${next}`, next, planned };
  return { status: "done", label: "finished", next, planned };
}

const PauseIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
    <path d="M3 2v6M7 2v6" stroke="currentColor" strokeWidth="1.8" />
  </svg>
);
const PlayIcon = () => (
  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
    <path d="M3 1.8v6.4L8.2 5z" fill="currentColor" />
  </svg>
);

interface Props {
  run: string;
  manifest: Manifest;
  ctl: ControlState | null;
  refresh: () => void;
}

/** Start / pause / resume the run's pipeline from the page. Pausing is graceful: the
 * trainer saves a checkpoint after the current step (or the pipeline stops before its
 * next stage), and resuming re-runs the pipeline with the same run id. */
export function RunControl({ run, manifest, ctl, refresh }: Props) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [until, setUntil] = useState<string>("");
  if (!ctl) return null;
  const pos = runPosition(manifest, ctl);
  const cur = manifest.stages.find((s) => s.status === "running") ?? manifest.stages[manifest.stages.length - 1];

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setErr(null);
    try {
      await fn();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      refresh();
    }
  };
  const pause = (when: PauseWhen) => act(() => api.pause(run, when));

  if (!ctl.enabled) {
    return (
      <div className="ctl-bar">
        <span className="ctl-msg muted">Run control is off: start the viewer on localhost (or with --control) to pause and resume from here.</span>
      </div>
    );
  }

  const remaining = pos.next ? pos.planned.slice(pos.planned.indexOf(pos.next)) : [];
  const untilOpts = [
    { value: "", label: "the end", hint: `${remaining.length} stage${remaining.length === 1 ? "" : "s"}` },
    ...remaining.slice(0, -1).map((s) => ({ value: s, label: `after ${s}` })),
  ];
  const other = ctl.busy && ctl.busy !== run ? ctl.busy : null;
  let body;
  if (pos.status === "starting") {
    body = <span className="ctl-msg">Starting the pipeline (loading data and the checkpoint)…</span>;
  } else if (ctl.running) {
    if (ctl.pending === "now") {
      body = (
        <>
          <span className="ctl-msg">
            Pausing: <b>{cur?.name}</b> saves a checkpoint after its current step, then stops.
          </span>
          <span className="grow" />
          <button className="btn" disabled={busy} onClick={() => act(() => api.cancelPause(run))}>
            Keep training
          </button>
        </>
      );
    } else if (ctl.pending === "after_stage") {
      body = (
        <>
          <span className="ctl-msg">
            Will stop when <b>{cur?.name}</b> finishes, before the next stage starts.
          </span>
          <span className="grow" />
          <button className="btn" disabled={busy} onClick={() => act(() => api.cancelPause(run))}>
            Cancel
          </button>
          <button className="btn" disabled={busy} onClick={() => pause("now")}>
            <PauseIcon /> Pause now instead
          </button>
        </>
      );
    } else {
      body = (
        <>
          <span className="ctl-msg">
            Pipeline running{ctl.pid ? <span className="muted"> · pid {ctl.pid}</span> : null}
            {ctl.started ? <span className="muted"> · since {new Date(ctl.started).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false })}</span> : null}
          </span>
          <span className="grow" />
          <button className="btn" disabled={busy} onClick={() => pause("after_stage")} title={`let ${cur?.name ?? "this stage"} finish, then stop`}>
            Stop after {cur?.name ?? "this stage"}
          </button>
          <button className="btn primary" disabled={busy} onClick={() => pause("now")} title="checkpoint after the current step, then stop; resume continues from it">
            <PauseIcon /> Pause now
          </button>
        </>
      );
    }
  } else if (!pos.next) {
    body = <span className="ctl-msg muted">All {pos.planned.length} stages of this pipeline are finished.</span>;
  } else if (!ctl.pipeline) {
    body = <span className="ctl-msg muted">Can't tell which pipeline made this run, so it can't be resumed from here.</span>;
  } else {
    const resumeStage = cur && cur.name === pos.next && (cur.status === "paused" || cur.status === "running");
    body = (
      <>
        <span className="ctl-msg">
          {resumeStage ? (
            <>
              <b>{pos.next}</b> continues from its last checkpoint.
            </>
          ) : (
            <>
              Next: <span className="dot" style={{ background: stageColor(pos.next) }} /> <b>{pos.next}</b>
              <span className="muted"> · starts from {cur ? `${cur.name}'s final checkpoint` : "scratch"}</span>
            </>
          )}
          {other && <span className="warn"> · {other} is running; pause it first</span>}
        </span>
        <span className="grow" />
        <Select label="run until" value={until} options={untilOpts} onChange={setUntil} />
        <button className="btn primary" disabled={busy || !!other} onClick={() => act(() => api.resume(run, until || null))}>
          <PlayIcon /> {resumeStage ? `Resume ${pos.next}` : `Start ${pos.next}`}
        </button>
      </>
    );
  }

  return (
    <div className="ctl-bar">
      {body}
      {err && <span className="ctl-err">{err}</span>}
      {ctl.log.length > 0 && (
        <details className="ctl-log" style={{ width: "100%" }}>
          <summary>pipeline log ▸ last {ctl.log.length} lines</summary>
          <pre>{ctl.log.join("\n")}</pre>
        </details>
      )}
    </div>
  );
}
