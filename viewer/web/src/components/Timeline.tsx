import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import { fmtNum } from "../lib";
import { STAGE_ORDER, stageColor } from "../theme";
import type { SnapshotEntry, Span } from "../types";

interface Props {
  timeline: Span[];
  snapshots: SnapshotEntry[];
  selected: number | null; // index into the snapshot list; null = follow the latest
  onSelect: (index: number | null) => void;
}

const PLAY_MS = 900;

/** Snapshot scrubber over the whole lifecycle. Stages are thin coloured segments on one
 * global-step axis (stages not run yet are listed in the legend); every snapshot is a
 * stop; drag the handle, click a stop, use the arrow keys, or press play. */
export function Timeline({ timeline, snapshots, selected, onSelect }: Props) {
  const track = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [dragging, setDragging] = useState(false);
  const drag = useRef(false); // read synchronously by pointer handlers
  const n = snapshots.length;
  const idx = selected ?? n - 1;
  const cur = snapshots[idx];
  const total = Math.max(1, ...timeline.map((s) => s.end), ...snapshots.map((s) => s.gstep));
  const pct = (g: number) => (100 * g) / total;
  const ran = new Set(timeline.map((s) => s.name));

  useEffect(() => {
    if (!playing) return;
    if (idx >= n - 1) {
      setPlaying(false);
      return;
    }
    const t = setTimeout(() => onSelect(idx + 1), PLAY_MS);
    return () => clearTimeout(t);
  }, [playing, idx, n, onSelect]);

  if (!n && !timeline.length) return null;

  const nearest = (clientX: number): number => {
    const r = track.current!.getBoundingClientRect();
    const g = ((clientX - r.left) / r.width) * total;
    let best = 0;
    snapshots.forEach((s, i) => {
      if (Math.abs(s.gstep - g) < Math.abs(snapshots[best]!.gstep - g)) best = i;
    });
    return best;
  };
  const go = (i: number) => onSelect(i >= n - 1 ? null : Math.max(0, i));
  const onDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!n) return;
    setPlaying(false);
    drag.current = true;
    setDragging(true);
    e.currentTarget.setPointerCapture?.(e.pointerId);
    go(nearest(e.clientX));
  };
  const onMove = (e: PointerEvent<HTMLDivElement>) => {
    if (drag.current) {
      const i = nearest(e.clientX);
      if (i !== idx) go(i);
    }
  };
  const onKey = (e: KeyboardEvent) => {
    const map: Record<string, number> = { ArrowLeft: idx - 1, ArrowRight: idx + 1, Home: 0, End: n - 1, PageDown: idx - 5, PageUp: idx + 5 };
    if (e.key in map) {
      e.preventDefault();
      setPlaying(false);
      go(Math.min(n - 1, map[e.key]!));
    }
  };

  return (
    <div className="scrub">
      <div className="scrub-head">
        <span className="label">snapshot</span>
        {cur ? (
          <span className="scrub-cur">
            <span className="dot" style={{ background: stageColor(cur.stage) }} />
            <b>{cur.stage}</b> step <span className="num">{fmtNum(cur.step)}</span>
            <span className="muted">
              {" "}
              · global {fmtNum(cur.gstep)} · {idx + 1}/{n}
            </span>
            {selected === null && <span className="scrub-follow">latest</span>}
          </span>
        ) : (
          <span className="muted">no snapshots yet</span>
        )}
        <span className="grow" />
        <span className="scrub-legend" aria-label="stages">
          {STAGE_ORDER.map((s) => (
            <span key={s} className={ran.has(s) ? "" : "later"} title={ran.has(s) ? s : `${s}: not started`}>
              <i style={{ background: ran.has(s) ? stageColor(s) : "transparent", borderColor: stageColor(s) }} />
              {s}
            </span>
          ))}
        </span>
        <span className="scrub-btns">
          <button className="icon-btn" onClick={() => go(idx - 1)} disabled={idx <= 0} aria-label="previous snapshot" title="previous (←)">
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><path d="M6.5 2 3.5 5l3 3" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
          </button>
          <button
            className="icon-btn"
            onClick={() => {
              if (!playing && idx >= n - 1) onSelect(0);
              setPlaying((p) => !p);
            }}
            disabled={n < 2}
            aria-label={playing ? "pause" : "play through snapshots"}
            title={playing ? "pause" : "play"}
          >
            {playing ? (
              <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><path d="M3 2v6M7 2v6" stroke="currentColor" strokeWidth="1.6" /></svg>
            ) : (
              <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><path d="M3 1.8v6.4L8.2 5z" fill="currentColor" /></svg>
            )}
          </button>
          <button className="icon-btn" onClick={() => go(idx + 1)} disabled={idx >= n - 1} aria-label="next snapshot" title="next (→)">
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden><path d="M3.5 2 6.5 5l-3 3" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>
          </button>
        </span>
      </div>
      <div
        className={`scrub-track${dragging ? " dragging" : ""}`}
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="snapshot"
        aria-valuemin={1}
        aria-valuemax={Math.max(1, n)}
        aria-valuenow={idx + 1}
        aria-valuetext={cur ? `${cur.stage} step ${cur.step}` : "none"}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={(e) => {
          if (drag.current) go(nearest(e.clientX));
          drag.current = false;
          setDragging(false);
        }}
        onPointerCancel={() => {
          drag.current = false;
          setDragging(false);
        }}
        onKeyDown={onKey}
      >
        <div className="scrub-rail" />
        {timeline.map((s) => {
          const w = pct(s.end - s.start);
          return (
            <div
              key={s.name}
              className={`scrub-seg status-${s.status}`}
              style={{ left: `${pct(s.start)}%`, width: `${Math.max(w, 0.4)}%`, ["--c" as string]: stageColor(s.name) }}
              title={`${s.name}: global steps ${s.start}–${s.end} (${s.status})`}
            >
              {w > 6 && <span>{s.name}</span>}
            </div>
          );
        })}
        {cur && <div className="scrub-mask" style={{ left: `${pct(cur.gstep)}%` }} />}
        {snapshots.map((s, i) => (
          <i key={`${s.stage}-${s.step}`} className={`scrub-stop${i <= idx ? " past" : ""}`} style={{ left: `${pct(s.gstep)}%` }} />
        ))}
        {cur && (
          <div className="scrub-handle" style={{ left: `${pct(cur.gstep)}%`, ["--c" as string]: stageColor(cur.stage) }}>
            <span className="scrub-bubble">{fmtNum(cur.gstep)}</span>
          </div>
        )}
      </div>
    </div>
  );
}
