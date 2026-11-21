import type { SnapshotEntry, Span } from "../types";
import { stageColor } from "../theme";

interface Props {
  timeline: Span[];
  snapshots: SnapshotEntry[];
  selected: number | null; // index into the de-duplicated snapshot list
  onSelect: (index: number) => void;
}

/** One horizontal bar across the whole lifecycle, with snapshot ticks and a scrubber. */
export function Timeline({ timeline, snapshots, selected, onSelect }: Props) {
  const total = Math.max(1, ...timeline.map((s) => s.end), ...snapshots.map((s) => s.gstep));
  const current = selected !== null ? snapshots[selected] : undefined;
  return (
    <div className="timeline">
      <div className="timeline-bar">
        {timeline.map((s) => {
          const width = ((s.end - s.start) / total) * 100;
          return (
            <div
              key={s.name}
              className={`timeline-stage status-${s.status}`}
              style={{ left: `${(s.start / total) * 100}%`, width: `${Math.max(width, 0.5)}%`, background: stageColor(s.name) }}
              title={`${s.name}: global steps ${s.start}–${s.end} (${s.status})`}
            >
              <span>{s.name}</span>
            </div>
          );
        })}
        {snapshots.map((s, i) => (
          <button
            key={`${s.stage}-${s.step}`}
            className={`timeline-tick${i === selected ? " active" : ""}`}
            style={{ left: `${(s.gstep / total) * 100}%` }}
            title={`${s.stage} step ${s.step}`}
            onClick={() => onSelect(i)}
          />
        ))}
      </div>
      {snapshots.length > 0 && (
        <div className="scrubber">
          <input
            type="range"
            min={0}
            max={snapshots.length - 1}
            value={selected ?? snapshots.length - 1}
            onChange={(e) => onSelect(Number(e.target.value))}
            aria-label="snapshot scrubber"
          />
          <span className="muted">
            {current ? `${current.stage} · step ${current.step} · global ${current.gstep}` : "latest"}
            {` (${(selected ?? snapshots.length - 1) + 1}/${snapshots.length})`}
          </span>
        </div>
      )}
    </div>
  );
}
