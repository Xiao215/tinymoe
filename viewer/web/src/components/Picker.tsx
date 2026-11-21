import type { ReactNode } from "react";
import type { SnapshotEntry } from "../types";

interface PickerProps<T extends string | number> {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}

/** Labelled select whose value type round-trips (numbers stay numbers). */
export function Picker<T extends string | number>({ label, value, options, onChange }: PickerProps<T>) {
  return (
    <label className="picker">
      {label}{" "}
      <select
        value={String(value)}
        onChange={(e) => {
          const hit = options.find((o) => String(o.value) === e.target.value);
          if (hit) onChange(hit.value);
        }}
      >
        {options.map((o) => (
          <option key={String(o.value)} value={String(o.value)}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Placeholder for panels whose data has not been produced (yet). */
export function Empty({ children }: { children: ReactNode }) {
  return <p className="muted empty">{children}</p>;
}

export const snapLabel = (s: SnapshotEntry | null) => (s ? `${s.stage} step ${s.step}` : "no snapshot");
