import type { ReactNode } from "react";
import type { SnapshotEntry } from "../types";
import { Select } from "./Select";

interface PickerProps<T extends string | number> {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}

/** Labelled dropdown used by the panels (the shared Select). */
export function Picker<T extends string | number>(props: PickerProps<T>) {
  return <Select {...props} />;
}

/** Placeholder for panels whose data has not been produced (yet). */
export function Empty({ children }: { children: ReactNode }) {
  return <p className="muted empty">{children}</p>;
}

export const snapLabel = (s: SnapshotEntry | null) => (s ? `${s.stage} step ${s.step}` : "no snapshot");
