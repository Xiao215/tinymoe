import { useCallback, useEffect, useRef, useState } from "react";
import type {
  Manifest,
  MetricRecord,
  MetricsResponse,
  RunSummary,
  SnapshotEntry,
  Span,
} from "./types";

export async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${path}`);
  return (await res.json()) as T;
}

const enc = encodeURIComponent;

export const api = {
  runs: () => getJSON<RunSummary[]>("/api/runs"),
  manifest: (run: string) => getJSON<Manifest>(`/api/runs/${enc(run)}/manifest`),
  metrics: (run: string, offset = 0) =>
    getJSON<MetricsResponse>(`/api/runs/${enc(run)}/metrics?offset=${offset}`),
  snapshots: (run: string) => getJSON<SnapshotEntry[]>(`/api/runs/${enc(run)}/snapshots`),
  snapshot: <T>(run: string, stage: string, step: number, name: string) =>
    getJSON<T>(`/api/runs/${enc(run)}/snapshots/${enc(stage)}/${step}/${enc(name)}`),
  array: <T>(run: string, stage: string, step: number, name: string, key: string) =>
    getJSON<T>(`/api/runs/${enc(run)}/snapshots/${enc(stage)}/${step}/${enc(name)}/${enc(key)}`),
  artifacts: (run: string) => getJSON<string[]>(`/api/runs/${enc(run)}/artifacts`),
  artifact: <T>(run: string, name: string) =>
    getJSON<T>(`/api/runs/${enc(run)}/artifacts/${enc(name)}`),
};

/** Fetch once per key; `null` while loading, error message on failure. */
export function useFetch<T>(
  key: string | null,
  load: () => Promise<T>,
): { data: T | null; error: string | null } {
  const [state, setState] = useState<{ key: string | null; data: T | null; error: string | null }>(
    { key: null, data: null, error: null },
  );
  const loader = useRef(load);
  loader.current = load;
  useEffect(() => {
    if (key === null) return;
    let live = true;
    loader
      .current()
      .then((data) => live && setState({ key, data, error: null }))
      .catch((e: unknown) => live && setState({ key, data: null, error: String(e) }));
    return () => {
      live = false;
    };
  }, [key]);
  return state.key === key ? state : { data: null, error: null };
}

export interface RunData {
  manifest: Manifest | null;
  records: MetricRecord[];
  timeline: Span[];
  snapshots: SnapshotEntry[];
  artifacts: string[];
  live: boolean;
  error: string | null;
  /** bumps whenever snapshot/artifact files change, so panels can refetch */
  version: number;
}

const EMPTY: RunData = {
  manifest: null,
  records: [],
  timeline: [],
  snapshots: [],
  artifacts: [],
  live: false,
  error: null,
  version: 0,
};

/** Everything the panels need for one run, kept current through the SSE stream. */
export function useRun(run: string | null): RunData {
  const [data, setData] = useState<RunData>(EMPTY);

  const refreshIndex = useCallback(async (id: string) => {
    const [manifest, snapshots, artifacts] = await Promise.all([
      api.manifest(id),
      api.snapshots(id),
      api.artifacts(id),
    ]);
    setData((d) => ({ ...d, manifest, snapshots, artifacts, version: d.version + 1 }));
  }, []);

  useEffect(() => {
    if (!run) {
      setData(EMPTY);
      return;
    }
    let closed = false;
    let source: EventSource | null = null;
    setData(EMPTY);
    (async () => {
      try {
        const m = await api.metrics(run);
        if (closed) return;
        setData((d) => ({ ...d, records: m.records, timeline: m.timeline }));
        await refreshIndex(run);
        if (closed) return;
        source = new EventSource(`/api/runs/${enc(run)}/events?offset=${m.offset}`);
        source.addEventListener("hello", () => setData((d) => ({ ...d, live: true })));
        source.addEventListener("metrics", (ev) => {
          const payload = JSON.parse((ev as MessageEvent<string>).data) as MetricsResponse;
          setData((d) => ({
            ...d,
            records: d.records.concat(payload.records),
            timeline: payload.timeline,
          }));
        });
        for (const kind of ["snapshots", "manifest", "artifacts"]) {
          source.addEventListener(kind, () => void refreshIndex(run));
        }
        source.onerror = () => setData((d) => ({ ...d, live: false }));
      } catch (e) {
        if (!closed) setData((d) => ({ ...d, error: String(e) }));
      }
    })();
    return () => {
      closed = true;
      source?.close();
    };
  }, [run, refreshIndex]);

  return data;
}

/** Metrics of other runs, for overlay comparison (fetched once, not live). */
export function useCompare(runs: string[]): Record<string, MetricRecord[]> {
  const [out, setOut] = useState<Record<string, MetricRecord[]>>({});
  const key = runs.join("|");
  useEffect(() => {
    let live = true;
    Promise.all(runs.map((r) => api.metrics(r).then((m) => [r, m.records] as const)))
      .then((pairs) => live && setOut(Object.fromEntries(pairs)))
      .catch(() => live && setOut({}));
    return () => {
      live = false;
    };
  }, [key]);
  return out;
}

/** One snapshot file (JSON), refetched when the run's files change. */
export function useSnapshot<T>(
  run: string,
  entry: SnapshotEntry | null,
  name: string,
  version: number,
): { data: T | null; error: string | null } {
  const key = entry && entry.files.includes(name) ? `${run}/${entry.stage}/${entry.step}/${name}/${version}` : null;
  return useFetch<T>(key, () => api.snapshot<T>(run, entry!.stage, entry!.step, name));
}

export interface Loaded<T> {
  entry: SnapshotEntry;
  data: T;
}

/** Every snapshot named `name`, in timeline order (for over-time views). Files that fail
 * to load are skipped, so a half-written snapshot never breaks a panel. */
export function useSnapshotSeries<T>(run: string, entries: SnapshotEntry[], name: string, version: number): Loaded<T>[] {
  const [out, setOut] = useState<Loaded<T>[]>([]);
  const chosen = entries.filter((e) => e.files.includes(name));
  const key = `${run}|${name}|${version}|${chosen.map((e) => `${e.stage}:${e.step}`).join(",")}`;
  useEffect(() => {
    let live = true;
    Promise.all(
      chosen.map((entry) =>
        api
          .snapshot<T>(run, entry.stage, entry.step, name)
          .then((data) => ({ entry, data }))
          .catch(() => null),
      ),
    ).then((rows) => live && setOut(rows.filter((r): r is Loaded<T> => r !== null)));
    return () => {
      live = false;
    };
  }, [key]);
  return out;
}

/** An artifact JSON, or null while absent (it may appear later in a live run). */
export function useArtifact<T>(run: string, name: string, data: RunData): { data: T | null; error: string | null } {
  const key = data.artifacts.includes(name) ? `${run}/${name}/${data.version}` : null;
  return useFetch<T>(key, () => api.artifact<T>(run, name));
}
