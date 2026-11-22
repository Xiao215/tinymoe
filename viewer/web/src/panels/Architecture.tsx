import { useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { api, useFetch, useSnapshot, type RunData } from "../api";
import { Empty } from "../components/Picker";
import { Select } from "../components/Select";
import { fmtNum } from "../lib";
import { categorical, sequential, sequentialInk } from "../theme";
import type { Architecture as Arch, InternalsSnapshot, ModelConfigJson, ParamInfo } from "../types";

interface Props {
  run: string;
  data: RunData;
}

// ------------------------------------------------------------ parameters

/** Parameter groups in a fixed order (categorical slots, never re-ordered by size). */
const GROUPS = [
  { id: "embedding", label: "embedding" },
  { id: "attention", label: "attention" },
  { id: "dense", label: "dense FFN" },
  { id: "routed", label: "routed experts" },
  { id: "shared", label: "shared expert" },
  { id: "router", label: "router" },
  { id: "mtp", label: "MTP head" },
  { id: "norms", label: "norms" },
] as const;
type Group = (typeof GROUPS)[number]["id"];
const groupColor = (g: Group) => categorical(GROUPS.findIndex((x) => x.id === g));

function groupOf(name: string): Group {
  if (name.startsWith("mtp.")) return "mtp";
  if (name.includes("norm")) return "norms";
  if (name.startsWith("embed.") || name.startsWith("lm_head.")) return "embedding";
  if (name.includes(".attn.")) return "attention";
  if (name.includes(".ffn.router.")) return "router";
  if (name.includes(".ffn.shared.")) return "shared";
  if (/\.ffn\.w_(gate|up|down)$/.test(name)) return "routed";
  return "dense";
}

const layerOf = (name: string): number | null => {
  const m = /^layers\.(\d+)\./.exec(name);
  return m ? Number(m[1]) : null;
};

const fmtP = (n: number): string =>
  n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 1 : 2)}M` : n >= 1e3 ? `${(n / 1e3).toFixed(1)}K` : String(n);
const pct = (a: number, b: number) => (b ? `${((100 * a) / b).toFixed(a / b < 0.01 ? 2 : 1)}%` : "–");
const dims = (shape: number[]) => shape.join(" × ");

interface LayerInfo {
  index: number;
  moe: boolean;
  attn: number;
  ffn: number;
  norms: number;
  active: number;
}

function layers(arch: Arch): LayerInfo[] {
  const c = arch.config;
  const out: LayerInfo[] = [];
  for (let i = 0; i < c.n_layers; i++) {
    const ps = arch.params.filter((p) => layerOf(p.name) === i);
    const sum = (g: Group[]) => ps.filter((p) => g.includes(groupOf(p.name))).reduce((a, p) => a + p.numel, 0);
    const moe = ps.some((p) => groupOf(p.name) === "routed");
    const routed = sum(["routed"]);
    const ffn = sum(["dense", "routed", "shared", "router"]);
    const active = sum(["attention", "norms", "dense", "shared", "router"]) + (moe ? (routed * c.moe.top_k) / c.moe.n_experts : 0);
    out.push({ index: i, moe, attn: sum(["attention"]), ffn, norms: sum(["norms"]), active });
  }
  return out;
}

// --------------------------------------------------------------- drawing

/** A weight matrix drawn to scale (rows = input dim, columns = output dim, as in x·W).
 * Dimensions beyond `cap` pixels are cut with a torn edge so the vocab fits. */
function Mat({
  rows,
  cols,
  s,
  name,
  color,
  strips,
  cap = 360,
  note,
}: {
  rows: number;
  cols: number;
  s: number;
  name: ReactNode;
  color: string;
  strips?: { n: number; color: (i: number) => string };
  cap?: number;
  note?: ReactNode;
}) {
  const w = cols * s;
  const h = rows * s;
  const cutW = w > cap;
  const cutH = h > cap;
  const style = { width: Math.max(10, Math.min(w, cap)), height: Math.max(10, Math.min(h, cap)), ["--c" as string]: color } as CSSProperties;
  return (
    <figure className="mat">
      <div className={`mat-box${cutW ? " cut-w" : ""}${cutH ? " cut-h" : ""}`} style={style}>
        {strips && (
          <div className="mat-strips">
            {Array.from({ length: strips.n }, (_, i) => (
              <i key={i} style={{ background: strips.color(i) }} />
            ))}
          </div>
        )}
      </div>
      <figcaption>
        <b>{name}</b>
        <span className="num">
          {fmtNum(rows)} × {fmtNum(cols)}
        </span>
        {note && <span className="muted">{note}</span>}
      </figcaption>
    </figure>
  );
}

const Op = ({ children }: { children: ReactNode }) => <span className="arch-op">{children}</span>;

function Shape({ children }: { children: ReactNode }) {
  return <span className="arch-shape num">{children}</span>;
}

// ----------------------------------------------------------------- block

function Attention({ c, s }: { c: ModelConfigJson; s: number }) {
  const hd = c.head_dim ?? c.d_model / c.n_heads;
  const per = c.n_heads / c.n_kv_heads;
  const kvColor = (g: number) => `color-mix(in srgb, ${categorical(g)} 34%, transparent)`;
  return (
    <div className="arch-sub">
      <div className="arch-sub-head">
        <h4>Attention</h4>
        <span className="muted">
          {c.n_kv_heads < c.n_heads ? `grouped-query: ${c.n_heads} query heads share ${c.n_kv_heads} key/value heads (${per} each)` : `${c.n_heads} heads`} · head dim {hd}
          {c.qk_norm ? " · RMSNorm on q and k" : ""} · RoPE base {fmtNum(c.rope_base)}
          {c.sliding_window ? ` · sliding window ${c.sliding_window}` : ""}
        </span>
      </div>
      <div className="mat-row">
        <Mat rows={c.d_model} cols={c.n_heads * hd} s={s} name="W_q" color={categorical(1)} strips={{ n: c.n_heads, color: (h) => kvColor(Math.floor(h / per)) }} note={`${c.n_heads} heads`} />
        <Mat rows={c.d_model} cols={c.n_kv_heads * hd} s={s} name="W_k" color={categorical(1)} strips={{ n: c.n_kv_heads, color: kvColor }} note={`${c.n_kv_heads} kv`} />
        <Mat rows={c.d_model} cols={c.n_kv_heads * hd} s={s} name="W_v" color={categorical(1)} strips={{ n: c.n_kv_heads, color: kvColor }} note={`${c.n_kv_heads} kv`} />
        <div className="arch-mid">
          <Op>softmax(q kᵀ / √{hd}) v</Op>
          <div className="causal" title="causal mask: each position attends to itself and earlier ones" />
          <span className="muted">T × T per head, causal</span>
        </div>
        <Mat rows={c.n_heads * hd} cols={c.d_model} s={s} name="W_o" color={categorical(1)} />
      </div>
      <div className="kv-legend">
        {Array.from({ length: c.n_kv_heads }, (_, g) => (
          <span key={g}>
            <i style={{ background: kvColor(g), borderColor: categorical(g) }} />
            q heads {g * per}–{g * per + per - 1} → kv head {g}
          </span>
        ))}
      </div>
    </div>
  );
}

function DenseFFN({ c, s, hidden, label }: { c: ModelConfigJson; s: number; hidden: number; label: string }) {
  return (
    <div className="mat-row">
      <Mat rows={c.d_model} cols={hidden} s={s} name="W_gate" color={categorical(2)} />
      <Mat rows={c.d_model} cols={hidden} s={s} name="W_up" color={categorical(2)} />
      <div className="arch-mid">
        <Op>silu(x W_gate) ⊙ x W_up</Op>
        <span className="muted">{label}</span>
      </div>
      <Mat rows={hidden} cols={c.d_model} s={s} name="W_down" color={categorical(2)} />
    </div>
  );
}

function MoEBlock({ c, s, load, loadAt }: { c: ModelConfigJson; s: number; load: number[] | null; loadAt: string | null }) {
  const m = c.moe;
  const ideal = 1 / m.n_experts;
  // shade relative to a fair share: white = starved, mid = fair, darkest = 2x fair or more
  const shade = (v: number) => 0.06 + 0.88 * Math.min(1, v / (2 * ideal));
  return (
    <div className="arch-sub">
      <div className="arch-sub-head">
        <h4>Mixture of experts</h4>
        <span className="muted">
          {m.n_experts} routed experts, each token picks top-{m.top_k} by {m.score_func} score
          {m.balance === "aux_free" ? " (+ a load-balancing bias)" : ""} · {m.n_shared_experts} shared expert{m.n_shared_experts === 1 ? "" : "s"} always on
        </span>
      </div>
      <div className="mat-row">
        <Mat rows={c.d_model} cols={m.n_experts} s={s} name="W_router" color={categorical(5)} note="scores" />
        <div className="arch-mid">
          <Op>top-{m.top_k}</Op>
          <span className="muted">
            {m.top_k}/{m.n_experts} experts run per token
          </span>
        </div>
        <div className="experts">
          <div className="experts-grid" style={{ gridTemplateColumns: `repeat(${Math.min(8, m.n_experts)}, 1fr)` }}>
            {Array.from({ length: m.n_experts }, (_, e) => {
              const v = load?.[e] ?? null;
              const t = v === null ? 0 : shade(v);
              return (
                <div
                  key={e}
                  className="expert"
                  style={v === null ? undefined : { background: sequential(t), color: sequentialInk(t) }}
                  title={v === null ? `expert ${e}` : `expert ${e}: ${(100 * v).toFixed(2)}% of routed slots (fair share ${(100 * ideal).toFixed(2)}%, ${(v / ideal).toFixed(2)}×)`}
                >
                  <span>{e}</span>
                </div>
              );
            })}
          </div>
          <figcaption>
            <b>routed experts</b>
            <span className="num">
              {m.n_experts} × SwiGLU {c.d_model} → {m.expert_hidden} → {c.d_model}
            </span>
            <span className="muted">{load ? `routed share at ${loadAt}: light = starved, mid = fair, dark = 2× fair` : "no routing snapshot yet"}</span>
          </figcaption>
        </div>
        <div className="experts shared">
          <div className="expert on">S</div>
          <figcaption>
            <b>shared</b>
            <span className="num">
              {c.d_model} → {m.expert_hidden * m.n_shared_experts} → {c.d_model}
            </span>
          </figcaption>
        </div>
      </div>
      <div className="arch-formula">
        y = shared(x) + Σ<sub>i ∈ top-{m.top_k}</sub> g<sub>i</sub>(x) · expert<sub>i</sub>(x)
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ panel

/** The model as built: every tensor's real shape (from the stage's resolved config,
 * built on the meta device server-side), drawn to scale, with the forward pass. */
export function Architecture({ run, data }: Props) {
  const stages = (data.manifest?.stages ?? []).filter((s) => s.config);
  const [stage, setStage] = useState<string | null>(null);
  const st = stage ?? stages[stages.length - 1]?.name ?? null;
  const [which, setWhich] = useState<"model" | "student">("model");
  const w = st === "distill" ? which : "model";
  const { data: arch, error } = useFetch(st ? `${run}/${st}/${w}` : null, () => api.architecture(run, st, w));
  const [sel, setSel] = useState(1);
  const [filter, setFilter] = useState("");

  const latest = useMemo(() => {
    const snaps = data.snapshots.filter((s) => s.files.includes("data"));
    return snaps[snaps.length - 1] ?? null;
  }, [data.snapshots]);
  const { data: snap } = useSnapshot<InternalsSnapshot>(run, latest, "data", data.version);

  const info = useMemo(() => (arch ? layers(arch) : []), [arch]);
  const groups = useMemo(() => {
    const tot: Record<string, number> = {};
    for (const p of arch?.params ?? []) tot[groupOf(p.name)] = (tot[groupOf(p.name)] ?? 0) + p.numel;
    return GROUPS.map((g) => ({ ...g, n: tot[g.id] ?? 0 })).filter((g) => g.n > 0);
  }, [arch]);

  if (!stages.length) return <Empty>No stage has recorded a config yet.</Empty>;
  if (error) return <p className="error">{error}</p>;
  if (!arch) return <Empty>building the model on the meta device…</Empty>;

  const c = arch.config;
  const total = arch.counts.total ?? 0;
  const active = arch.counts.active ?? 0;
  const T = arch.seq_len ?? c.max_seq_len;
  const hd = c.head_dim ?? c.d_model / c.n_heads;
  const layer = info[Math.min(sel, info.length - 1)] ?? info[0]!;
  const nMoE = info.filter((l) => l.moe).length;
  const s = 150 / c.d_model; // px per dimension: d_model draws 150px
  const maxLayer = Math.max(...info.map((l) => l.attn + l.ffn));
  const loadLi = snap?.moe?.layers.indexOf(layer.index) ?? -1;
  const load = loadLi >= 0 ? (snap?.moe?.load?.[loadLi] ?? null) : null;
  const rows = arch.params.filter((p) => !filter || p.name.includes(filter));
  const mtpParams = arch.params.filter((p) => p.name.startsWith("mtp.")).reduce((a, p) => a + p.numel, 0);

  return (
    <div className="arch">
      <div className="toolbar">
        <Select label="stage" value={st ?? ""} options={stages.map((x) => ({ value: x.name, label: x.name }))} onChange={(v) => setStage(v)} />
        {st === "distill" && (
          <div className="seg" role="group" aria-label="which model">
            {(["model", "student"] as const).map((k) => (
              <button key={k} className={w === k ? "on" : ""} onClick={() => setWhich(k)}>
                {k === "model" ? "teacher" : "student"}
              </button>
            ))}
          </div>
        )}
        <span className="muted" style={{ fontSize: 12 }}>
          real tensor shapes of this stage's model (built shape-only, no weights loaded) · matrices drawn to scale as in × out
        </span>
      </div>

      <p className="arch-summary">
        Decoder-only transformer · <b>{c.n_layers} blocks</b>
        {c.moe.enabled ? ` (${c.n_layers - nMoE} dense, ${nMoE} mixture-of-experts)` : " (dense)"} · width <b>{c.d_model}</b> · {c.n_heads} query / {c.n_kv_heads} kv heads of {hd} · vocab <b>{fmtNum(c.vocab_size)}</b> · trained at context <b>{fmtNum(T)}</b> (max {fmtNum(c.max_seq_len)}){c.tie_embeddings ? " · embedding tied to the output head" : ""}
        {c.mtp.depth ? ` · multi-token prediction depth ${c.mtp.depth}` : ""}.
      </p>

      <div className="kpis" style={{ marginTop: 10 }}>
        <div className="kpi">
          <span className="label">total parameters</span>
          <div className="value">{fmtP(total)}</div>
          <div className="delta">{fmtP(arch.counts.total_non_embedding ?? 0)} without embedding</div>
        </div>
        <div className="kpi">
          <span className="label">active per token</span>
          <div className="value">{fmtP(active)}</div>
          <div className="delta" title="parameters a single token's forward pass multiplies by">{pct(active, total)} of the weights</div>
        </div>
        {c.moe.enabled && (
          <div className="kpi">
            <span className="label">routed experts</span>
            <div className="value">{pct(arch.counts.routed_experts ?? 0, total)}</div>
            <div className="delta">of all parameters · {fmtP(arch.counts.routed_experts ?? 0)}</div>
          </div>
        )}
        <div className="kpi">
          <span className="label">forward FLOPs / token</span>
          <div className="value">≈{fmtP(2 * (arch.counts.active_non_embedding ?? active))}</div>
          <div className="delta" title="2 × active non-embedding parameters; attention over the context adds more">2 × active non-embedding</div>
        </div>
        <div className="kpi">
          <span className="label">tensors</span>
          <div className="value">{arch.params.length}</div>
          <div className="delta">{arch.buffers.length ? `+ ${arch.buffers.length} routing-bias buffers` : "no buffers"}</div>
        </div>
      </div>

      <div className="section-head">
        <h2>Where the parameters are</h2>
        <span className="muted">share of {fmtP(total)}</span>
      </div>
      <div className="card">
        <div className="budget" role="img" aria-label="parameter budget by component">
          {groups.map((g) => (
            <i key={g.id} style={{ flexGrow: g.n, background: groupColor(g.id) }} title={`${g.label}: ${fmtP(g.n)} (${pct(g.n, total)})`} />
          ))}
        </div>
        <div className="budget-legend">
          {groups.map((g) => (
            <span key={g.id}>
              <i style={{ background: groupColor(g.id) }} />
              {g.label} <b className="num">{fmtP(g.n)}</b> <span className="muted num">{pct(g.n, total)}</span>
            </span>
          ))}
        </div>
      </div>

      <div className="arch-body">
        <div>
          <div className="section-head">
            <h2>Forward pass</h2>
            <span className="muted">click a block</span>
          </div>
          <div className="card flow">
            <div className="flow-io">
              token ids <Shape>[B, {fmtNum(T)}]</Shape>
            </div>
            <div className="flow-step">
              <b>embed</b> look up a row of E <Shape>
                {fmtNum(c.vocab_size)} × {c.d_model}
              </Shape>
            </div>
            <div className="flow-wire">
              <Shape>[B, T, {c.d_model}]</Shape> residual stream
            </div>
            {info.map((l) => (
              <button key={l.index} className={`flow-block${l.index === layer.index ? " on" : ""}`} onClick={() => setSel(l.index)}>
                <span className="flow-idx num">{l.index}</span>
                <span className="flow-kind">{l.moe ? `MoE ${c.moe.n_experts}e · top-${c.moe.top_k}` : "dense"}</span>
                <span className="flow-bar" aria-hidden>
                  <i style={{ width: `${(100 * l.attn) / maxLayer}%`, background: groupColor("attention") }} />
                  <i style={{ width: `${(100 * l.ffn) / maxLayer}%`, background: groupColor(l.moe ? "routed" : "dense") }} />
                </span>
                <span className="flow-n num">{fmtP(l.attn + l.ffn + l.norms)}</span>
              </button>
            ))}
            <div className="flow-step">
              <b>norm</b> final RMSNorm
            </div>
            <div className="flow-step">
              <b>head</b> logits = h {c.tie_embeddings ? "Eᵀ (tied)" : "W_out"} <Shape>[B, T, {fmtNum(c.vocab_size)}]</Shape>
            </div>
            {c.mtp.depth > 0 && (
              <div className="flow-step mtp">
                <b>MTP</b> norm(h) ‖ norm(embed(t+1)) → proj {2 * c.d_model}→{c.d_model} → dense block → same head: predicts token t+2 · loss × {c.mtp.loss_weight} · {fmtP(mtpParams)} · training only
              </div>
            )}
          </div>
        </div>
        <div>
          <div className="section-head">
            <h2>Block {layer.index}</h2>
            <span className="muted">
              {fmtP(layer.attn + layer.ffn + layer.norms)} parameters · {fmtP(layer.active)} active per token · x ← x + Attn(RMSNorm(x)); x ← x + {layer.moe ? "MoE" : "FFN"}(RMSNorm(x))
            </span>
          </div>
          <div className="card arch-detail">
            <Attention c={c} s={s} />
            {layer.moe ? (
              <MoEBlock c={c} s={s} load={load} loadAt={latest ? `${latest.stage} step ${latest.step}` : null} />
            ) : (
              <div className="arch-sub">
                <div className="arch-sub-head">
                  <h4>Feed-forward (SwiGLU)</h4>
                  <span className="muted">dense: every token uses all {fmtNum(c.ffn_hidden)} hidden units</span>
                </div>
                <DenseFFN c={c} s={s} hidden={c.ffn_hidden} label={`hidden ${fmtNum(c.ffn_hidden)} = ${(c.ffn_hidden / c.d_model).toFixed(1)} × d`} />
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="section-head">
        <h2>All tensors</h2>
        <span className="muted">
          {rows.length} of {arch.params.length} · PyTorch shapes (out × in)
        </span>
        <span className="grow" />
        <input placeholder="filter, e.g. layers.3 or w_gate" value={filter} onChange={(e) => setFilter(e.target.value)} />
      </div>
      <div className="card scroll-x" style={{ maxHeight: 420, overflowY: "auto", padding: 0 }}>
        <table className="report params">
          <thead>
            <tr>
              <th>tensor</th>
              <th>group</th>
              <th>shape</th>
              <th className="r">params</th>
              <th className="r">share</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p: ParamInfo) => (
              <tr key={p.name}>
                <td className="mono">{p.name}</td>
                <td>
                  <span className="dot" style={{ background: groupColor(groupOf(p.name)) }} /> {GROUPS.find((g) => g.id === groupOf(p.name))!.label}
                </td>
                <td className="mono">{dims(p.shape)}</td>
                <td className="r num">{fmtNum(p.numel)}</td>
                <td className="r num muted">{pct(p.numel, total)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
