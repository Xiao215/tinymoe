"""Model-internals snapshots for the viewer (schema ``SCHEMA_VERSION`` in logging.py).

Captured on fixed probe inputs, so successive snapshots are directly comparable:

* per layer: residual-stream norm, attention entropy per head, attention maps (npz),
  attention / FFN output norms, weight norms;
* MoE: expert load since the last snapshot, router bias, score statistics, per-token
  expert assignment for the probes, expert co-activation, token-type specialisation;
* logit lens (nostalgebraist, 2020): top-k next-token predictions decoded from every
  layer's residual stream;
* token-embedding PCA for digits / operators / common words.

Numerics guarantee: capture runs under ``torch.no_grad``, in eval mode (restored after),
inside ``preserved_rng``; MoE load counters only advance in training mode. A test checks
that training losses are bit-identical with and without snapshots.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch

from tinymoe import hardware
from tinymoe.config import RunConfig
from tinymoe.data import mathworld, stories
from tinymoe.data.tokenizer import Tokenizer
from tinymoe.eval.probes import probes
from tinymoe.hardware import Runtime
from tinymoe.logging import RunLogger
from tinymoe.model.hooks import Capture
from tinymoe.model.moe import MoE
from tinymoe.model.transformer import Transformer

log = logging.getLogger(__name__)
CATEGORIES = ("digit", "operator", "word", "punct", "special", "space")


def token_category(text: str, is_special: bool) -> str:
    if is_special:
        return "special"
    s = text.strip()
    if not s:
        return "space"
    if s.isdigit():
        return "digit"
    if s in {"+", "-", "*", "=", "/", "(", ")"}:
        return "operator"
    if s.isalpha():
        return "word"
    return "punct"


def _entropy(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(1e-12).log()).sum(-1)


class Snapshotter:
    def __init__(
        self, cfg: RunConfig, tok: Tokenizer, runtime: Runtime, logger: RunLogger, stage: str
    ) -> None:
        self.cfg = cfg.snapshot
        self.tok = tok
        self.runtime = runtime
        self.logger = logger
        self.stage = stage
        limit = self.cfg.max_probe_tokens
        story = cfg.eval.probe_stories[0] if cfg.eval.probe_stories else "Once upon a time"
        math_probe = next((p for p in probes(cfg) if p.kind == "math"), None)
        texts = {"story": story}
        if math_probe is not None:
            texts["math"] = f"Question: {math_probe.prompt}\nAnswer: {math_probe.answer}"
        self.probe_ids = {k: [tok.bos_id, *tok.encode(t)][:limit] for k, t in texts.items()}
        spec_text = (
            stories.synthetic_story(cfg.eval.seed, 0, "probe")
            + "\n\n"
            + mathworld.pretrain_document(cfg.data.mathworld, cfg.eval.seed, 0, "probe")
        )
        self.spec_ids = [tok.bos_id, *tok.encode(spec_text)][: 4 * limit]
        specials = set(tok.special_ids.values())
        self.spec_categories = [
            CATEGORIES.index(token_category(tok.token_str(i), i in specials)) for i in self.spec_ids
        ]
        found = [(t, tok.token_id(t)) for t in self.cfg.embedding_tokens]
        self.embed_tokens = [(t, i) for t, i in found if i is not None]

    def _forward(self, model: Transformer, ids: list[int]) -> tuple[Capture, torch.Tensor]:
        cap = Capture()
        x = torch.tensor([ids], device=self.runtime.device)
        with self.runtime.autocast():
            out = model(x, capture=cap)
        return cap, out.logits[0].float()

    @torch.no_grad()
    def __call__(self, model: Transformer, step: int, recent_load: torch.Tensor | None) -> None:
        was_training = model.training
        model.eval()
        try:
            with hardware.preserved_rng():
                data, arrays = self.collect(model, recent_load)
            self.logger.save_snapshot(self.stage, step, data, arrays)
        finally:
            model.train(was_training)

    def collect(
        self, model: Transformer, recent_load: torch.Tensor | None
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        cfg = model.cfg
        moe_layers = [b.layer_idx for b in model.blocks if isinstance(b.ffn, MoE)]
        data: dict[str, Any] = {
            "model": {
                "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads,
                "n_kv_heads": cfg.n_kv_heads,
                "d_model": cfg.d_model,
                "moe_layers": moe_layers,
                "n_experts": cfg.moe.n_experts if cfg.moe.enabled else 0,
                "top_k": cfg.moe.top_k if cfg.moe.enabled else 0,
                "max_seq_len": cfg.max_seq_len,
            },
            "probes": [],
            "logit_lens": {},
        }
        arrays: dict[str, np.ndarray] = {}
        per_layer: dict[str, dict[str, Any]] = {}
        assignments: dict[str, Any] = {}
        for name, ids in self.probe_ids.items():
            cap, _ = self._forward(model, ids)
            tokens = [self.tok.token_str(i) for i in ids]
            data["probes"].append({"id": name, "tokens": tokens})
            per_layer[name] = {
                "residual_norm": [float(r[0].norm(dim=-1).mean()) for r in cap.residual],
                "attn_entropy": [
                    [float(v) for v in _entropy(cap.attn_probs[i][0]).mean(-1)]
                    for i in sorted(cap.attn_probs)
                ],
                "attn_out_norm": [
                    float(cap.attn_out_norm[i].mean()) for i in sorted(cap.attn_out_norm)
                ],
                "ffn_out_norm": [
                    float(cap.ffn_out_norm[i].mean()) for i in sorted(cap.ffn_out_norm)
                ],
            }
            t = min(len(ids), self.cfg.attention_max_tokens)
            arrays[f"attn_{name}"] = (
                torch.stack([cap.attn_probs[i][0, :, :t, :t] for i in sorted(cap.attn_probs)])
                .numpy()
                .astype(np.float16)
            )
            assignments[name] = {str(k): v[0].tolist() for k, v in cap.routing.items()}
            data["logit_lens"][name] = self._logit_lens(model, cap, ids)
        data["layers"] = per_layer
        data["weights"] = self._weight_norms(model)
        data["embeddings"] = self._embeddings(model)
        if moe_layers:
            data["moe"] = self._moe(model, moe_layers, recent_load, assignments)
        return data, arrays

    def _logit_lens(self, model: Transformer, cap: Capture, ids: list[int]) -> list[dict[str, Any]]:
        n = min(self.cfg.logit_lens_positions, len(ids) - 1)
        positions = list(range(len(ids) - 1 - n, len(ids) - 1))
        out = []
        for pos in positions:
            layers = []
            for resid in cap.residual:
                h = resid[0, pos].to(self.runtime.device)
                probs = model.head(model.norm(h)).float().softmax(-1)
                top = probs.topk(self.cfg.logit_lens_topk)
                layers.append(
                    [
                        [self.tok.token_str(int(i)), float(p)]
                        for p, i in zip(top.values, top.indices, strict=True)
                    ]
                )
            out.append(
                {
                    "pos": pos,
                    "token": self.tok.token_str(ids[pos]),
                    "target": self.tok.token_str(ids[pos + 1]),
                    "layers": layers,
                }
            )
        return out

    def _weight_norms(self, model: Transformer) -> dict[str, list[float]]:
        def norm(module: torch.nn.Module) -> float:
            return math.sqrt(sum(float(p.float().pow(2).sum()) for p in module.parameters()))

        return {
            "attn": [norm(b.attn) for b in model.blocks],
            "ffn": [norm(b.ffn) for b in model.blocks],
        }

    def _embeddings(self, model: Transformer) -> dict[str, Any]:
        if len(self.embed_tokens) < 3:
            return {"tokens": [], "xy": []}
        ids = torch.tensor([i for _, i in self.embed_tokens], device=model.embed.weight.device)
        e = model.embed.weight[ids].float().cpu()
        e = e - e.mean(0)
        _, _, v = torch.linalg.svd(e, full_matrices=False)
        xy = e @ v[:2].T
        xy = xy * torch.sign(xy[0]).where(xy[0] != 0, torch.ones(2))  # stable orientation
        return {"tokens": [t for t, _ in self.embed_tokens], "xy": xy.tolist()}

    def _moe(
        self,
        model: Transformer,
        moe_layers: list[int],
        recent_load: torch.Tensor | None,
        assignments: dict[str, Any],
    ) -> dict[str, Any]:
        cap, _ = self._forward(model, self.spec_ids)
        n_exp = model.cfg.moe.n_experts
        cats = torch.tensor(self.spec_categories)
        spec, coact, score_mean, score_hist = [], [], [], []
        for layer in moe_layers:
            topk = cap.routing[layer][0]  # (T, k)
            onehot = torch.zeros(topk.shape[0], n_exp).scatter_(1, topk, 1.0)
            spec.append([onehot[cats == c].sum(0).tolist() for c in range(len(CATEGORIES))])
            coact.append((onehot.T @ onehot).tolist())
            scores = cap.router_scores[layer][0]
            score_mean.append(scores.mean(0).tolist())
            score_hist.append(torch.histc(scores, bins=20, min=0.0, max=1.0).tolist())
        load = None
        if recent_load is not None:
            load = (recent_load / recent_load.sum(-1, keepdim=True).clamp_min(1)).cpu().tolist()
        biases = [m.expert_bias.float().cpu().tolist() for m in model.moe_layers()]
        return {
            "layers": moe_layers,
            "load": load,
            "bias": biases,
            "score_mean": score_mean,
            "score_hist": score_hist,
            "coactivation": coact,
            "specialization": {"categories": list(CATEGORIES), "counts": spec},
            "assignments": assignments,
            "expert_out_norm": [cap.expert_out_norm[i].tolist() for i in moe_layers],
        }
