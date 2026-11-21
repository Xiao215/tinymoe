from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn.functional as F

from tinymoe.config import ModelConfig, MoEConfig, MTPConfig, RopeScalingConfig, section_from_dict
from tinymoe.model.config import param_count
from tinymoe.model.hooks import Capture
from tinymoe.model.moe import MoE
from tinymoe.model.rope import inv_frequencies
from tinymoe.model.transformer import Transformer, generate, generate_speculative
from tinymoe.train.autobatch import worst_case_routing

V = 97


def make_cfg(use_moe: bool = True, **kw: object) -> ModelConfig:
    base = {
        "vocab_size": V,
        "d_model": 32,
        "n_layers": 3,
        "n_heads": 4,
        "n_kv_heads": 2,
        "ffn_hidden": 64,
        "max_seq_len": 64,
        "moe": {
            "enabled": use_moe,
            "n_experts": 6,
            "top_k": 2,
            "expert_hidden": 16,
            "n_dense_layers": 1,
        },
        "mtp": {"depth": 1},
    }
    base.update(kw)
    cfg: ModelConfig = section_from_dict(ModelConfig, base)
    return cfg


def model(moe: bool = True, seed: int = 0, **kw: object) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(make_cfg(moe, **kw)).eval()


@pytest.mark.parametrize("moe", [False, True])
def test_shapes(moe: bool) -> None:
    m = model(moe).train()
    x = torch.randint(0, V, (2, 10))
    out = m(x)
    assert out.logits.shape == (2, 10, V)
    assert len(out.mtp_logits) == 1 and out.mtp_logits[0].shape == (2, 9, V)
    assert out.aux_loss.ndim == 0
    assert set(out.routing) == ({1, 2} if moe else set())


@pytest.mark.parametrize("moe", [False, True])
def test_causal(moe: bool) -> None:
    m = model(moe)
    x = torch.randint(0, V, (1, 16))
    y = x.clone()
    y[0, 9:] = (y[0, 9:] + 1) % V
    a, b = m(x).logits, m(y).logits
    torch.testing.assert_close(a[:, :9], b[:, :9])
    assert not torch.allclose(a[:, 9:], b[:, 9:])


def test_doc_mask_isolates_documents() -> None:
    m = model(True)
    x = torch.randint(0, V, (1, 12))
    docs = torch.tensor([[1] * 5 + [2] * 7])
    pos = torch.tensor([list(range(5)) + list(range(7))])
    full = m(x, position_ids=pos, doc_ids=docs).logits
    alone = m(x[:, 5:]).logits  # second doc on its own
    torch.testing.assert_close(full[:, 5:], alone, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("window", [None, 4])
def test_kv_cache_matches_full_forward(window: int | None) -> None:
    m = model(True, sliding_window=window, global_every=2)
    x = torch.randint(0, V, (1, 20))
    full = m(x).logits
    cache = m.new_cache(1, 20)
    steps = [m(x[:, :6], cache=cache).logits]
    for t in range(6, 20):
        steps.append(m(x[:, t : t + 1], cache=cache).logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)


def test_left_padded_batch_matches_single() -> None:
    m = model(True)
    prompts = [[5, 6, 7, 8, 9, 10], [11, 12, 13]]
    gen = generate(m, prompts, max_new_tokens=8, temperature=0.0)
    for p, seq in zip(prompts, gen.sequences, strict=True):
        alone = generate(m, [p], max_new_tokens=8, temperature=0.0).sequences[0]
        assert seq == alone
        # greedy decoding == argmax of full forward at each step
        ids = torch.tensor([p + seq])
        logits = m(ids).logits[0, len(p) - 1 : -1]
        assert logits.argmax(-1).tolist() == seq


def test_generation_logprobs_and_routing() -> None:
    m = model(True)
    g = torch.Generator().manual_seed(0)
    gen = generate(m, [[1, 2, 3], [4, 5]], 6, temperature=1.0, generator=g, record_routing=True)
    assert gen.routing is not None
    for p, seq, lps, route in zip(
        [[1, 2, 3], [4, 5]], gen.sequences, gen.logprobs, gen.routing, strict=True
    ):
        ids = torch.tensor([p + seq])
        out = m(ids[:, :-1])
        ref = out.logits.log_softmax(-1)[0, len(p) - 1 :].gather(-1, ids[0, len(p) :, None])
        torch.testing.assert_close(torch.tensor(lps), ref.squeeze(-1), atol=1e-4, rtol=1e-4)
        for layer, r in route.items():
            assert (torch.from_numpy(r) == out.routing[layer][0]).all()


def naive_moe(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    """Per-token reference implementation (the definition of the layer)."""
    h = x.reshape(-1, x.shape[-1])
    _, _, topk, gates = moe.route(h, None)
    out = torch.zeros_like(h)
    for i in range(h.shape[0]):
        for j in range(moe.top_k):
            e = int(topk[i, j])
            out[i] += gates[i, j] * moe._expert(h[i : i + 1], e)[0]
    if moe.shared is not None:
        out += moe.shared(h)
    return out.view_as(x)


@pytest.mark.parametrize("dispatch", ["loop", "padded", "auto"])
@pytest.mark.parametrize("score", ["sigmoid", "softmax"])
def test_moe_matches_naive(dispatch: str, score: str) -> None:
    torch.manual_seed(0)
    cfg = make_cfg(True)
    cfg.moe.dispatch = dispatch  # type: ignore[assignment]
    cfg.moe.score_func = score  # type: ignore[assignment]
    moe = MoE(cfg, 1)
    for p in moe.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    moe.expert_bias.normal_()
    x = torch.randn(2, 7, cfg.d_model)
    y, _ = moe(x)
    torch.testing.assert_close(y, naive_moe(moe, x), atol=1e-5, rtol=1e-4)


def test_moe_dispatch_gradients_agree() -> None:
    torch.manual_seed(0)
    cfg = make_cfg(True)
    grads = {}
    for mode in ("loop", "padded"):
        cfg.moe.dispatch = mode  # type: ignore[assignment]
        torch.manual_seed(1)
        moe = MoE(cfg, 1)
        for p in moe.parameters():
            torch.nn.init.normal_(p, 0, 0.3)
        x = torch.randn(3, 5, cfg.d_model)
        y, aux = moe(x)
        (y.pow(2).sum() + aux.loss).backward()
        grads[mode] = [p.grad.clone() for p in moe.parameters() if p.grad is not None]
    for a, b in zip(grads["loop"], grads["padded"], strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("capacity", [None, 3.0])
def test_padded_overflow_is_exact_and_bounded(
    capacity: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A collapsed router (most tokens on one expert) must not widen the padded buffer past
    ``padded_limit`` x balanced load; the overflow runs per expert and matches ``loop``."""
    cfg = make_cfg(True)
    cfg.moe.capacity_factor = capacity
    n, k, e = 18, cfg.moe.top_k, cfg.moe.n_experts
    hot = torch.zeros(n, dtype=torch.long)  # every token's first choice is expert 0
    override = torch.stack([hot, torch.arange(n) % (e - 1) + 1], 1).view(2, 9, k)
    widths: list[int] = []
    real_bmm = torch.bmm

    def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        widths.append(a.shape[1])
        return real_bmm(a, b)

    monkeypatch.setattr(torch, "bmm", bmm)
    torch.manual_seed(0)
    x = torch.randn(2, 9, cfg.d_model)
    outs = {}
    for mode in ("loop", "padded"):
        cfg.moe.dispatch = mode  # type: ignore[assignment]
        torch.manual_seed(1)
        moe = MoE(cfg, 1)
        for p in moe.parameters():
            torch.nn.init.normal_(p, 0, 0.3)
        outs[mode] = moe(x, override)[0]
    torch.testing.assert_close(outs["padded"], outs["loop"], atol=1e-5, rtol=1e-4)
    assert widths and max(widths) <= math.ceil(cfg.moe.padded_limit * n * k / e) < n


def test_worst_case_routing_pins_padded_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Autobatch probes with padded buffers at their cap; numerics must not change."""
    torch.manual_seed(0)
    cfg = make_cfg(True)
    cfg.moe.dispatch = "padded"  # type: ignore[assignment]
    model = Transformer(cfg)
    x = torch.randint(0, V, (2, 9))
    widths: list[int] = []
    real_bmm = torch.bmm

    def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        widths.append(a.shape[1])
        return real_bmm(a, b)

    monkeypatch.setattr(torch, "bmm", bmm)
    with torch.no_grad():
        normal = model(x).logits
        natural = max(widths)
        widths.clear()
        with worst_case_routing(model):
            assert all(m.worst_case for m in model.moe_layers())
            pinned = model(x).logits
    assert not any(m.worst_case for m in model.moe_layers())
    n, k, e = 18, cfg.moe.top_k, cfg.moe.n_experts
    assert set(widths) == {math.ceil(cfg.moe.padded_limit * n * k / e)} and max(widths) >= natural
    torch.testing.assert_close(pinned, normal, atol=1e-5, rtol=1e-4)


def test_capacity_factor_drops_overflow() -> None:
    torch.manual_seed(0)
    cfg = make_cfg(True)
    cfg.moe.capacity_factor = 0.5
    moe = MoE(cfg, 1)
    x = torch.randn(1, 30, cfg.d_model)
    y, _ = moe(x)
    cfg_free = copy.deepcopy(cfg)
    cfg_free.moe.capacity_factor = 100.0
    moe_free = MoE(cfg_free, 1)
    moe_free.load_state_dict(moe.state_dict())
    y_free, _ = moe_free(x)
    torch.testing.assert_close(y_free, naive_moe(moe_free, x), atol=1e-5, rtol=1e-4)
    assert not torch.allclose(y, y_free)  # some tokens were dropped at cf=0.5


def test_aux_free_balancing_equalises_load() -> None:
    """Sign-based bias updates drive the *time-averaged* load to uniform (DeepSeek-V3)."""
    torch.manual_seed(0)
    cfg = make_cfg(True)
    cfg.moe.bias_update_rate = 0.005
    moe = MoE(cfg, 1).train()
    with torch.no_grad():
        moe.router.weight.normal_(0, 0.1)
        moe.router.weight[0] += 0.08  # inputs are positive -> expert 0 is preferred
    x = torch.randn(8, 32, cfg.d_model).abs()

    def step(update: bool) -> torch.Tensor:
        moe(x)
        counts = moe.reset_counts()
        if update:
            moe.update_bias(counts)
        return counts / counts.sum()

    before = step(update=False)
    for _ in range(400):
        step(update=True)
    after = torch.stack([step(update=True) for _ in range(100)]).mean(0)
    uniform = 1 / cfg.moe.n_experts
    assert before.max() > 2 * uniform
    assert after.max() < 1.3 * uniform and after.min() > 0.7 * uniform


@pytest.mark.parametrize("balance", ["switch", "aux_free"])
def test_aux_losses_have_gradient(balance: str) -> None:
    cfg = make_cfg(True)
    cfg.moe.balance = balance  # type: ignore[assignment]
    cfg.moe.z_loss_weight = 1e-3
    moe = MoE(cfg, 1)
    _, aux = moe(torch.randn(2, 6, cfg.d_model))
    aux.loss.backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0


def test_routing_replay() -> None:
    m = model(True).train()
    x = torch.randint(0, V, (2, 8))
    out = m(x)
    replay = m(x, routing_override={k: v.detach() for k, v in out.routing.items()})
    torch.testing.assert_close(replay.logits, out.logits)
    shifted = {k: (v + 1) % 6 for k, v in out.routing.items()}
    other = m(x, routing_override=shifted)
    assert not torch.allclose(other.logits, out.logits)
    other.logits.sum().backward()
    moe = m.moe_layers()[0]
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "cfg",
    [
        make_cfg(True),
        make_cfg(False),
        make_cfg(True, tie_embeddings=False, qk_norm=False, head_dim=16),
        make_cfg(
            True,
            moe={
                "enabled": True,
                "n_experts": 8,
                "top_k": 1,
                "n_shared_experts": 0,
                "expert_hidden": 8,
                "n_dense_layers": 0,
            },
            mtp={"depth": 2},
        ),
    ],
)
def test_param_count_formula(cfg: ModelConfig) -> None:
    m = Transformer(cfg)
    actual = sum(p.numel() for p in m.parameters())
    pc = param_count(cfg)
    assert pc.total == actual
    inactive = sum(
        (moe.n_experts - moe.top_k) * (moe.w_gate[0].numel() * 3) for moe in m.moe_layers()
    )
    assert pc.active == actual - inactive


def test_speculative_equals_greedy() -> None:
    m = model(True)
    m.mtp[0].proj.weight.data.zero_()  # any MTP weights must give identical output
    prompt = [3, 1, 4, 1, 5, 9, 2, 6]
    greedy = generate(m, [prompt], 20, temperature=0.0).sequences[0]
    spec, stats = generate_speculative(m, prompt, 20)
    assert spec == greedy
    assert stats["drafted"] > 0


def test_speculative_accepts_with_good_drafter() -> None:
    torch.manual_seed(0)
    m = Transformer(make_cfg(False, n_layers=1, mtp={"depth": 1}))
    # train a little so main model and MTP head agree on a deterministic sequence
    seq = torch.arange(0, 40).remainder(V)[None]
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    m.train()
    for _ in range(150):
        out = m(seq[:, :-1])
        loss = F.cross_entropy(out.logits[0], seq[0, 1:]) + F.cross_entropy(
            out.mtp_logits[0][0], seq[0, 2:]
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    spec, stats = generate_speculative(m, seq[0, :5].tolist(), 20)
    assert spec == generate(m, [seq[0, :5].tolist()], 20, temperature=0.0).sequences[0]
    assert stats["accepted"] / stats["drafted"] > 0.5


def test_rope_scaling_variants() -> None:
    base, _ = inv_frequencies(16, 10000.0, RopeScalingConfig(), 64)
    lin, _ = inv_frequencies(16, 10000.0, RopeScalingConfig(type="linear", factor=4.0), 64)
    torch.testing.assert_close(lin, base / 4)
    ntk, _ = inv_frequencies(16, 10000.0, RopeScalingConfig(type="ntk", factor=4.0), 64)
    assert ntk[0] == base[0] and (ntk[1:] < base[1:]).all()
    yarn, mscale = inv_frequencies(
        16, 10000.0, RopeScalingConfig(type="yarn", factor=4.0, original_max_seq_len=4096), 16384
    )
    assert mscale > 1 and torch.isfinite(yarn).all()
    assert yarn[0] == base[0]  # high-frequency dims are left alone by NTK-by-parts


def test_extend_context() -> None:
    m = model(False)
    m.extend_context(128, RopeScalingConfig(type="ntk", factor=2.0))
    out = m(torch.randint(0, V, (1, 100)))
    assert out.logits.shape == (1, 100, V) and torch.isfinite(out.logits).all()


def test_capture_does_not_change_output() -> None:
    m = model(True)
    x = torch.randint(0, V, (1, 9))
    cap = Capture()
    a = m(x).logits
    b = m(x, capture=cap).logits
    assert torch.equal(a, b)
    assert len(cap.residual) == 4 and set(cap.attn_probs) == {0, 1, 2}
    torch.testing.assert_close(cap.attn_probs[0].sum(-1), torch.ones(1, 4, 9))
    assert set(cap.routing) == {1, 2} and cap.router_scores[1].shape == (1, 9, 6)


def test_moe_config_defaults() -> None:
    assert MoEConfig().balance == "aux_free" and MTPConfig().depth == 0
