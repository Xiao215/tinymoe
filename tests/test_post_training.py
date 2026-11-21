"""M7/M8: LoRA, SFT data path, preference losses and pair batches."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from helpers import tiny_model

from tinylab.config import DPOConfig, LoRAConfig
from tinylab.data.batch import IGNORE_INDEX
from tinylab.data.tokenizer import Message, Tokenizer
from tinylab.model.lora import LoRALinear, apply_lora, merge_lora
from tinylab.stages.dpo import EncodedPair, PairDataset, encode_side
from tinylab.train.optim import classify
from tinylab.train.preference import preference_loss


def test_lora_starts_at_base_and_merges_exactly() -> None:
    model = tiny_model(True, seed=3)
    x = torch.randint(0, 61, (2, 12))
    base_out = model(x).logits.detach()
    base_keys = set(model.state_dict())
    targets = ["q_proj", "v_proj", "gate_proj"]
    expected = [
        name
        for name, m in model.named_modules()
        if name.rsplit(".", 1)[-1] in targets and isinstance(m, torch.nn.Linear)
    ]
    assert any(".mtp." in f".{n}" or n.startswith("mtp.") for n in expected)  # MTP block too
    assert any("shared" in n for n in expected)  # and the MoE shared expert
    n = apply_lora(model, LoRAConfig(rank=4, alpha=8.0, targets=targets))
    assert n == len(expected)
    assert torch.allclose(model(x).logits, base_out)  # B = 0 at init
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in name for name in trainable)
    assert all(
        classify(name, p) == "no_decay" for name, p in model.named_parameters() if "lora_" in name
    )
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_b.normal_(0, 0.5)
    adapted = model(x).logits.detach()
    assert not torch.allclose(adapted, base_out)
    assert merge_lora(model) == n
    assert set(model.state_dict()) == base_keys
    assert torch.allclose(model(x).logits, adapted, atol=1e-5)
    assert all(p.requires_grad for p in model.parameters())


def test_lora_rejects_missing_targets() -> None:
    with pytest.raises(ValueError, match=r"no nn\.Linear"):
        apply_lora(tiny_model(False), LoRAConfig(targets=["nope"]))


def _pl(
    cfg: DPOConfig,
    pc: float,
    pr: float,
    rc: float = 0.0,
    rr: float = 0.0,
    lc: float = 1,
    lr: float = 1,
) -> float:
    t = lambda v: torch.tensor([v])  # noqa: E731
    return float(preference_loss(cfg, t(pc), t(pr), t(rc), t(rr), t(lc), t(lr)).loss)


def test_dpo_ipo_simpo_losses() -> None:
    dpo = DPOConfig(loss="dpo", beta=0.5)
    assert _pl(dpo, -3.0, -3.0, -3.0, -3.0) == pytest.approx(math.log(2))  # policy == ref
    assert _pl(dpo, -1.0, -5.0, -3.0, -3.0) < math.log(2)  # chosen up, rejected down
    smooth = DPOConfig(loss="dpo", beta=0.5, label_smoothing=0.5)
    assert _pl(smooth, -1.0, -5.0) == pytest.approx(
        _pl(smooth, -5.0, -1.0)
    )  # eps=.5: no preference
    ipo = DPOConfig(loss="ipo", beta=0.25)
    target = 1 / (2 * 0.25)
    assert _pl(ipo, target, 0.0) == pytest.approx(0.0)
    assert _pl(ipo, target + 3, 0.0) > 0  # IPO also penalises overshooting the margin
    simpo = DPOConfig(loss="simpo", beta=2.0, simpo_gamma=0.5)
    # reference-free and length-normalised: doubling both sequence lengths and log-probs is a no-op
    assert _pl(simpo, -2.0, -4.0, rc=99, rr=-99) == pytest.approx(
        _pl(simpo, -4.0, -8.0, lc=2, lr=2)
    )


def test_dpo_gradient_prefers_chosen() -> None:
    pc = torch.tensor([-2.0], requires_grad=True)
    pr = torch.tensor([-2.0], requires_grad=True)
    z, one = torch.zeros(1), torch.ones(1)
    preference_loss(DPOConfig(beta=0.1), pc, pr, z - 2, z - 2, one, one).loss.backward()
    assert pc.grad is not None and pr.grad is not None
    assert float(pc.grad) < 0 < float(pr.grad)  # descent raises chosen, lowers rejected


def test_pair_encoding_and_batch(tok: Tokenizer) -> None:
    prompt = (Message("user", "What is 2 + 3?"),)
    x, y = encode_side(tok, prompt, "<think>\n2 + 3 = 5\n</think>\nAnswer: 5", 64)
    supervised = y[y != IGNORE_INDEX]
    assert tok.decode(list(supervised)).startswith("<think>")
    assert supervised[-1] == tok.end_id  # the model learns to stop
    assert (y[: len(tok.encode("What is 2 + 3?"))] == IGNORE_INDEX).all()  # prompt not trained
    short = encode_side(tok, prompt, "Answer: 5", 64)
    data = PairDataset(
        [EncodedPair((x, y), short, 0), EncodedPair(short, (x, y), 1)], tok.pad_id, 0
    )
    data.ref_logp[:] = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = data.collate([0, 1])
    assert b.input_ids.shape == (4, len(x))
    assert b.ref_logp.tolist() == [1.0, 3.0, 2.0, 4.0]  # chosen block, then rejected block
    assert b.kinds.tolist() == [0, 1]
    assert (b.labels[1, len(short[1]) :] == IGNORE_INDEX).all()  # right padding is ignored
