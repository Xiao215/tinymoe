"""Worker for the DDP test: run directly (1 process) or under ``torchrun`` (N processes).

Trains the tiny MoE for a few steps on deterministic rows with a fixed *global* batch:
one process uses grad_accum = 2, two ranks use grad_accum = 1 each. Writes losses and a
parameter checksum (rank 0) and whether all ranks ended with identical parameters.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from helpers import RandomRows, cpu_runtime, make_trainer, read_losses, tiny_model

from tinylab.config import BatchPlan, HardwareConfig
from tinylab.train import distributed
from tinylab.train.providers import RowProvider


def main(out: Path) -> None:
    info = distributed.init(HardwareConfig(device="cpu", dist_backend="gloo"))
    torch.set_num_threads(1)
    plan = BatchPlan(
        micro_batch_size=2, grad_accum=2 // info.world_size, world_size=info.world_size, unit=16
    )
    model = tiny_model(True, seed=0)
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        trainer = make_trainer(
            run_dir, model, RowProvider(RandomRows(16), plan, info.rank, torch.device("cpu")),
            runtime=cpu_runtime(), plan=plan, dist=info, max_steps=6, optimizer="muon",
        )  # fmt: skip
        trainer.fit()
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        bias = torch.cat([m.expert_bias for m in model.moe_layers()])
        agree = True
        if info.enabled:
            gathered = [torch.zeros_like(flat) for _ in range(info.world_size)]
            dist.all_gather(gathered, flat)
            agree = all(torch.equal(gathered[0], g) for g in gathered)
        if info.is_main:
            losses = read_losses(run_dir)
            out.write_text(
                json.dumps(
                    {
                        "world_size": info.world_size,
                        "losses": [losses[s] for s in sorted(losses)],
                        "param_sum": float(flat.double().sum()),
                        "bias_sum": float(bias.sum()),
                        "ranks_agree": agree,
                    }
                )
            )
    distributed.shutdown()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
