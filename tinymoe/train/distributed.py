"""Optional DDP. Activated only when launched by ``torchrun`` (``WORLD_SIZE`` > 1).

Single-device runs need no launcher and every helper here degrades to a no-op, so stage
code never branches on "am I distributed?" beyond calling these functions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from tinymoe import hardware
from tinymoe.config import HardwareConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistInfo:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def env_info() -> DistInfo:
    return DistInfo(
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


def init(cfg: HardwareConfig) -> DistInfo:
    """Initialise the process group if launched with WORLD_SIZE>1."""
    info = env_info()
    if info.enabled and not dist.is_initialized():
        device = hardware.resolve_device(cfg.device, info.local_rank)
        backend = hardware.dist_backend(cfg.dist_backend, device)
        dist.init_process_group(backend=backend)
        log.info("[dist] rank %d/%d backend=%s", info.rank, info.world_size, backend)
    return info


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    if is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor


def barrier() -> None:
    if is_initialized():
        dist.barrier()


def wrap(model: nn.Module, device: torch.device) -> nn.Module:
    """DDP-wrap when distributed. ``find_unused_parameters`` covers MoE experts that get
    no tokens in a micro batch (their grads would otherwise be missing on some ranks)."""
    if not is_initialized():
        return model
    ids = [device.index] if device.index is not None else None
    return nn.parallel.DistributedDataParallel(model, device_ids=ids, find_unused_parameters=True)


def unwrap(model: nn.Module) -> nn.Module:
    inner = getattr(model, "module", None)
    return inner if isinstance(inner, nn.Module) else model


def shutdown() -> None:
    if is_initialized():
        dist.destroy_process_group()
