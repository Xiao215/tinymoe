from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tinymoe.config import DataConfig, load_stage_config
from tinymoe.data.pretrain_mix import build_tokenizer
from tinymoe.data.tokenizer import Tokenizer

torch.set_num_threads(4)


@pytest.fixture(scope="session")
def data_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data_cache")


@pytest.fixture(scope="session")
def smoke_data(data_cache: Path) -> DataConfig:
    return load_stage_config(["data=smoke", f"data.cache_dir={data_cache}"]).data


@pytest.fixture(scope="session")
def tok(smoke_data: DataConfig) -> Tokenizer:
    return build_tokenizer(smoke_data)[0]
