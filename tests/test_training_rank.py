"""Global process identity stays distinct from the node-local CUDA index."""

import numpy as np
import pytest
import torch

from synthefy_nori.training.config import TrainingConfig
from synthefy_nori.training.trainer import NoriTrainer


def _trainer(rank, tmp_path):
    config = TrainingConfig(
        device="cpu",
        rank=rank,
        local_rank=rank % 4,
        world_size=8,
        optimizer="adamw",
        prefetch_workers=0,
        use_wandb=False,
        mixed_precision=False,
        checkpoint_dir=str(tmp_path),
        batch_size=2,
    )
    return NoriTrainer(torch.nn.Linear(2, 1), config)


def test_node_local_rank_zero_has_distinct_data_and_no_rank_zero_outputs(tmp_path):
    first = _trainer(0, tmp_path)
    second = _trainer(4, tmp_path)
    assert first.is_main
    assert not second.is_main
    np.testing.assert_array_equal(first.shared_rng.integers(10000, size=20), second.shared_rng.integers(10000, size=20))
    assert not np.array_equal(first.rng.integers(10000, size=20), second.rng.integers(10000, size=20))
    second.save_checkpoint()
    assert not list(tmp_path.iterdir())


def test_legacy_resume_keeps_same_local_rank_data_streams_distinct(tmp_path):
    first = _trainer(0, tmp_path)
    second = _trainer(4, tmp_path)
    checkpoint = tmp_path / "legacy.pt"
    first.save_checkpoint(str(checkpoint))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    del saved["runtime_states"]
    torch.save(saved, checkpoint)

    for trainer in (first, second):
        with pytest.warns(UserWarning, match="no RNG state"):
            trainer.load_checkpoint(str(checkpoint))
    np.testing.assert_array_equal(first.shared_rng.integers(10000, size=20), second.shared_rng.integers(10000, size=20))
    assert not np.array_equal(first.rng.integers(10000, size=20), second.rng.integers(10000, size=20))
