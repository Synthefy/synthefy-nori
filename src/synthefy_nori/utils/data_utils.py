from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DistributedInferenceDataset(Dataset):
    """Shard query rows for the optional distributed inference runner."""

    def __init__(self, x_test: torch.Tensor):
        if x_test.ndim != 2:
            raise ValueError(
                f"distributed inference expects X_test with shape [rows, features], got {tuple(x_test.shape)}"
            )
        self.x_test = x_test

    def __len__(self) -> int:
        return self.x_test.shape[0]

    def __getitem__(self, idx: int) -> dict[str, int | torch.Tensor]:
        return {"idx": int(idx), "X_test": self.x_test[idx]}
