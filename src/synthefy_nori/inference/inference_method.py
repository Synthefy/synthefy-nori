from __future__ import annotations

import contextlib
import gc
import os
import socket

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from synthefy_nori.utils.data_utils import DistributedInferenceDataset
from synthefy_nori.utils.inference_utils import (
    NonPaddingDistributedSampler,
    swap_rows_back,
)
from synthefy_nori.utils.loading import load_model


def _pick_free_port():
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def setup():
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), False

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_pick_free_port()))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, True


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedInference:
    """Run full-context query batches through the optional DDP path."""

    def __init__(
        self,
        model: torch.nn.Module | str,
        device: int | torch.device = 0,
        mix_precision: bool = True,
    ):
        self.model = load_model(model) if isinstance(model, str) else model
        self.requested_device = torch.device(f"cuda:{device}") if isinstance(device, int) else torch.device(device)
        if self.requested_device.type != "cuda":
            raise ValueError("distributed inference requires a CUDA device")
        self.mix_precision = bool(mix_precision)
        self.rank: int | None = None
        self.world_size: int | None = None
        self.device: torch.device | None = None
        self._owns_process_group = False

    def __enter__(self) -> DistributedInference:
        if self.rank is not None:
            return self

        rank, world_size, owns_process_group = setup()
        try:
            local_rank = int(
                os.environ.get(
                    "LOCAL_RANK",
                    self.requested_device.index
                    if self.requested_device.index is not None
                    else torch.cuda.current_device(),
                )
            )
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        except Exception:
            if owns_process_group:
                cleanup()
            raise

        self.rank = rank
        self.world_size = world_size
        self.device = device
        self._owns_process_group = owns_process_group
        return self

    def close(self) -> None:
        if self._owns_process_group:
            cleanup()
        self.rank = None
        self.world_size = None
        self.device = None
        self._owns_process_group = False

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def inference(
        self,
        x_train: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        x_test: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        if isinstance(x_train, np.ndarray):
            x_train = torch.from_numpy(x_train)
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train)
        if isinstance(x_test, np.ndarray):
            x_test = torch.from_numpy(x_test)

        if x_train.ndim != 2 or x_test.ndim != 2:
            raise ValueError("distributed inference expects 2D train and test features")
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train[:, 0]
        if y_train.ndim != 1 or y_train.shape[0] != x_train.shape[0]:
            raise ValueError(
                "distributed inference labels must have shape [rows] or [rows, 1] and match the context rows"
            )
        if x_train.shape[1] != x_test.shape[1]:
            raise ValueError("distributed inference train/test feature counts must match")
        if x_test.shape[0] == 0:
            raise ValueError("distributed inference requires at least one query row")

        dataset = DistributedInferenceDataset(x_test)
        manages_context = self.rank is None
        if manages_context:
            self.__enter__()
        assert self.rank is not None
        assert self.world_size is not None
        assert self.device is not None
        try:
            model = DDP(
                self.model.to(self.device),
                device_ids=[self.device.index],
                output_device=self.device.index,
                # Synchronize buffers once while every rank is participating.
                broadcast_buffers=True,
                find_unused_parameters=False,
            )
            # Some ranks can have no query rows. Per-forward buffer broadcasts
            # would deadlock when only non-empty ranks call forward.
            model.broadcast_buffers = False
            sampler = NonPaddingDistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            cells_per_query = max(x_train.shape[0] * x_train.shape[1], 1)
            batch_size = min(max(1_000_000 // cells_per_query, 1), 1024)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                sampler=sampler,
            )

            outputs = []
            indices = []
            x_context = x_train.to(self.device)
            y_context = y_train.to(self.device)
            for data in dataloader:
                indices.append(data["idx"])
                x_query = data["X_test"].to(self.device)
                batch = x_query.shape[0]
                x_context_batch = x_context.unsqueeze(0).expand(batch, -1, -1)
                y_context_batch = y_context.unsqueeze(0).expand(batch, -1)
                x_all = torch.cat((x_context_batch, x_query.unsqueeze(1)), dim=1)
                with (
                    torch.autocast(
                        self.device.type,
                        enabled=self.mix_precision,
                    ),
                    torch.inference_mode(),
                ):
                    output = model(
                        x=x_all,
                        y=y_context_batch,
                        eval_pos=y_context.shape[0],
                    )
                    if isinstance(output, dict):
                        output = output["reg_output"]
                    if output.ndim == 3:
                        output = output.view(-1, output.shape[-1])
                outputs.append(output.cpu())
                del output
                gc.collect()
                torch.cuda.empty_cache()

            local_outputs = torch.cat(outputs, dim=0) if outputs else None
            local_indices = torch.cat(indices, dim=0) if indices else torch.empty(0, dtype=torch.long)
            gathered_outputs = [None for _ in range(self.world_size)]
            gathered_indices = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered_outputs, local_outputs)
            dist.all_gather_object(gathered_indices, local_indices)
            result = torch.cat(
                [output for output in gathered_outputs if output is not None],
                dim=0,
            ).to(torch.float32)
            result_indices = torch.cat(gathered_indices, dim=0)
            return swap_rows_back(result, result_indices)
        finally:
            if manages_context:
                self.close()
