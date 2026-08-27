"""End-to-end training smoke test.

Runs a single optimizer step of the real training loop on CPU (data-gen ->
CCMM/pinball loss -> optimizer step -> checkpoint write) and asserts it exits
cleanly and writes a checkpoint. This is the training half of the CI merge gate;
the inference half lives in ``test_inference_e2e.py``.

No network access is required (it trains a tiny model from scratch, no checkpoint
download), but it is marked ``slow`` because it spins up the full training stack
and is opt-in alongside the other end-to-end tests.

By default it runs on CPU. Set ``SYNTHEFY_NORI_SMOKE_DEVICE`` (e.g. ``cuda:0``)
to exercise the GPU training path instead; on a CUDA device it also leaves mixed
precision enabled so the autocast path is covered.

Run explicitly with::

    pytest -m slow tests/test_training_smoke.py
    SYNTHEFY_NORI_SMOKE_DEVICE=cuda:0 pytest -m slow tests/test_training_smoke.py
"""

from __future__ import annotations

import os
import json
import subprocess
import sys

import pytest
import torch


pytestmark = pytest.mark.slow


def test_single_training_step_writes_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "smoke"
    device = os.environ.get("SYNTHEFY_NORI_SMOKE_DEVICE", "cpu")

    cmd = [
        sys.executable,
        "-m",
        "synthefy_nori.training.cli",
        "--device",
        device,
        "--no-prefetch",
        "--no-wandb",
        "--total-steps",
        "1",
        "--run-steps",
        "1",
        "--save-interval",
        "1",
        # Tiny architecture so the step is fast on a CPU runner.
        "--embed-dim",
        "32",
        "--hid-dim",
        "64",
        "--nlayers",
        "2",
        "--nhead",
        "2",
        "--batch-size",
        "2",
        "--max-features",
        "16",
        "--max-budget",
        "4000",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    # CPU has no autocast support here; on CUDA, keep mixed precision on (default)
    # so the GPU run also exercises the autocast path.
    if device == "cpu":
        cmd.append("--no-mixed-precision")

    env = dict(os.environ, WANDB_MODE="disabled")
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"training step failed (exit {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    assert checkpoints, f"no checkpoint written to {checkpoint_dir}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert result.stdout.count("Checkpoint saved to") == 1, (
        f"a boundary checkpoint must not be overwritten by the final save\nSTDOUT:\n{result.stdout}"
    )


@pytest.mark.parametrize(
    ("use_memory_plan", "expected_physical_micro_steps"),
    [(True, 2), (False, 1)],
)
def test_logical_batch_runs_with_or_without_memory_plan(tmp_path, use_memory_plan, expected_physical_micro_steps):
    checkpoint_dir = tmp_path / "planned-smoke"
    memory_plan = tmp_path / "memory-plan.json"
    if use_memory_plan:
        memory_plan.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "headroom_fraction": 0.15,
                    "max_per_rank_microbatch": 2,
                    "usable_memory_bytes": 1,
                    "shapes": {
                        "64x4": {
                            "rows": 64,
                            "features": 4,
                            "safe_microbatch": 2,
                            "peak_allocated_bytes": 0,
                            "peak_reserved_bytes": 0,
                        }
                    },
                }
            )
        )
    cmd = [
        sys.executable,
        "-m",
        "synthefy_nori.training.cli",
        "--device",
        "cpu",
        "--no-mixed-precision",
        "--no-prefetch",
        "--no-wandb",
        "--fixed-size",
        "64x4",
        "--max-budget",
        "4000",
        "--total-steps",
        "1",
        "--run-steps",
        "1",
        "--save-interval",
        "1",
        "--embed-dim",
        "32",
        "--hid-dim",
        "64",
        "--nlayers",
        "2",
        "--nhead",
        "2",
        "--batch-size",
        "4",
        "--gradient-accumulation",
        "1",
        "--target-global-batch-size",
        "4",
        "--no-oom-checkpoint-retry",
        "--no-oom-shape-blacklist",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if use_memory_plan:
        cmd.extend(["--memory-plan", str(memory_plan)])
    result = subprocess.run(
        cmd,
        env=dict(os.environ, WANDB_MODE="disabled"),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"planned training failed (exit {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    checkpoint = torch.load(
        next(checkpoint_dir.glob("checkpoint_step_*.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["optimizer_step"] == 1
    assert checkpoint["physical_micro_steps"] == expected_physical_micro_steps
    assert checkpoint["config"].global_batch_size == 4


def test_prefetch_refill_survives_a_second_step(tmp_path):
    """Two optimizer steps with the prefetcher enabled.

    The in-loop prefetch refill runs only when a prefetcher exists *and* another
    step is still to be submitted, so neither a ``--no-prefetch`` run nor a
    single-step run ever reaches it. Every other test here is one or both of
    those, which is how an arity mismatch in that block once passed the whole
    suite while making any real training run die before logging step 1.

    Two steps with prefetch on is the smallest configuration that covers it.
    """
    checkpoint_dir = tmp_path / "prefetch-refill"
    cmd = [
        sys.executable,
        "-m",
        "synthefy_nori.training.cli",
        "--device",
        "cpu",
        "--no-mixed-precision",
        "--no-wandb",
        # Deliberately NOT --no-prefetch: the refill path is the subject.
        "--prefetch-workers",
        "1",
        "--prefetch-count",
        "2",
        "--fixed-size",
        "64x4",
        "--max-budget",
        "4000",
        "--total-steps",
        "2",
        "--run-steps",
        "2",
        "--save-interval",
        "2",
        "--embed-dim",
        "32",
        "--hid-dim",
        "64",
        "--nlayers",
        "2",
        "--nhead",
        "2",
        "--batch-size",
        "2",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    result = subprocess.run(
        cmd,
        env=dict(os.environ, WANDB_MODE="disabled"),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, (
        f"prefetched training failed (exit {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    checkpoint = torch.load(
        next(checkpoint_dir.glob("checkpoint_step_*.pt")),
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["optimizer_step"] == 2
