"""End-to-end training smoke test.

Runs a single optimizer step of the real training loop on CPU (data-gen ->
CCMM/pinball loss -> optimizer step -> checkpoint write) and asserts it exits
cleanly and writes a checkpoint. This is the training half of the CI merge gate;
the inference half lives in ``test_inference_e2e.py``.

No network access is required (it trains a tiny model from scratch, no checkpoint
download), but it is marked ``slow`` because it spins up the full training stack
and is opt-in alongside the other end-to-end tests.

Run explicitly with::

    pytest -m slow tests/test_training_smoke.py
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest


pytestmark = pytest.mark.slow


def test_single_cpu_training_step_writes_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "smoke"

    cmd = [
        sys.executable, "-m", "synthefy_nori.training.cli",
        "--device", "cpu", "--no-mixed-precision", "--no-prefetch", "--no-wandb",
        "--task-type", "reg",
        "--total-steps", "1", "--run-steps", "1", "--save-interval", "1",
        # Tiny architecture so the step is fast on a CPU runner.
        "--embed-dim", "32", "--hid-dim", "64", "--nlayers", "2", "--nhead", "2",
        "--batch-size", "2", "--max-features", "16", "--max-budget", "4000",
        "--checkpoint-dir", str(checkpoint_dir),
    ]

    env = dict(os.environ, WANDB_MODE="disabled")
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=600,
    )

    assert result.returncode == 0, (
        f"training step failed (exit {result.returncode})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
    assert checkpoints, (
        f"no checkpoint written to {checkpoint_dir}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
