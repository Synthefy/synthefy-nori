"""Exercise checkpoint continuation through the public CLI on CPU."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

import torch


def test_cli_resume_matches_uninterrupted_updates(tmp_path):
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        WANDB_MODE="disabled",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    common = [sys.executable, "-m", "synthefy_nori.training.cli", "--device", "cpu"]
    scratch = shlex.split(
        "--no-wandb --no-mixed-precision --fixed-size 64x6 --max-budget 4000 "
        "--total-steps 4 --warmup-steps 10 --save-interval 2 "
        "--embed-dim 32 --hid-dim 64 --nlayers 1 --nhead 2 --batch-size 2 "
        "--prefetch-workers 1 --prefetch-count 2 "
        "--ema-decay 0.9 --seed 73"
    )
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    for options in [
        [*scratch, "--checkpoint-dir", str(uninterrupted)],
        [*scratch, "--run-steps", "2", "--checkpoint-dir", str(resumed)],
        ["--resume", str(resumed / "checkpoint_step_2.pt"), "--run-steps", "2"],
    ]:
        result = subprocess.run([*common, *options], env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    expected = torch.load(uninterrupted / "checkpoint_step_4.pt", weights_only=False, map_location="cpu")
    actual = torch.load(resumed / "checkpoint_step_4.pt", weights_only=False, map_location="cpu")
    for key in ["model_state_dict", "ema_state_dict"]:
        for name, value in expected[key].items():
            torch.testing.assert_close(actual[key][name], value, rtol=0, atol=0, msg=lambda msg: f"{key}.{name}: {msg}")
    assert actual["optimizer_step"] == expected["optimizer_step"] == 4
    assert actual["scheduler_state_dict"] == expected["scheduler_state_dict"]
