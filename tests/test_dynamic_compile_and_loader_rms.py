"""Dynamic regional compilation, and native RMSNorm at the loader choke point.

`dynamic` exists because `static` can only be enabled together with a shape
palette, and the palette changes the training curriculum. Dynamic compiles
shape-generic kernels, so it needs no palette and leaves the curriculum alone.

`load_model` is where native RMSNorm is turned on for inference: every
inference and evaluation path reaches the model through it, while training
builds via `build_model` and is therefore unaffected.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

import torch
import torch.nn as nn

from synthefy_nori.model.layer import RMSNorm


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "synthefy_nori.training.cli", *args],
        capture_output=True, text=True, timeout=180,
    )


@pytest.mark.slow
def test_dynamic_is_an_accepted_choice():
    """Rejecting a bogus value makes argparse print the full choice list."""
    result = _run_cli("--compile-encoder-layers", "definitely-not-a-mode")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "dynamic" in combined, combined[-2000:]
    assert "static" in combined


@pytest.mark.slow
def test_static_still_requires_a_bounded_shape_contract():
    """Unbounded shapes with dynamic=False would blow the Dynamo cache limit
    and silently degrade to eager, so the CLI must keep rejecting it."""
    result = _run_cli("--compile-encoder-layers", "static")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "requires --shape-palette" in combined, combined[-2000:]


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = RMSNorm((4,))
        self.b = nn.Sequential(RMSNorm((4,)), nn.Linear(4, 4))


def _patch_loader(monkeypatch, model):
    import synthefy_nori.utils.loading as loading

    monkeypatch.setattr(loading, "_safe_torch_load",
                        lambda *a, **k: {"config": {}, "state_dict": {}})
    monkeypatch.setattr(loading, "build_model", lambda cfg: model)
    monkeypatch.setattr(model, "load_state_dict", lambda *a, **k: None)
    return loading


def test_load_model_enables_native_rms_by_default(monkeypatch):
    model = _TinyNet()
    loading = _patch_loader(monkeypatch, model)

    loading.load_model("ignored.ckpt")

    assert model.a.use_native is True
    assert model.b[0].use_native is True


def test_load_model_can_opt_out(monkeypatch):
    model = _TinyNet()
    loading = _patch_loader(monkeypatch, model)

    loading.load_model("ignored.ckpt", native_rms_norm=False)

    assert model.a.use_native is False
    assert model.b[0].use_native is False


def test_native_and_decomposed_rms_agree_closely():
    """Same algorithm, different kernel: agreement to well inside bf16 ulp."""
    torch.manual_seed(0)
    x = torch.randn(8, 32, 64)

    norm = RMSNorm((64,))
    decomposed = norm(x)
    norm.use_native = True
    native = norm(x)

    assert torch.allclose(decomposed, native, rtol=1e-5, atol=1e-6)
