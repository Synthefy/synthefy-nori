"""Fast unit tests for synthefy-nori-eval device resolution (no GPU, no downloads)."""

import pytest
import torch

import synthefy_nori.api as api
import synthefy_nori.evaluation.datasets as eval_datasets
import synthefy_nori.evaluation.models as eval_models
import synthefy_nori.evaluation.runner as eval_runner
from synthefy_nori.evaluation import cli


class _FakeDatasetRegistry:
    def __init__(self, **kwargs):
        pass

    def load_tabarena(self, **kwargs):
        pass

    def load_talent(self, **kwargs):
        pass


class _FakeModelRegistry:
    instances = []

    def __init__(self, device):
        self.device = device
        self.checkpoint_devices = []
        _FakeModelRegistry.instances.append(self)

    def add_checkpoint(self, label, path, device=None, reg_config=None):
        self.checkpoint_devices.append(device)


class _FakeEvalRunner:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, sources=None):
        return None


@pytest.fixture
def fake_eval(monkeypatch):
    _FakeModelRegistry.instances = []
    monkeypatch.setattr(eval_datasets, "DatasetRegistry", _FakeDatasetRegistry)
    monkeypatch.setattr(eval_models, "ModelRegistry", _FakeModelRegistry)
    monkeypatch.setattr(eval_runner, "EvalRunner", _FakeEvalRunner)
    return _FakeModelRegistry.instances


def _set_backends(monkeypatch, *, cuda, mps):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1 if cuda else 0)
    monkeypatch.setattr(api, "_mps_available", lambda: mps)


def _run(*extra):
    cli.main(["--checkpoint", "label:/nonexistent.pt", *extra])


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda:0"), (False, True, "mps"), (False, False, "cpu")],
)
def test_default_device_is_autodetected(monkeypatch, fake_eval, cuda, mps, expected):
    _set_backends(monkeypatch, cuda=cuda, mps=mps)
    _run()
    (registry,) = fake_eval
    assert registry.device == expected
    assert registry.checkpoint_devices == [expected]


def test_explicit_device_is_passed_through(monkeypatch, fake_eval):
    _set_backends(monkeypatch, cuda=False, mps=False)
    _run("--device", "cpu")
    (registry,) = fake_eval
    assert registry.device == "cpu"
    assert registry.checkpoint_devices == ["cpu"]


def test_unavailable_cuda_fails_before_any_model_is_registered(monkeypatch, fake_eval):
    _set_backends(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        _run("--device", "cuda:0")
    assert fake_eval == []
