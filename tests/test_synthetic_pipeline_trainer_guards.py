from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from synthefy_nori.training.config import TrainingConfig
from synthefy_nori.training.data_generator import SyntheticDataFilterError
from synthefy_nori.training.prefetch import DataPrefetcher, _ErrorSentinel
from synthefy_nori.training.trainer import ICLFilterExhaustedError, NoriTrainer


class _PerfectRecordingFilter:
    def __init__(self):
        self.calls = []

    def __call__(self, x, y, eval_pos, task_type):
        self.calls.append(
            (x.detach().clone(), y.detach().clone(), eval_pos, task_type)
        )
        if task_type == "reg":
            return {"reg_output": y[:, eval_pos:, None]}

        n_classes = max(int(y.max().item()) + 1, 2)
        logits = torch.nn.functional.one_hot(
            y[:, eval_pos:].long(), num_classes=n_classes
        ).float()
        return {"cls_output": logits * 10.0}


def _bare_filter_trainer(config=None):
    trainer = object.__new__(NoriTrainer)
    trainer.config = config or TrainingConfig(device="cpu", distributed=False)
    trainer.device = torch.device("cpu")
    trainer.rng = np.random.default_rng(123)
    trainer.quality_rules = None
    trainer.is_main = False
    trainer.global_step = 1
    trainer._icl_total_episodes = 0
    trainer._icl_first_round_reject = 0
    trainer._icl_escape_count = 0
    trainer._icl_rounds_used_sum = 0
    trainer._icl_reject_rate_ema = None
    return trainer


def test_icl_filter_uses_exact_training_split_and_prescores_health():
    cfg = TrainingConfig(
        device="cpu",
        distributed=False,
        # A legacy false value must no longer restore the mismatched 70/30 gate.
        icl_filter_use_train_context=False,
    )
    trainer = _bare_filter_trainer(cfg)
    model = _PerfectRecordingFilter()
    trainer._icl_filter_model = model

    rows = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    X = np.broadcast_to(rows[None, :, None], (5, 10, 2)).copy()
    y = np.broadcast_to(rows[None, :], (5, 10)).copy()

    X[0, 0, 1] = np.nan  # ordinary missingness remains valid
    y[1, 0] = np.nan     # non-finite target
    y[2] = 1.0           # constant target
    X[3] = 0.0           # no varying feature
    X[4, 0, 0] = np.inf  # invalid numeric feature value

    passed = trainer._gpu_icl_filter_limix(X, y, context_ratio=0.3)

    assert passed.tolist() == [True, False, False, False, False]
    assert len(model.calls) == 1
    x_seen, _y_seen, eval_pos, task_type = model.calls[0]
    assert x_seen.shape[0] == 1  # unhealthy episodes never reach the model
    assert eval_pos == 3         # same floor(N * ratio) used by _prepare_batch
    assert task_type == "reg"


@pytest.mark.parametrize("task_type", ["reg", "cls"])
def test_icl_filter_propagates_public_task_type(task_type):
    trainer = _bare_filter_trainer()
    model = _PerfectRecordingFilter()
    trainer._icl_filter_model = model

    rows = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    X = np.broadcast_to(rows[None, :, None], (2, 10, 2)).copy()
    if task_type == "reg":
        y = np.broadcast_to(rows[None, :], (2, 10)).copy()
        n_classes = None
    else:
        labels = np.arange(10, dtype=np.float32) % 2
        y = np.broadcast_to(labels[None, :], (2, 10)).copy()
        n_classes = 2

    passed = trainer._gpu_icl_filter_limix(
        X,
        y,
        task_type=task_type,
        n_classes=n_classes,
        context_ratio=0.5,
    )

    assert passed.tolist() == [True, True]
    assert model.calls[0][3] == task_type

def test_synchronous_filter_fails_instead_of_returning_rejected_episode(monkeypatch):
    cfg = TrainingConfig(
        batch_size=1,
        device="cpu",
        distributed=False,
        icl_filter_max_rounds=1,
    )
    trainer = _bare_filter_trainer(cfg)
    trainer._gpu_icl_filter = types.MethodType(
        lambda self, X, y, task_type='reg', n_classes=None,
        context_ratio=None: np.zeros(X.shape[0], dtype=bool),
        trainer,
    )
    X = np.zeros((1, 8, 2), dtype=np.float32)
    y = np.linspace(-1.0, 1.0, 8, dtype=np.float32)[None, :]

    monkeypatch.setattr(
        "synthefy_nori.training.trainer.generate_batch",
        lambda **kwargs: (X.copy(), y.copy(), None),
    )

    with pytest.raises(RuntimeError, match="replacement budget"):
        trainer._filter_and_replace(
            X.copy(), y.copy(), n_samples=8, n_features=2, context_ratio=0.5
        )

    assert trainer._icl_escape_count == 1


def test_generator_scale_variation_is_disabled_after_context_normalization():
    cfg = TrainingConfig(
        device="cpu",
        distributed=False,
        scale_variation=True,  # legacy manifests may still contain this value
    )
    trainer = _bare_filter_trainer(cfg)

    kwargs = trainer._build_gen_kwargs(
        128, 8, "reg", None, context_ratio=0.4
    )

    assert kwargs["scale_variation"] is False
    assert kwargs["context_rows"] == 51


def test_effective_shape_support_exposes_reachable_bucket_endpoints():
    cfg = TrainingConfig(
        min_samples=50,
        max_samples=2000,
        min_features=2,
        max_features=250,
        max_sample_feature_budget=250_000,
    )

    support = NoriTrainer._resolve_effective_shape_support(cfg)

    assert (support["min_samples"], support["max_samples"]) == (64, 1536)
    assert (support["min_features"], support["max_features"]) == (4, 192)
    assert len(support["shapes"]) == 62
    assert all(
        rows * features <= cfg.max_sample_feature_budget
        for rows, features in support["shapes"]
    )


def test_random_shape_range_without_a_bucket_fails_clearly():
    cfg = TrainingConfig(max_samples=63)

    with pytest.raises(ValueError, match="contains no supported bucket"):
        NoriTrainer._resolve_effective_shape_support(cfg)


def test_prefetcher_preserves_public_three_value_batch_contract():
    prefetcher = DataPrefetcher(num_workers=1, prefetch_count=1)
    X = np.zeros((1, 8, 2), dtype=np.float32)
    y = np.zeros((1, 8), dtype=np.float32)
    expected = (X, y, 3)
    prefetcher._started = True
    prefetcher._pending_ids.append(0)
    prefetcher._results_cache[0] = expected

    try:
        result = prefetcher.get()
    finally:
        # This test uses the in-memory result cache; no process was spawned.
        prefetcher._started = False

    assert result is expected


@pytest.mark.parametrize(
    ("err_type", "expected_type"),
    [
        ("ICLFilterExhaustedError", ICLFilterExhaustedError),
        ("SyntheticDataFilterError", SyntheticDataFilterError),
        ("TypeError", RuntimeError),
    ],
)
def test_prefetch_worker_error_preserves_filter_outcomes(err_type, expected_type):
    sentinel = _ErrorSentinel(err_type, "worker failed", "worker traceback")

    error = NoriTrainer._prefetch_worker_exception(sentinel)

    assert isinstance(error, expected_type)
    assert "worker failed" in str(error)


def test_ddp_step_status_synchronizes_remote_skip_and_fatal(monkeypatch):
    trainer = object.__new__(NoriTrainer)
    trainer.config = TrainingConfig(
        device="cpu",
        distributed=True,
        world_size=2,
        local_rank=0,
    )
    trainer.device = torch.device("cpu")

    local_statuses = []

    def fake_all_reduce(status, op):
        assert op == torch.distributed.ReduceOp.MAX
        local_statuses.append(status.clone())
        status.copy_(torch.tensor([1, 1], dtype=status.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    should_skip, fatal_any = trainer._ddp_step_status(False, None)

    assert local_statuses[0].tolist() == [0, 0]
    assert should_skip is True
    assert fatal_any is True


def test_single_process_fatal_error_keeps_original_type():
    trainer = object.__new__(NoriTrainer)
    trainer.config = TrainingConfig(device="cpu", distributed=False)

    with pytest.raises(TypeError, match="programming defect"):
        trainer._raise_synchronized_step_error(
            TypeError("programming defect"), "data preparation"
        )
