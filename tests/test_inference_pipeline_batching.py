"""Conservative batching of independent inference-pipeline model calls."""

from __future__ import annotations

import weakref

import numpy as np
import pytest
import torch

import synthefy_nori.inference.predictor as predictor_module
from synthefy_nori.inference.predictor import NoriPredictor
from synthefy_nori.inference.memory_policy import estimate_cache_gb


class _PlainModel:
    mask_prediction = False
    num_reg_quantiles = 1

    def __init__(self, *, oom_above: int | None = None):
        self.calls = []
        self.oom_above = oom_above

    def to(self, _device):
        return self

    def __call__(self, *, x, y, eval_pos):
        del y
        self.calls.append(x.shape[0])
        if self.oom_above is not None and x.shape[0] > self.oom_above:
            raise torch.cuda.OutOfMemoryError("synthetic batched OOM")
        return x[:, eval_pos:, :1]


class _QuantileModel(_PlainModel):
    """A nonlinear K>1 head so collapse-before-average is observably wrong."""

    num_reg_quantiles = 5

    def __call__(self, *, x, y, eval_pos):
        del y
        self.calls.append(x.shape[0])
        base = x[:, eval_pos:, 0]
        return torch.stack(
            [
                base - 10.0,
                base - 1.0,
                base,
                base + base.square() / 10.0,
                base + base.square(),
            ],
            dim=-1,
        )


class _BatchSqueezingModel(_PlainModel):
    """A custom wrapper that violates the required batched output contract."""

    def __call__(self, *, x, y, eval_pos):
        del y
        self.calls.append(x.shape[0])
        output = x[:, eval_pos:, :1]
        return output[0] if x.shape[0] > 1 else output


class _RNGModel(_PlainModel):
    """Consume RNG by feature width, as feature positional embeddings do."""

    def __call__(self, *, x, y, eval_pos):
        del y
        self.calls.append((x.shape[0], x.shape[-1]))
        torch.rand(x.shape[-1])
        return x[:, eval_pos:, :1]


class _CachedModel(_PlainModel):
    features_per_group = 2
    embed_dim = 8
    nlayers = 1
    nhead = 2

    # Eligibility checks for the split cached API through this historical name.
    forward_cached_regression = object()

    def __init__(self):
        super().__init__()
        self.build_batches = []
        self.apply_batches = []

    def build_context_cache(self, x_train, y_train, **_kwargs):
        self.build_batches.append(x_train.shape[0])
        return x_train, y_train

    def apply_context_cache(self, x_test, context, **_kwargs):
        self.apply_batches.append(x_test.shape[0])
        x_train, _ = context
        return x_test[:, :, :1] + 0.0 * x_train[:, :1, :1]


class _TrackedContext:
    pass


class _LifetimeCachedModel(_CachedModel):
    """Track whether non-retained grouped contexts overlap in memory."""

    def __init__(self):
        super().__init__()
        self.active_contexts = 0
        self.active_before_build = []
        self.build_shapes = []

    def _release_context(self):
        self.active_contexts -= 1

    def build_context_cache(self, x_train, y_train, **_kwargs):
        del y_train
        self.active_before_build.append(self.active_contexts)
        self.build_batches.append(x_train.shape[0])
        self.build_shapes.append(tuple(x_train.shape))
        self.active_contexts += 1
        context = _TrackedContext()
        weakref.finalize(context, self._release_context)
        return context

    def apply_context_cache(self, x_test, context, **_kwargs):
        del context
        self.apply_batches.append(x_test.shape[0])
        return x_test[:, :, :1]


class _FailingCachedApplyModel(_CachedModel):
    """Fail only the grouped apply; the legacy B=1 retry remains healthy."""

    def __init__(self, error_type):
        super().__init__()
        self.error_type = error_type

    def apply_context_cache(self, x_test, context, **_kwargs):
        self.apply_batches.append(x_test.shape[0])
        if x_test.shape[0] > 1:
            raise self.error_type("synthetic grouped-cache failure")
        x_train, _ = context
        return x_test[:, :, :1] + 0.0 * x_train[:, :1, :1]


class _PipeTransform:
    """Give each member distinct values and optionally a distinct feature width."""

    def __init__(self, offset: float, *, drop_last: bool = False):
        self.offset = offset
        self.drop_last = drop_last
        self.categorical_idx = []
        self.fit_calls = 0
        self.transform_calls = 0

    def fit(self, _x, categorical_idx, _seed, **_kwargs):
        self.fit_calls += 1
        self.categorical_idx = list(categorical_idx)
        return self.categorical_idx

    def transform(self, x):
        self.transform_calls += 1
        transformed = x + self.offset
        if self.drop_last:
            transformed = transformed[:, :-1]
        return transformed, self.categorical_idx


def _predictor(model, *, memory_policy="off"):
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.device = torch.device("cuda")
    predictor.model = model
    predictor.model_path = None
    predictor.seed = 7
    predictor.mix_precision = False
    predictor.mask_prediction = False
    predictor.inference_with_DDP = False
    predictor.memory_policy = memory_policy
    predictor.preprocess_pipelines = [[], [], []]
    predictor.inference_config = [
        {"retrieval_config": {"use_retrieval": False}} for _ in predictor.preprocess_pipelines
    ]
    predictor.preprocess_num = 10
    predictor.seeds = [0] * 30
    predictor.min_seq_len_for_category_infer = 100
    predictor.min_unique_num_for_numerical_infer = 4
    predictor.quantile_collapse = "mean"
    predictor._warned_this_call = set()
    predictor._logged_this_call = set()
    return predictor


@pytest.fixture
def fake_cuda_tensors(monkeypatch):
    """Exercise CUDA routing on a CPU-only CI worker without emulating kernels."""
    original_to = torch.Tensor.to

    def keep_cpu(tensor, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", keep_cpu)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def _table(n_train=12, n_test=7, n_features=4):
    values = np.arange((n_train + n_test) * n_features, dtype=np.float32)
    x = values.reshape(n_train + n_test, n_features) / 10.0
    y = np.linspace(-1.0, 1.0, n_train, dtype=np.float32)
    return x[:n_train], y, x[n_train:]


def test_shape_grouping_is_stable_and_does_not_pad():
    def item(id_pipe, features):
        return (
            id_pipe,
            torch.zeros(4, features),
            torch.zeros(4),
            torch.zeros(2, features),
        )

    groups = NoriPredictor._group_ordinary_pipes(
        [
            item(2, 3),
            item(0, 5),
            item(1, 3),
            item(3, 4),
            item(4, 3),
            item(5, 3),
            item(6, 3),
            item(7, 5),
            item(8, 3),
            item(9, 257),
            item(10, 257),
        ]
    )
    assert [[entry[0] for entry in group] for group in groups] == [
        [2, 1, 4, 5],
        [6, 8],
        [0, 7],
        [3],
        [9],
        [10],
    ]


def test_plain_path_batches_and_kill_switch_restores_b1(fake_cuda_tensors, monkeypatch):
    x_train, y_train, x_test = _table()
    batched_model = _PlainModel()
    batched = _predictor(batched_model)._predict_reg_single(x_train, y_train, x_test)
    assert batched_model.calls == [3]

    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    legacy_model = _PlainModel()
    legacy = _predictor(legacy_model)._predict_reg_single(x_train, y_train, x_test)
    assert legacy_model.calls == [1, 1, 1]
    torch.testing.assert_close(batched, legacy, rtol=0, atol=0)


def test_distribution_member_order_and_collapse_match_b1(fake_cuda_tensors, monkeypatch):
    x_train, y_train, x_test = _table()
    offsets = (30.0, 10.0, 60.0, 20.0, 50.0, 40.0)

    def distinct_predictor():
        predictor = _predictor(_QuantileModel())
        # More members than PIPELINE_BATCH_MAX forces execution groups [4, 2].
        # Offsets are deliberately unsorted, so reconstructing results in group
        # order instead of public pipeline-id order is observable.
        predictor.preprocess_pipelines = [[_PipeTransform(offset)] for offset in offsets]
        predictor.inference_config = [{"retrieval_config": {"use_retrieval": False}} for _ in offsets]
        predictor.seeds = [0] * (len(offsets) * predictor.preprocess_num)
        predictor.quantile_collapse = "huber_mean"
        return predictor

    ordered = distinct_predictor()
    members = ordered._try_batched_ordinary_regression(
        ordered._bare_model(),
        x_train_base=x_train,
        x_test_base=x_test,
        y_train=y_train,
        categorical_idx=[],
        n_samples_train=len(x_train),
        n_samples_test=len(x_test),
        budget_n_features=x_train.shape[1],
        max_elements_budget=1_000_000,
        dropped_context_rows=0,
    )
    assert members is not None
    assert ordered.model.calls == [4, 2]
    base = torch.from_numpy(x_test[:, 0])
    for member, offset in zip(members, offsets):
        shifted = base + offset
        expected = torch.stack(
            [
                shifted - 10.0,
                shifted - 1.0,
                shifted,
                shifted + shifted.square() / 10.0,
                shifted + shifted.square(),
            ],
            dim=-1,
        )
        torch.testing.assert_close(member, expected, rtol=0, atol=0)

    expected_bank = torch.stack(members).mean(dim=0)
    expected_point = ordered._collapse_regression_output(expected_bank)
    collapse_first = torch.stack([ordered._collapse_regression_output(member) for member in members]).mean(dim=0)
    assert not torch.allclose(expected_point, collapse_first), (
        "fixture does not distinguish average-before-collapse from the wrong order"
    )

    batched = distinct_predictor()
    batched_bank = batched._predict_reg_single(x_train, y_train, x_test, return_distribution=True)
    batched_point = batched._predict_reg_single(x_train, y_train, x_test)

    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    legacy = distinct_predictor()
    legacy_bank = legacy._predict_reg_single(x_train, y_train, x_test, return_distribution=True)
    legacy_point = legacy._predict_reg_single(x_train, y_train, x_test)

    torch.testing.assert_close(batched_bank, expected_bank, rtol=0, atol=0)
    torch.testing.assert_close(batched_bank, legacy_bank, rtol=0, atol=0)
    torch.testing.assert_close(batched_point, expected_point, rtol=0, atol=0)
    torch.testing.assert_close(batched_point, legacy_point, rtol=0, atol=0)


def test_wide_b1_groups_use_prepared_transforms_once(fake_cuda_tensors):
    x_train, y_train, x_test = _table(n_features=257)
    model = _PlainModel()
    predictor = _predictor(model)
    steps = [_PipeTransform(float(id_pipe)) for id_pipe in range(3)]
    predictor.preprocess_pipelines = [[step] for step in steps]

    output = predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.calls == [1, 1, 1]
    assert [(step.fit_calls, step.transform_calls) for step in steps] == [(1, 2)] * 3
    torch.testing.assert_close(
        output,
        torch.from_numpy(x_test[:, 0]) + 1.0,
        rtol=0,
        atol=5e-5,
    )


def test_mixed_width_groups_preserve_legacy_post_call_rng_state(
    fake_cuda_tensors, monkeypatch
):
    x_train, y_train, x_test = _table()

    def mixed_predictor():
        predictor = _predictor(_RNGModel())
        predictor.preprocess_pipelines = [
            [_PipeTransform(0.0)],
            [_PipeTransform(0.0, drop_last=True)],
            [_PipeTransform(0.0, drop_last=True)],
            [_PipeTransform(0.0)],
        ]
        predictor.inference_config = [
            {"retrieval_config": {"use_retrieval": False}}
            for _ in predictor.preprocess_pipelines
        ]
        predictor.seeds = [0] * (
            len(predictor.preprocess_pipelines) * predictor.preprocess_num
        )
        return predictor

    batched = mixed_predictor()
    batched._predict_reg_single(x_train, y_train, x_test)
    batched_rng_state = torch.get_rng_state()
    assert batched.model.calls == [(2, 3), (2, 4)]

    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    legacy = mixed_predictor()
    legacy._predict_reg_single(x_train, y_train, x_test)
    legacy_rng_state = torch.get_rng_state()
    assert legacy.model.calls == [(1, 4), (1, 3), (1, 3), (1, 4)]
    assert torch.equal(batched_rng_state, legacy_rng_state)


def test_batched_oom_restarts_the_untouched_b1_loop(fake_cuda_tensors):
    x_train, y_train, x_test = _table()
    model = _PlainModel(oom_above=1)
    output = _predictor(model)._predict_reg_single(x_train, y_train, x_test)
    assert model.calls == [3, 1, 1, 1]
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-6)


def test_invalid_batched_output_shape_restarts_b1(fake_cuda_tensors):
    x_train, y_train, x_test = _table()
    model = _BatchSqueezingModel()
    output = _predictor(model)._predict_reg_single(x_train, y_train, x_test)
    assert model.calls == [3, 1, 1, 1]
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)


def test_host_retention_guard_restarts_b1(fake_cuda_tensors, monkeypatch):
    monkeypatch.setattr(predictor_module, "PIPELINE_BATCH_HOST_BUDGET_BYTES", 1)
    x_train, y_train, x_test = _table()
    model = _PlainModel()
    output = _predictor(model)._predict_reg_single(x_train, y_train, x_test)
    assert model.calls == [1, 1, 1]
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)


def test_resident_cache_batches_build_and_apply(fake_cuda_tensors):
    # n_test > the predictor's minimum query chunk (256) makes the cache eligible.
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    model = _CachedModel()
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    first = predictor._predict_reg_single(x_train, y_train, x_test)
    second_x_test = x_test + 0.25
    second = predictor._predict_reg_single(x_train, y_train, second_x_test)
    assert model.build_batches == [3]
    assert model.apply_batches == [3, 3]
    assert predictor.memory_report_["rung"] == "resident_bf16"
    assert predictor.memory_report_["est_cache_gb"] == pytest.approx(
        3
        * estimate_cache_gb(
            n_context_rows=12,
            n_groups=2,
            nlayers=1,
            embed_dim=8,
            bytes_per_element=4,
        )
    )
    assert list(predictor._context_cache) == [("pipeline_batch", (0, 1, 2))]
    torch.testing.assert_close(first, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)
    torch.testing.assert_close(second, torch.from_numpy(second_x_test[:, 0]), rtol=0, atol=1e-5)


def test_nonreused_group_context_is_released_before_next_build(fake_cuda_tensors):
    x_train, y_train, x_test = _table(n_train=12, n_test=300, n_features=5)
    model = _LifetimeCachedModel()
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_280,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
            "reuse_context_cache": False,
        },
    )
    predictor.preprocess_pipelines = [
        [_PipeTransform(0.0)],
        [_PipeTransform(0.0)],
        [_PipeTransform(0.0, drop_last=True)],
        [_PipeTransform(0.0, drop_last=True)],
    ]
    predictor.inference_config = [
        {"retrieval_config": {"use_retrieval": False}}
        for _ in predictor.preprocess_pipelines
    ]
    predictor.seeds = [0] * (
        len(predictor.preprocess_pipelines) * predictor.preprocess_num
    )

    predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.build_shapes == [(2, 12, 5), (2, 12, 4)]
    assert model.active_before_build == [0, 0]
    assert model.active_contexts == 0
    expected_peak_gb = 2 * estimate_cache_gb(
        n_context_rows=12,
        n_groups=3,
        nlayers=1,
        embed_dim=8,
        bytes_per_element=4,
    )
    assert predictor.memory_report_["est_cache_gb"] == pytest.approx(expected_peak_gb)


@pytest.mark.parametrize("transition", ["kill-switch", "int8"])
def test_cache_slots_are_exclusive_across_batched_b1_transitions(fake_cuda_tensors, monkeypatch, transition):
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    model = _CachedModel()
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    predictor._predict_reg_single(x_train, y_train, x_test)
    assert list(predictor._context_cache) == [("pipeline_batch", (0, 1, 2))]

    if transition == "kill-switch":
        monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    else:
        predictor.memory_policy = {
            "elements_budget": 1_072,
            "cache_dtype": "int8",
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        }
    predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.build_batches == [3, 1, 1, 1]
    assert set(predictor._context_cache) == {0, 1, 2}
    assert not any(isinstance(slot, tuple) for slot in predictor._context_cache)

    if transition == "kill-switch":
        monkeypatch.delenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING")
    else:
        predictor.memory_policy = {
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        }
    predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.build_batches == [3, 1, 1, 1, 3]
    assert list(predictor._context_cache) == [("pipeline_batch", (0, 1, 2))]


@pytest.mark.parametrize("error_type", [torch.cuda.OutOfMemoryError, NotImplementedError])
def test_cached_group_failure_clears_slot_and_restarts_b1(fake_cuda_tensors, error_type):
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    model = _FailingCachedApplyModel(error_type)
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    predictor.preprocess_pipelines = [
        [_PipeTransform(0.0, drop_last=True)],
        [_PipeTransform(0.0)],
        [_PipeTransform(0.0)],
    ]
    output = predictor._predict_reg_single(x_train, y_train, x_test)
    assert model.build_batches == [1, 2, 1, 1, 1]
    assert model.apply_batches == [1, 2, 1, 1, 1]
    assert set(predictor._context_cache) == {0, 1, 2}
    assert not any(isinstance(slot, tuple) for slot in predictor._context_cache)
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "excluded",
    [
        "cpu",
        "single-pipe",
        "ddp",
        "predictor-mask",
        "model-mask",
        "retrieval",
        "training",
        "dropout",
    ],
)
def test_execution_mode_exclusions_do_not_enter_batching(fake_cuda_tensors, excluded):
    x_train, y_train, x_test = _table()
    model = _PlainModel()
    predictor = _predictor(model)
    if excluded == "cpu":
        predictor.device = torch.device("cpu")
    elif excluded == "single-pipe":
        predictor.preprocess_pipelines = predictor.preprocess_pipelines[:1]
        predictor.inference_config = predictor.inference_config[:1]
    elif excluded == "ddp":
        predictor.inference_with_DDP = True
    elif excluded == "predictor-mask":
        predictor.mask_prediction = True
    elif excluded == "model-mask":
        model.mask_prediction = True
    elif excluded == "retrieval":
        predictor.inference_config[0]["retrieval_config"]["use_retrieval"] = True
    elif excluded == "training":
        model.training = True
    elif excluded == "dropout":
        model.dropout = 0.1

    result = predictor._try_batched_ordinary_regression(
        predictor._bare_model(),
        x_train_base=x_train,
        x_test_base=x_test,
        y_train=y_train,
        categorical_idx=[],
        n_samples_train=len(x_train),
        n_samples_test=len(x_test),
        budget_n_features=x_train.shape[1],
        max_elements_budget=1_000_000,
        dropped_context_rows=0,
    )
    assert result is None
    assert model.calls == []


@pytest.mark.parametrize(
    "memory_policy",
    [
        pytest.param(
            {
                "elements_budget": 1_072,
                "cache_dtype": "int8",
                "gpu_budget_absolute_gb": 1.0,
                "host_budget_absolute_gb": 2.0,
            },
            id="int8",
        ),
        pytest.param(
            {
                "elements_budget": 1_072,
                "gpu_budget_absolute_gb": 0.0,
                "host_budget_absolute_gb": 2.0,
            },
            id="host-offload",
        ),
        pytest.param(
            {
                "elements_budget": 1_072,
                "context_row_chunk": 4,
                "gpu_budget_absolute_gb": 1.0,
                "host_budget_absolute_gb": 2.0,
            },
            id="pinned-context-row-chunk",
        ),
    ],
)
def test_memory_mode_exclusions_stay_on_the_legacy_loop(fake_cuda_tensors, memory_policy):
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    model = _CachedModel()
    predictor = _predictor(model, memory_policy=memory_policy)
    predictor._predict_reg_single(x_train, y_train, x_test)
    assert model.build_batches == [1, 1, 1]
    assert model.apply_batches == [1, 1, 1]
