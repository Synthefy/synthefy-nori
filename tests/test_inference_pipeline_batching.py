"""Conservative batching of independent inference-pipeline model calls."""

from __future__ import annotations

import logging
import warnings
import weakref

import numpy as np
import pytest
import torch

import synthefy_nori.inference.predictor as predictor_module
from synthefy_nori.inference.predictor import (
    EXACT_CACHED_CUDAGRAPHS_ENV,
    NoriPredictor,
)
from synthefy_nori.inference.degradation import CacheQuantizedWarning
from synthefy_nori.inference.memory_policy import MemoryPolicy, estimate_cache_gb


class _PlainModel:
    mask_prediction = False
    num_reg_quantiles = 1

    def __init__(self, *, oom_above: int | None = None):
        self.calls = []
        self.oom_above = oom_above

    def to(self, _device):
        return self

    def __call__(self, *, x, y, eval_pos, task_type="reg"):
        del y
        self.calls.append(x.shape[0])
        if self.oom_above is not None and x.shape[0] > self.oom_above:
            raise torch.cuda.OutOfMemoryError("synthetic batched OOM")
        return x[:, eval_pos:, :1]


class _QuantileModel(_PlainModel):
    """A nonlinear K>1 head so collapse-before-average is observably wrong."""

    num_reg_quantiles = 5

    def __call__(self, *, x, y, eval_pos, task_type="reg"):
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

    def __call__(self, *, x, y, eval_pos, task_type="reg"):
        del y
        self.calls.append(x.shape[0])
        output = x[:, eval_pos:, :1]
        return output[0] if x.shape[0] > 1 else output


class _RNGModel(_PlainModel):
    """Consume RNG by feature width, as feature positional embeddings do."""

    def __call__(self, *, x, y, eval_pos, task_type="reg"):
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


class _RetryingCachedModel(_CachedModel):
    """Synthetic cache builder that succeeds only on one exact fallback rung."""

    def __init__(self, success_at):
        super().__init__()
        self.success_at = success_at
        self.cache_attempts = []

    def build_context_cache(self, x_train, y_train, *, cache_dtype, offload_kv_cache, fit_row_chunk):
        attempt = (cache_dtype, offload_kv_cache, fit_row_chunk)
        self.cache_attempts.append(attempt)
        if attempt != self.success_at:
            raise torch.cuda.OutOfMemoryError("synthetic cache-build OOM")
        return x_train, y_train


class _AdaptiveDecodeModel(_CachedModel):
    """Emit the model's adaptive query-chunk events without requiring CUDA."""

    def __init__(self, succeed_at: int | None):
        super().__init__()
        self.succeed_at = succeed_at

    def apply_context_cache(
        self,
        x_test,
        context,
        *,
        row_chunk_size,
        query_chunk_attempt_callback,
        **_kwargs,
    ):
        chunk = row_chunk_size
        while chunk != self.succeed_at:
            query_chunk_attempt_callback(chunk, "oom")
            if chunk <= 1:
                raise torch.cuda.OutOfMemoryError("synthetic exhausted adaptive decode")
            chunk = max(1, chunk // 2)
        query_chunk_attempt_callback(chunk, "success")
        x_train, _ = context
        return x_test[:, :, :1] + 0.0 * x_train[:, :1, :1]


class _RowChunkUnsupportedModel(_CachedModel):
    """Raise NotImplementedError whenever asked to row-chunk the build, as a
    checkpoint with serial sequence attention would."""

    def build_context_cache(self, x_train, y_train, *, cache_dtype, offload_kv_cache, fit_row_chunk):
        if fit_row_chunk is not None:
            raise NotImplementedError("context_row_chunk is not supported for serial sequence attention")
        self.build_batches.append(x_train.shape[0])
        return x_train, y_train


class _CountedOOMCachedModel(_CachedModel):
    """Raise a synthetic OOM on exactly the Nth build_context_cache call; every
    other call succeeds. Used to make one specific attempt -- and only that
    attempt -- fail, regardless of which pipeline or which rung it belongs to.
    """

    def __init__(self, oom_on_call):
        super().__init__()
        self.oom_on_call = oom_on_call
        self.build_calls = 0

    def build_context_cache(self, x_train, y_train, **_kwargs):
        self.build_calls += 1
        if self.build_calls == self.oom_on_call:
            raise torch.cuda.OutOfMemoryError("synthetic cross-pipe OOM")
        self.build_batches.append(x_train.shape[0])
        return x_train, y_train


class _StreamingRetryModel(_CachedModel):
    """Record the bounded host-streaming ladder and fail before one rung."""

    def __init__(self, success_at=None, *, error_type=torch.cuda.OutOfMemoryError):
        super().__init__()
        self.success_at = success_at
        self.error_type = error_type
        self.cache_attempts = []
        self.train_rows = []
        self.build_devices = []
        self.apply_devices = []
        self.apply_kwargs = []

    def build_context_cache(
        self,
        x_train,
        y_train,
        *,
        cache_dtype,
        offload_kv_cache,
        fit_row_chunk,
        stream_context,
    ):
        attempt = (cache_dtype, offload_kv_cache, fit_row_chunk, stream_context)
        self.cache_attempts.append(attempt)
        self.train_rows.append(x_train.shape[1])
        self.build_devices.append((x_train.device.type, y_train.device.type))
        if attempt != self.success_at:
            raise self.error_type("synthetic streamed-cache failure")
        return x_train, y_train

    def apply_context_cache(self, x_test, context, **kwargs):
        self.apply_devices.append(x_test.device.type)
        self.apply_kwargs.append(kwargs)
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


def test_exact_cached_cudagraph_mode_preserves_b1_execution(
    fake_cuda_tensors,
    monkeypatch,
):
    monkeypatch.setenv(EXACT_CACHED_CUDAGRAPHS_ENV, "1")
    x_train, y_train, x_test = _table()
    model = _PlainModel()
    predictor = _predictor(model)
    predictor._predict_reg_single(x_train, y_train, x_test)
    assert model.calls == [1, 1, 1]


def test_exact_cached_cudagraphs_compile_one_unbound_layer_method(monkeypatch):
    class Layer:
        def forward_test_with_cache(self, x_test, cache, feature_atten_mask=None):
            del self, cache, feature_atten_mask
            return x_test

    class Encoder:
        layers = [Layer(), Layer(), Layer()]

    class Model:
        transformer_encoder = Encoder()

    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.device = torch.device("cuda")
    predictor._logged_this_call = set()
    monkeypatch.setenv(EXACT_CACHED_CUDAGRAPHS_ENV, "1")
    monkeypatch.setattr(
        predictor_module.importlib.util,
        "find_spec",
        lambda _name: object(),
    )
    compile_calls = []

    def compile_once(function, **kwargs):
        compile_calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", compile_once)
    model = Model()
    assert predictor._maybe_enable_exact_cached_cudagraphs(model)
    assert predictor._maybe_enable_exact_cached_cudagraphs(model)
    assert len(compile_calls) == 1
    assert compile_calls[0][1] == {
        "backend": "cudagraphs",
        "dynamic": False,
        "fullgraph": False,
    }
    for layer in model.transformer_encoder.layers:
        assert layer.forward_test_with_cache("value", {}) == "value"

    predictor._disable_exact_cached_cudagraphs(model)
    assert not model._exact_cudagraphs_enabled


def test_exact_cached_cudagraphs_fall_back_without_setuptools(monkeypatch):
    class Layer:
        def forward_test_with_cache(self, x_test, cache):
            del self, cache
            return x_test

    class Encoder:
        layers = [Layer()]

    class Model:
        transformer_encoder = Encoder()

    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.device = torch.device("cuda")
    predictor._logged_this_call = set()
    monkeypatch.setenv(EXACT_CACHED_CUDAGRAPHS_ENV, "1")
    monkeypatch.setattr(
        predictor_module.importlib.util,
        "find_spec",
        lambda _name: None,
    )
    monkeypatch.setattr(
        torch,
        "compile",
        lambda *_args, **_kwargs: pytest.fail("torch.compile must not run"),
    )

    model = Model()
    assert not predictor._maybe_enable_exact_cached_cudagraphs(model)
    assert model.transformer_encoder.layers[0].forward_test_with_cache("value", {}) == "value"


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


def test_mixed_width_groups_preserve_legacy_post_call_rng_state(fake_cuda_tensors, monkeypatch):
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
            {"retrieval_config": {"use_retrieval": False}} for _ in predictor.preprocess_pipelines
        ]
        predictor.seeds = [0] * (len(predictor.preprocess_pipelines) * predictor.preprocess_num)
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


def test_worst_pipeline_report_and_history_survive_later_resident_success():
    predictor = _predictor(_PlainModel())
    predictor._memory_attempt_history = []
    predictor._memory_outcome_policy = None
    requested = MemoryPolicy(gpu_budget_absolute_gb=1.0, host_budget_absolute_gb=2.0)
    resident = requested.resolve(
        est_cache_gb=0.1,
        bytes_per_element=2,
        head_dim=4,
        total_vram_gb=8.0,
        total_ram_gb=16.0,
    )
    plain = resident.escalated("plain_loop")

    predictor._publish_memory_policy(plain)
    predictor._record_memory_attempt(
        plain,
        pipeline_ids=[0],
        path="plain_loop",
        rung="plain_loop",
        context_row_chunk=None,
        outcome="success",
        reason="fallback_after_oom",
        dropped_context_rows=0,
    )
    predictor._publish_memory_policy(resident)
    predictor._record_memory_attempt(
        resident,
        pipeline_ids=[1],
        path="cached",
        rung="resident_bf16",
        context_row_chunk=None,
        outcome="success",
        reason="resolved",
        dropped_context_rows=0,
    )

    assert predictor.memory_report_["rung"] == "plain_loop"
    assert [attempt["pipeline_ids"] for attempt in predictor.memory_report_["attempt_history"]] == [[0], [1]]


def test_explicit_fit_row_chunk_is_a_retry_cap():
    assert NoriPredictor._fit_row_chunk_attempts(None) == [2048, 1024, 512, 256]
    assert NoriPredictor._fit_row_chunk_attempts(16384) == [2048, 1024, 512, 256]
    assert NoriPredictor._fit_row_chunk_attempts(4096) == [2048, 1024, 512, 256]
    assert NoriPredictor._fit_row_chunk_attempts(1024) == [512, 256]
    assert NoriPredictor._fit_row_chunk_attempts(512) == [256]
    assert NoriPredictor._fit_row_chunk_attempts(256) == []


def test_apply_oom_releases_retained_context_and_output_before_retry(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")

    class Context:
        def __init__(self, x_train):
            self.x_train = x_train

    class ApplyOOMModel(_CachedModel):
        def __init__(self):
            super().__init__()
            self.cache_attempts = []
            self.context_refs = []

        def build_context_cache(self, x_train, y_train, *, cache_dtype, offload_kv_cache, fit_row_chunk):
            del y_train
            self.cache_attempts.append((cache_dtype, offload_kv_cache, fit_row_chunk))
            context = Context(x_train)
            self.context_refs.append(weakref.ref(context))
            return context

        def apply_context_cache(self, x_test, context, **_kwargs):
            return x_test[:, :, :1] + 0.0 * context.x_train[:, :1, :1]

    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    model = ApplyOOMModel()
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    predictor.preprocess_pipelines = [[]]
    predictor.inference_config = [{}]
    real_unwrap = predictor._unwrap_model_output
    output_refs = []

    def oom_after_apply(value, *args, **kwargs):
        if not output_refs:
            output_refs.append(weakref.ref(value))
            del value
            raise torch.cuda.OutOfMemoryError("synthetic post-apply OOM")
        return real_unwrap(value, *args, **kwargs)

    predictor._unwrap_model_output = oom_after_apply
    cleanup_checks = []

    def assert_released_before_empty_cache():
        cleanup_checks.append(True)
        assert predictor._context_cache == {}
        assert model.context_refs[0]() is None
        assert output_refs[0]() is None

    monkeypatch.setattr(torch.cuda, "empty_cache", assert_released_before_empty_cache)
    output = predictor._predict_reg_single(x_train, y_train, x_test)

    assert cleanup_checks == [True]
    assert model.cache_attempts[:2] == [
        ("bf16", False, None),
        ("int8", False, None),
    ]
    assert [attempt["outcome"] for attempt in predictor.memory_report_["attempt_history"]] == ["oom", "success"]
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)


def test_runtime_oom_walks_every_allowed_rung_then_bounded_fit_chunks(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    model = _RetryingCachedModel(("int8", True, 512))
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    predictor.preprocess_pipelines = [[]]
    predictor.inference_config = [{}]

    output = predictor._predict_reg_single(x_train, y_train, x_test)

    expected = [
        ("bf16", False, None),
        ("int8", False, None),
        ("bf16", True, None),
        ("int8", True, None),
        ("int8", True, 2048),
        ("int8", True, 1024),
        ("int8", True, 512),
    ]
    assert model.cache_attempts == expected
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)
    report = predictor.memory_report_
    assert report["rung"] == "context_row_chunk"
    assert report["context_row_chunk"] == 512
    history = report["attempt_history"]
    assert [attempt["outcome"] for attempt in history] == ["oom"] * 6 + ["success"]
    assert [
        (attempt["cache_dtype"], attempt["offload_to_host"], attempt["context_row_chunk"]) for attempt in history
    ] == expected
    assert all(attempt["dropped_context_rows"] == 0 for attempt in history)


def test_oom_on_one_pipe_does_not_evict_a_sibling_pipes_cache(fake_cuda_tensors, monkeypatch):
    """An OOM while building pipe 1's context cache must evict only pipe 1's own
    (failed) cache entry -- never pipe 0's already-built, reusable one.

    Regression coverage for the _evict_pipe_cache scope fix: the pre-fix code
    called cache.clear() on the whole shared _context_cache dict from every
    OOM/unsupported handler in this loop, which would have wiped pipe 0's entry
    here even though pipe 0 never touched the failing build.
    """
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    # Call 1: pipe 0's only attempt (bf16) -> succeeds, gets cached under id_pipe=0.
    # Call 2: pipe 1's first attempt (bf16) -> OOMs.
    # Call 3: pipe 1's retry (int8) -> succeeds, gets cached under id_pipe=1.
    model = _CountedOOMCachedModel(oom_on_call=2)
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    predictor.preprocess_pipelines = [[], []]
    predictor.inference_config = [{}, {}]

    predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.build_calls == 3
    assert set(predictor._context_cache) == {0, 1}
    assert len(predictor._context_cache[0]) == 1
    assert len(predictor._context_cache[1]) == 1
    assert [attempt["outcome"] for attempt in predictor.memory_report_["attempt_history"]] == [
        "success",
        "oom",
        "success",
    ]


def test_pinned_context_row_chunk_on_unsupported_checkpoint_raises(fake_cuda_tensors, monkeypatch):
    """A caller who explicitly pins context_row_chunk on a checkpoint that cannot
    honor it must see the NotImplementedError, not a silent fallback to the plain
    loop -- only a chunk WE chose ourselves as an OOM escalation may degrade
    quietly. See the "if pinned is not None: raise" branch this pins.
    """
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    model = _RowChunkUnsupportedModel()
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
            "context_row_chunk": 2048,
        },
    )
    predictor.preprocess_pipelines = [[]]
    predictor.inference_config = [{}]

    with pytest.raises(NotImplementedError, match="serial sequence attention"):
        predictor._predict_reg_single(x_train, y_train, x_test)


def test_precision_only_recovery_logs_regardless_of_chunk_change(fake_cuda_tensors, monkeypatch, caplog):
    """A recovery via precision/placement alone (no context_row_chunk change) must
    still log "Nori recovered on rung", not only chunk-based recoveries.
    """
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    # Attempt 1 (bf16, unoffloaded, no chunk) OOMs; attempt 2 (int8, unoffloaded,
    # no chunk) succeeds -- a pure precision recovery, attempt_fit_chunk is None
    # throughout.
    model = _RetryingCachedModel(("int8", False, None))
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    predictor.preprocess_pipelines = [[]]
    predictor.inference_config = [{}]

    with caplog.at_level(logging.WARNING, logger="synthefy_nori.inference.predictor"):
        predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.cache_attempts == [("bf16", False, None), ("int8", False, None)]
    assert any("Nori recovered on rung" in record.message for record in caplog.records)
    assert predictor.memory_report_["rung"] == "resident_int8"


def test_int8_cache_warns_but_bf16_cache_does_not(fake_cuda_tensors, monkeypatch):
    """Landing on an int8 rung must raise CacheQuantizedWarning (a real, catchable
    fidelity-loss signal for strict_pipeline()/scored callers), not just a log
    line -- and a bit-exact bf16 rung must NOT raise it.
    """
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=300)

    int8_model = _RetryingCachedModel(("int8", False, None))
    int8_predictor = _predictor(
        int8_model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    int8_predictor.preprocess_pipelines = [[]]
    int8_predictor.inference_config = [{}]
    with pytest.warns(CacheQuantizedWarning):
        int8_predictor._predict_reg_single(x_train, y_train, x_test)

    bf16_model = _CachedModel()
    bf16_predictor = _predictor(
        bf16_model,
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    bf16_predictor.preprocess_pipelines = [[]]
    bf16_predictor.inference_config = [{}]
    with warnings.catch_warnings():
        warnings.simplefilter("error", CacheQuantizedWarning)
        bf16_predictor._predict_reg_single(x_train, y_train, x_test)
    assert bf16_predictor.memory_report_["rung"] == "resident_bf16"


def test_resident_int8_row_chunk_uses_private_hybrid_without_changing_report(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")

    class HybridModel(_CachedModel):
        def __init__(self):
            super().__init__()
            self.builds = []

        def build_context_cache(
            self,
            x_train,
            y_train,
            *,
            cache_dtype,
            offload_kv_cache,
            fit_row_chunk,
            stream_context=False,
            _hybrid_resident_int8_prefill=False,
        ):
            self.builds.append(
                {
                    "cache_dtype": cache_dtype,
                    "offload": offload_kv_cache,
                    "fit_row_chunk": fit_row_chunk,
                    "stream_context": stream_context,
                    "hybrid": _hybrid_resident_int8_prefill,
                    "devices": (x_train.device.type, y_train.device.type),
                }
            )
            return x_train, y_train

    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    model = HybridModel()
    predictor = _predictor(
        model,
        memory_policy={
            "elements_budget": 1_072,
            "cache_dtype": "int8",
            "context_row_chunk": 4,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    predictor.preprocess_pipelines = [[]]
    predictor.inference_config = [{}]

    output = predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.builds == [
        {
            "cache_dtype": "int8",
            "offload": False,
            "fit_row_chunk": 4,
            "stream_context": False,
            "hybrid": True,
            "devices": ("cpu", "cpu"),
        }
    ]
    report = predictor.memory_report_
    assert report["rung"] == "resident_int8"
    assert report["stream_context"] is False
    assert report["context_row_chunk"] == 4
    assert report["attempt_history"][0]["rung"] == "resident_int8"
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)


def _streaming_predictor(model, **policy_overrides):
    policy = {
        "stream_context": True,
        "elements_budget": 8,
        "allow_subsample": False,
        "gpu_budget_absolute_gb": 0.0,
        "host_budget_absolute_gb": 2.0,
    }
    policy.update(policy_overrides)
    predictor = _predictor(model, memory_policy=policy)
    predictor.preprocess_pipelines = [[]]
    predictor.inference_config = [{}]
    predictor.seeds = [0] * predictor.preprocess_num
    return predictor


def test_stream_context_keeps_all_rows_on_cpu_and_runs_for_one_query(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=1)
    model = _StreamingRetryModel(("bf16", True, 8192, True))
    predictor = _streaming_predictor(model, allow_subsample=True)
    monkeypatch.setattr(predictor, "_total_vram_gb", lambda: 64.0)
    monkeypatch.setattr(predictor, "_resolve_max_elements_budget", lambda: 8)

    output = predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.cache_attempts == [("bf16", True, 8192, True)]
    assert model.train_rows == [len(x_train)]
    assert model.build_devices == [("cpu", "cpu")]
    assert model.apply_devices == ["cpu"]
    assert model.calls == []
    assert len(model.apply_kwargs) == 1
    assert callable(model.apply_kwargs[0].pop("query_chunk_attempt_callback"))
    assert model.apply_kwargs == [
        {
            "row_chunk_size": 1,
            "adaptive_query_chunk": True,
        }
    ]
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)
    report = predictor.memory_report_
    assert report["rung"] == "stream_bf16"
    assert report["context_row_chunk"] == 8192
    assert report["query_chunk"] == 1
    assert report["dropped_context_rows"] == 0
    assert report["attempt_history"][0]["outcome"] == "success"


def test_adaptive_decode_halving_is_reported_with_effective_query_chunk(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    predictor = _predictor(
        _AdaptiveDecodeModel(succeed_at=128),
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )
    predictor.preprocess_pipelines = [[]]
    predictor.inference_config = [{}]

    output = predictor._predict_reg_single(x_train, y_train, x_test)

    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)
    report = predictor.memory_report_
    assert report["query_chunk"] == 128
    assert [
        (attempt["query_chunk"], attempt["outcome"], attempt["reason"]) for attempt in report["attempt_history"]
    ] == [
        (256, "oom", "resolved"),
        (128, "success", "oom_retry"),
    ]


def test_pipeline_batched_adaptive_decode_reports_the_same_trace(
    fake_cuda_tensors,
):
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    predictor = _predictor(
        _AdaptiveDecodeModel(succeed_at=128),
        memory_policy={
            "elements_budget": 1_072,
            "gpu_budget_absolute_gb": 1.0,
            "host_budget_absolute_gb": 2.0,
        },
    )

    predictor._predict_reg_single(x_train, y_train, x_test)

    report = predictor.memory_report_
    assert report["query_chunk"] == 128
    assert [
        (
            attempt["path"],
            attempt["pipeline_ids"],
            attempt["query_chunk"],
            attempt["outcome"],
        )
        for attempt in report["attempt_history"]
    ] == [
        ("pipeline_batch", [0, 1, 2], 256, "oom"),
        ("pipeline_batch", [0, 1, 2], 128, "success"),
    ]


def test_exhausted_adaptive_decode_reports_every_halving_and_reraises(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=300)
    predictor = _streaming_predictor(
        _AdaptiveDecodeModel(succeed_at=None),
        cache_dtype="int8",
        context_row_chunk=256,
    )
    monkeypatch.setattr(predictor, "_total_vram_gb", lambda: 64.0)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="exhausted adaptive decode"):
        predictor._predict_reg_single(x_train, y_train, x_test)

    report = predictor.memory_report_
    assert report["query_chunk"] == 1
    assert [attempt["query_chunk"] for attempt in report["attempt_history"]] == [256, 128, 64, 32, 16, 8, 4, 2, 1]
    assert all(attempt["outcome"] == "oom" for attempt in report["attempt_history"])
    assert all(
        attempt["path"] == "cached" and attempt["rung"] == "stream_int8" for attempt in report["attempt_history"]
    )


def test_stream_context_uses_shared_precision_then_row_chunk_ladder(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=1)
    model = _StreamingRetryModel(("int8", True, 256, True))
    predictor = _streaming_predictor(model)
    monkeypatch.setattr(predictor, "_total_vram_gb", lambda: 64.0)

    output = predictor._predict_reg_single(x_train, y_train, x_test)

    expected = [
        ("bf16", True, 8192, True),
        ("int8", True, 8192, True),
        ("int8", True, 2048, True),
        ("int8", True, 1024, True),
        ("int8", True, 512, True),
        ("int8", True, 256, True),
    ]
    assert model.cache_attempts == expected
    assert model.train_rows == [len(x_train)] * len(expected)
    assert model.calls == []
    torch.testing.assert_close(output, torch.from_numpy(x_test[:, 0]), rtol=0, atol=1e-5)
    report = predictor.memory_report_
    assert report["rung"] == "stream_int8"
    assert report["context_row_chunk"] == 256
    assert report["query_chunk"] == 1
    assert [attempt["outcome"] for attempt in report["attempt_history"]] == [
        "oom",
        "oom",
        "oom",
        "oom",
        "oom",
        "success",
    ]
    assert [
        (
            attempt["cache_dtype"],
            attempt["offload_to_host"],
            attempt["context_row_chunk"],
        )
        for attempt in report["attempt_history"]
    ] == [attempt[:3] for attempt in expected]
    assert all(attempt["path"] == "cached" and attempt["rung"] != "plain_loop" for attempt in report["attempt_history"])
    assert all(attempt["dropped_context_rows"] == 0 for attempt in report["attempt_history"])


def test_stream_context_exhaustion_reraises_without_plain_loop(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=1)
    model = _StreamingRetryModel()
    predictor = _streaming_predictor(model)
    monkeypatch.setattr(predictor, "_total_vram_gb", lambda: 64.0)

    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic streamed-cache failure"):
        predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.cache_attempts == [
        ("bf16", True, 8192, True),
        ("int8", True, 8192, True),
        ("int8", True, 2048, True),
        ("int8", True, 1024, True),
        ("int8", True, 512, True),
        ("int8", True, 256, True),
    ]
    assert model.calls == []
    history = predictor.memory_report_["attempt_history"]
    assert [attempt["outcome"] for attempt in history] == ["oom"] * 6
    assert all(attempt["path"] == "cached" for attempt in history)
    assert all(attempt["rung"] != "plain_loop" for attempt in history)


def test_stream_context_unsupported_reraises_without_retry_or_plain_loop(fake_cuda_tensors, monkeypatch):
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    x_train, y_train, x_test = _table(n_train=12, n_test=1)
    model = _StreamingRetryModel(error_type=NotImplementedError)
    predictor = _streaming_predictor(model, context_row_chunk=256)

    with pytest.raises(NotImplementedError, match="synthetic streamed-cache failure"):
        predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.cache_attempts == [("bf16", True, 256, True)]
    assert model.calls == []
    history = predictor.memory_report_["attempt_history"]
    assert [attempt["outcome"] for attempt in history] == ["unsupported"]
    assert history[0]["context_row_chunk"] == 256


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SYNTHEFY_DISABLE_CACHED_INFERENCE", "1"),
        ("SYNTHEFY_ENABLE_CACHED_INFERENCE", "0"),
    ],
)
def test_stream_context_cache_kill_switch_fails_before_rows_or_model_call(fake_cuda_tensors, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    x_train, y_train, x_test = _table(n_train=12, n_test=1)
    model = _StreamingRetryModel(("bf16", True, 2048, True))
    predictor = _streaming_predictor(model)

    with pytest.raises(RuntimeError, match="stream_context=True cannot run"):
        predictor._predict_reg_single(x_train, y_train, x_test)

    assert model.cache_attempts == []
    assert model.train_rows == []
    assert model.calls == []
    assert predictor.memory_report_ is None


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
        {"retrieval_config": {"use_retrieval": False}} for _ in predictor.preprocess_pipelines
    ]
    predictor.seeds = [0] * (len(predictor.preprocess_pipelines) * predictor.preprocess_num)

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
        # Internal also parametrizes "retrieval" here. This tier has no
        # retrieval: its inference configs carry no "retrieval_config" key, so
        # there is no such execution mode to exclude.
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
