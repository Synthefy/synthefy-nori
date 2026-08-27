from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from sklearn.preprocessing import PowerTransformer

import synthefy_nori.inference.inference_method as inference_method
from synthefy_nori.inference.predictor import NoriPredictor
from synthefy_nori.inference.preprocess import FilterValidFeatures
from synthefy_nori.utils.data_utils import DistributedInferenceDataset


@pytest.mark.parametrize(
    ("device_type", "expected_dtype"),
    [
        ("cpu", torch.float64),
        ("cuda", torch.float64),
        ("mps", torch.float32),
    ],
)
@pytest.mark.parametrize(
    "collapse_mode",
    ["mean", "tail_aware", "qdist_simple"],
)
def test_quantile_collapse_uses_an_mps_supported_tau_dtype(
    monkeypatch,
    device_type,
    expected_dtype,
    collapse_mode,
):
    from synthefy_nori.model import quantile_dist

    class FakeQuantiles:
        def __init__(self):
            self.values = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
            self.shape = self.values.shape
            self.device = torch.device(device_type)

        def __getitem__(self, key):
            return self.values[key]

    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.quantile_collapse = collapse_mode
    predictor.model = type(
        "QuantileHead",
        (),
        {
            "regression_quantiles": (0.25, 0.5, 0.75),
            "num_reg_quantiles": 3,
        },
    )()
    tau_sentinel = object()
    tensor_request = {}

    def fake_as_tensor(values, *, device, dtype):
        tensor_request.update(values=values, device=device, dtype=dtype)
        return tau_sentinel

    def fake_quantile_mean(q, tau, *, enforce_monotone_first):
        assert isinstance(q, FakeQuantiles)
        assert tau is tau_sentinel
        assert enforce_monotone_first is True
        return torch.zeros(q.shape[:-1])

    monkeypatch.setattr(torch, "as_tensor", fake_as_tensor)
    monkeypatch.setattr(
        quantile_dist,
        "quantile_dist_mean_simple",
        fake_quantile_mean,
    )

    result = predictor._apply_quantile_collapse(FakeQuantiles())

    assert result.shape == (2,)
    assert tensor_request == {
        "values": predictor.regression_quantiles,
        "device": torch.device(device_type),
        "dtype": expected_dtype,
    }


def test_yj_fit_warnings_are_suppressed(monkeypatch):
    calls = {"count": 0}

    def fake_predict_single(self, x_train, y_train, x_test):
        calls["count"] += 1
        value = 0.0 if calls["count"] == 1 else 2.0
        return np.full(len(x_test), value)

    def warning_fit_transform(self, values):
        warnings.warn("synthetic power-transform warning", RuntimeWarning)
        return values

    monkeypatch.setattr(NoriPredictor, "_predict_reg_single", fake_predict_single)
    monkeypatch.setattr(PowerTransformer, "fit_transform", warning_fit_transform)
    monkeypatch.setattr(PowerTransformer, "inverse_transform", lambda self, values: values)

    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.augmentations = ("yj",)
    predictor.yj_skew_threshold = 0.0
    y_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = predictor._predict_reg(
            np.zeros((len(y_train), 2)),
            y_train,
            np.zeros((3, 2)),
        )

    assert calls["count"] == 2
    np.testing.assert_allclose(result, 1.0)


def test_removed_retrieval_mode_fails_instead_of_ignoring_the_config():
    with pytest.raises(ValueError, match="retrieval inference has been removed"):
        NoriPredictor(
            device=torch.device("cpu"),
            model=object(),
            inference_config=[{"retrieval_config": {"use_retrieval": True}}],
        )


def test_distributed_inference_dataset_shards_only_query_rows():
    x_test = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    dataset = DistributedInferenceDataset(x_test)

    assert len(dataset) == 3
    assert dataset[1]["idx"] == 1
    torch.testing.assert_close(dataset[1]["X_test"], x_test[1])


def test_distributed_inference_rejects_multioutput_labels_before_setup():
    runner = inference_method.DistributedInference(
        model=object(),
        device=torch.device("cuda:0"),
    )

    with pytest.raises(ValueError, match=r"shape \[rows\] or \[rows, 1\]"):
        runner.inference(
            x_train=torch.ones((2, 3)),
            y_train=torch.ones((1, 2)),
            x_test=torch.ones((1, 3)),
        )


def test_distributed_imputation_fails_explicitly():
    with pytest.raises(
        ValueError,
        match="inference_with_DDP does not support mask_prediction",
    ):
        NoriPredictor(
            device=torch.device("cpu"),
            model=object(),
            inference_config=[{}],
            inference_with_DDP=True,
            mask_prediction=True,
        )


def test_distributed_memory_policy_fails_explicitly():
    """DDP inference never resolves a MemoryPolicy or populates memory_report_,
    so requesting one must fail at construction, not surface later as an empty
    report a caller could mistake for a stale runtime."""
    with pytest.raises(
        ValueError,
        match="inference_with_DDP does not support memory_policy",
    ):
        NoriPredictor(
            device=torch.device("cpu"),
            model=object(),
            inference_config=[{}],
            inference_with_DDP=True,
            memory_policy={"cache_dtype": "int8"},
        )


@pytest.mark.parametrize("owns_process_group", [False, True])
def test_distributed_runner_only_closes_owned_process_group(
    monkeypatch,
    owns_process_group,
):
    cleanup_calls = []
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(
        inference_method,
        "setup",
        lambda: (5, 8, owns_process_group),
    )
    monkeypatch.setattr(
        inference_method,
        "cleanup",
        lambda: cleanup_calls.append(True),
    )
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)

    runner = inference_method.DistributedInference(
        model=object(),
        device=torch.device("cuda:0"),
    )
    with runner:
        assert runner.rank == 5
        assert runner.world_size == 8
        assert runner.device == torch.device("cuda:2")

    assert cleanup_calls == ([True] if owns_process_group else [])
    assert runner.rank is None


def _fitted_filter():
    step = FilterValidFeatures()
    source = np.array(
        [
            [1.0, 10.0, 5.0],
            [2.0, 10.0, 6.0],
            [3.0, 10.0, 7.0],
            [4.0, 10.0, 8.0],
        ]
    )
    step.fit(source, categorical_features=[], seed=0)
    step.transform(source)
    return step


def test_filter_valid_features_inverse_selects_matching_chunk_rows():
    predictor = NoriPredictor.__new__(NoriPredictor)
    step = _fitted_filter()
    reconstructed_valid = np.array([[100.0, 500.0], [300.0, 700.0]])

    restored = predictor.PostProcess(
        reconstructed_valid,
        [step],
        {},
        source_row_indices=np.array([0, 2]),
    )

    np.testing.assert_array_equal(
        restored,
        np.array([[100.0, 10.0, 500.0], [300.0, 10.0, 700.0]]),
    )


def test_filter_valid_features_inverse_refuses_ambiguous_row_shape():
    predictor = NoriPredictor.__new__(NoriPredictor)
    step = _fitted_filter()

    with pytest.raises(ValueError, match="source_row_indices"):
        predictor.PostProcess(np.ones((2, 2)), [step], {})


def test_feature_reconstruction_averages_context_and_concatenates_queries():
    chunks = [
        np.array([[1.0], [3.0], [10.0], [11.0]]),
        np.array([[5.0], [7.0], [12.0]]),
    ]

    result = NoriPredictor._aggregate_feature_reconstruction_chunks(
        chunks,
        n_context=2,
    )

    np.testing.assert_array_equal(
        result,
        np.array([[3.0], [5.0], [10.0], [11.0], [12.0]]),
    )


def test_default_elements_budget_uses_accelerator_memory(monkeypatch):
    predictor = NoriPredictor.__new__(NoriPredictor)
    monkeypatch.setattr(predictor, "_total_vram_gb", lambda: 48.0)

    assert predictor._default_max_elements_budget() == 4_000_000


def test_pipeline_batching_routes_regression_task_type(monkeypatch):
    original_to = torch.Tensor.to

    def keep_cpu(tensor, *args, **kwargs):
        if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
            return tensor
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", keep_cpu)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)

    class TaskAwareModel:
        mask_prediction = False
        training = False

        def __init__(self):
            self.task_types = []

        def to(self, _device):
            return self

        def __call__(self, *, x, y, eval_pos, task_type):
            del y
            self.task_types.append(task_type)
            query = x[:, eval_pos:, :1]
            return {
                "reg_output": query,
                "cls_output": query + 1000,
            }

    model = TaskAwareModel()
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.device = torch.device("cuda")
    predictor.model = model
    predictor.seed = 0
    predictor.mix_precision = False
    predictor.mask_prediction = False
    predictor.inference_with_DDP = False
    predictor.memory_policy = "off"
    predictor.preprocess_pipelines = [[], []]
    predictor.inference_config = [{}, {}]
    predictor.preprocess_num = 10
    predictor.seeds = [0] * 20
    predictor._warned_this_call = set()
    predictor._logged_this_call = set()

    x_train = np.arange(12, dtype=np.float32).reshape(4, 3)
    x_test = np.arange(6, dtype=np.float32).reshape(2, 3)
    y_train = np.arange(4, dtype=np.float32)
    outputs = predictor._try_batched_ordinary_regression(
        model,
        x_train_base=x_train,
        x_test_base=x_test,
        y_train=y_train,
        categorical_idx=[],
        n_samples_train=4,
        n_samples_test=2,
        budget_n_features=3,
        max_elements_budget=1_000_000,
        dropped_context_rows=0,
    )

    assert outputs is not None
    assert model.task_types == ["reg"]
    assert len(outputs) == 2
    for output in outputs:
        torch.testing.assert_close(output, torch.from_numpy(x_test[:, :1]))
