"""Regression tests for structural and numerical SCM-prior health."""

import random

import numpy as np
import pytest
import torch

from synthefy_nori.training import data_generator
from synthefy_nori.training import scm_prior_generator as scm_prior


def _mlp_hyperparams():
    return {
        "is_causal": False,
        "num_causes": 8,
        "y_is_effect": True,
        "in_clique": False,
        "sort_features": True,
        "num_layers": 3,
        "hidden_dim": 16,
        "init_std": 1.0,
        "block_wise_dropout": True,
        "dropout_prob": 0.3,
        "sampling": "normal",
        "pre_sample_cause_stats": False,
        "noise_std": 0.01,
        "pre_sample_noise_std": False,
    }


def test_linear_initialization_preserves_random_function_parameters():
    random.seed(43)
    torch.manual_seed(43)
    scm = scm_prior.MLPSCM(
        n_samples=128,
        n_features=8,
        is_causal=False,
        num_layers=3,
        hidden_dim=16,
        activation_factory=lambda: scm_prior.RandomFunctionActivation(32),
        init_std=1.0,
        block_wise_dropout=True,
        dropout_prob=0.5,
        noise_std=0.0,
    )

    activations = [module for module in scm.modules() if isinstance(module, scm_prior.RandomFunctionActivation)]
    assert activations
    for activation in activations:
        assert torch.count_nonzero(activation.freqs) == activation.freqs.numel()
        assert torch.count_nonzero(activation.bias) == activation.bias.numel()
        assert torch.count_nonzero(activation.l2_weights) == activation.l2_weights.numel()

    X, y = scm()
    assert torch.isfinite(X).all()
    assert torch.isfinite(y).all()
    assert y.std(unbiased=False) > 1e-6


def test_block_dropout_uses_probability_and_keeps_every_row_and_column(monkeypatch):
    monkeypatch.setattr(scm_prior.pyrandom, "randint", lambda _low, _high: 3)

    torch.manual_seed(7)
    dense_blocks = torch.empty(7, 5)
    scm_prior._block_wise_dropout_init(
        dense_blocks,
        init_std=1.0,
        dropout_prob=0.0,
    )

    torch.manual_seed(7)
    sparse_blocks = torch.empty(7, 5)
    scm_prior._block_wise_dropout_init(
        sparse_blocks,
        init_std=1.0,
        dropout_prob=0.9,
    )

    assert torch.all(torch.count_nonzero(sparse_blocks, dim=1) > 0)
    assert torch.all(torch.count_nonzero(sparse_blocks, dim=0) > 0)
    assert torch.count_nonzero(sparse_blocks) < torch.count_nonzero(dense_blocks)


def test_block_dropout_rejects_impossible_probability():
    with pytest.raises(ValueError, match="dropout_prob"):
        scm_prior._block_wise_dropout_init(
            torch.empty(3, 3),
            init_std=1.0,
            dropout_prob=1.0,
        )


def test_positive_meta_distribution_has_no_boundary_atom():
    sample = scm_prior.meta_trunc_norm_log_scaled(
        np.random.default_rng(5),
        min_mean=0.01,
        max_mean=0.01,
        lower_bound=0.0,
    )
    values = np.array([sample() for _ in range(2_000)])
    assert np.all(values > 0.0)
    assert np.count_nonzero(values == 0.0) == 0


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_robust_stabilization_preserves_rare_discrete_levels(array_type):
    values = np.array([0.0] * 100 + [1.0, 2.0], dtype=np.float32)[:, None]
    if array_type == "torch":
        values = torch.from_numpy(values)

    stabilized = scm_prior._robust_stabilize_features(values)
    if isinstance(stabilized, torch.Tensor):
        stabilized = stabilized.numpy()

    assert len(np.unique(stabilized[:, 0])) == 3
    assert np.max(np.abs(stabilized)) <= 50.0


def test_random_choice_activation_cannot_choose_itself_recursively():
    factories = scm_prior.get_activations()
    choice_factory = factories[len(factories) // 2]
    assert isinstance(choice_factory, scm_prior.RandomChoiceFactory)
    assert all(not isinstance(factory, scm_prior.RandomChoiceFactory) for factory in choice_factory.act_factories)


def test_failed_mlp_draws_use_an_observable_exactly_learnable_fallback(monkeypatch):
    class NumericallyInvalidSCM:
        def __init__(self, **_kwargs):
            raise RuntimeError("sampled numerical failure")

    monkeypatch.setattr(scm_prior, "MLPSCM", NumericallyInvalidSCM)
    X, y, meta = scm_prior._generate_mlp(
        256,
        8,
        _mlp_hyperparams(),
        np.random.default_rng(13),
        "cpu",
    )

    weights = np.arange(4, 0, -1, dtype=np.float32)
    weights /= np.linalg.norm(weights)
    np.testing.assert_allclose(y, X[:, :4] @ weights)
    assert meta == {
        "generation_attempts": 3,
        "fallback_used": True,
        "fallback_reason": "RuntimeError: sampled numerical failure",
    }


@pytest.mark.parametrize(
    "error",
    [
        TypeError("bad API call"),
        RuntimeError("shape mismatch"),
        RuntimeError("Inference tensors cannot be saved for backward"),
        RuntimeError("inferred shape mismatch"),
    ],
)
def test_programming_errors_are_not_swallowed(monkeypatch, error):
    class BrokenSCM:
        def __init__(self, **_kwargs):
            raise error

    monkeypatch.setattr(scm_prior, "MLPSCM", BrokenSCM)
    with pytest.raises(type(error), match=str(error)):
        scm_prior._generate_mlp(
            32,
            4,
            _mlp_hyperparams(),
            np.random.default_rng(13),
            "cpu",
        )


@pytest.mark.parametrize(
    "message",
    [
        "output contains NaN",
        "output contains inf",
        "array must not contain infs or NaNs",
        "output is infinite",
        "output is non-finite",
        "overflow encountered in sampled activation",
        "sampled numerical failure",
    ],
)
def test_known_numerical_errors_remain_retryable(message):
    assert scm_prior._retryable_generation_exception(RuntimeError(message))


def test_tree_presampled_noise_uses_per_output_scales(monkeypatch):
    class ZeroTreeLayer:
        def __init__(self, _tree_model, _max_depth, _n_estimators, out_dim, _rng):
            self.out_dim = out_dim

        def fit_transform(self, X, _rng, context_rows=None):
            del context_rows
            return np.zeros((len(X), self.out_dim), dtype=np.float32)

    class RecordingRNG:
        def __init__(self):
            self.normal_shapes = []

        def integers(self, *_args, **_kwargs):
            return 1

        def exponential(self, *_args, **_kwargs):
            return 0.0

        def normal(self, _mean, _std, size):
            self.normal_shapes.append(size)
            return np.full(size, 0.25)

        def standard_normal(self, shape):
            return np.ones(shape, dtype=np.float32)

    monkeypatch.setattr(scm_prior, "TreeLayer", ZeroTreeLayer)
    rng = RecordingRNG()
    scm = scm_prior.TreeSCM(
        n_samples=16,
        n_features=3,
        n_outputs=1,
        noise_std=2.0,
        pre_sample_noise_std=True,
        rng=rng,
    )
    scm.num_layers = 1

    _X, y = scm.generate()
    assert rng.normal_shapes == [(1, 1)]
    np.testing.assert_array_equal(y, np.full(16, 0.25, dtype=np.float32))


def test_scm_target_normalization_does_not_inspect_query_targets(monkeypatch):
    n_samples, n_features, context_rows = 12, 3, 6
    X = np.arange(n_samples * n_features, dtype=np.float32).reshape(n_samples, n_features)
    context_y = np.linspace(-2.0, 3.0, context_rows, dtype=np.float32)
    query_a = np.linspace(4.0, 7.0, n_samples - context_rows, dtype=np.float32)
    query_b = query_a * 100.0 + 10_000.0

    hp = {
        "scm_type": "mlp",
        "num_causes": 2,
        "noise_std": 0.1,
        "num_layers": 2,
        "hidden_dim": 4,
    }
    monkeypatch.setattr(scm_prior, "sample_hyperparams", lambda *_args, **_kwargs: hp)

    state = {"query": query_a}

    def fixed_mlp(*_args, **_kwargs):
        y = np.concatenate([context_y, state["query"]])
        return X.copy(), y, {"generation_attempts": 1, "fallback_used": False}

    monkeypatch.setattr(scm_prior, "_generate_mlp", fixed_mlp)
    first = scm_prior._generate_scm_prior_dataset(
        n_samples, n_features, "reg", None, np.random.default_rng(8), "cpu", context_rows=context_rows
    )
    state["query"] = query_b
    second = scm_prior._generate_scm_prior_dataset(
        n_samples, n_features, "reg", None, np.random.default_rng(8), "cpu", context_rows=context_rows
    )

    np.testing.assert_array_equal(first["y"][:context_rows], second["y"][:context_rows])
    assert float(np.mean(first["y"][:context_rows])) == pytest.approx(0.0)
    assert float(np.std(first["y"][:context_rows])) == pytest.approx(1.0)
    assert first["meta"]["context_rows"] == context_rows


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_scm_feature_stabilization_fits_context_only(array_type):
    context_rows = 8
    base = np.arange(48, dtype=np.float32).reshape(16, 3)
    shifted = base.copy()
    shifted[context_rows:] = shifted[context_rows:] * 100.0 + 10_000.0
    if array_type == "torch":
        base = torch.from_numpy(base)
        shifted = torch.from_numpy(shifted)

    first = scm_prior._robust_stabilize_features(
        base,
        context_rows=context_rows,
    )
    second = scm_prior._robust_stabilize_features(
        shifted,
        context_rows=context_rows,
    )
    if isinstance(first, torch.Tensor):
        first = first.numpy()
        second = second.numpy()
    np.testing.assert_array_equal(
        first[:context_rows],
        second[:context_rows],
    )


def test_scm_validation_requires_context_variation():
    context_rows = 8
    X = np.zeros((16, 3), dtype=np.float32)
    y = np.zeros(16, dtype=np.float32)
    X[context_rows:, 0] = np.arange(8, dtype=np.float32)
    y[context_rows:] = np.arange(8, dtype=np.float32)

    healthy, reason = scm_prior._validate_generated_data(
        X,
        y,
        16,
        3,
        context_rows=context_rows,
    )

    assert healthy is False
    assert reason == "constant features"

    X[:context_rows] = np.arange(24, dtype=np.float32).reshape(8, 3)
    healthy, reason = scm_prior._validate_generated_data(
        X,
        y,
        16,
        3,
        context_rows=context_rows,
    )
    assert healthy is False
    assert reason == "constant target"


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_scm_validation_accepts_finite_singleton_context(array_type):
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y = np.array([5.0, 6.0], dtype=np.float32)
    if array_type == "torch":
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

    healthy, reason = scm_prior._validate_generated_data(
        X,
        y,
        2,
        2,
        context_rows=1,
    )

    assert healthy is True
    assert reason == ""


@pytest.mark.parametrize("array_type", ["numpy", "torch"])
def test_scm_validation_ignores_query_only_nonfinite_values(array_type):
    context_rows = 8
    X = np.arange(48, dtype=np.float32).reshape(16, 3)
    y = np.arange(16, dtype=np.float32)
    X[context_rows:, 0] = np.nan
    X[context_rows:, 1] = np.inf
    y[context_rows:] = np.nan
    if array_type == "torch":
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

    healthy, reason = scm_prior._validate_generated_data(
        X,
        y,
        16,
        3,
        context_rows=context_rows,
    )

    assert healthy is True
    assert reason == ""


@pytest.mark.parametrize("scm_type", ["mlp", "tree"])
def test_scm_context_is_invariant_to_query_causes(monkeypatch, scm_type):
    n_samples, n_features, context_rows = 64, 4, 32
    state = {"query_mode": "unchanged"}
    original_sample = scm_prior.XSampler.sample

    def sample_with_query_shift(self, context_rows=None):
        values = original_sample(self, context_rows=context_rows)
        if state["query_mode"] == "shifted":
            values = values.clone()
            values[context_rows:] = values[context_rows:] * 50.0 + 500.0
        elif state["query_mode"] == "nonfinite":
            values = values.clone()
            values[context_rows:] = torch.nan
        return values

    monkeypatch.setattr(scm_prior.XSampler, "sample", sample_with_query_shift)
    if scm_type == "mlp":
        hp = {
            **_mlp_hyperparams(),
            "scm_type": "mlp",
            "sampling": "normal",
            "noise_std": 0.0,
        }
        monkeypatch.setattr(
            scm_prior,
            "_sample_activation_factory",
            lambda _rng: scm_prior.StdRandomScaleFactory(torch.nn.Tanh),
        )
    else:
        hp = {
            "scm_type": "tree",
            "num_causes": n_features,
            "tree_model": "extra_trees",
            "max_depth_lambda": 0.5,
            "n_estimators_lambda": 0.5,
            "sampling": "normal",
            "pre_sample_cause_stats": False,
            "noise_std": 0.0,
            "pre_sample_noise_std": False,
        }
    monkeypatch.setattr(
        scm_prior,
        "sample_hyperparams",
        lambda *_args, **_kwargs: hp,
    )

    first = scm_prior.generate_scm_prior_dataset(
        n_samples,
        n_features,
        "reg",
        rng=np.random.default_rng(31),
        context_rows=context_rows,
    )
    state["query_mode"] = "shifted"
    second = scm_prior.generate_scm_prior_dataset(
        n_samples,
        n_features,
        "reg",
        rng=np.random.default_rng(31),
        context_rows=context_rows,
    )

    np.testing.assert_array_equal(
        first["X"][:context_rows],
        second["X"][:context_rows],
    )
    np.testing.assert_array_equal(
        first["y"][:context_rows],
        second["y"][:context_rows],
    )

    state["query_mode"] = "nonfinite"
    nonfinite_query = scm_prior.generate_scm_prior_dataset(
        n_samples,
        n_features,
        "reg",
        rng=np.random.default_rng(31),
        context_rows=context_rows,
    )
    assert nonfinite_query["meta"]["fallback_used"] is False
    assert np.isfinite(nonfinite_query["X"]).all()
    assert np.isfinite(nonfinite_query["y"]).all()
    np.testing.assert_array_equal(
        first["X"][:context_rows],
        nonfinite_query["X"][:context_rows],
    )
    np.testing.assert_array_equal(
        first["y"][:context_rows],
        nonfinite_query["y"][:context_rows],
    )


@pytest.mark.parametrize("scm_type", ["mlp", "tree"])
def test_scm_singleton_context_preserves_a_usable_episode(monkeypatch, scm_type):
    if scm_type == "mlp":
        hp = {
            **_mlp_hyperparams(),
            "scm_type": "mlp",
            "sampling": "normal",
            "noise_std": 0.0,
        }
        monkeypatch.setattr(
            scm_prior,
            "_sample_activation_factory",
            lambda _rng: scm_prior.StdRandomScaleFactory(torch.nn.Tanh),
        )
    else:
        hp = {
            "scm_type": "tree",
            "num_causes": 4,
            "tree_model": "extra_trees",
            "max_depth_lambda": 0.5,
            "n_estimators_lambda": 0.5,
            "sampling": "normal",
            "pre_sample_cause_stats": False,
            "noise_std": 0.0,
            "pre_sample_noise_std": False,
        }
    monkeypatch.setattr(
        scm_prior,
        "sample_hyperparams",
        lambda *_args, **_kwargs: hp,
    )

    data = scm_prior.generate_scm_prior_dataset(
        32,
        4,
        "reg",
        rng=np.random.default_rng(41),
        context_rows=1,
    )

    assert data["meta"]["fallback_used"] is (scm_type == "tree")
    assert np.isfinite(data["X"]).all()
    assert np.isfinite(data["y"]).all()
    assert np.count_nonzero(data["X"]) > 0
    assert np.std(data["y"]) > 0.0


@pytest.mark.parametrize("module", [data_generator, scm_prior])
def test_context_rows_must_include_at_least_one_labeled_row(module):
    with pytest.raises(ValueError, match="at least 1"):
        module._resolve_context_rows(16, 0)


def test_generate_batch_passes_context_rows_to_scm_prior(monkeypatch):
    seen_context_rows = []

    def fake_scm(n_samples, n_features, task_type, n_classes=None, rng=None, device="cpu", context_rows=None):
        del task_type, n_classes, rng, device
        seen_context_rows.append(context_rows)
        X = np.arange(n_samples * n_features, dtype=np.float32).reshape(n_samples, n_features)
        y = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
        return {"X": X, "y": y, "meta": {"generator_family": "scm_prior"}}

    monkeypatch.setattr(scm_prior, "generate_scm_prior_dataset", fake_scm)
    data_generator.generate_batch(
        batch_size=1,
        n_samples=64,
        n_features=4,
        task_type="reg",
        rng=np.random.default_rng(9),
        scm_prior=True,
        scm_prior_prob=1.0,
        context_rows=23,
    )

    assert seen_context_rows == [23]
