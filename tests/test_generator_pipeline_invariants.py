"""Regression tests for synthetic-pipeline invariants.

These tests target failure modes that ordinary shape/finiteness smoke tests do
not detect: query leakage, stale target snapshots, silent filter fallthrough,
miscalibrated noise, and feature transforms that escape the numeric contract.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest
import sklearn.ensemble
import torch

from synthefy_nori.training import data_generator as dg


class _QueryPerturbingFeatureRng:
    """Replay one RNG stream while perturbing only initial query features."""

    def __init__(self, seed, n_samples, n_features, context_rows, shift):
        self._rng = np.random.default_rng(seed)
        self._n_samples = n_samples
        self._remaining_feature_draws = n_features
        self._context_rows = context_rows
        self._shift = shift

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def _maybe_perturb(self, values):
        values = np.asarray(values)
        if (self._remaining_feature_draws > 0
                and values.shape == (self._n_samples,)):
            self._remaining_feature_draws -= 1
            if self._shift:
                values = values.copy()
                values[self._context_rows:] = (
                    values[self._context_rows:] * 11.0 + self._shift)
        return values

    def standard_normal(self, *args, **kwargs):
        return self._maybe_perturb(
            self._rng.standard_normal(*args, **kwargs))

    def uniform(self, *args, **kwargs):
        return self._maybe_perturb(self._rng.uniform(*args, **kwargs))

    def beta(self, *args, **kwargs):
        return self._maybe_perturb(self._rng.beta(*args, **kwargs))

    def standard_t(self, *args, **kwargs):
        return self._maybe_perturb(
            self._rng.standard_t(*args, **kwargs))


def test_feature_tail_missingness_uses_observed_covariates_not_labels():
    assert list(inspect.signature(
        dg._apply_feature_dependent_tail_missingness).parameters) == [
            'X', 'rng', 'context_rows']
    X = np.column_stack([
        np.arange(20, dtype=np.float64),
        np.linspace(-2.0, 2.0, 20),
        np.ones(20, dtype=np.float64),
    ])

    first = X.copy()
    second = X.copy()
    second[12:, :2] = -1_000_000
    dg._apply_feature_dependent_tail_missingness(
        first, np.random.default_rng(17), context_rows=12)
    dg._apply_feature_dependent_tail_missingness(
        second, np.random.default_rng(17), context_rows=12)

    np.testing.assert_array_equal(
        np.isnan(first[:12]), np.isnan(second[:12]))
    assert not np.array_equal(
        np.isnan(first[12:]), np.isnan(second[12:]))
    assert np.isnan(first).any()
    assert np.isfinite(first).any()


def test_extra_trees_filter_rejects_scoreability_errors(monkeypatch):
    class NumericallyUnscoreableModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise ValueError('non-finite episode')

    monkeypatch.setattr(
        sklearn.ensemble, 'ExtraTreesRegressor', NumericallyUnscoreableModel)
    X = np.random.default_rng(0).standard_normal((64, 8))
    y = np.random.default_rng(1).standard_normal(64)
    assert dg._check_learnability(X, y) is False


def test_extra_trees_filter_propagates_programming_errors(monkeypatch):
    class BrokenFilterAPI:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise TypeError('wrong model API')

    monkeypatch.setattr(
        sklearn.ensemble, 'ExtraTreesRegressor', BrokenFilterAPI)
    X = np.random.default_rng(2).standard_normal((64, 8))
    y = np.random.default_rng(3).standard_normal(64)
    with pytest.raises(TypeError, match='wrong model API'):
        dg._check_learnability(X, y)


@pytest.mark.parametrize('message', [
    'bad API call',
    'input contains inferred shape',
])
def test_extra_trees_filter_propagates_api_value_errors(monkeypatch, message):
    class BrokenFilterAPI:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise ValueError(message)

    monkeypatch.setattr(
        sklearn.ensemble, 'ExtraTreesRegressor', BrokenFilterAPI)
    X = np.random.default_rng(6).standard_normal((64, 8))
    y = np.random.default_rng(7).standard_normal(64)
    with pytest.raises(ValueError, match=message):
        dg._check_learnability(X, y)


@pytest.mark.parametrize('message', [
    'Input contains NaN',
    'Input contains infinity or a value too large',
    'array must not contain infs or nans',
    'numerical overflow while scoring',
])
def test_numerical_scoreability_error_classifier_is_boundary_aware(message):
    assert dg._is_numerical_scoreability_error(ValueError(message)) is True
    assert dg._is_numerical_scoreability_error(
        ValueError('input contains inferred shape')) is False


def test_configured_learnability_gates_reject_undersized_episodes(monkeypatch):
    class MustNotRun:
        def __call__(self, *args, **kwargs):
            raise AssertionError('undersized data must reject before inference')

    monkeypatch.setattr(dg, '_get_icl_filter_model', lambda _path: MustNotRun())
    X = np.random.default_rng(8).standard_normal((19, 4))
    y = np.linspace(-1, 1, 19, dtype=np.float32)
    assert dg._check_learnability(X, y) is False
    assert dg._check_learnability_icl(X, y, 'fake-model') is False


def test_icl_filter_rejects_constant_sampled_context(monkeypatch):
    class MustNotRun:
        def __call__(self, *args, **kwargs):
            raise AssertionError('constant context must reject before inference')

    monkeypatch.setattr(dg, '_get_icl_filter_model', lambda _path: MustNotRun())
    X = np.random.default_rng(4).standard_normal((96, 8))
    y = np.concatenate([
        np.zeros(64, dtype=np.float32),
        np.linspace(-1, 1, 32, dtype=np.float32),
    ])
    assert dg._check_learnability_icl(X, y, 'fake-model') is False


def test_icl_filter_propagates_model_api_failures(monkeypatch):
    class BrokenModel:
        def __call__(self, *args, **kwargs):
            raise RuntimeError('broken forward contract')

    monkeypatch.setattr(dg, '_get_icl_filter_model', lambda _path: BrokenModel())
    X = np.random.default_rng(5).standard_normal((96, 8))
    y = np.linspace(-2, 2, 96, dtype=np.float32)
    with pytest.raises(RuntimeError, match='broken forward contract'):
        dg._check_learnability_icl(X, y, 'fake-model')


def test_max_parents_is_an_inclusive_limit():
    observed = set()
    for seed in range(100):
        lcs = dg.LocalCausalStructure(
            8, [], np.random.default_rng(seed), max_parents=4)
        observed.update(len(parents) for parents in lcs.parents.values())
    assert 4 in observed
    assert max(observed) == 4


def test_hard_negatives_are_nan_safe_and_do_not_overwrite_protected_columns():
    saw_hard_negative = False
    for seed in range(200):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((128, 12))
        # Exercise nan-safety without giving the helper any all-finite column.
        X[np.arange(12), np.arange(12)] = np.nan
        protected_before = X[:, :3].copy()
        metadata = {}
        X_after, y_after = dg._add_latent_bayes_error(
            X.copy(), rng.standard_normal(128), 12, rng,
            protected_cols={0, 1, 2}, metadata=metadata)
        assert np.isfinite(y_after).all()
        assert not np.any(np.all(np.isnan(X_after), axis=0))
        np.testing.assert_allclose(
            X_after[:, :3], protected_before, equal_nan=True)
        if metadata.get('hard_negative_cols'):
            saw_hard_negative = True
            assert set(metadata['hard_negative_cols']).isdisjoint({0, 1, 2})
    assert saw_hard_negative


@pytest.mark.parametrize('op', [0, 1, 2])
def test_groupby_context_effect_is_invariant_to_query_composition(op):
    class FixedOpRng:
        def __init__(self, seed, fixed_op):
            self.base = np.random.default_rng(seed)
            self.fixed_op = fixed_op

        def uniform(self, *args, **kwargs):
            return self.base.uniform(*args, **kwargs)

        def choice(self, *args, **kwargs):
            return self.base.choice(*args, **kwargs)

        def standard_normal(self, *args, **kwargs):
            return self.base.standard_normal(*args, **kwargs)

        def integers(self, low, high=None, *args, **kwargs):
            assert low == 0 and high == 3
            return self.fixed_op

    n_samples, n_features, context_rows = 240, 8, 96
    X_a = np.random.default_rng(44).standard_normal((n_samples, n_features))
    X_b = X_a.copy()
    X_b[context_rows:, 1:] = (
        X_b[context_rows:, 1:] * 50.0 + 1000.0)

    effect_a, _ = dg._create_groupby_relative_numeric(
        X_a, 0, n_samples, FixedOpRng(91, op), 10,
        n_features, set(), {0}, context_rows=context_rows)
    effect_b, _ = dg._create_groupby_relative_numeric(
        X_b, 0, n_samples, FixedOpRng(91, op), 10,
        n_features, set(), {0}, context_rows=context_rows)
    np.testing.assert_allclose(
        effect_a[:context_rows], effect_b[:context_rows], atol=0.0, rtol=0.0)


def test_entity_signal_mixture_scale_uses_context_only():
    context_rows = 80
    rng = np.random.default_rng(123)
    y_a = rng.standard_normal(200)
    effect_a = rng.standard_normal(200)
    y_b = y_a.copy()
    effect_b = effect_a.copy()
    y_b[context_rows:] = y_b[context_rows:] * 100.0 + 300.0
    effect_b[context_rows:] = effect_b[context_rows:] * 50.0 - 200.0
    out_a = dg._add_entity_lookup_signal(
        y_a, effect_a, np.random.default_rng(5), {},
        context_rows=context_rows)
    out_b = dg._add_entity_lookup_signal(
        y_b, effect_b, np.random.default_rng(5), {},
        context_rows=context_rows)
    np.testing.assert_allclose(
        out_a[:context_rows], out_b[:context_rows], atol=0.0, rtol=0.0)


def test_rich_target_component_scale_uses_context_only():
    context_rows = 80
    rng = np.random.default_rng(321)
    y_a = rng.standard_normal(200)
    extra_a = rng.standard_normal(200)
    y_b = y_a.copy()
    extra_b = extra_a.copy()
    y_b[context_rows:] = y_b[context_rows:] * 100.0 + 500.0
    extra_b[context_rows:] = extra_b[context_rows:] * 80.0 - 300.0
    scaled_a = dg._scale_rich_target_component(
        extra_a, y_a, np.random.default_rng(8), context_rows=context_rows)
    scaled_b = dg._scale_rich_target_component(
        extra_b, y_b, np.random.default_rng(8), context_rows=context_rows)
    np.testing.assert_allclose(
        scaled_a[:context_rows], scaled_b[:context_rows], atol=0.0, rtol=0.0)


def test_tail_missingness_control_flow_ignores_query_only_tails():
    context_rows = 8
    pattern = np.tile([0.0, 1.0], 8)
    base = np.zeros((16, 3), dtype=np.float64)
    base[:, 0] = pattern
    shifted = base.copy()
    shifted[context_rows:, 0] += 10.0

    base_applied = dg._apply_feature_dependent_tail_missingness(
        base, np.random.default_rng(0), context_rows=context_rows)
    shifted_applied = dg._apply_feature_dependent_tail_missingness(
        shifted, np.random.default_rng(0), context_rows=context_rows)

    assert base_applied is False
    assert shifted_applied is False
    np.testing.assert_array_equal(
        base[:context_rows], shifted[:context_rows])
    assert not np.isnan(base).any()
    assert np.isnan(shifted[context_rows:]).any()


@pytest.mark.parametrize('transform_kwargs', [
    {'realistic_augmentation_prob': 1.0},
    {'y_transform_prob': 1.0},
    {'low_unique_y_prob': 1.0},
    {'cap_injection_prob': 1.0, 'cap_high_fraction_prob': 1.0},
    {'heavy_tail_prior_prob': 1.0},
])
def test_final_target_transforms_fit_context_only(monkeypatch, transform_kwargs):
    n_samples, n_features, context_rows = 128, 8, 64
    base_y = np.linspace(-2.0, 2.0, n_samples, dtype=np.float32)
    query_shifted_y = base_y.copy()
    query_shifted_y[context_rows:] = (
        query_shifted_y[context_rows:] * 20.0 + 100.0)
    state = {'shift_query': False}

    def tree_episode(
            _n_samples, _n_features, task_type, n_classes, rng,
            context_rows=None, **_kwargs):
        del task_type, n_classes, rng, context_rows
        X = np.random.default_rng(77).standard_normal(
            (_n_samples, _n_features)).astype(np.float32)
        y = query_shifted_y if state['shift_query'] else base_y
        return {'X': X, 'y': y.copy(), 'filtered': False, 'meta': {}}

    monkeypatch.setattr(dg, '_generate_tree_prior_episode', tree_episode)
    saw_context_transform = False
    for seed in range(20):
        state['shift_query'] = False
        _, y_a, _ = dg.generate_batch(
            1, n_samples, n_features, 'reg', rng=np.random.default_rng(seed),
            tree_prior_prob=1.0, context_rows=context_rows,
            **transform_kwargs)
        state['shift_query'] = True
        _, y_b, _ = dg.generate_batch(
            1, n_samples, n_features, 'reg', rng=np.random.default_rng(seed),
            tree_prior_prob=1.0, context_rows=context_rows,
            **transform_kwargs)
        np.testing.assert_array_equal(
            y_a[0, :context_rows], y_b[0, :context_rows])
        saw_context_transform |= not np.array_equal(
            y_a[0, :context_rows], base_y[:context_rows])
    assert saw_context_transform


def test_complete_noise_vector_matches_requested_empirical_r2():
    rng = np.random.default_rng(7)
    signal = rng.standard_normal(512)
    raw_noise = rng.standard_normal(512)
    raw_noise[:20] *= 30.0
    for requested in (0.3, 0.5, 0.8, 0.98):
        noise, realized = dg._calibrate_noise_to_target_r2(
            signal, raw_noise, requested)
        noisy = signal + noise
        empirical = 1.0 - np.sum(noise ** 2) / np.sum(
            (noisy - noisy.mean()) ** 2)
        assert realized == pytest.approx(requested, abs=1e-12)
        assert empirical == pytest.approx(requested, abs=1e-12)


def test_noise_calibration_scale_is_invariant_to_query_composition():
    context_rows = 96
    rng = np.random.default_rng(17)
    signal_a = rng.standard_normal(256)
    raw_noise_a = rng.standard_normal(256)
    signal_b = signal_a.copy()
    raw_noise_b = raw_noise_a.copy()
    signal_b[context_rows:] = signal_b[context_rows:] * 100.0 + 500.0
    raw_noise_b[context_rows:] = raw_noise_b[context_rows:] * 80.0 - 300.0

    noise_a, realized_a = dg._calibrate_noise_to_target_r2(
        signal_a, raw_noise_a, 0.73, context_rows=context_rows)
    noise_b, realized_b = dg._calibrate_noise_to_target_r2(
        signal_b, raw_noise_b, 0.73, context_rows=context_rows)
    np.testing.assert_array_equal(
        noise_a[:context_rows], noise_b[:context_rows])
    assert realized_a == pytest.approx(0.73, abs=1e-12)
    assert realized_b == pytest.approx(0.73, abs=1e-12)


def test_special_prior_context_targets_ignore_query_feature_values():
    n_samples = 256
    n_features = 12
    context_rows = 128
    generators = (
        dg._generate_gp_prior_episode,
        dg._generate_quadratic_surface_episode,
        dg._generate_sparse_nonlinear_episode,
    )
    for generator in generators:
        for seed in range(3):
            control = generator(
                n_samples,
                n_features,
                'reg',
                None,
                _QueryPerturbingFeatureRng(
                    seed, n_samples, n_features, context_rows, shift=0.0),
                context_rows=context_rows,
            )
            perturbed = generator(
                n_samples,
                n_features,
                'reg',
                None,
                _QueryPerturbingFeatureRng(
                    seed, n_samples, n_features, context_rows, shift=100.0),
                context_rows=context_rows,
            )
            np.testing.assert_array_equal(
                control['y'][:context_rows],
                perturbed['y'][:context_rows],
            )
            np.testing.assert_array_equal(
                control['X'][:context_rows],
                perturbed['X'][:context_rows],
            )


@pytest.mark.parametrize('generator', [
    dg._generate_gp_prior_episode,
    dg._generate_quadratic_surface_episode,
    dg._generate_sparse_nonlinear_episode,
    dg._generate_lookup_prior_episode,
])
def test_special_prior_classification_ignores_regression_context_contract(
        generator):
    control = generator(
        128, 12, 'cls', 3, np.random.default_rng(23))
    with_context = generator(
        128, 12, 'cls', 3, np.random.default_rng(23), context_rows=64)

    np.testing.assert_array_equal(control['X'], with_context['X'])
    np.testing.assert_array_equal(control['y'], with_context['y'])
    assert with_context['task_type'] == 'cls'
    assert with_context['n_classes'] == 3


def test_generate_batch_keeps_public_classification_three_tuple():
    result = dg.generate_batch(
        1, 128, 12, 'cls', n_classes=3,
        rng=np.random.default_rng(31), gp_prior_prob=1.0)

    assert len(result) == 3
    X, y, n_classes = result
    assert X.shape == (1, 128, 12)
    assert y.shape == (1, 128)
    assert n_classes == 3


def test_duplicate_rows_never_copy_query_labels_into_context():
    context_rows = 37
    saw_context_to_query = False
    for seed in range(20):
        src, dst = dg._sample_context_safe_duplicate_indices(
            100, 50, np.random.default_rng(seed),
            context_rows=context_rows)
        assert not np.any((src >= context_rows) & (dst < context_rows))
        saw_context_to_query |= bool(np.any(
            (src < context_rows) & (dst >= context_rows)))
    assert saw_context_to_query


def test_basic_health_cannot_be_rescued_by_query_only_variation():
    X = np.zeros((12, 3), dtype=np.float64)
    y = np.zeros(12, dtype=np.float64)
    X[6:] = np.arange(18, dtype=np.float64).reshape(6, 3)
    y[6:] = np.arange(6, dtype=np.float64)

    assert dg._passes_basic_episode_health(X, y)
    assert not dg._passes_basic_episode_health(
        X, y, context_rows=6)


@pytest.mark.parametrize('family', [
    'regression',
    'tree',
])
def test_family_target_pipeline_ignores_query_composition_when_fitting_stats(
        monkeypatch, family):
    """Exercise each real family, perturbing only stats-visible query values."""
    n_samples, n_features, context_rows = 128, 12, 64
    perturb_query = {'enabled': False}
    calibration_calls = 0
    normalization_calls = 0
    original_calibrate = dg._calibrate_noise_to_target_r2
    original_normalize = dg._normalize_target_pair

    def calibrate(signal, raw_noise, target_r2, context_rows=None):
        nonlocal calibration_calls
        calibration_calls += 1
        assert context_rows == 64
        signal = np.asarray(signal).copy()
        raw_noise = np.asarray(raw_noise).copy()
        if perturb_query['enabled']:
            signal[64:] = signal[64:] * 100.0 + 500.0
            raw_noise[64:] = raw_noise[64:] * 80.0 - 300.0
        return original_calibrate(
            signal, raw_noise, target_r2, context_rows=context_rows)

    def normalize(y, y_clean, context_rows=None):
        nonlocal normalization_calls
        normalization_calls += 1
        assert context_rows == 64
        y = np.asarray(y).copy()
        y_clean = np.asarray(y_clean).copy()
        if perturb_query['enabled']:
            y[64:] = y[64:] * 20.0 + 100.0
            y_clean[64:] = y_clean[64:] * 20.0 + 100.0
        return original_normalize(y, y_clean, context_rows=context_rows)

    monkeypatch.setattr(dg, '_calibrate_noise_to_target_r2', calibrate)
    monkeypatch.setattr(dg, '_normalize_target_pair', normalize)

    def generate():
        rng = np.random.default_rng(103)
        if family == 'regression':
            return dg._generate_regression_prior(
                n_samples, n_features, rng,
                reg_deterministic_prob=0.0,
                preserve_target_features=True,
                context_rows=context_rows)
        if family == 'tree':
            return dg._generate_tree_prior_episode(
                n_samples, n_features, 'reg', None, rng,
                context_rows=context_rows)
        if family == 'gp':
            return dg._generate_gp_prior_episode(
                n_samples, n_features, 'reg', None, rng,
                context_rows=context_rows)
        if family == 'quadratic':
            return dg._generate_quadratic_surface_episode(
                n_samples, n_features, 'reg', None, rng,
                context_rows=context_rows)
        if family == 'sparse':
            return dg._generate_sparse_nonlinear_episode(
                n_samples, n_features, 'reg', None, rng,
                context_rows=context_rows)
        if family == 'lookup':
            return dg._generate_lookup_prior_episode(
                n_samples, n_features, 'reg', None, rng,
                context_rows=context_rows)
        return dg._generate_clean_lowdim_episode(
            n_samples, n_features, 'reg', None, rng,
            context_rows=context_rows)

    data_a = generate()
    calls_after_a = (calibration_calls, normalization_calls)
    perturb_query['enabled'] = True
    data_b = generate()

    assert calls_after_a[0] >= 1
    assert calls_after_a[1] >= 1
    assert calibration_calls == 2 * calls_after_a[0]
    assert normalization_calls == 2 * calls_after_a[1]
    np.testing.assert_array_equal(data_a['X'], data_b['X'])
    np.testing.assert_array_equal(
        data_a['y'][:context_rows], data_b['y'][:context_rows])
    assert not np.array_equal(
        data_a['y'][context_rows:], data_b['y'][context_rows:])


def test_tree_family_raw_query_signal_cannot_change_context_snr(monkeypatch):
    n_samples, n_features, context_rows = 128, 12, 64
    query_shifted = {'enabled': False}
    raw_signal = np.sin(np.linspace(-3.0, 3.0, n_samples))

    def controlled_tree_prediction(
            X, feature_cols, rng, depth, context_rows=None):
        del X, feature_cols, rng, depth
        assert context_rows == 64
        signal = raw_signal.copy()
        if query_shifted['enabled']:
            signal[context_rows:] = (
                signal[context_rows:] * 100.0 + 500.0)
        return signal

    monkeypatch.setattr(
        dg, '_random_decision_tree_predict', controlled_tree_prediction)
    data_a = dg._generate_tree_prior_episode(
        n_samples, n_features, 'reg', None, np.random.default_rng(211),
        context_rows=context_rows)
    query_shifted['enabled'] = True
    data_b = dg._generate_tree_prior_episode(
        n_samples, n_features, 'reg', None, np.random.default_rng(211),
        context_rows=context_rows)

    np.testing.assert_array_equal(data_a['X'], data_b['X'])
    np.testing.assert_array_equal(
        data_a['y'][:context_rows], data_b['y'][:context_rows])
    assert not np.array_equal(
        data_a['y'][context_rows:], data_b['y'][context_rows:])


def test_tree_split_thresholds_fit_context_features_only():
    context_rows = 64
    X_a = np.random.default_rng(212).standard_normal((128, 6))
    X_b = X_a.copy()
    X_b[context_rows:] = X_b[context_rows:] * 100.0 + 500.0
    pred_a = dg._random_decision_tree_predict(
        X_a, np.arange(6), np.random.default_rng(213), depth=4,
        context_rows=context_rows)
    pred_b = dg._random_decision_tree_predict(
        X_b, np.arange(6), np.random.default_rng(213), depth=4,
        context_rows=context_rows)
    np.testing.assert_array_equal(
        pred_a[:context_rows], pred_b[:context_rows])


@pytest.mark.parametrize('seed', [0, 5, 15])
def test_regression_hybrid_target_fit_ignores_query_feature_composition(seed):
    n_samples, n_features, context_rows = 128, 12, 64
    X_a = np.random.default_rng(214).standard_normal(
        (n_samples, n_features)).astype(np.float32)
    X_b = X_a.copy()
    X_b[context_rows:] = X_b[context_rows:] * 100.0 + 500.0
    data_a = dg._generate_regression_prior(
        n_samples, n_features, np.random.default_rng(seed),
        X_scm=X_a, preserve_target_features=True,
        reg_deterministic_prob=0.0, context_rows=context_rows)
    data_b = dg._generate_regression_prior(
        n_samples, n_features, np.random.default_rng(seed),
        X_scm=X_b, preserve_target_features=True,
        reg_deterministic_prob=0.0, context_rows=context_rows)
    np.testing.assert_array_equal(
        data_a['y'][:context_rows], data_b['y'][:context_rows])
    np.testing.assert_array_equal(
        data_a['y_clean'][:context_rows],
        data_b['y_clean'][:context_rows])
    np.testing.assert_array_equal(
        data_a['X'][:context_rows], data_b['X'][:context_rows])


def test_regression_prior_reports_realized_target_r2():
    for seed in range(30):
        data = dg._generate_regression_prior(
            256, 24, np.random.default_rng(seed),
            reg_deterministic_prob=0.0,
            preserve_target_features=True,
            context_rows=128)
        meta = data['meta']
        assert meta['realized_target_r2'] == pytest.approx(
            meta['target_r2'], abs=1e-10)
        assert meta['context_realized_target_r2'] == pytest.approx(
            meta['target_r2'], abs=1e-10)
        assert meta['context_rows'] == 128










def test_filter_exhaustion_raises_instead_of_returning_rejected_data(monkeypatch):
    calls = 0

    def constant_episode(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {
            'X': np.ones((32, 4), dtype=np.float32),
            'y': np.zeros(32, dtype=np.float32),
            'filtered': False,
            'meta': {},
        }

    monkeypatch.setattr(dg, 'generate_dataset', constant_episode)
    with pytest.raises(dg.SyntheticDataFilterError):
        dg.generate_dataset_filtered(32, 4, 'reg', max_retries=2)
    assert calls == 3


def test_special_prior_retries_its_actual_target_on_failed_health(monkeypatch):
    calls = 0

    def tree_episode(
            n_samples, n_features, task_type, n_classes, rng,
            context_rows=None, **_kwargs):
        del task_type, n_classes, rng, context_rows
        nonlocal calls
        calls += 1
        X = np.tile(np.arange(n_features), (n_samples, 1)).astype(np.float32)
        X[:, 0] = np.arange(n_samples)
        y = (
            np.zeros(n_samples, dtype=np.float32)
            if calls == 1 else np.linspace(-1, 1, n_samples, dtype=np.float32)
        )
        return {'X': X, 'y': y, 'filtered': False, 'meta': {}}

    monkeypatch.setattr(dg, '_generate_tree_prior_episode', tree_episode)
    X, y, _ = dg.generate_batch(
        1, 64, 8, 'reg', rng=np.random.default_rng(0),
        tree_prior_prob=1.0, filter_max_retries=2)
    assert calls == 2
    assert X.shape == (1, 64, 8)
    assert np.std(y[0]) > 0


def test_lookup_prior_keeps_ids_at_narrow_width_and_adds_cold_start():
    for seed in range(30):
        data = dg._generate_lookup_prior_episode(
            50, 2, 'reg', None, np.random.default_rng(seed),
            context_rows=15)
        meta = data['meta']
        assert meta['n_id_cols'] == 1
        assert np.equal(data['X'][:, 0], np.round(data['X'][:, 0])).all()
        assert all(count >= 1 for count in meta['minimum_context_counts'])
        assert all(count >= 1 for count in meta['unseen_cardinalities'])
        assert all(0.04 <= frac <= 0.21
                   for frac in meta['query_unseen_fractions'])


def test_final_feature_safety_is_affine_and_bounded():
    X = np.zeros((1, 256, 3), dtype=np.float64)
    X[0, :, 0] = np.exp(np.linspace(-20, 20, 256))
    X[0, :, 1] = np.sign(np.linspace(-1e6, 1e6, 256)) * (
        np.linspace(-1e6, 1e6, 256) ** 2)
    X[0, 10, 2] = np.nan
    stabilized, affine, bounded = dg._finalize_feature_batch(X)
    assert set(affine) == {(0, 0), (0, 1)}
    assert bounded == []
    assert np.nanmax(np.abs(stabilized)) <= 1e4
    assert np.isnan(stabilized[0, 10, 2])
    np.testing.assert_array_equal(
        np.argsort(X[0, :, 0]), np.argsort(stabilized[0, :, 0]))
    np.testing.assert_array_equal(
        np.argsort(X[0, :, 1]), np.argsort(stabilized[0, :, 1]))


@pytest.mark.parametrize(
    ("warp", "limit"),
    [
        (dg._exp_with_linear_tails, 20.0),
        (dg._sigmoid_with_linear_tails, 8.0),
    ],
)
def test_realistic_monotone_warps_preserve_distinct_query_tails(warp, limit):
    values = np.array(
        [
            -1e6, -200.0, -limit - 1.0, -limit, 0.0,
            limit, limit + 1.0, 200.0, 1e6,
        ],
        dtype=np.float64,
    )
    transformed = warp(values).astype(np.float32)

    assert np.all(np.diff(transformed) > 0.0)
    assert len(np.unique(transformed)) == len(values)


def test_feature_transforms_fit_only_the_context_prefix():
    context_rows = 32
    base = np.arange(384, dtype=np.float64).reshape(64, 6) / 10.0
    shifted = base.copy()
    shifted[context_rows:] = (
        shifted[context_rows:] * 100.0 + 10_000.0)

    correlated_base = base.copy()
    correlated_shifted = shifted.copy()
    dg._apply_invertible_feature_correlation(
        correlated_base, [0, 1, 2], np.random.default_rng(9), rho=0.5,
        context_rows=context_rows)
    dg._apply_invertible_feature_correlation(
        correlated_shifted, [0, 1, 2], np.random.default_rng(9), rho=0.5,
        context_rows=context_rows)
    np.testing.assert_array_equal(
        correlated_base[:context_rows],
        correlated_shifted[:context_rows],
    )

    stabilized_base = dg._affine_stabilize_feature_column(
        base[:, 0], context_rows=context_rows)
    stabilized_shifted = dg._affine_stabilize_feature_column(
        shifted[:, 0], context_rows=context_rows)
    np.testing.assert_array_equal(
        stabilized_base[:context_rows],
        stabilized_shifted[:context_rows],
    )


def test_final_feature_safety_rejects_infinity_but_preserves_nan():
    X = np.zeros((1, 16, 2), dtype=np.float64)
    X[0, 3, 0] = np.nan
    X[0, 4, 1] = np.inf
    with pytest.raises(
            dg.SyntheticDataFilterError, match='contains infinity'):
        dg._finalize_feature_batch(X)


def test_final_feature_safety_does_not_rescale_context_for_query_extremes():
    context_rows = 8
    base = np.arange(32, dtype=np.float64).reshape(1, 16, 2)
    shifted = base.copy()
    shifted[:, context_rows:, 0] = 1e12
    shifted[:, context_rows:, 1] = np.inf

    base_out, _, _ = dg._finalize_feature_batch(
        base, context_rows=context_rows)
    shifted_out, affine, bounded = dg._finalize_feature_batch(
        shifted, context_rows=context_rows)

    np.testing.assert_array_equal(
        base_out[:, :context_rows],
        shifted_out[:, :context_rows],
    )
    assert affine == []
    assert set(bounded) == {(0, 0), (0, 1)}
    assert np.nanmax(np.abs(shifted_out)) <= 1e4
    assert not np.isinf(shifted_out).any()


def test_query_only_feature_bound_invalidates_oracle(monkeypatch):
    context_rows = 8

    def tree_episode(
            n_samples, n_features, task_type, n_classes, rng,
            context_rows=None, **_kwargs):
        del task_type, n_classes, rng
        assert context_rows == 8
        X = np.column_stack([
            np.arange(n_samples, dtype=np.float64),
            np.arange(n_samples, dtype=np.float64) * 2.0,
        ])
        X[context_rows:, 0] = 1e12
        y = np.arange(n_samples, dtype=np.float32)
        return {
            "X": X,
            "y": y,
            "y_clean": y.copy(),
            "filtered": False,
            "meta": {
                "generator_family": "tree_prior",
                "oracle_exact": True,
                "r2_oracle_bound": 0.0,
            },
        }

    monkeypatch.setattr(dg, "_generate_tree_prior_episode", tree_episode)
    oracle = []
    X, _, _ = dg.generate_batch(
        1,
        16,
        2,
        'reg',
        rng=np.random.default_rng(0),
        tree_prior_prob=1.0,
        context_rows=context_rows,
        oracle_sink=oracle,
    )

    assert np.max(np.abs(X)) <= 1e4
    assert oracle[0]["oracle_exact"] is None
    assert oracle[0]["y_clean"] is None
    assert oracle[0]["transformation_lineage"] == ["final_feature_bound"]


def test_realistic_augmentation_cannot_escape_final_feature_contract():
    X, y, _ = dg.generate_batch(
        24, 256, 32, 'reg', rng=np.random.default_rng(123),
        augment_v4=True, synth_v5=True, synth_v5_mixture=True,
        realistic_augmentation_prob=1.0, context_rows=128)
    assert X.dtype == np.float32
    assert np.nanmax(np.abs(X)) <= 1e4
    assert not np.isinf(X).any()
    assert np.isfinite(y).all()


def test_health_stats_tolerate_all_nan_columns_with_warnings_as_errors():
    X = np.array([
        [np.nan, 0.0],
        [np.nan, 1.0],
        [np.nan, 2.0],
    ])
    y = np.arange(3, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        stats = dg._compute_health_stats_fast(X, y, context_rows=3)
        healthy = dg._passes_basic_episode_health(X, y, context_rows=3)

    assert stats["nonconstant_features"] == 1
    assert stats["const_feature_frac"] == 0.5
    assert healthy

    subnormal = np.array([[0.0], [np.nextafter(0.0, 1.0)]])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert not dg._finite_column_variation(subnormal).any()


@pytest.mark.parametrize(
    ("n_samples", "n_features", "context_rows", "seed"),
    [
        (128, 16, 1, 21),
        (512, 32, 64, 4180),
    ],
)
def test_context_fitted_target_transform_stays_finite_without_warnings(
        n_samples, n_features, context_rows, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        data = dg.generate_dataset(
            n_samples,
            n_features,
            'reg',
            rng=np.random.default_rng(seed),
            augment_v4=True,
            synth_v5=True,
            synth_v5_mixture=True,
            enhanced_missingness=True,
            rich_reg_targets=True,
            context_rows=context_rows,
        )

    assert np.isfinite(data["y"]).all()


def test_missingness_skips_all_nan_context_driver_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        data = dg.generate_dataset(
            128,
            16,
            'reg',
            rng=np.random.default_rng(14),
            augment_v4=True,
            synth_v5=True,
            synth_v5_mixture=True,
            enhanced_missingness=True,
            rich_reg_targets=True,
            context_rows=1,
        )

    assert np.isfinite(data["y"]).all()


def test_generate_batch_does_not_mutate_warning_filters(monkeypatch):
    def tree_episode(
            n_samples, n_features, task_type, n_classes, rng,
            context_rows=None, **_kwargs):
        del task_type, n_classes, rng, context_rows
        X = np.arange(
            n_samples * n_features, dtype=np.float32,
        ).reshape(n_samples, n_features)
        y = np.linspace(-1.0, 1.0, n_samples, dtype=np.float32)
        return {"X": X, "y": y, "filtered": False, "meta": {}}

    monkeypatch.setattr(dg, "_generate_tree_prior_episode", tree_episode)
    with warnings.catch_warnings():
        filters_before = list(warnings.filters)
        dg.generate_batch(
            1,
            32,
            4,
            'reg',
            rng=np.random.default_rng(0),
            tree_prior_prob=1.0,
            realistic_augmentation_prob=1.0,
            y_transform_prob=1.0,
            context_rows=16,
        )
        assert warnings.filters == filters_before
