"""HighDimFeatureSelector's SVD fallbacks must never be silent.

Both fallbacks keep inference alive by changing what the model sees: a **fit**
failure degrades the projection to a passthrough of the raw columns, and a
**transform** failure hands ``svd_all`` a single all-zero column (every feature
gone). Both still return predictions, so a broken SVD reads as "this config
scores badly" instead of "the SVD broke".

These tests pin the guarantees: every fallback warns under ``SvdFallbackWarning``,
one ``strict_pipeline()`` makes it fatal, the category tree lets a caller escalate
all degradation or one kind, and the eval runner escalates the SVD fallback so no
eval can score a degraded run.
"""
import inspect
import warnings

import numpy as np
import pytest

from synthefy_nori import (
    ContextSubsampledWarning,
    DegradedPipelineWarning,
    SvdFallbackWarning,
    strict_pipeline,
)
from synthefy_nori.inference import preprocess as pp
from synthefy_nori.inference.preprocess import HighDimFeatureSelector

SVD_BLOCK = {"strategy": "svd_all", "svd_components": 64,
             "n_features_threshold": 256, "binary_threshold": 1.01}


def _data(n=300, p=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    return X, rng.standard_normal(n)


def _selector(**kw):
    return HighDimFeatureSelector(**SVD_BLOCK, **kw)


def _break_fit(monkeypatch):
    """Make the SVD blow up at fit time."""
    class _Boom(pp._TorchTruncatedSVD):
        def fit(self, X, y=None):
            raise np.linalg.LinAlgError("synthetic fit failure")
    monkeypatch.setattr(pp, "_TorchTruncatedSVD", _Boom)


def _break_transform(sel):
    """Make the already-fitted SVD blow up at transform time."""
    def _boom(_X):
        raise np.linalg.LinAlgError("synthetic transform failure")
    sel.svd_model_.transform = _boom


# --- transform-time failure (the all-zero column) ----------------------------

def test_transform_failure_warns_and_zeros():
    X, y = _data()
    sel = _selector()
    sel.fit(X, [], 0, y=y)
    _break_transform(sel)
    with pytest.warns(SvdFallbackWarning, match="all-zero column"):
        out, cat = sel.transform(X[:5])
    assert out.shape == (5, 1) and not out.any()   # fallback still taken
    assert cat == []


def test_transform_failure_is_fatal_under_strict_pipeline():
    X, y = _data()
    sel = _selector()
    sel.fit(X, [], 0, y=y)
    _break_transform(sel)
    with strict_pipeline(), pytest.raises(SvdFallbackWarning, match="all-zero column"):
        sel.transform(X[:5])


def test_svd_binary_transform_failure_warns():
    rng = np.random.default_rng(1)
    X = (rng.random((300, 400)) < 0.3).astype(float)   # all-binary -> svd_binary
    sel = HighDimFeatureSelector(strategy="svd_binary", n_features_threshold=256,
                                 binary_threshold=0.5, svd_components=64)
    sel.fit(X, [], 0, y=rng.standard_normal(300))
    _break_transform(sel)
    with pytest.warns(SvdFallbackWarning, match="SVD-projected columns"):
        out, _ = sel.transform(X[:5])
    assert out.shape[1] == len(sel.svd_keep_idx_)


# --- fit-time failure (passthrough of the raw columns) -----------------------

def test_fit_failure_warns_and_passes_through(monkeypatch):
    X, y = _data()
    _break_fit(monkeypatch)
    sel = _selector()
    with pytest.warns(SvdFallbackWarning, match="passthrough of all 400 raw columns"):
        sel.fit(X, [], 0, y=y)
    assert sel.passthrough_
    out, _ = sel.transform(X[:5])
    assert out.shape[1] == 400          # model sees the raw width, not 64


def test_fit_failure_is_fatal_under_strict_pipeline(monkeypatch):
    X, y = _data()
    _break_fit(monkeypatch)
    with strict_pipeline(), pytest.raises(SvdFallbackWarning, match="SVD fit failed"):
        _selector().fit(X, [], 0, y=y)


# --- the happy path stays quiet ---------------------------------------------

def test_working_svd_is_silent_even_under_strict_pipeline():
    """The guard must not fire on a healthy run, strict or not."""
    X, y = _data()
    sel = _selector()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sel.fit(X, [], 0, y=y)
        out, _ = sel.transform(X[:5])
    assert not [w for w in caught if issubclass(w.category, DegradedPipelineWarning)]
    assert out.shape == (5, 64)

    sel2 = _selector()                      # same run, now strict: still no raise
    with strict_pipeline():
        sel2.fit(X, [], 0, y=y)
        out2, _ = sel2.transform(X[:5])
    assert out2.shape == (5, 64)


# --- the category tree is the whole mechanism --------------------------------

def test_category_tree():
    """One escalation covers every fallback; a subclass covers just one."""
    assert issubclass(SvdFallbackWarning, DegradedPipelineWarning)
    assert issubclass(ContextSubsampledWarning, DegradedPipelineWarning)
    # UserWarning so callers who already filter that keep matching
    assert issubclass(DegradedPipelineWarning, UserWarning)
    assert not issubclass(SvdFallbackWarning, ContextSubsampledWarning)


def test_strict_pipeline_can_escalate_one_kind_only(monkeypatch):
    X, y = _data()
    _break_fit(monkeypatch)
    # escalating a sibling category leaves the SVD fallback merely warning
    with strict_pipeline(ContextSubsampledWarning):
        with pytest.warns(SvdFallbackWarning):
            _selector().fit(X, [], 0, y=y)
    # escalating the base class catches it
    with strict_pipeline(DegradedPipelineWarning), pytest.raises(SvdFallbackWarning):
        _selector().fit(X, [], 0, y=y)


def test_strict_pipeline_restores_filters(monkeypatch):
    """Safe in a loop: one strict prediction must not harden or silence the next."""
    X, y = _data()
    _break_fit(monkeypatch)
    before = list(warnings.filters)
    with strict_pipeline(), pytest.raises(SvdFallbackWarning):
        _selector().fit(X, [], 0, y=y)
    assert warnings.filters == before
    with pytest.warns(SvdFallbackWarning):          # back to warning
        _selector().fit(X, [], 0, y=y)


def test_no_threaded_argument_survives():
    """The knob is the warning category — not a parameter to plumb through the stack.

    Guards against reintroducing the per-step argument (or the env var before it),
    which had to be repeated in preprocess.py, predictor.py, api.py and the eval
    wrapper and kept in sync in all four.
    """
    from synthefy_nori import api
    from synthefy_nori.evaluation import models
    from synthefy_nori.inference import predictor

    for mod in (pp, predictor, api, models):
        src = inspect.getsource(mod)
        assert "on_svd_failure" not in src, f"{mod.__name__} threads on_svd_failure"
        assert "FORBID_SVD_FALLBACK" not in src, f"{mod.__name__} reads an env var"
    for cls in (HighDimFeatureSelector, predictor.NoriPredictor, api.NoriRegressor):
        assert "on_svd_failure" not in inspect.signature(cls.__init__).parameters


# --- an eval can never score a degraded run ---------------------------------

class _WarningWrapper:
    """A model whose predict degrades: warns ``category``, then predicts anyway."""

    def __init__(self, category, name="degrader"):
        self.category = category
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def device_str(self):
        return "cpu"

    def predict_regression(self, X_train, y_train, X_test):
        warnings.warn(f"Nori: synthetic {self.category.__name__}", self.category)
        return np.full(len(X_test), float(np.nanmean(y_train)), dtype=np.float64)

    def cleanup(self):
        pass


def _run_harness_with(wrapper, tmp_path):
    from synthefy_nori.evaluation.harness import (
        OFFICIAL_ALLOW_SUBSAMPLE,
        OFFICIAL_ELEMENTS_BUDGET,
        run_benchmark,
    )
    from synthefy_nori.evaluation.models import ModelEntry, ModelRegistry
    from synthefy_nori.evaluation.protocol import BenchmarkEvalUnit, MaterializedSplit

    class _Loader:
        name = "synth"

        def units(self):
            yield BenchmarkEvalUnit(source=self.name, dataset="d")

        def materialize(self, unit):
            return MaterializedSplit(
                X_train=np.zeros((5, 2), np.float32),
                y_train=np.arange(5, dtype=np.float64),
                X_test=np.zeros((3, 2), np.float32),
                y_test=np.arange(3, dtype=np.float64),
                n_features=2,
            )

    registry = ModelRegistry(device="cpu")
    registry.register(
        ModelEntry(
            name=wrapper.name,
            wrapper=wrapper,
            model_type="custom",
            metadata={
                "memory_policy": {
                    "elements_budget": OFFICIAL_ELEMENTS_BUDGET,
                    "allow_subsample": OFFICIAL_ALLOW_SUBSAMPLE,
                },
            },
        )
    )
    frame = run_benchmark([_Loader()], registry, out_jsonl=str(tmp_path / "results.jsonl"))
    return frame.iloc[-1].to_dict()


def test_harness_records_a_broken_svd_as_failed_not_scored(tmp_path):
    """The guarantee: an eval can never report a degraded pipeline as a score."""
    row = _run_harness_with(_WarningWrapper(SvdFallbackWarning), tmp_path)
    assert row["r2"] is None or np.isnan(row["r2"])       # NOT scored
    assert "SvdFallbackWarning" in (row["error"] or "")   # recorded instead


def test_harness_still_scores_a_subsampled_context(tmp_path):
    """Control: only the SVD fallback is escalated.

    Trimming context to an element budget is expected on large tables, so
    ContextSubsampledWarning must not fail the row — memory_policy=
    {'allow_subsample': False} is the knob for refusing that.
    """
    with pytest.warns(ContextSubsampledWarning):        # warns, but does not fail
        row = _run_harness_with(_WarningWrapper(ContextSubsampledWarning), tmp_path)
    assert row["error"] is None
    assert row["r2"] is not None and not np.isnan(row["r2"])


class _DistPointDegrader:
    """A second custom wrapper whose point prediction degrades."""
    name = "dist-point-degrader"
    device_str = "cpu"

    def predict_regression(self, X_train, y_train, X_test):
        warnings.warn("Nori: synthetic SvdFallbackWarning", SvdFallbackWarning)
        return np.full(len(X_test), float(np.nanmean(y_train)), dtype=np.float64)

    def cleanup(self):
        pass


def test_harness_guards_every_scored_predict(tmp_path):
    """Every scored prediction is guarded by the strict SVD filter."""
    row = _run_harness_with(_DistPointDegrader(), tmp_path)
    assert "SvdFallbackWarning" in (row["error"] or "")   # recorded as failed
    assert row["r2"] is None or np.isnan(row["r2"])       # NOT scored


# --- an escalated warning must not be swallowed as a generic failure ---------

def test_yj_ensemble_does_not_swallow_an_escalated_warning(monkeypatch):
    """A Warning IS an Exception, so blanket handlers can eat the escalation.

    ``_predict_reg``'s YJ ensemble wraps a second prediction pass in
    ``except Exception`` and falls back to the identity pass. Escalated, the
    warning must propagate instead of being printed as "[YJ] augmentation failed"
    — otherwise strict_pipeline() silently degrades back to warn-and-continue.
    """
    from synthefy_nori.inference.predictor import NoriPredictor

    calls = {"n": 0}

    def _fake_single(self, x_train, y_train, x_test, return_distribution=False):
        calls["n"] += 1
        if calls["n"] == 2:                              # the YJ pass only
            warnings.warn("Nori: synthetic SVD fit failure", SvdFallbackWarning)
        return np.full(len(x_test), 1.0)

    monkeypatch.setattr(NoriPredictor, "_predict_reg_single", _fake_single)
    pred = NoriPredictor.__new__(NoriPredictor)          # no checkpoint needed
    pred.augmentations = ["yj"]
    pred.yj_skew_threshold = 0.0                         # force the YJ branch open
    y = np.array([0.0, 0, 0, 0, 0, 0, 0, 0, 1, 40.0])    # skewed -> gate opens
    x_train, x_test = np.zeros((10, 3)), np.zeros((4, 3))

    with strict_pipeline(), pytest.raises(SvdFallbackWarning):
        pred._predict_reg(x_train, y, x_test)
    assert calls["n"] == 2, "the YJ pass must actually have run"

    calls["n"] = 0                                       # default stays resilient
    with pytest.warns(SvdFallbackWarning):
        out = pred._predict_reg(x_train, y, x_test)
    assert len(out) == 4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
