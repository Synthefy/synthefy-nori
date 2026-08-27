"""End-to-end: does ``memory_policy=`` actually reach inference and change what happens?

``src/synthefy_nori/inference/test_memory_policy.py`` covers the policy in isolation
(pure arithmetic, no GPU). This file covers the wiring nothing else does — the path a
real user takes:

    NoriRegressor(memory_policy=...) -> NoriPredictor -> resolve() -> forward_cached_regression

Without this, the whole public surface could be inert and every other test would still
pass, because the policy unit tests never construct an estimator.

Needs a checkpoint, so it skips when one is not reachable (no network / no HF cache).
Runs on CPU; slow but not GPU-bound.
"""

from __future__ import annotations

import numpy as np
import pytest

from synthefy_nori.inference.memory_policy import MemoryPolicy

pytestmark = pytest.mark.slow

N_TRAIN, N_TEST, N_FEATURES = 600, 400, 8

# The cached path engages only when n_test exceeds chunk_size, and
# chunk_size = max(256, budget / features - n_train). At 50_000 that is 5650 for this
# table -- larger than N_TEST, so the cache never engaged and most assertions below
# were vacuously true against rung "no_cache". 5_000 drives chunk_size to its 256-row
# floor, so 400 query rows really do stream through a cache.
SMALL_ELEMENTS_BUDGET = 5_000


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_TRAIN + N_TEST, N_FEATURES)).astype(np.float32)
    y = np.sin(x[:, 0] * 2) - x[:, 1] + 0.1 * rng.normal(size=len(x))
    return x[:N_TRAIN], y[:N_TRAIN], x[N_TRAIN:]


@pytest.fixture(scope="module")
def regressor_cls():
    """NoriRegressor, or skip when no checkpoint is reachable."""
    from synthefy_nori.api import NoriRegressor

    try:
        NoriRegressor(model="nori-6m", device="cpu")._get_predictor()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no reachable Nori checkpoint: {type(exc).__name__}: {exc}")
    return NoriRegressor


def _predict(regressor_cls, table, memory_policy):
    """Fit + predict under one memory policy, returning (predictions, report)."""
    x_train, y_train, x_test = table
    model = regressor_cls(
        model="nori-6m",
        device="cpu",
        memory_policy={**({} if memory_policy is None else memory_policy), "elements_budget": SMALL_ELEMENTS_BUDGET},
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_test), dtype=np.float64), model.memory_report_


def _policy_from_report(report):
    """Round-trip the reported dict straight back through MemoryPolicy."""
    assert set(report) == set(MemoryPolicy().model_dump())
    assert isinstance(report["attempt_history"], list)
    return MemoryPolicy(**report)


class TestPolicyReachesInference:
    def test_report_is_none_before_predicting(self, regressor_cls):
        model = regressor_cls(model="nori-6m", device="cpu")
        assert model.memory_report_ is None

    def test_default_lands_on_a_bit_exact_rung(self, regressor_cls, table):
        _, report = _predict(regressor_cls, table, None)
        assert report is not None, "memory_policy= never reached the predictor"
        assert report["rung"] == "resident_bf16", (
            f"expected the cached path to engage; got {report['rung']}. If this is "
            f'"no_cache", SMALL_ELEMENTS_BUDGET is too large for this table and the '
            f'rest of this suite is testing nothing.')
        assert _policy_from_report(report).is_bit_exact

    def test_report_round_trips_into_the_policy_type(self, regressor_cls, table):
        _, report = _predict(regressor_cls, table, None)
        assert _policy_from_report(report).rung == report["rung"]

    def test_requesting_int8_changes_the_reported_rung(self, regressor_cls, table):
        # The load-bearing assertion: a different memory_policy= must produce a different
        # decision, or the parameter is decorative.
        _, default_report = _predict(regressor_cls, table, None)
        _, int8_report = _predict(regressor_cls, table, {"cache_dtype": "int8"})
        assert int8_report["cache_dtype"] == "int8"
        assert not _policy_from_report(int8_report).is_bit_exact
        assert default_report["rung"] == "resident_bf16"
        assert int8_report["rung"] == "resident_int8"

    def test_off_preset_disables_the_cache(self, regressor_cls, table):
        _, report = _predict(regressor_cls, table, {"cache": False})
        assert report["cache"] is False
        assert report["rung"] == "no_cache"

    def test_no_cache_reports_no_budget(self, regressor_cls, table):
        # It consulted no budget, so it must not state one.
        _, report = _predict(regressor_cls, table, {"cache": False})
        assert report["gpu_budget_absolute_gb"] is None
        assert report["host_budget_absolute_gb"] is None

    def test_tiny_gpu_budget_forces_a_fallback_rung(self, regressor_cls, table):
        _, report = _predict(regressor_cls, table, {"gpu_budget_absolute_gb": 0.0})
        assert report["rung"] != "resident_bf16"


class TestPredictionsAgreeAcrossPolicies:
    def test_cached_and_uncached_agree_within_mixed_precision_tolerance(self, regressor_cls, table):
        # NOT bit-identical: the two paths reduce in a different order, which the
        # README documents at ~1.5e-3. Asserting equality here would be wrong.
        cached, cached_report = _predict(regressor_cls, table, None)
        uncached, _ = _predict(regressor_cls, table, {"cache": False})
        if cached_report["rung"] == "no_cache":
            pytest.skip("cached path not eligible on this shape; nothing to compare")
        assert np.abs(cached - uncached).max() < 5e-3

    def test_same_policy_twice_is_deterministic(self, regressor_cls, table):
        first, _ = _predict(regressor_cls, table, None)
        second, _ = _predict(regressor_cls, table, None)
        assert np.array_equal(first, second)


class TestIncoherentConfigFailsAtFit:
    def test_row_chunking_without_caching_raises_before_any_compute(self, regressor_cls, table):
        # Must fail before inference, not silently do nothing at predict time.
        from pydantic import ValidationError

        x_train, y_train, x_test = table
        model = regressor_cls(
            model="nori-6m",
            device="cpu",
            memory_policy={"cache": False, "context_row_chunk": 2048},
        )
        # fit(), not predict(): the point is to fail before any inference runs.
        with pytest.raises(ValidationError, match="not a\n?\\s*reachable configuration"):
            model.fit(x_train, y_train)

    def test_unknown_key_raises(self, regressor_cls, table):
        from pydantic import ValidationError

        x_train, y_train, x_test = table
        model = regressor_cls(model="nori-6m", device="cpu", memory_policy={"gpu_budget": 20})
        with pytest.raises(ValidationError):
            model.fit(x_train, y_train)


class TestSklearnContract:
    def test_clone_round_trips_every_memory_form(self, regressor_cls):
        from sklearn.base import clone

        for policy in (None, "exact", {"gpu_budget_frac": 0.25}, MemoryPolicy(cache_dtype="bf16")):
            estimator = regressor_cls(model="nori-6m", memory_policy=policy)
            # sklearn's param name follows the constructor argument, so the rename moves
            # this key too -- get_params()["memory"] would silently KeyError-free to nothing.
            assert clone(estimator).get_params()["memory_policy"] == policy

    def test_set_params_accepts_a_preset(self, regressor_cls):
        estimator = regressor_cls(model="nori-6m")
        estimator.set_params(memory_policy="max_context")
        assert estimator.memory_policy == "max_context"


class TestSubsamplingIsNeverSilent:
    """`allow_subsample=False` must raise rather than quietly shrink the context.

    Promoted with #257 and untested anywhere until now -- including in the internal tier,
    where `allow_subsample` appears in no test file at all. It is the one setting in this
    feature whose failure mode is a *wrong answer* rather than an error: a silently trimmed
    context still returns confident, plausible predictions computed from a fraction of the
    data the caller supplied. Nothing downstream can detect that.
    """

    def test_forbidding_the_shrink_raises_instead_of_trimming(self, regressor_cls):
        from synthefy_nori.inference.memory_policy import ContextTooLargeError

        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(3000, 8)).astype(np.float32)
        y_train = (0.5 + 0.4 * (x_train[:, 0] - x_train[:, 1])).astype(np.float64)
        x_test = rng.normal(size=(64, 8)).astype(np.float32)

        model = regressor_cls(
            model="nori-6m",
            device="cpu",
            # A budget this context cannot fit, with the shrink forbidden.
            memory_policy={"elements_budget": 2000, "allow_subsample": False},
        )
        model.fit(x_train, y_train)
        with pytest.raises(ContextTooLargeError) as excinfo:
            model.predict(x_test)
        message = str(excinfo.value)
        # The message must name the setting that forbade it and how to proceed -- the caller
        # cannot act on "context too large" alone.
        assert "allow_subsample" in message
        assert "elements_budget" in message

    def test_permitting_the_shrink_serves_and_reports_what_it_dropped(self, regressor_cls):
        rng = np.random.default_rng(0)
        x_train = rng.normal(size=(3000, 8)).astype(np.float32)
        y_train = (0.5 + 0.4 * (x_train[:, 0] - x_train[:, 1])).astype(np.float64)
        x_test = rng.normal(size=(64, 8)).astype(np.float32)

        model = regressor_cls(model="nori-6m", device="cpu", memory_policy={"elements_budget": 2000})
        model.fit(x_train, y_train)
        with pytest.warns(UserWarning, match="context subsampled"):
            predictions = model.predict(x_test)

        assert len(predictions) == len(x_test)
        # The count is the point: it is the only signal that the answer used less data than
        # was supplied, and it has to survive past the warning.
        assert model.memory_report_["dropped_context_rows"] > 0


class TestWarningVolume:
    """One user-visible predict must not emit N copies of the same warning.

    `_predict_reg_single` runs once per inference pipeline (16 on the default config),
    so a warning raised there fires 16 times per predict. Python's own per-location
    de-duplication does not save us, because `stacklevel` attributes the warning to a
    varying caller frame, and logging has no de-duplication at all. Measured 36 stderr
    lines from two predict() calls before this was fixed.
    """

    def _broken_config_predict(self, regressor_cls, table, n_calls):
        """Predict under a config guaranteed to hit the plain-loop fallback."""
        x_train, y_train, x_test = table
        model = regressor_cls(
            model="nori-6m",
            device="cpu",
            # BOTH budgets capped: capping only the GPU lets offload succeed against
            # this box's real RAM, which never reaches the plain-loop rung.
            memory_policy={
                "elements_budget": SMALL_ELEMENTS_BUDGET,
                "gpu_budget_absolute_gb": 0.001,
                "host_budget_absolute_gb": 0.0005,
            },
        )
        model.fit(x_train, y_train)
        for _ in range(n_calls):
            model.predict(x_test)
        return model

    def test_fallback_warns_once_per_call_not_once_per_pipeline(self, regressor_cls, table):
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")  # defeat dedup, so we count raw emissions
            model = self._broken_config_predict(regressor_cls, table, n_calls=1)
        if model.memory_report_["rung"] != "plain_loop":
            pytest.skip("config did not reach the plain-loop rung on this shape")
        fallbacks = [w for w in caught if "fell back to the plain chunked loop" in str(w.message)]
        assert len(fallbacks) == 1, (
            f"one predict emitted {len(fallbacks)} copies of the fallback warning; "
            f"it must be de-duplicated per call, not per inference pipeline"
        )

    def test_a_second_call_warns_again(self, regressor_cls, table):
        # Per-CALL, not per-process: each request is a fact about that request, so a
        # second predict that also falls back must say so again.
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            model = self._broken_config_predict(regressor_cls, table, n_calls=2)
        if model.memory_report_["rung"] != "plain_loop":
            pytest.skip("config did not reach the plain-loop rung on this shape")
        fallbacks = [w for w in caught if "fell back to the plain chunked loop" in str(w.message)]
        assert len(fallbacks) == 2, f"expected one per call, got {len(fallbacks)}"

    def test_log_lines_are_deduplicated_too(self, regressor_cls, table, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="synthefy_nori.inference.predictor"):
            self._broken_config_predict(regressor_cls, table, n_calls=1)
        rung_lines = [r for r in caplog.records if "serving-memory rung" in r.message]
        assert len(rung_lines) == 1, (
            f"one predict logged the rung {len(rung_lines)} times; logging has no "
            f"de-duplication of its own, so it must be done at the call site"
        )
