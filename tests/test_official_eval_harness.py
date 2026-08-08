import json

import numpy as np
import pandas as pd
import pytest

from synthefy_nori.evaluation.cli import (
    DEFAULT_CONFIG, EXPECTED_CONFIG_SHA256, _current_invocation_rows, _file_sha256,
)
from synthefy_nori.evaluation.harness import (
    OFFICIAL_ALLOW_SUBSAMPLE,
    OFFICIAL_ELEMENTS_BUDGET,
    OFFICIAL_PROTOCOL,
    run_benchmark,
)
from synthefy_nori.evaluation.models import ModelEntry, ModelRegistry
from synthefy_nori.evaluation.protocol import BenchmarkEvalUnit, MaterializedSplit


class _Loader:
    name = "toy"
    ctx_cap = 2

    def __init__(self, fingerprint="toy-v1"):
        self._fingerprint = fingerprint

    def fingerprint(self):
        return self._fingerprint

    def units(self):
        yield BenchmarkEvalUnit(source=self.name, dataset="data")

    def materialize(self, unit):
        return MaterializedSplit(
            X_train=np.array([[1.0], [np.nan], [5.0]], dtype=np.float32),
            y_train=np.array([1.0, 2.0, 3.0]),
            X_test=np.array([[np.nan], [8.0]], dtype=np.float32),
            y_test=np.array([2.0, 4.0]),
            n_features=1,
        )


class _MixedLoader(_Loader):
    ctx_cap = None

    def units(self):
        yield BenchmarkEvalUnit(source=self.name, dataset="bad")
        yield BenchmarkEvalUnit(source=self.name, dataset="good")

    def materialize(self, unit):
        if unit.dataset == "bad":
            raise OSError("fixture is unavailable")
        return super().materialize(unit)


class _Wrapper:
    name = "toy-model"
    device_str = "cpu"

    def __init__(self):
        self.calls = 0
        self.seen = None

    def predict_regression(self, X_train, y_train, X_test):
        self.calls += 1
        self.seen = (X_train.copy(), X_test.copy())
        return np.full(len(X_test), y_train.mean())

    def cleanup(self):
        pass


class _NonFiniteWrapper(_Wrapper):
    def __init__(self, prediction):
        super().__init__()
        self.prediction = prediction

    def predict_regression(self, X_train, y_train, X_test):
        self.calls += 1
        return np.asarray(self.prediction, dtype=np.float64)


class _FailingWrapper(_Wrapper):
    model_path = "/private/checkpoints/model.pt"

    def predict_regression(self, X_train, y_train, X_test):
        raise OSError(f"cannot read {self.model_path}")


def _registry(wrapper):
    registry = ModelRegistry(device="cpu")
    registry.register(
        ModelEntry(
            name=wrapper.name,
            wrapper=wrapper,
            model_type="custom",
            metadata={
                "checkpoint_sha256": "a" * 64,
                "reg_config_sha256": "b" * 64,
                "memory_policy": {
                    "elements_budget": OFFICIAL_ELEMENTS_BUDGET,
                    "allow_subsample": OFFICIAL_ALLOW_SUBSAMPLE,
                },
            },
        )
    )
    return registry


def test_bundled_official_config_matches_identity():
    assert _file_sha256(DEFAULT_CONFIG) == EXPECTED_CONFIG_SHA256


def test_harness_records_identity_caps_and_resumes(tmp_path):
    wrapper = _Wrapper()
    output = tmp_path / "results.jsonl"
    frame = run_benchmark([_Loader()], _registry(wrapper), out_jsonl=str(output))

    assert wrapper.calls == 1
    assert frame.loc[0, "protocol"] == OFFICIAL_PROTOCOL
    assert frame.loc[0, "n_train"] == 2
    assert frame.loc[0, "elements_budget"] == OFFICIAL_ELEMENTS_BUDGET
    assert frame.loc[0, "allow_subsample"] == OFFICIAL_ALLOW_SUBSAMPLE
    assert np.isfinite(wrapper.seen[0]).all()
    assert np.isfinite(wrapper.seen[1]).all()
    record = json.loads(output.read_text())
    assert record["checkpoint_sha256"] == "a" * 64
    assert record["rng_mode"] == "per_unit"

    run_benchmark([_Loader()], _registry(wrapper), out_jsonl=str(output))
    assert wrapper.calls == 1
    assert len(output.read_text().splitlines()) == 1



def test_changed_data_fingerprint_is_a_distinct_resume_identity(tmp_path):
    wrapper = _Wrapper()
    output = tmp_path / "results.jsonl"
    run_benchmark([_Loader("toy-v1")], _registry(wrapper), out_jsonl=str(output))
    run_benchmark([_Loader("toy-v2")], _registry(wrapper), out_jsonl=str(output))
    assert wrapper.calls == 2
    assert len(output.read_text().splitlines()) == 2


def test_truncated_final_jsonl_record_is_removed_before_resume(tmp_path):
    wrapper = _Wrapper()
    output = tmp_path / "results.jsonl"
    run_benchmark([_Loader()], _registry(wrapper), out_jsonl=str(output))
    with output.open("a") as sink:
        sink.write('{"partial"')
    run_benchmark([_Loader()], _registry(wrapper), out_jsonl=str(output))

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert wrapper.calls == 1
    assert [record["seed"] for record in records] == [0]


def test_valid_final_jsonl_record_gets_newline_before_resume(tmp_path):
    wrapper = _Wrapper()
    output = tmp_path / "results.jsonl"
    run_benchmark([_Loader()], _registry(wrapper), out_jsonl=str(output))
    output.write_text(output.read_text().rstrip("\n"))
    run_benchmark([_Loader()], _registry(wrapper), out_jsonl=str(output))

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert wrapper.calls == 1
    assert [record["seed"] for record in records] == [0]


def test_complete_malformed_jsonl_record_is_not_ignored(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text('{"broken"}\n')
    with pytest.raises(json.JSONDecodeError):
        run_benchmark([_Loader()], _registry(_Wrapper()), out_jsonl=str(output))


def test_materialization_failure_is_recorded_and_next_unit_runs(tmp_path):
    wrapper = _Wrapper()
    frame = run_benchmark(
        [_MixedLoader()],
        _registry(wrapper),
        out_jsonl=str(tmp_path / "results.jsonl"),
    )

    assert len(frame) == 2
    assert "MaterializationError" in frame.loc[frame["dataset"] == "bad", "error"].item()
    assert frame.loc[frame["dataset"] == "good", "error"].isna().all()
    assert wrapper.calls == 1


def test_current_invocation_rows_excludes_stale_resume_records(tmp_path):
    loader = _Loader()
    registry = _registry(_Wrapper())
    frame = run_benchmark([loader], registry, out_jsonl=str(tmp_path / "results.jsonl"))
    stale = frame.copy()
    stale["source_tree_sha256"] = "stale"
    combined = pd.concat([frame, stale], ignore_index=True)
    selected = [(loader, next(loader.units()))]

    current = _current_invocation_rows(
        combined,
        selected=selected,
        registry=registry,
        reg_config_sha256="b" * 64,
    )
    assert len(current) == 1
    assert current.iloc[0]["source_tree_sha256"] != "stale"


def test_current_invocation_rows_rejects_missing_results():
    loader = _Loader()
    registry = _registry(_Wrapper())
    selected = [(loader, next(loader.units()))]

    with pytest.raises(RuntimeError, match="found 0 of 1"):
        _current_invocation_rows(
            pd.DataFrame(),
            selected=selected,
            registry=registry,
            reg_config_sha256="b" * 64,
        )


def test_partial_non_finite_predictions_are_dropped_pairwise(tmp_path):
    wrapper = _NonFiniteWrapper([2.0, np.nan])
    frame = run_benchmark([_Loader()], _registry(wrapper), out_jsonl=str(tmp_path / "results.jsonl"))
    assert pd.isna(frame.loc[0, "r2"])
    assert frame.loc[0, "rmse"] == 0.0
    assert frame.loc[0, "mae"] == 0.0
    assert pd.isna(frame.loc[0, "error"])


def test_all_non_finite_predictions_produce_null_metrics(tmp_path):
    wrapper = _NonFiniteWrapper([np.nan, np.inf])
    frame = run_benchmark([_Loader()], _registry(wrapper), out_jsonl=str(tmp_path / "results.jsonl"))
    assert frame.loc[0, ["r2", "rmse", "mae"]].isna().all()
    assert pd.isna(frame.loc[0, "error"])


def test_single_finite_pair_has_null_r2_and_finite_errors(tmp_path):
    wrapper = _NonFiniteWrapper([2.0])
    loader = _Loader()
    loader.materialize = lambda unit: _Loader().materialize(unit).model_copy(
        update={"X_test": np.array([[1.0]]), "y_test": np.array([2.0])}
    )
    frame = run_benchmark([loader], _registry(wrapper), out_jsonl=str(tmp_path / "results.jsonl"))
    assert pd.isna(frame.loc[0, "r2"])
    assert frame.loc[0, "rmse"] == 0.0
    assert frame.loc[0, "mae"] == 0.0
    assert pd.isna(frame.loc[0, "error"])


def test_harness_rejects_noncanonical_memory_policy(tmp_path):
    registry = _registry(_Wrapper())
    registry.get("toy-model").metadata["memory_policy"]["elements_budget"] = 2_000_000

    with pytest.raises(ValueError, match="official memory policy"):
        run_benchmark([_Loader()], registry, out_jsonl=str(tmp_path / "results.jsonl"))


def test_model_error_omits_local_checkpoint_path(tmp_path):
    frame = run_benchmark(
        [_Loader()],
        _registry(_FailingWrapper()),
        out_jsonl=str(tmp_path / "results.jsonl"),
    )

    assert "/private" not in frame.loc[0, "error"]
    assert "<checkpoint>" in frame.loc[0, "error"]


def test_registry_omits_local_paths_from_publishable_metadata():
    registry = ModelRegistry(device="cpu")
    registry.add_checkpoint(
        "local",
        "/private/checkpoints/model.pt",
        device="cpu",
        reg_config="/private/config.json",
        memory_policy={
            "elements_budget": OFFICIAL_ELEMENTS_BUDGET,
            "allow_subsample": OFFICIAL_ALLOW_SUBSAMPLE,
        },
        metadata={
            "checkpoint_path": "/private/checkpoints/model.pt",
            "reg_config_path": "/private/config.json",
            "checkpoint_sha256": "c" * 64,
            "memory_policy": {
                "elements_budget": 2_000_000,
                "allow_subsample": True,
            },
        },
    )

    metadata = registry.get("local").metadata
    assert metadata["checkpoint_sha256"] == "c" * 64
    assert not {
        "model_path",
        "checkpoint_path",
        "reg_config",
        "reg_config_path",
    } & metadata.keys()
    assert metadata["memory_policy"] == {
        "elements_budget": OFFICIAL_ELEMENTS_BUDGET,
        "allow_subsample": OFFICIAL_ALLOW_SUBSAMPLE,
    }
