import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from synthefy_nori.evaluation.harness import OFFICIAL_SUITE_COUNTS, validate_protocol_units
from synthefy_nori.evaluation.loaders.openml_task import OpenMLTaskLoader
from synthefy_nori.evaluation.loaders.tabarena import TabArenaLoader, _tabarena_num_repeats
from synthefy_nori.evaluation.loaders.talent_native import TalentNativeLoader
from synthefy_nori.evaluation.protocol import BenchmarkEvalUnit


def _fake_units(suite, n_datasets, n_units):
    return [
        BenchmarkEvalUnit(
            source=suite,
            dataset=f"dataset-{index % n_datasets}",
            fold=index // n_datasets,
        )
        for index in range(n_units)
    ]


@pytest.mark.parametrize(
    ("suite", "n_datasets", "n_units"),
    [(suite, *counts) for suite, counts in OFFICIAL_SUITE_COUNTS.items()],
)
def test_official_coverage_contract(suite, n_datasets, n_units):
    validate_protocol_units(suite, _fake_units(suite, n_datasets, n_units))


def test_official_coverage_rejects_missing_or_duplicate_units():
    n_datasets, n_units = OFFICIAL_SUITE_COUNTS["talent"]
    units = _fake_units("talent", n_datasets, n_units)
    with pytest.raises(RuntimeError, match="protocol changed"):
        validate_protocol_units("talent", units[:-1])

    units[-1] = units[0]
    with pytest.raises(RuntimeError, match="duplicate"):
        validate_protocol_units("talent", units)


def test_official_task_lists_are_pinned_without_network():
    ctr_ids = OpenMLTaskLoader.from_ctr23().task_ids
    tabarena_ids = TabArenaLoader().task_ids
    assert len(ctr_ids) == len(set(ctr_ids)) == 35
    assert len(tabarena_ids) == len(set(tabarena_ids)) == 13


def test_openml_default_cache_uses_public_config(monkeypatch):
    default = "/private/default/openml/org/openml/www"
    config = SimpleNamespace(get_cache_directory=lambda: default)
    client = SimpleNamespace(config=config)
    monkeypatch.setattr("synthefy_nori.evaluation.loaders.openml_task._openml", lambda: client)
    assert OpenMLTaskLoader([1])._local_paths() == (default,)
    assert OpenMLTaskLoader([1], cache_dir="cache/openml")._local_paths() == ()


def test_tabarena_repeat_policy_matches_official_thresholds():
    assert _tabarena_num_repeats(2_499) == 10
    assert _tabarena_num_repeats(2_500) == 3
    assert _tabarena_num_repeats(250_000) == 3
    assert _tabarena_num_repeats(250_001) == 1


def test_talent_uses_train_plus_validation_and_preserves_nan(tmp_path):
    folder = tmp_path / "toy"
    folder.mkdir()
    (folder / "info.json").write_text(json.dumps({"task_type": "regression"}))
    arrays = {
        "N_train": np.array([[1.0], [np.nan]]),
        "N_val": np.array([[3.0]]),
        "N_test": np.array([[4.0], [5.0]]),
        "C_train": np.array([["a"], ["b"]], dtype=object),
        "C_val": np.array([["a"]], dtype=object),
        "C_test": np.array([["b"], ["unseen"]], dtype=object),
        "y_train": np.array([1.0, 2.0]),
        "y_val": np.array([3.0]),
        "y_test": np.array([4.0, 5.0]),
    }
    for name, array in arrays.items():
        np.save(folder / f"{name}.npy", array)

    loader = TalentNativeLoader(str(tmp_path), expected_datasets=1)
    unit = next(loader.units())
    split = loader.materialize(unit)

    assert split.X_train.shape == (3, 2)
    assert split.X_test.shape == (2, 2)
    assert np.isnan(split.X_train[1, 0])
    assert split.X_test[1, 1] == -1
    np.testing.assert_array_equal(split.y_train, [1.0, 2.0, 3.0])
    assert loader.ctx_cap == 10_000
    assert loader.test_cap == 20_000


@pytest.mark.slow
@pytest.mark.parametrize(
    "loader,expected_task_id",
    [
        (OpenMLTaskLoader([361617], name="openml-ctr23"), 361617),
        (TabArenaLoader([363615]), 363615),
    ],
)
def test_real_official_openml_unit_materializes(loader, expected_task_id, tmp_path):
    """Exercise one small real task through metadata, split, and featurization."""
    pytest.importorskip("openml")
    cache_dir = str(tmp_path / "openml")
    if isinstance(loader, TabArenaLoader):
        loader._openml.cache_dir = cache_dir
    else:
        loader.cache_dir = cache_dir
    unit = next(loader.units())
    split = loader.materialize(unit)

    assert unit.meta.openml_task_id == expected_task_id
    assert unit.n_folds > 1
    assert split.X_train.shape == (len(split.y_train), split.n_features)
    assert split.X_test.shape == (len(split.y_test), split.n_features)
    assert len(split.y_train) >= 2
    assert len(split.y_test) >= 1
    assert np.isfinite(split.y_train).all()
    assert np.isfinite(split.y_test).all()


@pytest.mark.slow
def test_real_talent_archive_materializes():
    """Exercise one dataset from a downloaded, hash-verified TALENT-100 archive."""
    root = Path(os.environ.get("TALENT_NATIVE_ROOT", "cache/talent/data"))
    if not root.is_dir():
        pytest.skip(f"TALENT archive is not available under {root}")
    loader = TalentNativeLoader(str(root))
    units = list(loader.units())
    split = loader.materialize(units[0])

    assert len(units) == 100
    assert split.X_train.shape == (len(split.y_train), split.n_features)
    assert split.X_test.shape == (len(split.y_test), split.n_features)
    assert len(split.y_train) >= 2
    assert len(split.y_test) >= 2
    assert np.isfinite(split.y_train).all()
    assert np.isfinite(split.y_test).all()
