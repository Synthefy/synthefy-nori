"""The bundled config names are written down once, and everything routes to them.

The inference config used to record its tuning values in its own filename
(``reg_allordinal_poly10_adaptive_svd256.json``). When those values changed the name
became wrong in a dozen places at once — including the deployed variant bindings that
live one tier up in the serving registry. These guard the routing so the next rename
is one edit.
"""

import json
import pathlib

import pytest

import synthefy_nori
from synthefy_nori.configs import (
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_MODEL_CONFIG,
    config_path,
)

LEGACY_NAME = "reg_allordinal_poly10_adaptive_svd256.json"


def test_default_inference_config_ships():
    path = pathlib.Path(config_path(DEFAULT_INFERENCE_CONFIG))
    assert path.exists(), f"{DEFAULT_INFERENCE_CONFIG} is not bundled"
    cfg = json.loads(path.read_text())
    assert isinstance(cfg, list) and cfg, "the inference config is an ensemble list"


def test_default_model_config_ships():
    assert pathlib.Path(config_path(DEFAULT_MODEL_CONFIG)).exists()


def test_config_path_defaults_to_the_inference_config():
    assert config_path() == config_path(DEFAULT_INFERENCE_CONFIG)


def test_the_constants_are_publicly_exported():
    """The deprecation warning tells callers to use these, so they must be reachable."""
    assert synthefy_nori.DEFAULT_INFERENCE_CONFIG == DEFAULT_INFERENCE_CONFIG
    assert synthefy_nori.DEFAULT_MODEL_CONFIG == DEFAULT_MODEL_CONFIG


def test_every_helper_shares_one_implementation():
    """Three long-standing entry points, one resolver -- so a rename is one edit."""
    from synthefy_nori.api import config_path as api_path
    from synthefy_nori.evaluation.models import package_config_path as eval_path
    from synthefy_nori.training.config import package_config_path as training_path

    resolved = {
        api_path(DEFAULT_INFERENCE_CONFIG),
        eval_path(DEFAULT_INFERENCE_CONFIG),
        training_path(DEFAULT_INFERENCE_CONFIG),
        config_path(DEFAULT_INFERENCE_CONFIG),
    }
    assert len(resolved) == 1, f"helpers disagree on the config path: {resolved}"


def test_legacy_name_still_resolves_with_a_deprecation_warning():
    """Renamed in 0.18.0; the old name works for one minor version (removed in 0.19.0)."""
    with pytest.warns(DeprecationWarning, match="renamed"):
        legacy = config_path(LEGACY_NAME)
    assert legacy == config_path(DEFAULT_INFERENCE_CONFIG)


def test_the_new_name_does_not_warn(recwarn):
    config_path(DEFAULT_INFERENCE_CONFIG)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_no_source_file_writes_the_config_name_by_hand():
    """The point of the rename: the filename literal lives in exactly one module.

    ``configs/__init__.py`` owns both the constant and the legacy alias map; anywhere
    else spelling the name out is the duplication that made the last rename a 12-file
    edit.
    """
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "synthefy_nori"
    owner = src / "configs" / "__init__.py"
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if p != owner and DEFAULT_INFERENCE_CONFIG in p.read_text()
    ]
    assert not offenders, f"config filename hard-coded outside configs/__init__.py: {offenders}"
