"""The bundled config names are written down once, and everything routes to them.

The inference config used to record its tuning values in its own filename
(``reg_allordinal_poly10_adaptive_svd256.json``). When those values changed the name
became wrong in a dozen places at once — including the deployed variant bindings that
live one tier up in the serving registry. These guard the routing so the next rename
is one edit.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest
import torch

import synthefy_nori
from synthefy_nori.inference.predictor import NoriPredictor, read_inference_config
from synthefy_nori.api import NoriRegressor
from synthefy_nori.evaluation.models import NoriWrapper
from synthefy_nori.configs import (
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_MODEL_CONFIG,
    MODEL_V3_INFERENCE_CONFIG,
    config_path,
    get_default_inference_config_name,
    get_inference_config_digest,
)


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
    """Callers should use the stable phase names through the public package."""
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


def test_the_new_name_does_not_warn(recwarn):
    config_path(DEFAULT_INFERENCE_CONFIG)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_the_retired_0_18_name_no_longer_resolves():
    legacy = config_path("reg_allordinal_poly10_adaptive_svd256.json")
    assert not pathlib.Path(legacy).exists()


def test_no_source_file_writes_the_config_name_by_hand():
    """The point of the rename: the filename literal lives in exactly one module.

    ``configs/__init__.py`` owns the constant; anywhere else spelling the name out is
    the duplication that made the last rename a 12-file edit.
    """
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "synthefy_nori"
    owner = src / "configs" / "__init__.py"
    names = (DEFAULT_INFERENCE_CONFIG, MODEL_V3_INFERENCE_CONFIG)
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if p != owner and any(name in p.read_text() for name in names)
    ]
    assert not offenders, f"config filename hard-coded outside configs/__init__.py: {offenders}"


def test_model_v3_inference_config_ships_four_members():
    path = pathlib.Path(config_path(MODEL_V3_INFERENCE_CONFIG))
    assert path.exists(), f"{MODEL_V3_INFERENCE_CONFIG} is not bundled"
    members = json.loads(path.read_text())
    assert len(members) == 4
    # The fourth member is the "plain" view: no distribution rebalancing, everything else native.
    assert "RebalanceFeatureDistribution" not in members[3]
    assert all("RebalanceFeatureDistribution" in member for member in members[:3])


def test_pre_v3_default_config_is_unchanged():
    """Older checkpoints, including every released one, keep their exact pipeline."""
    digest = hashlib.sha256(pathlib.Path(config_path(DEFAULT_INFERENCE_CONFIG)).read_bytes()).hexdigest()
    assert digest == "6710b2f09a6c2e3070e888451ad2bace53b3ebc21ec9eafb22ca61553704ace0"


def test_default_config_is_chosen_by_model_name():
    assert get_default_inference_config_name("nori-1-5") == MODEL_V3_INFERENCE_CONFIG
    for model_name in ("nori-6m", "nori-30m", "nori-100m", "org/custom-repo", None):
        assert get_default_inference_config_name(model_name) == DEFAULT_INFERENCE_CONFIG


def _get_bundled_config_members(filename: str) -> list:
    """Return the member list of a bundled inference config."""
    return json.loads(pathlib.Path(config_path(filename)).read_text())


def _get_identity_model_storing_config(filename: str | None) -> torch.nn.Module:
    """Return a stand-in model whose checkpoint stored ``filename``'s config (``None``: stored none)."""
    model = torch.nn.Identity()
    model.stored_inference_config = (
        None if filename is None else {"name": filename, "members": _get_bundled_config_members(filename)}
    )
    return model


@pytest.mark.parametrize(
    ("model", "model_path", "inference_config", "expected_inference_config"),
    [
        ("nori-1-5", None, None, config_path(MODEL_V3_INFERENCE_CONFIG)),
        ("nori-30m", None, None, config_path(DEFAULT_INFERENCE_CONFIG)),
        (None, "local.pt", None, None),
        ("nori-1-5", "local.pt", None, config_path(MODEL_V3_INFERENCE_CONFIG)),
        ("nori-1-5", None, "caller.json", config_path(MODEL_V3_INFERENCE_CONFIG)),
        ("Synthefy/Nori-1-5", None, None, None),
    ],
)
def test_regressor_passes_its_config_and_the_one_its_name_expects(
    monkeypatch, model, model_path, inference_config, expected_inference_config
):
    """The caller's config passes through untouched; a released name adds the config to check against."""
    predictor_arguments = {}

    class _RecordingPredictor:
        def __init__(self, **kwargs):
            predictor_arguments.update(kwargs)

    monkeypatch.setattr("synthefy_nori.inference.predictor.NoriPredictor", _RecordingPredictor)
    monkeypatch.setattr(
        "synthefy_nori.api._resolve_model_path", lambda model_path, token, model: model_path or "downloaded.pt"
    )
    regressor = NoriRegressor(model=model, model_path=model_path, inference_config=inference_config, device="cpu")
    regressor._get_predictor()

    assert predictor_arguments["inference_config"] == inference_config
    assert predictor_arguments["expected_inference_config"] == expected_inference_config
    assert predictor_arguments["model_path"] == (model_path or "downloaded.pt")


@pytest.mark.parametrize(
    ("stored", "explicit", "expected", "resolved"),
    [
        (MODEL_V3_INFERENCE_CONFIG, None, MODEL_V3_INFERENCE_CONFIG, MODEL_V3_INFERENCE_CONFIG),
        (MODEL_V3_INFERENCE_CONFIG, None, None, MODEL_V3_INFERENCE_CONFIG),
        (None, None, MODEL_V3_INFERENCE_CONFIG, MODEL_V3_INFERENCE_CONFIG),
        (None, None, None, DEFAULT_INFERENCE_CONFIG),
        (MODEL_V3_INFERENCE_CONFIG, DEFAULT_INFERENCE_CONFIG, DEFAULT_INFERENCE_CONFIG, DEFAULT_INFERENCE_CONFIG),
    ],
    ids=["stored-matches-name", "stored-path-only", "name-only", "neither", "explicit-wins"],
)
def test_predictor_resolves_the_config_in_precedence_order(stored, explicit, expected, resolved):
    """Explicit config, then the checkpoint's stored config, then the name's config, then the default."""
    predictor = NoriPredictor(
        device=torch.device("cpu"),
        model=_get_identity_model_storing_config(stored),
        inference_config=None if explicit is None else config_path(explicit),
        expected_inference_config=None if expected is None else config_path(expected),
    )

    assert predictor.inference_config == _get_bundled_config_members(resolved)
    assert predictor.n_estimators == len(predictor.inference_config)


def test_a_stored_config_that_contradicts_the_model_name_raises():
    """A checkpoint pointed at by the wrong name, or a stale package, fails instead of running either config."""
    with pytest.raises(ValueError, match="stores inference config 'model_v3_inference.json'"):
        NoriPredictor(
            device=torch.device("cpu"),
            model=_get_identity_model_storing_config(MODEL_V3_INFERENCE_CONFIG),
            expected_inference_config=config_path(DEFAULT_INFERENCE_CONFIG),
        )


def test_the_config_digest_ignores_key_order_but_not_content():
    members = _get_bundled_config_members(MODEL_V3_INFERENCE_CONFIG)
    reordered = [dict(reversed(list(member.items()))) for member in members]

    assert get_inference_config_digest(reordered) == get_inference_config_digest(members)
    assert get_inference_config_digest(members) != get_inference_config_digest(
        _get_bundled_config_members(DEFAULT_INFERENCE_CONFIG)
    )


def test_eval_wrapper_without_a_config_uses_the_config_stored_in_the_checkpoint(monkeypatch):
    """The evaluation harness loads by path, so it relies on the stored config, as NoriRegressor does."""
    from synthefy_nori.utils import loading

    model = _get_identity_model_storing_config(MODEL_V3_INFERENCE_CONFIG)
    monkeypatch.setattr(loading, "load_model", lambda *args, **kwargs: model)
    wrapper = NoriWrapper("nori-1-5", "unused.pt", device="cpu")

    assert wrapper.reg_config_path is None
    assert wrapper._get_reg_predictor().n_estimators == 4


def test_predictor_without_a_config_uses_the_default_even_for_model_v3():
    """The predictor never guesses a config from the architecture; only a stored config or a name selects one."""
    model = torch.nn.Identity()
    model.two_phase_config = {"target_tokens": 2}
    predictor = NoriPredictor(device=torch.device("cpu"), model=model)

    assert predictor.inference_config == json.loads(pathlib.Path(config_path(DEFAULT_INFERENCE_CONFIG)).read_text())
    assert predictor.n_estimators == len(predictor.inference_config)
    assert len(predictor.preprocess_pipelines) == predictor.n_estimators


def test_an_explicit_config_reaches_the_predictor():
    predictor = NoriPredictor(
        device=torch.device("cpu"), model=torch.nn.Identity(), inference_config=config_path(MODEL_V3_INFERENCE_CONFIG)
    )

    assert predictor.n_estimators == 4


def test_read_inference_config_does_not_alias_the_callers_list():
    members = json.loads(pathlib.Path(config_path(DEFAULT_INFERENCE_CONFIG)).read_text())
    resolved = read_inference_config(members)
    resolved[0]["RebalanceFeatureDistribution"]["discrete_flag"] = True

    assert members[0]["RebalanceFeatureDistribution"]["discrete_flag"] is False
