from __future__ import annotations

import argparse
import gc
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from synthefy_nori.training import cli as training_cli
from synthefy_nori.training.cli import (
    configure_architecture_extras,
    configure_feature_decoder_architecture,
    configure_feature_loss_schedule,
    configure_regression_head,
    load_resume_configs,
    parse_quantiles,
    resolve_model_config_source,
)


def _head_args(**overrides):
    values = {
        "regression_loss": None,
        "regression_quantiles": None,
        "num_bars": None,
        "bar_borders_low": None,
        "bar_borders_high": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _feature_loss_args(**overrides):
    values = {
        "feature_loss_weight": 0.5,
        "feature_loss_weight_end": None,
        "feature_loss_decay_start_step": 0,
        "feature_loss_decay_end_step": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("raw", ["0.1,nan,0.9", "0.1,inf,0.9", "0.1,-inf,0.9"])
def test_parse_quantiles_rejects_nonfinite_levels(raw):
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        parse_quantiles(raw)


def test_resume_checkpoint_is_the_architecture_source():
    assert resolve_model_config_source(None, "/tmp/resume.pt") == "/tmp/resume.pt"
    assert resolve_model_config_source("/tmp/base.json", None) == "/tmp/base.json"


def test_different_checkpoint_and_resume_sources_are_rejected():
    with pytest.raises(ValueError, match="different files"):
        resolve_model_config_source("/tmp/base.pt", "/tmp/resume.pt")


def test_resume_metadata_load_releases_tensor_payload(monkeypatch):
    payload_refs = {}

    def fake_load(_source, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is False
        payload = torch.ones(1)
        payload_refs["tensor"] = weakref.ref(payload)
        return {
            "model_config": {"embed_dim": 16},
            "config": {"feature_loss_weight": 0.0},
            "model_state_dict": {"weight": payload},
        }

    monkeypatch.setattr(training_cli.torch, "load", fake_load)

    model_config, training_config = load_resume_configs("resume.pt")
    gc.collect()

    assert model_config == {"embed_dim": 16}
    assert training_config == {"feature_loss_weight": 0.0}
    assert payload_refs["tensor"]() is None


def test_legacy_resume_model_and_training_configs_do_not_alias(monkeypatch):
    legacy_config = {
        "embed_dim": 16,
        "decoder_config": {"num_reg_quantiles": 3},
    }
    monkeypatch.setattr(
        training_cli.torch,
        "load",
        lambda *_args, **_kwargs: {"config": legacy_config},
    )

    model_config, training_config = load_resume_configs("legacy.pt")
    model_config["decoder_config"]["num_reg_quantiles"] = 5

    assert training_config["decoder_config"]["num_reg_quantiles"] == 3


def test_resume_upgrades_legacy_head_metadata_from_training_config():
    model_config = {"decoder_config": {"num_reg_quantiles": 3}}
    training_config = SimpleNamespace(
        regression_loss="pinball",
        regression_quantiles=(0.02, 0.5, 0.98),
        num_bars=17,
        bar_borders_low=-4.0,
        bar_borders_high=6.0,
    )
    args = _head_args()

    configure_regression_head(
        model_config,
        args,
        resume_training_config=training_config,
        preserve_existing=True,
    )

    assert model_config["decoder_config"] == {
        "num_reg_quantiles": 3,
        "regression_loss": "pinball",
        "regression_quantiles": [0.02, 0.5, 0.98],
    }
    assert args.regression_quantiles == (0.02, 0.5, 0.98)


def test_resume_legacy_scalar_head_defaults_to_mse_when_metadata_is_absent():
    model_config = {"decoder_config": {"num_reg_quantiles": 1}}
    args = _head_args()

    configure_regression_head(
        model_config,
        args,
        preserve_existing=True,
    )

    assert model_config["decoder_config"] == {
        "num_reg_quantiles": 1,
        "regression_loss": "mse",
    }


def test_resume_ignores_unselected_parser_defaults_for_head_architecture():
    model_config = {
        "decoder_config": {
            "num_reg_quantiles": 3,
            "regression_loss": "pinball",
            "regression_quantiles": [0.02, 0.5, 0.98],
        }
    }
    # These are argparse's scratch-run defaults. With no corresponding CLI
    # option present, they must not rewrite a resumed pinball head to MSE.
    args = _head_args(
        regression_loss="mse",
        regression_quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        num_bars=5000,
        bar_borders_low=-10.0,
        bar_borders_high=10.0,
    )

    configure_regression_head(
        model_config,
        args,
        preserve_existing=True,
        explicit_options=set(),
    )

    assert model_config["decoder_config"] == {
        "num_reg_quantiles": 3,
        "regression_loss": "pinball",
        "regression_quantiles": [0.02, 0.5, 0.98],
    }


def test_new_run_persists_loss_and_exact_quantile_metadata():
    model_config = {"decoder_config": {}}
    args = _head_args(
        regression_loss="pinball",
        regression_quantiles=(0.03, 0.4, 0.91),
    )

    configure_regression_head(model_config, args)

    assert model_config["decoder_config"] == {
        "num_reg_quantiles": 3,
        "regression_loss": "pinball",
        "regression_quantiles": [0.03, 0.4, 0.91],
    }


def test_resume_preserves_boolean_architecture_defaults_without_flags():
    model_config = {
        "use_qassmax": False,
        "use_target_aware_embedding": False,
        "use_column_specific_y_aware": True,
    }
    args = SimpleNamespace(
        no_qassmax=False,
        no_target_aware_embedding=False,
        column_specific_y_aware=False,
    )

    configure_architecture_extras(
        model_config,
        args,
        preserve_existing=True,
    )

    assert model_config == {
        "use_qassmax": False,
        "use_target_aware_embedding": False,
        "use_column_specific_y_aware": True,
    }


def test_resume_applies_explicit_boolean_architecture_switches():
    model_config = {
        "use_qassmax": True,
        "use_target_aware_embedding": True,
        "use_column_specific_y_aware": False,
    }
    args = SimpleNamespace(
        no_qassmax=True,
        no_target_aware_embedding=True,
        column_specific_y_aware=True,
    )

    configure_architecture_extras(
        model_config,
        args,
        preserve_existing=True,
    )

    assert model_config == {
        "use_qassmax": False,
        "use_target_aware_embedding": False,
        "use_column_specific_y_aware": True,
    }


def test_omitted_head_resume_inherits_zero_feature_loss_schedule():
    args = _feature_loss_args()
    checkpoint_config = SimpleNamespace(
        feature_loss_weight=0.0,
        feature_loss_weight_end=0.0,
        feature_loss_decay_start_step=100,
        feature_loss_decay_end_step=200,
    )

    stays_zero = configure_feature_loss_schedule(
        {"omit_feature_decoder": True},
        args,
        resume_training_config=checkpoint_config,
        preserve_existing=True,
        explicit_options=set(),
    )

    assert stays_zero is True
    assert args.feature_loss_weight == 0.0
    assert args.feature_loss_weight_end == 0.0
    assert args.feature_loss_decay_start_step == 100
    assert args.feature_loss_decay_end_step == 200


def test_omitted_head_without_training_config_defaults_schedule_to_zero():
    args = _feature_loss_args()

    stays_zero = configure_feature_loss_schedule(
        {"omit_feature_decoder": True},
        args,
        resume_training_config=None,
        preserve_existing=True,
        explicit_options=set(),
    )

    assert stays_zero is True
    assert args.feature_loss_weight == 0.0
    assert args.feature_loss_weight_end is None


@pytest.mark.parametrize(
    ("option", "overrides"),
    [
        ("--feature-loss-weight", {"feature_loss_weight": 0.1}),
        ("--feature-loss-weight-end", {"feature_loss_weight_end": 0.1}),
    ],
)
def test_omitted_head_resume_rejects_explicit_positive_schedule(
    option,
    overrides,
):
    args = _feature_loss_args(**overrides)
    checkpoint_config = SimpleNamespace(
        feature_loss_weight=0.0,
        feature_loss_weight_end=0.0,
        feature_loss_decay_start_step=0,
        feature_loss_decay_end_step=10,
    )

    with pytest.raises(ValueError, match="omits feature_decoder"):
        configure_feature_loss_schedule(
            {"omit_feature_decoder": True},
            args,
            resume_training_config=checkpoint_config,
            preserve_existing=True,
            explicit_options={option},
        )


class _OmittedFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_decoder = None


class _TrainerCaptured(Exception):
    pass


def _write_omitted_head_checkpoint(path):
    torch.save(
        {
            "model_config": {
                "omit_feature_decoder": True,
                "decoder_config": {
                    "num_reg_quantiles": 1,
                    "regression_loss": "mse",
                },
                "features_per_group": 1,
                "mask_prediction": True,
                "use_qassmax": False,
                "use_target_aware_embedding": False,
                "use_column_specific_y_aware": False,
            },
            "config": {
                "regression_loss": "mse",
                "feature_loss_weight": 0.0,
                "feature_loss_weight_end": 0.0,
                "feature_loss_decay_start_step": 10,
                "feature_loss_decay_end_step": 20,
            },
            "model_state_dict": {},
        },
        path,
    )


@pytest.mark.parametrize(
    "legacy_flags",
    [
        (),
        ("--no-scale-variation", "--icl-filter-use-train-context"),
    ],
)
def test_minimal_cli_resume_carries_zero_feature_schedule_to_trainer(
    tmp_path,
    monkeypatch,
    legacy_flags,
):
    checkpoint = tmp_path / "omitted-head.pt"
    _write_omitted_head_checkpoint(checkpoint)
    captured = {}

    def capture_trainer(model, config, *, model_config, **_kwargs):
        captured["config"] = config
        captured["model_config"] = model_config
        raise _TrainerCaptured

    monkeypatch.setattr(
        training_cli,
        "build_model",
        lambda _config: _OmittedFeatureModel(),
    )
    monkeypatch.setattr(training_cli, "NoriTrainer", capture_trainer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthefy-nori-train",
            "--resume",
            str(checkpoint),
            "--device",
            "cpu",
            "--no-wandb",
            *legacy_flags,
        ],
    )

    with pytest.raises(_TrainerCaptured):
        training_cli.main()

    assert captured["model_config"]["omit_feature_decoder"] is True
    assert captured["config"].feature_loss_weight == 0.0
    assert captured["config"].feature_loss_weight_end == 0.0
    assert captured["config"].feature_loss_decay_start_step == 10
    assert captured["config"].feature_loss_decay_end_step == 20
    assert captured["config"].scale_variation is False
    assert captured["config"].icl_filter_use_train_context is True


def test_cli_rejects_positive_feature_loss_before_building_omitted_head(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint = tmp_path / "omitted-head.pt"
    _write_omitted_head_checkpoint(checkpoint)

    def unexpected_build(_config):
        pytest.fail("model construction must not run for an invalid schedule")

    monkeypatch.setattr(training_cli, "build_model", unexpected_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "synthefy-nori-train",
            "--resume",
            str(checkpoint),
            "--device",
            "cpu",
            "--no-wandb",
            "--feature-loss-weight",
            "0.1",
        ],
    )

    with pytest.raises(SystemExit):
        training_cli.main()

    assert "omits feature_decoder" in capsys.readouterr().err


def test_new_zero_feature_loss_run_persists_decoder_omission():
    model_config = {}
    args = SimpleNamespace(skip_zero_feature_decoder=True)

    configure_feature_decoder_architecture(
        model_config,
        args,
        feature_loss_stays_zero=True,
        preserve_existing=False,
    )

    assert model_config["omit_feature_decoder"] is True


def test_new_run_persists_present_decoder_when_skip_is_disabled():
    model_config = {}
    args = SimpleNamespace(skip_zero_feature_decoder=False)

    configure_feature_decoder_architecture(
        model_config,
        args,
        feature_loss_stays_zero=True,
        preserve_existing=False,
    )

    assert model_config["omit_feature_decoder"] is False


def test_legacy_resume_does_not_invent_decoder_omission():
    model_config = {}
    args = SimpleNamespace(skip_zero_feature_decoder=True)

    configure_feature_decoder_architecture(
        model_config,
        args,
        feature_loss_stays_zero=True,
        preserve_existing=True,
    )

    assert "omit_feature_decoder" not in model_config
