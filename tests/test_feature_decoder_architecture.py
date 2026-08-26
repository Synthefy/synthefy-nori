from __future__ import annotations

import copy
import json

import pytest
import torch

from synthefy_nori.inference.predictor import NoriPredictor
from synthefy_nori.training.config import package_config_path
from synthefy_nori.training.loss import compute_ccmm_loss
from synthefy_nori.utils.loading import build_model, load_model


def _tiny_model_config(**overrides) -> dict:
    with open(package_config_path("model_base.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    decoder_overrides = overrides.pop("decoder_config", None)
    config.update(
        nlayers=1,
        embed_dim=32,
        hid_dim=64,
        nhead=2,
        mask_prediction=True,
        **overrides,
    )
    if decoder_overrides is not None:
        config["decoder_config"].update(decoder_overrides)
    for sub_key in ("encoder_config_x", "encoder_config_y"):
        sub_config = config[sub_key]
        for field in ("embedding_size", "mask_embedding_size"):
            if field in sub_config:
                sub_config[field] = 32
    return config


def test_legacy_config_keeps_decoder_and_strict_state_schema():
    config = _tiny_model_config()
    original = build_model(config)
    reloaded = build_model(copy.deepcopy(config))

    assert original.feature_decoder is not None
    assert original.regression_loss == "mse"
    assert any(key.startswith("feature_decoder.") for key in original.state_dict())
    reloaded.load_state_dict(original.state_dict(), strict=True)


def test_legacy_wide_head_is_inferred_as_evenly_spaced_pinball():
    model = build_model(_tiny_model_config(decoder_config={"num_reg_quantiles": 2}))

    assert model.regression_loss == "pinball"
    assert model.regression_quantiles == pytest.approx((1 / 3, 2 / 3))


def test_model_rejects_nonfinite_quantile_metadata():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_model(
            _tiny_model_config(
                decoder_config={
                    "num_reg_quantiles": 3,
                    "regression_loss": "pinball",
                    "regression_quantiles": [0.1, float("nan"), 0.9],
                }
            )
        )


def test_explicit_omission_removes_only_feature_decoder_parameters():
    legacy = build_model(_tiny_model_config())
    omitted = build_model(_tiny_model_config(omit_feature_decoder=True))
    decoder_parameters = sum(parameter.numel() for parameter in legacy.feature_decoder.parameters())

    assert omitted.feature_decoder is None
    assert not any(key.startswith("feature_decoder.") for key in omitted.state_dict())
    assert (
        sum(parameter.numel() for parameter in legacy.parameters())
        - sum(parameter.numel() for parameter in omitted.parameters())
        == decoder_parameters
    )


def test_omitted_decoder_forward_and_zero_weight_loss():
    model = build_model(_tiny_model_config(omit_feature_decoder=True)).eval()
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, 8, 4, generator=generator)
    y = torch.randn(1, 8, generator=generator)

    with torch.inference_mode():
        output = model(x=x, y=y, eval_pos=4, task_type="reg")

    assert output["feature_pred"] is None
    loss, loss_parts = compute_ccmm_loss(
        output,
        y[:, 4:],
        x,
        torch.zeros_like(x, dtype=torch.bool),
        "reg",
        feature_loss_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert loss_parts["feat_loss"] == 0.0

    with pytest.raises(ValueError, match="feature_pred is required"):
        compute_ccmm_loss(
            output,
            y[:, 4:],
            x,
            torch.zeros_like(x, dtype=torch.bool),
            "reg",
            feature_loss_weight=0.1,
        )


def test_predictor_rejects_imputation_with_omitted_decoder():
    model = build_model(_tiny_model_config(omit_feature_decoder=True)).eval()

    with pytest.raises(ValueError, match="requires a model with feature_decoder"):
        NoriPredictor(
            device=torch.device("cpu"),
            model=model,
            inference_config=[{}],
            mask_prediction=True,
        )


def test_checkpoint_loader_rejects_imputation_with_omitted_decoder(tmp_path):
    config = _tiny_model_config(omit_feature_decoder=True)
    model = build_model(config)
    checkpoint = tmp_path / "omitted-feature-decoder.pt"
    torch.save({"config": config, "state_dict": model.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="requires a checkpoint with feature_decoder"):
        load_model(checkpoint, mask_prediction=True)
