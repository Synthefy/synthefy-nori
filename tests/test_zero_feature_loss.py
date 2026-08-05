import pytest
import torch

from synthefy_nori.training.loss import compute_ccmm_loss


def _model_output(feature_pred):
    return {
        "cls_output": torch.empty(2, 3, 0),
        "reg_output": torch.zeros(2, 3, 1),
        "feature_pred": feature_pred,
        "process_config": {
            "n_x_padding": 0,
            "num_used_features": None,
            "mean_for_normalization": None,
            "std_for_normalization": None,
            "features_per_group": 1,
        },
    }


def _loss(feature_pred, feature_loss_weight):
    return compute_ccmm_loss(
        _model_output(feature_pred),
        y_true=torch.ones(2, 3),
        x_original=torch.zeros(2, 5, 4),
        feature_mask=torch.zeros(2, 5, 4, dtype=torch.bool),
        task_type="reg",
        feature_loss_weight=feature_loss_weight,
        regression_loss="mse",
    )


def test_zero_feature_loss_does_not_require_decoder_output():
    loss, metrics = _loss(feature_pred=None, feature_loss_weight=0.0)

    assert loss.item() == pytest.approx(10.0)
    assert metrics["feat_loss"] == 0.0


def test_positive_feature_loss_requires_decoder_output():
    with pytest.raises(ValueError, match="feature_pred is required"):
        _loss(feature_pred=None, feature_loss_weight=0.1)
