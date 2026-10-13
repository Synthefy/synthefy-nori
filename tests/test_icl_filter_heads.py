from types import SimpleNamespace

import numpy as np
import pytest
import torch

from synthefy_nori.training import data_generator
from synthefy_nori.training.filtering import decode_filter_regression_mean
from synthefy_nori.training.trainer import NoriTrainer


class SyntheticRegressionHead:
    def __init__(self, regression_loss, accurate_predictions=True):
        self.regression_loss = regression_loss
        self.accurate_predictions = accurate_predictions
        self.num_reg_quantiles = {"mse": 1, "pinball": 3, "bar_distribution": 2}[regression_loss]
        self.regression_quantiles = (0.1, 0.2, 0.8)
        self.bar_borders_buffer = torch.tensor([-4.0, 0.0, 4.0])

    def __call__(self, features, targets, eval_pos, **kwargs):
        query_targets = targets[:, eval_pos:]
        predicted_mean = query_targets if self.accurate_predictions else -query_targets
        if self.regression_loss == "mse":
            output = predicted_mean.unsqueeze(-1)
        elif self.regression_loss == "pinball":
            # Nonuniform trapezoid weights: .15, .35, .50. Mean offset is zero;
            # channel zero and the arithmetic mean both give wrong answers.
            # The inaccurate head's first channel is perfect, but its decoded
            # mean is shifted by 2.55 and must fail the learnability threshold.
            offsets = [-3.0, -3.0, 3.0] if self.accurate_predictions else [0.0, 3.0, 3.0]
            output = query_targets.unsqueeze(-1) + torch.tensor(offsets)
        else:
            upper_bin_probability = (predicted_mean + 2.0) / 4.0
            output = torch.stack((1 - upper_bin_probability, upper_bin_probability), dim=-1).log()
        return {"reg_output": output}


@pytest.mark.parametrize("regression_loss", ["mse", "pinball", "bar_distribution"])
@pytest.mark.parametrize("expected_acceptance", [True, False])
def test_both_filter_paths_score_decoded_mean(monkeypatch, regression_loss, expected_acceptance):
    head = SyntheticRegressionHead(regression_loss, accurate_predictions=expected_acceptance)
    monkeypatch.setattr(data_generator, "_get_icl_filter_model", lambda model_path: head)
    targets = np.tile(np.linspace(-1.5, 1.5, 32), 2).astype(np.float32)
    features = np.column_stack((targets, targets**2))
    assert (
        data_generator._check_learnability_icl(features, targets, "test", context_rows=32, max_context=32, max_query=32)
        == expected_acceptance
    )
    trainer = object.__new__(NoriTrainer)
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(icl_filter_reg_min_r2=0.05)
    trainer._icl_filter_model = head
    assert trainer._gpu_icl_filter_limix(features[None], targets[None], context_ratio=0.5).tolist() == [
        expected_acceptance
    ]


@pytest.mark.parametrize("regression_loss", [None, "mse", "smooth_l1", "huber"])
def test_scalar_decode_is_bitwise_unchanged(regression_loss):
    output = torch.randn(2, 17, 1)
    model = (
        SimpleNamespace()
        if regression_loss is None
        else SimpleNamespace(regression_loss=regression_loss, num_reg_quantiles=1)
    )
    assert torch.equal(decode_filter_regression_mean(output, model), output[..., 0])


def test_decoder_uses_saved_nonuniform_grid_and_sorts_crossings():
    head = SyntheticRegressionHead("pinball")
    output = torch.tensor([[[3.0, -3.0, -3.0]]])
    torch.testing.assert_close(decode_filter_regression_mean(output, head), torch.zeros(1, 1))
    assert not torch.allclose(output.mean(-1), torch.zeros(1, 1))


def test_bar_decoder_uses_nonuniform_borders():
    head = SyntheticRegressionHead("bar_distribution")
    head.bar_borders_buffer = torch.tensor([-4.0, 0.0, 2.0])
    logits = torch.tensor([[[0.25, 0.75]]]).log()
    torch.testing.assert_close(decode_filter_regression_mean(logits, head), torch.tensor([[0.25]]))


@pytest.mark.parametrize(
    "head,output",
    [
        (SimpleNamespace(), torch.zeros(1, 2, 3)),
        (SyntheticRegressionHead("mse"), torch.zeros(1, 2, 3)),
        (SimpleNamespace(regression_loss="pinball"), torch.zeros(1, 2, 3)),
        (SimpleNamespace(regression_loss="pinball", regression_quantiles=(0.1, 0.9)), torch.zeros(1, 2, 3)),
        (SimpleNamespace(regression_loss="bar_distribution"), torch.zeros(1, 2, 3)),
        (SimpleNamespace(regression_loss="unknown"), torch.zeros(1, 2, 1)),
    ],
)
def test_incompatible_heads_fail_clearly(head, output):
    with pytest.raises(ValueError):
        decode_filter_regression_mean(output, head)
