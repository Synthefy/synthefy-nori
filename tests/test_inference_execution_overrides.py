"""Execution-only inference speedups on NoriPredictor.

Two settings, neither of which may change what the model predicts:

* ``skip_unused_feature_decoder`` - skip the feature decoder when its output
  cannot be read (mask_prediction off). Default ON, so the parity test below
  is the one that has to hold.
* ``native_rms_norm`` - fused RMSNorm kernel. Default OFF because it is not
  bit-identical.

Both are applied per call and undone afterwards: ``NoriPredictor(model=...)``
may be handed a module the caller also trains with, and a leaked
``_skip_feature_decoder`` would silently zero its feature-reconstruction loss.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from synthefy_nori.inference.predictor import NoriPredictor
from synthefy_nori.model.layer import RMSNorm


def _stub_predictor(model, *, mask_prediction=False,
                    skip_unused_feature_decoder=True, native_rms_norm=None):
    """A NoriPredictor with only the attributes _execution_overrides reads.

    Bypasses __init__ so the test stays on CPU and off the config/pipeline
    machinery, which is irrelevant to the override contract.
    """
    predictor = NoriPredictor.__new__(NoriPredictor)
    predictor.model = model
    predictor.mask_prediction = mask_prediction
    predictor.skip_unused_feature_decoder = skip_unused_feature_decoder
    predictor.native_rms_norm = native_rms_norm
    return predictor


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_a = RMSNorm((4,))
        self.inner = nn.Sequential(RMSNorm((4,)), nn.Linear(4, 4))


def test_constructor_defaults():
    import inspect

    params = inspect.signature(NoriPredictor.__init__).parameters
    assert params["skip_unused_feature_decoder"].default is True
    # None = inherit the model's setting; see test_inherits_* below.
    assert params["native_rms_norm"].default is None


def test_skip_applied_then_restored():
    model = _TinyModule()
    predictor = _stub_predictor(model)

    assert getattr(model, "_skip_feature_decoder", False) is False
    with predictor._execution_overrides():
        assert model._skip_feature_decoder is True
    assert model._skip_feature_decoder is False


def test_skip_not_applied_when_mask_prediction_is_on():
    """The imputation path reads feature_pred, so it must never be skipped."""
    model = _TinyModule()
    predictor = _stub_predictor(model, mask_prediction=True)

    with predictor._execution_overrides():
        assert getattr(model, "_skip_feature_decoder", False) is False


def test_skip_respects_opt_out():
    model = _TinyModule()
    predictor = _stub_predictor(model, skip_unused_feature_decoder=False)

    with predictor._execution_overrides():
        assert getattr(model, "_skip_feature_decoder", False) is False


def test_state_restored_on_exception():
    model = _TinyModule()
    predictor = _stub_predictor(model, native_rms_norm=True)

    with pytest.raises(RuntimeError):
        with predictor._execution_overrides():
            assert model._skip_feature_decoder is True
            assert model.norm_a.use_native is True
            raise RuntimeError("boom")

    assert model._skip_feature_decoder is False
    assert model.norm_a.use_native is False
    assert model.inner[0].use_native is False


def test_native_rms_norm_reaches_every_rms_module():
    model = _TinyModule()
    predictor = _stub_predictor(model, native_rms_norm=True)

    with predictor._execution_overrides():
        assert model.norm_a.use_native is True
        assert model.inner[0].use_native is True
    assert model.norm_a.use_native is False
    assert model.inner[0].use_native is False


def test_native_rms_norm_can_be_forced_off():
    """False must force the decomposed path even on a model already using the
    fused kernel -- otherwise opting out is impossible on the model= path."""
    model = _TinyModule()
    model.norm_a.use_native = True
    model.inner[0].use_native = True
    predictor = _stub_predictor(model, native_rms_norm=False)

    with predictor._execution_overrides():
        assert model.norm_a.use_native is False
        assert model.inner[0].use_native is False
    # and restored afterwards
    assert model.norm_a.use_native is True
    assert model.inner[0].use_native is True


def test_inherits_model_setting_by_default():
    """Default None must not override an explicit load_model(native=False)."""
    off = _TinyModule()
    with _stub_predictor(off)._execution_overrides():
        assert off.norm_a.use_native is False

    on = _TinyModule()
    on.norm_a.use_native = True
    on.inner[0].use_native = True
    with _stub_predictor(on)._execution_overrides():
        assert on.norm_a.use_native is True


def test_preexisting_native_flag_is_not_clobbered():
    """A model already configured for native RMSNorm keeps it after the call."""
    model = _TinyModule()
    model.norm_a.use_native = True
    predictor = _stub_predictor(model, native_rms_norm=True)

    with predictor._execution_overrides():
        assert model.norm_a.use_native is True
    assert model.norm_a.use_native is True
    assert model.inner[0].use_native is False


@pytest.mark.slow
def test_forward_parity_reg_output_is_bit_identical():
    """Skipping the decoder must not perturb the regression output at all.

    This is what makes the default safe: the decoder's result is dropped when
    mask_prediction is off, so removing the work is not a speed/accuracy
    trade-off.
    """
    from synthefy_nori.training.cli import load_model_config
    from synthefy_nori.utils.loading import build_model

    mc = load_model_config(None)
    mc['mask_prediction'] = True
    mc['embed_dim'] = 32
    mc['hid_dim'] = 64
    mc['nlayers'] = 2
    mc['nhead'] = 2
    for sub_key in ('encoder_config_x', 'encoder_config_y'):
        sub = mc.get(sub_key, {})
        for field in ('embedding_size', 'mask_embedding_size'):
            if field in sub:
                sub[field] = 32

    model = build_model(mc).to('cpu').eval()

    torch.manual_seed(0)
    x = torch.randn(1, 16, 4)
    y = torch.randn(1, 16)
    eval_pos = 8

    def run():
        torch.manual_seed(1234)  # feature positional embeddings are re-drawn
        with torch.inference_mode():
            return model(x=x, y=y, eval_pos=eval_pos, task_type='reg')

    baseline = run()
    assert baseline['feature_pred'] is not None

    predictor = _stub_predictor(model)
    with predictor._execution_overrides():
        skipped = run()

    assert skipped['feature_pred'] is None
    assert torch.equal(baseline['reg_output'], skipped['reg_output'])
    assert torch.equal(baseline['cls_output'], skipped['cls_output'])
