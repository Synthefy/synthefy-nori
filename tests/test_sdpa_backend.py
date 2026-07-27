import pytest
import torch

import synthefy_nori.model.layer as layer_module
from synthefy_nori.model.layer import MultiheadAttention


@pytest.mark.parametrize(
    ("allow_cudnn", "expected_cudnn_enabled"),
    [(False, False), (True, True)],
)
def test_sdpa_cudnn_policy_is_scoped(monkeypatch, allow_cudnn, expected_cudnn_enabled):
    original_cudnn_enabled = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    observed_cudnn_states = []

    def fake_sdpa(q, k, v, **kwargs):
        observed_cudnn_states.append(torch.backends.cuda.cudnn_sdp_enabled())
        return q

    try:
        monkeypatch.setattr(layer_module, "_ALLOW_CUDNN_SDP", allow_cudnn)
        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

        attention = MultiheadAttention(embed_dim=8, num_heads=2)
        q = torch.randn(1, 3, 2, 4)
        kv = torch.randn(1, 5, 2, 2, 4)

        output = attention.compute_attention_by_torch(None, q, kv, None)

        assert output.shape == q.shape
        assert observed_cudnn_states == [expected_cudnn_enabled]
        assert torch.backends.cuda.cudnn_sdp_enabled()
    finally:
        torch.backends.cuda.enable_cudnn_sdp(original_cudnn_enabled)
