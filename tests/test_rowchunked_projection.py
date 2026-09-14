"""Project real activations whole and in chunks (SynthefyPFN#118).

Reassembling slices of an already-projected tensor only tests storage. Here the
production projection runs again at each chunk size, including a partial tail.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from synthefy_nori import NoriRegressor
from synthefy_nori.model.layer import EncoderBaseLayer, MultiheadAttention


@pytest.mark.parametrize(
    "device,dtype",
    [
        pytest.param("cpu", torch.float32, id="cpu-fp32"),
        pytest.param("cpu", torch.bfloat16, id="cpu-bf16"),
        pytest.param("cuda", torch.float32, marks=pytest.mark.slow, id="cuda-fp32"),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.slow, id="cuda-bf16"),
    ],
)
@pytest.mark.parametrize("copy_kv", [False, True], ids=["mha", "shared-kv"])
@pytest.mark.parametrize("pre_norm", [False, True])
@pytest.mark.parametrize("row_chunk", [400, 128, 64])
def test_rowchunked_projection_matches_whole(device, dtype, copy_kv, pre_norm, row_chunk, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA projection coverage requires a GPU")
    torch.manual_seed(118)
    layer = EncoderBaseLayer(nhead=6, embed_dim=96, hid_dim=192).to(device).eval()
    attention = MultiheadAttention(embed_dim=96, num_heads=6, qkv_combined=False).to(device).eval()
    source = torch.randn(2, 400, 3, 96, device=device)
    norm = torch.nn.LayerNorm(96).to(device).eval() if pre_norm else None
    projected_rows = []
    original_project = attention.project_kv_cache

    def tracked_project(x, **kwargs):
        projected_rows.append(x.shape[-2])
        return original_project(x, **kwargs)

    monkeypatch.setattr(attention, "project_kv_cache", tracked_project)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=dtype, enabled=dtype == torch.bfloat16):
        whole = original_project(
            (norm(source) if norm is not None else source).transpose(1, 2), copy_first_head_kv=copy_kv
        )
        chunked = layer._project_kv_cache_rowchunked(attention, source, copy_kv, row_chunk, norm=norm)
        repeated = layer._project_kv_cache_rowchunked(attention, source, copy_kv, row_chunk, norm=norm)

    expected_rows = [min(row_chunk, 400 - start) for start in range(0, 400, row_chunk)]
    assert projected_rows == expected_rows * 2  # no full projection hidden behind slicing
    assert chunked["batch"] == whole["batch"] == 2
    assert chunked["groups"] == whole["groups"] == 3
    assert chunked["kv"].shape == whole["kv"].shape
    assert torch.equal(chunked["kv"], repeated["kv"])
    if row_chunk == 400:
        assert torch.equal(chunked["kv"], whole["kv"])
    else:
        # BF16 has ~0.8% relative spacing. Allow two rounding steps, with an
        # absolute allowance near cancellation; FP32 uses a much tighter bound.
        # These are fixture-specific regression tolerances, not an API error bound.
        rtol, atol = (2e-2, 2e-3) if dtype == torch.bfloat16 else (1e-5, 1e-6)
        torch.testing.assert_close(chunked["kv"], whole["kv"], rtol=rtol, atol=atol)


@pytest.mark.slow
def test_prediction_guarantees_with_fixed_ensemble_batching(monkeypatch):
    """Offload transports unchanged K/V; other execution paths may round differently."""
    if not torch.cuda.is_available():
        pytest.skip("cached prediction comparisons require CUDA")
    monkeypatch.setenv("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "1")
    # Same deterministic 400-context / 512-query / 8-feature fixture as #118.
    rng = random.Random(0)
    x_train = [[rng.gauss(0, 1) for _ in range(8)] for _ in range(400)]
    table = {
        "X_train": x_train,
        "y_train": [row[0] * 2.0 - row[1] for row in x_train],
        "X_test": [[rng.gauss(0, 1) for _ in range(8)] for _ in range(512)],
    }

    def predict(policy):
        model = NoriRegressor(
            model="nori-6m",
            device="cuda:0",
            memory_policy={
                "elements_budget": 4000,
                "reuse_context_cache": False,
                **({"allow_quantization": False} if policy.get("cache", True) else {}),
                "allow_subsample": False,
                **policy,
            },
        )
        model.fit(table["X_train"], table["y_train"])
        first = np.asarray(model.predict(table["X_test"]))
        repeated = np.asarray(model.predict(table["X_test"]))
        np.testing.assert_array_equal(first, repeated)
        report = model.memory_report_
        assert report["dropped_context_rows"] == 0
        assert report["cache_dtype"] == "bf16"
        assert first.shape == (512,) and np.isfinite(first).all()
        return first, report

    baseline, report = predict({})
    assert report["rung"] == "resident_bf16"
    assert report["query_chunk"] < 512
    offloaded, report = predict({"gpu_budget_absolute_gb": 0.001})
    assert report["rung"] == "offload_bf16"
    np.testing.assert_array_equal(offloaded, baseline)

    cases = [
        ({"cache": False}, "no_cache"),
        ({"gpu_budget_absolute_gb": 0.001, "offload_to_host": False}, "plain_loop"),
        *[({"context_row_chunk": rows}, "resident_bf16") for rows in (400, 128, 64)],
    ]
    for policy, expected_rung in cases:
        predictions, report = predict(policy)
        assert report["rung"] == expected_rung
        if "context_row_chunk" in policy:
            assert report["context_row_chunk"] == policy["context_row_chunk"]
        # Current 6M fixture: max measured H100 difference 1.2e-3. The mixed
        # tolerance covers BF16 rounding; it is not a universal output bound.
        # Do not assert inequality: different execution paths may agree exactly.
        np.testing.assert_allclose(predictions, baseline, rtol=5e-3, atol=5e-3)
