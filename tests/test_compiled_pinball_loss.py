import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.exc import BackendCompilerFailed

import synthefy_nori.training.loss as loss_module


@pytest.fixture(autouse=True)
def _reset_compiled_pinball_state(monkeypatch):
    monkeypatch.setattr(loss_module, "_compiled_pinball_objective", None)
    monkeypatch.setattr(loss_module, "_compiled_pinball_preflight_signatures", set())
    monkeypatch.setattr(loss_module, "_pinball_compile_failed", False)


def _inputs(batch=3, n_query=11, n_quantiles=999):
    generator = torch.Generator().manual_seed(20260819)
    target = torch.randn(batch, n_query, generator=generator)
    pred = torch.randn(batch, n_query, n_quantiles, generator=generator)
    pred[0, 0, n_quantiles // 2] = target[0, 0]
    quantiles = torch.linspace(0.001, 0.999, n_quantiles).view(1, 1, -1)
    per_ep_var = target.var(dim=1, unbiased=False).clamp(min=0.01)
    return pred, target, quantiles, per_ep_var


def _reference_objective(
    pred,
    target,
    quantiles,
    per_ep_var,
    tail_weight,
    monotonicity_weight,
    mse_weight,
):
    error = target.unsqueeze(-1) - pred
    objective = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    if tail_weight > 0:
        weights = 1.0 + tail_weight * (2.0 * (quantiles - 0.5).abs())
        objective = objective * weights
    per_episode = objective.mean(dim=(1, 2))
    if monotonicity_weight > 0:
        diff = pred[..., :-1] - pred[..., 1:]
        monotonicity = F.relu(diff).pow(2).mean(dim=-1).mean(dim=1)
        per_episode = per_episode + monotonicity_weight * monotonicity
    if mse_weight > 0:
        mean_pred = pred.mean(dim=-1)
        mse = ((mean_pred - target) ** 2).mean(dim=1) / per_ep_var
        per_episode = per_episode + mse_weight * mse
    return per_episode


def _objective_args():
    pred, target, quantiles, per_ep_var = _inputs()
    return pred, target, quantiles, per_ep_var, 0.4, 0.05, 0.1


def _backend_compiler_failure(inner_exception):
    def inductor_backend(*unused_args, **unused_kwargs):
        raise AssertionError("backend callable is only used to name the failure")

    return BackendCompilerFailed(inductor_backend, inner_exception, None)


def _cuda_error_without_loading_cudart():
    error = torch.cuda.CudaError.__new__(torch.cuda.CudaError)
    RuntimeError.__init__(error, "wrapped CUDA fault")
    return error


def test_extracted_pinball_region_preserves_output_and_prediction_gradient():
    args = _objective_args()
    pred_reference = args[0].clone().requires_grad_(True)
    pred_candidate = args[0].clone().requires_grad_(True)
    upstream = torch.tensor([0.5, -1.25, 2.0])

    reference = _reference_objective(pred_reference, *args[1:])
    candidate = loss_module._pinball_objective_per_episode(pred_candidate, *args[1:])
    reference.backward(upstream)
    candidate.backward(upstream)

    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
    torch.testing.assert_close(pred_candidate.grad, pred_reference.grad, rtol=0, atol=0)


def test_compile_dispatch_defaults_on_but_keeps_cpu_and_explicit_disable_eager(monkeypatch):
    args = _objective_args()
    compile_requests = []
    original_requested = loss_module._compiled_pinball_requested

    def tracked_requested(pred, compile_pinball_loss):
        compile_requests.append(compile_pinball_loss)
        return original_requested(pred, compile_pinball_loss)

    monkeypatch.setattr(
        loss_module,
        "_compiled_pinball_requested",
        tracked_requested,
    )
    monkeypatch.setattr(
        torch,
        "compile",
        lambda *unused_args, **unused_kwargs: pytest.fail("default path must not compile"),
    )

    actual = loss_module._apply_pinball_objective(*args)
    disabled = loss_module._apply_pinball_objective(
        *args,
        compile_pinball_loss=False,
    )
    expected = loss_module._pinball_objective_per_episode(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(disabled, expected, rtol=0, atol=0)

    assert compile_requests == [True, False]
    assert not loss_module._compiled_pinball_requested(args[0], True)


def test_compile_dispatch_keeps_small_cuda_quantile_banks_eager():
    class FakeCudaPrediction:
        is_cuda = True
        requires_grad = True

        def __init__(self, n_quantiles):
            self.shape = (2, 8, n_quantiles)

    assert not loss_module._compiled_pinball_requested(FakeCudaPrediction(255), True)
    assert loss_module._compiled_pinball_requested(FakeCudaPrediction(256), True)
    assert not loss_module._compiled_pinball_requested(FakeCudaPrediction(999), False)
    detached_pred = FakeCudaPrediction(999)
    detached_pred.requires_grad = False
    assert not loss_module._compiled_pinball_requested(detached_pred, True)
    with torch.no_grad():
        assert not loss_module._compiled_pinball_requested(FakeCudaPrediction(999), True)


def test_backend_compiler_failure_with_plain_runtime_error_falls_back(monkeypatch):
    args = _objective_args()
    attempts = 0
    failure = _backend_compiler_failure(RuntimeError("C++ compiler exited with status 1"))

    def fake_compile(function, **kwargs):
        del function, kwargs

        def compiled(*call_args):
            nonlocal attempts
            del call_args
            attempts += 1
            raise failure

        return compiled

    monkeypatch.setattr(
        loss_module,
        "_compiled_pinball_requested",
        lambda pred, compile_pinball_loss: compile_pinball_loss,
    )
    monkeypatch.setattr(torch, "compile", fake_compile)

    with pytest.warns(RuntimeWarning, match="using eager loss"):
        actual = loss_module._apply_pinball_objective(*args)
    expected = loss_module._pinball_objective_per_episode(*args)

    assert attempts == 1
    assert loss_module._pinball_compile_failed
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_plain_runtime_wrapper_around_backend_failure_is_a_compiler_failure():
    backend_failure = _backend_compiler_failure(RuntimeError("compiler subprocess failed"))
    outer_failure = RuntimeError("backend invocation failed")
    outer_failure.__cause__ = backend_failure

    assert loss_module._is_compiler_failure(outer_failure)


@pytest.mark.parametrize(
    "execution_fault",
    [
        torch.OutOfMemoryError("wrapped OOM"),
        torch.AcceleratorError("wrapped accelerator fault"),
        _cuda_error_without_loading_cudart(),
        RuntimeError("CUDA error: device-side assert triggered"),
    ],
    ids=["oom", "accelerator", "cuda-type", "cuda-runtime"],
)
def test_backend_compiler_failure_wrapping_execution_fault_is_reraised(
    monkeypatch,
    execution_fault,
):
    args = _objective_args()
    failure = _backend_compiler_failure(execution_fault)

    def fake_compile(function, **kwargs):
        del function, kwargs

        def compiled(*call_args):
            del call_args
            raise failure

        return compiled

    monkeypatch.setattr(
        loss_module,
        "_compiled_pinball_requested",
        lambda pred, compile_pinball_loss: compile_pinball_loss,
    )
    monkeypatch.setattr(torch, "compile", fake_compile)

    with pytest.raises(BackendCompilerFailed) as captured:
        loss_module._apply_pinball_objective(*args)

    assert captured.value.inner_exception is execution_fault
    assert not loss_module._pinball_compile_failed
    assert not loss_module._compiled_pinball_preflight_signatures


def test_compile_dispatch_caches_one_dynamic_fullgraph_callable(monkeypatch):
    args = list(_objective_args())
    args[0] = args[0].requires_grad_(True)
    args = tuple(args)
    compiled_calls = []
    compiled_invocations = []

    def fake_compile(function, **kwargs):
        compiled_calls.append((function, kwargs))

        def compiled(*call_args):
            compiled_invocations.append(call_args)
            return function(*call_args)

        return compiled

    monkeypatch.setattr(
        loss_module,
        "_compiled_pinball_requested",
        lambda pred, compile_pinball_loss: compile_pinball_loss,
    )
    monkeypatch.setattr(torch, "compile", fake_compile)

    first = loss_module._apply_pinball_objective(*args)
    second = loss_module._apply_pinball_objective(*args)
    expected = loss_module._pinball_objective_per_episode(*args)

    assert compiled_calls == [
        (
            loss_module._pinball_objective_per_episode,
            {"dynamic": True, "fullgraph": True},
        )
    ]
    assert len(compiled_invocations) == 3
    assert compiled_invocations[0][0] is not args[0]
    assert compiled_invocations[0][0].shape == args[0].shape
    assert compiled_invocations[0][0].requires_grad
    assert compiled_invocations[1][0] is args[0]
    assert compiled_invocations[2][0] is args[0]
    assert len(loss_module._compiled_pinball_preflight_signatures) == 1
    assert args[0].grad is None
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)


def test_backward_preflight_failure_warns_once_and_permanently_falls_back(monkeypatch):
    args = list(_objective_args())
    args[0] = args[0].requires_grad_(True)
    args = tuple(args)
    attempts = 0

    def fake_compile(function, **kwargs):
        del kwargs

        class FakeCompilerError(RuntimeError):
            pass

        FakeCompilerError.__module__ = "torch._inductor.exc"

        class FailingBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *call_args):
                del ctx
                return function(*call_args)

            @staticmethod
            def backward(ctx, grad_output):
                del ctx, grad_output
                raise FakeCompilerError("backward compiler exploded")

        def compiled(*call_args):
            nonlocal attempts
            attempts += 1
            return FailingBackward.apply(*call_args)

        return compiled

    monkeypatch.setattr(
        loss_module,
        "_compiled_pinball_requested",
        lambda pred, compile_pinball_loss: compile_pinball_loss,
    )
    monkeypatch.setattr(torch, "compile", fake_compile)

    with pytest.warns(RuntimeWarning, match="using eager loss"):
        first = loss_module._apply_pinball_objective(
            *args,
            compile_pinball_loss=True,
        )
    second = loss_module._apply_pinball_objective(
        *args,
        compile_pinball_loss=True,
    )
    expected = loss_module._pinball_objective_per_episode(*args)

    assert attempts == 1
    assert loss_module._pinball_compile_failed
    assert loss_module._compiled_pinball_objective is None
    assert not loss_module._compiled_pinball_preflight_signatures
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    first.sum().backward()
    assert args[0].grad is not None
    assert torch.isfinite(args[0].grad).all()


def test_preflight_oom_is_reraised_without_eager_retry(monkeypatch):
    args = _objective_args()
    attempts = 0

    def fake_compile(function, **kwargs):
        del function, kwargs

        def compiled(*call_args):
            nonlocal attempts
            del call_args
            attempts += 1
            raise torch.OutOfMemoryError("preflight OOM")

        return compiled

    monkeypatch.setattr(
        loss_module,
        "_compiled_pinball_requested",
        lambda pred, compile_pinball_loss: compile_pinball_loss,
    )
    monkeypatch.setattr(torch, "compile", fake_compile)

    with pytest.raises(torch.OutOfMemoryError, match="preflight OOM"):
        loss_module._apply_pinball_objective(*args, compile_pinball_loss=True)

    assert attempts == 1
    assert not loss_module._pinball_compile_failed
    assert loss_module._compiled_pinball_objective is not None
    assert not loss_module._compiled_pinball_preflight_signatures


def test_preflight_cache_keys_exact_shape_stride_and_weights(monkeypatch):
    args = list(_objective_args())
    args[0] = args[0].requires_grad_(True)
    args = tuple(args)
    preflight_calls = []
    original_preflight = loss_module._preflight_compiled_pinball

    def fake_compile(function, **kwargs):
        del kwargs
        return function

    def tracked_preflight(compiled_objective, call_args):
        preflight_calls.append(loss_module._pinball_preflight_signature(call_args))
        return original_preflight(compiled_objective, call_args)

    monkeypatch.setattr(
        loss_module,
        "_compiled_pinball_requested",
        lambda pred, compile_pinball_loss: compile_pinball_loss,
    )
    monkeypatch.setattr(loss_module, "_preflight_compiled_pinball", tracked_preflight)
    monkeypatch.setattr(torch, "compile", fake_compile)

    loss_module._apply_pinball_objective(*args, compile_pinball_loss=True)
    loss_module._apply_pinball_objective(*args, compile_pinball_loss=True)
    assert len(preflight_calls) == 1

    shape_tensors = _inputs(batch=2, n_query=7)
    shape_args = (*shape_tensors, *args[4:])
    loss_module._apply_pinball_objective(
        *shape_args,
        compile_pinball_loss=True,
    )
    assert len(preflight_calls) == 2

    strided_pred = torch.randn(
        args[0].shape[0],
        args[0].shape[2],
        args[0].shape[1],
    ).transpose(1, 2).requires_grad_(True)
    assert strided_pred.shape == args[0].shape
    assert strided_pred.stride() != args[0].stride()
    stride_args = (strided_pred, *args[1:])
    loss_module._apply_pinball_objective(
        *stride_args,
        compile_pinball_loss=True,
    )
    assert len(preflight_calls) == 3

    weight_args = (*args[:6], 0.2)
    loss_module._apply_pinball_objective(
        *weight_args,
        compile_pinball_loss=True,
    )
    assert len(preflight_calls) == 4
    assert loss_module._compiled_pinball_preflight_signatures == set(preflight_calls)


def test_compute_ccmm_loss_uses_complete_999_quantile_objective(monkeypatch):
    pred, target, quantiles, per_ep_var = _inputs()
    pred = pred.requires_grad_(True)
    compile_requests = []
    original_apply = loss_module._apply_pinball_objective

    def tracked_apply(
        pred,
        target,
        quantiles,
        per_ep_var,
        tail_weight,
        monotonicity_weight,
        mse_weight,
        compile_pinball_loss=True,
    ):
        compile_requests.append(compile_pinball_loss)
        return original_apply(
            pred,
            target,
            quantiles,
            per_ep_var,
            tail_weight,
            monotonicity_weight,
            mse_weight,
            compile_pinball_loss,
        )

    monkeypatch.setattr(loss_module, "_apply_pinball_objective", tracked_apply)
    model_output = {
        "reg_output": pred,
        "feature_pred": None,
        "process_config": {
            "n_x_padding": 0,
            "num_used_features": None,
            "mean_for_normalization": None,
            "std_for_normalization": None,
            "features_per_group": 1,
        },
    }

    actual, metrics = loss_module.compute_ccmm_loss(
        model_output,
        y_true=target,
        x_original=torch.empty(pred.shape[0], pred.shape[1], 0),
        feature_mask=torch.empty(pred.shape[0], pred.shape[1], 0, dtype=torch.bool),
        task_type="reg",
        feature_loss_weight=0.0,
        regression_loss="pinball",
        regression_quantiles=quantiles.flatten().tolist(),
        pinball_tail_weight=0.4,
        pinball_monotonicity_weight=0.05,
        pinball_mse_weight=0.1,
    )
    expected_per_episode = _reference_objective(
        pred,
        target,
        quantiles,
        per_ep_var,
        0.4,
        0.05,
        0.1,
    )

    assert compile_requests == [True]
    torch.testing.assert_close(actual, expected_per_episode.mean(), rtol=0, atol=0)
    assert metrics["y_loss"] == pytest.approx(actual.detach().item())


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_cuda_inductor_pinball_forward_and_backward():
    pred, target, quantiles, per_ep_var = _inputs(batch=2, n_query=32)
    source = pred.cuda().to(torch.bfloat16)
    eager_source = source.clone().requires_grad_(True)
    compiled_source = source.clone().requires_grad_(True)
    eager_pred = eager_source.float()
    compiled_pred = compiled_source.float()
    target = target.cuda()
    quantiles = quantiles.cuda()
    per_ep_var = per_ep_var.cuda()
    weights = (0.0, 0.05, 0.0)

    assert eager_source.is_leaf and compiled_source.is_leaf
    assert not eager_pred.is_leaf and not compiled_pred.is_leaf

    eager = loss_module._pinball_objective_per_episode(
        eager_pred,
        target,
        quantiles,
        per_ep_var,
        *weights,
    )
    eager.sum().backward()

    compiled = loss_module._apply_pinball_objective(
        compiled_pred,
        target,
        quantiles,
        per_ep_var,
        *weights,
    )
    signature = loss_module._pinball_preflight_signature(
        (compiled_pred, target, quantiles, per_ep_var, *weights)
    )
    assert loss_module._pinball_compile_failed is False
    assert loss_module._compiled_pinball_objective is not None
    assert signature in loss_module._compiled_pinball_preflight_signatures
    compiled.sum().backward()

    torch.testing.assert_close(compiled, eager, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        compiled_source.grad,
        eager_source.grad,
        rtol=1e-5,
        atol=1e-7,
    )
