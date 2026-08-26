"""Unified CCMM loss for Nori training.

The CCMM objective treats ALL masked entries uniformly. The loss sums NLL
across all masked positions:
- Discrete cells (classification target): Cross-entropy
- Continuous cells (regression target, features): MSE

The feature decoder outputs predictions in the model's normalized+grouped space,
so we transform x_original into the same space for comparison.
"""

from __future__ import annotations

import threading
import warnings

import torch
import torch.nn.functional as F


_MIN_COMPILED_PINBALL_QUANTILES = 256
_compiled_pinball_objective = None
_compiled_pinball_preflight_signatures = set()
_pinball_compile_failed = False
_pinball_compile_lock = threading.RLock()
_COMPILER_FAILURE_MODULE_PREFIXES = (
    "torch._dynamo",
    "torch._functorch",
    "torch._inductor",
)
_UNSAFE_ACCELERATOR_RUNTIME_MARKERS = (
    "out of memory",
    "cuda error",
    "device-side assert",
    "illegal memory access",
    "cublas",
    "cudnn",
    "nccl",
    "hip error",
    "mps backend",
    "xpu error",
)


def _pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor, tail_weight: float = 0.0
) -> torch.Tensor:
    """Pinball loss for predicted quantiles.

    Args:
        pred: [..., Q] predicted quantile values
        target: [...] ground truth targets
        quantiles: [1, 1, Q] quantile levels (e.g. 0.01, ..., 0.99)
        tail_weight: if > 0, upweight extreme quantiles. Weight for quantile τ:
            w_τ = 1.0 + tail_weight * (2 * |τ - 0.5|)
            So τ=0.5 gets weight 1.0 and τ=0.01/0.99 gets weight 1.0 + 0.98*tail_weight.
            Default 0.0 = uniform weighting (no tail emphasis).
    """
    error = target.unsqueeze(-1) - pred
    loss = torch.maximum(quantiles * error, (quantiles - 1.0) * error)
    if tail_weight > 0:
        w = 1.0 + tail_weight * (2.0 * (quantiles - 0.5).abs())
        loss = loss * w
    return loss


def _quantile_monotonicity_penalty(pred: torch.Tensor) -> torch.Tensor:
    """Soft penalty on non-monotone adjacent τ predictions.

    For τ-ordered pred [..., Q], penalize relu(pred[..., i] - pred[..., i+1])^2
    for adjacent i. Returns [...] (averaged over Q-1 adjacent pairs). The
    pinball loss alone allows quantile crossing — this term explicitly
    discourages it, which improves the smoothness of the τ-mean point estimate
    used at inference and reduces noise in calibration on smooth large-n
    datasets (houses, space_ga).
    """
    if pred.shape[-1] < 2:
        return pred.new_zeros(pred.shape[:-1])
    diff = pred[..., :-1] - pred[..., 1:]
    return F.relu(diff).pow(2).mean(dim=-1)


def _pinball_objective_per_episode(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    per_ep_var: torch.Tensor,
    tail_weight: float,
    monotonicity_weight: float,
    mse_weight: float,
) -> torch.Tensor:
    """Return the complete pinball objective before the batch reduction."""
    per_ep_loss = _pinball_loss(
        pred,
        target,
        quantiles,
        tail_weight=tail_weight,
    ).mean(dim=(1, 2))
    if monotonicity_weight > 0:
        mono = _quantile_monotonicity_penalty(pred).mean(dim=1)
        per_ep_loss = per_ep_loss + monotonicity_weight * mono
    if mse_weight > 0:
        mean_pred = pred.mean(dim=-1)
        mse_aux = ((mean_pred - target) ** 2).mean(dim=1) / per_ep_var
        per_ep_loss = per_ep_loss + mse_weight * mse_aux
    return per_ep_loss


def _warn_and_disable_compiled_pinball(reason: str) -> None:
    global _compiled_pinball_objective
    global _pinball_compile_failed

    with _pinball_compile_lock:
        if _pinball_compile_failed:
            return
        _compiled_pinball_objective = None
        _compiled_pinball_preflight_signatures.clear()
        _pinball_compile_failed = True
        warnings.warn(
            f"Compiled pinball loss unavailable ({reason}); using eager loss for this process.",
            RuntimeWarning,
            stacklevel=3,
        )


def _is_unsafe_execution_failure(exc: Exception) -> bool:
    """Return whether eager retry could repeat or hide an accelerator fault."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, getattr(torch, "OutOfMemoryError", ())):
        return True
    if isinstance(exc, getattr(torch, "AcceleratorError", ())):
        return True
    if isinstance(exc, getattr(torch.cuda, "CudaError", ())):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        return any(marker in message for marker in _UNSAFE_ACCELERATOR_RUNTIME_MARKERS)
    return False


def _is_compiler_failure(exc: Exception) -> bool:
    """Distinguish compiler/setup failures from unsafe execution faults."""
    pending = [exc]
    seen = set()
    saw_compiler_error = False
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if _is_unsafe_execution_failure(current):
            return False
        module = type(current).__module__
        if module.startswith(_COMPILER_FAILURE_MODULE_PREFIXES):
            saw_compiler_error = True
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        inner_exception = getattr(current, "inner_exception", None)
        if isinstance(inner_exception, Exception):
            pending.append(inner_exception)

    return saw_compiler_error or not isinstance(exc, RuntimeError)


def _compiled_pinball_requested(
    pred: torch.Tensor,
    compile_pinball_loss: bool,
) -> bool:
    return (
        compile_pinball_loss
        and torch.is_grad_enabled()
        and pred.requires_grad
        and pred.is_cuda
        and pred.shape[-1] >= _MIN_COMPILED_PINBALL_QUANTILES
    )


def _preflight_compiled_pinball(
    compiled_objective,
    args: tuple,
) -> None:
    """Compile and execute both autograd halves without touching model state."""
    pred, target, quantiles, per_ep_var, *weights = args
    probe_leaf = pred.detach().requires_grad_(True)
    probe_pred = probe_leaf if pred.is_leaf else probe_leaf + 0.0
    probe_args = (
        probe_pred,
        target.detach(),
        quantiles.detach(),
        per_ep_var.detach(),
        *weights,
    )
    probe_output = compiled_objective(*probe_args)
    torch.autograd.grad(probe_output.sum(), probe_pred)
    if pred.is_cuda:
        torch.cuda.synchronize(pred.device)


def _pinball_preflight_signature(args: tuple) -> tuple:
    tensors = args[:4]
    weights = args[4:]
    tensor_signatures = tuple(
        (
            tensor.device,
            tensor.dtype,
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        for tensor in tensors
    )
    return tensor_signatures + tuple(weights)


def _apply_pinball_objective(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
    per_ep_var: torch.Tensor,
    tail_weight: float,
    monotonicity_weight: float,
    mse_weight: float,
    compile_pinball_loss: bool = True,
) -> torch.Tensor:
    """Apply the default compiled pinball region with safe eager fallback."""
    global _compiled_pinball_objective

    args = (
        pred,
        target,
        quantiles,
        per_ep_var,
        tail_weight,
        monotonicity_weight,
        mse_weight,
    )
    if not _compiled_pinball_requested(pred, compile_pinball_loss) or _pinball_compile_failed:
        return _pinball_objective_per_episode(*args)

    with _pinball_compile_lock:
        if _pinball_compile_failed:
            return _pinball_objective_per_episode(*args)
        if _compiled_pinball_objective is None:
            _compiled_pinball_preflight_signatures.clear()
            try:
                _compiled_pinball_objective = torch.compile(
                    _pinball_objective_per_episode,
                    dynamic=True,
                    fullgraph=True,
                )
            except Exception as exc:
                if not _is_compiler_failure(exc):
                    raise
                _warn_and_disable_compiled_pinball(type(exc).__name__)
                return _pinball_objective_per_episode(*args)
        signature = _pinball_preflight_signature(args)
        if signature not in _compiled_pinball_preflight_signatures:
            try:
                _preflight_compiled_pinball(_compiled_pinball_objective, args)
            except Exception as exc:
                if not _is_compiler_failure(exc):
                    raise
                _warn_and_disable_compiled_pinball(type(exc).__name__)
                return _pinball_objective_per_episode(*args)
            _compiled_pinball_preflight_signatures.add(signature)

        compiled_objective = _compiled_pinball_objective

    try:
        return compiled_objective(*args)
    except Exception as exc:
        if not _is_compiler_failure(exc):
            raise
        _warn_and_disable_compiled_pinball(type(exc).__name__)
        return _pinball_objective_per_episode(*args)


def _bar_distribution_loss(logits: torch.Tensor, target: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss over arbitrary (possibly non-uniform) bin borders.

    The model head outputs [..., num_bars] logits. Targets (already
    context-normalized: mean 0, std 1 per episode via trainer._prepare_batch)
    are bucketized into bin indices, then CE is computed.

    Targets outside the borders range are clamped to the outermost bin —
    heavy-tail episodes still contribute useful signal.

    Args:
        logits: [..., num_bars] unnormalized class scores
        target: [...] ground-truth y values (context-normalized)
        borders: [num_bars+1] tensor of bin edges (uniform OR non-uniform,
                 e.g. N(0,1)-quantile-spaced).

    Returns:
        per-example loss tensor with shape equal to target.
    """
    num_bars = borders.shape[0] - 1
    # Bin i covers [borders[i], borders[i+1]). bucketize(right=True)-1 gives
    # the correct index even at exact-edge values.
    idx = torch.bucketize(target.float().contiguous(), borders, right=True) - 1
    idx = idx.clamp(0, num_bars - 1)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        idx.reshape(-1),
        reduction="none",
    ).reshape(target.shape)


def _bar_distribution_soft_loss(
    logits: torch.Tensor, target: torch.Tensor, borders: torch.Tensor, sigma_y: float
) -> torch.Tensor:
    """Soft cross-entropy with Gaussian-smoothed bin targets in y-space.

    Hard CE has K categorical cliffs in the loss landscape — predicting
    bin 251 when truth is bin 250 gets full penalty. Soft CE replaces the
    one-hot target with a Gaussian density centered at the true y with
    sigma_y std units. The model gets partial credit for predicting nearby
    bins, with credit decaying as Gaussian in y-space distance.

    Critically, the smoothing is in *y-space* (not "bin units") so the
    semantics are well-defined even when bins are non-uniformly spaced
    (e.g. N(0,1)-quantile borders, where bins near y=0 are much narrower
    than bins at y=±2).

    Args:
        logits: [..., num_bars] unnormalized scores
        target: [...] ground-truth y values (context-normalized)
        borders: [num_bars+1] bin edges
        sigma_y: smoothing width in y-space std units. e.g. 0.12 means
                 soft-target Gaussian has std 0.12.

    Returns:
        per-example loss tensor, shape == target.shape.
    """
    bin_centers = 0.5 * (borders[:-1] + borders[1:])  # [num_bars]
    diff = target.float().unsqueeze(-1) - bin_centers
    log_target = -(diff**2) / (2.0 * float(sigma_y) ** 2 + 1e-12)
    target_dist = torch.softmax(log_target, dim=-1)
    log_pred = torch.log_softmax(logits.float(), dim=-1)
    loss = -(target_dist * log_pred).sum(dim=-1)
    return loss


def _bar_aux_mse_loss(logits: torch.Tensor, target: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """Auxiliary MSE on the bar head's expected-value point estimate.

    Bar CE is satisfied as long as the right BIN gets high mass — but the
    expected value (Σ softmax(logits)·bin_centers) used at inference can
    still be biased if mass is concentrated bimodally. This term forces
    the expected value to be calibrated against the true target,
    regardless of the underlying distribution shape.

    Returns per-example MSE tensor (shape == target).
    """
    bin_centers = 0.5 * (borders[:-1] + borders[1:])
    probs = torch.softmax(logits.float(), dim=-1)
    expected = (probs * bin_centers).sum(dim=-1)  # [...]
    return (expected - target.float()) ** 2


def compute_ccmm_loss(
    model_output,
    y_true,
    x_original,
    feature_mask,
    task_type,
    n_classes=None,
    feature_loss_weight=0.5,
    regression_loss="mse",
    regression_loss_beta=1.0,
    regression_quantiles=None,
    pinball_tail_weight=0.0,
    pinball_monotonicity_weight: float = 0.0,
    pinball_mse_weight: float = 0.0,
    num_bars: int = 5000,
    bar_borders_low: float = -10.0,
    bar_borders_high: float = 10.0,
    bar_target_sigma: float = 0.0,
    bar_borders: torch.Tensor | None = None,
    bar_target_sigma_y: float = 0.0,
    bar_aux_mse_weight: float = 0.0,
    compile_pinball_loss: bool = True,
):
    """Compute the unified CCMM loss.

    Args:
        model_output: dict from model forward with mask_prediction=True
            - 'cls_output': [batch, n_query, max_num_classes]
            - 'reg_output': [batch, n_query, 1]
            - 'feature_pred': [batch, seq_len, n_feature_groups, features_per_group]
            - 'process_config': dict with normalization params
        y_true: [batch, n_query] ground truth targets for query rows
        x_original: [batch, seq_len, n_features] original unmasked features
        feature_mask: [batch, seq_len, n_features] bool, True where masked
        task_type: 'cls' or 'reg'
        n_classes: int, number of classes (for cls)
        regression_loss: 'mse', 'smooth_l1', 'huber', or 'pinball'
        regression_loss_beta: beta for smooth_l1/huber
        regression_quantiles: list/tuple of quantile levels for pinball loss
        compile_pinball_loss: compile the large-quantile pinball region (default: true)

    Returns:
        total_loss: scalar
        loss_dict: dict with individual loss components for logging
    """
    process_config = model_output["process_config"]
    n_x_padding = process_config["n_x_padding"]
    # num_used_features: [B, n_groups, 1]
    num_used_features = process_config["num_used_features"]
    if num_used_features is not None:
        num_used_features = num_used_features.detach()
    # mean_norm: [B, n_groups, fpg], std_norm: [B, n_groups, fpg]
    mean_norm = process_config["mean_for_normalization"]
    if mean_norm is not None:
        mean_norm = mean_norm.detach()
    std_norm = process_config["std_for_normalization"]
    if std_norm is not None:
        std_norm = std_norm.detach()
    # features_per_group: actual ValidFeatureEncoder num_features (a scalar-like tensor).
    # By construction (cli.py propagates features_per_group into the encoder's
    # num_features), this equals the model's reshape grouping dimension, so we
    # derive the grouping size `fpg` from it rather than hardcoding 2.
    model_fpg = process_config["features_per_group"]
    fpg = int(model_fpg.item()) if isinstance(model_fpg, torch.Tensor) else int(model_fpg)

    batch_size = y_true.shape[0]
    n_query = y_true.shape[1]
    # ------------------------------------------------------------------
    # 1. Y prediction loss (target column, always masked for query rows)
    # ------------------------------------------------------------------
    y_loss = torch.tensor(0.0, device=y_true.device)
    n_y_cells = 0

    if task_type == "cls":
        cls_output = model_output["cls_output"]  # [B, n_query, max_classes]
        if cls_output.shape[0] > 0:
            max_classes = cls_output.shape[-1]
            if n_classes is not None:
                max_classes = min(int(n_classes), max_classes)
                cls_output = cls_output[..., :max_classes]
            y_true_long = y_true.long().clamp(0, max_classes - 1)
            y_loss = F.cross_entropy(
                cls_output.float().reshape(-1, cls_output.shape[-1]), y_true_long.reshape(-1), reduction="sum"
            )
            n_y_cells = batch_size * n_query
    else:
        reg_output = model_output["reg_output"]  # [B, n_query, 1]
        if reg_output.shape[0] > 0:
            pred = reg_output.float()  # [B, n_query, Q]
            target = y_true.float()  # [B, n_query]
            # Normalize per-episode losses so all regression datasets contribute
            # comparably even when query-target scale varies across episodes.
            per_ep_var = target.var(dim=1, unbiased=False).clamp(min=0.01)  # [B]
            if regression_loss == "mse":
                pred = pred.squeeze(-1)
                per_ep_loss = ((pred - target) ** 2).mean(dim=1)
                per_ep_loss = per_ep_loss / per_ep_var
            elif regression_loss == "smooth_l1":
                pred = pred.squeeze(-1)
                beta = max(float(regression_loss_beta), 1e-6)
                per_ep_loss = F.smooth_l1_loss(
                    pred,
                    target,
                    reduction="none",
                    beta=beta,
                ).mean(dim=1)
                per_ep_loss = per_ep_loss / torch.sqrt(per_ep_var)
            elif regression_loss == "huber":
                pred = pred.squeeze(-1)
                delta = max(float(regression_loss_beta), 1e-6)
                per_ep_loss = F.huber_loss(
                    pred,
                    target,
                    reduction="none",
                    delta=delta,
                ).mean(dim=1)
                per_ep_loss = per_ep_loss / torch.sqrt(per_ep_var)
            elif regression_loss == "pinball":
                if regression_quantiles is None:
                    regression_quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
                quantiles = torch.as_tensor(
                    regression_quantiles,
                    device=pred.device,
                    dtype=pred.dtype,
                ).view(1, 1, -1)
                if pred.shape[-1] != quantiles.shape[-1]:
                    raise ValueError(
                        f"Pinball loss expects {quantiles.shape[-1]} quantiles, "
                        f"got reg_output last dim {pred.shape[-1]}"
                    )
                # Keep the complete quantile objective in one tensor-only
                # region so Inductor can fuse its large-Q temporaries. The
                # eager implementation retains the original operation order.
                per_ep_loss = _apply_pinball_objective(
                    pred,
                    target,
                    quantiles,
                    per_ep_var,
                    pinball_tail_weight,
                    pinball_monotonicity_weight,
                    pinball_mse_weight,
                    compile_pinball_loss,
                )
            elif regression_loss == "bar_distribution":
                # pred: [B, n_query, num_bars] logits (no squeeze)
                if pred.shape[-1] != num_bars:
                    raise ValueError(
                        f"bar_distribution expects reg head output dim = num_bars "
                        f"({num_bars}); got {pred.shape[-1]}. Did you set "
                        f"decoder_config.num_reg_quantiles = {num_bars} at model build?"
                    )
                # Resolve borders: prefer explicit bar_borders tensor (passed
                # from synthefy_nori.model.bar_borders_buffer), else fall back to uniform.
                if bar_borders is not None:
                    borders_t = bar_borders.to(pred.device)
                else:
                    borders_t = torch.linspace(
                        bar_borders_low,
                        bar_borders_high,
                        num_bars + 1,
                        device=pred.device,
                        dtype=torch.float32,
                    )
                # Resolve sigma_y: prefer explicit y-space sigma; else convert
                # legacy sigma_bins to sigma_y via mean bin width (only well-
                # defined for uniform-ish borders).
                if bar_target_sigma_y > 0:
                    sigma_y = float(bar_target_sigma_y)
                elif bar_target_sigma > 0:
                    mean_w = float((borders_t[1:] - borders_t[:-1]).mean().item())
                    sigma_y = float(bar_target_sigma) * mean_w
                else:
                    sigma_y = 0.0

                # Soft CE when sigma_y>0; hard CE otherwise.
                if sigma_y > 0:
                    per_ex_loss = _bar_distribution_soft_loss(
                        pred,
                        target,
                        borders_t,
                        sigma_y=sigma_y,
                    )
                else:
                    per_ex_loss = _bar_distribution_loss(
                        pred,
                        target,
                        borders_t,
                    )  # [B, n_query]
                per_ep_loss = per_ex_loss.mean(dim=1)
                # bar_distribution loss is CE in log-space: not episode-variance
                # normalized. Typical scale is log(num_bars) ≈ 8.5 for 5000 bars;
                # clamp to 2× that ceiling to cap pathological episodes.
                per_ep_loss = per_ep_loss.clamp(max=20.0)
                # Auxiliary MSE on expected value: forces the bar head's
                # softmax(logits)·bin_centers point estimate to be
                # calibrated against the true target. Without this, bar CE
                # is satisfied even when the expected value is biased
                # (compression failure mode for bimodal predicted
                # distributions). Episode-variance-normalized so it composes
                # cleanly with the CE term across heterogeneous y scales.
                if bar_aux_mse_weight > 0:
                    aux = _bar_aux_mse_loss(pred, target, borders_t).mean(dim=1)
                    aux_norm = aux / per_ep_var
                    per_ep_loss = per_ep_loss + bar_aux_mse_weight * aux_norm.clamp(max=10.0)
            else:
                raise ValueError(f"Unsupported regression_loss: {regression_loss}")

            if regression_loss not in ("pinball", "bar_distribution"):
                # If the model is dramatically worse than predicting the mean, the
                # gradient is usually dominated by noise rather than useful signal.
                # Not applied to pinball (raw quantile loss) or bar_distribution
                # (CE already self-capped at ~log(num_bars) + local clamp 20).
                per_ep_loss = per_ep_loss.clamp(max=10.0)
            y_loss = per_ep_loss.sum()
            n_y_cells = batch_size  # one NMSE value per episode

    if feature_loss_weight <= 0:
        y_loss_avg = y_loss / max(n_y_cells, 1)
        loss_dict = {
            "total_loss": y_loss_avg.item(),
            "y_loss": y_loss_avg.item(),
            "feat_loss": 0.0,
            "n_y_cells": n_y_cells,
            "n_feat_cells": 0,
        }
        return y_loss_avg, loss_dict

    feature_pred = model_output.get("feature_pred")
    if feature_pred is None:
        raise ValueError("feature_pred is required when feature_loss_weight is positive")

    # ------------------------------------------------------------------
    # 2. Feature reconstruction loss (masked cells only)
    # ------------------------------------------------------------------
    # feature_pred is in the model's normalized+grouped space.
    # Transform x_original into the same space:
    # 1. Pad features if needed (to make divisible by fpg)
    # 2. Reshape to [B, seq_len, n_groups, fpg]
    # 3. Apply mean-0 std-1 normalization
    # 4. Apply valid feature scaling
    # ------------------------------------------------------------------

    n_features = x_original.shape[-1]

    # Pad x_original and feature_mask to be divisible by fpg
    pad_amount = n_x_padding
    if pad_amount > 0:
        x_padded = torch.cat(
            [
                x_original,
                torch.zeros(
                    batch_size, x_original.shape[1], pad_amount, device=x_original.device, dtype=x_original.dtype
                ),
            ],
            dim=-1,
        )
        mask_padded = torch.cat(
            [
                feature_mask,
                torch.zeros(
                    batch_size, feature_mask.shape[1], pad_amount, device=feature_mask.device, dtype=feature_mask.dtype
                ).bool(),
            ],
            dim=-1,
        )
    else:
        x_padded = x_original
        mask_padded = feature_mask

    # Reshape to grouped form: [B, seq_len, n_groups, fpg]
    n_features_padded = x_padded.shape[-1]
    n_groups = n_features_padded // fpg
    x_grouped = x_padded.reshape(batch_size, -1, n_groups, fpg)
    mask_grouped = mask_padded.reshape(batch_size, -1, n_groups, fpg)

    # Apply normalization in grouped space
    # mean_norm: [B, n_groups, fpg], std_norm: [B, n_groups, fpg]
    if mean_norm is not None and std_norm is not None:
        x_normed = (x_grouped - mean_norm.unsqueeze(1)) / (std_norm.unsqueeze(1) + 1e-20)
        x_normed = x_normed.clamp(-100, 100)  # Match model's internal clipping
    else:
        x_normed = x_grouped

    # Apply valid feature scaling: sqrt(model_fpg / num_used_features)
    if num_used_features is not None:
        fpg_val = float(model_fpg) if not isinstance(model_fpg, torch.Tensor) else model_fpg.float()
        scale = torch.sqrt(fpg_val / num_used_features.float().clamp(min=1))
        # scale: [B, n_groups, 1] -> broadcast over [B, seq_len, n_groups, fpg]
        x_normed = x_normed * scale.unsqueeze(1)

    # Match dimensions with feature_pred
    seq_len = min(feature_pred.shape[1], x_normed.shape[1])
    n_groups_min = min(feature_pred.shape[2], x_normed.shape[2])

    fp = feature_pred[:, :seq_len, :n_groups_min]
    gt = x_normed[:, :seq_len, :n_groups_min]
    fm = mask_grouped[:, :seq_len, :n_groups_min]

    # MSE on masked positions only
    feat_loss = torch.tensor(0.0, device=y_true.device)
    n_feat_cells = 0

    if fm.any():
        masked_pred = fp[fm].float()
        masked_true = gt[fm].float()
        # Exclude NaN positions (truly missing values in original data)
        valid = ~torch.isnan(masked_true)
        n_valid = valid.sum()
        if n_valid > 0:
            feat_loss = F.mse_loss(masked_pred[valid], masked_true[valid], reduction="sum")
            n_feat_cells = n_valid.item()

    # ------------------------------------------------------------------
    # 3. Combined loss — normalize each term independently
    # ------------------------------------------------------------------
    y_loss_avg = y_loss / max(n_y_cells, 1)
    feat_loss_avg = feat_loss / max(n_feat_cells, 1)
    total_loss = y_loss_avg + feature_loss_weight * feat_loss_avg

    loss_dict = {
        "total_loss": total_loss.item(),
        "y_loss": y_loss_avg.item(),
        "feat_loss": feat_loss_avg.item(),
        "n_y_cells": n_y_cells,
        "n_feat_cells": n_feat_cells,
    }

    return total_loss, loss_dict
