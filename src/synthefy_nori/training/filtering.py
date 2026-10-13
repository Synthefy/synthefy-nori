"""Point predictions shared by CPU and batched frozen-model filters."""

import torch

from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple


def decode_filter_regression_mean(output: torch.Tensor, model) -> torch.Tensor:
    """Decode regression outputs in the filter's context-normalized units.

    Loaded Nori models resolve historical head metadata themselves. Only a
    scalar output is unambiguous when metadata is absent; never interpret an
    unknown multi-output head as quantiles or invent its probability grid.
    """
    if output.ndim != 3 or output.shape[-1] < 1:
        raise ValueError("ICL filter expects regression output [batch, query, channels]")
    width = output.shape[-1]
    if width != getattr(model, "num_reg_quantiles", width):
        raise ValueError("ICL filter output width does not match loaded head metadata")
    head = getattr(model, "regression_loss", "mse" if width == 1 else None)
    if head == "pinball":
        levels = getattr(model, "regression_quantiles", None)
        if levels is None:
            raise ValueError("ICL quantile filter requires loaded regression_quantiles")
        return quantile_dist_mean_simple(
            output, torch.as_tensor(levels, device=output.device), enforce_monotone_first=True
        )
    if head == "bar_distribution":
        borders = getattr(model, "bar_borders_buffer", None)
        if borders is None or borders.ndim != 1 or len(borders) != width + 1:
            raise ValueError("ICL bar filter requires borders matching its output width")
        borders = borders.to(device=output.device, dtype=torch.float32)
        if not torch.isfinite(borders).all() or not (borders[1:] > borders[:-1]).all():
            raise ValueError("ICL bar filter requires finite, strictly increasing borders")
        centers = (borders[:-1] + borders[1:]) * 0.5
        return (output.float().softmax(dim=-1) * centers).sum(dim=-1)
    if head in ("mse", "smooth_l1", "huber") and width == 1:
        return output[..., 0]
    raise ValueError(f"Unsupported ICL filter regression head: {head!r} with {width} outputs")
