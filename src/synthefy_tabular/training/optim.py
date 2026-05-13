from __future__ import annotations

from collections.abc import Iterable

import torch


_MUON_EXCLUDE_SUBSTRINGS = (
    "embedding",
    "mask_embedding",
    "y_mask",
    "feature_positional_embedding",
)


def split_muon_adamw_params(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    include_embeddings: bool = False,
    include_nd: bool = False,
) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    """Split parameters into Muon-safe and AdamW fallback groups.

    Default policy: Muon operates on 2D hidden-layer weights only
    (PyTorch stock Muon requires ndim==2). Embeddings, 1D norms/biases, and
    higher-rank (attention qkv / out_proj, stored as 3D/4D) go to AdamW.

    With include_embeddings=True: 2D embedding tables also route to Muon.
    Useful because embedding tables are mathematically 2D matrices; the
    default exclusion is conservative, not mandatory.

    With include_nd=True: higher-rank (ndim>=2) tensors also route to Muon.
    Requires the downstream optimizer to support ndim>2 (our MuonND class;
    PyTorch's stock torch.optim.Muon will raise). This covers attention
    out_proj (3D: heads, head_dim, embed_dim) and qkv_proj (4D), which are
    mathematically just reshaped 2D projections.
    """

    muon_params: list[tuple[str, torch.nn.Parameter]] = []
    adamw_params: list[tuple[str, torch.nn.Parameter]] = []

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        lname = name.lower()
        is_embedding = any(token in lname for token in _MUON_EXCLUDE_SUBSTRINGS)
        shape_ok = (param.ndim == 2) or (include_nd and param.ndim >= 2)
        use_muon = shape_ok and (include_embeddings or not is_embedding)

        if use_muon:
            muon_params.append((name, param))
        else:
            adamw_params.append((name, param))

    return muon_params, adamw_params


@torch.no_grad()
def _newton_schulz_ns5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz iteration for approximate zeroth-power of a matrix.

    Computes an orthogonalized version of G — for 2D G of shape [m, n],
    returns X ≈ G @ (G^T G)^(-1/2), the classical Muon update. Works in
    bfloat16 for speed, returns in G's dtype.

    Coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) are the standard
    5-step polynomial used by Jordan (nanoGPT Muon) — chosen so 5 iterations
    converge to 6 decimal places for the relevant singular-value range.
    """
    assert G.ndim == 2, f"NS requires 2D, got shape {tuple(G.shape)}"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)
    # Operate on the shorter-last-dim orientation for efficiency.
    transpose = X.shape[-2] < X.shape[-1]
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class MuonND(torch.optim.Optimizer):
    """Muon variant that accepts any ndim>=2 parameter by reshape-to-2D.

    Drop-in replacement for torch.optim.Muon on 2D params. For higher-rank
    params (3D/4D attention weights), reshapes the gradient to
    [prod(shape[:-1]), shape[-1]] for the Newton-Schulz step, then reshapes
    the update back. This is mathematically correct because attention
    projections ARE 2D matrices — they're stored as 3D/4D for implementation
    convenience (per-head splitting in einsum).

    The per-update RMS scaling follows nanoGPT Muon (sqrt(max(m,n)/min(m,n))),
    which approximately matches AdamW's unit-RMS update magnitude so the same
    LR tunes both optimizers comparably — similar in spirit to PyTorch's
    adjust_lr_fn='match_rms_adamw' but implemented inline.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            weight_decay=weight_decay, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None or p.ndim < 2:
                    continue

                orig_shape = p.shape
                # Flatten leading dims for the Newton-Schulz step.
                if p.ndim == 2:
                    grad2d = p.grad
                else:
                    grad2d = p.grad.reshape(-1, orig_shape[-1])

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad2d)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad2d)
                if nesterov:
                    g_eff = grad2d.add(buf, alpha=momentum)
                else:
                    g_eff = buf

                update2d = _newton_schulz_ns5(g_eff, steps=ns_steps)

                # nanoGPT-style RMS scaling: make update magnitude shape-invariant,
                # so the same lr works across matrices of different aspect ratios.
                m, n = update2d.shape
                rms_scale = (max(m, n) / min(m, n)) ** 0.5
                update2d = update2d * rms_scale

                if p.ndim == 2:
                    update_nd = update2d
                else:
                    update_nd = update2d.reshape(orig_shape)

                # Decoupled weight decay.
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)

                p.data.add_(update_nd, alpha=-lr)

        return loss


class HybridMuonAdamW(torch.optim.Optimizer):
    """Optimizer wrapper that applies Muon to hidden matrices and AdamW elsewhere.

    Backend selected by `muon_backend`:
      'torch'  — stock torch.optim.Muon (strict 2D only; ndim>2 goes to AdamW)
      'nd'     — our MuonND (Python loop, ~50µs/op dispatch overhead per tensor)
      'split'  — RECOMMENDED when muon_include_nd: stock C++ Muon for 2D
                 tensors, MuonND only for ndim>2 attention tensors. Avoids the
                 per-tensor Python dispatch overhead on the bulk of weights.
    """

    def __init__(
        self,
        *,
        muon_named_params: list[tuple[str, torch.nn.Parameter]],
        adamw_named_params: list[tuple[str, torch.nn.Parameter]],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_adjust_lr_fn: str = "match_rms_adamw",
        muon_backend: str = "torch",
    ) -> None:
        all_params = [p for _, p in muon_named_params] + [p for _, p in adamw_named_params]
        super().__init__(all_params, defaults={"lr": lr, "weight_decay": weight_decay})

        # Sub-optimizers can be a list (split backend uses both stock + MuonND).
        self._sub_optimizers: list[torch.optim.Optimizer] = []
        if muon_named_params:
            if muon_backend == "nd":
                muon_nd = MuonND(
                    muon_named_params,
                    lr=lr,
                    momentum=muon_momentum,
                    nesterov=muon_nesterov,
                    weight_decay=weight_decay,
                )
                self._sub_optimizers.append(muon_nd)
            elif muon_backend == "torch":
                muon_t = torch.optim.Muon(
                    muon_named_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=muon_momentum,
                    nesterov=muon_nesterov,
                    adjust_lr_fn=muon_adjust_lr_fn,
                )
                self._sub_optimizers.append(muon_t)
            elif muon_backend == "split":
                # Split: 2D → stock torch.optim.Muon (C++ fast path), ndim>2 → MuonND
                two_d = [(n, p) for n, p in muon_named_params if p.ndim == 2]
                nd = [(n, p) for n, p in muon_named_params if p.ndim > 2]
                if two_d:
                    self._sub_optimizers.append(torch.optim.Muon(
                        two_d,
                        lr=lr,
                        weight_decay=weight_decay,
                        momentum=muon_momentum,
                        nesterov=muon_nesterov,
                        adjust_lr_fn=muon_adjust_lr_fn,
                    ))
                if nd:
                    self._sub_optimizers.append(MuonND(
                        nd,
                        lr=lr,
                        momentum=muon_momentum,
                        nesterov=muon_nesterov,
                        weight_decay=weight_decay,
                    ))
            else:
                raise ValueError(f"Unknown muon_backend: {muon_backend!r}")
        # Keep the historic .muon attribute for state_dict compatibility,
        # pointing at the first sub-optimizer if any (used only as a hint —
        # state_dict serializes the full _sub_optimizers list).
        self.muon = self._sub_optimizers[0] if self._sub_optimizers else None
        self.adamw = (
            torch.optim.AdamW(
                adamw_named_params,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
            )
            if adamw_named_params
            else None
        )
        self.muon_param_count = len(muon_named_params)
        self.adamw_param_count = len(adamw_named_params)
        self._refresh_param_groups()

    def _refresh_param_groups(self) -> None:
        param_groups = []
        for sub in self._sub_optimizers:
            param_groups.extend(sub.param_groups)
        if self.adamw is not None:
            param_groups.extend(self.adamw.param_groups)
        self.param_groups = param_groups

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for sub in self._sub_optimizers:
            sub.step()
        if self.adamw is not None:
            self.adamw.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for sub in self._sub_optimizers:
            sub.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {
            "muon_subs": [sub.state_dict() for sub in self._sub_optimizers],
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        # New format: list of sub-optimizer states
        sub_states = state_dict.get("muon_subs")
        if sub_states is not None:
            for sub, st in zip(self._sub_optimizers, sub_states):
                sub.load_state_dict(st)
        else:
            # Backward compat: old single-muon format
            old_muon = state_dict.get("muon")
            if old_muon is not None and len(self._sub_optimizers) == 1:
                self._sub_optimizers[0].load_state_dict(old_muon)
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
        self._refresh_param_groups()


def build_optimizer(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_adjust_lr_fn: str = "match_rms_adamw",
    muon_include_embeddings: bool = False,
    muon_include_nd: bool = False,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    """Build the training optimizer.

    optimizer_name='muon' uses HybridMuonAdamW, which routes matrix-shaped
    weights to Muon and everything else to AdamW. The 'muon_include_nd' flag
    also routes higher-rank tensors (3D/4D attention weights) to Muon by
    switching to our MuonND backend (stock PyTorch Muon rejects ndim != 2).
    """
    optimizer_name = optimizer_name.lower()

    named_parameters = list(named_parameters)
    if optimizer_name == "adamw":
        params = [param for _, param in named_parameters if param.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
        )
        stats = {
            "muon_tensors": 0,
            "adamw_tensors": len(params),
        }
        return optimizer, stats

    if optimizer_name == "muon":
        muon_params, adamw_params = split_muon_adamw_params(
            named_parameters,
            include_embeddings=muon_include_embeddings,
            include_nd=muon_include_nd,
        )
        # When include_nd is True, use the 'split' backend: stock Muon for 2D
        # (C++ fast path), MuonND only for ndim>2 attention tensors. This
        # avoids paying ~50µs/op Python dispatch overhead on the bulk of
        # weights (which are 2D MLPs/decoders); only the smaller set of
        # attention 3D/4D weights pays the per-tensor Python loop cost.
        backend = "split" if muon_include_nd else "torch"
        optimizer = HybridMuonAdamW(
            muon_named_params=muon_params,
            adamw_named_params=adamw_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
            muon_backend=backend,
        )
        stats = {
            "muon_tensors": len(muon_params),
            "adamw_tensors": len(adamw_params),
            "muon_backend": backend,
            "muon_include_embeddings": bool(muon_include_embeddings),
            "muon_include_nd": bool(muon_include_nd),
        }
        return optimizer, stats

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
