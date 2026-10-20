"""Model v3 encoder: two phases, cell layers with a target stream, then row-level ICL.

Phase A keeps Nori's per-cell tokens and adds a K-token target stream per row. Each
cell layer runs, in Nori's S -> M -> F order:

* S: column attention for cells (every row's cell reads the context rows' cells of its
  column) and train-only row attention for the flattened target stream;
* M: separate SwiGLU feed-forwards for cells and for the target stream;
* F: decoupled feature attention: cells read [cells, targets], each target token reads
  cells only, and the K readouts are mixed by one linear.

At the boundary the K target tokens of a row are flattened into one vector of width
``K * embed_dim`` and the context labels are injected again. Phase B runs row layers of
train-only row attention, a read-back from the row's own final cell states, and SwiGLU.

Every attention normalises its queries and keys (per-head RMS norm with a learned gain).
Without it, the row attention's scores grow without bound during training and the
softmax collapses to one-hot.

Module and parameter names mirror ``EncoderBaseLayer``
(``transformer_encoder.layers.<i>.sequence_attentions`` / ``feature_attentions`` /
``mlp``) so the hidden Muon policy routes the same kinds of weights.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from synthefy_nori.model.layer import MultiheadAttention, RMSNorm, SwiGLUMLP


def _build_two_phase_attention(
    embed_dim: int, heads: int, *, device, dtype, prefer_cudnn_inference: bool = False
) -> MultiheadAttention:
    """Cross attention over the third tensor axis with log-n query scaling and query-key norm."""
    return MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=heads,
        qkv_combined=False,
        dropout=0.0,
        use_qassmax=True,
        qass_mode="log_only",
        device=device,
        dtype=dtype,
        qk_norm=True,
        reuse_projected_kv_storage=True,
        prefer_cudnn_inference=prefer_cudnn_inference,
    )


def _attend_to_keys(module: MultiheadAttention, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """``queries``/``keys``: ``[batch, groups, length, width]``; attention runs over ``length``."""
    return module(x=queries, x_kv=keys, copy_first_head_kv=False)[0]


class TargetStreamEncoder(nn.Module):
    """K target tokens per row from a regression value or a classification label."""

    def __init__(
        self,
        cell_dim: int,
        target_tokens: int,
        *,
        task_type: str,
        max_num_classes: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.cell_dim = int(cell_dim)
        self.target_tokens = int(target_tokens)
        self.task_type = task_type
        width = self.cell_dim * self.target_tokens
        self.width = width
        # Keep the regression module under its original ``mlp`` name so Model
        # v3 regression checkpoints retain their state-dict key contract.
        self.mlp = (
            nn.Sequential(
                nn.Linear(1, width // 2, device=device, dtype=dtype),
                nn.GELU(),
                nn.Linear(width // 2, width, device=device, dtype=dtype),
            )
            if task_type == "reg"
            else None
        )
        self.class_embedding = (
            nn.Embedding(max_num_classes, width, device=device, dtype=dtype) if task_type == "cls" else None
        )
        self.query_embedding = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        nn.init.normal_(self.query_embedding, std=0.02)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """``y``: ``[batch, rows]`` with NaN on query rows -> ``[batch, rows, K, cell_dim]``."""
        valid = torch.isfinite(y)
        labels = torch.where(valid, y, torch.zeros_like(y))
        if self.task_type == "cls":
            assert self.class_embedding is not None
            embedded = self.class_embedding(labels.long().clamp(0, self.class_embedding.num_embeddings - 1))
        else:
            assert self.mlp is not None
            embedded = self.mlp(labels.unsqueeze(-1).to(self.mlp[0].weight.dtype))
        embedded = torch.where(valid.unsqueeze(-1), embedded, self.query_embedding.to(embedded.dtype))
        return embedded.view(*y.shape, self.target_tokens, self.cell_dim)


class FullColumnAttention(nn.Module):
    """Per column: every row's cell attends to the context rows' cells of that column."""

    def __init__(self, cell_dim: int, heads: int, *, device, dtype):
        super().__init__()
        self.sequence_attentions = nn.ModuleList(
            [_build_two_phase_attention(cell_dim, heads, device=device, dtype=dtype, prefer_cudnn_inference=True)]
        )

    def forward(self, cells: torch.Tensor, eval_pos: int) -> torch.Tensor:
        columns = cells.transpose(1, 2)  # [batch, groups, rows, width]
        return _attend_to_keys(self.sequence_attentions[0], columns, columns[:, :, :eval_pos]).transpose(1, 2)


class DecoupledFeatureAttention(nn.Module):
    """Within each row: cells read [cells, targets]; target tokens read cells only."""

    def __init__(self, cell_dim: int, heads: int, target_tokens: int, *, device, dtype):
        super().__init__()
        self.target_tokens = int(target_tokens)
        self.feature_attentions = nn.ModuleList(
            [_build_two_phase_attention(cell_dim, heads, device=device, dtype=dtype) for _ in range(2)]
        )
        width = cell_dim * self.target_tokens
        self.target_mix = nn.Linear(width, width, bias=False, device=device, dtype=dtype)
        nn.init.normal_(self.target_mix.weight, std=0.02)

    def forward(self, cells: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``cells``: ``[batch, rows, groups, width]``; ``targets``: ``[batch, rows, K, width]``."""
        keys = torch.cat((cells, targets), dim=2)
        cell_update = _attend_to_keys(self.feature_attentions[0], cells, keys)
        readout = _attend_to_keys(self.feature_attentions[1], targets, cells)
        target_update = self.target_mix(readout.flatten(-2)).view_as(targets)
        return cell_update, target_update


class CellLayer2P(nn.Module):
    """One phase-A layer: S (column attention, row attention on targets) -> M -> F."""

    def __init__(
        self,
        cell_dim: int,
        heads: int,
        hidden_dim: int,
        *,
        target_tokens: int,
        target_heads: int,
        target_hidden_dim: int,
        deepnorm_alpha: float,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.cell_dim = int(cell_dim)
        self.target_tokens = int(target_tokens)
        self.target_width = self.cell_dim * self.target_tokens
        self.deepnorm_alpha = float(deepnorm_alpha)
        norm = lambda width: RMSNorm(width, eps=layer_norm_eps, elementwise_affine=False)  # noqa: E731
        self.cell_norms = nn.ModuleList([norm(self.cell_dim) for _ in range(3)])
        self.target_norms = nn.ModuleList([norm(self.target_width), norm(self.target_width), norm(self.cell_dim)])
        self.column_attention = FullColumnAttention(self.cell_dim, heads, device=device, dtype=dtype)
        self.sequence_attentions = nn.ModuleList(
            [_build_two_phase_attention(self.target_width, target_heads, device=device, dtype=dtype)]
        )
        self.mlp = nn.ModuleList(
            [
                SwiGLUMLP(self.cell_dim, hidden_dim, self.cell_dim, has_bias=False, device=device, dtype=dtype),
                SwiGLUMLP(
                    self.target_width, target_hidden_dim, self.target_width, has_bias=False, device=device, dtype=dtype
                ),
            ]
        )
        self.feature_attention = DecoupledFeatureAttention(
            self.cell_dim, heads, self.target_tokens, device=device, dtype=dtype
        )

    def forward(self, cells: torch.Tensor, targets: torch.Tensor, eval_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        # S: column memory for cells; context-only row attention for the target stream.
        cells = self.apply_column_attention(cells, eval_pos)
        rows = targets.flatten(-2)
        normalized = self.target_norms[0](rows).unsqueeze(1)
        rows = self.attend_target_rows_to_context(rows, normalized, normalized[:, :, :eval_pos])
        return self.apply_row_mlps_and_feature_attention(cells, rows)

    # The steps below are the layer split by what each one reads, so the chunked forward
    # (TwoPhaseEncoder.forward_chunked) can run the same functions on slices of the table.

    def apply_column_attention(self, cells: torch.Tensor, eval_pos: int) -> torch.Tensor:
        """S for cells. Reads every row of a column, so it can be sliced by column only."""
        return cells + self.deepnorm_alpha * self.column_attention(self.cell_norms[0](cells), eval_pos)

    def attend_target_rows_to_context(
        self, rows: torch.Tensor, normalized: torch.Tensor, context_keys: torch.Tensor
    ) -> torch.Tensor:
        """S for the target stream: ``normalized`` rows attend to the context rows' ``context_keys``.

        ``rows``: ``[batch, n, width]``; ``normalized``: ``target_norms[0](rows).unsqueeze(1)``;
        ``context_keys``: the same for the context rows. Any slice of rows may be passed.
        """
        update = _attend_to_keys(self.sequence_attentions[0], normalized, context_keys)
        return rows + self.deepnorm_alpha * update.squeeze(1)

    def apply_row_mlps_and_feature_attention(
        self, cells: torch.Tensor, rows: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """M then F. Each row reads only itself, so any slice of rows may be passed.

        Returns ``(cells, targets)`` with targets as ``[batch, n, K, cell_dim]``.
        """
        alpha = self.deepnorm_alpha
        # M: separate feed-forwards.
        cells = cells + alpha * self.mlp[0](self.cell_norms[1](cells))
        rows = rows + alpha * self.mlp[1](self.target_norms[1](rows))
        # F: decoupled feature attention inside each row.
        targets = rows.view(*rows.shape[:-1], self.target_tokens, self.cell_dim)
        cell_update, target_update = self.feature_attention(self.cell_norms[2](cells), self.target_norms[2](targets))
        return cells + alpha * cell_update, targets + alpha * target_update


class RowLayer2P(nn.Module):
    """One phase-B layer: train-only row attention on the row vector, a read-back from the
    row's own final cell states, then SwiGLU."""

    def __init__(
        self,
        width: int,
        heads: int,
        hidden_dim: int,
        *,
        deepnorm_alpha: float,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.deepnorm_alpha = float(deepnorm_alpha)
        self.attn_norm = RMSNorm(width, eps=layer_norm_eps, elementwise_affine=False)
        self.mlp_norm = RMSNorm(width, eps=layer_norm_eps, elementwise_affine=False)
        self.readback_norm = RMSNorm(width, eps=layer_norm_eps, elementwise_affine=False)
        self.sequence_attentions = nn.ModuleList([_build_two_phase_attention(width, heads, device=device, dtype=dtype)])
        # Named feature_attentions so the hidden Muon policy routes it like the cell phase.
        self.feature_attentions = nn.ModuleList([_build_two_phase_attention(width, heads, device=device, dtype=dtype)])
        self.mlp = nn.ModuleList([SwiGLUMLP(width, hidden_dim, width, has_bias=False, device=device, dtype=dtype)])

    def forward(self, rows: torch.Tensor, eval_pos: int, cell_memory: torch.Tensor) -> torch.Tensor:
        normalized = self.attn_norm(rows).unsqueeze(1)
        rows = self.attend_rows_to_context(rows, normalized, normalized[:, :, :eval_pos])
        return self.apply_readback_and_mlp(rows, cell_memory)

    def attend_rows_to_context(
        self, rows: torch.Tensor, normalized: torch.Tensor, context_keys: torch.Tensor
    ) -> torch.Tensor:
        """Row attention: ``normalized`` rows (``attn_norm(rows).unsqueeze(1)``) attend to the
        context rows' ``context_keys``. Any slice of rows may be passed."""
        update = _attend_to_keys(self.sequence_attentions[0], normalized, context_keys)
        return rows + self.deepnorm_alpha * update.squeeze(1)

    def apply_readback_and_mlp(self, rows: torch.Tensor, cell_memory: torch.Tensor) -> torch.Tensor:
        """Read-back and SwiGLU. Each row reads only itself and its own cells."""
        alpha = self.deepnorm_alpha
        # queries: [batch, rows, 1, width]; keys: the row's own cells [batch, rows, features, width]
        query = self.readback_norm(rows).unsqueeze(2)
        rows = rows + alpha * _attend_to_keys(self.feature_attentions[0], query, cell_memory).squeeze(2)
        return rows + alpha * self.mlp[0](self.mlp_norm(rows))


class TwoPhaseEncoder(nn.Module):
    """Phase A cell layers, the boundary, and phase B row layers.

    Exposes ``layers`` and ``gradient_checkpointing`` like ``LayerStack`` so the
    trainer's compilation and checkpointing hooks apply unchanged.
    """

    def __init__(
        self,
        cell_dim: int,
        heads: int,
        hidden_dim: int,
        *,
        target_tokens: int,
        cell_layers: int,
        row_layers: int,
        row_heads: int,
        row_hidden_dim: int,
        task_type: str,
        max_num_classes: int,
        layer_norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if min(cell_layers, target_tokens, row_heads) <= 0 or row_layers < 0:
            raise ValueError("two-phase encoder needs positive cell-layer, token and head counts and row_layers >= 0")
        self.cell_dim = int(cell_dim)
        self.target_tokens = int(target_tokens)
        self.task_type = task_type
        self.width = self.cell_dim * self.target_tokens
        if self.width % int(row_heads):
            raise ValueError("two-phase target width (target_tokens * embed_dim) must be divisible by row_heads")
        self.num_cell_layers = int(cell_layers)
        self.num_row_layers = int(row_layers)
        self.gradient_checkpointing = False
        alpha = float(self.num_cell_layers + self.num_row_layers) ** -0.5
        self.deepnorm_alpha = alpha
        self.target_encoder = TargetStreamEncoder(
            self.cell_dim,
            self.target_tokens,
            task_type=task_type,
            max_num_classes=max_num_classes,
            device=device,
            dtype=dtype,
        )
        self.row_y_encoder = (
            nn.Embedding(max_num_classes, self.width, device=device, dtype=dtype)
            if task_type == "cls"
            else nn.Sequential(
                nn.Linear(1, self.width, device=device, dtype=dtype),
                nn.LayerNorm(self.width, device=device, dtype=dtype),
            )
        )
        cell = [
            CellLayer2P(
                self.cell_dim,
                heads,
                hidden_dim,
                target_tokens=self.target_tokens,
                target_heads=row_heads,
                target_hidden_dim=row_hidden_dim,
                deepnorm_alpha=alpha,
                layer_norm_eps=layer_norm_eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.num_cell_layers)
        ]
        self.cell_to_row = nn.Linear(self.cell_dim, self.width, bias=False, device=device, dtype=dtype)
        row = [
            RowLayer2P(
                self.width,
                row_heads,
                row_hidden_dim,
                deepnorm_alpha=alpha,
                layer_norm_eps=layer_norm_eps,
                device=device,
                dtype=dtype,
            )
            for _ in range(self.num_row_layers)
        ]
        self.layers = nn.ModuleList(cell + row)
        self.cell_out_norm = RMSNorm(self.cell_dim, eps=layer_norm_eps, elementwise_affine=False)
        self.row_out_norm = RMSNorm(self.width, eps=layer_norm_eps, elementwise_affine=False)

    def _run_encoder_layer(self, layer: nn.Module, *args):
        if self.gradient_checkpointing and self.training:
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def _embed_context_labels(self, context_y: torch.Tensor) -> torch.Tensor:
        """The boundary's label re-injection for context rows ``[batch, n]`` -> ``[batch, n, width]``."""
        context_labels = torch.nan_to_num(context_y)
        if self.task_type == "cls":
            assert isinstance(self.row_y_encoder, nn.Embedding)
            return self.row_y_encoder(context_labels.long().clamp(0, self.row_y_encoder.num_embeddings - 1))
        assert isinstance(self.row_y_encoder, nn.Sequential)
        return self.row_y_encoder(context_labels.unsqueeze(-1).to(self.row_y_encoder[0].weight.dtype))

    def forward(
        self, cells: torch.Tensor, y: torch.Tensor, eval_pos: int, **_: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``cells``: ``[batch, rows, groups, cell_dim]``; ``y``: ``[batch, rows]`` (NaN on query rows).

        Returns ``(row_states [batch, rows, width], cell_states [batch, rows, groups, cell_dim])``.
        """
        if not 0 < eval_pos <= cells.shape[1]:
            raise ValueError("two-phase encoder requires a non-empty context prefix")
        targets = self.target_encoder(y).to(cells.dtype)
        for layer in self.layers[: self.num_cell_layers]:
            cells, targets = self._run_encoder_layer(layer, cells, targets, eval_pos)
        rows = targets.flatten(-2)
        context_bias = self._embed_context_labels(y[:, :eval_pos]).to(rows.dtype)
        rows = torch.cat((rows[:, :eval_pos] + context_bias, rows[:, eval_pos:]), dim=1)
        cell_states = self.cell_out_norm(cells)
        cell_memory = self.cell_to_row(cell_states)
        for layer in self.layers[self.num_cell_layers :]:
            rows = self._run_encoder_layer(layer, rows, eval_pos, cell_memory)
        return self.row_out_norm(rows), cell_states

    @torch.no_grad()
    def forward_chunked(
        self,
        cells: torch.Tensor,
        y: torch.Tensor,
        eval_pos: int,
        *,
        row_chunk: int,
        column_chunk: int,
        compute_device: torch.device,
    ) -> torch.Tensor:
        """:meth:`forward`'s normalized row states, never holding the whole table on ``compute_device``.

        ``cells`` is the embedded table ``[batch, rows, groups, cell_dim]`` on its residual device
        (the GPU, or host RAM when it does not fit) and is overwritten. ``y``: ``[batch, rows]``
        with NaN on query rows. Every step is the layer's own function on a slice:

        * column attention reads every row of its column, so it runs over ``column_chunk`` columns;
        * the target-stream and phase-B row attentions read only the context rows' keys, so those
          keys are normalized once per layer and each ``row_chunk`` of rows attends to them;
        * everything else is per row (MLPs, feature attention, the read-back from the row's own
          cells), so it runs over ``row_chunk`` rows. ``cell_memory`` is never materialized for
          the whole table: each row chunk projects its own cells.

        The target stream ``[batch, rows, width]`` lives beside ``cells`` on the residual device.
        Results match :meth:`forward` up to floating-point reassociation from the smaller batches.
        Returns ``[batch, rows, width]`` on the residual device.
        """
        if not 0 < eval_pos <= cells.shape[1]:
            raise ValueError("two-phase encoder requires a non-empty context prefix")
        residual_device = cells.device
        batch, total_rows, n_groups, _ = cells.shape
        row_slices = [slice(start, min(start + row_chunk, total_rows)) for start in range(0, total_rows, row_chunk)]
        y = y.to(compute_device)

        rows = torch.empty((batch, total_rows, self.width), dtype=cells.dtype, device=residual_device)
        for part in row_slices:
            rows[:, part] = self.target_encoder(y[:, part]).to(cells.dtype).flatten(-2).to(residual_device)

        for layer in self.layers[: self.num_cell_layers]:
            for start in range(0, n_groups, column_chunk):
                cols = slice(start, min(start + column_chunk, n_groups))
                cells[:, :, cols] = layer.apply_column_attention(cells[:, :, cols].to(compute_device), eval_pos).to(
                    residual_device
                )
            # Keys from the layer's input, taken before any row chunk below is overwritten.
            context_keys = layer.target_norms[0](rows[:, :eval_pos].to(compute_device)).unsqueeze(1)
            for part in row_slices:
                part_rows = rows[:, part].to(compute_device)
                part_rows = layer.attend_target_rows_to_context(
                    part_rows, layer.target_norms[0](part_rows).unsqueeze(1), context_keys
                )
                part_cells, part_targets = layer.apply_row_mlps_and_feature_attention(
                    cells[:, part].to(compute_device), part_rows
                )
                cells[:, part] = part_cells.to(residual_device)
                rows[:, part] = part_targets.flatten(-2).to(residual_device)
            del context_keys

        # Boundary: re-inject the context labels; cells become the final cell states in place.
        for part in row_slices:
            part_rows = rows[:, part].to(compute_device)
            n_context = min(max(eval_pos - part.start, 0), part.stop - part.start)
            if n_context:
                bias = self._embed_context_labels(y[:, part.start : part.start + n_context])
                part_rows = torch.cat(
                    (part_rows[:, :n_context] + bias.to(part_rows.dtype), part_rows[:, n_context:]), 1
                )
            rows[:, part] = part_rows.to(residual_device)
            cells[:, part] = self.cell_out_norm(cells[:, part].to(compute_device)).to(residual_device)

        for layer in self.layers[self.num_cell_layers :]:
            context_keys = layer.attn_norm(rows[:, :eval_pos].to(compute_device)).unsqueeze(1)
            for part in row_slices:
                part_rows = rows[:, part].to(compute_device)
                part_rows = layer.attend_rows_to_context(
                    part_rows, layer.attn_norm(part_rows).unsqueeze(1), context_keys
                )
                cell_memory = self.cell_to_row(cells[:, part].to(compute_device))
                rows[:, part] = layer.apply_readback_and_mlp(part_rows, cell_memory).to(residual_device)
            del context_keys

        # A new tensor, not written back into ``rows``: the output keeps the norm's dtype, as in forward().
        out = None
        for part in row_slices:
            normalized = self.row_out_norm(rows[:, part].to(compute_device))
            if out is None:
                out = torch.empty((batch, total_rows, self.width), dtype=normalized.dtype, device=residual_device)
            out[:, part] = normalized.to(residual_device)
        assert out is not None
        return out
