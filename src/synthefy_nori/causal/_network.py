"""NoriCausal network: (X, y) table -> (T, T) directed edge logits (inference only).

    table -> input layer -> row-and-column blocks -> pool rows -> column-only blocks -> edge head
             per-cell tokens   attend across rows     mean+std     nodes attend to      bilinear +
             (N, T, 128)       and across columns     (T, d)       each other           src/dst bias

T = C feature columns + 1 target slot; ``logits[i, j]`` scores the edge ``node i -> node j``.
Parameter names match the training checkpoints so they load with ``load_state_dict`` directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from synthefy_nori.causal import nori_causal_encoders as encoders

EMBED_DIM = 128
FEATURES_PER_GROUP = 2
# Input-layer configuration the NoriCausal checkpoints were trained with (fixed; part of the weights' meaning).
PREPROCESS_CONFIG_X = dict(
    num_features=2,
    nan_handling_enabled=True,
    normalize_on_train_only=True,
    normalize_x=True,
    remove_outliers=False,
    normalize_by_used_features=True,
)
ENCODER_CONFIG_X = dict(
    num_features=2,
    embedding_size=EMBED_DIM,
    mask_embedding_size=EMBED_DIM,
    encoder_use_bias=True,
    numeric_embed_type="RBF",
    RBF_config={
        "token_embed_dim": 16,
        "n_kernels": 64,
        "sigma": 0.5,
        "use_learn_sigma": False,
        "use_learn_embeddings": False,
        "use_random_kernels": False,
        "use_original_features": False,
    },
)
ENCODER_CONFIG_Y = dict(num_inputs=1, embedding_size=EMBED_DIM, nan_handling_y_encoder=True, max_num_classes=10)
# Training-time modules present in the checkpoints but never reached by the NoriCausal forward pass.
UNUSED_PREFIXES = (
    "enc.backbone.cls_y_encoder.",
    "enc.backbone.cls_target_aware_embedding.",
    "enc.backbone.cls_y_decoder.",
    "enc.backbone.reg_y_decoder.",
    "enc.backbone.feature_decoder.",
)


class _InputBackbone(nn.Module):
    """The Nori input embedding the checkpoints were trained with (``nori_causal_encoders``): preprocessing,
    per-cell RBF tokens (two values per token), the regression y token and the target-aware bias."""

    def __init__(self):
        super().__init__()
        self.embed_dim = EMBED_DIM
        self.x_preprocess = encoders.preprocesss_4_x(**PREPROCESS_CONFIG_X)
        self.encoder_x = encoders.get_x_encoder(**ENCODER_CONFIG_X)
        self.reg_y_encoder = encoders.get_reg_y_encoder(**ENCODER_CONFIG_Y)
        self.reg_target_aware_embedding = nn.Linear(1, EMBED_DIM)
        self.column_y_aware_alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x: torch.Tensor, y: torch.Tensor, eval_pos: int) -> torch.Tensor:
        """Embed every cell and the target.

        x: (1, S, F) with F a multiple of FEATURES_PER_GROUP; y: (1, eval_pos). Returns (1, S, F/2 + 1, E):
        one token per feature group plus the y token, for every row."""
        batch, seq, n_feat = x.shape
        x_inputs = {"data": x, "mask": torch.isnan(x).to(torch.int32)}
        for key in ("data", "mask"):
            x_inputs[key] = x_inputs[key].reshape(batch, seq, n_feat // FEATURES_PER_GROUP, FEATURES_PER_GROUP)
        x_inputs["eval_pos"] = eval_pos
        x_preprocessed = self.x_preprocess(x_inputs)
        x_preprocessed["data"] = torch.where(
            x_preprocessed["mask"].to(torch.bool), float("nan"), x_preprocessed["data"]
        )
        emb_x = self.encoder_x(x_preprocessed)["data"]
        y_column = y.unsqueeze(-1)
        if y_column.shape[1] < seq:
            y_column = torch.cat(
                (
                    y_column,
                    torch.full((batch, seq - y_column.shape[1], 1), float("nan"), device=y.device, dtype=y.dtype),
                ),
                1,
            )
        target_aware_y = y_column.squeeze(-1).clone()
        pos = torch.arange(seq, device=y.device)
        y_masked = torch.where(
            pos.view(1, -1, 1) >= eval_pos, torch.tensor(float("nan"), device=y.device, dtype=y.dtype), y_column
        )
        emb_y = self.reg_y_encoder({"data": y_masked, "eval_pos": eval_pos})["data"].squeeze(2)
        # target-aware bias: context rows get Linear(y), query rows zero; column-specific gate mixed by alpha
        ctx = self.reg_target_aware_embedding(target_aware_y[:, :eval_pos].unsqueeze(-1).to(emb_x.dtype)).to(
            emb_x.dtype
        )
        bias = torch.cat([ctx, ctx.new_zeros(batch, seq - eval_pos, self.embed_dim)], 1).unsqueeze(2)
        gate = torch.sigmoid((emb_x * bias).sum(-1) / self.embed_dim**0.5).unsqueeze(-1)
        alpha = torch.sigmoid(self.column_y_aware_alpha).to(emb_x.dtype)
        emb_x = emb_x + (1.0 - alpha) * bias + alpha * bias * gate
        return torch.cat((emb_x, emb_y.unsqueeze(2)), 2)


class InputEncoder(nn.Module):
    """(X (N, C), y (N,)) -> (N, C+1, E) column tokens. Each column is duplicated into one two-value
    group so that one token corresponds to one column; one dummy query row is appended and dropped."""

    def __init__(self):
        super().__init__()
        self.backbone = _InputBackbone()
        self.embed_dim = EMBED_DIM

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """(N, C) features and (N,) target -> (N, C+1, E) tokens, the target token last."""
        n, c = X.shape
        xg = torch.cat([X, X[:1]], 0).repeat_interleave(FEATURES_PER_GROUP, dim=1)[None]
        yb = torch.cat([y, torch.zeros(1, dtype=y.dtype, device=y.device)])[None]
        z = self.backbone(xg, yb, eval_pos=n)[0, :n]
        assert z.shape[1] == c + 1, f"{z.shape[1]} tokens for {c} columns + target"
        return z


class _Attn(nn.Module):
    """Pre-norm self-attention + FFN over whatever axis the caller folds into dim 1."""

    def __init__(self, d, nhead, ff_mult=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def forward(self, z):
        """(B, L, d) -> (B, L, d): attention over the L axis, then the FFN, both residual."""
        h = self.n1(z)
        z = z + self.attn(h, h, h, need_weights=False)[0]
        return z + self.ff(self.n2(z))


class STBlock(nn.Module):
    """Attention across rows, an MLP, then attention across columns."""

    def __init__(self, d, nhead):
        super().__init__()
        self.row, self.feat = _Attn(d, nhead), _Attn(d, nhead)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, z):
        """(N rows, T nodes, d) -> same shape."""
        z = self.row(z.transpose(0, 1)).transpose(0, 1)
        z = z + self.mlp(self.norm(z))
        return self.feat(z)


class FeatureAttnBlock(nn.Module):
    """Self-attention across the T node tokens after row pooling."""

    def __init__(self, d, nhead=4, ff_mult=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def forward(self, z):
        """(1, T, d) node tokens -> same shape."""
        h = self.n1(z)
        z = z + self.attn(h, h, h, need_weights=False)[0]
        return z + self.ff(self.n2(z))


class RowPool(nn.Module):
    """Order-free row pooling: mean and std per node token, projected back to d."""

    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(2 * d, d)

    def forward(self, z):
        """(N, T, d) -> (T, d)."""
        return self.proj(torch.cat([z.mean(0), z.std(0, unbiased=False)], dim=-1))


class GraphEncoder(nn.Module):
    """(N, T, E) cell tokens -> (T, d) node states: project, row-and-column blocks, pool the rows, then
    column-only blocks. Tables longer than ``max_rows`` are evenly subsampled first."""

    def __init__(self, embed_dim=EMBED_DIM, d=128, nhead=4, n_blocks=3, n_pooled_blocks=2, max_rows=256):
        super().__init__()
        self.max_rows = max_rows
        self.inp = nn.Linear(embed_dim, d)
        self.blocks = nn.ModuleList([STBlock(d, nhead) for _ in range(n_blocks)])
        self.pool = RowPool(d)
        self.pooled_blocks = nn.ModuleList([FeatureAttnBlock(d, nhead) for _ in range(n_pooled_blocks)])
        self.out_norm = nn.LayerNorm(d)
        self.feat_head = nn.Linear(d, 1)  # present in the checkpoints; unused by the edge head
        self.bias_head = nn.Linear(d, 1)

    def forward(self, Z):
        """(N, T, E) -> (T, d)."""
        if Z.shape[0] > self.max_rows:
            Z = Z[torch.linspace(0, Z.shape[0] - 1, self.max_rows, device=Z.device).long()]
        z = self.inp(Z)
        for blk in self.blocks:
            z = blk(z)
        S = self.pool(z)[None]
        for blk in self.pooled_blocks:
            S = blk(S)
        return self.out_norm(S)[0]


class DAGHead(nn.Module):
    """(T, d) node states -> (T, T) logits = A(S) B(S)^T / sqrt(r) + src(S) + dst(S)^T."""

    def __init__(self, d=128, r=64):
        super().__init__()
        self.r = r
        # A / B keep the checkpoint's parameter names (head.A.weight, head.B.weight)
        self.A, self.B = nn.Linear(d, r, bias=False), nn.Linear(d, r, bias=False)
        self.src, self.dst = nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, S):
        """(T, d) node states -> (T, T) edge logits."""
        return (self.A(S) @ self.B(S).t()) / math.sqrt(self.r) + self.src(S) + self.dst(S).t()


class NoriCausalNetwork(nn.Module):
    """Input layer -> graph encoder -> edge head. ``forward(X, y, null_target)`` -> (C+1, C+1) logits.
    With ``null_target=True`` the target token is replaced by a learned null vector, so every column
    is an ordinary node and ``y`` is only a placeholder."""

    def __init__(self, d=352, r=176, nhead=4, n_blocks=6, n_pooled_blocks=15, max_rows=2048):
        super().__init__()
        self.enc = InputEncoder()
        self.hyp = GraphEncoder(EMBED_DIM, d, nhead, n_blocks, n_pooled_blocks, max_rows)
        self.head = DAGHead(d, r)
        self.null_token = nn.Parameter(torch.zeros(EMBED_DIM))
        self.max_rows = max_rows

    def forward(self, X: torch.Tensor, y: torch.Tensor, null_target: bool = False) -> torch.Tensor:
        """(N, C) features and (N,) target -> (C+1, C+1) edge logits, ``[i, j]`` for node i -> node j."""
        if len(X) > self.max_rows:
            idx = torch.linspace(0, len(X) - 1, self.max_rows, device=X.device).long()
            X, y = X[idx], y[idx]
        z = self.enc(X, y)
        if null_target:
            z = z.clone()
            z[:, -1, :] = self.null_token
        return self.head(self.hyp(z))

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Rebuild from a training checkpoint; every parameter the forward pass reads must be present."""
        ck = torch.load(path, map_location=device, weights_only=True)
        net = cls(
            d=ck["d"],
            r=ck["r"],
            nhead=ck.get("nhead", 4),
            n_blocks=ck["n_blocks"],
            n_pooled_blocks=ck.get("n_pooled_blocks", 2),
            max_rows=ck.get("max_rows", 1024),
        )
        state = {k: v for k, v in ck["state_dict"].items() if not k.startswith(UNUSED_PREFIXES)}
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"checkpoint does not match the NoriCausal network: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        return net.to(device).eval(), {
            k: ck.get(k) for k in ("step", "d", "r", "n_blocks", "n_pooled_blocks", "nhead", "max_rows")
        }
