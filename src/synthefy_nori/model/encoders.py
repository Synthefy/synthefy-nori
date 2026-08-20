from __future__ import annotations

import torch
import torch.nn as nn
from synthefy_nori.model.layer import EncoderBaseLayer, MLP
from typing import Any,Literal
from torch.nn.init import orthogonal_
import numpy as np
import einops


def calc_mean(x:torch.Tensor, dim:int):
    finite = torch.isfinite(x)
    num = torch.sum(finite, dim=dim).clip(min=1)
    values = torch.where(finite, x, torch.zeros_like(x))
    return torch.sum(values, dim=dim) / num, num

def calc_std(x:torch.Tensor, dim:int, mean_v:torch.Tensor|None = None, value_num:torch.Tensor|None=None ):
    if mean_v is None or value_num is None:
        mean_v, value_num = calc_mean(x, dim)
    finite = torch.isfinite(x)
    squared_error = torch.where(
        finite,
        torch.square(mean_v.unsqueeze(dim) - x),
        torch.zeros_like(x),
    )
    return torch.sqrt(torch.sum(squared_error, dim=dim) / (value_num - 1).clip(min=1))

def drop_outliers(
                    x:torch.Tensor, 
                    std_sigma:float=4,
                    eval_pos:int=-1,
                    lower:torch.Tensor|None = None,
                    upper:torch.Tensor|None = None,
                    dim:int=1
                    ):
        # assert len(x.shape)==3, "x.shape must be B,S,F" 

        if lower is None:
            data = x[:,:eval_pos].clone()
            data_mean, value_num = calc_mean(data, dim=dim)
            data_std = calc_std(data, dim=dim, mean_v=data_mean, value_num=value_num)
            cut_off = data_std * std_sigma
            lower, upper = data_mean - cut_off, data_mean + cut_off
            
            data[torch.logical_or(data > upper, data < lower)] = np.nan
            data_mean, value_num = calc_mean(data, dim=dim)
            data_std = calc_std(data, dim=dim, mean_v=data_mean, value_num=value_num)
            cut_off = data_std * std_sigma
            lower, upper = data_mean - cut_off, data_mean + cut_off
        
        x = torch.maximum(-torch.log(1 + torch.abs(x)) + lower, x)
        x = torch.minimum(torch.log(1 + torch.abs(x)) + upper, x)
        
        return x, lower, upper
    
def normalize_mean0_std1(
                        x:torch.Tensor, 
                        eval_pos:int=-1,
                        clip:bool=True,
                        dim:int=1,
                        mean: torch.Tensor | None = None,
                        std: torch.Tensor | None = None
                        ):
    if mean is None:
        mean, value_num = calc_mean(x[:,:eval_pos], dim=dim)
        std = calc_std(x[:,:eval_pos], dim=dim, mean_v=mean, value_num=value_num) + 1e-20
        
        if x.shape[1] == 1 or eval_pos == 1:
            std[:] = 1.0
    x = (x - mean.unsqueeze(1).expand_as(x)) / std.unsqueeze(1).expand_as(x)
    if clip:
        x = torch.clip(x, min=-100, max=100)
    return x, mean, std
    

class LinearEncoder(nn.Module):
    """linear input encoder"""
    def __init__(
                self,
                num_features: int,
                emsize: int,
                nan_to_zero: bool = False,
                bias: bool = True,
                in_keys:list[str]=['data'],
                out_key:str='data',
    ):
        """Initialize the LinearEncoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.nan_to_zero = nan_to_zero
        self.in_keys = in_keys
        self.out_key = out_key
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        missing_keys = [key for key in self.in_keys if key not in input]
        assert not missing_keys, f"Missing encoder inputs: {missing_keys}"
        x = [input[key] for key in self.in_keys] 
        x = torch.cat(x, dim=-1) # type: ignore
        if self.nan_to_zero:
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
        input[self.out_key] = self.layer(x)
        return input

class RBFembedding(nn.Module):
    def __init__(
        self, 
        embedding_size: int = 96,
        exponent_digits: int = 1,
        token_embed_dim: int = 32,
        n_kernels: int = 32,
        sigma: float = 1.05,
        use_learn_sigma: bool = False,
        center_range: tuple = (0.0, 10.0),
        use_learn_embeddings: bool = False,
        as_tokenizer: bool = False,
        use_original_features: bool = False,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.dtype = dtype
        self.n_kernels = n_kernels
        self.exponent_digits = exponent_digits
        self.as_tokenizer = as_tokenizer
        self.use_original_features = use_original_features

        min_val, max_val = center_range
        if n_kernels <= 1:
            centers = torch.tensor([(min_val + max_val) / 2.0], dtype=torch.float32)
        else:
            centers = torch.linspace(min_val, max_val, steps=n_kernels, dtype=torch.float32)
        self.register_buffer("centers", centers, persistent=False)
        if use_learn_sigma:
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=dtype))
        else:
            sigma = torch.tensor(sigma, dtype=dtype)
            self.register_buffer("sigma", sigma)

        if exponent_digits > 0:
            self.sign_embedding = nn.Embedding(2, token_embed_dim, dtype=dtype)       # 0:+, 1:-
            self.exp_sign_embedding = nn.Embedding(2, token_embed_dim, dtype=dtype)   # 0:exp+, 1:exp-
            self.exp_digit_embedding = nn.Embedding(10, token_embed_dim, dtype=dtype) # 0-9
            
            if not use_learn_embeddings:
                self.sign_embedding.weight.requires_grad = False
                self.exp_sign_embedding.weight.requires_grad = False
                self.exp_digit_embedding.weight.requires_grad = False
            
            ctrl_in_dim = (exponent_digits + 2) * token_embed_dim
            self.gate_mlp = nn.Sequential(
                nn.Linear(ctrl_in_dim, 4 * token_embed_dim, dtype=dtype),
                nn.GELU(),
                nn.Linear(4 * token_embed_dim, 2 * n_kernels, dtype=dtype)
            )
        else:
            self.sign_embedding = self.exp_sign_embedding = self.exp_digit_embedding = None
            self.gate_mlp = None
        
        self.norm = nn.LayerNorm(n_kernels, dtype=dtype)
        if self.use_original_features:
            self.out_layer = nn.Linear(n_kernels + 1, embedding_size, dtype=dtype)
        else:
            self.out_layer = nn.Linear(n_kernels, embedding_size, dtype=dtype)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # A short-lived checkpoint format persisted these now-deterministic
        # centers. Accept the key only when it encodes the same canonical grid;
        # a random or corrupt grid represents different model behavior and must
        # continue to fail a strict load.
        centers_key = prefix + "centers"
        persisted_centers = state_dict.get(centers_key)
        canonical_centers = self.centers.detach()
        if (
            torch.is_tensor(persisted_centers)
            and persisted_centers.shape == canonical_centers.shape
            and torch.equal(
                persisted_centers.to(
                    device=canonical_centers.device,
                    dtype=canonical_centers.dtype,
                ),
                canonical_centers,
            )
        ):
            state_dict = state_dict.copy()
            state_dict.pop(centers_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @torch.compiler.disable
    def _rbf_scientific_decompose(self, x_work):
        """Decompose x into (mantissa, exponent) in scientific notation.

        Isolated from torch.compile because log10 + integer power
        causes Inductor CantSplit in the backward pass.
        """
        abs_x = torch.abs(x_work)
        is_zero = (abs_x == 0)
        safe = torch.where(is_zero, torch.ones_like(abs_x), abs_x)
        exp_f = torch.floor(torch.log10(safe))
        max_exp = 10**self.exponent_digits - 1
        exp_i = torch.clamp(exp_f, -max_exp, max_exp).to(torch.int64)
        x_scaled = abs_x / (10.0 ** exp_i.to(x_work.dtype))
        x_scaled = torch.where(is_zero, torch.zeros_like(x_scaled), x_scaled)
        return x_scaled, exp_i

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (batch_size, n_features)
        x = x.squeeze(-1)
        S, F = x.shape[0], (x.shape[1] if x.ndim > 1 else 1)
        # MPS does not implement float64. Keep the higher-precision path on
        # CPU/CUDA and use float32 for the same calculation on Apple GPUs.
        work_dtype = torch.float32 if x.device.type == "mps" else torch.float64
        x_work = x.to(work_dtype)
        if self.gate_mlp is not None:
            x_scaled, exp_i = self._rbf_scientific_decompose(x_work)
        else:
            exp_i = None
            x_scaled = x_work

        centers = self.centers.to(work_dtype)
        diff = x_scaled.unsqueeze(-1) - centers.view((1,)*x_scaled.dim() + (self.n_kernels,))
        rbf = torch.exp(-(diff ** 2) / (2 * (self.sigma ** 2))).to(self.dtype)

        if self.gate_mlp is not None:
            sign_idx = (x_work < 0).to(torch.long)
            exp_sign_idx = (exp_i < 0).to(torch.long)
            abs_exp = exp_i.abs()
            sign_emb = self.sign_embedding(sign_idx)            # (S, F, D)
            exp_sign_emb = self.exp_sign_embedding(exp_sign_idx)
            exp_digit_emb_list = []
            for power in range(self.exponent_digits):
                digit = (abs_exp // (10 ** power)) % 10
                exp_digit_emb_list.append(self.exp_digit_embedding(digit))
            exp_digits_emb = torch.stack(exp_digit_emb_list[::-1], dim=-2)     # (S,F,e,D)
            exp_digits_emb_flat = einops.rearrange(exp_digits_emb, "... e D -> ... (e D)")
            ctrl = torch.cat([sign_emb, exp_sign_emb, exp_digits_emb_flat], dim=-1).to(self.dtype)
            gamma_beta = self.gate_mlp(ctrl)                # (S, F, 2*k)
            gamma, beta = torch.split(gamma_beta, self.n_kernels, dim=-1)
            gamma = torch.sigmoid(gamma)                    # (S, F, k), 0~1
            beta = torch.tanh(beta)                         # (S, F, k), -1~1
            # rbf = self.norm(rbf)
            rbf = rbf * gamma + beta
            rbf = self.norm(rbf)
        if self.use_original_features:
            rbf = torch.cat([rbf, x.unsqueeze(-1)], dim=-1)
        out = self.out_layer(rbf.to(self.dtype))            # (S, F, embedding_size)
        return out.reshape(S, -1) if not self.as_tokenizer else out
    
class PBLDEmbedding(nn.Module):
    """Periodic + Linear + DenseNet embedding (inspired by RealMLP).
    For each scalar x:
      periodic = cos(2pi * w * x + b)  ->  Linear + residual  ->  LayerNorm
      output = Linear(concat(x, periodic))  ->  [embedding_size]
    All params shared across features (ICL-compatible)."""
    def __init__(self, embedding_size=96, n_frequencies=48, as_tokenizer=False,
                 dtype=torch.float32):
        super().__init__()
        self.as_tokenizer = as_tokenizer
        self.n_frequencies = n_frequencies
        self.freq_weight = nn.Parameter(torch.randn(n_frequencies) * 0.5)
        self.freq_bias = nn.Parameter(torch.zeros(n_frequencies))
        self.periodic_dense = nn.Linear(n_frequencies, n_frequencies, dtype=dtype)
        self.periodic_norm = nn.LayerNorm(n_frequencies, dtype=dtype)
        self.out_layer = nn.Linear(n_frequencies + 1, embedding_size, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1) or (...,) — broadcasts over arbitrary leading dims
        x = x.squeeze(-1)
        leading_shape = x.shape
        periodic = torch.cos(2 * torch.pi * self.freq_weight * x.unsqueeze(-1) + self.freq_bias)
        periodic = self.periodic_dense(periodic) + periodic  # DenseNet residual
        periodic = self.periodic_norm(periodic)
        out = self.out_layer(torch.cat([x.unsqueeze(-1), periodic], dim=-1))
        if not self.as_tokenizer:
            return out.reshape(leading_shape[0], -1)
        return out


class MaskEmbEncoder(nn.Module):
    """
    For masked features, use the mask vector to obtain their representations; 
    for numerical features, use a nonlinear network to obtain their representations
    """
    def __init__(
                self,
                num_features: int,
                emsize: int,
                mask_embedding_size: int,
                numeric_embed_type: str = "linear",
                RBF_config: dict|None = None,
                PBLD_config: dict|None = None,
                nan_to_zero: bool = False,
                bias: bool = True,
                in_keys: list[str] = ['data'],
                out_key: str = 'data',
    ):
        """Initialize the MaskEmbEncoder.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            nan_to_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.embedding_dim = emsize
        self.mask_embedding_size = mask_embedding_size
        self.in_keys = in_keys
        self.out_key = out_key

        # All masked positions use the same vector
        self.mask_embedding = nn.Parameter(torch.randn(self.mask_embedding_size))

        # MLP for numerical features: input is 1, output is embedding_dim
        self.numeric_mlp = nn.Sequential(
            nn.Linear(1, self.embedding_dim // 2, bias=bias),
            nn.LayerNorm(self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim, bias=bias),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU()
        )

        self.numeric_embed_type = numeric_embed_type
        if numeric_embed_type == "linear":
            self.numeric_mlp = nn.Sequential(
                nn.Linear(1, self.embedding_dim // 2),
                nn.LayerNorm(self.embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embedding_dim // 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.ReLU()
            )
        elif numeric_embed_type == "RBF":
            self.numeric_mlp = RBFembedding(
                embedding_size=self.embedding_dim,
                exponent_digits=1,
                token_embed_dim=RBF_config['token_embed_dim'],
                n_kernels=RBF_config['n_kernels'],
                sigma=RBF_config['sigma'],
                use_learn_sigma=RBF_config['use_learn_sigma'],
                use_learn_embeddings=RBF_config['use_learn_embeddings'],
                center_range=(0.0, 10.0),
                use_original_features=RBF_config['use_original_features'],
                as_tokenizer=True,
            )
        elif numeric_embed_type == "PBLD":
            self.numeric_mlp = PBLDEmbedding(
                embedding_size=self.embedding_dim,
                n_frequencies=PBLD_config.get('n_frequencies', 48) if PBLD_config else 48,
                as_tokenizer=True,
            )
        else:
            raise ValueError(f"Invalid numeric_embed_type: {numeric_embed_type}")

        # Merging layer: maps the concatenated feature vectors back to embedding_dim.
        self.fusion_network = nn.Sequential(
            nn.Linear(num_features * self.embedding_dim, self.embedding_dim, bias=bias),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias),
            nn.LayerNorm(self.embedding_dim)
        )
        self.nan_to_zero = nan_to_zero
    
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        missing_keys = [key for key in self.in_keys if key not in input]
        assert not missing_keys, f"Missing encoder inputs: {missing_keys}"
        x = [input[key] for key in self.in_keys]
        x = torch.cat(x, dim=-1) # type: ignore
        batch_size, seq_len, group, feature_num = x.shape

        x = x.unsqueeze(-1)
        is_mask = torch.isnan(x)
        x = x.masked_fill(is_mask, 0.0)
        
        x_emb = self.numeric_mlp(x)

        if self.nan_to_zero:
            # The input was already filled with zero above. Honor the flag by
            # sending missing values through the ordinary numeric-zero path rather
            # than replacing them with the learned mask embedding.
            combined_emb = x_emb
        else:
            mask_emb = self.mask_embedding.expand_as(x_emb)
            combined_emb = torch.where(is_mask, mask_emb, x_emb)
        del x, is_mask, x_emb

        concat_vector = combined_emb.flatten(3)

        sample_representation = self.fusion_network(concat_vector)
        output = sample_representation.view(batch_size, seq_len, group, -1)
        
        input[self.out_key] = output
        return input

class NanEncoder(nn.Module):
    """Encoder stage that deals with NaN and infinite values in the input"""
    def __init__(
        self,
        nan_value: float = -2.0,
        inf_value: float = 2.0,
        neg_info_value: float = 4.0,
        in_keys:list[str]=['data'],
        out_key:str='nan_encoding'
    ):
        """Initialize the NanEncoder.

        Args:
            keep_nans: Flag to maintain NaN values as individual indicators. 
        """
        super().__init__()
        self.nan_value = nan_value
        self.inf_value = inf_value
        self.neg_info_value = neg_info_value
        self.in_keys = in_keys
        self.out_key = out_key
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x:torch.Tensor = input[self.in_keys[0]] # type: ignore
        eval_pos = input['eval_pos']

        # Context-cache "frozen stats" path, same contract as NormalizationEncoder's:
        # the imputation fill is the CONTEXT column mean, so query rows arriving
        # without a train prefix must be given the train mean rather than recomputing
        # one from themselves. Without this, apply_context_cache fills from the QUERY
        # column mean while the transductive path fills from the train mean.
        #
        # Every non-finite value is excluded from the statistic. In particular, an
        # infinity must not poison the fill for every other missing value in its
        # column. A column with no finite context value receives the neutral fill 0;
        # the original-value mask restores that column to missing before embedding.
        frozen_mean = input.get('_frozen_nan_mean')
        if frozen_mean is not None:
            mean_value = frozen_mean
        else:
            mean_value, _ = calc_mean(x[:,:eval_pos,:], dim=1)
        # Expose the fill actually used so build_context_cache can capture it.
        input['_nan_mean'] = mean_value

        # The data channel consistently treats NaN and both infinities as missing.
        # Keep the signed indicator as a separate channel for encoder configurations
        # that explicitly include `nan_encoding` (the y encoder does); it never makes
        # an infinity masquerade as an ordinary numeric feature.
        is_nan = torch.isnan(x)
        is_pos_inf = torch.isinf(x) & (x > 0)
        is_neg_inf = torch.isinf(x) & (x < 0)
        is_nonfinite = is_nan | is_pos_inf | is_neg_inf
        nans_indicator = torch.where(
            is_nan, self.nan_value,
            torch.where(is_pos_inf, self.inf_value,
            torch.where(is_neg_inf, self.neg_info_value,
            torch.zeros_like(x))))

        # Replace NaN/Inf with context mean (functional, no clone + masked assign)
        fill = mean_value.unsqueeze(1).expand_as(x)
        x = torch.where(is_nonfinite, fill, x)

        input[self.in_keys[0]] = x
        input[self.out_key ] = nans_indicator
        return input
        
    
class ValidFeatureEncoder(nn.Module):
    """Valid feature encoder"""
    def __init__(
        self,
        num_features: int,
        nan_normalize: bool=True,
        sqrt_normalize: bool=True,
        in_keys:list[str]=['data'],
        out_key:str='data'
    ):
        """Initialize the ValidFeatureEncoder.

        Args:
            num_features: The target number of features to transform the input into.
            nan_normalize: Indicates whether to normalize based on the number of features actually used.
            sqrt_normalize: Legacy option to normalize using the square root rather than the count of used features.
        """
        super().__init__()
        self.num_features = num_features
        self.nan_normalize = nan_normalize
        self.sqrt_normalize = sqrt_normalize
        self.in_keys = in_keys
        self.out_key = out_key
        self.valid_feature_num = None
    
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x:torch.Tensor = input[self.in_keys[0]]  # type: ignore
        frozen_valid_feature_num = input.get('_frozen_valid_feature_num')
        if frozen_valid_feature_num is None:
            eval_pos = int(input['eval_pos'])
            context = x[:, :eval_pos]
            original_mask = input.get('mask')
            if (self.in_keys[0] == 'data'
                    and isinstance(original_mask, torch.Tensor)
                    and original_mask.shape == x.shape):
                context = torch.where(
                    original_mask[:, :eval_pos].to(torch.bool),
                    torch.full_like(context, float('nan')),
                    context,
                )
            finite = torch.isfinite(context)
            finite_count = finite.sum(dim=1)
            finite_min = torch.where(
                finite, context, torch.full_like(context, float('inf'))).amin(dim=1)
            finite_max = torch.where(
                finite, context, torch.full_like(context, float('-inf'))).amax(dim=1)
            valid_feature = (finite_count > 1) & (finite_min != finite_max)
            valid_feature_num = torch.clip(valid_feature.sum(-1).unsqueeze(-1), min=1)
        else:
            assert isinstance(frozen_valid_feature_num, torch.Tensor)
            valid_feature_num = frozen_valid_feature_num
        # Store on self for backward compat (inference predictor reads it),
        # and also pass through dict for compile-friendly access.
        self.valid_feature_num = valid_feature_num.detach()
        input['_valid_feature_num'] = valid_feature_num

        if self.nan_normalize:
            if self.sqrt_normalize:
                x = x * torch.sqrt(self.num_features / valid_feature_num).unsqueeze(1).expand_as(x)
            else:
                x = x * (self.num_features / valid_feature_num)

        zeros = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros], -1)

        input[self.out_key] = x
        return input
    

class EmbYEncoderStep(nn.Module):
    """A simple linear input encoder step."""

    def __init__(
        self,
        *,
        emsize: int,
        n_classes: int = 10,
        in_keys: list[str] = ['data'],
        out_key: str = 'data',
    ):
        """Initialize the EmbYEncoderStep.

        Args:
            emsize: The embedding size, i.e. the number of output features.
            n_classes: Number of classes
        """
        super().__init__()
        
        # Ensure the embedding dimension is large enough to support orthogonal initialization.
        assert emsize > n_classes + 1, (f"emsize ({emsize}) must be >= n_classes+1 ({n_classes+1}) for orthogonal initialization")

        # Generate an orthogonal matrix of size (n_classes + 1) × emsize
        ortho_matrix = torch.empty(n_classes + 1, emsize)
        orthogonal_(ortho_matrix)  # Initialize in-place as an orthogonal matrix

        # Decompose the matrix: the first n_classes rows are used for y_embedding, and the last row is used for y_mask.
        y_embed_weights = ortho_matrix[:n_classes, :]  # Shape (n_classes, emsize)
        y_mask_weight = ortho_matrix[n_classes:n_classes+1, :]  # Shape (1, emsize)

        self.y_embedding = nn.Embedding(n_classes, emsize)
        self.y_embedding.weight.data = y_embed_weights.clone()

        self.y_mask = nn.Embedding(1, emsize)
        self.y_mask.weight.data = y_mask_weight.clone()
        self.in_keys = in_keys
        self.out_key = out_key
        if len(self.in_keys) > 1:
            print("\033[30;43mWarning: The EmbYEncoderStepl function is only for processing Y, and in_keys must contain exactly one key.\033[0m")
        
    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        y = input[self.in_keys[0]]
        eval_pos = input['eval_pos']
        y = y.int() # type: ignore
        y_train = y[:,:eval_pos]
        y_test = torch.zeros_like(y[:, eval_pos:], dtype=torch.int)
        y_train_emb = self.y_embedding(y_train)
        y_test_emb = self.y_mask(y_test)
        y_emb = torch.cat([y_train_emb, y_test_emb], dim=1)
        
        input[self.out_key] = y_emb
        return input

class MulticlassTargetEncoder(nn.Module):
    """Use the target's index as the class value, with each class corresponding to an index.

    Rank-remaps arbitrary integer class labels to contiguous 0..K-1 based on
    the sorted unique labels observed in the context rows.  The vectorised
    implementation avoids per-batch Python loops and torch.unique so that
    torch.compile can trace through this module without a graph break.
    """
    MAX_CLASSES: int = 10  # model hard-cap (matches decoder width)

    def __init__(
        self,
        in_keys:list[str]=['data'],
        out_key:str='data'
    ):
        super().__init__()
        self.in_keys = in_keys
        self.out_key = out_key

    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x:torch.Tensor = input[self.in_keys[0]]  # type: ignore  [B, S, 1]
        eval_pos = input['eval_pos']
        max_cls = self.MAX_CLASSES

        # Which classes appear in each batch's context?  [B, max_cls] bool
        # Broadcasting: ctx[:, :, :, None] == arange[None, None, None, max_cls]
        ctx = x[:, :eval_pos, :]                                      # [B, ep, 1]
        class_ids = torch.arange(max_cls, device=x.device, dtype=x.dtype)  # [max_cls]
        class_present = (ctx == class_ids).any(dim=1)                   # [B, max_cls]

        # Exclusive prefix-sum: ranks[b, c] = #present classes with index < c
        cum = class_present.to(torch.long).cumsum(dim=1)               # [B, max_cls]
        ranks = torch.zeros_like(cum)
        ranks[:, 1:] = cum[:, :-1]                                    # [B, max_cls]

        # Look up the rank for every position
        x_idx = x[:, :, 0].to(torch.long).clamp(0, max_cls - 1)      # [B, S]
        x_ranked = torch.gather(ranks, 1, x_idx).unsqueeze(-1)        # [B, S, 1]

        input[self.out_key] = x_ranked.to(x.dtype)
        return input

class NormalizationEncoder(nn.Module):
    """normalize encoder"""
    def __init__(
                self, 
                train_only:bool,
                normalize_x:bool,
                remove_outliers:bool,
                std_sigma:float=4.0,
                in_keys:list[str]=['data'],
                out_key:str='data'
                
    ):
        super().__init__()
        self.train_only = train_only
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.std_sigma = std_sigma
        self.in_keys = in_keys
        self.out_key = out_key
        self.mean = None
        self.std = None

    def forward(self, input:dict[str, torch.Tensor|int])->dict[str, torch.Tensor]:
        x = input[self.in_keys[0]]
        eval_pos = input['eval_pos']
        # A negative slice endpoint excludes the final row. `train_only=False`
        # means all rows, including that final row, participate in the statistics.
        pos = eval_pos if self.train_only else x.shape[1]
        # Context-cache "frozen stats" path: when a caller supplies precomputed
        # train-derived stats (via `_frozen_norm_stats`), apply them directly to
        # these rows instead of recomputing from the [:eval_pos] prefix. This is
        # what lets the context encode be built once (over train) and reused
        # across query batches that arrive WITHOUT a train prefix -- the applied
        # stats are bit-identical to the transductive path, which also normalizes
        # test rows with train stats. `drop_outliers`/`normalize_mean0_std1`
        # already skip their eval_pos computation when given lower/upper/mean/std.
        frozen = input.get('_frozen_norm_stats')
        captured: dict[str, torch.Tensor] = {}
        if self.remove_outliers:
            lower = frozen.get('lower') if frozen else None
            upper = frozen.get('upper') if frozen else None
            x, lower, upper = drop_outliers(
                x, eval_pos=pos, std_sigma=self.std_sigma, lower=lower, upper=upper)
            captured['lower'], captured['upper'] = lower, upper
        if self.normalize_x:
            mean = frozen.get('mean') if frozen else None
            std = frozen.get('std') if frozen else None
            x, mean, std = normalize_mean0_std1(x, eval_pos=pos, mean=mean, std=std)
            # Store on self for backward compat (inference predictor reads it),
            # and also pass through dict for compile-friendly access.
            self.mean = mean.detach()
            self.std = std.detach()
            input['_norm_mean'] = mean
            input['_norm_std'] = std
            captured['mean'], captured['std'] = mean, std

        input[self.out_key] = x
        # Expose the stats actually used so build_context_cache can capture them
        # for later frozen reuse (the values are train-derived whether freshly
        # computed here or passed in via `_frozen_norm_stats`).
        input['_norm_stats'] = captured
        return input



def get_x_encoder(
    *,
    num_features: int,
    embedding_size: int,
    mask_embedding_size: int,
    encoder_use_bias: bool,
    numeric_embed_type: str = "linear",
    RBF_config: dict|None = None,
    PBLD_config: dict|None = None,
    in_keys: list = ['data'],
    nan_to_zero: bool = False,
):
    assert isinstance(in_keys, list), "The type of in_keys must be a list!"
    inputs_to_merge = {}
    for in_key in in_keys:
        inputs_to_merge[in_key] = {'dim': num_features}

    encoder_steps = []
    encoder_steps += [
        # The masked features (i.e., features with None values) are directly mapped to 
        # vectors via the embedding matrix, while the numerical features obtain their 
        # embedding representations through a nonlinear transformation matrix.
        MaskEmbEncoder(
            num_features=sum([i["dim"] for i in inputs_to_merge.values()]),
            emsize=embedding_size,
            mask_embedding_size=mask_embedding_size,
            nan_to_zero=nan_to_zero,
            bias=encoder_use_bias,
            in_keys=in_keys,
            RBF_config=RBF_config,
            PBLD_config=PBLD_config,
            numeric_embed_type=numeric_embed_type,
        ),
    ]
    return nn.Sequential(*encoder_steps,)


def get_cls_y_encoder(
    *,
    num_inputs: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int
) -> nn.Module:
    steps = []
    inputs_to_merge = [{"name": "data", "dim": num_inputs}]
    if nan_handling_y_encoder:
        steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]
        inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]

    if max_num_classes >= 2:
        steps += [MulticlassTargetEncoder()]

    steps += [
            EmbYEncoderStep(
                emsize=embedding_size,
                n_classes=max_num_classes
        )
    ]
    return nn.Sequential(*steps)

def get_reg_y_encoder(
    *,
    num_inputs: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int = 10,
) -> nn.Module:
    steps = []
    inputs_to_merge = [{"name": "data", "dim": num_inputs}]
    if nan_handling_y_encoder:
        steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]
        inputs_to_merge += [{"name": "nan_encoding", "dim": num_inputs}]

    in_keys = [item["name"] for item in inputs_to_merge]

    steps += [
        LinearEncoder(
            num_features=sum([i["dim"] for i in inputs_to_merge]),  # type: ignore
            emsize=embedding_size,
            nan_to_zero=not nan_handling_y_encoder,
            in_keys=in_keys,
            out_key='data'
        ),
    ]
    return nn.Sequential(*steps)


def preprocesss_4_x(
    *,
    num_features: int,
    nan_handling_enabled: bool,
    normalize_on_train_only: bool,
    normalize_x: bool,
    remove_outliers: bool,
    normalize_by_used_features: bool,
    ):
    """feature preprocess"""
    preprocess_steps = []

    if nan_handling_enabled:
        # Obtain the positions of non-finite values and replace those data-channel
        # values with the corresponding finite context mean for normalization.
        preprocess_steps += [NanEncoder(in_keys=['data'], out_key='nan_encoding')]

    preprocess_steps += [
        NormalizationEncoder(
            train_only=normalize_on_train_only,
            normalize_x=normalize_x,
            remove_outliers=remove_outliers,
        ),
    ]

    preprocess_steps += [
        # Convert the input into a fixed number of features by adding zero values, with normalization applied (variance is constant).
        ValidFeatureEncoder(
            num_features=num_features,
            nan_normalize=normalize_by_used_features,
        ),
    ]

    return nn.Sequential(*preprocess_steps)
