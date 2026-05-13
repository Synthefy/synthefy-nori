"""TabICL-style synthetic data prior for LimiX training.

Ports the key components of TabICL's prior system (MLP SCM, Tree SCM,
Reg2Cls, meta-distribution HP sampling, rich activation library) to work
with our generate_batch interface.  Internally uses PyTorch (as TabICL
does) but outputs numpy arrays: dict(X, y, n_classes).

Reference: https://github.com/soda-inria/tabicl
           Schlegel et al., "TabICL" (ICML 2025)
"""

import math
import random as pyrandom
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ============================================================================
# Activation functions  (matches TabICL activations.py exactly)
# ============================================================================

class SignActivation(nn.Module):
    def forward(self, x):
        return 2 * (x >= 0.0).float() - 1.0

class RBFActivation(nn.Module):
    def forward(self, x):
        return torch.exp(-(x ** 2))

class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)

class SqrtAbsActivation(nn.Module):
    def forward(self, x):
        return torch.sqrt(torch.abs(x))

class UnitIntervalIndicator(nn.Module):
    def forward(self, x):
        return (torch.abs(x) <= 1.0).float()

class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SquareActivation(nn.Module):
    def forward(self, x):
        return x ** 2

class AbsActivation(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class StdScaleLayer(nn.Module):
    """Standardize input on first forward (fit mean/std along dim=0).

    TabICL applies this BEFORE the activation, not wrapping it.
    Pipeline: StdScaleLayer -> RandomScaleLayer -> Activation.
    """

    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def forward(self, x):
        if self.mean is None or self.std is None:
            self.mean = x.mean(dim=0, keepdim=True)
            self.std = x.std(dim=0, keepdim=True) + 1e-6
        return (x - self.mean) / self.std


class RandomScaleLayer(nn.Module):
    """Random affine: scale * (x + bias).  Lazy-initialized on first forward."""

    def __init__(self):
        super().__init__()
        self.initialized = False

    def forward(self, x):
        if not self.initialized:
            self.scale = torch.exp(
                torch.log(torch.tensor(1.0, device=x.device))
                + 2 * torch.randn(1, 1, device=x.device)
            )
            self.bias = torch.randn(1, 1, device=x.device)
            self.initialized = True
        return self.scale * (x + self.bias)


class RandomFunctionActivation(nn.Module):
    """Random Fourier features with power-law frequency decay.

    Matches TabICL: freqs ~ Uniform(0, n_freq), decay exponent from
    log-uniform, L2-normalized weights, internal StdScaleLayer.
    """

    def __init__(self, n_frequencies=256):
        super().__init__()
        self.freqs = nn.Parameter(
            n_frequencies * torch.rand(n_frequencies), requires_grad=False
        )
        self.bias = nn.Parameter(
            2 * math.pi * torch.rand(n_frequencies), requires_grad=False
        )
        self.stdscaler = StdScaleLayer()

        decay_exponent = -math.exp(
            pyrandom.uniform(math.log(0.7), math.log(3.0))
        )
        with torch.no_grad():
            freq_factors = self.freqs ** decay_exponent
            freq_factors = freq_factors / (freq_factors ** 2).sum().sqrt()
        self.l2_weights = nn.Parameter(
            freq_factors * torch.randn(n_frequencies), requires_grad=False
        )

    def forward(self, x):
        x = self.stdscaler(x)
        x = torch.sin(self.freqs * x[..., None] + self.bias)
        x = (self.l2_weights * x).sum(dim=-1)
        return x


class RandomFreqSineActivation(nn.Module):
    """Sine with log-uniform frequency, random phase, and internal StdScaleLayer."""

    def __init__(self, min_scale=0.1, max_scale=100):
        super().__init__()
        log_min = math.log(min_scale)
        log_max = math.log(max_scale)
        self.scale = nn.Parameter(
            torch.exp(torch.tensor(log_min + (log_max - log_min) * pyrandom.random())),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.tensor(2 * math.pi * pyrandom.random()), requires_grad=False
        )
        self.stdscaler = StdScaleLayer()

    def forward(self, x):
        return torch.sin(self.scale * self.stdscaler(x) + self.bias)


class StdRandomScaleFactory:
    """Factory: StdScaleLayer -> RandomScaleLayer -> Activation (TabICL pattern)."""

    def __init__(self, act_class):
        self.act_class = act_class

    def __call__(self):
        return nn.Sequential(StdScaleLayer(), RandomScaleLayer(), self.act_class())


class RandomChoiceActivation(nn.Module):
    """Randomly selects one activation from a list at construction time."""

    def __init__(self, act_list):
        super().__init__()
        self.act = act_list[pyrandom.randint(0, len(act_list) - 1)]()

    def forward(self, x):
        return self.act(x)


class RandomChoiceFactory:
    """Factory that creates RandomChoiceActivation from a list of factories."""

    def __init__(self, act_factories):
        self.act_factories = act_factories

    def __call__(self):
        return RandomChoiceActivation(self.act_factories)


def get_activations():
    """Build the full activation factory list matching TabICL.

    Returns ~56 callable factories (each produces an nn.Module).
    Pipeline per factory: StdScaleLayer -> RandomScaleLayer -> Activation.
    Plus RandomChoiceFactory entries for per-layer diversity.
    """
    simple_activations = [
        nn.Tanh,
        nn.LeakyReLU,
        nn.ELU,
        nn.Identity,
        nn.SELU,
        nn.SiLU,
        nn.ReLU,
        nn.Softplus,
        nn.ReLU6,
        nn.Hardtanh,
        SignActivation,
        RBFActivation,
        ExpActivation,
        SqrtAbsActivation,
        UnitIntervalIndicator,
        SineActivation,
        SquareActivation,
        AbsActivation,
    ]

    # Add RandomFunctionActivation * 10 for higher selection probability
    activations = simple_activations + [RandomFunctionActivation] * 10  # 28 total

    # Wrap all with StdRandomScaleFactory
    scaled = [StdRandomScaleFactory(act) for act in activations]  # 28 factories

    # Add RandomChoiceFactory for per-layer diversity (28 more)
    scaled += [RandomChoiceFactory(scaled)] * len(scaled)  # 56 total

    return scaled


# ============================================================================
# XSampler — root distribution sampling  (matches TabICL utils.py)
# ============================================================================

class XSampler:
    """Sample root cause variables for the SCM."""

    def __init__(self, n_samples, n_features, sampling="mixed",
                 pre_sample_stats=False, device="cpu"):
        self.n_samples = n_samples
        self.n_features = n_features
        self.sampling = sampling
        self.pre_sample_stats = pre_sample_stats
        self.device = device

        if pre_sample_stats:
            means = np.random.normal(0, 1, n_features)
            stds = np.abs(np.random.normal(0, 1, n_features) * means)
            self.means = torch.tensor(means, dtype=torch.float, device=device).unsqueeze(0)
            self.stds = torch.tensor(stds, dtype=torch.float, device=device).unsqueeze(0)

    def sample(self):
        n, d = self.n_samples, self.n_features
        device = self.device

        if self.sampling == "normal":
            if self.pre_sample_stats:
                return torch.normal(
                    self.means.expand(n, d),
                    self.stds.abs().expand(n, d),
                ).float()
            return torch.randn(n, d, device=device)

        elif self.sampling == "uniform":
            return torch.rand(n, d, device=device)

        else:  # "mixed"
            X = []
            zipf_p = pyrandom.random() * 0.66
            multi_p = pyrandom.random() * 0.66
            normal_p = pyrandom.random() * 0.66

            for j in range(d):
                if pyrandom.random() > normal_p:
                    # Normal
                    if self.pre_sample_stats:
                        x = torch.normal(
                            self.means[0, j].expand(n),
                            self.stds[0, j].abs().expand(n),
                        ).float()
                    else:
                        x = torch.randn(n, device=device)
                elif pyrandom.random() > multi_p:
                    # Multinomial
                    n_cats = pyrandom.randint(2, 20)
                    probs = torch.rand(n_cats, device=device)
                    x = torch.multinomial(probs, n, replacement=True).float()
                    x = (x - x.mean()) / x.std().clamp(min=1e-6)
                elif pyrandom.random() > zipf_p:
                    # Zipf
                    a = 2.0 + pyrandom.random() * 2.0
                    vals = np.random.zipf(a, n)
                    x = torch.tensor(vals, device=device).clamp(max=10).float()
                    x = x - x.mean()
                else:
                    # Uniform
                    x = torch.rand(n, device=device)
                X.append(x)

            return torch.stack(X, dim=-1)


# ============================================================================
# GaussianNoise  (supports per-dim std tensors like TabICL)
# ============================================================================

class GaussianNoise(nn.Module):
    def __init__(self, std):
        super().__init__()
        # std can be a scalar or a tensor [1, out_dim]
        if isinstance(std, torch.Tensor):
            self.register_buffer("std", std)
        else:
            self.std = std

    def forward(self, x):
        if isinstance(self.std, (int, float)):
            if self.std > 0:
                return x + torch.normal(torch.zeros_like(x), self.std)
            return x
        return x + torch.normal(torch.zeros_like(x), self.std.expand_as(x))


# ============================================================================
# Block-wise dropout initialization  (matches TabICL mlp_scm.py)
# ============================================================================

def _block_wise_dropout_init(weight, init_std, dropout_prob,
                              scale_by_dropout=True):
    """Non-overlapping grid-block init matching TabICL.

    Divides weight into n_blocks along each dim, fills diagonal blocks
    with Normal(0, init_std/sqrt(keep_prob)).
    """
    nn.init.zeros_(weight)
    h_out, h_in = weight.shape
    n_blocks = pyrandom.randint(1, math.ceil(math.sqrt(min(h_out, h_in))))
    block_size = [h_out // n_blocks, h_in // n_blocks]
    keep_prob = (n_blocks * block_size[0] * block_size[1]) / weight.numel()

    std = init_std / (keep_prob ** 0.5) if scale_by_dropout else init_std
    for block in range(n_blocks):
        row_slice = slice(block_size[0] * block, block_size[0] * (block + 1))
        col_slice = slice(block_size[1] * block, block_size[1] * (block + 1))
        nn.init.normal_(weight[row_slice, col_slice], std=std)


# ============================================================================
# MLPSCM — MLP-based Structural Causal Model  (matches TabICL mlp_scm.py)
# ============================================================================

class MLPSCM(nn.Module):
    """MLP-based SCM.

    Layer architecture (TabICL):
      layers[0] = Linear(num_causes, hidden_dim)
      layers[1..N-1] = Sequential(Activation, Linear, GaussianNoise)
      layers[N] = Sequential(Activation, Linear, GaussianNoise)  [non-causal only]

    Forward: outputs = [causes]; for layer in layers: outputs.append(layer(outputs[-1]))
    Then outputs = outputs[2:]  (skip causes and first linear projection).
    """

    def __init__(self, n_samples, n_features, n_outputs=1,
                 is_causal=True, num_causes=10, y_is_effect=True,
                 in_clique=False, sort_features=True,
                 num_layers=10, hidden_dim=20,
                 activation_factory=None,
                 init_std=1.0,
                 block_wise_dropout=True, dropout_prob=0.1,
                 scale_init_std_by_dropout=True,
                 sampling="normal", pre_sample_cause_stats=False,
                 noise_std=0.01, pre_sample_noise_std=False,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.is_causal = is_causal
        self.y_is_effect = y_is_effect
        self.in_clique = in_clique
        self.sort_features = sort_features

        # Key TabICL behavior: non-causal sets num_causes = num_features
        if is_causal:
            self.num_causes = num_causes
            hidden_dim = max(hidden_dim, n_outputs + 2 * n_features)
        else:
            self.num_causes = n_features

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        assert num_layers >= 2, "num_layers must be >= 2"

        if activation_factory is None:
            activation_factory = nn.Tanh

        # Build layers as TabICL does: [Linear, Sequential(Act,Linear,Noise), ...]
        layers = [nn.Linear(self.num_causes, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.append(self._make_layer_module(
                hidden_dim, hidden_dim, activation_factory,
                noise_std, pre_sample_noise_std, device,
            ))
        if not is_causal:
            layers.append(self._make_layer_module(
                hidden_dim, n_outputs, activation_factory,
                noise_std, pre_sample_noise_std, device,
                is_output=True,
            ))
        self.layers = nn.Sequential(*layers).to(device)

        # Initialize all parameters
        self._initialize_parameters(
            init_std, block_wise_dropout, dropout_prob, scale_init_std_by_dropout
        )

        # X sampler
        self.x_sampler = XSampler(
            n_samples, self.num_causes, sampling=sampling,
            pre_sample_stats=pre_sample_cause_stats, device=device,
        )

    def _make_layer_module(self, in_dim, out_dim, activation_factory,
                            noise_std, pre_sample_noise_std, device,
                            is_output=False):
        activation = activation_factory()
        linear = nn.Linear(in_dim, out_dim)
        if pre_sample_noise_std:
            ns = torch.abs(
                torch.normal(torch.zeros(1, out_dim, device=device),
                             float(noise_std))
            )
        else:
            ns = noise_std
        return nn.Sequential(activation, linear, GaussianNoise(ns))

    def _initialize_parameters(self, init_std, block_wise_dropout,
                                dropout_prob, scale_by_dropout):
        for i, (name, param) in enumerate(self.layers.named_parameters()):
            if param.dim() != 2:
                # Biases — zero init
                nn.init.zeros_(param)
                continue
            if block_wise_dropout:
                _block_wise_dropout_init(param, init_std, dropout_prob,
                                         scale_by_dropout)
            else:
                nn.init.normal_(param, std=init_std)

    @torch.no_grad()
    def forward(self):
        causes = self.x_sampler.sample()  # [n_samples, num_causes]

        # Forward through all layers, collecting intermediate outputs.
        # Clamp after each layer to prevent magnitude explosion through
        # deep networks with Exp/Square/polynomial activations.
        outputs = [causes]
        for layer in self.layers:
            h = layer(outputs[-1])
            h = h.clamp(-100, 100)
            outputs.append(h)

        # Skip causes (idx 0) and first linear projection (idx 1)
        outputs = outputs[2:]

        X, y = self._handle_outputs(causes, outputs)

        # NaN handling: TabICL poisons entire dataset if any NaN
        if torch.any(torch.isnan(X)) or torch.any(torch.isnan(y)):
            X = torch.zeros_like(X)
            y = torch.full_like(y, -100.0)

        if self.n_outputs == 1 and y.dim() > 1:
            y = y.squeeze(-1)

        return X, y

    def _handle_outputs(self, causes, outputs):
        """Extract X and y from intermediate outputs."""
        if self.is_causal:
            outputs_flat = torch.cat(outputs, dim=-1)
            total_dim = outputs_flat.shape[-1]

            if self.in_clique:
                start = pyrandom.randint(
                    0, max(0, total_dim - self.n_outputs - self.n_features)
                )
                random_perm = start + torch.randperm(
                    self.n_outputs + self.n_features, device=self.device
                )
            else:
                random_perm = torch.randperm(
                    max(total_dim - 1, 1), device=self.device
                )

            indices_X = random_perm[self.n_outputs:self.n_outputs + self.n_features]
            if self.y_is_effect:
                indices_y = list(range(-self.n_outputs, 0))
            else:
                indices_y = random_perm[:self.n_outputs]

            if self.sort_features:
                indices_X, _ = torch.sort(indices_X)

            # Clamp to valid range
            indices_X = indices_X.clamp(0, total_dim - 1)
            if isinstance(indices_y, list):
                # Negative indexing for y_is_effect
                X = outputs_flat[:, indices_X]
                y = outputs_flat[:, indices_y]
            else:
                indices_y = indices_y.clamp(0, total_dim - 1)
                X = outputs_flat[:, indices_X]
                y = outputs_flat[:, indices_y]
        else:
            X = causes
            y = outputs[-1]

        return X, y


# ============================================================================
# TreeSCM — Tree-based Structural Causal Model  (matches TabICL tree_scm.py)
# ============================================================================

class TreeLayer:
    """A single tree-based transformation layer."""

    def __init__(self, tree_model, max_depth, n_estimators, out_dim, rng):
        self.out_dim = out_dim
        self.models = []

        for _ in range(out_dim):
            if tree_model == "xgboost" and HAS_XGB:
                model = XGBRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=0.3,
                    verbosity=0,
                    random_state=int(rng.integers(0, 2**31)),
                )
            elif tree_model == "extra_trees" and HAS_SKLEARN:
                model = ExtraTreesRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=int(rng.integers(0, 2**31)),
                )
            elif tree_model == "random_forest" and HAS_SKLEARN:
                model = RandomForestRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    random_state=int(rng.integers(0, 2**31)),
                )
            else:
                model = DecisionTreeRegressor(
                    max_depth=max_depth,
                    splitter="random",
                    random_state=int(rng.integers(0, 2**31)),
                ) if HAS_SKLEARN else None
                if model is None:
                    raise ImportError("sklearn is required for TreeSCM")
            self.models.append(model)

    def fit_transform(self, X, rng):
        n = X.shape[0]
        out = np.zeros((n, self.out_dim), dtype=np.float32)
        for i, model in enumerate(self.models):
            y_fake = rng.standard_normal(n).astype(np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X, y_fake)
            out[:, i] = model.predict(X).astype(np.float32)
        return out


class TreeSCM:
    """Tree-based SCM (always non-causal, shallow layers for speed)."""

    def __init__(self, n_samples, n_features, n_outputs=1,
                 num_causes=10, tree_model="xgboost",
                 max_depth_lambda=0.5, n_estimators_lambda=0.5,
                 sampling="normal", pre_sample_cause_stats=False,
                 noise_std=0.01, pre_sample_noise_std=False,
                 rng=None, device="cpu"):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_outputs = n_outputs
        # TabICL hardcodes is_causal=False for trees, so num_causes = num_features
        self.num_causes = n_features
        self.noise_std = noise_std
        self.device = device
        self.rng = rng if rng is not None else np.random.default_rng()

        # TabICL overrides: shallow for speed
        self.num_layers = int(self.rng.integers(1, 3))
        self.hidden_dim = int(self.rng.integers(3, 11))

        if HAS_XGB:
            self.tree_model = tree_model
        else:
            self.tree_model = "extra_trees"

        self.max_depth_lambda = max_depth_lambda
        self.n_estimators_lambda = n_estimators_lambda

        self.x_sampler = XSampler(
            n_samples, self.num_causes, sampling=sampling,
            pre_sample_stats=pre_sample_cause_stats, device=device,
        )

    def generate(self):
        rng = self.rng
        causes = self.x_sampler.sample().cpu().numpy()

        h = causes
        for layer_idx in range(self.num_layers):
            max_depth = 2 + int(np.random.exponential(
                1.0 / max(self.max_depth_lambda, 0.01)))
            max_depth = min(max_depth, 4)
            n_estimators = 1 + int(np.random.exponential(
                1.0 / max(self.n_estimators_lambda, 0.01)))
            n_estimators = min(n_estimators, 4)

            out_dim = self.hidden_dim if layer_idx < self.num_layers - 1 else self.n_outputs

            tree_layer = TreeLayer(
                self.tree_model, max_depth, n_estimators, out_dim, rng)
            h = tree_layer.fit_transform(h, rng)

            if self.noise_std > 0:
                h = h + rng.standard_normal(h.shape).astype(np.float32) * self.noise_std

        # Non-causal: X = causes, y = final output
        X = causes
        y = h.squeeze(-1) if h.ndim > 1 and h.shape[-1] == 1 else h
        if y.ndim > 1:
            y = y[:, 0]

        return X.astype(np.float32), y.astype(np.float32)


# ============================================================================
# Reg2Cls — Convert regression targets to classification
# ============================================================================

def _outlier_remove(X, threshold=4.0):
    """Two-pass outlier clamping (matches TabICL reg2cls.py)."""
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    mask = (lower <= X) & (X <= upper) & ~np.isnan(X)
    X_clean = np.where(mask, X, np.nan)
    mean2 = np.nanmean(X_clean, axis=0, keepdims=True)
    std2 = np.nanstd(X_clean, axis=0, keepdims=True)
    std2 = np.where(np.isnan(std2) | (std2 < 1e-6), 1.0, std2)
    mean2 = np.where(np.isnan(mean2), mean, mean2)

    cut_off2 = std2 * threshold
    lower2, upper2 = mean2 - cut_off2, mean2 + cut_off2
    return np.clip(X, lower2, upper2)


def _standard_scale(X, clip_value=100.0):
    """Standardize columns to zero mean, unit variance."""
    mean = np.nanmean(X, axis=0, keepdims=True)
    std = np.nanstd(X, axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return np.clip((X - mean) / std, -clip_value, clip_value)


def _multiclass_assign(y, n_classes, mode="rank", ordered_prob=0.0, rng=None):
    """Convert continuous y to discrete labels (matches TabICL MulticlassAssigner)."""
    if rng is None:
        rng = np.random.default_rng()

    n = len(y)
    if n_classes == 2:
        labels = (y > np.median(y)).astype(np.int64)
        if len(np.unique(labels)) < 2:
            labels = (y >= np.percentile(y, 50)).astype(np.int64)
        return labels

    if mode == "rank":
        boundary_indices = rng.choice(n, size=n_classes - 1, replace=True)
        boundaries = y[boundary_indices]
    else:
        boundaries = np.sort(rng.standard_normal(n_classes - 1))

    labels = np.sum(y[:, None] > boundaries[None, :], axis=1)
    labels = np.clip(labels, 0, n_classes - 1)

    if pyrandom.random() > ordered_prob:
        perm = rng.permutation(n_classes)
        labels = perm[labels]

    if pyrandom.random() > 0.5:
        labels = n_classes - 1 - labels

    return labels.astype(np.int64)


def _num2cat(X, cat_prob=0.2, max_categories=float("inf"), rng=None):
    """Randomly categorize some features (matches TabICL Reg2Cls._num2cat)."""
    if rng is None:
        rng = np.random.default_rng()

    if pyrandom.random() >= cat_prob:
        return X

    X = X.copy()
    n_features = X.shape[1]
    col_prob = pyrandom.random()

    for j in range(n_features):
        if pyrandom.random() < col_prob:
            n_cats = min(max(round(pyrandom.gammavariate(1, 10)), 2),
                         int(min(max_categories, 100)))
            # Rank-based discretization using MulticlassAssigner logic
            col = X[:, j]
            ranks = np.argsort(np.argsort(col))
            n = len(col)
            X[:, j] = (ranks * n_cats // n).astype(np.float32)

    return X


def _permute_classes(labels, n_classes):
    """Label-encode and randomly permute class labels (TabICL permute_classes)."""
    unique = np.unique(labels)
    if len(unique) <= 1:
        return labels
    # Re-encode to 0..n_unique-1
    sort_idx = np.argsort(unique)
    mapped = sort_idx[np.searchsorted(unique, labels)]
    perm = np.random.permutation(len(unique))
    return perm[mapped].astype(np.int64)


def reg2cls(X, y, n_classes, rng=None):
    """Convert regression (X, y) to classification.

    Matches TabICL Reg2Cls.forward() pipeline:
      X = num2cat(X) -> outlier_remove -> standard_scale -> permute_features
      y = standard_scale -> class_assigner -> permute_labels
    """
    if rng is None:
        rng = np.random.default_rng()

    # Feature processing
    X = _num2cat(X, cat_prob=0.2, rng=rng)
    X = _outlier_remove(X, threshold=4.0)
    X = _standard_scale(X, clip_value=100.0)

    # Permute features (TabICL default: permute_features=True)
    perm = rng.permutation(X.shape[1])
    X = X[:, perm]

    # Target processing
    y_std = np.std(y)
    if y_std > 1e-8:
        y = (y - np.mean(y)) / y_std

    # Assign classes
    mode = rng.choice(["rank", "value"])
    labels = _multiclass_assign(y, n_classes, mode=mode, ordered_prob=0.0, rng=rng)

    # Permute labels (TabICL default: permute_labels=True)
    labels = _permute_classes(labels, n_classes)

    # Ensure contiguous classes
    unique = np.unique(labels)
    if len(unique) < n_classes:
        mapping = {old: new for new, old in enumerate(unique)}
        labels = np.array([mapping[l] for l in labels], dtype=np.int64)
        n_classes = len(unique)

    return X, labels, n_classes


# ============================================================================
# Meta-distribution hyperparameter sampling
# ============================================================================

def meta_trunc_norm_log_scaled(rng, min_mean=0.01, max_mean=10.0,
                                round_val=False, lower_bound=0.0):
    """Log-scaled truncated normal (TabICL meta_trunc_norm_log_scaled)."""
    log_mean = rng.uniform(
        math.log(max(min_mean, 1e-8)), math.log(max(max_mean, 1e-7)))
    mu = math.exp(log_mean)
    log_std = rng.uniform(math.log(0.01), math.log(1.0))
    sigma = mu * math.exp(log_std)

    def sample():
        val = rng.normal(mu, max(sigma, 1e-8))
        val = max(val, lower_bound)
        if round_val:
            val = max(int(round(val)), int(math.ceil(lower_bound)))
        return val

    return sample


def meta_beta(rng, scale=1.0, min_b=0.1, max_b=5.0, min_k=0.1, max_k=5.0):
    """Beta with meta-sampled shape params (TabICL meta_beta)."""
    b = rng.uniform(min_b, max_b)
    k = rng.uniform(min_k, max_k)

    def sample():
        return float(rng.beta(b, k) * scale)

    return sample


def meta_choice(rng, values):
    """Weighted choice with softmax-sampled weights (TabICL meta_choice)."""
    n = len(values)
    if n == 0:
        raise ValueError("meta_choice requires at least one value")
    weights = np.zeros(n)
    weights[0] = 1.0
    for i in range(1, n):
        weights[i] = rng.uniform(-3, 5)
    weights = weights - weights.max()
    exp_w = np.exp(weights)
    probs = exp_w / exp_w.sum()

    def sample():
        return values[rng.choice(n, p=probs)]

    return sample


def sample_hyperparams(rng, n_features, device="cpu"):
    """Sample a complete set of HPs via meta-distributions."""
    hp = {}

    # SCM type: 70% MLP, 30% Tree (TabICL mix_probs default)
    hp["scm_type"] = "mlp" if rng.random() < 0.7 else "tree"

    # Shared SCM parameters
    hp["is_causal"] = meta_choice(rng, [True, False])()
    hp["num_causes"] = meta_trunc_norm_log_scaled(
        rng, max_mean=12, min_mean=1, round_val=True, lower_bound=1)()
    hp["y_is_effect"] = meta_choice(rng, [True, False])()
    hp["in_clique"] = meta_choice(rng, [True, False])()
    hp["sort_features"] = meta_choice(rng, [True, False])()
    hp["num_layers"] = meta_trunc_norm_log_scaled(
        rng, max_mean=6, min_mean=1, round_val=True, lower_bound=2)()
    hp["hidden_dim"] = meta_trunc_norm_log_scaled(
        rng, max_mean=130, min_mean=5, round_val=True, lower_bound=4)()
    hp["init_std"] = meta_trunc_norm_log_scaled(
        rng, max_mean=10.0, min_mean=0.01, lower_bound=0.0)()
    hp["noise_std"] = meta_trunc_norm_log_scaled(
        rng, max_mean=0.3, min_mean=0.0001, lower_bound=0.0)()
    hp["sampling"] = meta_choice(rng, ["normal", "mixed", "uniform"])()
    hp["pre_sample_cause_stats"] = meta_choice(rng, [True, False])()
    hp["pre_sample_noise_std"] = meta_choice(rng, [True, False])()

    # MLP-specific
    hp["block_wise_dropout"] = meta_choice(rng, [True, False])()
    hp["dropout_prob"] = meta_beta(rng, scale=0.9, min_b=0.1, max_b=5.0)()

    # Tree-specific
    hp["tree_model"] = "xgboost" if HAS_XGB else "extra_trees"
    hp["max_depth_lambda"] = 0.5
    hp["n_estimators_lambda"] = 0.5

    # Reg2Cls
    hp["multiclass_type"] = meta_choice(rng, ["value", "rank"])()

    return hp


# ============================================================================
# Top-level generation function
# ============================================================================

def generate_tabicl_dataset(n_samples, n_features, task_type, n_classes=None,
                             rng=None, device="cpu"):
    """Generate a single dataset using TabICL's prior system.

    Returns dict(X=[n_samples, n_features], y=[n_samples], n_classes=int|None).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample n_classes (TabICL: 50% binary, 50% uniform 2-10)
    if task_type == 'cls' and n_classes is None:
        if rng.random() > 0.5:
            n_classes = int(rng.integers(2, 11))
        else:
            n_classes = 2

    hp = sample_hyperparams(rng, n_features, device=device)

    if hp["scm_type"] == "mlp":
        X, y = _generate_mlp(n_samples, n_features, hp, rng, device)
    else:
        X, y = _generate_tree(n_samples, n_features, hp, rng, device)

    # --- Robust stabilization in torch (before numpy conversion) ---
    # MLP SCM with ExpActivation / polynomials can produce huge values.
    # Standardize per-column with median/MAD to avoid float32 overflow in
    # numpy's std() (which squares values internally).
    if isinstance(X, torch.Tensor):
        X = X.float()
        X = torch.where(torch.isfinite(X), X, torch.zeros_like(X))
        X = X.clamp(-1e6, 1e6)
        med = X.median(dim=0).values.unsqueeze(0)
        mad = (X - med).abs().median(dim=0).values.unsqueeze(0)
        mad = mad.clamp(min=1e-6)
        X = (X - med) / mad
        X = X.clamp(-50, 50)
        X = X.cpu().numpy()
    else:
        X = np.clip(X, -1e6, 1e6)
        med = np.median(X, axis=0, keepdims=True)
        mad = np.median(np.abs(X - med), axis=0, keepdims=True)
        mad = np.where(mad < 1e-6, 1.0, mad)
        X = (X - med) / mad
        X = np.clip(X, -50, 50)

    if isinstance(y, torch.Tensor):
        y = y.float()
        y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
        y = y.clamp(-1e6, 1e6)
        y = y.cpu().numpy()
    else:
        y = np.clip(y, -1e6, 1e6)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Safety: catch any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0)
    y = np.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)

    # Delete constant features (TabICL delete_unique_features)
    col_std = np.std(X, axis=0)
    keep_mask = col_std > 1e-8
    if keep_mask.sum() < n_features:
        X[:, ~keep_mask] = 0.0

    if task_type == 'cls':
        X, y, n_classes = reg2cls(X, y, n_classes, rng=rng)

    if task_type == 'reg':
        y_std = np.std(y)
        if y_std > 1e-8:
            y = (y - np.mean(y)) / y_std
        y = np.clip(y, -1e4, 1e4)

    X = np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=-50.0)

    return {
        'X': X,
        'y': y,
        'n_classes': n_classes if task_type == 'cls' else None,
    }


def _generate_mlp(n_samples, n_features, hp, rng, device):
    """Generate data using MLP SCM."""
    act_factories = get_activations()

    # TabICL meta_choice_mixed: sample weights, then pick per-call
    n_acts = len(act_factories)
    act_weights = np.zeros(n_acts)
    act_weights[0] = 1.0
    for i in range(1, n_acts):
        act_weights[i] = rng.uniform(-5, 6)
    act_weights = act_weights - act_weights.max()
    act_probs = np.exp(act_weights) / np.exp(act_weights).sum()
    act_idx = rng.choice(n_acts, p=act_probs)
    act_factory = act_factories[act_idx]

    try:
        scm = MLPSCM(
            n_samples=n_samples,
            n_features=n_features,
            n_outputs=1,
            is_causal=hp["is_causal"],
            num_causes=hp["num_causes"],
            y_is_effect=hp["y_is_effect"],
            in_clique=hp["in_clique"],
            sort_features=hp["sort_features"],
            num_layers=hp["num_layers"],
            hidden_dim=hp["hidden_dim"],
            activation_factory=act_factory,
            init_std=hp["init_std"],
            block_wise_dropout=hp["block_wise_dropout"],
            dropout_prob=hp["dropout_prob"],
            sampling=hp["sampling"],
            pre_sample_cause_stats=hp["pre_sample_cause_stats"],
            noise_std=hp["noise_std"],
            pre_sample_noise_std=hp["pre_sample_noise_std"],
            device=device,
        )
        scm.to(device)
        X, y = scm()
    except Exception:
        # Fallback: random data if SCM construction fails
        X = torch.randn(n_samples, n_features, device=device)
        y = torch.randn(n_samples, device=device)

    return X, y


def _generate_tree(n_samples, n_features, hp, rng, device):
    """Generate data using Tree SCM."""
    try:
        scm = TreeSCM(
            n_samples=n_samples,
            n_features=n_features,
            n_outputs=1,
            num_causes=hp["num_causes"],
            tree_model=hp["tree_model"],
            max_depth_lambda=hp["max_depth_lambda"],
            n_estimators_lambda=hp["n_estimators_lambda"],
            sampling=hp["sampling"],
            pre_sample_cause_stats=hp["pre_sample_cause_stats"],
            noise_std=hp["noise_std"],
            pre_sample_noise_std=hp["pre_sample_noise_std"],
            rng=rng,
            device=device,
        )
        X, y = scm.generate()
    except Exception:
        X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        y = rng.standard_normal(n_samples).astype(np.float32)

    return X, y
