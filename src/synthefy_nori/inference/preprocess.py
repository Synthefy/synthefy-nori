from __future__ import annotations

import numpy as np
import torch
import warnings
from typing_extensions import override
from typing import Literal, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
    PowerTransformer,
    StandardScaler,
    QuantileTransformer, 
    MinMaxScaler,
    RobustScaler
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted
from synthefy_nori.inference.degradation import SvdFallbackWarning
from synthefy_nori.utils.data_utils import NoriInferenceDataset
from torch.cuda import OutOfMemoryError

import hashlib
import os
from kditransform import KDITransformer

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)


# Module-level helpers for FunctionTransformer steps. These must be importable
# (i.e. not local lambdas) so that fitted pipelines — and the NoriRegressor that
# owns them — can be serialized with stdlib pickle, which only pickles functions
# by reference. See GitHub issue #45.
def _inf_to_nan(x):
    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)


def _identity(x):
    return x


def _shift_to_nonnegative(x):
    return x + np.abs(np.nanmin(x))


def _add_epsilon(x):
    return x + 1e-10


class CappedQuantileTransformer(QuantileTransformer):
    """QuantileTransformer that caps ``n_quantiles`` and ``subsample`` at fit time.

    sklearn's QuantileTransformer fit (``np.nanpercentile``) is slow when
    ``n_quantiles`` is large and fit on all rows. Quantile boundaries are
    statistically stable far below the raw request, so capping is
    ~accuracy-neutral while much faster. Gated by ``SYNTHEFY_CAP_QUANTILES``
    (default on) so it can be A/B'd against the uncapped path.
    """

    def fit(self, X, y=None):
        if os.environ.get("SYNTHEFY_CAP_QUANTILES", "1") == "1":
            cap_q = int(os.environ.get("SYNTHEFY_QUANTILE_MAX", "256"))
            cap_s = int(os.environ.get("SYNTHEFY_QUANTILE_SUBSAMPLE", "10000"))
            n = X.shape[0] if hasattr(X, "shape") else len(X)
            self.n_quantiles = max(2, min(int(self.n_quantiles), n, cap_q))
            self.subsample = min(int(self.subsample), cap_s)
        return super().fit(X, y)


# Device used for GPU SVD; set by NoriPredictor.__init__ to its own device.
# Falls back to the current CUDA device (or CPU) when unset.
_GPU_SVD_DEVICE = None


def _resolve_svd_device():
    if _GPU_SVD_DEVICE is not None:
        return torch.device(_GPU_SVD_DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


class _TorchTruncatedSVD(BaseEstimator, TransformerMixin):
    """Drop-in for sklearn ``TruncatedSVD`` computing the exact truncated SVD on
    GPU via ``torch.linalg.svd``.

    ``transform(X) == X @ components_.T`` and ``fit_transform`` returns ``U * S``
    — the same contract as sklearn's TruncatedSVD, with the same
    ``svd_flip(u_based_decision=True)`` sign convention (so output matches up to
    the randomized-vs-exact gap). sklearn's CPU TruncatedSVD dominates inference
    cost on high-dimensional datasets; the GPU exact SVD collapses that. Falls
    back to sklearn on any error and is gated by ``SYNTHEFY_GPU_SVD`` (default
    on). Randomized-only kwargs (algorithm/n_iter/n_oversamples) are accepted and
    ignored so it slots into sklearn Pipelines unchanged.
    """

    def __init__(self, n_components=2, *, random_state=None, algorithm=None,
                 n_iter=None, n_oversamples=None):
        self.n_components = int(n_components)
        self.random_state = random_state
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples

    def _sklearn_fallback(self, X):
        algo = self.algorithm or "randomized"
        m = TruncatedSVD(n_components=self.n_components, algorithm=algo,
                         random_state=self.random_state)
        out = m.fit_transform(X)
        self.components_ = m.components_
        return out

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        if os.environ.get("SYNTHEFY_GPU_SVD", "1") != "1":
            return self._sklearn_fallback(X)
        try:
            Xnp = np.asarray(X, dtype=np.float64)
            n, p = Xnp.shape
            k = max(1, min(int(self.n_components), p, n))
            dev = _resolve_svd_device()
            Xt = torch.from_numpy(Xnp).to(dev)
            U, S, Vh = torch.linalg.svd(Xt, full_matrices=False)
            U = U[:, :k]; S = S[:k]; Vh = Vh[:k]
            # svd_flip, u_based_decision=True (sklearn's default for both solvers):
            # sign each component so the largest-|.| entry of its U column is +.
            idx = U.abs().argmax(dim=0)
            signs = torch.sign(U[idx, torch.arange(k, device=dev)])
            signs = torch.where(signs == 0, torch.ones_like(signs), signs)
            Vh = Vh * signs[:, None]
            US = U * (S * signs)[None, :]
            self.components_ = Vh.detach().cpu().numpy()
            return US.detach().cpu().numpy()
        except Exception:
            return self._sklearn_fallback(X)

    def transform(self, X):
        Xnp = np.asarray(X, dtype=np.float64)
        # Projection matmul X @ components_.T is large on wide test sets; do it on
        # GPU when available (same result as numpy), else fall back to numpy.
        if os.environ.get("SYNTHEFY_GPU_SVD", "1") == "1":
            try:
                dev = _resolve_svd_device()
                if dev.type == "cuda":
                    Xt = torch.from_numpy(Xnp).to(dev)
                    Ct = torch.from_numpy(self.components_).to(dev)
                    return (Xt @ Ct.T).detach().cpu().numpy()
            except Exception:
                pass
        return Xnp @ self.components_.T

class SelectiveInversePipeline(Pipeline):
    def __init__(self, steps, skip_inverse=None):
        super().__init__(steps)
        self.skip_inverse = skip_inverse or []
    
    def inverse_transform(self, X):
        """inverse_transform that skips the configured steps."""
        if X.shape[1] == 0:
            return X
        for step_idx in range(len(self.steps) - 1, -1, -1):
            name, transformer = self.steps[step_idx]
            try:
                check_is_fitted(transformer)
            except Exception:
                continue
            
            if name in self.skip_inverse:
                continue
                
            if hasattr(transformer, 'inverse_transform'):
                X = transformer.inverse_transform(X)
                if np.any(np.isnan(X)):
                    print(f"After reverse RebalanceFeatureDistribution of {name}, there is nan")
        return X

class RobustPowerTransformer(PowerTransformer):
    """PowerTransformer with automatic feature reversion when variance or value constraints fail."""

    def __init__(self, var_tolerance: float = 1e-3,
                 max_abs_value: float = 100,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.var_tolerance = var_tolerance
        self.max_abs_value = max_abs_value
        self.restore_indices_: np.ndarray | None = None


    def fit(self, X, y=None):
        fitted = super().fit(X, y)
        self.restore_indices_ = np.array([], dtype=int)
        return fitted

    def fit_transform(self, X, y=None):
        Z = super().fit_transform(X,y)
        self.restore_indices_ = self._should_revert(Z)
        return Z

    def _should_revert(self, Z: np.ndarray) -> np.ndarray:
        """Determine which columns to revert to their original values."""
        variances = np.nanvar(Z, axis=0)
        bad_var = np.flatnonzero(np.abs(variances - 1.0) > self.var_tolerance)

        bad_large = np.flatnonzero(np.any(Z > self.max_abs_value, axis=0))

        return np.unique(np.concatenate([bad_var, bad_large]))

    def _apply_reversion(self, Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.restore_indices_.size > 0:
            Z[:, self.restore_indices_] = X[:, self.restore_indices_]
        return Z

    def transform(self, X):
        Z = super().transform(X)
        # self.restore_indices_ = self._should_revert(Z)
        return self._apply_reversion(Z, X)

    def _yeo_johnson_optimize(self, x: np.ndarray) -> float:
        "Overload_yeo_johnson_optimize to avoid crashes caused by values such as NaN and Inf."
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message=r"overflow encountered",
                                        category=RuntimeWarning)
                return super()._yeo_johnson_optimize(x)  # type: ignore
        except Exception as e:
            return np.nan

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        "_yeo_johnson_transform to avoid crashes caused by NaN"
        if np.isnan(lmbda):
            return x
        return super()._yeo_johnson_transform(x, lmbda)  # type: ignore

class AdaptiveColumnTransformer(BaseEstimator, TransformerMixin):
    """TabPFN-style per-column adaptive preprocessor.

    Routes each numeric column through a transformer chosen by the column's
    own distribution (ordinal / skewed-positive / skewed / normal / other),
    instead of applying one global transform to all features. Closes the gap
    on heterogeneous-feature datasets (houses, MIP-2016, topo_2_1) where a
    uniform quantile/power transform is sub-optimal for any specific column.

    Column type rules (matches TabPFN's get_adaptive_preprocessors):
      ordinal       (<10 unique values)              → identity
      skewed_pos_01 (skew>1.1, x∈[0,1])              → exp
      skewed_pos    (skew>1.1, x>0)                  → safe box-cox + standardize
      skewed        (|skew|>1.1)                     → safe yeo-johnson + standardize
      normal        (Shapiro stat>0.95 on first 3K)  → identity
      other                                           → quantile-normal (n_quantiles)

    `fit_transform` is sklearn-compatible. Inverse-transform is supported
    column-wise via the per-column inverse path; this is needed for masked
    feature reconstruction (mask_prediction=True). For regression-only
    inference no inversion is needed.
    """

    def __init__(self, *, random_state: int | None = None, n_quantiles: int = 100):
        self.random_state = random_state
        self.n_quantiles = max(2, int(n_quantiles))

    @staticmethod
    def _skew(x: np.ndarray) -> float:
        """Median-based skew proxy, bounded ~[-3,3]. Cheap, stable on small N."""
        x = x[np.isfinite(x)]
        if len(x) < 3:
            return 0.0
        std = float(np.nanstd(x))
        if std < 1e-8:
            return 0.0
        return float(3.0 * (np.nanmean(x) - np.nanmedian(x)) / std)

    @staticmethod
    def _fp_skew(x: np.ndarray) -> float:
        """Fisher-Pearson skew (m3 / m2^1.5). Unbounded; senses heavy tails."""
        x = x[np.isfinite(x)]
        if len(x) < 3:
            return 0.0
        try:
            from scipy.stats import skew as _scipy_skew
            return float(_scipy_skew(x, bias=False, nan_policy='omit'))
        except Exception:
            return 0.0

    @classmethod
    def _categorize(cls, x_col: np.ndarray) -> str:
        x_finite = x_col[np.isfinite(x_col)]
        if len(x_finite) < 3:
            return 'identity'
        n_unique = int(len(np.unique(x_finite)))
        if n_unique < 10:
            return 'ordinal'
        skew_val = cls._skew(x_finite)
        x_min, x_max = float(np.min(x_finite)), float(np.max(x_finite))
        # Heavy positive-skew (income/price/count distributions): use log1p
        # before standardization. Detected via Fisher-Pearson skew (unbounded);
        # threshold |fp_skew|>5 catches genuinely heavy-right-tail data where
        # yeo-johnson under-fits. Median-based proxy maxes near 1 and can't
        # distinguish "moderate" from "extreme" skew.
        if x_min >= 0:
            fp = cls._fp_skew(x_finite)
            if fp > 5.0:
                return 'positive_heavy_skew'
        if abs(skew_val) > 1.1:
            if x_min >= 0:
                if x_max <= 1.0:
                    return 'skewed_pos_01'
                return 'skewed_pos'
            return 'skewed'
        # Shapiro normality on a sample (it's expensive on large N)
        try:
            from scipy.stats import shapiro
            sub = x_finite[:3000] if len(x_finite) > 3000 else x_finite
            stat = float(shapiro(sub).statistic)
            if stat > 0.95:
                return 'normal'
        except Exception:
            pass
        return 'other'

    def _make_transformer(self, cat: str):
        """Return a fresh sklearn transformer for the given column category."""
        if cat in ('identity', 'ordinal', 'normal'):
            return FunctionTransformer()
        if cat == 'skewed_pos_01':
            # exp pulls (0,1)-bounded right-skew into broader range; identity inverse via log.
            return FunctionTransformer(
                func=np.exp, inverse_func=np.log, check_inverse=False,
            )
        # scikit-learn PowerTransformer (Yeo-Johnson) for skewed columns.
        # (Verified numerically identical to the previously-optional
        # SafePowerTransformer on the winsorized feature ranges seen here.)
        _PowerImpl = PowerTransformer
        if cat == 'positive_heavy_skew':
            # log1p first (textbook for heavy positive skew: income/price/counts),
            # then standardize. yeo-johnson under-fits very heavy tails.
            return Pipeline([
                ('log1p', FunctionTransformer(func=np.log1p, inverse_func=np.expm1, check_inverse=False)),
                ('std', StandardScaler()),
            ])
        if cat == 'skewed_pos':
            # Box-cox needs strictly positive input; ensure with MinMax→(0.1,1).
            return Pipeline([
                ('mm', MinMaxScaler(feature_range=(0.1, 1.0), clip=True)),
                ('bc', _PowerImpl(method='box-cox', standardize=True)),
            ])
        if cat == 'skewed':
            return _PowerImpl(method='yeo-johnson', standardize=True)
        # 'other'
        return CappedQuantileTransformer(
            output_distribution='normal',
            n_quantiles=self.n_quantiles,
            random_state=self.random_state,
            subsample=int(1e6),
        )

    def fit(self, X: np.ndarray, y=None) -> 'AdaptiveColumnTransformer':
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        self.column_categories_: list[str] = []
        self.column_transformers_: list[Any] = []
        # Row-subsample for *fitting* the per-column transformers. The box-cox /
        # yeo-johnson lambda is a 1-parameter MLE and the standardize mean/std
        # are 2-parameter estimates, all converging tightly well below the full
        # row count, so fitting on a capped subsample is ~indistinguishable in
        # output but avoids O(n) scipy iterations over every row (a dominant
        # inference cost on many-row datasets). Categorization (the column *type*
        # decision) still runs on the FULL column so routing is byte-identical;
        # only fitted parameters come from the subsample, and transform() always
        # applies to all rows. SYNTHEFY_ADAPTIVE_FIT_SUBSAMPLE=0 -> exact legacy.
        cap = int(os.environ.get("SYNTHEFY_ADAPTIVE_FIT_SUBSAMPLE", "2000"))
        if cap > 0 and n_samples > cap:
            fit_idx = np.random.default_rng(self.random_state).choice(
                n_samples, size=cap, replace=False)
            X_fit = X[fit_idx]
        else:
            X_fit = X
        for i in range(n_features):
            cat = self._categorize(X[:, i])          # exact: full column
            t = self._make_transformer(cat)
            fit_col = X_fit[:, i:i + 1]               # cheap: fit on subsample
            try:
                # sklearn ColumnTransformer-style: fit a 2D slice
                t.fit(fit_col)
            except Exception:
                # Degenerate column (all-NaN, all-equal): fall back to identity
                t = FunctionTransformer()
                t.fit(X[:, i:i + 1])
                cat = 'identity'
            self.column_categories_.append(cat)
            self.column_transformers_.append(t)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_features = X.shape[1]
        if n_features != len(self.column_transformers_):
            raise ValueError(
                f"AdaptiveColumnTransformer fitted on {len(self.column_transformers_)} "
                f"features, got {n_features} at transform time."
            )
        out = np.empty_like(X, dtype=np.float64)
        for i, t in enumerate(self.column_transformers_):
            try:
                col = t.transform(X[:, i:i + 1])
                out[:, i:i + 1] = np.asarray(col, dtype=np.float64)
            except Exception:
                # Test-time corrupted column (all NaN, etc.): fall back to median fill 0
                out[:, i] = 0.0
        # Replace any inf/nan from extreme transforms
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X, dtype=np.float64)
        for i, t in enumerate(self.column_transformers_):
            try:
                if hasattr(t, 'inverse_transform'):
                    out[:, i:i + 1] = t.inverse_transform(X[:, i:i + 1])
                else:
                    out[:, i] = X[:, i]
            except Exception:
                out[:, i] = X[:, i]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class BasePreprocess:
    """Abstract base class for preprocessing class"""

    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs)->list[int]:
        """Fit the preprocessing model to the data"""
        raise NotImplementedError
    
    def transform(self, x:np.ndarray, **kwargs)->tuple[np.ndarray, list[int]]:
        """Transform the data using the fitted preprocessing model"""
        raise NotImplementedError
    
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs)->tuple[np.ndarray, list[int]]:
        """Fit the preprocessing model to the data and transform the data"""
        self.fit(x, categorical_features, seed, **kwargs)
        return self.transform(x, **kwargs)

def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer the random state and return the seed and generator"""
    if random_state is None:
        np_rng = np.random.default_rng()
        return int(np_rng.integers(0, MAXINT_RANDOM_SEED)), np_rng
        
    if isinstance(random_state, (int, np.integer)):
        return int(random_state), np.random.default_rng(random_state)
        
    if isinstance(random_state, np.random.RandomState):
        seed = int(random_state.randint(0, MAXINT_RANDOM_SEED))
        return seed, np.random.default_rng(seed)
        
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, MAXINT_RANDOM_SEED)), random_state
        
    raise ValueError(f"Invalid random_state {random_state}")

class FilterValidFeatures(BasePreprocess):
    def __init__(self):
        self.valid_features: list[bool] | None = None
        self.categorical_idx: list[int] | None = None
        self.invalid_indices: list[int] | None = None
        self.invalid_features: list[int] | None = None

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, y:np.ndarray | None = None, **kwargs) -> list[int]:
        self.categorical_idx = categorical_features
        self.valid_features = np.asarray(
            (x[0:1, :] == x).mean(axis=0) < 1.0,
            dtype=bool,
        )
        self.invalid_indices = np.asarray(
            (x[0:1, :] == x).mean(axis=0) == 1.0,
            dtype=bool,
        )

        # Train-only fitting needs to drop all-NaN columns directly.
        all_nan_fit = np.all(np.isnan(x), axis=0)
        self.valid_features = self.valid_features & ~all_nan_fit
        self.invalid_indices = self.invalid_indices | all_nan_fit

        if y is not None and len(x) > len(y):
            eval_pos = len(y)
            nan_train = np.isnan(x[:eval_pos, :])
            all_nan_train = np.all(nan_train, axis=0)
            nan_test = np.isnan(x[eval_pos:, :])
            all_nan_test = np.all(nan_test, axis=0)
            
            features_nan = all_nan_train | all_nan_test
            self.valid_features = self.valid_features & ~features_nan
            self.invalid_indices = self.invalid_indices | features_nan

        if not np.any(self.valid_features):
            raise ValueError("All features are constant! Please check your data.")

        self.categorical_idx = [
            index
            for index, idx in enumerate(np.where(self.valid_features)[0])
            if idx in categorical_features
        ]

        return self.categorical_idx
    
    @override
    def transform(self, x:np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        assert self.valid_features is not None, "You must call fit first to get effective_features"
        self.invalid_features = x[:, self.invalid_indices]
        return x[:, self.valid_features], self.categorical_idx


class MADWinsorizer(BasePreprocess):
    """Per-column winsorization at ±N MAD from median (matches training).

    The training data generator (`training/data_generator.py`) does a final
    safety step: for each feature column, clip values outside ±6 MAD from the
    column's median, plus a hard ±1e4 clip. Inference does NOT do this — test
    rows can have feature values 8–15 std beyond training median, which the
    model has never seen.

    This step closes that training/inference distribution mismatch by
    explicitly clipping each feature column at fit-time computed ±N MAD
    bounds from the train split's per-column median. Categorical columns
    (passed in via `categorical_features`) are passed through unmodified.

    Parameters
    ----------
    n_mad : float
        Number of MADs from the median to use as the clip bound (default 6.0,
        matching the training-side winsorization constant).
    skip_categorical : bool
        If True, leave categorical columns unmodified (default True). MAD on
        a low-cardinality discrete column is degenerate and clipping can
        eliminate valid categorical values.
    """

    def __init__(self, *, n_mad: float = 6.0, skip_categorical: bool = True,
                  soft_log_clip: bool = False):
        self.n_mad = float(n_mad)
        self.skip_categorical = bool(skip_categorical)
        # Replace hard clip at boundary with soft logarithmic clip.
        # For values beyond [lo, hi], replace with hi + log1p(|x - hi|) /
        # bounds-equivalent on the low side. Preserves ordering of extremes
        # (rank-sensitive heads benefit) while still suppressing magnitude.
        self.soft_log_clip = bool(soft_log_clip)
        self.median_: np.ndarray | None = None
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None
        self.cat_mask_: np.ndarray | None = None

    @override
    def fit(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> list[int]:
        x = np.asarray(x, dtype=np.float64)
        n_features = x.shape[1]
        # MAD * 1.4826 ≈ standard-deviation equivalent for normal data.
        median = np.nanmedian(x, axis=0)
        mad = np.nanmedian(np.abs(x - median), axis=0)
        scale = mad * 1.4826
        # Avoid 0-scale columns (constants, near-constants): no clipping there.
        scale = np.where(scale > 1e-8, scale, np.inf)
        self.median_ = median
        self.lo_ = median - self.n_mad * scale
        self.hi_ = median + self.n_mad * scale
        cat_mask = np.zeros(n_features, dtype=bool)
        if self.skip_categorical and categorical_features:
            valid_idx = [i for i in categorical_features if 0 <= i < n_features]
            cat_mask[valid_idx] = True
        self.cat_mask_ = cat_mask
        return categorical_features

    @override
    def transform(self, x: np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        if self.median_ is None:
            raise RuntimeError("MADWinsorizer.fit must be called before transform.")
        x = np.asarray(x, dtype=np.float64)
        if x.shape[1] != len(self.median_):
            # Shape changed (e.g., upstream filter dropped columns). Skip
            # winsorization rather than crash; this is a safety fallback.
            return x.astype(np.float32), kwargs.get('categorical_features', [])
        if self.soft_log_clip:
            # Soft logarithmic clip: values beyond bounds are mapped to
            # bound + sign(excess) * log1p(|excess|). Preserves ordering
            # of extremes; smooth transition at the boundary.
            out = x.copy()
            hi_excess = np.maximum(out - self.hi_, 0.0)
            lo_excess = np.maximum(self.lo_ - out, 0.0)
            # Suppress only where actually exceeding
            out = np.where(hi_excess > 0, self.hi_ + np.log1p(hi_excess), out)
            out = np.where(lo_excess > 0, self.lo_ - np.log1p(lo_excess), out)
        else:
            out = np.clip(x, self.lo_, self.hi_)
        if self.skip_categorical and self.cat_mask_ is not None and self.cat_mask_.any():
            # Restore categorical columns to original (un-clipped) values
            out[:, self.cat_mask_] = x[:, self.cat_mask_]
        # Preserve NaN positions (clip doesn't change NaN, but be explicit)
        nan_mask = np.isnan(x)
        if nan_mask.any():
            out = np.where(nan_mask, np.nan, out)
        # Use the categorical_features passed at fit time
        cat_idx = list(np.where(self.cat_mask_)[0]) if self.cat_mask_ is not None else []
        return out.astype(np.float32), cat_idx


class FeatureShuffler(BasePreprocess):
    """
    Feature column reordering preprocessor
    """

    def __init__(
        self,
        mode: Literal['rotate', 'shuffle', 'latin'] | None = "shuffle",
        offset: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.offset = offset
        self.random_seed = None
        self.feature_indices = None
        self.categorical_indices = None

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        n_features = x.shape[1]
        self.random_seed = seed

        indices = np.arange(n_features)

        if self.mode == "rotate":
            self.feature_indices = np.roll(indices, self.offset)
        elif self.mode == "shuffle":
            _, rng = infer_random_state(self.random_seed)
            self.feature_indices = rng.permutation(indices)
        elif self.mode == "latin":
            # Latin-square-like permutation: random base shuffle (deterministic
            # per dataset via seed) + rotation by offset. Across the
            # n_estimators ensemble (each with a different offset 0..K-1),
            # every feature visits each position roughly equally — best
            # feature-position coverage for fixed n_estimators. Ports
            # TabICL's _latin_squares() pattern with simpler rotational
            # construction.
            _, rng = infer_random_state(self.random_seed)
            base = rng.permutation(indices)
            self.feature_indices = np.roll(base, self.offset)
        elif self.mode is None:
            self.feature_indices = np.arange(n_features)
        else:
            raise ValueError(f"Unsupported reordering mode: {self.mode}")

        is_categorical = np.isin(np.arange(n_features), categorical_features)
        self.categorical_indices = np.where(is_categorical[self.feature_indices])[0].tolist()
        
        return self.categorical_indices

    @override
    def transform(self, x:np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        if self.feature_indices is None:
            raise RuntimeError("Please call the fit method first to initialize")
        if len(self.feature_indices) != x.shape[1]:
            raise ValueError("The number of features in the input data does not match the training data")
            
        return x[:, self.feature_indices], self.categorical_indices or []

class CategoricalFeatureEncoder(BasePreprocess):
    """
    Categorical feature encoder
    """

    def __init__(
        self,
        encoding_strategy: Literal['ordinal', 'ordinal_strict_feature_shuffled', 'ordinal_shuffled', 'onehot', 'numeric', 'none']|None = "ordinal",
    ):
        super().__init__()
        self.encoding_strategy = encoding_strategy
        self.random_seed = None
        self.transformer = None
        self.category_mappings = None
        self.categorical_features = None
        self.feature_indices = None

    def _fit_impl(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> list[int]:
        ct, categorical_features = self._create_transformer(X, categorical_features)
        self.category_mappings = None
        if ct is None:
            self.transformer = None
            self.categorical_features = categorical_features
            return categorical_features

        _, rng = infer_random_state(self.random_seed)

        if self.encoding_strategy.startswith("ordinal"):
            ct.fit(X)
            categorical_features = list(range(len(categorical_features)))

            if self.encoding_strategy.endswith("_shuffled"):
                self.category_mappings = {}
                for col_ix in categorical_features:
                    col_cats = len(
                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
                    )
                    self.category_mappings[col_ix] = rng.permutation(col_cats)

        elif self.encoding_strategy == "onehot":
            ct.fit(X)
            Xt = ct.transform(X)
            if Xt.size >= 1_000_000:
                ct = None
            else:
                categorical_features = list(range(Xt.shape[1]))[
                    ct.output_indices_["one_hot_encoder"]
                ]
        else:
            raise ValueError(
                f"Unknown categorical transform {self.encoding_strategy}",
            )

        self.transformer = ct
        self.categorical_features = categorical_features
        return categorical_features

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        self.random_seed = seed
        return self._fit_impl(x, categorical_features)

    @override
    def transform(self, x:np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        if self.transformer is None:
            return x, self.categorical_features or []

        Xt = self.transformer.transform(x)
        categorical_features = self.categorical_features or []

        if self.encoding_strategy.startswith("ordinal") and self.category_mappings is not None:
            for col_ix, perm in self.category_mappings.items():
                col_data = Xt[:, col_ix]
                valid_mask = ~np.isnan(col_data)
                col_data[valid_mask] = perm[col_data[valid_mask].astype(int)].astype(col_data.dtype)

        return Xt, categorical_features

    @override
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> tuple[np.ndarray, list[int]]:
        self.fit(x, categorical_features, seed, **kwargs)
        return self.transform(x, **kwargs)

    @staticmethod
    def get_least_common_category_count(column: np.ndarray) -> int:
        """Retrieve the smallest count value among categorical features"""
        if len(column) == 0:
            return 0
        return int(np.unique(column, return_counts=True)[1].min())

    def _create_transformer(self, data: np.ndarray, categorical_columns: list[int]) -> tuple[ColumnTransformer | None, list[int]]:
        """Create an appropriate column transformer"""
        if self.encoding_strategy.startswith("ordinal"):
            suffix = self.encoding_strategy[len("ordinal"):]
            
            if "feature_shuffled" in suffix:
                categorical_columns = [
                    idx for idx in categorical_columns 
                    if self._is_valid_common_category(data[:, idx], suffix)
                ]
            remainder_columns = [idx for idx in range(data.shape[1]) if idx not in categorical_columns]
            self.feature_indices = categorical_columns + remainder_columns
                
            return ColumnTransformer(
                [("ordinal_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), categorical_columns)],
                remainder="passthrough"
            ), categorical_columns
            
        elif self.encoding_strategy == "onehot":
            return ColumnTransformer(
                [("one_hot_encoder", OneHotEncoder(drop="if_binary", sparse_output=False, handle_unknown="ignore"), categorical_columns)],
                remainder="passthrough"
            ), categorical_columns
            
        elif self.encoding_strategy in ("numeric", "none"):
            return None, categorical_columns
            
        raise ValueError(f"Unsupported encoding strategy: {self.encoding_strategy}")

    def _is_valid_common_category(self, column: np.ndarray, suffix: str) -> bool:
        """Check whether the input data meets the common category conditions"""
        min_count = self.get_least_common_category_count(column)
        unique_count = len(np.unique(column))
        
        if "strict_feature_shuffled" in suffix:
            return min_count >= 10 and unique_count < (len(column) // 10)
        return min_count >= 10

class QTx(QuantileTransformer):
    """
    Works like QuantileTransformer, but quietly fixes n_quantiles > n_samples.
    """

    def __init__(self, *, n_quantiles: int = 1000, **kwargs: Any) -> None:
        # tuck away the original request
        self._preferred = n_quantiles
        # pass a placeholder to parent (will be overwritten later anyway)
        super().__init__(n_quantiles=n_quantiles, **kwargs)

    def fit(self, X, y=None):
        # sample count
        m = getattr(X, "shape", [0])[0]

        # pick the actual quantiles we’ll use (safe value)
        q = [self._preferred, m, self.subsample]
        q = max(1, min(*q))

        # overwrite parent attr just-in-time
        object.__setattr__(self, "n_quantiles", q)

        # random_state adjustments
        rs = getattr(self, "random_state", None)
        if isinstance(rs, np.random.Generator):
            rs = np.random.RandomState(int(rs.integers(0, 2**32)))
        elif hasattr(rs, "bit_generator"):
            raise ValueError(
                f"Unsupported random_state type: {type(rs)}"
            )
        self.random_state = rs

        # delegate to parent
        return super().fit(X, y)

class KDIX(KDITransformer):
    """
    Variant of KDITransformer that won't crash on NaNs.
    """

    def _more_tags(self):
        # obscure way of saying "NaNs are okay"
        d = {}
        d.update(allow_nan=True)
        return d

    def fit(self, X, y=None):
        # accept both numpy and torch
        if hasattr(X, "detach"):   # torch.Tensor case
            base = X.cpu().numpy()
        else:
            base = np.asarray(X)

        # replace NaNs with col means for training
        means = np.nanmean(base, axis=0)
        cleaned = np.where(np.isnan(base), means, base)

        return super().fit(cleaned, y)  # type: ignore

    def transform(self, X):
        # lazy conversion
        if isinstance(X, torch.Tensor):
            mat = X.cpu().numpy()
        else:
            mat = np.array(X, copy=False)

        # track NaNs
        nan_pos = np.isnan(mat)

        # impute with column means (zero fallback)
        col_means = np.nanmean(mat, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        filled = np.where(np.isnan(mat), col_means, mat)

        # apply KDI
        res = super().transform(filled)

        # put NaNs back in
        np.putmask(res, nan_pos, np.nan)
        return res  # type: ignore


class RebalanceFeatureDistribution(BasePreprocess):
    def __init__(
            self,
            *,
            worker_tags: list[str] | None = None,
            discrete_flag: bool = False,
            original_flag: bool = False,
            svd_tag: Literal['svd'] | None = None,
            joined_svd_feature: bool = True,
            joined_log_normal: bool = True,
    ):
        super().__init__()
        self.worker_tags = worker_tags if worker_tags is not None else ["quantile"]
        self.discrete_flag = discrete_flag
        self.original_flag = original_flag
        self.random_state = None
        self.svd_tag = svd_tag
        self.worker: Pipeline | ColumnTransformer | None = None
        self.joined_svd_feature = joined_svd_feature
        self.joined_log_normal = joined_log_normal
        self.feature_indices = None
        self.n_quantile_features = 0

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        self.random_state = seed
        n_samples, n_features = x.shape
        worker, self.dis_ix = self._set(n_samples,n_features,categorical_features)
        worker.fit(x)
        self.worker = worker
        return self.dis_ix

    @override
    def transform(self, x:np.ndarray, **kwargs) -> np.ndarray:
        assert self.worker is not None
        return self.worker.transform(x), self.dis_ix  # type: ignore

    @override
    def fit_transform(self, x:np.ndarray, categorical_features:list[int], seed:int, *, y:np.ndarray, **kwargs)->tuple[np.ndarray, list[int]]:
        """Fit the preprocessing model to the data and transform the data"""
        assert y is not None, "The input y cannot be None"
        x_train_ = x[:len(y)]
        x_test_ = x[len(y):]
        if x_train_.shape[1] != x_test_.shape[1]:
            x_test_ = x_test_[:, :x_train_.shape[1]]
        categorical_idx_ = self.fit(x_train_, categorical_features, seed)
        x_train_, categorical_idx_ = self.transform(x_train_)
        x_test_, categorical_idx_ = self.transform(x_test_)
        x_ = np.concatenate([x_train_, x_test_], axis=0)

        return (x_, categorical_idx_)

    def _set(self,n_samples: int,
        n_features: int,
        categorical_features: list[int],
        ):
        static_seed, rng = infer_random_state(self.random_state)
        all_ix = list(range(n_features))
        workers = []
        cont_ix = [i for i in all_ix if i not in categorical_features]
        if self.original_flag:
            trans_ixs = categorical_features + cont_ix if self.discrete_flag else cont_ix
            workers.append(("original", "passthrough", all_ix))
            dis_ix = categorical_features
        elif self.discrete_flag:
            # trans_ixs = all_ix
            # dis_ix = categorical_features
            trans_ixs = categorical_features + cont_ix
            self.feature_indices = categorical_features + cont_ix
            dis_ix = []
        else:
            workers.append(("discrete", "passthrough", categorical_features))
            trans_ixs, dis_ix = cont_ix, list(range(len(categorical_features)))
        for worker_tag in self.worker_tags:
            # print(f"== worker_tag: \033[31m{worker_tag}\033[0m")
            if worker_tag == "logNormal":
                sworker = Pipeline(steps=[
                                        ("save_standard", Pipeline(steps=[
                                            ("i2n_pre",
                                             FunctionTransformer(
                                                 func=_inf_to_nan,
                                                 inverse_func=_identity, check_inverse=False)),
                                            ("fill_missing_pre",
                                             SimpleImputer(missing_values=np.nan, strategy="mean",
                                                           keep_empty_features=True)),
                                            ("feature_shift",
                                             FunctionTransformer(func=_shift_to_nonnegative)),
                                            ("add_epsilon", FunctionTransformer(func=_add_epsilon)),
                                            ("logNormal", FunctionTransformer(np.log, validate=False)),
                                            ("i2n_post",
                                             FunctionTransformer(
                                                 func=_inf_to_nan,
                                                 inverse_func=_identity, check_inverse=False)),
                                            ("fill_missing_post",
                                             SimpleImputer(missing_values=np.nan, strategy="mean",
                                                           keep_empty_features=True))])),
                                        ])
                # trans_ixs = cont_ix
            elif worker_tag == "quantile_uniform_10":
                sworker = CappedQuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 10, 2),
                    random_state=static_seed,
                )
            elif worker_tag == "quantile_uniform_5":
                sworker = CappedQuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                )
            elif worker_tag == "quantile_uniform_all_data":
                sworker = CappedQuantileTransformer(
                    output_distribution="uniform",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    subsample=n_samples,
                )
            elif worker_tag == 'power':
                self.feature_indices = categorical_features+cont_ix
                self.dis_ix = dis_ix
                nan_to_mean_transformer = SimpleImputer(
                                                    missing_values=np.nan,
                                                    strategy="mean",
                                                    keep_empty_features=True,
                                                )
            
                sworker = SelectiveInversePipeline(
                                steps=[
                                    ("power_transformer", RobustPowerTransformer(standardize=False)),
                                    ("inf_to_nan_1", FunctionTransformer(
                                                        func=_inf_to_nan,
                                                        inverse_func=_identity,
                                                        check_inverse=False,
                                                    )),
                                    ("nan_to_mean_1", nan_to_mean_transformer),
                                    ("scaler", StandardScaler()),
                                    ("inf_to_nan_2", FunctionTransformer(
                                                        func=_inf_to_nan,
                                                        inverse_func=_identity,
                                                        check_inverse=False,
                                                    )),
                                    ("nan_to_mean_2", nan_to_mean_transformer),
                                ],
                        skip_inverse=['nan_to_mean_1', 'nan_to_mean_2']
                )

            elif worker_tag=="quantile_norm_10":
                sworker = QTx(
                    output_distribution="normal",
                    n_quantiles=max(n_samples // 10, 2),
                    random_state=static_seed,
                )
            elif worker_tag=="quantile_norm_5":
                sworker = QTx(
                    output_distribution="normal",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                )
            elif worker_tag == "quantile_norm_all_data":
                sworker = CappedQuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=max(n_samples // 5, 2),
                    random_state=static_seed,
                    subsample=n_samples,
                )
            elif worker_tag=="norm_and_kdi":
                sworker = FeatureUnion(
                    [
                        (
                            "norm",
                            QTx(
                                output_distribution="normal",
                                n_quantiles=max(n_samples // 10, 2),
                                random_state=static_seed,
                            ),
                        ),
                        (
                            "kdi",
                            KDIX(alpha=1.0, output_distribution="uniform"),
                        ),
                    ],
                )

            elif worker_tag=="robust":
                sworker = RobustScaler(unit_variance=True)
            elif worker_tag=="adaptive":
                # Per-column adaptive routing (TabPFN-style). Each numeric
                # feature is categorized at fit-time by its distribution and
                # routed through the appropriate transform: identity for
                # ordinal/normal columns, box-cox/yeo-johnson for skewed,
                # quantile-normal for the rest. Better than a single global
                # transform on heterogeneous-feature datasets.
                sworker = AdaptiveColumnTransformer(
                    random_state=static_seed,
                    n_quantiles=max(n_samples // 10, 2),
                )
            elif worker_tag=="squash":
                # Robust scaling for outlier / heavy-tail features:
                # median-centred, IQR-scaled to unit variance.
                sworker = RobustScaler(unit_variance=True)
            elif worker_tag=="kdi_uni":
                sworker = KDIX(alpha=1.0, output_distribution="uniform")
            elif worker_tag is None:
                sworker = FunctionTransformer(_identity)
            elif worker_tag.startswith("kdi_uni_alpha_"):
                alpha = float(worker_tag.split("_")[-1])
                sworker = KDIX(alpha=alpha, output_distribution="uniform")
            elif worker_tag.startswith("kdi_norm_alpha_"):
                alpha = float(worker_tag.split("_")[-1])
                sworker = KDIX(alpha=alpha, output_distribution="normal")
            elif worker_tag=="kdi_norm":
                sworker = KDIX(alpha=1.0, output_distribution="normal")
            else:
                sworker = FunctionTransformer(_identity)
            if worker_tag in ["quantile_uniform_10", "quantile_uniform_5", "quantile_uniform_all_data"]:
                self.n_quantile_features = len(trans_ixs)
            workers.append((f"feat_transform_{worker_tag}", sworker, trans_ixs))

        CT_worker = ColumnTransformer(workers,remainder="drop",sparse_threshold=0.0)
        if self.svd_tag == "svd" and n_features >= 2:
            svd_worker = FeatureUnion([
                    ("default", FunctionTransformer(func=_identity)),
                    ("svd",Pipeline(steps=[
                                    ("save_standard",Pipeline(steps=[
                                    ("i2n_pre", FunctionTransformer(func=_inf_to_nan,inverse_func=_identity, check_inverse=False)),
                                    ("fill_missing_pre", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True)),
                                    ("standard", StandardScaler(with_mean=False)) ,
                                    ("i2n_post", FunctionTransformer(func=_inf_to_nan,inverse_func=_identity, check_inverse=False)),
                                    ("fill_missing_post", SimpleImputer(missing_values=np.nan, strategy="mean", keep_empty_features=True))])),
                                    ("svd",_TorchTruncatedSVD(algorithm="arpack",n_components=max(1,min(n_samples // 10 + 1,n_features // 2)),random_state=static_seed))]))
                    ])
            self.svd_n_comp = max(1,min(n_samples // 10 + 1,n_features // 2))
            worker = Pipeline([("worker", CT_worker), ("svd_worker", svd_worker)])
        else:   
            self.svd_n_comp = 0
            worker = CT_worker

        self.worker = worker
        return worker, dis_ix


class SubSampleData():
    def __init__(
            self,
            subsample_type: Literal["feature", "sample"] = "sample",
            use_type: Literal["mixed", "only_sample"] = "mixed",
    ):
        super().__init__()
        self.subsample_type = subsample_type
        self.use_type = use_type

    def fit(self,
            x: torch.Tensor=None,
            y: torch.Tensor = None,
            feature_attention_score: torch.Tensor = None,
            sample_attention_score: torch.Tensor = None,
            subsample_ratio: float | int = 200,
            subsample_idx:list[int] | np.ndarray[int] = None,
            ):
        if isinstance(subsample_ratio, float):
            if self.subsample_type == "sample":
                self.subsample_num = int(subsample_ratio * x.shape[0])
            else:
                self.subsample_num = int(subsample_ratio * x.shape[1])
        else:
            self.subsample_num = subsample_ratio
        if self.subsample_type == "sample":
            if self.use_type == "mixed":
                y_feature_attention_score = feature_attention_score[:, -1, :].squeeze().permute(1, 0).unsqueeze(
                    -1) # shape [features,test_sample_lens,1] broadcast to [features,test_sample_lens,train_sample_lens]
                # TODO: this elementwise product may cause OOM on large inputs
                try:
                    self.attention_score = torch.mean(sample_attention_score.to("cuda") * y_feature_attention_score.to("cuda"),
                                                      dim=0).cpu()  # shape [test_sample_lens,train_sample_lens]
                except OutOfMemoryError as e:
                    print("calculate attention score OOM, use cpu")
                    self.attention_score = torch.mean(
                        sample_attention_score.cpu() * y_feature_attention_score.cpu(),
                        dim=0)
                del sample_attention_score,y_feature_attention_score

            else:
                self.attention_score = sample_attention_score[-1, :, :]
            self.X_train = x
            self.y_train = y
        else:
            y_feature_attention_score = torch.mean(feature_attention_score[:, -1, :].squeeze(),dim=0)  # shape [test_sample_lens,features]
            if subsample_idx is None:
                self.subsample_idx = torch.argsort(y_feature_attention_score)[-min(self.subsample_num, x.shape[0]):]
            else:
                self.subsample_idx = subsample_idx
            self.X_train = x

    def transform(self, x: torch.Tensor=None) -> np.ndarray |torch.Tensor | NoriInferenceDataset:
        if self.subsample_type == "feature":
            return torch.cat([self.X_train, x], dim=0)[:, self.subsample_idx].numpy()
        else:
            return self.attention_score



# Large constant for hash normalization
_HASH_MODULUS = 10**12

def float_hash_arr(input_array: np.ndarray) -> float:
    """
    Generate a normalized floating-point hash value from a numpy array.
    
    This function computes a SHA256 hash of the array's byte representation,
    converts it to an integer, and normalizes it to a float between 0 and 1.
    
    Args:
        input_array: Input numpy array to be hashed
        
    Returns:
        Normalized hash value in the range [0, 1)
    """
    # Convert array to bytes and compute SHA256 hash
    array_bytes = input_array.tobytes()
    hash_hex = hashlib.sha256(array_bytes).hexdigest()
    
    # Convert hex digest to integer
    hash_int = int(hash_hex, 16)
    
    # Normalize to [0, 1) range using modulus operation
    normalized_hash = (hash_int % _HASH_MODULUS) / _HASH_MODULUS
    
    return normalized_hash


class FingerprintFeatureEncoder(BasePreprocess):
    """
    Appends a fingerprint column derived from row-wise hashing of input data.
    
    For test data: Uses first computed hash even if collisions occur.
    For training data: Resolves hash collisions through iterative rehashing.
    """
    
    def __init__(self, rng_seed: int | np.random.Generator | None = None):
        super().__init__()
        # self.rng_seed = rng_seed
        self.salt_value = None
        self.categorical_features = None
    
    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        """Initialize random salt and return categorical feature indices."""
        _, rng = infer_random_state(seed)
        self.salt_value = int(rng.integers(0, 65536))  # 2^16 range
        self.categorical_features = categorical_features
        return categorical_features.copy()
    
    @override
    def transform(self, x:np.ndarray, is_test:bool=False, **kwargs) -> tuple[np.ndarray, list[int]]:
        """
        Transform input by appending fingerprint column.
        
        Args:
            X_data: Input array of shape (n_samples, n_features)
            is_test: Whether processing test data (affects collision handling)
            
        Returns:
            Transformed array with fingerprint column and updated categorical indices
        """
        # print(f"add finger")
        if self.salt_value is None:
            raise RuntimeError("Must call fit() before transform()")
        
        n_samples = x.shape[0]
        fingerprint_col = np.zeros(n_samples, dtype=x.dtype)
        
        # Apply salt to input data
        salted_data = x + self.salt_value
        
        if is_test:
            # Test mode: use first hash regardless of collisions
            for idx in range(n_samples):
                row_hash = float_hash_arr(salted_data[idx] + self.salt_value)
                fingerprint_col[idx] = row_hash
        else:
            # Training mode: resolve hash collisions
            existing_hashes = set()
            for idx in range(n_samples):
                current_row = salted_data[idx]
                hash_val = float_hash_arr(current_row)
                increment = 0
                
                # Handle collisions by incrementing and rehashing
                while hash_val in existing_hashes:
                    increment += 1
                    hash_val = float_hash_arr(current_row + increment)
                
                fingerprint_col[idx] = hash_val
                existing_hashes.add(hash_val)
        
        # Append fingerprint column and update categorical indices
        transformed = np.column_stack([x, fingerprint_col.reshape(-1, 1)])
        # cat_indices_updated = list(range(x.shape[1]))  # Original features remain categorical
        
        return transformed, self.categorical_features

class HighDimFeatureSelector(BasePreprocess):
    """Conditional high-dimensional feature selector / projector.

    Self-gates: passthrough when ``n_features <= n_features_threshold`` AND
    ``binary_frac < binary_threshold``. When active, picks a top-k feature
    subset (or projection) using the chosen strategy.

    Strategies (regression-aware; cls falls back to passthrough):
      ``corr``        top_k features by ``|Pearson(X[:, i], y)|``
      ``mi``          top_k features by ``mutual_info_regression(X, y)``
      ``extratrees``  top_k features by ``ExtraTreesRegressor`` importance
      ``svd_binary``  TruncatedSVD on binary columns; replace them with
                      ``svd_components`` components, keep non-binary cols
      ``svd_all``     TruncatedSVD on all columns; output is ``svd_components``
      ``passthrough`` explicit no-op (useful for ensemble diversity)

    Inductive: fit selector / SVD on ``(x_train, y_train)``; transform on
    ``x_test`` via stored indices / components. ``y`` is passed via fit kwargs;
    if absent (e.g., classification path), the step falls back to passthrough.

    Parameters
    ----------
    strategy : str
        One of ``corr`` / ``mi`` / ``extratrees`` / ``svd_binary`` /
        ``svd_all`` / ``passthrough``.
    top_k : int
        Number of features to select. Default 256.
    n_features_threshold : int
        Activate gate when ``n_features > n_features_threshold``. Default 128.
    binary_threshold : float
        Activate gate when binary-column fraction ``>= binary_threshold``
        (default 0.5). For the ``svd_binary`` strategy, also gates the SVD
        transform itself.
    svd_components : int
        SVD output dimension for ``svd_binary`` / ``svd_all``. Default 64.
    svd_rows_per_component : int
        Minimum training rows required per retained SVD component. **Defaults to 1
        (previous behaviour)**; the shipped inference config sets 3. Opt-in rather than a
        global default because the evidence for it is low-rank spectral data, while a
        fixed low cap is known to *regress* genuinely high-rank wide tables (QSAR /
        isolet / Santander). With it set, the fitted rank is
        ``min(svd_components, p-1, n-1, n // svd_rows_per_component)``. Without it the
        rank was bounded by ``n-1``, which takes the *maximum* available rank exactly
        when rows are scarcest — on a 1901-feature x 190-row table that is 189, an
        orthogonal rotation rather than a reduction, retaining the whole noise tail.
        Set to 1 to restore the previous behaviour.
    extratrees_n_estimators : int
        Number of trees for ``extratrees`` strategy (default 100).
    subsample_rows : int
        Cap rows when fitting MI / ExtraTrees (default 5000) to bound runtime
        on large train splits.

    When the SVD itself throws, this step works around it rather than failing —
    and always warns, under :class:`~synthefy_nori.SvdFallbackWarning`. Callers
    who must not be handed a degraded prediction escalate that category to an
    exception (``with strict_pipeline(): ...``); see
    ``synthefy_nori.inference.degradation``.
    """

    def __init__(
        self,
        *,
        strategy: Literal['corr', 'mi', 'extratrees', 'svd_binary', 'svd_all', 'passthrough'] = 'passthrough',
        top_k: int = 256,
        n_features_threshold: int = 128,
        binary_threshold: float = 0.5,
        svd_components: int = 64,
        svd_rows_per_component: int = 1,
        extratrees_n_estimators: int = 100,
        subsample_rows: int = 5000,
    ):
        super().__init__()
        self.strategy = strategy
        self.top_k = int(top_k)
        self.n_features_threshold = int(n_features_threshold)
        self.binary_threshold = float(binary_threshold)
        self.svd_components = int(svd_components)
        self.svd_rows_per_component = max(1, int(svd_rows_per_component))
        self.extratrees_n_estimators = int(extratrees_n_estimators)
        self.subsample_rows = int(subsample_rows)
        self.passthrough_: bool = False
        self.selected_idx_: np.ndarray | None = None
        self.svd_model_ = None
        self.svd_binary_mask_: np.ndarray | None = None
        self.svd_keep_idx_: np.ndarray | None = None
        self.categorical_features_: list[int] = []

    @staticmethod
    def _detect_binary_cols(x: np.ndarray) -> np.ndarray:
        n_cols = x.shape[1]
        mask = np.zeros(n_cols, dtype=bool)
        for i in range(n_cols):
            col = x[:, i]
            col = col[np.isfinite(col)]
            if len(col) == 0:
                continue
            uniq = np.unique(col)
            if len(uniq) == 2:
                mask[i] = True
        return mask

    @staticmethod
    def _topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
        n = len(scores)
        k = max(1, min(top_k, n))
        idx = np.argsort(-scores, kind='stable')[:k]
        return np.sort(idx)

    @staticmethod
    def _compute_corr_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        n_features = x.shape[1]
        scores = np.zeros(n_features, dtype=np.float64)
        for i in range(n_features):
            xi = x[:, i]
            mask = np.isfinite(xi) & np.isfinite(y)
            if mask.sum() < 3:
                continue
            xs = xi[mask]
            ys = y[mask]
            if xs.std() < 1e-8 or ys.std() < 1e-8:
                continue
            try:
                c = np.corrcoef(xs, ys)[0, 1]
                scores[i] = abs(c) if np.isfinite(c) else 0.0
            except Exception:
                scores[i] = 0.0
        return scores

    def _maybe_subsample(self, x, y, rng):
        n = x.shape[0]
        if n <= self.subsample_rows:
            return x, y
        idx = rng.choice(n, size=self.subsample_rows, replace=False)
        return x[idx], y[idx]

    def _remap_categorical(self, categorical_features: list[int]) -> list[int]:
        idx_set = set(int(i) for i in self.selected_idx_)
        old_to_new = {old: new for new, old in enumerate(self.selected_idx_.tolist())}
        new_cat = sorted(old_to_new[int(c)] for c in categorical_features if int(c) in idx_set)
        self.categorical_features_ = new_cat
        return new_cat

    def _remap_categorical_svd_binary(self, categorical_features: list[int]) -> list[int]:
        non_bin_old_idx = self.svd_keep_idx_
        idx_set = set(int(i) for i in non_bin_old_idx)
        old_to_new = {old: new for new, old in enumerate(non_bin_old_idx.tolist())}
        new_cat = sorted(old_to_new[int(c)] for c in categorical_features if int(c) in idx_set)
        self.categorical_features_ = new_cat
        return new_cat

    def _activate_passthrough(self, categorical_features: list[int]) -> list[int]:
        self.passthrough_ = True
        self.categorical_features_ = list(categorical_features)
        return list(categorical_features)

    def _svd_degraded(self, stage: str, exc: Exception, fallback: str) -> None:
        """Report an SVD failure that silently degrades the configured pipeline.

        Both SVD fallbacks change what the model sees without changing the config:
        a **fit** failure turns the projection into a passthrough of the RAW
        (e.g. 1024) columns, and a **transform** failure feeds ``svd_all`` a single
        zero column — every feature gone. Either one still returns predictions, so
        the run completes with a plausible-looking bad R2 that reads as "this
        config is weak" rather than "the SVD broke".

        So it is never silent. Emitting :class:`SvdFallbackWarning` is also the
        whole opt-out mechanism: escalating that category to an exception (see
        ``synthefy_nori.inference.degradation``) makes the fallback fatal, with no
        per-step argument to plumb through the predictor and the public API.
        """
        warnings.warn(
            f"Nori: HighDimFeatureSelector(strategy={self.strategy!r}): SVD {stage} failed "
            f"({type(exc).__name__}: {str(exc)[:120]}) -> {fallback}. Predictions are "
            f"degraded, NOT the configured pipeline. Use "
            f"synthefy_nori.strict_pipeline() to make this an error instead.",
            SvdFallbackWarning,
            stacklevel=3,
        )

    @override
    def fit(self, x: np.ndarray, categorical_features: list[int], seed: int, *, y=None, **kwargs) -> list[int]:
        x = np.asarray(x, dtype=np.float64)
        n_samples, n_features = x.shape

        if self.strategy == 'passthrough' or n_features == 0:
            return self._activate_passthrough(categorical_features)

        binary_mask = self._detect_binary_cols(x)
        binary_frac = float(binary_mask.mean()) if n_features > 0 else 0.0
        gate_open = (n_features > self.n_features_threshold) or (binary_frac >= self.binary_threshold)
        if not gate_open:
            return self._activate_passthrough(categorical_features)

        rng = np.random.default_rng(int(seed) % (2**32))
        seed_int = int(seed) % (2**31)

        if self.strategy == 'svd_binary':
            if binary_frac < self.binary_threshold or int(binary_mask.sum()) < 2:
                return self._activate_passthrough(categorical_features)
            x_binary = x[:, binary_mask]
            x_binary = np.where(np.isnan(x_binary), 0.0, x_binary)
            n_components = max(1, min(
                self.svd_components,
                int(binary_mask.sum()) - 1,
                n_samples - 1,
            ))
            try:
                self.svd_model_ = _TorchTruncatedSVD(n_components=n_components, random_state=seed_int)
                self.svd_model_.fit(x_binary)
            except Exception as exc:
                self._svd_degraded(
                    "fit", exc,
                    f"passthrough of all {n_features} raw columns "
                    f"({int(binary_mask.sum())} binary columns left unprojected)")
                return self._activate_passthrough(categorical_features)
            self.svd_binary_mask_ = binary_mask
            self.svd_keep_idx_ = np.where(~binary_mask)[0]
            self.passthrough_ = False
            return self._remap_categorical_svd_binary(categorical_features)

        if self.strategy == 'svd_all':
            x_imp = np.where(np.isnan(x), 0.0, x)
            # Rank must be supported by the ROWS, not merely bounded by them.
            #
            # The old rule was min(svd_components, p-1, n-1), which takes the MAXIMUM
            # available rank exactly when rows are scarcest: on a 1901-feature x 190-row
            # table it returns 189, so the "reduction" is an orthogonal rotation into the
            # full row space that retains every noise direction and the in-context model
            # then overfits it. Dividing by svd_rows_per_component ties k to the samples
            # that actually support it, while svd_components still caps the tall end.
            #
            # Measured on RamanBench (nori-100m, production config, 107 datasets,
            # dataset-level, paired vs the old rule): +0.0269 macro R2, p=0.0010, and
            # negative-R2 datasets fall from 9 to 5. The gain is concentrated where the
            # old rule degenerated -- n<256 rows: +0.0316 (43/67, p=0.004) -- and is
            # neutral-to-positive where the cap already bound (n>=256: +0.0191). Removing
            # the cap instead of adding this term is NOT equivalent: min(p, n//3) alone
            # scores -0.0453 on n>=256 because k then runs to 2000 on tall tables.
            #
            # No effect on standard suites: the gate needs p > n_features_threshold, which
            # fires on 1 of 145 tracked datasets, and there n//3 > svd_components anyway --
            # so k is unchanged and predictions are bit-identical.
            # n_samples - 1 is kept as a VALIDITY bound (mean-centering costs one degree
            # of freedom) and is separate from the rows-support term, so
            # svd_rows_per_component=1 reproduces the previous rank exactly.
            n_components = max(1, min(self.svd_components,
                                      n_features - 1,
                                      n_samples - 1,
                                      n_samples // self.svd_rows_per_component))
            try:
                self.svd_model_ = _TorchTruncatedSVD(n_components=n_components, random_state=seed_int)
                self.svd_model_.fit(x_imp)
            except Exception as exc:
                self._svd_degraded(
                    "fit", exc,
                    f"passthrough of all {n_features} raw columns "
                    f"(no reduction to {n_components} components)")
                return self._activate_passthrough(categorical_features)
            self.svd_binary_mask_ = np.ones(n_features, dtype=bool)
            self.svd_keep_idx_ = np.array([], dtype=int)
            self.categorical_features_ = []
            self.passthrough_ = False
            return []

        if y is None:
            return self._activate_passthrough(categorical_features)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        if y_arr.size != n_samples:
            return self._activate_passthrough(categorical_features)
        finite_y = np.isfinite(y_arr)
        if finite_y.sum() < max(10, self.top_k // 4):
            return self._activate_passthrough(categorical_features)

        if self.strategy == 'corr':
            scores = self._compute_corr_scores(x, y_arr)
            self.selected_idx_ = self._topk_indices(scores, self.top_k)
            self.passthrough_ = False
            return self._remap_categorical(categorical_features)

        if self.strategy == 'mi':
            try:
                from sklearn.feature_selection import mutual_info_regression
            except Exception:
                return self._activate_passthrough(categorical_features)
            x_sub, y_sub = self._maybe_subsample(x[finite_y], y_arr[finite_y], rng)
            x_imp = np.where(np.isnan(x_sub), 0.0, x_sub)
            try:
                scores = mutual_info_regression(x_imp, y_sub, random_state=seed_int)
            except Exception:
                return self._activate_passthrough(categorical_features)
            self.selected_idx_ = self._topk_indices(scores, self.top_k)
            self.passthrough_ = False
            return self._remap_categorical(categorical_features)

        if self.strategy == 'extratrees':
            try:
                from sklearn.ensemble import ExtraTreesRegressor
            except Exception:
                return self._activate_passthrough(categorical_features)
            x_sub, y_sub = self._maybe_subsample(x[finite_y], y_arr[finite_y], rng)
            x_imp = np.where(np.isnan(x_sub), 0.0, x_sub)
            try:
                et = ExtraTreesRegressor(
                    n_estimators=self.extratrees_n_estimators,
                    random_state=seed_int,
                    n_jobs=-1,
                )
                et.fit(x_imp, y_sub)
                scores = et.feature_importances_
            except Exception:
                return self._activate_passthrough(categorical_features)
            self.selected_idx_ = self._topk_indices(scores, self.top_k)
            self.passthrough_ = False
            return self._remap_categorical(categorical_features)

        return self._activate_passthrough(categorical_features)

    @override
    def transform(self, x: np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        if self.passthrough_:
            return x, self.categorical_features_
        if self.strategy in ('corr', 'mi', 'extratrees'):
            return x[:, self.selected_idx_], self.categorical_features_
        if self.strategy in ('svd_binary', 'svd_all'):
            x_arr = np.asarray(x, dtype=np.float64)
            x_in = x_arr[:, self.svd_binary_mask_]
            x_in = np.where(np.isnan(x_in), 0.0, x_in)
            try:
                x_svd = self.svd_model_.transform(x_in)
            except Exception as exc:
                # SVD test-time failure: fall back to keep cols (or zeros if svd_all).
                # Never silently — a zero column makes the model predict from nothing.
                if self.strategy == 'svd_all':
                    self._svd_degraded(
                        "transform", exc,
                        f"a single all-zero column for {x_arr.shape[0]} rows "
                        f"(all {x_in.shape[1]} features dropped)")
                    return np.zeros((x_arr.shape[0], 1), dtype=np.float32), []
                self._svd_degraded(
                    "transform", exc,
                    f"dropping the {x_in.shape[1]} SVD-projected columns, keeping the "
                    f"{len(self.svd_keep_idx_)} non-binary ones")
                return x_arr[:, self.svd_keep_idx_].astype(np.float32), self.categorical_features_
            if self.strategy == 'svd_all':
                return x_svd.astype(np.float32), []
            x_keep = x_arr[:, self.svd_keep_idx_]
            out = np.concatenate([x_keep, x_svd], axis=1).astype(np.float32)
            return out, self.categorical_features_
        return x, self.categorical_features_


class MaxFeatureSubsampler(BasePreprocess):
    """Randomly subsample features if the input has more than `max_features` columns.

    Matches TabPFN-2.6's `max_features_per_estimator` behavior: each ensemble
    member sees at most `max_features` randomly-selected columns from the
    original feature set. Different per-estimator seeds yield different
    subsets — ensembling over them recovers coverage.

    Critical for high-dim datasets like QSAR-TID-11 (f=1024): instead of
    feeding all 1024 features to every estimator, each sees ~500 random
    features, which tends to reduce noise and improve the committee's R².

    No-op if input has <= max_features columns.
    """

    def __init__(self, *, max_features: int = 500):
        super().__init__()
        assert max_features > 0, "max_features must be > 0"
        self.max_features = int(max_features)
        self.selected_idx: np.ndarray | None = None
        self.categorical_features = None

    @override
    def fit(self, x: np.ndarray, categorical_features: list[int], seed: int, **kwargs) -> list[int]:
        assert x.ndim == 2, "Input must be 2D"
        _, rng = infer_random_state(seed)
        n_features = x.shape[1]
        if n_features <= self.max_features:
            # No subsampling needed
            self.selected_idx = np.arange(n_features)
        else:
            self.selected_idx = np.sort(
                rng.choice(n_features, size=self.max_features, replace=False)
            )
        # Map categorical indices to new index space
        if categorical_features:
            idx_set = set(int(i) for i in self.selected_idx)
            old_to_new = {old: new for new, old in enumerate(self.selected_idx.tolist())}
            new_cat = sorted(old_to_new[int(c)] for c in categorical_features if int(c) in idx_set)
        else:
            new_cat = []
        self.categorical_features = new_cat
        return new_cat

    @override
    def transform(self, x: np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        if self.selected_idx is None:
            raise RuntimeError("Must call fit() before transform()")
        return x[:, self.selected_idx], self.categorical_features


class PolynomialInteractionGenerator(BasePreprocess):
    """
    Generates polynomial interaction features through randomized pairwise combinations
    with standardized preprocessing and memory-efficient implementation.
    """
    
    def __init__(
        self,
        *,
        max_interaction_features: int | None = None,
        random_generator: int | np.random.Generator | None = None,
        disable_above_n_features: int | None = None,
    ):
        super().__init__()
        self.max_interactions = max_interaction_features
        # self.rng_config = random_generator
        # print(f"max_interactions: {self.max_interactions}")
        if self.max_interactions:
            assert max_interaction_features > 0, "max_interaction_features must be greater than 0"
        else:
            self.max_interactions = 100
        # When set, the step becomes a no-op for inputs with > N features (the
        # high-dim route disables poly so we don't add 10 noisy product
        # features on top of 1024 fingerprint columns).
        self.disable_above_n_features = (
            int(disable_above_n_features) if disable_above_n_features is not None else None
        )
        self.disabled_: bool = False

        self.primary_factor_indices: np.ndarray | None = None
        self.secondary_factor_indices: np.ndarray | None = None
        self.feature_normalizer = StandardScaler(with_mean=False)
        self.categorical_features = None

    @override
    def fit(self, x:np.ndarray, categorical_features:list[int], seed:int, **kwargs) -> list[int]:
        """Configure polynomial feature generation parameters from training data."""
        assert x.ndim == 2, "Input matrix must be 2-dimensional"

        _, random_engine = infer_random_state(seed)

        # Handle empty dataset scenarios
        if x.size == 0:
            self.disabled_ = False
            return list(categorical_features)
        if (self.disable_above_n_features is not None
                and x.shape[1] > self.disable_above_n_features):
            self.disabled_ = True
            self.categorical_features = list(categorical_features)
            return list(categorical_features)
        self.disabled_ = False

        feature_count = x.shape[1]
        
        # Calculate maximum possible interaction combinations
        max_possible_combinations = (feature_count * (feature_count + 1)) // 2
        
        # print(f"max_possible_combinations: {max_possible_combinations}")
        # Determine actual interaction count with constraint
        actual_interaction_count = (
            min(self.max_interactions, max_possible_combinations) 
            if self.max_interactions is not None 
            else max_possible_combinations
        )
        
        # Standardize features before interaction generation
        normalized_data = self.feature_normalizer.fit_transform(x)
        
        # Generate randomized factor pairs efficiently
        self._generate_interaction_pairs(feature_count, actual_interaction_count, random_engine)
        self.categorical_features = categorical_features
        return categorical_features
    
    def _generate_interaction_pairs(
        self, 
        total_features: int, 
        required_pairs: int, 
        rng: np.random.Generator
    ) -> None:
        """Efficiently generate unique factor pairs for polynomial feature creation."""
        self.primary_factor_indices = rng.choice(
            np.arange(total_features),
            size=required_pairs,
            replace=True,
        )

        self.secondary_factor_indices = np.full_like(self.primary_factor_indices, -1)

        for i in range(required_pairs):
            while self.secondary_factor_indices[i] == -1:
                a = self.primary_factor_indices[i]
                used_b = self.secondary_factor_indices[self.primary_factor_indices == a]
                allowed_b = [b for b in range(a, total_features) if b not in used_b]

                if len(allowed_b) == 0:
                    self.primary_factor_indices[i] = rng.choice(np.arange(total_features))
                    continue
                else:
                    self.secondary_factor_indices[i] = rng.choice(allowed_b)

    @override
    def transform(self, x:np.ndarray, **kwargs) -> tuple[np.ndarray, list[int]]:
        """Apply polynomial feature transformation to input data."""
        assert x.ndim == 2, "Input matrix must be 2-dimensional"

        if x.size == 0:
            return x, []
        if self.disabled_:
            return x, self.categorical_features

        # Standardize input features
        standardized_features = self.feature_normalizer.transform(x)
        
        # Generate polynomial interaction features
        interaction_features = (
            standardized_features[:, self.primary_factor_indices] * 
            standardized_features[:, self.secondary_factor_indices]
        )
        
        # Combine original and interaction features
        transformed_output = np.column_stack([standardized_features, interaction_features])
        
        return transformed_output, self.categorical_features