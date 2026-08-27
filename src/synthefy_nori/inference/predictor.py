from __future__ import annotations

from synthefy_nori.inference.inference_method import DistributedInference
from synthefy_nori.inference.preprocess import (
    FeatureShuffler,
    FilterValidFeatures,
    CategoricalFeatureEncoder,
    RebalanceFeatureDistribution,
    FingerprintFeatureEncoder,
    HighDimFeatureSelector,
    MaxFeatureSubsampler,
    MADWinsorizer,
    PolynomialInteractionGenerator)
from synthefy_nori.inference.degradation import (
    CacheQuantizedWarning,
    ContextSubsampledWarning,
    DegradedPipelineWarning,
)
from synthefy_nori.inference.memory_policy import (
    FIT_ROW_CHUNK_RETRY_LADDER,
    RUNGS,
    ContextTooLargeError,
    MemoryAttempt,
    MemoryPolicy,
    estimate_cache_gb,
    total_host_ram_gb,
)
from synthefy_nori.model.layer import RMSNorm
from synthefy_nori.utils.loading import load_model
import contextlib
import torch
from typing import List, Literal
import random
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd
import einops
import json
import os
import logging
import warnings


logger = logging.getLogger(__name__)

NA_PLACEHOLDER = "__MISSING__"
# The default eight-member ensemble naturally forms two shape groups of four.
# Cap custom configs at that measured sweet spot so batching cannot multiply
# activation/context-cache memory without bound.
PIPELINE_BATCH_MAX = 4
# Batched GEMMs are launch-bound and much faster on narrow/medium tables, but
# the gain disappears on very wide transformed tables while memory rises. The
# H200 crossover was between 146 and 352 processed features, so keep the first
# rollout on the conservative side and leave wide members at B=1.
PIPELINE_BATCH_MAX_FEATURES = 256
# Bound the extra host copies retained while pipelines wait for same-shape
# partners. The legacy path streams one transformed table at a time; falling
# back above this cap preserves that behavior for very large query sets.
PIPELINE_BATCH_HOST_BUDGET_BYTES = 512 * 1024**2


class NoriPredictor:
    """Nori model inferencer, supporting tasks such as classification, regression, and missing value prediction."""

    def __init__(
        self,
        device: torch.device,
        model_path: str = None,
        inference_config: list | str = None,
        mix_precision: bool = True,
        outlier_remove_std: float = 12,
        softmax_temperature: float = 0.9,
        mask_prediction: bool = False,
        categorical_features_indices: List[int] | None = None,
        inference_with_DDP: bool = False,
        seed: int = 0,
        model: torch.nn.Module = None,
        augmentations: tuple | list | None = None,
        yj_skew_threshold: float = 10.0,
        quantile_collapse: str = "mean",
        bar_temperature: float = 1.0,
        bar_point_estimator: str = "mean",
        discrete_y_snap_max_unique: int = 0,
        memory_policy: "MemoryPolicy | dict | str | None" = None,
        skip_unused_feature_decoder: bool = True,
        native_rms_norm: bool | None = None,
    ):
        """
        init NoriPredictor

        Args:
            device: The device for performing inference; GPU is recommended
            model_path: The model path of the Nori model (unused when model is provided)
            mix_precision: Whether to use mixed precision inference
            outlier_remove_std: Standard deviation used for removing outliers
            softmax_temperature: Softmax temperature coefficient
            mask_prediction: Whether to enable missing value prediction
            categorical_features_indices: Index numbers of categorical features, currently not in use
            inference_config: inference_config_setting,
            inference_with_DDP: If using DDP to inference,
            seed: Random seed
            model: Pre-loaded model instance (skips load_model when provided)
            skip_unused_feature_decoder: Skip the feature decoder when
                mask_prediction is off, where its output is computed and then
                discarded. Output-preserving, so it defaults to True. Only
                reaches models built with mask_prediction=True — that is, the
                model= path; load_model() already builds with it off.
            native_rms_norm: Tri-state. None (default) inherits whatever the
                model already has, which is the right thing in both directions:
                load_model() turns the fused kernel on, and a training-built
                model carries whatever the training run chose. True forces the
                fused kernel on, False forces the decomposed path on for a
                bit-for-bit match with history. Measured: R2 shift <= 2e-5,
                largest per-row difference one bf16 ulp, ~+1.3% end-to-end
                (preprocessing dominates inference, so the ~+10% kernel win
                does not survive).
        """
        if isinstance(inference_config, str):
            if os.path.isfile(inference_config):
                with open(inference_config, "r") as f:
                    inference_config = json.load(f)
            else:
                raise ValueError(f"inference_config is not a config file path: {inference_config}")
        for inference_config_item in inference_config:
            retrieval_config = inference_config_item.get("retrieval_config")
            if retrieval_config and retrieval_config.get("use_retrieval"):
                raise ValueError("retrieval inference has been removed; omit retrieval_config and use the full context")
        self.model_path = model_path
        self.device = device
        # Route GPU SVD (preprocess._TorchTruncatedSVD) to this predictor's
        # device so high-dim SVD runs on the same GPU as the model.
        try:
            import synthefy_nori.inference.preprocess as _pp

            _pp._GPU_SVD_DEVICE = device
        except Exception as _e:
            print(
                f"WARNING: could not route GPU SVD to {device} ({type(_e).__name__}: {_e}); using default SVD device."
            )
        self.mix_precision = mix_precision
        self.categorical_features_indices = categorical_features_indices
        self.seed = seed
        self.inference_config = inference_config
        n_estimators = len(inference_config)
        assert n_estimators > 0, f"Invalid configuration file! the number of pipelines is 0!"
        self.n_estimators = n_estimators
        self.model = None
        self.outlier_remove_std = outlier_remove_std
        self.class_shuffle_factor = 3
        self.min_seq_len_for_category_infer = 100
        self.max_unique_num_for_category_infer = 30
        self.min_unique_num_for_numerical_infer = 4
        self.preprocess_num = 10
        self.softmax_temperature = softmax_temperature
        self.mask_prediction = mask_prediction
        # V12r3: snap regression predictions to nearest training-y value
        # when training y has at most this many unique values. Helps
        # discrete-target benchmarks (wine_quality 6 levels, sensory 11,
        # ERA 9, CookbookReviews 6, chscase_foot 3) without retraining.
        # Set to 0 to disable. Default 30 covers all standard discrete-y
        # benchmarks while leaving continuous y untouched.
        self.discrete_y_snap_max_unique = int(discrete_y_snap_max_unique)
        # Stored verbatim (a preset name, dict, MemoryPolicy or None) and coerced
        # lazily in _memory_policy(). Transforming it here would break sklearn's
        # clone(), whose identity check requires the stored attribute to be the
        # object it was handed.
        self.memory_policy = memory_policy
        # --- Execution-only options: same weights, same algorithm, different
        # --- kernels. Applied per call and undone afterwards. (native_rms_norm
        # --- does shift output by ~1 bf16 ulp; see below.)
        # Skip the feature decoder when its output cannot be read. It is read
        # only by the mask_prediction (imputation) branch of
        # _predict_reg_single, so with mask_prediction=False the decoder runs
        # and its result is dropped. Default ON: predictions are bit-identical
        # either way, so this is pure removed work, not a trade-off.
        self.skip_unused_feature_decoder = bool(skip_unused_feature_decoder)
        # Use torch.nn.functional.rms_norm instead of the decomposed
        # pow/mean/rsqrt/mul chain. None means "inherit the model's setting" so
        # an explicit load_model(native_rms_norm=False) is not silently undone
        # here. Controlled measurements show:
        #   accuracy - R2 shift <= 2e-5 across 1k/4k/8k-row tables; the largest
        #              per-row difference is 0.0078125, exactly one bf16 ulp,
        #              i.e. the two formulations land on adjacent representable
        #              values under autocast(bfloat16). Not an approximation.
        #   speed    - only about +1.3% end-to-end on predict(). The kernel win
        #              is ~+10% on forward+backward, but inference is dominated
        #              by the 16 CPU preprocessing pipelines, so little of it
        #              survives. Do NOT cite a large inference speedup here.
        self.native_rms_norm = None if native_rms_norm is None else bool(native_rms_norm)
        self.inference_with_DDP = inference_with_DDP
        if self.inference_with_DDP and self.mask_prediction:
            raise ValueError(
                "inference_with_DDP does not support mask_prediction"
            )
        if self.inference_with_DDP and memory_policy is not None:
            # The DDP branch of predict() never resolves a MemoryPolicy or
            # populates memory_report_ -- it is a structurally separate path, not
            # a missing feature. Reject this combination here, loudly and at
            # construction time, rather than silently leaving memory_report_ at
            # None and letting a caller downstream (e.g. the client) misdiagnose
            # it as an out-of-date runtime.
            raise ValueError(
                "inference_with_DDP does not support memory_policy: the DDP "
                "path never resolves a memory policy, so memory_report_ would "
                "stay None regardless of what was requested. Pass "
                "memory_policy=None with inference_with_DDP=True, or drop "
                "inference_with_DDP to use the memory-policy-aware path."
            )
        # Optional inference-time augmentations. Currently supports:
        #   'yj': Yeo-Johnson target transform ensemble — fit PowerTransformer
        #         on y_train, predict in transformed space, inverse-transform,
        #         average with the identity (untransformed) pass. Targets
        #         heavy-tailed regression datasets (stock_fardamento02, CPS1988).
        #         CONDITIONAL: only applied when |skew(y_train)| > yj_skew_threshold.
        #         Ablation showed unconditional YJ hurt net R² (MIP-2016 −0.10,
        #         others −0.03 to −0.06) despite helping stock_fardamento02
        #         (+0.077). Gating on skew preserves the wins while skipping
        #         moderately-skewed datasets where YJ is harmful.
        self.augmentations = tuple(augmentations) if augmentations else ()
        self.yj_skew_threshold = float(yj_skew_threshold)
        # How to collapse K-quantile output to a point estimate per row.
        #   'mean'          — simple average of all τ quantiles (≈ E[y]
        #                     under uniform τ spacing; current default)
        #   'median'        — the τ=0.5 quantile (robust to quantile noise;
        #                     more conservative on skewed distributions)
        #   'trimmed_mean'  — drop outer 5% of τ, average the rest
        #   'huber_mean'    — MAD-normalized Huber-weighted mean around q_0.5
        #   'tail_aware'    — per-row skewness test on the predicted quantile
        #                     distribution: if left-heavy (q_0.5-q_0.01 >>
        #                     q_0.99-q_0.5), return q_0.01; right-heavy →
        #                     q_0.99; balanced → mean. Targets extreme-outlier
        #                     rows (Job_Profitability, capped houses) where
        #                     the model's quantile spread signals an
        #                     out-of-bulk prediction.
        # All strategies are zero-retrain: they only change how the K-way
        # quantile head is collapsed to a single prediction.
        valid_collapse = ("mean", "median", "trimmed_mean", "huber_mean", "tail_aware", "qdist", "qdist_simple")
        if quantile_collapse not in valid_collapse:
            raise ValueError(f"quantile_collapse must be one of {valid_collapse}, got {quantile_collapse!r}")
        self.quantile_collapse = quantile_collapse
        # 'qdist' invokes the quantile-distribution decoder (sort-monotone
        # + analytical mean with exp tail extrapolation; ports TabICL's
        # _model/quantile_dist.py). 'qdist_simple' uses the same sort+
        # analytical-mean but without tail extrapolation (faster, pure-torch).
        # Bar-distribution inference controls (only used when the loaded model
        # was trained with regression_loss='bar_distribution', auto-detected
        # via model.regression_loss). Ignored for pinball/MSE/etc. checkpoints.
        valid_bar = ("mean", "mode", "median")
        if bar_point_estimator not in valid_bar:
            raise ValueError(f"bar_point_estimator must be one of {valid_bar}, got {bar_point_estimator!r}")
        self.bar_temperature = float(bar_temperature)
        self.bar_point_estimator = bar_point_estimator

        device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
        if device_type == "cpu":
            self.mix_precision = False
            print("Mixed precision is not supported for CPU inference, so it has been automatically disabled")
        elif device_type == "mps" and self.mix_precision:
            # MPS autocast uses float16, while Nori's CPU path and checkpoint
            # weights use float32. The reduced precision can materially change
            # regression quality even when every output remains finite.
            self.mix_precision = False
            logger.info("Mixed precision is disabled on MPS to preserve inference quality")

        if model is not None:
            self.model = model
        else:
            self.model = load_model(model_path=model_path, mask_prediction=mask_prediction)
        if self.mask_prediction and getattr(self._bare_model(), "feature_decoder", None) is None:
            raise ValueError(
                "mask_prediction=True requires a model with feature_decoder; "
                "the supplied model explicitly omits that head"
            )

        self.preprocess_pipelines = []
        self.preprocess_configs = []

        self.build_preprocess_pipeline()

    def build_preprocess_pipeline(self):
        self.preprocess_pipelines = []
        self.preprocess_configs = []

        random.seed(self.seed)
        rand_gen = np.random.default_rng(self.seed)
        self.seeds = [random.randint(0, 10000) for _ in range(self.n_estimators * self.preprocess_num)]
        start_idx = rand_gen.integers(0, 1000)
        all_shifts = list(range(start_idx, start_idx + self.n_estimators))
        self.all_shifts = rand_gen.choice(all_shifts, size=self.n_estimators, replace=False)

        if self.mask_prediction:
            for inference_config_item in self.inference_config:
                if len(inference_config_item["RebalanceFeatureDistribution"]["worker_tags"]) > 0:
                    for i, v in enumerate(inference_config_item["RebalanceFeatureDistribution"]["worker_tags"]):
                        if v == "power":
                            print(
                                "WARNING: Missing value imputation does not currently support the preprocessing method of power! Using the default worker_tags method"
                            )
                            inference_config_item["RebalanceFeatureDistribution"]["worker_tags"].pop(i)
                            inference_config_item["RebalanceFeatureDistribution"]["worker_tags"].append(None)
                inference_config_item["RebalanceFeatureDistribution"]["discrete_flag"] = True

        for idx in range(self.n_estimators):
            pipeline = []
            inference_config_item = self.inference_config[idx]
            # HighDimFeatureSelector runs BEFORE MaxFeatureSubsampler so we
            # can do supervised top-k selection (corr / MI / ExtraTrees) or
            # SVD projection on binary fingerprints before any random pruning.
            # Self-gates: passthrough on low-dim datasets, identical to today's
            # behavior. Activates only when n_features > threshold OR
            # binary_frac >= threshold.
            if "HighDimFeatureSelector" in inference_config_item:
                pipeline.append(HighDimFeatureSelector(**inference_config_item["HighDimFeatureSelector"]))
            # MaxFeatureSubsampler runs BEFORE poly generator so poly pairs are
            # drawn from the subsampled feature set (matches TabPFN semantics:
            # each estimator sees ≤ max_features original columns).
            if "MaxFeatureSubsampler" in inference_config_item:
                pipeline.append(MaxFeatureSubsampler(**inference_config_item["MaxFeatureSubsampler"]))
            if "PolynomialInteractionGenerator" in inference_config_item:
                pipeline.append(
                    PolynomialInteractionGenerator(**inference_config_item["PolynomialInteractionGenerator"])
                )

            pipeline.append(FilterValidFeatures())

            # MAD winsorization: clip per-column at ±N MAD from median, matching
            # the training-side safety winsorization. Runs BEFORE rebalance so
            # subsequent transforms see clipped values.
            if "MADWinsorizer" in inference_config_item:
                pipeline.append(MADWinsorizer(**inference_config_item["MADWinsorizer"]))

            if "RebalanceFeatureDistribution" in inference_config_item:
                pipeline.append(RebalanceFeatureDistribution(**inference_config_item["RebalanceFeatureDistribution"]))
            if "CategoricalFeatureEncoder" in inference_config_item:
                pipeline.append(CategoricalFeatureEncoder(**inference_config_item["CategoricalFeatureEncoder"]))
            if inference_config_item.get("FingerprintFeatureEncoder", False):
                pipeline.append(FingerprintFeatureEncoder())
            if "FeatureShuffler" in inference_config_item:
                shuffler = FeatureShuffler(**inference_config_item["FeatureShuffler"])
                shuffler.offset = self.all_shifts[idx]
                pipeline.append(shuffler)

            self.preprocess_pipelines.append(pipeline)

    def _check_n_features(self, X, reset):
        """Check whether the number of features matches the previous evaluation"""
        n_features = X.shape[1]
        if reset:
            self.n_features_in_ = n_features
        else:
            if self.n_features_in_ != n_features:
                raise ValueError(
                    f"X has {n_features} features, but this estimator is expecting {self.n_features_in_} features."
                )

    def validate_data(self, x=None, y=None, reset=True, validate_separately=False, **check_params):
        """
        {'accept_sparse': False, 'dtype': None, 'ensure_all_finite': 'allow-nan'}
        """
        # Validate both x and y simultaneously
        if y is not None:
            x, y = check_X_y(x, y, **check_params)
            self._check_n_features(x, reset=reset)
            return x, y

        # Validate X
        if x is not None:
            x = check_array(x, **check_params)
            self._check_n_features(x, reset=reset)
            return x

        return None

    def convert_x_dtypes(self, x: np.ndarray, dtypes: Literal["float32", "float64"] = "float64"):
        NUMERIC_DTYPE_KINDS = "?bBiufm"
        OBJECT_DTYPE_KINDS = "OV"
        STRING_DTYPE_KINDS = "SaU"

        if x.dtype.kind in NUMERIC_DTYPE_KINDS:
            x = pd.DataFrame(x, copy=False, dtype=dtypes)
        elif x.dtype.kind in OBJECT_DTYPE_KINDS:
            x = pd.DataFrame(x, copy=True)
            x = x.convert_dtypes()
        else:
            raise ValueError(f"Unsupport string dtypes! {x.dtype}")

        integer_columns = x.select_dtypes(include=["number"]).columns
        if len(integer_columns) > 0:
            x[integer_columns] = x[integer_columns].astype(dtypes)
        return x

    def _drop_high_cardinality_string_columns(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        string_cols_pre = x_train.select_dtypes(include=["string", "object"]).columns
        cols_to_drop = []
        for col in string_cols_pre:
            n_unique = x_train[col].nunique()
            n_samples = len(x_train[col])
            if n_samples > 50 and n_unique / n_samples > 0.90:
                cols_to_drop.append(col)
        if cols_to_drop:
            x_train = x_train.drop(columns=cols_to_drop)
            x_test = x_test.drop(columns=cols_to_drop)
        return x_train, x_test

    def _fit_category_encoder(
        self,
        x_train: pd.DataFrame,
        dtype: np.floating = np.float64,
        placeholder: str = NA_PLACEHOLDER,
    ) -> dict:
        ordinal_encoder = OrdinalEncoder(
            categories="auto",
            dtype=dtype,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=np.nan,
        )
        encoded_cols = list(x_train.select_dtypes(include=["category", "string", "bool"]).columns)
        string_cols = list(x_train.select_dtypes(include=["string", "object"]).columns)
        x_train_fit = x_train.copy()
        if string_cols:
            x_train_fit[string_cols] = x_train_fit[string_cols].fillna(placeholder)
        col_encoder = ColumnTransformer(
            transformers=[
                ("encoder", ordinal_encoder, make_column_selector(dtype_include=["category", "string", "bool"]))
            ],
            remainder=FunctionTransformer(),
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )
        col_encoder.fit(x_train_fit)
        return {
            "encoder": col_encoder,
            "encoded_cols": encoded_cols,
            "string_cols": string_cols,
            "placeholder": placeholder,
        }

    def _transform_category2num(
        self,
        x: pd.DataFrame,
        encoder_state: dict,
    ) -> np.ndarray:
        x_enc = x.copy()
        string_cols = encoder_state["string_cols"]
        placeholder = encoder_state["placeholder"]
        if string_cols:
            x_enc[string_cols] = x_enc[string_cols].fillna(placeholder)

        X_encoded = encoder_state["encoder"].transform(x_enc)
        if string_cols:
            encoded_cols = encoder_state["encoded_cols"]
            string_output_positions = [encoded_cols.index(col) for col in string_cols]
            placeholder_mask = (x_enc[string_cols] == placeholder).to_numpy()
            X_encoded[:, string_output_positions] = np.where(
                placeholder_mask,
                np.nan,
                X_encoded[:, string_output_positions],
            )
        return X_encoded

    def _prepare_inductive_features(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        x_train_df = self.convert_x_dtypes(x_train)
        x_test_df = self.convert_x_dtypes(x_test)
        x_train_df, x_test_df = self._drop_high_cardinality_string_columns(
            x_train_df,
            x_test_df,
        )
        encoder_state = self._fit_category_encoder(x_train_df)
        x_train_enc = self._transform_category2num(x_train_df, encoder_state).astype(np.float32)
        x_test_enc = self._transform_category2num(x_test_df, encoder_state).astype(np.float32)
        categorical_idx = self.get_categorical_features_indices(x_train_enc)
        return x_train_enc, x_test_enc, categorical_idx

    def _seed_step_index(self, pipe: list, id_step: int) -> int:
        """Seed-array index for pipeline step `id_step`.

        HighDimFeatureSelector gets the spare last slot (preprocess_num is a
        fixed 10, larger than any real pipeline). Every other step is indexed
        by its position EXCLUDING the HDF step — so adding HDF to a config
        does not shift any downstream step's seed. A config with HDF is then
        byte-identical to one without it on datasets where HDF passes through
        (gate inactive: n_features below threshold).
        """
        if isinstance(pipe[id_step], HighDimFeatureSelector):
            return self.preprocess_num - 1
        return sum(1 for s in pipe[:id_step] if not isinstance(s, HighDimFeatureSelector))

    def _fit_transform_step_inductive(
        self,
        step,
        x_train: np.ndarray,
        x_test: np.ndarray,
        categorical_idx: list[int],
        seed: int,
        y_train: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        # `y_train` is forwarded to selectors that require it (e.g.
        # HighDimFeatureSelector with corr / mi / extratrees). Steps that don't
        # accept a y= kwarg simply ignore it via **kwargs.
        categorical_idx = step.fit(x_train, categorical_idx, seed, y=y_train)
        if isinstance(step, FingerprintFeatureEncoder):
            x_train_out, categorical_idx = step.transform(x_train, is_test=False)
            x_test_out, categorical_idx = step.transform(x_test, is_test=True)
            return x_train_out, x_test_out, categorical_idx

        if isinstance(step, FilterValidFeatures):
            x_train_out, categorical_idx = step.transform(x_train)
            train_invalid = None if step.invalid_features is None else step.invalid_features.copy()
            x_test_out, categorical_idx = step.transform(x_test)
            test_invalid = None if step.invalid_features is None else step.invalid_features.copy()
            if train_invalid is None:
                step.invalid_features = test_invalid
            elif test_invalid is None:
                step.invalid_features = train_invalid
            else:
                step.invalid_features = np.concatenate([train_invalid, test_invalid], axis=0)
            return x_train_out, x_test_out, categorical_idx

        x_train_out, categorical_idx = step.transform(x_train)
        x_test_out, categorical_idx = step.transform(x_test)
        return x_train_out, x_test_out, categorical_idx

    def get_categorical_features_indices(self, x: np.ndarray):
        if x.shape[0] < self.min_seq_len_for_category_infer:
            return []
        categorical_idx = []
        for idx, col in enumerate(x.T):
            if len(np.unique(col)) < self.min_unique_num_for_numerical_infer:
                categorical_idx.append(idx)
        return categorical_idx

    def _warn_once_per_call(self, key: str, message: str, category) -> None:
        """Emit ``message`` at most once per public ``predict`` call.

        The runtime warnings (plain-loop fallback, context subsampling) describe the
        REQUEST, so once per request is right -- but they are raised inside
        ``_predict_reg_single``, which runs once per inference pipeline (16 on the
        default config). Emitting there directly produced 16 identical copies per
        predict, and Python's per-location de-duplication does not help because
        ``stacklevel`` attributes them to a varying frame.

        Args:
            key: stable identifier for this warning kind.
            message: the full warning text.
            category: warning class to raise.
        """
        seen = getattr(self, "_warned_this_call", None)
        if seen is None:
            seen = self._warned_this_call = set()
        if key in seen:
            return
        seen.add(key)
        warnings.warn(message, category, stacklevel=4)

    def _log_once_per_call(self, key: str, level: int, message: str) -> None:
        """Log ``message`` at most once per public ``predict`` call.

        Same reason as :meth:`_warn_once_per_call`, for the logger: these sites run once
        per inference pipeline, and Python's logging has no de-duplication at all. With
        no handler configured, logging's handler-of-last-resort writes WARNING to
        stderr, so a user saw 16 identical lines per predict (measured 32 across two
        calls) before this.

        Args:
            key: stable identifier for this log kind.
            level: logging level.
            message: the full text.
        """
        seen = getattr(self, "_logged_this_call", None)
        if seen is None:
            seen = self._logged_this_call = set()
        if key in seen:
            return
        seen.add(key)
        logger.log(level, "%s", message)

    def predict(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, return_distribution: bool = False
    ) -> np.ndarray:
        """
        Perform regression inference using the Nori model

        Args:
        x_train: Training data x
        y_train: Training data y
        x_test:  Testing data x
        return_distribution: when True, return the full per-row quantile bank
            (shape [n_test, num_reg_quantiles] for a pinball head, or the
            ensemble-averaged bar-distribution logits [n_test, num_bars] for a
            bar_distribution head) WITHOUT collapsing to a point estimate. The
            point-estimate post-processing (quantile collapse, discrete-y snap)
            and the Yeo-Johnson point ensemble are skipped — the raw decoder
            output is the predictive distribution callers want for scoring.
        """
        # Reset the per-call de-duplication sets. The runtime warnings and logs
        # below are raised once per inference pipeline (16 on the default
        # config); these make them once per user-visible predict instead.
        self._warned_this_call = set()
        self._logged_this_call = set()
        self._exact_cudagraph_step_marked = False
        self.memory_report_ = None
        self._memory_attempt_history = []
        self._memory_outcome_policy = None
        if not self._exact_cached_cudagraphs_requested():
            bare_model = self._bare_model()
            if getattr(bare_model, "_exact_cudagraphs_enabled", False):
                self._disable_exact_cached_cudagraphs(bare_model)
        with self._execution_overrides():
            if return_distribution:
                return self._predict_reg(x_train, y_train, x_test, return_distribution=True)
            preds = self._predict_reg(x_train, y_train, x_test)
        # V12r3: discrete-y snap. If enabled (discrete_y_snap_max_unique > 0)
        # and training y has a low unique count (≤ the threshold), snap each
        # prediction to the nearest training-y value. OFF by default since
        # v0.3: snapping is opt-in (via NoriRegressor's discretize= or by
        # setting this threshold explicitly) — the raw conditional mean
        # is the R²-optimal point output, and benchmarking showed lattice
        # snapping costs ~0.05 R² on K≤10 targets.
        if getattr(self, "discrete_y_snap_max_unique", 0) > 0:
            preds = self._maybe_snap_discrete_y(y_train, preds)
        return preds

    @staticmethod
    def _unwrap_model_output(output, *, task_type: str):
        """Normalize model forward outputs across training/inference contracts.

        Most backbones return a tensor when mask_prediction=False. V14 always
        returns the trainer dict unless return_tensor=True is passed, so the
        predictor must unwrap it before stacking/chunk concatenation.
        """
        if isinstance(output, dict):
            if task_type == "reg":
                return output["reg_output"]
            if task_type == "cls":
                return output["cls_output"]
        return output

    def _reject_nonfinite_output(
        self,
        out: torch.Tensor,
        *,
        path: str,
        n_train: int,
        n_test: int,
    ) -> torch.Tensor:
        """Raise when the model emits NaN/inf, instead of zero-filling it.

        Zero-filling was never a safe default here: the predictor works in
        standardized target space, so a zeroed prediction denormalizes to
        exactly ``y_mean``. A fully non-finite forward therefore became a
        *constant* prediction that was finite, plausible-looking, and useless —
        ROC AUC 0.5 with no error anywhere in the call. That silent conversion
        is what made issue #439 invisible for as long as it was.

        There is deliberately no opt-out. A caller who wants the old behavior
        can only want a number that is indistinguishable from a working
        prediction while carrying no signal, so the only honest thing this can
        return is an exception.
        """
        if torch.isfinite(out).all():
            return out

        n_bad = int((~torch.isfinite(out)).sum())
        raise RuntimeError(
            f"Nori produced {n_bad}/{out.numel()} non-finite prediction values "
            f"(nan={int(torch.isnan(out).sum())}, inf={int(torch.isinf(out).sum())}) "
            f"on the {path} path: n_context={n_train}, n_query={n_test}, "
            f"dtype={out.dtype}, mixed_precision={self.mix_precision}. "
            "These cannot be returned: in standardized target space they would "
            "denormalize to the context target mean, i.e. a constant prediction "
            "that scores at chance while reporting success."
        )

    def _maybe_snap_discrete_y(self, y_train: np.ndarray, preds: np.ndarray) -> np.ndarray:
        """Snap regression predictions to nearest training y value when
        training y is discrete (low unique count).

        Detection: y_train has <= self.discrete_y_snap_max_unique unique
        values. When triggered, each prediction is replaced by the closest
        unique y value seen in training. Continuous targets (many unique
        values) pass through unchanged.

        This is a pure post-hoc adjustment — does not change the model
        forward pass; only the final decoded output. Safe to apply broadly.
        """
        try:
            y_arr = np.asarray(y_train, dtype=np.float64).reshape(-1)
            y_finite = y_arr[np.isfinite(y_arr)]
            if y_finite.size == 0:
                return preds
            unique_y = np.unique(y_finite)
            threshold = int(getattr(self, "discrete_y_snap_max_unique", 0))
            if len(unique_y) > threshold or len(unique_y) < 2:
                return preds
            # Sort uniques and snap each prediction to nearest by absolute
            # distance. Vectorized for speed even on large query sets.
            unique_sorted = np.sort(unique_y)
            preds_arr = np.asarray(preds).reshape(-1).astype(np.float64)
            # Find insertion index for each pred, compare neighbors.
            idx = np.searchsorted(unique_sorted, preds_arr)
            idx = np.clip(idx, 0, len(unique_sorted) - 1)
            left_idx = np.clip(idx - 1, 0, len(unique_sorted) - 1)
            d_right = np.abs(preds_arr - unique_sorted[idx])
            d_left = np.abs(preds_arr - unique_sorted[left_idx])
            chosen = np.where(d_left < d_right, unique_sorted[left_idx], unique_sorted[idx])
            return chosen.reshape(np.asarray(preds).shape).astype(np.asarray(preds).dtype)
        except Exception as _e:
            # Fail open — never break inference because of the snap helper.
            print(f"WARNING: discrete-y snap failed ({type(_e).__name__}: {_e}); returning unsnapped predictions.")
            return preds

    def _bare_model(self):
        """Unwrap DDP / torch.compile wrappers to reach the backbone module."""
        model_ref = self.model
        if hasattr(model_ref, "module"):
            model_ref = model_ref.module
        # torch.compile wraps in an OptimizedModule; attributes set on the
        # wrapper never reach the module whose forward actually reads them, so
        # _skip_feature_decoder would silently do nothing.
        model_ref = getattr(model_ref, "_orig_mod", model_ref)
        return model_ref

    @contextlib.contextmanager
    def _execution_overrides(self):
        """Apply the execution-only speedups for the duration of one call.

        Both settings live on the model object, and ``model=`` may hand us a
        module the caller also trains with or uses for imputation elsewhere.
        Mutating it permanently would be a silent action at a distance — a
        leaked ``_skip_feature_decoder`` would zero the feature-reconstruction
        loss of a subsequent training step. So every change is undone on exit,
        including on exception.
        """
        model = self._bare_model()
        undo = []

        # Gate on mask_prediction at call time, not construction time: callers
        # do flip the attribute on an existing predictor.
        if self.skip_unused_feature_decoder and not self.mask_prediction:
            previous = getattr(model, "_skip_feature_decoder", False)
            if not previous:
                model._skip_feature_decoder = True
                undo.append(lambda: setattr(model, "_skip_feature_decoder", previous))

        # None = inherit the model's current setting and touch nothing.
        if self.native_rms_norm is not None:
            want = self.native_rms_norm
            for module in model.modules():
                if isinstance(module, RMSNorm) and module.use_native != want:
                    had = module.use_native
                    module.use_native = want
                    undo.append(lambda m=module, v=had: setattr(m, "use_native", v))

        try:
            yield
        finally:
            for revert in reversed(undo):
                revert()

    @property
    def regression_head(self) -> str:
        """'bar_distribution' or 'pinball' (quantile) — the decoder head type."""
        model = self._bare_model()
        regression_loss = getattr(model, "regression_loss", None)
        if regression_loss is None:
            # Legacy architecture records have only the decoder width. Their
            # default scalar head was MSE; width > 1 identified pinball. A
            # historical one-level pinball head cannot be distinguished, which
            # is why current checkpoints persist regression_loss explicitly.
            regression_loss = "mse" if int(getattr(model, "num_reg_quantiles", 1)) == 1 else "pinball"
        return str(regression_loss)

    @property
    def num_reg_quantiles(self) -> int:
        return int(getattr(self._bare_model(), "num_reg_quantiles", 1))

    @property
    def regression_quantiles(self) -> tuple[float, ...]:
        """Quantile levels carried by the checkpoint's pinball head.

        Checkpoints predating explicit level metadata stored only ``K``. Their
        training grid was ``i / (K + 1)``, so retain that as the legacy
        fallback rather than guessing from any current training defaults.
        """
        model = self._bare_model()
        levels = getattr(model, "regression_quantiles", None)
        if levels is not None:
            return tuple(float(level) for level in levels)
        count = int(getattr(model, "num_reg_quantiles", 1))
        return tuple((index + 1.0) / (count + 1.0) for index in range(count))

    def _collapse_regression_output(self, output: torch.Tensor) -> torch.Tensor:
        """Convert regression decoder output to one point prediction per test row.

        Pinball-trained checkpoints emit one column per quantile τ_i
        (ordered ascending, e.g. τ=0.01..0.99). This method collapses the
        [..., K] quantile bank to a [...] point estimate using
        self.quantile_collapse (set at __init__). ``mean`` integrates the
        piecewise-linear quantile function on its recorded τ grid; other
        strategies trade accuracy on symmetric bulk for robustness/tails.

        Bar-distribution-trained checkpoints emit K bin logits over a fixed
        [bar_borders_low, bar_borders_high] range. When we detect that mode on
        the loaded model, we skip the quantile-collapse path and decode via
        softmax → point estimate (mean/mode/median of the predicted density).
        """
        model_ref = self._bare_model()
        num_reg_quantiles = int(getattr(model_ref, "num_reg_quantiles", 1))
        regression_loss = self.regression_head
        if output.ndim == 0:
            output = output.unsqueeze(0)

        if num_reg_quantiles > 1:
            if regression_loss == "bar_distribution":
                num_bars = int(getattr(model_ref, "num_bars", num_reg_quantiles))
                lo = float(getattr(model_ref, "bar_borders_low", -10.0))
                hi = float(getattr(model_ref, "bar_borders_high", 10.0))
                # Prefer stored borders buffer (V12+ ships custom non-uniform
                # borders for normal-quantile mode). Fall back to uniform
                # [lo, hi] for older checkpoints.
                stored_borders = getattr(model_ref, "bar_borders_buffer", None)
                if output.ndim == 1 and output.shape[0] == num_bars:
                    # Single test row with num_bars logits
                    output = (
                        self._apply_bar_distribution_decode(
                            output.unsqueeze(0),
                            num_bars,
                            lo,
                            hi,
                            borders=stored_borders,
                        )
                        .squeeze(0)
                        .unsqueeze(0)
                    )
                elif output.shape[-1] == num_bars:
                    output = self._apply_bar_distribution_decode(
                        output,
                        num_bars,
                        lo,
                        hi,
                        borders=stored_borders,
                    )
            else:
                if output.ndim == 1 and output.shape[0] == num_reg_quantiles:
                    # Single test row with K quantiles — collapse, then re-expand
                    output = self._apply_quantile_collapse(output.unsqueeze(0)).squeeze(0).unsqueeze(0)
                elif output.shape[-1] == num_reg_quantiles:
                    output = self._apply_quantile_collapse(output)

        output = output.squeeze()
        if output.ndim == 0:
            output = output.unsqueeze(0)
        return output

    def _apply_bar_distribution_decode(
        self,
        logits: torch.Tensor,
        num_bars: int,
        lo: float,
        hi: float,
        borders: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode [..., num_bars] logits to [...] point estimate.

        probs = softmax(logits / T), where T = self.bar_temperature (default 1.0).
        Point estimate is selected by self.bar_point_estimator:
          'mean'   — E[y] = Σ prob_i * center_i   (default)
          'mode'   — bin_center at argmax(prob)
          'median' — bin_center where CDF first crosses 0.5

        If `borders` is provided (V12+ stored buffer), uses those non-uniform
        edges. Else constructs uniform [lo, hi] borders fresh on the logits
        device (legacy path, compile-compatible).
        """
        if borders is not None and borders.shape[0] == num_bars + 1:
            borders = borders.to(device=logits.device, dtype=logits.dtype)
        else:
            borders = torch.linspace(
                lo,
                hi,
                num_bars + 1,
                device=logits.device,
                dtype=logits.dtype,
            )
        bin_centers = 0.5 * (borders[:-1] + borders[1:])  # [num_bars]
        T = float(getattr(self, "bar_temperature", 1.0))
        if T <= 0:
            T = 1.0
        probs = torch.softmax(logits.float() / T, dim=-1)
        mode = getattr(self, "bar_point_estimator", "mean")
        if mode == "mode":
            idx = probs.argmax(dim=-1)
            return bin_centers[idx]
        if mode == "median":
            cdf = probs.cumsum(dim=-1)
            idx = (cdf >= 0.5).int().argmax(dim=-1)  # first bin where CDF crosses 0.5
            return bin_centers[idx]
        # mean (default)
        return (probs * bin_centers).sum(dim=-1)

    def _apply_quantile_collapse(self, q: torch.Tensor) -> torch.Tensor:
        """Collapse [..., K] quantile tensor to [...] point estimate.

        Strategies:
          mean          — current default; integral of the piecewise-linear
                          quantile function on the checkpoint's exact τ grid
          median        — q at the middle τ index; robust to quantile
                          crossing, but more conservative on skewed rows
          trimmed_mean  — drop outer 5% of τ indices, average rest
          huber_mean    — Huber-weighted mean with center = τ=0.5 quantile,
                          scale = MAD of quantile values
        """
        mode = getattr(self, "quantile_collapse", "mean")
        K = q.shape[-1]
        if K <= 1:
            return q.mean(dim=-1)
        # MPS does not implement float64 tensors. Float32 still keeps every
        # level in the released 999-quantile grid distinct; retain float64 on
        # CPU/CUDA so their existing integration precision is unchanged.
        tau_dtype = torch.float32 if q.device.type == "mps" else torch.float64
        if mode == "mean":
            from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple

            tau_levels = torch.as_tensor(
                self.regression_quantiles,
                device=q.device,
                dtype=tau_dtype,
            )
            return quantile_dist_mean_simple(
                q,
                tau_levels,
                enforce_monotone_first=True,
            )
        if mode == "median":
            # For an even-sized bank there is no single middle level. Average
            # the two central predictions; choosing K // 2 alone makes K=2
            # return the upper endpoint instead of its median.
            middle = K // 2
            if K % 2:
                return q[..., middle]
            return q[..., middle - 1 : middle + 1].mean(dim=-1)
        if mode == "trimmed_mean":
            # Keep at least one value. The historical max(1, ...) made K=2
            # trim both endpoints and reduce an empty tensor to NaN.
            trim = min(max(1, int(K * 0.05)), (K - 1) // 2)
            if trim == 0:
                return q.mean(dim=-1)
            return q[..., trim : K - trim].mean(dim=-1)
        if mode == "huber_mean":
            # Sort defensively against quantile crossing — q should already be
            # monotonic in τ for a well-trained pinball head.
            q_sorted, _ = q.sort(dim=-1)
            center = q_sorted[..., K // 2 : K // 2 + 1]
            mad = (q_sorted - center).abs().median(dim=-1, keepdim=True).values
            mad = torch.clamp(mad, min=1e-8)
            dev = (q_sorted - center) / mad
            k_huber = 1.5
            w = torch.where(
                dev.abs() < k_huber,
                torch.ones_like(dev),
                k_huber / dev.abs().clamp(min=k_huber),
            )
            return (q_sorted * w).sum(dim=-1) / w.sum(dim=-1).clamp(min=1e-8)
        if mode == "tail_aware":
            # Use the shape of the predicted quantile distribution to detect
            # rows where the model is signaling an out-of-bulk prediction.
            # For such rows, collapse to the heavy-side quantile instead of
            # the mean (which gets pulled back by bulk-side quantile mass).
            # Pure bulk rows fall through to mean — no bias there.
            #
            # Rule:
            #   left_weight  = q[τ=0.5] - q[τ=0.01]   (median → low tail)
            #   right_weight = q[τ=0.99] - q[τ=0.5]   (median → high tail)
            #   LEFT_HEAVY  if left_weight  > SKEW_RATIO * right_weight
            #   RIGHT_HEAVY if right_weight > SKEW_RATIO * left_weight
            # Both require spread > MIN_SPREAD_RATIO × (batch-median spread)
            # so a uniformly-tiny quantile spread can't accidentally trigger.
            lo_idx = max(0, int(round(K * 0.01)))  # τ≈0.01
            mid_idx = K // 2  # τ=0.5
            hi_idx = min(K - 1, int(round(K * 0.99)) - 1)  # τ≈0.99
            q_lo = q[..., lo_idx]
            q_mid = q[..., mid_idx]
            q_hi = q[..., hi_idx]
            left_w = q_mid - q_lo
            right_w = q_hi - q_mid
            spread = (q_hi - q_lo).abs()
            # Relative threshold: a row's spread has to stand out from the batch.
            spread_thresh = spread.median() * 1.5 if q.shape[0] > 1 else spread.new_zeros(()) - 1.0
            SKEW_RATIO = 3.0
            left_heavy = (left_w.abs() > SKEW_RATIO * right_w.abs().clamp(min=1e-8)) & (spread > spread_thresh)
            right_heavy = (right_w.abs() > SKEW_RATIO * left_w.abs().clamp(min=1e-8)) & (spread > spread_thresh)
            from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple

            tau_levels = torch.as_tensor(
                self.regression_quantiles,
                device=q.device,
                dtype=tau_dtype,
            )
            mean_est = quantile_dist_mean_simple(
                q,
                tau_levels,
                enforce_monotone_first=True,
            )
            result = torch.where(left_heavy, q_lo.to(mean_est.dtype), mean_est)
            result = torch.where(right_heavy, q_hi.to(mean_est.dtype), result)
            return result
        if mode == "qdist":
            # Quantile-distribution decoder: sort + analytical mean with
            # exp tail extrapolation. K should be ≥ ~100 for stable tail fit.
            # Falls back to qdist_simple at K < 8.
            from synthefy_nori.model.quantile_dist import quantile_dist_mean_batch

            tau_levels = np.asarray(
                self.regression_quantiles,
                dtype=np.float64,
            )
            return quantile_dist_mean_batch(
                q,
                tau_levels,
                enforce_monotone_first=True,
                tail_outer_n=20,
            )
        if mode == "qdist_simple":
            # Quantile-distribution decoder, pure-torch (no tail correction).
            # Faster, fully on-device. Use when tail extrapolation isn't needed
            # (e.g. K=999 already covers 99.9% of mass).
            from synthefy_nori.model.quantile_dist import quantile_dist_mean_simple

            tau_levels = torch.as_tensor(
                self.regression_quantiles,
                device=q.device,
                dtype=tau_dtype,
            )
            return quantile_dist_mean_simple(
                q,
                tau_levels,
                enforce_monotone_first=True,
            )
        # Should be unreachable (validated in __init__), but fall back safely.
        return q.mean(dim=-1)

    def _effective_budget_n_features(self, n_features: int, x_train: np.ndarray) -> int:
        """Feature count the model actually sees after a HighDimFeatureSelector
        in the inference config reduces dimensionality.

        The OOM row-budget must use this reduced count. Using the raw count
        subsamples training rows to fit memory the model never uses:
        QSAR-TID-11 (1024 raw features, svd_all -> 256) was wrongly cut from
        4019 context rows to 976, costing ~0.10 R2.
        """
        eff = n_features
        binary_frac = None
        for item in self.inference_config:
            if not isinstance(item, dict):
                continue
            hdf = item.get("HighDimFeatureSelector")
            if not hdf:
                continue
            strategy = hdf.get("strategy", "passthrough")
            thr = int(hdf.get("n_features_threshold", 128))
            bthr = float(hdf.get("binary_threshold", 0.5))
            if binary_frac is None:
                try:
                    bm = HighDimFeatureSelector._detect_binary_cols(np.asarray(x_train, dtype=np.float64))
                    binary_frac = float(bm.mean()) if n_features else 0.0
                except Exception:
                    binary_frac = 0.0
            if not ((n_features > thr) or (binary_frac >= bthr)):
                continue
            if strategy == "svd_all":
                eff = min(eff, int(hdf.get("svd_components", 64)))
            elif strategy in ("corr", "mi", "extratrees"):
                eff = min(eff, int(hdf.get("top_k", 256)))
            # svd_binary output size is data-dependent (n_nonbinary +
            # components) — leave the budget conservative (no reduction).
        return max(1, eff)

    def _default_max_elements_budget(self) -> int:
        """Default per-forward element budget, scaled to accelerator memory.

        Anchored at 2M elements for a ~24GB GPU (the historical conservative
        default) and linear in total VRAM, so e.g. a 140GB H200 gets ~11.7M
        without manual tuning. Apple MPS uses Metal's recommended maximum
        working-set size. Falls back to the 2M floor on CPU or if the device
        cannot be queried. Overridden by
        SYNTHEFY_MAX_ELEMENTS_BUDGET when that env var is set.
        """
        base = 2_000_000
        total_gb = self._total_vram_gb()
        if total_gb is not None:
            return max(base, int(base * (total_gb / 24.0)))
        return base

    #: How many DISTINCT encoded contexts ``_get_or_build_context`` retains per pipeline
    #: when ``memory_policy.reuse_context_cache`` is on.
    #:
    #: 1 is the historical behavior and right for ordinary ``predict``, which reuses one
    #: context. A large-context policy (:mod:`synthefy_nori.inference.large_context`) rotates between
    #: several shared pools, and at 1 each pool evicts the previous one, so a policy with
    #: 8 pools gets zero hits and pays 8 rebuilds *plus* 8 verification clones. Costs one
    #: full K/V cache per entry, so raising it is opt-in
    #: (``NoriRegressor(large_context_cache_entries=K)``) rather than automatic -- growing this
    #: on the caller's behalf would trade a missed optimization for an OOM.
    #:
    #: Deliberately an attribute here and NOT a :class:`MemoryPolicy` field: that class is
    #: mirrored in the ``synthefy`` client and published in the serving request schema, and
    #: this is an implementation detail of a library-only feature the hosted API does not
    #: expose. Putting it there broke client/server parity for no caller benefit.
    context_cache_entries: int = 1

    def budget_n_features(self, x_train: np.ndarray) -> int:
        """The feature count :meth:`max_context_rows` divides the budget by.

        Public because it is the expensive, *table-derived* half of that arithmetic and
        a caller re-checking the window every call should not repay it: detecting the
        binary columns costs a float64 copy of the whole table and a ``np.unique`` per
        column (~5s on 1M x 130). It depends only on ``inference_config`` and the
        values in ``x_train``, so it is fixed for the life of a fit, while callers may
        change the element budget between predictions. Cache this, pass it back to
        :meth:`max_context_rows`, and drop it on re-``fit``.
        """
        x_train = np.asarray(x_train)
        n_features = x_train.shape[1] if x_train.ndim > 1 else 1
        return self._effective_budget_n_features(n_features, x_train)

    def max_context_rows(self, x_train: np.ndarray, *, budget_n_features: int | None = None) -> int:
        """Context rows one call can take for this table before subsampling engages.

        Mirrors the budget arithmetic in :meth:`_predict_reg_single` exactly: returns
        ``len(x_train)`` when the whole context already fits the element budget, and
        otherwise the same ``max_train_samples`` that path would trim it to.

        Exists so a large-context policy (:mod:`synthefy_nori.inference.large_context`) can size its
        shared context to the budget that would otherwise have cut it, instead of
        guessing a window and being silently subsampled underneath. Read-only: it
        resolves the budget but predicts nothing and changes no state.

        Args:
            x_train: the full context matrix, for its row and feature counts (the
                feature count is the post-``HighDimFeatureSelector`` one, and detecting
                that needs the values, not just the shape).
            budget_n_features: a previously computed :meth:`budget_n_features` for this
                same table, to skip re-deriving it. ``None`` derives it here.

        Returns:
            Rows per call, at least 10 -- the same floor ``_predict_reg_single`` uses.
        """
        x_train = np.asarray(x_train)
        # The budget is resolved on every call, never cached: it follows `memory_policy`
        # (re-declared per call) or the default derived from the device's total VRAM.
        budget = self._resolve_max_elements_budget()
        if budget_n_features is None:
            budget_n_features = self.budget_n_features(x_train)
        if (len(x_train) + 1) * budget_n_features <= budget:
            return len(x_train)
        return max(10, budget // (2 * budget_n_features))

    def _resolve_max_elements_budget(self) -> int:
        """Per-forward element budget: the policy's value, else the VRAM-aware default.

        Single source of truth shared by the predict (``_predict_reg_single``) and
        embedding (``get_embeddings``) paths, which had duplicated — and drifted on
        — this resolution. Routed through :class:`MemoryPolicy` so
        ``memory_policy={"elements_budget": N}`` and the legacy
        ``SYNTHEFY_MAX_ELEMENTS_BUDGET`` land in the same place; this budget is
        upstream of the cache knobs, since the cached path engages only when the
        query set exceeds the chunk size it implies.
        """
        budget = self._coerced_memory_policy().elements_budget
        if budget is not None:
            return int(budget)
        # Legacy, still supported: SYNTHEFY_MAX_ELEMENTS_BUDGET shipped on main long
        # before this policy existed, is documented in public/README.md, and is how
        # the eval CLI's --max-elements-budget reaches inference
        # (evaluation/cli.py sets it, evaluation/harness.py records it). It is NOT
        # part of the MemoryPolicy surface; prefer memory_policy={"elements_budget": N}.
        env_budget = os.environ.get("SYNTHEFY_MAX_ELEMENTS_BUDGET")
        return int(env_budget) if env_budget else self._default_max_elements_budget()

    #: Set by ``predict`` to the resolved :class:`MemoryPolicy` for the last call (as
    #: ``model_dump()``): which ladder rung ran, the precision chosen, the budgets
    #: used, and any context rows dropped. None until a prediction has run.
    #:
    #: Exists because the fallback ladder is otherwise invisible — a request that
    #: quietly dropped to ``plain_loop``, the one rung that may subsample the context,
    #: is indistinguishable from a fast one except by its numbers.
    memory_report_: dict | None = None

    # Derived from RUNGS (not hand-duplicated) so a rung added there cannot silently
    # fall back to severity -1 here and be treated as never-worse-than-anything.
    _MEMORY_RUNG_SEVERITY = {rung: index for index, rung in enumerate(RUNGS)}

    @staticmethod
    def _fit_row_chunk_attempts(pinned: int | None) -> list[int]:
        """Return the smaller-than-`pinned` retry ladder (the pinned/None value
        itself is never a new attempt: it is already the first cache_attempts entry
        built from the placement ladder)."""
        return [
            chunk for chunk in FIT_ROW_CHUNK_RETRY_LADDER
            if pinned is None or chunk < pinned
        ]

    def _evict_pipe_cache(self, id_pipe) -> None:
        """Drop only this pipe's own retained context-cache entry.

        Used by the per-pipe cached-retry loop's OOM/unsupported handlers: a bad
        attempt for ONE pipe must not evict every OTHER pipe's already-built,
        reusable K/V cache.
        """
        cache = getattr(self, "_context_cache", None)
        if isinstance(cache, dict):
            cache.pop(id_pipe, None)

    def _memory_rung_sort_key(self, policy: MemoryPolicy) -> tuple[int, float]:
        """Comparable "how bad is this outcome" key: rank first, then chunk size.

        A smaller ``context_row_chunk`` is a more degraded outcome within that one
        rung (a harder-won escalation), so it sorts worse than a larger one. Every
        other rung always has ``context_row_chunk is None``, which sorts lowest, so
        rank alone decides ties there.
        """
        rank = self._MEMORY_RUNG_SEVERITY.get(policy.rung or "no_cache", -1)
        chunk = policy.context_row_chunk
        return rank, (-chunk if chunk is not None else float("-inf"))

    def _publish_memory_policy(self, policy: MemoryPolicy) -> None:
        """Publish the worst successful/selected rung seen in this predict call."""
        current = getattr(self, "_memory_outcome_policy", None)
        if current is None or self._memory_rung_sort_key(policy) > self._memory_rung_sort_key(current):
            self._memory_outcome_policy = policy
            current = policy
        report = current.model_dump()
        # Reference, not copy: ``_memory_attempt_history`` is reassigned to a fresh
        # list at the start of every predict() call (never appended-to across
        # calls), so sharing this call's list is safe and avoids an O(attempts^2)
        # copy across a multi-attempt OOM-retry sequence.
        report["attempt_history"] = getattr(self, "_memory_attempt_history", [])
        self.memory_report_ = report

    def _record_memory_attempt(
        self,
        policy: MemoryPolicy,
        *,
        pipeline_ids: list[int],
        path: str,
        rung: str,
        context_row_chunk: int | None,
        outcome: str,
        reason: str,
        dropped_context_rows: int,
    ) -> None:
        """Append one validated attempt and refresh the public report."""
        attempt = MemoryAttempt(
            pipeline_ids=pipeline_ids,
            path=path,
            rung=rung,
            cache_dtype=policy.cache_dtype,
            offload_to_host=policy.offload_to_host,
            context_row_chunk=context_row_chunk,
            outcome=outcome,
            reason=reason,
            dropped_context_rows=dropped_context_rows,
        ).model_dump()
        history = getattr(self, "_memory_attempt_history", None)
        if history is None:
            history = self._memory_attempt_history = []
        history.append(attempt)
        current = getattr(self, "_memory_outcome_policy", None)
        if current is not None:
            self._publish_memory_policy(current)

    def _coerced_memory_policy(self) -> MemoryPolicy:
        """This predictor's :class:`MemoryPolicy`.

        No environment variables feed the policy: it is configured through
        ``NoriPredictor(memory_policy=...)`` alone. The two env vars still honoured are
        pre-existing, shipped, and documented in ``public/README.md`` — the
        ``SYNTHEFY_DISABLE_CACHED_INFERENCE`` / ``SYNTHEFY_ENABLE_CACHED_INFERENCE``
        kill switches, applied here because an operator must be able to turn the
        cached path off in production without a redeploy.

        Returns:
            The still-unresolved policy for this predictor.

        Raises:
            RuntimeError: if ``SYNTHEFY_CACHE_MAX_GB`` is set. That variable shipped
                on main with a *different* meaning (skip the cache above this size,
                measured against the full-precision footprint) and no longer has an
                equivalent. Silently ignoring a memory-safety knob could turn a
                working job into an OOM, so this fails loudly instead.
        """
        if os.environ.get("SYNTHEFY_CACHE_MAX_GB") is not None:
            raise RuntimeError(
                "SYNTHEFY_CACHE_MAX_GB is no longer supported. It used to SKIP the "
                "KV cache when the full-precision footprint exceeded it; the cache "
                "is now OFFLOADED to host RAM instead of skipped, so the old value "
                "does not translate. Use memory_policy={'gpu_budget_frac': 0.4} for a share "
                "of VRAM (portable across GPUs) or "
                "memory_policy={'gpu_budget_absolute_gb': N} for a hard cap on a "
                "co-tenanted GPU. Unset the variable to continue."
            )
        policy = MemoryPolicy.coerce(self.memory_policy)
        disabled = (
            os.environ.get("SYNTHEFY_DISABLE_CACHED_INFERENCE", "0") == "1"
            or os.environ.get("SYNTHEFY_ENABLE_CACHED_INFERENCE", "1") != "1"
        )
        if disabled and policy.cache:
            logger.info("Nori KV cache disabled by environment kill switch")
            # Rebuild rather than model_copy(cache=False): flipping cache off while the
            # caller's cache-only levers are still set is precisely the state rule 1
            # rejects, and model_copy would smuggle it past validation -- the
            # inconsistency this branch removed everywhere else. Carry over only the
            # settings that still mean something without a cache.
            policy = MemoryPolicy(
                cache=False,
                elements_budget=policy.elements_budget,
                allow_subsample=policy.allow_subsample,
            )
        return policy

    def _total_vram_gb(self) -> float | None:
        """Accelerator memory in GiB for this predictor, or None if unknowable.

        None rather than a guess, so :meth:`MemoryPolicy.resolve` applies its own
        documented fallback instead of this method inventing a number.
        """
        try:
            dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            if dev.type == "cuda" and torch.cuda.is_available():
                return torch.cuda.get_device_properties(dev).total_memory / (1024**3)
            if dev.type == "mps" and torch.backends.mps.is_available():
                recommended = getattr(torch.mps, "recommended_max_memory", None)
                if callable(recommended):
                    return recommended() / (1024**3)
        except Exception:
            pass
        return None

    @staticmethod
    def _chunk_size(max_elements: int, budget_n_features: int, n_train: int) -> int:
        """Query rows per forward so ``(n_train + chunk) * budget_n_features``
        stays within ``max_elements``; floored at 256. Shared by predict and
        ``get_embeddings`` so the chunking formula cannot drift between them.
        """
        return max(256, (max_elements // max(budget_n_features, 1)) - n_train)

    def PostProcessInModel(self, feature_pred: torch.tensor, config: dict) -> torch.tensor:
        # Revert preprocess in model forward
        feature_pred = feature_pred / torch.sqrt(
            config["features_per_group"] / config["num_used_features"].to(self.device)
        )
        feature_pred = feature_pred * config["std_for_normalization"] + config["mean_for_normalization"]
        feature_pred = einops.rearrange(feature_pred, "b s f n -> s b (f n)").squeeze(1).float().cpu().numpy()
        if config["n_x_padding"] > 0:
            feature_pred = feature_pred[:, : -config["n_x_padding"]]
        return feature_pred

    def PostProcess(
        self,
        feature_pred: np.ndarray,
        pipeline: List,
        config: dict,
        gt: bool = False,
        source_row_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        # Revert preprocess in the Classifier
        for id_step, step in enumerate(reversed(pipeline)):
            if isinstance(step, FeatureShuffler):
                if step.mode == "shuffle":
                    inv_p = np.argsort(step.feature_indices)
                    feature_pred = feature_pred[:, inv_p]
                else:
                    raise NotImplementedError
            elif isinstance(step, CategoricalFeatureEncoder):
                if step.encoding_strategy != "onehot":
                    if step.category_mappings is not None:
                        categorical_indices = list(step.category_mappings.keys())
                        feature_pred[:, categorical_indices] = np.round(feature_pred[:, categorical_indices])
                    if step.transformer is not None:
                        for idx, p in step.category_mappings.items():
                            feature_pred[:, idx] = np.clip(feature_pred[:, idx], a_min=0, a_max=max(p))
                            inv_p = np.argsort(p)
                            feature_pred[:, idx] = inv_p[feature_pred[:, idx].astype(int)].astype(feature_pred.dtype)
                        inv_col = np.argsort(step.feature_indices)
                        feature_pred = feature_pred[:, inv_col]
                else:
                    if len(step.categorical_features) == 0 or step.transformer is None:
                        continue
                    cont_features_indices = [
                        idx for idx in range(feature_pred.shape[1]) if idx not in step.categorical_features
                    ]

                    assert np.array_equal(step.categorical_features, np.arange(len(step.categorical_features)))
                    start_idx = 0
                    for idx, out_category in enumerate(
                        step.transformer.named_transformers_["one_hot_encoder"].categories_
                    ):
                        assert len(out_category) >= 2
                        if not np.any(np.isnan(out_category)):
                            if len(out_category) == 2:  # e.g. [3, 5.5]
                                feature_pred[:, start_idx] = np.round(
                                    np.clip(feature_pred[:, start_idx], a_min=0, a_max=1)
                                )
                                start_idx += 1
                            else:
                                arr = feature_pred[:, start_idx : start_idx + len(out_category)]
                                feature_pred[:, start_idx : start_idx + len(out_category)] = (
                                    arr == arr.max(axis=1, keepdims=True)
                                ).astype(float)
                                start_idx += len(out_category)
                        else:
                            if len(out_category) == 2:  # e.g. [0, nan]
                                feature_pred[:, start_idx] = 0
                                start_idx += 1
                            else:
                                arr = feature_pred[:, start_idx : start_idx + len(out_category) - 1]
                                feature_pred[:, start_idx : start_idx + len(out_category) - 1] = (
                                    arr == arr.max(axis=1, keepdims=True)
                                ).astype(float)
                                feature_pred[:, start_idx + len(out_category) - 1] = 0
                                start_idx += len(out_category)
                    feature_pred = np.column_stack(
                        [
                            step.transformer.named_transformers_["one_hot_encoder"].inverse_transform(
                                feature_pred[:, step.categorical_features]
                            ),
                            feature_pred[:, cont_features_indices],
                        ]
                    )

            elif isinstance(step, RebalanceFeatureDistribution):
                if step.svd_tag == "svd" and step.svd_n_comp > 0:
                    feature_pred = feature_pred[:, : -step.svd_n_comp]
                if (
                    step.worker_tags[0] in ["quantile_uniform_10", "quantile_uniform_5", "quantile_uniform_all_data"]
                    and step.n_quantile_features > 0
                ):
                    feature_pred = feature_pred[:, : -step.n_quantile_features]
                elif step.worker_tags[0] == "power":
                    raise ValueError(
                        f"Missing value imputation does not currently support the preprocessing method of power!"
                    )
                if step.feature_indices is not None:
                    inv_p = np.argsort(step.feature_indices)
                    feature_pred = feature_pred[:, inv_p]

            elif isinstance(step, FilterValidFeatures):
                invalid_mask = np.asarray(step.invalid_indices, dtype=bool)
                if np.any(invalid_mask):
                    invalid_features = np.asarray(step.invalid_features)
                    if source_row_indices is not None:
                        source_row_indices = np.asarray(source_row_indices, dtype=np.int64)
                        if source_row_indices.shape != (feature_pred.shape[0],):
                            raise ValueError(
                                "FilterValidFeatures inverse row index shape "
                                f"{source_row_indices.shape} does not match reconstructed "
                                f"rows {feature_pred.shape[0]}"
                            )
                        invalid_features = invalid_features[source_row_indices]
                    elif invalid_features.shape[0] != feature_pred.shape[0]:
                        raise ValueError(
                            "FilterValidFeatures inverse needs source_row_indices when "
                            f"reconstructing {feature_pred.shape[0]} rows from stored "
                            f"features for {invalid_features.shape[0]} rows"
                        )

                    valid_mask = ~invalid_mask
                    if feature_pred.shape[1] != int(valid_mask.sum()):
                        raise ValueError(
                            "FilterValidFeatures inverse column mismatch: "
                            f"{feature_pred.shape[1]} reconstructed valid columns for "
                            f"{int(valid_mask.sum())} expected"
                        )
                    if invalid_features.shape[1] != int(invalid_mask.sum()):
                        raise ValueError(
                            "FilterValidFeatures stored invalid-column mismatch: "
                            f"{invalid_features.shape[1]} for {int(invalid_mask.sum())} expected"
                        )
                    restored = np.empty(
                        (feature_pred.shape[0], invalid_mask.shape[0]),
                        dtype=np.result_type(feature_pred.dtype, invalid_features.dtype),
                    )
                    restored[:, valid_mask] = feature_pred
                    restored[:, invalid_mask] = invalid_features
                    feature_pred = restored
        return feature_pred

    @staticmethod
    def _aggregate_feature_reconstruction_chunks(
        chunks: list[np.ndarray],
        n_context: int,
    ) -> np.ndarray:
        """Rebuild one full context+query reconstruction from query chunks.

        Every model call reconstructs the complete context plus only that
        call's query rows. Context rows are therefore averaged across calls,
        while disjoint query blocks are concatenated in request order.
        """
        if not chunks:
            raise ValueError("At least one feature reconstruction chunk is required")
        context = np.stack([chunk[:n_context] for chunk in chunks]).mean(axis=0)
        query = np.concatenate([chunk[n_context:] for chunk in chunks], axis=0)
        return np.concatenate([context, query], axis=0)

    def get_embeddings(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray | None = None,
        data_source: Literal["test", "train"] = "test",
    ) -> np.ndarray:
        """Thin wrapper applying the execution overrides once for the call.

        Entering the context manager per query chunk would walk model.modules()
        on every chunk. Note the feature-decoder skip is inert here: the
        embedding forwards pass return_embeddings=True and return before the
        decode branch, so only native_rms_norm has any effect.
        """
        with self._execution_overrides():
            return self._get_embeddings_impl(x_train, y_train, x_test, data_source)

    def _get_embeddings_impl(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray | None = None,
        data_source: Literal["test", "train"] = "test",
    ) -> np.ndarray:
        """Extract per-row Nori embeddings for a context/query split.

        Runs each preprocessing pipeline (the inference ensemble) through the
        backbone with ``return_embeddings=True`` and collects the final-layer
        target-token representation for the requested rows.

        Args:
            x_train, y_train: context rows the model conditions on.
            x_test: query rows to embed. Required for ``data_source="test"``;
                genuinely ignored for ``data_source="train"`` (may be ``None``) —
                the context embeddings depend only on ``x_train``.
            data_source: ``"test"`` returns one embedding per query row,
                ``"train"`` returns one embedding per context row.

        Returns:
            ``np.ndarray`` of shape ``(n_estimators, n_samples, embed_dim)`` —
            one slice per preprocessing pipeline, never averaged. ``n_samples``
            is ``len(x_test)`` for ``data_source="test"`` and ``len(x_train)``
            for ``data_source="train"``.
        """
        if data_source not in ("test", "train"):
            raise ValueError(f"data_source must be 'test' or 'train', got {data_source!r}.")

        x_train, y_train = self.validate_data(
            x_train,
            y_train,
            reset=True,
            validate_separately=False,
            accept_sparse=False,
            dtype=None,
            ensure_all_finite=False,
        )
        if data_source == "train":
            # Context embeddings ignore the query rows entirely (the train branch
            # below uses a dummy query drawn from the context). Replace whatever
            # the caller passed with a single context row so no feature-count
            # check or full-size query preprocessing applies to unused input —
            # this is what lets x_test be omitted / mismatched for "train".
            x_test = x_train[:1]
        else:
            if x_test is None:
                raise ValueError("get_embeddings requires x_test for data_source='test'.")
            x_test = self.validate_data(
                x_test, reset=False, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False
            )

        x_train_base, x_test_base, categorical_idx = self._prepare_inductive_features(
            x_train,
            x_test,
        )

        n_train = len(y_train)
        n_test = len(x_test)
        # Chunk the query rows along the same element budget the predictor uses,
        # so wide/large tables don't OOM while embedding.
        budget_n_features = self._effective_budget_n_features(x_train.shape[1] if x_train.ndim > 1 else 1, x_train)
        max_elements = self._resolve_max_elements_budget()

        # Context-size guard. Chunking below only splits the QUERY rows; every
        # forward still concatenates the full train context (and data_source=
        # "train" runs one unchunked forward over it). So when the context alone
        # is over budget, chunking cannot prevent an OOM. Unlike predict(), we do
        # NOT subsample the context here: it would silently change which rows
        # condition each embedding (breaking the NoriEmbedding OOF leakage
        # guarantee), and for data_source="train" it would return fewer rows than
        # requested. Raise a clear error instead — anything but OOM. This also
        # honors SYNTHEFY_FORBID_SUBSAMPLE by construction (we never subsample).
        base_elements = (n_train + 1) * budget_n_features
        if base_elements > max_elements:
            raise RuntimeError(
                f"get_embeddings: train context too large for the element budget "
                f"(n_train={n_train}, eff_features={budget_n_features}, "
                f"base_elements={base_elements} > budget={max_elements}). "
                f"Unlike predict(), embedding does not subsample the context "
                f"(it would change embedding semantics). Raise "
                f"SYNTHEFY_MAX_ELEMENTS_BUDGET or reduce the context size to run."
            )
        chunk_size = self._chunk_size(max_elements, budget_n_features, n_train)

        # Run embeddings through the UNWRAPPED backbone (skip torch.compile). The
        # compiled graph is specialized for predict() and guards on the
        # return_embeddings bool (a dynamo constant); calling it with
        # return_embeddings=True fails those guards and triggers a second
        # multi-minute cold compile of a structurally different graph (early
        # return before the decoder) — a stall the predict-only warmup cannot
        # hide. The embedding path early-returns before the decoder, so it gains
        # nothing from the compiled decoder graph; eager is the right default for
        # this offline/OOF path.
        bare_model = self._bare_model()

        per_pipeline = []
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):
            x_train_ = x_train_base.copy()
            x_test_ = x_test_base.copy()
            y_ = y_train.copy()
            categorical_idx_ = categorical_idx.copy()
            for id_step, step in enumerate(pipe):
                x_train_, x_test_, categorical_idx_ = self._fit_transform_step_inductive(
                    step,
                    x_train_,
                    x_test_,
                    categorical_idx_,
                    self.seeds[id_pipe * self.preprocess_num + self._seed_step_index(pipe, id_step)],
                    y_train=y_,
                )

            y_dev = torch.from_numpy(y_).float().to(self.device)
            train_t = torch.from_numpy(x_train_).float().to(self.device)
            # Feature positional embeddings are regenerated randomly on EVERY
            # forward pass (add_embeddings -> randn + orthogonal_). Reseed
            # immediately before each forward (below) so every query chunk — and
            # the data_source="train" forward — is embedded under the SAME random
            # basis. Seeding once here would leave chunks 2+ in an incomparable
            # subspace, adding a spurious chunk-boundary artifact and making
            # transform(X) depend on len(X). Mirrors _predict_reg_single.

            if data_source == "train":
                # Context embeddings are independent of the query rows, so one
                # forward with a single dummy query row suffices.
                x_all = torch.cat([train_t, train_t[:1]], dim=0).unsqueeze(0)
                y_all = torch.cat([y_dev, y_dev[:1]], dim=0).unsqueeze(0)
                bare_model.to(self.device)
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                with (
                    torch.autocast(
                        device_type=self.device.type if isinstance(self.device, torch.device) else self.device,
                        enabled=self.mix_precision,
                    ),
                    torch.inference_mode(),
                ):
                    emb = bare_model(x=x_all, y=y_all, eval_pos=n_train, task_type="reg", return_embeddings=True)
                per_pipeline.append(emb[0, :n_train].float().cpu().numpy())
                continue

            # data_source == "test": embed every query row, chunked.
            test_t = torch.from_numpy(x_test_).float().to(self.device)
            chunks = []
            for i in range(0, n_test, chunk_size):
                end = min(i + chunk_size, n_test)
                x_all = torch.cat([train_t, test_t[i:end]], dim=0).unsqueeze(0)
                y_pad = torch.zeros(end - i, dtype=y_dev.dtype, device=y_dev.device)
                y_all = torch.cat([y_dev, y_pad], dim=0).unsqueeze(0)
                bare_model.to(self.device)
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                with (
                    torch.autocast(
                        device_type=self.device.type if isinstance(self.device, torch.device) else self.device,
                        enabled=self.mix_precision,
                    ),
                    torch.inference_mode(),
                ):
                    emb = bare_model(x=x_all, y=y_all, eval_pos=n_train, task_type="reg", return_embeddings=True)
                chunks.append(emb[0, n_train:].float().cpu().numpy())
            per_pipeline.append(np.concatenate(chunks, axis=0))

        return np.stack(per_pipeline, axis=0)

    def _predict_reg(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        return_distribution: bool = False,
    ) -> np.ndarray:
        if not getattr(self, "inference_with_DDP", False):
            return self._predict_reg_impl(
                x_train,
                y_train,
                x_test,
                return_distribution=return_distribution,
            )

        with DistributedInference(
            model=self.model,
            device=self.device,
            mix_precision=self.mix_precision,
        ) as inference:
            return self._predict_reg_impl(
                x_train,
                y_train,
                x_test,
                return_distribution=return_distribution,
                distributed_inference=inference,
            )

    def _predict_reg_impl(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        return_distribution: bool = False,
        distributed_inference: DistributedInference | None = None,
    ) -> np.ndarray:
        """Regression predict with optional inference-time augmentations.

        Default: single pass on raw y_train.
        If 'yj' in self.augmentations AND |skew(y_train)| > yj_skew_threshold:
            ensemble [identity, Yeo-Johnson] passes.
        Otherwise (skew below threshold): return identity pass only.

        Ablation (96 datasets) showed unconditional YJ was net negative despite
        winning stock_fardamento02 (skew 17.7, +0.077 R²). The skew threshold
        default (10.0) preserves the wins on extreme-skew datasets while
        skipping moderately-skewed ones where YJ harms predictions.

        When return_distribution=True the YJ point ensemble is bypassed: it
        averages point predictions in original-y space and has no
        distribution-level analogue, so the distribution path returns the raw
        single-pass quantile bank.
        """
        if distributed_inference is None:
            predict_single = self._predict_reg_single
        else:

            def predict_single(
                x_train_single,
                y_train_single,
                x_test_single,
                return_distribution=False,
            ):
                return self._predict_reg_single_impl(
                    x_train_single,
                    y_train_single,
                    x_test_single,
                    return_distribution=return_distribution,
                    distributed_inference=distributed_inference,
                )

        if return_distribution:
            return predict_single(
                x_train,
                y_train,
                x_test,
                return_distribution=True,
            )
        base_pred = predict_single(
            x_train,
            y_train,
            x_test,
        )
        if "yj" not in self.augmentations:
            return base_pred

        # Conditional YJ gate: only apply if y_train is sufficiently skewed.
        # Use bias=False for unbiased estimator; robust to NaN via nan_to_num.
        try:
            from scipy import stats as _stats

            y_np = np.asarray(y_train, dtype=np.float64)
            y_np = y_np[np.isfinite(y_np)]
            if len(y_np) < 10:
                return base_pred
            y_skew = float(_stats.skew(y_np, bias=False))
        except Exception:
            return base_pred
        if not np.isfinite(y_skew) or abs(y_skew) < self.yj_skew_threshold:
            # Skew below threshold — YJ not needed / harmful
            return base_pred

        # --- Yeo-Johnson target transform ensemble pass ---
        try:
            from sklearn.preprocessing import PowerTransformer
            import warnings as _warnings

            y_train_arr = np.asarray(y_train, dtype=np.float64).reshape(-1, 1)
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                y_train_yj = pt.fit_transform(y_train_arr).ravel()

            # Predict in YJ-transformed target space
            pred_yj_space = predict_single(
                x_train,
                y_train_yj,
                x_test,
            )
            # base_pred may be torch.Tensor; normalize to numpy for transform
            pred_yj_np = (
                pred_yj_space.detach().cpu().numpy() if torch.is_tensor(pred_yj_space) else np.asarray(pred_yj_space)
            )

            # Clip in-transformed-space before inverse to avoid explosion
            y_tr_min, y_tr_max = y_train_yj.min(), y_train_yj.max()
            clip_range = (y_tr_max - y_tr_min) * 3.0 + 1e-6
            pred_yj_np_clipped = np.clip(pred_yj_np, y_tr_min - clip_range, y_tr_max + clip_range)

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                pred_inv = pt.inverse_transform(pred_yj_np_clipped.reshape(-1, 1)).ravel()

            # Convert base_pred to numpy for averaging
            base_np = base_pred.detach().cpu().numpy() if torch.is_tensor(base_pred) else np.asarray(base_pred)
            base_np = base_np.ravel()
            pred_inv = pred_inv.ravel()

            # Per-row fallback: if inverse produced NaN/inf, use identity pass
            bad = ~np.isfinite(pred_inv)
            if bad.any():
                pred_inv[bad] = base_np[bad]

            ensembled = 0.5 * (base_np + pred_inv)

            # Return same type as base_pred
            if torch.is_tensor(base_pred):
                return torch.as_tensor(ensembled, dtype=base_pred.dtype, device=base_pred.device)
            return ensembled
        except DegradedPipelineWarning:
            # A warning escalated to an exception is a caller DEMANDING to hear
            # about degradation -- it is not a YJ failure to swallow. Warnings are
            # Exceptions, so without this the blanket handler below would catch it
            # and turn strict_pipeline() back into a silent degradation on this path.
            raise
        except Exception as _e:
            print(f"  [YJ] augmentation failed ({type(_e).__name__}: {_e}), falling back to identity-only prediction")
            return base_pred

    def _get_or_build_context(
        self,
        bare_model,
        id_pipe,
        *,
        x_train_t,
        y_train_t,
        cache_dtype,
        offload_kv_cache,
        fit_row_chunk,
        reuse_context_cache,
        cache_entries=1,
    ):
        """Build the per-pipe context (train) K/V cache, reusing it across predict()
        calls whose context and cache params are unchanged.

        ``forward_cached_regression`` already reuses the context K/V across test
        CHUNKS within a single call; this reuses it across separate predict() calls
        too (fit-once / serve-many), so the O(N_train) context forward runs once per
        context instead of once per query batch. It is bit-identical to rebuilding:
        the (random) feature positional embedding is drawn under the per-pipe seed
        the predictor sets before every forward, so a fresh build would draw the
        same one -- the cache only skips recomputing an identical result.

        Reuse is decided by EXACT content verification (``_same_context``), not by a
        digest. A long-lived predictor may be re-fitted with a different table, so a
        false cache hit would answer new query rows against an earlier context. That is
        a wrong-answer bug, not a slow one, so the check may not be probabilistic --
        see ``_same_context`` for why bytes are compared directly instead of using a
        summary hash.

        ``MemoryPolicy(reuse_context_cache=False)`` rebuilds on every call and clears
        any retained bundle. This is a typed policy rather than an environment knob,
        so sklearn cloning, serialized configuration, local inference and serving all
        share one explicit contract.

        ``NoriRegressor(large_context_cache_entries=K)`` retains K DISTINCT contexts per
        pipeline instead of one, most-recently-used first. Ordinary ``predict`` reuses
        a single context and wants the default of 1; a large-context policy rotates between
        several shared pools within one ``predict``
        (:mod:`synthefy_nori.inference.large_context`), and at K=1 each pool evicts the
        previous one, so a policy with 8 pools gets zero hits and pays 8 rebuilds plus
        8 verification clones. Each entry is a full K/V cache, so capacity is bounded
        only by the caller's explicit, memory-aware choice.
        """

        def _build():
            return bare_model.build_context_cache(
                x_train_t,
                y_train_t,
                cache_dtype=cache_dtype,
                offload_kv_cache=offload_kv_cache,
                fit_row_chunk=fit_row_chunk,
            )

        if not reuse_context_cache:
            # A policy can change between calls (the shared engine re-declares one
            # per request). Drop a bundle retained under an earlier policy as soon as
            # reuse turns off, both to prevent a later resurrection and to release
            # another caller's context-derived state from this process.
            cache = getattr(self, "_context_cache", None)
            if cache is not None:
                cache.clear()
            return _build()
        key = self._context_cache_key(cache_dtype, offload_kv_cache, fit_row_chunk)
        cache = getattr(self, "_context_cache", None)
        if cache is None:
            cache = self._context_cache = {}
        # Per pipe: a most-recently-used-first list of entries. A list and a linear
        # scan, not an OrderedDict keyed by a digest, because the hit test is the exact
        # byte compare below -- there is no key to hash that would not reintroduce the
        # false-positive risk _same_context exists to avoid. K is single digits.
        #
        # Read with .get, and insert the list only once there is a bundle to put in it:
        # an empty list per pipe would make `self._context_cache` truthy, which is how
        # callers and tests ask "is anything retained?" -- including after a build that
        # raised, where the answer must stay no.
        entries = cache.get(id_pipe) or []
        capacity = max(1, int(cache_entries))
        for position, (hit_key, hit_x, hit_y, hit_bundle) in enumerate(entries):
            # Cheap scalar key first, then the O(N_train*F) byte compare -- a param
            # change short-circuits without touching the tensors.
            if hit_key == key and self._same_context(hit_x, x_train_t) and self._same_context(hit_y, y_train_t):
                if position:  # promote: this pool is in active rotation
                    entries.insert(0, entries.pop(position))
                # Honour a capacity shrink on the hit path too. Returning early here is
                # how a drop from 3 to 1 used to retain all three K/V bundles for as
                # long as the caller kept asking for a context already in the list --
                # exactly the case where the caller is trying to free memory. Promotion
                # above runs first, so the entry being returned always survives.
                del entries[capacity:]
                return hit_bundle
        bundle = _build()  # cache only on a successful build
        entries = cache.setdefault(id_pipe, entries)
        # Keep our OWN contiguous copies to verify future calls against. Clones, not
        # the caller's tensors: x_train_t is a slice of the concatenated train+test
        # table, so retaining the view would pin the (unbounded) query rows alive
        # too. One N_train x F copy is negligible against the per-layer K/V cache it
        # guards, which is ~nlayers x n_groups times larger.
        #
        # BOUND THE CACHE BY TOTAL BYTES, across every pipe and every retained entry.
        # It is keyed per PIPE, and each pipe may retain up to `cache_entries` distinct
        # contexts (see the K-pool docstring above), so the default regression path's
        # 8-member preprocessing ensemble times a large-N policy's K pools can retain
        # many bundles at once. Each is ~nlayers x n_groups x n_ctx: small on a normal
        # table, very large on a WIDE one. Measured on TALENT's topo_2_1 (7108 x 266 ->
        # ~134 feature groups at features_per_group=2) one bundle is 15.2 GiB, so eight
        # is 121.6 GiB and a 143 GiB H200 is exhausted before any query runs. The memory
        # ladder then OOMs on every rung and even the plain-loop fallback dies -- though
        # it needs only 14.5 GiB by itself -- because `torch.cuda.empty_cache()` cannot
        # release tensors this dict still references.
        #
        # Evicting to a BYTE budget rather than a fixed count keeps the fit-once /
        # serve-many amortization intact in the common case (bundles all fit) and sheds
        # bundles only when they are individually huge, which is precisely when
        # retaining them is what breaks the run.
        if self._evict_context_cache_for(cache, bundle, keep_pipe=id_pipe):
            entries.insert(
                0,
                (
                    key,
                    x_train_t.detach().clone(memory_format=torch.contiguous_format),
                    y_train_t.detach().clone(memory_format=torch.contiguous_format),
                    bundle,
                ),
            )
            # Evict from the cold end. Shrinking capacity between calls (the shared
            # engine re-declares its policy per request) is honoured here rather than
            # needing its own path: the list is trimmed to whatever the CURRENT call
            # asked for.
            del entries[capacity:]
        return bundle

    @staticmethod
    def _bundle_device_bytes(bundle) -> int:
        """Best-effort GPU byte count for a context bundle.

        Walks the bundle's tensors rather than trusting a declared size: the cache may
        hold int8-quantized or host-offloaded K/V (``ScalableSeqKV``) whose real
        footprint differs from the bf16 estimate the policy reports. Host-resident
        tensors are excluded -- they are not what exhausts VRAM. Anything unrecognised
        counts as 0 rather than raising: mis-measuring a bundle must never break a
        prediction.
        """
        total = 0
        seen: set[int] = set()

        def walk(obj, depth: int = 0) -> None:
            nonlocal total
            if depth > 6 or id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, torch.Tensor):
                if obj.is_cuda:
                    total += obj.element_size() * obj.nelement()
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    walk(v, depth + 1)
            elif isinstance(obj, (list, tuple, set)):
                for v in obj:
                    walk(v, depth + 1)
            elif hasattr(obj, "__dict__"):
                for v in vars(obj).values():
                    walk(v, depth + 1)

        try:
            walk(bundle)
        except Exception:  # never let accounting break a predict
            return 0
        return total

    def _context_cache_budget_bytes(self) -> "int | None":
        """Bytes of retained context cache to allow, or ``None`` when no limit applies.

        ``None`` means this device has no measurable accelerator memory to exhaust -- CPU or
        an unknown backend. CUDA uses total device memory; Apple MPS uses Metal's recommended
        maximum working-set size.

        A CUDA device gets a share of total VRAM rather than an absolute, matching how every
        other budget in ``memory_policy`` is expressed, so one setting ports from a 24 GB
        laptop card to a 143 GB H200. Deliberately a MINORITY of the device: the cache is an
        optimization, and leaving the majority free for the forward pass is what stops a
        retained cache from turning a slow path into a fatal one.
        """
        frac = float(getattr(self, "_context_cache_budget_frac", 0.25))
        try:
            dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            if dev.type == "cuda" and torch.cuda.is_available():
                return int(torch.cuda.get_device_properties(dev).total_memory * frac)
            if dev.type == "mps" and torch.backends.mps.is_available():
                recommended = getattr(torch.mps, "recommended_max_memory", None)
                if callable(recommended):
                    return int(recommended() * frac)
        except Exception:
            pass
        return None  # cannot measure -> leave behaviour unchanged

    def _evict_context_cache_for(self, cache: dict, incoming, *, keep_pipe=None) -> bool:
        """Drop cached bundles (oldest first, across every pipe) until ``incoming``
        fits the byte budget.

        ``cache`` maps ``id_pipe -> [(key, x, y, bundle), ...]``, most-recently-used
        first per pipe (see ``_get_or_build_context``'s K-pool docstring), so this
        walks every entry in every pipe's list rather than treating ``cache`` itself
        as one bundle per key.

        ``keep_pipe`` is the id_pipe the caller is about to insert ``incoming`` into:
        its list is drained like any other's if that is what eviction needs, but the
        now-empty list is never popped from ``cache`` here. The caller pre-registers
        that (possibly still-empty) list via ``cache.setdefault`` before calling this,
        and inserts into it right after -- popping the key out from under that insert
        would silently orphan the new bundle onto a list ``cache`` no longer holds.

        Returns whether ``incoming`` should be retained. False means the bundle is
        still built and returned to the caller -- it simply is not kept for a later
        call. That is the wide-table case: reuse is worth nothing if holding one
        bundle leaves no room to run the model.
        """
        budget = self._context_cache_budget_bytes()
        if budget is None:
            return True  # no device limit -> exactly the old behaviour
        incoming_bytes = self._bundle_device_bytes(incoming)
        if incoming_bytes > budget:
            # The branch that fixes topo_2_1: a 15.2 GiB bundle against a 0.25 x 143 GiB
            # budget is kept out, so eight of them can never accumulate. Release whatever
            # is held so the forward pass gets the whole card.
            cache.clear()
            return False
        total = sum(self._bundle_device_bytes(entry[3]) for entries in cache.values() for entry in entries)
        # Dict preserves pipe insertion order, and each per-pipe list is
        # most-recently-used-first (see _get_or_build_context), so popping from the
        # END of the first pipe's list, then the next pipe's, evicts oldest-first
        # without tracking a single cross-pipe timeline.
        for id_pipe in list(cache.keys()):
            entries = cache[id_pipe]
            while entries and total + incoming_bytes > budget:
                evicted = entries.pop()
                total -= self._bundle_device_bytes(evicted[3])
            if not entries and id_pipe != keep_pipe:
                cache.pop(id_pipe, None)
            if total + incoming_bytes <= budget:
                break
        return True

    def _context_cache_key(self, cache_dtype, offload_kv_cache, fit_row_chunk) -> tuple:
        """The non-tensor half of the cache key: everything OUTSIDE the context table
        that changes what ``build_context_cache`` produces.

        ``cache_dtype``/``offload_kv_cache``/``fit_row_chunk`` come from the resolved
        :class:`MemoryPolicy`, so changing ``memory_policy=`` between calls (which a
        server does per request) invalidates the bundle. ``self.seed`` is in here
        because the bit-identity argument for reusing a bundle rests on the rebuild
        drawing the SAME random feature positional embedding, which only holds while
        the seed the predictor reseeds with is unchanged.
        """
        return (str(cache_dtype), bool(offload_kv_cache), fit_row_chunk, self.seed)

    @staticmethod
    def _same_context(cached: torch.Tensor, candidate: torch.Tensor) -> bool:
        """Exact sameness test between a cached context tensor and a candidate.

        Compares raw BIT PATTERNS rather than values or a digest, because both
        alternatives are wrong here:

        - ``torch.equal`` uses float equality, and NaN != NaN. Missing values are
          ordinary in a Nori context, so a value compare would miss on every
          NaN-bearing table and rebuild forever -- silently deleting the speedup.
        - a summary digest (shape/NaN-count/sum/sum-of-squares/endpoints) collides on
          trivially constructed inputs: any permutation of the interior preserves all
          of them, so two genuinely different contexts hash alike. On a false hit the
          cached bundle answers the new query rows, i.e. predictions computed against
          the WRONG context, returned as if correct. A cryptographic digest would fix
          the collision odds but still costs a full device->host copy per call, which
          a byte compare on-device does not.

        Bitwise identity has neither failure mode: NaN payloads match themselves, and
        there are no false positives at all. Its false NEGATIVES (+0.0 vs -0.0, or two
        distinct NaN payloads) merely rebuild something we could have reused, which
        costs time and never correctness.

        The ``uint8`` view works for every dtype -- 1-byte elements always divide the
        last dimension evenly -- so one path covers float32/float64/float16/bfloat16
        and the integer dtypes alike. Shape, dtype and device are checked separately
        so that two tables of equal byte length but different layout cannot match.
        """
        if cached.shape != candidate.shape or cached.dtype != candidate.dtype or cached.device != candidate.device:
            return False
        if cached.numel() == 0:
            return True
        return bool(
            torch.equal(
                cached.contiguous().view(torch.uint8),
                candidate.detach().contiguous().view(torch.uint8),
            )
        )

    @staticmethod
    def _group_ordinary_pipes(prepared: list[tuple]) -> list[list[tuple]]:
        """Stable groups whose context/query tensors have identical shapes.

        The group containing the final pipeline executes last. Each model call
        reseeds first, so this preserves the legacy loop's post-call RNG state
        when different transformed widths consume different amounts of RNG.
        """
        grouped = {}
        for item in prepared:
            _, x_train, y_train, x_test = item
            key = (
                x_train.shape, y_train.shape, x_test.shape,
                x_train.dtype, y_train.dtype, x_test.dtype,
                x_train.device, y_train.device, x_test.device,
            )
            grouped.setdefault(key, []).append(item)
        result = []
        for group in grouped.values():
            width = group[0][1].shape[-1]
            batch_size = (
                PIPELINE_BATCH_MAX
                if width <= PIPELINE_BATCH_MAX_FEATURES
                else 1
            )
            result.extend(
                group[start:start + batch_size]
                for start in range(0, len(group), batch_size)
            )
        if result:
            final_pipe_id = max(item[0] for item in prepared)
            final_group_index = next(
                index
                for index, group in enumerate(result)
                if any(item[0] == final_pipe_id for item in group)
            )
            result.append(result.pop(final_group_index))
        return result

    def _clear_pipeline_batch_contexts(
            self, keep: set[tuple] | None = None) -> None:
        """Release grouped caches that cannot serve the current execution mode."""
        cache = getattr(self, "_context_cache", None)
        if not isinstance(cache, dict):
            return
        keep = keep or set()
        for slot in list(cache):
            if (
                isinstance(slot, tuple)
                and len(slot) == 2
                and slot[0] == "pipeline_batch"
                and slot not in keep
            ):
                cache.pop(slot, None)

    def _clear_pipeline_member_contexts(self, batch_slots: set[tuple]) -> None:
        """Release B=1 caches superseded by retained grouped cache slots."""
        cache = getattr(self, "_context_cache", None)
        if not isinstance(cache, dict):
            return
        grouped_ids = {
            id_pipe
            for _, ids in batch_slots
            for id_pipe in ids
        }
        for id_pipe in grouped_ids:
            cache.pop(id_pipe, None)

    @staticmethod
    def _model_supports_pipeline_batching(bare_model) -> bool:
        """Whether one RNG seed is equivalent to reseeding every B=1 member."""
        if bool(getattr(bare_model, "training", False)):
            return False
        modules = bare_model.modules() if hasattr(bare_model, "modules") else (bare_model,)
        for module in modules:
            dropout = getattr(module, "dropout", 0.0)
            if isinstance(dropout, (int, float)) and float(dropout) != 0.0:
                # MultiheadAttention passes its numeric dropout directly to SDPA,
                # which applies it even when the enclosing module is in eval mode.
                return False
        return True

    def _try_batched_ordinary_regression(
            self, bare_model, *, x_train_base, x_test_base, y_train,
            categorical_idx, n_samples_train, n_samples_test,
            budget_n_features, max_elements_budget, dropped_context_rows,
    ) -> list[torch.Tensor] | None:
        """Execute prepared ordinary members in bounded same-shape groups.

        This deliberately covers only the two full-precision, on-device execution
        modes: an ordinary model forward (``no_cache``) and a resident bf16 context
        cache.
        Retrieval, DDP, imputation, int8, host offload, pinned prefill chunking, and
        every OOM/unsupported case use the established loop below. The environment
        kill switch gives operators a no-code rollback while this optimization gains
        production mileage. Batching preserves the model math, although mixed-
        precision GEMMs may reassociate at the last few bits versus B=1 execution.
        """
        try:
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        except (TypeError, RuntimeError):
            self._clear_pipeline_batch_contexts()
            return None
        if (
            device.type != "cuda"
            or os.environ.get("SYNTHEFY_DISABLE_PIPELINE_BATCHING", "0") == "1"
            or len(self.preprocess_pipelines) < 2
            or self.inference_with_DDP
            or self.mask_prediction
            or bool(getattr(bare_model, "mask_prediction", False))
            or not self._model_supports_pipeline_batching(bare_model)
            or any(config["retrieval_config"]["use_retrieval"] for config in self.inference_config)
        ):
            self._clear_pipeline_batch_contexts()
            return None

        chunk_size = self._chunk_size(
            max_elements_budget, budget_n_features, n_samples_train)
        requested = self._coerced_memory_policy()
        cache_eligible = (
            hasattr(bare_model, "forward_cached_regression")
            and n_samples_test > chunk_size
            and requested.cache
        )
        if cache_eligible:
            fpg = max(int(getattr(bare_model, "features_per_group", 2)), 1)
            embed_dim = int(getattr(bare_model, "embed_dim", 128))
            nlayers = int(getattr(bare_model, "nlayers", 16))
            nhead = max(int(getattr(bare_model, "nhead", 2)), 1)
            bytes_per = 2 if self.mix_precision else 4
            policy = requested.resolve(
                est_cache_gb=estimate_cache_gb(
                    n_context_rows=n_samples_train,
                    n_groups=(budget_n_features + fpg - 1) // fpg,
                    nlayers=nlayers,
                    embed_dim=embed_dim,
                    bytes_per_element=bytes_per,
                ),
                bytes_per_element=bytes_per,
                head_dim=max(embed_dim // nhead, 1),
                total_vram_gb=self._total_vram_gb(),
                total_ram_gb=total_host_ram_gb(),
            )
        else:
            policy = requested.resolve(
                est_cache_gb=0.0, bytes_per_element=1, head_dim=1,
                cache_eligible=False,
            )
        if (
            policy.rung not in ("no_cache", "resident_bf16")
            or policy.cache_dtype != "bf16"
            or policy.offload_to_host
            or policy.context_row_chunk is not None
        ):
            self._clear_pipeline_batch_contexts()
            return None

        prepared = []
        retained_host_bytes = 0
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):
            x_train_ = x_train_base.copy()
            x_test_ = x_test_base.copy()
            y_ = y_train.copy()
            categorical_idx_ = categorical_idx.copy()
            # Internal bails out here on retrieval steps (InferenceAttentionMap,
            # SubSampleData), which reshape the context between pipelines and so
            # break the identical-shape grouping this path depends on. Neither
            # class exists in this tier's inference_method, so no pipeline can
            # contain one and the check would be dead code.
            for id_step, step in enumerate(pipe):
                x_train_, x_test_, categorical_idx_ = self._fit_transform_step_inductive(
                    step, x_train_, x_test_, categorical_idx_,
                    self.seeds[id_pipe * self.preprocess_num + self._seed_step_index(pipe, id_step)],
                    y_train=y_,
                )
            prospective_host_bytes = (
                x_train_.size + x_test_.size + y_.size
            ) * np.dtype(np.float32).itemsize
            if (
                retained_host_bytes + prospective_host_bytes
                > PIPELINE_BATCH_HOST_BUDGET_BYTES
            ):
                self._clear_pipeline_batch_contexts()
                return None
            x_all = torch.from_numpy(
                np.concatenate([x_train_, x_test_], axis=0)
            ).float()
            y_tensor = torch.from_numpy(y_).float()
            prepared.append((
                id_pipe,
                x_all[:len(y_)],
                y_tensor,
                x_all[len(y_):],
            ))
            retained_host_bytes += (
                x_all.numel() * x_all.element_size()
                + y_tensor.numel() * y_tensor.element_size()
            )

        groups = self._group_ordinary_pipes(prepared)
        if policy.rung == "resident_bf16":
            # Resolve again from the actual post-transform widths and the total
            # retained ensemble cache. The earlier lower-bound resolution uses the
            # raw/effective input width only to avoid preprocessing modes that could
            # never stay resident; it must not under-report a wider prepared group.
            group_cache_sizes = [
                estimate_cache_gb(
                    n_context_rows=n_samples_train,
                    n_groups=(group[0][1].shape[-1] + fpg - 1) // fpg,
                    nlayers=nlayers,
                    embed_dim=embed_dim,
                    bytes_per_element=bytes_per,
                ) * len(group)
                for group in groups
            ]
            batched_cache_gb = (
                sum(group_cache_sizes)
                if requested.reuse_context_cache
                else max(group_cache_sizes)
            )
            policy = requested.resolve(
                est_cache_gb=batched_cache_gb,
                bytes_per_element=bytes_per,
                head_dim=max(embed_dim // nhead, 1),
                total_vram_gb=self._total_vram_gb(),
                total_ram_gb=total_host_ram_gb(),
            )
            if policy.rung != "resident_bf16":
                self._clear_pipeline_batch_contexts()
                return None

        active_batch_slots = {
            ("pipeline_batch", tuple(item[0] for item in group))
            for group in groups
            if len(group) > 1
        } if policy.rung == "resident_bf16" and policy.reuse_context_cache else set()
        self._clear_pipeline_batch_contexts(keep=active_batch_slots)
        self._clear_pipeline_member_contexts(active_batch_slots)

        self._publish_memory_policy(policy)
        outputs: list[torch.Tensor | None] = [None] * len(prepared)
        ids: tuple[int, ...] = ()
        try:
            for group in groups:
                ids = tuple(item[0] for item in group)
                x_train_t = torch.stack([item[1] for item in group]).to(self.device)
                y_train_t = torch.stack([item[2] for item in group]).to(self.device)
                x_test_t = torch.stack([item[3] for item in group]).to(self.device)
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
                self.model.to(self.device)
                with torch.autocast(device_type=device.type, enabled=self.mix_precision), torch.inference_mode():
                    if policy.rung == "resident_bf16":
                        slot = ids[0] if len(ids) == 1 else ("pipeline_batch", ids)
                        context = self._get_or_build_context(
                            bare_model, slot,
                            x_train_t=x_train_t, y_train_t=y_train_t,
                            cache_dtype="bf16", offload_kv_cache=False,
                            fit_row_chunk=None,
                            reuse_context_cache=policy.reuse_context_cache,
                        )
                        try:
                            group_output = self._unwrap_model_output(
                                bare_model.apply_context_cache(
                                    x_test_t, context, row_chunk_size=chunk_size,
                                    adaptive_query_chunk=policy.adaptive_query_chunk,
                                ),
                                task_type="reg",
                            )
                        finally:
                            if not policy.reuse_context_cache:
                                # Without reuse only one group's full K/V bundle is
                                # budgeted. Drop it before the next group is built.
                                del context
                    else:
                        chunks = []
                        for start in range(0, n_samples_test, chunk_size):
                            end = min(start + chunk_size, n_samples_test)
                            x_in = torch.cat([x_train_t, x_test_t[:, start:end]], dim=1)
                            y_in = torch.cat([
                                y_train_t,
                                y_train_t.new_zeros(len(group), end - start),
                            ], dim=1)
                            chunks.append(self._unwrap_model_output(
                                self.model(x=x_in, y=y_in, eval_pos=n_samples_train, task_type="reg"),
                                task_type="reg",
                            ))
                        group_output = torch.cat(chunks, dim=1)
                if (
                    not isinstance(group_output, torch.Tensor)
                    or group_output.ndim < 2
                    or group_output.shape[0] != len(group)
                    or group_output.shape[1] != n_samples_test
                ):
                    raise NotImplementedError(
                        "pipeline-batched model output must be [B, n_test, ...]"
                    )
                group_output = self._reject_nonfinite_output(
                    group_output, path=f"pipeline-batched ({policy.rung})",
                    n_train=n_samples_train, n_test=n_samples_test,
                )
                for (id_pipe, *_), member_output in zip(group, group_output.unbind(0)):
                    outputs[id_pipe] = member_output
                self._record_memory_attempt(
                    policy,
                    pipeline_ids=list(ids),
                    path="pipeline_batch",
                    rung=policy.rung,
                    context_row_chunk=None,
                    outcome="success",
                    reason="resolved",
                    dropped_context_rows=dropped_context_rows,
                )
        except (NotImplementedError, torch.cuda.OutOfMemoryError) as exc:
            self._record_memory_attempt(
                policy,
                pipeline_ids=list(ids),
                path="pipeline_batch",
                rung=policy.rung,
                context_row_chunk=None,
                outcome=("oom" if isinstance(exc, torch.cuda.OutOfMemoryError)
                         else "unsupported"),
                reason="resolved",
                dropped_context_rows=dropped_context_rows,
            )
            cache = getattr(self, "_context_cache", None)
            if isinstance(cache, dict):
                # The failed attempt may already have built singleton groups.
                # Legacy retry must start without any partial grouped-run state.
                cache.clear()
            torch.cuda.empty_cache()
            return None

        used = policy.escalated(
            policy.rung,
            dropped_context_rows=dropped_context_rows,
            **({"query_chunk": chunk_size} if policy.rung == "resident_bf16" else {}),
        )
        self._publish_memory_policy(used)
        self._log_once_per_call(
            "rung", logging.INFO,
            f"Nori serving-memory rung: {used.describe()}",
        )
        if any(output is None for output in outputs):
            raise AssertionError("pipeline batching lost an ensemble member")
        return [output for output in outputs if output is not None]


    def _predict_reg_single(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        return_distribution: bool = False,
    ) -> np.ndarray:
        if not getattr(self, "inference_with_DDP", False):
            return self._predict_reg_single_impl(
                x_train,
                y_train,
                x_test,
                return_distribution=return_distribution,
            )

        with DistributedInference(
            model=self.model,
            device=self.device,
            mix_precision=self.mix_precision,
        ) as inference:
            return self._predict_reg_single_impl(
                x_train,
                y_train,
                x_test,
                return_distribution=return_distribution,
                distributed_inference=inference,
            )

    def _predict_reg_single_impl(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        return_distribution: bool = False,
        distributed_inference: DistributedInference | None = None,
    ) -> np.ndarray:
        # Check size constraints to avoid OOM
        n_features = x_train.shape[1] if x_train.ndim > 1 else 1
        n_samples_train = x_train.shape[0]
        n_samples_test = x_test.shape[0]

        # If the number of elements is too large, we must chunk the test set to avoid OOM.
        # The default is VRAM-aware: anchored at 2M elements for a ~24GB GPU (the
        # historical conservative value) and scaled linearly with total VRAM, so a
        # large GPU (A100/H100/H200) uses big tables' full context instead of
        # silently subsampling, without manual tuning. SYNTHEFY_MAX_ELEMENTS_BUDGET
        # still overrides explicitly when set.
        MAX_ELEMENTS_BUDGET = self._resolve_max_elements_budget()

        # Calculate elements for one full forward pass (train + 1 test row at
        # minimum). Use the post-HighDimFeatureSelector feature count — the
        # model never sees the raw count when the config reduces dimensionality.
        budget_n_features = self._effective_budget_n_features(n_features, x_train)
        base_elements = (n_samples_train + 1) * budget_n_features
        dropped_context_rows = 0
        if base_elements > MAX_ELEMENTS_BUDGET:
            # Guard: SYNTHEFY_FORBID_SUBSAMPLE=1 makes context subsampling FAIL LOUDLY
            # instead of silently shrinking the training set. Use for full-context
            # evaluation where any subsampling must be visible, never silent.
            _forbid_env = os.environ.get("SYNTHEFY_FORBID_SUBSAMPLE") == "1"
            if not self._coerced_memory_policy().allow_subsample or _forbid_env:
                # Name whichever setting actually forbade it. Reporting the env var to
                # someone who set memory_policy={"allow_subsample": False} sends them hunting
                # a variable they never set.
                _source = "SYNTHEFY_FORBID_SUBSAMPLE=1" if _forbid_env else "memory_policy={'allow_subsample': False}"
                _remedy = "Raise memory_policy={'elements_budget': N} for full context, or allow subsampling."
                raise ContextTooLargeError(
                    f"{_source}: context subsampling required but forbidden "
                    f"(n_train={n_samples_train}, eff_features={budget_n_features}, "
                    f"base_elements={base_elements} > budget={MAX_ELEMENTS_BUDGET}). "
                    f"{_remedy}"
                )
            # If even the train set + 1 row is too big, we must subsample the training set
            max_train_samples = max(10, MAX_ELEMENTS_BUDGET // (2 * budget_n_features))
            # Not forbidden, so we proceed — but never SILENTLY: warn so a trimmed
            # context is always visible. Raise SYNTHEFY_MAX_ELEMENTS_BUDGET to keep
            # full context, or set SYNTHEFY_FORBID_SUBSAMPLE=1 to make this an error.
            # Never SILENTLY: both warn and log, with the exact row counts, so a
            # trimmed context is visible whichever channel the caller watches. The
            # count is also carried into memory_report_["dropped_context_rows"]
            # below, so it survives past the warning.
            dropped_context_rows = n_samples_train - max_train_samples
            _subsample_msg = (
                f"Nori: context subsampled {n_samples_train} -> {max_train_samples} rows "
                f"({dropped_context_rows} dropped) to fit an element budget of "
                f"{MAX_ELEMENTS_BUDGET} (base_elements={base_elements}, "
                f"eff_features={budget_n_features}). Raise "
                f"memory_policy={{'elements_budget': N}} for full context, or set "
                f"memory_policy={{'allow_subsample': False}} to make this an error."
            )
            self._log_once_per_call("subsample", logging.WARNING, _subsample_msg)
            # Same category tree as the SVD fallback: one escalation forbids every
            # fallback that hands the model less than the config promised.
            self._warn_once_per_call("subsample", _subsample_msg, ContextSubsampledWarning)
            # Randomly subsample the training data
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(n_samples_train, max_train_samples, replace=False)
            x_train = x_train[idx]
            y_train = y_train[idx]
            n_samples_train = len(x_train)

        np_rng = np.random.default_rng(self.seed)

        x_train, y_train = self.validate_data(
            x_train,
            y_train,
            reset=True,
            validate_separately=False,
            accept_sparse=False,
            dtype=None,
            ensure_all_finite=False,
        )
        x_test = self.validate_data(
            x_test, reset=False, validate_separately=False, accept_sparse=False, dtype=None, ensure_all_finite=False
        )

        x_train_base, x_test_base, categorical_idx = self._prepare_inductive_features(
            x_train,
            x_test,
        )

        # The batched path calls the bare model directly, so it cannot honour a
        # DistributedInference session. Internal has no DDP inference and so no
        # equivalent guard; skip the fast path rather than silently dropping the
        # caller's distributed context.
        if distributed_inference is None:
            bare_model = self._bare_model()
            batched_outputs = self._try_batched_ordinary_regression(
                bare_model,
                x_train_base=x_train_base,
                x_test_base=x_test_base,
                y_train=y_train,
                categorical_idx=categorical_idx,
                n_samples_train=n_samples_train,
                n_samples_test=n_samples_test,
                budget_n_features=budget_n_features,
                max_elements_budget=MAX_ELEMENTS_BUDGET,
                dropped_context_rows=dropped_context_rows,
            )
            if batched_outputs is not None:
                output = torch.stack(batched_outputs).mean(dim=0)
                if return_distribution:
                    return output
                return self._collapse_regression_output(output)


        outputs = []
        mask_predictions = []
        for id_pipe, pipe in enumerate(self.preprocess_pipelines):
            pipeline_mask_chunks = []
            x_train_ = x_train_base.copy()
            x_test_ = x_test_base.copy()
            y_ = y_train.copy()
            categorical_idx_ = categorical_idx.copy()
            for id_step, step in enumerate(pipe):
                x_train_, x_test_, categorical_idx_ = self._fit_transform_step_inductive(
                    step,
                    x_train_,
                    x_test_,
                    categorical_idx_,
                    self.seeds[id_pipe * self.preprocess_num + self._seed_step_index(pipe, id_step)],
                    y_train=y_,
                )

            x_ = np.concatenate([x_train_, x_test_], axis=0)
            x_ = torch.from_numpy(x_[:, :]).float().to(self.device)
            y_ = torch.from_numpy(y_).float().to(self.device)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            if self.inference_with_DDP:
                assert distributed_inference is not None
                output = distributed_inference.inference(
                    x_[: len(y_train)],
                    y_,
                    x_[len(y_train) :],
                )
                outputs.append(output)
            if not self.inference_with_DDP:
                # Calculate max allowed test samples per batch to avoid OOM.
                # We need: (n_train + chunk_size) * budget_n_features <= MAX_ELEMENTS_BUDGET.
                # Use the post-HighDimFeatureSelector feature count — the model
                # never sees the raw count when the config reduces dimensionality.
                chunk_size = self._chunk_size(MAX_ELEMENTS_BUDGET, budget_n_features, n_samples_train)

                # --- Cached train-KV fast path (ON by default) ---
                # Numerically equivalent to the chunked path below (cache==chunked,
                # R2-identical; verified |dR2|=0 on GPU): it only engages when the
                # standard path is ALREADY chunking (n_test > chunk_size), and reuses
                # the train-side sequence K/V projection across all test chunks instead
                # of recomputing it per chunk. No accuracy change; ~2-3x faster on
                # multi-chunk (large test) inference, scaling with the chunk count.
                # Disable with SYNTHEFY_ENABLE_CACHED_INFERENCE=0 or the
                # SYNTHEFY_DISABLE_CACHED_INFERENCE=1 kill switch.
                # Unwrap the DDP wrapper so the cached-KV fast path (an
                # attribute on the backbone) is reachable.
                bare_model = self._bare_model()
                # Gate on the MODEL's mask_prediction, not the predictor's flag.
                # forward_cached_regression hard-requires mask_prediction=False;
                # NoriPredictor.mask_prediction defaults to False but a training-
                # style checkpoint builds the model with mask_prediction=True, so
                # keying off self.mask_prediction wrongly enters the cached path
                # and the model raises NotImplementedError under chunking. Use the
                # model's real flag so such ckpts fall to the plain chunked loop.
                model_mask_pred = bool(getattr(bare_model, "mask_prediction", self.mask_prediction))
                # Serving-memory policy. The ladder and the reasoning behind its
                # ORDER live in ``synthefy_nori.inference.memory_policy``; here we
                # only supply the measurements it needs and record what it picked.
                #
                # By default the cache stays bit-exact while it fits VRAM,
                # quantizes to int8 only to keep it resident, and offloads to host
                # only when it cannot be resident at any precision. So a table that
                # serves correctly today keeps bit-exact predictions: accuracy is
                # spent only to avoid a fallback that would otherwise be slower or
                # fatal.
                #
                # Configure via NoriPredictor(memory_policy=...) — omit it for the
                # defaults, or pass a preset name ("exact", "max_context", "off"), a
                # dict, or a MemoryPolicy. No env var configures this; only the
                # SYNTHEFY_DISABLE_CACHED_INFERENCE kill switch is honoured.
                requested_policy = self._coerced_memory_policy()
                policy = requested_policy
                use_cached = (
                    hasattr(bare_model, "forward_cached_regression")
                    and not self.mask_prediction
                    and n_samples_test > chunk_size
                    and policy.cache
                )
                if use_cached:
                    fpg = max(int(getattr(bare_model, "features_per_group", 2)), 1)
                    n_groups = (x_train_.shape[-1] + fpg - 1) // fpg
                    embed_dim = int(getattr(bare_model, "embed_dim", 128))
                    nlayers = int(getattr(bare_model, "nlayers", 16))
                    nhead = max(int(getattr(bare_model, "nhead", 2)), 1)
                    bytes_per = 2 if self.mix_precision else 4
                    head_dim = max(embed_dim // nhead, 1)
                    policy = policy.resolve(
                        est_cache_gb=estimate_cache_gb(
                            n_context_rows=n_samples_train,
                            n_groups=n_groups,
                            nlayers=nlayers,
                            embed_dim=embed_dim,
                            bytes_per_element=bytes_per,
                        ),
                        bytes_per_element=bytes_per,
                        head_dim=head_dim,
                        total_vram_gb=self._total_vram_gb(),
                        total_ram_gb=total_host_ram_gb(),
                    )
                    use_cached = policy.cache
                else:
                    policy = policy.resolve(
                        est_cache_gb=0.0,
                        bytes_per_element=1,
                        head_dim=1,
                        cache_eligible=False,
                    )
                self._publish_memory_policy(policy)
                # Announce the opening rung. WARNING when it is already a fallback
                # (offload / plain loop), because that is a real slowdown the caller
                # should see by default; INFO on the fast rungs so a normal run stays
                # quiet but is still explainable after the fact.
                self._log_once_per_call(
                    "rung",
                    logging.WARNING if policy.is_degraded else logging.INFO,
                    f"Nori serving-memory rung: {policy.describe()}",
                )

                cached_done = False
                fallback_reason = "resolved"
                if use_cached:
                    self.model.to(self.device)
                    # Walk every allowed, budget-feasible precision/placement rung,
                    # then bound the final rung's prefill working set at 2048, 1024,
                    # and 512 rows before giving up on the cache. An explicit chunk is
                    # the first-attempt cap; no retry can exceed it.
                    #
                    # Chunking is tried LAST, combined only with the worst placement,
                    # not first against the bit-exact rung: it only bounds the
                    # transient BUILD peak, not the finished cache's resident size,
                    # so it cannot rescue an OOM caused by the cache itself being too
                    # big to fit -- only the placement ladder above can. It also is
                    # not supported on every checkpoint (serial sequence attention
                    # raises NotImplementedError). RUNGS already ranks
                    # context_row_chunk below offload_int8, so this matches that.
                    pinned = requested_policy.context_row_chunk
                    placement_policies = requested_policy.runtime_fallbacks(
                        policy,
                        bytes_per_element=bytes_per,
                        head_dim=head_dim,
                    )
                    cache_attempts = [
                        (candidate, pinned) for candidate in placement_policies
                    ]
                    final_policy = placement_policies[-1]
                    for retry_chunk in self._fit_row_chunk_attempts(pinned):
                        cache_attempts.append((final_policy, retry_chunk))
                    fallback_reason = "fallback_after_oom"
                    for attempt_index, (attempt_policy, attempt_fit_chunk) in enumerate(cache_attempts):
                        policy = attempt_policy
                        attempt_reason = "resolved" if attempt_index == 0 else "oom_retry"
                        ctx_bundle = None
                        output = None
                        try:
                            with (
                                torch.autocast(
                                    device_type=self.device.type
                                    if isinstance(self.device, torch.device)
                                    else self.device,
                                    enabled=self.mix_precision,
                                ),
                                torch.inference_mode(),
                            ):
                                # Cross-call context amortization: build the O(N_train)
                                # per-layer K/V cache once and reuse it across predict()
                                # calls whose context (x_train/y_train) is unchanged --
                                # the fit-once / serve-many pattern -- instead of
                                # rebuilding it per query batch. This splits the former
                                # forward_cached_regression(full) into its build+apply
                                # halves (bit-identical: that method is now exactly this
                                # pair) and memoizes the build. See _get_or_build_context.
                                n_ctx = len(y_train)
                                ctx_bundle = self._get_or_build_context(
                                    bare_model,
                                    id_pipe,
                                    x_train_t=x_[:n_ctx].unsqueeze(0),
                                    y_train_t=y_[:n_ctx].unsqueeze(0),
                                    cache_dtype=policy.cache_dtype,
                                    offload_kv_cache=policy.offload_to_host,
                                    fit_row_chunk=attempt_fit_chunk,
                                    reuse_context_cache=policy.reuse_context_cache,
                                    cache_entries=self.context_cache_entries,
                                )
                                output = bare_model.apply_context_cache(
                                    x_[n_ctx:].unsqueeze(0),
                                    ctx_bundle,
                                    row_chunk_size=chunk_size,
                                    adaptive_query_chunk=policy.adaptive_query_chunk,
                                )
                            output = self._unwrap_model_output(output, task_type="reg").squeeze(0)
                            output = self._reject_nonfinite_output(
                                output, path=f"cached ({policy.rung})", n_train=n_samples_train, n_test=n_samples_test
                            )
                            outputs.append(output)
                            cached_done = True
                            self._record_memory_attempt(
                                attempt_policy,
                                pipeline_ids=[id_pipe],
                                path="cached",
                                rung=attempt_policy.rung,
                                context_row_chunk=attempt_fit_chunk,
                                outcome="success",
                                reason=attempt_reason,
                                dropped_context_rows=dropped_context_rows,
                            )
                            if attempt_policy.cache_dtype == "int8":
                                # Unlike offload (moves bytes unchanged) or chunking
                                # (bit-exact by design), quantizing the cache is a
                                # real fidelity loss -- the DegradedPipelineWarning
                                # family exists so a scored caller (strict_pipeline())
                                # can catch exactly this, not just see it in the logs.
                                self._warn_once_per_call(
                                    "cache_quantized",
                                    "Nori quantized the context cache to int8: "
                                    f"{attempt_policy.describe()}. Predictions carry "
                                    "a small numerical difference from bf16 (|dR2| ~ "
                                    "6e-6 measured). Raise memory_policy={'gpu_budget_frac': "
                                    "...} / 'host_budget_frac' / 'elements_budget' for more "
                                    "headroom, or set memory_policy={'allow_quantization': "
                                    "False} to keep every rung bit-exact (offloads sooner "
                                    "instead).",
                                    CacheQuantizedWarning,
                                )
                            if (
                                attempt_fit_chunk is not None
                                and (pinned is None or attempt_fit_chunk < pinned)
                            ):
                                policy = attempt_policy.escalated(
                                    "context_row_chunk",
                                    context_row_chunk=attempt_fit_chunk)
                            else:
                                policy = attempt_policy
                            if attempt_index > 0:
                                # Recovered after at least one earlier attempt failed --
                                # whether via a smaller chunk (above) or precision/
                                # placement alone. Log it either way, or a lossy/slow
                                # recovery that never touched context_row_chunk is
                                # invisible in the logs.
                                logger.warning(
                                    "Nori recovered on rung %s", policy.describe())
                            policy = policy.escalated(
                                policy.rung, dropped_context_rows=dropped_context_rows,
                                query_chunk=chunk_size)
                            self._publish_memory_policy(policy)
                            break
                        except NotImplementedError as exc:
                            self._record_memory_attempt(
                                attempt_policy,
                                pipeline_ids=[id_pipe],
                                path="cached",
                                rung=attempt_policy.rung,
                                context_row_chunk=attempt_fit_chunk,
                                outcome="unsupported",
                                reason=attempt_reason,
                                dropped_context_rows=dropped_context_rows,
                            )
                            fallback_reason = "fallback_after_unsupported"
                            # This checkpoint cannot do context-row chunking (serial
                            # sequence attention). If the CALLER pinned it, that is
                            # their error and it propagates. If we chose it ourselves
                            # as an OOM escalation, degrade quietly to the next rung.
                            self._evict_pipe_cache(id_pipe)
                            del ctx_bundle, output
                            if pinned is not None:
                                raise
                            logger.warning(
                                "Nori cannot escalate to context_row_chunk on this "
                                "checkpoint (%s); falling back to the plain loop",
                                exc,
                            )
                            cached_done = False
                            break
                        except torch.cuda.OutOfMemoryError:
                            self._record_memory_attempt(
                                attempt_policy,
                                pipeline_ids=[id_pipe],
                                path="cached",
                                rung=attempt_policy.rung,
                                context_row_chunk=attempt_fit_chunk,
                                outcome="oom",
                                reason=attempt_reason,
                                dropped_context_rows=dropped_context_rows,
                            )
                            self._evict_pipe_cache(id_pipe)
                            del ctx_bundle, output
                            torch.cuda.empty_cache()
                            cached_done = False
                            # Name the OOM and what happens next, or the escalation is
                            # invisible in the logs and a slow run looks inexplicable.
                            if attempt_index + 1 < len(cache_attempts):
                                next_policy, next_chunk = cache_attempts[attempt_index + 1]
                                next_step = (
                                    f"retrying rung {next_policy.rung} "
                                    f"with context_row_chunk={next_chunk}"
                                )
                            else:
                                next_step = "falling back to the plain chunked loop"
                            logger.warning(
                                "Nori OOM on rung %s (context_row_chunk=%s); %s",
                                policy.rung, attempt_fit_chunk, next_step,
                            )
                if not cached_done:
                    # Every cached rung failed, or the policy never chose one. This
                    # is the plain chunked loop: much slower, and the only rung that
                    # may have to subsample the context. Say so — a silent drop here
                    # reads as an unexplained accuracy regression to whoever is
                    # looking at the numbers later.
                    if policy.rung != "no_cache":
                        policy = policy.escalated("plain_loop", dropped_context_rows=dropped_context_rows)
                        msg = (
                            f"Nori fell back to the plain chunked loop: "
                            f"{policy.describe()}. Every query chunk now recomputes "
                            f"the context K/V, several times slower"
                            + (
                                f"; {dropped_context_rows} context rows were DROPPED to fit"
                                if dropped_context_rows
                                else ""
                            )
                            + ". Raise memory_policy={'gpu_budget_frac': ...} / "
                            "'host_budget_frac' / 'elements_budget', or read "
                            "predictor.memory_report_ for what was chosen."
                        )
                        self._log_once_per_call("plain_loop", logging.WARNING, msg)
                        self._warn_once_per_call("plain_loop", msg, RuntimeWarning)
                    else:
                        policy = policy.escalated(
                            policy.rung, dropped_context_rows=dropped_context_rows)
                    self._publish_memory_policy(policy)

                # Chunk the test data (skipped entirely if the cached path ran)
                try:
                    all_outputs = []
                    for i in ([] if cached_done else range(0, n_samples_test, chunk_size)):
                        end_idx = min(i + chunk_size, n_samples_test)
                        x_chunk_test = x_[len(y_train) + i : len(y_train) + end_idx]
                    
                        # Recombine train + test chunk
                        x_chunk_combined = torch.cat([x_[:len(y_train)], x_chunk_test], dim=0)
                    
                        # Create dummy y for test chunk
                        y_chunk_test = torch.zeros(end_idx - i, dtype=y_.dtype, device=y_.device)
                        y_chunk_combined = torch.cat([y_[:len(y_train)], y_chunk_test], dim=0)

                        self.model.to(self.device)
                        with torch.autocast(device_type=self.device.type if isinstance(self.device, torch.device) else self.device, enabled=self.mix_precision), torch.inference_mode():
                            x_in = x_chunk_combined.unsqueeze(0)
                            y_in = y_chunk_combined.unsqueeze(0)

                            chunk_output = self.model(
                                x=x_in, y=y_in, eval_pos=len(y_train), task_type='reg'
                            )

                        if self.mask_prediction:
                            process_config = chunk_output['process_config']
                            chunk_output_feature_pred = self.PostProcessInModel(chunk_output['feature_pred'], process_config)
                            source_row_indices = np.concatenate((
                                np.arange(len(y_train), dtype=np.int64),
                                np.arange(
                                    len(y_train) + i,
                                    len(y_train) + end_idx,
                                    dtype=np.int64,
                                ),
                            ))
                            chunk_output_feature_pred = self.PostProcess(
                                chunk_output_feature_pred,
                                pipe,
                                process_config,
                                source_row_indices=source_row_indices,
                            )
                            pipeline_mask_chunks.append(chunk_output_feature_pred)
                            chunk_output = chunk_output['reg_output']

                        chunk_output = self._unwrap_model_output(
                            chunk_output, task_type="reg"
                        ).squeeze(0)
                        chunk_output = self._reject_nonfinite_output(
                            chunk_output, path="plain chunked loop",
                            n_train=n_samples_train, n_test=end_idx - i)
                        all_outputs.append(chunk_output)

                    # Concatenate all test chunks (cached path already appended above)
                    if not cached_done:
                        output = torch.cat(all_outputs, dim=0)
                        outputs.append(output)
                    if pipeline_mask_chunks:
                        mask_predictions.append(
                            self._aggregate_feature_reconstruction_chunks(
                                pipeline_mask_chunks,
                                len(y_train),
                            )
                        )
                    if not cached_done:
                        self._record_memory_attempt(
                            policy,
                            pipeline_ids=[id_pipe],
                            path="plain_loop",
                            rung=policy.rung,
                            context_row_chunk=None,
                            outcome="success",
                            reason=fallback_reason,
                            dropped_context_rows=dropped_context_rows,
                        )
                except torch.cuda.OutOfMemoryError:
                    self._record_memory_attempt(
                        policy,
                        pipeline_ids=[id_pipe],
                        path="plain_loop",
                        rung=policy.rung,
                        context_row_chunk=None,
                        outcome="oom",
                        reason=fallback_reason,
                        dropped_context_rows=dropped_context_rows,
                    )
                    self._evict_pipe_cache(id_pipe)
                    torch.cuda.empty_cache()
                    raise
            
        output = torch.stack(outputs).mean(dim=0)
        if return_distribution:
            # Return the ensemble-averaged decoder output BEFORE collapse: the
            # per-row quantile bank [n_test, num_reg_quantiles] (pinball head) or
            # bar-distribution logits [n_test, num_bars]. mask_prediction is not
            # combined with the distribution path.
            return output
        output = self._collapse_regression_output(output)
        mask_prediction = np.stack(mask_predictions).mean(axis=0) if mask_predictions != [] else None

        if self.mask_prediction:
            return output, mask_prediction
        else:
            return output
