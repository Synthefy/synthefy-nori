from dataclasses import dataclass, field
from importlib.resources import files


def package_config_path(filename: str) -> str:
    return str(files("synthefy_tabular.configs").joinpath(filename))


@dataclass
class TrainingConfig:
    # Optimizer
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_adjust_lr_fn: str = "match_rms_adamw"
    # Route 2D embedding tables (encoder_x numeric-mlp sign/exp embeddings,
    # cls_y_encoder y_embedding, etc.) to Muon instead of AdamW. Safe on
    # vanilla embeddings; gated off by default for backward compat.
    muon_include_embeddings: bool = False
    # Route higher-rank (3D/4D) attention weights (qkv_proj_weight,
    # out_proj_weight) to Muon by reshape-to-2D. Requires MuonND backend
    # because torch.optim.Muon rejects ndim != 2. Covers the bulk of
    # attention parameters, which are the largest tensors in the model.
    muon_include_nd: bool = False

    # Schedule
    warmup_steps: int = 2000
    decay_start_step: int = 0  # 0 = decay starts right after warmup (old behavior)
    total_steps: int = 200_000
    run_steps: int | None = None  # optional per-invocation optimizer-step cap
    lr_min: float = 1e-5

    # Batch
    batch_size: int = 8
    gradient_accumulation: int = 1

    # Data generation ranges
    min_samples: int = 50
    max_samples: int = 2000
    min_features: int = 2
    max_features: int = 250
    min_classes: int = 2
    max_classes: int = 10

    # Dimension sampling bias: exponent applied to normalized log-uniform [0,1]
    # before mapping to [min, max]. >1.0 biases toward larger tables.
    # 1.0 = standard log-uniform (equal mass per order of magnitude)
    # 1.5 = ~40% of steps use n_samples > 1000 (vs 19% at 1.0)
    dim_bias_samples: float = 1.5
    dim_bias_features: float = 1.3

    # OOM protection: skip if n_samples * n_features exceeds this budget
    max_sample_feature_budget: int = 200_000

    # CCMM masking
    mask_ratio_min: float = 0.1
    mask_ratio_max: float = 0.4

    # Context feature masking: lower ratio keeps normalization stats stable
    # while still teaching the model to handle missing context features.
    context_mask_ratio_min: float = 0.01
    context_mask_ratio_max: float = 0.15

    # Loss
    feature_loss_weight: float = 0.5
    feature_loss_weight_end: float | None = None
    feature_loss_decay_start_step: int = 0
    feature_loss_decay_end_step: int = 0

    # Reproducibility
    seed: int = 42

    # Hardware
    device: str = "cuda:2"
    mixed_precision: bool = True
    gradient_clip: float = 1.0

    # Logging
    log_interval: int = 100
    save_interval: int = 5000
    checkpoint_dir: str = "./checkpoints"
    use_wandb: bool = True
    wandb_project: str = "limix-training"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None
    wandb_tags: tuple[str, ...] = field(default_factory=tuple)
    ema_decay: float = 0.0

    # Compilation
    compile: bool = False

    # Distributed
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

    # Model
    checkpoint_path: str = "cache/LimiX-2M.ckpt"
    features_per_group: int = 2
    target_aware_init_scale: float = 1.0
    target_aware_warmup_steps: int = 0

    # Model v2 architecture (SwiGLU, RMSNorm, pre-norm+DeepNorm, PBLD)
    model_v2: bool = False

    # Synthetic data v2 augmentations
    synth_v2: bool = True

    # Synthetic data v3 augmentations (true categoricals, feature interactions,
    # skewed regression targets)
    synth_v3: bool = True

    # Regression-specific synth_v3 sub-flags (allow independent control)
    rich_reg_targets: bool = True  # multi-feature deps + interaction terms in y
    scale_variation: bool = True   # random y scale (0.37x to 2.72x)

    # TabICL prior generator (alternative data source from TabICL's prior system)
    tabicl_prior: bool = False   # Enable TabICL MLP/Tree SCM prior
    tabicl_prior_prob: float = 0.5  # Per-dataset probability of using TabICL prior

    # Synthetic data v4 augmentations (TabICLv2-inspired diversity improvements)
    synth_v4: bool = False
    v4_filter: bool = True         # ExtraTrees filtering (reject unlearnable data)
    learnability_filter: bool = False  # Standalone ExtraTrees filter (independent of synth_v4)
    learnability_filter_cls_min_score: float = 0.60
    learnability_filter_cls_margin: float = 0.10
    learnability_filter_reg_min_score: float = 0.10
    icl_filter_model: str = ''       # Path to frozen LimiX checkpoint for ICL-based filtering
    icl_filter_cls_min_auc: float = 0.55
    icl_filter_reg_min_r2: float = 0.05
    icl_scaling_filter: bool = False  # Second gate: does more context help?
    icl_scaling_min_improvement: float = 0.03  # Min R² improvement from small to large context
    v4_no_edge_noise: bool = True  # skip Gaussian edge noise (TabICLv2 found no benefit)

    # Synthetic data v5: structural SCM improvements
    #   - Informative categoricals (vs random noise)
    #   - Concat-then-transform aggregation (multivariate edge functions)
    #   - Multi-dimensional SCM nodes (natural Bayes error)
    synth_v5: bool = False
    synth_v5_mixture: bool = False  # mixture prior: 25% v4, 35% v5c, 40% v5a per dataset

    # Task-type specialization: 'both' (default 50/50), 'cls', or 'reg'
    task_type: str = 'both'

    # Regression training improvements
    regression_ratio: float = 0.5  # fraction of 'both' steps that are regression
    regression_loss: str = 'mse'  # 'mse', 'smooth_l1', 'huber', 'pinball', 'bar_distribution'
    regression_loss_beta: float = 1.0  # beta for smooth_l1/huber (ignored for mse)
    regression_quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    # Tail-weighted pinball: upweight extreme quantiles to fix right/left-tail
    # underestimation on bounded-skewed targets (sulfur, Goodreads).
    # 0.0 = uniform (default). 1.0 = near-2x weight at extremes.
    pinball_tail_weight: float = 0.0
    # Quantile monotonicity penalty weight. Adds relu(q[i] - q[i+1])^2 over
    # adjacent τ to discourage quantile crossing. Improves τ-mean smoothness
    # at inference (helps smooth large-n calibration: houses, space_ga,
    # physiochemical_protein). 0.05 is a gentle starting value.
    pinball_monotonicity_weight: float = 0.0
    # Auxiliary MSE on the τ-mean point estimate. Forces the average of the
    # K predicted quantiles to be unbiased w.r.t. target. Targets the
    # "prediction compression" failure mode (QSAR-TID-11): without this term,
    # extreme-target rows are pulled toward the bulk because the τ-mean of
    # an asymmetric quantile distribution is biased. 0.1 = MSE contributes
    # ~10% of pinball gradient at typical magnitudes.
    pinball_mse_weight: float = 0.0
    # Bar distribution: discrete CE over uniformly-spaced bins on
    # context-normalized y. Only used when regression_loss='bar_distribution'.
    # With 5000 bins over [-10, 10], bin width = 0.004 std units — fine enough
    # for continuous prediction while giving explicit mass to extreme bins
    # (unlike pinball which must extrapolate continuously at the tails).
    num_bars: int = 5000
    bar_borders_low: float = -10.0
    bar_borders_high: float = 10.0
    # Soft CE width (in bin units) for bar_distribution loss. 0.0 = hard CE
    # (one-hot target). >0 = Gaussian-smoothed target with this sigma —
    # gives partial credit for predicting nearby bins, smooths the loss
    # landscape that's otherwise 5000 categorical cliffs.
    bar_target_sigma: float = 0.0
    reg_prior_prob: float = 0.4  # fraction of reg episodes using prior generator
    reg_denoise: bool = False  # reduce noise for regression episodes
    reg_deterministic_prob: float = 0.20  # probability a regression-prior episode is zero-noise
    reg_dense: bool = False  # dense-signal regression mode: flat importances,
                             # fewer noise features, more parents, higher R²

    # Probabilistic classification labelers (non-percentile class boundaries)
    # When True, ~50% of cls episodes use logistic/tree/Gaussian/threshold labelers
    # instead of quantile bucketing. Removes a known memorization fingerprint.
    probabilistic_labels: bool = False

    # Nominal categorical generation (non-ordinal categories with random effects)
    # When True, ~40% of categorical columns use nominal categories (independent
    # or crossed) instead of ordinal bins derived from continuous SCM values.
    nominal_categoricals: bool = False

    # Enhanced missingness patterns (row-level, block, target-dependent, missing-as-category)
    enhanced_missingness: bool = False

    # Clean low-dim regime: probability of generating simple 5-30 feature episodes
    # with high categorical fraction and low noise. Covers Amazon-like tabular tasks.
    clean_lowdim_prob: float = 0.0

    # Tree-ensemble prior: probability of generating tree-based episodes per step.
    # Produces piecewise-constant targets from random decision tree ensembles.
    tree_prior_prob: float = 0.0

    # Categorical lookup prior: probability of entity-lookup episodes per step.
    # Produces data where y = f(entity_id) + noise, covering repeated-entity tasks.
    lookup_prior_prob: float = 0.0

    # Quadratic response surface prior: y = x^T M x + w^T x + b
    quadratic_surface_prob: float = 0.0

    # Sparse nonlinear prior: 3-15 relevant features out of many, nonlinear target
    sparse_nonlinear_prob: float = 0.0

    # GP smooth function prior: y sampled from Gaussian Process with RBF/Matern kernels
    # via Random Fourier Features. Produces smooth joint multivariate functions.
    gp_prior_prob: float = 0.0

    # Context missingness augmentation: probability of injecting random NaN cells
    # across the full feature matrix (including context rows). Teaches the model
    # to handle missing values in train rows at inference time.
    context_missingness_prob: float = 0.0

    # Realistic augmentation: post-generation transforms that make synthetic data
    # look more like real benchmark datasets. Applies:
    #   1. Heavy-tailed target transforms (exp, power, log-normal)
    #   2. Correlated feature group injection
    #   3. Skewed feature distributions (log-normal, squared, sigmoid)
    realistic_augmentation_prob: float = 0.0

    # Realistic y-target transforms: post-generation transforms on y only.
    # Simulates: integer counts (stock_fardamento02, colleges), censored values
    # (boston MEDV cap at 50, MIP-2016 timeout), ordinal ratings (sensory 11
    # unique levels), bounded rating averages (Goodreads 0-5 quarter-step),
    # zero-inflated counts (socmob rates).
    y_transform_prob: float = 0.0

    # Cap injection: clip top/bottom tail of y to a single quantile value.
    # Simulates real-world censoring (boston MEDV capped at 50, MIP-2016 timeout
    # saturation, rating caps). Much narrower than y_transform — no rounding,
    # no value transformation, just creates ties at quantile caps.
    cap_injection_prob: float = 0.0

    # Heavy-tail regression y priors (gated). Applies continuous heavy-tail
    # transforms to y: log-normal (exp), Pareto-tailed additive noise, stronger
    # outlier injection. No rounding. Targets LimiX-2M strict wins:
    # stock_fardamento02 (skew 17.7), CPS1988, sulfur, Food_Delivery_Time.
    heavy_tail_prior_prob: float = 0.0

    # Pareto feature-importance regime: when triggered for a regression-prior
    # episode, the per-feature beta/coefficient vector is multiplied by a
    # Pareto-distributed importance vector so a few features dominate while
    # many are weak distractors. Targets the high-dim "many distractors,
    # sparse signal" failure (QSAR-TID-11, Allstate). 0.25 is a reasonable
    # frequency — about 1 in 4 prior episodes shows the pattern.
    pareto_importance_prob: float = 0.0

    # Latent-factor target prior: y = f(X @ V) where V is a d×k random
    # projection with k ≪ d. Models datasets where many features share a
    # low-dim latent structure (drug binding fingerprints, multi-task
    # embeddings, demographic indices). Adds a 10th branch to the regression
    # prior dispatch; selected with this probability among prior episodes.
    latent_factor_prob: float = 0.0

    # High-fraction censored y: extends cap_injection to support timeout
    # patterns where 20–50% of values sit at the cap (MIP-2016 runtime
    # timeout, SAT11). When this fires (gated independently from
    # cap_injection_prob within an episode), the cap percentile is drawn
    # from [50, 80] instead of the default [80, 97].
    high_cap_prob: float = 0.0

    # Low-unique y: round/bin y to 5–15 unique levels before final
    # normalization. Models discrete-regression datasets (Wine_Quality
    # 6 levels, sensory 1–10 ratings). Per-episode gate with this
    # probability.
    low_unique_y_prob: float = 0.0

    # Run one eval before training starts (before optimizer step 0). Useful
    # for fine-tune runs to establish a baseline on the eval dataset.
    eval_at_step_0: bool = False

    # Optional mined synthetic quality rules (from eval_synthetic_benchmark.py).
    # If provided, generated episodes that violate rules are rejected/regenerated.
    quality_filter_rules_path: str | None = None
    quality_filter_max_retries: int = 3

    # Fixed size override (skip random sampling, use one shape for all steps)
    fixed_n_samples: int | None = None
    fixed_n_features: int | None = None

    # Context/query split
    context_ratio_min: float = 0.3
    context_ratio_max: float = 0.8

    # Async data prefetching
    prefetch_workers: int = 4  # number of background data generation processes (0 = disabled)
    prefetch_count: int = 4    # number of batches to prefetch ahead

    # Evaluation
    eval_enabled: bool = True
    eval_interval: int = 5000
    eval_cls_data_dir: str = "cache/tabarena_cls"
    eval_reg_data_dir: str = "cache/tabarena_reg"
    eval_cls_config: str = field(
        default_factory=lambda: package_config_path("cls_default_noretrieval.json")
    )
    eval_reg_config: str = field(
        default_factory=lambda: package_config_path("reg_default_noretrieval.json")
    )

    # Early stopping (measured in validation runs, not optimizer steps)
    early_stop_patience_evals: int = 0
    early_stop_metric: str = "combined"  # 'combined', 'mean_auc', or 'mean_r2'
    early_stop_min_delta: float = 0.0
    early_stop_min_evals: int = 0
