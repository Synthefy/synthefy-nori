# Data preparation for Nori

Nori's public estimator takes a **numeric matrix** and a **finite numeric
target**: `fit()` casts `X` to float32 and `y` to float64 up front. What that
means in practice:

## Features (X)

- **Encode categoricals yourself.** Strings/categories must become numbers
  before `fit`. Ordinal-encode (`sklearn.preprocessing.OrdinalEncoder`) is the
  cheap default and works well — Nori's inference stack treats low-cardinality
  numeric columns as ordinal categories internally. One-hot is fine for a
  handful of levels; avoid exploding hundreds of columns from one
  high-cardinality feature (target/frequency encoding is the better move
  there).
- **Missing values are allowed.** NaN in `X` passes validation and is treated
  as missing by the model — you don't have to impute. ±inf is converted to NaN
  internally. Prefer leaving genuinely-missing cells as NaN over blanket
  zero-filling (0 is a value, NaN is "unknown").
- **Don't scale features.** The inference pipeline normalizes and transforms
  features internally (that's what the bundled inference config does). Feeding
  pre-standardized features is harmless but pointless.
- **Feature count:** tens of features is the sweet spot. With hundreds,
  select or project first (see `feature_selection` in
  `references/interpretability.md`, or a domain-driven cut).
- **Engineer richer features, then sweep them.** Domain-informed combinations
  of raw columns often beat the raw columns: differences of related
  quantities (process − ambient temperature), physical products (torque ×
  rotational speed = power), ratios, and load×wear interactions. Build a few
  candidate feature sets, score each under the one fixed evaluation protocol
  (`references/evaluation.md`), and keep a richer set only when it wins by
  more than the fold noise — iterate until gains stop.

## Target (y)

- **Must be finite.** NaN/inf in `y` corrupts the stored normalization stats
  (`y_mean_`, `y_std_`) without an error — drop or impute target-missing rows
  before `fit`.
- **Skewed targets are handled — up to a point.** The default
  `augmentations=("yj",)` applies a Yeo-Johnson transform when target skew
  exceeds `yj_skew_threshold` (10.0). For heavily skewed positive targets
  (revenue, counts, durations) it is still often worth fitting on
  `np.log1p(y)` and inverting predictions with `np.expm1` — A/B it. If you
  transform `y` yourself, remember quantile outputs come back on the
  transformed scale and each quantile can be inverted through the monotone
  inverse directly (`np.expm1(q)`), unlike the mean.
- **Point estimate choice:** `output_type="median"` is the robust default for
  skewed targets; `"mean"` minimizes squared error for symmetric ones.

## Context size (rows)

All compute happens in `predict`, attending from each query row to the whole
stored context, so cost grows with `n_context × n_query`:

- Up to a few thousand context rows: CPU is fine.
- Tens of thousands: use a GPU (`device="cuda:0"`) and/or **subsample the
  context** — a random few-thousand-row sample usually loses little accuracy;
  stratify the sample by target quantile if the target is skewed.
- Batch big query sets rather than predicting one row at a time — per-call
  overhead dominates tiny queries.

## Splits

Use ordinary sklearn tooling — `train_test_split` for a quick holdout,
`KFold` for evaluation. Because `fit` is instant (no training), cross-validation
costs only `k` `predict` calls, so prefer CV over a single split when the
dataset is small. Seed everything (`random_state=`) so runs are reproducible.
