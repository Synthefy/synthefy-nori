# `synthefy_nori.explainability`

Turn a trained Nori model into an **auditable glass-box** with no accuracy tax:

1. **Attribute importance** to the raw input features of a fitted `NoriRegressor`
   (permutation or Shapley — both measure the *deployed* model, through all of
   Nori's preprocessing).
2. **Select** the fewest top features that retain **≥95 % of Nori's skill** (`n95`).
3. **Distill** an interpretable **EBM** (Explainable Boosting Machine / GA²M) on
   exactly those features. On high-dimensional data this typically needs only
   ~5–10 % of the features and *matches or beats* the full-feature EBM.

Mirrors the layout of `synthefy_nori.interpretability` — a thin, well-tested
adapter over `interpret` (EBM) and, for the SHAP path, over
`synthefy_nori.interpretability.shapiq`. No bespoke attribution math.

## Install

```bash
pip install "synthefy-nori[explainability]"
# for method="shap" also add the interpretability extra (shapiq):
pip install "synthefy-nori[explainability,interpretability]"
```

## One-call API — `NoriInterpreter`

A scikit-learn-style estimator that does all three steps in sequence. Pass the
**full table**; `fit` makes an internal 70/30 split, then runs importance →
pruning → glass-box EBM and stores every artifact on the fitted object.

```python
from synthefy_nori.explainability import NoriInterpreter

interp = NoriInterpreter().fit(X, y)        # X, y = the full table (task auto-detected)

interp.feature_importances_    # np.ndarray, per input column
interp.importance_ranking_     # [{feature, index, importance}], most-important first
interp.selected_features_      # the pruned feature set (kept ≥95% of Nori's skill)
interp.ebm_                    # the fitted glass-box model (shippable)
interp.ebm_model_             # serialized shape functions / intercept / term importances
interp.summary()              # {task, metric, n_selected, nori/ebm scores, top_features}
interp.model_figure_          # the glass-box model diagram, rendered during fit()
interp.plot_model()           # re-draw it (e.g. plot_model(out_path=..., target_name=...))
interp.predict(X_new)         # score with the glass-box on the selected features
```

`fit` renders the model diagram once and stores it on `model_figure_`
(a matplotlib `Figure`); save it with `interp.model_figure_.savefig("model.png")`.
Pass `target_name="…"` for the label on the output node, or `render_figure=False`
to skip rendering (avoids importing matplotlib).

Key params: `model="nori-6m"`, `test_size=0.3`, `use_test=True` (measure importance on the
held-out test split — the post-hoc reading; see the note below), `reduce_threshold=16` (only prune
when `d > 16`), `retain=0.95`. Fitted scores: `nori_full_score_`,
`nori_selected_score_`, `ebm_score_`, `ebm_full_score_`.

## Run it end-to-end

Bundled scikit-learn demo (zero setup — downloads nothing for `diabetes` / `breast_cancer`):

```bash
python -m synthefy_nori.explainability.pipeline --demo diabetes        # regression (R²)
python -m synthefy_nori.explainability.pipeline --demo breast_cancer   # classification (ROC-AUC)
```

Your own data:

```bash
# npz with arrays Xtr, ytr, Xte, yte (+ optional feature_names)
python -m synthefy_nori.explainability.pipeline --npz mydata.npz --tag mydata

# a single CSV with a target column (auto train/test split, auto ordinal-encoding)
python -m synthefy_nori.explainability.pipeline --csv data.csv --target price
```

Each run writes to `--out-dir` (default `explainability_out/`):

- **`<tag>.json`** — ranked importance scores, `n95`/`pct95`, Nori vs EBM skill
  (full and at `n95`), the selection sweep, and the **EBM model itself**
  (intercept, per-term importances, and every feature's shape function).
- **`<tag>.ebm.joblib`** — the fitted EBM object + its feature indices/names, for
  exact reuse:

  ```python
  import joblib
  bundle = joblib.load("explainability_out/diabetes.ebm.joblib")
  bundle["model"].predict(X[:, bundle["feature_indices"]])
  ```

## Worked example — credit default

`examples/explainability_credit.py` runs the whole flow on the UCI *Default of
Credit Card Clients* dataset (30k rows, 23 features, binary) via a single
`NoriInterpreter().fit(X, y)`. It **downloads the dataset fresh every run** via
`ucimlrepo` (no local path) and writes `credit.json` + `credit.ebm.joblib` plus
two figures (importance bars, and the glass-box model diagram from
`interp.model_figure_`).

```bash
pip install "synthefy-nori[explainability]" ucimlrepo
python examples/explainability_credit.py                              # full run + figures
python examples/explainability_credit.py --nori-model nori-30m --out-dir /tmp/credit
```

## Library API

```python
from synthefy_nori import NoriRegressor
from synthefy_nori.explainability import NoriInterpreter
from synthefy_nori.explainability.importance import nori_permutation_importance, nori_shap_importance
from synthefy_nori.explainability.ebm import fit_ebm, ebm_structure
from synthefy_nori.explainability.viz import plot_ebm_model

# the whole flow in one call: importance -> prune -> glass-box EBM
interp = NoriInterpreter(model="nori-6m", target_name="y").fit(Xtr, ytr)

# or compose the pieces yourself
nori = NoriRegressor(model="nori-6m").fit(Xtr, ytr)
imp, base = nori_permutation_importance(nori, Xte, yte, metric=lambda a, b: r2_score(a, b))
ebm = fit_ebm(Xtr[:, top_cols], ytr, [names[c] for c in top_cols], task="regression")
plot_ebm_model(ebm, [names[c] for c in top_cols], X_density=Xtr[:, top_cols],
               task="regression", target_name="y", skill=r2, out_path="ebm.png")
```

## Modules

| module        | contents |
|---------------|----------|
| `pipeline.py` | `run(...)` end-to-end + the `python -m` CLI |
| `importance.py` | `nori_permutation_importance`, `nori_shap_importance` |
| `ebm.py`      | `fit_ebm`, `ebm_score`, `ebm_structure` |
| `viz.py`      | `plot_ebm_model` — the annotated model diagram |
| `data.py`     | `load_npz`, `load_csv`, `load_demo` (self-contained loaders) |

## Method notes

- **Permutation** (default, recommended for pruning): shuffle a raw column, run
  Nori's fixed fitted pipeline (Yeo-Johnson → polynomial interactions →
  TruncatedSVD → transformer), measure the skill drop. Directly accuracy-relevant;
  a feature reconstructable from correlated ones correctly reads as low-importance.
- **SHAP**: imputation-based Shapley values via shapiq — contribution *magnitude*
  rather than accuracy-after-removal. Slower; needs the `interpretability` extra.
- **Selection only when it pays off:** feature reduction runs only when
  `d > reduce_threshold` (default **16**). At or below that, low-dimensional data
  tends to lose a little accuracy from trimming, so all features are kept and the
  EBM is fit on everything (`reduced: false` in the JSON). Override with
  `--reduce-threshold N` / `reduce_threshold=N`.
- **Task** (regression vs binary classification) is auto-detected from `y`. The
  95 % target is `0.95·R²` for regression and `0.5 + 0.95·(AUC−0.5)` for
  classification (95 % of the skill above chance).


## `use_test` and what the scores mean

Interpretation is post-hoc, so by default (`use_test=True`) permutation importance is measured
on the held-out test split: shuffle a raw column of unseen rows, re-predict with the deployed
model, keep the skill drop. That is the standard way to ask what the shipped model relies on.

The one consequence to know is confined to a single number. The pruning sweep stops at the
first `k` whose score clears the `retain` bar, so with `use_test=True` that score —
`nori_selected_score_`, mirrored as `selection_score_` — is the value of the selection
criterion at the chosen `k`, not an independent estimate of it. On UCI credit default it reads
about 0.004 AUC optimistic. Unaffected: the ranking, the selected feature set,
`nori_full_score_`, and both EBM scores.

Pass `use_test=False` when you need every reported number to be a clean held-out estimate. The
training rows are then split 70/30 again, importance and the sweep are judged on that
carve-out, and the test split is touched only by the final scores. It costs one extra Nori fit.


## Multiclass

`detect_task` picks up a multiclass target (a non-float column with 3–20 distinct values;
three *floats* stay regression, since a coarse measurement is likelier than three classes), and
the metric becomes the macro one-vs-rest ROC-AUC.

Nori is a **regressor**, and the two classification cases are not the same problem:

* **Binary** already fits: regress the 0/1 indicator and rank on the continuous output, which is
  all ROC-AUC needs. Measured on breast-cancer with `nori-6m`, that scores 0.9923 against 0.9547
  for a prediction snapped to {0,1} — so routing through Nori's `discretize=` lattice would cost
  accuracy, not gain it.
* **Multiclass** cannot reuse it. Regressing the class codes `0..K-1` asserts that class 1 lies
  between class 0 and class 2 — false for nominal labels — and the objective then penalises a
  0-vs-4 error more than a 3-vs-4 one. `OneVsRestNori` applies the binary treatment K times, one
  indicator regression per class, and predicts an `(n, K)` score matrix. Measured on noisy
  digits, that beats code-regression by +0.09 to +0.17 macro-OVR-AUC; on clean data both hit
  1.0000, which is a ceiling effect rather than equivalence.

The cost is linear: **K fits per call**, so K times the sweep. For a genuinely *ordinal* target
(a rating, a grade) prefer a single regression or Nori's own `discretize=` lattice, which
exploits the ordering one-vs-rest deliberately throws away.

The glass-box diagram supports multiclass: each panel holds one curve per class with a shared
legend, and Σ feeds a softmax. There are no interaction heatmaps — interpret's EBM does not
support pairwise terms for K>2, so `fit_ebm` forces `interactions=0` there.
