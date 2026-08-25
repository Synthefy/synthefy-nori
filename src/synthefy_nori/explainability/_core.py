"""The shared importance -> prune -> distill core.

Both front doors call this, so the sequence exists once:

* :class:`~synthefy_nori.explainability.interpreter.NoriInterpreter` — the sklearn-style
  estimator that owns its own split.
* :func:`~synthefy_nori.explainability.pipeline.run` — explicit train/test arrays, the
  ``method="shap"`` path, the ``python -m`` CLI, and the on-disk artifacts.

Keeping the split/impute/importance/sweep/distill logic in one place is deliberate: when it
lived in two copies they drifted, and the fixes for two separate bugs had to be written twice.

The row-sampling order here is load-bearing. ``prepare_rows`` draws the final-model rows
BEFORE the selection rows, and callers draw their own samples after; changing that order
changes every reported number even though no rule changed.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from synthefy_nori import NoriRegressor
from synthefy_nori.explainability._common import fill_nan, target_from_full, train_means
from synthefy_nori.explainability.ebm import ebm_score, fit_ebm

# Fractions of d swept when looking for the smallest feature set that clears the target.
SWEEP_FRACTIONS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
# Share of the TRAINING rows held back to judge selection when use_test is False.
SELECTION_FRACTION = 0.3


class OneVsRestNori:
    """Multiclass on top of a regressor, without inventing an order between the classes.

    Nori is a regressor: it predicts one number with a predictive distribution over a 1-D
    axis. For a binary target that is already the right shape — regress the 0/1 indicator and
    the continuous output ranks the positive class, which is all ROC-AUC needs. For K > 2 there
    is no such luck: regressing the class CODES 0..K-1 would assert that class 1 lies between
    class 0 and class 2, which for nominal labels is simply false, and the model would be
    penalised for ranking a "distant" class badly.

    So each class gets its own indicator regression — exactly the binary treatment, K times —
    and ``predict`` returns the ``(n_rows, K)`` score matrix that the macro one-vs-rest metric
    consumes. No ordering is assumed anywhere.

    The cost is honest and linear: K fits per call, so K times the sweep. For an ordinal target
    (a rating, a grade) prefer a single regression or Nori's own ``discretize=`` lattice, which
    exploits the ordering this class deliberately throws away.
    """

    def __init__(self, model):
        self.model = model
        self.models_ = []
        self.classes_ = None

    def fit(self, features, target):
        codes = np.asarray(target).astype(int).ravel()
        self.classes_ = np.unique(codes)
        self.models_ = [new_nori(self.model).fit(features, (codes == k).astype(np.float32))
                        for k in self.classes_]
        return self

    def predict(self, features):
        """``(n_rows, K)`` scores — column k ranks "is class k"."""
        return np.column_stack([np.asarray(m.predict(features)).ravel() for m in self.models_])


def new_nori(model, task=None):
    """A FRESH model for every fit.

    Never the caller's instance: the pruning sweep refits on column subsets, so a shared
    estimator would leave the reported model fit on whichever subset came last. Accepts a
    checkpoint name (``"nori-6m"``) or a pre-built estimator to clone. For ``task="multiclass"``
    the result is a :class:`OneVsRestNori` over that spec.
    """
    if task == "multiclass":
        return OneVsRestNori(model)
    if hasattr(model, "fit"):
        return clone(model)
    return NoriRegressor(model=model)


@dataclass
class SelectionRows:
    """Which rows each stage trains and evaluates on.

    ``final_*`` trains the models whose scores get reported. ``select_fit_*`` trains the
    model that ranks and prunes, and ``select_eval_*`` is what that ranking is measured on
    — the test split when ``use_test`` is True, otherwise a carve-out of the training rows.
    """

    final_features: np.ndarray
    final_target: np.ndarray
    select_fit_features: np.ndarray
    select_fit_target: np.ndarray
    select_eval_features: np.ndarray
    select_eval_target: np.ndarray
    use_test: bool
    n_select_fit: int = 0
    n_select_eval: int = 0


@dataclass
class Selection:
    """What the ranking + pruning stage decided, and on what evidence."""

    importance: np.ndarray
    order: list
    columns: list
    reduced: bool
    base_score: float
    select_full_score: float
    select_score: float
    target: float
    curve: list = field(default_factory=list)


def cap_rows(rng, rows, cap):
    """Nori is in-context: cap the rows it conditions on."""
    rows = np.asarray(rows)
    if len(rows) <= cap:
        return rows
    return rng.choice(rows, cap, replace=False)


def prepare_rows(raw_train, target_train, raw_test, target_test, *, use_test, stratify, rng,
                 nori_cap, selection_fraction=SELECTION_FRACTION, random_state=0):
    """Impute, sample, and assemble :class:`SelectionRows` from the RAW feature arrays.

    Takes raw (un-imputed) features on purpose: each stage is imputed from its own training
    rows, so the selection carve-out never sees a column mean computed over its own eval rows.
    Returns ``(rows, mu_train, features_train, features_test)`` — ``mu_train`` is what a caller
    must keep to impute at predict time.
    """
    mu_train = train_means(raw_train)
    features_train = fill_nan(raw_train, mu_train)
    features_test = fill_nan(raw_test, mu_train)

    all_rows = np.arange(len(features_train))
    final_rows = cap_rows(rng, all_rows, nori_cap)      # drawn FIRST — see module docstring
    final_features, final_target = features_train[final_rows], target_train[final_rows]

    if use_test:
        # Post-hoc reading: perturb the deployed model on held-out data.
        rows = SelectionRows(
            final_features=final_features, final_target=final_target,
            select_fit_features=final_features, select_fit_target=final_target,
            select_eval_features=features_test, select_eval_target=target_test,
            use_test=True, n_select_fit=len(final_features), n_select_eval=len(features_test),
        )
        return rows, mu_train, features_train, features_test

    fit_rows, eval_rows = train_test_split(all_rows, test_size=selection_fraction,
                                           random_state=random_state, stratify=stratify)
    mu_select = train_means(raw_train[fit_rows])        # the carve-out's OWN column means
    capped = cap_rows(rng, fit_rows, nori_cap)
    rows = SelectionRows(
        final_features=final_features, final_target=final_target,
        select_fit_features=fill_nan(raw_train[capped], mu_select),
        select_fit_target=target_train[capped],
        select_eval_features=fill_nan(raw_train[eval_rows], mu_select),
        select_eval_target=target_train[eval_rows],
        use_test=False, n_select_fit=len(fit_rows), n_select_eval=len(eval_rows),
    )
    return rows, mu_train, features_train, features_test


def select_features(rows, *, model, metric, metric_name, task, importance_fn, retain,
                    reduce_threshold, n_features, log=None, sweep_fractions=SWEEP_FRACTIONS):
    """Rank the features, then keep the fewest that retain ``retain`` of the model's skill.

    ``importance_fn(fitted_model, features, target) -> (importance, base_score)`` decides HOW
    features are ranked (permutation or SHAP); everything else here is shared.

    Returns ``(selection, select_model)``. When ``rows.use_test`` is True the returned model was
    trained on the final rows, so pass it to :func:`score_on_test` instead of refitting it.
    """
    say = log or (lambda *a: None)
    select_model = new_nori(model, task).fit(rows.select_fit_features, rows.select_fit_target)
    importance, base_score = importance_fn(select_model, rows.select_eval_features,
                                           rows.select_eval_target)
    order = list(np.argsort(-np.asarray(importance)))

    full = metric(rows.select_eval_target, select_model.predict(rows.select_eval_features))
    target = target_from_full(task, full, retain)
    reduced = n_features > reduce_threshold
    n_selected, at_k, curve = n_features, full, []
    if reduced:
        for k in sorted({max(1, math.ceil(f * n_features)) for f in sweep_fractions}):
            cols = order[:k]
            fitted = new_nori(model, task).fit(rows.select_fit_features[:, cols],
                                              rows.select_fit_target)
            score = metric(rows.select_eval_target,
                           fitted.predict(rows.select_eval_features[:, cols]))
            curve.append({"k": k, metric_name: round(float(score), 4)})
            say(f"  top {k:>4}/{n_features:<4} {metric_name}={score:+.4f} (target {target:+.4f})")
            if score >= target:
                n_selected, at_k = k, score
                break
        columns = order[:n_selected]
    else:
        columns = list(range(n_features))
        say(f"  d={n_features} <= reduce_threshold={reduce_threshold}: keeping every feature")

    selection = Selection(importance=np.asarray(importance), order=order, columns=list(columns),
                          reduced=reduced, base_score=float(base_score),
                          select_full_score=float(full), select_score=float(at_k),
                          target=float(target), curve=curve)
    return selection, select_model


def score_on_test(rows, selection, *, model, metric, features_test, target_test, task=None,
                  full_model=None):
    """Report the model's skill on the test split, on all features and on the selected ones.

    Two reuses avoid pointless refits when ``use_test=True``: ``full_model`` (the model
    :func:`select_features` already trained on the final rows), and the accepted sweep score,
    which already IS the selected-feature measurement on this very data.
    """
    if full_model is None:
        full_model = new_nori(model, task).fit(rows.final_features, rows.final_target)
    full = metric(target_test, full_model.predict(features_test))
    if not selection.reduced:
        selected = full
    elif rows.use_test:
        selected = selection.select_score
    else:
        cols = selection.columns
        fitted = new_nori(model, task).fit(rows.final_features[:, cols], rows.final_target)
        selected = metric(target_test, fitted.predict(features_test[:, cols]))
    return full_model, float(full), float(selected)


def distill_glassbox(rows, selection, *, names, task, metric, features_test, target_test):
    """Fit the glass-box EBM on the selected features, plus an all-features reference.

    When nothing was pruned the two are the same fit on the same data, so the reference is
    reused instead of paying for an identical second EBM.
    """
    cols = selection.columns
    selected_names = [names[c] for c in cols]
    ebm = fit_ebm(rows.final_features[:, cols], rows.final_target, selected_names, task)
    score = metric(target_test, ebm_score(ebm, features_test[:, cols], task))
    if selection.reduced:
        ebm_all = fit_ebm(rows.final_features, rows.final_target, list(names), task)
        score_all = metric(target_test, ebm_score(ebm_all, features_test, task))
    else:
        ebm_all, score_all = ebm, score
    return ebm, float(score), ebm_all, float(score_all), selected_names
