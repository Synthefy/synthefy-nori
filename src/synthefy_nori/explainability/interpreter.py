"""NoriInterpreter — one-call tabular interpretability with Nori.

``fit(X, y)`` takes the FULL table, makes an internal train/test split, then runs the
whole interpretability pipeline in sequence and stores every artifact on the fitted
estimator (scikit-learn style, trailing-underscore attributes):

    1. fit Nori and read **permutation feature importance**,
    2. **prune** to the fewest top features that keep >=`retain` of Nori's skill (when
       there are more than `reduce_threshold` features),
    3. distill a **glass-box EBM** (GA2M) on the selected features.

Regression (metric = R2) and binary classification (metric = ROC-AUC) are auto-detected.

    from synthefy_nori.explainability import NoriInterpreter
    interp = NoriInterpreter().fit(X, y)          # X, y = the full table
    interp.feature_importances_                   # per-feature importance
    interp.selected_features_                     # the pruned feature set
    interp.ebm_                                   # the fitted glass-box model
    interp.model_figure_                          # the model diagram (rendered in fit)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from synthefy_nori.explainability._common import (
    decode_labels,
    detect_task,
    encode_labels,
    fill_nan,
    make_metric,
    target_classes,
)
from synthefy_nori.explainability._core import distill_glassbox, prepare_rows, score_on_test, select_features
from synthefy_nori.explainability.ebm import ebm_structure
from synthefy_nori.explainability.importance import nori_permutation_importance
from synthefy_nori.explainability.viz import plot_ebm_model

_DENSITY_CAP = 2000  # rows kept for the shape-function histograms (keeps a fitted estimator light)


class NoriInterpreter(BaseEstimator):
    """One call from a raw table to importance + a pruned, readable glass-box model.

    Parameters
    ----------
    model : str or NoriRegressor, default="nori-6m"
        Nori checkpoint variant (``"nori-6m"`` / ``"nori-30m"``) or a pre-built
        ``NoriRegressor`` to clone for each fit.
    use_test : bool, default=True
        Measure importance (and the pruning sweep) on the held-out test split. Interpretation
        is post-hoc: permuting a column of unseen data and re-predicting is the standard way
        to ask what the DEPLOYED model relies on, so this is the honest reading of the
        ranking and the default.

        The trade-off is confined to one number. Because the sweep stops at the first ``k``
        whose test score clears the ``retain`` bar, ``nori_selected_score_`` is then the value
        of the selection criterion at that ``k`` rather than an independent estimate, and it
        reads optimistically (measured at ~0.004 AUC on UCI credit default). The ranking, the
        selected feature set, ``nori_full_score_`` and the EBM scores are unaffected.

        Set ``False`` to carve a selection split out of the training rows instead: importance
        and the sweep are judged there, the test split is touched only by the final scores,
        and every reported number is a clean held-out estimate. Costs one extra Nori fit.
    test_size : float, default=0.3
        Held-out fraction of the internal split (so the default is a 70/30 split).
        Both Nori and the EBM train on the ``nori_cap``-capped subsample, so a larger
        test fraction costs no training data on tables above the cap — it just buys a
        lower-variance estimate of the scores below.
    reduce_threshold : int, default=16
        Only prune features when the table has more than this many columns; at or
        below it, all features are kept (low-dim tables lose accuracy from trimming).
    retain : float, default=0.95
        The pruned feature set must retain at least this fraction of Nori's skill
        (of R2 for regression; of the AUC margin over 0.5 for classification).
    n_repeats : int, default=3
        Permutation shuffles averaged per feature.
    nori_cap : int, default=8000
        Cap on training rows used as Nori's in-context set (subsampled if larger).
    perm_eval : int, default=2000
        Cap on held-out rows used to measure the permutation drop.
    task : {"auto", "regression", "classification"}, default="auto"
    random_state : int, default=0
    render_figure : bool, default=True
        Render the glass-box model diagram at the end of ``fit`` and store it on
        ``model_figure_``. Set ``False`` to skip (avoids importing matplotlib).
    target_name : str, default="target"
        Label for the predicted quantity in the model diagram.

    Attributes
    ----------
    task_, metric_ : str
        Detected task and its scoring metric ("r2" or "roc_auc").
    feature_names_ : list[str]
    feature_importances_ : np.ndarray, shape (n_features,)
        Permutation importance aligned to the input columns (skill drop when shuffled).
    importance_ranking_ : list[dict]
        ``{"feature", "index", "importance"}`` sorted most-important first.
    selected_indices_ / selected_features_ : list[int] / list[str]
    n_selected_ : int
    classes_ : np.ndarray or None
        For classification, the two labels as originally passed to ``fit`` (ascending);
        ``predict`` maps back to them. ``None`` for regression.
    reduced_ : bool
        Whether pruning ran (``n_features > reduce_threshold``).
    nori_full_score_, nori_selected_score_ : float
        Nori's score on all features / on the selected features, measured on the test split.
        ``nori_full_score_`` involves no selection and is always a clean held-out estimate.
        ``nori_selected_score_`` is clean when ``use_test=False``; when ``use_test=True`` it
        coincides with ``selection_score_`` and carries that attribute's caveat.
    sweep_ : list[dict]
        The pruning sweep actually walked: ``{"k", <metric>}`` per candidate size, in
        increasing ``k``, stopping at the accepted one. Empty when nothing was pruned.
        Measured on the same data as ``selection_score_``, so it carries the same caveat.
    selection_score_, selection_full_score_ : float
        The score at the accepted ``k`` / on all features, on whatever data the sweep judged
        (the test split when ``use_test=True``, else the training carve-out). This is the
        criterion value pruning thresholded on — not an independent estimate. Reported so the
        optimism is visible rather than implied.
    ebm_score_, ebm_full_score_ : float
        Glass-box EBM score on the selected features / on all features.
    base_score_ : float
        Nori's unshuffled score on the importance-eval sample (drawn from the selection
        split, so it shares the caveat on ``selection_score_``).
    nori_ : NoriRegressor
        Nori fit on the (subsampled) full-feature training set.
    ebm_ : ExplainableBoosting{Classifier,Regressor}
        The shippable glass-box model, fit on the selected features.
    ebm_full_ : ExplainableBoosting{...}
        Reference EBM fit on all features.
    ebm_model_ : dict
        Serialized EBM: intercept, per-term importances, per-feature shape functions AND the
        pairwise interaction tables — together these reproduce the EBM's own prediction.
    model_figure_ : matplotlib.figure.Figure or None
        For multiclass the panels hold one curve per class (coloured, with a shared legend) and
        Σ feeds a softmax rather than a sigmoid; there are no interaction heatmaps, because
        interpret's EBM does not support pairwise terms for more than two classes.
        The glass-box model diagram (shape functions + interactions -> output),
        rendered during ``fit`` when ``render_figure`` is True (else ``None``).
        Save it with ``interp.model_figure_.savefig(...)`` or re-draw via ``plot_model``.
    """

    def __init__(
        self,
        *,
        model="nori-6m",
        test_size=0.3,
        reduce_threshold=16,
        retain=0.95,
        n_repeats=3,
        nori_cap=8000,
        perm_eval=2000,
        task="auto",
        random_state=0,
        render_figure=True,
        target_name="target",
        use_test=True,
    ):
        self.model = model
        self.test_size = test_size
        self.reduce_threshold = reduce_threshold
        self.retain = retain
        self.n_repeats = n_repeats
        self.nori_cap = nori_cap
        self.perm_eval = perm_eval
        self.task = task
        self.random_state = random_state
        self.render_figure = render_figure
        self.target_name = target_name
        self.use_test = use_test

    def fit(self, X, y, feature_names=None):
        """Fit on the FULL table.

        An outer 70/30 split holds out the test set the reported scores are measured on.
        With ``use_test=True`` (default) importance and the pruning sweep are measured there
        too — the post-hoc reading, at the cost of ``nori_selected_score_`` being a criterion
        value. With ``use_test=False`` the training portion is split 70/30 again so that
        nothing which chooses features has seen the test split.
        """
        if feature_names is None and hasattr(X, "columns"):
            feature_names = list(X.columns)
        X = np.asarray(X, np.float32)
        y = np.asarray(y)
        d = X.shape[1]
        names = list(feature_names) if feature_names is not None else [f"f{j}" for j in range(d)]

        task = detect_task(y, self.task)
        metric, metric_name = make_metric(task)
        if task in ("classification", "multiclass"):
            # keep the caller's labels; work internally in 0..K-1 so ROC-AUC stays valid for
            # any encoding ({1,2}, {-1,1}, strings, ...)
            self.classes_ = target_classes(y, expect=2 if task == "classification" else None)
            y = encode_labels(y, self.classes_)
        else:
            self.classes_ = None
            y = y.astype(np.float32)
        strat = y if task in ("classification", "multiclass") else None
        raw_train, raw_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=strat
        )

        rng = np.random.RandomState(self.random_state)
        rows, self._impute_mu_, features_train, features_test = prepare_rows(
            raw_train,
            y_train,
            raw_test,
            y_test,
            use_test=self.use_test,
            stratify=(y_train if task in ("classification", "multiclass") else None),
            rng=rng,
            nori_cap=self.nori_cap,
            selection_fraction=self.test_size,
            random_state=self.random_state,
        )

        def importance_fn(fitted, eval_features, eval_target):
            """Permutation importance, on a capped sample of the selection-eval rows."""
            if len(eval_features) > self.perm_eval:
                take = rng.choice(len(eval_features), self.perm_eval, replace=False)
                eval_features, eval_target = eval_features[take], eval_target[take]
            return nori_permutation_importance(
                fitted, eval_features, eval_target, metric, n_repeats=self.n_repeats, random_state=self.random_state
            )

        selection, select_model = select_features(
            rows,
            model=self.model,
            metric=metric,
            metric_name=metric_name,
            task=task,
            importance_fn=importance_fn,
            retain=self.retain,
            reduce_threshold=self.reduce_threshold,
            n_features=d,
        )

        full_model, nori_full, nori_sel = score_on_test(
            rows,
            selection,
            model=self.model,
            metric=metric,
            features_test=features_test,
            target_test=y_test,
            task=task,
            full_model=select_model if rows.use_test else None,
        )

        ebm, ebm_sel, ebm_all, ebm_all_score, fn_sel = distill_glassbox(
            rows, selection, names=names, task=task, metric=metric, features_test=features_test, target_test=y_test
        )

        # store artifacts
        self.task_, self.metric_ = task, metric_name
        self.feature_names_ = names
        self.feature_importances_ = selection.importance
        self.importance_ranking_ = [
            {"feature": names[j], "index": int(j), "importance": float(selection.importance[j])}
            for j in selection.order
        ]
        self.selected_indices_ = [int(c) for c in selection.columns]
        self.selected_features_ = fn_sel
        self.n_selected_ = len(selection.columns)
        self.reduced_ = selection.reduced
        self.base_score_ = selection.base_score
        self.sweep_ = selection.curve
        self.selection_score_ = selection.select_score
        self.selection_full_score_ = selection.select_full_score
        self.nori_full_score_ = nori_full
        self.nori_selected_score_ = nori_sel
        self.ebm_score_ = ebm_sel
        self.ebm_full_score_ = ebm_all_score
        self.nori_ = full_model
        self.ebm_ = ebm
        self.ebm_full_ = ebm_all
        self.ebm_model_ = ebm_structure(ebm)
        # capped sample of the selected columns, kept only for the shape-function histograms
        dens = features_train[:, selection.columns]
        if len(dens) > _DENSITY_CAP:
            dens = dens[rng.choice(len(dens), _DENSITY_CAP, replace=False)]
        self._density_ = dens

        # glass-box model diagram, rendered once and kept on the estimator
        self.model_figure_ = None
        if self.render_figure:
            fig = self._render_figure(target_name=self.target_name)
            # deferred on purpose: matplotlib is an optional extra, and importing it at
            # module level would make `import synthefy_nori.explainability` require it.
            # plot_ebm_model has already validated it by this point.
            import matplotlib.pyplot as plt

            plt.close(fig)  # keep the Figure object; don't auto-display it on fit
            self.model_figure_ = fig
        return self

    # ---- use the shippable glass-box (selected features) ----
    def _prep(self, X):
        return fill_nan(X, self._impute_mu_)[:, self.selected_indices_]

    def predict(self, X):
        """Predict with the glass-box EBM on the selected features.

        Classification returns the labels originally passed to ``fit`` (``classes_``), not the
        internal {0, 1} encoding — so a model fit on ``y in {1, 2}`` predicts 1s and 2s."""
        check_is_fitted(self, "ebm_")
        out = self.ebm_.predict(self._prep(X))
        if self.classes_ is None:
            return out
        return decode_labels(out, self.classes_)

    def predict_proba(self, X):
        """Class probabilities from the glass-box EBM (classification only)."""
        check_is_fitted(self, "ebm_")
        if self.task_ not in ("classification", "multiclass"):
            raise AttributeError("predict_proba is only available for classification tasks")
        return self.ebm_.predict_proba(self._prep(X))

    def _render_figure(self, *, target_name=None, feature_ranges="auto", **kwargs):
        """Draw the glass-box model diagram (shape functions + interactions -> output)."""
        if feature_ranges == "auto":  # clip each shape function to its 10-90 pct
            feature_ranges = {
                self.selected_features_[k]: (
                    float(np.percentile(self._density_[:, k], 10)),
                    float(np.percentile(self._density_[:, k], 90)),
                )
                for k in range(self.n_selected_)
            }
        skill = kwargs.pop("skill", self.ebm_score_)
        # multiclass panels hold one curve per class; label them with the caller's own labels
        kwargs.setdefault("class_names", None if self.task_ != "multiclass" else [str(c) for c in self.classes_])
        # allow an explicit override instead of colliding with the stored sample
        density = kwargs.pop("X_density", self._density_)
        return plot_ebm_model(
            self.ebm_,
            self.selected_features_,
            X_density=density,
            task=self.task_,
            target_name=target_name or self.target_name,
            skill=skill,
            feature_ranges=feature_ranges,
            **kwargs,
        )

    def plot_model(self, *, target_name=None, feature_ranges="auto", **kwargs):
        """Re-draw the glass-box model diagram (e.g. to a file via ``out_path=...``).

        ``fit`` already renders and stores the diagram on ``self.model_figure_``; call
        this to regenerate it with a different ``target_name``, ``feature_ranges``, or an
        ``out_path`` to save it."""
        check_is_fitted(self, "ebm_")
        return self._render_figure(target_name=target_name, feature_ranges=feature_ranges, **kwargs)

    def summary(self):
        """A compact dict of the headline results."""
        check_is_fitted(self, "ebm_")
        return {
            "task": self.task_,
            "metric": self.metric_,
            "n_features": len(self.feature_names_),
            "n_selected": self.n_selected_,
            "reduced": self.reduced_,
            "nori_full": round(self.nori_full_score_, 4),
            "nori_selected": round(self.nori_selected_score_, 4),
            "ebm_selected": round(self.ebm_score_, 4),
            "ebm_full": round(self.ebm_full_score_, 4),
            "selection_split_at_k": round(self.selection_score_, 4),  # criterion value, not held-out
            "top_features": [e["feature"] for e in self.importance_ranking_[:7]],
        }
