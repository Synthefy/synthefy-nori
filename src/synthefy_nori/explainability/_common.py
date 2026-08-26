"""Small shared helpers: task detection, skill metric, 95%-target, NaN imputation."""

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score


TASKS = ("regression", "classification", "multiclass")
MAX_CLASSES = 20  # above this, an integer target is treated as a regression target


def detect_task(y, forced="auto"):
    """``'regression'``, ``'classification'`` (exactly 2 classes) or ``'multiclass'`` (3..20).

    Deliberately conservative for multiclass: only a non-float target (integer, bool, string,
    category) qualifies, because 3 distinct floats are far more likely to be a coarse
    continuous measurement than 3 classes. Anything above ``MAX_CLASSES`` distinct values is
    regression. Pass ``forced`` to override the guess.
    """
    if forced in TASKS:
        return forced
    y = np.asarray(y)
    is_float = np.issubdtype(y.dtype, np.floating)
    finite = y[~np.isnan(y)] if is_float else y
    n = len(np.unique(finite))
    if n == 2:
        return "classification"
    if not is_float and 3 <= n <= MAX_CLASSES:
        return "multiclass"
    return "regression"


def _macro_ovr_auc(y_true, scores):
    """Mean one-vs-rest ROC-AUC over the classes present in ``y_true``.

    Computed per class rather than through ``roc_auc_score(multi_class="ovr")`` on purpose:
    that path demands calibrated probabilities summing to 1, while these columns are raw
    regression scores. AUC only needs a ranking within each column, so the per-class mean is
    both valid and free of a normalisation that would change nothing but could fail.
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores)
    if scores.ndim == 1:
        raise ValueError("multiclass scoring needs one score column per class, got a 1-D array")
    aucs = [
        roc_auc_score((y_true == k).astype(int), scores[:, k])
        for k in range(scores.shape[1])
        if 0 < (y_true == k).sum() < len(y_true)
    ]
    if not aucs:
        raise ValueError("no class in y_true has both positive and negative examples")
    return float(np.mean(aucs))


def make_metric(task):
    """Return ``(metric_fn(y_true, y_pred_or_score), name)``.

    Regression is R² on the point prediction. Binary is ROC-AUC on a single score column.
    Multiclass is the macro one-vs-rest ROC-AUC over a ``(n, n_classes)`` score matrix — which
    is what both the one-vs-rest Nori wrapper and the EBM's ``predict_proba`` produce.
    """
    if task == "multiclass":
        return _macro_ovr_auc, "macro_ovr_auc"
    if task == "classification":
        return (lambda yt, yp: float(roc_auc_score(yt, np.asarray(yp).ravel()))), "roc_auc"
    return (lambda yt, yp: float(r2_score(yt, np.asarray(yp).ravel()))), "r2"


def target_classes(*ys, expect=None):
    """The classes present across *ys*, ascending — the caller's own labels.

    Kept by the estimator so predictions can be mapped back out of the internal 0..K-1 coding.
    ``expect`` asserts a class count (2 for binary).
    """
    arrs = [np.asarray(y) for y in ys]
    classes = np.unique(np.concatenate([a.ravel() for a in arrs]))
    if len(classes) < 2:
        raise ValueError(f"classification needs at least 2 classes, got {len(classes)}")
    if expect is not None and len(classes) != expect:
        raise ValueError(
            f"expected exactly {expect} classes, got {len(classes)}: "
            f"{classes.tolist()}. Pass task='regression' or 'multiclass' instead."
        )
    return classes


def encode_labels(y, classes):
    """Map labels onto 0..K-1 by position in ``classes`` (from :func:`target_classes`)."""
    return np.searchsorted(np.asarray(classes), np.asarray(y)).astype(int)


def decode_labels(codes, classes):
    """Inverse of :func:`encode_labels`: 0..K-1 back to the caller's own labels."""
    return np.asarray(classes)[np.asarray(codes).astype(int)]


def binary_classes(*ys):
    """The two classes present across *ys*, ascending. Raises if there are not exactly two."""
    return target_classes(*ys, expect=2)


def binarize01(*ys):
    """Remap 2-class target(s) to {0,1} (largest label -> 1) so ROC-AUC stays valid
    for any binary encoding ({1,2}, {-1,1}, {0,1}, ...). Pass one or more arrays that
    share the same classes (e.g. ytr, yte); returns them remapped, in order."""
    classes = binary_classes(*ys)
    out = tuple((np.asarray(y) == classes[-1]).astype(int) for y in ys)
    return out if len(out) > 1 else out[0]


def target_from_full(task, full, frac=0.95):
    """95% of the model's *skill*. Regression: frac·R². Classification: frac of AUC's margin over 0.5."""
    if task == "classification":
        return 0.5 + frac * (full - 0.5)
    return frac * full


def train_means(Xtr):
    """Per-column means of Xtr over the FINITE entries (0.0 for columns with none).
    +/-inf is treated as missing, so one infinite cell cannot poison a column's mean."""
    X = np.asarray(Xtr, np.float32)
    with warnings.catch_warnings():  # an all-missing column is expected here
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(np.where(np.isfinite(X), X, np.nan), 0)
    return np.where(np.isnan(mu), 0.0, mu)


def fill_nan(X, mu):
    """Replace every non-finite entry of X (NaN and +/-inf alike) with the per-column
    means ``mu``; returns a float32 array. Imputing inf matters because the alternative
    (``np.nan_to_num``) turns it into ~3.4e38, a bogus but finite feature value."""
    X = np.asarray(X, np.float32)
    return np.where(np.isfinite(X), X, mu).astype(np.float32)


def impute_mean(Xtr, *others):
    """Fill NaNs with train-column means (0.0 for all-NaN columns). Returns float32 arrays."""
    mu = train_means(Xtr)
    return tuple(fill_nan(a, mu) for a in (Xtr, *others))


def clip_inf_edges(a):
    """Replace ±inf entries with a small pad (5% of span) beyond the finite min/max.
    Used to make EBM bin edges plottable/serialisable. Returns a 1-D float array."""
    a = np.asarray(a, float).ravel()
    fin = a[np.isfinite(a)]
    lo, hi = (fin.min(), fin.max()) if len(fin) else (0.0, 1.0)
    span = (hi - lo) or 1.0
    a = np.where(a == -np.inf, lo - 0.05 * span, a)
    a = np.where(a == np.inf, hi + 0.05 * span, a)
    return a


def shape_direction(scores):
    """Classify an EBM shape function's trend from its per-bin scores:
    'flat' (too few bins) | 'negligible' (swing < 6% of |max|) | 'increasing' |
    'decreasing' | 'non-monotone'. Single source of truth for both the serialized
    structure and the rendered figure."""
    ys = np.asarray(scores, float).ravel()
    if len(ys) < 3:
        return "flat"
    if (ys.max() - ys.min()) < 0.06 * (abs(ys).max() or 1.0):
        return "negligible"
    rho = spearmanr(np.arange(len(ys)), ys).correlation
    if rho is not None and rho > 0.75:
        return "increasing"
    if rho is not None and rho < -0.75:
        return "decreasing"
    return "non-monotone"
