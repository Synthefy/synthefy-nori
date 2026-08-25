"""Portable data loading for the explainability pipeline — NO benchmark infra.

Everything downstream only needs ``Xtr, ytr, Xte, yte`` (numpy) plus feature
names, so this module offers three self-contained entry points:

  * :func:`load_npz`  — an ``.npz`` holding arrays ``Xtr, ytr, Xte, yte`` (+ optional ``feature_names``)
  * :func:`load_table` — a single CSV **or Parquet** file with a target column, split into
    train/test (``load_csv`` remains as an alias)
  * :func:`load_demo` — a bundled scikit-learn dataset, so the pipeline runs with zero setup
"""
import os

import numpy as np
import pandas as pd
import sklearn.datasets as skd
from sklearn.model_selection import train_test_split

from synthefy_nori.explainability._common import detect_task


def _split(X, y, test_size, random_state):
    """train_test_split, stratified when y looks like a classification target."""
    strat = y if detect_task(y) == "classification" else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    for k in ("Xtr", "ytr", "Xte", "yte"):
        if k not in z:
            raise KeyError(f"{path}: npz must contain arrays Xtr, ytr, Xte, yte (missing {k!r})")
    Xtr = np.asarray(z["Xtr"], np.float32)
    names = list(z["feature_names"]) if "feature_names" in z else [f"f{j}" for j in range(Xtr.shape[1])]
    return Xtr, np.asarray(z["ytr"]), np.asarray(z["Xte"], np.float32), np.asarray(z["yte"]), names


def load_table(path, target, test_size=0.3, random_state=0):
    """Load a CSV or Parquet file and split it. Non-numeric feature columns are
    ordinal-encoded; the target column is taken as-is.

    Parquet keeps dtypes and is much faster on wide/long tables, so it is worth preferring
    for anything big. Chosen by suffix: ``.parquet`` / ``.pq`` read via pandas + pyarrow
    (a base dependency), everything else via ``read_csv``.
    """
    suffix = os.path.splitext(path)[1].lower()
    df = pd.read_parquet(path) if suffix in (".parquet", ".pq") else pd.read_csv(path)
    if target not in df.columns:
        raise KeyError(f"{path}: target column {target!r} not found (have {list(df.columns)})")
    feats = [c for c in df.columns if c != target]
    for c in feats:  # ordinal-encode any non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[c]):
            codes = df[c].astype("category").cat.codes.astype(np.float32)
            df[c] = codes.where(codes >= 0, np.nan)      # cat.codes marks missing as -1;
                                                         # keep it NaN for mean imputation
    X = df[feats].to_numpy(np.float32)
    y = df[target].to_numpy()
    Xtr, Xte, ytr, yte = _split(X, y, test_size, random_state)
    return Xtr, ytr, Xte, yte, feats


def load_csv(path, target, test_size=0.3, random_state=0):
    """Deprecated alias for :func:`load_table`, which also reads Parquet."""
    return load_table(path, target, test_size=test_size, random_state=random_state)


_DEMOS = {
    "diabetes": ("regression", "load_diabetes"),        # 10 features, quantitative disease progression
    "california": ("regression", "fetch_california_housing"),  # 8 features (downloads once)
    "breast_cancer": ("classification", "load_breast_cancer"),  # 30 features, binary
}


def load_demo(name="diabetes", test_size=0.3, random_state=0):
    """A bundled scikit-learn dataset so the pipeline runs end-to-end with no external data."""
    if name not in _DEMOS:
        raise ValueError(f"unknown demo {name!r}; choose from {sorted(_DEMOS)}")
    _, loader = _DEMOS[name]
    bunch = getattr(skd, loader)()
    X = np.asarray(bunch.data, np.float32)
    y = np.asarray(bunch.target)
    names = list(getattr(bunch, "feature_names", [f"f{j}" for j in range(X.shape[1])]))
    names = [str(n) for n in names]
    Xtr, Xte, ytr, yte = _split(X, y, test_size, random_state)
    return Xtr, ytr, Xte, yte, names


def load_from_args(args):
    """Resolve a data source from CLI args (--npz | --csv/--parquet + --target | --demo)."""
    if getattr(args, "npz", None):
        return load_npz(args.npz)
    table = getattr(args, "csv", None) or getattr(args, "parquet", None)
    if table:
        return load_table(table, args.target)
    return load_demo(getattr(args, "demo", None) or "diabetes")
