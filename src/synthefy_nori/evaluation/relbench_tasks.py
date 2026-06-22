"""Run Nori on the RelBench entity tasks (classification + regression).

RelBench (https://relbench.stanford.edu) is a relational deep-learning
benchmark. Tabular foundation models join its leaderboard via the *entity-table
tabular protocol*: each predictive task is flattened into a single feature table
(the task's train/val/test rows merged with their entity table), which a tabular
model then fits in-context. This module implements that protocol for Nori across
the seven canonical RelBench datasets, covering the binary-classification
(AUROC) and regression (MAE) entity tasks. Recommendation / link-prediction
tasks are out of scope for a tabular model and are skipped.

Two feature regimes are supported, gated by ``mode``:

* ``"entity"`` (Phase 1) — merge each task row with its entity table only. This
  is RelBench's out-of-the-box tabular baseline (the regime TabPFN is listed
  under on the leaderboard).
* ``"temporal"`` (Phase 2) — additionally synthesize per-entity temporal
  aggregations from related tables, computed strictly before each row's
  timestamp to avoid leakage. Implemented per-dataset in ``TEMPORAL_BUILDERS``.

Scoring uses RelBench's own ``task.evaluate`` so metrics match the leaderboard
exactly: validation is scored locally (labels available) and the held-out test
split is scored against RelBench's hidden labels.
"""

from __future__ import annotations

import os
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Optional

import duckdb
import jinja2
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from relbench.base import TaskType
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from synthefy_nori.api import NoriClassifier, NoriRegressor

# The seven canonical RelBench datasets and their entity (non-recommendation)
# tasks — the set tracked by the public leaderboard. Used as a fallback when the
# packaged list file is unavailable; the authoritative copy lives in
# ``benchmark_lists/relbench_entity.csv``.
DEFAULT_ENTITY_TASKS: tuple[tuple[str, str], ...] = (
    ("rel-amazon", "user-churn"),
    ("rel-amazon", "item-churn"),
    ("rel-amazon", "user-ltv"),
    ("rel-amazon", "item-ltv"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "user-visits"),
    ("rel-avito", "ad-ctr"),
    ("rel-event", "user-repeat"),
    ("rel-event", "user-ignore"),
    ("rel-event", "user-attendance"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-position"),
    ("rel-hm", "user-churn"),
    ("rel-hm", "item-sales"),
    ("rel-stack", "user-engagement"),
    ("rel-stack", "user-badge"),
    ("rel-stack", "post-votes"),
    ("rel-trial", "study-outcome"),
    ("rel-trial", "study-adverse"),
    ("rel-trial", "site-success"),
)

# Max numeric columns to aggregate per related fact table (keeps the feature
# count bounded on wide tables). Override via SYNTHEFY_RELBENCH_MAX_AGG_COLS.
_MAX_AGG_COLS = int(os.environ.get("SYNTHEFY_RELBENCH_MAX_AGG_COLS", "12"))

# Relational hops to traverse for temporal features: 1 = entity<-fact only,
# 2 = also entity<-fact->dimension (DFS-style multi-hop). Default 1: across the
# suite, generic 2-hop aggregates did not improve accuracy (even with feature
# selection) and the entity->fact->dimension join does not scale to the largest
# fact tables (e.g. rel-hm transactions). Override via SYNTHEFY_RELBENCH_HOPS.
_MAX_HOPS = int(os.environ.get("SYNTHEFY_RELBENCH_HOPS", "1"))

# Keep at most this many features (top-k by mutual information on the train
# split). DFS generates many aggregates; unlike GBDTs, Nori's ICL has no built-in
# feature selection, so we filter to the most informative before inference.
# 0 disables. Override via SYNTHEFY_RELBENCH_TOPK.
_TOPK_FEATURES = int(os.environ.get("SYNTHEFY_RELBENCH_TOPK", "128"))


def _select_features(Xtr, ytr, eval_arrays, is_cls, k):
    """Keep the top-``k`` features by mutual information with the target.

    Fit the selector on the (already subsampled) train split and apply the same
    column subset to every eval array. No-op when there are <= k features.
    """
    if not k or Xtr.shape[1] <= k:
        return Xtr, eval_arrays
    n = min(len(Xtr), 8000)  # cap rows for MI estimation speed
    rng = np.random.default_rng(0)
    ridx = rng.choice(len(Xtr), n, replace=False) if len(Xtr) > n else np.arange(len(Xtr))
    scorer = mutual_info_classif if is_cls else mutual_info_regression
    mi = scorer(Xtr[ridx], ytr[ridx], random_state=0)
    cols = np.argsort(mi)[-k:]
    return Xtr[:, cols], [Xe[:, cols] for Xe in eval_arrays]


def _numeric_feature_cols(table) -> list[str]:
    """Numeric, non-key columns of a table that are meaningful to aggregate."""
    keys = set(table.fkey_col_to_pkey_table.keys())
    if table.pkey_col:
        keys.add(table.pkey_col)
    if table.time_col:
        keys.add(table.time_col)
    cols = [
        c for c in table.df.columns
        if c not in keys and pd.api.types.is_numeric_dtype(table.df[c])
    ]
    return cols[:_MAX_AGG_COLS]


def _asof_aggregate(base, right, by_key, time_key, entity_col, time_col, prefix, num_cols):
    """As-of cumulative aggregation of ``right`` onto ``base`` rows.

    For each base (entity, seed_time), looks back over ``right`` rows for that
    entity occurring **strictly before** seed_time and returns event count,
    recency (days since the last), and the running sum / mean / latest of every
    ``num_cols`` value. Returns a DataFrame keyed by ``_row``. Leakage-safe
    (``allow_exact_matches=False``) and O(n log n) via ``merge_asof``.
    """
    right = right.dropna(subset=[by_key]).copy()
    if right.empty:
        return None
    right[by_key] = pd.to_numeric(right[by_key], errors="coerce").astype("int64")
    right[time_key] = pd.to_datetime(right[time_key]).astype("datetime64[ns]")
    right = right.sort_values(time_key, kind="stable")
    grp = right.groupby(by_key, sort=False)
    # DFS aggregation primitives, accumulated chronologically per entity so an
    # as-of join yields the value over all events strictly before the seed time:
    # count, sum, running max/min, and sum-of-squares (for std). mean/std derive
    # from these; last = the as-of row's own value; recency from its timestamp.
    right[f"{prefix}__cnt"] = grp.cumcount() + 1
    for c in num_cols:
        right[f"{prefix}__sum_{c}"] = grp[c].cumsum()
        right[f"{prefix}__sq_{c}"] = grp[c].transform(lambda s: (s * s).cumsum())
        right[f"{prefix}__max_{c}"] = grp[c].cummax()
        right[f"{prefix}__min_{c}"] = grp[c].cummin()

    merged = pd.merge_asof(
        base, right,
        left_on=time_col, right_on=time_key,
        left_by=entity_col, right_by=by_key,
        direction="backward", allow_exact_matches=False,
    )
    out = pd.DataFrame({"_row": merged["_row"].to_numpy()})
    cnt = merged[f"{prefix}__cnt"]
    out[f"{prefix}__cnt"] = cnt.fillna(0).to_numpy()
    out[f"{prefix}__recency_days"] = (merged[time_col] - merged[time_key]).dt.days.to_numpy()
    for c in num_cols:
        csum = merged[f"{prefix}__sum_{c}"]
        mean = csum / cnt
        var = (merged[f"{prefix}__sq_{c}"] / cnt) - (mean * mean)
        out[f"{prefix}__sum_{c}"] = csum.to_numpy()
        out[f"{prefix}__mean_{c}"] = mean.to_numpy()
        out[f"{prefix}__max_{c}"] = merged[f"{prefix}__max_{c}"].to_numpy()
        out[f"{prefix}__min_{c}"] = merged[f"{prefix}__min_{c}"].to_numpy()
        out[f"{prefix}__std_{c}"] = np.sqrt(var.clip(lower=0)).to_numpy()
        out[f"{prefix}__last_{c}"] = merged[c].to_numpy()
    return out


def add_temporal_features(task, db, base_df: pd.DataFrame) -> pd.DataFrame:
    """Append leakage-safe per-entity temporal aggregations to ``base_df``.

    Walks the schema from the task's entity table and aggregates related events
    occurring **strictly before** each row's seed timestamp:

    * **1-hop:** every fact table with a foreign key into the entity table —
      event count, recency, and running sum/mean/latest of its numeric columns.
    * **2-hop (DFS-style):** for each such fact table that *also* references a
      dimension table, the dimension's numeric attributes are joined in and
      aggregated over the entity's history (e.g. the price/attributes of the
      items a user bought). This cross-table signal is what the strong
      TabPFN+DFS tabular baseline on the RelBench leaderboard relies on.

    All aggregation is via ``merge_asof`` (``allow_exact_matches=False``), so it
    is leakage-safe and scales to multi-million-row tables. One generic pass
    covers all seven datasets.
    """
    entity_col, time_col = task.entity_col, task.time_col
    entity_table = task.entity_table
    if time_col not in base_df.columns:
        return base_df

    # Stable key to merge aggregates back onto base_df rows. merge_asof requires
    # the `by` keys to share an exact dtype and the `on` keys to be comparable,
    # so normalize entity ids to int64 and timestamps to datetime64[ns].
    base = base_df[[entity_col, time_col]].copy()
    base[entity_col] = pd.to_numeric(base[entity_col], errors="coerce").astype("int64")
    base[time_col] = pd.to_datetime(base[time_col]).astype("datetime64[ns]")
    base["_row"] = np.arange(len(base))
    base = base.sort_values(time_col, kind="stable")

    feat_frames: list[pd.DataFrame] = []
    for fname, ftable in db.table_dict.items():
        if fname == entity_table or ftable.time_col is None:
            continue
        fks = [c for c, tab in ftable.fkey_col_to_pkey_table.items() if tab == entity_table]
        if not fks:
            continue
        ftime = ftable.time_col
        fnum = _numeric_feature_cols(ftable)
        for fk in fks:
            # 1-hop: aggregate the fact table's own numeric columns.
            cols = [fk, ftime] + fnum
            out = _asof_aggregate(base, ftable.df[cols], fk, ftime, entity_col,
                                  time_col, f"{fname}.{fk}", fnum)
            if out is not None:
                feat_frames.append(out)

            # 2-hop: for each other dimension this fact references, join the
            # dimension's numeric attributes and aggregate over the entity's
            # history (DFS-style entity -> fact -> dimension).
            if _MAX_HOPS < 2:
                continue
            for dk, dtab_name in ftable.fkey_col_to_pkey_table.items():
                if dk == fk or dtab_name == entity_table or dtab_name not in db.table_dict:
                    continue
                dtab = db.table_dict[dtab_name]
                dnum = _numeric_feature_cols(dtab)
                if not dnum or not dtab.pkey_col:
                    continue
                left = ftable.df[[fk, ftime, dk]]
                right2 = left.merge(
                    dtab.df[[dtab.pkey_col] + dnum].rename(columns={c: f"{dtab_name}_{c}" for c in dnum}),
                    left_on=dk, right_on=dtab.pkey_col, how="left",
                )
                dcols = [f"{dtab_name}_{c}" for c in dnum]
                out2 = _asof_aggregate(base, right2[[fk, ftime] + dcols], fk, ftime,
                                       entity_col, time_col, f"{fname}.{fk}.{dtab_name}", dcols)
                if out2 is not None:
                    feat_frames.append(out2)

    if not feat_frames:
        return base_df
    feats = feat_frames[0]
    for f in feat_frames[1:]:
        feats = feats.merge(f, on="_row", how="outer")
    feats = feats.sort_values("_row").drop(columns="_row").reset_index(drop=True)
    return pd.concat([base_df.reset_index(drop=True), feats], axis=1)


@dataclass
class TaskResult:
    dataset: str
    task: str
    task_type: str
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    n_features: int = 0
    latency_ms: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def row(self) -> dict:
        r = {
            "dataset": self.dataset,
            "task": self.task,
            "task_type": self.task_type,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error or "",
        }
        r.update({k: v for k, v in self.metrics.items()})
        return r


def load_task_list(list_path: Optional[str] = None) -> list[tuple[str, str]]:
    """Read the pinned ``dataset,task`` list, falling back to the constant."""
    path = list_path
    if not path:
        try:
            packaged = files("synthefy_nori.evaluation.benchmark_lists") / "relbench_entity.csv"
            if packaged.is_file():
                path = str(packaged)
        except Exception:
            path = None
    if not path or not os.path.exists(path):
        return list(DEFAULT_ENTITY_TASKS)
    tasks: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] and parts[1]:
                tasks.append((parts[0], parts[1]))
    return tasks or list(DEFAULT_ENTITY_TASKS)


def _key_columns(task, entity_table) -> set[str]:
    """Columns that are primary/foreign keys or join/time/target — not features."""
    keys = {task.entity_col, task.time_col, task.target_col}
    if entity_table.pkey_col:
        keys.add(entity_table.pkey_col)
    keys.update(entity_table.fkey_col_to_pkey_table.keys())
    return keys


# RelBench's published "Data Scientist" hand-engineered SQL features
# (snap-stanford/relbench-user-study). Running these and feeding them to Nori
# gives a model-vs-model comparison against LightGBM on identical features.
_DS_SQL_BASE = "https://raw.githubusercontent.com/snap-stanford/relbench-user-study/main"
_DS_SQL_CACHE = "cache/relbench_ds_sql"


def fetch_ds_feats_sql(dataset_name: str, task_name: str) -> Optional[str]:
    """Fetch (and cache) the task's hand-written feats.sql, or None if absent."""
    short = dataset_name.replace("rel-", "")
    local = os.path.join(_DS_SQL_CACHE, short, task_name, "feats.sql")
    if os.path.exists(local):
        with open(local, "r", encoding="utf-8") as f:
            return f.read()
    url = f"{_DS_SQL_BASE}/{short}/{task_name}/feats.sql"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None
    os.makedirs(os.path.dirname(local), exist_ok=True)
    with open(local, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def build_ds_sql_features(dataset_name, task_name, task, db, split, sql_text):
    """Run the task's Data-Scientist SQL in DuckDB and return ``(X_df, y)``.

    Registers the RelBench base tables and the label splits, renders the
    Jinja-templated query for ``split`` (no subsampling), executes it, then
    re-aligns the result to ``task.get_table(split)`` row order (DuckDB joins do
    not preserve order) so predictions line up with ``task.evaluate``. The SQL's
    per-row ``timestamp > event_time`` cutoffs make it leakage-safe by design.
    """
    con = duckdb.connect()
    try:
        for name, t in db.table_dict.items():
            con.register(name, t.df)
        task_us = task_name.replace("-", "_")
        for s in ("train", "val", "test"):
            con.register(f"{task_us}_{s}", task.get_table(s).df)
        rendered = jinja2.Template(sql_text).render(set=split, subsample=0)
        con.execute(rendered)
        feats = con.execute(f"SELECT * FROM {task_us}_{split}_feats").df()
    finally:
        con.close()

    # Re-align to the canonical task-table order via the (entity, time) key.
    ekey, tkey = task.entity_col, task.time_col
    canon = task.get_table(split).df[[ekey, tkey]].copy()
    feats = feats.drop_duplicates(subset=[ekey, tkey])
    merged = canon.merge(feats, on=[ekey, tkey], how="left")
    y = merged[task.target_col].to_numpy() if task.target_col in merged.columns else None
    drop = {ekey, tkey, task.target_col}
    feature_cols = [c for c in merged.columns if c not in drop]
    return merged[feature_cols], y


def build_feature_table(task, db, split: str, mode: str = "entity"):
    """Flatten a RelBench entity task split into ``(X_df, y)`` (``y`` may be None).

    ``mode="entity"`` merges the task rows with their entity table only;
    ``mode="temporal"`` additionally applies the dataset's temporal builder.
    """
    table = task.get_table(split)  # test split is label-masked by RelBench
    df = table.df.copy()

    entity_table = db.table_dict[task.entity_table]
    ent_df = entity_table.df.copy()
    pk = entity_table.pkey_col
    df = df.merge(
        ent_df, how="left", left_on=task.entity_col, right_on=pk, suffixes=("", "__ent")
    )

    if mode == "temporal":
        df = add_temporal_features(task, db, df)

    y = df[task.target_col].to_numpy() if task.target_col in df.columns else None

    drop = _key_columns(task, entity_table) | {pk}
    feature_cols = [c for c in df.columns if c not in drop]
    return df[feature_cols], y


def _encode(train_df: pd.DataFrame, eval_dfs: list[pd.DataFrame]) -> list[np.ndarray]:
    """Numeric-encode features, fitting category/datetime maps on all splits.

    Mirrors ``DatasetRegistry._make_entry_from_df`` so the relational path
    matches the established CSV path: label-encode object/category/string
    columns (shared vocabulary across splits), convert datetimes to int64,
    coerce the rest to numeric, fill NaN with 0, cast to float32. Returns the
    encoded train array followed by one array per eval frame.
    """
    frames = [train_df] + eval_dfs
    combined = pd.concat(frames, axis=0, ignore_index=True)

    # Booleans (incl. pandas nullable 'boolean') -> object so the later
    # to_numeric coerces True/False -> 1/0 and NA -> NaN. Without this, the
    # final fillna(0) raises "Invalid value '0' for dtype boolean".
    for col in combined.columns:
        if combined[col].dtype == bool or str(combined[col].dtype) == "boolean":
            for fr in frames:
                if col in fr.columns:
                    fr[col] = fr[col].astype(object)
            combined[col] = combined[col].astype(object)

    # Datetime -> int64 nanoseconds (keep the temporal signal as a feature).
    for col in combined.select_dtypes(include=["datetime", "datetimetz"]).columns:
        for fr in frames:
            if col in fr.columns:
                fr[col] = pd.to_datetime(fr[col], errors="coerce").astype("int64")
        combined[col] = pd.to_datetime(combined[col], errors="coerce").astype("int64")

    # Label-encode strings/categoricals with a shared vocabulary.
    for col in list(combined.select_dtypes(include=["object", "category", "string"]).columns):
        try:
            le = LabelEncoder()
            le.fit(combined[col].astype(object).fillna("__MISSING__").astype(str))
            for fr in frames:
                if col in fr.columns:
                    fr[col] = le.transform(fr[col].astype(object).fillna("__MISSING__").astype(str))
        except Exception:
            for fr in frames:
                if col in fr.columns:
                    fr.drop(columns=[col], inplace=True, errors="ignore")

    out = []
    for fr in frames:
        arr = fr.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32).to_numpy()
        out.append(arr)
    return out


def _subsample(X: np.ndarray, y: np.ndarray, max_rows: int, *, stratify: bool = False, seed: int = 0):
    """Subsample context rows to ``max_rows``.

    For classification (``stratify=True``) the context is balanced across
    classes rather than sampled uniformly: with a uniform 50k draw on a
    rare-positive task (churn / DNF) Nori's in-context learner sees almost no
    positives and ranks near-randomly. Balancing the context surfaces the
    minority signal; AUROC is rank-based, so re-balancing the prior does not
    distort the well-behaved tasks.
    """
    n = len(X)
    if not max_rows or n <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    if not stratify:
        idx = rng.choice(n, max_rows, replace=False)
        return X[idx], y[idx]

    classes = np.unique(y)
    per_class = max(1, max_rows // len(classes))
    picks = []
    for c in classes:
        ci = np.where(y == c)[0]
        picks.append(rng.choice(ci, min(len(ci), per_class), replace=False))
    idx = np.concatenate(picks)
    if len(idx) < max_rows:  # a class was smaller than its quota — backfill
        remaining = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        if len(remaining):
            extra = rng.choice(remaining, min(len(remaining), max_rows - len(idx)), replace=False)
            idx = np.concatenate([idx, extra])
    rng.shuffle(idx)
    return X[idx], y[idx]


def run_task(
    dataset_name: str,
    task_name: str,
    *,
    mode: str = "entity",
    device: str = "cuda:0",
    inference_config: Optional[str] = None,
    max_train: int = 50000,
    download: bool = True,
) -> TaskResult:
    """Run one RelBench entity task end-to-end and score val + hidden test."""
    task = get_task(dataset_name, task_name, download=download)
    task_type = str(task.task_type).split(".")[-1]
    res = TaskResult(dataset=dataset_name, task=task_name, task_type=task_type)

    if task.task_type not in (TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION):
        res.error = f"unsupported task_type {task_type} (tabular path covers cls/reg only)"
        return res

    try:
        dataset = get_dataset(dataset_name, download=download)
        db = dataset.get_db()

        if mode == "ds_sql":
            sql_text = fetch_ds_feats_sql(dataset_name, task_name)
            if sql_text is None:
                res.error = "no Data-Scientist feats.sql published for this task"
                return res
            Xtr_df, ytr = build_ds_sql_features(dataset_name, task_name, task, db, "train", sql_text)
            Xval_df, yval = build_ds_sql_features(dataset_name, task_name, task, db, "val", sql_text)
            Xte_df, _ = build_ds_sql_features(dataset_name, task_name, task, db, "test", sql_text)
        else:
            Xtr_df, ytr = build_feature_table(task, db, "train", mode)
            Xval_df, yval = build_feature_table(task, db, "val", mode)
            Xte_df, _ = build_feature_table(task, db, "test", mode)

        # Align to the features available in every split. RelBench masks extra
        # input columns in the test table, so train/val can carry columns the
        # model would never see at inference — restrict to the common set (in
        # train order) so fit and predict use an identical, inference-available
        # feature schema.
        common = [c for c in Xtr_df.columns if c in set(Xval_df.columns) & set(Xte_df.columns)]
        Xtr_df, Xval_df, Xte_df = Xtr_df[common], Xval_df[common], Xte_df[common]
        Xtr, Xval, Xte = _encode(Xtr_df, [Xval_df, Xte_df])

        res.n_train, res.n_val, res.n_test = len(Xtr), len(Xval), len(Xte)
        res.n_features = Xtr.shape[1] if Xtr.ndim > 1 else 1

        is_cls = task.task_type == TaskType.BINARY_CLASSIFICATION
        Xtr_s, ytr_s = _subsample(Xtr, ytr, max_train, stratify=is_cls)
        Xtr_s, (Xval, Xte) = _select_features(Xtr_s, ytr_s, (Xval, Xte), is_cls, _TOPK_FEATURES)
        res.n_features = Xtr_s.shape[1] if Xtr_s.ndim > 1 else 1

        t0 = time.time()
        cls = NoriClassifier if is_cls else NoriRegressor
        model = cls(device=device, inference_config=inference_config).fit(Xtr_s, ytr_s)
        # The predictor chunks the forward over the (large) RelBench eval splits
        # internally; chunk_size is sequence-capped so it stays under the CUDA
        # grid limit even with these few-feature tables.
        if is_cls:
            val_pred = model.predict_proba(Xval)[:, 1]   # positive-class score for AUROC
            test_pred = model.predict_proba(Xte)[:, 1]
        else:
            val_pred = model.predict(Xval)
            test_pred = model.predict(Xte)
        res.latency_ms = (time.time() - t0) * 1000.0

        val_metrics = task.evaluate(val_pred, task.get_table("val"))
        test_metrics = task.evaluate(test_pred)
        for k, v in val_metrics.items():
            res.metrics[f"val_{k}"] = float(v)
        for k, v in test_metrics.items():
            res.metrics[f"test_{k}"] = float(v)
        # NMAE = MAE / train-split std — the RelBench leaderboard's regression
        # metric, so the reported number is directly comparable to it.
        if not is_cls:
            train_std = float(np.std(ytr))
            if train_std > 0:
                res.metrics["test_nmae"] = float(test_metrics.get("mae", np.nan)) / train_std
                res.metrics["val_nmae"] = float(val_metrics.get("mae", np.nan)) / train_std
    except Exception as e:  # noqa: BLE001 — record and continue the suite
        res.error = f"{type(e).__name__}: {e}".replace("\n", " | ")[:300]
        res.metrics["_traceback"] = traceback.format_exc()[-2000:]
    return res


def run_suite(
    tasks: Optional[list[tuple[str, str]]] = None,
    *,
    mode: str = "entity",
    device: str = "cuda:0",
    inference_config: Optional[str] = None,
    max_train: int = 50000,
    out_dir: str = "results/relbench",
    download: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the RelBench entity suite, writing per-type CSVs + a submission doc.

    Returns a dict with ``classification`` and ``regression`` DataFrames.
    Honors the dataset-crash-reporting rule: failed tasks are kept (with their
    error) and the crash count is printed in the summary.
    """
    tasks = tasks or load_task_list()
    os.makedirs(out_dir, exist_ok=True)

    results: list[TaskResult] = []
    for i, (ds, tk) in enumerate(tasks, 1):
        print(f"[relbench {i}/{len(tasks)}] {ds}/{tk} (mode={mode}) ...", flush=True)
        res = run_task(
            ds, tk, mode=mode, device=device,
            inference_config=inference_config, max_train=max_train, download=download,
        )
        if res.error:
            print(f"  CRASH {ds}/{tk}: {res.error}", flush=True)
        else:
            headline = res.metrics.get("test_roc_auc", res.metrics.get("test_mae"))
            print(f"  ok {ds}/{tk}: test={headline:.4f}  ({res.latency_ms:.0f} ms)", flush=True)
        results.append(res)

    rows = [r.row() for r in results]
    # Drop the bulky traceback column from the saved tables (kept only in-memory).
    for row in rows:
        row.pop("_traceback", None)
    df = pd.DataFrame(rows)

    cls_df = df[df["task_type"] == "BINARY_CLASSIFICATION"].copy()
    reg_df = df[df["task_type"] == "REGRESSION"].copy()
    cls_path = os.path.join(out_dir, "classification.csv")
    reg_path = os.path.join(out_dir, "regression.csv")
    cls_df.to_csv(cls_path, index=False)
    reg_df.to_csv(reg_path, index=False)

    _write_submission(cls_df, reg_df, results, mode, out_dir)

    n_crash = sum(1 for r in results if r.error)
    print("\n=== RelBench summary ===")
    print(f"  tasks run: {len(results)}  |  crashed: {n_crash}  |  ok: {len(results) - n_crash}")
    ok_cls = cls_df[cls_df["error"] == ""]
    ok_reg = reg_df[reg_df["error"] == ""]
    if len(ok_cls) and "test_roc_auc" in ok_cls:
        print(f"  classification: {len(ok_cls)} tasks  mean test AUROC = {ok_cls['test_roc_auc'].mean():.4f}")
    if len(ok_reg) and "test_mae" in ok_reg:
        print(f"  regression:     {len(ok_reg)} tasks  mean test MAE  = {ok_reg['test_mae'].mean():.4f}")
    print(f"  wrote {cls_path}, {reg_path}, {os.path.join(out_dir, 'SUBMISSION.md')}")
    return {"classification": cls_df, "regression": reg_df}


def _write_submission(cls_df, reg_df, results, mode, out_dir) -> None:
    regime = "Tabular (entity-table)" if mode == "entity" else "Tabular (temporal)"
    n_crash = sum(1 for r in results if r.error)
    lines = [
        "# Nori on RelBench — submission package",
        "",
        f"**Method:** Nori  ·  **Regime:** {regime}  ·  "
        f"**Protocol:** entity-table tabular flattening + in-context learning.",
        "",
        "Metrics are computed with RelBench's own `task.evaluate` (validation "
        "scored locally; test scored against RelBench's hidden labels), so they "
        "match the leaderboard definitions: **AUROC** for classification, "
        "**MAE** for regression (the public leaderboard reports normalized MAE).",
        "",
        f"Tasks attempted: {len(results)}  ·  crashed: {n_crash}  ·  "
        f"ok: {len(results) - n_crash}.",
        "",
        "## Classification (AUROC ↑)",
        "",
        "| Dataset | Task | Val AUROC | Test AUROC |",
        "|---|---|---:|---:|",
    ]
    for _, r in cls_df.iterrows():
        if r["error"]:
            lines.append(f"| {r['dataset']} | {r['task']} | — | _crashed_ |")
        else:
            lines.append(
                f"| {r['dataset']} | {r['task']} | "
                f"{r.get('val_roc_auc', float('nan')):.4f} | {r.get('test_roc_auc', float('nan')):.4f} |"
            )
    lines += [
        "",
        "## Regression (MAE ↓)",
        "",
        "| Dataset | Task | Val MAE | Test MAE | Test R² |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in reg_df.iterrows():
        if r["error"]:
            lines.append(f"| {r['dataset']} | {r['task']} | — | _crashed_ | — |")
        else:
            lines.append(
                f"| {r['dataset']} | {r['task']} | "
                f"{r.get('val_mae', float('nan')):.4f} | {r.get('test_mae', float('nan')):.4f} | "
                f"{r.get('test_r2', float('nan')):.4f} |"
            )
    lines += [
        "",
        "## Submitting to the leaderboard",
        "",
        "RelBench has **no self-service submission endpoint** at this time. The "
        "leaderboard (https://huggingface.co/spaces/relbench/leaderboard) is a "
        "maintainer-generated static page and is being redesigned ("
        "\"expecting submissions in the near future\"). `CONTRIBUTING.md` covers "
        "contributing datasets/tasks, not results. To submit these numbers, open "
        "an issue/PR on https://github.com/snap-stanford/relbench referencing "
        "this package and the tables above, and watch the leaderboard banner for "
        "the redesign going live.",
        "",
    ]
    with open(os.path.join(out_dir, "SUBMISSION.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
