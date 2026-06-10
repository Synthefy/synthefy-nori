"""Dataset registry and loaders for the unified evaluation pipeline.

Supports:
  - TabArena (local CSV dirs)
  - TALENT benchmark (local CSV dirs + benchmark lists, optional OpenML download)
  - RelBench CTU (70+ relational datasets, flattened to single tables)
  - OpenML-CC18 (curated classification benchmark suite)
  - OpenML regression suites
  - Custom local CSV directories
"""

from __future__ import annotations

import os
import gc
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class DatasetEntry:
    """A single evaluation dataset."""
    name: str
    source: str
    task_type: str  # "classification" or "regression"
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_classes: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_train(self):
        return self.X_train.shape[0]

    @property
    def n_test(self):
        return self.X_test.shape[0]

    @property
    def n_features(self):
        return self.X_train.shape[1]

    def summary(self):
        return {
            "name": self.name, "source": self.source,
            "task_type": self.task_type, "n_train": self.n_train,
            "n_test": self.n_test, "n_features": self.n_features,
            "n_classes": self.n_classes,
        }


class DatasetRegistry:
    """Central registry that loads and caches datasets from multiple sources."""

    def __init__(self, cache_dir="./cache/eval_datasets", max_train_samples=50000,
                 max_classes=10, random_state=42):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_train_samples = max_train_samples
        self.max_classes = max_classes
        self.random_state = random_state
        self._datasets: Dict[str, DatasetEntry] = {}

    @property
    def datasets(self):
        return self._datasets

    def list_datasets(self):
        return sorted(self._datasets.keys())

    def get(self, name):
        return self._datasets.get(name)

    def filter(self, source=None, task_type=None):
        results = list(self._datasets.values())
        if source:
            results = [d for d in results if d.source == source]
        if task_type:
            results = [d for d in results if d.task_type == task_type]
        return results

    # ------------------------------------------------------------------
    # TabArena
    # ------------------------------------------------------------------
    def load_tabarena(self, cls_dir="cache/tabarena_cls", reg_dir="cache/tabarena_reg"):
        loaded = 0
        for ttype, data_dir in [("classification", cls_dir), ("regression", reg_dir)]:
            if data_dir and os.path.isdir(data_dir):
                for name in sorted(os.listdir(data_dir)):
                    folder = os.path.join(data_dir, name)
                    if not os.path.isdir(folder):
                        continue
                    entry = self._load_csv_dataset(folder, name, "tabarena", ttype)
                    if entry is not None:
                        self._datasets[f"tabarena/{entry.name}"] = entry
                        loaded += 1
        print(f"[DatasetRegistry] Loaded {loaded} TabArena datasets")
        return loaded

    # ------------------------------------------------------------------
    # TALENT benchmark (local CSV dirs)
    # ------------------------------------------------------------------
    def load_talent(
        self,
        cls_dir="cache/talent_cls",
        reg_dir="cache/talent_reg",
        cls_list_path="benchmark_list/talent_cls.csv",
        reg_list_path="benchmark_list/talent_reg.csv",
    ):
        loaded = 0
        total_missing = 0
        for ttype, data_dir, list_path in [
            ("classification", cls_dir, cls_list_path),
            ("regression", reg_dir, reg_list_path),
        ]:
            if not data_dir:
                continue
            if not os.path.isdir(data_dir):
                print(f"[DatasetRegistry] TALENT {ttype} dir not found: {data_dir}")
                continue

            expected_names = self._read_dataset_list(list_path)
            if expected_names:
                names = expected_names
            else:
                names = sorted(os.listdir(data_dir))

            for name in names:
                folder = os.path.join(data_dir, name)
                if not os.path.isdir(folder):
                    total_missing += 1
                    continue
                entry = self._load_csv_dataset(folder, name, "talent", ttype)
                if entry is not None:
                    self._datasets[f"talent/{entry.name}"] = entry
                    loaded += 1

        if total_missing > 0:
            print(f"[DatasetRegistry] TALENT: skipped {total_missing} missing dataset folders")
        print(f"[DatasetRegistry] Loaded {loaded} TALENT datasets")
        return loaded

    def download_talent(
        self,
        cls_dir="cache/talent_cls",
        reg_dir="cache/talent_reg",
        cls_list_path="benchmark_list/talent_cls.csv",
        reg_list_path="benchmark_list/talent_reg.csv",
        force=False,
    ):
        """Download TALENT datasets from OpenML into local CSV cache dirs."""
        try:
            import openml
        except ImportError:
            print("[DatasetRegistry] openml not installed. Use: uv add openml")
            return {"downloaded": 0, "skipped": 0, "failed": 0}

        try:
            openml_index = openml.datasets.list_datasets(output_format="dataframe")
        except Exception as e:
            print(f"[DatasetRegistry] Failed to list OpenML datasets: {e}")
            return {"downloaded": 0, "skipped": 0, "failed": 0}

        totals = {"downloaded": 0, "skipped": 0, "failed": 0}
        for task_type, out_dir, list_path in [
            ("classification", cls_dir, cls_list_path),
            ("regression", reg_dir, reg_list_path),
        ]:
            if not out_dir:
                continue
            names = self._read_dataset_list(list_path)
            if not names:
                print(f"[DatasetRegistry] TALENT {task_type}: list is empty or missing ({list_path})")
                continue

            os.makedirs(out_dir, exist_ok=True)
            print(f"[DatasetRegistry] Downloading TALENT {task_type}: {len(names)} datasets")
            for name in names:
                status = self._download_openml_csv_dataset(
                    dataset_name=name,
                    output_dir=out_dir,
                    task_type=task_type,
                    openml_index=openml_index,
                    force=force,
                )
                totals[status] += 1

        print(
            "[DatasetRegistry] TALENT download complete: "
            f"{totals['downloaded']} downloaded, {totals['skipped']} skipped, {totals['failed']} failed"
        )
        return totals

    # ------------------------------------------------------------------
    # RelBench CTU (via redelex)
    # ------------------------------------------------------------------
    def load_ctu(self, max_datasets=70, task_types=None):
        """Load CTU datasets via redelex (installed with uv add relbench[ctu])."""
        try:
            from redelex.datasets import ctu_datasets as _ctu_mod
        except ImportError:
            print("[DatasetRegistry] redelex not importable. "
                  "Use: uv add 'relbench[ctu]' torchvision pytorch_frame tensorboard")
            return 0

        ctu_classes = self._get_ctu_classes(_ctu_mod)
        loaded = 0
        for cls_name, cls_obj in ctu_classes[:max_datasets]:
            try:
                entry = self._load_ctu_dataset(cls_name, cls_obj, task_types)
                if entry is not None:
                    self._datasets[f"ctu/{entry.name}"] = entry
                    loaded += 1
            except Exception as e:
                print(f"  [CTU] Skipping {cls_name}: {e}")
            gc.collect()
        print(f"[DatasetRegistry] Loaded {loaded} CTU datasets")
        return loaded

    @staticmethod
    def _get_ctu_classes(ctu_mod):
        """Discover CTU dataset classes from redelex.datasets.ctu_datasets."""
        from redelex.datasets.ctu_datasets import CTUDataset
        classes = []
        for name in sorted(dir(ctu_mod)):
            obj = getattr(ctu_mod, name)
            if (isinstance(obj, type) and issubclass(obj, CTUDataset)
                    and obj is not CTUDataset):
                classes.append((name, obj))
        return classes

    def _load_ctu_dataset(self, cls_name, cls_obj, task_types=None):
        """Load a single CTU dataset: download, pick largest table, auto-detect target."""
        ds = cls_obj()
        db = ds.make_db()
        table_names = list(db.table_dict.keys())
        if not table_names:
            return None

        main_name = max(table_names, key=lambda t: len(db.table_dict[t].df))
        main_df = db.table_dict[main_name].df.copy()
        if len(main_df) < 50:
            return None

        df = self._prepare_ctu_dataframe(main_df)
        if df is None or df.shape[1] < 2:
            return None

        target_col = self._pick_ctu_target(df)
        if target_col is None:
            return None

        y, X = df[target_col], df.drop(columns=[target_col])
        n_unique = y.nunique()

        if 2 <= n_unique <= self.max_classes:
            task_type = "classification"
        elif n_unique > self.max_classes and pd.api.types.is_numeric_dtype(y):
            task_type = "regression"
        else:
            return None

        if task_types and task_type not in task_types:
            return None
        return self._make_entry_from_df(X, y, cls_name, "ctu", task_type)

    @staticmethod
    def _pick_ctu_target(df):
        """Auto-select the best target column (last numeric col with reasonable cardinality)."""
        # Skip PK / FK columns
        skip = {c for c in df.columns if c.startswith("__") or c.startswith("FK_")}
        candidates = [c for c in df.columns if c not in skip]
        if not candidates:
            return None
        # Prefer low-cardinality numeric columns (classification targets)
        for col in reversed(candidates):
            nuniq = df[col].nunique()
            if 2 <= nuniq <= 10 and pd.api.types.is_numeric_dtype(df[col]):
                return col
        # Fall back to last numeric column
        for col in reversed(candidates):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
        return None

    def _prepare_ctu_dataframe(self, df):
        df = df.dropna(axis=1, how="all")
        # Drop PK columns, datetime columns, and string-heavy columns
        drop_cols = [c for c in df.columns if c == "__PK__"]
        dt_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns
        drop_cols.extend(dt_cols)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        for col in df.select_dtypes(include=["object", "category", "string"]).columns:
            try:
                le = LabelEncoder()
                filled = df[col].fillna("__MISSING__").astype(str)
                if filled.nunique() > 100:
                    df = df.drop(columns=[col])
                    continue
                df[col] = le.fit_transform(filled)
            except Exception:
                df = df.drop(columns=[col])

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, thresh=int(len(df) * 0.5))
        df = df.fillna(df.median(numeric_only=True))
        if df.shape[1] < 2 or len(df) < 50:
            return None
        return df

    # ------------------------------------------------------------------
    # OpenML-CC18
    # ------------------------------------------------------------------
    def load_openml_cc18(self, max_datasets=72, show_progress=True):
        """Load OpenML-CC18 classification benchmark. Requires: uv add openml"""
        try:
            import openml
        except ImportError:
            print("[DatasetRegistry] openml not installed. Use: uv add openml")
            return 0

        loaded = 0
        try:
            suite = openml.study.get_suite(99)
            task_ids = suite.tasks[:max_datasets] if suite.tasks else []
            it = task_ids
            if show_progress:
                try:
                    from tqdm import tqdm
                    it = tqdm(task_ids, desc="OpenML-CC18", unit="task")
                except ImportError:
                    pass
            for tid in it:
                try:
                    entry = self._load_openml_task(tid, "openml_cc18", "classification")
                    if entry is not None:
                        self._datasets[f"openml_cc18/{entry.name}"] = entry
                        loaded += 1
                except Exception as e:
                    if not show_progress:
                        print(f"  [OpenML-CC18] Skipping task {tid}: {e}")
                gc.collect()
        except Exception as e:
            print(f"[DatasetRegistry] OpenML-CC18 loading failed: {e}")
        print(f"[DatasetRegistry] Loaded {loaded} OpenML-CC18 datasets")
        return loaded

    # ------------------------------------------------------------------
    # OpenML Regression
    # ------------------------------------------------------------------
    def load_openml_regression(self, max_datasets=30, show_progress=True):
        """Load curated OpenML regression datasets. Requires: uv add openml"""
        try:
            import openml
        except ImportError:
            print("[DatasetRegistry] openml not installed. Use: uv add openml")
            return 0

        # Note: 4550 (MiceProtein) was removed — it's an 8-class classification dataset
        REGRESSION_IDS = [287, 422, 507, 546, 541, 1030, 23515, 42225, 42571, 43071, 43093]
        ids = REGRESSION_IDS[:max_datasets]
        loaded = 0
        it = ids
        if show_progress:
            try:
                from tqdm import tqdm
                it = tqdm(ids, desc="OpenML regression", unit="dataset")
            except ImportError:
                pass
        for did in it:
            try:
                entry = self._load_openml_dataset_by_id(did, "openml_reg", "regression")
                if entry is not None:
                    self._datasets[f"openml_reg/{entry.name}"] = entry
                    loaded += 1
            except Exception as e:
                if not show_progress:
                    print(f"  [OpenML-Reg] Skipping dataset {did}: {e}")
            gc.collect()
        print(f"[DatasetRegistry] Loaded {loaded} OpenML regression datasets")
        return loaded

    # ------------------------------------------------------------------
    # Custom local CSV dir
    # ------------------------------------------------------------------
    def load_custom_dir(self, data_dir, task_type="classification", source_name="custom"):
        loaded = 0
        for name in sorted(os.listdir(data_dir)):
            folder = os.path.join(data_dir, name)
            if not os.path.isdir(folder):
                continue
            entry = self._load_csv_dataset(folder, name, source_name, task_type)
            if entry is not None:
                self._datasets[f"{source_name}/{entry.name}"] = entry
                loaded += 1
        print(f"[DatasetRegistry] Loaded {loaded} datasets from {data_dir}")
        return loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _read_dataset_list(list_path):
        if not list_path or not os.path.exists(list_path):
            return []
        try:
            with open(list_path, "r", encoding="utf-8") as f:
                names = []
                for line in f:
                    name = line.strip()
                    if not name or name.startswith("#"):
                        continue
                    names.append(name)
                return names
        except OSError:
            return []

    @staticmethod
    def _pick_best_openml_row(matches, task_type=None):
        """Pick the best OpenML dataset row from a set of matches.

        For regression: prefer version 1 (original numeric target).
        Later versions are often binarized classification variants with
        string targets (P/N), which silently produce all-zero y arrays.
        For classification: prefer the latest version (most cleaned up).
        """
        if matches is None or matches.empty:
            return None
        if "version" not in matches.columns:
            return matches.iloc[0]
        if task_type == "regression":
            return matches.sort_values("version", ascending=True).iloc[0]
        return matches.sort_values("version", ascending=False).iloc[0]

    @staticmethod
    def _latest_openml_row(matches):
        """Legacy alias — prefer _pick_best_openml_row with task_type."""
        if matches is None or matches.empty:
            return None
        if "version" in matches.columns:
            return matches.sort_values("version", ascending=False).iloc[0]
        return matches.iloc[0]

    @staticmethod
    def _pick_openml_row(dataset_name, openml_index, task_type=None):
        if openml_index is None or "name" not in openml_index.columns:
            return None

        picker = lambda m: DatasetRegistry._pick_best_openml_row(m, task_type=task_type)

        names = openml_index["name"].astype(str)
        exact = openml_index[names == dataset_name]
        row = picker(exact)
        if row is not None:
            return row

        target_lower = dataset_name.lower()
        lower_names = names.str.lower()
        ci_exact = openml_index[lower_names == target_lower]
        row = picker(ci_exact)
        if row is not None:
            return row

        # Normalize separators to improve matching consistency across naming variants.
        target_norm = re.sub(r"[^a-z0-9]+", "", target_lower)
        norm_names = lower_names.str.replace(r"[^a-z0-9]+", "", regex=True)
        norm_exact = openml_index[norm_names == target_norm]
        row = picker(norm_exact)
        if row is not None:
            return row

        pattern = re.escape(target_lower)
        pattern = pattern.replace(r"\-", r"[-_.\s]*")
        pattern = pattern.replace(r"\_", r"[-_.\s]*")
        partial = openml_index[lower_names.str.contains(pattern, regex=True, na=False)]
        row = picker(partial)
        if row is not None:
            return row
        return None

    def _download_openml_csv_dataset(self, dataset_name, output_dir, task_type, openml_index, force=False):
        dataset_dir = os.path.join(output_dir, dataset_name)
        train_path = os.path.join(dataset_dir, f"{dataset_name}_train.csv")
        test_path = os.path.join(dataset_dir, f"{dataset_name}_test.csv")

        if not force and os.path.exists(train_path) and os.path.exists(test_path):
            print(f"  [SKIP] {dataset_name} already exists")
            return "skipped"

        row = self._pick_openml_row(dataset_name, openml_index, task_type=task_type)
        if row is None:
            print(f"  [FAIL] Could not find '{dataset_name}' on OpenML")
            return "failed"

        try:
            did = int(row["did"]) if "did" in row.index else int(row.name)
        except (ValueError, KeyError):
            print(f"  [FAIL] Invalid OpenML dataset id for '{dataset_name}'")
            return "failed"

        try:
            import openml

            dataset = openml.datasets.get_dataset(did, download_data=True)
            target = dataset.default_target_attribute
            if target is None:
                print(f"  [FAIL] {dataset_name} has no default target attribute")
                return "failed"

            X, y, _, _ = dataset.get_data(target=target)
            if X is None or y is None:
                print(f"  [FAIL] No data returned for {dataset_name}")
                return "failed"

            X_df = pd.DataFrame(X)
            y_series = pd.Series(y)

            # Drop rows where target is missing to keep splits and metrics well-defined.
            valid_mask = ~y_series.isna()
            X_df = X_df.loc[valid_mask]
            y_series = y_series.loc[valid_mask]
            if len(X_df) < 20:
                print(f"  [FAIL] {dataset_name} too small after filtering missing targets")
                return "failed"

            # Validate that regression targets are actually numeric.
            # Binarized OpenML versions have string targets (P/N) that silently
            # become all-zero arrays via pd.to_numeric(errors='coerce').fillna(0).
            if task_type == "regression":
                numeric_y = pd.to_numeric(y_series, errors="coerce")
                n_failed = numeric_y.isna().sum() - y_series.isna().sum()
                frac_failed = n_failed / max(len(y_series), 1)
                if frac_failed > 0.5:
                    print(f"  [FAIL] {dataset_name} (did={did}): {frac_failed:.0%} of "
                          f"target values are non-numeric (got: {y_series.unique()[:5].tolist()}). "
                          f"This is likely a binarized classification variant, not regression.")
                    return "failed"

            df = X_df.copy()
            df["target"] = y_series.values

            stratify = None
            if task_type == "classification":
                n_unique = y_series.nunique()
                if n_unique >= 2:
                    stratify = y_series.astype(str)

            try:
                train_df, test_df = train_test_split(
                    df,
                    test_size=0.3,
                    random_state=self.random_state,
                    stratify=stratify,
                )
            except ValueError:
                train_df, test_df = train_test_split(
                    df, test_size=0.3, random_state=self.random_state
                )

            os.makedirs(dataset_dir, exist_ok=True)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            print(
                f"  [OK] {dataset_name}: {len(train_df)} train, "
                f"{len(test_df)} test, {X_df.shape[1]} features (did={did})"
            )
            return "downloaded"

        except Exception as e:
            print(f"  [FAIL] Error downloading {dataset_name} (did={did}): {e}")
            return "failed"

    def _load_csv_dataset(self, folder, name, source, task_type):
        train_path = os.path.join(folder, f"{name}_train.csv")
        test_path = os.path.join(folder, f"{name}_test.csv")
        if not os.path.exists(train_path):
            return None
        try:
            train_df = pd.read_csv(train_path)
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path)
            else:
                train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=self.random_state)
            X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
            X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
            return self._make_entry_from_df(X_train, y_train, name, source, task_type,
                                            X_test=X_test, y_test=y_test)
        except Exception as e:
            print(f"  [{source}] Error loading {name}: {e}")
            return None

    def _load_openml_task(self, task_id, source, task_type):
        import openml
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(target=task.target_name)
        if X is None or y is None:
            return None
        return self._make_entry_from_df(X, y, dataset.name, source, task_type)

    def _load_openml_dataset_by_id(self, dataset_id, source, task_type):
        import openml
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        target = dataset.default_target_attribute
        if target is None:
            return None
        X, y, _, _ = dataset.get_data(target=target)
        if X is None or y is None:
            return None
        return self._make_entry_from_df(X, y, dataset.name, source, task_type)

    def _make_entry_from_df(self, X, y, name, source, task_type, X_test=None, y_test=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        if X_test is not None:
            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test.copy()
            y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test.copy()

        # Encode object/category columns
        for col in list(X.select_dtypes(include=["object", "category"]).columns):
            try:
                le = LabelEncoder()
                parts = [X[col]] + ([X_test[col]] if X_test is not None else [])
                combined = pd.concat(parts).fillna("__MISSING__").astype(str)
                le.fit(combined)
                X[col] = le.transform(X[col].fillna("__MISSING__").astype(str))
                if X_test is not None:
                    X_test[col] = le.transform(X_test[col].fillna("__MISSING__").astype(str))
            except Exception:
                X = X.drop(columns=[col])
                if X_test is not None:
                    X_test = X_test.drop(columns=[col])

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        if X_test is not None:
            X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

        n_classes = None
        if task_type == "classification":
            le = LabelEncoder()
            parts_y = [y] + ([y_test] if y_test is not None else [])
            le.fit(pd.concat(parts_y).astype(str))
            y_arr = le.transform(y.astype(str)).astype(np.int64)
            y_test_arr = le.transform(y_test.astype(str)).astype(np.int64) if y_test is not None else None
            n_classes = len(le.classes_)
            if n_classes > self.max_classes or n_classes < 2:
                return None
        else:
            numeric_y = pd.to_numeric(y, errors="coerce")
            n_coerced = int(numeric_y.isna().sum() - y.isna().sum())
            if n_coerced > 0:
                frac_coerced = n_coerced / max(len(y), 1)
                if frac_coerced > 0.5:
                    warnings.warn(
                        f"[{source}/{name}] Skipping: {frac_coerced:.0%} of regression "
                        f"targets are non-numeric — likely a misclassified dataset. "
                        f"Sample values: {y[numeric_y.isna()].unique()[:5].tolist()}"
                    )
                    return None
                warnings.warn(
                    f"[{source}/{name}] {n_coerced}/{len(y)} regression target values "
                    f"are non-numeric — dropping those rows."
                )
            # Convert y and drop rows with non-numeric targets (instead of filling with 0)
            y_arr = numeric_y.values.astype(np.float64)
            if y_test is not None:
                numeric_y_test = pd.to_numeric(y_test, errors="coerce")
                n_coerced_test = int(numeric_y_test.isna().sum() - y_test.isna().sum())
                if n_coerced_test > 0:
                    frac_test = n_coerced_test / max(len(y_test), 1)
                    if frac_test > 0.5:
                        warnings.warn(
                            f"[{source}/{name}] Skipping: {frac_test:.0%} of test regression "
                            f"targets are non-numeric."
                        )
                        return None
                    warnings.warn(
                        f"[{source}/{name}] {n_coerced_test}/{len(y_test)} test regression "
                        f"target values are non-numeric — dropping those rows."
                    )
                y_test_arr = numeric_y_test.values.astype(np.float64)
            else:
                y_test_arr = None

        # Convert X to numpy. Row dropping for bad regression targets happens below.
        X_arr = X.values.astype(np.float32)
        X_test_arr = X_test.values.astype(np.float32) if X_test is not None else None

        # For regression: drop rows where y was non-numeric (NaN after coercion)
        if task_type != "classification":
            valid_y = np.isfinite(y_arr)
            if not valid_y.all():
                X_arr = X_arr[valid_y]
                y_arr = y_arr[valid_y]
            if y_test_arr is not None:
                valid_y_test = np.isfinite(y_test_arr)
                if not valid_y_test.all():
                    X_test_arr = X_test_arr[valid_y_test] if X_test_arr is not None else None
                    y_test_arr = y_test_arr[valid_y_test]

        if X_test_arr is None:
            try:
                X_arr, X_test_arr, y_arr, y_test_arr = train_test_split(
                    X_arr, y_arr, test_size=0.2, random_state=self.random_state,
                    stratify=y_arr if task_type == "classification" else None)
            except ValueError:
                X_arr, X_test_arr, y_arr, y_test_arr = train_test_split(
                    X_arr, y_arr, test_size=0.2, random_state=self.random_state)

        if X_arr.shape[0] > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X_arr.shape[0], self.max_train_samples, replace=False)
            X_arr, y_arr = X_arr[idx], y_arr[idx]

        if X_arr.shape[1] < 1 or X_arr.shape[0] < 10:
            return None

        return DatasetEntry(
            name=name, source=source, task_type=task_type,
            X_train=X_arr, y_train=y_arr,
            X_test=X_test_arr, y_test=y_test_arr,
            n_classes=n_classes,
            metadata={"n_train_original": X_arr.shape[0], "n_features_original": X_arr.shape[1]},
        )
