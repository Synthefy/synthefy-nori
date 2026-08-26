"""Dataset registry and loaders for the unified evaluation pipeline.

Supports:
  - TabArena (local CSV dirs)
  - TALENT benchmark (local CSV dirs + benchmark lists, optional OpenML download)
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

# Keep in sync with synthefy_nori.inference.predictor.NA_PLACEHOLDER (not
# imported: this module must stay importable without pulling in torch).
NA_PLACEHOLDER = "__MISSING__"


def encode_categorical_column(col, classes=None, *, unknown_value=None):
    """Ordinal-encode one categorical feature column to sorted-unique int64
    codes 0..K-1 — exactly what sklearn's LabelEncoder produced here, via the
    same primitives (np.unique / searchsorted), minus the 0-row raise.

    Missing values are mapped to NA_PLACEHOLDER so they get a real code
    (callers that want NaN restore it afterwards). Cast to object before
    fillna: on category dtype, fillna with an unseen value raises. When
    ``classes`` is given the codes index into that context-fitted vocabulary.
    Values absent from it raise unless ``unknown_value`` is provided (evaluation
    query rows use -1). Otherwise classes are derived from ``col``. Returns
    ``(codes, classes)``.
    """
    arr = col.astype(object).fillna(NA_PLACEHOLDER).astype(str).to_numpy()
    if classes is None:
        classes, codes = np.unique(arr, return_inverse=True)
        # numpy 2.0.0 briefly returned a 2-D inverse; ravel so codes are 1-D
        # on any numpy>=2.0 (keeps written CSV codes reproducible).
        codes = codes.ravel()
    else:
        classes = np.asarray(classes)
        codes = np.searchsorted(classes, arr)
        bounded = np.minimum(codes, max(len(classes) - 1, 0))
        known = (
            np.zeros(arr.shape, dtype=bool) if len(classes) == 0 else (codes < len(classes)) & (classes[bounded] == arr)
        )
        if not np.all(known):
            if unknown_value is None:
                unknown = np.unique(arr[~known]).tolist()
                raise ValueError(f"categorical values are absent from the fitted vocabulary: {unknown}")
            codes = np.where(known, codes, int(unknown_value))
    return codes.astype(np.int64), classes


@dataclass
class DatasetEntry:
    """A single evaluation dataset."""

    name: str
    source: str
    task_type: str  # always "regression"
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
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
            "name": self.name,
            "source": self.source,
            "task_type": self.task_type,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
        }


class DatasetRegistry:
    """Central registry that loads and caches datasets from multiple sources."""

    def __init__(self, cache_dir="./cache/eval_datasets", max_train_samples=50000, random_state=42):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_train_samples = max_train_samples
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
    def load_tabarena(self, reg_dir="cache/tabarena_reg"):
        loaded = 0
        if reg_dir and os.path.isdir(reg_dir):
            for name in sorted(os.listdir(reg_dir)):
                folder = os.path.join(reg_dir, name)
                if not os.path.isdir(folder):
                    continue
                entry = self._load_csv_dataset(folder, name, "tabarena")
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
        reg_dir="cache/talent_reg",
        reg_list_path="benchmark_list/talent_reg.csv",
    ):
        loaded = 0
        total_missing = 0
        reg_list_path = self._resolve_list_path(reg_list_path, "talent_reg.csv")
        if not reg_dir or not os.path.isdir(reg_dir):
            print(f"[DatasetRegistry] TALENT regression dir not found: {reg_dir}")
            return loaded

        expected_names = self._read_dataset_list(reg_list_path)
        names = expected_names if expected_names else sorted(os.listdir(reg_dir))

        for name in names:
            folder = os.path.join(reg_dir, name)
            if not os.path.isdir(folder):
                total_missing += 1
                continue
            entry = self._load_csv_dataset(folder, name, "talent")
            if entry is not None:
                self._datasets[f"talent/{entry.name}"] = entry
                loaded += 1

        if total_missing > 0:
            print(f"[DatasetRegistry] TALENT: skipped {total_missing} missing dataset folders")
        print(f"[DatasetRegistry] Loaded {loaded} TALENT datasets")
        return loaded

    def download_talent(
        self,
        reg_dir="cache/talent_reg",
        reg_list_path="benchmark_list/talent_reg.csv",
        force=False,
    ):
        """Download the TALENT regression datasets from OpenML into a local CSV cache dir."""
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
        reg_list_path = self._resolve_list_path(reg_list_path, "talent_reg.csv")
        names = self._read_dataset_list(reg_list_path)
        if not names:
            print(f"[DatasetRegistry] TALENT regression: list is empty or missing ({reg_list_path})")
            return totals

        os.makedirs(reg_dir, exist_ok=True)
        print(f"[DatasetRegistry] Downloading TALENT regression: {len(names)} datasets")
        for name in names:
            status = self._download_openml_csv_dataset(
                dataset_name=name,
                output_dir=reg_dir,
                openml_index=openml_index,
                force=force,
            )
            totals[status] += 1

        print(
            "[DatasetRegistry] TALENT download complete: "
            f"{totals['downloaded']} downloaded, {totals['skipped']} skipped, {totals['failed']} failed"
        )
        return totals

    def download_tabarena(
        self,
        reg_dir="cache/tabarena_reg",
        reg_list_path="benchmark_list/tabarena_reg.csv",
        force=False,
    ):
        """Download the official TabArena regression datasets from OpenML.

        Uses the TabArena team's curated dataset uploads, pinned by OpenML
        dataset ID (data on OpenML is immutable per ID), so the CSVs and the
        seeded 70/30 split are bit-reproducible. Categorical columns are
        label-encoded at download time with NaN preserved, matching the CSVs
        the published benchmark numbers were computed on.
        """
        try:
            import openml  # noqa: F401
        except ImportError:
            print("[DatasetRegistry] openml not installed. Use: uv add openml")
            return {"downloaded": 0, "skipped": 0, "failed": 0}

        totals = {"downloaded": 0, "skipped": 0, "failed": 0}
        reg_list_path = self._resolve_list_path(reg_list_path, "tabarena_reg.csv")
        pinned = self._read_pinned_dataset_list(reg_list_path)
        if not pinned:
            print(f"[DatasetRegistry] TabArena regression: list is empty or missing ({reg_list_path})")
            return totals

        os.makedirs(reg_dir, exist_ok=True)
        print(f"[DatasetRegistry] Downloading TabArena regression: {len(pinned)} datasets")
        for name, did in pinned:
            status = self._download_pinned_openml_dataset(name, did, reg_dir, force=force)
            totals[status] += 1

        print(
            "[DatasetRegistry] TabArena download complete: "
            f"{totals['downloaded']} downloaded, {totals['skipped']} skipped, {totals['failed']} failed"
        )
        return totals

    @staticmethod
    def _read_pinned_dataset_list(list_path):
        """Read a 'name,openml_dataset_id' list; returns [(name, did), ...]."""
        out = []
        for line in DatasetRegistry._read_dataset_list(list_path):
            if "," not in line:
                continue
            name, _, did = line.rpartition(",")
            try:
                out.append((name.strip(), int(did)))
            except ValueError:
                continue
        return out

    def _download_pinned_openml_dataset(self, dataset_name, did, output_dir, force=False):
        """Download one ID-pinned OpenML dataset and write seeded 70/30 train/test CSVs.

        Mirrors the procedure that produced the published benchmark CSVs:
        categoricals are label-encoded here (NaN preserved, lexicographic class
        order), the target is written as-is, and the split is test_size=0.3
        with the registry seed.
        """
        import openml

        dataset_dir = os.path.join(output_dir, dataset_name)
        train_path = os.path.join(dataset_dir, f"{dataset_name}_train.csv")
        test_path = os.path.join(dataset_dir, f"{dataset_name}_test.csv")
        if not force and os.path.exists(train_path) and os.path.exists(test_path):
            print(f"  [SKIP] {dataset_name} already exists")
            return "skipped"

        try:
            dataset = openml.datasets.get_dataset(did, download_data=True)
            target = dataset.default_target_attribute
            if target is None:
                print(f"  [FAIL] {dataset_name} (did={did}) has no default target attribute")
                return "failed"
            X, y, _, _ = dataset.get_data(target=target)
            if X is None or y is None:
                print(f"  [FAIL] No data returned for {dataset_name} (did={did})")
                return "failed"

            X = X.copy()
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    mask = X[col].isna()
                    # int64 codes keep the written CSVs byte-identical to the
                    # published ones (same sorted-unique codes as before).
                    X[col], _ = encode_categorical_column(X[col])
                    if mask.any():
                        X.loc[mask, col] = np.nan

            df = X
            df["target"] = y
            train_df, test_df = train_test_split(df, test_size=0.3, random_state=self.random_state)

            os.makedirs(dataset_dir, exist_ok=True)
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            print(f"  [OK] {dataset_name}: {len(train_df)} train, {len(test_df)} test (did={did})")
            return "downloaded"
        except Exception as e:
            print(f"  [FAIL] Error downloading {dataset_name} (did={did}): {e}")
            return "failed"

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

        # Note: 4550 (MiceProtein) was removed — its target is not a regression target
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
                entry = self._load_openml_dataset_by_id(did, "openml_reg")
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
    def load_custom_dir(self, data_dir, source_name="custom"):
        loaded = 0
        for name in sorted(os.listdir(data_dir)):
            folder = os.path.join(data_dir, name)
            if not os.path.isdir(folder):
                continue
            entry = self._load_csv_dataset(folder, name, source_name)
            if entry is not None:
                self._datasets[f"{source_name}/{entry.name}"] = entry
                loaded += 1
        print(f"[DatasetRegistry] Loaded {loaded} datasets from {data_dir}")
        return loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_list_path(list_path, packaged_name):
        """Prefer an existing local list file; fall back to the packaged list."""
        if list_path and os.path.exists(list_path):
            return list_path
        try:
            from importlib.resources import files

            packaged = files("synthefy_nori.evaluation.benchmark_lists") / packaged_name
            if packaged.is_file():
                return str(packaged)
        except Exception:
            pass
        return list_path

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
    def _pick_best_openml_row(matches):
        """Pick the best OpenML dataset row from a set of matches.

        Prefer version 1 (original numeric target). Later versions are often
        binarized variants with string targets (P/N), which silently produce
        all-zero y arrays.
        """
        if matches is None or matches.empty:
            return None
        if "version" not in matches.columns:
            return matches.iloc[0]
        return matches.sort_values("version", ascending=True).iloc[0]

    @staticmethod
    def _pick_openml_row(dataset_name, openml_index):
        if openml_index is None or "name" not in openml_index.columns:
            return None

        picker = DatasetRegistry._pick_best_openml_row

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

    def _download_openml_csv_dataset(self, dataset_name, output_dir, openml_index, force=False):
        dataset_dir = os.path.join(output_dir, dataset_name)
        train_path = os.path.join(dataset_dir, f"{dataset_name}_train.csv")
        test_path = os.path.join(dataset_dir, f"{dataset_name}_test.csv")

        if not force and os.path.exists(train_path) and os.path.exists(test_path):
            print(f"  [SKIP] {dataset_name} already exists")
            return "skipped"

        row = self._pick_openml_row(dataset_name, openml_index)
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
            numeric_y = pd.to_numeric(y_series, errors="coerce")
            n_failed = numeric_y.isna().sum() - y_series.isna().sum()
            frac_failed = n_failed / max(len(y_series), 1)
            if frac_failed > 0.5:
                print(
                    f"  [FAIL] {dataset_name} (did={did}): {frac_failed:.0%} of "
                    f"target values are non-numeric (got: {y_series.unique()[:5].tolist()}). "
                    f"This is not a usable regression target."
                )
                return "failed"

            df = X_df.copy()
            df["target"] = y_series.values

            train_df, test_df = train_test_split(df, test_size=0.3, random_state=self.random_state)

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

    def _load_csv_dataset(self, folder, name, source):
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
            return self._make_entry_from_df(X_train, y_train, name, source, X_test=X_test, y_test=y_test)
        except Exception as e:
            print(f"  [{source}] Error loading {name}: {e}")
            return None

    def _load_openml_dataset_by_id(self, dataset_id, source):
        import openml

        dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
        target = dataset.default_target_attribute
        if target is None:
            return None
        X, y, _, _ = dataset.get_data(target=target)
        if X is None or y is None:
            return None
        return self._make_entry_from_df(X, y, dataset.name, source)

    def _make_entry_from_df(self, X, y, name, source, X_test=None, y_test=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        y = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        if X_test is not None:
            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test.copy()
            y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test.copy()

        # Fit categorical vocabularies on context/train only. Query-only
        # values are explicitly unknown (-1), so preprocessing remains
        # inductive and cannot leak the query distribution into the context.
        for col in list(X.select_dtypes(include=["object", "category", "string"]).columns):
            try:
                _, classes = encode_categorical_column(X[col])
                X[col], _ = encode_categorical_column(X[col], classes)
                if X_test is not None:
                    X_test[col], _ = encode_categorical_column(
                        X_test[col],
                        classes,
                        unknown_value=-1,
                    )
            except Exception:
                X = X.drop(columns=[col])
                if X_test is not None:
                    X_test = X_test.drop(columns=[col])

        # Impute numeric NaNs with the per-column TRAIN median, then 0 for
        # all-missing columns — matching the production evaluator's
        # X.fillna(medians).fillna(0.0). Test rows are filled with TRAIN
        # medians so no test statistics leak into preprocessing.
        X = X.apply(pd.to_numeric, errors="coerce")
        train_medians = X.median()
        X = X.fillna(train_medians).fillna(0.0).astype(np.float32)
        if X_test is not None:
            X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(train_medians).fillna(0.0).astype(np.float32)

        numeric_y = pd.to_numeric(y, errors="coerce")
        n_coerced = int(numeric_y.isna().sum() - y.isna().sum())
        if n_coerced > 0:
            frac_coerced = n_coerced / max(len(y), 1)
            if frac_coerced > 0.5:
                warnings.warn(
                    f"[{source}/{name}] Skipping: {frac_coerced:.0%} of regression "
                    f"targets are non-numeric — not a usable regression target. "
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
                        f"[{source}/{name}] Skipping: {frac_test:.0%} of test regression targets are non-numeric."
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

        # Drop rows where y was non-numeric (NaN after coercion)
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
            X_arr, X_test_arr, y_arr, y_test_arr = train_test_split(
                X_arr, y_arr, test_size=0.2, random_state=self.random_state
            )

        if X_arr.shape[0] > self.max_train_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X_arr.shape[0], self.max_train_samples, replace=False)
            X_arr, y_arr = X_arr[idx], y_arr[idx]

        if X_arr.shape[1] < 1 or X_arr.shape[0] < 10:
            return None

        return DatasetEntry(
            name=name,
            source=source,
            task_type="regression",
            X_train=X_arr,
            y_train=y_arr,
            X_test=X_test_arr,
            y_test=y_test_arr,
            metadata={"n_train_original": X_arr.shape[0], "n_features_original": X_arr.shape[1]},
        )
