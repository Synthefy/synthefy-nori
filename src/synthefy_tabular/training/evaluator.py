"""TabArena validation evaluation for use during training.

Runs all classification and regression datasets from TabArena,
returning aggregate and per-dataset metrics.
"""

import os
import gc
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

from synthefy_tabular.inference.predictor import LimiXPredictor


def _auc_metric(target, pred):
    """Compute AUC, handling binary and multiclass."""
    try:
        if len(np.unique(target)) > 2:
            return roc_auc_score(target, pred, multi_class='ovo')
        else:
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            return roc_auc_score(target, pred)
    except ValueError:
        return float('nan')


def _eval_cls_dataset(predictor, X_train, y_train, X_test, y_test):
    """Evaluate a single classification dataset. Returns dict or None on skip."""
    le = LabelEncoder()
    scaler = MinMaxScaler()

    # Encode string columns
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            try:
                col_le = LabelEncoder()
                X_train[col] = col_le.fit_transform(X_train[col])
                X_test[col] = col_le.transform(X_test[col])
            except Exception:
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])

    X_train_np = scaler.fit_transform(X_train)
    X_test_np = scaler.transform(X_test)
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    X_train_np = np.asarray(X_train_np, dtype=np.float32)
    y_train_enc = np.asarray(y_train_enc, dtype=np.int64)
    X_test_np = np.asarray(X_test_np, dtype=np.float32)
    y_test_enc = np.asarray(y_test_enc, dtype=np.int64)

    n_classes = len(le.classes_)
    if n_classes > 10 or n_classes < 2:
        return None
    if len(X_train_np) >= 50000:
        return None

    pred = predictor.predict(X_train_np, y_train_enc, X_test_np, task_type="Classification")
    pred_label = np.argmax(pred, axis=1)

    auc = _auc_metric(y_test_enc, pred)
    acc = accuracy_score(y_test_enc, pred_label)

    return {'auc': float(auc), 'acc': float(acc)}


def _preprocess_X(X_train, X_test):
    """Convert DataFrame to float32 numpy, handling categorical/datetime columns.

    - Drop all-NaN columns
    - Drop datetime / datetimetz columns
    - LabelEncode object/category/string columns (drop if >100 unique)
    - Coerce remaining to numeric, fill NaN with median
    """
    import pandas as _pd
    from sklearn.preprocessing import LabelEncoder as _LE

    if isinstance(X_train, np.ndarray):
        return X_train.astype(np.float32), X_test.astype(np.float32)
    # Combine to fit encoders consistently across train/test
    X_train = X_train.copy()
    X_test = X_test.copy()
    # Drop all-NaN
    all_nan_train = X_train.columns[X_train.isna().all()]
    drop_cols = list(all_nan_train)
    # Drop datetime
    dt_cols = list(X_train.select_dtypes(include=["datetime64", "datetimetz"]).columns)
    drop_cols.extend(dt_cols)
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # Label-encode string/object/category columns (drop if >100 unique)
    str_cols = list(X_train.select_dtypes(include=["object", "category", "string"]).columns)
    for col in str_cols:
        try:
            train_vals = X_train[col].fillna("__MISSING__").astype(str)
            test_vals = X_test[col].fillna("__MISSING__").astype(str)
            if train_vals.nunique() > 100:
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
                continue
            le = _LE()
            le.fit(_pd.concat([train_vals, test_vals], ignore_index=True))
            X_train[col] = le.transform(train_vals)
            X_test[col] = le.transform(test_vals)
        except Exception:
            X_train = X_train.drop(columns=[col], errors='ignore')
            X_test = X_test.drop(columns=[col], errors='ignore')

    # Coerce to numeric
    X_train = X_train.apply(_pd.to_numeric, errors="coerce")
    X_test = X_test.apply(_pd.to_numeric, errors="coerce")
    # Fill NaN with train median
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    # Remaining NaN → 0
    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)

    return X_train.to_numpy(dtype=np.float32), X_test.to_numpy(dtype=np.float32)


def _eval_reg_dataset(predictor, X_train, y_train, X_test, y_test):
    """Evaluate a single regression dataset. Returns dict or None on skip."""
    X_train_np, X_test_np = _preprocess_X(X_train, X_test)
    y_train_np = np.asarray(y_train, dtype=np.float64)
    y_test_np = np.asarray(y_test, dtype=np.float64)

    if len(X_train_np) >= 50000:
        return None

    # Normalize y
    y_mean = y_train_np.mean()
    y_std = y_train_np.std()
    if y_std < 1e-12:
        return None
    y_train_norm = (y_train_np - y_mean) / y_std
    y_test_norm = (y_test_np - y_mean) / y_std

    y_pred = predictor.predict(X_train_np, y_train_norm, X_test_np, task_type="Regression")
    y_pred = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else np.asarray(y_pred)
    y_pred = y_pred.squeeze()

    r2 = r2_score(y_test_norm, y_pred)
    return {'r2': float(r2)}


def _load_dataset(data_dir, dataset_name):
    """Load train/test CSVs for a dataset folder."""
    folder = os.path.join(data_dir, dataset_name)
    train_path = os.path.join(folder, f'{dataset_name}_train.csv')
    test_path = os.path.join(folder, f'{dataset_name}_test.csv')

    if not os.path.exists(train_path):
        return None, None, None, None

    train_df = pd.read_csv(train_path)
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(train_df, test_size=0.5, random_state=42)

    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]

    return X_train, y_train, X_test, y_test


def run_full_evaluation(
    model,
    device,
    cls_data_dir=None,
    reg_data_dir=None,
    cls_config_path="config/cls_default_noretrieval.json",
    reg_config_path="config/reg_default_noretrieval.json",
):
    """Run full TabArena evaluation on classification and regression datasets.

    Args:
        model: The model (nn.Module) to evaluate. Will be unwrapped from DDP/compile.
        device: torch.device for inference.
        cls_data_dir: Path to classification dataset folder (or None to skip cls).
        reg_data_dir: Path to regression dataset folder (or None to skip reg).
        cls_config_path: Path to classification inference config JSON.
        reg_config_path: Path to regression inference config JSON.

    Returns:
        dict with keys:
            'mean_auc', 'mean_acc', 'cls_datasets': {name: {auc, acc}},
            'mean_r2', 'reg_datasets': {name: {r2}},
            'elapsed_seconds'
    """
    # Unwrap DDP / torch.compile
    bare_model = model
    if hasattr(bare_model, 'module'):
        bare_model = bare_model.module
    if hasattr(bare_model, '_orig_mod'):
        bare_model = bare_model._orig_mod

    # Training model has mask_prediction=True (returns dict output).
    # Predictor expects mask_prediction=False (returns tensor output).
    # Temporarily disable it for evaluation.
    orig_mask_prediction = getattr(bare_model, 'mask_prediction', False)
    bare_model.mask_prediction = False

    results = {
        'mean_auc': float('nan'),
        'mean_acc': float('nan'),
        'cls_datasets': {},
        'mean_r2': float('nan'),
        'reg_datasets': {},
        'elapsed_seconds': 0,
    }

    start = time.time()
    try:
        # --- Classification ---
        if cls_data_dir and os.path.isdir(cls_data_dir):
            cls_predictor = LimiXPredictor(
                device=device,
                inference_config=cls_config_path,
                model=bare_model,
            )
            aucs, accs = [], []
            for dataset_name in sorted(os.listdir(cls_data_dir)):
                folder_path = os.path.join(cls_data_dir, dataset_name)
                if os.path.isfile(folder_path):
                    continue
                try:
                    X_tr, y_tr, X_te, y_te = _load_dataset(cls_data_dir, dataset_name)
                    if X_tr is None:
                        continue
                    rst = _eval_cls_dataset(cls_predictor, X_tr.copy(), y_tr.copy(),
                                            X_te.copy(), y_te.copy())
                    if rst is None:
                        continue
                    results['cls_datasets'][dataset_name] = rst
                    if np.isfinite(rst['auc']):
                        aucs.append(rst['auc'])
                    accs.append(rst['acc'])
                except Exception as e:
                    print(f"  [EVAL] Error on cls/{dataset_name}: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if aucs:
                results['mean_auc'] = float(np.mean(aucs))
            if accs:
                results['mean_acc'] = float(np.mean(accs))
            del cls_predictor

        # --- Regression ---
        if reg_data_dir and os.path.isdir(reg_data_dir):
            reg_predictor = LimiXPredictor(
                device=device,
                inference_config=reg_config_path,
                model=bare_model,
            )
            r2s = []
            for dataset_name in sorted(os.listdir(reg_data_dir)):
                folder_path = os.path.join(reg_data_dir, dataset_name)
                if os.path.isfile(folder_path):
                    continue
                try:
                    X_tr, y_tr, X_te, y_te = _load_dataset(reg_data_dir, dataset_name)
                    if X_tr is None:
                        continue
                    rst = _eval_reg_dataset(reg_predictor, X_tr.copy(),
                                            y_tr.copy().astype(float),
                                            X_te.copy(), y_te.copy().astype(float))
                    if rst is None:
                        continue
                    results['reg_datasets'][dataset_name] = rst
                    if np.isfinite(rst['r2']):
                        r2s.append(rst['r2'])
                except Exception as e:
                    print(f"  [EVAL] Error on reg/{dataset_name}: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if r2s:
                results['mean_r2'] = float(np.mean(r2s))
            del reg_predictor

        results['elapsed_seconds'] = time.time() - start
        return results
    finally:
        bare_model.mask_prediction = orig_mask_prediction
