"""Zero-shot text features on a synthetic text + numeric dataset.

Nori is a numeric tabular model, but ``NoriRegressor.fit`` can take free-text
columns directly: pass ``text_columns=`` and Nori embeds them with a frozen
sentence encoder, reduces the embedding to ``svd_dim`` columns (TruncatedSVD, fit
on train), and appends them to the numeric/categorical block — no training, the
encoder and Nori stay frozen. ``predict`` replays the same transform.

This example builds a small synthetic dataset whose target depends on numeric
columns (``x1``, ``x2``), a categorical column (``brand``), AND the sentiment word
buried in a free-text ``review``. Nori on the tabular columns alone cannot see the
review, so adding the text feature recovers the missing signal and lifts R².

    uv sync --extra text                              # adds sentence-transformers
    uv run python examples/text_features_synthetic.py

The first run downloads the public ~47MB Nori checkpoint and the MiniLM encoder;
GPU if available, else CPU.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from synthefy_nori import NoriRegressor

# sentiment word -> its (hidden) contribution to the target
_SENTIMENTS = [("terrible", -2.0), ("poor", -1.0), ("okay", 0.0), ("good", 1.0), ("excellent", 2.0)]
_FILLERS = ["fast shipping", "arrived late", "nice packaging", "exactly as described", "would buy again"]
_BRANDS = {"acme": 0.5, "globex": -0.5, "initech": 0.0}


def make_dataset(n: int, seed: int):
    """Synthetic rows with numeric (``x1``,``x2``), categorical (``brand``) and
    free-text (``review``) columns.

    The target is a linear function of the numerics and brand PLUS a large term
    driven by the review's sentiment word — signal that lives only in the text.
    Returns ``(DataFrame, y)``.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    brand = rng.choice(list(_BRANDS), size=n)
    idx = rng.integers(0, len(_SENTIMENTS), size=n)
    words = [_SENTIMENTS[i][0] for i in idx]
    sval = np.array([_SENTIMENTS[i][1] for i in idx])
    fillers = rng.choice(_FILLERS, size=n)
    review = [f"Customer review: the item was {w}. {f}." for w, f in zip(words, fillers)]
    brand_effect = np.array([_BRANDS[b] for b in brand])

    y = 2.0 * x1 - 1.5 * x2 + brand_effect + 3.0 * sval + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "brand": brand, "review": review})
    return df, y.astype(np.float64)


def run(n_train: int = 800, n_test: int = 200, svd_dim: int = 64, device=None, seed: int = 0):
    """Fit Nori with and without the text column; return ``(r2_tabular, r2_text)``."""
    df_train, y_train = make_dataset(n_train, seed)
    df_test, y_test = make_dataset(n_test, seed + 1)
    cols = ["x1", "x2", "brand", "review"]

    tab_cols = ["x1", "x2", "brand"]
    # Tabular-only baseline: review dropped, text_columns=[] -> numeric passthrough
    # + categorical label-encoding, no embedder. Same estimator as the +text run
    # below, just without the text column. Text config lives in the constructor.
    tab = NoriRegressor(device=device, model="nori-6m", text_columns=[]).fit(df_train[tab_cols], y_train)
    r_tab = float(r2_score(y_test, tab.predict(df_test[tab_cols])))

    # + text: same columns plus the review, embedded -> SVD -> appended columns.
    reg = NoriRegressor(device=device, model="nori-6m", text_columns=["review"], svd_dim=svd_dim)
    reg.fit(df_train[cols], y_train)
    r_text = float(r2_score(y_test, reg.predict(df_test[cols])))
    return r_tab, r_text


def main():
    r_tab, r_text = run()
    print(f"Nori tabular-only (x1, x2, brand)   R2 = {r_tab:.4f}")
    print(f"Nori + text review (svd-64)         R2 = {r_text:.4f}")
    print(f"text lift                           {r_text - r_tab:+.4f}")


if __name__ == "__main__":
    main()
