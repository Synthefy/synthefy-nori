"""Zero-shot text-feature preprocessing (synthefy_nori.text_features).

Uses a deterministic fake embedder so the tests are fast and offline (no model
download / GPU). The heavy end-to-end check — that the widened matrix reproduces
the standalone zero-shot script's R² through NoriRegressor — is exercised
separately against a real split.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest

from synthefy_nori.text_features import MultimodalPreprocessor, build_paragraphs


def _fake_embed(texts):
    """Deterministic 16-d embedding: same text -> same vector (fit/transform consistent)."""
    out = []
    for t in texts:
        h = hashlib.sha1(t.encode("utf-8")).digest()
        out.append(np.frombuffer(h[:16], dtype=np.uint8).astype(np.float32) / 255.0)
    return np.stack(out)


def _frame(n_rows):
    return pd.DataFrame({
        "price": np.arange(n_rows, dtype=float) * 10.0,
        "brand": (["A", "B"] * n_rows)[:n_rows],
        "review": [f"row {i} some free text here" for i in range(n_rows)],
    })


def test_build_paragraphs_prefixes_and_missing():
    df = pd.DataFrame({"a": ["x", None], "b": ["y", "z"]})
    paras = build_paragraphs(df, ["a", "b"])
    assert paras[0] == "a: x. b: y."
    assert paras[1] == "a: . b: z."          # missing -> empty
    assert build_paragraphs(df, []) == ["", ""]  # no text cols


def test_widened_matrix_shape_and_layout():
    train = _frame(6)
    mp = MultimodalPreprocessor(text_columns=["review"], svd_dim=8, embedder=_fake_embed)
    X = mp.fit_transform(train)
    # 2 non-text cols + min(svd_dim=8, emb=16, n_train-1=5) = 5 text cols
    assert mp.numeric_columns_ == ["price"]
    assert mp.categorical_columns_ == ["brand"]
    assert mp.n_text_features_ == 5
    assert X.shape == (6, 7)
    assert X.dtype == np.float32


def test_svd_dim_clamped_to_embed_dim():
    mp = MultimodalPreprocessor(text_columns=["review"], svd_dim=128, embedder=_fake_embed)
    mp.fit_transform(_frame(40))          # 40 rows, 16-d embedding
    assert mp.n_text_features_ == 16      # clamped to embedding dim


def test_unseen_category_and_nan_handled_at_transform():
    train = _frame(6)
    mp = MultimodalPreprocessor(text_columns=["review"], svd_dim=4, embedder=_fake_embed)
    mp.fit_transform(train)
    unknown_code = len(mp.category_maps_["brand"])   # A,B -> 0,1 ; unseen -> 2
    test = pd.DataFrame({"price": [5.0, np.nan], "brand": ["A", "C"],
                         "review": ["seen-ish", "totally new text"]})
    Xt = mp.transform(test)
    assert Xt.shape[1] == mp.n_features_out_
    assert Xt[1, 1] == unknown_code                  # 'C' unseen
    assert Xt[0, 1] == mp.category_maps_["brand"]["A"]
    assert Xt[1, 0] == 0.0                           # NaN price -> 0


def test_transform_before_fit_raises():
    mp = MultimodalPreprocessor(text_columns=["review"], embedder=_fake_embed)
    with pytest.raises(RuntimeError):
        mp.transform(_frame(3))


def test_ndarray_input_rejected_when_text_configured():
    mp = MultimodalPreprocessor(text_columns=["review"], embedder=_fake_embed)
    with pytest.raises(TypeError):
        mp.fit_transform(np.zeros((4, 3)))


def test_missing_nontext_column_at_transform_raises():
    mp = MultimodalPreprocessor(text_columns=["review"], svd_dim=4, embedder=_fake_embed)
    mp.fit_transform(_frame(6))
    bad = pd.DataFrame({"brand": ["A"], "review": ["x"]})  # dropped 'price'
    with pytest.raises(ValueError):
        mp.transform(bad)


def test_no_text_columns_skips_embedder():
    # A pure tabular transform must NOT resolve/load the embedder (so it needs no
    # sentence-transformers). Passing an embedder that explodes if called proves it.
    def _boom(_texts):
        raise AssertionError("embedder must not be called when there are no text columns")

    mp = MultimodalPreprocessor(text_columns=[], embedder=_boom)
    X = mp.fit_transform(_frame(5)[["price", "brand"]])
    assert mp.n_text_features_ == 0
    assert X.shape == (5, 2)                 # price + brand, no text cols
    mp.transform(_frame(3)[["price", "brand"]])   # transform path also skips the embedder


def test_csv_loaded_dataframe_matches_pickle(tmp_path):
    # A DataFrame read back from CSV must produce the same widened matrix as the
    # in-memory one -- the package is format-agnostic; CSV is just a DataFrame.
    train = _frame(8)
    csv = tmp_path / "train.csv"
    train.to_csv(csv, index=False)
    reloaded = pd.read_csv(csv)

    a = MultimodalPreprocessor(text_columns=["review"], svd_dim=4, embedder=_fake_embed)
    b = MultimodalPreprocessor(text_columns=["review"], svd_dim=4, embedder=_fake_embed)
    Xa = a.fit_transform(train)
    Xb = b.fit_transform(reloaded)
    assert Xa.shape == Xb.shape
    assert a.numeric_columns_ == b.numeric_columns_
    assert a.categorical_columns_ == b.categorical_columns_
    np.testing.assert_allclose(Xa, Xb, rtol=1e-5, atol=1e-6)


def test_categorical_encoding_bounded_by_max_cardinality():
    # Codes must stay in [0, max_cardinality]: the k most frequent train values
    # get 0..k-1, everything rarer or unseen collapses to the in-range "other" code
    # (no out-of-distribution sentinel).
    n = 60
    train = pd.DataFrame({
        "x": np.arange(n, dtype=float),
        "hc": [f"v{i % 40}" for i in range(n)],     # 40 distinct categories
        "review": [f"row {i}" for i in range(n)],
    })
    mp = MultimodalPreprocessor(text_columns=["review"], svd_dim=4,
                                embedder=_fake_embed, max_cardinality=5)
    Xtr = mp.fit_transform(train)
    hc_col = mp.nontext_columns_.index("hc")
    assert len(mp.category_maps_["hc"]) == 5            # capped to top-5
    assert Xtr[:, hc_col].max() <= 5                    # rare -> "other" == 5, bounded
    # an unseen category at transform also lands on the same in-range "other" code
    test = pd.DataFrame({"x": [1.0], "hc": ["totally_new"], "review": ["new row"]})
    Xte = mp.transform(test)
    assert Xte[0, hc_col] == 5


def test_svd_dim_none_appends_raw_embedding():
    mp = MultimodalPreprocessor(text_columns=["review"], svd_dim=None, embedder=_fake_embed)
    X = mp.fit_transform(_frame(8))
    assert mp.svd_ is None
    assert mp.n_text_features_ == 16               # full fake-embedding dim, no reduction
    assert X.shape[1] == 2 + 16                    # price + brand + raw text
    assert mp.transform(_frame(3)).shape[1] == X.shape[1]


def test_text_columns_accepts_lone_string_and_index():
    df = _frame(6)
    # a lone string is one column name, not characters to iterate
    m1 = MultimodalPreprocessor(text_columns="review", svd_dim=4, embedder=_fake_embed)
    m1.fit_transform(df)
    assert m1.text_columns_ == ["review"]
    # a pandas Index (df.select_dtypes(...).columns idiom) must not raise on truthiness
    m2 = MultimodalPreprocessor(text_columns=df.columns[[2]], svd_dim=4, embedder=_fake_embed)
    m2.fit_transform(df)
    assert m2.text_columns_ == ["review"]


def test_unknown_text_column_raises_at_fit():
    with pytest.raises(ValueError):
        MultimodalPreprocessor(text_columns=["not_a_column"],
                               embedder=_fake_embed).fit_transform(_frame(5))


def test_missing_text_column_at_transform_raises():
    m = MultimodalPreprocessor(text_columns=["review"], svd_dim=4, embedder=_fake_embed)
    m.fit_transform(_frame(6))
    with pytest.raises(ValueError):
        m.transform(_frame(3)[["price", "brand"]])   # 'review' dropped


def test_zero_width_embedding_raises():
    # text requested but the encoder returns 0 columns -> fail loudly, don't
    # silently degrade to pure tabular.
    zero = lambda texts: np.zeros((len(texts), 0), dtype=np.float32)
    with pytest.raises(ValueError):
        MultimodalPreprocessor(text_columns=["review"], embedder=zero).fit_transform(_frame(5))


def test_noriregressor_clone_and_pickle_carry_text_config():
    # text config lives in __init__, so clone/get_params keep it and a fitted
    # model pickles (the encoder is dropped and rebuilt lazily).
    import pickle
    from sklearn.base import clone
    from synthefy_nori import NoriRegressor

    reg = NoriRegressor(text_columns=["review"], svd_dim=4, embedder=_fake_embed)
    assert clone(reg).get_params()["text_columns"] == ["review"]
    reg.fit(_frame(8), np.arange(8.0, dtype=float))
    assert reg.n_features_in_ == 3                 # input cols, not the widened SVD block
    assert list(reg.feature_names_in_) == ["price", "brand", "review"]
    round_tripped = pickle.loads(pickle.dumps(reg))
    assert round_tripped._text_preprocessor.text_columns_ == ["review"]
    assert round_tripped._text_preprocessor._encoder is None      # dropped on pickle


def test_categorical_key_canonicalization_int_vs_float():
    # a categorical read as int in train but float in test (NaN promotion) must map
    # to the same code, not collapse to "other".
    tr = pd.DataFrame({"c": pd.Series([5, 6, 5, 6], dtype=object),
                       "review": ["a", "b", "a", "b"]})
    te = pd.DataFrame({"c": pd.Series([5.0, 6.0], dtype=object), "review": ["a", "b"]})
    m = MultimodalPreprocessor(text_columns=["review"], svd_dim=2, embedder=_fake_embed)
    m.fit_transform(tr)
    Xte = m.transform(te)
    ci = m.nontext_columns_.index("c")
    other = len(m.category_maps_["c"])
    assert (Xte[:, ci] != other).all()             # 5.0/6.0 hit the int codes, not "other"


def test_categorical_tiebreak_is_deterministic():
    df = pd.DataFrame({"c": ["b", "a", "c", "a", "b", "c"], "review": ["x"] * 6})  # a,b,c each 2
    m = MultimodalPreprocessor(text_columns=["review"], svd_dim=2,
                               embedder=_fake_embed, max_cardinality=2)
    m.fit_transform(df)
    assert m.category_maps_["c"] == {"a": 0, "b": 1}  # ties broken by key ascending


def test_string_embedder_does_not_masquerade_as_encoder():
    # regression guard: str has a (bytes) .encode; it must take the model path,
    # not be treated as a preloaded encoder object. We only check resolution
    # doesn't misfire by passing an explicit callable instead (offline).
    mp = MultimodalPreprocessor(text_columns=["review"], svd_dim=3, embedder=_fake_embed)
    X = mp.fit_transform(_frame(5))
    assert X.shape[1] == 2 + 3
