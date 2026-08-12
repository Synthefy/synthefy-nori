"""Private compatibility implementation for clients without synthefy.text_features.

Turns free-text columns of a :class:`pandas.DataFrame` into a handful of extra
numeric columns that a frozen, pretrained Nori can consume like any other feature
— no gradient training anywhere:

    text columns
      -> one column-prefixed paragraph per row     (:func:`build_paragraphs`)
      -> frozen sentence embedding                  (a sentence-transformer)
      -> TruncatedSVD to ``svd_dim`` columns        (fit on train only, unsupervised)
      -> appended to the numeric / categorical block

:class:`MultimodalPreprocessor` owns the whole train-only transform so a fitted
instance reproduces it exactly at predict time. Non-text columns are handled the
same way the standalone zero-shot script does — numeric columns pass through,
categorical (string) columns are label-encoded — except the label maps are fit on
train only and unseen categories at transform time map to a reserved code.

``sentence-transformers`` is an optional dependency; it is imported lazily only
when a string / model embedder is used (a plain callable embedder needs nothing).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# short name -> HF model id (embedding backbones)
MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",   # 384-d, fast baseline
    "qwen0.6b": "Qwen/Qwen3-Embedding-0.6B",              # 1024-d, public, fast
    "qwen4b": "Qwen/Qwen3-Embedding-4B",                  # 2560-d, powerful, public
    "qwen8b": "Qwen/Qwen3-Embedding-8B",                  # 4096-d, public
    "gemma": "google/embeddinggemma-300m",                # 768-d — GATED
    "bge-m3": "BAAI/bge-m3",                              # 1024-d, public
    "bge-large": "BAAI/bge-large-en-v1.5",               # 1024-d
}

_MISSING = "__MISSING__"


def _canon_cat(s: pd.Series) -> pd.Series:
    """Canonical string keys for a categorical column, stable across train/test.

    NaN/None -> the _MISSING sentinel; an int-valued float renders like the int
    ("5.0" -> "5"), so a column read as int in one split and float in another
    (e.g. a NaN promoted it) still maps to the same code; everything else via str.
    """
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return _MISSING
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    return s.astype(object).map(one)


def build_paragraphs(df: pd.DataFrame, text_columns) -> list[str]:
    """One column-prefixed paragraph per row: ``"col: value. col2: value2."``.

    Matches the zero-shot pipeline's ``paragraph`` helper. Missing values become
    empty strings; an empty ``text_columns`` yields ``[""] * len(df)``.
    """
    cols = [c for c in (text_columns or []) if c in df.columns]
    if not cols:
        return [""] * len(df)
    # Row-vectorized: the loop is over the (few) text columns, not the rows — each
    # "col: value" is a vectorized pandas string op, joined in one str.cat call.
    # (astype(object) first: fillna on a category-dtype column raises.)
    parts = [str(c) + ": " + df[c].astype(object).fillna("").astype(str) for c in cols]
    joined = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=". ")
    return (joined + ".").tolist()


def _make_encoder(embedder, device=None, *, batch_size: int | None = None,
                  normalize: bool | None = None) -> Callable[[list[str]], np.ndarray]:
    """Resolve ``embedder`` to a ``texts -> (n, dim) float32 ndarray`` callable.

    Accepts a short name / HF id (str), a preloaded SentenceTransformer-like object
    (anything with ``.encode``), or a plain callable ``texts -> ndarray``.
    """
    # plain callable (not a str, not a SentenceTransformer) — used as-is.
    # NB: check str first, since str has a (bytes) .encode method that would
    # otherwise masquerade as a SentenceTransformer-like encoder.
    if not isinstance(embedder, str) and callable(embedder) and not hasattr(embedder, "encode"):
        def _enc_call(texts):
            return np.asarray(embedder(texts), dtype=np.float32)
        return _enc_call

    st = embedder
    model_id = None
    if isinstance(embedder, str):
        # string: lazily load a sentence-transformer
        model_id = MODELS.get(embedder, embedder)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - dependency hint
            raise ImportError(
                "Text features need the optional 'sentence-transformers' package. "
                "Install it with `pip install synthefy-nori[text]` (or "
                "`pip install sentence-transformers`), or pass a preloaded encoder "
                "object / callable as the embedder."
            ) from e
        kwargs = {"trust_remote_code": True}
        if "Qwen" in model_id or "bge-large" in model_id.lower():
            kwargs["model_kwargs"] = {"torch_dtype": "float16"}  # fit large LLM encoders
        st = SentenceTransformer(model_id, device=str(device) if device else "cpu", **kwargs)
        if "Qwen" in model_id or "gemma" in model_id.lower():
            st.max_seq_length = min(getattr(st, "max_seq_length", 512) or 512, 512)

    is_llm = bool(model_id) and (
        "Qwen" in model_id or "gemma" in model_id.lower() or "bge" in model_id.lower())
    bs = batch_size or (8 if (model_id and "Qwen" in model_id) else 32 if is_llm else 256)
    norm = normalize if normalize is not None else is_llm  # cosine-normalize LLM encoders

    def _enc_st(texts):
        return st.encode(texts, batch_size=bs, convert_to_numpy=True,
                         show_progress_bar=False,
                         normalize_embeddings=norm).astype(np.float32)
    return _enc_st


class MultimodalPreprocessor:
    """Fit-once / apply-many transform: DataFrame -> widened numeric matrix.

    ``fit`` / ``fit_transform`` learn the column split, the train-only categorical
    label maps, and the text SVD; ``transform`` replays them on new rows.

    Args:
        text_columns: column names to treat as free text.
        svd_dim: number of SVD columns to append (clamped to the embedding dim and
            ``n_train - 1``).
        embedder: short name / HF id (str), a preloaded SentenceTransformer-like
            object, or a callable ``texts -> ndarray``.
        device: torch device string for a string embedder (ignored otherwise).
        seed: TruncatedSVD ``random_state``.
        max_cardinality: keep only the ``max_cardinality`` most frequent values of
            each categorical column; rarer values AND any value unseen at fit map
            to a single in-range "other" code. This bounds the encoded range so an
            unseen test value can't become an out-of-distribution outlier (which
            otherwise wrecks high-cardinality columns under Nori's context
            normalization). Very high-cardinality columns are better passed as text.
    """

    def __init__(self, text_columns, svd_dim: int | None = 128, embedder="minilm",
                 device=None, seed: int = 0, max_cardinality: int = 128,
                 normalize: bool | None = None):
        # kept raw (None / str / list / Index); resolved to self.text_columns_ at fit
        self.text_columns = text_columns
        self.svd_dim = None if svd_dim is None else int(svd_dim)
        self.embedder = embedder
        self.device = device
        self.seed = int(seed)
        self.max_cardinality = int(max_cardinality)
        # cosine-normalize embeddings: None = auto (on for LLM encoders keyed by a
        # known model id, off otherwise). Set True/False to override — needed for a
        # preloaded encoder OBJECT, whose model id can't be inspected.
        self.normalize = normalize

    def __getstate__(self):
        # Drop the cached encoder callable (a closure / live torch model) so a
        # fitted preprocessor pickles; _embed rebuilds it lazily from self.embedder.
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state

    # -- non-text (numeric passthrough + train-only categorical label maps) --
    def _fit_tabular(self, df: pd.DataFrame) -> None:
        self.nontext_columns_ = [c for c in df.columns if c not in set(self.text_columns_)]
        self.numeric_columns_ = []
        self.categorical_columns_ = []
        # col -> {value: code}; codes are 0..k-1 for the k<=max_cardinality most
        # frequent train values. Rare train values and any unseen value map to the
        # single "other" code len(map) at transform, so the encoded range stays
        # bounded (no out-of-distribution sentinel).
        self.category_maps_ = {}
        for c in self.nontext_columns_:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                self.numeric_columns_.append(c)
            else:
                self.categorical_columns_.append(c)
                vals = _canon_cat(s)
                vc = vals.value_counts()
                # keep the max_cardinality most frequent, breaking count ties by
                # key so the surviving set and their codes are deterministic across
                # runs / pandas versions (value_counts tie order is not stable).
                top = sorted(vc.index, key=lambda k: (-int(vc[k]), k))[: self.max_cardinality]
                self.category_maps_[c] = {v: i for i, v in enumerate(top)}

    def _transform_tabular(self, df: pd.DataFrame) -> np.ndarray:
        # build columns in the original non-text order for a stable feature layout
        out = np.empty((len(df), len(self.nontext_columns_)), dtype=np.float32)
        for j, c in enumerate(self.nontext_columns_):
            if c in self.category_maps_:
                m = self.category_maps_[c]
                other = len(m)  # in-range code for rare/unseen values
                # vectorized dict lookup; unseen -> NaN -> the bounded "other" code
                out[:, j] = _canon_cat(df[c]).map(m).fillna(other).to_numpy(dtype=np.float32)
            else:
                col = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                out[:, j] = col.to_numpy(dtype=np.float32)
        return out

    # -- text (embed + train-fit SVD) --
    def _embed(self, df: pd.DataFrame) -> np.ndarray:
        if getattr(self, "_encoder", None) is None:
            self._encoder = _make_encoder(self.embedder, self.device, normalize=self.normalize)
        return self._encoder(build_paragraphs(df, self.text_columns_))

    def _resolve_text_columns(self, df: pd.DataFrame) -> list[str]:
        """Normalize the `text_columns` argument to a validated list of names.

        Handles None (numeric-only), a lone string (single column), and any
        iterable of names (list / tuple / pandas Index). Raises if a named column
        is absent, rather than silently dropping it. Text columns must be named
        explicitly — the preprocessor does not guess which columns are text.
        """
        tc = self.text_columns
        if tc is None:
            return []
        if isinstance(tc, str):
            tc = [tc]                       # a lone column name, not chars to iterate
        cols = list(tc)                     # handles pandas Index / tuple / ndarray
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"text_columns not found in the DataFrame: {missing}")
        return cols

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "MultimodalPreprocessor requires a pandas DataFrame (so text "
                f"columns can be located by name); got {type(df).__name__}.")
        self._encoder = None
        self.svd_ = None
        self.text_columns_ = self._resolve_text_columns(df)
        self._fit_tabular(df)
        Xnum = self._transform_tabular(df)

        # No text columns -> pure tabular; never resolve/load the embedder (so the
        # numeric+categorical path needs no sentence-transformers dependency).
        if not self.text_columns_:
            self.n_text_features_ = 0
            self.n_features_out_ = Xnum.shape[1]
            return Xnum

        E = self._embed(df)
        if E.shape[1] == 0:
            # text columns were requested but the encoder returned a zero-width
            # embedding — fail loudly rather than silently degrade to pure tabular.
            raise ValueError(
                f"embedder produced a zero-width embedding for text columns "
                f"{self.text_columns_}; check the encoder / that the text is non-empty.")

        if self.svd_dim is None:
            # raw mode: append the full embedding, no reduction
            Xtext = E.astype(np.float32)
        else:
            k = min(self.svd_dim, E.shape[1], max(1, E.shape[0] - 1))
            self.svd_ = TruncatedSVD(n_components=k, random_state=self.seed).fit(E)
            Xtext = self.svd_.transform(E).astype(np.float32)
        self.n_text_features_ = Xtext.shape[1]
        self.n_features_out_ = Xnum.shape[1] + self.n_text_features_
        return np.hstack([Xnum, Xtext]).astype(np.float32)

    def fit(self, df: pd.DataFrame) -> "MultimodalPreprocessor":
        self.fit_transform(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "n_features_out_"):
            raise RuntimeError("Call fit()/fit_transform() before transform().")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "MultimodalPreprocessor.transform requires a pandas DataFrame with "
                f"the same columns seen at fit; got {type(df).__name__}.")
        missing = [c for c in (self.nontext_columns_ + self.text_columns_)
                   if c not in df.columns]
        if missing:
            raise ValueError(f"transform() is missing columns seen at fit: {missing}")
        Xnum = self._transform_tabular(df)
        if not self.text_columns_:          # no text at all -> tabular only
            return Xnum
        E = self._embed(df)
        # raw mode (svd_dim=None) appends the embedding directly; else SVD-transform
        Xtext = E.astype(np.float32) if self.svd_ is None else self.svd_.transform(E).astype(np.float32)
        return np.hstack([Xnum, Xtext]).astype(np.float32)
