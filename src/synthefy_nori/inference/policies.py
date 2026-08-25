"""Shared-pool inference policies for large-context Nori, and the API to add your own.

A **policy** answers one question: given a table far larger than Nori's context
window, which rows go in the shared context, and how are the calls chained? Every
policy uses one context shared across queries, never a per-query context. That keeps
the K/V cache reusable across all queries in a call and preserves the predictor's
cached regression path.

This module is the **shared** home for that logic:
`NoriRegressor(large_context_policy=...)` runs these policies on the inference path (see
`synthefy_nori.inference.large_context`), and the benchmark harness imports from here too.
Keeping one implementation for both roles prevents benchmark and production behavior
from drifting apart: a policy measured in the harness is the exact policy that
deploys.

Five policies are registered. `cluster_route` is the only one with full benchmark
coverage and is the default when the feature is enabled; the rest are opt-in. The
numbers below are from the recorded within-checkpoint policy benchmark:

  random             one random window. The baseline, and the cheapest thing that works:
                     a shared 8k window is within 0.013 R² of full context (median 0.004).
  cluster_route      cluster the QUERIES into `groups` groups, build one shared local pool
                     per group, route each query to its group's cache. Best policy of the
                     nine benchmarked (+0.017 mean Δ vs `random`, best on 7 of 15 tables,
                     never regresses) at `groups` cache builds instead of one. THE DEFAULT.
  cluster_route_g4   `cluster_route` at groups=4. Best mean of the sweep (+0.027) but on
                     the 9-table subset only, and 4 cache builds instead of 8.
  safeboost          residual boosting whose corrections are applied only when they do
                     not hurt out-of-fold R² on the next shard. +0.0146 mean Δ, min Δ
                     0.000 — it keeps boost's upside with no disaster tail — but was
                     measured on 8 of the 15 tables only, so it is opt-in, not default.
  boost              plain residual boosting over successive context shards.
                     **Measured at −0.019 mean Δ and −0.229 min Δ**: bimodal, 5 wins and
                     2 detonations (diamonds 0.946→0.717, BNG(stock) 0.806→0.605). It is
                     registered so the comparison is runnable, NOT because it is
                     deployable — prefer `safeboost`, which dominates it, or gate it.

Both boosting arms shard the table into `n_train // window` disjoint shards. Their
observed failures were at 4–6 shards, so they warn below `MIN_BOOST_SHARDS` (8) and
fall back to `random` below two.

Other evaluated approaches (coreset, stratified-y, per-cluster boosting) either lost
or had severe regressions and are deliberately not shipped. Use this module's
extension point to evaluate a new approach without expanding the built-in menu.

## Writing your own policy

A policy is any callable `(problem, rng) -> np.ndarray` of length `problem.n_test`.
`Problem` gives you the arrays plus the one primitive a policy is allowed to call:

    from synthefy_nori.inference.policies import register_policy

    @register_policy("bag")
    def bag(problem, rng, k=3):
        \"\"\"Average k independent random windows — the cheapest variance reducer.\"\"\"
        windows = [rng.permutation(problem.n_train)[: problem.window] for _ in range(k)]
        return np.mean([problem.predict(w) for w in windows], axis=0)  # k Nori calls

Registration is optional. `NoriRegressor(large_context_policy=...)` also takes a dotted
module path, file path, or callable, so nothing here needs editing to run your own
idea:

    NoriRegressor(large_context_policy="my_policies.py:bag")
    NoriRegressor(large_context_policy="mypkg.mod:my_policy[k=3]")

`Problem.predict(context_idx, query_idx)` chunks huge query blocks and (when
`impute=True`) median-imputes from the context rows as the eval harness does, so a
policy stays about row selection. `Problem.predict_arrays` is the escape hatch for
policies that need synthetic labels (residual boosting) or want to query train rows.

Imputation is a constructor flag because the two callers differ: the benchmark harness
imputes here to match `evaluation.harness._apply_impute`, while the production path
passes `impute=False` — `NoriPredictor` owns missing-value handling there, and imputing
first would change what the model sees relative to an ordinary `predict`.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import r2_score

from synthefy_nori.inference.degradation import DegradedPipelineWarning


class LargeContextPolicyWarning(UserWarning):
    """A large-context policy ran, but outside the regime it was measured in.

    Advisory: the prediction is the policy the caller asked for. Filter this category
    to silence "you are boosting a table with only 5 shards" without silencing the
    substitution below.
    """


class LargeContextPolicyFallbackWarning(DegradedPipelineWarning, LargeContextPolicyWarning):
    """The requested large-context policy could not run and another one was substituted.

    This is a genuine degradation — the caller named one policy and a different one
    produced the numbers — so it joins the :mod:`~synthefy_nori.inference.degradation`
    tree and ``strict_pipeline()`` turns it into an error. It also subclasses
    :class:`LargeContextPolicyWarning`, so a caller filtering large-context advisories catches both.
    """


class LargeContextCallLimitError(RuntimeError):
    """A policy attempted more internal Nori calls than its caller permits."""


QUERY_CHUNK = 25_000
"""Max query rows per Nori call. The transductive RBF+poly preprocessing is
O(n_query * F * poly) and upstream of the transformer (the #235 preprocessing wall),
so a 170k-row query block OOMs a shared GPU. The shared context cache is rebuilt per
chunk, which is cheap at window <= 10k."""


# ----------------------------------------------------------------------- helpers
def r2(y_true, y_pred) -> float:
    """R^2 over the pairwise-finite rows; NaN when fewer than two survive.

    Deliberately sklearn directly rather than `evaluation.metrics.compute_reg_metrics`,
    whose "r2" key this reproduces (same finite mask, same >=2-point rule): this module
    ships to staging and public, where `synthefy_nori.evaluation` carries no `metrics`
    submodule, so importing it would break `safeboost`, `boost` and the gate on
    promotion. The dependency has to point at something every tier has -- and sklearn is
    already a hard dependency (`api.py` imports it at module top).

    The finite mask matters here: a boosting stage grades a running fit that may hold
    non-finite entries, and bare `r2_score` returns NaN for the whole shard if even one
    row is. Below two surviving points the score is NaN, which fails every `>=`
    comparison -- so `safeboost` declines the correction and `boost` stops, both of
    which are the safe reading of "this shard could not be graded".
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(mask.sum()) < 2:      # R^2 needs the variance of >= 2 points
        return float("nan")
    return float(r2_score(y_true[mask], y_pred[mask]))


def column_medians(fit_on: np.ndarray) -> np.ndarray:
    """Per-column medians of `fit_on`, non-finite ignored, all-NaN columns -> 0.

    Split out of `median_impute` so a caller that already has the medians can apply them
    to a new block without re-deriving them -- and, more to the point, without the full
    float32 copy of `fit_on` that `median_impute` produces and such a caller discards.
    """
    with np.errstate(invalid="ignore"):
        # An all-NaN column is expected (and warns); the finite-check below zeroes it.
        med = np.nanmedian(np.where(np.isfinite(fit_on), fit_on, np.nan), axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def apply_medians(med: np.ndarray, X: np.ndarray) -> np.ndarray:
    """`X` as float32 with every non-finite entry replaced by its column's median."""
    X = np.array(X, dtype=np.float32, copy=True)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(med, np.where(bad)[1])
    return X


def median_impute(fit_on: np.ndarray, *apply_to: np.ndarray):
    """Column medians from `fit_on`, applied to every array (non-finite -> median).

    Context-fitted statistics, matching `evaluation.harness._apply_impute` semantics.
    """
    med = column_medians(fit_on)
    return tuple(apply_medians(med, X) for X in (fit_on, *apply_to))


class SharedTrainState(dict):
    """A dict that counts the reads which hit work an EARLIER call left behind.

    The train-derived stores are shared by reference across query views, so "was
    anything in here?" is not the same question as "did this call reuse any of it" --
    an entry left by another policy, seed or cache scope makes the first question true on a
    complete miss. `run_policy` reports the second question, and this counter is how it
    knows: `begin_call()`, run the policy, read `hits`.

    Two kinds of read are deliberately not counted, because neither saved this call any
    work:

    * a **write** -- deriving a chain and storing it is what the cache exists to avoid,
      not evidence of avoiding it;
    * a read of a key **this call derived**. `routing_space()` reads `train_medians`
      twice, once through `select_view` and once for the query half; the second is a
      hit on an entry a microsecond old.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hits = 0
        self._derived_this_call: set = set()

    def begin_call(self) -> None:
        """Start a fresh accounting window: everything stored so far now counts as
        earlier work, and `hits` restarts at zero."""
        self.hits = 0
        self._derived_this_call = set()

    def _count(self, key) -> None:
        if key not in self._derived_this_call:
            self.hits += 1

    def get(self, key, default=None):
        # Not `self[key]` -- that would route through __getitem__ and count twice.
        if dict.__contains__(self, key):
            self._count(key)
            return dict.__getitem__(self, key)
        return default

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)   # a miss raises before it can be counted
        self._count(key)
        return value

    def __setitem__(self, key, value) -> None:
        self._derived_this_call.add(key)
        super().__setitem__(key, value)

    # The remaining mutators route through the counted paths rather than dict's, so a
    # policy that reaches for one of them cannot silently under-report reuse.
    def setdefault(self, key, default=None):
        if dict.__contains__(self, key):
            self._count(key)
            return dict.__getitem__(self, key)
        self[key] = default
        return default

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def pop(self, key, *default):
        # Dropped, so it is no longer this call's derivation -- a later re-derive and
        # re-read must not be counted as reuse of earlier work.
        self._derived_this_call.discard(key)
        return dict.pop(self, key, *default)


def _nearest_centroid(points: np.ndarray, centroids: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Assign each row of `points` to its nearest centroid, chunked (N can be 1M)."""
    out = np.empty(len(points), dtype=int)
    for s in range(0, len(points), chunk):
        block = points[s : s + chunk]
        out[s : s + chunk] = np.argmin(
            ((block[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1
        )
    return out


# ----------------------------------------------------------------------- problem
class Problem:
    """One (table, split) instance, plus the only Nori primitive a policy may call.

    `predict_fn(X_context, y_context, X_query) -> preds` is injected rather than
    imported so the harness can pass a `NoriWrapper` and tests can pass a stub.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: Optional[np.ndarray],
        window: int,
        seed: int = 0,
        embedder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        query_chunk: int = QUERY_CHUNK,
        impute: bool = True,
        max_nori_calls: Optional[int] = None,
    ):
        self.predict_fn = predict_fn
        self.X_train, self.y_train = X_train, np.asarray(y_train, dtype=np.float64)
        self.X_test = X_test
        # None at inference time -- nobody knows the query labels there, and the three
        # policies that could want them do not: boosting grades residuals out-of-fold on
        # TRAIN shards, and the gate scores candidates on a subproblem whose y_test is
        # carved from y_train. Storing zeros instead would let a custom policy read a
        # fabricated target and silently score against it; the property raises instead.
        self._y_test = None if y_test is None else np.asarray(y_test, dtype=np.float64)
        self.window, self.seed = window, seed
        self.embedder, self.query_chunk = embedder, query_chunk
        self.impute = impute
        if max_nori_calls is not None and (
            isinstance(max_nori_calls, bool)
            or not isinstance(max_nori_calls, int)
            or max_nori_calls < 1
        ):
            raise ValueError("max_nori_calls must be a positive integer or None")
        self.max_nori_calls = max_nori_calls
        # The seed the CURRENT run's rng was drawn from. Part of every chain-cache key:
        # a chain's shards come from that rng, so a different seed is a different chain
        # and must not be served from the cache. `run_policy` sets it per call.
        self.run_seed = seed
        self.nori_calls = 0
        self._parent: Optional["Problem"] = None
        # Everything derived from (X_train, y_train) alone lives in these two dicts,
        # which `with_queries` shares BY REFERENCE. They have to be dicts rather than
        # plain attributes: a query view populates them lazily, and if the view held its
        # own copy that work would die with the view and be redone on every predict.
        #   train_state -- derived arrays (imputed train block, routing space, embeddings)
        #   train_cache -- derived decisions (boosting chains, a gate's winner)
        self.train_state = SharedTrainState()
        # Decisions are additionally partitioned by `cache_scope`, because unlike the
        # arrays above they depend on what `predict_fn` returns, not only on the table.
        # See the `cache_scope` / `train_cache` docs below.
        self.cache_scope: tuple = ()
        self._train_caches: dict[tuple, SharedTrainState] = {}
        # Query-derived, deliberately NOT shared: wrong for the next query block.
        self._routing_test: Optional[np.ndarray] = None
        self._embed_test: Optional[np.ndarray] = None

    @property
    def n_train(self) -> int:
        return len(self.y_train)

    @property
    def train_cache(self) -> SharedTrainState:
        """Train-derived DECISIONS — boosting chains, a gate's winner — for the
        current `cache_scope`.

        Scoped, where `train_state` is not, because these are derived from what
        `predict_fn` *returned*, not from the table alone. One estimator can change
        its output decoder or memory policy between calls. Decoder choice changes the
        point estimate, while INT8 cache precision is lossy; either can change a
        chain's residual labels or a gate's winner even with the same fitted table.
        Replaying a decision derived under different predictive settings is therefore
        a wrong answer, not merely stale work. `run_policy(cache_scope=...)` keys those
        settings apart; each scope derives its own decisions once, and switching back
        to an earlier scope replays its decisions rather than rebuilding them.


        The arrays in `train_state` need no such scoping: medians, the imputed view and
        the routing space are functions of `X_train` alone and never touch `predict_fn`.
        """
        cache = self._train_caches.get(self.cache_scope)
        if cache is None:
            cache = self._train_caches[self.cache_scope] = SharedTrainState()
        return cache

    @property
    def y_test(self) -> np.ndarray:
        """Query labels — benchmark only. Raises when they are genuinely unknown."""
        if self._y_test is None:
            raise ValueError(
                "y_test is unknown on this Problem: it was built by the inference path, "
                "where the query labels are what you are predicting. A policy must score "
                "itself out-of-fold on train rows instead — see `Problem.subproblem` "
                "(how `holdout_gate` does it) or `Problem.shards` (how `safeboost` does)."
            )
        return self._y_test

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    def rng(self, offset: int = 0) -> np.random.Generator:
        """Seed-derived generator — use for per-group draws that must be reproducible."""
        return np.random.default_rng(self.seed + offset)

    # -- the Nori primitive ---------------------------------------------------
    def _count_call(self):
        """One shared context = one cache build, however many query chunks it serves.
        A subproblem also bills its parent, so a gate's cost includes its holdout sweep.
        """
        lineage = []
        problem = self
        while problem is not None:
            if (
                problem.max_nori_calls is not None
                and problem.nori_calls >= problem.max_nori_calls
            ):
                raise LargeContextCallLimitError(
                    f"large-context policy exceeded the "
                    f"{problem.max_nori_calls}-call limit"
                )
            lineage.append(problem)
            problem = problem._parent
        for problem in lineage:
            problem.nori_calls += 1

    def predict_arrays(self, X_context, y_context, X_query) -> np.ndarray:
        """One shared-context Nori call over raw arrays; chunks the query block.

        Escape hatch for policies that need labels other than `y_train` (residual
        boosting), want to score train rows, or genuinely need a context larger than
        `window` — unlike `predict`, this does not enforce the window. Callers own
        imputation.
        """
        X_query = np.asarray(X_query)
        self._count_call()
        if len(X_query) <= self.query_chunk:
            return np.asarray(
                self.predict_fn(X_context, y_context, X_query), dtype=np.float64
            ).reshape(-1)
        out = np.empty(len(X_query), dtype=np.float64)
        for s in range(0, len(X_query), self.query_chunk):
            out[s : s + self.query_chunk] = np.asarray(
                self.predict_fn(X_context, y_context, X_query[s : s + self.query_chunk]),
                dtype=np.float64,
            ).reshape(-1)
        return out

    def predict(self, context_idx, query_idx=None, labels=None) -> np.ndarray:
        """Predict test rows `query_idx` (default: all) from shared context rows
        `context_idx`, imputing with the context's own column medians.

        `labels` overrides `y_train[context_idx]` (for residual-style policies).

        Raises if the context exceeds `window`. This harness leaves
        SYNTHEFY_FORBID_SUBSAMPLE unset (every call is meant to be within budget), so an
        oversized context would be SILENTLY subsampled by the predictor and its score
        would not be comparable to the other policies'. Use `predict_arrays` if you
        really mean to exceed the window.
        """
        context_idx = np.asarray(context_idx, dtype=int)
        if len(context_idx) > self.window:
            raise ValueError(
                f"context of {len(context_idx)} rows exceeds window={self.window}; the "
                "predictor would silently subsample it. Use predict_arrays() to opt out."
            )
        X_query = self.X_test if query_idx is None else self.X_test[np.asarray(query_idx, dtype=int)]
        X_context, X_query = self.impute_from_context(self.X_train[context_idx], X_query)
        y_context = self.y_train[context_idx] if labels is None else np.asarray(labels, dtype=np.float64)
        return self.predict_arrays(X_context, y_context, X_query)

    def impute_from_context(self, fit_on: np.ndarray, *apply_to: np.ndarray):
        """`median_impute` when `self.impute`, else a plain float32 coercion.

        The one place the two callers' missing-value semantics diverge; every policy
        goes through here rather than calling `median_impute` directly, so a policy
        does not have to know which caller it is running under.
        """
        if self.impute:
            return median_impute(fit_on, *apply_to)
        return tuple(np.asarray(X, dtype=np.float32) for X in (fit_on, *apply_to))

    # -- sharding (the boosting policies) -------------------------------------
    def shards(self, rng: np.random.Generator, count: Optional[int] = None) -> list:
        """Split train rows into up to `count` disjoint shards of `window` rows.

        `count=None` means "as many as the table affords" (`n_train // window`), which
        is what makes a boosting chain read the whole table. Partial trailing rows are
        dropped: a short final shard would grade the running fit on fewer rows than
        every other stage, and the stage-to-stage OOF comparison assumes a fixed size.
        """
        affordable = self.n_train // self.window
        count = affordable if count is None else max(0, min(int(count), affordable))
        perm = rng.permutation(self.n_train)
        return [perm[i * self.window : (i + 1) * self.window] for i in range(count)]

    # -- selection / routing spaces ------------------------------------------
    @property
    def train_medians(self) -> np.ndarray:
        """Column medians of the train block, cached in `train_state`."""
        med = self.train_state.get("train_medians")
        if med is None:
            med = self.train_state["train_medians"] = column_medians(self.X_train)
        return med

    @property
    def select_view(self) -> np.ndarray:
        """Imputed train features, for distance/clustering math.

        Cached in `train_state`, so the nanmedian plus full float32 copy (~0.5 GB on a
        1M x 130 table) is paid once per fitted table rather than once per predict.
        """
        view = self.train_state.get("select_view")
        if view is None:
            view = self.train_state["select_view"] = apply_medians(
                self.train_medians, self.X_train)
        return view

    def routing_space(self) -> tuple[np.ndarray, np.ndarray]:
        """(train, test) in a common space for clustering: Nori embeddings when an
        embedder was supplied, else train-median-imputed features.

        The two halves are cached in different places on purpose: the train half goes
        in `train_state` and outlives the query block, the test half is local to it.
        """
        train = self.train_state.get("routing_train")
        if train is None:
            # select_view is the same array in the feature path, so share whichever
            # exists rather than allocating a second copy of the imputed train block.
            train = self.train_state["routing_train"] = (
                self.embed("train") if self.embedder is not None else self.select_view)
        if self._routing_test is None:
            # The cached train medians, applied to the query block only -- NOT
            # median_impute(X_train, X_test)[1], which re-derives them and builds a full
            # imputed copy of the train block for the caller to throw away.
            self._routing_test = (self.embed("test") if self.embedder is not None
                                  else apply_medians(self.train_medians, self.X_test))
        return train, self._routing_test

    def adopt_train_state(self, other: "Problem") -> None:
        """Take over another Problem's train-derived ARRAYS, leaving its decisions.

        For the rebuild a changed `window` forces: `select_view`, `train_medians` and
        train embeddings are functions of `X_train` alone, so they are still valid and
        re-deriving them costs a nanmedian plus a full float32 copy (~0.5 GB on a
        1M x 130 table) for nothing. The cached DECISIONS are not carried: a boosting
        chain's shards are window-sized, so a new window is a different chain.
        """
        if other.X_train is not self.X_train:
            raise ValueError(
                "adopt_train_state expects the same fitted table -- these arrays are "
                "only valid for the X_train they were derived from."
            )
        self.train_state = other.train_state

    def with_queries(self, X_test: np.ndarray,
                     run_seed: Optional[int] = None,
                     cache_scope: Optional[tuple] = None) -> "Problem":
        """This same fitted table, pointed at a new query block.

        Everything derived from `(X_train, y_train)` alone carries over — the imputed
        train view, the train routing space, train embeddings, boosting chains and a
        gate's winner. It carries over by SHARING `train_state`/`train_cache`, not by
        copying them, so work this view does lazily is visible to every later query set.
        Everything query-derived is dropped, because it is wrong for the new block.

        This is what makes fit-once/predict-many affordable: without it every `predict`
        re-imputes the whole train table and re-derives the boosting chain's residuals
        from scratch, which is the dominant cost on a long table (the chain's
        train-side decode is O(shards^2 x window) rows).

        `run_seed` records which seed this run's rng was drawn from; it is part of every
        train-cache key, so a re-run under a different seed builds its own chain instead of
        being served the first one's. Defaults to this Problem's own seed.

        `cache_scope` selects which partition of the cached DECISIONS this view reads and
        writes — anything outside the table that changes what `predict_fn` returns, such
        as the decoder or a lossy memory precision (see `train_cache`). Defaults to this
        Problem's own scope. Derived ARRAYS are unscoped and shared across all of them.

        `nori_calls` starts at zero — it measures what THIS prediction costs, which is
        the number a caller comparing a policy to a single `predict` wants.
        """
        fresh = Problem(
            self.predict_fn,
            self.X_train, self.y_train,
            np.asarray(X_test), None,
            window=self.window, seed=self.seed, embedder=self.embedder,
            query_chunk=self.query_chunk, impute=self.impute,
            max_nori_calls=self.max_nori_calls,
        )
        # BY REFERENCE, both of them: the view is throwaway, so anything it derives
        # lazily has to land in state the fitted Problem keeps. Assigning copies here is
        # the bug this replaced -- select_view was recomputed on every single predict.
        # The whole scope->cache mapping is shared, not just the active scope's, so a
        # call that flips back to an earlier scope still finds that scope's chain.
        fresh.train_state = self.train_state
        fresh._train_caches = self._train_caches
        fresh.cache_scope = self.cache_scope if cache_scope is None else tuple(cache_scope)
        fresh.run_seed = self.run_seed if run_seed is None else int(run_seed)
        return fresh

    def embed(self, which: str) -> np.ndarray:
        """Nori 128-d target-token embeddings — sanctioned for clustering/routing
        only, never as regression input (embed-recycle postmortem)."""
        if self.embedder is None:
            raise ValueError("no embedder was supplied to this Problem")
        cached = (self.train_state.get("embed_train") if which == "train"
                  else self._embed_test)
        if cached is not None:
            return cached
        fit = getattr(self.embedder, "fit_context", None)
        if fit is not None and not getattr(self.embedder, "is_fit", True):
            fit(self.X_train, self.y_train)
        X = self.X_train if which == "train" else self.X_test
        embedded = np.asarray(self.embedder(X))
        if which == "train":
            self.train_state["embed_train"] = embedded
        else:
            self._embed_test = embedded
        return embedded

    def subproblem(self, train_idx, test_idx) -> "Problem":
        """A Problem over row subsets — how `holdout_gate` scores candidates without
        touching the real test set. Its Nori calls are billed to this Problem too."""
        train_idx, test_idx = np.asarray(train_idx, int), np.asarray(test_idx, int)
        sub = Problem(
            self.predict_fn,
            self.X_train[train_idx],
            self.y_train[train_idx],
            self.X_train[test_idx],
            self.y_train[test_idx],
            self.window,
            self.seed,
            self.embedder,
            self.query_chunk,
            self.impute,
            self.max_nori_calls,
        )
        sub._parent = self
        return sub


# ----------------------------------------------------------------------- registry
Policy = Callable[[Problem, np.random.Generator], np.ndarray]

POLICIES: dict[str, Policy] = {}


def register_policy(name: str):
    """Decorator: add a policy to the registry so `--policies <name>` finds it."""

    def wrap(fn: Policy) -> Policy:
        if name in POLICIES:
            raise ValueError(f"policy {name!r} already registered")
        POLICIES[name] = fn
        return fn

    return wrap


# ----------------------------------------------------------------------- policies
@register_policy("random")
def random_window(problem: Problem, rng: np.random.Generator) -> np.ndarray:
    """One random shared window of `problem.window` rows. One Nori call.

    The easiest policy, and a strong one: an 8k shared window is within 0.013 mean R²
    of full context on 59 datasets, ~4x better than isab's -0.048 arch tax (#235).
    """
    pool = rng.permutation(problem.n_train)[: problem.window]
    return problem.predict(pool)


@register_policy("cluster_route")
def cluster_route(problem: Problem, rng: np.random.Generator, groups: int = 8) -> np.ndarray:
    """Clustered shared pools: `groups` query clusters, one shared local pool each.

    Cluster the QUERY rows into `groups` groups, assign every train row to its nearest
    query centroid, and give each group a shared pool of `window` rows drawn from its
    own region (backfilled with the rows nearest the centroid when a region is too
    small). One cache build per group; each query still decodes against a shared
    context, so cache-sharing is intact.

    Best policy in the 15-table benchmark: +0.017 mean Δ vs `random`, best on 7 of the
    15 tables, and its worst table is 0.000 — it never regresses. `groups` is a
    per-table knob (Buzz likes 4, nyc-taxi likes 16); 8 is the robust default.
    """
    train_space, test_space = problem.routing_space()
    groups = max(1, min(int(groups), problem.n_test))
    km = MiniBatchKMeans(n_clusters=groups, random_state=problem.seed, n_init=3).fit(test_space)
    query_group, centroids = km.labels_, km.cluster_centers_
    train_group = _nearest_centroid(train_space, centroids)

    preds = np.zeros(problem.n_test, dtype=np.float64)
    for g in range(groups):
        queries = np.flatnonzero(query_group == g)
        if not len(queries):
            continue
        pool = np.flatnonzero(train_group == g)
        if len(pool) < problem.window:
            # Region too thin: fall back to the train rows nearest this centroid, in
            # the SAME space the clustering used (so an embedder stays consistent).
            d2 = ((train_space - centroids[g]) ** 2).sum(1)
            pool = np.argsort(d2)[: problem.window]
        else:
            pool = problem.rng(g + 1).permutation(pool)[: problem.window]
        preds[queries] = problem.predict(pool, query_idx=queries)
    return preds


@register_policy("cluster_route_g4")
def cluster_route_g4(problem: Problem, rng: np.random.Generator, groups: int = 4) -> np.ndarray:
    """`cluster_route` at groups=4 — the coarse-routing benchmark variant.

    Best MEAN Δ of the whole sweep (+0.027 vs `random`, min Δ 0.000), but measured on
    the 9-table discriminating subset only, which is why `cluster_route` (groups=8) —
    +0.017 across all 15 — stays the default rather than this.

    Worth running because the group count is a genuine per-table knob and the coarse end
    is where the biggest single win lives: Buzz scores 0.942 at G=4 vs 0.834 at G=8 and
    0.875 at G=16. Tables with many small regimes go the other way (nyc-taxi peaks at
    G=16, 0.640). Cheaper too — 4 cache builds instead of 8. Gate over
    `random,cluster_route,cluster_route_g4` to pick per table instead of guessing.

    `groups` is accepted so `cluster_route_g4[groups=N]` does the obvious thing rather
    than erroring, but prefer `cluster_route[groups=N]` for anything other than 4 — the
    recorded method name should not say `g4` while running a different count.
    """
    return cluster_route(problem, rng, groups=groups)


# ----------------------------------------------------------------------- boosting
MIN_BOOST_SHARDS = 8
"""Below this many shards a boosting chain is outside its measured-good regime.

Both observed benchmark failures were at 4–6 shards. Since shards are
`n_train // window`, this is really a statement about the table: at a 10k window it
takes about 80k rows to earn 8 stages. Fewer than this warns rather than refuses —
the caller may have measured their own table — and fewer than 2 falls back to `random`,
which is arithmetic: one shard has nothing to correct.
"""


def _check_shard_count(shards: list, name: str) -> bool:
    """True if the chain should run. Warns inside the danger zone, False below 2."""
    if len(shards) < 2:
        warnings.warn(
            f"{name}: the table affords {len(shards)} shard(s) of `window` rows, so there "
            f"is no second shard to correct against; falling back to `random`. Lower the "
            f"large-context threshold only if the table is large enough to shard.",
            LargeContextPolicyFallbackWarning,
            stacklevel=3,
        )
        return False
    if len(shards) < MIN_BOOST_SHARDS:
        warnings.warn(
            f"{name}: running a boosting chain over {len(shards)} shards, below the "
            f"{MIN_BOOST_SHARDS}-shard measured safe floor (both observed benchmark "
            f"failures were at 4-6 shards). Prefer `cluster_route` on a table this "
            f"size, or raise the large-context threshold.",
            LargeContextPolicyWarning,
            stacklevel=3,
        )
    return True


def _boost_stage(problem: Problem, shard, labels, later):
    """One boosting stage: `(test_contribution, later_train_contribution)`.

    A stage queries the test rows AND every not-yet-consumed train row in ONE Nori
    call: the test rows are the answer, the later train rows carry the running fit
    forward so the next stage can grade it out-of-fold. Both are imputed from the SAME
    context statistics, which is why it is one call and one impute rather than two.
    """
    X_ctx, X_test, X_later = problem.impute_from_context(
        problem.X_train[shard], problem.X_test, problem.X_train[later]
    )
    h = problem.predict_arrays(X_ctx, labels, np.vstack([X_test, X_later]))
    return h[: problem.n_test], h[problem.n_test:]


def replay_chain(problem: Problem, stages) -> np.ndarray:
    """Apply an already-built boosting chain to this Problem's query rows.

    A chain is `[(shard_idx, labels, weight), ...]`, and every element of it — which
    shards, each stage's residual labels, whether a guarded stage was accepted, where
    early stopping cut the chain — is derived from the TRAIN rows alone. So a chain
    built while serving one query block is exactly the chain a different query block
    would have produced, and replaying it costs one call per stage with no residuals to
    re-derive and no train rows to decode (the O(shards^2 x window) half of the work).

    Equivalent to the fused build, not an approximation: preprocessing is inductive
    (fitted on the context, never on the query block — `_fit_transform_step_inductive`),
    so a query row scores the same whether it shared its call with the later train rows
    or not, exactly as the existing `query_chunk` splitting already relies on.
    """
    preds = np.zeros(problem.n_test, dtype=np.float64)
    for shard, labels, weight in stages:
        preds += weight * problem.predict(shard, labels=labels)
    return preds


@register_policy("safeboost")
def safeboost(problem: Problem, rng: np.random.Generator, nu: float = 0.5,
              shards: Optional[int] = None) -> np.ndarray:
    """Residual boosting with an out-of-fold guard on every correction.

    Shard the table, predict from shard 1 at full weight, then let each later shard
    learn the *residual* of the running fit and correct everyone — but apply a stage's
    correction only when it does not hurt R² on the next (still unseen) shard. A stage
    that would amplify noise is simply skipped.

    That guard is the whole difference from `boost`, and it is what makes this the
    boosting arm worth deploying: +0.0146 mean Δ vs `random` with **min Δ 0.000**,
    keeping boost's heterogeneous-table wins (nyc 0.596 vs boost 0.575, Buzz 0.816 vs
    0.790) while refusing its detonations (diamonds 0.947 vs boost's 0.717; BNG(stock)
    0.812 vs 0.605). Its caveat is coverage, not tail: 8 of the 15 tables, which is why
    `cluster_route` (+0.017 over all 15) is the default and this is opt-in.

    Cost is one Nori call per shard — `n_train // window`, so ~50 on a 500k table at a
    10k window. That is far more than `cluster_route`'s 8 and is the other reason it is
    not the default.

    The base stage is applied at FULL weight and only later corrections are shrunk by
    `nu`. Shrinking the base too leaves the running fit at a fraction of the target's
    scale and can collapse R² to approximately zero.
    """
    key = ("safeboost", nu, shards, problem.run_seed)
    cached = problem.train_cache.get(key)
    if cached is not None:
        return replay_chain(problem, cached)

    chain = problem.shards(rng, shards)
    if not _check_shard_count(chain, "safeboost"):
        return random_window(problem, rng)

    stages: list = []
    F_test = np.zeros(problem.n_test, dtype=np.float64)
    F_train = np.zeros(problem.n_train, dtype=np.float64)
    # The last shard is graded against, never taught on: a correction it learned would
    # have no later shard left to validate it, so it would go in unguarded.
    for k in range(len(chain) - 1):
        shard, nxt = chain[k], chain[k + 1]
        later = np.concatenate(chain[k + 1:])
        labels = problem.y_train[shard] - F_train[shard]
        h_test, h_later = _boost_stage(problem, shard, labels, later)
        weight = 1.0 if k == 0 else nu
        if k == 0:
            F_train[later] += weight * h_later
            F_test += weight * h_test
            stages.append((shard, labels, weight))
            continue
        oof_before = r2(problem.y_train[nxt], F_train[nxt])
        candidate = F_train.copy()
        candidate[later] += weight * h_later
        if r2(problem.y_train[nxt], candidate[nxt]) >= oof_before - 1e-4:
            F_train = candidate
            F_test += weight * h_test
            stages.append((shard, labels, weight))
    problem.train_cache[key] = stages
    return F_test


@register_policy("boost")
def boost(problem: Problem, rng: np.random.Generator, nu: float = 0.5,
          shards: Optional[int] = None, patience: int = 3) -> np.ndarray:
    """Plain residual boosting — **measured negative on the mean; prefer `safeboost`**.

    Each stage grades the running fit on a fresh shard, learns the residual pattern,
    and corrects everyone by `nu` times the predicted error. Early stopping ends the
    chain after `patience` stages that fail to improve out-of-fold R², and the best
    earlier chain is retained.

    Registered so the comparison stays runnable and so a caller who has measured their
    own table can opt in — **not because it is deployable by default.** Across the
    15-table benchmark it scored −0.019 mean Δ versus one random window, with a −0.229
    minimum: bimodal, with 5 wins and 2 severe regressions on saturated tables.
    `safeboost` preserves the upside without that observed tail; `holdout_gate` is the
    other safe way to use this arm because it can decline it on tables where it
    underperforms.
    """
    key = ("boost", nu, shards, patience, problem.run_seed)
    cached = problem.train_cache.get(key)
    if cached is not None:
        return replay_chain(problem, cached)

    chain = problem.shards(rng, shards)
    if not _check_shard_count(chain, "boost"):
        return random_window(problem, rng)

    stages: list = []
    F_test = np.zeros(problem.n_test, dtype=np.float64)
    F_train = np.zeros(problem.n_train, dtype=np.float64)
    # `best_*` snapshots the chain at its best out-of-fold point; everything after it is
    # discarded. `best_len` is the same cut expressed as a stage count, so the recorded
    # chain replays to exactly `best_test` and not to the longer chain that was walked.
    best_test, best_len, best_oof, stale = F_test.copy(), 0, -np.inf, 0
    for k, shard in enumerate(chain):
        labels = problem.y_train[shard] - F_train[shard]
        later = (np.concatenate(chain[k + 1:]) if k + 1 < len(chain)
                 else np.empty(0, dtype=int))
        h_test, h_later = _boost_stage(problem, shard, labels, later)
        F_test = F_test + nu * h_test
        stages.append((shard, labels, nu))
        if len(later):
            F_train[later] += nu * h_later
        if k + 1 >= len(chain):
            best_test, best_len = F_test.copy(), len(stages)
            break
        # Grade the running fit on the NEXT shard before teaching on it -- the only
        # honestly out-of-fold estimate available mid-chain. Measured performance can
        # reverse after depth saturates, so stopping early is part of the recipe, not
        # merely an optimization.
        oof = r2(problem.y_train[chain[k + 1]], F_train[chain[k + 1]])
        if oof > best_oof + 1e-4:
            best_oof, best_test, best_len, stale = oof, F_test.copy(), len(stages), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if not best_len:
        # No stage ever improved out-of-fold R2 -- reachable when every OOF score is NaN
        # (a constant shard target, say). best_test is still all zeros, so returning it
        # would serve a silent all-zero prediction and cache an empty chain that replays
        # zeros forever. Fall back instead, loudly.
        warnings.warn(
            "boost: no stage improved out-of-fold R2, so the chain is empty and would "
            "predict all zeros; falling back to `random`. This usually means the target "
            "is constant or near-constant within a shard.",
            LargeContextPolicyFallbackWarning,
            stacklevel=2,
        )
        return random_window(problem, rng)
    problem.train_cache[key] = stages[:best_len]
    return best_test


# ----------------------------------------------------------------------- combinator
def holdout_gate(candidates: Sequence[str | Policy], holdout: int = 2000) -> Policy:
    """Build a meta-policy that picks the per-table winner on a train holdout.

    Carves up to `holdout` rows out of train — capped so at least one full `window`
    remains behind for the candidates to build a context from — scores every candidate
    on them, then re-runs the winner on the real test set. This is how a custom policy
    earns its way into production without a global claim: on the 15-table run the gate scored +0.0166,
    matching the best single policy, because it deployed the strong arm where it won
    and fell back where it would have blown up (diamonds: 0.947 not 0.717).

    Cost is the whole menu PLUS the winner run again — the winner is evaluated twice, once
    on the holdout and once for real. `nori_calls` prices this correctly (a subproblem
    bills its parent), so a gate over 3 arms reads ~2x the winner's own cost.

    The subproblem keeps the parent's `window` on purpose: the gate has to score the exact
    policy configuration it will deploy, and shrinking the window would estimate a
    different policy. The cost is that on a table whose regions are already thinner than
    `window`, routing policies hit their backfill path during gating and their candidate
    pools overlap more than they will at deploy time — the gate is conservative there,
    not wrong.

    The winner is recorded on the returned function as `.last_winner`.
    """
    resolved = [resolve_policy(c) for c in candidates]

    def gate(problem: Problem, rng: np.random.Generator) -> np.ndarray:
        # The winner is decided on a holdout carved out of TRAIN, so it is a function of
        # the fitted table alone -- exactly like a boosting chain, and cached the same
        # way. Without this the sweep re-runs on every predict and the gate costs its
        # whole menu forever instead of once (measured: 33.8s cold, 32.4s warm).
        key = ("gate", tuple(name for name, _ in resolved), holdout, problem.run_seed)
        by_name = dict(resolved)
        cached = problem.train_cache.get(key)
        if cached is not None:
            gate.last_winner = cached
            return by_name[cached](problem, rng)

        # Never hold out so much that the candidates are left without a full window to
        # fill their context from. At n_train <= holdout the uncapped draw takes every
        # row, and each candidate is then scored on an EMPTY context -- a real predictor
        # raises there, and one that tolerates it picks a winner off noise.
        n_held = min(holdout, problem.n_train - problem.window)
        if n_held < 1:
            raise ValueError(
                f"holdout_gate needs more than window={problem.window} train rows to "
                f"carve a holdout from, but this table has {problem.n_train}. At that "
                "size the whole table fits one call and no policy is needed."
            )
        if n_held < holdout:
            warnings.warn(
                f"holdout_gate: {problem.n_train} train rows at window="
                f"{problem.window} leaves only {n_held} for the holdout, not "
                f"{holdout}; the winner is chosen on that much evidence. Lower "
                "`window`, or select a policy directly instead of gating.",
                LargeContextPolicyWarning,
                stacklevel=2,
            )
        held = problem.rng(7).permutation(problem.n_train)[:n_held]
        keep = np.setdiff1d(np.arange(problem.n_train), held, assume_unique=False)
        sub = problem.subproblem(keep, held)
        best_name, best_fn, best_score = None, None, -np.inf
        for name, fn in resolved:
            try:
                score = r2(sub.y_test, fn(sub, np.random.default_rng(problem.seed + 11)))
            except LargeContextCallLimitError:
                raise
            except Exception as exc:  # noqa: BLE001 - a broken candidate must not sink the gate
                print(f"    gate candidate {name} FAILED: {exc}", flush=True)
                continue
            print(f"    gate holdout {name}: {score:.4f}", flush=True)
            if score > best_score:
                best_name, best_fn, best_score = name, fn, score
        if best_fn is None:
            raise RuntimeError("every gate candidate failed on the holdout")
        problem.train_cache[key] = best_name
        gate.last_winner = best_name
        return best_fn(problem, rng)

    gate.last_winner = None
    return gate


# ----------------------------------------------------------------------- resolution
def _parse_params(text: str) -> dict:
    """`groups=16,tol=0.5,flag=true` -> dict with ints/floats/bools/strings."""
    params = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"policy parameter {item!r} is not key=value")
        key, raw = (s.strip() for s in item.split("=", 1))
        if raw.lower() in ("true", "false"):
            params[key] = raw.lower() == "true"
            continue
        for cast in (int, float):
            try:
                params[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            params[key] = raw
    return params


def _load_attr(location: str, attr: str) -> Policy:
    """Import `attr` from a dotted module path or a .py file path."""
    if location.endswith(".py") or "/" in location:
        path = Path(location).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"no policy file at {path}")
        name = f"_shared_subset_policies_{path.stem}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(location)
    if not hasattr(module, attr):
        raise AttributeError(f"{location} has no attribute {attr!r}")
    return getattr(module, attr)


def resolve_policy(spec: str | Policy) -> tuple[str, Policy]:
    """Turn a policy spec into `(display_name, callable)`.

    Accepts, in order of precedence:
      * a callable                      -> used as-is
      * `"random"`                      -> a name in POLICIES
      * `"pkg.mod:fn"` / `"file.py:fn"` -> imported (importing a module also runs any
                                           `@register_policy` in it)
      * any of the above with `"[k=v,...]"` appended -> partially applied

    Raises ValueError on an unknown name, listing what is available.
    """
    if callable(spec):
        return getattr(spec, "__name__", "policy"), spec

    text = str(spec).strip()
    params: dict = {}
    raw = ""
    if text.endswith("]") and "[" in text:
        text, raw = text[:-1].split("[", 1)
        params = _parse_params(raw)

    label = text
    if ":" in text:
        location, attr = text.rsplit(":", 1)
        fn = _load_attr(location, attr)
        if location.endswith(".py") or "/" in location:
            # Record the BASENAME, not the path the user happened to type: this string
            # lands in the jsonl `method` field, becomes the resume key, and is a column
            # header in analyze.py — an absolute path makes all three unreadable.
            label = f"{Path(location).name}:{attr}"
    elif text in POLICIES:
        fn = POLICIES[text]
    else:
        raise ValueError(
            f"unknown policy {text!r}. Built-ins: {sorted(POLICIES)}. "
            "For your own, pass 'pkg.module:function' or 'path/to/file.py:function'."
        )

    if params:
        signature = inspect.signature(fn)
        takes_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD
                           for p in signature.parameters.values())
        unknown = set() if takes_kwargs else set(params) - set(signature.parameters)
        if unknown:
            raise ValueError(f"policy {text!r} has no parameter(s) {sorted(unknown)}")
        fn = partial(fn, **params)
    return (f"{label}[{raw}]" if params else label), fn
