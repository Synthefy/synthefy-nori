"""What ``predict`` does when the table is bigger than the context window.

Nori reads at most a window of labeled rows per call. Above that, *something* has to
choose which rows the model sees. Today that choice is made by memory pressure:
``NoriPredictor`` finds the context does not fit its element budget, drops rows at
random to make it fit, and warns (``ContextSubsampledWarning``). The result is the
`random` policy — one arbitrary window — arrived at by accident rather than on purpose.

This module makes it a decision. Above ``large_context_threshold`` rows, a **policy** from
:mod:`synthefy_nori.inference.policies` decides the context and chains the calls, and
the per-call subsample never engages because every context handed to the predictor is
already within the window.

    NoriRegressor(large_context_policy="cluster_route").fit(X, y).predict(X_test)

``cluster_route`` is the default because it is the only arm measured across all 15
benchmark tables without a regression. ``safeboost``, ``boost``, ``random``,
``cluster_route_g4``, a ``holdout_gate`` over any of them, and custom callables
are all selectable. **The menu, each arm's measured cost and its tail risk live in**
``policies.py`` — one home per number, so they cannot drift apart. ``boost``'s tail is
severe; read it there before selecting it.

**This is off by default** (``large_context_policy=None`` keeps today's behavior exactly).
The evidence is a within-checkpoint policy comparison on 15 tables, not a guarantee
about your table, and turning it on multiplies the number of forward passes per predict.
Opt in, and read :attr:`NoriRegressor.large_context_report_` to see what actually ran.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

from synthefy_nori.inference.policies import (
    LargeContextPolicyWarning,
    Policy,
    Problem,
    holdout_gate,
    resolve_policy,
)

class LargeContextUnsupportedOutputError(NotImplementedError):
    """A large-context policy cannot produce the output the caller asked for.

    Its own subclass rather than a bare ``NotImplementedError`` because
    ``_predict_categorical`` wraps that type to blame a bar_distribution checkpoint for
    not exposing a quantile bank. Both refusals surface from the same call, so without a
    distinguishable type a policy problem is reported as a checkpoint problem and the
    remedy points at the wrong subsystem.
    """


DEFAULT_LARGE_CONTEXT_THRESHOLD = 50_000
"""Row count above which a large-context policy engages, when one is selected.

Chosen as the round number a caller asks for, not derived from the evidence: the policies
were benchmarked on tables of 47k-1M rows, so 50k is the bottom of the measured range.

It is a constructor argument because the right value is table- and hardware-dependent. In
particular a boosting chain is governed by ``n_train // window``, not by a row count, and
50k rows at a 10k window is below the floor ``policies.MIN_BOOST_SHARDS`` documents --
which is why the boosting arms warn there rather than this constant trying to encode it.
Routing policies have no such floor.
"""

DEFAULT_LARGE_CONTEXT_POLICY = "cluster_route"
"""The policy used when one is requested without naming which."""


def resolve_large_context_policy(spec) -> tuple[str, Policy]:
    """Turn ``large_context_policy=`` into ``(name, callable)``.

    Accepts everything :func:`~synthefy_nori.inference.policies.resolve_policy` does --
    a registry name, ``"pkg.mod:fn"``, ``"file.py:fn"``, each optionally with
    ``"[k=v,...]"``, or a bare callable -- plus:

    * ``True`` -- the default policy, so ``large_context_policy=True`` means "yes, handle
      large tables sensibly" without the caller having to know the menu.
    * a sequence of specs -- wrapped in a
      :func:`~synthefy_nori.inference.policies.holdout_gate`, which scores each on a
      train holdout and deploys the per-table winner. This is the safe way to use
      ``boost``: the gate declined it on exactly the tables where it detonated.
    """
    if spec is True:
        spec = DEFAULT_LARGE_CONTEXT_POLICY
    if isinstance(spec, (list, tuple, set)):
        candidates = sorted(spec) if isinstance(spec, set) else list(spec)
        if not candidates:
            raise ValueError(
                "large_context_policy=[] selects nothing. Pass a policy name, or None to "
                "leave large-table handling to the memory policy as before."
            )
        names = [resolve_policy(c)[0] for c in candidates]
        return f"gate[{','.join(names)}]", holdout_gate(candidates)
    return resolve_policy(spec)


def large_context_applies(n_train: int, policy_spec, threshold: int) -> bool:
    """Whether ``predict`` should route through a policy for a table this size."""
    return policy_spec is not None and n_train > int(threshold)


def build_problem(
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    window: int,
    seed: int = 0,
    embedder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Problem:
    """The fitted half of a large-context prediction, with no query rows attached yet.

    Hold on to this across ``predict`` calls and pass it to :func:`run_policy`, which
    derives a per-query view with :meth:`Problem.with_queries`. That is what keeps
    train-derived work — the imputed train view, the train routing space, a boosting
    chain's residuals — from being redone on every call.

    Args:
        predict_fn: one Nori call, ``(X_context, y_context, X_query) -> preds``. Bound
            to :meth:`NoriPredictor.predict` on the production path; a stub in tests.
        X_train, y_train: the fitted context, in whatever y-space the caller uses --
            nothing here rescales, so a caller that normalized y gets normalized
            predictions back. Residual boosting is scale-free, so that is safe.
        window: context rows one Nori call may take, normally
            :meth:`NoriPredictor.max_context_rows` -- the same budget that would
            otherwise have trimmed the context sizes it instead.
        seed: fixes the row draws, so two identical predicts agree.
        embedder: optional callable for embedding-space routing. Omitted on the
            production path -- the embed-recycle postmortem predicts no separability
            gain, and it would double the forward passes.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    X_train = np.asarray(X_train)
    return Problem(
        predict_fn,
        X_train,
        y_train,
        np.empty((0, X_train.shape[1]), dtype=np.float32),
        None,  # query labels are what we are predicting
        window=window,
        seed=seed,
        embedder=embedder,
        # NoriPredictor owns missing-value handling on this path; imputing first would
        # change what the model sees relative to an ordinary predict() on the same rows.
        impute=False,
    )


def run_policy(
    base: Problem,
    X_test: np.ndarray,
    *,
    policy_spec,
    seed: int = 0,
    cache_scope: tuple = (),
) -> tuple[np.ndarray, dict]:
    """Run one large-context policy over ``X_test`` and return ``(predictions, report)``.

    ``base`` is a :func:`build_problem` result, reused across calls; this attaches the
    query rows to it, so anything already derived from the fitted table is carried over
    rather than recomputed.

    ``cache_scope`` partitions the cached train-derived *decisions* (a boosting chain, a
    gate's winner) by anything outside the table that changes what ``predict_fn``
    returns. The caller owns it because only the caller knows those inputs. The
    production scope includes both the output decoder and memory policy, whose lossy
    INT8 precision can change residual labels and gate scores. See
    :attr:`~synthefy_nori.inference.policies.Problem.train_cache`.

    Returns:
        ``(preds, report)``. The report names the policy, the window it ran under, and
        ``nori_calls`` for THIS call -- cache builds, which is what a caller comparing
        the policy to a single ``predict`` is paying. A second call on the same
        estimator reports fewer, because the chain it replays was already built.
    """
    name, policy = resolve_large_context_policy(policy_spec)
    # run_seed travels with the query view because the rng below is drawn from it and a
    # boosting chain's shards come from that rng -- so it keys the train cache.
    problem = base.with_queries(X_test, run_seed=seed, cache_scope=cache_scope)
    if problem.n_test == 0:
        # Reachable as predict(X) on an empty frame. Without this, cluster_route dies
        # inside MiniBatchKMeans on an empty query set with nothing naming the cause.
        return np.zeros(0, dtype=np.float64), _report(
            name, problem, problem.window, full_context=False, reused_train_state=False)
    window = problem.window
    if problem.n_train <= window:
        # Nothing to select: the whole table already fits one call. Every policy would
        # degenerate to it, and cluster_route would waste `groups` calls re-predicting
        # partitions of a context it could take whole.
        warnings.warn(
            f"large_context_policy={name!r} was requested for a table of {problem.n_train} "
            f"rows, but all of them fit the {window}-row window, so no policy is needed; "
            f"predicting from full context in one call. Raise large_context_threshold above "
            f"{problem.n_train} to silence this.",
            LargeContextPolicyWarning,
            stacklevel=2,
        )
        preds = problem.predict_arrays(problem.X_train, problem.y_train, problem.X_test)
        return preds, _report(name, problem, window, full_context=True,
                              reused_train_state=False)

    # Whether THIS call actually read train-derived state, not merely whether some is
    # on hand. "Is the cache non-empty?" answers the wrong question in both directions:
    # a second `random` call reuses nothing and a second `cluster_route` call reuses the
    # routing space, yet an entry left by another policy, seed or cache scope makes both
    # report reuse. The counters move only on a read that hit.
    problem.train_state.begin_call()
    problem.train_cache.begin_call()
    preds = np.asarray(policy(problem, np.random.default_rng(seed)), dtype=np.float64)
    reused = (problem.train_state.hits + problem.train_cache.hits) > 0
    if preds.shape != (problem.n_test,):
        raise ValueError(
            f"large-context policy {name!r} returned predictions of shape {preds.shape}, "
            f"expected ({problem.n_test},) -- one per query row."
        )
    report = _report(name, problem, window, full_context=False, reused_train_state=reused)
    winner = getattr(policy, "last_winner", None)
    if winner is not None:
        report["gate_winner"] = winner
    return preds, report


def _report(name: str, problem: Problem, window: int, *, full_context: bool,
            reused_train_state: bool) -> dict:
    return {
        "policy": name,
        "window": window,
        "n_train": problem.n_train,
        "n_test": problem.n_test,
        "shards_available": problem.n_train // window,
        "nori_calls": problem.nori_calls,
        "full_context": full_context,
        "reused_train_state": reused_train_state,
    }


def predictor_call_fn(predictor) -> Callable[..., np.ndarray]:
    """Adapt a predictor into the one primitive a policy calls.

    A policy's ``predict_fn`` is ``(X_context, y_context, X_query) -> flat float64
    ndarray``; :meth:`NoriPredictor.predict` may hand back a torch tensor still on the
    device. Bridging the two is this module's job — it is what makes ``policies.py``
    numpy-only and testable without a checkpoint.

    Kept as a closure over the predictor alone (not the surrounding predict call) so a
    long-lived :class:`Problem` holding this does not pin a query block alive.
    """
    def one_call(X_context, y_context, X_query) -> np.ndarray:
        out = predictor.predict(X_context, y_context, X_query)
        if hasattr(out, "detach"):       # a torch tensor: np.asarray() would raise on CUDA
            out = out.detach().cpu().numpy()
        return np.asarray(out, dtype=np.float64).reshape(-1)

    return one_call


__all__ = [
    "DEFAULT_LARGE_CONTEXT_POLICY",
    "DEFAULT_LARGE_CONTEXT_THRESHOLD",
    "build_problem",
    "large_context_applies",
    "predictor_call_fn",
    "resolve_large_context_policy",
    "run_policy",
]
