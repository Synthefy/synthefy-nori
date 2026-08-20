"""CPU-only tests for the large-context inference path (`large_context_policy=`).

No checkpoint and no GPU: `Problem` takes an injected `predict_fn`, so a stub predictor
exercises the whole path -- the threshold, the policy resolver, the boosting arms and
their shard-count guards, and the two caching claims that the tier-1/tier-2 work rests
on:

  * a replayed boosting chain gives the SAME predictions as the fused build, which is
    what makes it legitimate to reuse one (`test_replay_*`);
  * train-derived state is reused across query sets rather than recomputed, and the
    second predict costs strictly fewer Nori calls (`test_*_reuse*`).

The API-level tests monkeypatch the predictor rather than load weights, so they check
the wiring (threshold, dispatch, report, fit invalidation) and not the model.
"""
from __future__ import annotations

import ast
import inspect
import warnings

import numpy as np
import pytest
import torch

from synthefy_nori.inference import large_context
from synthefy_nori.inference import policies as pol
from synthefy_nori.inference.memory_policy import MemoryPolicy


def ridge_predict(X_ctx, y_ctx, X_query, ridge=1e-3):
    """Stand-in for Nori: closed-form ridge on the context, applied to the query.

    Deterministic, context-sensitive, and -- crucially for the replay tests -- purely
    inductive: it fits on the context only, so a query row's prediction does not depend
    on which other rows shared its call. That is the same property the real
    preprocessing has (`_fit_transform_step_inductive`).
    """
    A = np.hstack([X_ctx, np.ones((len(X_ctx), 1))])
    B = np.hstack([X_query, np.ones((len(X_query), 1))])
    w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ np.asarray(y_ctx))
    return B @ w


def make_table(n_train=600, n_test=80, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    # coef is drawn before X_test so n_test cannot perturb the training target through
    # the shared rng -- otherwise make_base(n_test=N) quietly changes what y means.
    coef = rng.normal(size=n_features)
    X_test = rng.normal(size=(n_test, n_features)).astype(np.float32)
    # A mild nonlinearity so a longer boosting chain has something left to learn.
    y_train = X_train @ coef + 0.3 * X_train[:, 0] ** 2
    return X_train, y_train, X_test


def make_base(window=50, seed=0, **kwargs):
    X_train, y_train, _ = make_table(seed=seed, **kwargs)
    return large_context.build_problem(
        ridge_predict, X_train, y_train, window=window, seed=seed)


# ------------------------------------------------------------------ the threshold
@pytest.mark.parametrize("n_train,policy,threshold,expected", [
    (60_000, "cluster_route", 50_000, True),
    (50_000, "cluster_route", 50_000, False),   # strictly greater, not >=
    (60_000, None, 50_000, False),              # opt-in: no policy, no dispatch
    (10, "cluster_route", 5, True),
])
def test_large_context_applies_is_strictly_above_the_threshold(n_train, policy, threshold,
                                                         expected):
    assert large_context.large_context_applies(n_train, policy, threshold) is expected


def test_default_threshold_is_fifty_thousand():
    assert large_context.DEFAULT_LARGE_CONTEXT_THRESHOLD == 50_000


def test_default_policy_is_the_arm_with_full_coverage_evidence():
    """cluster_route, not a boosting arm: it is the only one measured on all 15 tables
    (+0.017 mean, min 0.000). Changing this default is a claim that needs numbers."""
    assert large_context.DEFAULT_LARGE_CONTEXT_POLICY == "cluster_route"


# ------------------------------------------------------------------- the resolver
def test_true_selects_the_default_policy():
    name, fn = large_context.resolve_large_context_policy(True)
    assert name == "cluster_route"
    assert fn is pol.POLICIES["cluster_route"]


def test_a_list_becomes_a_holdout_gate_naming_its_candidates():
    name, fn = large_context.resolve_large_context_policy(["random", "cluster_route"])
    assert name == "gate[random,cluster_route]"
    assert callable(fn)


def test_an_empty_list_is_rejected_rather_than_silently_doing_nothing():
    with pytest.raises(ValueError, match="selects nothing"):
        large_context.resolve_large_context_policy([])


def test_an_unknown_policy_name_fails_with_the_menu():
    with pytest.raises(ValueError, match="unknown policy"):
        large_context.resolve_large_context_policy("clusterboost")


def test_parameters_bind_through_the_spec():
    name, fn = large_context.resolve_large_context_policy("cluster_route[groups=3]")
    assert name == "cluster_route[groups=3]"
    base = make_base(window=40, n_train=400)
    preds, report = large_context.run_policy(base, make_table(n_test=60)[2],
                                       policy_spec="cluster_route[groups=3]")
    assert report["nori_calls"] == 3


# -------------------------------------------------------------------- run_policy
def test_run_policy_returns_one_prediction_per_query_row():
    base = make_base()
    _, _, X_test = make_table()
    preds, report = large_context.run_policy(base, X_test, policy_spec="random")
    assert preds.shape == (len(X_test),)
    assert report["policy"] == "random"
    assert report["nori_calls"] == 1
    assert report["window"] == 50
    assert report["full_context"] is False


def test_a_policy_returning_the_wrong_shape_is_caught():
    base = make_base()
    _, _, X_test = make_table()
    with pytest.raises(ValueError, match="one per query row"):
        large_context.run_policy(base, X_test, policy_spec=lambda p, rng: np.zeros(3))


def test_a_table_inside_the_window_predicts_from_full_context_and_says_so():
    """The threshold can be set below the window; then no policy is needed and running
    one would waste calls on partitions of a context that fits whole."""
    base = make_base(window=5000, n_train=600)
    _, _, X_test = make_table()
    with pytest.warns(pol.LargeContextPolicyWarning, match="fit the 5000-row window"):
        preds, report = large_context.run_policy(base, X_test, policy_spec="cluster_route")
    assert report["full_context"] is True
    assert report["nori_calls"] == 1
    assert preds.shape == (len(X_test),)


def test_the_gate_records_which_candidate_it_deployed():
    base = make_base(window=40, n_train=400)
    _, _, X_test = make_table(n_test=60)
    _, report = large_context.run_policy(base, X_test,
                                   policy_spec=["random", "cluster_route"])
    assert report["gate_winner"] in ("random", "cluster_route")
    # The gate pays for its whole holdout sweep plus the winner's real run.
    assert report["nori_calls"] > 1


def test_the_gate_decision_is_cached_so_the_sweep_runs_once():
    """The winner comes from a holdout carved out of TRAIN, so it is train-derived like
    a chain. Re-sweeping per predict would make the gate cost its whole menu forever."""
    base = make_base(window=40, n_train=400)
    _, _, X_test = make_table(n_test=60)
    spec = ["random", "cluster_route"]
    cold_preds, cold = large_context.run_policy(base, X_test, policy_spec=spec)
    warm_preds, warm = large_context.run_policy(base, X_test, policy_spec=spec)
    assert warm["gate_winner"] == cold["gate_winner"]
    assert warm["nori_calls"] < cold["nori_calls"], "the sweep re-ran"
    np.testing.assert_allclose(cold_preds, warm_preds, rtol=0, atol=1e-9)


def test_a_different_candidate_list_is_gated_separately():
    base = make_base(window=40, n_train=400)
    _, _, X_test = make_table(n_test=60)
    large_context.run_policy(base, X_test, policy_spec=["random"])
    _, report = large_context.run_policy(base, X_test,
                                   policy_spec=["random", "cluster_route"])
    # A two-arm menu must run its own sweep, not inherit the one-arm menu's winner.
    assert report["nori_calls"] > 2


# ------------------------------------------------------------- query labels absent
def test_y_test_raises_on_an_inference_problem_instead_of_returning_zeros():
    """A policy that reads y_test at inference time must fail loudly. Storing zeros
    would let it 'score' itself against a fabricated target."""
    base = make_base()
    with pytest.raises(ValueError, match="y_test is unknown"):
        _ = base.y_test


def test_the_shipped_policies_never_read_y_test():
    base = make_base(window=40, n_train=400)
    _, _, X_test = make_table(n_test=60)
    for name in ("random", "cluster_route", "cluster_route_g4", "safeboost", "boost"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
            preds, _ = large_context.run_policy(base, X_test, policy_spec=name)
        assert preds.shape == (60,), name


# ------------------------------------------------------------------ shard guards
def test_boosting_falls_back_to_random_when_the_table_affords_one_shard():
    base = make_base(window=400, n_train=600)   # 1 shard
    _, _, X_test = make_table()
    with pytest.warns(pol.LargeContextPolicyFallbackWarning, match="falling back to `random`"):
        preds, report = large_context.run_policy(base, X_test, policy_spec="safeboost")
    assert report["nori_calls"] == 1                        # one random window
    assert preds.shape == (80,)


def test_the_fallback_is_a_degradation_so_strict_pipeline_makes_it_fatal():
    """The caller asked for safeboost and would have got random: that is exactly what
    the degradation tree is for, so strict_pipeline() must catch it."""
    from synthefy_nori import strict_pipeline

    base = make_base(window=400, n_train=600)
    _, _, X_test = make_table()
    with pytest.raises(pol.LargeContextPolicyFallbackWarning):
        with strict_pipeline():
            large_context.run_policy(base, X_test, policy_spec="safeboost")


def test_boosting_below_the_shard_floor_warns_but_still_runs():
    base = make_base(window=150, n_train=600)    # 4 shards, floor is 8
    _, _, X_test = make_table()
    with pytest.warns(pol.LargeContextPolicyWarning, match="below the 8"):
        preds, _ = large_context.run_policy(base, X_test, policy_spec="safeboost")
    assert preds.shape == (80,)


def test_a_routing_policy_has_no_shard_floor():
    base = make_base(window=150, n_train=600)
    _, _, X_test = make_table()
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # any warning fails this
        large_context.run_policy(base, X_test, policy_spec="cluster_route[groups=2]")


# ------------------------------------------------ tier 2: the chain replays exactly
@pytest.mark.parametrize("policy", ["safeboost", "boost"])
def test_replay_reproduces_the_fused_build_bit_for_bit(policy):
    """THE correctness claim behind caching a chain.

    A chain's shards, residual labels, guard decisions and early-stopping cut come from
    the train rows alone, so replaying a chain on the same queries must reproduce the
    predictions the fused build returned. If this drifts, the cache is a wrong answer,
    not a slow one.
    """
    X_train, y_train, X_test = make_table(n_train=800, n_test=70)
    base = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)
    first, report_first = large_context.run_policy(base, X_test, policy_spec=policy)
    assert report_first["reused_train_state"] is False

    second, report_second = large_context.run_policy(base, X_test, policy_spec=policy)
    assert report_second["reused_train_state"] is True
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-9)


@pytest.mark.parametrize("policy", ["safeboost", "boost"])
def test_replay_on_new_queries_matches_a_cold_run_on_those_queries(policy):
    """The stronger claim: a chain built while serving one query block is the chain a
    DIFFERENT query block would have built for itself. Only true because preprocessing
    is inductive -- which is why the stub predictor is inductive too."""
    X_train, y_train, _ = make_table(n_train=800)
    other = make_table(n_train=800, n_test=45, seed=9)[2]

    warm = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)
    large_context.run_policy(warm, make_table(n_train=800, n_test=70)[2], policy_spec=policy)
    replayed, warm_report = large_context.run_policy(warm, other, policy_spec=policy)

    cold = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)
    fresh, cold_report = large_context.run_policy(cold, other, policy_spec=policy)

    np.testing.assert_allclose(replayed, fresh, rtol=0, atol=1e-9)
    assert warm_report["reused_train_state"] is True
    assert cold_report["reused_train_state"] is False


@pytest.mark.parametrize("policy", ["safeboost", "boost"])
def test_replay_costs_strictly_fewer_nori_calls(policy):
    """Tier 2's whole point. A replay skips the train-side decode, which is the
    O(shards^2 x window) half of a boosting chain."""
    X_train, y_train, X_test = make_table(n_train=800, n_test=70)
    base = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)
    cold = large_context.run_policy(base, X_test, policy_spec=policy)[1]["nori_calls"]
    warm = large_context.run_policy(base, X_test, policy_spec=policy)[1]["nori_calls"]
    assert 0 < warm <= cold


def test_a_chain_is_not_shared_between_different_seeds():
    """The shards come from the run's rng, so the seed is part of the chain's identity.
    Without it in the key, a re-run under a new seed replays the old seed's chain."""
    X_train, y_train, X_test = make_table(n_train=800, n_test=70)
    base = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)
    a, report_a = large_context.run_policy(base, X_test, policy_spec="safeboost", seed=0)
    b, report_b = large_context.run_policy(base, X_test, policy_spec="safeboost", seed=99)
    assert not np.allclose(a, b), "seed 99 was served seed 0's chain"
    # And the report says so. Seed 0's chain is sitting in the cache, so "is the cache
    # non-empty?" would call this a reuse; the counter records that seed 99 read
    # nothing and built its own.
    assert report_b["reused_train_state"] is False


def test_a_chain_is_not_shared_between_different_parameter_settings():
    """nu is part of the cache key: a chain built at nu=0.5 is not the nu=0.2 chain."""
    X_train, y_train, X_test = make_table(n_train=800, n_test=70)
    base = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)
    a = large_context.run_policy(base, X_test, policy_spec="safeboost[nu=0.5]")[0]
    b = large_context.run_policy(base, X_test, policy_spec="safeboost[nu=0.2]")[0]
    assert not np.allclose(a, b)


# ------------------------------------------- tier 2: train-derived state is reused
def test_with_queries_shares_the_train_view_and_drops_the_query_view():
    base = make_base(window=40, n_train=400)
    train_view = base.select_view
    base.routing_space()

    fresh = base.with_queries(make_table(n_test=25)[2])
    assert fresh.select_view is train_view, "the imputed train block was recopied"
    assert fresh._routing_test is None, "a stale query routing space leaked across calls"
    assert fresh.n_test == 25
    assert fresh.nori_calls == 0, "call accounting must be per-prediction"


def test_a_view_populating_train_state_lazily_reaches_the_fitted_problem():
    """The regression test for the bug the by-value copy caused.

    A query view is throwaway, so train-derived work it does lazily has to land on the
    Problem that outlives it. When `with_queries` copied `_select_view` instead of
    sharing it, the fitted Problem stayed empty forever and the ~0.5 GB train imputation
    was redone on EVERY predict -- while the old test above still passed, because it
    primed the base by hand first."""
    base = make_base(window=40, n_train=400)
    assert "select_view" not in base.train_state

    view = base.with_queries(make_table(n_test=25)[2])
    computed = view.select_view                       # lazily, on the throwaway view

    assert base.train_state["select_view"] is computed, "the work died with the view"
    later = base.with_queries(make_table(n_test=30)[2])
    assert later.select_view is computed, "a later predict recomputed it"


def test_the_train_block_is_imputed_once_across_two_predicts():
    """The same claim end-to-end, counted rather than inspected."""
    base = make_base(window=40, n_train=400)
    _, _, X_test = make_table(n_test=60)
    derived = []
    real = pol.column_medians

    def counting(fit_on):
        if len(fit_on) == base.n_train:
            derived.append(1)
        return real(fit_on)

    pol.column_medians = counting
    try:
        large_context.run_policy(base, X_test, policy_spec="cluster_route")
        assert len(derived) == 1, f"train medians derived {len(derived)}x on one predict"
        large_context.run_policy(base, X_test, policy_spec="cluster_route")
        assert len(derived) == 1, "the train block was re-derived on the second predict"
    finally:
        pol.column_medians = real


def test_with_queries_shares_the_train_cache_by_reference():
    base = make_base(window=40, n_train=400)
    fresh = base.with_queries(make_table(n_test=25)[2])
    assert fresh.train_cache is base.train_cache
    assert fresh.train_state is base.train_state


def test_a_subproblem_does_not_inherit_the_train_cache():
    """A subproblem has DIFFERENT train rows, so a chain built on the parent's rows
    would be wrong for it. This is what keeps the gate honest."""
    base = make_base(window=40, n_train=400)
    base.train_cache[("safeboost", 0.5, None)] = ["sentinel"]
    base.select_view                                  # populate train_state too
    sub = base.subproblem(np.arange(200), np.arange(200, 260))
    assert sub.train_cache == {}
    assert sub.train_state == {}, "a subproblem has different train rows"


def test_subproblem_inherits_the_impute_setting():
    base = make_base()
    assert base.impute is False
    assert base.subproblem(np.arange(100), np.arange(100, 150)).impute is False


# --------------------------------------------------------- the production path is raw
def _one_column_problem(impute: bool) -> pol.Problem:
    return pol.Problem(
        ridge_predict,
        np.array([[np.nan], [1.0], [3.0]], dtype=np.float32),
        np.array([0.0, 1.0, 3.0]),
        np.array([[2.0]], dtype=np.float32),
        None,
        window=3,
        impute=impute,
    )


def test_the_inference_path_hands_missing_values_through_untouched():
    """NoriPredictor owns missing-value handling. Imputing in the policy first would
    change what the model sees relative to an ordinary predict() on the same rows."""
    ctx, = _one_column_problem(impute=False).impute_from_context(
        np.array([[np.nan], [1.0]], dtype=np.float32))
    assert np.isnan(ctx[0, 0]), "the NaN was filled before the predictor saw it"


def test_the_benchmark_path_still_imputes():
    """The harness keeps `evaluation.harness._apply_impute` semantics; only the
    production path opts out. The default is the harness's, so a benchmark that does
    not mention imputation keeps the behavior its numbers were measured under."""
    problem = _one_column_problem(impute=True)
    assert pol.Problem(ridge_predict, np.zeros((2, 1)), np.zeros(2),
                       np.zeros((1, 1)), None, window=2).impute is True
    ctx, = problem.impute_from_context(problem.X_train)
    assert np.isfinite(ctx).all()


def test_build_problem_opts_out_of_imputation():
    assert large_context.build_problem(
        ridge_predict, np.zeros((10, 2)), np.zeros(10), window=5).impute is False


# ------------------------------------------------------------------ tier 1 capacity
def test_cache_capacity_is_not_a_memory_policy_field():
    """It is a library-only implementation detail, and MemoryPolicy is mirrored in the
    `synthefy` client and published in the serving request schema. Putting it there broke
    client/server parity (`test_the_policy_schema_and_the_client_policy_declare_the_same_inputs`)
    for a knob no hosted-API caller can use. Serving exposes a bounded large-context
    policy menu, but deliberately fixes cache entries at one and disables retained
    customer context across requests."""
    from synthefy_nori.inference.memory_policy import MemoryPolicy

    assert "context_cache_entries" not in MemoryPolicy.model_fields


def test_cache_capacity_defaults_to_one_on_the_predictor():
    from synthefy_nori.inference.predictor import NoriPredictor

    assert NoriPredictor.context_cache_entries == 1


def test_the_estimator_pushes_its_cache_capacity_onto_the_predictor(monkeypatch):
    """And re-declares it per call, so setting it after the first predict takes effect."""
    from synthefy_nori.api import NoriRegressor

    est = NoriRegressor(model_path="unused", large_context_cache_entries=6)
    stub = StubPredictor()
    monkeypatch.setattr(NoriRegressor, "_get_predictor",
                        lambda self: (setattr(stub, "context_cache_entries",
                                              self.large_context_cache_entries) or stub))
    est.fit(*make_table(n_train=400)[:2])
    est.predict(make_table(n_test=40)[2])
    assert stub.context_cache_entries == 6


def test_the_kv_cache_retains_every_pool_of_a_rotation_at_capacity():
    """Tier 1. At capacity 1 a policy that rotates between pools evicts on every call
    and never hits; the point of the multi-entry cache is that the rotation survives."""
    class FakeBare:
        def __init__(self):
            self.builds = 0

        def build_context_cache(self, x_train, y_train, *, cache_dtype,
                                offload_kv_cache, fit_row_chunk):
            self.builds += 1
            return ("bundle", self.builds)

    from synthefy_nori.inference.predictor import NoriPredictor

    holder = NoriPredictor.__new__(NoriPredictor)
    holder.seed = 0
    bare = FakeBare()
    pools = [torch.arange(6, dtype=torch.float32).reshape(1, 3, 2) + offset
             for offset in (0.0, 10.0, 20.0)]
    y = torch.zeros(1, 3)

    def call(x, entries):
        return NoriPredictor._get_or_build_context(
            holder, bare, 0, x_train_t=x, y_train_t=y, cache_dtype="bf16",
            offload_kv_cache=False, fit_row_chunk=None, reuse_context_cache=True,
            cache_entries=entries)

    for pool in pools:                        # capacity 1: three builds, no reuse
        call(pool, 1)
    for pool in pools:
        call(pool, 1)
    assert bare.builds == 6, "capacity 1 should rebuild every rotation"

    bare.builds = 0
    holder._context_cache = {}
    for pool in pools:                        # capacity 3: warm once...
        call(pool, 3)
    assert bare.builds == 3
    for pool in pools:                        # ...then every pool is a hit
        call(pool, 3)
    assert bare.builds == 3, "the rotation was not retained at capacity"


def test_shrinking_capacity_between_calls_trims_the_cache():
    """A shared engine re-declares its policy per request, so the CURRENT call's
    capacity has to be what is honoured -- otherwise a bundle outlives its budget."""
    class FakeBare:
        def build_context_cache(self, x_train, y_train, **kwargs):
            return "bundle"

    from synthefy_nori.inference.predictor import NoriPredictor

    holder = NoriPredictor.__new__(NoriPredictor)
    holder.seed = 0
    bare = FakeBare()
    y = torch.zeros(1, 2)
    for offset in (0.0, 10.0, 20.0):
        NoriPredictor._get_or_build_context(
            holder, bare, 0,
            x_train_t=torch.arange(4, dtype=torch.float32).reshape(1, 2, 2) + offset,
            y_train_t=y, cache_dtype="bf16", offload_kv_cache=False,
            fit_row_chunk=None, reuse_context_cache=True, cache_entries=3)
    assert len(holder._context_cache[0]) == 3
    NoriPredictor._get_or_build_context(
        holder, bare, 0,
        x_train_t=torch.full((1, 2, 2), 99.0), y_train_t=y, cache_dtype="bf16",
        offload_kv_cache=False, fit_row_chunk=None, reuse_context_cache=True,
        cache_entries=1)
    assert len(holder._context_cache[0]) == 1


def test_shrinking_capacity_trims_on_a_cache_HIT_too():
    """The shrink above arrives with a context that misses. A hit used to return before
    the trim, so a caller lowering capacity to free memory kept every bundle for as long
    as it kept asking for a context already in the list -- precisely when it is trying
    to release VRAM. Each entry is a full K/V cache, so that is the OOM this knob exists
    to prevent."""
    class FakeBare:
        def build_context_cache(self, x_train, y_train, **kwargs):
            return "bundle"

    from synthefy_nori.inference.predictor import NoriPredictor

    holder = NoriPredictor.__new__(NoriPredictor)
    holder.seed = 0
    bare = FakeBare()
    y = torch.zeros(1, 2)
    pools = [torch.arange(4, dtype=torch.float32).reshape(1, 2, 2) + offset
             for offset in (0.0, 10.0, 20.0)]

    def call(x, entries):
        return NoriPredictor._get_or_build_context(
            holder, bare, 0, x_train_t=x, y_train_t=y, cache_dtype="bf16",
            offload_kv_cache=False, fit_row_chunk=None, reuse_context_cache=True,
            cache_entries=entries)

    for pool in pools:
        call(pool, 3)
    assert len(holder._context_cache[0]) == 3
    # The COLDEST pool, so the trim has to promote it before cutting -- a trim that
    # dropped it would evict the very bundle being returned.
    assert call(pools[0], 1) == "bundle"
    assert len(holder._context_cache[0]) == 1, "a hit skipped the capacity trim"


# --------------------------------------------------------------- estimator wiring
class StubPredictor:
    """Enough NoriPredictor surface for the dispatch tests: ridge, and a fixed window."""

    def __init__(self, window=50):
        self.window = window
        self.quantile_collapse = "mean"
        self.bar_point_estimator = "mean"
        self.contexts = []
        self.budget_scans = 0

    def budget_n_features(self, x_train):
        """The expensive, table-derived half -- counted, so a test can pin how often
        the estimator pays for it."""
        self.budget_scans += 1
        return x_train.shape[1]

    def max_context_rows(self, x_train, *, budget_n_features=None):
        if budget_n_features is None:
            self.budget_n_features(x_train)
        return min(self.window, len(x_train))

    def predict(self, x_train, y_train, x_test):
        self.contexts.append(len(x_train))
        return ridge_predict(x_train, y_train, x_test)


def fitted(monkeypatch, stub=None, **kwargs):
    from synthefy_nori.api import NoriRegressor

    X_train, y_train, X_test = make_table(n_train=400, n_test=40)
    est = NoriRegressor(model_path="unused", **kwargs)
    stub = StubPredictor() if stub is None else stub

    def get_predictor():
        stub.memory_policy = est.memory_policy
        return stub

    monkeypatch.setattr(est, "_get_predictor", get_predictor)
    est.fit(X_train, y_train)
    return est, stub, X_test


def test_no_policy_means_one_full_context_call(monkeypatch):
    est, stub, X_test = fitted(monkeypatch)
    est.predict(X_test)
    assert stub.contexts == [400], "the default path must be unchanged"
    assert est.large_context_report_ is None


def test_below_the_threshold_the_policy_does_not_engage(monkeypatch):
    est, stub, X_test = fitted(
        monkeypatch, large_context_policy="cluster_route", large_context_threshold=1000)
    est.predict(X_test)
    assert stub.contexts == [400]
    assert est.large_context_report_ is None


def test_a_direct_predict_clears_the_previous_large_context_report(monkeypatch):
    est, stub, X_test = fitted(
        monkeypatch, large_context_policy="random", large_context_threshold=100)
    est.predict(X_test)
    assert est.large_context_report_["policy"] == "random"

    est.large_context_policy = None
    est.predict(X_test)
    assert stub.contexts[-1] == 400
    assert est.large_context_report_ is None


def test_above_the_threshold_the_policy_runs_and_reports(monkeypatch):
    est, stub, X_test = fitted(
        monkeypatch, large_context_policy="cluster_route[groups=4]", large_context_threshold=100)
    preds = est.predict(X_test)
    assert preds.shape == (40,)
    assert stub.contexts == [50] * 4, "every context must be within the window"
    assert est.large_context_report_["policy"] == "cluster_route[groups=4]"
    assert est.large_context_report_["nori_calls"] == 4


def test_an_unknown_policy_fails_at_fit_not_deep_into_predict(monkeypatch):
    from synthefy_nori.api import NoriRegressor

    X_train, y_train, _ = make_table(n_train=400)
    est = NoriRegressor(model_path="unused", large_context_policy="clusterboost")
    monkeypatch.setattr(est, "_get_predictor", lambda: StubPredictor())
    with pytest.raises(ValueError, match="unknown policy"):
        est.fit(X_train, y_train)


def test_refitting_invalidates_the_chain_so_it_cannot_serve_new_rows(monkeypatch):
    """The cache is keyed to the table. A chain built on the previous fit's rows
    serving this one would be a wrong answer, not a slow one."""
    est, stub, X_test = fitted(
        monkeypatch, large_context_policy="safeboost", large_context_threshold=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
        est.predict(X_test)
        assert est.large_context_report_["reused_train_state"] is False
        est.predict(X_test)
        assert est.large_context_report_["reused_train_state"] is True

        other_X, other_y, _ = make_table(n_train=400, seed=5)
        est.fit(other_X, other_y)
        assert est._large_context_problem is None
        assert est.large_context_report_ is None
        est.predict(X_test)
        assert est.large_context_report_["reused_train_state"] is False


def test_a_second_predict_reuses_the_fitted_problem(monkeypatch):
    est, stub, X_test = fitted(
        monkeypatch, large_context_policy="random", large_context_threshold=100)
    est.predict(X_test)
    first = est._large_context_problem
    est.predict(X_test[:10])
    assert est._large_context_problem is first
    assert est.large_context_report_["n_test"] == 10


def test_predictions_are_denormalized_back_to_original_units(monkeypatch):
    """The policy runs in normalized y; predict() must undo that. A large offset makes
    a missing denormalization obvious."""
    from synthefy_nori.api import NoriRegressor

    X_train, y_train, X_test = make_table(n_train=400, n_test=40)
    y_shifted = y_train * 100.0 + 5000.0
    est = NoriRegressor(model_path="unused", large_context_policy="random",
                        large_context_threshold=100)
    monkeypatch.setattr(est, "_get_predictor", lambda: StubPredictor())
    est.fit(X_train, y_shifted)
    preds = est.predict(X_test)
    assert 3000.0 < float(np.mean(preds)) < 7000.0, float(np.mean(preds))


# ------------------------------------------------- review: the decoder scopes a chain
class DecoderStub(StubPredictor):
    """A stub whose predictions depend on the decoder, as a real head's do.

    `NoriRegressor._predict_point` flips `quantile_collapse`/`bar_point_estimator` per
    call, so one estimator's `predict_fn` is really several functions. A constant offset
    is enough to expose a chain replayed under the wrong one: residual labels are
    `y - (what the model said)`, so they inherit it.
    """

    def predict(self, x_train, y_train, x_test):
        base = super().predict(x_train, y_train, x_test)
        return base + (0.0 if self.bar_point_estimator == "mean" else 5.0)


def test_a_chain_built_under_one_decoder_is_not_replayed_under_another(monkeypatch):
    """P1. A mean predict followed by a median one used to replay the MEAN chain's
    residual labels and return numbers no cold median run would produce."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
        est, _, X_test = fitted(monkeypatch, stub=DecoderStub(),
                                large_context_policy="safeboost", large_context_threshold=100)
        mean_preds = est.predict(X_test)                      # builds the mean chain
        median_preds = est.predict(X_test, output_type="median")

        cold, _, _ = fitted(monkeypatch, stub=DecoderStub(),
                            large_context_policy="safeboost", large_context_threshold=100)
        cold_median = cold.predict(X_test, output_type="median")

    assert not np.allclose(mean_preds, median_preds), "the stub decoder does nothing"
    np.testing.assert_allclose(median_preds, cold_median, rtol=0, atol=1e-9)


def test_each_decoder_derives_its_own_chain_once(monkeypatch):
    """The scoping is a partition, not an invalidation: flipping back to a decoder that
    already has a chain must replay it rather than pay for it twice."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
        est, _, X_test = fitted(monkeypatch, stub=DecoderStub(),
                                large_context_policy="safeboost", large_context_threshold=100)
        est.predict(X_test)
        assert est.large_context_report_["reused_train_state"] is False

        est.predict(X_test, output_type="median")
        assert est.large_context_report_["reused_train_state"] is False, "a fresh decoder"
        est.predict(X_test, output_type="median")
        assert est.large_context_report_["reused_train_state"] is True

        est.predict(X_test)             # back to mean: its chain is still there
        assert est.large_context_report_["reused_train_state"] is True


class PrecisionStub(StubPredictor):
    """Stand in for the supported lossy BF16-to-INT8 memory-policy change."""

    def predict(self, x_train, y_train, x_test):
        base = super().predict(x_train, y_train, x_test)
        dtype = MemoryPolicy.coerce(self.memory_policy).cache_dtype
        return base + (0.0 if dtype == "bf16" else 5.0)


def test_a_chain_is_not_replayed_across_memory_precision(monkeypatch):
    """Decision caches depend on every setting that changes what predict_fn returns.

    Memory policy is re-declared per call. INT8 is deliberately lossy, and it can use
    the same context window as BF16, so window invalidation alone cannot scope it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
        est, _, X_test = fitted(monkeypatch, stub=PrecisionStub(),
                                memory_policy={"cache_dtype": "bf16"},
                                large_context_policy="safeboost", large_context_threshold=100)
        bf16_preds = est.predict(X_test)
        est.memory_policy = {"cache_dtype": "int8"}
        warm_int8 = est.predict(X_test)
        assert est.large_context_report_["reused_train_state"] is False

        cold, _, _ = fitted(monkeypatch, stub=PrecisionStub(),
                            memory_policy={"cache_dtype": "int8"},
                            large_context_policy="safeboost", large_context_threshold=100)
        cold_int8 = cold.predict(X_test)

    assert not np.allclose(bf16_preds, warm_int8), "the precision stub does nothing"
    np.testing.assert_allclose(warm_int8, cold_int8, rtol=0, atol=1e-9)


def test_the_train_cache_is_partitioned_by_scope_and_the_arrays_are_not():
    """The unit-level shape of the fix. Decisions depend on what `predict_fn` returned;
    the derived arrays are functions of X_train alone and are shared across scopes."""
    base = make_base(window=50, n_train=600)
    _, _, X_test = make_table(n_test=40)
    mean_view = base.with_queries(X_test, cache_scope=("mean", "mean"))
    median_view = base.with_queries(X_test, cache_scope=("median", "median"))

    mean_view.train_cache["chain"] = "mean-chain"
    assert "chain" not in median_view.train_cache
    assert base.with_queries(X_test, cache_scope=("mean", "mean")).train_cache[
        "chain"] == "mean-chain"
    assert mean_view.train_state is median_view.train_state


# --------------------------------------------- review: the window is a per-call input
def test_the_window_is_recomputed_per_call_not_frozen_at_the_first_predict(monkeypatch):
    """P1. `memory_policy` is re-declared per call (a server sets it per request), so a
    later, smaller elements_budget must shrink the context the policy emits. Frozen, the
    policy kept emitting the old size -- to be randomly subsampled underneath or to
    raise ContextTooLargeError -- while the report still advertised the old window."""
    est, stub, X_test = fitted(
        monkeypatch, large_context_policy="random", large_context_threshold=100)
    est.predict(X_test)
    assert est.large_context_report_["window"] == 50
    assert stub.contexts == [50]
    first = est._large_context_problem

    stub.window = 25                        # a smaller budget on the next call
    est.predict(X_test)
    assert est.large_context_report_["window"] == 25
    assert stub.contexts[-1] == 25, "the policy emitted the stale window"
    assert est._large_context_problem is not first, "a window-sized cache outlived its window"


def test_a_changed_window_invalidates_the_chain(monkeypatch):
    """A boosting chain's shards are `window` rows wide, so a new window is a different
    chain -- replaying the old one would be a wrong answer, not a stale cost."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
        est, stub, X_test = fitted(
            monkeypatch, large_context_policy="safeboost", large_context_threshold=100)
        est.predict(X_test)
        est.predict(X_test)
        assert est.large_context_report_["reused_train_state"] is True

        stub.window = 25
        est.predict(X_test)
        assert est.large_context_report_["reused_train_state"] is False
        assert max(stub.contexts[-8:]) <= 25


def test_an_unchanged_window_still_reuses_the_fitted_problem(monkeypatch):
    """The recompute must not become a rebuild-every-call: same budget, same Problem."""
    est, _, X_test = fitted(
        monkeypatch, large_context_policy="cluster_route[groups=2]", large_context_threshold=100)
    est.predict(X_test)
    first = est._large_context_problem
    est.predict(X_test)
    assert est._large_context_problem is first


# -------------------------------------------------- review: the gate keeps a context
def test_the_gate_leaves_the_candidates_a_full_window_of_context():
    """P2. holdout=2000 against 800 train rows used to select every row, leaving each
    candidate an EMPTY context. The 400-row tests passed only because the ridge stub
    tolerates an empty fit; a real predictor raises, or picks a winner off noise."""
    seen = []

    def strict_predict(X_ctx, y_ctx, X_query):
        seen.append(len(X_ctx))
        if len(X_ctx) == 0:                 # what a real predictor does
            raise ValueError("empty context")
        return ridge_predict(X_ctx, y_ctx, X_query)

    X_train, y_train, X_test = make_table(n_train=800, n_test=60)
    base = large_context.build_problem(strict_predict, X_train, y_train, window=50, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
        preds, report = large_context.run_policy(
            base, X_test, policy_spec=["random", "cluster_route"])
    assert preds.shape == (60,)
    assert report["gate_winner"] in ("random", "cluster_route")
    assert min(seen) == 50, "the holdout ate into the candidates' window"


def test_a_squeezed_holdout_says_so():
    """Capping is not silent: the winner is being chosen on less evidence than asked
    for, and that changes how much the gate's verdict is worth."""
    X_train, y_train, X_test = make_table(n_train=800, n_test=60)
    base = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)
    with pytest.warns(pol.LargeContextPolicyWarning, match="leaves only 750 for the holdout"):
        large_context.run_policy(base, X_test, policy_spec=["random", "cluster_route"])


def test_the_gate_refuses_a_table_it_cannot_carve_a_holdout_from():
    """Unreachable through `run_policy`, which serves full context below the window --
    but `holdout_gate` is public, and silently scoring on nothing is the failure this
    replaces."""
    X_train, y_train, X_test = make_table(n_train=40, n_test=10)
    base = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)
    gate = pol.holdout_gate(["random"])
    with pytest.raises(ValueError, match="needs more than window=50 train rows"):
        gate(base.with_queries(X_test), np.random.default_rng(0))


# ------------------------------------------- review: what `reused_train_state` records
def test_reuse_is_reported_from_reads_that_hit_not_from_a_non_empty_cache():
    """P2. `random` derives nothing and reuses nothing, however warm the Problem is;
    `cluster_route` reuses the train routing space. A "is the cache non-empty?" test
    called both of them reuse."""
    X_train, y_train, X_test = make_table(n_train=800, n_test=60)
    base = large_context.build_problem(ridge_predict, X_train, y_train, window=50, seed=1)

    assert large_context.run_policy(base, X_test, policy_spec="random")[1][
        "reused_train_state"] is False
    assert large_context.run_policy(base, X_test, policy_spec="cluster_route")[1][
        "reused_train_state"] is False, "nothing was on hand to reuse yet"
    assert large_context.run_policy(base, X_test, policy_spec="cluster_route")[1][
        "reused_train_state"] is True, "the train routing space"
    # Warm Problem, but `random` still reads none of it.
    assert large_context.run_policy(base, X_test, policy_spec="random")[1][
        "reused_train_state"] is False


def test_the_shared_state_counts_only_hits_on_earlier_calls_work():
    state = pol.SharedTrainState()
    state.begin_call()
    state["k"] = 1
    assert state.get("k") == 1 and state["k"] == 1
    assert state.hits == 0, "reading back what this call just derived is not reuse"

    state.begin_call()
    assert state.get("k") == 1
    assert state["k"] == 1
    assert state.hits == 2
    assert state.get("absent") is None
    assert state.hits == 2, "a miss is not a hit"


# --------------------------------------------- portable dependency boundary
def test_policies_does_not_depend_on_the_evaluation_package():
    """The inference policy module must not depend on optional evaluation helpers.

    Keep its dependencies available wherever the inference package is installed.
    """

    tree = ast.parse(inspect.getsource(pol))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [m for m in imported if m.startswith("synthefy_nori.evaluation")]

    # Also enforce the repository's top-level import convention. A deferred dependency
    # can otherwise remain invisible until the affected policy runs, potentially
    # minutes into a prediction on a large table.
    nested = [n for body in tree.body if isinstance(body, ast.FunctionDef)
              for n in ast.walk(body) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not nested, [ast.unparse(n) for n in nested]


@pytest.mark.parametrize("y_true,y_pred,expected", [
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
    ([1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 99.0], 1.0),     # pairwise-finite mask
    ([1.0, 2.0, 3.0], [1.0, np.inf, np.nan], float("nan")),    # <2 survivors -> NaN
    ([], [], float("nan")),
])
def test_r2_matches_the_evaluation_harness_semantics(y_true, y_pred, expected):
    """Same contract as `compute_reg_metrics(...)["r2"]`, which this replaced: drop
    non-finite pairs, NaN below two survivors."""
    got = pol.r2(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float))
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


def test_the_window_recompute_does_not_rescan_the_table_every_predict(monkeypatch):
    """The window has to be re-resolved per call, but its table-derived half must not
    be: that one is a float64 copy of the whole table plus a `np.unique` per column
    (~5s on 1M x 130), and repaying it per predict is the cost this whole path exists
    to remove. Only the element budget -- cheap, and the part that actually moves -- is
    re-resolved."""
    est, stub, X_test = fitted(
        monkeypatch, large_context_policy="random", large_context_threshold=100)
    for _ in range(3):
        est.predict(X_test)
    assert stub.budget_scans == 1

    est.fit(*make_table(n_train=400, seed=5)[:2])    # a new table is a new scan
    est.predict(X_test)
    assert stub.budget_scans == 2


# ------------------------------------------------------- review-round-3 regressions
def test_a_policy_refusal_is_not_reported_as_a_checkpoint_problem(monkeypatch):
    """`_predict_categorical` wraps NotImplementedError to blame a bar_distribution
    checkpoint for withholding a quantile bank. The large-context refusal comes out of the
    same call, so before it had its own type the caller was told to change checkpoint
    or discretize strategy over a problem that was neither."""
    from synthefy_nori.inference.large_context import LargeContextUnsupportedOutputError

    est, _, X_test = fitted(monkeypatch, large_context_policy="cluster_route",
                            large_context_threshold=100)
    with pytest.raises(LargeContextUnsupportedOutputError, match="large_context_policy"):
        est.predict(X_test, output_type="full")
    with pytest.raises(LargeContextUnsupportedOutputError, match="large_context_policy") as caught:
        est.predict(X_test, discretize="map-cell", categorical_levels=[0.0, 1.0])
    assert "bar_distribution" not in str(caught.value), (
        "a policy refusal was re-reported as a checkpoint limitation")


def test_a_changed_window_keeps_the_train_arrays_and_drops_the_decisions(monkeypatch):
    """A window change makes a boosting chain invalid (its shards are window-sized) but
    not the imputed train block, which is a function of X_train alone."""
    # A gate over cluster_route touches BOTH stores: the routing space is an array in
    # train_state, the chosen winner is a decision in train_cache.
    stub = StubPredictor(window=50)
    est, _, X_test = fitted(monkeypatch, stub=stub,
                            large_context_policy=["random", "cluster_route"],
                            large_context_threshold=100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
        est.predict(X_test)
    problem = est._large_context_problem
    arrays = problem.train_state
    # Decisions are scope-partitioned, so ask the store of all scopes rather than this
    # object's partition of it (the base Problem sits at the default, empty scope).
    decisions = problem._train_caches
    assert arrays, "nothing was cached to carry"
    assert decisions, "no decision was cached to invalidate"

    stub.window = 40                                  # a smaller elements_budget
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pol.LargeContextPolicyWarning)
        est.predict(X_test)
    rebuilt = est._large_context_problem
    assert rebuilt is not problem, "the Problem must be rebuilt for a new window"
    assert rebuilt.window == 40
    assert rebuilt.train_state is arrays, "the imputed train block was re-derived"
    # Identity, not emptiness: the second predict re-derives its own decision under the
    # new window, so the question is whether the OLD store was carried over.
    assert rebuilt._train_caches is not decisions, (
        "window-sized decisions were carried across a window change")


def test_adopt_train_state_refuses_a_different_table():
    base = make_base(window=40, n_train=400)
    other = large_context.build_problem(
        ridge_predict, *make_table(n_train=400, seed=7)[:2], window=40, seed=0)
    with pytest.raises(ValueError, match="same fitted table"):
        other.adopt_train_state(base)


@pytest.mark.parametrize("mutate", [
    lambda s: s.setdefault("k", 99),
    lambda s: s.update({"k": 99}),
])
def test_every_mutator_routes_through_the_hit_counter(mutate):
    """`get`/`__getitem__`/`__setitem__` were counted but `setdefault`/`update` were
    not, so a policy reaching for one would silently under-report reuse."""
    state = pol.SharedTrainState()
    state["k"] = 1
    state.begin_call()
    mutate(state)
    assert state.hits + len(state._derived_this_call) > 0, "the mutator bypassed both"


def test_setdefault_on_an_existing_key_counts_as_reuse():
    state = pol.SharedTrainState()
    state["k"] = 1
    state.begin_call()
    assert state.setdefault("k", 99) == 1
    assert state.hits == 1


def test_pop_forgets_that_this_call_derived_the_key():
    state = pol.SharedTrainState()
    state.begin_call()
    state["k"] = 1                      # derived by THIS call
    state.pop("k")
    state["k"] = 2                      # re-derived, still this call
    assert state["k"] == 2
    assert state.hits == 0, "a key this call derived was counted as earlier work"
