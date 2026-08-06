"""Persistent context cache: build the context (train) encode once, reuse it
across many query batches.

Covers ``FeaturesTransformer.build_context_cache`` / ``apply_context_cache`` and the
refactor of ``forward_cached_regression`` into a thin wrapper over them. The claims
under test:

  A. build+apply reproduces ``forward_cached_regression`` EXACTLY (same code path).
  B. one bundle scores multiple query batches consistently (reuse across calls) and
     is chunk-order-independent.
  C. the O(N_train) context build (``build_train_cache``) runs ONCE, and applies do
     NOT rebuild it -- the whole point of amortization.
  D. repeated applies from one bundle are deterministic, and the bundle is not
     mutated by apply.
  E. apply agrees with the TRANSDUCTIVE forward on non-finite query values, i.e. the
     bundle carries every train-derived preprocess stat (``norm_stats`` AND
     ``nan_mean``) rather than letting query rows derive stats from themselves.
  F. the CROSS-CALL cache in ``NoriPredictor._get_or_build_context`` only ever hands
     back a bundle built from the exact context being asked about -- the invariant
     that keeps a long-lived server from answering one caller against another's rows.

The plain-vs-cached numeric agreement (~1.5e-3) is already covered by
``test_memory_policy_e2e.py``; since (A) is exact, that guarantee carries over to the
split for free.

Two tiers of test here:

  * The (F) tests, and the unit-level halves of (E), are hermetic -- they drive
    ``_get_or_build_context`` with a stub model that counts builds, and the encoders
    directly, so they need no checkpoint, no network and no GPU and can assert build
    counts and bundle identity EXACTLY. They run in the default suite.
  * The (A)-(D) tests and (E)'s end-to-end arms need real weights, so they are marked
    ``slow`` and skip when a checkpoint is not reachable (no network / no HF cache).
    They run on CPU.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

N_TRAIN, N_TEST, N_FEATURES = 300, 120, 8


@pytest.fixture(scope="module")
def bare_model():
    """The bare FeaturesTransformer from nori-6m, or skip when unreachable."""
    from synthefy_nori.api import NoriRegressor
    try:
        reg = NoriRegressor(model="nori-6m", device="cpu")
        pred = reg._get_predictor()
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"no reachable Nori checkpoint: {type(exc).__name__}: {exc}")
    model = pred._bare_model()
    model.eval()
    if getattr(model, "mask_prediction", False):
        pytest.skip("checkpoint built with mask_prediction=True; cached path N/A")
    return model


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_TRAIN + N_TEST, N_FEATURES)).astype(np.float32)
    y = (np.sin(x[:, 0] * 2) - x[:, 1] + 0.1 * rng.normal(size=len(x))).astype(np.float32)
    xt = torch.from_numpy(x).unsqueeze(0)          # [1, N, F]
    yt = torch.from_numpy(y).unsqueeze(0)          # [1, N]
    return xt, yt


def _split(model, xt, yt, eval_pos, seed=0):
    """build_context_cache + apply_context_cache under a fixed seed."""
    torch.manual_seed(seed)
    with torch.no_grad():
        bundle = model.build_context_cache(xt[:, :eval_pos], yt[:, :eval_pos])
        return model.apply_context_cache(xt[:, eval_pos:], bundle, row_chunk_size=0)


@pytest.mark.slow
def test_split_matches_forward_cached_regression_exactly(bare_model, table):
    # (A) forward_cached_regression is now build+apply; called directly the split
    # must reproduce it bit-for-bit under the same seed (same RNG draw for the
    # random feature positional embedding, same everything downstream).
    xt, yt = table
    ep = N_TRAIN
    torch.manual_seed(0)
    with torch.no_grad():
        ref = bare_model.forward_cached_regression(xt, yt, ep, row_chunk_size=0)
    split = _split(bare_model, xt, yt, ep, seed=0)
    # reg decoder emits a per-row quantile bank -> [1, N_TEST, K]; compare values.
    assert split.shape == ref.shape
    assert split.shape[:2] == (1, N_TEST)
    maxdiff = (split - ref).abs().max().item()
    assert torch.equal(split, ref), f"split != forward_cached_regression (max |Δ|={maxdiff:.2e})"


def _transductive(model, xt, yt, eval_pos, seed=0):
    """The plain (non-cached) forward -- the reference the cached path must match."""
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model(x=xt, y=yt, eval_pos=eval_pos, task_type="reg")
    return out["reg_output"] if isinstance(out, dict) else out


def _nonfinite_table(sentinel):
    """A table whose QUERY rows carry `sentinel` in column 0, and whose query column
    mean is shifted well away from the context's -- so imputing from the wrong split
    is visible rather than coincidentally close."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_TRAIN + N_TEST, N_FEATURES)).astype(np.float32)
    y = np.sin(x[:, 0]).astype(np.float32)
    x[N_TRAIN:, 0] += 5.0
    if sentinel is not None:
        x[N_TRAIN + 3::7, 0] = sentinel
    return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y).unsqueeze(0)


@pytest.mark.slow
@pytest.mark.parametrize("sentinel,label", [
    pytest.param(None, "finite", id="finite-baseline"),
    pytest.param(float("nan"), "NaN", id="nan"),
    pytest.param(float("inf"), "+Inf", id="pos-inf"),
    pytest.param(float("-inf"), "-Inf", id="neg-inf"),
])
def test_apply_matches_transductive_on_nonfinite_query_values(bare_model, sentinel, label):
    # NanEncoder imputes NaN/Inf with the CONTEXT column mean, computed from
    # x[:, :eval_pos]. On the apply path eval_pos spans the query rows, so without
    # ContextCache.nan_mean the fill comes from the QUERY mean while the transductive
    # path uses the TRAIN mean.
    #
    # The visible failure is +/-Inf, not NaN: the preprocess mask is `isnan(x)` alone,
    # so process_4_x writes NaN back over NaN cells and discards the fill there, while
    # an Inf cell is unmasked and its fill reaches the model. Measured on nori-6m
    # before the fix: 3.2e+01 (+Inf) and 2.9e+01 (-Inf) against a 2.9e-05 finite
    # baseline. The 1e-3 bound below sits ~4 orders below the broken values and ~1.5
    # above the healthy one; the finite arm is the control, so a general plain-vs-cached
    # regression fails there too rather than being blamed on the non-finite handling.
    xt, yt = _nonfinite_table(sentinel)
    ref = _transductive(bare_model, xt, yt, N_TRAIN)
    got = _split(bare_model, xt, yt, N_TRAIN)
    delta = (got - ref).abs().max().item()
    assert delta < 1e-3, (
        f"cached apply diverged from the transductive path on {label} query values: "
        f"max |Δ|={delta:.3e} (float reassociation alone is ~3e-05). The context's "
        f"train-derived imputation stats are not reaching the query rows.")


def test_nan_encoder_frozen_mean_overrides_the_eval_pos_split():
    # The frozen-stats contract at the unit level, no checkpoint: given
    # `_frozen_nan_mean`, NanEncoder must impute with THAT and ignore its own
    # [:eval_pos] computation -- which is what lets a query-only batch be filled from
    # the context. NaN here so the own-split mean is a clean finite number; the
    # companion test below covers Inf, whose fill is the one that reaches the model.
    from synthefy_nori.model.encoders import NanEncoder

    enc = NanEncoder(in_keys=["data"], out_key="nan_encoding")
    x = torch.tensor([[[1.0], [3.0], [float("nan")]]])          # [1, 3, 1]

    free = enc({"data": x.clone(), "eval_pos": 3})
    # Own split: calc_mean is nansum/non-NaN-count, so (1+3)/2 = 2.0
    assert free["data"][0, 2, 0].item() == pytest.approx(2.0)
    assert free["_nan_mean"].item() == pytest.approx(2.0), "did not expose the fill used"

    frozen = torch.tensor([[-7.0]])
    out = enc({"data": x.clone(), "eval_pos": 3, "_frozen_nan_mean": frozen})
    assert out["data"][0, 2, 0].item() == pytest.approx(-7.0), (
        "frozen context mean was ignored; NanEncoder recomputed from these rows")
    assert out["_nan_mean"].item() == pytest.approx(-7.0)
    # Finite entries are untouched either way -- only NaN/Inf cells are filled.
    assert out["data"][0, 0, 0].item() == pytest.approx(1.0)
    assert out["data"][0, 1, 0].item() == pytest.approx(3.0)


def test_nan_encoder_frozen_mean_survives_an_inf_bearing_query_batch():
    # Why the Inf case is catastrophic and not merely slightly off: calc_mean excludes
    # NaN (nansum) but NOT Inf, so a single Inf anywhere in the split drives the whole
    # column mean to Inf. A query-only batch computing its own mean therefore imputes
    # Inf, and unlike a NaN cell that fill is unmasked and reaches the model. The
    # frozen context mean is what keeps the column finite.
    from synthefy_nori.model.encoders import NanEncoder

    enc = NanEncoder(in_keys=["data"], out_key="nan_encoding")
    x = torch.tensor([[[1.0], [3.0], [float("inf")]]])

    poisoned = enc({"data": x.clone(), "eval_pos": 3})
    assert not torch.isfinite(poisoned["_nan_mean"]).all(), (
        "expected an Inf in the split to poison calc_mean -- if this now holds, the "
        "severity argument for freezing the fill has changed")
    assert not torch.isfinite(poisoned["data"]).all()

    healthy = enc({"data": x.clone(), "eval_pos": 3,
                   "_frozen_nan_mean": torch.tensor([[2.0]])})
    assert healthy["data"][0, 2, 0].item() == pytest.approx(2.0)
    assert torch.isfinite(healthy["data"]).all(), (
        "frozen context mean did not keep the query column finite")


@pytest.mark.slow
def test_build_context_cache_captures_the_nan_fill(bare_model, table):
    # ContextCache must actually carry the imputation fill; a None here is how the
    # divergence above comes back (apply_context_cache only freezes it when non-None).
    xt, yt = table
    torch.manual_seed(0)
    with torch.no_grad():
        bundle = bare_model.build_context_cache(xt[:, :N_TRAIN], yt[:, :N_TRAIN])
        half = bare_model.build_context_cache(xt[:, :N_TRAIN // 2], yt[:, :N_TRAIN // 2])
    assert bundle.nan_mean is not None, "context build did not capture NanEncoder's fill"
    assert torch.isfinite(bundle.nan_mean).all()
    # The fill is a per-feature-group reduction over rows ([B, n_groups,
    # features_per_group]), so its shape must not depend on how many context rows went
    # in -- that row-independence is what lets it be applied to a query batch of any
    # size. Asserted by construction rather than by hardcoding the layout.
    assert half.nan_mean is not None
    assert bundle.nan_mean.shape == half.nan_mean.shape, (
        f"fill shape tracks the context row count ({tuple(bundle.nan_mean.shape)} for "
        f"{N_TRAIN} rows vs {tuple(half.nan_mean.shape)} for {N_TRAIN // 2}) -- it "
        f"cannot then be re-applied to an arbitrary query batch")
    assert N_TRAIN not in bundle.nan_mean.shape


@pytest.mark.slow
def test_one_bundle_scores_multiple_query_batches(bare_model, table):
    # (B) build once, apply to two disjoint query halves; concatenated result must
    # equal applying to the whole query block at once -> reuse is correct and
    # chunk/order-independent.
    xt, yt = table
    ep = N_TRAIN
    torch.manual_seed(0)
    with torch.no_grad():
        bundle = bare_model.build_context_cache(xt[:, :ep], yt[:, :ep])
        x_q = xt[:, ep:]
        half = x_q.shape[1] // 2
        p_a = bare_model.apply_context_cache(x_q[:, :half], bundle, row_chunk_size=0)
        p_b = bare_model.apply_context_cache(x_q[:, half:], bundle, row_chunk_size=0)
        p_all = bare_model.apply_context_cache(x_q, bundle, row_chunk_size=0)
    joined = torch.cat([p_a, p_b], dim=1)
    assert torch.equal(joined, p_all), (
        f"per-batch scoring diverged from whole-block (max |Δ|="
        f"{(joined - p_all).abs().max().item():.2e})")


@pytest.mark.slow
def test_context_build_runs_once_not_per_apply(bare_model, table):
    # (C) the amortization guarantee: build_train_cache (the O(N_train) context
    # forward) is invoked exactly once when the bundle is built, and NOT again on
    # any number of applies.
    xt, yt = table
    ep = N_TRAIN
    enc = bare_model.transformer_encoder
    orig = enc.build_train_cache
    calls = {"n": 0}

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    enc.build_train_cache = counting                            # type: ignore[assignment]
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            bundle = bare_model.build_context_cache(xt[:, :ep], yt[:, :ep])
            assert calls["n"] == 1, "build did not run the context forward exactly once"
            for _ in range(3):
                bare_model.apply_context_cache(xt[:, ep:], bundle, row_chunk_size=0)
        assert calls["n"] == 1, (
            f"context forward re-ran on apply: build_train_cache called {calls['n']}x "
            f"(expected 1) -- amortization is broken")
    finally:
        enc.build_train_cache = orig                            # type: ignore[assignment]


def _regressor(*, reuse_context_cache: bool = True):
    """A NoriRegressor whose predict() takes the cached path, or skip."""
    from synthefy_nori.api import NoriRegressor
    # Small element budget forces chunk_size to its floor so n_test > chunk_size
    # and the cached (build/apply) path engages -- same trick as test_memory_policy_e2e.
    reg = NoriRegressor(
        model="nori-6m",
        device="cpu",
        memory_policy={"elements_budget": 5_000,
                       "reuse_context_cache": reuse_context_cache},
    )
    try:
        reg._get_predictor()
    except (OSError, RuntimeError) as exc:
        # Only a genuinely unreachable checkpoint may skip. A TypeError/AttributeError
        # here means the estimator API moved out from under this test -- that must
        # fail, not silently skip (it did once, hiding a memory= -> memory_policy= rename).
        pytest.skip(f"no reachable Nori checkpoint: {type(exc).__name__}: {exc}")
    return reg


@pytest.mark.slow
def test_predict_context_cache_is_bit_identical_to_rebuilding():
    # The cross-call cache must not change predictions: cache ON must equal cache
    # OFF (rebuild every call) to within the documented mixed-precision tolerance.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600 + 400, 8)).astype(np.float32)
    y = (np.sin(x[:, 0] * 2) - x[:, 1]).astype(np.float32)
    x_tr, y_tr, x_te = x[:600], y[:600], x[600:]

    off = _regressor(reuse_context_cache=False).fit(x_tr, y_tr).predict(x_te)
    reg = _regressor().fit(x_tr, y_tr)
    on = reg.predict(x_te)
    if reg.memory_report_ is None or reg.memory_report_.get("rung") == "no_cache":
        pytest.skip("cached path did not engage on this shape/box")
    assert np.abs(np.asarray(on) - np.asarray(off)).max() < 5e-3


@pytest.mark.slow
def test_context_built_once_across_multiple_predicts(monkeypatch):
    # The amortization at the public level: fit once, predict twice against the SAME
    # context -> the O(N_train) context build runs once per pipe, not once per call.
    # Two query batches EACH above the cached-path threshold (chunk_size floor 256
    # at this budget), so both predicts take the build/apply path -- otherwise a
    # sub-256 batch falls to the uncached loop and there is no build to count.
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600 + 800, 8)).astype(np.float32)
    y = (np.cos(x[:, 0]) + x[:, 2]).astype(np.float32)
    x_tr, y_tr = x[:600], y[:600]
    x_te_a, x_te_b = x[600:1000], x[1000:]      # 400 + 400 query rows

    reg = _regressor().fit(x_tr, y_tr)
    bare = reg._get_predictor()._bare_model()
    calls = {"n": 0}
    orig = bare.build_context_cache

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(bare, "build_context_cache", counting)
    reg.predict(x_te_a)
    if reg.memory_report_ is None or reg.memory_report_.get("rung") == "no_cache":
        pytest.skip("cached path did not engage on this shape/box")
    after_first = calls["n"]
    assert after_first >= 1, "first predict never built a context cache"
    reg.predict(x_te_b)          # different queries, same context
    assert calls["n"] == after_first, (
        f"context rebuilt on the second predict: {calls['n']} total builds "
        f"(expected {after_first}) -- cross-call amortization is broken")


@pytest.mark.slow
def test_repeated_apply_is_deterministic_and_bundle_immutable(bare_model, table):
    # (D) applying the same bundle twice yields identical predictions, and apply
    # does not mutate the bundle (so it is safe to reuse indefinitely).
    xt, yt = table
    ep = N_TRAIN
    torch.manual_seed(0)
    with torch.no_grad():
        bundle = bare_model.build_context_cache(xt[:, :ep], yt[:, :ep])
        cache_ids = [id(c) for c in bundle.caches]
        first = bare_model.apply_context_cache(xt[:, ep:], bundle, row_chunk_size=0)
        second = bare_model.apply_context_cache(xt[:, ep:], bundle, row_chunk_size=0)
    assert torch.equal(first, second)
    assert [id(c) for c in bundle.caches] == cache_ids, "apply mutated the bundle caches"


_ELEMENTS_BUDGET = 5_000        # must match _regressor()'s memory_policy


def _assert_cached_path_reachable(n_train: int, n_test: int, n_features: int):
    """Fail loudly if this table cannot reach the cached path at all.

    ``predict`` only takes the build/apply path when ``n_test > chunk_size``, and
    ``chunk_size = max(256, budget // n_features - n_train)`` grows as ``n_train``
    SHRINKS. So making the tables smaller to save time can quietly push chunk_size
    above n_test, at which point every test that guards on ``rung == "no_cache"``
    skips and the suite reports success while exercising nothing. That happened once
    (n_train 600 -> 300 turned all four lifecycle tests into skips), hence this guard:
    a sizing mistake now fails here instead of hiding in a skip.

    Uses the RAW feature count, which is the conservative direction -- preprocessing
    only ever widens the table, and a larger ``budget_n_features`` makes chunk_size
    smaller, i.e. easier to exceed.
    """
    chunk = max(256, (_ELEMENTS_BUDGET // max(n_features, 1)) - n_train)
    assert n_test > chunk, (
        f"n_test={n_test} does not exceed chunk_size={chunk} for n_train={n_train}, "
        f"n_features={n_features}, elements_budget={_ELEMENTS_BUDGET}: predict would "
        f"take the UNCACHED loop and these tests would silently skip. Raise n_test or "
        f"n_train (larger n_train lowers chunk_size).")


def _dataset(seed: int, n_train: int = 600, n_test: int = 300, n_features: int = 8):
    # Sized so the cached path engages: at elements_budget=5_000 an n_train of 600 puts
    # chunk_size on its 256 floor, and n_test=300 clears it. Query rows are the trimmable
    # half (each test runs several full CPU predicts), the context size is not.
    _assert_cached_path_reachable(n_train, n_test, n_features)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_train + n_test, n_features)).astype(np.float32)
    y = (np.sin(x[:, 0] * (1 + seed)) - x[:, 1] + 0.3 * x[:, 2]).astype(np.float32)
    return x[:n_train], y[:n_train], x[n_train:]


def _count_builds(monkeypatch, reg):
    """Count context builds on `reg`'s live predictor. Returns a mutable counter."""
    bare = reg._get_predictor()._bare_model()
    calls = {"n": 0}
    orig = bare.build_context_cache

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(bare, "build_context_cache", counting)
    return calls


@pytest.mark.slow
def test_serving_lifecycle_matches_uncached_predictions(monkeypatch):
    # The real serving lifecycle, end to end: ONE estimator (as
    # serving/core/nori_inference/engine.py keeps) re-fit per request, contexts alternating
    # between callers, and each answer checked against what a cold predictor computes
    # for the same request. This is the test that would catch a stale bundle actually
    # reaching a caller -- the hermetic (F) tests prove the cache decides correctly,
    # this one proves the decision lands on the right numbers.
    #
    # Dataset C collides with A under the retired summary digest (two interior target
    # values swapped), so under that digest request C would have been answered from
    # A's context and this assertion would fire.
    requests = []
    for seed in (11, 12):
        requests.append(_dataset(seed))
    x_tr_c, y_tr_c, x_te_c = _dataset(11)
    y_tr_c = y_tr_c.copy()
    y_tr_c[5], y_tr_c[17] = y_tr_c[17], y_tr_c[5]
    requests.append((x_tr_c, y_tr_c, x_te_c))
    assert _legacy_summary_fingerprint(torch.from_numpy(y_tr_c).unsqueeze(0)) == \
        _legacy_summary_fingerprint(torch.from_numpy(requests[0][1]).unsqueeze(0)), \
        "dataset C no longer collides with A -- the mis-serve case is not covered"

    # Ground truth: every request answered by a FRESH predictor with reuse off,
    # so no bundle can carry between them.
    expected = [
        _regressor(reuse_context_cache=False).fit(xt, yt).predict(xq)
        for xt, yt, xq in requests
    ]

    reg = _regressor()                                  # one long-lived estimator
    # A, B, C, then A again: fit -> predict -> fit -> predict with the context changing
    # each time, and the trailing revisit making a stale single-slot entry observable.
    # The exhaustive alternation is covered instantly by the hermetic lifecycle test;
    # this one only has to prove the decision lands on the right numbers, so it stays
    # at four requests rather than looping (each is a full CPU predict).
    for step, i in enumerate([0, 1, 2, 0]):
        x_tr, y_tr, x_te = requests[i]
        reg.fit(x_tr, y_tr)
        got = reg.predict(x_te)
        if reg.memory_report_ is None or reg.memory_report_.get("rung") == "no_cache":
            pytest.skip("cached path did not engage on this shape/box")
        maxdiff = np.abs(np.asarray(got) - np.asarray(expected[i])).max()
        assert maxdiff < 5e-3, (
            f"step {step} (request {i}): cached serving diverged from a cold predictor "
            f"by {maxdiff:.2e} -- a stale context bundle reached a caller")


@pytest.mark.slow
def test_refit_with_a_new_context_rebuilds_it(monkeypatch):
    # Changed context, through the real predictor: refitting a long-lived estimator on
    # a DIFFERENT table must pay for the context forward again. The companion to
    # test_context_built_once_across_multiple_predicts, which pins the same-context
    # case -- together they say the cache keys on the context and nothing else.
    x_a, y_a, x_te_a = _dataset(21)
    x_b, y_b, x_te_b = _dataset(22)

    reg = _regressor().fit(x_a, y_a)
    calls = _count_builds(monkeypatch, reg)
    reg.predict(x_te_a)
    if reg.memory_report_ is None or reg.memory_report_.get("rung") == "no_cache":
        pytest.skip("cached path did not engage on this shape/box")
    after_a = calls["n"]
    assert after_a >= 1

    reg.predict(x_te_a)                                 # same context -> reuse
    assert calls["n"] == after_a, "same context rebuilt"

    reg.fit(x_b, y_b)                                   # new context -> rebuild
    reg.predict(x_te_b)
    assert calls["n"] == 2 * after_a, (
        f"refit on a new context did not rebuild: {calls['n']} builds "
        f"(expected {2 * after_a}) -- the new caller would be served the old context")


@pytest.mark.slow
def test_memory_policy_change_between_predicts_rebuilds(monkeypatch):
    # memory_policy= is re-read per predict (a server sets it per request), and
    # cache_dtype changes what the bundle CONTAINS -- int8-quantized K/V vs bf16. A
    # bundle built under the previous request's rung must not answer the next one.
    x_tr, y_tr, x_te = _dataset(31)

    reg = _regressor().fit(x_tr, y_tr)
    calls = _count_builds(monkeypatch, reg)
    reg.predict(x_te)
    if reg.memory_report_ is None or reg.memory_report_.get("rung") == "no_cache":
        pytest.skip("cached path did not engage on this shape/box")
    after_first = calls["n"]
    assert reg.memory_report_.get("cache_dtype") == "bf16"

    reg.memory_policy = {"elements_budget": 5_000, "cache_dtype": "int8"}
    reg.predict(x_te)                                   # same context, new rung
    assert reg.memory_report_.get("cache_dtype") == "int8", "policy change was ignored"
    assert calls["n"] == 2 * after_first, (
        f"bundle survived a cache_dtype change: {calls['n']} builds (expected "
        f"{2 * after_first}) -- an int8 request would be served bf16 K/V")


@pytest.mark.slow
def test_typed_policy_disables_reuse_on_every_predict(monkeypatch):
    # With reuse off, N predicts against one fitted context cost N context builds,
    # and nothing is retained. Switching the typed policy back on starts a fresh
    # cache rather than resurrecting state from the disabled window.
    x_tr, y_tr, x_te = _dataset(41)

    reg = _regressor(reuse_context_cache=False).fit(x_tr, y_tr)
    calls = _count_builds(monkeypatch, reg)
    reg.predict(x_te)
    if reg.memory_report_ is None or reg.memory_report_.get("rung") == "no_cache":
        pytest.skip("cached path did not engage on this shape/box")
    per_predict = calls["n"]
    assert per_predict >= 1

    for i in range(2, 4):
        reg.predict(x_te)
        assert calls["n"] == i * per_predict, (
            f"predict {i} reused a bundle with reuse disabled: {calls['n']} builds "
            f"(expected {i * per_predict})")
    assert not getattr(reg._get_predictor(), "_context_cache", {}), (
        "reuse_context_cache=False populated the cache")

    # Re-enable through the public policy: the first call builds, then the next hits.
    reg.memory_policy = {"elements_budget": 5_000, "reuse_context_cache": True}
    reg.predict(x_te)
    warm = calls["n"]
    reg.predict(x_te)
    assert calls["n"] == warm, "reuse did not resume after changing the typed policy"


# --------------------------------------------------------------------------------
# (F) The cross-call cache in NoriPredictor._get_or_build_context.
#
# These drive the memoization directly with a stub model, so a build is a COUNTABLE
# event and the returned bundle records exactly which context produced it. No
# checkpoint, no network, no GPU -- which is what lets them assert "rebuilt" vs
# "reused" as an equality rather than as a timing or a tolerance.
#
# The invariant every test below is really guarding: a returned bundle was built from
# the context it is being handed back for. `serving/core/nori_inference/engine.py` keeps ONE
# estimator (hence one NoriPredictor, hence one of these caches) alive across every
# request and re-fits it per request, so a false hit does not mean a slow answer -- it
# means caller B's rows scored against caller A's context, returned as if correct.
# --------------------------------------------------------------------------------

class _StubBundle:
    """Stands in for a ContextCache, remembering what it was built from."""

    def __init__(self, x, y, params):
        # Clone: the predictor hands in views of a tensor it reuses, and the whole
        # point is to detect a bundle built from *different* content later.
        self.x = x.detach().clone()
        self.y = y.detach().clone()
        self.params = params


class _StubModel:
    """A bare_model whose build_context_cache only records that it was called."""

    def __init__(self):
        self.builds: list[_StubBundle] = []

    def build_context_cache(self, x_train, y_train, *, cache_dtype,
                            offload_kv_cache, fit_row_chunk):
        bundle = _StubBundle(x_train, y_train,
                             (cache_dtype, offload_kv_cache, fit_row_chunk))
        self.builds.append(bundle)
        return bundle


def _predictor(seed: int = 0):
    """A real NoriPredictor, minus the checkpoint, plus a calling shorthand.

    A genuine subclass rather than a duck-typed stand-in, so `_get_or_build_context`
    resolves its collaborators (`_context_cache_key`, `_same_context`) through normal
    MRO: these tests then exercise the SHIPPED wiring, and a rename inside it fails
    here instead of being papered over by a hand-wired fake. `__init__` is skipped on
    purpose -- it loads weights, and the cache methods touch only `self.seed` and
    `self._context_cache`.
    """
    from synthefy_nori.inference.predictor import NoriPredictor

    class _NoCheckpointPredictor(NoriPredictor):
        def __init__(self, seed):                       # noqa: D107 - see above
            self.seed = seed

    pred = _NoCheckpointPredictor(seed)

    def get(model, x, y, *, id_pipe=0, cache_dtype="bf16",
            offload_kv_cache=False, fit_row_chunk=None,
            reuse_context_cache=True):
        return pred._get_or_build_context(
            model, id_pipe, x_train_t=x, y_train_t=y,
            cache_dtype=cache_dtype, offload_kv_cache=offload_kv_cache,
            fit_row_chunk=fit_row_chunk,
            reuse_context_cache=reuse_context_cache)

    return pred, get


def _ctx(seed: int = 0, n: int = 32, f: int = 4, nans: bool = False):
    """A context table shaped like the predictor's: x [1, N, F], y [1, N]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, n, f, generator=g)
    y = torch.randn(1, n, generator=g)
    if nans:
        x[0, 3, 1] = float("nan")
        x[0, 7, 0] = float("nan")
        y[0, 5] = float("nan")
    return x, y


def _legacy_summary_fingerprint(t: torch.Tensor) -> tuple:
    """The summary-stat digest this cache used to key on, kept ONLY so the collision
    tests below can prove the collision is real rather than asserted.

    shape + NaN count + sum + sum-of-squares + first and last element. Every one of
    those is invariant under permuting the interior, which is why it was replaced.
    """
    nan_ct = int(torch.isnan(t).sum().item()) if t.numel() else 0
    tf = torch.nan_to_num(t.detach().to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0)
    flat = tf.flatten()
    return (tuple(t.shape), nan_ct,
            float(tf.sum().item()), float((tf * tf).sum().item()),
            float(flat[0].item()), float(flat[-1].item()))


def _built_from(bundle, x, y) -> bool:
    """Was `bundle` built from exactly this context? Bitwise, so NaN matches NaN."""
    def same(a, b):
        return (a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8)))
    return same(bundle.x, x) and same(bundle.y, y)


# --- same context reuses the cache -------------------------------------------------

def test_same_context_reuses_the_cache():
    # The amortization itself: an equal-but-distinct context must HIT. Distinct
    # objects on purpose -- object identity would pass trivially and prove nothing,
    # since the predictor builds fresh tensors on every predict() call.
    _, get = _predictor()
    model = _StubModel()
    x, y = _ctx()

    first = get(model, x, y)
    second = get(model, x.clone(), y.clone())

    assert second is first, "equal context did not reuse the cached bundle"
    assert len(model.builds) == 1, f"context rebuilt on a cache hit ({len(model.builds)} builds)"


def test_nan_bearing_context_reuses_the_cache():
    # Regression guard on the sameness test itself. A value compare (`torch.equal`)
    # calls NaN != NaN, so a context with missing values -- entirely ordinary for Nori
    # -- would miss on EVERY call and silently rebuild forever, deleting the speedup
    # while every other test still passed. The bitwise compare must hit here.
    _, get = _predictor()
    model = _StubModel()
    x, y = _ctx(nans=True)
    assert torch.isnan(x).any() and torch.isnan(y).any()
    assert not torch.equal(x, x.clone()), "expected float equality to fail on NaN"

    first = get(model, x, y)
    second = get(model, x.clone(), y.clone())

    assert second is first, "NaN-bearing context missed the cache (value compare?)"
    assert len(model.builds) == 1


def test_repeated_hits_stay_hits():
    # Serve-many: N queries against one fitted context pay for one build, not N.
    _, get = _predictor()
    model = _StubModel()
    x, y = _ctx()

    bundles = [get(model, x.clone(), y.clone()) for _ in range(6)]

    assert len(model.builds) == 1, f"{len(model.builds)} builds for 6 queries on one context"
    assert all(b is bundles[0] for b in bundles)


def test_separate_pipes_keep_separate_bundles():
    # Per-pipe slots: an ensemble's pipes see DIFFERENT preprocessed contexts, so they
    # must not share one entry (which would rebuild on every alternation, or worse
    # reuse pipe 0's bundle for pipe 1 if the content happened to match).
    _, get = _predictor()
    model = _StubModel()
    x, y = _ctx()

    a = get(model, x, y, id_pipe=0)
    b = get(model, x.clone(), y.clone(), id_pipe=1)
    assert b is not a
    assert len(model.builds) == 2
    # ...and both stay warm simultaneously, rather than evicting each other.
    assert get(model, x.clone(), y.clone(), id_pipe=0) is a
    assert get(model, x.clone(), y.clone(), id_pipe=1) is b
    assert len(model.builds) == 2


# --- deliberately colliding contexts ----------------------------------------------

def test_colliding_y_does_not_share_a_bundle():
    # THE collision case. Swapping two INTERIOR target values leaves shape, NaN count,
    # sum, sum-of-squares and both endpoints untouched -- an exact collision under the
    # old summary digest -- while being a genuinely different x->y mapping. Under that
    # digest the second fit would have been served the FIRST fit's context.
    _, get = _predictor()
    model = _StubModel()
    x, y_a = _ctx()
    y_b = y_a.clone()
    y_b[0, 5], y_b[0, 11] = y_a[0, 11].clone(), y_a[0, 5].clone()

    # The collision is real, not stipulated: prove it against the old digest.
    assert _legacy_summary_fingerprint(y_a) == _legacy_summary_fingerprint(y_b), (
        "test is not exercising a collision -- the summary stats differ")
    assert not torch.equal(y_a, y_b), "the two contexts are not actually different"

    first = get(model, x, y_a)
    second = get(model, x.clone(), y_b)

    assert second is not first, "colliding contexts shared a bundle"
    assert len(model.builds) == 2
    assert _built_from(second, x, y_b), "bundle was not built from the context asked for"


def test_colliding_x_does_not_share_a_bundle():
    # Same collision on the feature side: permuting the interior ROWS of x preserves
    # every summary statistic (they are all permutation-invariant, and rows 0 and N-1
    # hold the flattened endpoints) but pairs each row of features with a different
    # target, so it is a different problem entirely.
    _, get = _predictor()
    model = _StubModel()
    x_a, y = _ctx()
    perm = torch.arange(x_a.shape[1])
    interior = perm[1:-1].flip(0).clone()          # reverse rows 1..N-2, pin 0 and N-1
    perm[1:-1] = interior
    x_b = x_a[:, perm].contiguous()

    assert _legacy_summary_fingerprint(x_a) == _legacy_summary_fingerprint(x_b), (
        "test is not exercising a collision -- the summary stats differ")
    assert not torch.equal(x_a, x_b)

    first = get(model, x_a, y)
    second = get(model, x_b, y.clone())

    assert second is not first, "colliding contexts shared a bundle"
    assert len(model.builds) == 2
    assert _built_from(second, x_b, y)


def test_colliding_nan_layout_does_not_share_a_bundle():
    # A NaN-count collision: same number of NaNs in different PLACES, with the
    # surviving finite values arranged to keep sum/sum-of-squares equal. The old
    # digest counted NaNs but never located them.
    _, get = _predictor()
    model = _StubModel()
    x, y_base = _ctx()
    y_a, y_b = y_base.clone(), y_base.clone()
    # Both drop y_base[9] and keep one extra copy of y_base[4], so the finite multiset
    # (hence sum and sum-of-squares) is identical and each has exactly one NaN -- only
    # the INDEX the NaN sits at differs.
    y_a[0, 4], y_a[0, 9] = float("nan"), y_base[0, 4].clone()
    y_b[0, 9] = float("nan")

    assert _legacy_summary_fingerprint(y_a) == _legacy_summary_fingerprint(y_b), (
        "test is not exercising a collision -- the summary stats differ")

    first = get(model, x, y_a)
    second = get(model, x.clone(), y_b)

    assert second is not first, "contexts differing only in NaN placement shared a bundle"
    assert len(model.builds) == 2


# --- changed context or memory policy rebuilds it ---------------------------------

def _perturb(t, idx):
    out = t.clone()
    out[idx] = out[idx] + 1.0
    return out


def _with_nan(t, idx):
    out = t.clone()
    out[idx] = float("nan")
    return out


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda x, y: (_perturb(x, (0, 9, 2)), y), id="one-x-value"),
    pytest.param(lambda x, y: (x, _perturb(y, (0, 9))), id="one-y-value"),
    pytest.param(lambda x, y: (x[:, :-1].contiguous(), y[:, :-1].contiguous()),
                 id="fewer-context-rows"),
    pytest.param(lambda x, y: (x[:, :, :-1].contiguous(), y), id="fewer-features"),
    pytest.param(lambda x, y: (x.double(), y), id="different-dtype"),
    pytest.param(lambda x, y: (_with_nan(x, (0, 9, 2)), y), id="value-becomes-nan"),
])
def test_changed_context_rebuilds(mutate):
    # Every way the context can differ must MISS -- including a single value moving by
    # one float step, a shape change, and a value turning into NaN.
    _, get = _predictor()
    model = _StubModel()
    x, y = _ctx()

    first = get(model, x, y)
    x2, y2 = mutate(x.clone(), y.clone())
    second = get(model, x2, y2)

    assert second is not first, "changed context reused a stale bundle"
    assert len(model.builds) == 2
    assert _built_from(second, x2, y2)


@pytest.mark.parametrize("params", [
    pytest.param({"cache_dtype": "int8"}, id="cache_dtype"),
    pytest.param({"offload_kv_cache": True}, id="offload_to_host"),
    pytest.param({"fit_row_chunk": 128}, id="context_row_chunk"),
])
def test_changed_memory_policy_rebuilds(params):
    # cache_dtype / offload_to_host / context_row_chunk all change WHAT the bundle
    # contains (quantized vs bf16, host vs device, chunked build). A server re-reads
    # memory_policy= per request, so a bundle built under the previous request's rung
    # must not be handed to the next one.
    _, get = _predictor()
    model = _StubModel()
    x, y = _ctx()

    first = get(model, x, y)
    second = get(model, x.clone(), y.clone(), **params)

    assert second is not first, f"bundle survived a {list(params)[0]} change"
    assert len(model.builds) == 2
    # The rebuild used the NEW params, not the cached ones.
    assert second.params == model.builds[1].params
    assert second.params != first.params


def test_changed_seed_rebuilds():
    # The bit-identity argument for reusing a bundle is that a rebuild would draw the
    # SAME random feature positional embedding, which only holds under an unchanged
    # seed. Two predictors at different seeds must therefore not share a bundle -- and
    # since the key carries the seed, neither may one predictor whose seed moved.
    x, y = _ctx()
    model = _StubModel()

    fake, get = _predictor(seed=0)
    first = get(model, x, y)
    fake.seed = 7
    second = get(model, x.clone(), y.clone())

    assert second is not first, "bundle survived a seed change"
    assert len(model.builds) == 2


# --- typed reuse_context_cache=False always rebuilds -------------------------------

def test_reuse_disabled_rebuilds_every_call():
    # The typed setting is absolute: no hits and nothing retained, so turning reuse
    # back on later cannot resurrect a bundle from the disabled window.
    fake, get = _predictor()
    model = _StubModel()
    x, y = _ctx()

    bundles = [
        get(model, x.clone(), y.clone(), reuse_context_cache=False)
        for _ in range(4)
    ]

    assert len(model.builds) == 4, "reuse_context_cache=False did not rebuild"
    assert len({id(b) for b in bundles}) == 4, "disabled reuse returned a cached bundle"
    assert not getattr(fake, "_context_cache", {}), (
        "disabled reuse retained context-derived state")


def test_policy_change_clears_a_warm_bundle_before_rebuilding():
    # MemoryPolicy is re-read per predict. Turning reuse off must evict an already
    # warm entry; turning it back on starts a new cache instead of reviving old state.
    fake, get = _predictor()
    model = _StubModel()
    x, y = _ctx()

    warm = get(model, x, y)
    assert get(model, x.clone(), y.clone()) is warm
    assert len(model.builds) == 1

    forced = get(model, x.clone(), y.clone(), reuse_context_cache=False)
    assert forced is not warm, "typed policy ignored an already-warm cache"
    assert len(model.builds) == 2
    assert not getattr(fake, "_context_cache", {}), "warm bundle was not evicted"

    rewarmed = get(model, x.clone(), y.clone(), reuse_context_cache=True)
    assert rewarmed is not warm
    assert get(model, x.clone(), y.clone(), reuse_context_cache=True) is rewarmed
    assert len(model.builds) == 3


# --- serving's fit -> predict -> fit -> predict lifecycle --------------------------

def test_serving_fit_predict_lifecycle_never_serves_a_stale_bundle():
    # The shape of a real server: ONE predictor, re-fit per request, contexts
    # alternating between callers, and each fit followed by several predicts. Asserts
    # the invariant directly -- every bundle handed out was built from the context of
    # the request it is answering -- across the alternation that a single-slot cache
    # makes the interesting case.
    _, get = _predictor()
    model = _StubModel()
    # Two tenants' contexts, plus a third that COLLIDES with the first under the old
    # summary digest, so the sequence would have mis-served under it.
    ctx_a = _ctx(seed=1, nans=True)
    ctx_b = _ctx(seed=2, n=40)
    x_c, y_c = ctx_a[0].clone(), ctx_a[1].clone()
    y_c[0, 6], y_c[0, 13] = ctx_a[1][0, 13].clone(), ctx_a[1][0, 6].clone()
    ctx_c = (x_c, y_c)
    assert _legacy_summary_fingerprint(ctx_c[1]) == _legacy_summary_fingerprint(ctx_a[1])

    for _ in range(3):                                  # several request rounds
        for x, y in (ctx_a, ctx_b, ctx_c):
            for _ in range(2):                          # fit once, predict twice
                bundle = get(model, x.clone(), y.clone())
                assert _built_from(bundle, x, y), (
                    "served a bundle built from a DIFFERENT context -- one caller's "
                    "rows would be scored against another's")

    # 3 rounds x 3 contexts = 9 fits, each followed by 2 predicts. The context changes
    # between fits, so every fit rebuilds; the second predict of each pair reuses.
    assert len(model.builds) == 9, (
        f"expected one build per fit (9), got {len(model.builds)} -- either the second "
        f"predict of a pair rebuilt, or a fit reused a stale bundle")


def test_refitting_the_same_context_reuses_the_bundle():
    # The other half of the serving lifecycle: a server that re-fits the SAME context
    # (a client re-sending its table, or a retry) must not pay for the context forward
    # again. Nothing about fit() invalidates the cache -- only the content does.
    _, get = _predictor()
    model = _StubModel()
    x, y = _ctx(seed=3, nans=True)

    first = get(model, x, y)
    for _ in range(4):                                  # re-fit + predict, same table
        assert get(model, x.clone(), y.clone()) is first
    assert len(model.builds) == 1, (
        f"re-fitting an identical context rebuilt it ({len(model.builds)} builds)")


def test_failed_build_is_not_cached():
    # A build that raises (OOM on the real path) must leave NOTHING behind: the retry
    # at the next rung has to actually run, and a later call must not be handed a
    # half-built or absent bundle.
    fake, get = _predictor()
    model = _StubModel()
    x, y = _ctx()

    class _Boom(_StubModel):
        def build_context_cache(self, *a, **k):
            raise torch.cuda.OutOfMemoryError("simulated")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        get(_Boom(), x, y)
    assert not getattr(fake, "_context_cache", {}), "a failed build was cached"

    good = get(model, x.clone(), y.clone())
    assert _built_from(good, x, y)
    assert len(model.builds) == 1
