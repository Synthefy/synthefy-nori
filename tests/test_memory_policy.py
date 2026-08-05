"""Tests for MemoryPolicy — the ladder, the budgets, and the config surface.

Deliberately GPU-free: resolution is pure arithmetic, so every rung boundary can be
pinned in CI on every PR rather than only being exercised by whoever happens to run
a 500k-row table on a big card.

The load-bearing test is
:func:`TestAutoLadder.test_small_table_stays_bit_exact_even_though_int8_would_fit` —
it encodes the decision that int8 is *not* charged to requests with no memory
problem. That is exactly the kind of default that regresses silently a year later,
because "int8 is basically free" is true per-request and wrong as a policy.
"""
from __future__ import annotations

import io

import pytest
from pydantic import ValidationError

from synthefy_nori.inference import memory_policy as _mp
from synthefy_nori.inference.memory_policy import (
    DEFAULT_GPU_BUDGET_FRAC,
    MEMORY_PRESETS,
    MemoryPolicy,
    estimate_cache_gb,
    int8_footprint_gb,
    total_host_ram_gb,
)

# The deployed ~5.87M checkpoint's shape, as used by the WS1 Stage 2 matrix.
NORI_6M = dict(nlayers=16, embed_dim=128, bytes_per_element=2)
H100_80GB_VRAM = 79.6
H100_BUDGET = DEFAULT_GPU_BUDGET_FRAC * H100_80GB_VRAM          # ~31.8 GiB at the default fraction
HEAD_DIM = 64                                # embed_dim 128 / nhead 2
BIG_RAM = 1024.0


@pytest.fixture(autouse=True)
def _forget_emitted_warnings():
    """Clear the once-per-process warning registry between tests.

    Config warnings are de-duplicated process-wide (see memory_policy.warn_once), so
    without this a `pytest.warns` assertion would pass or fail depending on whether an
    earlier test in the same session already emitted the same text — order-dependent
    flakiness, which pytest-randomly would surface eventually.
    """
    _mp._WARNED_ONCE.clear()
    yield
    _mp._WARNED_ONCE.clear()


def cache_gb(n_rows: int, n_groups: int) -> float:
    """Full-precision cache footprint for the 6M model at this shape."""
    return estimate_cache_gb(n_context_rows=n_rows, n_groups=n_groups, **NORI_6M)


def resolve(est_gb: float, policy: MemoryPolicy | None = None, **kwargs) -> MemoryPolicy:
    """Resolve a policy for the 6M model on an 80 GB H100 with ample host RAM."""
    kwargs.setdefault("total_vram_gb", H100_80GB_VRAM)
    kwargs.setdefault("total_ram_gb", BIG_RAM)
    return (policy or MemoryPolicy()).resolve(
        est_cache_gb=est_gb, bytes_per_element=2, head_dim=HEAD_DIM, **kwargs
    )


class TestFootprintArithmetic:
    def test_cache_scales_with_layers_rows_groups_and_width(self):
        # One key and one value vector of width embed_dim per (layer, group, row)
        # -> the factor 2.
        expected = (16 * 8 * 50_000 * 2 * 128 * 2) / (1024 ** 3)
        assert cache_gb(50_000, 8) == pytest.approx(expected)

    def test_int8_saves_about_1_9x_against_bf16_not_4x(self):
        # 4x holds only against fp32. Against bf16 it is 2x minus the fp32 absmax
        # scale (one per head_dim vector) -> ~1.9x. Over-claiming here is exactly
        # what makes a resident budget optimistic.
        int8_gb = int8_footprint_gb(8.0, bytes_per_element=2, head_dim=HEAD_DIM)
        assert int8_gb == pytest.approx(4.25)          # 4.0 payload + 0.25 scale
        assert 8.0 / int8_gb == pytest.approx(1.88, abs=0.01)

    def test_scale_overhead_shrinks_as_head_dim_grows(self):
        wide = int8_footprint_gb(8.0, bytes_per_element=2, head_dim=256)
        narrow = int8_footprint_gb(8.0, bytes_per_element=2, head_dim=16)
        assert wide < narrow

    def test_host_ram_probe_is_positive_or_a_declared_unknown(self):
        # 0.0 means "cannot tell" and must disable offload rather than invent a
        # number -- guessing gets the process OOM-killed instead of degrading.
        assert total_host_ram_gb() >= 0.0

    def test_cgroup_limit_file_is_closed(self, monkeypatch):
        limit_file = io.StringIO(str(2 * _mp.BYTES_PER_GIB))
        monkeypatch.setattr(_mp, "_CGROUP_LIMIT_PATHS", ("/fake/memory.max",))
        monkeypatch.setattr("builtins.open", lambda _path: limit_file)

        assert _mp._cgroup_memory_limit_gb() == 2.0
        assert limit_file.closed

    def test_cgroup_limit_does_not_leak_resource_warning_into_notes(
        self, monkeypatch, tmp_path
    ):
        limit_path = tmp_path / "memory.max"
        limit_path.write_text(str(2 * _mp.BYTES_PER_GIB), encoding="utf-8")
        monkeypatch.setattr(_mp, "_CGROUP_LIMIT_PATHS", (str(limit_path),))

        with _mp.capture_policy_notes() as notes:
            assert _mp._cgroup_memory_limit_gb() == 2.0
        assert notes == []


class TestBudgets:
    def test_fraction_is_relative_to_the_card(self):
        # The whole point of a fraction: one setting, portable across hardware.
        policy = MemoryPolicy()
        assert policy.gpu_budget(79.6) == pytest.approx(31.84)
        assert policy.gpu_budget(143.0) == pytest.approx(57.2)

    def test_absolute_override_wins_for_a_co_tenanted_gpu(self):
        policy = MemoryPolicy(gpu_budget_absolute_gb=20.0)
        assert policy.gpu_budget(143.0) == 20.0

    def test_host_budget_is_also_a_fraction(self):
        # A flat 128 GB "fits" on a 32 GB laptop by arithmetic; offload then
        # proceeds and the kernel OOM-kills the process. A fraction cannot.
        laptop = MemoryPolicy().host_budget(32.0)
        assert laptop == pytest.approx(8.0)
        assert laptop < 128.0

    def test_unknown_host_ram_yields_no_host_budget(self):
        assert MemoryPolicy().host_budget(0.0) == 0.0

    def test_out_of_range_fractions_are_rejected(self):
        with pytest.raises(ValidationError):
            MemoryPolicy(gpu_budget_frac=40)        # 40x VRAM, surely a typo
        with pytest.raises(ValidationError):
            MemoryPolicy(gpu_budget_frac=0)
        with pytest.raises(ValidationError):
            MemoryPolicy(host_budget_absolute_gb=-5)


class TestAutoLadder:
    def test_small_table_stays_bit_exact_even_though_int8_would_fit(self):
        """The decision: never spend accuracy on a request that fits anyway.

        6M at 50k rows x 16 features is ~3 GiB of cache against a ~32 GiB budget on
        an 80 GB H100. Both precisions fit comfortably, so the tie must break toward
        bit-exact. Under the previous default this shape was quantized and returned
        a max prediction difference of 0.046875 versus the un-quantized path.
        """
        policy = resolve(cache_gb(50_000, 8))
        assert policy.rung == "resident_bf16"
        assert policy.cache_dtype == "bf16"
        assert policy.is_bit_exact
        assert policy.cache and not policy.offload_to_host

    def test_quantizes_only_to_keep_the_cache_resident(self):
        # bf16 overflows the budget, int8 does not -> quantize rather than pay PCIe
        # streaming. The one lossy rung, reached only because the rung above it
        # could not serve the request.
        policy = resolve(H100_BUDGET * 1.5)
        assert policy.rung == "resident_int8"
        assert policy.cache_dtype == "int8"
        assert not policy.is_bit_exact
        assert not policy.offload_to_host

    def test_offloads_at_full_precision_when_host_ram_can_hold_it(self):
        # Offload transport is bit-exact at either precision, so quantizing here
        # would buy only PCIe bandwidth -- accuracy is not spent for speed.
        policy = resolve(H100_BUDGET * 4)
        assert policy.rung == "offload_bf16"
        assert policy.is_bit_exact
        assert policy.offload_to_host and policy.cache

    def test_offloads_quantized_only_when_bf16_will_not_fit_host(self):
        # bf16 = 127 GiB exceeds a 100 GiB host budget; int8 (~68 GiB) fits.
        policy = resolve(H100_BUDGET * 4, total_ram_gb=400.0)
        assert policy.rung == "offload_int8"
        assert not policy.is_bit_exact

    def test_falls_to_plain_loop_when_host_cannot_take_it_either(self):
        with pytest.warns(UserWarning, match="could not help"):
            policy = resolve(H100_BUDGET * 4, total_ram_gb=4.0)
        assert policy.rung == "plain_loop"
        assert not policy.cache

    def test_unknown_host_ram_disables_offload_rather_than_guessing(self):
        with pytest.warns(UserWarning, match="could not help"):
            policy = resolve(H100_BUDGET * 4, total_ram_gb=0.0)
        assert policy.rung == "plain_loop"
        assert not policy.cache

    def test_offload_to_host_false_reproduces_legacy_skip_the_cache(self):
        policy = resolve(H100_BUDGET * 4, MemoryPolicy(offload_to_host=False))
        assert policy.rung == "plain_loop"
        assert not policy.cache and not policy.offload_to_host

    def test_rungs_are_monotonic_in_table_size(self):
        # Growing the table may only move us down the ladder, never back up. These
        # four row counts walk every rung in order on an 80 GB card.
        expected = ["resident_bf16", "resident_int8", "offload_bf16", "plain_loop"]
        seen = [resolve(cache_gb(n, 8)).rung
                for n in (10_000, 655_000, 2_000_000, 20_000_000)]
        assert seen == expected

    def test_ineligible_request_reports_no_cache(self):
        policy = resolve(0.0, cache_eligible=False)
        assert policy.rung == "no_cache"
        assert not policy.cache and policy.is_bit_exact

    def test_resolved_policy_is_fully_concrete(self):
        for est in (0.1, H100_BUDGET * 1.5, H100_BUDGET * 4):
            policy = resolve(est)
            assert policy.is_resolved
            assert policy.cache_dtype in ("bf16", "int8")
            assert isinstance(policy.offload_to_host, bool)
            assert policy.gpu_budget_absolute_gb is not None
            assert policy.host_budget_absolute_gb is not None

    def test_resolve_returns_a_memory_policy_not_a_second_type(self):
        # One type in, one type out: a resolved policy is just one with no "auto"
        # left, which is why there is no separate decision object to keep straight.
        assert isinstance(resolve(1.0), MemoryPolicy)


class TestPinnedPrecision:
    def test_exact_preset_offloads_rather_than_quantizing(self):
        # "exact" must stay exact: when it will not fit the GPU the answer is host
        # RAM (bit-exact transport), never a silent downgrade to int8.
        policy = resolve(H100_BUDGET * 4, MemoryPolicy.coerce("exact"))
        assert policy.rung == "offload_bf16"
        assert policy.cache_dtype == "bf16" and policy.is_bit_exact

    def test_int8_pinned_quantizes_even_when_bf16_would_fit(self):
        policy = resolve(cache_gb(50_000, 8), MemoryPolicy(cache_dtype="int8"))
        assert policy.rung == "resident_int8"

    def test_max_context_starts_quantized(self):
        # It does not force offload: starting int8 already frees the VRAM, and the
        # ladder will offload only if even that will not fit.
        policy = resolve(cache_gb(50_000, 8), MemoryPolicy.coerce("max_context"))
        assert policy.rung == "resident_int8"
        assert policy.cache_dtype == "int8"

    def test_off_preset_never_caches(self):
        policy = resolve(cache_gb(50_000, 8), MemoryPolicy.coerce("off"))
        assert policy.rung == "no_cache"
        assert not policy.cache


class TestConfigSurface:
    def test_presets_cover_the_documented_names(self):
        assert set(MEMORY_PRESETS) == {"exact", "max_context", "off"}
        for name in MEMORY_PRESETS:
            assert isinstance(MemoryPolicy.coerce(name), MemoryPolicy)

    def test_none_means_the_defaults(self):
        # There is deliberately no "auto" preset: omitting memory_policy= already means
        # the defaults, so a name for it would be a second spelling of nothing.
        assert MemoryPolicy.coerce(None) == MemoryPolicy()
        with pytest.raises(ValueError, match="unknown memory preset"):
            MemoryPolicy.coerce("auto")

    def test_dict_config_is_accepted(self):
        # The config-file path: parsed YAML/JSON lands here.
        assert MemoryPolicy.coerce({"gpu_budget_frac": 0.25}).gpu_budget_frac == 0.25

    def test_a_policy_passes_through_unchanged(self):
        policy = MemoryPolicy(cache_dtype="bf16")
        assert MemoryPolicy.coerce(policy) is policy

    def test_typos_are_rejected_not_ignored(self):
        # The failure mode this guards: a silently-ignored key on a knob whose whole
        # job is "do not lose accuracy" would be wrong forever with no signal.
        with pytest.raises(ValidationError):
            MemoryPolicy.coerce({"int_8": False})
        with pytest.raises(ValidationError):
            MemoryPolicy.coerce({"gpu_budget": 20})
        with pytest.raises(ValidationError):
            MemoryPolicy.coerce({"offload": True})       # renamed to offload_to_host

    def test_unknown_preset_and_wrong_type_are_rejected(self):
        with pytest.raises(ValueError, match="unknown memory preset"):
            MemoryPolicy.coerce("fastest")
        with pytest.raises(TypeError):
            MemoryPolicy.coerce(3.5)

    def test_unknown_rung_is_rejected(self):
        with pytest.raises(ValidationError):
            MemoryPolicy(rung="resident_fp8")

    def test_policy_is_frozen_and_hashable(self):
        policy = MemoryPolicy()
        assert hash(policy) is not None
        with pytest.raises(ValidationError):
            policy.cache_dtype = "int8"

    def test_every_field_documents_itself(self):
        # These descriptions are the user-facing documentation for the knobs and
        # become the JSON Schema if the preset is ever exposed over the API.
        for name, field in MemoryPolicy.model_fields.items():
            assert field.description, f"{name} has no description"


class TestEscalation:
    def test_context_row_chunk_escalation_is_recorded(self):
        policy = resolve(cache_gb(50_000, 8)).escalated("context_row_chunk", context_row_chunk=2048)
        assert policy.rung == "context_row_chunk"
        assert policy.context_row_chunk == 2048
        assert policy.cache

    def test_plain_loop_escalation_turns_the_cache_off(self):
        policy = resolve(cache_gb(50_000, 8)).escalated("plain_loop")
        assert policy.rung == "plain_loop"
        assert not policy.cache

    def test_subsample_count_is_recorded(self):
        # The reviewer's question: where does a dropped context show up? Here, so it
        # survives past the log line into memory_report_.
        policy = resolve(cache_gb(50_000, 8)).escalated(
            "plain_loop", dropped_context_rows=1234)
        assert policy.dropped_context_rows == 1234
        assert "DROPPED 1234 context rows" in policy.describe()

    def test_describe_names_the_rung_and_the_numbers(self):
        text = resolve(cache_gb(50_000, 8)).describe()
        assert text.startswith("resident_bf16")
        assert "GPU budget" in text and "host budget" in text

    def test_report_round_trips_through_a_dict(self):
        # predictor.memory_report_ is exactly this dump, so it must reconstruct.
        policy = resolve(cache_gb(50_000, 8))
        assert MemoryPolicy(**policy.model_dump()) == policy


class TestPermissionsInsteadOfSentinels:
    """Every default is a literal value; adaptivity is expressed as permissions."""

    def test_declared_defaults_are_real_values_not_sentinels(self):
        # The reviewer's point: reading the signature should tell you what happens
        # without looking anything up. No field defaults to "auto".
        policy = MemoryPolicy()
        assert policy.cache_dtype == "bf16"          # a precision, not "auto"
        assert policy.allow_quantization is True
        assert policy.offload_to_host is True
        assert policy.context_row_chunk is None          # off, plainly
        assert policy.gpu_budget_frac == DEFAULT_GPU_BUDGET_FRAC
        for field in MemoryPolicy.model_fields.values():
            assert field.default != "auto", "an 'auto' sentinel crept back in"

    def test_allow_quantization_false_keeps_every_rung_bit_exact(self):
        for est in (cache_gb(50_000, 8), H100_BUDGET * 1.5, H100_BUDGET * 4):
            policy = resolve(est, MemoryPolicy(allow_quantization=False))
            assert policy.is_bit_exact, policy.rung
            assert policy.cache_dtype == "bf16"

    def test_starting_at_int8_never_considers_bf16(self):
        # Asking for int8 outright must not be silently upgraded to bf16 even when
        # bf16 would fit -- the caller wanted the smaller cache.
        policy = resolve(cache_gb(50_000, 8), MemoryPolicy(cache_dtype="int8"))
        assert policy.rung == "resident_int8"

    def test_int8_with_quantization_forbidden_is_contradictory(self):
        # One forbids quantizing, the other asks for a quantized cache. Reading them
        # together tells you nothing, so it is rejected rather than resolved by a
        # precedence rule nobody would remember.
        with pytest.raises(ValidationError, match="contradictory"):
            MemoryPolicy(cache_dtype="int8", allow_quantization=False)

    def test_offload_prefers_the_smallest_candidate_that_host_can_hold(self):
        # bf16 (128 GiB) will not fit a 100 GiB host budget but int8 (~68 GiB) will.
        est = H100_BUDGET * 4
        policy = resolve(est, total_ram_gb=400.0)     # host budget = 100 GiB
        assert policy.rung == "offload_int8"

    def test_offload_stays_bf16_when_quantization_is_forbidden(self):
        est = H100_BUDGET * 2
        policy = resolve(est, MemoryPolicy(allow_quantization=False))
        assert policy.rung == "offload_bf16"
        assert policy.is_bit_exact


class TestPredictorEnvSurface:
    """The policy itself reads no env vars; only the kill switch survives."""

    def _policy_of(self, memory_policy=None):
        # _coerced_memory_policy() touches only self.memory_policy and the environment, so a bare
        # instance is enough — no checkpoint load required.
        from synthefy_nori.inference.predictor import NoriPredictor
        predictor = NoriPredictor.__new__(NoriPredictor)
        predictor.memory_policy = memory_policy
        return predictor._coerced_memory_policy()

    def test_no_env_var_configures_the_policy(self, monkeypatch):
        for name in ("SYNTHEFY_KV_CACHE_DTYPE", "SYNTHEFY_KV_INT8",
                     "SYNTHEFY_KV_OFFLOAD", "SYNTHEFY_KV_GPU_BUDGET_FRAC",
                     "SYNTHEFY_CACHE_HOST_MAX_GB", "SYNTHEFY_FIT_ROW_CHUNK"):
            monkeypatch.setenv(name, "int8" if "DTYPE" in name else "1")
        policy = self._policy_of()
        assert policy == MemoryPolicy(), "an env var leaked into the policy"

    def test_disable_kill_switch_is_honoured(self, monkeypatch):
        monkeypatch.setenv("SYNTHEFY_DISABLE_CACHED_INFERENCE", "1")
        assert self._policy_of().cache is False

    def test_legacy_enable_zero_also_disables(self, monkeypatch):
        # Shipped and documented in public/README.md, so it keeps working.
        monkeypatch.setenv("SYNTHEFY_ENABLE_CACHED_INFERENCE", "0")
        assert self._policy_of().cache is False

    def test_removed_cache_max_gb_fails_loudly(self, monkeypatch):
        # It shipped with a DIFFERENT meaning (skip vs offload), so silently ignoring
        # it could turn a working job into an OOM. Fail instead.
        monkeypatch.setenv("SYNTHEFY_CACHE_MAX_GB", "6.0")
        with pytest.raises(RuntimeError, match="no longer supported"):
            self._policy_of()

    def test_preset_and_dict_reach_the_policy(self):
        assert self._policy_of("exact").allow_quantization is False
        assert self._policy_of({"gpu_budget_frac": 0.25}).gpu_budget_frac == 0.25


class TestIncoherentConfigsAreRejected:
    """A correctly-spelled request that cannot be honoured must fail, not be dropped.

    `extra="forbid"` catches typos. This catches the worse case: every key spelled
    right, and the caller still gets none of what they asked for.
    """

    def test_row_chunking_without_caching_is_rejected(self):
        # The headline case. context_row_chunk bounds the fit-time K/V build, which only
        # runs on the cached path, so "row chunking, no KV cache" is unreachable.
        with pytest.raises(ValidationError, match="not a\n?\\s*reachable configuration"):
            MemoryPolicy(cache=False, context_row_chunk=2048)

    @pytest.mark.parametrize("lever", [
        {"context_row_chunk": 2048},
        {"cache_dtype": "int8"},
        {"offload_to_host": True},
        {"allow_quantization": False},
    ])
    def test_every_cache_only_lever_is_rejected_with_cache_off(self, lever):
        with pytest.raises(ValidationError, match="cache=False cannot be combined"):
            MemoryPolicy(cache=False, **lever)

    def test_the_error_names_every_unhonourable_field(self):
        with pytest.raises(ValidationError) as exc:
            MemoryPolicy(cache=False, context_row_chunk=2048, cache_dtype="int8")
        for name in ("cache_dtype", "context_row_chunk"):
            assert name in str(exc.value)

    def test_off_preset_stays_legal(self):
        # cache=False alone is the "off" preset and must not trip the check --
        # offload_to_host defaults to True, so only EXPLICITLY set levers count.
        assert MemoryPolicy.coerce("off").cache is False
        assert MemoryPolicy(cache=False).cache is False

    def test_every_combination_either_works_or_raises_clearly(self):
        """Sweep the field space: no combination may silently drop a lever."""
        from itertools import product
        checked = rejected = 0
        for cache, dtype, allow_q, offload, chunk in product(
                [True, False], ["bf16", "int8"], [True, False], [True, False],
                [None, 2048]):
            kwargs = dict(cache=cache, cache_dtype=dtype,
                          allow_quantization=allow_q, offload_to_host=offload)
            if chunk is not None:
                kwargs["context_row_chunk"] = chunk
            checked += 1
            contradictory = dtype == "int8" and not allow_q
            try:
                policy = MemoryPolicy(**kwargs)
            except ValidationError:
                rejected += 1
                # Every rejection must have one of the two documented reasons.
                assert (not cache) or contradictory, (
                    f"unexplained rejection: {kwargs}")
                continue
            assert cache and not contradictory, f"should have been rejected: {kwargs}"
            # Accepted => the cache is on, so every lever can actually take effect.
            resolved = resolve(cache_gb(50_000, 8), policy)
            assert resolved.is_resolved
        assert checked == 32
        # 16 with cache=False (each sets cache-only levers explicitly here), plus the
        # 4 cache=True ones pairing cache_dtype="int8" with allow_quantization=False.
        assert rejected == 20, f"expected 20 rejections, got {rejected}"


class TestResolutionIsValidated:
    """model_copy does not re-validate; the copies used by resolve/escalated must."""

    def test_escalated_rejects_an_unknown_rung(self):
        with pytest.raises(ValidationError, match="rung must be one of"):
            MemoryPolicy().escalated("resident_fp8")

    def test_escalated_rejects_a_negative_row_count(self):
        with pytest.raises(ValidationError):
            resolve(cache_gb(50_000, 8)).escalated("plain_loop",
                                                   dropped_context_rows=-5)

    def test_resolved_policy_may_report_cache_false_with_a_concrete_dtype(self):
        # The coherence check must NOT fire on outputs: plain_loop legitimately has
        # cache=False alongside a concrete cache_dtype.
        policy = resolve(H100_BUDGET * 4, MemoryPolicy(offload_to_host=False))
        assert policy.rung == "plain_loop"
        assert policy.cache is False and policy.cache_dtype in ("bf16", "int8")

    def test_escalating_a_resolved_policy_round_trips(self):
        policy = resolve(cache_gb(50_000, 8)).escalated("context_row_chunk",
                                                        context_row_chunk=2048)
        assert MemoryPolicy(**policy.model_dump()) == policy


class TestNoCacheReportsNoBudget:
    def test_no_cache_does_not_invent_a_budget(self):
        # Previously reported 9.6 GiB (0.4 x ASSUMED_VRAM_GB) even on an 80 GB card,
        # for a decision where no budget was consulted at all.
        policy = MemoryPolicy().resolve(
            est_cache_gb=0.0, bytes_per_element=1, head_dim=1, cache_eligible=False)
        assert policy.rung == "no_cache"
        assert policy.gpu_budget_absolute_gb is None
        assert policy.host_budget_absolute_gb is None

    def test_describe_survives_absent_budgets(self):
        policy = MemoryPolicy().resolve(
            est_cache_gb=0.0, bytes_per_element=1, head_dim=1, cache_eligible=False)
        assert policy.describe() == "no_cache"

    def test_a_real_rung_still_reports_both_budgets(self):
        policy = resolve(cache_gb(50_000, 8))
        assert policy.gpu_budget_absolute_gb is not None
        assert policy.host_budget_absolute_gb is not None


class TestDeadSettingsWarn:
    """A setting another makes unreachable WARNS -- it is not silent, not fatal.

    The error/warn split follows one rule: error when we cannot know what the caller
    meant, or when guessing wrong would cost accuracy; warn when intent is clear and
    the extra setting is merely inert. Both of these have an obvious precedence AND a
    legitimate way to be reached innocently (a base config sets the fraction, a
    per-run override sets the absolute), so refusing them would break config layering
    for no safety gain.
    """

    def test_both_gpu_budget_forms_warns_and_absolute_wins(self):
        with pytest.warns(UserWarning, match="gpu_budget_frac is ignored"):
            policy = MemoryPolicy(gpu_budget_frac=0.3, gpu_budget_absolute_gb=20)
        assert policy.gpu_budget(100.0) == 20.0

    def test_both_host_budget_forms_warns_and_absolute_wins(self):
        with pytest.warns(UserWarning, match="host_budget_frac is ignored"):
            policy = MemoryPolicy(host_budget_frac=0.3, host_budget_absolute_gb=64)
        assert policy.host_budget(1000.0) == 64.0

    def test_host_budget_with_offload_disabled_warns(self):
        with pytest.warns(UserWarning, match="offload_to_host=False"):
            MemoryPolicy(offload_to_host=False, host_budget_absolute_gb=64)

    def test_cache_that_can_never_be_placed_warns(self):
        with pytest.warns(UserWarning, match="can never be placed"):
            MemoryPolicy(gpu_budget_absolute_gb=0.0, offload_to_host=False)

    def test_either_budget_form_alone_is_fine(self):
        assert MemoryPolicy(gpu_budget_frac=0.3).gpu_budget(100.0) == 30.0
        assert MemoryPolicy(gpu_budget_absolute_gb=20).gpu_budget(100.0) == 20.0

    def test_resolved_policies_carry_both_forms_without_tripping_the_check(self):
        # resolve() writes the decided absolute budget while the fraction is still
        # set, which is legal precisely because rung is no longer None.
        policy = resolve(cache_gb(50_000, 8), MemoryPolicy(gpu_budget_frac=0.3))
        assert policy.gpu_budget_frac == 0.3
        assert policy.gpu_budget_absolute_gb is not None


class TestOffloadReachability:
    """Rule 6: offload_to_host is dead when the host budget is not the larger one.

    Found by the differential prober (`scratchpad/probe_inert.py`), not by reading the
    code -- offload only engages above the GPU budget, so a host budget at or below it
    can never rescue anything.
    """

    def test_warns_only_when_a_request_actually_needed_the_dead_fallback(self):
        # 200 GiB cache against a 70 GiB GPU budget and a 50 GiB host budget: the
        # request needs to spill and offload cannot help, so say so.
        policy = MemoryPolicy(gpu_budget_absolute_gb=70.0)
        with pytest.warns(UserWarning, match="offload_to_host could not help"):
            policy.resolve(est_cache_gb=200.0, bytes_per_element=2, head_dim=HEAD_DIM,
                           total_vram_gb=H100_80GB_VRAM, total_ram_gb=200.0)

    def test_silent_on_untouched_defaults_even_when_ram_is_small(self):
        # The regression this replaced: warning up-front fired for anyone whose RAM is
        # under ~1.6x their VRAM, on a request that never left the GPU.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            MemoryPolicy().resolve(est_cache_gb=0.2, bytes_per_element=2,
                                   head_dim=HEAD_DIM, total_vram_gb=80.0,
                                   total_ram_gb=128.0)

    def test_silent_when_offload_can_actually_rescue_something(self):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")            # any warning fails the test
            resolve(cache_gb(50_000, 8))        # host 256 GiB > gpu 31.8 GiB

    def test_no_warning_when_offload_is_disabled(self):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            MemoryPolicy(offload_to_host=False).resolve(
                est_cache_gb=1.0, bytes_per_element=2, head_dim=HEAD_DIM,
                total_vram_gb=H100_80GB_VRAM, total_ram_gb=200.0)


class TestOffloadReachabilityAtConstruction:
    """Rule 6a: the half of the offload-reachability check pydantic can do alone."""

    def test_both_absolute_budgets_are_compared_at_construction(self):
        with pytest.warns(UserWarning, match="cannot engage"):
            MemoryPolicy(gpu_budget_absolute_gb=70.0, host_budget_absolute_gb=50.0)

    def test_equal_budgets_also_warn(self):
        # Equal is still unreachable: offload needs a footprint ABOVE the GPU budget
        # that is at or below the host budget, and there is none.
        with pytest.warns(UserWarning, match="cannot engage"):
            MemoryPolicy(gpu_budget_absolute_gb=50.0, host_budget_absolute_gb=50.0)

    def test_host_above_gpu_is_silent(self):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            MemoryPolicy(gpu_budget_absolute_gb=20.0, host_budget_absolute_gb=200.0)

    def test_fraction_case_is_deferred_to_resolve(self):
        # Cannot be decided here: 0.25 x RAM vs 0.4 x VRAM depends on the box.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            MemoryPolicy(gpu_budget_frac=0.9, host_budget_frac=0.05)


class TestEstimatorRedeclaresMemory:
    """``memory_policy=`` must not be frozen by the first predict.

    ``NoriRegressor`` caches its predictor because the predictor owns the loaded
    checkpoint. Every other constructor argument is therefore fixed at first use,
    which is correct for them and wrong for ``memory_policy``: it is a per-call resource
    decision, and a long-lived server (``serving/nori_serving/engine.py``) sets it
    per request on one reused estimator. If the cached predictor kept the first
    value, request 2 would silently run request 1's policy.

    No checkpoint is loaded here: the re-declaration path is the branch taken when
    ``_predictor`` already exists, so a stand-in is enough and the test stays fast.
    """

    class _PredictorStub:
        """Just enough of NoriPredictor: it only has to hold the attribute."""

        def __init__(self):
            self.memory_policy = "the-first-call's-policy"

    def _regressor_with_stub(self, memory_policy):
        from synthefy_nori import NoriRegressor

        estimator = NoriRegressor(model="nori-6m", memory_policy=memory_policy)
        estimator._predictor = self._PredictorStub()
        return estimator

    def test_a_later_memory_change_reaches_the_cached_predictor(self):
        estimator = self._regressor_with_stub(None)
        estimator.memory_policy = "off"
        assert estimator._get_predictor().memory_policy == "off"

    def test_it_tracks_every_change_not_just_the_first(self):
        estimator = self._regressor_with_stub(None)
        for value in ("off", "exact", {"cache_dtype": "int8"}, None):
            estimator.memory_policy = value
            assert estimator._get_predictor().memory_policy == value

    def test_the_value_is_passed_through_verbatim_not_coerced(self):
        # sklearn's contract: params round-trip unchanged. Coercion happens inside
        # predict, so clone()/get_params() keep seeing what the caller passed.
        estimator = self._regressor_with_stub(None)
        estimator.memory_policy = {"gpu_budget_frac": 0.3}
        assert estimator._get_predictor().memory_policy == {"gpu_budget_frac": 0.3}


class TestCoerceForService:
    """``coerce_for_service`` — the entry point a server uses instead of ``coerce``.

    It lives here, not in a serving target, because everything it does depends on this
    module's own fields: which budgets to bound, and which warnings belong to the caller.
    A serving-side copy would go stale the moment the policy gains a field, with every
    test on both sides still passing.
    """

    CEILING = 0.5

    def _coerce(self, value, **kwargs):
        kwargs.setdefault("max_host_budget_frac", self.CEILING)
        kwargs.setdefault("total_ram_gb", 200.0)
        return MemoryPolicy.coerce_for_service(value, **kwargs)

    def test_none_stays_none_so_a_server_can_tell_it_was_not_asked_for(self):
        # Distinct from coerce(None), which returns the default policy: a server needs
        # "the request said nothing" to leave its response untouched.
        assert self._coerce(None) == (None, (), ())

    def test_host_budget_frac_over_the_ceiling_is_clamped_and_named(self):
        policy, clamped, _ = self._coerce({"host_budget_frac": 0.95})
        assert policy.host_budget_frac == self.CEILING
        assert clamped == ("host_budget_frac",)

    def test_host_budget_absolute_gb_is_clamped_against_the_ram_it_is_given(self):
        policy, clamped, _ = self._coerce({"host_budget_absolute_gb": 10_000.0})
        assert policy.host_budget_absolute_gb == self.CEILING * 200.0
        assert clamped == ("host_budget_absolute_gb",)

    def test_a_budget_under_the_ceiling_is_honoured_verbatim(self):
        policy, clamped, _ = self._coerce({"host_budget_absolute_gb": 40.0})
        assert (policy.host_budget_absolute_gb, clamped) == (40.0, ())

    def test_only_the_host_budgets_are_bounded(self):
        # The point of the asymmetry: everything else spends the caller's own GPU
        # memory, so overspending is self-inflicted and passed through.
        policy, clamped, _ = self._coerce({"gpu_budget_frac": 0.99})
        assert (policy.gpu_budget_frac, clamped) == (0.99, ())

    def test_the_callers_dict_is_never_mutated(self):
        sent = {"host_budget_frac": 0.95}
        self._coerce(sent)
        assert sent == {"host_budget_frac": 0.95}

    def test_a_non_numeric_budget_is_left_for_pydantic_to_reject(self):
        # Not compared against the ceiling (that would raise TypeError from the
        # comparison); pydantic's message names the field, which is more useful.
        with pytest.raises(ValidationError):
            self._coerce({"host_budget_frac": "lots"})

    def test_notes_are_returned_to_every_caller_not_just_the_first(self):
        # warn_once de-duplicates for the life of the process, which on a server means
        # the first caller to make a mistake absorbs the only copy. The registry is
        # cleared per call so each caller hears about their own config.
        both = {"gpu_budget_frac": 0.6, "gpu_budget_absolute_gb": 10.0}
        first = self._coerce(dict(both))[2]
        second = self._coerce(dict(both))[2]
        assert first and first == second
        assert any("gpu_budget_frac" in note for note in first)

    def test_an_unambiguous_policy_produces_no_notes(self):
        assert self._coerce({"cache_dtype": "int8"})[2] == ()

    @pytest.mark.parametrize("preset", MEMORY_PRESETS)
    def test_no_preset_sets_a_host_budget_so_none_needs_clamping(self, preset):
        """Backs the claim that presets skip clamping.

        ``coerce_for_service`` only bounds dicts. That is safe exactly while no preset
        sets a host budget itself — so if a future preset does, this fails rather than
        letting an unbounded budget through under a friendly name.
        """
        policy, clamped, _ = self._coerce(preset)
        assert clamped == ()
        assert policy.host_budget_frac == _mp.DEFAULT_HOST_BUDGET_FRAC
        assert policy.host_budget_absolute_gb is None

    def test_measures_host_ram_when_none_is_given(self):
        # The default path a server actually takes; cgroup-aware, so a container sees
        # its own limit rather than the machine's.
        policy, clamped, _ = MemoryPolicy.coerce_for_service(
            {"host_budget_absolute_gb": 10_000_000.0}, max_host_budget_frac=self.CEILING
        )
        assert clamped == ("host_budget_absolute_gb",)
        assert 0 < policy.host_budget_absolute_gb < 10_000_000.0

    def test_rejections_pass_through_unchanged_for_the_caller_to_map(self):
        with pytest.raises(ValueError, match="unknown memory preset"):
            self._coerce("aggressive")
        with pytest.raises(TypeError, match="preset name"):
            self._coerce(["exact"])
        with pytest.raises(ValidationError):
            self._coerce({"int8": True})


class TestContextTooLargeError:
    """The typed error that lets a server tell a caller's budget from its own OOM."""

    def test_it_is_a_runtime_error_so_existing_handlers_keep_working(self):
        # It replaced a bare RuntimeError; anything already catching that must still catch.
        assert issubclass(_mp.ContextTooLargeError, RuntimeError)

    def test_it_is_importable_from_the_package_root(self):
        # A caller who can trigger it must be able to catch it by name without reaching
        # into a private module path.
        import synthefy_nori

        assert synthefy_nori.ContextTooLargeError is _mp.ContextTooLargeError
        assert "ContextTooLargeError" in synthefy_nori.__all__
