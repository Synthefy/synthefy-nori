"""The client's MemoryPolicy mirror must not drift from the library's.

`src/synthefy/nori_data_models.py` is a deliberate copy of
`synthefy_nori.inference.memory_policy.MemoryPolicy`, so the client can offer a typed policy
object without depending on the model package (which would pull torch into a thin API client).

A copy needs policing. The cross-repo CI job that will do that on every merge to either repo is
specced in SynthefyPFN#119; until it exists, these run the same comparison wherever both
packages happen to be installed together — which is exactly where a divergence would bite.

The comparison is on `model_json_schema()`, not source text: canonical, order-independent, and
it covers precisely what this mirror claims to copy (names, types, bounds, enums, defaults).
Behaviour is deliberately NOT copied, so there is nothing behavioural to compare.
"""

from __future__ import annotations

import importlib.util

import pytest

from synthefy.nori_data_models import (
    MEMORY_PRESETS,
    MEMORY_RUNGS,
    MemoryAttempt,
    MemoryPolicy,
    MemoryReport,
)


def _library_available() -> bool:
    """Is a synthefy-nori that actually HAS MemoryPolicy importable?

    Probe the leaf module, not ``synthefy_nori.inference`` -- that package exists in every
    version back to 0.10, so probing it reports "available" against a published 0.12 (which is
    what the `local`/`text` extras install in CI) and these tests then fail on an import of a
    module that does not exist yet. The client's own capability check
    (``_local_memory_policy_available``) already probes the leaf; this file did not, and CI
    caught it.

    Two steps because find_spec on a dotted name imports the parent, which raises rather than
    returning None when the top-level package is absent.
    """
    if importlib.util.find_spec("synthefy_nori") is None:
        return False
    return importlib.util.find_spec("synthefy_nori.inference.memory_policy") is not None


requires_library = pytest.mark.skipif(
    not _library_available(),
    reason="needs synthefy-nori installed to compare against the authoritative model",
)

#: Fields the server DECIDES; they belong to the report, not the policy. Mirrors the split in
#: the library's openapi generator, which is the one place that split is written down.
DECIDED_FIELDS = {
    "rung", "est_cache_gb", "resident_gb", "query_chunk",
    "dropped_context_rows", "attempt_history",
}


# ------------------------------------------------- the mirror stands on its own
def test_the_policy_rejects_unknown_fields_like_the_server_does():
    with pytest.raises(ValueError):
        MemoryPolicy(int8=True)


def test_the_policy_is_frozen_so_a_shared_instance_cannot_be_mutated():
    policy = MemoryPolicy(cache_dtype="int8")
    with pytest.raises(ValueError):
        policy.cache_dtype = "bf16"


def test_bounds_are_enforced_locally():
    with pytest.raises(ValueError):
        MemoryPolicy(gpu_budget_frac=1.5)  # a fraction of VRAM cannot exceed 1
    with pytest.raises(ValueError):
        MemoryPolicy(gpu_budget_frac=0)  # exclusive minimum: 0 would mean "never cache"
    assert MemoryPolicy(gpu_budget_absolute_gb=0).gpu_budget_absolute_gb == 0


def test_the_report_keeps_fields_this_version_does_not_know():
    """extra="allow" on the RESPONSE model: a newer server may report more than we know.

    Dropping an unrecognised field would be worse than carrying it — the report is diagnostic,
    and silently shortening it hides exactly what someone is debugging.
    """
    report = MemoryReport(rung="resident_int8", something_new=17)
    assert report.rung == "resident_int8"
    assert report.model_dump()["something_new"] == 17


def test_attempt_history_keeps_newer_server_fields():
    report = MemoryReport(
        attempt_history=[{
            "pipeline_ids": [0],
            "path": "cached",
            "rung": "resident_bf16",
            "cache_dtype": "bf16",
            "offload_to_host": False,
            "context_row_chunk": None,
            "outcome": "success",
            "reason": "resolved",
            "dropped_context_rows": 0,
            "new_attempt_detail": "kept",
        }]
    )
    assert report.model_dump()["attempt_history"][0]["new_attempt_detail"] == "kept"



def test_defaults_match_the_documented_behaviour():
    policy = MemoryPolicy()
    assert policy.cache is True and policy.reuse_context_cache is True
    assert policy.cache_dtype == "bf16"
    assert policy.allow_quantization is True and policy.offload_to_host is True
    assert policy.gpu_budget_frac == 0.4 and policy.host_budget_frac == 0.25
    assert policy.allow_subsample is True


# ------------------------------------------------- and matches the authority
@requires_library
def test_the_policy_fields_match_the_library_exactly():
    from synthefy_nori.inference.memory_policy import MemoryPolicy as Authoritative

    theirs = set(Authoritative.model_fields) - DECIDED_FIELDS
    ours = set(MemoryPolicy.model_fields)
    assert ours == theirs, (
        f"the client's MemoryPolicy drifted: missing={sorted(theirs - ours)}, "
        f"stale={sorted(ours - theirs)}. Update src/synthefy/nori_data_models.py."
    )


@requires_library
def test_the_report_fields_match_the_library_exactly():
    from synthefy_nori.inference.memory_policy import MemoryPolicy as Authoritative

    # clamped/notes are added by the serving layer, not the model.
    ours = set(MemoryReport.model_fields) - {"clamped", "notes"}
    assert ours == DECIDED_FIELDS, f"report drifted: {sorted(ours)}"
    assert DECIDED_FIELDS <= set(Authoritative.model_fields)


@requires_library
@pytest.mark.parametrize("field", sorted(set(MemoryPolicy.model_fields)))
def test_each_field_matches_the_library_on_type_bounds_and_default(field):
    """Per-field so a failure names the field, not just "the schemas differ"."""
    from synthefy_nori.inference.memory_policy import MemoryPolicy as Authoritative

    mine = MemoryPolicy.model_json_schema()["properties"][field]
    theirs = Authoritative.model_json_schema()["properties"][field]
    for key in ("type", "enum", "default", "exclusiveMinimum", "minimum", "maximum", "anyOf"):
        assert mine.get(key) == theirs.get(key), (
            f"{field}.{key}: client has {mine.get(key)!r}, library has {theirs.get(key)!r}"
        )


#: `rung` is deliberately looser on the client (bare `str`) than the library's Literal, so a
#: server that adds a new rung still parses -- same forward-compat reasoning as
#: `test_attempt_history_keeps_newer_server_fields`. Excluded from the per-field type/bounds
#: comparison below; still covered by the field-NAME comparison, since the field itself must
#: still exist under the same name.
_MEMORY_ATTEMPT_LOOSENED_FIELDS = {"rung"}


@requires_library
def test_the_memory_attempt_fields_match_the_library_exactly():
    """Field-name parity for the nested type the other mirror tests never touch.

    `MemoryPolicy`/`MemoryReport` parity checks only ever key off `MemoryPolicy.model_fields`,
    so a drift in the nested `MemoryAttempt` type (a rename, a new required field) would pass
    every one of them silently -- `extra="allow"` on the client's `MemoryAttempt` means an
    unrecognised field is carried through rather than rejected, not caught.
    """
    from synthefy_nori.inference.memory_policy import MemoryAttempt as Authoritative

    ours = set(MemoryAttempt.model_fields)
    theirs = set(Authoritative.model_fields)
    assert ours == theirs, (
        f"the client's MemoryAttempt drifted: missing={sorted(theirs - ours)}, "
        f"stale={sorted(ours - theirs)}. Update src/synthefy/nori_data_models.py."
    )


@requires_library
@pytest.mark.parametrize(
    "field",
    sorted(set(MemoryAttempt.model_fields) - _MEMORY_ATTEMPT_LOOSENED_FIELDS),
)
def test_each_memory_attempt_field_matches_the_library_on_type_bounds_and_default(field):
    """Per-field, mirroring test_each_field_matches_the_library_on_type_bounds_and_default."""
    from synthefy_nori.inference.memory_policy import MemoryAttempt as Authoritative

    mine = MemoryAttempt.model_json_schema()["properties"][field]
    theirs = Authoritative.model_json_schema()["properties"][field]
    for key in ("type", "enum", "default", "exclusiveMinimum", "minimum", "maximum", "anyOf"):
        assert mine.get(key) == theirs.get(key), (
            f"MemoryAttempt.{field}.{key}: client has {mine.get(key)!r}, "
            f"library has {theirs.get(key)!r}"
        )


@requires_library
def test_the_presets_and_rungs_match_the_library():
    from synthefy_nori.inference.memory_policy import MEMORY_PRESETS as THEIR_PRESETS
    from synthefy_nori.inference.memory_policy import RUNGS as THEIR_RUNGS

    assert tuple(MEMORY_PRESETS) == tuple(THEIR_PRESETS)
    assert tuple(MEMORY_RUNGS) == tuple(THEIR_RUNGS)
