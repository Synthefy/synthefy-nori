"""One definition of "this policy must produce this rung", shared by two very different tests.

The serving-memory ladder is decided by hardware, so its cases were verified twice in two
places that drifted apart by construction: a local GPU test driving the engine in-process
(``tests/test_memory_policy_e2e.py``) and a smoke suite POSTing to a real Baseten deployment
(the live-deployment smoke suite, scoped to #312). Two hand-maintained lists of the same
expectations is exactly
the drift this module removes — add a rung here and both sides test it.

Stdlib only, no torch and no numpy, so the deployment smoke can import it on a bare ``python``
with no ``uv sync``. See this package's ``__init__`` for why it lives here rather than under
``ci/`` (which holds workflow entrypoints) or ``tests/`` (which pytest would try to collect).

**Forcing a rung takes a table, not just a policy.** The cached path engages only when the
QUERY set spans more than one chunk, because a single chunk has no second pass to reuse a
context cache across. The chunk size is

    chunk = max(256, elements_budget // effective_features - context_rows)

so with the DEFAULT budget (~6.6M elements) the chunk is ~312k-408k query rows and no
realistic request reaches it — everything reports ``no_cache``, correctly. The table below
therefore pins ``elements_budget`` low enough to put the chunk on its 256 floor while sending
512 query rows: two chunks, one reusable cache. Without that, every case in this file would be
vacuous while appearing to pass.
"""

from __future__ import annotations

import random

#: Context rows. Small: these cases assert WHICH rung ran, not how fast it was, and the smoke
#: has to fit inside the gateway's request-body cap.
N_CONTEXT = 400

#: Query rows — must exceed the chunk size below, or no cache is built.
N_QUERY = 512

N_FEATURES = 8

#: Puts the chunk on its 256-row floor: 4000 // 8 - 400 = 100 -> max(256, 100) = 256 < 512.
#: Also above ``(N_CONTEXT + 1) * N_FEATURES = 3208``, below which the context itself would be
#: subsampled and these cases would be measuring the wrong thing.
ELEMENTS_BUDGET = 4_000

#: Cache footprint here is ~0.0122 GiB in bf16, ~0.0065 GiB in int8 (measured). The budgets
#: below are chosen against those two numbers, so they must move together.
_EST_CACHE_GB_BF16 = 0.0122
_EST_CACHE_GB_INT8 = 0.0065


def build_table(seed: int = 0) -> dict:
    """A deterministic request body for the cases below.

    Deterministic because several cases compare predictions ACROSS requests; a fresh random
    table per call would make the exactness assertions meaningless.
    """
    rng = random.Random(seed)
    x_train = [[rng.gauss(0, 1) for _ in range(N_FEATURES)] for _ in range(N_CONTEXT)]
    return {
        "X_train": x_train,
        # A learnable signal, so a broken rung shows up as a bad answer rather than as noise.
        "y_train": [row[0] * 2.0 - row[1] for row in x_train],
        "X_test": [
            [rng.gauss(0, 1) for _ in range(N_FEATURES)] for _ in range(N_QUERY)
        ],
    }


def policy(**overrides) -> dict:
    """A policy for these cases: the forcing budget plus whatever the case sets."""
    return {"elements_budget": ELEMENTS_BUDGET, **overrides}


class Case:
    """One rung, how to force it, and what its predictions must equal.

    Attributes:
        label: what appears in test output.
        overrides: the policy fields on top of ``policy()``.
        rung: the rung this MUST resolve to. Verified by measurement on an H100, not assumed
            from reading the ladder — two of these did not match the obvious guess.
        bit_exact: whether predictions must be byte-identical to the ``resident_bf16``
            baseline. Only claimed where it was measured true; see the module note in
            ``tests/test_memory_policy_e2e.py`` about the claims that did NOT hold.
        why: why this rung is worth pinning at all.
    """

    def __init__(self, label, overrides, rung, bit_exact, why):
        self.label = label
        self.overrides = overrides
        self.rung = rung
        self.bit_exact = bit_exact
        self.why = why

    @property
    def memory_policy(self) -> dict:
        return policy(**self.overrides)


#: Every rung reachable by configuration alone, with the policy that forces it.
#:
#: ``context_row_chunk`` is deliberately ABSENT. Setting the field bounds the fit-time working
#: set but leaves the reported rung at ``resident_bf16`` -- that rung is an escalation after a
#: real OutOfMemoryError, so it cannot be forced from a request. Measured, not assumed:
#: ``context_row_chunk=128`` reports ``resident_bf16``.
CASES: tuple[Case, ...] = (
    Case(
        "default", {}, "resident_bf16", True,
        "the rung almost every request gets: full-precision cache, resident, exact",
    ),
    Case(
        "cache_dtype=int8", {"cache_dtype": "int8"}, "resident_int8", False,
        "quantize on request; the caller asked to trade accuracy for a smaller cache",
    ),
    Case(
        "allow_quantization=False", {"allow_quantization": False}, "resident_bf16", True,
        "must NOT quantize when it fits; the bit-exact promise for a caller who asked for it",
    ),
    Case(
        "gpu budget too small", {"gpu_budget_absolute_gb": 0.001}, "offload_bf16", True,
        "cannot stay resident -> host RAM. Bit-exact: offload moves bytes, it does not "
        "approximate, so this is the rung that proves memory pressure need not cost accuracy",
    ),
    Case(
        "gpu and host both too small for bf16",
        {"gpu_budget_absolute_gb": 0.001, "host_budget_absolute_gb": 0.008},
        "offload_int8", False,
        "host RAM fits int8 but not bf16, so quantizing is what makes offload possible at all",
    ),
    Case(
        "no room and offload forbidden",
        {"gpu_budget_absolute_gb": 0.001, "offload_to_host": False},
        "plain_loop", False,
        "the bottom of the ladder: no cache, the context re-processed per query chunk",
    ),
    Case(
        "cache disabled", {"cache": False}, "no_cache", False,
        "the caller opting out entirely, e.g. to isolate a suspected cache problem",
    ),
)

#: The case whose predictions the others are compared against.
BASELINE = CASES[0]

assert BASELINE.rung == "resident_bf16" and BASELINE.bit_exact, "baseline must be the exact rung"
assert len({c.label for c in CASES}) == len(CASES), "case labels must be unique"
assert {c.rung for c in CASES} >= {
    "resident_bf16", "resident_int8", "offload_bf16", "offload_int8", "plain_loop", "no_cache"
}, "a rung lost its coverage"
