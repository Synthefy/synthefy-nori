"""Silent degradation: the one place that names it, and the one way to forbid it.

Parts of inference keep going by handing the model **less than the configured
pipeline promised** — a projection that fell back to raw columns, a context that
was trimmed to fit. Predictions still come out, so the run reads as "this config
is weak" rather than "the pipeline broke", and nothing downstream can tell the two
apart. In serving that trade is right; in anything *scored* it is a fabricated
number.

Every such fallback therefore warns, and warns under a category from this module.
That is the whole mechanism: no per-step argument to thread through the stack, no
environment variable, nothing to keep in sync. A caller who must not be handed a
degraded prediction escalates the category to an exception::

    from synthefy_nori import strict_pipeline

    with strict_pipeline():                 # every degradation is fatal
        model.predict(X_test)

    with strict_pipeline(SvdFallbackWarning):   # just this one
        model.predict(X_test)

or, equivalently, with the standard library::

    warnings.simplefilter("error", DegradedPipelineWarning)

Because the categories form a tree, escalation is inherited: a future fallback
adds one subclass here and every caller who already asked for a strict pipeline
gets it, with no other file to edit.

``synthefy_nori.evaluation``'s runner runs every scored predict call inside
``strict_pipeline(SvdFallbackWarning)``, so an eval records a failed row instead
of scoring a broken projection. It deliberately does NOT escalate
:class:`ContextSubsampledWarning`: trimming context to an element budget is
expected on large tables, and ``memory_policy={"allow_subsample": False}`` is the knob
for refusing it.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager


class DegradedPipelineWarning(UserWarning):
    """Nori gave the model less than the configured pipeline promised.

    The prediction is still returned. Subclasses say which fallback ran; catch
    this base class to mean "any loss of fidelity I did not ask for".

    Subclasses ``UserWarning`` so callers who already filter that keep matching.
    """


class SvdFallbackWarning(DegradedPipelineWarning):
    """The high-dimensional feature SVD failed and was worked around.

    A ``fit`` failure passes the raw (unprojected) columns through; a
    ``transform`` failure hands ``svd_all`` a single all-zero column — every
    feature dropped — or ``svd_binary`` only its non-binary keep-columns. See
    ``HighDimFeatureSelector``.
    """


class ContextSubsampledWarning(DegradedPipelineWarning):
    """Context rows were dropped so the request would fit the element budget.

    The declarative way to refuse this is ``memory_policy={"allow_subsample": False}``,
    which raises before any prediction is computed.
    """


@contextmanager
def strict_pipeline(*categories: type[Warning]):
    """Make degradation warnings fatal for the duration of the block.

    Args:
        *categories: which to escalate; defaults to
            :class:`DegradedPipelineWarning`, i.e. all of them. Pass a subclass
            to escalate just that fallback.

    The filter is restored on exit, so this is safe inside a library or a loop
    over many datasets — one strict prediction does not silence or harden the
    next one.
    """
    with warnings.catch_warnings():
        for category in categories or (DegradedPipelineWarning,):
            warnings.simplefilter("error", category)
        yield
