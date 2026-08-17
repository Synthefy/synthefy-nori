"""Bundled configuration files, and the one place their names are written down.

Two artifacts for two phases, and they are not interchangeable:

``default_inference.json``
    The inference preprocessing ensemble (8 members) that every prediction path
    uses unless the caller passes its own. This is the *only* inference config
    the package ships.
``model_base.json``
    The training/architecture config — encoders, embedding sizes, RBF kernels.
    Consumed by ``training/cli.py``; not an inference config.

Resolve both through :func:`config_path` rather than repeating a filename. The
inference config used to record its tuning values in its own name
(``reg_allordinal_poly10_adaptive_svd256.json``); when those values changed the
name became wrong, and it was wrong in a dozen places at once. Names here
describe the phase, and the values live inside the file.
"""

from __future__ import annotations

import warnings
from importlib.resources import files

#: The one inference config the package ships.
DEFAULT_INFERENCE_CONFIG = "default_inference.json"

#: The bundled training/architecture config — a different phase, not an inference config.
DEFAULT_MODEL_CONFIG = "model_base.json"

# Renamed in 0.18.0. The old name keeps resolving for one minor version so callers
# pinning it — including installed 0.17.x code — do not break on upgrade. Delete
# this map in 0.19.0.
_RENAMED_IN_0_18 = {
    "reg_allordinal_poly10_adaptive_svd256.json": DEFAULT_INFERENCE_CONFIG,
}


def config_path(filename: str = DEFAULT_INFERENCE_CONFIG) -> str:
    """Return an absolute path for a bundled config file.

    Defaults to :data:`DEFAULT_INFERENCE_CONFIG`. Names retired in 0.18.0 still
    resolve, with a :class:`DeprecationWarning`.
    """
    renamed = _RENAMED_IN_0_18.get(filename)
    if renamed is not None:
        warnings.warn(
            f"The bundled config {filename!r} was renamed to {renamed!r} in "
            f"synthefy-nori 0.18.0 — its name recorded tuning values (svd256) that no "
            f"longer match the file. The old name still resolves for this minor version "
            f"and is removed in 0.19.0; use synthefy_nori.DEFAULT_INFERENCE_CONFIG.",
            DeprecationWarning,
            stacklevel=2,
        )
        filename = renamed
    return str(files("synthefy_nori.configs").joinpath(filename))


__all__ = ["DEFAULT_INFERENCE_CONFIG", "DEFAULT_MODEL_CONFIG", "config_path"]
