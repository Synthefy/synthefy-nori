"""Bundled configuration files, and the one place their names are written down.

Two phases, and their artifacts are not interchangeable:

``default_inference.json``
    The inference preprocessing ensemble (8 members) for every model not listed in
    :data:`INFERENCE_CONFIG_BY_MODEL_NAME`, including the released ``nori-6m`` /
    ``nori-30m`` / ``nori-100m`` weights and a checkpoint loaded by file path alone that
    carries no config of its own.
``model_v3_inference.json``
    The 4-member ensemble that ``nori-1-5`` uses when the caller passes no config.
    Bound to that model by name in :data:`INFERENCE_CONFIG_BY_MODEL_NAME`, and also
    stored inside the released ``nori-1-5`` checkpoint (``ci/convert_checkpoint.py
    --inference-config``), which is checked against this binding at load time.
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

import hashlib
import json
from importlib.resources import files

#: The inference config for every model without its own entry in INFERENCE_CONFIG_BY_MODEL_NAME.
DEFAULT_INFERENCE_CONFIG = "default_inference.json"

#: The inference config ``nori-1-5`` defaults to: the fixed four-member set selected on the
#: 21M v8.5.2 204k checkpoint (nori-monorepo PR 432, ``204k-fixed-search``), simplified to
#: run on the native pipeline. The simplification is
#: not prediction-identical to the searched set: every member uses legacy categorical
#: detection and native seeds (no SeedSource), and member 4 drops its Latin permutation for
#: the native random shuffle. Its scores come from a direct evaluation of this file, not
#: from the search.
MODEL_V3_INFERENCE_CONFIG = "model_v3_inference.json"

#: The bundled training/architecture config — a different phase, not an inference config.
DEFAULT_MODEL_CONFIG = "model_base.json"

#: Released model name -> the bundled inference config it uses when the caller passes none.
#: A model not listed here uses DEFAULT_INFERENCE_CONFIG. Keyed by name, not architecture, so a
#: new checkpoint never inherits a config that was tuned for a different one.
INFERENCE_CONFIG_BY_MODEL_NAME = {
    "nori-1-5": MODEL_V3_INFERENCE_CONFIG,
}


def get_default_inference_config_name(model_name: str | None) -> str:
    """Return the bundled inference-config filename a released model uses by default.

    Args:
        model_name: a released model name such as ``"nori-1-5"``, or ``None`` when the
            checkpoint was loaded by file path with no name.

    Returns:
        The filename from :data:`INFERENCE_CONFIG_BY_MODEL_NAME`, or
        :data:`DEFAULT_INFERENCE_CONFIG` for any model not listed there (including ``None``).
    """
    return INFERENCE_CONFIG_BY_MODEL_NAME.get(model_name, DEFAULT_INFERENCE_CONFIG)


def get_inference_config_digest(members: list) -> str:
    """Return ``sha256:<hex>`` of an inference config's members in canonical JSON.

    Two configs with the same members and settings get the same digest regardless of
    key order, so a config stored in a checkpoint can be compared with a bundled file.
    """
    canonical = json.dumps(members, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_path(filename: str = DEFAULT_INFERENCE_CONFIG) -> str:
    """Return an absolute path for a bundled config file.

    Defaults to :data:`DEFAULT_INFERENCE_CONFIG`.
    """
    return str(files("synthefy_nori.configs").joinpath(filename))


__all__ = [
    "DEFAULT_INFERENCE_CONFIG",
    "DEFAULT_MODEL_CONFIG",
    "INFERENCE_CONFIG_BY_MODEL_NAME",
    "MODEL_V3_INFERENCE_CONFIG",
    "config_path",
    "get_default_inference_config_name",
    "get_inference_config_digest",
]
