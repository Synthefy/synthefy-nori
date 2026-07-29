"""Bundled inference configuration files.

``DEFAULT_CONFIG_NAME`` is the config every entry point falls back to — the public
``NoriRegressor``, the eval CLI and the eval model wrapper all resolve to the same file,
so evaluation measures what inference actually does.

It pins ``svd_components=48``. The ``HighDimFeatureSelector`` gate fires only above 256
features, and on those tables a rank of 48 beats the 256 previously pinned here:
+0.020 mean R² over 109 wide tables (p=0.025) and ~2.9x faster inference. At or below
256 features the selector is a passthrough, so the change is a no-op for most tables.

``reg_allordinal_poly10_adaptive_svd256.json`` is still bundled: pass it explicitly to
reproduce numbers produced before this became the default.
"""

from __future__ import annotations

from importlib.resources import files

DEFAULT_CONFIG_NAME = "reg_allordinal_poly10_adaptive_svd48.json"
LEGACY_SVD256_CONFIG_NAME = "reg_allordinal_poly10_adaptive_svd256.json"

__all__ = ["DEFAULT_CONFIG_NAME", "LEGACY_SVD256_CONFIG_NAME", "config_path"]


def config_path(filename: str) -> str:
    """Return an absolute path for a bundled config file."""
    return str(files("synthefy_nori.configs").joinpath(filename))
