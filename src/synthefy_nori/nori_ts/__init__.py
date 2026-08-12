"""Compatibility exports for Nori time-series forecasting.

The implementation is owned by :mod:`synthefy.nori_ts`. This historical path
remains import-compatible during the package migration.
"""

try:
    from synthefy_nori.nori_ts.core import (
        DEFAULT_QUANTILES,
        NoriTSForecaster,
    )
except ModuleNotFoundError as exc:
    # The client-sync compatibility lane deliberately installs released
    # Synthefy 6.3 without dependencies. It has no canonical forecaster yet,
    # but the historical tsfeatures fallback must remain importable there.
    if exc.name not in {"synthefy.nori_ts", "synthefy.nori_ts.core"}:
        raise

    __all__ = []
else:
    __all__ = ["NoriTSForecaster", "DEFAULT_QUANTILES"]
