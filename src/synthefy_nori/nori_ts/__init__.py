"""Nori for time series — TabPFN-TS-style tabular-regression forecasting with Nori.

Frames univariate forecasting as tabular regression (after PriorLabs'
tabpfn-time-series): turn each series into a table, add time features
(running index, calendar sin/cos, auto-detected seasonal sin/cos), and regress
the target with Nori — using Nori's quantile head for probabilistic forecasts.
"""

from synthefy_nori.nori_ts.core import (
    DEFAULT_QUANTILES,
    NoriTSForecaster,
)

__all__ = ["NoriTSForecaster", "DEFAULT_QUANTILES"]
