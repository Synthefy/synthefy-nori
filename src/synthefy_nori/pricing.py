"""Billable pricing for Nori inference requests."""

from __future__ import annotations

_M = 1.4
_GPU_USD_PER_SECOND = 6.50 / 3600.0

_BILLED_PER_ROW = _M * _GPU_USD_PER_SECOND * 0.0030
_BILLED_PER_COLUMN = _M * _GPU_USD_PER_SECOND * 0.12
_BILLED_FLOOR = _M * _GPU_USD_PER_SECOND * 0.6


def billable_price(row_count: int, column_count: int) -> float:
    """Return the dollar price billed for a request of the given shape."""
    linear = _BILLED_PER_ROW * row_count + _BILLED_PER_COLUMN * column_count
    return float(max(_BILLED_FLOOR, linear))
