"""CCMM masking strategies for Nori training.

Three mask patterns (randomly selected per episode):
1. Cell-wise: Each cell independently masked with probability = mask_ratio
2. Column-wise: Select mask_ratio fraction of columns, mask ALL cells in those columns
3. Block masking: Select contiguous rectangular block(s) covering ~mask_ratio fraction
"""

from __future__ import annotations

import numpy as np


def create_masks(n_query, n_features, mask_ratio, mask_type, rng=None):
    """Create feature masks for query rows.

    Args:
        n_query: number of query rows
        n_features: number of features
        mask_ratio: fraction of cells to mask, in [0.1, 0.4]
        mask_type: 'cell', 'column', or 'block'
        rng: numpy random generator

    Returns:
        mask: np.ndarray [n_query, n_features] bool, True where masked
    """
    if rng is None:
        rng = np.random.default_rng()

    if mask_type == "cell":
        return _cell_mask(n_query, n_features, mask_ratio, rng)
    elif mask_type == "column":
        return _column_mask(n_query, n_features, mask_ratio, rng)
    elif mask_type == "block":
        return _block_mask(n_query, n_features, mask_ratio, rng)
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")


def _cell_mask(n_query, n_features, mask_ratio, rng):
    """Each cell independently masked with probability = mask_ratio."""
    return rng.random((n_query, n_features)) < mask_ratio


def _column_mask(n_query, n_features, mask_ratio, rng):
    """Select mask_ratio fraction of columns, mask all cells in those columns."""
    n_cols_to_mask = max(1, int(round(n_features * mask_ratio)))
    cols = rng.choice(n_features, size=n_cols_to_mask, replace=False)
    mask = np.zeros((n_query, n_features), dtype=bool)
    mask[:, cols] = True
    return mask


def _block_mask(n_query, n_features, mask_ratio, rng):
    """Select contiguous rectangular block(s) covering ~mask_ratio fraction."""
    total_cells = n_query * n_features
    target_cells = int(round(total_cells * mask_ratio))

    mask = np.zeros((n_query, n_features), dtype=bool)
    cells_masked = 0

    max_attempts = 10
    for _ in range(max_attempts):
        if cells_masked >= target_cells:
            break
        remaining = target_cells - cells_masked

        # Random block dimensions
        block_rows = rng.integers(1, n_query + 1)
        block_cols = max(1, min(remaining // block_rows, n_features))
        if block_cols == 0:
            block_cols = 1

        # Random position
        row_start = rng.integers(0, max(1, n_query - block_rows + 1))
        col_start = rng.integers(0, max(1, n_features - block_cols + 1))

        mask[row_start : row_start + block_rows, col_start : col_start + block_cols] = True
        cells_masked = mask.sum()

    return mask


def random_mask_type(rng=None):
    """Randomly select a mask type."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(["cell", "column", "block"])


def random_mask_ratio(min_ratio=0.1, max_ratio=0.4, rng=None):
    """Randomly sample a mask ratio."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(min_ratio, max_ratio)
