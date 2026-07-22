"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/views/pca_policy.py

PCA policy helpers shared by reduction and memory-preflight paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

_DENSE_PCA_FIT_THRESHOLD_BYTES = 256 * 1024**2
_STREAMING_TARGET_BATCH_BYTES = 128 * 1024**2
_STREAMING_MIN_BATCH_ROWS = 512


def select_pca_method(*, rows: int, dims: int, itemsize: int) -> str:
    fit_bytes = max(rows, 0) * max(dims, 0) * max(itemsize, 1)
    if fit_bytes <= _DENSE_PCA_FIT_THRESHOLD_BYTES:
        return "dense_svd"
    return "randomized_svd"


def streaming_batch_rows(*, total_rows: int, dims: int, itemsize: int, output_dims: int) -> int:
    row_bytes = max(dims, 1) * max(itemsize, 1)
    target_rows = max(_STREAMING_TARGET_BATCH_BYTES // max(row_bytes, 1), output_dims + 1)
    target_rows = max(target_rows, _STREAMING_MIN_BATCH_ROWS)
    return max(min(int(target_rows), max(total_rows, 1)), output_dims + 1)
