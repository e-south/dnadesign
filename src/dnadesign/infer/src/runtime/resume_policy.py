"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/resume_policy.py

Resolves resume parquet filter chunk-size policy from explicit environment contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os

_RESUME_FILTER_CHUNK_ENV = "DNADESIGN_INFER_RESUME_FILTER_CHUNK"
_RESUME_FILTER_CHUNK_DEFAULT = 10_000
_RESUME_FILTER_CHUNK_MIN = 1
_RESUME_FILTER_CHUNK_MAX = 100_000


def resolve_resume_filter_chunk_size() -> int:
    raw_chunk_size = os.environ.get(
        _RESUME_FILTER_CHUNK_ENV,
        str(_RESUME_FILTER_CHUNK_DEFAULT),
    )
    try:
        chunk_size = int(raw_chunk_size)
    except ValueError as exc:
        raise ValueError(
            f"{_RESUME_FILTER_CHUNK_ENV} must be an integer, got {raw_chunk_size!r}",
        ) from exc

    if chunk_size < _RESUME_FILTER_CHUNK_MIN:
        raise ValueError(
            f"{_RESUME_FILTER_CHUNK_ENV} must be >= {_RESUME_FILTER_CHUNK_MIN}, got {chunk_size}",
        )
    if chunk_size > _RESUME_FILTER_CHUNK_MAX:
        raise ValueError(
            f"{_RESUME_FILTER_CHUNK_ENV} must be <= {_RESUME_FILTER_CHUNK_MAX}, got {chunk_size}",
        )
    return chunk_size
