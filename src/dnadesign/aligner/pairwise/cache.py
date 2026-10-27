"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/pairwise/cache.py

Cache helpers for pairwise alignment score batches.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_CACHE_VERSION = 1
_SCORE_KEYS = frozenset({"mean", "matrix", "condensed", "dissimilarity"})


def generate_cache_filename(
    sequences: Sequence[str],
    normalize: bool,
    match: int,
    mismatch: int,
    gap_open: int,
    gap_extend: int,
    matrix_id: str = "nt",
    return_formats: tuple[str, ...] = ("mean", "condensed"),
    return_raw: bool = False,
    return_dissimilarity: bool = False,
) -> str:
    """Identify the scoring inputs and cache schema independently of wall time."""
    digest = _cache_digest(
        sequences=sequences,
        normalize=normalize,
        match=match,
        mismatch=mismatch,
        gap_open=gap_open,
        gap_extend=gap_extend,
        matrix_id=matrix_id,
        return_formats=return_formats,
        return_raw=return_raw,
        return_dissimilarity=return_dissimilarity,
    )
    return f"swcache-v{_CACHE_VERSION}-{digest}.npz"


def save_cache(cache_dir: Path, filename: str, data: Any) -> None:
    """Atomically cache numeric score values without executable serialization."""

    if not isinstance(data, dict):
        kind, values = "scalar", {"value": data}
    elif "normalized" in data:
        kind = "raw_scores"
        values = {f"normalized_{key}": value for key, value in data["normalized"].items()}
        values["raw"] = data["raw"]
    else:
        kind, values = "scores", data
    arrays = {key: np.asarray(value) for key, value in values.items()}
    _validate_arrays(kind, arrays)
    arrays["kind"] = np.asarray(kind)
    arrays["version"] = np.asarray(_CACHE_VERSION)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".alignment-", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, cache_dir / filename)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_cache(cache_dir: Path, filename: str) -> Any:
    """Read numeric score values, returning ``None`` only when no cache exists."""

    cache_file = cache_dir / filename
    if not cache_file.exists():
        return None
    try:
        with np.load(cache_file, allow_pickle=False) as archive:
            if archive["version"].item() != _CACHE_VERSION:
                raise ValueError("unsupported cache version")
            kind = archive["kind"].item()
            arrays = {key: archive[key] for key in archive.files if key not in {"kind", "version"}}
        _validate_arrays(kind, arrays)
        values = {key: value.item() if value.ndim == 0 else value for key, value in arrays.items()}
        if kind == "scalar":
            return values["value"]
        if kind == "raw_scores":
            return {
                "raw": values["raw"],
                "normalized": {key.removeprefix("normalized_"): value for key, value in values.items() if key != "raw"},
            }
        return values
    except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError) as exc:
        raise ValueError(f"Invalid alignment cache: {cache_file}: {exc}") from exc


def _validate_arrays(kind: str, arrays: dict[str, np.ndarray]) -> None:
    keys = set(arrays)
    if kind == "scalar":
        valid = keys == {"value"}
    elif kind == "scores":
        valid = bool(keys) and keys <= _SCORE_KEYS
    elif kind == "raw_scores":
        valid = "raw" in keys and len(keys) > 1 and keys - {"raw"} <= {f"normalized_{key}" for key in _SCORE_KEYS}
    else:
        valid = False
    if not valid:
        raise ValueError("invalid score fields or cache kind")
    for key, value in arrays.items():
        if value.dtype.kind not in "bifu":
            raise ValueError(f"cache field {key!r} must be numeric")
        label = key.removeprefix("normalized_")
        expected_ndim = 2 if label in {"raw", "matrix"} else 1 if label == "condensed" else 0
        if value.ndim != expected_ndim:
            raise ValueError(f"cache field {key!r} must have {expected_ndim} dimensions")


def _cache_digest(
    *,
    sequences: Sequence[str],
    normalize: bool,
    match: int,
    mismatch: int,
    gap_open: int,
    gap_extend: int,
    matrix_id: str,
    return_formats: tuple[str, ...],
    return_raw: bool,
    return_dissimilarity: bool,
) -> str:
    payload = "\n".join(
        [
            f"cache_version={_CACHE_VERSION}",
            f"matrix_id={matrix_id}",
            f"normalize={normalize}",
            f"match={match}",
            f"mismatch={mismatch}",
            f"gap_open={gap_open}",
            f"gap_extend={gap_extend}",
            f"return_formats={','.join(return_formats)}",
            f"return_raw={return_raw}",
            f"return_dissimilarity={return_dissimilarity}",
            "sequences:",
            *sequences,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
