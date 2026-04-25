"""
Matrix IO helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)


def read_matrix(path: Path, *, mmap_mode: str | None = "r") -> np.ndarray:
    loaded = np.load(path, mmap_mode=mmap_mode)
    if isinstance(loaded, np.ndarray):
        return loaded
    if "matrix" in loaded.files:
        return loaded["matrix"]
    if len(loaded.files) == 1:
        return loaded[loaded.files[0]]
    raise ValueError(f"matrix archive must expose exactly one array or a 'matrix' array: {path}")
