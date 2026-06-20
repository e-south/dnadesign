"""Cache helpers for pairwise alignment score batches."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


def generate_cache_filename(
    n: int,
    normalize: bool,
    gap_open: int,
    gap_extend: int,
    matrix_id: str = "nt",
    return_formats: tuple[str, ...] = ("mean", "condensed"),
) -> str:
    """Generate a human-readable cache filename."""

    date_str = datetime.now().strftime("%Y-%m-%d")
    rf = "_".join(return_formats)
    return f"swcache_n{n}_norm{normalize}_go{gap_open}_ge{gap_extend}_matrix{matrix_id}_{rf}_{date_str}.pkl"


def save_cache(cache_dir: Path, filename: str, data: Any) -> None:
    """Save cache data to disk."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / filename).open("wb") as handle:
        pickle.dump(data, handle)


def load_cache(cache_dir: Path, filename: str) -> Any:
    """Load cache data from disk, returning ``None`` when absent."""

    cache_file = cache_dir / filename
    if not cache_file.exists():
        return None
    with cache_file.open("rb") as handle:
        return pickle.load(handle)
