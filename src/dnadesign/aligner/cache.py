"""
--------------------------------------------------------------------------------
<dnadesign project>
aligner/cache.py

Cache module for storing and retrieving computed alignment scores.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

def generate_cache_filename(n: int, normalize: bool, gap_open: int, gap_extend: int, 
                            matrix_id: str = "nt", return_formats: tuple = ("mean", "condensed")) -> str:
    """
    Generate a human-readable cache filename.
    
    Example: swcache_n50_normTrue_go11_ge1_matrixnt_meancondensed_2025-04-15.pkl
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    rf = "_".join(return_formats)
    filename = f"swcache_n{n}_norm{normalize}_go{gap_open}_ge{gap_extend}_matrix{matrix_id}_{rf}_{date_str}.pkl"
    return filename

def save_cache(cache_dir: Path, filename: str, data: Any) -> None:
    """
    Save cache data to disk.
    """
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / filename
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

def load_cache(cache_dir: Path, filename: str) -> Any:
    """
    Load cache data from disk.
    """
    cache_file = cache_dir / filename
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None