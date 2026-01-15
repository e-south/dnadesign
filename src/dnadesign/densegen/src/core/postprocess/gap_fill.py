"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/postprocess/gap_fill.py

Gap fill policy implementation.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import random


def _gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)


def random_fill(
    length: int,
    gc_min: float = 0.40,
    gc_max: float = 0.60,
    *,
    max_tries: int = 2000,
    mode: str = "strict",
    rng: random.Random | None = None,
) -> tuple[str, dict]:
    """
    Random filler with strict/adaptive GC control.

    Returns
    -------
    seq : str
    info : dict
        {
          "attempts": int,
          "gc_actual": float,
          "relaxed": bool,
          "final_gc_min": float,
          "final_gc_max": float,
          "target_gc_min": float,
          "target_gc_max": float,
        }

    Behavior
    --------
    - strict: infeasible windows or inability to hit GC target raise ValueError.
    - adaptive: infeasible windows or misses after max_tries relax to [0, 1],
      and are recorded with relaxed=True.
    """
    nucleotides = "ATGC"
    rng = rng or random

    if length <= 0:
        return "", {
            "attempts": 0,
            "gc_actual": 0.0,
            "relaxed": False,
            "final_gc_min": gc_min,
            "final_gc_max": gc_max,
            "target_gc_min": gc_min,
            "target_gc_max": gc_max,
        }

    # Convert fraction window -> integer GC count window
    lo = math.ceil(length * gc_min)
    hi = math.floor(length * gc_max)

    relaxed = False
    final_min = gc_min
    final_max = gc_max

    # If infeasible window (common for very small lengths), handle per policy.
    if lo > hi:
        if mode == "strict":
            raise ValueError(f"GC target infeasible for gap length {length} (min={gc_min}, max={gc_max}).")
        relaxed = True
        lo, hi = 0, length
        final_min, final_max = 0.0, 1.0

    # Try within the (possibly relaxed) window
    attempts = 0
    for attempts in range(1, max_tries + 1):
        seq = "".join(rng.choices(nucleotides, k=length))
        gc_count = seq.count("G") + seq.count("C")
        if lo <= gc_count <= hi:
            return seq, {
                "attempts": attempts,
                "gc_actual": _gc_fraction(seq),
                "relaxed": relaxed,
                "final_gc_min": final_min,
                "final_gc_max": final_max,
                "target_gc_min": gc_min,
                "target_gc_max": gc_max,
            }

    if mode == "strict":
        raise ValueError(f"Failed to hit GC target after {max_tries} tries (len={length}, min={gc_min}, max={gc_max}).")

    # Adaptive: relax and return next draw.
    if not relaxed:
        relaxed = True
        final_min, final_max = 0.0, 1.0
    seq = "".join(rng.choices(nucleotides, k=length))
    return seq, {
        "attempts": attempts,
        "gc_actual": _gc_fraction(seq),
        "relaxed": relaxed,
        "final_gc_min": final_min,
        "final_gc_max": final_max,
        "target_gc_min": gc_min,
        "target_gc_max": gc_max,
    }
