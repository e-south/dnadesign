"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/postprocess/gap_fill.py

Pad policy implementation (fills remaining length budget).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import random

from ...utils.sequence_utils import gc_fraction


def generate_pad(
    length: int,
    *,
    mode: str = "adaptive",
    gc_mode: str = "range",
    gc_min: float = 0.40,
    gc_max: float = 0.60,
    gc_target: float = 0.50,
    gc_tolerance: float = 0.10,
    gc_min_pad_length: int = 0,
    max_tries: int = 2000,
    rng: random.Random | None = None,
) -> tuple[str, dict]:
    """
    Generate pad bases to meet a length budget with optional GC constraints.

    Returns
    -------
    seq : str
    info : dict
        {
          "attempts": int,
          "gc_actual": float,
          "relaxed": bool,
          "relaxed_reason": str | None,
          "final_gc_min": float | None,
          "final_gc_max": float | None,
          "target_gc_min": float | None,
          "target_gc_max": float | None,
          "gc_mode": str,
        }
    """
    rng = rng or random

    if length <= 0:
        return "", {
            "attempts": 0,
            "gc_actual": 0.0,
            "relaxed": False,
            "relaxed_reason": None,
            "final_gc_min": None,
            "final_gc_max": None,
            "target_gc_min": None,
            "target_gc_max": None,
            "gc_mode": gc_mode,
        }

    if gc_mode == "off":
        bases = [rng.choice("ACGT") for _ in range(length)]
        seq = "".join(bases)
        return seq, {
            "attempts": 1,
            "gc_actual": gc_fraction(seq),
            "relaxed": False,
            "relaxed_reason": None,
            "final_gc_min": None,
            "final_gc_max": None,
            "target_gc_min": None,
            "target_gc_max": None,
            "gc_mode": gc_mode,
        }

    if gc_mode not in {"range", "target"}:
        raise ValueError(f"Unsupported gc_mode: {gc_mode!r}")

    if gc_mode == "target":
        target_min = gc_target - gc_tolerance
        target_max = gc_target + gc_tolerance
        if target_min < 0.0 or target_max > 1.0:
            raise ValueError("gc_target +/- gc_tolerance must stay within [0, 1]")
    else:
        target_min = gc_min
        target_max = gc_max

    relaxed = False
    relaxed_reason = None
    final_min = target_min
    final_max = target_max

    if length < gc_min_pad_length:
        if mode == "strict":
            raise ValueError(f"Pad length {length} is shorter than gc.min_pad_length={gc_min_pad_length}.")
        relaxed = True
        relaxed_reason = "short_pad"
        final_min, final_max = 0.0, 1.0

    lo = math.ceil(length * final_min)
    hi = math.floor(length * final_max)

    if lo > hi:
        if mode == "strict":
            raise ValueError(f"GC target infeasible for pad length {length} (min={target_min}, max={target_max}).")
        relaxed = True
        if relaxed_reason is None:
            relaxed_reason = "infeasible_gc_window"
        final_min, final_max = 0.0, 1.0
        lo, hi = 0, length

    gc_count = rng.randint(lo, hi) if lo <= hi else 0
    bases = [rng.choice("GC") for _ in range(gc_count)] + [rng.choice("AT") for _ in range(length - gc_count)]
    rng.shuffle(bases)
    seq = "".join(bases)
    return seq, {
        "attempts": 1,
        "gc_actual": gc_fraction(seq),
        "relaxed": relaxed,
        "relaxed_reason": relaxed_reason,
        "final_gc_min": final_min,
        "final_gc_max": final_max,
        "target_gc_min": target_min,
        "target_gc_max": target_max,
        "gc_mode": gc_mode,
    }
