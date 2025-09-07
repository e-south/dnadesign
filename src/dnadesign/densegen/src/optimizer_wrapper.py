"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/optimizer_wrapper.py

dense-arrays Optimizer wrapper and helpers.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import math
import random

import dense_arrays as da

log = logging.getLogger(__name__)


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
) -> tuple[str, dict]:
    """
    Best-effort random filler with GC control.

    Returns
    -------
    seq : str
    info : dict
        {
          "attempts": int,
          "gc_actual": float,
          "relaxed": bool,
          "final_gc_min": float,
          "final_gc_max": float
        }

    Behavior
    --------
    - If the integer GC window is infeasible for this length (very short pads),
      we relax to [0, 1] (i.e., [0..length] counts), mark relaxed=True.
    - Otherwise we sample up to `max_tries`. If we cannot hit the window, we relax
      to [0,1] and return the next draw, marking relaxed=True.
    - No warnings are logged here; the caller can record metadata.
    """
    nucleotides = "ATGC"

    if length <= 0:
        return "", {
            "attempts": 0,
            "gc_actual": 0.0,
            "relaxed": False,
            "final_gc_min": gc_min,
            "final_gc_max": gc_max,
        }

    # Convert fraction window → integer GC count window
    lo = math.ceil(length * gc_min)
    hi = math.floor(length * gc_max)

    relaxed = False
    final_min = gc_min
    final_max = gc_max

    # If infeasible window (common for very small lengths), relax immediately.
    if lo > hi:
        relaxed = True
        lo, hi = 0, length
        final_min, final_max = 0.0, 1.0

    # Try within the (possibly relaxed) window
    attempts = 0
    for attempts in range(1, max_tries + 1):
        seq = "".join(random.choices(nucleotides, k=length))
        gc_count = seq.count("G") + seq.count("C")
        if lo <= gc_count <= hi:
            return seq, {
                "attempts": attempts,
                "gc_actual": _gc_fraction(seq),
                "relaxed": relaxed,
                "final_gc_min": final_min,
                "final_gc_max": final_max,
            }

    # As a last resort, fully relax and return next draw.
    if not relaxed:
        relaxed = True
        final_min, final_max = 0.0, 1.0
    seq = "".join(random.choices(nucleotides, k=length))
    return seq, {
        "attempts": attempts,
        "gc_actual": _gc_fraction(seq),
        "relaxed": relaxed,
        "final_gc_min": final_min,
        "final_gc_max": final_max,
    }


class DenseArrayOptimizer:
    def __init__(
        self,
        library: list,
        sequence_length: int,
        solver: str = "CBC",
        solver_options: list | None = None,
        fixed_elements: dict | None = None,
    ):
        valid = {"A", "T", "G", "C"}
        filtered = []
        for motif in library:
            if isinstance(motif, str):
                s = motif.strip().upper()
                if s and s != "NONE" and set(s).issubset(valid):
                    filtered.append(s)
        if not filtered:
            raise ValueError("After filtering, the motif library is empty or invalid.")

        self.library = filtered
        self.sequence_length = sequence_length
        self.solver = solver
        self.solver_options = solver_options or []
        self.fixed_elements = (fixed_elements or {}).copy()

    @staticmethod
    def _convert_none(v):
        return None if (isinstance(v, str) and v.strip().lower() == "none") else v

    def get_optimizer_instance(self) -> da.Optimizer:
        lib = self.library.copy()
        converted = []
        cons = (self.fixed_elements or {}).get("promoter_constraints")
        if cons:
            # Accept a single dict or a list with single dict
            if isinstance(cons, list) and cons and isinstance(cons[0], dict):
                cons = cons[0]
            if isinstance(cons, dict):
                pc = {k: self._convert_none(v) for k, v in cons.items() if k != "name"}
                # Ensure motifs in library
                for mkey in ("upstream", "downstream"):
                    mv = pc.get(mkey)
                    if mv and mv not in lib:
                        lib.append(mv)
                # Normalize list→tuple for *_pos and spacer_length
                norm = {}
                for k, v in pc.items():
                    if v is None:
                        continue
                    if k in {
                        "upstream_pos",
                        "downstream_pos",
                        "spacer_length",
                    } and isinstance(v, list):
                        norm[k] = tuple(v)
                    else:
                        norm[k] = v
                converted.append(norm)

        opt = da.Optimizer(library=lib, sequence_length=self.sequence_length)
        for c in converted:
            opt.add_promoter_constraints(**c)

        sb = (self.fixed_elements or {}).get("side_biases") or {}
        left, right = sb.get("left"), sb.get("right")
        if (left and any(left)) or (right and any(right)):
            opt.add_side_biases(left=left, right=right)
        return opt
