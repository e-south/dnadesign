"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/_param_utils.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def _first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def get_str(
    d: Mapping[str, Any], keys: Sequence[str], default: Optional[str] = None
) -> Optional[str]:
    """
    Return the first present key coerced to str; otherwise `default`.
    (Previous behavior incorrectly ignored `default`.)
    """
    v = _first_present(d, keys)
    return str(v) if v is not None else default


def get_float(d: Mapping[str, Any], keys: Sequence[str], default: float) -> float:
    v = _first_present(d, keys)
    return float(v) if v is not None else float(default)


def get_int(d: Mapping[str, Any], keys: Sequence[str], default: int) -> int:
    v = _first_present(d, keys)
    return int(v) if v is not None else int(default)


def get_bool(d: Mapping[str, Any], keys: Sequence[str], default: bool) -> bool:
    v = _first_present(d, keys)
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean for {keys}, got {v!r}")


# -------------------------------
# Metric aliasing / field helpers
# -------------------------------


def normalize_metric_field(field: Optional[str]) -> Optional[str]:
    """
    Normalize human-friendly metric names to canonical ledger column names.
    Accepts dotted or double-underscore forms; keeps explicit obj__/pred__/sel__.
    """
    if field is None:
        return None
    s = str(field).strip()
    if not s:
        return None
    s = s.replace(".", "__")  # allow "obj.effect_scaled" style too
    alias = {
        # objective scalar / score
        "score": "pred__y_obj_scalar",
        "scalar": "pred__y_obj_scalar",
        "objective_scalar": "pred__y_obj_scalar",
        # rank
        "rank": "sel__rank_competition",
        "rank_competition": "sel__rank_competition",
        # per-row objective diagnostics
        "logic_fidelity": "obj__logic_fidelity",
        "effect_raw": "obj__effect_raw",
        "effect_scaled": "obj__effect_scaled",
        "clip_lo": "obj__clip_lo_mask",
        "clip_hi": "obj__clip_hi_mask",
    }
    return alias.get(s, s)


def event_columns_for(*fields: Optional[str]) -> set[str]:
    """
    Return the subset of fields that are columns we must request from
    ledger.predictions (i.e., with obj__/pred__/sel__ prefixes).
    """
    cols: set[str] = set()
    for f in fields:
        if not f:
            continue
        ff = str(f)
        if ff.startswith(("obj__", "pred__", "sel__")):
            cols.add(ff)
    return cols
