"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/transforms_y/sfxi_vec8_from_tidy_v2.py

Builds y = [v00, v10, v01, v11, y00*, y10*, y01*, y11*] where:
  • v.. ∈ [0,1] is the per-design logic vector (order: 00,10,01,11),
    computed from log2 signal then per-design min-max normalized.
    Logic signal = log2( (YFP+eps)/(CFP+eps) ) if CFP is present for all four states.
  • y..* = log2(YFP + eps) (replicate-mean per state).

Input (tidy CSV):
  design_id,
  experiment_id,
  hours_post_induction,
  replicate,
  logic_state ("00"|"10"|"01"|"11"),
  reporter_channel ("yfp" [required], "cfp" [optional]),
  fluorescence_raw

Params (campaign.yaml → transforms_y.params):
  enforce_single_timepoint: bool = True
  replicate_aggregation: "mean"           # (fixed)
  logic_signal:
    source: "ratio_if_available"|"yfp_only" = "ratio_if_available"
    eps_division: float = 1e-8
    apply_log2: bool = True               # (fixed true for v)
  logic_normalization:
    method: "minmax_per_design"           # (fixed)
    minmax_epsilon: float = 1e-6
    equal_states_fallback: "uniform_quarters_and_warn"|"uniform_quarters"|"error"
  intensity_tail:
    base_channel: "yfp"                    # (fixed)
    eps_log2: float = 1e-8

Returns:
  labels_df: DataFrame with columns ["id","y"] (y is length-8 list[float])
  preview:   TransformPreview-like object (counts, warnings, sample)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..registries.transforms_y import register_transform_y
from ..utils import OpalError

# Try to use the project's preview type; fall back to a lightweight namespace.
try:
    from ..registries.transforms_y import TransformPreview  # type: ignore

    _HAS_PREVIEW_CLASS = True
except Exception:  # pragma: no cover
    from types import SimpleNamespace as TransformPreview  # type: ignore

    _HAS_PREVIEW_CLASS = False


_STATES: Tuple[str, str, str, str] = ("00", "10", "01", "11")


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise OpalError(f"Missing required columns in tidy input: {missing}")


def _canon_logic_state(s: str) -> str:
    s2 = str(s).strip()
    if s2 not in _STATES:
        raise OpalError(f"Invalid logic_state {s!r}; expected one of {_STATES}")
    return s2


def _canon_channel(ch: str) -> str:
    return str(ch).strip().lower()


def _build_preview(kwargs: Dict):
    """
    Instantiate the project's TransformPreview if available; otherwise return a
    SimpleNamespace with the same fields so CLI printing (.__dict__) still works.
    """
    if _HAS_PREVIEW_CLASS:
        # Filter to fields accepted by the dataclass constructor
        try:
            import inspect

            params = set(inspect.signature(TransformPreview).parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in params}
            return TransformPreview(**filtered)  # type: ignore
        except Exception:
            pass  # fall through to SimpleNamespace
    from types import SimpleNamespace

    return SimpleNamespace(**kwargs)  # type: ignore


@register_transform_y("sfxi_vec8_from_tidy_v2")
def sfxi_vec8_from_tidy_v2(
    tidy: pd.DataFrame,
    params: Dict,
    **_: Dict,
):
    """
    Main transform entrypoint. See module docstring for details.
    """
    # -------- 1) Preflight & canonicalize --------
    required = {
        "design_id",
        "experiment_id",
        "hours_post_induction",
        "replicate",
        "logic_state",
        "reporter_channel",
        "fluorescence_raw",
    }
    _require_columns(tidy, required)

    # Canonicalize minimal fields
    df = tidy.copy()
    df["logic_state"] = df["logic_state"].map(_canon_logic_state)
    df["reporter_channel"] = df["reporter_channel"].map(_canon_channel)

    # Enforce single timepoint if requested
    if bool(params.get("enforce_single_timepoint", True)):
        hpi_vals = pd.to_numeric(df["hours_post_induction"], errors="coerce").dropna()
        n_hpi = int(hpi_vals.nunique())
        if n_hpi != 1:
            raise OpalError(
                f"Expected exactly one hours_post_induction; found {n_hpi} in tidy file."
            )

    # Replicate aggregation policy (fixed: mean)
    rep_policy = str(params.get("replicate_aggregation", "mean")).lower()
    if rep_policy != "mean":
        raise OpalError("replicate_aggregation supports only 'mean' in v2.")

    # Logic signal params
    logic_cfg = params.get("logic_signal", {}) or {}
    source = str(logic_cfg.get("source", "ratio_if_available")).lower()
    eps_div = float(logic_cfg.get("eps_division", 1e-8))
    # apply_log2 is fixed True for v; ignore if provided.

    # Normalization params for v
    norm_cfg = params.get("logic_normalization", {}) or {}
    minmax_eps = float(norm_cfg.get("minmax_epsilon", 1e-6))
    fallback = str(
        norm_cfg.get("equal_states_fallback", "uniform_quarters_and_warn")
    ).lower()

    # Intensity tail params (y* = log2(YFP + eps))
    tail_cfg = params.get("intensity_tail", {}) or {}
    eps_y = float(tail_cfg.get("eps_log2", 1e-8))

    # -------- 2) Aggregate replicates to state×channel means --------
    # groupby mean(fluorescence_raw) across replicates
    g = (
        df.groupby(["design_id", "logic_state", "reporter_channel"], dropna=False)[
            "fluorescence_raw"
        ]
        .mean()
        .reset_index()
    )
    pivot = g.pivot_table(
        index=["design_id", "logic_state"],
        columns="reporter_channel",
        values="fluorescence_raw",
        aggfunc="mean",
    )

    # -------- 3) Per-design assembly of v and y* --------
    labels: List[Dict] = []
    warnings: List[str] = []

    # Pre-compute whether CFP is fully present per design (all states)
    def _has_full_cfp(design_id: str) -> bool:
        try:
            sub = pivot.loc[design_id]
            if "cfp" not in sub.columns:
                return False
            st = set(s for s, _ in sub.index.tolist())
            return set(_STATES).issubset(st) and sub["cfp"].notna().all()
        except Exception:
            return False

    # Iterate designs
    for did, sub in pivot.groupby(level=0):
        # Ensure all four states present (after replicate aggregation)
        present_states = set(s for s, _ in sub.index.tolist())
        if not set(_STATES).issubset(present_states):
            missing = sorted(set(_STATES) - present_states)
            raise OpalError(f"{did}: missing logic_state(s) {missing}")

        # Pull YFP per state (mean)
        def _state_val(state: str, column: str) -> float:
            try:
                return float(sub.loc[(state,), column])
            except Exception:
                return float("nan")

        yfp = np.asarray([_state_val(s, "yfp") for s in _STATES], dtype=float)
        if np.any(~np.isfinite(yfp)):
            raise OpalError(f"{did}: non-finite YFP values after aggregation")

        # Logic signal
        use_ratio = (source == "ratio_if_available") and _has_full_cfp(did)
        if use_ratio:
            cfp = np.asarray([_state_val(s, "cfp") for s in _STATES], dtype=float)
            if np.any(~np.isfinite(cfp)):
                raise OpalError(f"{did}: CFP present but contains non-finite values")
            logic_sig = np.log2((yfp + eps_div) / (cfp + eps_div))
        else:
            logic_sig = np.log2(yfp + eps_y)

        # Per-design min–max normalization to [0,1]
        a = float(np.min(logic_sig))
        b = float(np.max(logic_sig))
        if (b - a) <= float(minmax_eps):
            if fallback == "error":
                raise OpalError(
                    f"{did}: near-constant logic signal (max-min ≤ {minmax_eps})"
                )
            v = [0.25, 0.25, 0.25, 0.25]
            if fallback == "uniform_quarters_and_warn":
                warnings.append(f"{did}: equal-states fallback used")
        else:
            v = [float((x - a) / (b - a)) for x in logic_sig]

        # Intensity tail in log2 units
        ystar = list(np.log2(yfp + eps_y))

        labels.append({"id": str(did), "y": [*v, *ystar]})

    labels_df = pd.DataFrame(labels, columns=["id", "y"])

    # -------- 4) Build preview (counts + sample) --------
    preview = _build_preview(
        n_input_rows=int(len(df)),
        n_unique_designs=int(labels_df.shape[0]),
        n_missing_states=0,
        replicate_aggregation="mean",
        logic_signal_source=source,
        warnings=warnings[:50],
        sample=labels_df.head(8).to_dict(orient="records"),
    )

    return labels_df, preview
