"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/recompute.py

Canonical SFXI ledger recomputation checks for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..core.contracts import SfxiEvidenceFrame


def validate_canonical_sfxi_recompute(
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    canonical_scored: dict[str, pd.DataFrame],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    max_abs = 0.0
    for evidence in sfxi_evidence:
        recomputed = canonical_scored[evidence.target_view.id].sort_values("id").reset_index(drop=True)
        ledger = evidence.predictions[["id", "pred__score_selected"]].sort_values("id").reset_index(drop=True)
        diff = np.abs(recomputed["score"].to_numpy(dtype=float) - ledger["pred__score_selected"].to_numpy(dtype=float))
        if not np.all(np.isfinite(diff)):
            raise ValueError(
                f"{evidence.target_view.id}: canonical recomputation produced non-finite score differences."
            )
        value = float(np.max(diff))
        max_abs = max(max_abs, value)
        rows.append({"selection_view_id": evidence.target_view.id, "max_abs_error": value})
    return {
        "max_abs_error": max_abs,
        "per_selection_view": rows,
        "matches_canonical_ledger": bool(max_abs <= 1e-12),
    }


def assert_canonical_sfxi_recompute(validation: Mapping[str, object]) -> None:
    if bool(validation.get("matches_canonical_ledger")):
        return
    raise RuntimeError(
        "canonical SFXI recomputation mismatch: "
        f"max_abs_error={validation.get('max_abs_error')}; "
        f"per_selection_view={validation.get('per_selection_view')}"
    )
