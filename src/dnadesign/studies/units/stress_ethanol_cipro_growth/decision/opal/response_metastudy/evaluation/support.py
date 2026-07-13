"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/support.py

Candidate response-shape support summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.contracts import SfxiEvidenceFrame


def build_setpoint_support(
    sfxi_evidence: Sequence[SfxiEvidenceFrame],
    canonical_scores: Mapping[str, pd.DataFrame],
    *,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    if not thresholds:
        raise ValueError("setpoint-support thresholds must not be empty.")
    checked = np.asarray(thresholds, dtype=float)
    if not np.all(np.isfinite(checked)) or np.any(checked < 0.0) or np.any(checked > 1.0):
        raise ValueError("setpoint-support thresholds must be finite and in [0, 1].")
    rows: list[dict[str, object]] = []
    for evidence_frame in sfxi_evidence:
        label = evidence_frame.target_view.id
        if label not in canonical_scores:
            raise ValueError(f"missing canonical SFXI score surface for target view {label!r}.")
        logic = canonical_scores[label]["logic_fidelity"].astype(float).to_numpy()
        if logic.size == 0 or not np.all(np.isfinite(logic)):
            raise ValueError(f"target view {label!r} has an empty or non-finite logic-fidelity surface.")
        for threshold in checked:
            count = int(np.count_nonzero(logic >= threshold))
            rows.append(
                {
                    "selection_view_id": label,
                    "logic_threshold": float(threshold),
                    "candidate_count": count,
                    "candidate_fraction": float(count / logic.size),
                    "max_logic_fidelity": float(np.max(logic)),
                    "p99_logic_fidelity": float(np.percentile(logic, 99)),
                    "median_logic_fidelity": float(np.median(logic)),
                    "candidate_total": int(logic.size),
                }
            )
    return pd.DataFrame(rows)
