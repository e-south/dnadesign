"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/candidates.py

Top-candidate table construction for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..core.contracts import PolicySpec, SfxiEvidenceFrame
from .selection import select_top_rows

CANDIDATE_COLUMNS = [
    "policy_id",
    "selection_view_id",
    "rank",
    "id",
    "sequence",
    "score",
    "logic_fidelity",
    "effect_raw",
    "effect_scaled",
    "off_state_logic_level",
    "v00",
    "v10",
    "v01",
    "v11",
]


def build_top_candidate_table(
    policies: Iterable[PolicySpec],
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    scored: dict[str, dict[str, pd.DataFrame]],
    *,
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in policies:
        for evidence in sfxi_evidence:
            top = select_top_rows(scored[policy.id][evidence.target_view.id], top_k=top_k)
            for _, row in top.iterrows():
                rows.append(
                    {
                        "policy_id": policy.id,
                        "selection_view_id": evidence.target_view.id,
                        "rank": int(row["rank"]),
                        "id": str(row["id"]),
                        "sequence": str(row["sequence"]),
                        "score": float(row["score"]),
                        "logic_fidelity": float(row["logic_fidelity"]),
                        "effect_raw": float(row["effect_raw"]),
                        "effect_scaled": float(row["effect_scaled"]),
                        "off_state_logic_level": float(row["off_state_logic_level"]),
                        "v00": float(row["v00"]),
                        "v10": float(row["v10"]),
                        "v01": float(row["v01"]),
                        "v11": float(row["v11"]),
                    }
                )
    table = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    if table.empty:
        table["selection_view_count"] = pd.Series(dtype="int64")
        return table
    counts = (
        table.groupby(["policy_id", "id"])["selection_view_id"].nunique().rename("selection_view_count").reset_index()
    )
    return table.merge(counts, on=["policy_id", "id"], how="left")
