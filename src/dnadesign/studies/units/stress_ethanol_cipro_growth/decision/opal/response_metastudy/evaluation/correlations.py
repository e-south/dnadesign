"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/correlations.py

Correlation diagnostics for the response metric metastudy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..core.contracts import PolicySpec, SfxiEvidenceFrame


def build_pairwise_correlations(
    policies: Iterable[PolicySpec],
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    scored: dict[str, dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in policies:
        for _, row in target_view_score_correlations(sfxi_evidence, scored[policy.id]).iterrows():
            rows.append({"policy_id": policy.id, **row.to_dict()})
        for evidence in sfxi_evidence:
            frame = scored[policy.id][evidence.target_view.id]
            rows.append(
                {
                    "policy_id": policy.id,
                    "metric": "within_selection_view",
                    "selection_view_a": evidence.target_view.id,
                    "selection_view_b": "logic_fidelity",
                    "pearson": _safe_corr(frame["score"], frame["logic_fidelity"], method="pearson"),
                    "spearman": _safe_corr(frame["score"], frame["logic_fidelity"], method="spearman"),
                }
            )
            rows.append(
                {
                    "policy_id": policy.id,
                    "metric": "within_selection_view",
                    "selection_view_a": evidence.target_view.id,
                    "selection_view_b": "effect_scaled",
                    "pearson": _safe_corr(frame["score"], frame["effect_scaled"], method="pearson"),
                    "spearman": _safe_corr(frame["score"], frame["effect_scaled"], method="spearman"),
                }
            )
    return pd.DataFrame(rows)


def target_view_score_correlations(
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    scored_for_policy: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, left in enumerate(sfxi_evidence):
        for right in sfxi_evidence[idx + 1 :]:
            a = scored_for_policy[left.target_view.id][["id", "score"]].rename(columns={"score": "score_a"})
            b = scored_for_policy[right.target_view.id][["id", "score"]].rename(columns={"score": "score_b"})
            merged = a.merge(b, on="id", how="inner")
            if len(merged) != len(a) or len(merged) != len(b):
                raise ValueError(
                    f"SFXI evidence ids do not align for {left.target_view.id} and {right.target_view.id}."
                )
            rows.append(
                {
                    "metric": "between_selection_views",
                    "selection_view_a": left.target_view.id,
                    "selection_view_b": right.target_view.id,
                    "pearson": _safe_corr(merged["score_a"], merged["score_b"], method="pearson"),
                    "spearman": _safe_corr(merged["score_a"], merged["score_b"], method="spearman"),
                }
            )
    return pd.DataFrame(rows)


def _safe_corr(left: pd.Series | np.ndarray, right: pd.Series | np.ndarray, *, method: str) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    if np.nanstd(a[mask]) == 0.0 or np.nanstd(b[mask]) == 0.0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(a[mask], b[mask]).statistic)
    if method == "spearman":
        return float(spearmanr(a[mask], b[mask]).statistic)
    raise ValueError(f"Unsupported correlation method: {method}")
