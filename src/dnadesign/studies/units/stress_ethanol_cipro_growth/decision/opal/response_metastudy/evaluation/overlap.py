"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/overlap.py

Overlap diagnostics for SFXI top-k selections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from ..core.contracts import PolicySpec, SfxiEvidenceFrame
from .selection import select_top_rows


def build_overlap_by_k(
    policies: Iterable[PolicySpec],
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    scored: dict[str, dict[str, pd.DataFrame]],
    *,
    k_values: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy in policies:
        for k in k_values:
            top_by_target_view = {
                evidence.target_view.id: select_top_rows(scored[policy.id][evidence.target_view.id], top_k=k)
                for evidence in sfxi_evidence
            }
            stats = overlap_stats(top_by_target_view)
            labels = sorted(top_by_target_view)
            sets = {key: set(frame["id"].astype(str).tolist()) for key, frame in top_by_target_view.items()}
            for idx, left in enumerate(labels):
                for right in labels[idx + 1 :]:
                    pair_count = int(len(sets[left] & sets[right]))
                    rows.append(
                        {
                            "policy_id": policy.id,
                            "k": int(k),
                            "selected_count_a": int(len(sets[left])),
                            "selected_count_b": int(len(sets[right])),
                            "overlap_type": "pairwise",
                            "selection_view_a": left,
                            "selection_view_b": right,
                            "observed_overlap": pair_count,
                            "unique_topk": int(stats["unique_topk"]),
                            "all_target_views_overlap": int(stats["all_target_views_overlap"]),
                        }
                    )
            all_target_views = set.intersection(*(sets[label] for label in labels))
            rows.append(
                {
                    "policy_id": policy.id,
                    "k": int(k),
                    "selected_count_a": int(min(len(sets[label]) for label in labels)),
                    "selected_count_b": int(max(len(sets[label]) for label in labels)),
                    "overlap_type": "all_target_views",
                    "selection_view_a": "all",
                    "selection_view_b": "all",
                    "observed_overlap": int(len(all_target_views)),
                    "unique_topk": int(stats["unique_topk"]),
                    "all_target_views_overlap": int(stats["all_target_views_overlap"]),
                }
            )
    return pd.DataFrame(rows)


def overlap_stats(top_by_target_view: dict[str, pd.DataFrame]) -> dict[str, int]:
    sets = {key: set(frame["id"].astype(str).tolist()) for key, frame in top_by_target_view.items()}
    labels = sorted(sets)
    all_ids = set().union(*sets.values())
    all_target_views = set.intersection(*(sets[label] for label in labels)) if labels else set()
    pairwise_total = 0
    for idx, left in enumerate(labels):
        for right in labels[idx + 1 :]:
            pairwise_total += len(sets[left] & sets[right])
    return {
        "unique_topk": len(all_ids),
        "all_target_views_overlap": len(all_target_views),
        "pairwise_overlap_total": int(pairwise_total),
    }
