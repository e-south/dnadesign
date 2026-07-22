"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/sfxi_greedy_replay.py

Build a compact replay of the persisted historical SFXI greedy selections.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..core.contracts import SfxiEvidenceFrame

REPLAY_COLUMNS = (
    "selection_view_id",
    "source_campaign_slug",
    "run_id",
    "rank",
    "id",
    "sequence",
    "score",
    "logic_fidelity",
    "effect_scaled",
    "effect_rank",
    "logic_rank",
    "selection_view_count",
    "pool_candidate_count",
    "score_vs_effect_spearman",
    "score_vs_logic_spearman",
    "top_k_effect_overlap",
    "total_selection_slots",
    "unique_selected_sequences",
    "selected_in_all_views",
    "pairwise_overlap_total",
    "source_y_contract",
    "evidence_lifecycle",
)


def build_historical_sfxi_greedy_replay(
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    canonical_scored: dict[str, pd.DataFrame],
    *,
    top_k: int,
) -> pd.DataFrame:
    """Bind recomputed canonical scores to the exact persisted OPAL selections."""

    if top_k <= 0:
        raise ValueError("historical SFXI greedy replay requires top_k > 0.")
    if not sfxi_evidence:
        raise ValueError("historical SFXI greedy replay requires at least one source run.")
    view_ids = [evidence.target_view.id for evidence in sfxi_evidence]
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("historical SFXI greedy replay requires unique target views.")
    missing_views = sorted(set(view_ids) - set(canonical_scored))
    extra_views = sorted(set(canonical_scored) - set(view_ids))
    if missing_views or extra_views:
        raise ValueError(
            "historical SFXI greedy replay score views do not match source runs; "
            f"missing={missing_views}, extra={extra_views}."
        )

    selected_frames: list[pd.DataFrame] = []
    selected_sequences_by_view: dict[str, set[str]] = {}
    for evidence in sfxi_evidence:
        view_id = evidence.target_view.id
        scored = _validated_scored_frame(canonical_scored[view_id], view_id=view_id)
        persisted = _persisted_selected_rows(evidence, top_k=top_k)
        recomputed = scored.head(top_k).copy()
        persisted_ids = persisted["id"].astype(str).tolist()
        recomputed_ids = recomputed["id"].astype(str).tolist()
        if persisted_ids != recomputed_ids:
            raise ValueError(
                f"{view_id}: persisted selected identities do not match the recomputed canonical Top-{top_k}; "
                f"persisted={persisted_ids}, recomputed={recomputed_ids}."
            )
        persisted_sequences = persisted["sequence"].astype(str).tolist()
        recomputed_sequences = recomputed["sequence"].astype(str).tolist()
        if persisted_sequences != recomputed_sequences:
            raise ValueError(f"{view_id}: persisted selected sequences do not match canonical score rows.")

        effect_rank = scored["effect_scaled"].rank(method="min", ascending=False).astype(int)
        logic_rank = scored["logic_fidelity"].rank(method="min", ascending=False).astype(int)
        scored = scored.assign(effect_rank=effect_rank, logic_rank=logic_rank)
        recomputed = scored.head(top_k).copy()
        score_vs_effect = _spearman(scored["score"], scored["effect_scaled"], label=f"{view_id} score/effect")
        score_vs_logic = _spearman(scored["score"], scored["logic_fidelity"], label=f"{view_id} score/logic")
        highest_effect_ids = set(
            scored.sort_values(["effect_scaled", "id"], ascending=[False, True], kind="mergesort")
            .head(top_k)["id"]
            .astype(str)
        )
        recomputed["source_campaign_slug"] = evidence.source.source_campaign_slug
        recomputed["run_id"] = evidence.run_id
        recomputed["selection_view_id"] = view_id
        recomputed["pool_candidate_count"] = len(scored)
        recomputed["score_vs_effect_spearman"] = score_vs_effect
        recomputed["score_vs_logic_spearman"] = score_vs_logic
        recomputed["top_k_effect_overlap"] = len(set(recomputed_ids) & highest_effect_ids)
        selected_frames.append(recomputed)
        selected_sequences_by_view[view_id] = set(recomputed_sequences)

    selected = pd.concat(selected_frames, ignore_index=True)
    total_slots = len(selected)
    unique_sequences = int(selected["sequence"].nunique())
    sequence_view_counts = selected.groupby("sequence")["selection_view_id"].nunique()
    selected_in_all_views = int((sequence_view_counts == len(view_ids)).sum())
    pairwise_overlap_total = sum(
        len(selected_sequences_by_view[left] & selected_sequences_by_view[right])
        for left, right in combinations(view_ids, 2)
    )
    selected["selection_view_count"] = selected["sequence"].map(sequence_view_counts).astype(int)
    selected["total_selection_slots"] = total_slots
    selected["unique_selected_sequences"] = unique_sequences
    selected["selected_in_all_views"] = selected_in_all_views
    selected["pairwise_overlap_total"] = pairwise_overlap_total
    selected["source_y_contract"] = "sfxi_vec8"
    selected["evidence_lifecycle"] = "provenance_only"
    selected = selected.sort_values(["selection_view_id", "rank"], kind="mergesort").reset_index(drop=True)
    return selected.loc[:, REPLAY_COLUMNS]


def _validated_scored_frame(frame: pd.DataFrame, *, view_id: str) -> pd.DataFrame:
    required = {"id", "sequence", "score", "logic_fidelity", "effect_scaled", "rank"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{view_id}: canonical SFXI score frame is missing columns {missing}.")
    if frame["id"].astype(str).duplicated().any():
        raise ValueError(f"{view_id}: canonical SFXI score frame contains duplicate candidate ids.")
    result = frame.loc[:, sorted(required)].copy()
    result["id"] = result["id"].astype(str)
    result["sequence"] = result["sequence"].astype(str)
    for column in ("score", "logic_fidelity", "effect_scaled"):
        values = result[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{view_id}: canonical SFXI {column} contains non-finite values.")
    expected_score = result["logic_fidelity"].to_numpy(dtype=float) * result["effect_scaled"].to_numpy(dtype=float)
    if not np.allclose(result["score"].to_numpy(dtype=float), expected_score, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{view_id}: canonical SFXI score is not logic fidelity multiplied by scaled effect.")
    result = result.sort_values(["score", "id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    expected_rank = np.arange(1, len(result) + 1)
    if not np.array_equal(result["rank"].to_numpy(dtype=int), expected_rank):
        raise ValueError(f"{view_id}: canonical SFXI ranks do not match deterministic score order.")
    return result


def _persisted_selected_rows(evidence: SfxiEvidenceFrame, *, top_k: int) -> pd.DataFrame:
    required = {"id", "sequence", "sel__is_selected", "sel__rank_competition"}
    missing = sorted(required - set(evidence.predictions.columns))
    if missing:
        raise ValueError(f"{evidence.source.source_id}: prediction ledger is missing selection columns {missing}.")
    selected = evidence.predictions.loc[
        evidence.predictions["sel__is_selected"].astype(bool),
        ["id", "sequence", "sel__rank_competition"],
    ].copy()
    if len(selected) != top_k:
        raise ValueError(
            f"{evidence.target_view.id}: persisted selected identities contain {len(selected)} rows; expected {top_k}."
        )
    selected = selected.sort_values(["sel__rank_competition", "id"], kind="mergesort").reset_index(drop=True)
    if selected["sel__rank_competition"].astype(int).tolist() != list(range(1, top_k + 1)):
        raise ValueError(f"{evidence.target_view.id}: persisted selection ranks are not exactly 1..{top_k}.")
    return selected


def _spearman(left: pd.Series, right: pd.Series, *, label: str) -> float:
    value = float(left.corr(right, method="spearman"))
    if not np.isfinite(value):
        raise ValueError(f"{label} Spearman correlation is undefined.")
    return value


__all__ = ["REPLAY_COLUMNS", "build_historical_sfxi_greedy_replay"]
