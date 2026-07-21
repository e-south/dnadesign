"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/observed_sfxi.py

Bind verified source artifacts to the historical observed-label SFXI replay.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..core.contracts import SfxiEvidenceFrame
from ..evaluation.observed_sfxi_replay import (
    ObservedSfxiViewContext,
    build_observed_sfxi_decomposition,
    summarize_observed_sfxi_decomposition,
)
from .candidate_identity import ResponseCandidateIdentityBindings
from .label_truth import LabelTruthState

_HISTORICAL_SFXI_HIGHLIGHT_COUNT = 6


@dataclass(frozen=True)
class HistoricalObservedSfxiEvidence:
    """Measured SFXI detail and its corpus-sensitivity summary."""

    components: pd.DataFrame
    robustness: pd.DataFrame


def build_historical_observed_sfxi_evidence(
    source_rows: pd.DataFrame,
    label_rows: pd.DataFrame,
    *,
    sfxi_evidence: tuple[SfxiEvidenceFrame, ...],
    label_truth_state: LabelTruthState,
    candidate_bindings: ResponseCandidateIdentityBindings,
) -> HistoricalObservedSfxiEvidence:
    """Build the replay without passing prediction rows into the pure evaluator."""

    if label_truth_state.ready and not label_truth_state.candidate_ids:
        raise ValueError("A promoted response-window label state must contain candidate IDs.")
    active_identities = (
        _load_bound_identities(candidate_bindings, candidate_ids=label_truth_state.candidate_ids)
        if label_truth_state.ready
        else None
    )
    contexts = tuple(
        ObservedSfxiViewContext(
            selection_view_id=evidence.target_view.id,
            target_mask=evidence.target_view.target_mask,
            denom=evidence.denom,
            scaling_percentile=evidence.scaling_percentile,
            scaling_min_n=evidence.scaling_min_n,
            scaling_eps=evidence.scaling_eps,
            intensity_log2_offset_delta=evidence.intensity_log2_offset_delta,
            source_campaign_slug=evidence.source.source_campaign_slug,
            source_run_id=evidence.run_id,
        )
        for evidence in sfxi_evidence
    )
    components = build_observed_sfxi_decomposition(
        source_rows,
        label_rows,
        view_contexts=contexts,
        active_identities=active_identities,
        top_k=_HISTORICAL_SFXI_HIGHLIGHT_COUNT,
    )
    return HistoricalObservedSfxiEvidence(
        components=components,
        robustness=summarize_observed_sfxi_decomposition(components),
    )


def _load_bound_identities(
    candidate_bindings: ResponseCandidateIdentityBindings,
    *,
    candidate_ids: tuple[str, ...],
) -> pd.DataFrame:
    ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
    if len(ids) != len(set(ids)) or any(not candidate_id for candidate_id in ids):
        raise ValueError("Promoted response-window candidate IDs must be non-empty and unique.")
    bindings = pd.read_parquet(
        candidate_bindings.records_path,
        columns=["candidate_id", "canonical_sequence", "binding_status"],
        filters=[("candidate_id", "in", list(ids))],
    )
    bindings = bindings.loc[bindings["binding_status"].astype(str).eq("resolved")].copy()
    grouped = bindings.groupby("candidate_id", sort=False)["canonical_sequence"].agg(
        lambda values: tuple(sorted(set(str(value) for value in values)))
    )
    ambiguous = grouped.loc[grouped.map(len).ne(1)]
    if not ambiguous.empty:
        raise ValueError(
            f"Promoted response-window candidate bindings have ambiguous sequences: {ambiguous.index[:5].tolist()}"
        )
    missing = sorted(set(ids) - set(grouped.index.astype(str)))
    if missing:
        raise ValueError(
            f"Promoted response-window candidates are absent from the verified binding artifact: {missing[:5]}"
        )
    result = pd.DataFrame(
        {
            "id": list(ids),
            "sequence": [grouped.loc[candidate_id][0] for candidate_id in ids],
        }
    )
    if result["sequence"].duplicated().any():
        raise ValueError("Promoted response-window binding sequences must be unique.")
    return result


__all__ = ["HistoricalObservedSfxiEvidence", "build_historical_observed_sfxi_evidence"]
