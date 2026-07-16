"""Derive campaign backlog state from selections and observed labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...core.utils import OpalError
from ...storage.state import CampaignState
from ...storage.workspace import CampaignWorkspace

BACKLOG_COUNT_KEY = "number_of_selected_but_not_yet_labeled_candidates_total"


def _candidate_ids(frame: pd.DataFrame, *, source: str) -> set[str]:
    if "id" not in frame.columns:
        raise OpalError(f"Cannot derive campaign backlog: {source} is missing required column 'id'.")
    if frame["id"].isna().any():
        raise OpalError(f"Cannot derive campaign backlog: {source} contains null candidate ids.")
    candidate_ids = {str(value) for value in frame["id"].tolist()}
    if "" in candidate_ids:
        raise OpalError(f"Cannot derive campaign backlog: {source} contains an empty candidate id.")
    return candidate_ids


def derive_pending_candidate_count(
    *,
    state: CampaignState,
    workspace: CampaignWorkspace,
    current_selection_batch: pd.DataFrame,
    observed_events: pd.DataFrame,
) -> int:
    """Count selected candidate IDs that have no observed-label event."""

    selected_ids = _candidate_ids(current_selection_batch, source="current selection batch")
    rounds_root = workspace.rounds_dir.resolve()
    for round_entry in state.rounds:
        artifact_path_raw = round_entry.artifacts.get("selection_batch_parquet")
        if not artifact_path_raw:
            raise OpalError(
                "Cannot derive campaign backlog: a retained state round is missing artifacts.selection_batch_parquet."
            )
        artifact_path = Path(str(artifact_path_raw)).resolve()
        try:
            artifact_path.relative_to(rounds_root)
        except ValueError as exc:
            raise OpalError(
                f"Cannot derive campaign backlog from selection artifact outside {rounds_root}: {artifact_path}"
            ) from exc
        if not artifact_path.is_file():
            raise OpalError(f"Cannot derive campaign backlog: selection artifact does not exist: {artifact_path}")
        try:
            prior_selection = pd.read_parquet(artifact_path, columns=["id"])
        except Exception as exc:
            raise OpalError(f"Cannot derive campaign backlog from selection artifact {artifact_path}: {exc}") from exc
        selected_ids.update(
            _candidate_ids(
                prior_selection,
                source=f"selection batch for round {int(round_entry.round_index)}",
            )
        )

    observed_ids = _candidate_ids(observed_events, source="observed-label events")
    return len(selected_ids - observed_ids)


__all__ = ["BACKLOG_COUNT_KEY", "derive_pending_candidate_count"]
