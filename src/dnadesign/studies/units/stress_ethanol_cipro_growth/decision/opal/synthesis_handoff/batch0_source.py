"""Batch-0 selected-candidate source for synthesis handoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..batch0.select import (
    build_candidate_frame,
    load_sampling_config,
    select_batch0,
    validate_configured_candidate_feature_table,
    validate_selected_ids_against_candidate_feature_table,
)
from .campaigns import STRESS_OPAL_CAMPAIGN_ALIAS_CODES
from .campaigns import batch0_synthesis_name as _campaign_batch0_synthesis_name
from .contracts import SelectedCandidate

DEFAULT_BATCH0_BATCH_ID = "stress-opal-batch0-sfxi-v1"
DEFAULT_BATCH0_RUN_ID = "batch0_pre_assay_review"
DEFAULT_BATCH0_SELECTION_SOURCE = "batch0_pre_assay"
DEFAULT_BATCH0_SELECTION_CONFIG = Path(__file__).resolve().parents[1] / "batch0" / "sampling.yaml"

BATCH0_CAMPAIGN_ALIAS_CODES = STRESS_OPAL_CAMPAIGN_ALIAS_CODES


def _repo_root_from(path: Path) -> Path:
    for parent in [path.resolve(), *path.resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"could not resolve repo root from {path}")


def batch0_synthesis_name(campaign_slug: str, selection_rank: int) -> str:
    """Return the deterministic batch-0 synthesis alias for a campaign row."""

    try:
        return _campaign_batch0_synthesis_name(campaign_slug, selection_rank)
    except ValueError as exc:
        if "unknown stress OPAL campaign slug" in str(exc):
            raise ValueError(f"unknown batch-0 campaign slug for synthesis alias: {campaign_slug}") from None
        raise


def selected_candidates_from_batch0_review(
    review: pd.DataFrame,
    *,
    as_of_round: int = 0,
    run_id: str = DEFAULT_BATCH0_RUN_ID,
) -> list[SelectedCandidate]:
    """Convert batch-0 review rows into synthesis selected-candidate records."""

    required = ("campaign", "id", "sequence")
    missing = [column for column in required if column not in review.columns]
    if missing:
        raise ValueError("batch-0 review table missing required columns: " + ", ".join(missing))

    selected: list[SelectedCandidate] = []
    ranks_by_campaign: dict[str, int] = {}
    for _, row in review.iterrows():
        campaign_slug = str(row["campaign"])
        rank = ranks_by_campaign.get(campaign_slug, 0) + 1
        ranks_by_campaign[campaign_slug] = rank
        selected.append(
            SelectedCandidate(
                campaign_slug=campaign_slug,
                as_of_round=as_of_round,
                run_id=run_id,
                selection_rank=rank,
                id=str(row["id"]),
                sequence=str(row["sequence"]),
                synthesis_name=batch0_synthesis_name(campaign_slug, rank),
                selection_source=DEFAULT_BATCH0_SELECTION_SOURCE,
                selection_epoch="pre_assay_seed",
                assay_batch_index=0,
                model_as_of_round=None,
            )
        )
    return selected


def build_batch0_selected_candidates(
    *,
    config_path: str | Path = DEFAULT_BATCH0_SELECTION_CONFIG,
    repo_root: str | Path | None = None,
) -> tuple[list[SelectedCandidate], dict[str, Any]]:
    """Build synthesis candidates from the checked-in batch-0 selector."""

    cfg_path = Path(config_path)
    root = Path(repo_root) if repo_root is not None else _repo_root_from(cfg_path)
    config = load_sampling_config(cfg_path)
    candidate_table_report = validate_configured_candidate_feature_table(config, repo_root=root)
    candidates = build_candidate_frame(config, repo_root=root)
    review = select_batch0(candidates, config)
    selection_table_report = validate_selected_ids_against_candidate_feature_table(
        review,
        config,
        repo_root=root,
    )
    selected = selected_candidates_from_batch0_review(review)
    campaign_counts = review.groupby("campaign", sort=False).size().astype(int).to_dict()
    report: dict[str, Any] = {
        "source": DEFAULT_BATCH0_SELECTION_SOURCE,
        "config_path": str(cfg_path),
        "candidate_feature_table": candidate_table_report,
        "selection_candidate_table": selection_table_report,
        "campaign_counts": campaign_counts,
        "row_count": int(len(selected)),
    }
    return selected, report
