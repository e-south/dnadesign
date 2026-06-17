"""Measured OPAL round selected-candidate source for synthesis handoff."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from dnadesign.opal import load_config, load_selection_set

from .campaigns import opal_round_synthesis_name
from .contracts import SelectedCandidate

OPAL_ROUND_SELECTION_SOURCE = "opal_ledger"


def _study_value_error_from_opal(exc: RuntimeError) -> ValueError:
    message = str(exc)
    if message.startswith("Missing runs sink:") or message.startswith("Missing predictions sink:"):
        _, path = message.split(":", 1)
        return ValueError(f"required OPAL parquet artifact is missing:{path}")
    return ValueError(message)


def _resolve_repo_path(repo_root: Path | None, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if repo_root is None:
        return path
    return repo_root / path


def _records_path(config_path: Path) -> Path:
    cfg = load_config(config_path)
    loc = cfg.data.location
    loc_path = Path(str(getattr(loc, "path")))
    dataset = getattr(loc, "dataset", None)
    if dataset is not None:
        return loc_path / str(dataset) / "records.parquet"
    return loc_path


def _read_parquet(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"required OPAL parquet artifact is missing: {path}")
    return pd.read_parquet(path, columns=list(columns) if columns is not None else None)


def _records_sequence_map(config_path: Path) -> dict[str, str]:
    records = _read_parquet(_records_path(config_path), columns=["id", "sequence"])
    missing = [column for column in ("id", "sequence") if column not in records.columns]
    if missing:
        raise ValueError(f"records table missing required columns for synthesis handoff: {missing}")
    records = records.copy()
    records["id"] = records["id"].astype(str).str.strip()
    if records["id"].duplicated().any():
        dupes = sorted(records.loc[records["id"].duplicated(), "id"].unique().tolist())
        raise ValueError(f"records table contains duplicate ids: {dupes[:10]}")
    records["sequence"] = records["sequence"].astype(str).str.strip()
    return dict(zip(records["id"], records["sequence"], strict=True))


def _validate_selected_sequences_against_records(
    selected: pd.DataFrame,
    *,
    records_by_id: Mapping[str, str],
    campaign_slug: str,
) -> None:
    missing_ids = [
        candidate_id for candidate_id in selected["id"].astype(str).tolist() if candidate_id not in records_by_id
    ]
    if missing_ids:
        raise ValueError(
            f"OPAL selected ids missing from records table for campaign={campaign_slug}: {missing_ids[:10]}"
        )
    mismatches: list[str] = []
    for candidate_id, sequence in selected[["id", "sequence"]].itertuples(index=False, name=None):
        expected = str(records_by_id[str(candidate_id)])
        observed = str(sequence)
        if observed != expected:
            mismatches.append(str(candidate_id))
    if mismatches:
        raise ValueError(
            f"OPAL selected sequence mismatch against records table for campaign={campaign_slug}: {mismatches[:10]}"
        )


def _campaign_selected_candidates(
    config_path: Path,
    *,
    as_of_round: int,
    requested_run_id: str | None,
) -> tuple[list[SelectedCandidate], dict[str, Any]]:
    cfg = load_config(config_path)
    try:
        selection_set = load_selection_set(
            config_path,
            round_selector=str(as_of_round),
            run_id=requested_run_id,
            verify_artifact=False,
        )
    except RuntimeError as exc:
        raise _study_value_error_from_opal(exc) from exc

    selected = pd.DataFrame(selection_set["rows"])
    _validate_selected_sequences_against_records(
        selected,
        records_by_id=_records_sequence_map(config_path),
        campaign_slug=cfg.campaign.slug,
    )

    candidates: list[SelectedCandidate] = []
    for order_index, row in enumerate(selected.itertuples(index=False), start=1):
        candidates.append(
            SelectedCandidate(
                campaign_slug=cfg.campaign.slug,
                as_of_round=int(selection_set["as_of_round"]),
                run_id=str(selection_set["run_id"]),
                selection_rank=order_index,
                id=str(row.id),
                sequence=str(row.sequence),
                synthesis_name=opal_round_synthesis_name(cfg.campaign.slug, int(as_of_round), order_index),
                selection_source=OPAL_ROUND_SELECTION_SOURCE,
                selection_epoch="opal_model_round",
                assay_batch_index=None,
                model_as_of_round=int(selection_set["as_of_round"]),
            )
        )
    report = {
        "campaign_slug": cfg.campaign.slug,
        "config_path": str(config_path),
        "workdir": str(selection_set["campaign"]["workdir"]),
        "as_of_round": int(selection_set["as_of_round"]),
        "run_id": str(selection_set["run_id"]),
        "selected_count": int(len(candidates)),
        "selection_set_schema_version": str(selection_set["schema_version"]),
        "selection_path": selection_set.get("selection_path"),
        "selection_verification": selection_set.get("verification"),
    }
    return candidates, report


def selected_candidates_from_opal_round_campaigns(
    campaign_configs: Sequence[str | Path],
    *,
    as_of_round: int,
    run_id: str | None = None,
    run_id_by_campaign: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> tuple[list[SelectedCandidate], dict[str, Any]]:
    """Build synthesis candidates from measured OPAL round ledgers."""

    if int(as_of_round) < 0:
        raise ValueError("as_of_round must be non-negative")
    if not campaign_configs:
        raise ValueError("at least one OPAL campaign config is required")
    root = Path(repo_root) if repo_root is not None else None
    config_paths = [_resolve_repo_path(root, config_path) for config_path in campaign_configs]
    if run_id is not None and len(config_paths) != 1:
        raise ValueError("run_id without a campaign key is only supported for a single campaign config")

    selected: list[SelectedCandidate] = []
    campaign_reports: list[dict[str, Any]] = []
    run_map = {str(key): str(value) for key, value in (run_id_by_campaign or {}).items()}
    for config_path in config_paths:
        cfg = load_config(config_path)
        requested_run_id = run_map.get(cfg.campaign.slug, run_id)
        campaign_selected, campaign_report = _campaign_selected_candidates(
            config_path,
            as_of_round=int(as_of_round),
            requested_run_id=requested_run_id,
        )
        selected.extend(campaign_selected)
        campaign_reports.append(campaign_report)

    report = {
        "source": OPAL_ROUND_SELECTION_SOURCE,
        "as_of_round": int(as_of_round),
        "row_count": int(len(selected)),
        "campaign_counts": {row["campaign_slug"]: int(row["selected_count"]) for row in campaign_reports},
        "campaigns": campaign_reports,
    }
    return selected, report
