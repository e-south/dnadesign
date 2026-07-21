"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/opal_round_source.py

Measured OPAL selection-batch source for the study synthesis handoff.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from dnadesign.opal import load_config, load_selection_batch, load_selection_set
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    PROMOTER_ALIAS_REGISTRY_PATH,
    load_study_promoter_alias_registry,
)

from .contracts import SelectedCandidate, SelectionMembership

OPAL_ROUND_SELECTION_SOURCE = "opal_selection_batch"


def _study_value_error_from_opal(exc: RuntimeError) -> ValueError:
    message = str(exc)
    if message.startswith("Missing runs sink:") or message.startswith("Missing predictions sink:"):
        _, path = message.split(":", 1)
        return ValueError(f"required OPAL parquet artifact is missing:{path}")
    return ValueError(message)


def _resolve_repo_path(repo_root: Path | None, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or repo_root is None:
        return path
    return repo_root / path


def _records_path(config_path: Path) -> Path:
    cfg = load_config(config_path)
    location = cfg.data.location
    path = Path(str(getattr(location, "path")))
    dataset = getattr(location, "dataset", None)
    return path / str(dataset) / "records.parquet" if dataset is not None else path


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
        duplicates = sorted(records.loc[records["id"].duplicated(), "id"].unique().tolist())
        raise ValueError(f"records table contains duplicate ids: {duplicates[:10]}")
    records["sequence"] = records["sequence"].astype(str).str.strip()
    return dict(zip(records["id"], records["sequence"], strict=True))


def _membership_from(value: Mapping[str, Any]) -> SelectionMembership:
    return SelectionMembership.from_mapping(value)


def _require_verified_selection_sets(selection_sets: Mapping[str, dict[str, Any]]) -> None:
    for view_id, payload in selection_sets.items():
        verification = payload.get("verification")
        if not isinstance(verification, Mapping):
            raise ValueError(f"selection replay verification missing for selection_view={view_id}")
        mismatch_count = verification.get("mismatch_count")
        if (
            verification.get("status") != "pass"
            or isinstance(mismatch_count, bool)
            or not isinstance(mismatch_count, int)
            or mismatch_count != 0
        ):
            raise ValueError(
                "selection replay verification failed for "
                f"selection_view={view_id}: status={verification.get('status')!r}, "
                f"mismatch_count={mismatch_count!r}"
            )


def _validate_batch_memberships(
    batch_rows: list[dict[str, Any]],
    *,
    selection_sets: Mapping[str, dict[str, Any]],
) -> None:
    selected_by_view = {
        view_id: {str(row["id"]): row for row in payload["rows"]} for view_id, payload in selection_sets.items()
    }
    batch_ids_by_view: dict[str, set[str]] = {view_id: set() for view_id in selection_sets}
    for row in batch_rows:
        candidate_id = str(row["id"])
        memberships = row.get("selection_memberships")
        if not isinstance(memberships, list) or not memberships:
            raise ValueError(f"selection batch candidate {candidate_id!r} has no selection memberships")
        declared_view_ids = row.get("selection_view_ids")
        if not isinstance(declared_view_ids, list):
            raise ValueError(f"selection batch candidate {candidate_id!r} has invalid selection_view_ids")
        membership_view_ids = [str(item.get("selection_view_id", "")) for item in memberships]
        if declared_view_ids != membership_view_ids:
            raise ValueError(f"selection batch candidate {candidate_id!r} view IDs do not match its membership records")
        for membership_raw in memberships:
            membership = _membership_from(membership_raw)
            selected_rows = selected_by_view.get(membership.selection_view_id)
            if selected_rows is None:
                raise ValueError(
                    f"selection batch candidate {candidate_id!r} references unknown view "
                    f"{membership.selection_view_id!r}"
                )
            selected_row = selected_rows.get(candidate_id)
            if selected_row is None:
                raise ValueError(
                    f"selection batch candidate {candidate_id!r} is absent from selection set "
                    f"{membership.selection_view_id!r}"
                )
            if int(selected_row["rank_competition"]) != membership.rank:
                raise ValueError(
                    f"selection batch rank mismatch for candidate {candidate_id!r} in view "
                    f"{membership.selection_view_id!r}"
                )
            if membership.score is None or abs(float(selected_row["score"]) - membership.score) > 1e-9:
                raise ValueError(
                    f"selection batch score mismatch for candidate {candidate_id!r} in view "
                    f"{membership.selection_view_id!r}"
                )
            batch_ids_by_view[membership.selection_view_id].add(candidate_id)
    for view_id, selected_rows in selected_by_view.items():
        if batch_ids_by_view[view_id] != set(selected_rows):
            raise ValueError(f"selection batch membership coverage mismatch for view {view_id!r}")


def selected_candidates_from_opal_round(
    campaign_config: str | Path,
    *,
    as_of_round: int,
    run_id: str | None = None,
    repo_root: str | Path | None = None,
    alias_registry_path: str | Path = PROMOTER_ALIAS_REGISTRY_PATH,
) -> tuple[list[SelectedCandidate], dict[str, Any]]:
    """Load one campaign run's verified logical selection batch."""

    if int(as_of_round) < 0:
        raise ValueError("as_of_round must be non-negative")
    root = Path(repo_root) if repo_root is not None else None
    config_path = _resolve_repo_path(root, campaign_config)
    cfg = load_config(config_path)
    view_ids = [view.id for view in cfg.selection_views]
    try:
        batch = load_selection_batch(
            config_path,
            round_selector=str(as_of_round),
            run_id=run_id,
        )
        selection_sets = {
            view_id: load_selection_set(
                config_path,
                selection_view_id=view_id,
                round_selector=str(as_of_round),
                run_id=str(batch["run_id"]),
                verify_artifact=True,
            )
            for view_id in view_ids
        }
    except RuntimeError as exc:
        raise _study_value_error_from_opal(exc) from exc

    _require_verified_selection_sets(selection_sets)
    batch_rows = list(batch["rows"])
    _validate_batch_memberships(batch_rows, selection_sets=selection_sets)
    records_by_id = _records_sequence_map(config_path)
    for view_id, payload in selection_sets.items():
        mismatches = [
            str(row["id"]) for row in payload["rows"] if records_by_id.get(str(row["id"])) != str(row["sequence"])
        ]
        if mismatches:
            raise ValueError(
                f"OPAL selected sequence mismatch against records table for selection_view={view_id}: {mismatches[:10]}"
            )
    view_order = {view_id: index for index, view_id in enumerate(view_ids)}
    if root is None:
        root = _repo_root_from_config(config_path)
    alias_registry = load_study_promoter_alias_registry(root, registry_path=alias_registry_path)

    def order_key(row: dict[str, Any]) -> tuple[int, int, str]:
        memberships = [_membership_from(item) for item in row["selection_memberships"]]
        first = min(memberships, key=lambda item: (view_order[item.selection_view_id], item.rank))
        return view_order[first.selection_view_id], first.rank, str(row["id"])

    candidates: list[SelectedCandidate] = []
    for batch_rank, row in enumerate(sorted(batch_rows, key=order_key), start=1):
        candidate_id = str(row["id"])
        sequence = records_by_id.get(candidate_id)
        if sequence is None:
            raise ValueError(f"OPAL selection batch id missing from records table: {candidate_id}")
        memberships = tuple(_membership_from(item) for item in row["selection_memberships"])
        candidates.append(
            SelectedCandidate(
                campaign_slug=cfg.campaign.slug,
                selection_memberships=memberships,
                as_of_round=int(batch["as_of_round"]),
                run_id=str(batch["run_id"]),
                selection_rank=batch_rank,
                id=candidate_id,
                sequence=sequence,
                synthesis_name=alias_registry.alias_for(candidate_id=candidate_id, sequence=sequence),
                selection_source=OPAL_ROUND_SELECTION_SOURCE,
                selection_epoch="opal_model_round",
                assay_batch_index=None,
                model_as_of_round=int(batch["as_of_round"]),
            )
        )

    selection_view_counts = {
        view_id: int(sum(view_id in row.selection_view_ids for row in candidates)) for view_id in view_ids
    }
    study_aliases = [row.synthesis_name for row in candidates]
    replay_mismatch_count = sum(
        int(payload.get("verification", {}).get("mismatch_count", 0)) for payload in selection_sets.values()
    )
    return candidates, {
        "source": OPAL_ROUND_SELECTION_SOURCE,
        "campaign_slug": cfg.campaign.slug,
        "config_path": str(config_path),
        "workdir": str(batch["campaign"]["workdir"]),
        "as_of_round": int(batch["as_of_round"]),
        "run_id": str(batch["run_id"]),
        "row_count": int(len(candidates)),
        "unique_candidate_count": len({row.id for row in candidates}),
        "unique_sequence_count": len({row.sequence for row in candidates}),
        "unique_study_alias_count": len(set(study_aliases)),
        "study_aliases": study_aliases,
        "selection_view_counts": selection_view_counts,
        "replay_mismatch_count": replay_mismatch_count,
        "selection_batch_schema_version": str(batch["schema_version"]),
        "selection_batch_path": str(batch["selection_batch_path"]),
        "candidate_records_path": str(_records_path(config_path)),
        "promoter_alias_registry": {
            "path": str(root / alias_registry.path),
            "sha256": hashlib.sha256((root / alias_registry.path).read_bytes()).hexdigest(),
            "assignment_count": len(alias_registry.assignments),
            "next_alias": alias_registry.alias_format.render(alias_registry.next_ordinal),
        },
        "selection_sets": {
            view_id: {
                "selected_count": int(payload["selected_count"]),
                "selection_path": payload.get("selection_path"),
                "verification": payload.get("verification"),
            }
            for view_id, payload in selection_sets.items()
        },
    }


def _repo_root_from_config(config_path: Path) -> Path:
    for parent in (config_path.resolve(), *config_path.resolve().parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError("repo_root is required when the campaign config is outside a dnadesign checkout")
