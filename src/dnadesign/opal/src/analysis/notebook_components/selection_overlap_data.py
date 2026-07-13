"""Canonical selection-artifact loading for notebook overlap views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ...core.utils import ExitCodes, OpalError


def build_selection_overlap_rows(
    campaigns: Iterable[Mapping[str, Any]],
    *,
    round_selector: str | int | None,
) -> list[dict[str, Any]]:
    """Return one row per selected campaign-view candidate."""

    import polars as pl

    rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        if not isinstance(campaign, Mapping):
            raise OpalError("Campaign overlap inputs must be mappings.", ExitCodes.CONTRACT_VIOLATION)
        campaign_meta = campaign.get("campaign")
        if not isinstance(campaign_meta, Mapping):
            raise OpalError("Campaign overlap input is missing campaign metadata.", ExitCodes.CONTRACT_VIOLATION)
        workdir_text = str(campaign_meta.get("workdir") or "").strip()
        if not workdir_text:
            raise OpalError("Campaign overlap input is missing campaign.workdir.", ExitCodes.CONTRACT_VIOLATION)
        selection_path = resolve_selection_artifact_path(Path(workdir_text), round_selector=round_selector)
        if selection_path is None:
            continue
        frame = pl.read_parquet(selection_path)
        required = {
            "selection_view_id",
            "id",
            "as_of_round",
            "run_id",
            "rank_competition",
            "score",
            "selection_score",
            "score_ref",
            "sequence",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise OpalError(
                f"Selection artifact {selection_path} is missing columns: {missing}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        declared_views = _declared_view_ids(campaign_meta)
        artifact_views = sorted(str(value) for value in frame.get_column("selection_view_id").unique().to_list())
        undeclared = sorted(set(artifact_views) - set(declared_views)) if declared_views else []
        if undeclared:
            raise OpalError(
                f"Selection artifact contains undeclared views: {undeclared}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        campaign_slug = str(campaign_meta.get("slug") or selection_path.parts[-5])
        base_label = _campaign_label(campaign_meta, fallback=campaign_slug)
        multi_view = len(artifact_views) > 1
        for raw in frame.to_dicts():
            candidate_id = str(raw["id"]).strip()
            if not candidate_id:
                raise OpalError("Selection artifact contains a blank candidate id.", ExitCodes.CONTRACT_VIOLATION)
            view_id = str(raw["selection_view_id"])
            campaign_key = f"{campaign_slug}:{view_id}" if multi_view else campaign_slug
            campaign_label = f"{base_label} | {_view_label(view_id)}" if multi_view else base_label
            rows.append(
                {
                    "campaign": campaign_key,
                    "campaign_label": campaign_label,
                    "selection_view_id": view_id,
                    "round": int(raw["as_of_round"]),
                    "run_id": str(raw["run_id"]),
                    "id": candidate_id,
                    "short_id": _short_id(candidate_id),
                    "rank": int(raw["rank_competition"]),
                    "score": float(raw["score"]),
                    "selection_score": float(raw["selection_score"]),
                    "score_ref": str(raw["score_ref"]),
                    "selection_path": str(selection_path),
                    "sequence": str(raw["sequence"] or ""),
                }
            )
    return rows


def resolve_selection_artifact_path(
    workdir: Path,
    *,
    round_selector: str | int | None,
) -> Path | None:
    rounds_dir = workdir / "outputs" / "rounds"
    if not rounds_dir.exists():
        return None
    round_value = str(round_selector or "latest")
    if round_value not in {"latest", "all"}:
        path = rounds_dir / f"round_{int(round_value)}" / "selection" / "selections.parquet"
        return path if path.exists() else None
    candidates: list[tuple[int, Path]] = []
    for child in rounds_dir.glob("round_*"):
        try:
            round_index = int(child.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        path = child / "selection" / "selections.parquet"
        if path.exists():
            candidates.append((round_index, path))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def _declared_view_ids(campaign: Mapping[str, Any]) -> list[str]:
    raw = campaign.get("selection_views") or []
    return [str(view["id"]) for view in raw if isinstance(view, Mapping) and str(view.get("id") or "").strip()]


def _campaign_label(campaign: Mapping[str, Any], *, fallback: str) -> str:
    name = str(campaign.get("name") or fallback).split("|", 1)[0].strip()
    for prefix in ("SECG ", "Stress ethanol/ciprofloxacin "):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name or fallback


def _view_label(view_id: str) -> str:
    return "AND" if view_id.lower() == "and" else view_id.replace("_", " ").strip().title()


def _short_id(value: str) -> str:
    return value[:8] if len(value) > 8 else value


__all__ = ["build_selection_overlap_rows", "resolve_selection_artifact_path"]
