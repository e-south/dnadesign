"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/campaign_navigation.py

Resolve the study-owned current OPAL review route.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.opal import load_config
from dnadesign.ops.status import resolve_repo_relative_path

STUDY_ID = "stress_ethanol_cipro_growth"
CAMPAIGN_RECORD_PATH = Path("docs/studies/stress_ethanol_cipro_growth/record/campaign.yaml")


@dataclass(frozen=True)
class CurrentCampaignNavigation:
    """Verified source-tree navigation for the study's current OPAL campaign."""

    campaign_slug: str
    config_path: Path
    notebook_path: Path
    notebook_materialized: bool
    selection_view_ids: tuple[str, ...]
    objective_names: tuple[str, ...]
    run_command: str


def load_current_campaign_navigation(
    repo_root: str | Path,
    *,
    record_path: str | Path = CAMPAIGN_RECORD_PATH,
) -> CurrentCampaignNavigation:
    """Resolve one current OPAL route from the checked-in study record."""

    root = Path(repo_root).expanduser().resolve()
    record = _resolve_inside_repo(root, record_path, label="campaign record")
    if not record.is_file():
        raise FileNotFoundError(f"Study campaign record not found: {record}")
    payload = _mapping(yaml.safe_load(record.read_text(encoding="utf-8")), label="campaign record")
    if payload.get("campaign_id") != STUDY_ID or payload.get("path_base") != "repo":
        raise ValueError("Study campaign record identity or path base is invalid.")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("Study campaign record steps must be a list.")
    opal_steps = [
        _mapping(step, label="campaign step")
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("inputs"), dict) and "opal_config" in step["inputs"]
    ]
    if len(opal_steps) != 1:
        raise ValueError("Study campaign record must declare exactly one OPAL config step.")
    inputs = _mapping(opal_steps[0]["inputs"], label="campaign step inputs")
    config_path = resolve_repo_relative_path(
        repo_root=root,
        raw_path=str(inputs["opal_config"]),
        status_kind="stress-ethanol-cipro-growth-campaign-navigation",
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Current OPAL campaign config not found: {config_path}")
    config = load_config(config_path)
    if config.ownership.owner_scope != "study_campaign" or config.ownership.study_id != STUDY_ID:
        raise ValueError("Current OPAL campaign is not owned by this study.")
    if config.campaign.metadata.get("study_id") != STUDY_ID:
        raise ValueError("Current OPAL campaign metadata study ID does not match the study record.")

    workdir = Path(config.campaign.workdir).expanduser().resolve()
    try:
        workdir.relative_to(root)
    except ValueError as exc:
        raise ValueError("Current OPAL campaign workdir escapes the repository.") from exc
    notebook_path = workdir / "notebooks" / f"opal_{config.campaign.slug}_analysis.py"

    relative_config = config_path.relative_to(root)
    objective_names = tuple(dict.fromkeys(view.objective.name for view in config.selection_views))
    return CurrentCampaignNavigation(
        campaign_slug=config.campaign.slug,
        config_path=relative_config,
        notebook_path=notebook_path.relative_to(root),
        notebook_materialized=notebook_path.is_file(),
        selection_view_ids=tuple(view.id for view in config.selection_views),
        objective_names=objective_names,
        run_command=f"uv run opal notebook run -c {relative_config}",
    )


def discover_current_campaign_navigation(start: str | Path) -> CurrentCampaignNavigation | None:
    """Return live source-tree navigation, or None when no checkout is available."""

    start_path = Path(start).expanduser().resolve()
    cursor = start_path if start_path.is_dir() else start_path.parent
    for candidate in (cursor, *cursor.parents):
        if (candidate / CAMPAIGN_RECORD_PATH).is_file():
            return load_current_campaign_navigation(candidate)
    return None


def _resolve_inside_repo(root: Path, value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository.") from exc
    return path


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping.")
    return {str(key): item for key, item in value.items()}


__all__ = [
    "CAMPAIGN_RECORD_PATH",
    "CurrentCampaignNavigation",
    "discover_current_campaign_navigation",
    "load_current_campaign_navigation",
]
