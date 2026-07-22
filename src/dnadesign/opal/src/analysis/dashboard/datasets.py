"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/dashboard/datasets.py

Dashboard helpers for discovering campaigns and datasets. Resolves campaign.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import polars as pl

from ...config.loader import load_config
from ...config.types import LocationUSR, RootConfig
from ...core.config_resolve import resolve_campaign_root
from .diagnostics import Diagnostics


@dataclass(frozen=True)
class CampaignSelectionViewInfo:
    id: str
    objective_name: str
    objective_params: dict
    selection_name: str
    selection_params: dict


@dataclass(frozen=True)
class CampaignInfo:
    label: str
    path: Path
    workdir: Path | None
    slug: str
    owner_scope: str
    study_id: str | None
    portable: bool
    x_column: str
    y_column: str
    y_expected_length: int | None
    model_name: str
    model_params: dict
    selection_views: tuple[CampaignSelectionViewInfo, ...]
    training_policy: dict
    y_ops: list[dict]


@dataclass(frozen=True)
class CampaignDatasetRef:
    campaign_label: str
    campaign_path: Path
    kind: str | None
    dataset_name: str | None
    records_path: Path | None


@dataclass(frozen=True)
class CampaignSelection:
    label: str | None
    path: Path | None
    info: CampaignInfo | None
    workdir: Path | None
    records_path: Path | None
    diagnostics: Diagnostics


@dataclass(frozen=True)
class RoundOptions:
    rounds: list[int]
    run_ids_by_round: dict[int, list[str]]
    source: str
    diagnostics: Diagnostics


def find_repo_root(start: Path) -> Path | None:
    start = Path(start).resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def resolve_usr_root(repo_root: Path | None, env_override: str | None) -> Path:
    if env_override:
        override_path = Path(env_override).expanduser().resolve()
        if not override_path.exists():
            raise ValueError(f"DNADESIGN_USR_ROOT does not exist: {override_path}")
        return override_path
    if repo_root is None:
        raise ValueError("Could not find repo root (pyproject.toml). Provide an absolute path.")
    return repo_root / "src" / "dnadesign" / "usr" / "datasets"


def list_usr_datasets(usr_root: Path) -> list[str]:
    if not usr_root.exists():
        return []
    datasets: list[str] = []
    for child in usr_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "records.parquet").is_file():
            datasets.append(child.name)
    return sorted(datasets)


def resolve_dataset_path(
    *,
    repo_root: Path | None,
    usr_root: Path | None,
    dataset_name: str | None,
    custom_path: str | None,
) -> tuple[Path, str]:
    custom = (custom_path or "").strip()
    if custom:
        custom_path_obj = Path(custom).expanduser()
        if custom_path_obj.is_absolute():
            return custom_path_obj, "custom"
        if repo_root is None:
            raise ValueError("Relative custom paths require a repo root.")
        return (repo_root / custom_path_obj).resolve(), "custom"
    if usr_root is None:
        raise ValueError("USR root is unavailable; provide a custom path.")
    if not dataset_name or dataset_name in {"(none found)", "(none)"}:
        raise ValueError("Select a dataset or provide a custom path.")
    return (usr_root / dataset_name / "records.parquet").resolve(), "usr"


def list_campaign_paths(repo_root: Path | None) -> list[Path]:
    if repo_root is None:
        return []
    campaigns_root = repo_root / "src" / "dnadesign" / "opal" / "campaigns"
    if not campaigns_root.exists():
        return []
    campaign_paths = sorted(campaigns_root.glob("*/configs/campaign.yaml"))
    slugs: set[str] = set()
    for campaign_path in campaign_paths:
        config = load_config(campaign_path)
        directory_slug = campaign_path.parents[1].name
        if directory_slug != config.campaign.slug:
            raise ValueError(
                "Checked-in campaign directory must match campaign.slug: "
                f"directory={directory_slug!r}, slug={config.campaign.slug!r}."
            )
        if config.campaign.slug in slugs:
            raise ValueError(f"Duplicate checked-in campaign slug: {config.campaign.slug!r}")
        slugs.add(config.campaign.slug)
    return campaign_paths


def list_campaign_dataset_refs(repo_root: Path | None) -> list[CampaignDatasetRef]:
    refs: list[CampaignDatasetRef] = []
    for campaign_path in list_campaign_paths(repo_root):
        campaign_label = campaign_label_from_path(campaign_path, repo_root)
        config = load_config(campaign_path)
        location = config.data.location
        if isinstance(location, LocationUSR):
            kind = "usr"
            dataset_name = location.dataset
            records_path = (Path(location.path) / location.dataset / "records.parquet").resolve()
        else:
            kind = "local"
            dataset_name = None
            records_path = Path(location.path).resolve()
        refs.append(
            CampaignDatasetRef(
                campaign_label=campaign_label,
                campaign_path=campaign_path,
                kind=kind,
                dataset_name=dataset_name,
                records_path=records_path,
            )
        )
    return refs


def load_parquet_cached(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise ValueError(f"Parquet path does not exist: {path}")
    stat = path.stat()
    return _read_parquet_cached(str(path), int(stat.st_mtime_ns))


@lru_cache(maxsize=4)
def _read_parquet_cached(path_str: str, mtime_ns: int) -> pl.DataFrame:
    _unused = mtime_ns
    return pl.read_parquet(path_str)


def resolve_campaign_records_path(config: RootConfig) -> Path:
    location = config.data.location
    if isinstance(location, LocationUSR):
        return (Path(location.path) / location.dataset / "records.parquet").resolve()
    return Path(location.path).resolve()


def campaign_label_from_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return str(resolve_campaign_root(path))
    root = resolve_campaign_root(path)
    try:
        return str(root.relative_to(repo_root))
    except ValueError:
        return str(root)


def campaign_info_from_config(*, config: RootConfig, path: Path, label: str) -> CampaignInfo:
    return CampaignInfo(
        label=label,
        path=path,
        workdir=Path(config.campaign.workdir),
        slug=config.campaign.slug,
        owner_scope=config.ownership.owner_scope,
        study_id=config.ownership.study_id,
        portable=config.ownership.portable,
        x_column=config.data.x_column_name,
        y_column=config.data.y_column_name,
        y_expected_length=config.data.y_expected_length,
        model_name=config.model.name,
        model_params=dict(config.model.params),
        selection_views=tuple(
            CampaignSelectionViewInfo(
                id=view.id,
                objective_name=view.objective.name,
                objective_params=dict(view.objective.params),
                selection_name=view.selection.name,
                selection_params=dict(view.selection.params),
            )
            for view in config.selection_views
        ),
        training_policy=dict(config.training.policy),
        y_ops=[{"name": item.name, "params": dict(item.params)} for item in config.training.y_ops],
    )


def load_campaign_selection(*, campaign_path: Path | None, repo_root: Path | None) -> CampaignSelection:
    diagnostics = Diagnostics()
    if campaign_path is None:
        return CampaignSelection(
            label=None,
            path=None,
            info=None,
            workdir=None,
            records_path=None,
            diagnostics=diagnostics.add_warning("Select a campaign to load records.parquet."),
        )
    label = campaign_label_from_path(campaign_path, repo_root)
    try:
        config = load_config(campaign_path)
    except Exception as exc:
        return CampaignSelection(
            label=label,
            path=campaign_path,
            info=None,
            workdir=None,
            records_path=None,
            diagnostics=diagnostics.add_error(f"Campaign config invalid: {exc}"),
        )
    info = campaign_info_from_config(config=config, path=campaign_path, label=label)
    workdir = Path(config.campaign.workdir)
    try:
        records_path = resolve_campaign_records_path(config)
    except Exception as exc:
        return CampaignSelection(
            label=label,
            path=campaign_path,
            info=info,
            workdir=workdir,
            records_path=None,
            diagnostics=diagnostics.add_error(f"Failed to resolve records.parquet: {exc}"),
        )
    if not records_path.exists():
        diagnostics = diagnostics.add_error(f"records.parquet not found: {records_path}")
    return CampaignSelection(
        label=label,
        path=campaign_path,
        info=info,
        workdir=workdir,
        records_path=records_path,
        diagnostics=diagnostics,
    )
