# ABOUTME: Dashboard helpers for discovering campaigns and datasets.
# ABOUTME: Resolves campaign roots and records paths for notebook UIs.
"""Dataset and campaign helpers for dashboard notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ...core.config_resolve import resolve_campaign_root
from .diagnostics import Diagnostics


@dataclass(frozen=True)
class CampaignInfo:
    label: str
    path: Path
    workdir: Path | None
    slug: str
    x_column: str
    y_column: str
    y_expected_length: int | None
    model_name: str
    model_params: dict
    objective_name: str
    objective_params: dict
    selection_name: str
    selection_params: dict
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
    return sorted(campaigns_root.rglob("campaign.yaml"))


def list_campaign_dataset_refs(repo_root: Path | None) -> list[CampaignDatasetRef]:
    refs: list[CampaignDatasetRef] = []
    for campaign_path in list_campaign_paths(repo_root):
        campaign_root = resolve_campaign_root(campaign_path)
        campaign_label = campaign_label_from_path(campaign_path, repo_root)
        try:
            raw = load_campaign_yaml(campaign_path)
        except Exception:
            continue
        data = raw.get("data") or {}
        location = data.get("location") or {}
        kind = str(location.get("kind")) if location.get("kind") is not None else None
        dataset_name = None
        records_path = None
        if kind == "usr":
            dataset_name = location.get("dataset")
            base_path_raw = location.get("path")
            base_path = Path(str(base_path_raw)) if base_path_raw else None
            if base_path is not None and not base_path.is_absolute():
                base_path = (campaign_root / base_path).resolve()
            if base_path is not None and dataset_name:
                records_path = (base_path / str(dataset_name) / "records.parquet").resolve()
        elif kind == "local":
            local_path_raw = location.get("path")
            if local_path_raw:
                local_path = Path(str(local_path_raw))
                if not local_path.is_absolute():
                    local_path = (campaign_root / local_path).resolve()
                records_path = local_path
        refs.append(
            CampaignDatasetRef(
                campaign_label=campaign_label,
                campaign_path=campaign_path,
                kind=kind,
                dataset_name=str(dataset_name) if dataset_name else None,
                records_path=records_path,
            )
        )
    return refs


def resolve_campaign_records_path(*, raw: dict, campaign_path: Path) -> Path:
    data = raw.get("data") or {}
    location = data.get("location") or {}
    kind = str(location.get("kind") or "").strip().lower()
    campaign_root = resolve_campaign_root(campaign_path)
    if kind == "usr":
        dataset_name = location.get("dataset")
        base_path_raw = location.get("path")
        if not dataset_name or not base_path_raw:
            raise ValueError("Campaign YAML missing data.location.dataset or data.location.path for usr.")
        base_path = Path(str(base_path_raw))
        if not base_path.is_absolute():
            base_path = (campaign_root / base_path).resolve()
        return (base_path / str(dataset_name) / "records.parquet").resolve()
    if kind == "local":
        local_path_raw = location.get("path")
        if not local_path_raw:
            raise ValueError("Campaign YAML missing data.location.path for local.")
        local_path = Path(str(local_path_raw))
        if not local_path.is_absolute():
            local_path = (campaign_root / local_path).resolve()
        return local_path
    raise ValueError(f"Unsupported data.location.kind: {kind!r}.")


def campaign_label_from_path(path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return str(resolve_campaign_root(path))
    root = resolve_campaign_root(path)
    try:
        return str(root.relative_to(repo_root))
    except ValueError:
        return str(root)


def load_campaign_yaml(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"Campaign config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Campaign YAML must be a mapping.")
    return raw


def parse_campaign_info(*, raw: dict, path: Path, label: str) -> CampaignInfo:
    campaign = raw.get("campaign") or {}
    slug = campaign.get("slug")
    if not slug:
        raise ValueError("Campaign YAML missing campaign.slug.")
    workdir = None
    workdir_raw = campaign.get("workdir")
    if workdir_raw:
        workdir_path = Path(str(workdir_raw))
        if not workdir_path.is_absolute():
            workdir_path = (resolve_campaign_root(path) / workdir_path).resolve()
        workdir = workdir_path
    data = raw.get("data") or {}
    x_column = data.get("x_column_name")
    y_column = data.get("y_column_name")
    if not x_column or not y_column:
        raise ValueError("Campaign YAML missing data.x_column_name or data.y_column_name.")
    y_expected_length = data.get("y_expected_length")

    model_block = raw.get("model") or {}
    model_name = model_block.get("name") or "random_forest"
    model_params = dict(model_block.get("params") or {})

    objective_block = raw.get("objective") or {}
    objective_name = objective_block.get("name") or "sfxi_v1"
    objective_params = dict(objective_block.get("params") or {})

    selection_block = raw.get("selection") or {}
    selection_name = selection_block.get("name") or "top_k"
    selection_params = dict(selection_block.get("params") or {})

    training_block = raw.get("training") or {}
    training_policy = dict(training_block.get("policy") or {})
    y_ops = list(training_block.get("y_ops") or [])

    return CampaignInfo(
        label=label,
        path=path,
        workdir=workdir,
        slug=str(slug),
        x_column=str(x_column),
        y_column=str(y_column),
        y_expected_length=int(y_expected_length) if y_expected_length is not None else None,
        model_name=str(model_name),
        model_params=model_params,
        objective_name=str(objective_name),
        objective_params=objective_params,
        selection_name=str(selection_name),
        selection_params=selection_params,
        training_policy=training_policy,
        y_ops=y_ops,
    )


def resolve_campaign_workdir(info: CampaignInfo) -> Path:
    if info.workdir is not None:
        return info.workdir
    return info.path.parent


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
        raw = load_campaign_yaml(campaign_path)
    except Exception as exc:
        return CampaignSelection(
            label=label,
            path=campaign_path,
            info=None,
            workdir=None,
            records_path=None,
            diagnostics=diagnostics.add_error(f"Failed to load campaign.yaml: {exc}"),
        )
    try:
        info = parse_campaign_info(raw=raw, path=campaign_path, label=label)
    except Exception as exc:
        return CampaignSelection(
            label=label,
            path=campaign_path,
            info=None,
            workdir=None,
            records_path=None,
            diagnostics=diagnostics.add_error(f"Campaign config invalid: {exc}"),
        )
    workdir = resolve_campaign_workdir(info)
    try:
        records_path = resolve_campaign_records_path(raw=raw, campaign_path=campaign_path)
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
