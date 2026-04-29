"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/contracts.py

Public infer contracts for USR write-back destination resolution and default
notify profile path derivation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.usr import resolve_usr_root_from_config

from .src.config import RootConfig


@dataclass(frozen=True)
class InferUSROutputContract:
    config_path: Path
    usr_root: Path
    usr_dataset: str


@dataclass(frozen=True)
class InferUSREventsContract:
    config_path: Path
    usr_root: Path
    usr_dataset: str
    source_kind: str


@dataclass(frozen=True)
class InferConfigValidationContract:
    config_path: Path
    model_id: str
    device: str
    job_ids: tuple[str, ...]
    usr_datasets: tuple[str, ...]


@dataclass(frozen=True)
class InferRuntimeLaneContract:
    config_label: str
    config_path: Path
    runtime_label: str
    lane_kind: str
    model_family: str
    family_suffix: str
    phase_id: str


_INFER_LANE_CONFIG_RE = re.compile(r"^config\.(?P<lane>.+)\.evo2_(?P<family>\d+b)$")
_INFER_SINGLE_STREAM_LANE_KINDS = frozenset({"anchor_only", "anchor_plus_template"})


def _required_non_empty_string(raw: object, *, label: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _normalize_relative_dataset_path(dataset_value: object, *, label: str) -> str:
    dataset_raw = _required_non_empty_string(dataset_value, label=label)
    dataset_path = Path(dataset_raw.replace("\\", "/"))
    if dataset_path.is_absolute():
        raise ValueError(f"{label} must be a relative path")
    if any(part in {".", ".."} for part in dataset_path.parts):
        raise ValueError(f"{label} must not contain '.' or '..'")
    return Path(*dataset_path.parts).as_posix()


def _load_infer_root_config(config_path: Path) -> tuple[Path, dict[str, object]]:
    resolved_config_path = config_path.expanduser().resolve()
    if not resolved_config_path.exists():
        raise ValueError(f"tool config not found: {resolved_config_path}")
    if not resolved_config_path.is_file():
        raise ValueError(f"tool config is not a file: {resolved_config_path}")
    try:
        raw = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"failed to parse infer config '{resolved_config_path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"infer config must be a YAML mapping at top-level: {resolved_config_path}")
    return resolved_config_path, raw


def _load_validated_infer_config(config_path: Path) -> tuple[Path, RootConfig]:
    resolved_config_path, raw = _load_infer_root_config(config_path)
    try:
        root = RootConfig(**raw)
    except Exception as exc:
        raise ValueError(f"failed to validate infer config '{resolved_config_path}': {exc}") from exc
    return resolved_config_path, root


def _optional_non_empty_string(raw: object) -> str | None:
    text = str(raw or "").strip()
    return text or None


def _build_infer_config_validation_contract(
    *,
    config_path: Path,
    root: RootConfig,
) -> InferConfigValidationContract:
    usr_datasets = tuple(
        str(job.ingest.dataset)
        for job in root.jobs
        if job.ingest.source == "usr" and str(job.ingest.dataset or "").strip()
    )
    return InferConfigValidationContract(
        config_path=config_path,
        model_id=str(root.model.id),
        device=str(root.model.device),
        job_ids=tuple(str(job.id) for job in root.jobs),
        usr_datasets=usr_datasets,
    )


def infer_model_family_suffix(model_family: str | None) -> str | None:
    family = _optional_non_empty_string(model_family)
    if family is None:
        return None
    prefix, _, suffix = family.rpartition("_")
    return suffix if prefix else family


def _infer_notify_profile_namespace_for_config_name(config_name: str) -> str | None:
    stem = Path(config_name).stem
    match = _INFER_LANE_CONFIG_RE.fullmatch(stem)
    if match is None:
        return None
    lane = str(match.group("lane")).strip().replace(".", "_")
    family = str(match.group("family")).strip().lower()
    if not lane or not family:
        return None
    return f"{lane}_{family}"


def _infer_runtime_lane_contract_from_config(
    *,
    config_label: str,
    config_path: Path,
) -> InferRuntimeLaneContract | None:
    stem = Path(config_path.name).stem
    match = _INFER_LANE_CONFIG_RE.fullmatch(stem)
    if match is None:
        return None
    lane_kind = str(match.group("lane")).strip().replace(".", "_")
    if lane_kind not in _INFER_SINGLE_STREAM_LANE_KINDS:
        return None
    family_suffix = str(match.group("family")).strip().lower()
    if not family_suffix:
        return None
    runtime_label = f"{lane_kind}_{family_suffix}"
    return InferRuntimeLaneContract(
        config_label=str(config_label).strip(),
        config_path=config_path.expanduser().resolve(),
        runtime_label=runtime_label,
        lane_kind=lane_kind,
        model_family=f"evo2_{family_suffix}",
        family_suffix=family_suffix,
        phase_id=f"infer_{runtime_label}",
    )


def resolve_infer_runtime_lane_contracts(
    infer_config_paths: Mapping[str, Path],
    *,
    preferred_model_family: str | None = None,
) -> tuple[InferRuntimeLaneContract, ...]:
    contracts: list[InferRuntimeLaneContract] = []
    for config_label, config_path in infer_config_paths.items():
        contract = _infer_runtime_lane_contract_from_config(
            config_label=str(config_label),
            config_path=Path(config_path),
        )
        if contract is not None:
            contracts.append(contract)
    preferred_suffix = infer_model_family_suffix(preferred_model_family)
    if preferred_suffix is None:
        return tuple(contracts)
    preferred = [contract for contract in contracts if contract.family_suffix == preferred_suffix]
    deferred = [contract for contract in contracts if contract.family_suffix != preferred_suffix]
    return tuple(preferred + deferred)


def resolve_infer_usr_output_contract(config_path: Path) -> InferUSROutputContract:
    resolved_config_path, root = _load_infer_root_config(config_path)

    destinations: set[tuple[Path, str]] = set()
    jobs = root.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"infer config must include a jobs list: {resolved_config_path}")

    for job in jobs:
        if not isinstance(job, dict):
            continue
        ingest = job.get("ingest")
        if not isinstance(ingest, dict):
            continue
        source = str(ingest.get("source") or "").strip().lower()
        if source != "usr":
            continue
        io_cfg = job.get("io")
        io = io_cfg if isinstance(io_cfg, dict) else {}
        if not bool(io.get("write_back")):
            continue
        dataset = _normalize_relative_dataset_path(
            ingest.get("dataset"),
            label="infer resolver requires ingest.dataset for source='usr' jobs",
        )
        root_value = ingest.get("root")
        if root_value is None:
            raise ValueError("infer resolver requires ingest.root for source='usr' write-back jobs")
        usr_root = resolve_usr_root_from_config(
            root_value,
            config_path=resolved_config_path,
            label="infer resolver ingest.root",
        )
        if usr_root is None:
            raise ValueError("infer resolver requires ingest.root for source='usr' write-back jobs")
        destinations.add((usr_root, dataset))

    if not destinations:
        raise ValueError("infer resolver requires at least one job with ingest.source='usr' and io.write_back=true")
    if len(destinations) > 1:
        rendered = ", ".join(sorted(f"{root_path}/{dataset}" for root_path, dataset in destinations))
        raise ValueError(
            f"infer resolver found multiple USR destinations in config: {rendered}. "
            "Pass --events explicitly to select one stream."
        )
    usr_root, dataset = next(iter(destinations))
    return InferUSROutputContract(
        config_path=resolved_config_path,
        usr_root=usr_root,
        usr_dataset=dataset,
    )


def resolve_infer_usr_events_contract(config_path: Path) -> InferUSREventsContract:
    resolved_config_path, root = _load_infer_root_config(config_path)

    try:
        output = resolve_infer_usr_output_contract(resolved_config_path)
    except ValueError as exc:
        output_error = str(exc)
    else:
        return InferUSREventsContract(
            config_path=output.config_path,
            usr_root=output.usr_root,
            usr_dataset=output.usr_dataset,
            source_kind="usr_writeback",
        )

    destinations = _sequence_view_event_destinations(root, config_path=resolved_config_path)
    if not destinations:
        raise ValueError(
            "infer resolver requires at least one job with ingest.source='usr' and io.write_back=true "
            f"or exactly one feature_bundle.sequence_view_inputs dataset; write-back resolver error: {output_error}"
        )
    if len(destinations) > 1:
        rendered = ", ".join(sorted(f"{root_path}/{dataset}" for root_path, dataset in destinations))
        raise ValueError(
            f"infer resolver found multiple sequence-view event sources in config: {rendered}. "
            "Split the batch into single-dataset sequence-view runbooks or pass --events explicitly."
        )
    usr_root, dataset = next(iter(destinations))
    return InferUSREventsContract(
        config_path=resolved_config_path,
        usr_root=usr_root,
        usr_dataset=dataset,
        source_kind="sequence_view_input",
    )


def _sequence_view_event_destinations(
    root: Mapping[str, object],
    *,
    config_path: Path,
) -> set[tuple[Path, str]]:
    jobs = root.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"infer config must include a jobs list: {config_path}")
    destinations: set[tuple[Path, str]] = set()
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        feature_bundle = job.get("feature_bundle")
        if not isinstance(feature_bundle, Mapping):
            continue
        inputs = feature_bundle.get("sequence_view_inputs")
        if not isinstance(inputs, list):
            continue
        for input_payload in inputs:
            if not isinstance(input_payload, Mapping):
                continue
            dataset = _normalize_relative_dataset_path(
                input_payload.get("dataset"),
                label="infer resolver requires sequence_view_inputs.dataset",
            )
            root_value = input_payload.get("root")
            if root_value is None:
                raise ValueError("infer resolver requires sequence_view_inputs.root for sequence-view jobs")
            usr_root = resolve_usr_root_from_config(
                root_value,
                config_path=config_path,
                label="infer resolver sequence_view_inputs.root",
            )
            if usr_root is None:
                raise ValueError("infer resolver requires sequence_view_inputs.root for sequence-view jobs")
            destinations.add((usr_root, dataset))
    return destinations


def validate_infer_config_contract(config_path: Path) -> InferConfigValidationContract:
    from .src.runtime.adapter_runtime import validate_adapter_runtime_contract

    resolved_config_path, root = _load_validated_infer_config(config_path)
    validate_adapter_runtime_contract(model=root.model)
    return _build_infer_config_validation_contract(config_path=resolved_config_path, root=root)


def validate_infer_dry_run_contract(config_path: Path) -> InferConfigValidationContract:
    from .src.cli.config_inputs import resolve_config_job_inputs
    from .src.ingest.sources import preflight_usr_input
    from .src.runtime.adapter_runtime import validate_adapter_runtime_contract

    resolved_config_path, root = _load_validated_infer_config(config_path)
    validate_adapter_runtime_contract(model=root.model)
    for job in root.jobs:
        resolve_config_job_inputs(
            job=job,
            config_dir=resolved_config_path.parent,
            i_know_this_is_pickle=False,
            guard_pickle=lambda _ack: None,
        )
        if job.ingest.source != "usr":
            continue
        preflight_usr_input(
            dataset_name=str(job.ingest.dataset),
            field=str(job.ingest.field or "sequence"),
            root=job.ingest.root,
        )
    return _build_infer_config_validation_contract(config_path=resolved_config_path, root=root)


def plan_sequence_view_feature_completion_from_config(
    config_path: Path,
    *,
    job: str | None = None,
) -> tuple[dict[str, object], ...]:
    from .src.cli.config_inputs import resolve_config_sequence_view_roots
    from .src.features.completion_planner import plan_sequence_view_feature_completion
    from .src.features.sequence_views import bundle_uses_sequence_views

    resolved_config_path, root = _load_validated_infer_config(config_path)
    selected_jobs = [selected_job for selected_job in root.jobs if job in {None, str(selected_job.id)}]
    if not selected_jobs:
        raise ValueError("No jobs selected. Check the job id or the config file.")

    plans: list[dict[str, object]] = []
    for selected_job in selected_jobs:
        if selected_job.feature_bundle is None or not bundle_uses_sequence_views(selected_job.feature_bundle):
            continue
        resolve_config_sequence_view_roots(job=selected_job, config_dir=resolved_config_path.parent)
        command = f"uv run infer run --config {resolved_config_path} --job {selected_job.id}"
        plans.append(
            plan_sequence_view_feature_completion(
                bundle=selected_job.feature_bundle,
                model_id=root.model.id,
                job_id=selected_job.id,
                bundle_id=selected_job.id,
                infer_command=command,
            ).to_dict()
        )

    if not plans:
        raise ValueError("No selected jobs use feature_bundle.sequence_view_inputs.")
    return tuple(plans)


def plan_sequence_view_feature_inventory_completion_from_config(
    config_path: Path,
    *,
    job: str | None = None,
) -> tuple[dict[str, object], ...]:
    from .src.cli.config_inputs import resolve_config_sequence_view_roots
    from .src.features.completion_planner import plan_sequence_view_feature_inventory_completion
    from .src.features.sequence_views import bundle_uses_sequence_views

    resolved_config_path, root = _load_validated_infer_config(config_path)
    selected_jobs = [selected_job for selected_job in root.jobs if job in {None, str(selected_job.id)}]
    if not selected_jobs:
        raise ValueError("No jobs selected. Check the job id or the config file.")

    plans: list[dict[str, object]] = []
    for selected_job in selected_jobs:
        if selected_job.feature_bundle is None or not bundle_uses_sequence_views(selected_job.feature_bundle):
            continue
        resolve_config_sequence_view_roots(job=selected_job, config_dir=resolved_config_path.parent)
        command = f"uv run infer run --config {resolved_config_path} --job {selected_job.id}"
        plans.append(
            plan_sequence_view_feature_inventory_completion(
                bundle=selected_job.feature_bundle,
                model_id=root.model.id,
                job_id=selected_job.id,
                bundle_id=selected_job.id,
                infer_command=command,
            ).to_dict()
        )

    if not plans:
        raise ValueError("No selected jobs use feature_bundle.sequence_view_inputs.")
    return tuple(plans)


def resolve_infer_notify_profile_path(config_path: Path) -> Path:
    resolved_config_path, _root = _load_infer_root_config(config_path)
    workspace_root = resolved_config_path.parent
    namespace = _infer_notify_profile_namespace_for_config_name(resolved_config_path.name)
    if namespace is None:
        return (workspace_root / "outputs" / "notify" / "infer" / "profile.json").resolve()
    return (workspace_root / "outputs" / "notify" / "infer" / namespace / "profile.json").resolve()


__all__ = [
    "InferConfigValidationContract",
    "InferRuntimeLaneContract",
    "InferUSREventsContract",
    "InferUSROutputContract",
    "infer_model_family_suffix",
    "plan_sequence_view_feature_completion_from_config",
    "plan_sequence_view_feature_inventory_completion_from_config",
    "resolve_infer_notify_profile_path",
    "resolve_infer_runtime_lane_contracts",
    "resolve_infer_usr_events_contract",
    "resolve_infer_usr_output_contract",
    "validate_infer_config_contract",
    "validate_infer_dry_run_contract",
]
