"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/contracts.py

Public DenseGen contracts for USR destination and producer-shape resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def _required_mapping(raw: object, *, label: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def _required_non_empty_string(raw: object, *, label: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _resolve_path_from_config(config_path: Path, value: object, *, label: str) -> Path:
    text = _required_non_empty_string(value, label=label)
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def _normalize_relative_dataset_path(dataset_value: object, *, label: str) -> str:
    dataset_raw = _required_non_empty_string(dataset_value, label=label)
    dataset_path = Path(dataset_raw.replace("\\", "/"))
    if dataset_path.is_absolute():
        raise ValueError(f"{label} must be a relative path")
    if any(part in {".", ".."} for part in dataset_path.parts):
        raise ValueError(f"{label} must not contain '.' or '..'")
    return Path(*dataset_path.parts).as_posix()


def load_densegen_config_mapping(config_path: Path) -> tuple[Path, dict[str, object]]:
    resolved_config_path = config_path.expanduser().resolve()
    if not resolved_config_path.exists():
        raise ValueError(f"DenseGen config not found: {resolved_config_path}")
    if not resolved_config_path.is_file():
        raise ValueError(f"DenseGen config is not a file: {resolved_config_path}")
    try:
        raw = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"failed to parse DenseGen config '{resolved_config_path}': {exc}") from exc
    root = _required_mapping(raw, label="DenseGen config")
    return resolved_config_path, root


@dataclass(frozen=True)
class DensegenUSROutputContract:
    config_path: Path
    run_root: Path
    usr_root: Path
    usr_dataset: str


@dataclass(frozen=True)
class DensegenUSRProducerContract:
    config_path: Path
    run_root: Path
    usr_root: Path
    usr_dataset: str
    usr_chunk_size: int
    records_path: Path | None
    parquet_chunk_size: int | None
    round_robin: bool
    max_accepted_per_library: int
    generation_total_quota: int
    supports_overlay_parts: bool
    supports_records_parts: bool


def _densegen_usr_destination(
    config_path: Path,
    *,
    root: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object], dict[str, object], Path, Path]:
    if root is None:
        resolved_config_path, root_mapping = load_densegen_config_mapping(config_path)
    else:
        resolved_config_path = config_path.expanduser().resolve()
        root_mapping = root
    densegen_cfg = _required_mapping(root_mapping.get("densegen"), label="densegen")
    run_cfg = _required_mapping(densegen_cfg.get("run"), label="densegen.run")
    output_cfg = _required_mapping(densegen_cfg.get("output"), label="densegen.output")

    run_root = _resolve_path_from_config(resolved_config_path, run_cfg.get("root"), label="densegen.run.root")
    if run_root.exists() and not run_root.is_dir():
        raise ValueError(f"densegen.run.root must be a directory: {run_root}")
    if not run_root.exists():
        raise ValueError(f"densegen.run.root does not exist: {run_root}")

    targets = output_cfg.get("targets")
    if not isinstance(targets, list):
        raise ValueError("densegen.output.targets must be a list")
    targets_set = {str(item).strip() for item in targets}
    usr_cfg_raw = output_cfg.get("usr")
    if "usr" not in targets_set or not isinstance(usr_cfg_raw, dict):
        raise ValueError("densegen.output.targets must include 'usr' with densegen.output.usr configured")

    usr_cfg = _required_mapping(usr_cfg_raw, label="densegen.output.usr")
    usr_root = _resolve_path_from_config(resolved_config_path, usr_cfg.get("root"), label="densegen.output.usr.root")
    return resolved_config_path, densegen_cfg, output_cfg, run_root, usr_root


def resolve_densegen_usr_output_contract(config_path: Path) -> DensegenUSROutputContract:
    resolved_config_path, _densegen_cfg, output_cfg, run_root, usr_root = _densegen_usr_destination(config_path)
    usr_cfg = _required_mapping(output_cfg.get("usr"), label="densegen.output.usr")
    return DensegenUSROutputContract(
        config_path=resolved_config_path,
        run_root=run_root,
        usr_root=usr_root,
        usr_dataset=_normalize_relative_dataset_path(
            usr_cfg.get("dataset"),
            label="densegen.output.usr.dataset",
        ),
    )


def resolve_densegen_usr_producer_contract(config_path: Path) -> DensegenUSRProducerContract:
    resolved_config_path, root = load_densegen_config_mapping(config_path)
    destination = resolve_densegen_usr_output_contract(resolved_config_path)
    densegen_cfg = _required_mapping(root.get("densegen"), label="densegen")
    runtime_cfg = _required_mapping(densegen_cfg.get("runtime"), label="densegen.runtime")
    generation_cfg = _required_mapping(densegen_cfg.get("generation"), label="densegen.generation")
    plan = generation_cfg.get("plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("densegen.generation.plan must be a non-empty list")
    output_cfg = _required_mapping(densegen_cfg.get("output"), label="densegen.output")
    usr_cfg = _required_mapping(output_cfg.get("usr"), label="densegen.output.usr")
    targets = output_cfg.get("targets")
    if not isinstance(targets, list):
        raise ValueError("densegen.output.targets must be a list")
    targets_set = {str(item).strip() for item in targets}

    usr_chunk_size = int(usr_cfg.get("chunk_size", 1))
    if usr_chunk_size <= 0:
        raise ValueError("densegen.output.usr.chunk_size must be > 0")

    records_path: Path | None = None
    parquet_chunk_size: int | None = None
    supports_records_parts = False
    if "parquet" in targets_set:
        parquet_cfg = _required_mapping(output_cfg.get("parquet"), label="densegen.output.parquet")
        records_path = _resolve_path_from_config(
            resolved_config_path,
            parquet_cfg.get("path"),
            label="densegen.output.parquet.path",
        )
        parquet_chunk_size = int(parquet_cfg.get("chunk_size", 2048))
        if parquet_chunk_size <= 0:
            raise ValueError("densegen.output.parquet.chunk_size must be > 0")
        supports_records_parts = True

    max_accepted_per_library = int(runtime_cfg.get("max_accepted_per_library", 1))
    if max_accepted_per_library <= 0:
        raise ValueError("densegen.runtime.max_accepted_per_library must be > 0")
    round_robin = bool(runtime_cfg.get("round_robin", False))

    generation_total_quota = 0
    for idx, entry in enumerate(plan):
        if not isinstance(entry, dict):
            raise ValueError(f"densegen.generation.plan[{idx}] must be a mapping")
        try:
            sequences = int(entry.get("sequences", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"densegen.generation.plan[{idx}].sequences must be an integer") from exc
        if sequences < 0:
            raise ValueError(f"densegen.generation.plan[{idx}].sequences must be >= 0")
        generation_total_quota += sequences
    if generation_total_quota <= 0:
        raise ValueError("densegen.generation.plan total quota must be > 0")

    return DensegenUSRProducerContract(
        config_path=resolved_config_path,
        run_root=destination.run_root,
        usr_root=destination.usr_root,
        usr_dataset=destination.usr_dataset,
        usr_chunk_size=usr_chunk_size,
        records_path=records_path,
        parquet_chunk_size=parquet_chunk_size,
        round_robin=round_robin,
        max_accepted_per_library=max_accepted_per_library,
        generation_total_quota=generation_total_quota,
        supports_overlay_parts=True,
        supports_records_parts=supports_records_parts,
    )


__all__ = [
    "DensegenUSROutputContract",
    "DensegenUSRProducerContract",
    "load_densegen_config_mapping",
    "resolve_densegen_usr_output_contract",
    "resolve_densegen_usr_producer_contract",
]
