"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_contracts/usr_producer.py

Shared USR producer-contract parsing for tool configs used by ops and notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .densegen_usr_output import load_densegen_config_mapping, resolve_densegen_usr_output_contract


@dataclass(frozen=True)
class InferUSROutputContract:
    config_path: Path
    usr_root: Path
    usr_dataset: str


@dataclass(frozen=True)
class USRProducerContract:
    tool: str
    config_path: Path
    run_root: Path | None
    usr_root: Path
    usr_dataset: str
    supports_overlay_parts: bool
    supports_records_parts: bool
    usr_chunk_size: int | None
    records_path: Path | None
    parquet_chunk_size: int | None
    round_robin: bool | None
    max_accepted_per_library: int | None
    generation_total_quota: int | None


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


def _resolve_densegen_usr_producer_contract(config_path: Path) -> USRProducerContract:
    resolved_config_path, root = load_densegen_config_mapping(config_path)
    destination = resolve_densegen_usr_output_contract(resolved_config_path, root=root)
    densegen = _required_mapping(root.get("densegen"), label="densegen")
    runtime = _required_mapping(densegen.get("runtime"), label="densegen.runtime")
    generation = _required_mapping(densegen.get("generation"), label="densegen.generation")
    plan = generation.get("plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("densegen.generation.plan must be a non-empty list")
    output = _required_mapping(densegen.get("output"), label="densegen.output")
    usr = _required_mapping(output.get("usr"), label="densegen.output.usr")
    targets = output.get("targets")
    if not isinstance(targets, list):
        raise ValueError("densegen.output.targets must be a list")
    targets_set = {str(item).strip() for item in targets}

    usr_chunk_size = int(usr.get("chunk_size", 1))
    if usr_chunk_size <= 0:
        raise ValueError("densegen.output.usr.chunk_size must be > 0")

    records_path: Path | None = None
    parquet_chunk_size: int | None = None
    supports_records_parts = False
    if "parquet" in targets_set:
        parquet_cfg = _required_mapping(output.get("parquet"), label="densegen.output.parquet")
        records_path = _resolve_path_from_config(
            resolved_config_path,
            parquet_cfg.get("path"),
            label="densegen.output.parquet.path",
        )
        parquet_chunk_size = int(parquet_cfg.get("chunk_size", 2048))
        if parquet_chunk_size <= 0:
            raise ValueError("densegen.output.parquet.chunk_size must be > 0")
        supports_records_parts = True

    max_accepted_per_library = int(runtime.get("max_accepted_per_library", 1))
    if max_accepted_per_library <= 0:
        raise ValueError("densegen.runtime.max_accepted_per_library must be > 0")
    round_robin = bool(runtime.get("round_robin", False))

    total_quota = 0
    for idx, entry in enumerate(plan):
        if not isinstance(entry, dict):
            raise ValueError(f"densegen.generation.plan[{idx}] must be a mapping")
        try:
            sequences = int(entry.get("sequences", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"densegen.generation.plan[{idx}].sequences must be an integer") from exc
        if sequences < 0:
            raise ValueError(f"densegen.generation.plan[{idx}].sequences must be >= 0")
        total_quota += sequences
    if total_quota <= 0:
        raise ValueError("densegen.generation.plan total quota must be > 0")

    return USRProducerContract(
        tool="densegen",
        config_path=resolved_config_path,
        run_root=destination.run_root,
        usr_root=destination.usr_root,
        usr_dataset=destination.usr_dataset,
        supports_overlay_parts=True,
        supports_records_parts=supports_records_parts,
        usr_chunk_size=usr_chunk_size,
        records_path=records_path,
        parquet_chunk_size=parquet_chunk_size,
        round_robin=round_robin,
        max_accepted_per_library=max_accepted_per_library,
        generation_total_quota=total_quota,
    )


def _load_infer_config_mapping(config_path: Path) -> tuple[Path, dict[str, object]]:
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


def _infer_usr_root_from_env() -> Path | None:
    env = str(os.environ.get("DNADESIGN_USR_ROOT", "")).strip()
    if not env:
        return None
    return Path(env).expanduser().resolve()


def resolve_infer_usr_output_contract(config_path: Path) -> InferUSROutputContract:
    resolved_config_path, root = _load_infer_config_mapping(config_path)
    jobs = root.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"infer config must include a jobs list: {resolved_config_path}")

    destinations: set[tuple[Path, str]] = set()
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
            usr_root = _infer_usr_root_from_env()
            if usr_root is None:
                raise ValueError("infer resolver requires ingest.root or DNADESIGN_USR_ROOT for source='usr' jobs")
        else:
            root_text = str(root_value).strip()
            if not root_text:
                raise ValueError("infer resolver received empty ingest.root")
            candidate = Path(root_text).expanduser()
            if candidate.is_absolute():
                usr_root = candidate.resolve()
            else:
                usr_root = (resolved_config_path.parent / candidate).resolve()
        destinations.add((usr_root, dataset))

    if not destinations:
        raise ValueError(
            "infer resolver requires at least one job with ingest.source='usr' and io.write_back=true"
        )
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


def _resolve_infer_usr_producer_contract(config_path: Path) -> USRProducerContract:
    destination = resolve_infer_usr_output_contract(config_path)
    return USRProducerContract(
        tool="infer",
        config_path=destination.config_path,
        run_root=None,
        usr_root=destination.usr_root,
        usr_dataset=destination.usr_dataset,
        supports_overlay_parts=False,
        supports_records_parts=False,
        usr_chunk_size=None,
        records_path=None,
        parquet_chunk_size=None,
        round_robin=None,
        max_accepted_per_library=None,
        generation_total_quota=None,
    )


_USR_PRODUCER_ADAPTERS: dict[str, Callable[[Path], USRProducerContract]] = {
    "densegen": _resolve_densegen_usr_producer_contract,
    "infer": _resolve_infer_usr_producer_contract,
}


def resolve_usr_producer_contract(*, tool: str, config_path: Path) -> USRProducerContract:
    tool_name = str(tool or "").strip().lower()
    adapter = _USR_PRODUCER_ADAPTERS.get(tool_name)
    if adapter is None:
        supported = ", ".join(sorted(_USR_PRODUCER_ADAPTERS))
        raise ValueError(f"unsupported usr producer tool: {tool_name} (supported: {supported})")
    return adapter(config_path)
