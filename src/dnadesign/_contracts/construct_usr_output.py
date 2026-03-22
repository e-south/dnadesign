"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_contracts/construct_usr_output.py

Shared construct output->USR contract parsing used by notify and ops
integrations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.usr_roots import resolve_usr_root_from_config


@dataclass(frozen=True)
class ConstructUSROutputContract:
    config_path: Path
    usr_root: Path
    usr_dataset: str


def _required_mapping(raw: object, *, label: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


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


def load_construct_config_mapping(config_path: Path) -> tuple[Path, dict[str, object]]:
    resolved_config_path = config_path.expanduser().resolve()
    if not resolved_config_path.exists():
        raise ValueError(f"construct config not found: {resolved_config_path}")
    if not resolved_config_path.is_file():
        raise ValueError(f"construct config is not a file: {resolved_config_path}")
    try:
        raw = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"failed to parse construct config '{resolved_config_path}': {exc}") from exc
    root = _required_mapping(raw, label="construct config")
    return resolved_config_path, root


def resolve_construct_usr_output_contract(
    config_path: Path,
    *,
    root: dict[str, object] | None = None,
) -> ConstructUSROutputContract:
    if root is None:
        resolved_config_path, root_mapping = load_construct_config_mapping(config_path)
    else:
        resolved_config_path = config_path.expanduser().resolve()
        root_mapping = root

    job_cfg = _required_mapping(root_mapping.get("job"), label="job")
    input_cfg = _required_mapping(job_cfg.get("input"), label="job.input")
    output_cfg = _required_mapping(job_cfg.get("output"), label="job.output")

    usr_dataset = _normalize_relative_dataset_path(output_cfg.get("dataset"), label="job.output.dataset")
    root_value = output_cfg.get("root", input_cfg.get("root"))
    usr_root = resolve_usr_root_from_config(
        root_value,
        config_path=resolved_config_path,
        label="job.output.root or job.input.root",
    )
    if usr_root is None:
        raise ValueError("construct resolver requires job.input.root or job.output.root")

    return ConstructUSROutputContract(
        config_path=resolved_config_path,
        usr_root=usr_root,
        usr_dataset=usr_dataset,
    )
