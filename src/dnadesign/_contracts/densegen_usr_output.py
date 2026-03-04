"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_contracts/densegen_usr_output.py

Shared DenseGen output->USR contract parsing used by ops and notify integrations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DensegenUSROutputContract:
    config_path: Path
    run_root: Path
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


def _resolve_path_from_config(config_path: Path, value: object, *, label: str) -> Path:
    text = _required_non_empty_string(value, label=label)
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


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


def resolve_densegen_usr_output_contract(
    config_path: Path,
    *,
    root: dict[str, object] | None = None,
) -> DensegenUSROutputContract:
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
    outputs_root = (run_root / "outputs").resolve()
    if not _is_relative_to(usr_root, outputs_root):
        raise ValueError(
            "output.usr.root must be within outputs/ under "
            f"densegen.run.root ({outputs_root}), got: {usr_root}"
        )

    dataset_raw = _required_non_empty_string(usr_cfg.get("dataset"), label="densegen.output.usr.dataset")
    dataset_path = Path(dataset_raw.replace("\\", "/"))
    if dataset_path.is_absolute():
        raise ValueError("densegen.output.usr.dataset must be a relative path")
    if any(part in {".", ".."} for part in dataset_path.parts):
        raise ValueError("densegen.output.usr.dataset must not contain '.' or '..'")
    usr_dataset = Path(*dataset_path.parts).as_posix()

    return DensegenUSROutputContract(
        config_path=resolved_config_path,
        run_root=run_root,
        usr_root=usr_root,
        usr_dataset=usr_dataset,
    )

