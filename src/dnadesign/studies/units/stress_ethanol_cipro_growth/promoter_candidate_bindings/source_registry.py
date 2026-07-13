"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/source_registry.py

Checked-in source registry for the study candidate-binding authority.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .contracts import STUDY_ID, PromoterCandidateBindingsError
from .values import required_text

REGISTRY_SCHEMA_ID = "dnadesign.study.promoter_candidate_binding_sources.v1"
REGISTRY_PATH = Path("docs/studies/stress_ethanol_cipro_growth/record/promoter_candidate_binding_sources.yaml")


@dataclass(frozen=True)
class CandidateTableSource:
    dataset_id: str
    records_path: Path


@dataclass(frozen=True)
class AliasSource:
    source_id: str
    adapter: str
    config: dict[str, Any]


@dataclass(frozen=True)
class BindingSourceRegistry:
    path: Path
    candidate_table: CandidateTableSource
    alias_sources: tuple[AliasSource, ...]


def load_source_registry(repo_root: Path) -> BindingSourceRegistry:
    root = Path(repo_root).expanduser().resolve()
    registry_path = root / REGISTRY_PATH
    if not registry_path.is_file():
        raise PromoterCandidateBindingsError(f"Promoter candidate-binding source registry not found: {registry_path}")
    try:
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromoterCandidateBindingsError(f"Could not parse promoter binding source registry: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_id",
        "schema_version",
        "study_id",
        "candidate_table",
        "alias_sources",
    }:
        raise PromoterCandidateBindingsError("Promoter binding source registry fields do not match v1.")
    if (
        payload["schema_id"] != REGISTRY_SCHEMA_ID
        or str(payload["schema_version"]) != "1"
        or payload["study_id"] != STUDY_ID
    ):
        raise PromoterCandidateBindingsError("Promoter binding source registry identity mismatch.")
    candidate = _mapping(payload["candidate_table"], context="candidate_table")
    if set(candidate) != {"dataset_id", "records_path"}:
        raise PromoterCandidateBindingsError("candidate_table fields must be dataset_id and records_path.")
    sources = _alias_sources(payload["alias_sources"])
    return BindingSourceRegistry(
        path=REGISTRY_PATH,
        candidate_table=CandidateTableSource(
            dataset_id=required_text(candidate["dataset_id"], field="candidate table dataset ID"),
            records_path=_relative_path(candidate["records_path"], context="candidate_table.records_path"),
        ),
        alias_sources=sources,
    )


def _alias_sources(value: object) -> tuple[AliasSource, ...]:
    if not isinstance(value, list) or not value:
        raise PromoterCandidateBindingsError("alias_sources must be a non-empty list.")
    sources: list[AliasSource] = []
    for index, raw in enumerate(value):
        item = _mapping(raw, context=f"alias_sources[{index}]")
        if set(item) != {"source_id", "adapter", "config"}:
            raise PromoterCandidateBindingsError(
                f"alias_sources[{index}] fields must be source_id, adapter, and config."
            )
        config = _mapping(item["config"], context=f"alias_sources[{index}].config")
        sources.append(
            AliasSource(
                source_id=required_text(item["source_id"], field="alias source ID"),
                adapter=required_text(item["adapter"], field="alias source adapter"),
                config=config,
            )
        )
    ids = [source.source_id for source in sources]
    if len(ids) != len(set(ids)):
        raise PromoterCandidateBindingsError("Promoter binding alias source IDs must be unique.")
    return tuple(sources)


def relative_config_path(value: object, *, context: str) -> Path:
    return _relative_path(value, context=context)


def _relative_path(value: object, *, context: str) -> Path:
    text = required_text(value, field=context)
    path = PurePosixPath(text)
    first = path.parts[0] if path.parts else ""
    if "\\" in text or text.startswith("~") or path.is_absolute() or ".." in path.parts or ":" in first:
        raise PromoterCandidateBindingsError(f"{context} must be a confined relative POSIX path.")
    return Path(path)


def _mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromoterCandidateBindingsError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


__all__ = [
    "AliasSource",
    "BindingSourceRegistry",
    "CandidateTableSource",
    "REGISTRY_PATH",
    "REGISTRY_SCHEMA_ID",
    "load_source_registry",
    "relative_config_path",
]
