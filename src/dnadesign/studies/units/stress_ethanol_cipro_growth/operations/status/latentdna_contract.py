"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/operations/status/latentdna_contract.py

LatentDNA binding and snapshot helpers for the stress_ethanol_cipro_growth.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from .record_normalizer import StressEthanolCiproGrowthResolvedContext

DEFAULT_LATENTDNA_DOC = "src/dnadesign/latentdna/docs/workflows/stress-ethanol-cipro-representation-comparison.md"
_REQUIRED_BINDING_FIELDS = {
    "workspace_id",
    "workspace_ref",
    "snapshot_ref",
    "source_datasets",
    "supported_model_families",
    "default_model_family",
    "required_wildtype_references",
    "decision_deliverables",
}
_WORKSPACE_SNAPSHOT_SCHEMA_VERSION = "latentdna.workspace_snapshot.v1"
_REQUIRED_WORKSPACE_SNAPSHOT_FIELDS = {
    "schema_version",
    "workspace_id",
    "output_root",
    "sources",
    "model_families",
    "canonical_views",
    "deliverables",
    "exports",
    "browser",
    "decision_ladder",
    "last_updated_at",
}


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _string_or_none(item)
        if text is not None:
            result.append(text)
    return result


def latentdna_binding_path(study_context: StressEthanolCiproGrowthResolvedContext) -> Path | None:
    repo_root = study_context.study_repo_root
    if repo_root is None:
        return None
    latentdna_config = study_context.study_pipeline.get("latentdna")
    if isinstance(latentdna_config, Mapping):
        configured = _string_or_none(latentdna_config.get("binding"))
        if configured is not None:
            return (repo_root / configured).resolve()
    return (
        repo_root / "docs" / "studies" / study_context.study_id / "contexts" / "latentdna" / "binding.yaml"
    ).resolve()


def load_latentdna_binding(
    study_context: StressEthanolCiproGrowthResolvedContext,
) -> tuple[Path | None, dict[str, object] | None]:
    binding_path = latentdna_binding_path(study_context)
    if binding_path is None or not binding_path.is_file():
        return binding_path, None
    payload = yaml.safe_load(binding_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("latentdna binding payload must be a mapping")
    return binding_path, _validate_binding(dict(payload))


def _require_binding_text(binding: Mapping[str, object], field: str) -> str:
    value = _string_or_none(binding.get(field))
    if value is None:
        raise ValueError(f"latentdna binding requires non-empty {field!r}")
    return value


def _require_binding_list(binding: Mapping[str, object], field: str) -> list[str]:
    values = _string_list(binding.get(field))
    if not values:
        raise ValueError(f"latentdna binding requires non-empty {field!r}")
    return values


def _normalize_source_datasets(
    source_datasets: object,
    *,
    field: str = "source_datasets",
    required: bool = True,
) -> dict[str, str]:
    if source_datasets is None and not required:
        return {}
    if not isinstance(source_datasets, Mapping):
        raise ValueError(f"latentdna binding {field!r} must be a mapping")
    normalized = {
        str(scope): text
        for scope, value in sorted(source_datasets.items(), key=lambda item: str(item[0]))
        if (text := _string_or_none(value)) is not None
    }
    if required and not normalized:
        raise ValueError(f"latentdna binding requires at least one non-empty {field} entry")
    return normalized


def _binding_source_dataset_keys(binding: Mapping[str, object] | None) -> list[str]:
    if binding is None:
        raise ValueError("latentdna snapshot validation requires a validated study binding")
    source_datasets = binding.get("source_datasets")
    if not isinstance(source_datasets, Mapping):
        raise ValueError("latentdna binding 'source_datasets' must be a mapping")
    return sorted(_normalize_source_datasets(source_datasets))


def _binding_appendix_source_dataset_keys(binding: Mapping[str, object] | None) -> list[str]:
    if binding is None:
        raise ValueError("latentdna snapshot validation requires a validated study binding")
    return sorted(
        _normalize_source_datasets(
            binding.get("appendix_source_datasets"),
            field="appendix_source_datasets",
            required=False,
        )
    )


def _validate_binding(binding: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(field for field in _REQUIRED_BINDING_FIELDS if field not in binding)
    if missing:
        raise ValueError(f"latentdna binding missing required top-level fields: {missing}")

    normalized_source_datasets = _normalize_source_datasets(binding.get("source_datasets"))
    normalized_appendix_source_datasets = _normalize_source_datasets(
        binding.get("appendix_source_datasets"),
        field="appendix_source_datasets",
        required=False,
    )

    workspace_id = _require_binding_text(binding, "workspace_id")
    workspace_ref = _require_binding_text(binding, "workspace_ref")
    snapshot_ref = _require_binding_text(binding, "snapshot_ref")
    model_families = _require_binding_list(binding, "supported_model_families")
    default_model_family = _require_binding_text(binding, "default_model_family")
    if default_model_family not in model_families:
        raise ValueError("latentdna binding default_model_family must be declared in supported_model_families")

    return {
        **dict(binding),
        "workspace_id": workspace_id,
        "workspace_ref": workspace_ref,
        "snapshot_ref": snapshot_ref,
        "source_datasets": normalized_source_datasets,
        "appendix_source_datasets": normalized_appendix_source_datasets,
        "supported_model_families": model_families,
        "default_model_family": default_model_family,
        "required_wildtype_references": _string_list(binding.get("required_wildtype_references")),
        "decision_deliverables": _require_binding_list(binding, "decision_deliverables"),
    }


def validate_binding(binding: Mapping[str, object]) -> dict[str, object]:
    return _validate_binding(binding)


def _resolve_repo_relative_path(
    study_context: StressEthanolCiproGrowthResolvedContext,
    *,
    raw_path: object,
) -> Path | None:
    repo_root = study_context.study_repo_root
    resolved = _string_or_none(raw_path)
    if repo_root is None or resolved is None:
        return None
    return (repo_root / resolved).resolve()


def latentdna_workspace_ref(
    study_context: StressEthanolCiproGrowthResolvedContext,
    *,
    binding: Mapping[str, object] | None,
) -> Path | None:
    return _resolve_repo_relative_path(
        study_context,
        raw_path=(binding or {}).get("workspace_ref"),
    )


def latentdna_snapshot_ref(
    study_context: StressEthanolCiproGrowthResolvedContext,
    *,
    binding: Mapping[str, object] | None,
) -> Path | None:
    return _resolve_repo_relative_path(
        study_context,
        raw_path=(binding or {}).get("snapshot_ref"),
    )


def _validate_workspace_snapshot(
    *,
    binding: Mapping[str, object] | None,
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    schema_version = _string_or_none(snapshot.get("schema_version"))
    if schema_version != _WORKSPACE_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"latentdna snapshot schema_version must be {_WORKSPACE_SNAPSHOT_SCHEMA_VERSION!r}; got {schema_version!r}"
        )
    missing = sorted(field for field in _REQUIRED_WORKSPACE_SNAPSHOT_FIELDS if field not in snapshot)
    if missing:
        raise ValueError(f"latentdna snapshot missing required top-level fields: {missing}")
    expected_workspace_id = _string_or_none((binding or {}).get("workspace_id"))
    snapshot_workspace_id = _string_or_none(snapshot.get("workspace_id"))
    if expected_workspace_id is not None and snapshot_workspace_id != expected_workspace_id:
        raise ValueError(
            "latentdna snapshot workspace_id does not match the study binding: "
            f"expected {expected_workspace_id!r}, got {snapshot_workspace_id!r}"
        )
    sources = snapshot.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("latentdna snapshot 'sources' must be a mapping")
    missing_binding_sources: list[str] = []
    for source_name in _binding_source_dataset_keys(binding):
        source_payload = sources.get(source_name)
        if source_payload is None:
            missing_binding_sources.append(source_name)
            continue
        _validate_snapshot_source_payload(source_name=source_name, source_payload=source_payload)
    missing_appendix_sources: list[str] = []
    for source_name in _binding_appendix_source_dataset_keys(binding):
        source_payload = sources.get(source_name)
        if source_payload is None:
            missing_appendix_sources.append(source_name)
            continue
        _validate_snapshot_source_payload(source_name=source_name, source_payload=source_payload)

    browser = snapshot.get("browser")
    if not isinstance(browser, Mapping):
        raise ValueError("latentdna snapshot 'browser' must be a mapping")
    for field in ("default_geometry_ids", "preferred_hues"):
        if not isinstance(browser.get(field), list):
            raise ValueError(f"latentdna snapshot browser field {field!r} must be a list")

    deliverables = snapshot.get("deliverables")
    if not isinstance(deliverables, Mapping):
        raise ValueError("latentdna snapshot 'deliverables' must be a mapping")
    missing_decision_deliverables: list[str] = []
    for deliverable_id in _string_list((binding or {}).get("decision_deliverables")):
        deliverable_payload = deliverables.get(deliverable_id)
        if deliverable_payload is None:
            missing_decision_deliverables.append(deliverable_id)
            continue
        if not isinstance(deliverable_payload, Mapping):
            raise ValueError(f"latentdna snapshot deliverable {deliverable_id!r} must be a mapping")
        for field in ("title", "status", "freshness", "acceptance_checks", "artifact_paths", "docs_refs", "warnings"):
            if field not in deliverable_payload:
                raise ValueError(f"latentdna snapshot deliverable {deliverable_id!r} missing required field {field!r}")

    exports = snapshot.get("exports")
    if not isinstance(exports, Mapping):
        raise ValueError("latentdna snapshot 'exports' must be a mapping")
    for export_id, export_payload in exports.items():
        if not isinstance(export_payload, Mapping):
            raise ValueError(f"latentdna snapshot export {export_id!r} must be a mapping")
        for field in ("status", "artifact_path", "manifest_path"):
            if field not in export_payload:
                raise ValueError(f"latentdna snapshot export {export_id!r} missing required field {field!r}")
    return {
        **dict(snapshot),
        "missing_binding_sources": missing_binding_sources,
        "missing_appendix_sources": missing_appendix_sources,
        "missing_decision_deliverables": missing_decision_deliverables,
    }


def _validate_snapshot_source_payload(*, source_name: str, source_payload: object) -> None:
    if not isinstance(source_payload, Mapping):
        raise ValueError(f"latentdna snapshot source {source_name!r} must be a mapping")
    for field in ("kind", "path", "row_count"):
        if field not in source_payload:
            raise ValueError(f"latentdna snapshot source {source_name!r} missing required field {field!r}")


def validate_workspace_snapshot(
    *,
    binding: Mapping[str, object] | None,
    snapshot: Mapping[str, object],
) -> dict[str, object]:
    return _validate_workspace_snapshot(binding=binding, snapshot=snapshot)


def load_latentdna_snapshot(
    study_context: StressEthanolCiproGrowthResolvedContext,
    *,
    binding: Mapping[str, object] | None,
) -> tuple[Path | None, dict[str, object] | None]:
    snapshot_path = latentdna_snapshot_ref(study_context, binding=binding)
    if snapshot_path is None or not snapshot_path.is_file():
        return snapshot_path, None
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("latentdna snapshot payload must be a JSON object")
    return snapshot_path, _validate_workspace_snapshot(binding=binding, snapshot=payload)


def latentdna_doc_path(study_context: StressEthanolCiproGrowthResolvedContext) -> str:
    latentdna_config = study_context.study_pipeline.get("latentdna")
    if isinstance(latentdna_config, Mapping):
        configured = _string_or_none(latentdna_config.get("doc"))
        if configured is not None:
            return configured
    return DEFAULT_LATENTDNA_DOC


def binding_source_datasets(binding: Mapping[str, object] | None) -> dict[str, str]:
    payload = (binding or {}).get("source_datasets")
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, value in payload.items():
        text = _string_or_none(value)
        if text is not None:
            result[str(key)] = text
    return result


def binding_appendix_source_datasets(binding: Mapping[str, object] | None) -> dict[str, str]:
    payload = (binding or {}).get("appendix_source_datasets")
    if not isinstance(payload, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, value in payload.items():
        text = _string_or_none(value)
        if text is not None:
            result[str(key)] = text
    return result


def binding_decision_deliverables(
    binding: Mapping[str, object] | None,
    *,
    snapshot: Mapping[str, object] | None = None,
) -> list[str]:
    declared = _string_list((binding or {}).get("decision_deliverables"))
    if declared:
        return declared
    if isinstance(snapshot, Mapping):
        return _string_list(snapshot.get("decision_ladder"))
    return []


def binding_model_families(binding: Mapping[str, object] | None) -> list[str]:
    return _string_list((binding or {}).get("supported_model_families"))


def binding_default_model_family(binding: Mapping[str, object] | None) -> str | None:
    return _string_or_none((binding or {}).get("default_model_family"))


def binding_required_wildtypes(binding: Mapping[str, object] | None) -> list[str]:
    return _string_list((binding or {}).get("required_wildtype_references"))
