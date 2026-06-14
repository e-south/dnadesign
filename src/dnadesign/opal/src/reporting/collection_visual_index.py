"""Loaded collection-visual index contracts for OPAL campaign-set notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..analysis.campaign_set import list_collection_visual_surface_kinds
from ..core.utils import ExitCodes, OpalError, read_json

COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION = "opal.collection_visual_manifest_index.v1"


def load_collection_visual_manifest_index(
    path: str | Path,
    *,
    expected_collection_id: str | None = None,
    allowed_surface_kinds: Iterable[str] | None = None,
) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise OpalError(
            f"Collection visual manifest index is not a JSON object: {path}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if payload.get("schema_version") != COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION:
        raise OpalError(
            f"Unsupported collection visual manifest index schema: {payload.get('schema_version')!r}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    _validate_collection_visual_manifest_index(
        payload,
        index_path=Path(path),
        expected_collection_id=expected_collection_id,
        allowed_surface_kinds=allowed_surface_kinds,
    )
    return payload


def _validate_collection_visual_manifest_index(
    payload: Mapping[str, Any],
    *,
    index_path: Path,
    expected_collection_id: str | None,
    allowed_surface_kinds: Iterable[str] | None,
) -> None:
    collection_id = _required_string(payload.get("collection_id"), field="collection_id")
    if expected_collection_id is not None and collection_id != str(expected_collection_id):
        raise OpalError(
            "Collection visual manifest index collection_id mismatch: "
            f"expected {expected_collection_id!r}, found {collection_id!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    comparison_sets = _required_mapping_list(payload.get("comparison_sets"), field="comparison_sets")
    visuals = _required_mapping_list(payload.get("visuals"), field="visuals")
    _validate_declared_count(payload, field="comparison_set_count", rows=comparison_sets)
    _validate_declared_count(payload, field="visual_count", rows=visuals)

    declared_surface_kinds = _string_list(payload.get("surface_kinds"), field="surface_kinds", allow_empty=True)
    extension_surface_kinds = {str(kind).strip() for kind in (allowed_surface_kinds or []) if str(kind).strip()}
    undeclared_extension_kinds = sorted(set(declared_surface_kinds) - list_collection_visual_surface_kinds())
    unauthorized_extension_kinds = sorted(set(undeclared_extension_kinds) - extension_surface_kinds)
    if unauthorized_extension_kinds:
        raise OpalError(
            "Collection visual manifest index declares extension surface kind(s) without caller approval: "
            + ", ".join(repr(kind) for kind in unauthorized_extension_kinds),
            ExitCodes.CONTRACT_VIOLATION,
        )
    allowed = {
        *list_collection_visual_surface_kinds(),
        *declared_surface_kinds,
        *extension_surface_kinds,
    }
    comparison_set_keys = {str(row.get("key") or "").strip() for row in comparison_sets}
    for index, visual in enumerate(visuals):
        _validate_collection_visual_manifest_entry(
            visual,
            index=index,
            index_path=index_path,
            allowed_surface_kinds=allowed,
            comparison_set_keys=comparison_set_keys,
        )


def _validate_collection_visual_manifest_entry(
    visual: Mapping[str, Any],
    *,
    index: int,
    index_path: Path,
    allowed_surface_kinds: set[str],
    comparison_set_keys: set[str],
) -> None:
    field = f"visuals[{index}]"
    _required_string(visual.get("visual_id"), field=f"{field}.visual_id")
    surface_kind = _required_string(visual.get("surface_kind"), field=f"{field}.surface_kind")
    if surface_kind not in allowed_surface_kinds:
        raise OpalError(
            f"Collection visual manifest index {field}.surface_kind is not declared: {surface_kind!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    set_key = _required_string(visual.get("comparison_set_key"), field=f"{field}.comparison_set_key")
    if comparison_set_keys and set_key not in comparison_set_keys:
        raise OpalError(
            "Collection visual manifest index "
            f"{field}.comparison_set_key is not listed in comparison_sets: {set_key!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    _required_string(visual.get("comparison_set_label"), field=f"{field}.comparison_set_label")
    for path_field in ("path", "tidy_csv", "manifest_path"):
        _require_existing_visual_path(visual.get(path_field), field=f"{field}.{path_field}", index_path=index_path)
    _validate_visual_artifact_manifest(visual, field=field, index_path=index_path)
    freshness = visual.get("freshness")
    if not isinstance(freshness, Mapping):
        raise OpalError(
            f"Collection visual manifest index {field}.freshness must be a mapping.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    freshness_status = _required_string(freshness.get("status"), field=f"{field}.freshness.status")
    if freshness_status not in {"current", "fresh"}:
        raise OpalError(
            f"Collection visual manifest index {field}.freshness.status must be 'current' or 'fresh'.",
            ExitCodes.CONTRACT_VIOLATION,
        )


def _validate_visual_artifact_manifest(visual: Mapping[str, Any], *, field: str, index_path: Path) -> None:
    manifest_path = _require_existing_visual_path(
        visual.get("manifest_path"),
        field=f"{field}.manifest_path",
        index_path=index_path,
    )
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise OpalError(
            f"Collection visual artifact manifest is not a JSON object: {manifest_path}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    schema_version = str(manifest.get("schema_version") or "").strip()
    if not schema_version:
        raise OpalError(
            f"Collection visual manifest path has no schema_version: {manifest_path}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if schema_version != "opal.collection_visual_artifact.v1":
        surface_kind = _required_string(visual.get("surface_kind"), field=f"{field}.surface_kind")
        if surface_kind in list_collection_visual_surface_kinds():
            raise OpalError(
                "Unsupported collection visual artifact manifest schema for generic OPAL surface: "
                f"{schema_version!r} at {manifest_path}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        return
    for key in ("collection_id", "visual_id", "surface_kind", "comparison_set_key"):
        expected = _required_string(visual.get(key), field=f"{field}.{key}")
        actual = _required_string(manifest.get(key), field=f"{field}.manifest.{key}")
        if actual != expected:
            raise OpalError(
                f"Collection visual artifact manifest {key} mismatch for {field}: "
                f"expected {expected!r}, found {actual!r}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
    for index_key, manifest_key in (("path", "path"), ("tidy_csv", "tidy_csv")):
        expected_path = _resolve_visual_path(
            visual.get(index_key),
            field=f"{field}.{index_key}",
            base_path=index_path.parent,
        )
        actual_path = _resolve_visual_path(
            manifest.get(manifest_key),
            field=f"{field}.manifest.{manifest_key}",
            base_path=manifest_path.parent,
        )
        if actual_path != expected_path:
            raise OpalError(
                f"Collection visual artifact manifest {manifest_key} mismatch for {field}: "
                f"expected {expected_path}, found {actual_path}.",
                ExitCodes.CONTRACT_VIOLATION,
            )


def _require_existing_visual_path(value: Any, *, field: str, index_path: Path) -> Path:
    path = _resolve_visual_path(value, field=field, base_path=index_path.parent)
    if not path.exists():
        raise OpalError(
            f"Collection visual manifest index {field} does not exist: {path}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return path


def _resolve_visual_path(value: Any, *, field: str, base_path: Path) -> Path:
    text = _required_string(value, field=field)
    path = Path(text)
    if not path.is_absolute():
        path = base_path / path
    return path.resolve(strict=False)


def _validate_declared_count(payload: Mapping[str, Any], *, field: str, rows: list[Mapping[str, Any]]) -> None:
    value = payload.get(field)
    try:
        declared = int(value)
    except (TypeError, ValueError) as exc:
        raise OpalError(
            f"Collection visual manifest index field {field} must be an integer.",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    if declared != len(rows):
        raise OpalError(
            f"Collection visual manifest index field {field}={declared} does not match {len(rows)} row(s).",
            ExitCodes.CONTRACT_VIOLATION,
        )


def _required_mapping_list(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise OpalError(
            f"Collection visual manifest index field {field} must be a list of objects.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return value


def _string_list(value: Any, *, field: str, allow_empty: bool) -> list[str]:
    if value in (None, "") and allow_empty:
        return []
    if not isinstance(value, list):
        raise OpalError(
            f"Collection visual manifest index field {field} must be a list of strings.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    out = [str(item).strip() for item in value]
    if any(not item for item in out) or (not out and not allow_empty):
        raise OpalError(
            f"Collection visual manifest index field {field} must contain non-empty strings.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return out


def _required_string(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise OpalError(
            f"Collection visual manifest index field {field} must be a non-empty string.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return text


__all__ = [
    "COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION",
    "load_collection_visual_manifest_index",
]
