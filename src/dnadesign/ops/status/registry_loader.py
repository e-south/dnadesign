"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/registry_loader.py

Metadata-only loader for checked-in OPS status registry fragments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from .models import InputFieldSpec, StatusKindSpec

_STATUS_REGISTRY_FILENAME = "status.registry.yaml"


@lru_cache(maxsize=1)
def list_status_kind_specs() -> tuple[StatusKindSpec, ...]:
    specs: list[StatusKindSpec] = []
    loaded_status_kinds: set[str] = set()
    loaded_provider_ids: set[str] = set()
    for fragment_path in _iter_status_registry_fragment_paths():
        payload = yaml.safe_load(fragment_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"status registry fragment must be a mapping: {fragment_path}")
        version = int(payload.get("version") or 0)
        if version != 1:
            raise ValueError(f"unsupported status registry fragment version {version} in {fragment_path}")
        provider_id = str(payload.get("provider_id") or "").strip()
        if not provider_id:
            raise ValueError(f"status registry fragment must define provider_id: {fragment_path}")
        if provider_id in loaded_provider_ids:
            raise ValueError(f"status registry provider already registered: {provider_id}")
        loaded_provider_ids.add(provider_id)

        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"status registry fragment must define a non-empty entries list: {fragment_path}")
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"status registry entry {index} must be a mapping: {fragment_path}")
            spec = StatusKindSpec(
                status_kind=str(entry.get("status_kind") or "").strip(),
                provider_id=provider_id,
                provider_ref=str(entry.get("provider_ref") or "").strip(),
                description=str(
                    entry.get("description") or "Read one explicit, artifact-backed status surface."
                ).strip(),
                input_schema=_load_input_schema(entry.get("input_schema"), fragment_path=fragment_path),
                notes=_load_notes(entry.get("notes")),
                surface_type=str(entry.get("surface_type") or "artifact_state").strip() or "artifact_state",
                cost_class=str(entry.get("cost_class") or "cheap").strip().lower(),  # type: ignore[arg-type]
                summary_scope=str(entry.get("summary_scope") or "workspace").strip().lower(),  # type: ignore[arg-type]
            )
            if spec.status_kind in loaded_status_kinds:
                raise ValueError(
                    f"status registry fragment entry {index} duplicates status kind {spec.status_kind}: {fragment_path}"
                )
            loaded_status_kinds.add(spec.status_kind)
            specs.append(spec)
    return tuple(sorted(specs, key=lambda item: item.status_kind))


def load_status_kind_spec(status_kind: str) -> StatusKindSpec:
    normalized_kind = str(status_kind or "").strip()
    if not normalized_kind:
        raise ValueError("status kind must be non-empty")
    for spec in list_status_kind_specs():
        if spec.status_kind == normalized_kind:
            return spec
    raise ValueError(
        f"unsupported status kind: {normalized_kind}. "
        "Add an explicit checked-in status registry fragment before using this surface."
    )


def _iter_status_registry_fragment_paths() -> tuple[Path, ...]:
    ops_root = Path(__file__).resolve().parents[1]
    dnadesign_root = Path(__file__).resolve().parents[2]
    fragment_paths: list[Path] = []
    for search_root in (
        ops_root / "providers",
        dnadesign_root / "studies",
    ):
        if not search_root.exists():
            continue
        fragment_paths.extend(path for path in search_root.rglob(_STATUS_REGISTRY_FILENAME) if path.is_file())
    return tuple(sorted(fragment_paths))


def _load_input_schema(payload: object, *, fragment_path: Path) -> tuple[InputFieldSpec, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"status registry input_schema must be a list: {fragment_path}")
    fields: list[InputFieldSpec] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"status registry input field {index} must be a mapping: {fragment_path}")
        cli_flag = str(item.get("cli_flag") or "").strip()
        name = str(item.get("name") or item.get("manifest_key") or "").strip()
        manifest_key = str(item.get("manifest_key") or name).strip()
        if not cli_flag and name:
            cli_flag = "--" + name.replace("_", "-")
        placeholder = str(item.get("placeholder") or f"<{manifest_key}>")
        fields.append(
            InputFieldSpec(
                name=name,
                manifest_key=manifest_key,
                cli_flag=cli_flag,
                placeholder=placeholder,
                summary=str(item.get("summary") or item.get("help") or "").strip(),
                type=str(item.get("type") or "str").strip(),  # type: ignore[arg-type]
                required=bool(item.get("required", True)),
                default=item.get("default"),
                choices=tuple(str(choice) for choice in item.get("choices") or ()),
                path_base=item.get("path_base"),  # type: ignore[arg-type]
                scaffold_required=item.get("scaffold_required"),
            )
        )
    return tuple(fields)


def _load_notes(payload: object) -> tuple[str, ...]:
    if payload is None:
        return ()
    if isinstance(payload, list):
        return tuple(str(item).strip() for item in payload if str(item).strip())
    note = str(payload).strip()
    return (note,) if note else ()


__all__ = ["list_status_kind_specs", "load_status_kind_spec"]
