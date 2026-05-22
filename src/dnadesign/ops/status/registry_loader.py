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
_REGISTRY_TOP_LEVEL_KEYS = frozenset({"version", "provider_id", "entries"})
_REGISTRY_ENTRY_KEYS = frozenset(
    {
        "status_kind",
        "owner_boundary",
        "observes_plane",
        "provider_ref",
        "description",
        "input_schema",
        "notes",
        "surface_type",
        "cost_class",
        "summary_scope",
    }
)
_REQUIRED_REGISTRY_ENTRY_KEYS = frozenset(
    {
        "status_kind",
        "owner_boundary",
        "observes_plane",
        "provider_ref",
        "description",
        "surface_type",
        "cost_class",
        "summary_scope",
    }
)
_REGISTRY_INPUT_KEYS = frozenset(
    {
        "name",
        "manifest_key",
        "cli_flag",
        "placeholder",
        "summary",
        "help",
        "type",
        "required",
        "default",
        "choices",
        "path_base",
        "scaffold_required",
    }
)


@lru_cache(maxsize=1)
def list_status_kind_specs() -> tuple[StatusKindSpec, ...]:
    dnadesign_root = Path(__file__).resolve().parents[2]
    return _load_status_kind_specs(
        fragment_paths=_iter_status_registry_fragment_paths(),
        dnadesign_root=dnadesign_root,
    )


def list_status_kind_specs_for_repo(repo_root: Path) -> tuple[StatusKindSpec, ...]:
    dnadesign_root = repo_root.expanduser().resolve() / "src" / "dnadesign"
    if not dnadesign_root.is_dir():
        return ()
    return _load_status_kind_specs(
        fragment_paths=_iter_status_registry_fragment_paths_for_root(dnadesign_root=dnadesign_root),
        dnadesign_root=dnadesign_root,
    )


def _load_status_kind_specs(
    *,
    fragment_paths: tuple[Path, ...],
    dnadesign_root: Path,
) -> tuple[StatusKindSpec, ...]:
    specs: list[StatusKindSpec] = []
    loaded_status_kinds: set[str] = set()
    loaded_provider_ids: set[str] = set()
    for fragment_path in fragment_paths:
        payload = yaml.safe_load(fragment_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"status registry fragment must be a mapping: {fragment_path}")
        _reject_unknown_keys(
            payload,
            allowed_keys=_REGISTRY_TOP_LEVEL_KEYS,
            label="status registry fragment",
            fragment_path=fragment_path,
        )
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
            _reject_unknown_keys(
                entry,
                allowed_keys=_REGISTRY_ENTRY_KEYS,
                label=f"status registry entry {index}",
                fragment_path=fragment_path,
            )
            _require_keys(
                entry,
                required_keys=_REQUIRED_REGISTRY_ENTRY_KEYS,
                label=f"status registry entry {index}",
                fragment_path=fragment_path,
            )
            provider_ref = str(entry.get("provider_ref") or "").strip()
            _validate_provider_ref_location(
                provider_ref=provider_ref,
                status_kind=str(entry.get("status_kind") or "").strip(),
                fragment_path=fragment_path,
                dnadesign_root=dnadesign_root,
            )
            spec = StatusKindSpec(
                status_kind=str(entry.get("status_kind") or "").strip(),
                provider_id=provider_id,
                owner_boundary=str(entry.get("owner_boundary") or "").strip(),
                observes_plane=str(entry.get("observes_plane") or "").strip().lower(),  # type: ignore[arg-type]
                provider_ref=provider_ref,
                description=str(entry.get("description") or "").strip(),
                input_schema=_load_input_schema(entry.get("input_schema"), fragment_path=fragment_path),
                notes=_load_notes(entry.get("notes")),
                surface_type=str(entry.get("surface_type") or "").strip(),
                cost_class=str(entry.get("cost_class") or "").strip().lower(),  # type: ignore[arg-type]
                summary_scope=str(entry.get("summary_scope") or "").strip().lower(),  # type: ignore[arg-type]
            )
            if spec.status_kind in loaded_status_kinds:
                raise ValueError(
                    f"status registry fragment entry {index} duplicates status kind {spec.status_kind}: {fragment_path}"
                )
            loaded_status_kinds.add(spec.status_kind)
            specs.append(spec)
    return tuple(sorted(specs, key=lambda item: item.status_kind))


def _reject_unknown_keys(
    payload: dict,
    *,
    allowed_keys: frozenset[str],
    label: str,
    fragment_path: Path,
) -> None:
    unknown_keys = sorted(str(key) for key in payload if str(key) not in allowed_keys)
    if unknown_keys:
        raise ValueError(f"{label} has unknown key(s) {', '.join(unknown_keys)}: {fragment_path}")


def _require_keys(
    payload: dict,
    *,
    required_keys: frozenset[str],
    label: str,
    fragment_path: Path,
) -> None:
    missing_keys = sorted(key for key in required_keys if not str(payload.get(key) or "").strip())
    if missing_keys:
        raise ValueError(f"{label} is missing required key(s) {', '.join(missing_keys)}: {fragment_path}")


def _validate_provider_ref_location(
    *,
    provider_ref: str,
    status_kind: str,
    fragment_path: Path,
    dnadesign_root: Path,
) -> None:
    expected_prefix = _expected_provider_ref_prefix(fragment_path=fragment_path, dnadesign_root=dnadesign_root)
    if expected_prefix is None:
        return
    module_name = provider_ref.split(":", maxsplit=1)[0]
    if module_name == expected_prefix.rstrip(".") or module_name.startswith(expected_prefix):
        return
    raise ValueError(
        "status registry provider_ref must stay under the fragment owner package: "
        f"{status_kind} provider_ref={provider_ref!r} expected_prefix={expected_prefix!r}: {fragment_path}"
    )


def _expected_provider_ref_prefix(*, fragment_path: Path, dnadesign_root: Path) -> str | None:
    try:
        relative_parts = fragment_path.resolve().relative_to(dnadesign_root).parts
    except ValueError:
        return None
    if len(relative_parts) >= 4 and relative_parts[0] == "ops" and relative_parts[1] == "providers":
        return f"dnadesign.ops.providers.{relative_parts[2]}."
    if (
        len(relative_parts) >= 6
        and relative_parts[0] == "studies"
        and relative_parts[1] == "studies"
        and relative_parts[3] == "status"
        and relative_parts[4] == "ops"
    ):
        return f"dnadesign.studies.studies.{relative_parts[2]}.status.ops."
    if len(relative_parts) >= 4 and relative_parts[1] == "src" and relative_parts[2] == "ops":
        return f"dnadesign.{relative_parts[0]}.src.ops."
    if len(relative_parts) >= 3 and relative_parts[1] == "ops":
        return f"dnadesign.{relative_parts[0]}.ops."
    return None


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
    dnadesign_root = Path(__file__).resolve().parents[2]
    return _iter_status_registry_fragment_paths_for_root(dnadesign_root=dnadesign_root)


def _iter_status_registry_fragment_paths_for_root(*, dnadesign_root: Path) -> tuple[Path, ...]:
    fragment_paths = [
        path
        for path in dnadesign_root.rglob(_STATUS_REGISTRY_FILENAME)
        if path.is_file() and _is_status_registry_fragment_path(path=path, dnadesign_root=dnadesign_root)
    ]
    return tuple(sorted(fragment_paths))


def _is_status_registry_fragment_path(*, path: Path, dnadesign_root: Path) -> bool:
    if path.parent.name == "ops":
        return True
    relative_parts = path.resolve().relative_to(dnadesign_root).parts
    return (
        len(relative_parts) == 4
        and relative_parts[0] == "ops"
        and relative_parts[1] == "providers"
        and relative_parts[3] == _STATUS_REGISTRY_FILENAME
    )


def _load_input_schema(payload: object, *, fragment_path: Path) -> tuple[InputFieldSpec, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"status registry input_schema must be a list: {fragment_path}")
    fields: list[InputFieldSpec] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"status registry input field {index} must be a mapping: {fragment_path}")
        _reject_unknown_keys(
            item,
            allowed_keys=_REGISTRY_INPUT_KEYS,
            label=f"status registry input field {index}",
            fragment_path=fragment_path,
        )
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
