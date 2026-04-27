"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/nickases/catalog.py

Nickase catalog loading and normalization shared across Cruncher workflow
families.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.nickases.errors import NickaseCatalogError
from dnadesign.cruncher.nickases.models import NickaseCatalog, NickaseCatalogDocument

_RAW_CUT_NOTATION_RE = re.compile(r"^\s*([A-Za-z]+)\((none|-?\d+)\/(none|-?\d+)\)\s*$", re.IGNORECASE)
_LOCAL_CATALOG_TOP_LEVEL = "nickases"
_PRESET_RESOURCE_ROOT = ("resources", "cassette", "catalogs")
_ENTRY_METADATA_EXCLUDE = {
    "id",
    "specificity_id",
    "motif_top_5to3",
    "vendor_diagram_top_5to3",
    "motif_len",
    "top_cut_offset",
    "bottom_cut_offset",
    "source",
    "raw_cut_notation",
    "raw_cut_offset_reference",
    "metadata",
    "recognition_sequence",
    "nicked_site_strand",
    "cut_offset",
    "vendor",
    "vendor_catalog_number",
    "source_url",
    "origin_class",
    "source_family",
    "notes",
    "selection",
    "operational",
    "outside_site",
    "snapback_tier",
    "commercial_confidence",
    "warning_codes",
    "incubation_temp_c",
    "buffer_family",
    "heat_inactivation",
    "methylation_sensitivity",
    "star_activity_warning",
    "double_strand_cleavage_warning",
}


def resolve_workspace_relative_path(raw_path: Path, *, workspace_root: Path, label: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    if any(part == ".." for part in path.parts):
        raise NickaseCatalogError(f"{label} must not traverse outside the workspace: {raw_path}")
    return (workspace_root / path).resolve()


def _parse_cut_value(raw_value: str) -> int | None:
    value = raw_value.strip().lower()
    if value == "none":
        return None
    return int(value)


def _parse_raw_cut_notation(raw_notation: str) -> tuple[str, int | None, int | None]:
    match = _RAW_CUT_NOTATION_RE.fullmatch(str(raw_notation or ""))
    if match is None:
        raise NickaseCatalogError(f"Unsupported raw cut notation: {raw_notation!r}")
    motif, top_raw, bottom_raw = match.groups()
    return motif.upper(), _parse_cut_value(top_raw), _parse_cut_value(bottom_raw)


def _normalize_raw_cut_offset_reference(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value or "").strip().lower()
    if value not in {"motif_start", "motif_end"}:
        raise NickaseCatalogError(
            "CATALOG_ENTRY_NOT_NORMALIZABLE: raw_cut_offset_reference must be 'motif_start' or 'motif_end'."
        )
    return value


def _normalize_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(entry.get("metadata") or {})
    for key, value in entry.items():
        if key not in _ENTRY_METADATA_EXCLUDE and value is not None:
            metadata[key] = value
    return metadata


def _normalize_string_list(value: Any, *, label: str) -> list[str]:
    items = value or []
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise NickaseCatalogError(f"{label} must be a list of strings.")
    normalized = [str(item).strip() for item in items]
    if any(not item for item in normalized):
        raise NickaseCatalogError(f"{label} must not contain blank values.")
    return normalized


def _normalize_selection(entry: dict[str, Any]) -> dict[str, Any] | None:
    raw_selection = entry.get("selection")
    if raw_selection is None:
        selection: dict[str, Any] = {}
    elif isinstance(raw_selection, dict):
        selection = dict(raw_selection)
    else:
        raise NickaseCatalogError("entry.selection must be a mapping when present.")

    for key in ("outside_site", "snapback_tier", "commercial_confidence", "warning_codes"):
        if key in entry:
            if key in selection and selection[key] != entry[key]:
                raise NickaseCatalogError(f"entry.selection.{key} must not conflict with top-level {key}.")
            selection[key] = entry[key]
    if not selection:
        return None
    if "warning_codes" in selection:
        selection["warning_codes"] = _normalize_string_list(
            selection["warning_codes"], label="entry.selection.warning_codes"
        )
    return selection


def _normalize_operational(entry: dict[str, Any]) -> dict[str, Any] | None:
    raw_operational = entry.get("operational")
    if raw_operational is None:
        operational: dict[str, Any] = {}
    elif isinstance(raw_operational, dict):
        operational = dict(raw_operational)
    else:
        raise NickaseCatalogError("entry.operational must be a mapping when present.")

    for key in (
        "incubation_temp_c",
        "buffer_family",
        "heat_inactivation",
        "methylation_sensitivity",
        "star_activity_warning",
        "double_strand_cleavage_warning",
    ):
        if key in entry:
            if key in operational and operational[key] != entry[key]:
                raise NickaseCatalogError(f"entry.operational.{key} must not conflict with top-level {key}.")
            operational[key] = entry[key]
    if not operational:
        return None
    methylation = operational.get("methylation_sensitivity")
    if methylation is not None and not isinstance(methylation, dict):
        raise NickaseCatalogError("entry.operational.methylation_sensitivity must be a mapping when present.")
    return operational


def _normalize_preset_ids(value: Any, *, label: str) -> list[str]:
    raw_ids = value or []
    if not isinstance(raw_ids, list):
        raise NickaseCatalogError(f"{label} must be a list when present.")
    preset_ids = [str(item or "").strip() for item in raw_ids]
    if any(not item for item in preset_ids):
        raise NickaseCatalogError(f"{label} must not contain blank values.")
    if len(set(preset_ids)) != len(preset_ids):
        raise NickaseCatalogError(f"{label} must not repeat preset ids.")
    return preset_ids


def _normalize_catalog_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    if "id" not in normalized:
        raise NickaseCatalogError("Catalog entries must define an id.")

    raw_cut_notation = normalized.get("raw_cut_notation")
    parsed_motif: str | None = None
    parsed_top: int | None = None
    parsed_bottom: int | None = None
    if raw_cut_notation is not None:
        parsed_motif, parsed_top, parsed_bottom = _parse_raw_cut_notation(str(raw_cut_notation))

    motif = normalized.get("motif_top_5to3")
    vendor_diagram = normalized.get("vendor_diagram_top_5to3")
    recognition_sequence = normalized.get("recognition_sequence")
    if motif is not None and recognition_sequence is not None:
        if str(motif).strip().upper() != str(recognition_sequence).strip().upper():
            raise NickaseCatalogError(
                f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} defines both motif_top_5to3 "
                "and recognition_sequence with different values."
            )
    if parsed_motif is not None and motif is not None:
        if str(motif).strip().upper() != parsed_motif:
            raise NickaseCatalogError(
                f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} defines raw_cut_notation "
                "with a motif that disagrees with motif_top_5to3."
            )
    if parsed_motif is not None and recognition_sequence is not None:
        if str(recognition_sequence).strip().upper() != parsed_motif:
            raise NickaseCatalogError(
                f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} defines raw_cut_notation "
                "with a motif that disagrees with recognition_sequence."
            )
    canonical_motif = motif or recognition_sequence or parsed_motif
    if canonical_motif is None:
        raise NickaseCatalogError(
            f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} must define a motif or raw_cut_notation."
        )
    canonical_vendor_diagram = str(vendor_diagram).strip().upper() if vendor_diagram is not None else None
    if (
        canonical_vendor_diagram is not None
        and canonical_vendor_diagram[: len(str(canonical_motif).strip())] != str(canonical_motif).strip().upper()
    ):
        raise NickaseCatalogError(
            f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} defines vendor_diagram_top_5to3 "
            "that does not start with motif_top_5to3."
        )

    selection = _normalize_selection(normalized)
    specificity_id = str(normalized.get("specificity_id") or normalized["id"]).strip()
    raw_cut_offset_reference = _normalize_raw_cut_offset_reference(normalized.get("raw_cut_offset_reference"))
    top_cut_offset = normalized.get("top_cut_offset")
    bottom_cut_offset = normalized.get("bottom_cut_offset")
    if raw_cut_notation is not None:
        if top_cut_offset is not None or bottom_cut_offset is not None:
            raise NickaseCatalogError(
                f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} cannot mix raw_cut_notation "
                "with top_cut_offset/bottom_cut_offset."
            )
        top_cut_offset = parsed_top
        bottom_cut_offset = parsed_bottom
        motif_end_reference = raw_cut_offset_reference == "motif_end" or (
            raw_cut_offset_reference is None and selection is not None and selection.get("outside_site") is True
        )
        if motif_end_reference:
            motif_len = len(str(canonical_motif).strip())
            if top_cut_offset is not None and top_cut_offset >= 0:
                top_cut_offset += motif_len
            if bottom_cut_offset is not None and bottom_cut_offset >= 0:
                bottom_cut_offset += motif_len

    has_legacy_geometry = "nicked_site_strand" in normalized or "cut_offset" in normalized
    has_canonical_geometry = top_cut_offset is not None or bottom_cut_offset is not None
    if has_legacy_geometry and has_canonical_geometry:
        raise NickaseCatalogError(
            f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} cannot mix legacy and canonical cut fields."
        )

    if has_legacy_geometry:
        if "nicked_site_strand" not in normalized or "cut_offset" not in normalized:
            raise NickaseCatalogError(
                f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} must define both nicked_site_strand "
                "and cut_offset when using legacy geometry."
            )
        motif_len = len(str(canonical_motif).strip())
        cut_offset = int(normalized["cut_offset"])
        if cut_offset < 0 or cut_offset > motif_len:
            raise NickaseCatalogError(
                f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} has legacy cut_offset outside motif bounds."
            )
        nicked_site_strand = str(normalized["nicked_site_strand"]).strip().lower()
        if nicked_site_strand == "forward":
            top_cut_offset = cut_offset
            bottom_cut_offset = None
        elif nicked_site_strand == "reverse":
            top_cut_offset = None
            bottom_cut_offset = motif_len - cut_offset
        else:
            raise NickaseCatalogError(
                f"CATALOG_ENTRY_NOT_NORMALIZABLE: entry {normalized['id']} uses unknown "
                f"legacy nicked_site_strand {normalized['nicked_site_strand']!r}."
            )

    source = normalized.get("source")
    notes = _normalize_string_list(normalized.get("notes"), label="entry.notes")
    operational = _normalize_operational(normalized)
    metadata = _normalize_metadata(normalized)
    vendor = normalized.get("vendor")
    if source is None and vendor is not None:
        source = str(vendor)

    return {
        "id": normalized["id"],
        "specificity_id": specificity_id,
        "motif_top_5to3": str(canonical_motif).strip().upper(),
        "vendor_diagram_top_5to3": canonical_vendor_diagram,
        "motif_len": len(str(canonical_motif).strip()),
        "top_cut_offset": top_cut_offset,
        "bottom_cut_offset": bottom_cut_offset,
        "source": source,
        "vendor": vendor,
        "vendor_catalog_number": normalized.get("vendor_catalog_number"),
        "source_url": normalized.get("source_url"),
        "origin_class": normalized.get("origin_class"),
        "source_family": normalized.get("source_family"),
        "notes": notes,
        "selection": selection,
        "operational": operational,
        "raw_cut_notation": raw_cut_notation,
        "raw_cut_offset_reference": raw_cut_offset_reference,
        "metadata": metadata,
    }


def _normalize_product_alias(entry: dict[str, Any]) -> dict[str, Any]:
    alias = dict(entry)
    if "alias_id" not in alias or "canonical_variant_id" not in alias:
        raise NickaseCatalogError("Product alias entries must define alias_id and canonical_variant_id.")
    notes = alias.get("notes") or []
    if not isinstance(notes, list) or not all(isinstance(item, str) for item in notes):
        raise NickaseCatalogError("Product alias notes must be a list of strings.")
    return {
        "alias_id": alias["alias_id"],
        "canonical_variant_id": alias["canonical_variant_id"],
        "vendor": alias.get("vendor"),
        "vendor_catalog_number": alias.get("vendor_catalog_number"),
        "source_url": alias.get("source_url"),
        "alias_kind": alias.get("alias_kind"),
        "notes": notes,
    }


def _normalize_local_catalog_document(payload: dict[str, Any]) -> dict[str, Any]:
    if _LOCAL_CATALOG_TOP_LEVEL not in payload:
        raise NickaseCatalogError("Nickase catalog must define top-level key 'nickases'.")
    nickases = payload[_LOCAL_CATALOG_TOP_LEVEL]
    if not isinstance(nickases, dict):
        raise NickaseCatalogError("nickases must be a mapping.")
    entries = nickases.get("entries")
    if not isinstance(entries, list):
        raise NickaseCatalogError("nickases.entries must be a list.")
    normalized_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise NickaseCatalogError("nickases.entries must contain mappings.")
        normalized_entries.append(_normalize_catalog_entry(entry))
    product_aliases = nickases.get("product_aliases", [])
    if not isinstance(product_aliases, list):
        raise NickaseCatalogError("nickases.product_aliases must be a list when present.")
    normalized_aliases = []
    for alias in product_aliases:
        if not isinstance(alias, dict):
            raise NickaseCatalogError("nickases.product_aliases must contain mappings.")
        normalized_aliases.append(_normalize_product_alias(alias))
    return {
        "nickases": {
            "schema_version": nickases.get("schema_version", 1),
            "entries": normalized_entries,
            "preset_id": str(nickases["preset_id"]) if nickases.get("preset_id") is not None else None,
            "preset_ids": _normalize_preset_ids(nickases.get("preset_ids"), label="nickases.preset_ids"),
            "catalog_version": nickases.get("catalog_version"),
            "generated_from": str(nickases["generated_from"]) if nickases.get("generated_from") is not None else None,
            "generated_on": str(nickases["generated_on"]) if nickases.get("generated_on") is not None else None,
            "normalization_policy": (
                str(nickases["normalization_policy"]) if nickases.get("normalization_policy") is not None else None
            ),
            "product_aliases": normalized_aliases,
        }
    }


def _normalize_preset_catalog_document(payload: dict[str, Any]) -> dict[str, Any]:
    entries = payload.get("variants")
    if not isinstance(entries, list):
        raise NickaseCatalogError("Preset catalog variants must be a list.")
    normalized_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise NickaseCatalogError("Preset catalog variants must contain mappings.")
        normalized_entries.append(_normalize_catalog_entry(entry))
    product_aliases = payload.get("product_aliases", [])
    if not isinstance(product_aliases, list):
        raise NickaseCatalogError("Preset product_aliases must be a list when present.")
    normalized_aliases = []
    for alias in product_aliases:
        if not isinstance(alias, dict):
            raise NickaseCatalogError("Preset product_aliases must contain mappings.")
        normalized_aliases.append(_normalize_product_alias(alias))
    return {
        "nickases": {
            "schema_version": 1,
            "entries": normalized_entries,
            "preset_id": str(payload["preset_id"]) if payload.get("preset_id") is not None else None,
            "preset_ids": _normalize_preset_ids(payload.get("preset_ids"), label="preset.preset_ids"),
            "catalog_version": payload.get("catalog_version"),
            "generated_from": str(payload["generated_from"]) if payload.get("generated_from") is not None else None,
            "generated_on": str(payload["generated_on"]) if payload.get("generated_on") is not None else None,
            "normalization_policy": (
                str(payload["normalization_policy"]) if payload.get("normalization_policy") is not None else None
            ),
            "product_aliases": normalized_aliases,
        }
    }


def _normalize_catalog_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if _LOCAL_CATALOG_TOP_LEVEL in payload:
        return _normalize_local_catalog_document(payload)
    if "variants" in payload:
        return _normalize_preset_catalog_document(payload)
    raise NickaseCatalogError("Nickase catalog must define top-level key 'nickases' or 'variants'.")


def _load_catalog_from_payload(payload: dict[str, Any], *, source_label: str) -> NickaseCatalog:
    try:
        document = NickaseCatalogDocument.model_validate(_normalize_catalog_payload(payload))
    except NickaseCatalogError:
        raise
    except Exception as exc:
        raise NickaseCatalogError(f"Nickase catalog validation failed for {source_label}: {exc}") from exc
    return document.nickases


def _load_catalog_from_text(text: str, *, source_label: str) -> NickaseCatalog:
    try:
        payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise NickaseCatalogError(f"Invalid YAML in nickase catalog {source_label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NickaseCatalogError(f"Nickase catalog {source_label} must be a YAML mapping.")
    return _load_catalog_from_payload(payload, source_label=source_label)


def resolve_builtin_catalog_resource(preset_id: str) -> resources.abc.Traversable:
    resource = resources.files("dnadesign.cruncher")
    for part in _PRESET_RESOURCE_ROOT:
        resource = resource.joinpath(part)
    resource = resource.joinpath(f"{preset_id}.yaml")
    if not resource.is_file():
        raise NickaseCatalogError(f"Unknown built-in nickase preset: {preset_id}")
    return resource


def load_builtin_nickase_catalog_preset(preset_id: str) -> NickaseCatalog:
    resource = resolve_builtin_catalog_resource(preset_id)
    return _load_catalog_from_text(resource.read_text(encoding="utf-8"), source_label=f"preset:{preset_id}")


def read_builtin_nickase_catalog_preset_text(preset_id: str) -> str:
    resource = resolve_builtin_catalog_resource(preset_id)
    return resource.read_text(encoding="utf-8")


def load_nickase_catalog(path: Path) -> NickaseCatalog:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Nickase catalog not found: {resolved}")
    return _load_catalog_from_text(resolved.read_text(encoding="utf-8"), source_label=str(resolved))


def merge_nickase_catalogs(*catalogs: NickaseCatalog) -> NickaseCatalog:
    entries = []
    product_aliases = []
    seen_entry_ids: set[str] = set()
    seen_alias_ids: set[str] = set()
    preset_id: str | None = None
    preset_ids: list[str] = []
    catalog_version: int | None = None
    generated_from: str | None = None
    generated_on: str | None = None
    normalization_policy: str | None = None

    for catalog in catalogs:
        if catalog.preset_id is not None and preset_id is None:
            preset_id = catalog.preset_id
        for catalog_preset_id in catalog.preset_ids:
            if catalog_preset_id not in preset_ids:
                preset_ids.append(catalog_preset_id)
        if catalog.catalog_version is not None and catalog_version is None:
            catalog_version = catalog.catalog_version
        if catalog.generated_from is not None and generated_from is None:
            generated_from = catalog.generated_from
        if catalog.generated_on is not None and generated_on is None:
            generated_on = catalog.generated_on
        if catalog.normalization_policy is not None and normalization_policy is None:
            normalization_policy = catalog.normalization_policy
        for entry in catalog.entries:
            if entry.id in seen_entry_ids:
                raise NickaseCatalogError(f"Duplicate nickase id across merged catalogs: {entry.id}")
            entries.append(entry)
            seen_entry_ids.add(entry.id)
        for alias in catalog.product_aliases:
            if alias.alias_id in seen_alias_ids:
                raise NickaseCatalogError(f"Duplicate product alias id across merged catalogs: {alias.alias_id}")
            product_aliases.append(alias)
            seen_alias_ids.add(alias.alias_id)

    return NickaseCatalog(
        schema_version=1,
        entries=entries,
        preset_id=preset_id,
        preset_ids=preset_ids,
        catalog_version=catalog_version,
        generated_from=generated_from,
        generated_on=generated_on,
        normalization_policy=normalization_policy,
        product_aliases=product_aliases,
    )


def load_merged_nickase_catalog(
    *,
    preset_id: str | None,
    additional_preset_ids: list[str] | None = None,
    additional_paths: list[Path],
    workspace_root: Path,
) -> tuple[NickaseCatalog, list[Path]]:
    catalogs: list[NickaseCatalog] = []
    resolved_paths: list[Path] = []
    for builtin_preset_id in [preset_id, *(additional_preset_ids or [])]:
        if builtin_preset_id:
            catalogs.append(load_builtin_nickase_catalog_preset(builtin_preset_id))
    for raw_path in additional_paths:
        resolved = resolve_workspace_relative_path(
            raw_path,
            workspace_root=workspace_root,
            label="catalog.additional_paths",
        )
        resolved_paths.append(resolved)
        catalogs.append(load_nickase_catalog(resolved))
    if not catalogs:
        raise NickaseCatalogError("Solve catalogs must define at least one preset or additional catalog path.")
    return merge_nickase_catalogs(*catalogs), resolved_paths


def dump_nickase_catalog_payload(catalog: NickaseCatalog) -> dict[str, Any]:
    return {
        "nickases": {
            "schema_version": catalog.schema_version,
            "preset_id": catalog.preset_id,
            "preset_ids": list(catalog.preset_ids),
            "catalog_version": catalog.catalog_version,
            "generated_from": catalog.generated_from,
            "generated_on": catalog.generated_on,
            "normalization_policy": catalog.normalization_policy,
            "entries": [entry.model_dump(mode="json") for entry in catalog.entries],
            "product_aliases": [alias.model_dump(mode="json") for alias in catalog.product_aliases],
        }
    }


def dump_nickase_catalog_yaml(catalog: NickaseCatalog) -> str:
    return yaml.safe_dump(dump_nickase_catalog_payload(catalog), sort_keys=False)
