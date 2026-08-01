"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/selected_lineage.py

Typed lineage projection for selected materialized MSD variants.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal

from Bio import SeqIO
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from ..catalog.strict_mapping_io import DuplicateMappingKeyError, load_unique_yaml

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
StemBase = Annotated[str, StringConstraints(pattern=r"^[ACGT]+$")]
ScarNickProfile = Annotated[str, StringConstraints(pattern=r"^[MWX]{4}$")]

_VARIANT_ID_RE = re.compile(r"^retron(?P<number>\d+)$")
_DISPLAY_ID_RE = re.compile(r"^pES-retron-(?P<number>\d+)$")
_SOURCE_RECORD_ID_RE = re.compile(r"^msd-retron-(?P<number>\d+)$")


class MaterializedVariantLineageError(ValueError):
    """Raised when a selected materialized-variant lineage is inconsistent."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MsdStructuralPrimitiveRefsV1(_StrictModel):
    """Stable identifiers for the payload and structural primitives used by one MSD."""

    scaffold_context_id: NonBlank
    payload_id: NonBlank
    cap_id: NonBlank
    cap_selector_id: NonBlank
    stem_base_selector_id: NonBlank
    left_stem_base_5to3: StemBase
    right_stem_base_5to3: StemBase
    scar_nick_profile_s3s2s1s0: ScarNickProfile
    literal_stem_base_source_id: NonBlank | None = None


class MaterializedVariantLineageEntryV1(_StrictModel):
    """Exact source-to-materialized-variant mapping for one MSD record."""

    variant_id: NonBlank
    display_id: NonBlank
    source_record_id: NonBlank
    design_set_ref: NonBlank
    compiler_spec_ref: NonBlank
    deliverable_plan_ref: NonBlank
    deliverable_variant_key: NonBlank
    source_construct_id: NonBlank
    source_msd_design_id: NonBlank
    source_precedent_id: NonBlank
    primitives: MsdStructuralPrimitiveRefsV1
    source_genbank_ref: NonBlank
    source_genbank_sha256: Sha256
    source_sequence_sha256: Sha256
    msd_region_record_ref: NonBlank
    msd_sequence_sha256: Sha256

    @model_validator(mode="after")
    def _identity_numbers_agree(self) -> "MaterializedVariantLineageEntryV1":
        variant = _matched_number(_VARIANT_ID_RE, self.variant_id, field="variant_id")
        display = _matched_number(_DISPLAY_ID_RE, self.display_id, field="display_id")
        source_record = _matched_number(_SOURCE_RECORD_ID_RE, self.source_record_id, field="source_record_id")
        if len({variant, display, source_record}) != 1:
            raise ValueError("variant_id, display_id, and source_record_id must encode the same retron number.")
        return self


class MaterializedVariantLineageV1(_StrictModel):
    """Hairpin-study projection from a selected cohort to source-owned records."""

    contract: Literal["retron_hairpin_materialized_variant_lineage_v1"]
    schema_version: Literal[1] = 1
    owner_study_id: Literal["retron_hairpin_design"]
    source_bundle_manifest_ref: NonBlank
    selected_variant_ids: tuple[NonBlank, ...]
    expected_selected_variant_count: int = Field(gt=0)
    entries: tuple[MaterializedVariantLineageEntryV1, ...]

    @model_validator(mode="after")
    def _selection_and_entries_are_complete_and_unique(self) -> "MaterializedVariantLineageV1":
        if len(self.selected_variant_ids) != self.expected_selected_variant_count:
            raise ValueError(
                "expected_selected_variant_count="
                f"{self.expected_selected_variant_count} but found {len(self.selected_variant_ids)} selected IDs."
            )
        if len(self.selected_variant_ids) != len(set(self.selected_variant_ids)):
            raise ValueError("selected_variant_ids contain duplicates.")
        if len(self.entries) != self.expected_selected_variant_count:
            raise ValueError(
                "expected_selected_variant_count="
                f"{self.expected_selected_variant_count} but found {len(self.entries)} entries."
            )
        for field in (
            "variant_id",
            "display_id",
            "source_record_id",
            "source_genbank_ref",
            "msd_region_record_ref",
        ):
            values = [getattr(entry, field) for entry in self.entries]
            if len(values) != len(set(values)):
                raise ValueError(f"entries contain duplicate {field} values.")
        entry_ids = {entry.variant_id for entry in self.entries}
        selected_ids = set(self.selected_variant_ids)
        if entry_ids != selected_ids:
            raise ValueError(
                "selected_variant_ids must exactly match entry variant IDs: "
                f"missing={sorted(selected_ids - entry_ids)}, unselected={sorted(entry_ids - selected_ids)}."
            )
        return self


def load_materialized_variant_lineage(
    path: str | Path,
    *,
    repo_root: str | Path,
) -> MaterializedVariantLineageV1:
    """Load and verify selected lineage against its source manifest and owner artifacts."""

    root = Path(repo_root).expanduser().resolve()
    lineage_path = _repo_file(root, path, field="materialized-variant lineage")
    try:
        payload = load_unique_yaml(lineage_path)
        lineage = MaterializedVariantLineageV1.model_validate(payload)
    except (DuplicateMappingKeyError, OSError, ValidationError) as exc:
        raise MaterializedVariantLineageError(f"Invalid materialized-variant lineage {lineage_path}: {exc}") from exc
    _validate_lineage(lineage, repo_root=root)
    return lineage


def _validate_lineage(
    lineage: MaterializedVariantLineageV1,
    *,
    repo_root: Path,
) -> None:
    manifest_path = _linked_file(
        repo_root,
        lineage.source_bundle_manifest_ref,
        repo_root=repo_root,
        field="source_bundle_manifest_ref",
    )
    manifest = _load_mapping(manifest_path, label="MSD-region bundle manifest")
    _require_value(manifest, "contract", "retron_msd_region_record_bundle_v1", label=manifest_path.as_posix())
    manifest_ids = _manifest_variant_ids(manifest.get("records"))
    selected_ids = set(lineage.selected_variant_ids)
    missing = selected_ids - manifest_ids
    if missing:
        raise MaterializedVariantLineageError(
            f"MSD-region source manifest is missing selected variant IDs: {sorted(missing)}."
        )

    cache: dict[Path, dict[str, Any]] = {manifest_path: manifest}
    for entry in lineage.entries:
        _validate_entry(
            entry,
            manifest=manifest,
            bundle_root=manifest_path.parent,
            repo_root=repo_root,
            cache=cache,
        )


def _manifest_variant_ids(raw: object) -> set[str]:
    if not isinstance(raw, list):
        raise MaterializedVariantLineageError("MSD-region source manifest records must be a list.")
    variant_ids: list[str] = []
    for position, row in enumerate(raw):
        if not isinstance(row, Mapping):
            raise MaterializedVariantLineageError(f"MSD-region source manifest records[{position}] must be a mapping.")
        variant_id = row.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id.strip():
            raise MaterializedVariantLineageError(
                f"MSD-region source manifest records[{position}].variant_id must be a non-empty string."
            )
        variant_ids.append(variant_id)
    if len(variant_ids) != len(set(variant_ids)):
        raise MaterializedVariantLineageError("MSD-region source manifest record variant IDs must be unique.")
    return set(variant_ids)


def _validate_entry(
    entry: MaterializedVariantLineageEntryV1,
    *,
    manifest: Mapping[str, Any],
    bundle_root: Path,
    repo_root: Path,
    cache: dict[Path, dict[str, Any]],
) -> None:
    label = entry.variant_id
    design_set_path = _linked_file(
        repo_root,
        entry.design_set_ref,
        repo_root=repo_root,
        field=f"{label}.design_set_ref",
    )
    compiler_spec_path = _linked_file(
        repo_root,
        entry.compiler_spec_ref,
        repo_root=repo_root,
        field=f"{label}.compiler_spec_ref",
    )
    deliverable_plan_path = _linked_file(
        repo_root,
        entry.deliverable_plan_ref,
        repo_root=repo_root,
        field=f"{label}.deliverable_plan_ref",
    )
    source_genbank_path = _linked_file(
        repo_root,
        entry.source_genbank_ref,
        repo_root=repo_root,
        field=f"{label}.source_genbank_ref",
    )
    msd_record_path = _linked_file(
        repo_root,
        entry.msd_region_record_ref,
        repo_root=repo_root,
        field=f"{label}.msd_region_record_ref",
    )

    design_set = _cached_mapping(cache, design_set_path, label="Retron MSD design set")
    compiler_spec = _cached_mapping(cache, compiler_spec_path, label="Retron MSD compiler spec")
    deliverable_plan = _cached_mapping(cache, deliverable_plan_path, label="Retron hairpin deliverable plan")
    msd_record = _cached_mapping(cache, msd_record_path, label="MSD-region variant record")

    _require_value(design_set, "contract", "retron_msd_design_set_v1", label=design_set_path.as_posix())
    _require_value(design_set, "study_id", "retron_hairpin_design", label=design_set_path.as_posix())
    _require_value(design_set, "compiler_spec_ref", entry.compiler_spec_ref, label=design_set_path.as_posix())
    _require_value(design_set, "deliverable_plan_ref", entry.deliverable_plan_ref, label=design_set_path.as_posix())
    _require_value(compiler_spec, "contract", "retron_msd_compiler_spec_v1", label=compiler_spec_path.as_posix())
    _require_value(
        deliverable_plan, "contract", "retron_hairpin_deliverable_plan_v1", label=deliverable_plan_path.as_posix()
    )
    _require_value(deliverable_plan, "study_id", "retron_hairpin_design", label=deliverable_plan_path.as_posix())
    _require_value(deliverable_plan, "design_set_ref", entry.design_set_ref, label=deliverable_plan_path.as_posix())
    _require_value(
        deliverable_plan,
        "compiler_spec_ref",
        entry.compiler_spec_ref,
        label=deliverable_plan_path.as_posix(),
    )

    design = _unique_record(
        design_set.get("designs"),
        key="construct_id",
        value=entry.source_construct_id,
        label=f"{label} design-set source construct",
    )
    compiler_design = _unique_record(
        compiler_spec.get("designs"),
        key="construct_id",
        value=entry.source_construct_id,
        label=f"{label} compiler-spec source construct",
    )
    _validate_design_projection(entry, design=design, compiler_design=compiler_design)
    _validate_deliverable_projection(entry, deliverable_plan=deliverable_plan)
    _validate_msd_record(entry, record=msd_record, source_genbank_path=source_genbank_path)
    _validate_manifest_projection(
        entry,
        manifest=manifest,
        source_genbank_path=source_genbank_path,
        msd_record_path=msd_record_path,
        bundle_root=bundle_root,
        repo_root=repo_root,
    )


def _validate_design_projection(
    entry: MaterializedVariantLineageEntryV1,
    *,
    design: Mapping[str, Any],
    compiler_design: Mapping[str, Any],
) -> None:
    primitive_fields = {
        "scaffold_context": entry.primitives.scaffold_context_id,
        "payload_id": entry.primitives.payload_id,
        "cap_id": entry.primitives.cap_id,
        "left_base": entry.primitives.left_stem_base_5to3,
        "right_base": entry.primitives.right_stem_base_5to3,
    }
    for field, expected in primitive_fields.items():
        _require_value(design, field, expected, label=f"{entry.variant_id} design set")
        _require_value(compiler_design, field, expected, label=f"{entry.variant_id} compiler spec")
    _require_value(
        design,
        "profile_s3s2s1s0",
        entry.primitives.scar_nick_profile_s3s2s1s0,
        label=f"{entry.variant_id} design set",
    )
    _require_value(
        design,
        "expected_msd_design_id",
        entry.source_msd_design_id,
        label=f"{entry.variant_id} design set",
    )
    _require_value(
        compiler_design,
        "cap_selector_id",
        entry.primitives.cap_selector_id,
        label=f"{entry.variant_id} compiler spec",
    )
    _require_value(
        compiler_design,
        "stem_base_selector_id",
        entry.primitives.stem_base_selector_id,
        label=f"{entry.variant_id} compiler spec",
    )
    _require_value(
        compiler_design,
        "literal_stem_base_source_id",
        entry.primitives.literal_stem_base_source_id,
        label=f"{entry.variant_id} compiler spec",
    )


def _validate_deliverable_projection(
    entry: MaterializedVariantLineageEntryV1,
    *,
    deliverable_plan: Mapping[str, Any],
) -> None:
    families = _require_mapping(deliverable_plan.get("artifact_families"), label="deliverable artifact_families")
    handoff = _require_mapping(families.get("benchling_genbank_import"), label="benchling_genbank_import")
    expected_by_field = {
        "assigned_retron_ids": entry.display_id,
        "record_ids": entry.source_record_id,
        "source_precedent_ids": entry.source_precedent_id,
    }
    for field, expected in expected_by_field.items():
        values = _require_mapping(handoff.get(field), label=f"benchling_genbank_import.{field}")
        _require_value(values, entry.deliverable_variant_key, expected, label=f"{entry.variant_id} {field}")


def _validate_msd_record(
    entry: MaterializedVariantLineageEntryV1,
    *,
    record: Mapping[str, Any],
    source_genbank_path: Path,
) -> None:
    _require_value(record, "contract", "retron_msd_region_record_v1", label=entry.msd_region_record_ref)
    _require_value(record, "variant_id", entry.variant_id, label=entry.msd_region_record_ref)
    _require_value(record, "display_id", entry.display_id, label=entry.msd_region_record_ref)
    _require_value(record, "source_record_id", entry.source_record_id, label=entry.msd_region_record_ref)
    _require_value(
        record,
        "source_sequence_sha256",
        entry.source_sequence_sha256,
        label=entry.msd_region_record_ref,
    )
    _require_value(record, "msd_sequence_sha256", entry.msd_sequence_sha256, label=entry.msd_region_record_ref)

    genbank_digest = sha256(source_genbank_path.read_bytes()).hexdigest()
    if genbank_digest != entry.source_genbank_sha256:
        raise MaterializedVariantLineageError(
            f"{entry.variant_id} source GenBank digest drift: {genbank_digest} != {entry.source_genbank_sha256}"
        )
    genbank_records = list(SeqIO.parse(source_genbank_path, "genbank"))
    if len(genbank_records) != 1:
        raise MaterializedVariantLineageError(
            f"{entry.variant_id} source GenBank must contain exactly one record, found {len(genbank_records)}."
        )
    genbank_record = genbank_records[0]
    if genbank_record.id != entry.source_record_id:
        raise MaterializedVariantLineageError(
            f"{entry.variant_id} source GenBank record id drift: {genbank_record.id} != {entry.source_record_id}"
        )
    source_sequence_digest = sha256(str(genbank_record.seq).upper().encode("ascii")).hexdigest()
    if source_sequence_digest != entry.source_sequence_sha256:
        raise MaterializedVariantLineageError(
            f"{entry.variant_id} source sequence digest drift: "
            f"{source_sequence_digest} != {entry.source_sequence_sha256}"
        )
    msd_sequence = str(record.get("msd_sequence_5to3") or "").strip().upper()
    msd_sequence_digest = sha256(msd_sequence.encode("ascii")).hexdigest()
    if msd_sequence_digest != entry.msd_sequence_sha256:
        raise MaterializedVariantLineageError(
            f"{entry.variant_id} MSD sequence digest drift: {msd_sequence_digest} != {entry.msd_sequence_sha256}"
        )


def _validate_manifest_projection(
    entry: MaterializedVariantLineageEntryV1,
    *,
    manifest: Mapping[str, Any],
    source_genbank_path: Path,
    msd_record_path: Path,
    bundle_root: Path,
    repo_root: Path,
) -> None:
    source_input = _unique_record(
        manifest.get("source_inputs"),
        key="variant_id",
        value=entry.variant_id,
        label=f"{entry.variant_id} manifest source input",
    )
    _require_value(source_input, "display_id", entry.display_id, label=f"{entry.variant_id} manifest source input")
    _require_value(
        source_input,
        "source_sha256",
        entry.source_genbank_sha256,
        label=f"{entry.variant_id} manifest source input",
    )
    linked_source = _linked_file(
        bundle_root / "source_inputs",
        source_input.get("source_file"),
        repo_root=repo_root,
        field=f"{entry.variant_id} manifest source_file",
    )
    if linked_source != source_genbank_path:
        raise MaterializedVariantLineageError(
            f"{entry.variant_id} manifest source_file points to {linked_source}, not {source_genbank_path}."
        )

    record_row = _unique_record(
        manifest.get("records"),
        key="variant_id",
        value=entry.variant_id,
        label=f"{entry.variant_id} manifest record",
    )
    _require_value(record_row, "display_id", entry.display_id, label=f"{entry.variant_id} manifest record")
    _require_value(
        record_row,
        "msd_sequence_sha256",
        entry.msd_sequence_sha256,
        label=f"{entry.variant_id} manifest record",
    )
    linked_record = _linked_file(
        bundle_root,
        record_row.get("record"),
        repo_root=repo_root,
        field=f"{entry.variant_id} manifest record path",
    )
    if linked_record != msd_record_path:
        raise MaterializedVariantLineageError(
            f"{entry.variant_id} manifest record points to {linked_record}, not {msd_record_path}."
        )


def _cached_mapping(cache: dict[Path, dict[str, Any]], path: Path, *, label: str) -> dict[str, Any]:
    if path not in cache:
        cache[path] = _load_mapping(path, label=label)
    return cache[path]


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = load_unique_yaml(path)
    except (DuplicateMappingKeyError, OSError) as exc:
        raise MaterializedVariantLineageError(f"Could not load {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MaterializedVariantLineageError(f"{label} must be a mapping: {path}")
    return payload


def _repo_file(repo_root: Path, raw: object, *, field: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise MaterializedVariantLineageError(f"Missing {field} path.")
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            resolved = candidate.expanduser().resolve(strict=True)
        except OSError as exc:
            raise MaterializedVariantLineageError(f"{field} path does not exist: {candidate}") from exc
    else:
        try:
            resolved = (repo_root / candidate).resolve(strict=True)
        except OSError as exc:
            raise MaterializedVariantLineageError(f"{field} path does not exist: {candidate}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise MaterializedVariantLineageError(f"{field} path escapes the repository: {candidate}") from exc
    if not resolved.is_file():
        raise MaterializedVariantLineageError(f"{field} path is not a file: {candidate}")
    return resolved


def _linked_file(base: Path, raw: object, *, repo_root: Path, field: str) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise MaterializedVariantLineageError(f"Missing {field} path.")
    candidate = Path(value)
    if candidate.is_absolute():
        raise MaterializedVariantLineageError(f"{field} must be relative: {candidate}")
    try:
        resolved = (base / candidate).resolve(strict=True)
    except OSError as exc:
        raise MaterializedVariantLineageError(f"{field} path does not exist: {candidate}") from exc
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise MaterializedVariantLineageError(f"{field} path escapes the repository: {candidate}") from exc
    if not resolved.is_file():
        raise MaterializedVariantLineageError(f"{field} path is not a file: {candidate}")
    return resolved


def _unique_record(raw: object, *, key: str, value: str, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, list):
        raise MaterializedVariantLineageError(f"{label} source must be a list.")
    matches = [item for item in raw if isinstance(item, Mapping) and item.get(key) == value]
    if len(matches) != 1:
        raise MaterializedVariantLineageError(f"{label} expected exactly one {key}={value!r}, found {len(matches)}.")
    return matches[0]


def _require_mapping(raw: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise MaterializedVariantLineageError(f"{label} must be a mapping.")
    return raw


def _require_value(source: Mapping[str, Any], field: str, expected: object, *, label: str) -> None:
    actual = source.get(field)
    if actual != expected:
        raise MaterializedVariantLineageError(f"{label} {field} drift: {actual!r} != {expected!r}.")


def _matched_number(pattern: re.Pattern[str], value: str, *, field: str) -> str:
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError(f"{field} has invalid form: {value!r}.")
    return match.group("number")


__all__ = [
    "MaterializedVariantLineageEntryV1",
    "MaterializedVariantLineageError",
    "MaterializedVariantLineageV1",
    "MsdStructuralPrimitiveRefsV1",
    "load_materialized_variant_lineage",
]
