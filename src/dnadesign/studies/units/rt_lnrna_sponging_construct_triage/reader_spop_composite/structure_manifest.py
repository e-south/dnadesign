"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/structure_manifest.py

Retron-hairpin structure thumbnail manifest for the Reader SPOP composite.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .identifiers import assay_subject_key_for_display_id, display_id_for_assay_subject, variant_sort_key
from .paths import DEFAULT_HAIRPIN_OUTPUT_DIR, DEFAULT_MSD_REGION_RECORD_DIR, relative_path, resolve_repo_root

STRUCTURE_THUMBNAIL_MANIFEST_TABLE = "retron_structure_thumbnail_manifest.parquet"


class RetronStructureManifestError(ValueError):
    """Raised when retron-hairpin outputs cannot satisfy the manifest contract."""


@dataclass(frozen=True, slots=True)
class RetronStructureThumbnailRow:
    assay_subject_key: str
    display_variant_id: str
    hairpin_variant_id: str | None
    construct_id: str | None
    source_precedent_id: str | None
    sequence_sha256: str | None
    sequence_length_nt: int | None
    folding_status: str
    structure_status: str
    structure_png_path: str
    composition_png_path: str
    source_bundle_path: str
    review_manifest_path: str
    structure_svg_path: str = ""
    left_base_sequence: str = ""
    stem_length_bp: int | None = None
    foldback_sequence: str = ""
    right_base_sequence: str = ""
    primitive_source_path: str = ""
    primitive_warning: str = ""
    stem_extension_pairing_status: str = ""
    payload_pairing_status: str = ""
    foldback_pairing_status: str = ""
    pairing_summary: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_retron_structure_thumbnail_manifest(
    *,
    repo_root: Path | None = None,
    assay_subject_keys: Sequence[str],
    hairpin_output_dir: Path | None = None,
) -> tuple[RetronStructureThumbnailRow, ...]:
    """Build a plot-facing thumbnail manifest from retron-hairpin outputs."""

    root = resolve_repo_root(repo_root)
    hairpin_root = root / (hairpin_output_dir or DEFAULT_HAIRPIN_OUTPUT_DIR)
    review_manifest_path = hairpin_root / "reviews/review_manifest.json"
    direct_sequence_index_path = hairpin_root / "manifest/indexes/sequence_index.tsv"
    if not review_manifest_path.exists() and direct_sequence_index_path.exists():
        return _build_manifest_from_sequence_index(
            root=root,
            materialized_root=hairpin_root,
            sequence_index_path=direct_sequence_index_path,
            assay_subject_keys=assay_subject_keys,
        )
    review_manifest = _read_json(review_manifest_path)
    display_by_hairpin_variant = dict(review_manifest["sequence_montage"]["review_variant_ids"])
    source_precedent_by_hairpin_variant = dict(
        review_manifest.get("benchling_genbank_import", {}).get("source_precedent_ids", {})
    )
    handoff_path = hairpin_root / "reviews/handoff/teto_pwm_trim_rescue_v1.handoff.tsv"
    handoff_by_variant = _load_tsv_by_key(
        handoff_path,
        key="variant_id",
    )
    materialized_root = hairpin_root / "materialized"
    sequence_index_path = materialized_root / "manifest/indexes/sequence_index.tsv"
    sequence_by_construct = _load_tsv_by_key(
        sequence_index_path,
        key="construct_id",
    )
    reference_by_construct = _load_optional_tsv_by_key(
        materialized_root / "manifest/indexes/reference_index.tsv",
        key="construct_id",
    )
    msd_region_record_by_construct = _load_msd_region_records_by_display_id(root=root)
    hairpin_variant_by_assay_subject = {
        assay_subject_key_for_display_id(display_id): hairpin_variant_id
        for hairpin_variant_id, display_id in display_by_hairpin_variant.items()
    }
    rows: list[RetronStructureThumbnailRow] = []
    for assay_subject_key in sorted(set(assay_subject_keys), key=variant_sort_key):
        hairpin_variant_id = hairpin_variant_by_assay_subject.get(assay_subject_key)
        if hairpin_variant_id is None:
            rows.append(
                RetronStructureThumbnailRow(
                    assay_subject_key=assay_subject_key,
                    display_variant_id=display_id_for_assay_subject(assay_subject_key),
                    hairpin_variant_id=None,
                    construct_id=None,
                    source_precedent_id=None,
                    sequence_sha256=None,
                    sequence_length_nt=None,
                    folding_status="missing",
                    structure_status="missing_hairpin_materialization",
                    structure_png_path="",
                    structure_svg_path="",
                    composition_png_path="",
                    source_bundle_path="",
                    review_manifest_path=relative_path(review_manifest_path, root),
                )
            )
            continue
        handoff = _require_row(handoff_by_variant, key=hairpin_variant_id, source=handoff_path)
        construct_id = _require_field(handoff, field="construct_id", row_key=hairpin_variant_id, source=handoff_path)
        sequence = _require_row(sequence_by_construct, key=construct_id, source=sequence_index_path)
        structure_path = materialized_root / _require_field(
            sequence,
            field="secondary_structure_native_png",
            row_key=construct_id,
            source=sequence_index_path,
        )
        composition_path = materialized_root / _require_field(
            sequence,
            field="composition_overview_png",
            row_key=construct_id,
            source=sequence_index_path,
        )
        source_bundle_path = materialized_root / _require_field(
            sequence,
            field="artifact_bundle",
            row_key=construct_id,
            source=sequence_index_path,
        )
        structure_svg_path = _native_structure_svg_path(
            sequence=sequence,
            materialized_root=materialized_root,
            root=root,
        )
        sequence_sha256 = _require_field(
            sequence, field="sequence_sha256", row_key=construct_id, source=sequence_index_path
        )
        sequence_length = _require_field(
            sequence, field="sequence_length", row_key=construct_id, source=sequence_index_path
        )
        folding_status = _require_field(
            sequence, field="folding_status", row_key=construct_id, source=sequence_index_path
        )
        primitive_fields = _primitive_fields(
            sequence=sequence,
            reference_by_construct=reference_by_construct,
            msd_region_record_by_construct=msd_region_record_by_construct,
            construct_id=construct_id,
            materialized_root=materialized_root,
            root=root,
            source=sequence_index_path,
        )
        rows.append(
            RetronStructureThumbnailRow(
                assay_subject_key=assay_subject_key,
                display_variant_id=display_by_hairpin_variant[hairpin_variant_id],
                hairpin_variant_id=hairpin_variant_id,
                construct_id=construct_id,
                source_precedent_id=source_precedent_by_hairpin_variant.get(
                    hairpin_variant_id,
                    display_by_hairpin_variant[hairpin_variant_id],
                ),
                sequence_sha256=sequence_sha256,
                sequence_length_nt=int(sequence_length),
                folding_status=folding_status,
                structure_status="available" if structure_path.exists() else "missing_thumbnail_path",
                structure_png_path=relative_path(structure_path, root),
                structure_svg_path=structure_svg_path,
                composition_png_path=relative_path(composition_path, root),
                source_bundle_path=relative_path(source_bundle_path, root),
                review_manifest_path=relative_path(review_manifest_path, root),
                **primitive_fields,
            )
        )
    return tuple(rows)


def _build_manifest_from_sequence_index(
    *,
    root: Path,
    materialized_root: Path,
    sequence_index_path: Path,
    assay_subject_keys: Sequence[str],
) -> tuple[RetronStructureThumbnailRow, ...]:
    sequence_by_construct = _load_tsv_by_key(
        sequence_index_path,
        key="construct_id",
    )
    sequence_by_assay_subject = {
        assay_subject_key_for_display_id(construct_id): row for construct_id, row in sequence_by_construct.items()
    }
    reference_by_construct = _load_optional_tsv_by_key(
        materialized_root / "manifest/indexes/reference_index.tsv",
        key="construct_id",
    )
    msd_region_record_by_construct = _load_msd_region_records_by_display_id(root=root)
    rows: list[RetronStructureThumbnailRow] = []
    for assay_subject_key in sorted(set(assay_subject_keys), key=variant_sort_key):
        sequence = sequence_by_assay_subject.get(assay_subject_key)
        display_id = display_id_for_assay_subject(assay_subject_key)
        if sequence is None:
            rows.append(
                RetronStructureThumbnailRow(
                    assay_subject_key=assay_subject_key,
                    display_variant_id=display_id,
                    hairpin_variant_id=None,
                    construct_id=None,
                    source_precedent_id=None,
                    sequence_sha256=None,
                    sequence_length_nt=None,
                    folding_status="missing",
                    structure_status="missing_hairpin_materialization",
                    structure_png_path="",
                    structure_svg_path="",
                    composition_png_path="",
                    source_bundle_path="",
                    review_manifest_path=relative_path(sequence_index_path, root),
                )
            )
            continue
        construct_id = _require_field(sequence, field="construct_id", row_key=display_id, source=sequence_index_path)
        structure_path = materialized_root / _require_field(
            sequence,
            field="secondary_structure_native_png",
            row_key=construct_id,
            source=sequence_index_path,
        )
        composition_path = materialized_root / _require_field(
            sequence,
            field="composition_overview_png",
            row_key=construct_id,
            source=sequence_index_path,
        )
        source_bundle_path = materialized_root / _require_field(
            sequence,
            field="artifact_bundle",
            row_key=construct_id,
            source=sequence_index_path,
        )
        structure_svg_path = _native_structure_svg_path(
            sequence=sequence,
            materialized_root=materialized_root,
            root=root,
        )
        sequence_sha256 = _require_field(
            sequence, field="sequence_sha256", row_key=construct_id, source=sequence_index_path
        )
        sequence_length = _require_field(
            sequence, field="sequence_length", row_key=construct_id, source=sequence_index_path
        )
        folding_status = _require_field(
            sequence, field="folding_status", row_key=construct_id, source=sequence_index_path
        )
        primitive_fields = _primitive_fields(
            sequence=sequence,
            reference_by_construct=reference_by_construct,
            msd_region_record_by_construct=msd_region_record_by_construct,
            construct_id=construct_id,
            materialized_root=materialized_root,
            root=root,
            source=sequence_index_path,
        )
        rows.append(
            RetronStructureThumbnailRow(
                assay_subject_key=assay_subject_key,
                display_variant_id=construct_id,
                hairpin_variant_id=assay_subject_key,
                construct_id=construct_id,
                source_precedent_id=construct_id,
                sequence_sha256=sequence_sha256,
                sequence_length_nt=int(sequence_length),
                folding_status=folding_status,
                structure_status="available" if structure_path.exists() else "missing_thumbnail_path",
                structure_png_path=relative_path(structure_path, root),
                structure_svg_path=structure_svg_path,
                composition_png_path=relative_path(composition_path, root),
                source_bundle_path=relative_path(source_bundle_path, root),
                review_manifest_path=relative_path(sequence_index_path, root),
                **primitive_fields,
            )
        )
    return tuple(rows)


def write_retron_structure_thumbnail_manifest(
    rows: Sequence[RetronStructureThumbnailRow],
    *,
    output_dir: Path,
) -> str:
    """Write the thumbnail manifest table and return its path."""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    path = resolved_output_dir / STRUCTURE_THUMBNAIL_MANIFEST_TABLE
    pq.write_table(_thumbnail_table(rows), path)
    return path.as_posix()


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise RetronStructureManifestError(f"required retron-hairpin manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_msd_region_records_by_display_id(*, root: Path) -> dict[str, dict[str, object]]:
    record_root = root / DEFAULT_MSD_REGION_RECORD_DIR
    manifest_path = record_root / "manifest.yaml"
    if not manifest_path.exists():
        return {}
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RetronStructureManifestError(f"{manifest_path} is not a YAML mapping")
    rows = manifest.get("records", [])
    if not isinstance(rows, list):
        raise RetronStructureManifestError(f"{manifest_path} field 'records' is not a list")
    by_display_id: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RetronStructureManifestError(f"{manifest_path} contains a non-mapping record row")
        display_id = row.get("display_id")
        record = row.get("record")
        if not isinstance(display_id, str) or not isinstance(record, str):
            raise RetronStructureManifestError(f"{manifest_path} has a record row missing display_id or record")
        record_path = record_root / record
        payload = yaml.safe_load(record_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RetronStructureManifestError(f"{record_path} is not a YAML mapping")
        payload["_record_path"] = record_path
        by_display_id[display_id] = payload
    return by_display_id


def _load_tsv_by_key(path: Path, *, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise RetronStructureManifestError(f"required retron-hairpin table is missing: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or key not in reader.fieldnames:
            raise RetronStructureManifestError(f"{path} is missing required column {key!r}")
        return {row[key]: row for row in reader}


def _load_optional_tsv_by_key(path: Path, *, key: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return _load_tsv_by_key(path, key=key)


def _require_row(mapping: dict[str, dict[str, str]], *, key: str, source: Path) -> dict[str, str]:
    try:
        return mapping[key]
    except KeyError as exc:
        raise RetronStructureManifestError(f"{source} is missing required row {key!r}") from exc


def _require_field(row: dict[str, str], *, field: str, row_key: str, source: Path) -> str:
    value = row.get(field)
    if value is None or value == "":
        raise RetronStructureManifestError(f"{source} row {row_key!r} is missing required field {field!r}")
    return value


def _native_structure_svg_path(
    *,
    sequence: dict[str, str],
    materialized_root: Path,
    root: Path,
) -> str:
    artifact_bundle = sequence.get("artifact_bundle")
    if not artifact_bundle:
        return ""
    bundle_root = materialized_root / artifact_bundle
    candidates = (
        bundle_root / "manifest/visual/secondary_structure/native.svg",
        bundle_root / "runtime/construct/visual/viennarna_secondary_structure/secondary_structure.native.svg",
    )
    for candidate in candidates:
        if candidate.exists():
            return relative_path(candidate, root)
    return ""


def _primitive_fields(
    *,
    sequence: dict[str, str],
    reference_by_construct: dict[str, dict[str, str]],
    msd_region_record_by_construct: dict[str, dict[str, object]],
    construct_id: str,
    materialized_root: Path,
    root: Path,
    source: Path,
) -> dict[str, object]:
    if reference_by_construct and construct_id not in reference_by_construct:
        raise RetronStructureManifestError(f"{source} has no reference-index row for construct {construct_id!r}")
    reference = reference_by_construct.get(construct_id, {})
    features_path = _features_path(sequence=sequence, materialized_root=materialized_root)
    feature_rows = _feature_rows_by_role(features_path)
    left_base = reference.get("left_base") or _feature_sequence(feature_rows, "stem_base_left")
    right_base = reference.get("right_base") or _feature_sequence(feature_rows, "stem_base_right")
    foldback = _feature_sequence(feature_rows, "snapback_foldback_geometry")
    msd_region_record = msd_region_record_by_construct.get(construct_id, {})
    source_record_path = _source_record_path(msd_region_record)
    pairing_segments = _source_pairing_segments(msd_region_record)
    stem_length, primitive_warning = _stem_length_bp(
        feature_rows=feature_rows,
        pairing_segments=pairing_segments,
        foldback_return_sequence=_source_record_feature_sequence(msd_region_record, "snapback_foldback_return"),
        foldback_retained_stem_sequence=_source_record_feature_sequence(msd_region_record, "snapback_retained_stem"),
        construct_id=construct_id,
        source=source_record_path or features_path or source,
    )
    _validate_stem_base_lengths(
        left_base=left_base,
        right_base=right_base,
        construct_id=construct_id,
        source=source,
    )
    return {
        "left_base_sequence": left_base,
        "stem_length_bp": stem_length,
        "foldback_sequence": foldback,
        "right_base_sequence": right_base,
        "primitive_source_path": _primitive_source_path(
            features_path=features_path,
            source_record_path=source_record_path,
            root=root,
        ),
        "primitive_warning": primitive_warning,
        "stem_extension_pairing_status": _pairing_status(pairing_segments, "stem_extension"),
        "payload_pairing_status": _pairing_status(pairing_segments, "payload_stem"),
        "foldback_pairing_status": _pairing_status(pairing_segments, "foldback_stem"),
        "pairing_summary": _pairing_summary(pairing_segments),
    }


def _features_path(*, sequence: dict[str, str], materialized_root: Path) -> Path:
    if features_csv := sequence.get("features_csv"):
        return materialized_root / features_csv
    artifact_bundle = sequence.get("artifact_bundle", "")
    return materialized_root / artifact_bundle / "sequences/features.csv"


def _feature_rows_by_role(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "role" not in reader.fieldnames or "sequence" not in reader.fieldnames:
            raise RetronStructureManifestError(f"{path} is missing required feature columns 'role' and 'sequence'")
        return {row["role"]: row for row in reader if row.get("role")}


def _feature_sequence(feature_rows: dict[str, dict[str, str]], role: str) -> str:
    row = feature_rows.get(role)
    if not row:
        return ""
    return row.get("sequence", "")


def _source_record_feature_sequence(record: dict[str, object], role: str) -> str:
    features = record.get("features", [])
    if not isinstance(features, list):
        return ""
    for feature in features:
        if isinstance(feature, dict) and feature.get("role") == role:
            value = feature.get("sequence_5to3")
            return value if isinstance(value, str) else ""
    return ""


def _source_pairing_segments(record: dict[str, object]) -> tuple[dict[str, object], ...]:
    segments = record.get("pairing_segments", ())
    if not isinstance(segments, list):
        return ()
    return tuple(segment for segment in segments if isinstance(segment, dict))


def _source_record_path(record: dict[str, object]) -> Path | None:
    value = record.get("_record_path")
    return value if isinstance(value, Path) else None


def _primitive_source_path(*, features_path: Path, source_record_path: Path | None, root: Path) -> str:
    if source_record_path is not None and source_record_path.exists():
        return relative_path(source_record_path, root)
    return relative_path(features_path, root) if features_path.exists() else ""


def _stem_length_bp(
    *,
    feature_rows: dict[str, dict[str, str]],
    pairing_segments: Sequence[dict[str, object]],
    foldback_return_sequence: str,
    foldback_retained_stem_sequence: str,
    construct_id: str,
    source: Path,
) -> tuple[int | None, str]:
    paired_length = _stem_length_from_pairing_segments(pairing_segments)
    if paired_length is not None:
        warning = _pairing_review_warning(pairing_segments, construct_id=construct_id, source=source)
        return paired_length, warning
    payload_primary = _feature_sequence(feature_rows, "payload_primary")
    payload_complement = _feature_sequence(feature_rows, "payload_complement")
    if not payload_primary:
        return None, ""
    if payload_complement and len(payload_primary) != len(payload_complement):
        raise RetronStructureManifestError(
            f"{source} row {construct_id!r} has mismatched stem-body arm lengths: "
            f"payload_primary={len(payload_primary)}, payload_complement={len(payload_complement)}"
        )
    foldback_stem, primitive_warning = _foldback_stem_bp(
        foldback_return_sequence=foldback_return_sequence,
        foldback_retained_stem_sequence=foldback_retained_stem_sequence,
        construct_id=construct_id,
        source=source,
    )
    return len(payload_primary) + foldback_stem, primitive_warning


def _stem_length_from_pairing_segments(pairing_segments: Sequence[dict[str, object]]) -> int | None:
    if not pairing_segments:
        return None
    total = 0
    matched = False
    for segment_name in ("stem_extension", "payload_stem", "foldback_stem"):
        segment = _pairing_segment_row(pairing_segments, segment_name)
        if segment is None:
            continue
        length = segment.get("length_bp")
        if isinstance(length, int):
            total += length
            matched = True
    return total if matched else None


def _pairing_status(pairing_segments: Sequence[dict[str, object]], segment_name: str) -> str:
    segment = _pairing_segment_row(pairing_segments, segment_name)
    if segment is None:
        return ""
    value = segment.get("pairing_status")
    return value if isinstance(value, str) else ""


def _pairing_summary(pairing_segments: Sequence[dict[str, object]]) -> str:
    parts: list[str] = []
    for segment_name in ("stem_extension", "payload_stem", "foldback_stem"):
        segment = _pairing_segment_row(pairing_segments, segment_name)
        if segment is None:
            continue
        status = segment.get("pairing_status")
        wc = segment.get("watson_crick_bp")
        wobble = segment.get("wobble_bp")
        mismatch = segment.get("mismatch_bp")
        if isinstance(status, str) and isinstance(wc, int) and isinstance(wobble, int) and isinstance(mismatch, int):
            parts.append(f"{segment_name}:{status},WC={wc},wobble={wobble},mismatch={mismatch}")
    return "; ".join(parts)


def _pairing_review_warning(pairing_segments: Sequence[dict[str, object]], *, construct_id: str, source: Path) -> str:
    review_segments: list[str] = []
    for segment in pairing_segments:
        status = segment.get("pairing_status")
        segment_name = segment.get("segment")
        if status == "review_required" and isinstance(segment_name, str):
            review_segments.append(segment_name)
    if not review_segments:
        return ""
    return f"pairing_review_required:{','.join(review_segments)},source={source.name},construct={construct_id}"


def _pairing_segment_row(pairing_segments: Sequence[dict[str, object]], segment_name: str) -> dict[str, object] | None:
    for segment in pairing_segments:
        if segment.get("segment") == segment_name:
            return segment
    return None


def _foldback_stem_bp(
    *,
    foldback_return_sequence: str,
    foldback_retained_stem_sequence: str,
    construct_id: str,
    source: Path,
) -> tuple[int, str]:
    if foldback_return_sequence and foldback_retained_stem_sequence:
        if len(foldback_return_sequence) != len(foldback_retained_stem_sequence):
            counted_bp = min(len(foldback_return_sequence), len(foldback_retained_stem_sequence))
            return (
                counted_bp,
                "foldback_stem_length_mismatch:"
                f"return={len(foldback_return_sequence)},"
                f"retained={len(foldback_retained_stem_sequence)},"
                f"counted_shorter_arm={counted_bp},"
                f"source={source.name},"
                f"construct={construct_id}",
            )
        return len(foldback_return_sequence), ""
    return len(foldback_return_sequence or foldback_retained_stem_sequence), ""


def _validate_stem_base_lengths(*, left_base: str, right_base: str, construct_id: str, source: Path) -> None:
    if not left_base and not right_base:
        return
    if len(left_base) != len(right_base):
        raise RetronStructureManifestError(
            f"{source} row {construct_id!r} has mismatched stem-base lengths: left={left_base!r}, right={right_base!r}"
        )


def _thumbnail_table(rows: Sequence[RetronStructureThumbnailRow]) -> pa.Table:
    return pa.Table.from_pylist([row.to_dict() for row in rows], schema=_thumbnail_schema())


def _thumbnail_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("assay_subject_key", pa.string()),
            pa.field("display_variant_id", pa.string()),
            pa.field("hairpin_variant_id", pa.string()),
            pa.field("construct_id", pa.string()),
            pa.field("source_precedent_id", pa.string()),
            pa.field("sequence_sha256", pa.string()),
            pa.field("sequence_length_nt", pa.int64()),
            pa.field("folding_status", pa.string()),
            pa.field("structure_status", pa.string()),
            pa.field("structure_png_path", pa.string()),
            pa.field("structure_svg_path", pa.string()),
            pa.field("composition_png_path", pa.string()),
            pa.field("source_bundle_path", pa.string()),
            pa.field("review_manifest_path", pa.string()),
            pa.field("left_base_sequence", pa.string()),
            pa.field("stem_length_bp", pa.int64()),
            pa.field("foldback_sequence", pa.string()),
            pa.field("right_base_sequence", pa.string()),
            pa.field("primitive_source_path", pa.string()),
            pa.field("primitive_warning", pa.string()),
            pa.field("stem_extension_pairing_status", pa.string()),
            pa.field("payload_pairing_status", pa.string()),
            pa.field("foldback_pairing_status", pa.string()),
            pa.field("pairing_summary", pa.string()),
        ]
    )
