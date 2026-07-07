"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/msd_region_genbank.py

Parse monolithic MSD-region GenBank files into decomposed retron-hairpin records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from .feature_roles import feature_label, normalized_role_for_feature
from .models import (
    MsdRegionAnnotationWarning,
    MsdRegionBundleWriteResult,
    MsdRegionComparisonReport,
    MsdRegionDiscrepancy,
    MsdRegionIngestError,
    MsdRegionSourceBundle,
    NormalizedMsdFeature,
    NormalizedMsdRegionRecord,
    SkippedMsdSourceRecord,
)

_VARIANT_RE = re.compile(r"(?:pES-)?retron[-_ ]?(\d+)|msd[-_ ]?retron[-_ ]?(\d+)", re.IGNORECASE)
_DISPLAY_ID_RE = re.compile(r"pES-retron-(\d+)", re.IGNORECASE)
_PRIMITIVE_ROLES = (
    "stem_base_left",
    "payload_primary",
    "snapback_foldback_geometry",
    "payload_complement",
    "stem_base_right",
)


def parse_msd_region_genbank(path: str | Path) -> MsdRegionSourceBundle:
    """Parse an MSD-region GenBank file into decomposed per-variant records."""

    source_path = Path(path).expanduser().resolve()
    records = list(SeqIO.parse(source_path, "genbank"))
    normalized: list[NormalizedMsdRegionRecord] = []
    skipped: list[SkippedMsdSourceRecord] = []
    seen: set[str] = set()
    for record in records:
        variant_id = _variant_id(record)
        if variant_id is None:
            skipped.append(
                SkippedMsdSourceRecord(
                    record_id=record.id,
                    reason="unresolved_variant_id",
                    sequence_length_nt=len(record.seq),
                )
            )
            continue
        if variant_id in seen:
            raise MsdRegionIngestError(f"Duplicate MSD-region variant id in {source_path}: {variant_id}")
        seen.add(variant_id)
        normalized.append(_normalize_record(record, variant_id=variant_id))
    normalized.sort(key=lambda item: _variant_sort_key(item.variant_id))
    return MsdRegionSourceBundle(
        source_path=source_path.as_posix(),
        source_sha256=_sha256_file(source_path),
        source_record_count=len(records),
        records=tuple(normalized),
        skipped_records=tuple(skipped),
    )


def compiler_spec_payload_from_records(records: Sequence[NormalizedMsdRegionRecord]) -> dict[str, object]:
    """Build an explicit compiler-spec payload from normalized MSD records."""

    designs: list[dict[str, object]] = []
    payload_sequences: dict[str, dict[str, object]] = {}
    cap_sequences: dict[str, dict[str, object]] = {}
    for record in sorted(records, key=lambda item: _variant_sort_key(item.variant_id)):
        number = _variant_number(record.variant_id)
        payload_id = f"MSDRegion{number}_payload"
        cap_id = f"C{number}_msd_region"
        payload = record.primitive("payload_primary").sequence_5to3
        cap = record.primitive("snapback_foldback_geometry").sequence_5to3
        payload_sequences[payload_id] = {
            "sequence": payload,
            "display_name": f"{record.display_id} payload_primary",
            "selection_basis": "msd_region_genbank_annotation",
        }
        cap_sequences[cap_id] = {"sequence": cap}
        designs.append(
            {
                "construct_id": record.display_id,
                "payload_id": payload_id,
                "cap_id": cap_id,
                "left_base": record.primitive("stem_base_left").sequence_5to3.upper(),
                "right_base": record.primitive("stem_base_right").sequence_5to3.upper(),
                "source_notes": (
                    f"Generated from decomposed retron-hairpin MSD-region source record {record.file_stem}.yaml."
                ),
            }
        )
    return {
        "contract": "retron_msd_compiler_spec_v1",
        "schema_version": 1,
        "allow_non_ligatable_s0": True,
        "designs": designs,
        "payload_sequences": payload_sequences,
        "cap_sequences": cap_sequences,
    }


def write_msd_region_record_bundle(
    bundle: MsdRegionSourceBundle,
    *,
    output_dir: str | Path,
    comparison_report: MsdRegionComparisonReport | None = None,
) -> MsdRegionBundleWriteResult:
    """Write decomposed records, manifest, compiler spec, and optional review report."""

    root = Path(output_dir).expanduser().resolve()
    records_dir = root / "variants"
    compiler_dir = root / "compiler"
    reports_dir = root / "reports"
    records_dir.mkdir(parents=True, exist_ok=True)
    compiler_dir.mkdir(parents=True, exist_ok=True)
    variant_paths: dict[str, str] = {}
    for record in bundle.records:
        path = records_dir / f"{record.file_stem}.yaml"
        _write_yaml(path, record.to_dict())
        variant_paths[record.variant_id] = path.as_posix()
    compiler_spec_path = compiler_dir / "reader_spop_msd_structure_panel_v1.spec.yaml"
    _write_yaml(compiler_spec_path, compiler_spec_payload_from_records(bundle.records))
    discrepancy_report_path: str | None = None
    if comparison_report is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "discrepancies.yaml"
        _write_yaml(report_path, comparison_report.to_dict())
        discrepancy_report_path = report_path.as_posix()
    manifest_path = root / "manifest.yaml"
    _write_yaml(
        manifest_path,
        {
            "contract": "retron_msd_region_record_bundle_v1",
            "schema_version": 1,
            "source_policy": "decomposed_records_are_authority",
            "source_path": bundle.source_path,
            "source_sha256": bundle.source_sha256,
            "source_record_count": bundle.source_record_count,
            "included_record_count": bundle.included_record_count,
            "skipped_record_count": len(bundle.skipped_records),
            "skipped_records": [record.to_dict() for record in bundle.skipped_records],
            "compiler_spec": _relative_to(compiler_spec_path, root),
            "discrepancy_report": _relative_to(Path(discrepancy_report_path), root)
            if discrepancy_report_path is not None
            else None,
            "records": [
                {
                    "variant_id": record.variant_id,
                    "display_id": record.display_id,
                    "record": _relative_to(Path(variant_paths[record.variant_id]), root),
                    "annotation_status": record.annotation_status,
                    "annotation_warning_count": len(record.annotation_warnings),
                    "annotation_warnings": [warning.to_dict() for warning in record.annotation_warnings],
                    "msd_sequence_sha256": record.msd_sequence_sha256,
                }
                for record in bundle.records
            ],
        },
    )
    return MsdRegionBundleWriteResult(
        output_dir=root.as_posix(),
        manifest_path=manifest_path.as_posix(),
        compiler_spec_path=compiler_spec_path.as_posix(),
        discrepancy_report_path=discrepancy_report_path,
        variant_record_paths=variant_paths,
    )


def compare_records_to_existing_sources(
    records: Sequence[NormalizedMsdRegionRecord],
    *,
    existing_roots: Sequence[str | Path],
    cap_source_path: str | Path | None = None,
) -> MsdRegionComparisonReport:
    """Compare normalized MSD records to existing materialized/catalog sources."""

    by_variant = {record.variant_id: record for record in records}
    discrepancies: list[MsdRegionDiscrepancy] = []
    comparison_count = 0
    for root in existing_roots:
        for item in _iter_existing_sequence_rows(Path(root).expanduser().resolve()):
            record = by_variant.get(item["variant_id"])
            if record is None:
                continue
            comparison_count += 1
            discrepancies.extend(_compare_existing_sequence(record, item))
    if cap_source_path is not None:
        cap_comparisons, cap_discrepancies = _compare_cap_sources(by_variant, Path(cap_source_path).expanduser())
        comparison_count += cap_comparisons
        discrepancies.extend(cap_discrepancies)
    return MsdRegionComparisonReport(comparison_count=comparison_count, discrepancies=tuple(discrepancies))


def _normalize_record(record: SeqRecord, *, variant_id: str) -> NormalizedMsdRegionRecord:
    source_sequence = str(record.seq).upper()
    display_sequence = str(record.seq.reverse_complement()).upper()
    features = tuple(
        _normalize_feature(feature, display_sequence=display_sequence, source_length=len(record.seq))
        for feature in record.features
    )
    features = tuple(feature for feature in features if feature.label or feature.role)
    features = _with_derived_stem_bases(features, display_sequence=display_sequence, source_length=len(record.seq))
    features = _deduplicate_equivalent_features(features)
    annotation_warnings = _annotation_warnings(features)
    direct_roles = any(feature.source_role for feature in features)
    inferred_roles = any(feature.role is not None and feature.source_role is None for feature in features)
    if direct_roles and inferred_roles:
        annotation_status = "mixed_typed_and_label_normalized"
    elif direct_roles:
        annotation_status = "typed_dnadesign_roles"
    else:
        annotation_status = "label_only_normalized"
    number = _variant_number(variant_id)
    return NormalizedMsdRegionRecord(
        variant_id=variant_id,
        display_id=f"pES-retron-{number}",
        file_stem=f"pes-retron-{number}-msd-region",
        source_record_id=record.id,
        source_description=record.description,
        source_sequence_sha256=_sha256_text(source_sequence),
        msd_sequence_sha256=_sha256_text(display_sequence),
        sequence_length_nt=len(display_sequence),
        msd_sequence_5to3=display_sequence,
        rna_sequence_5to3=display_sequence.replace("T", "U"),
        annotation_status=annotation_status,
        annotation_warnings=annotation_warnings,
        features=features,
    )


def _with_derived_stem_bases(
    features: tuple[NormalizedMsdFeature, ...],
    *,
    display_sequence: str,
    source_length: int,
) -> tuple[NormalizedMsdFeature, ...]:
    roles = {feature.role for feature in features}
    derived: list[NormalizedMsdFeature] = []
    if "stem_base_left" not in roles:
        flank_5p = _single_feature_or_none(features, "flank_5p")
        if flank_5p is not None and flank_5p.display_end_0 - flank_5p.display_start_0 >= 4:
            derived.append(
                _derived_feature(
                    role="stem_base_left",
                    label="Left Base",
                    display_start_0=flank_5p.display_end_0 - 4,
                    display_end_0=flank_5p.display_end_0,
                    display_sequence=display_sequence,
                    source_length=source_length,
                )
            )
        else:
            annotated = _single_feature_or_none(features, "stem_base_left_annotated_span")
            if annotated is not None and annotated.display_end_0 - annotated.display_start_0 >= 4:
                derived.append(
                    _derived_feature(
                        role="stem_base_left",
                        label="Left Base",
                        display_start_0=annotated.display_start_0,
                        display_end_0=annotated.display_start_0 + 4,
                        display_sequence=display_sequence,
                        source_length=source_length,
                    )
                )
    if "stem_base_right" not in roles:
        flank_3p = _single_feature_or_none(features, "flank_3p")
        if flank_3p is not None and flank_3p.display_end_0 - flank_3p.display_start_0 >= 4:
            derived.append(
                _derived_feature(
                    role="stem_base_right",
                    label="Right Base",
                    display_start_0=flank_3p.display_start_0,
                    display_end_0=flank_3p.display_start_0 + 4,
                    display_sequence=display_sequence,
                    source_length=source_length,
                )
            )
        else:
            annotated = _single_feature_or_none(features, "stem_base_right_annotated_span")
            if annotated is not None and annotated.display_end_0 - annotated.display_start_0 >= 4:
                derived.append(
                    _derived_feature(
                        role="stem_base_right",
                        label="Right Base",
                        display_start_0=annotated.display_end_0 - 4,
                        display_end_0=annotated.display_end_0,
                        display_sequence=display_sequence,
                        source_length=source_length,
                    )
                )
    return (*features, *derived)


def _annotation_warnings(features: Sequence[NormalizedMsdFeature]) -> tuple[MsdRegionAnnotationWarning, ...]:
    warnings: list[MsdRegionAnnotationWarning] = []
    by_role = {feature.role: feature for feature in features if feature.role is not None}
    for annotated_role, primitive_role in (
        ("stem_base_left_annotated_span", "stem_base_left"),
        ("stem_base_right_annotated_span", "stem_base_right"),
    ):
        annotated = by_role.get(annotated_role)
        primitive = by_role.get(primitive_role)
        if annotated is None or primitive is None:
            continue
        warnings.append(
            MsdRegionAnnotationWarning(
                kind="stem_base_annotation_span_adjusted",
                role=primitive_role,
                label=annotated.label,
                source_span_0=(annotated.source_start_0, annotated.source_end_0),
                display_span_0=(annotated.display_start_0, annotated.display_end_0),
                annotated_sequence_5to3=annotated.sequence_5to3,
                compiler_sequence_5to3=primitive.sequence_5to3,
                note=(
                    "Source annotation span is not 4 bp; compiler-facing stem base was derived as the "
                    "4 bp boundary sequence."
                ),
            )
        )
    return tuple(warnings)


def _deduplicate_equivalent_features(
    features: tuple[NormalizedMsdFeature, ...],
) -> tuple[NormalizedMsdFeature, ...]:
    by_key: dict[tuple[str | None, int, int, str], NormalizedMsdFeature] = {}
    ordered: list[tuple[str | None, int, int, str]] = []
    for feature in features:
        key = (feature.role, feature.display_start_0, feature.display_end_0, feature.sequence_5to3)
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = feature
            ordered.append(key)
            continue
        if previous.source_role is None and feature.source_role is not None:
            by_key[key] = feature
    return tuple(by_key[key] for key in ordered)


def _single_feature_or_none(features: Sequence[NormalizedMsdFeature], role: str) -> NormalizedMsdFeature | None:
    matches = [feature for feature in features if feature.role == role]
    if len(matches) == 1:
        return matches[0]
    return None


def _derived_feature(
    *,
    role: str,
    label: str,
    display_start_0: int,
    display_end_0: int,
    display_sequence: str,
    source_length: int,
) -> NormalizedMsdFeature:
    source_start_0 = source_length - display_end_0
    source_end_0 = source_length - display_start_0
    return NormalizedMsdFeature(
        role=role,
        source_role=None,
        label=label,
        feature_type="derived_feature",
        source_start_0=source_start_0,
        source_end_0=source_end_0,
        source_strand=None,
        display_start_0=display_start_0,
        display_end_0=display_end_0,
        display_strand=1,
        sequence_5to3=display_sequence[display_start_0:display_end_0].upper(),
    )


def _normalize_feature(feature: Any, *, display_sequence: str, source_length: int) -> NormalizedMsdFeature:
    start, end = _simple_span(feature)
    display_start = source_length - end
    display_end = source_length - start
    labels = _qualifier_values(feature, "label") + _qualifier_values(feature, "note")
    source_roles = _qualifier_values(feature, "dnadesign_role")
    role, source_role = normalized_role_for_feature(
        labels=labels,
        source_roles=source_roles,
        source_start_0=start,
        source_end_0=end,
        source_length=source_length,
        source_strand=feature.location.strand,
    )
    source_strand = feature.location.strand
    display_strand = -source_strand if source_strand in {-1, 1} else source_strand
    return NormalizedMsdFeature(
        role=role,
        source_role=source_role,
        label=feature_label(labels),
        feature_type=str(feature.type),
        source_start_0=start,
        source_end_0=end,
        source_strand=source_strand,
        display_start_0=display_start,
        display_end_0=display_end,
        display_strand=display_strand,
        sequence_5to3=display_sequence[display_start:display_end].upper(),
    )


def _iter_existing_sequence_rows(root: Path) -> Iterable[dict[str, object]]:
    for index_path in (
        root / "manifest/indexes/sequence_index.tsv",
        root / "materialized/manifest/indexes/sequence_index.tsv",
    ):
        if not index_path.exists():
            continue
        bundle_root = index_path.parents[2]
        display_by_variant_key = _review_display_map(root)
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                variant_id = _variant_id_for_existing_row(row, display_by_variant_key)
                if variant_id is None:
                    continue
                genbank_path = bundle_root / str(row.get("genbank", ""))
                features_path = bundle_root / str(row.get("features_csv") or "")
                if not features_path.exists() and genbank_path.exists():
                    features_path = genbank_path.parent / "features.csv"
                yield {
                    "variant_id": variant_id,
                    "index_path": index_path,
                    "genbank_path": genbank_path,
                    "features_path": features_path,
                }


def _compare_existing_sequence(
    record: NormalizedMsdRegionRecord,
    item: Mapping[str, object],
) -> list[MsdRegionDiscrepancy]:
    discrepancies: list[MsdRegionDiscrepancy] = []
    genbank_path = Path(str(item["genbank_path"]))
    if genbank_path.exists():
        parsed = list(SeqIO.parse(genbank_path, "genbank"))
        if parsed:
            existing_sequence = str(parsed[0].seq).upper()
            if existing_sequence != record.msd_sequence_5to3:
                discrepancies.append(
                    MsdRegionDiscrepancy(
                        kind="sequence_mismatch",
                        variant_id=record.variant_id,
                        compared_path=genbank_path.as_posix(),
                        details={
                            "canonical_sha256": record.msd_sequence_sha256,
                            "existing_sha256": _sha256_text(existing_sequence),
                            "canonical_length_nt": record.sequence_length_nt,
                            "existing_length_nt": len(existing_sequence),
                        },
                    )
                )
    features_path = Path(str(item["features_path"]))
    if features_path.exists():
        mismatches = _feature_mismatches(record, features_path)
        if mismatches:
            discrepancies.append(
                MsdRegionDiscrepancy(
                    kind="annotation_mismatch",
                    variant_id=record.variant_id,
                    compared_path=features_path.as_posix(),
                    details={"mismatches": mismatches},
                )
            )
    return discrepancies


def _feature_mismatches(record: NormalizedMsdRegionRecord, features_path: Path) -> list[dict[str, object]]:
    expected = {
        feature.role: {
            "start_0": feature.display_start_0,
            "end_0": feature.display_end_0,
            "sequence": feature.sequence_5to3.upper(),
        }
        for feature in record.features
        if feature.role in _PRIMITIVE_ROLES
    }
    observed: dict[str, dict[str, object]] = {}
    with features_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            role = str(row.get("role") or "").strip()
            if role in _PRIMITIVE_ROLES:
                observed[role] = {
                    "start_0": int(row["start_0"]),
                    "end_0": int(row["end_0"]),
                    "sequence": str(row.get("sequence") or "").upper(),
                }
    mismatches: list[dict[str, object]] = []
    for role, expected_payload in expected.items():
        observed_payload = observed.get(role)
        if observed_payload is not None and observed_payload != expected_payload:
            mismatches.append({"role": role, "expected": expected_payload, "observed": observed_payload})
    return mismatches


def _compare_cap_sources(
    by_variant: Mapping[str, NormalizedMsdRegionRecord],
    path: Path,
) -> tuple[int, list[MsdRegionDiscrepancy]]:
    if not path.exists():
        return 0, []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources") or {}
    cap_map = {
        "retron26": "C26",
        "retron43": "C43",
        "retron172": "C172",
        "retron173": "C173",
        "retron174": "C174",
        "retron175": "C175",
        "retron176": "C176",
    }
    comparisons = 0
    discrepancies: list[MsdRegionDiscrepancy] = []
    for variant_id, cap_id in cap_map.items():
        record = by_variant.get(variant_id)
        source = sources.get(cap_id) if isinstance(sources, dict) else None
        if record is None or not isinstance(source, dict):
            continue
        comparisons += 1
        cap_sequence = str(source.get("sequence_5to3") or "").upper()
        observed_cap = record.primitive("snapback_foldback_geometry").sequence_5to3.upper()
        details: dict[str, object] = {}
        if cap_sequence and cap_sequence != observed_cap:
            details["cap_sequence"] = {"expected": observed_cap, "existing": cap_sequence}
        full_sequence = str(source.get("full_msd_sequence_5to3") or "").upper()
        if full_sequence and full_sequence != record.msd_sequence_5to3:
            details["full_msd_sequence"] = {
                "expected_sha256": record.msd_sequence_sha256,
                "existing_sha256": _sha256_text(full_sequence),
            }
        if details:
            discrepancies.append(
                MsdRegionDiscrepancy(
                    kind="cap_source_mismatch",
                    variant_id=variant_id,
                    compared_path=f"{path.as_posix()}#sources.{cap_id}",
                    details=details,
                )
            )
    return comparisons, discrepancies


def _variant_id(record: SeqRecord) -> str | None:
    match = _VARIANT_RE.search(_record_text(record))
    if match is None:
        return None
    return f"retron{match.group(1) or match.group(2)}"


def _variant_id_for_existing_row(row: Mapping[str, str], display_by_variant_key: Mapping[str, str]) -> str | None:
    construct_id = str(row.get("construct_id") or "")
    trim_match = re.fullmatch(r"pES-tetr-(.+)", construct_id)
    if trim_match is not None:
        display_id = display_by_variant_key.get(trim_match.group(1))
        if display_id:
            display_match = _DISPLAY_ID_RE.fullmatch(display_id)
            if display_match is not None:
                return f"retron{display_match.group(1)}"
    display_match = _DISPLAY_ID_RE.search(construct_id)
    if display_match is not None:
        return f"retron{display_match.group(1)}"
    return None


def _review_display_map(root: Path) -> dict[str, str]:
    path = root / "reviews/review_manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("sequence_montage", {}).get("review_variant_ids", {})
    return dict(mapping) if isinstance(mapping, dict) else {}


def _record_text(record: SeqRecord) -> str:
    parts = [record.id, record.name, record.description]
    for feature in record.features:
        for values in feature.qualifiers.values():
            parts.extend(str(value) for value in values)
    return "\n".join(parts)


def _simple_span(feature: Any) -> tuple[int, int]:
    if len(getattr(feature.location, "parts", ())) > 1:
        raise MsdRegionIngestError("Compound GenBank features are not supported for MSD-region ingest.")
    start = int(feature.location.start)
    end = int(feature.location.end)
    if end < start:
        raise MsdRegionIngestError(f"Invalid feature span {start}:{end}.")
    return start, end


def _qualifier_values(feature: Any, key: str) -> list[str]:
    return [str(value) for value in feature.qualifiers.get(key, [])]


def _variant_number(variant_id: str) -> str:
    match = re.fullmatch(r"retron(\d+)", variant_id)
    if match is None:
        raise MsdRegionIngestError(f"Invalid retron variant id: {variant_id}")
    return match.group(1)


def _variant_sort_key(variant_id: str) -> int:
    return int(_variant_number(variant_id))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    path.write_text(_allowlist_checksum_lines(text), encoding="utf-8")


def _allowlist_checksum_lines(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        key = line.split(":", 1)[0].strip()
        if key.endswith("sha256") and "pragma: allowlist secret" not in line:
            line = f"{line}  # pragma: allowlist secret"
        lines.append(line)
    return "\n".join(lines) + "\n"


def _relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


__all__ = [
    "compiler_spec_payload_from_records",
    "compare_records_to_existing_sources",
    "parse_msd_region_genbank",
    "write_msd_region_record_bundle",
]
