"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/retron_hairpin_design/compiler.py

Study-local Retron MSD design-reference compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from dnadesign.construct.src.composition import run_linear_ssdna_composition
from dnadesign.contracts.sequence import MsdDesignCatalogV1, MsdDesignReferenceV1

from .msd_ids import parse_msd_construct_label
from .registry import load_retron_msd_registry

REFERENCE_FILENAME = "msd_design_reference_v1.json"
CATALOG_FILENAME = "msd_design_catalog_v1.json"
BUNDLE_MANIFEST_FILENAME = "manifest.json"
BUNDLE_README_FILENAME = "README.md"
REFERENCE_INDEX_FILENAME = "reference_index.tsv"
REFERENCE_DIRNAME = "references"
LEGACY_ASSETS_DIRNAME = "assets"
SEQUENCE_MANIFEST_FILENAME = "sequence_manifest.json"
SEQUENCE_INDEX_FILENAME = "sequence_index.tsv"
COMPOSITION_CONFIG_DIRNAME = "composition_configs"
VARIANT_DIRNAME = "variants"
DEFAULT_FLANK_5P_PREFIX = "gtcagaaaaaa"
DEFAULT_FLANK_3P_SUFFIX = "acagtaactcaga"
IGNORED_OUTPUT_FILENAMES = {".DS_Store"}
_SEQUENCE_TOP_LEVEL = {
    SEQUENCE_MANIFEST_FILENAME,
    SEQUENCE_INDEX_FILENAME,
    COMPOSITION_CONFIG_DIRNAME,
    VARIANT_DIRNAME,
}
_BASERENDER_CONTRACT_KIND = "nucleotide_evidence_map_render_v3"
_MSD_UNIT_REPEAT_COUNT = 1


class RetronMsdCompilerError(ValueError):
    """Raised when MSD design-reference compilation cannot proceed safely."""


@dataclass(frozen=True)
class MsdSequenceBundleResult:
    catalog: MsdDesignCatalogV1
    bundle_root: Path
    manifest_path: Path
    index_path: Path
    variants: list[dict[str, object]]


def build_msd_design_reference(label: str, *, study_dir: str | Path) -> MsdDesignReferenceV1:
    parsed = parse_msd_construct_label(label)
    registry = load_retron_msd_registry(study_dir)
    return registry.build_reference(parsed)


def compile_msd_design_catalog(labels: list[str], *, study_dir: str | Path) -> MsdDesignCatalogV1:
    if not labels:
        raise RetronMsdCompilerError("Provide at least one construct label.")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise RetronMsdCompilerError(f"Duplicate construct label(s): {', '.join(duplicates)}")
    references = [build_msd_design_reference(label, study_dir=study_dir) for label in labels]
    return MsdDesignCatalogV1(records=references)


def write_msd_design_catalog(catalog: MsdDesignCatalogV1, *, out_dir: str | Path) -> Path:
    return _write_msd_design_catalog(catalog, out_dir=out_dir, extra_allowed_top_level=set())


def materialize_msd_design_artifacts(
    catalog: MsdDesignCatalogV1,
    *,
    out_dir: str | Path,
    payload_sequences: Mapping[str, str],
    cap_sequences: Mapping[str, str],
    flank_5p_prefix: str = DEFAULT_FLANK_5P_PREFIX,
    flank_3p_suffix: str = DEFAULT_FLANK_3P_SUFFIX,
    render_formats: Sequence[str] = ("png",),
    run_baserender: bool = True,
) -> MsdSequenceBundleResult:
    formats = _normalize_render_formats(render_formats)
    payload_sequences = {str(key).strip(): str(value).strip() for key, value in payload_sequences.items()}
    cap_sequences = {str(key).strip(): str(value).strip() for key, value in cap_sequences.items()}
    _require_sequence_subcomponents(catalog, payload_sequences=payload_sequences, cap_sequences=cap_sequences)

    root = Path(out_dir).expanduser().resolve()
    reference_filenames = {_reference_bundle_filename(record) for record in catalog.records}
    expected_design_ids = {record.msd_design_id for record in catalog.records}
    _guard_output_layout(
        root,
        expected_reference_filenames=reference_filenames,
        extra_allowed_top_level=_SEQUENCE_TOP_LEVEL,
    )
    _guard_sequence_output_layout(root, expected_design_ids=expected_design_ids)
    root.mkdir(parents=True, exist_ok=True)

    _write_msd_design_catalog(catalog, out_dir=root, extra_allowed_top_level=_SEQUENCE_TOP_LEVEL)
    configs_dir = root / COMPOSITION_CONFIG_DIRNAME
    variants_dir = root / VARIANT_DIRNAME
    configs_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    updated_records: list[MsdDesignReferenceV1] = []
    for record in catalog.records:
        artifact_bundle = variants_dir / record.msd_design_id
        config_path = configs_dir / f"{record.msd_design_id}.linear_ssdna_composition.yaml"
        config_payload = _composition_config_payload(
            record,
            artifact_bundle=artifact_bundle,
            payload_sequence=payload_sequences[record.payload_or_target.id],
            cap_sequence=cap_sequences[record.cap.id],
            flank_5p_prefix=flank_5p_prefix,
            flank_3p_suffix=flank_3p_suffix,
            render_formats=formats,
        )
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
        try:
            composition = run_linear_ssdna_composition(config_path)
        except Exception as exc:  # pragma: no cover - depends on producer failure mode.
            raise RetronMsdCompilerError(
                f"Construct failed to emit Retron MSD unit '{record.msd_design_id}': {exc}"
            ) from exc
        rendered = _run_baserender_jobs(composition.artifact_bundle, formats=formats, enabled=run_baserender)
        row = _sequence_index_row(
            record,
            composition_id=composition.composition_id,
            sequence_length=composition.sequence_length,
            sequence_sha256=composition.sequence_sha256,
            config_path=config_path,
            artifact_bundle=composition.artifact_bundle,
            rendered=rendered,
            root=root,
        )
        rows.append(row)
        updated_records.append(_record_with_sequence_artifacts(record, row=row))

    updated_catalog = MsdDesignCatalogV1(records=updated_records)
    _write_msd_design_catalog(updated_catalog, out_dir=root, extra_allowed_top_level=_SEQUENCE_TOP_LEVEL)
    _write_sequence_index(root / SEQUENCE_INDEX_FILENAME, rows)
    _write_sequence_manifest(root / SEQUENCE_MANIFEST_FILENAME, rows=rows, render_formats=formats, root=root)
    _write_bundle_readme(root / BUNDLE_README_FILENAME, catalog=updated_catalog, sequence_rows=rows)
    return MsdSequenceBundleResult(
        catalog=updated_catalog,
        bundle_root=root,
        manifest_path=root / SEQUENCE_MANIFEST_FILENAME,
        index_path=root / SEQUENCE_INDEX_FILENAME,
        variants=rows,
    )


def _write_msd_design_catalog(
    catalog: MsdDesignCatalogV1,
    *,
    out_dir: str | Path,
    extra_allowed_top_level: set[str],
) -> Path:
    root = Path(out_dir).expanduser().resolve()
    reference_filenames = [_reference_bundle_filename(record) for record in catalog.records]
    duplicate_reference_filenames = sorted(
        {filename for filename in reference_filenames if reference_filenames.count(filename) > 1}
    )
    if duplicate_reference_filenames:
        raise RetronMsdCompilerError(
            f"Duplicate MSD design reference filename(s): {', '.join(duplicate_reference_filenames)}"
        )

    _guard_output_layout(
        root,
        expected_reference_filenames=set(reference_filenames),
        extra_allowed_top_level=extra_allowed_top_level,
    )
    root.mkdir(parents=True, exist_ok=True)
    references_dir = root / REFERENCE_DIRNAME
    references_dir.mkdir(parents=True, exist_ok=True)

    reference_rows: list[dict[str, object]] = []
    for record, reference_filename in zip(catalog.records, reference_filenames, strict=True):
        reference_path = references_dir / reference_filename
        reference_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        reference_rows.append(_reference_index_row(record, reference_path=reference_path, root=root))

    catalog_path = root / CATALOG_FILENAME
    catalog_path.write_text(catalog.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_reference_index(root / REFERENCE_INDEX_FILENAME, reference_rows)
    _write_bundle_manifest(root / BUNDLE_MANIFEST_FILENAME, catalog=catalog, reference_rows=reference_rows)
    _write_bundle_readme(root / BUNDLE_README_FILENAME, catalog=catalog)
    return catalog_path


def _guard_output_layout(
    root: Path,
    *,
    expected_reference_filenames: set[str],
    extra_allowed_top_level: set[str],
) -> None:
    if not root.exists():
        return
    if not root.is_dir():
        raise RetronMsdCompilerError(f"MSD compiler output path exists but is not a directory: {root}")

    legacy_assets_dir = root / LEGACY_ASSETS_DIRNAME
    if legacy_assets_dir.exists():
        raise RetronMsdCompilerError(
            f"Legacy MSD compiler output layout exists at {legacy_assets_dir}. "
            "Choose a fresh --out-dir or explicitly archive/remove the old generated assets directory before compiling."
        )

    allowed_top_level = {
        BUNDLE_README_FILENAME,
        BUNDLE_MANIFEST_FILENAME,
        CATALOG_FILENAME,
        REFERENCE_INDEX_FILENAME,
        REFERENCE_DIRNAME,
        *extra_allowed_top_level,
        *IGNORED_OUTPUT_FILENAMES,
    }
    unexpected_top_level = sorted(item.name for item in root.iterdir() if item.name not in allowed_top_level)
    if unexpected_top_level:
        raise RetronMsdCompilerError(
            f"Unexpected MSD compiler output entries at {root}: {', '.join(unexpected_top_level)}. "
            "Choose a fresh --out-dir or explicitly archive/remove unrelated generated entries before compiling."
        )

    references_dir = root / REFERENCE_DIRNAME
    if not references_dir.exists():
        return
    stale_reference_entries = sorted(
        item.name
        for item in references_dir.iterdir()
        if item.name not in IGNORED_OUTPUT_FILENAMES
        and (item.is_dir() or item.name not in expected_reference_filenames)
    )
    if stale_reference_entries:
        raise RetronMsdCompilerError(
            f"Stale MSD design reference output at {references_dir}: {', '.join(stale_reference_entries)}. "
            "Choose a fresh --out-dir or explicitly archive/remove stale generated references before compiling."
        )


def _guard_sequence_output_layout(root: Path, *, expected_design_ids: set[str]) -> None:
    variants_dir = root / VARIANT_DIRNAME
    if variants_dir.exists():
        stale_variants = sorted(
            item.name
            for item in variants_dir.iterdir()
            if item.name not in expected_design_ids | IGNORED_OUTPUT_FILENAMES
        )
        if stale_variants:
            raise RetronMsdCompilerError(
                f"Stale MSD sequence output at {variants_dir}: {', '.join(stale_variants)}. "
                "Choose a fresh --out-dir or explicitly archive/remove stale generated variants before materializing."
            )
    configs_dir = root / COMPOSITION_CONFIG_DIRNAME
    if configs_dir.exists():
        expected_config_names = {
            f"{design_id}.linear_ssdna_composition.yaml" for design_id in expected_design_ids
        } | IGNORED_OUTPUT_FILENAMES
        stale_configs = sorted(item.name for item in configs_dir.iterdir() if item.name not in expected_config_names)
        if stale_configs:
            raise RetronMsdCompilerError(
                f"Stale MSD composition config output at {configs_dir}: {', '.join(stale_configs)}. "
                "Choose a fresh --out-dir or explicitly archive/remove stale generated configs before materializing."
            )


def _require_sequence_subcomponents(
    catalog: MsdDesignCatalogV1,
    *,
    payload_sequences: Mapping[str, str],
    cap_sequences: Mapping[str, str],
) -> None:
    missing_payloads = sorted(
        {
            record.payload_or_target.id
            for record in catalog.records
            if not payload_sequences.get(record.payload_or_target.id)
        }
    )
    missing_caps = sorted({record.cap.id for record in catalog.records if not cap_sequences.get(record.cap.id)})
    if not missing_payloads and not missing_caps:
        return
    pieces: list[str] = []
    if missing_payloads:
        pieces.append(f"payload(s): {', '.join(missing_payloads)}")
    if missing_caps:
        pieces.append(f"cap(s): {', '.join(missing_caps)}")
    raise RetronMsdCompilerError(
        "MSD sequence artifact generation requires concrete sequence subcomponents for "
        f"{'; '.join(pieces)}. Provide --payload-sequence ID=ACGT and --cap-sequence ID=ACGT overrides, "
        "or route missing cap/shortening inputs to Snapback and missing base-junction inputs to scar-nick first."
    )


def _normalize_render_formats(render_formats: Sequence[str]) -> list[str]:
    requested = list(render_formats) or ["png"]
    formats: list[str] = []
    for raw_format in requested:
        fmt = str(raw_format or "").strip().lower()
        if fmt not in {"png", "svg", "pdf"}:
            raise RetronMsdCompilerError(f"Unsupported render format '{raw_format}'. Expected png, svg, or pdf.")
        if fmt not in formats:
            formats.append(fmt)
    return formats


def _reference_bundle_filename(record: MsdDesignReferenceV1) -> str:
    return f"{record.msd_design_id}.{REFERENCE_FILENAME}"


def _reference_index_row(
    record: MsdDesignReferenceV1,
    *,
    reference_path: Path,
    root: Path,
) -> dict[str, object]:
    return {
        "construct_id": record.construct_id,
        "msd_design_id": record.msd_design_id,
        "payload_id": record.payload_or_target.id,
        "cap_id": record.cap.id,
        "left_base": record.scar_nick.left_base,
        "right_base": record.scar_nick.right_base,
        "profile_s3s2s1s0": record.scar_nick.profile_s3s2s1s0,
        "route_status": record.scar_nick.route_status,
        "nick_orientation": record.scar_nick.nick_orientation or "",
        "nickase": record.scar_nick.nickase or "",
        "reference_path": reference_path.relative_to(root).as_posix(),
    }


def _msd_unit_segments(
    record: MsdDesignReferenceV1,
    *,
    flank_5p: str,
    flank_3p: str,
    payload_sequence: str,
    cap_sequence: str,
) -> list[dict[str, object]]:
    return [
        {
            "segment_id": "flank_5p",
            "role": "flank_5p",
            "sequence": flank_5p,
            "source": {
                "kind": "study_record",
                "study_id": "retron_hairpin_design",
                "ref": record.construct_label,
            },
        },
        {
            "segment_id": "payload_primary",
            "role": "payload_primary",
            "sequence": payload_sequence,
            "source": {"kind": "literal", "label": f"{record.payload_or_target.id} override"},
        },
        {
            "segment_id": "snapback_cap_segment",
            "role": "cap",
            "sequence": cap_sequence,
            "source": {"kind": "literal", "label": f"{record.cap.id} override"},
        },
        {
            "segment_id": "payload_complement",
            "role": "payload_complement",
            "transform": {
                "kind": "reverse_complement",
                "source_segment_id": "payload_primary",
                "assert_expected_sequence": True,
            },
            "source": {"kind": "derived", "from_segment_id": "payload_primary"},
        },
        {
            "segment_id": "flank_3p",
            "role": "flank_3p",
            "sequence": flank_3p,
            "source": {
                "kind": "study_record",
                "study_id": "retron_hairpin_design",
                "ref": record.construct_label,
            },
        },
    ]


def _msd_unit_annotations(*, flank_5p_prefix: str, flank_5p: str, right_base: str) -> list[dict[str, object]]:
    return [
        {
            "annotation_id": "stem_base_left",
            "role": "stem_base_left",
            "location": {
                "basis": "segment",
                "segment_id": "flank_5p",
                "start": len(flank_5p_prefix),
                "end": len(flank_5p),
            },
        },
        {
            "annotation_id": "stem_base_right",
            "role": "stem_base_right",
            "location": {
                "basis": "segment",
                "segment_id": "flank_3p",
                "start": 0,
                "end": len(right_base),
            },
        },
    ]


def _msd_display_profile(record: MsdDesignReferenceV1, *, payload_label: str) -> dict[str, object]:
    return {
        "title": f"{record.construct_id} {payload_label}",
        "component_labels": {
            "flank_5p": "5' Flanking",
            "payload_primary": payload_label,
            "snapback_cap_segment": "Cap",
            "payload_complement": f"{payload_label} complement",
            "flank_3p": "3' Flanking",
        },
        "annotation_labels": {
            "stem_base_left": "Left Base",
            "stem_base_right": "Right Base",
        },
        "component_hues": {
            "flank_5p": "#4C78A8",
            "flank_3p": "#72B7B2",
            "payload_primary": "#F58518",
            "payload_complement": "#E45756",
            "snapback_cap_segment": "#54A24B",
            "stem_base_left": "#B279A2",
            "stem_base_right": "#9D755D",
        },
        "base_highlight_color": "#111827",
    }


def _composition_config_payload(
    record: MsdDesignReferenceV1,
    *,
    artifact_bundle: Path,
    payload_sequence: str,
    cap_sequence: str,
    flank_5p_prefix: str,
    flank_3p_suffix: str,
    render_formats: Sequence[str],
) -> dict[str, object]:
    left_base = record.scar_nick.left_base
    right_base = record.scar_nick.right_base
    flank_5p = f"{flank_5p_prefix}{left_base}"
    flank_3p = f"{right_base}{flank_3p_suffix}"
    payload_label = record.payload_or_target.display_name or record.payload_or_target.id
    return {
        "contract": "linear_ssdna_composition_v1",
        "schema_version": 1,
        "composition_id": record.msd_design_id,
        "alphabet": "dna",
        "topology": "linear_ssdna",
        "coordinate_system": "zero_based_half_open",
        "case_policy": "preserve_input_display_case",
        "canonicalization": {
            "compare_sequences_case_insensitive": True,
            "output_sequence_preserves_case": True,
        },
        "units": [
            {
                "unit_id": f"{record.msd_design_id}_unit",
                "repeat_count": _MSD_UNIT_REPEAT_COUNT,
                "segments": _msd_unit_segments(
                    record,
                    flank_5p=flank_5p,
                    flank_3p=flank_3p,
                    payload_sequence=payload_sequence,
                    cap_sequence=cap_sequence,
                ),
                "annotations": _msd_unit_annotations(
                    flank_5p_prefix=flank_5p_prefix,
                    flank_5p=flank_5p,
                    right_base=right_base,
                ),
                "assertions": [
                    {
                        "assertion_id": "payload_rc",
                        "kind": "reverse_complement",
                        "left_segment_id": "payload_primary",
                        "right_segment_id": "payload_complement",
                        "severity": "error",
                    }
                ],
            }
        ],
        "qa": {
            "require_no_unknown_bases": True,
            "allow_degenerate_bases": False,
            "require_segment_span_coverage": True,
            "require_non_overlapping_physical_segments": True,
            "require_annotation_bounds": True,
            "require_declared_transform_checks": True,
            "allow_cross_copy_intended_pairings": False,
        },
        "folding": {"enabled": False, "required": False, "scope": "canonical_component_unit"},
        "visual": {
            "emit": ["sequence_evidence_map_v1"],
            "display_profile": _msd_display_profile(record, payload_label=payload_label),
            "render_exports": {"formats": list(render_formats)},
        },
        "benchling_export": {
            "enabled": True,
            "primary_format": "genbank",
            "sidecars": ["fasta", "features_csv"],
        },
        "output": {"artifact_bundle": artifact_bundle.as_posix(), "usr": {"enabled": False}},
    }


def _run_baserender_jobs(
    artifact_bundle: Path,
    *,
    formats: Sequence[str],
    enabled: bool,
) -> dict[str, str]:
    rendered: dict[str, str] = {}
    if not enabled:
        return rendered
    import dnadesign.baserender as baserender

    for fmt in formats:
        job_path = artifact_bundle / "baserender_jobs" / f"component_span_qa_{fmt}.yaml"
        try:
            report = baserender.run_job(
                job_path,
                kind=_BASERENDER_CONTRACT_KIND,
                strict=True,
                caller_root=artifact_bundle,
            )
        except Exception as exc:  # pragma: no cover - depends on renderer backend failure mode.
            raise RetronMsdCompilerError(f"BaseRender failed for Retron MSD bundle '{artifact_bundle}': {exc}") from exc
        image_path = Path(report.outputs["images_path"])
        flat_path = artifact_bundle / f"component_span_qa.{fmt}"
        if image_path != flat_path:
            shutil.copyfile(image_path, flat_path)
        rendered[f"component_span_{fmt}"] = flat_path.as_posix()
    return rendered


def _sequence_index_row(
    record: MsdDesignReferenceV1,
    *,
    composition_id: str,
    sequence_length: int,
    sequence_sha256: str,
    config_path: Path,
    artifact_bundle: Path,
    rendered: Mapping[str, str],
    root: Path,
) -> dict[str, object]:
    genbank_path = artifact_bundle / "sequence.gb"
    row: dict[str, object] = {
        "construct_id": record.construct_id,
        "msd_design_id": record.msd_design_id,
        "composition_id": composition_id,
        "unit_count": _MSD_UNIT_REPEAT_COUNT,
        "sequence_length": sequence_length,
        "sequence_sha256": sequence_sha256,
        "composition_config": config_path.relative_to(root).as_posix(),
        "artifact_bundle": artifact_bundle.relative_to(root).as_posix(),
        "genbank": genbank_path.relative_to(root).as_posix(),
        "forward_fasta": (artifact_bundle / "sequence.fa").relative_to(root).as_posix(),
        "features_csv": (artifact_bundle / "features.csv").relative_to(root).as_posix(),
        "visual_contract": (artifact_bundle / "visual" / "sequence_evidence_map_v1.json").relative_to(root).as_posix(),
        "finder_reveal": f"open -R {shlex.quote(genbank_path.as_posix())}",
    }
    for key, raw_path in rendered.items():
        row[key] = Path(raw_path).relative_to(root).as_posix()
    return row


def _record_with_sequence_artifacts(record: MsdDesignReferenceV1, *, row: Mapping[str, object]) -> MsdDesignReferenceV1:
    payload = record.model_dump(mode="json")
    payload["sequence"] = {
        "length": row["sequence_length"],
        "sha256": row["sequence_sha256"],
    }
    payload["source"] = {
        "dnadesign_bundle": row["artifact_bundle"],
        "composition_id": row["composition_id"],
    }
    artifacts: dict[str, object] = {
        "genbank": row["genbank"],
        "forward_fasta": row["forward_fasta"],
        "features_csv": row["features_csv"],
        "visual_contract": row["visual_contract"],
    }
    for field in ("component_span_png", "component_span_svg"):
        value = row.get(field)
        if value:
            artifacts[field] = value
    payload["artifacts"] = artifacts
    return MsdDesignReferenceV1.model_validate(payload)


def _write_reference_index(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "construct_id",
        "msd_design_id",
        "payload_id",
        "cap_id",
        "left_base",
        "right_base",
        "profile_s3s2s1s0",
        "route_status",
        "nick_orientation",
        "nickase",
        "reference_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _write_sequence_index(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "construct_id",
        "msd_design_id",
        "composition_id",
        "unit_count",
        "sequence_length",
        "sequence_sha256",
        "composition_config",
        "artifact_bundle",
        "genbank",
        "forward_fasta",
        "features_csv",
        "visual_contract",
        "component_span_png",
        "component_span_svg",
        "finder_reveal",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_bundle_manifest(
    path: Path,
    *,
    catalog: MsdDesignCatalogV1,
    reference_rows: list[dict[str, object]],
) -> None:
    payload = {
        "contract": "msd_design_catalog_bundle_v1",
        "schema_version": 1,
        "catalog": CATALOG_FILENAME,
        "reference_index": REFERENCE_INDEX_FILENAME,
        "references_dir": REFERENCE_DIRNAME,
        "reference_count": len(catalog.records),
        "layout": {
            "top_level_files": [
                BUNDLE_README_FILENAME,
                BUNDLE_MANIFEST_FILENAME,
                CATALOG_FILENAME,
                REFERENCE_INDEX_FILENAME,
            ],
            "grouped_dirs": [REFERENCE_DIRNAME],
            "max_reference_depth": 1,
        },
        "references": reference_rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_sequence_manifest(
    path: Path,
    *,
    rows: list[dict[str, object]],
    render_formats: Sequence[str],
    root: Path,
) -> None:
    payload = {
        "contract": "msd_single_unit_sequence_bundle_v1",
        "schema_version": 1,
        "catalog": CATALOG_FILENAME,
        "sequence_index": SEQUENCE_INDEX_FILENAME,
        "variants_dir": VARIANT_DIRNAME,
        "composition_configs_dir": COMPOSITION_CONFIG_DIRNAME,
        "unit_count_per_design": _MSD_UNIT_REPEAT_COUNT,
        "render_formats": list(render_formats),
        "variant_count": len(rows),
        "variants": rows,
        "operator_hints": {
            "macos_open_bundle": f"open {shlex.quote(root.as_posix())}",
            "macos_finder_reveal_first_genbank": rows[0]["finder_reveal"] if rows else "",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_bundle_readme(
    path: Path,
    *,
    catalog: MsdDesignCatalogV1,
    sequence_rows: list[dict[str, object]] | None = None,
) -> None:
    lines = [
        "# Retron MSD Design Catalog",
        "",
        "Generated bundle for frozen `msd_design_reference_v1` records.",
        "",
        "Open first:",
        f"- `{CATALOG_FILENAME}`: full `msd_design_catalog_v1` contract.",
        f"- `{REFERENCE_INDEX_FILENAME}`: scan table with one row per design.",
        f"- `{REFERENCE_DIRNAME}/`: flat per-design reference JSON files.",
        "",
        f"Record count: {len(catalog.records)}",
        "",
        "Layout policy:",
        "- keep the top level limited to entrypoint files and grouped directories;",
        "- keep per-design references flat under `references/`; do not create one directory per design;",
        "- sequence bundles contain one MSD unit per design; do not repeat-expand complete MSD units.",
        "",
    ]
    if sequence_rows:
        lines.extend(
            [
                "Sequence outputs:",
                f"- `{SEQUENCE_MANIFEST_FILENAME}`: machine-readable manifest for generated GenBank/visual artifacts.",
                f"- `{SEQUENCE_INDEX_FILENAME}`: scan table with per-design GenBank, FASTA, feature CSV, "
                "and visual paths.",
                f"- `{VARIANT_DIRNAME}/`: one shallow per-design bundle with `sequence.gb`, `sequence.fa`, "
                "`features.csv`, and `component_span_qa.png` at the variant root.",
                f"- `{COMPOSITION_CONFIG_DIRNAME}/`: generated single-unit composition configs.",
                "",
                "Finder:",
                f"- `open {shlex.quote(path.parent.as_posix())}` opens the transient bundle directory.",
                f"- `finder_reveal` in `{SEQUENCE_INDEX_FILENAME}` reveals each `sequence.gb` file.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_README_FILENAME",
    "CATALOG_FILENAME",
    "COMPOSITION_CONFIG_DIRNAME",
    "DEFAULT_FLANK_3P_SUFFIX",
    "DEFAULT_FLANK_5P_PREFIX",
    "MsdSequenceBundleResult",
    "REFERENCE_DIRNAME",
    "REFERENCE_FILENAME",
    "REFERENCE_INDEX_FILENAME",
    "RetronMsdCompilerError",
    "SEQUENCE_INDEX_FILENAME",
    "SEQUENCE_MANIFEST_FILENAME",
    "VARIANT_DIRNAME",
    "build_msd_design_reference",
    "compile_msd_design_catalog",
    "materialize_msd_design_artifacts",
    "write_msd_design_catalog",
]
