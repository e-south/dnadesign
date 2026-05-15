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
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from dnadesign.construct import run_linear_ssdna_composition
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
MANIFEST_DIRNAME = "manifest"
COMPOSITION_CONFIG_DIRNAME = "composition_configs"
VARIANT_DIRNAME = "variants"
VARIANT_MANIFEST_DIRNAME = "manifest"
VARIANT_PLOTS_DIRNAME = "plots"
VARIANT_RUNTIME_DIRNAME = "runtime"
VARIANT_SEQUENCES_DIRNAME = "sequences"
CONSTRUCT_RUNTIME_DIRNAME = "construct"
DEFAULT_FLANK_5P_PREFIX = "gtcagaaaaaa"
DEFAULT_FLANK_3P_SUFFIX = "acagtaactcaga"
IGNORED_OUTPUT_FILENAMES = {".DS_Store"}
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
    expected_design_ids = {record.msd_design_id for record in catalog.records}
    reference_filenames = {_reference_bundle_filename(record) for record in catalog.records}
    _guard_materialize_output_layout(
        root,
        expected_design_ids=expected_design_ids,
        expected_reference_filenames=reference_filenames,
    )
    root.mkdir(parents=True, exist_ok=True)

    manifest_dir = root / MANIFEST_DIRNAME
    configs_dir = manifest_dir / COMPOSITION_CONFIG_DIRNAME
    variants_dir = root / VARIANT_DIRNAME
    manifest_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    updated_records: list[MsdDesignReferenceV1] = []
    for record in catalog.records:
        variant_dir = variants_dir / record.msd_design_id
        artifact_bundle = variant_dir / VARIANT_RUNTIME_DIRNAME / CONSTRUCT_RUNTIME_DIRNAME
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
        curated = _publish_variant_outputs(
            variant_dir,
            construct_bundle=composition.artifact_bundle,
            rendered=rendered,
            root=root,
        )
        row = _sequence_index_row(
            record,
            composition_id=composition.composition_id,
            sequence_length=composition.sequence_length,
            sequence_sha256=composition.sequence_sha256,
            config_path=config_path,
            variant_dir=variant_dir,
            construct_bundle=composition.artifact_bundle,
            curated=curated,
            root=root,
        )
        rows.append(row)
        updated_records.append(_record_with_sequence_artifacts(record, row=row))

    updated_catalog = MsdDesignCatalogV1(records=updated_records)
    _write_materialized_catalog(updated_catalog, root=root, manifest_dir=manifest_dir)
    _write_sequence_index(manifest_dir / SEQUENCE_INDEX_FILENAME, rows)
    _write_sequence_manifest(manifest_dir / SEQUENCE_MANIFEST_FILENAME, rows=rows, render_formats=formats, root=root)
    _write_bundle_readme(root / BUNDLE_README_FILENAME, catalog=updated_catalog, sequence_rows=rows)
    return MsdSequenceBundleResult(
        catalog=updated_catalog,
        bundle_root=root,
        manifest_path=manifest_dir / SEQUENCE_MANIFEST_FILENAME,
        index_path=manifest_dir / SEQUENCE_INDEX_FILENAME,
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


def _write_materialized_catalog(
    catalog: MsdDesignCatalogV1,
    *,
    root: Path,
    manifest_dir: Path,
) -> Path:
    reference_filenames = [_reference_bundle_filename(record) for record in catalog.records]
    duplicate_reference_filenames = sorted(
        {filename for filename in reference_filenames if reference_filenames.count(filename) > 1}
    )
    if duplicate_reference_filenames:
        raise RetronMsdCompilerError(
            f"Duplicate MSD design reference filename(s): {', '.join(duplicate_reference_filenames)}"
        )

    references_dir = manifest_dir / REFERENCE_DIRNAME
    references_dir.mkdir(parents=True, exist_ok=True)
    reference_rows: list[dict[str, object]] = []
    for record, reference_filename in zip(catalog.records, reference_filenames, strict=True):
        reference_path = references_dir / reference_filename
        reference_path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        reference_rows.append(_reference_index_row(record, reference_path=reference_path, root=root))

    catalog_path = manifest_dir / CATALOG_FILENAME
    catalog_path.write_text(catalog.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _write_reference_index(manifest_dir / REFERENCE_INDEX_FILENAME, reference_rows)
    _write_bundle_manifest(
        manifest_dir / BUNDLE_MANIFEST_FILENAME,
        catalog=catalog,
        reference_rows=reference_rows,
        catalog_path=(manifest_dir / CATALOG_FILENAME).relative_to(root).as_posix(),
        reference_index_path=(manifest_dir / REFERENCE_INDEX_FILENAME).relative_to(root).as_posix(),
        references_dir=(manifest_dir / REFERENCE_DIRNAME).relative_to(root).as_posix(),
        grouped_dirs=[MANIFEST_DIRNAME, VARIANT_DIRNAME],
        top_level_files=[BUNDLE_README_FILENAME],
    )
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


def _guard_materialize_output_layout(
    root: Path,
    *,
    expected_design_ids: set[str],
    expected_reference_filenames: set[str],
) -> None:
    if not root.exists():
        return
    if not root.is_dir():
        raise RetronMsdCompilerError(f"MSD materialize output path exists but is not a directory: {root}")

    legacy_assets_dir = root / LEGACY_ASSETS_DIRNAME
    if legacy_assets_dir.exists():
        raise RetronMsdCompilerError(
            f"Legacy MSD compiler output layout exists at {legacy_assets_dir}. "
            "Choose a fresh --out-dir or explicitly archive/remove the old generated assets directory before compiling."
        )

    allowed_top_level = {BUNDLE_README_FILENAME, MANIFEST_DIRNAME, VARIANT_DIRNAME, *IGNORED_OUTPUT_FILENAMES}
    unexpected_top_level = sorted(item.name for item in root.iterdir() if item.name not in allowed_top_level)
    if unexpected_top_level:
        raise RetronMsdCompilerError(
            f"Unexpected MSD materialize output entries at {root}: {', '.join(unexpected_top_level)}. "
            "Choose a fresh --out-dir or explicitly archive/remove unrelated generated entries before materializing."
        )

    manifest_dir = root / MANIFEST_DIRNAME
    if manifest_dir.exists():
        allowed_manifest_entries = {
            BUNDLE_MANIFEST_FILENAME,
            CATALOG_FILENAME,
            COMPOSITION_CONFIG_DIRNAME,
            REFERENCE_DIRNAME,
            REFERENCE_INDEX_FILENAME,
            SEQUENCE_INDEX_FILENAME,
            SEQUENCE_MANIFEST_FILENAME,
            *IGNORED_OUTPUT_FILENAMES,
        }
        stale_manifest_entries = sorted(
            item.name for item in manifest_dir.iterdir() if item.name not in allowed_manifest_entries
        )
        if stale_manifest_entries:
            raise RetronMsdCompilerError(
                f"Unexpected MSD materialize manifest entries at {manifest_dir}: "
                f"{', '.join(stale_manifest_entries)}. Choose a fresh --out-dir or archive/remove stale output first."
            )

    references_dir = manifest_dir / REFERENCE_DIRNAME
    if references_dir.exists():
        stale_reference_entries = sorted(
            item.name
            for item in references_dir.iterdir()
            if item.name not in IGNORED_OUTPUT_FILENAMES
            and (item.is_dir() or item.name not in expected_reference_filenames)
        )
        if stale_reference_entries:
            raise RetronMsdCompilerError(
                f"Stale MSD design reference output at {references_dir}: {', '.join(stale_reference_entries)}. "
                "Choose a fresh --out-dir or explicitly archive/remove stale generated references before materializing."
            )

    configs_dir = manifest_dir / COMPOSITION_CONFIG_DIRNAME
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

    variants_dir = root / VARIANT_DIRNAME
    if not variants_dir.exists():
        return
    stale_variants = sorted(
        item.name for item in variants_dir.iterdir() if item.name not in expected_design_ids | IGNORED_OUTPUT_FILENAMES
    )
    if stale_variants:
        raise RetronMsdCompilerError(
            f"Stale MSD sequence output at {variants_dir}: {', '.join(stale_variants)}. "
            "Choose a fresh --out-dir or explicitly archive/remove stale generated variants before materializing."
        )
    allowed_variant_entries = {
        VARIANT_MANIFEST_DIRNAME,
        VARIANT_PLOTS_DIRNAME,
        VARIANT_RUNTIME_DIRNAME,
        VARIANT_SEQUENCES_DIRNAME,
        *IGNORED_OUTPUT_FILENAMES,
    }
    for variant_dir in variants_dir.iterdir():
        if variant_dir.name in IGNORED_OUTPUT_FILENAMES or not variant_dir.is_dir():
            continue
        stale_variant_entries = sorted(
            item.name for item in variant_dir.iterdir() if item.name not in allowed_variant_entries
        )
        if stale_variant_entries:
            raise RetronMsdCompilerError(
                f"Unexpected MSD variant output entries at {variant_dir}: {', '.join(stale_variant_entries)}. "
                "Choose a fresh --out-dir or archive/remove stale generated variant output first."
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
        "folding": {
            "enabled": True,
            "required": False,
            "scope": "canonical_component_unit",
            "backend": {
                "name": "ViennaRNA",
                "interface": "cli",
                "executable": "RNAfold",
                "backend_contract": "secondary_structure_prediction_v1",
            },
            "dna_policy": {"mode": "convert_t_to_u_for_rna_backend"},
        },
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


def _publish_variant_outputs(
    variant_dir: Path,
    *,
    construct_bundle: Path,
    rendered: Mapping[str, str],
    root: Path,
) -> dict[str, object]:
    sequences_dir = variant_dir / VARIANT_SEQUENCES_DIRNAME
    plots_dir = variant_dir / VARIANT_PLOTS_DIRNAME
    manifest_dir = variant_dir / VARIANT_MANIFEST_DIRNAME
    sequences_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    forward_genbank = sequences_dir / "forward.gb"
    reverse_complement_genbank = sequences_dir / "reverse_complement.gb"
    forward_fasta = sequences_dir / "forward.fa"
    reverse_complement_fasta = sequences_dir / "reverse_complement.fa"
    features_csv = sequences_dir / "features.csv"
    _copy_required_file(construct_bundle / "sequence.gb", forward_genbank)
    _copy_required_file(construct_bundle / "sequence.reverse_complement.gb", reverse_complement_genbank)
    _copy_required_file(construct_bundle / "sequence.fa", forward_fasta)
    _copy_required_file(construct_bundle / "sequence.reverse_complement.fa", reverse_complement_fasta)
    _copy_required_file(construct_bundle / "features.csv", features_csv)

    component_png_source = Path(str(rendered.get("component_span_png") or construct_bundle / "component_span_qa.png"))
    component_png = plots_dir / "component_span.png"
    _copy_required_file(component_png_source, component_png)
    for fmt in ("svg", "pdf"):
        raw_path = rendered.get(f"component_span_{fmt}")
        if raw_path:
            _copy_required_file(Path(raw_path), plots_dir / f"component_span.{fmt}")

    folding_prediction = construct_bundle / "folding" / "secondary_structure_prediction_v1.json"
    folding_status = _folding_prediction_status(folding_prediction)
    folding_png = plots_dir / "secondary_structure.png"
    _write_folding_prediction_png(folding_prediction, folding_png)
    combined_png = plots_dir / "component_span_and_folding.png"
    _write_combined_png(component_png, folding_png, combined_png)

    manifest_sources = [
        (construct_bundle / "manifest.json", manifest_dir / "construct_manifest.json"),
        (construct_bundle / "assembled_sequence.json", manifest_dir / "assembled_sequence.json"),
        (construct_bundle / "segment_spans.json", manifest_dir / "segment_spans.json"),
        (construct_bundle / "annotation_spans.json", manifest_dir / "annotation_spans.json"),
        (construct_bundle / "provenance.json", manifest_dir / "provenance.json"),
        (construct_bundle / "validation_report.json", manifest_dir / "validation_report.json"),
        (
            construct_bundle / "visual" / "sequence_evidence_map_v1.json",
            manifest_dir / "sequence_evidence_map_v1.json",
        ),
        (folding_prediction, manifest_dir / "folding_prediction.json"),
        (construct_bundle / "folding" / "folding_preflight.json", manifest_dir / "folding_preflight.json"),
        (
            construct_bundle / "folding" / "secondary_structure_prediction_request_v1.yaml",
            manifest_dir / "folding_request.yaml",
        ),
    ]
    for source, destination in manifest_sources:
        _copy_required_file(source, destination)

    return {
        "genbank": forward_genbank.relative_to(root).as_posix(),
        "reverse_complement_genbank": reverse_complement_genbank.relative_to(root).as_posix(),
        "forward_fasta": forward_fasta.relative_to(root).as_posix(),
        "reverse_complement_fasta": reverse_complement_fasta.relative_to(root).as_posix(),
        "features_csv": features_csv.relative_to(root).as_posix(),
        "visual_contract": (manifest_dir / "sequence_evidence_map_v1.json").relative_to(root).as_posix(),
        "construct_manifest": (manifest_dir / "construct_manifest.json").relative_to(root).as_posix(),
        "folding_prediction": (manifest_dir / "folding_prediction.json").relative_to(root).as_posix(),
        "folding_status": folding_status,
        "component_span_png": component_png.relative_to(root).as_posix(),
        "folding_png": folding_png.relative_to(root).as_posix(),
        "combined_plot_png": combined_png.relative_to(root).as_posix(),
    }


def _copy_required_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RetronMsdCompilerError(f"Expected MSD materialize artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _folding_prediction_status(path: Path) -> str:
    if not path.is_file():
        raise RetronMsdCompilerError(
            f"Expected folding prediction artifact is missing: {path}. "
            "Materialize enables folding by default and requires an explicit folding status artifact."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload.get("status") or "").strip()
    if not status:
        raise RetronMsdCompilerError(f"Folding prediction artifact is missing status: {path}")
    return status


def _write_folding_prediction_png(prediction_path: Path, output_path: Path) -> None:
    payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    status = str(payload.get("status") or "unknown")
    input_payload = payload.get("input") if isinstance(payload.get("input"), dict) else {}
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    qa_payload = payload.get("qa") if isinstance(payload.get("qa"), dict) else {}
    dot_bracket = str(result_payload.get("dot_bracket") or "")
    mfe = result_payload.get("mfe_kcal_mol")
    pair_map = result_payload.get("pair_map") if isinstance(result_payload.get("pair_map"), list) else []
    warnings = qa_payload.get("warnings") if isinstance(qa_payload.get("warnings"), list) else []
    errors = qa_payload.get("errors") if isinstance(qa_payload.get("errors"), list) else []
    lines = [
        "Secondary structure prediction",
        f"status: {status}",
        f"sequence: {input_payload.get('sequence_id', '')}",
        f"length: {input_payload.get('length', '')}",
    ]
    if mfe is not None:
        lines.append(f"mfe_kcal_mol: {mfe}")
    if pair_map:
        lines.append(f"predicted_pair_count: {len(pair_map)}")
    if dot_bracket:
        lines.append("dot_bracket:")
        lines.extend(textwrap.wrap(dot_bracket, width=96))
    if warnings:
        lines.append("warnings:")
        lines.extend(textwrap.wrap("; ".join(str(item) for item in warnings), width=96))
    if errors:
        lines.append("errors:")
        lines.extend(textwrap.wrap("; ".join(str(item) for item in errors), width=96))
    _write_text_png(lines, output_path)


def _write_text_png(lines: Sequence[str], output_path: Path) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:  # pragma: no cover - dependency is pinned in the managed environment.
        raise RetronMsdCompilerError("Pillow is required to publish Retron MSD PNG plot artifacts.") from exc

    font = ImageFont.load_default()
    line_height = 18
    width = 1200
    height = max(240, 32 + line_height * len(lines))
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    y = 16
    for line in lines:
        draw.text((18, y), line, fill="#111827", font=font)
        y += line_height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _write_combined_png(component_png: Path, folding_png: Path, output_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency is pinned in the managed environment.
        raise RetronMsdCompilerError("Pillow is required to publish Retron MSD PNG plot artifacts.") from exc

    component = Image.open(component_png).convert("RGB")
    folding = Image.open(folding_png).convert("RGB")
    gutter = 24
    width = max(component.width, folding.width)
    height = component.height + folding.height + gutter
    image = Image.new("RGB", (width, height), color="white")
    image.paste(component, ((width - component.width) // 2, 0))
    image.paste(folding, ((width - folding.width) // 2, component.height + gutter))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _sequence_index_row(
    record: MsdDesignReferenceV1,
    *,
    composition_id: str,
    sequence_length: int,
    sequence_sha256: str,
    config_path: Path,
    variant_dir: Path,
    construct_bundle: Path,
    curated: Mapping[str, object],
    root: Path,
) -> dict[str, object]:
    genbank_path = root / str(curated["genbank"])
    row: dict[str, object] = {
        "construct_id": record.construct_id,
        "msd_design_id": record.msd_design_id,
        "composition_id": composition_id,
        "unit_count": _MSD_UNIT_REPEAT_COUNT,
        "sequence_length": sequence_length,
        "sequence_sha256": sequence_sha256,
        "composition_config": config_path.relative_to(root).as_posix(),
        "artifact_bundle": variant_dir.relative_to(root).as_posix(),
        "construct_bundle": construct_bundle.relative_to(root).as_posix(),
        "genbank": curated["genbank"],
        "reverse_complement_genbank": curated["reverse_complement_genbank"],
        "forward_fasta": curated["forward_fasta"],
        "reverse_complement_fasta": curated["reverse_complement_fasta"],
        "features_csv": curated["features_csv"],
        "visual_contract": curated["visual_contract"],
        "construct_manifest": curated["construct_manifest"],
        "folding_prediction": curated["folding_prediction"],
        "folding_status": curated["folding_status"],
        "component_span_png": curated["component_span_png"],
        "folding_png": curated["folding_png"],
        "combined_plot_png": curated["combined_plot_png"],
        "finder_reveal": f"open -R {shlex.quote(genbank_path.as_posix())}",
    }
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
        "reverse_complement_genbank": row["reverse_complement_genbank"],
        "forward_fasta": row["forward_fasta"],
        "reverse_complement_fasta": row["reverse_complement_fasta"],
        "features_csv": row["features_csv"],
        "visual_contract": row["visual_contract"],
        "folding_prediction": row["folding_prediction"],
    }
    for field in ("component_span_png", "component_span_svg", "folding_png", "combined_plot_png"):
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
        "construct_bundle",
        "genbank",
        "reverse_complement_genbank",
        "forward_fasta",
        "reverse_complement_fasta",
        "features_csv",
        "visual_contract",
        "construct_manifest",
        "folding_prediction",
        "folding_status",
        "component_span_png",
        "component_span_svg",
        "folding_png",
        "combined_plot_png",
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
    catalog_path: str = CATALOG_FILENAME,
    reference_index_path: str = REFERENCE_INDEX_FILENAME,
    references_dir: str = REFERENCE_DIRNAME,
    grouped_dirs: Sequence[str] | None = None,
    top_level_files: Sequence[str] | None = None,
) -> None:
    top_level_file_list = list(
        top_level_files
        or [BUNDLE_README_FILENAME, BUNDLE_MANIFEST_FILENAME, CATALOG_FILENAME, REFERENCE_INDEX_FILENAME]
    )
    grouped_dir_list = list(grouped_dirs or [REFERENCE_DIRNAME])
    payload = {
        "contract": "msd_design_catalog_bundle_v1",
        "schema_version": 1,
        "catalog": catalog_path,
        "reference_index": reference_index_path,
        "references_dir": references_dir,
        "reference_count": len(catalog.records),
        "layout": {
            "top_level_files": top_level_file_list,
            "grouped_dirs": grouped_dir_list,
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
        "catalog": f"{MANIFEST_DIRNAME}/{CATALOG_FILENAME}",
        "sequence_index": f"{MANIFEST_DIRNAME}/{SEQUENCE_INDEX_FILENAME}",
        "manifest_dir": MANIFEST_DIRNAME,
        "variants_dir": VARIANT_DIRNAME,
        "composition_configs_dir": f"{MANIFEST_DIRNAME}/{COMPOSITION_CONFIG_DIRNAME}",
        "unit_count_per_design": _MSD_UNIT_REPEAT_COUNT,
        "render_formats": list(render_formats),
        "variant_count": len(rows),
        "variants": rows,
        "variant_layout": {
            "sequences_dir": VARIANT_SEQUENCES_DIRNAME,
            "plots_dir": VARIANT_PLOTS_DIRNAME,
            "manifest_dir": VARIANT_MANIFEST_DIRNAME,
            "runtime_dir": VARIANT_RUNTIME_DIRNAME,
            "construct_runtime_dir": f"{VARIANT_RUNTIME_DIRNAME}/{CONSTRUCT_RUNTIME_DIRNAME}",
        },
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
    if sequence_rows is not None:
        first_variant = sequence_rows[0]["artifact_bundle"] if sequence_rows else f"{VARIANT_DIRNAME}/"
        lines = [
            "# Retron MSD Sequence Bundle",
            "",
            "Generated bundle for one single-unit MSD sequence per design.",
            "",
            "Open first:",
            f"- `{MANIFEST_DIRNAME}/{SEQUENCE_INDEX_FILENAME}`: scan table with GenBank, plot, and Finder paths.",
            f"- `{MANIFEST_DIRNAME}/{SEQUENCE_MANIFEST_FILENAME}`: machine-readable bundle manifest.",
            f"- `{first_variant}/{VARIANT_SEQUENCES_DIRNAME}/forward.gb`: first forward GenBank export.",
            (
                f"- `{first_variant}/{VARIANT_SEQUENCES_DIRNAME}/reverse_complement.gb`: "
                "first reverse-complement GenBank export."
            ),
            f"- `{first_variant}/{VARIANT_PLOTS_DIRNAME}/component_span_and_folding.png`: first combined plot.",
            "",
            f"Record count: {len(catalog.records)}",
            "",
            "Layout policy:",
            (
                f"- keep the top level limited to `{BUNDLE_README_FILENAME}`, `{MANIFEST_DIRNAME}/`, "
                f"and `{VARIANT_DIRNAME}/`;"
            ),
            (
                f"- keep runtime and provenance metadata under `{MANIFEST_DIRNAME}/` or each variant "
                f"`{VARIANT_MANIFEST_DIRNAME}/`;"
            ),
            f"- keep each variant grouped by `{VARIANT_SEQUENCES_DIRNAME}/`, `{VARIANT_PLOTS_DIRNAME}/`, "
            f"`{VARIANT_MANIFEST_DIRNAME}/`, and `{VARIANT_RUNTIME_DIRNAME}/`;",
            "- sequence bundles contain one MSD unit per design; do not repeat-expand complete MSD units.",
            "",
            "Finder:",
            f"- `open {shlex.quote(path.parent.as_posix())}` opens the transient bundle directory.",
            f"- `finder_reveal` in `{MANIFEST_DIRNAME}/{SEQUENCE_INDEX_FILENAME}` reveals each forward GenBank file.",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

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
    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "BUNDLE_MANIFEST_FILENAME",
    "BUNDLE_README_FILENAME",
    "CATALOG_FILENAME",
    "COMPOSITION_CONFIG_DIRNAME",
    "DEFAULT_FLANK_3P_SUFFIX",
    "DEFAULT_FLANK_5P_PREFIX",
    "MANIFEST_DIRNAME",
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
