"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/compiler.py

Study-local Retron MSD design-reference compiler.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from dnadesign.construct import run_linear_ssdna_composition
from dnadesign.contracts.sequence import MsdDesignCatalogV1, MsdDesignReferenceV1

from .composition_payload import (
    composition_config_payload,
    normalize_render_formats,
    render_formats_for_review,
    require_sequence_subcomponents,
)
from .errors import RetronMsdCompilerError
from .layout import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_README_FILENAME,
    CATALOG_FILENAME,
    COMPOSITION_CONFIG_DIRNAME,
    CONSTRUCT_RUNTIME_DIRNAME,
    DEFAULT_FLANK_3P_SUFFIX,
    DEFAULT_FLANK_5P_PREFIX,
    MANIFEST_BUNDLE_DIRNAME,
    MANIFEST_CATALOG_DIRNAME,
    MANIFEST_CONFIGS_DIRNAME,
    MANIFEST_DIRNAME,
    MANIFEST_INDEXES_DIRNAME,
    REFERENCE_DIRNAME,
    REFERENCE_INDEX_FILENAME,
    SEQUENCE_INDEX_FILENAME,
    SEQUENCE_MANIFEST_FILENAME,
    VARIANT_DIRNAME,
    VARIANT_RUNTIME_DIRNAME,
)
from .manifests import (
    record_with_sequence_artifacts,
    reference_bundle_filename,
    reference_index_row,
    sequence_index_row,
    write_bundle_manifest,
    write_bundle_readme,
    write_reference_index,
    write_sequence_index,
    write_sequence_manifest,
)
from .materialized_outputs import publish_variant_outputs, run_baserender_jobs
from .msd_ids import parse_msd_construct_label
from .output_guards import guard_catalog_output_layout, guard_materialize_output_layout
from .registry import load_retron_msd_registry


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
    formats = normalize_render_formats(render_formats)
    payload_sequences = {str(key).strip(): str(value).strip() for key, value in payload_sequences.items()}
    cap_sequences = {str(key).strip(): str(value).strip() for key, value in cap_sequences.items()}
    require_sequence_subcomponents(catalog, payload_sequences=payload_sequences, cap_sequences=cap_sequences)

    root = Path(out_dir).expanduser().resolve()
    expected_design_ids = {record.msd_design_id for record in catalog.records}
    reference_filenames = {reference_bundle_filename(record) for record in catalog.records}
    guard_materialize_output_layout(
        root,
        expected_design_ids=expected_design_ids,
        expected_reference_filenames=reference_filenames,
    )
    root.mkdir(parents=True, exist_ok=True)

    manifest_dir = root / MANIFEST_DIRNAME
    bundle_manifest_dir = manifest_dir / MANIFEST_BUNDLE_DIRNAME
    catalog_dir = manifest_dir / MANIFEST_CATALOG_DIRNAME
    configs_dir = manifest_dir / MANIFEST_CONFIGS_DIRNAME / COMPOSITION_CONFIG_DIRNAME
    indexes_dir = manifest_dir / MANIFEST_INDEXES_DIRNAME
    variants_dir = root / VARIANT_DIRNAME
    for directory in (bundle_manifest_dir, catalog_dir, configs_dir, indexes_dir):
        directory.mkdir(parents=True, exist_ok=True)
    variants_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    updated_records: list[MsdDesignReferenceV1] = []
    producer_render_formats = render_formats_for_review(formats)
    for record in catalog.records:
        variant_dir = variants_dir / record.msd_design_id
        artifact_bundle = variant_dir / VARIANT_RUNTIME_DIRNAME / CONSTRUCT_RUNTIME_DIRNAME
        config_path = configs_dir / f"{record.msd_design_id}.linear_ssdna_composition.yaml"
        config_payload = composition_config_payload(
            record,
            artifact_bundle=artifact_bundle,
            payload_sequence=payload_sequences[record.payload_or_target.id],
            cap_sequence=cap_sequences[record.cap.id],
            flank_5p_prefix=flank_5p_prefix,
            flank_3p_suffix=flank_3p_suffix,
            render_formats=producer_render_formats,
        )
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
        try:
            composition = run_linear_ssdna_composition(config_path)
        except Exception as exc:  # pragma: no cover - depends on producer failure mode.
            raise RetronMsdCompilerError(
                f"Construct failed to emit Retron MSD unit '{record.msd_design_id}': {exc}"
            ) from exc
        run_baserender_jobs(
            composition.artifact_bundle,
            formats=producer_render_formats,
            enabled=run_baserender,
        )
        curated = publish_variant_outputs(
            variant_dir,
            construct_bundle=composition.artifact_bundle,
            root=root,
        )
        row = sequence_index_row(
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
        updated_records.append(record_with_sequence_artifacts(record, row=row))

    updated_catalog = MsdDesignCatalogV1(records=updated_records)
    _write_materialized_catalog(
        updated_catalog,
        root=root,
        bundle_manifest_dir=bundle_manifest_dir,
        catalog_dir=catalog_dir,
        indexes_dir=indexes_dir,
    )
    write_sequence_index(indexes_dir / SEQUENCE_INDEX_FILENAME, rows)
    write_sequence_manifest(
        bundle_manifest_dir / SEQUENCE_MANIFEST_FILENAME,
        rows=rows,
        render_formats=producer_render_formats,
        root=root,
    )
    write_bundle_readme(root / BUNDLE_README_FILENAME, catalog=updated_catalog, sequence_rows=rows)
    return MsdSequenceBundleResult(
        catalog=updated_catalog,
        bundle_root=root,
        manifest_path=bundle_manifest_dir / SEQUENCE_MANIFEST_FILENAME,
        index_path=indexes_dir / SEQUENCE_INDEX_FILENAME,
        variants=rows,
    )


def _write_msd_design_catalog(
    catalog: MsdDesignCatalogV1,
    *,
    out_dir: str | Path,
    extra_allowed_top_level: set[str],
) -> Path:
    root = Path(out_dir).expanduser().resolve()
    reference_filenames = [reference_bundle_filename(record) for record in catalog.records]
    duplicate_reference_filenames = sorted(
        {filename for filename in reference_filenames if reference_filenames.count(filename) > 1}
    )
    if duplicate_reference_filenames:
        raise RetronMsdCompilerError(
            f"Duplicate MSD design reference filename(s): {', '.join(duplicate_reference_filenames)}"
        )

    guard_catalog_output_layout(
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
        reference_path.write_text(record.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
        reference_rows.append(reference_index_row(record, reference_path=reference_path, root=root))

    catalog_path = root / CATALOG_FILENAME
    catalog_path.write_text(catalog.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    write_reference_index(root / REFERENCE_INDEX_FILENAME, reference_rows)
    write_bundle_manifest(root / BUNDLE_MANIFEST_FILENAME, catalog=catalog, reference_rows=reference_rows)
    write_bundle_readme(root / BUNDLE_README_FILENAME, catalog=catalog)
    return catalog_path


def _write_materialized_catalog(
    catalog: MsdDesignCatalogV1,
    *,
    root: Path,
    bundle_manifest_dir: Path,
    catalog_dir: Path,
    indexes_dir: Path,
) -> Path:
    reference_filenames = [reference_bundle_filename(record) for record in catalog.records]
    duplicate_reference_filenames = sorted(
        {filename for filename in reference_filenames if reference_filenames.count(filename) > 1}
    )
    if duplicate_reference_filenames:
        raise RetronMsdCompilerError(
            f"Duplicate MSD design reference filename(s): {', '.join(duplicate_reference_filenames)}"
        )

    references_dir = catalog_dir / REFERENCE_DIRNAME
    references_dir.mkdir(parents=True, exist_ok=True)
    reference_rows: list[dict[str, object]] = []
    for record, reference_filename in zip(catalog.records, reference_filenames, strict=True):
        reference_path = references_dir / reference_filename
        reference_path.write_text(record.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
        reference_rows.append(reference_index_row(record, reference_path=reference_path, root=root))

    catalog_path = catalog_dir / CATALOG_FILENAME
    catalog_path.write_text(catalog.model_dump_json(indent=2, exclude_none=True) + "\n", encoding="utf-8")
    write_reference_index(indexes_dir / REFERENCE_INDEX_FILENAME, reference_rows)
    write_bundle_manifest(
        bundle_manifest_dir / BUNDLE_MANIFEST_FILENAME,
        catalog=catalog,
        reference_rows=reference_rows,
        catalog_path=catalog_path.relative_to(root).as_posix(),
        reference_index_path=(indexes_dir / REFERENCE_INDEX_FILENAME).relative_to(root).as_posix(),
        references_dir=references_dir.relative_to(root).as_posix(),
        grouped_dirs=[MANIFEST_DIRNAME, VARIANT_DIRNAME],
        top_level_files=[BUNDLE_README_FILENAME],
    )
    return catalog_path


__all__ = [
    "MsdSequenceBundleResult",
    "build_msd_design_reference",
    "compile_msd_design_catalog",
    "materialize_msd_design_artifacts",
    "write_msd_design_catalog",
]
