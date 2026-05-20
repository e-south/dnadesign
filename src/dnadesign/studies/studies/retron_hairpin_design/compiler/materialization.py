"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/compiler/materialization.py

Retron MSD sequence-bundle materialization orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml

from dnadesign.construct import run_linear_ssdna_composition
from dnadesign.contracts.sequence import MsdDesignCatalogV1, MsdDesignReferenceV1

from ..catalog.sequence_inputs import validate_dna_sequence
from ..outputs.composition_payload import (
    composition_config_payload,
    normalize_render_formats,
    render_formats_for_review,
    require_sequence_subcomponents,
)
from ..outputs.layout import (
    BUNDLE_README_FILENAME,
    COMPOSITION_CONFIG_DIRNAME,
    CONSTRUCT_RUNTIME_DIRNAME,
    DEFAULT_FLANK_3P_SUFFIX,
    DEFAULT_FLANK_5P_PREFIX,
    MANIFEST_BUNDLE_DIRNAME,
    MANIFEST_CATALOG_DIRNAME,
    MANIFEST_CONFIGS_DIRNAME,
    MANIFEST_DIRNAME,
    MANIFEST_INDEXES_DIRNAME,
    SEQUENCE_INDEX_FILENAME,
    SEQUENCE_MANIFEST_FILENAME,
    VARIANT_DIRNAME,
    VARIANT_RUNTIME_DIRNAME,
)
from ..outputs.manifests import (
    record_with_sequence_artifacts,
    reference_bundle_filename,
    sequence_index_row,
    write_bundle_readme,
    write_sequence_index,
    write_sequence_manifest,
)
from ..outputs.materialized_outputs import publish_variant_outputs, run_baserender_jobs
from ..outputs.output_guards import guard_materialize_output_layout
from .catalog_bundle import write_materialized_catalog
from .exceptions import RetronMsdCompilerError


@dataclass(frozen=True)
class MsdSequenceBundleResult:
    catalog: MsdDesignCatalogV1
    bundle_root: Path
    manifest_path: Path
    index_path: Path
    variants: list[dict[str, object]]


def variant_bundle_dirname(record: MsdDesignReferenceV1) -> str:
    construct_token = _path_token(record.construct_id, label="construct_id")
    design_token = _path_token(record.msd_design_id, label="msd_design_id")
    return f"{construct_token}__{design_token}"


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
    payload_sequences = _normalize_sequence_mapping(payload_sequences, label="payload_sequences")
    cap_sequences = _normalize_sequence_mapping(cap_sequences, label="cap_sequences")
    require_sequence_subcomponents(catalog, payload_sequences=payload_sequences, cap_sequences=cap_sequences)

    root = Path(out_dir).expanduser().resolve()
    expected_variant_dirnames = {variant_bundle_dirname(record) for record in catalog.records}
    reference_filenames = {reference_bundle_filename(record) for record in catalog.records}
    guard_materialize_output_layout(
        root,
        expected_variant_dirnames=expected_variant_dirnames,
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
    updated_records = []
    producer_render_formats = render_formats_for_review(formats)
    for record in catalog.records:
        variant_dirname = variant_bundle_dirname(record)
        variant_dir = variants_dir / variant_dirname
        artifact_bundle = variant_dir / VARIANT_RUNTIME_DIRNAME / CONSTRUCT_RUNTIME_DIRNAME
        config_path = configs_dir / f"{variant_dirname}.linear_ssdna_composition.yaml"
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
    write_materialized_catalog(
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


def _normalize_sequence_mapping(values: Mapping[str, str], *, label: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for raw_key, raw_sequence in values.items():
        sequence_id = str(raw_key).strip()
        if not sequence_id:
            raise RetronMsdCompilerError(f"{label} contains a blank key.")
        if sequence_id in resolved:
            raise RetronMsdCompilerError(f"{label} contains duplicate key after trimming: {sequence_id}.")
        try:
            resolved[sequence_id] = validate_dna_sequence(str(raw_sequence), label=f"{label}.{sequence_id}")
        except ValueError as exc:
            raise RetronMsdCompilerError(str(exc)) from exc
    return resolved


def _path_token(value: str, *, label: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip(".-")
    if not token:
        raise RetronMsdCompilerError(f"{label} cannot be represented as an MSD variant path token.")
    return token


__all__ = ["MsdSequenceBundleResult", "materialize_msd_design_artifacts", "variant_bundle_dirname"]
