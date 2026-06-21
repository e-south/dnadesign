"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/artifact_contracts/manifests.py

Retron MSD catalog, sequence index, and manifest writers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import Mapping, Sequence

from dnadesign.contracts.sequence import MsdDesignCatalogV1, MsdDesignReferenceV1

from .layout import (
    BUNDLE_MANIFEST_FILENAME,
    BUNDLE_README_FILENAME,
    CATALOG_FILENAME,
    COMPOSITION_CONFIG_DIRNAME,
    CONSTRUCT_RUNTIME_DIRNAME,
    MANIFEST_BUNDLE_DIRNAME,
    MANIFEST_CATALOG_DIRNAME,
    MANIFEST_CONFIGS_DIRNAME,
    MANIFEST_DIRNAME,
    MANIFEST_INDEXES_DIRNAME,
    MSD_UNIT_REPEAT_COUNT,
    REFERENCE_DIRNAME,
    REFERENCE_FILENAME,
    REFERENCE_INDEX_FILENAME,
    SEQUENCE_INDEX_FILENAME,
    SEQUENCE_MANIFEST_FILENAME,
    VARIANT_DIRNAME,
    VARIANT_MANIFEST_COMPOSITION_DIRNAME,
    VARIANT_MANIFEST_CONSTRUCT_DIRNAME,
    VARIANT_MANIFEST_DIRNAME,
    VARIANT_MANIFEST_FOLDING_DIRNAME,
    VARIANT_MANIFEST_PROVENANCE_DIRNAME,
    VARIANT_MANIFEST_REVIEWS_DIRNAME,
    VARIANT_MANIFEST_VISUAL_DIRNAME,
    VARIANT_PLOTS_DIRNAME,
    VARIANT_RUNTIME_DIRNAME,
    VARIANT_SEQUENCES_DIRNAME,
)


def reference_bundle_filename(record: MsdDesignReferenceV1) -> str:
    return f"{record.msd_design_id}.{REFERENCE_FILENAME}"


def reference_index_row(
    record: MsdDesignReferenceV1,
    *,
    reference_path: Path,
    root: Path,
) -> dict[str, object]:
    row = {
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
    row.update(_variant_index_fields(record))
    return row


def sequence_index_row(
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
        "construct_label": record.construct_label,
        "msd_design_id": record.msd_design_id,
        "composition_id": composition_id,
        "unit_count": MSD_UNIT_REPEAT_COUNT,
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
        "composition_overview_svg": curated["composition_overview_svg"],
        "composition_overview_png": curated["composition_overview_png"],
        "secondary_structure_native_png": curated["secondary_structure_native_png"],
        "finder_reveal": f"open -R {shlex.quote(genbank_path.as_posix())}",
    }
    row.update(_variant_index_fields(record))
    return row


def _variant_index_fields(record: MsdDesignReferenceV1) -> dict[str, object]:
    payload = record.payload_or_target
    variant = record.variant_metadata
    return {
        "payload_trim_id": payload.payload_trim_id or (variant.payload_trim_id if variant else "") or "",
        "payload_trim_class": payload.trim_class or "",
        "parent_payload_id": payload.parent_payload_id or "",
        "pwm_source_ref": payload.pwm_source_ref or "",
        "variant_role": (variant.variant_role if variant else None) or "",
        "scaffold_context": (variant.scaffold_context if variant else None) or "",
        "cap_selector_id": (variant.cap_selector_id if variant else None) or "",
        "stem_base_selector_id": (variant.stem_base_selector_id if variant else None) or "",
        "rt_mode": (variant.rt_mode if variant else None) or "",
        "decision_group": (variant.decision_group if variant else None) or "",
        "control_id": (variant.control_id if variant else None) or "",
    }


def record_with_sequence_artifacts(record: MsdDesignReferenceV1, *, row: Mapping[str, object]) -> MsdDesignReferenceV1:
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
    for field in (
        "composition_overview_svg",
        "composition_overview_png",
        "secondary_structure_native_png",
    ):
        value = row.get(field)
        if value:
            artifacts[field] = value
    payload["artifacts"] = artifacts
    return MsdDesignReferenceV1.model_validate(payload)


def write_reference_index(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "construct_id",
        "msd_design_id",
        "payload_id",
        "payload_trim_id",
        "payload_trim_class",
        "parent_payload_id",
        "pwm_source_ref",
        "cap_id",
        "variant_role",
        "scaffold_context",
        "cap_selector_id",
        "stem_base_selector_id",
        "rt_mode",
        "decision_group",
        "control_id",
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


def write_sequence_index(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "construct_id",
        "construct_label",
        "msd_design_id",
        "payload_trim_id",
        "payload_trim_class",
        "parent_payload_id",
        "pwm_source_ref",
        "variant_role",
        "scaffold_context",
        "cap_selector_id",
        "stem_base_selector_id",
        "rt_mode",
        "decision_group",
        "control_id",
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
        "composition_overview_svg",
        "composition_overview_png",
        "secondary_structure_native_png",
        "finder_reveal",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_bundle_manifest(
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


def write_sequence_manifest(
    path: Path,
    *,
    rows: list[dict[str, object]],
    render_formats: Sequence[str],
    root: Path,
) -> None:
    payload = {
        "contract": "msd_single_unit_sequence_bundle_v1",
        "schema_version": 1,
        "catalog": f"{MANIFEST_DIRNAME}/{MANIFEST_CATALOG_DIRNAME}/{CATALOG_FILENAME}",
        "sequence_index": f"{MANIFEST_DIRNAME}/{MANIFEST_INDEXES_DIRNAME}/{SEQUENCE_INDEX_FILENAME}",
        "manifest_dir": MANIFEST_DIRNAME,
        "variants_dir": VARIANT_DIRNAME,
        "composition_configs_dir": (f"{MANIFEST_DIRNAME}/{MANIFEST_CONFIGS_DIRNAME}/{COMPOSITION_CONFIG_DIRNAME}"),
        "manifest_layout": {
            "bundle_dir": f"{MANIFEST_DIRNAME}/{MANIFEST_BUNDLE_DIRNAME}",
            "catalog_dir": f"{MANIFEST_DIRNAME}/{MANIFEST_CATALOG_DIRNAME}",
            "configs_dir": f"{MANIFEST_DIRNAME}/{MANIFEST_CONFIGS_DIRNAME}",
            "indexes_dir": f"{MANIFEST_DIRNAME}/{MANIFEST_INDEXES_DIRNAME}",
        },
        "unit_count_per_design": MSD_UNIT_REPEAT_COUNT,
        "render_formats": list(render_formats),
        "variant_count": len(rows),
        "variants": rows,
        "variant_layout": {
            "sequences_dir": VARIANT_SEQUENCES_DIRNAME,
            "plots_dir": VARIANT_PLOTS_DIRNAME,
            "manifest_dir": VARIANT_MANIFEST_DIRNAME,
            "manifest_groups": [
                VARIANT_MANIFEST_COMPOSITION_DIRNAME,
                VARIANT_MANIFEST_CONSTRUCT_DIRNAME,
                VARIANT_MANIFEST_FOLDING_DIRNAME,
                VARIANT_MANIFEST_PROVENANCE_DIRNAME,
                VARIANT_MANIFEST_REVIEWS_DIRNAME,
                VARIANT_MANIFEST_VISUAL_DIRNAME,
            ],
            "runtime_dir": VARIANT_RUNTIME_DIRNAME,
            "construct_runtime_dir": f"{VARIANT_RUNTIME_DIRNAME}/{CONSTRUCT_RUNTIME_DIRNAME}",
        },
        "operator_hints": {
            "macos_open_bundle": f"open {shlex.quote(root.as_posix())}",
            "macos_finder_reveal_first_genbank": rows[0]["finder_reveal"] if rows else "",
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_bundle_readme(
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
            (
                f"- `{MANIFEST_DIRNAME}/{MANIFEST_INDEXES_DIRNAME}/{SEQUENCE_INDEX_FILENAME}`: "
                "scan table with GenBank, plot, and Finder paths."
            ),
            (
                f"- `{MANIFEST_DIRNAME}/{MANIFEST_BUNDLE_DIRNAME}/{SEQUENCE_MANIFEST_FILENAME}`: "
                "machine-readable bundle manifest."
            ),
            f"- `{first_variant}/{VARIANT_SEQUENCES_DIRNAME}/forward.gb`: first forward GenBank export.",
            (
                f"- `{first_variant}/{VARIANT_SEQUENCES_DIRNAME}/reverse_complement.gb`: "
                "first reverse-complement GenBank export."
            ),
            (
                f"- `{first_variant}/{VARIANT_PLOTS_DIRNAME}/secondary_structure.native.png`: "
                "first native ViennaRNA structure plot."
            ),
            (
                f"- `{first_variant}/{VARIANT_PLOTS_DIRNAME}/composition_overview.svg`: "
                "first two-row structure/component review."
            ),
            (
                f"- `{first_variant}/{VARIANT_PLOTS_DIRNAME}/composition_overview.png`: "
                "high-resolution PNG sibling for the first two-row review."
            ),
            "",
            f"Record count: {len(catalog.records)}",
            "",
            "Layout policy:",
            (
                f"- keep the top level limited to `{BUNDLE_README_FILENAME}`, `{MANIFEST_DIRNAME}/`, "
                f"and `{VARIANT_DIRNAME}/`;"
            ),
            (
                f"- keep root metadata grouped under `{MANIFEST_DIRNAME}/{MANIFEST_BUNDLE_DIRNAME}/`, "
                f"`{MANIFEST_DIRNAME}/{MANIFEST_CATALOG_DIRNAME}/`, "
                f"`{MANIFEST_DIRNAME}/{MANIFEST_CONFIGS_DIRNAME}/`, and "
                f"`{MANIFEST_DIRNAME}/{MANIFEST_INDEXES_DIRNAME}/`;"
            ),
            (
                f"- keep each variant grouped by `{VARIANT_SEQUENCES_DIRNAME}/`, `{VARIANT_PLOTS_DIRNAME}/`, "
                f"`{VARIANT_MANIFEST_DIRNAME}/`, and `{VARIANT_RUNTIME_DIRNAME}/`; variant manifests group "
                "composition, construct, folding, provenance, review, and visual records;"
            ),
            "- sequence bundles contain one MSD unit per design; do not repeat-expand complete MSD units.",
            "",
            "Finder:",
            f"- `open {shlex.quote(path.parent.as_posix())}` opens the transient bundle directory.",
            (
                f"- `finder_reveal` in `{MANIFEST_DIRNAME}/{MANIFEST_INDEXES_DIRNAME}/{SEQUENCE_INDEX_FILENAME}` "
                "reveals each forward GenBank file."
            ),
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
    "record_with_sequence_artifacts",
    "reference_bundle_filename",
    "reference_index_row",
    "sequence_index_row",
    "write_bundle_manifest",
    "write_bundle_readme",
    "write_reference_index",
    "write_sequence_index",
    "write_sequence_manifest",
]
