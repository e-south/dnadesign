"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/reader_promoter_evidence/_fixtures.py

Build verified study and Reader promoter-evidence test fixtures.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    BindingSourceArtifact,
    load_promoter_candidate_bindings,
    materialize_promoter_candidate_bindings,
    preview_promoter_candidate_bindings,
)


def write_reader_bundle(
    root: Path,
    *,
    candidate_id: str,
    design_id: str,
    experiment_id: str,
    claim_status: str = "objective_neutral",
    adapter_kind: str = "densegen_tfbs",
    bindings_bundle: Path | None = None,
) -> tuple[Path, Path]:
    bindings_bundle = bindings_bundle or write_candidate_bindings(
        root.with_name(f"{root.name}-bindings"),
        [(candidate_id, design_id, adapter_kind)],
    )
    binding_rows = load_promoter_candidate_bindings(bindings_bundle)
    matches = binding_rows.loc[
        (binding_rows["alias_namespace"] == "reader.design_id") & (binding_rows["alias"] == design_id)
    ]
    assert len(matches) == 1
    binding = matches.iloc[0]
    binding_manifest_path = bindings_bundle / "manifest.json"
    binding_manifest = json.loads(binding_manifest_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True)
    pdf = root / "promoter_evidence.pdf"
    png = root / "promoter_evidence.png"
    pdf.write_bytes(b"%PDF-1.7\nreader evidence\n")
    png.write_bytes(b"\x89PNG\r\n\x1a\nreader evidence\n")
    manifest = {
        "schema_version": "reader.response_window.promoter_evidence_bundle.v3",
        "created_at": "2026-07-13T12:00:00+00:00",
        "claim_status": claim_status,
        "non_claim_boundary": (
            "Reader presents response-window evidence and sequence context; downstream objective scoring, "
            "normalization or calibration, and promotion remain outside Reader."
        ),
        "selection": {
            "experiment_id": experiment_id,
            "design_id": design_id,
            "candidate_id": candidate_id,
            "reduction_id": "event_logmean_6_12h_post",
        },
        "selected_binding": {
            "reader_design_id": design_id,
            "candidate_id": candidate_id,
            "sequence_sha256": "sha256:" + str(binding["sequence_sha256"]),
            "sequence_authority_dataset_id": str(binding["sequence_authority_dataset_id"]),
            "sequence_authority_id": str(binding["sequence_authority_id"]),
            "sequence_authority_sha256": "sha256:" + str(binding["sequence_authority_sha256"]),
            "source_class": str(binding["source_class"]),
            "design_family": str(binding["design_family"]),
            "binding_status": str(binding["binding_status"]),
            "binding_method": str(binding["binding_method"]),
            "densegen_plan": _optional(binding["densegen__plan"]),
            "densegen_run_id": _optional(binding["densegen__run_id"]),
            "densegen_sampling_library_hash": _optional(binding["densegen__sampling_library_hash"]),
        },
        "sources": {
            "response_window": {
                "schema_version": "reader.response_window.bundle.v5",
                "study_id": "stress_ethanol_cipro_growth",
                "request_id": "stress_ethanol_cipro_growth.response_window.v3",
                "experiment_id": experiment_id,
                "reduction_id": "event_logmean_6_12h_post",
                "manifest_sha256": "sha256:" + "1" * 64,
            },
            "candidate_bindings": {
                "schema_id": "dnadesign.study.promoter_candidate_bindings.v1",
                "schema_version": "1",
                "study_id": "stress_ethanol_cipro_growth",
                "manifest_sha256": sha256(binding_manifest_path),
                "records_sha256": "sha256:" + binding_manifest["record"]["sha256"],
                "candidate_table_id": binding_manifest["candidate_table"]["dataset_id"],
                "candidate_selection_sha256": "sha256:" + binding_manifest["candidate_table"]["selection_sha256"],
            },
            "baserender": {
                "contract_id": "dnadesign.baserender.sequence_panel.v1",
                "contract_version": "1",
                "style_profile": "promoter_compact_slide.v1",
                "renderer_name": "sequence_rows",
                "adapter_kind": adapter_kind,
                "sequence_length_bp": 60,
                "feature_count": 2,
                "strand_count": 2,
                "legend_entries": ["tf:CpxR"],
                "image_width_px": 2200,
                "image_height_px": 430,
            },
        },
        "objective_overlay": None,
        "artifacts": {
            pdf.name: {"path": pdf.name, "bytes": pdf.stat().st_size, "sha256": sha256(pdf)},
            png.name: {"path": png.name, "bytes": png.stat().st_size, "sha256": sha256(png)},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root, bindings_bundle


def write_candidate_bindings(
    bundle: Path,
    specs: list[tuple[str, str, str]],
) -> Path:
    sequence = "ACGTACGT" + "CTGACA" + "AAAA" + "TATAAT"
    aliases: list[dict[str, str]] = []
    candidates: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for candidate_id, design_id, adapter_kind in specs:
        authority = f"authority:{design_id}"
        aliases.append(
            {
                "alias_namespace": "reader.design_id",
                "alias": design_id,
                "display_label": design_id,
                "candidate_id": candidate_id,
                "authority_sequence": sequence,
                "sequence_authority_dataset_id": "reader-test-authority",
                "sequence_authority_id": authority,
                "sequence_authority_sha256": hashlib.sha256(authority.encode()).hexdigest(),
            }
        )
        densegen = adapter_kind == "densegen_tfbs"
        candidates.append(
            {
                "id": candidate_id,
                "sequence": sequence,
                "usr_label__primary": None if densegen else design_id,
                "opal_candidate__source_class": "densegen" if densegen else "construct_derived",
                "opal_candidate__design_family": "ethanol_ciprofloxacin" if densegen else "control",
                "densegen__plan": "ethanol_ciprofloxacin" if densegen else None,
                "densegen__run_id": "reader_sfxi_pdual10_archive_port" if densegen else None,
                "densegen__sampling_library_hash": "archive_library_hash" if densegen else None,
                "densegen__used_tfbs_detail": _densegen_annotations() if densegen else None,
                "densegen__required_regulators": ["baeR"] if densegen else None,
            }
        )
        if not densegen:
            annotations.append(
                {
                    "id": candidate_id,
                    "seq_annot__features": [
                        {
                            "feature_id": f"{candidate_id}-promoter",
                            "feature_type": "promoter",
                            "label": design_id,
                            "start_0": 0,
                            "end_0": 6,
                            "strand": 1,
                        }
                    ],
                    "seq_annot__source_artifact_uri": f"artifacts/genbank/{candidate_id}.gb",
                }
            )
    preview = preview_promoter_candidate_bindings(
        alias_rows=pd.DataFrame(aliases),
        candidate_records=pd.DataFrame(candidates),
        genbank_annotations=pd.DataFrame(annotations),
        candidate_table_id="usr_prom_eth_cip_opal_candidates",
        candidate_selection_sha256="4" * 64,
        source_artifacts=(BindingSourceArtifact("test-authority", "inputs/aliases.parquet", "5" * 64),),
    )
    materialize_promoter_candidate_bindings(
        preview,
        out_dir=bundle,
        allowed_output_root=bundle.parent,
    )
    return bundle


def _densegen_annotations() -> list[dict[str, object]]:
    return [
        {
            "part_kind": "tfbs",
            "sequence": "ACGT",
            "regulator": "baeR",
            "offset": 0,
            "offset_raw": 0,
            "length": 4,
            "end": 4,
            "orientation": "fwd",
        },
        {
            "part_kind": "fixed_element",
            "role": "upstream",
            "constraint_name": "sigma70_core",
            "sequence": "CTGACA",
            "offset": 8,
            "offset_raw": 8,
            "length": 6,
            "end": 14,
            "spacer_length": 4,
            "placement_index": 0,
        },
        {
            "part_kind": "fixed_element",
            "role": "downstream",
            "constraint_name": "sigma70_core",
            "sequence": "TATAAT",
            "offset": 18,
            "offset_raw": 18,
            "length": 6,
            "end": 24,
            "spacer_length": 4,
            "placement_index": 0,
        },
    ]


def _optional(value: object) -> str | None:
    return None if value is None or bool(pd.isna(value)) else str(value)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["sha256", "write_candidate_bindings", "write_reader_bundle"]
