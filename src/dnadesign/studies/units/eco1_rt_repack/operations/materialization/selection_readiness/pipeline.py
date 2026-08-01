"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/pipeline.py

Materialize Eco1 panel-selection artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME,
    CANDIDATE_SELECTION_PANEL_FILE_NAME,
    CANDIDATE_TRIAGE_TABLE_FILE_NAME,
    DEFAULT_CREATED_AT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SELECTION_DIR_NAME,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    HYPOTHESIS_PANEL_SELECTION_TRACE_FILE_NAME,
    LOCAL_STRUCTURE_REGION_METRICS_FILE_NAME,
    LOCAL_STRUCTURE_THRESHOLD_SENSITIVITY_FILE_NAME,
    MANIFEST_FILE_NAME,
    PLOTS_DIR_NAME,
    REGION_MSA_SUPPORT_FILE_NAME,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.io import (
    read_rows,
    write_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    build_local_structure_region_rows,
    build_local_structure_review_by_candidate,
    mapped_positions_from_residue_map,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.manifest import (
    build_local_structure_source_basis_rows,
    write_selection_readiness_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.models import (
    MaterializedSelectionReadiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    build_selected_panel_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_trace import (
    build_selected_panel_trace_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plots import (
    write_selection_readiness_plots,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.review_axes import (
    build_review_axis_by_candidate,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.sequence_export import (
    read_fasta_sequences,
    write_candidate_handoff_sequence_csv,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.triage import (
    build_triage_rows,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri

from ..shared.rt_annotation_context import RTAnnotationContext, load_rt_annotation_context
from .local_structure_sensitivity import build_local_structure_threshold_sensitivity_rows
from .region_msa_support import build_region_msa_support_rows


def materialize_selection_readiness(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    selection_root: Path | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedSelectionReadiness:
    """Materialize triage and the policy-defined eight-sequence panel."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_OUTPUT_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    selected_root = _resolve(root, selection_root) if selection_root else class_root / DEFAULT_SELECTION_DIR_NAME
    _remove_retired_selection_artifacts(selected_root)
    paths = _input_paths(class_root=class_root, source_root=source_root)
    required_paths = [
        paths["candidate_pool"],
        paths["foldcheck_review"],
        paths["mask_set"],
        paths["conservation_profile"],
        paths["clade9_alignment"],
        paths["subtype_alignment"],
        paths["contact_geometry_profile"],
        paths["residue_map"],
        paths["foldcheck_input_sequences"],
    ]
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    candidate_rows = read_rows(paths["candidate_pool"])
    fold_review_rows = read_rows(paths["foldcheck_review"])
    llr_300m_rows = read_rows(paths["llr_300m"], required=False)
    llr_6b_rows = read_rows(paths["llr_6b"], required=False)
    sae_window_rows = read_rows(paths["sae_window"], required=False)
    conservation_profile_rows = read_rows(paths["conservation_profile"])
    contact_geometry_rows = read_rows(paths["contact_geometry_profile"])
    mask_payload = yaml.safe_load(paths["mask_set"].read_text(encoding="utf-8"))
    mask_residues = list(mask_payload.get("residues") or [])
    triage_path = selected_root / CANDIDATE_TRIAGE_TABLE_FILE_NAME
    local_structure_path = selected_root / LOCAL_STRUCTURE_REGION_METRICS_FILE_NAME
    local_structure_threshold_sensitivity_path = selected_root / LOCAL_STRUCTURE_THRESHOLD_SENSITIVITY_FILE_NAME
    region_msa_support_path = selected_root / REGION_MSA_SUPPORT_FILE_NAME
    hypothesis_panel_selection_trace_path = selected_root / HYPOTHESIS_PANEL_SELECTION_TRACE_FILE_NAME
    panel_path = selected_root / CANDIDATE_SELECTION_PANEL_FILE_NAME
    handoff_sequence_csv_path = selected_root / CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME
    candidate_handoff_path = source_root / "candidate_handoff.yaml"
    plots_root = selected_root / PLOTS_DIR_NAME
    input_hashes = {
        "candidate_pool": sha256_uri(paths["candidate_pool"]),
        "foldcheck_review": sha256_uri(paths["foldcheck_review"]),
        "sae_window_summary": sha256_uri(paths["sae_window"]) if paths["sae_window"].exists() else None,
        "conservation_profile": sha256_uri(paths["conservation_profile"]),
        "clade9_alignment": sha256_uri(paths["clade9_alignment"]),
        "subtype_alignment": sha256_uri(paths["subtype_alignment"]),
        "contact_geometry_profile": sha256_uri(paths["contact_geometry_profile"]),
        "residue_map": sha256_uri(paths["residue_map"]),
        "foldcheck_input_sequences": sha256_uri(paths["foldcheck_input_sequences"]),
        "foldcheck_full_structure_set": (
            sha256_uri(paths["foldcheck_full_structure_set"])
            if paths["foldcheck_full_structure_set"].exists()
            else None
        ),
        "foldcheck_reference_backbone": (
            sha256_uri(paths["foldcheck_reference_backbone"])
            if paths["foldcheck_reference_backbone"].exists()
            else None
        ),
    }
    review_axis_by_candidate = build_review_axis_by_candidate(
        candidate_rows=candidate_rows,
        conservation_profile_rows=conservation_profile_rows,
        clade9_alignment_path=paths["clade9_alignment"],
        subtype_alignment_path=paths["subtype_alignment"],
        contact_geometry_rows=contact_geometry_rows,
        mask_residues=mask_residues,
    )
    region_msa_support_rows = build_region_msa_support_rows(
        candidate_rows=candidate_rows,
        conservation_profile_rows=conservation_profile_rows,
        clade9_alignment_path=paths["clade9_alignment"],
        subtype_alignment_path=paths["subtype_alignment"],
        contact_geometry_rows=contact_geometry_rows,
        mask_residues=mask_residues,
    )
    write_rows(
        region_msa_support_path,
        region_msa_support_rows,
        schema_id="eco1_rt.region_msa_support",
    )
    local_structure_rows = build_local_structure_region_rows(
        fold_review_rows=_local_structure_source_rows(
            candidate_rows=candidate_rows,
            fold_review_rows=fold_review_rows,
        ),
        candidate_rows=candidate_rows,
        reference_backbone_path=paths["foldcheck_reference_backbone"],
        model_root=paths["foldcheck_full_structure_root"],
        mapped_positions=mapped_positions_from_residue_map(paths["residue_map"]),
        contact_geometry_rows=contact_geometry_rows,
    )
    write_rows(
        local_structure_path,
        local_structure_rows,
        schema_id="eco1_rt.local_structure_region_metrics",
    )
    local_structure_review_by_candidate = build_local_structure_review_by_candidate(local_structure_rows)
    triage_rows = build_triage_rows(
        candidate_rows=candidate_rows,
        fold_review_rows=fold_review_rows,
        llr_300m_rows=llr_300m_rows,
        llr_6b_rows=llr_6b_rows,
        sae_window_rows=sae_window_rows,
        review_axis_by_candidate=review_axis_by_candidate,
        local_structure_review_by_candidate=local_structure_review_by_candidate,
        region_msa_support_rows=region_msa_support_rows,
        input_hashes=input_hashes,
    )
    write_rows(triage_path, triage_rows, schema_id="eco1_rt.candidate_triage_table")
    panel_hashes = dict(input_hashes)
    panel_hashes["candidate_triage_table"] = sha256_uri(triage_path)
    rt_annotation_context = _load_rt_annotation_context_if_available(root)
    panel_rows = build_selected_panel_rows(
        triage_rows=triage_rows,
        candidate_rows=candidate_rows,
        input_hashes=panel_hashes,
    )
    write_rows(panel_path, panel_rows, schema_id="eco1_rt.candidate_selection_panel")
    hypothesis_panel_selection_trace_rows = build_selected_panel_trace_rows(
        triage_rows=triage_rows,
        panel_rows=panel_rows,
    )
    write_rows(
        hypothesis_panel_selection_trace_path,
        hypothesis_panel_selection_trace_rows,
        schema_id="eco1_rt.hypothesis_panel_selection_trace",
    )
    local_structure_threshold_sensitivity_rows = build_local_structure_threshold_sensitivity_rows(
        local_structure_rows=local_structure_rows,
        selected_candidate_ids=[str(row["candidate_id"]) for row in panel_rows],
    )
    write_rows(
        local_structure_threshold_sensitivity_path,
        local_structure_threshold_sensitivity_rows,
        schema_id="eco1_rt.local_structure_threshold_sensitivity",
    )
    canonical_sequences_by_id = read_fasta_sequences(paths["foldcheck_input_sequences"])
    handoff_sequence_rows = write_candidate_handoff_sequence_csv(
        handoff_sequence_csv_path,
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        canonical_sequences_by_id=canonical_sequences_by_id,
        source_candidate_pool_sha256=sha256_uri(paths["candidate_pool"]),
        source_panel_sha256=sha256_uri(panel_path),
        source_foldcheck_input_sequences_sha256=sha256_uri(paths["foldcheck_input_sequences"]),
    )
    plot_hashes = dict(panel_hashes)
    plot_hashes["candidate_selection_panel"] = sha256_uri(panel_path)
    plot_hashes["hypothesis_panel_selection_trace"] = sha256_uri(hypothesis_panel_selection_trace_path)
    plot_hashes["local_structure_region_metrics"] = sha256_uri(local_structure_path)
    plot_hashes["local_structure_threshold_sensitivity"] = sha256_uri(local_structure_threshold_sensitivity_path)
    plot_hashes["region_msa_support"] = sha256_uri(region_msa_support_path)
    if rt_annotation_context is not None:
        plot_hashes["rt_annotation_tracks"] = sha256_uri(rt_annotation_context.annotation_tracks_path)
        plot_hashes["manual_mask_authority_source"] = sha256_uri(
            rt_annotation_context.manual_mask_authority_source_path
        )
    plot_rows = write_selection_readiness_plots(
        plot_root=plots_root,
        triage_rows=triage_rows,
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        canonical_sequences_by_id=canonical_sequences_by_id,
        mask_residues=mask_residues,
        local_structure_rows=local_structure_rows,
        local_structure_threshold_sensitivity_rows=local_structure_threshold_sensitivity_rows,
        region_msa_support_rows=region_msa_support_rows,
        hypothesis_panel_selection_trace_rows=hypothesis_panel_selection_trace_rows,
        input_hashes=plot_hashes,
        rt_annotation_context=rt_annotation_context,
    )
    manifest_path = selected_root / MANIFEST_FILE_NAME
    write_selection_readiness_manifest(
        manifest_path,
        paths=paths,
        triage_path=triage_path,
        local_structure_path=local_structure_path,
        local_structure_threshold_sensitivity_path=local_structure_threshold_sensitivity_path,
        region_msa_support_path=region_msa_support_path,
        hypothesis_panel_selection_trace_path=hypothesis_panel_selection_trace_path,
        panel_path=panel_path,
        handoff_sequence_csv_path=handoff_sequence_csv_path,
        candidate_handoff_path=candidate_handoff_path,
        plot_rows=plot_rows,
        triage_rows=triage_rows,
        local_structure_rows=local_structure_rows,
        local_structure_threshold_sensitivity_rows=local_structure_threshold_sensitivity_rows,
        region_msa_support_rows=region_msa_support_rows,
        hypothesis_panel_selection_trace_rows=hypothesis_panel_selection_trace_rows,
        local_structure_source_basis_rows=build_local_structure_source_basis_rows(
            repo_root=root,
            local_structure_rows=local_structure_rows,
        ),
        candidate_rows=candidate_rows,
        panel_rows=panel_rows,
        handoff_sequence_rows=handoff_sequence_rows,
        created_at=created_at,
    )
    return MaterializedSelectionReadiness(
        candidate_triage_table_path=triage_path,
        local_structure_region_metrics_path=local_structure_path,
        local_structure_threshold_sensitivity_path=local_structure_threshold_sensitivity_path,
        region_msa_support_path=region_msa_support_path,
        hypothesis_panel_selection_trace_path=hypothesis_panel_selection_trace_path,
        candidate_selection_panel_path=panel_path,
        candidate_handoff_sequence_csv_path=handoff_sequence_csv_path,
        plots_root=plots_root,
        manifest_path=manifest_path,
    )


def _load_rt_annotation_context_if_available(repo_root: Path) -> RTAnnotationContext | None:
    annotation_tracks_path = repo_root / "docs/studies/eco1_rt_repack/workbench/ontology/rt-annotation-tracks.yaml"
    manual_mask_authority_source_path = (
        repo_root / "docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml"
    )
    if not annotation_tracks_path.exists() or not manual_mask_authority_source_path.exists():
        return None
    return load_rt_annotation_context(
        annotation_tracks_path=annotation_tracks_path,
        manual_mask_authority_source_path=manual_mask_authority_source_path,
    )


def _local_structure_source_rows(
    *,
    candidate_rows: list[dict[str, object]],
    fold_review_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    fold_review_by_candidate = {str(row["candidate_id"]): row for row in fold_review_rows if row.get("candidate_id")}
    rows: list[dict[str, object]] = []
    for candidate_row in candidate_rows:
        if str(candidate_row.get("status")) != "accepted":
            continue
        candidate_id = str(candidate_row["candidate_id"])
        fold_row = fold_review_by_candidate.get(candidate_id, {})
        policy_context_id = str(candidate_row.get("primary_policy_id") or candidate_row.get("policy_id") or "")
        if not policy_context_id:
            raise ValueError(f"Candidate {candidate_id!r} is missing generation policy provenance")
        rows.append(
            {
                "candidate_id": candidate_id,
                "policy_id": policy_context_id,
                "model_artifact_path": fold_row.get("model_artifact_path") or "",
            }
        )
    return rows


def _input_paths(*, class_root: Path, source_root: Path) -> dict[str, Path]:
    scoring_root = class_root / "review_deliverables/biohub_esmc_sequence_scoring"
    return {
        "candidate_pool": class_root / "candidate_pool.parquet",
        "foldcheck_review": class_root / "foldcheck_review/foldcheck_candidate_ranking.parquet",
        "foldcheck_input_sequences": class_root / "foldcheck_request/input_sequences.fasta",
        "mask_set": source_root / "mask_set.yaml",
        "conservation_profile": source_root / "conservation_profile.parquet",
        "clade9_alignment": source_root / "conservation_alignments/ec86_clade9_conservation_v1.aligned.fasta",
        "subtype_alignment": source_root
        / "conservation_alignments/ec86_iia3_cluster42_1_conservation_v1.aligned.fasta",
        "contact_geometry_profile": source_root / "contact_geometry_profile.parquet",
        "residue_map": source_root / "residue_map.parquet",
        "foldcheck_full_structure_set": class_root / "foldcheck_review/foldcheck_full_structure_set.yaml",
        "foldcheck_reference_backbone": class_root
        / "foldcheck_review/structures/ec86kit_chain_a_backbone_reference.pdb",
        "foldcheck_full_structure_root": class_root / "foldcheck_review/structures/full_fold_set",
        "llr_300m": scoring_root / "biohub_esmc_variant_llr_scores.parquet",
        "llr_6b": scoring_root / "esmc_6b_2024_12/biohub_esmc_variant_llr_scores.parquet",
        "sae_window": class_root / "biohub_esmc/sae_feature_window_summary.parquet",
    }


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()


def _remove_retired_selection_artifacts(selection_root: Path) -> None:
    for file_name in ("feasibility_report.parquet",):
        path = selection_root / file_name
        if path.is_file():
            path.unlink()
