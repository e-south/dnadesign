"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/pipeline.py

Materialize Eco1 panel-selection artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME,
    CANDIDATE_SELECTION_PANEL_FILE_NAME,
    CANDIDATE_TRIAGE_TABLE_FILE_NAME,
    CREATED_BY,
    DEFAULT_CREATED_AT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SELECTION_DIR_NAME,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    FEASIBILITY_REPORT_FILE_NAME,
    LOCAL_STRUCTURE_REGION_METRICS_FILE_NAME,
    LOCAL_STRUCTURE_THRESHOLD_SENSITIVITY_FILE_NAME,
    MANIFEST_FILE_NAME,
    PLOTS_DIR_NAME,
    PRIMARY_PANEL_SELECTION_TRACE_FILE_NAME,
    REGION_MSA_SUPPORT_FILE_NAME,
    SELECTION_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.feasibility import (
    build_feasibility_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.handoff_readiness import (
    build_handoff_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.io import (
    read_rows,
    write_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.local_structure import (
    LOCAL_STRUCTURE_REGION_IDS,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
    LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE,
    LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM,
    build_local_structure_region_rows,
    build_local_structure_review_by_candidate,
    mapped_positions_from_residue_map,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.models import (
    MaterializedSelectionReadiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    build_primary_panel_selection_trace_rows,
    build_selection_panel_rows,
    panel_coverage_summary,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plots import (
    write_selection_readiness_plots,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.review_axes import (
    build_review_axis_by_candidate,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.sequence_export import (
    write_candidate_handoff_sequence_csv,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.triage import (
    build_triage_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_METADATA,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri

from ..review_deliverables.rt_annotation_context import RTAnnotationContext, load_rt_annotation_context
from .local_structure_sensitivity import build_local_structure_threshold_sensitivity_rows
from .region_msa_support import build_region_msa_support_rows

_OPTIONAL_REVIEW_SOURCE_KEYS = ("llr_300m", "llr_6b", "sae_window")


def materialize_selection_readiness(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    selection_root: Path | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedSelectionReadiness:
    """Materialize feasibility, triage, and primary-panel selection artifacts."""

    root = repo_root.expanduser().resolve()
    class_root = _resolve(root, output_root or DEFAULT_OUTPUT_ROOT)
    source_root = _resolve(root, source_output_root or DEFAULT_SOURCE_OUTPUT_ROOT)
    selected_root = _resolve(root, selection_root) if selection_root else class_root / DEFAULT_SELECTION_DIR_NAME
    paths = _input_paths(class_root=class_root, source_root=source_root)
    required_paths = [
        paths["candidate_pool"],
        paths["foldcheck_report"],
        paths["foldcheck_review"],
        paths["mask_set"],
        paths["conservation_profile"],
        paths["clade9_alignment"],
        paths["subtype_alignment"],
        paths["contact_geometry_profile"],
        paths["residue_map"],
    ]
    for required in required_paths:
        if not required.exists():
            raise FileNotFoundError(required)
    candidate_rows = read_rows(paths["candidate_pool"])
    foldcheck_report_rows = read_rows(paths["foldcheck_report"])
    fold_review_rows = read_rows(paths["foldcheck_review"])
    llr_300m_rows = read_rows(paths["llr_300m"], required=False)
    llr_6b_rows = read_rows(paths["llr_6b"], required=False)
    sae_window_rows = read_rows(paths["sae_window"], required=False)
    conservation_profile_rows = read_rows(paths["conservation_profile"])
    contact_geometry_rows = read_rows(paths["contact_geometry_profile"])
    mask_payload = yaml.safe_load(paths["mask_set"].read_text(encoding="utf-8"))
    mask_residues = list(mask_payload.get("residues") or [])
    feasibility_path = selected_root / FEASIBILITY_REPORT_FILE_NAME
    triage_path = selected_root / CANDIDATE_TRIAGE_TABLE_FILE_NAME
    local_structure_path = selected_root / LOCAL_STRUCTURE_REGION_METRICS_FILE_NAME
    local_structure_threshold_sensitivity_path = selected_root / LOCAL_STRUCTURE_THRESHOLD_SENSITIVITY_FILE_NAME
    region_msa_support_path = selected_root / REGION_MSA_SUPPORT_FILE_NAME
    primary_panel_selection_trace_path = selected_root / PRIMARY_PANEL_SELECTION_TRACE_FILE_NAME
    panel_path = selected_root / CANDIDATE_SELECTION_PANEL_FILE_NAME
    handoff_sequence_csv_path = selected_root / CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME
    candidate_handoff_path = source_root / "candidate_handoff.yaml"
    plots_root = selected_root / PLOTS_DIR_NAME
    (selected_root / "class_local_elimination_trace.parquet").unlink(missing_ok=True)
    feasibility_rows = build_feasibility_rows(
        candidate_rows=candidate_rows,
        foldcheck_report_rows=foldcheck_report_rows,
        input_candidate_pool_hash=sha256_uri(paths["candidate_pool"]),
        input_mask_policy_hash=sha256_uri(paths["mask_set"]),
        input_foldcheck_report_hash=sha256_uri(paths["foldcheck_report"]),
        created_at=created_at,
    )
    write_rows(feasibility_path, feasibility_rows, schema_id="eco1_rt.feasibility_report")
    input_hashes = {
        "candidate_pool": sha256_uri(paths["candidate_pool"]),
        "foldcheck_review": sha256_uri(paths["foldcheck_review"]),
        "feasibility_report": sha256_uri(feasibility_path),
        "sae_window_summary": sha256_uri(paths["sae_window"]) if paths["sae_window"].exists() else None,
        "conservation_profile": sha256_uri(paths["conservation_profile"]),
        "clade9_alignment": sha256_uri(paths["clade9_alignment"]),
        "subtype_alignment": sha256_uri(paths["subtype_alignment"]),
        "contact_geometry_profile": sha256_uri(paths["contact_geometry_profile"]),
        "residue_map": sha256_uri(paths["residue_map"]),
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
        feasibility_rows=feasibility_rows,
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
    panel_rows = build_selection_panel_rows(
        triage_rows=triage_rows,
        candidate_rows=candidate_rows,
        input_hashes=panel_hashes,
    )
    write_rows(panel_path, panel_rows, schema_id="eco1_rt.candidate_selection_panel")
    primary_panel_selection_trace_rows = build_primary_panel_selection_trace_rows(
        triage_rows=triage_rows,
        panel_rows=panel_rows,
    )
    write_rows(
        primary_panel_selection_trace_path,
        primary_panel_selection_trace_rows,
        schema_id="eco1_rt.primary_panel_selection_trace",
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
    handoff_sequence_rows = write_candidate_handoff_sequence_csv(
        handoff_sequence_csv_path,
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        source_candidate_pool_sha256=sha256_uri(paths["candidate_pool"]),
        source_panel_sha256=sha256_uri(panel_path),
    )
    plot_hashes = dict(panel_hashes)
    plot_hashes["candidate_selection_panel"] = sha256_uri(panel_path)
    plot_hashes["primary_panel_selection_trace"] = sha256_uri(primary_panel_selection_trace_path)
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
        mask_residues=mask_residues,
        local_structure_rows=local_structure_rows,
        local_structure_threshold_sensitivity_rows=local_structure_threshold_sensitivity_rows,
        region_msa_support_rows=region_msa_support_rows,
        primary_panel_selection_trace_rows=primary_panel_selection_trace_rows,
        input_hashes=plot_hashes,
        rt_annotation_context=rt_annotation_context,
    )
    manifest_path = selected_root / MANIFEST_FILE_NAME
    _write_manifest(
        manifest_path,
        paths=paths,
        feasibility_path=feasibility_path,
        triage_path=triage_path,
        local_structure_path=local_structure_path,
        local_structure_threshold_sensitivity_path=local_structure_threshold_sensitivity_path,
        region_msa_support_path=region_msa_support_path,
        primary_panel_selection_trace_path=primary_panel_selection_trace_path,
        panel_path=panel_path,
        handoff_sequence_csv_path=handoff_sequence_csv_path,
        candidate_handoff_path=candidate_handoff_path,
        plot_rows=plot_rows,
        feasibility_rows=feasibility_rows,
        triage_rows=triage_rows,
        local_structure_rows=local_structure_rows,
        local_structure_threshold_sensitivity_rows=local_structure_threshold_sensitivity_rows,
        region_msa_support_rows=region_msa_support_rows,
        primary_panel_selection_trace_rows=primary_panel_selection_trace_rows,
        local_structure_source_basis_rows=_local_structure_source_basis_rows(
            repo_root=root,
            local_structure_rows=local_structure_rows,
        ),
        panel_rows=panel_rows,
        handoff_sequence_rows=handoff_sequence_rows,
        created_at=created_at,
    )
    return MaterializedSelectionReadiness(
        feasibility_report_path=feasibility_path,
        candidate_triage_table_path=triage_path,
        local_structure_region_metrics_path=local_structure_path,
        local_structure_threshold_sensitivity_path=local_structure_threshold_sensitivity_path,
        region_msa_support_path=region_msa_support_path,
        primary_panel_selection_trace_path=primary_panel_selection_trace_path,
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
        rows.append(
            {
                "candidate_id": candidate_id,
                "design_class_id": str(candidate_row.get("design_class_id") or ""),
                "model_artifact_path": fold_row.get("model_artifact_path") or "",
            }
        )
    return rows


def _input_paths(*, class_root: Path, source_root: Path) -> dict[str, Path]:
    scoring_root = class_root / "review_deliverables/biohub_esmc_sequence_scoring"
    return {
        "candidate_pool": class_root / "candidate_pool.parquet",
        "foldcheck_report": class_root / "foldcheck_report.parquet",
        "foldcheck_review": class_root / "foldcheck_review/foldcheck_candidate_ranking.parquet",
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


def _write_manifest(
    path: Path,
    *,
    paths: dict[str, Path],
    feasibility_path: Path,
    triage_path: Path,
    local_structure_path: Path,
    local_structure_threshold_sensitivity_path: Path,
    region_msa_support_path: Path,
    primary_panel_selection_trace_path: Path,
    panel_path: Path,
    handoff_sequence_csv_path: Path,
    candidate_handoff_path: Path,
    plot_rows: list[dict[str, object]],
    feasibility_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
    local_structure_rows: list[dict[str, object]],
    local_structure_threshold_sensitivity_rows: list[dict[str, object]],
    region_msa_support_rows: list[dict[str, object]],
    primary_panel_selection_trace_rows: list[dict[str, object]],
    local_structure_source_basis_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    handoff_sequence_rows: list[dict[str, object]],
    created_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    optional_review_sources = _optional_review_sources(manifest_root=path.parent, paths=paths)
    missing_optional_review_sources = [
        source_id for source_id, source in optional_review_sources.items() if not source["materialized"]
    ]
    payload = {
        "schema_id": "eco1_rt.selection_readiness_manifest",
        "schema_version": 1,
        "status": "materialized_degraded" if missing_optional_review_sources else "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "selection_policy_id": SELECTION_POLICY_ID,
        "governing_rule": (
            "Select six primary conservative candidates globally after protein-contract gates and a stricter "
            "C-terminal/thumb primer-RNA recognition local RMSD check. Proximal MSA support, near retained DNA/RNA "
            "chemistry warnings, mutation-set dissimilarity, local structure, and fold metrics define the final "
            "selection order. Design classes remain mask-policy context, not quotas. Direct Wang thumb-contact-track "
            "edits are not ordinary-panel eligible. Do not use ESMC or SAE as positive selection evidence."
        ),
        "primary_panel_policy": {
            "policy_id": SELECTION_POLICY_ID,
            "scope": "global_primary_candidate_pool",
            "interpretation": (
                "Primary panel candidates pass the preservation contract. Rows that fail the contract remain review "
                "or context rows and are not conservative handoff choices."
            ),
        },
        "local_structure_rmsd_threshold_policy": {
            "policy_id": LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_ID,
            "policy_note": LOCAL_STRUCTURE_RMSD_THRESHOLD_POLICY_NOTE,
            "coordinate_scope": "mapped_rt_chain_ca_after_global_fit",
            "thresholds_angstrom": dict(LOCAL_STRUCTURE_RMSD_THRESHOLDS_ANGSTROM),
        },
        "local_structure_source_basis": local_structure_source_basis_rows,
        "local_structure_regions": _local_structure_region_manifest_rows(local_structure_rows),
        "sae_window_policy": (
            "SAE windows are retained as review evidence but not used for selection because the current pool "
            "does not meaningfully stratify in SAE-window space."
        ),
        "esmc_policy": "ESMC additive LLR rows are retained for review and are not used as panel-selection tie-breaks.",
        "optional_review_sources": optional_review_sources,
        "missing_optional_review_sources": missing_optional_review_sources,
        "source_tables": {
            key: _manifest_relative_path(path.parent, value)
            for key, value in paths.items()
            if value.exists() and value.is_file()
        },
        "artifacts": {
            "feasibility_report": _manifest_relative_path(path.parent, feasibility_path),
            "candidate_triage_table": _manifest_relative_path(path.parent, triage_path),
            "local_structure_region_metrics": _manifest_relative_path(path.parent, local_structure_path),
            "local_structure_threshold_sensitivity": _manifest_relative_path(
                path.parent,
                local_structure_threshold_sensitivity_path,
            ),
            "region_msa_support": _manifest_relative_path(path.parent, region_msa_support_path),
            "primary_panel_selection_trace": _manifest_relative_path(path.parent, primary_panel_selection_trace_path),
            "candidate_selection_panel": _manifest_relative_path(path.parent, panel_path),
            "candidate_handoff_sequences": _manifest_relative_path(path.parent, handoff_sequence_csv_path),
            "plots_root": PLOTS_DIR_NAME,
        },
        "path_policy": "paths_relative_to_selection_manifest",
        "plots": [_plot_manifest_row(row, manifest_root=path.parent) for row in plot_rows],
        "artifact_hashes": {
            key: sha256_uri(value)
            for key, value in {
                **{key: value for key, value in paths.items() if value.exists() and value.is_file()},
                "feasibility_report": feasibility_path,
                "candidate_triage_table": triage_path,
                "local_structure_region_metrics": local_structure_path,
                "local_structure_threshold_sensitivity": local_structure_threshold_sensitivity_path,
                "region_msa_support": region_msa_support_path,
                "primary_panel_selection_trace": primary_panel_selection_trace_path,
                "candidate_selection_panel": panel_path,
                "candidate_handoff_sequences": handoff_sequence_csv_path,
                **{str(row["plot_id"]): Path(str(row["path"])) for row in plot_rows},
            }.items()
        },
        "row_counts": {
            "feasibility_report": len(feasibility_rows),
            "candidate_triage_table": len(triage_rows),
            "local_structure_region_metrics": len(local_structure_rows),
            "local_structure_threshold_sensitivity": len(local_structure_threshold_sensitivity_rows),
            "region_msa_support": len(region_msa_support_rows),
            "primary_panel_selection_trace": len(primary_panel_selection_trace_rows),
            "candidate_selection_panel": len(panel_rows),
            "candidate_handoff_sequences": len(handoff_sequence_rows),
        },
        "gate_counts": {
            "feasibility_status": _count_by(feasibility_rows, "feasibility_status"),
            "hard_gate_status": _count_by(triage_rows, "hard_gate_status"),
            "fold_review_class": _count_by(triage_rows, "fold_review_class"),
            "local_structure_gate_status": _count_by(triage_rows, "local_structure_gate_status"),
            "sae_window_status": _count_by(triage_rows, "sae_window_status"),
            "selection_candidate_tier": _count_by(triage_rows, "selection_candidate_tier"),
            "primary_c_terminal_local_rmsd_gate_status": _count_by(
                triage_rows,
                "primary_c_terminal_local_rmsd_gate_status",
            ),
        },
        "selection_funnel_stages": primary_panel_selection_trace_rows,
        "selected_candidate_ids": [str(row["candidate_id"]) for row in panel_rows],
        "panel_coverage": panel_coverage_summary(panel_rows),
        "handoff_readiness": build_handoff_readiness(
            selection_root=path.parent,
            panel_rows=panel_rows,
            candidate_handoff_path=candidate_handoff_path,
        ),
        "hard_gate_allowed_fold_classes": ["strong_fold_preserved"],
        "default_excluded_fold_classes": ["good_fold_preserved", "low_confidence", "review_band"],
        "panel_tie_break_order": [
            "fewest proximal unsupported substitutions",
            "fewest proximal rare or unobserved substitutions",
            "fewest acidic gains near retained DNA/RNA or thumb-track",
            "fewest basic losses near retained DNA/RNA or thumb-track",
            "fewest Pro/Gly gains near retained DNA/RNA or thumb-track",
            "largest nearest selected mutation-position Jaccard distance",
            "largest nearest selected exact-substitution Jaccard distance",
            "lowest C-terminal primer-RNA recognition-region C-alpha RMSD",
            "lowest substrate-relevant local C-alpha RMSD",
            "fewest chemistry warnings",
            "fewest near retained DNA/RNA mutations",
            "selection-support MSA observed fraction",
            "selection-support MSA mean alternate-residue frequency",
            "fold metrics",
            "sequence hash",
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _manifest_relative_path(manifest_root: Path, target: Path) -> str:
    return os.path.relpath(target, start=manifest_root)


def _count_by(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "missing") for row in rows)
    return {value: counts[value] for value in sorted(counts)}


def _optional_review_sources(*, manifest_root: Path, paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    return {
        key: {
            "path": _manifest_relative_path(manifest_root, paths[key]),
            "materialized": paths[key].exists(),
            "panel_selection_role": "review_annotation_not_selector",
        }
        for key in _OPTIONAL_REVIEW_SOURCE_KEYS
    }


def _local_structure_region_manifest_rows(local_structure_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    first_row_by_region: dict[str, dict[str, object]] = {}
    for row in local_structure_rows:
        region_id = str(row.get("region_id") or "")
        if region_id and region_id not in first_row_by_region:
            first_row_by_region[region_id] = row
    rows: list[dict[str, object]] = []
    for region_id in LOCAL_STRUCTURE_REGION_IDS:
        row = first_row_by_region.get(region_id)
        if row is None:
            raise ValueError(f"Missing local-structure region rows for manifest region: {region_id}")
        rows.append(
            {
                "region_id": region_id,
                "region_label": str(row.get("region_label") or ""),
                "region_role": str(row.get("region_role") or ""),
                "region_position_count": int(row.get("region_position_count") or 0),
                "region_position_spec": str(row.get("region_position_spec") or ""),
                "region_position_source": str(row.get("region_position_source") or ""),
                "region_source_basis_ids": _json_list(row.get("region_source_basis_ids_json")),
                "coordinate_scope": str(row.get("coordinate_scope") or ""),
                "local_ca_rmsd_threshold_angstrom": float(row["local_ca_rmsd_threshold_angstrom"]),
            }
        )
    return rows


def _local_structure_source_basis_rows(
    *,
    repo_root: Path,
    local_structure_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    used_ids = {
        source_id for row in local_structure_rows for source_id in _json_list(row.get("region_source_basis_ids_json"))
    }
    source_path = repo_root / "docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml"
    if not used_ids or not source_path.exists():
        return []
    loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected manual mask authority mapping at {source_path}")
    rows: list[dict[str, object]] = []
    for source in loaded.get("source_basis", []):
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id") or "")
        if source_id not in used_ids:
            continue
        rows.append(
            {
                "id": source_id,
                "role": str(source.get("role") or ""),
                "source_ref": str(source.get("source_ref") or ""),
                "note": str(source.get("note") or ""),
            }
        )
    missing = sorted(used_ids - {str(row["id"]) for row in rows})
    if missing:
        raise ValueError(f"Missing manual-mask source_basis rows for local-structure refs: {', '.join(missing)}")
    return rows


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()


def _json_list(value: object) -> list[str]:
    if not value:
        return []
    loaded = yaml.safe_load(str(value))
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


def _plot_manifest_row(row: dict[str, object], *, manifest_root: Path) -> dict[str, object]:
    normalized = dict(row)
    normalized["path"] = str(Path(str(row["path"])).relative_to(manifest_root))
    normalized.update(SELECTION_PLOT_METADATA.get(str(row.get("plot_id") or ""), {}))
    return normalized
