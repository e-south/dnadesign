"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/pipeline.py

Materialize Eco1 panel-selection artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.constants import (
    CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME,
    CANDIDATE_SELECTION_PANEL_FILE_NAME,
    CANDIDATE_TRIAGE_TABLE_FILE_NAME,
    CODON_POLICY_ID,
    CREATED_BY,
    DEFAULT_CREATED_AT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SELECTION_DIR_NAME,
    DEFAULT_SOURCE_OUTPUT_ROOT,
    FEASIBILITY_REPORT_FILE_NAME,
    MANIFEST_FILE_NAME,
    PLOTS_DIR_NAME,
    SELECTION_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.feasibility import (
    build_feasibility_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.io import (
    read_rows,
    write_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.models import (
    MaterializedSelectionReadiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel import (
    build_selection_panel_rows,
    panel_coverage_summary,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plots import (
    write_selection_readiness_plots,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.review_axes import (
    build_review_axis_by_candidate,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.triage import (
    build_triage_rows,
)
from dnadesign.thread.adapters.proteinmpnn.hashing import sha256_uri

from ..review_deliverables.rt_annotation_context import RTAnnotationContext, load_rt_annotation_context


def materialize_selection_readiness(
    *,
    repo_root: Path,
    output_root: Path | None = None,
    source_output_root: Path | None = None,
    selection_root: Path | None = None,
    created_at: str = DEFAULT_CREATED_AT,
) -> MaterializedSelectionReadiness:
    """Materialize feasibility, triage, and class-balanced panel artifacts."""

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
    panel_path = selected_root / CANDIDATE_SELECTION_PANEL_FILE_NAME
    handoff_sequence_csv_path = selected_root / CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME
    plots_root = selected_root / PLOTS_DIR_NAME
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
    }
    review_axis_by_candidate = build_review_axis_by_candidate(
        candidate_rows=candidate_rows,
        conservation_profile_rows=conservation_profile_rows,
        clade9_alignment_path=paths["clade9_alignment"],
        subtype_alignment_path=paths["subtype_alignment"],
        contact_geometry_rows=contact_geometry_rows,
        mask_residues=mask_residues,
    )
    triage_rows = build_triage_rows(
        candidate_rows=candidate_rows,
        fold_review_rows=fold_review_rows,
        feasibility_rows=feasibility_rows,
        llr_300m_rows=llr_300m_rows,
        llr_6b_rows=llr_6b_rows,
        sae_window_rows=sae_window_rows,
        review_axis_by_candidate=review_axis_by_candidate,
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
    handoff_sequence_rows = _write_candidate_handoff_sequence_csv(
        handoff_sequence_csv_path,
        panel_rows=panel_rows,
        candidate_rows=candidate_rows,
        source_candidate_pool_sha256=sha256_uri(paths["candidate_pool"]),
        source_panel_sha256=sha256_uri(panel_path),
    )
    plot_hashes = dict(panel_hashes)
    plot_hashes["candidate_selection_panel"] = sha256_uri(panel_path)
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
        input_hashes=plot_hashes,
        rt_annotation_context=rt_annotation_context,
    )
    manifest_path = selected_root / MANIFEST_FILE_NAME
    _write_manifest(
        manifest_path,
        paths=paths,
        feasibility_path=feasibility_path,
        triage_path=triage_path,
        panel_path=panel_path,
        handoff_sequence_csv_path=handoff_sequence_csv_path,
        plot_rows=plot_rows,
        feasibility_rows=feasibility_rows,
        triage_rows=triage_rows,
        panel_rows=panel_rows,
        handoff_sequence_rows=handoff_sequence_rows,
        created_at=created_at,
    )
    return MaterializedSelectionReadiness(
        feasibility_report_path=feasibility_path,
        candidate_triage_table_path=triage_path,
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
    panel_path: Path,
    handoff_sequence_csv_path: Path,
    plot_rows: list[dict[str, object]],
    feasibility_rows: list[dict[str, object]],
    triage_rows: list[dict[str, object]],
    panel_rows: list[dict[str, object]],
    handoff_sequence_rows: list[dict[str, object]],
    created_at: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_id": "eco1_rt.selection_readiness_manifest",
        "schema_version": 1,
        "status": "materialized",
        "created_by": CREATED_BY,
        "created_at": created_at,
        "selection_policy_id": SELECTION_POLICY_ID,
        "governing_rule": (
            "Select one feasible fold-preserved representative from each design class, then prefer natural "
            "sequence support, mutation geography near retained DNA/RNA or thumb-track, simple local chemistry, "
            "and sequence nonredundancy. Do not use ESMC or SAE as positive selection evidence."
        ),
        "sae_window_policy": (
            "SAE windows are retained as review evidence but not used for selection because the current pool "
            "does not meaningfully stratify in SAE-window space."
        ),
        "esmc_policy": "ESMC additive LLR rows are retained for review and are not used as panel-selection tie-breaks.",
        "source_tables": {key: str(value) for key, value in paths.items() if value.exists()},
        "artifacts": {
            "feasibility_report": str(feasibility_path),
            "candidate_triage_table": str(triage_path),
            "candidate_selection_panel": str(panel_path),
            "candidate_handoff_sequences": str(handoff_sequence_csv_path),
            "plots_root": str(path.parent / PLOTS_DIR_NAME),
        },
        "path_policy": "manifest_relative_for_plots",
        "plots": [_plot_manifest_row(row, manifest_root=path.parent) for row in plot_rows],
        "artifact_hashes": {
            key: sha256_uri(value)
            for key, value in {
                **{key: value for key, value in paths.items() if value.exists()},
                "feasibility_report": feasibility_path,
                "candidate_triage_table": triage_path,
                "candidate_selection_panel": panel_path,
                "candidate_handoff_sequences": handoff_sequence_csv_path,
                **{str(row["plot_id"]): Path(str(row["path"])) for row in plot_rows},
            }.items()
        },
        "row_counts": {
            "feasibility_report": len(feasibility_rows),
            "candidate_triage_table": len(triage_rows),
            "candidate_selection_panel": len(panel_rows),
            "candidate_handoff_sequences": len(handoff_sequence_rows),
        },
        "gate_counts": {
            "feasibility_status": _count_by(feasibility_rows, "feasibility_status"),
            "hard_gate_status": _count_by(triage_rows, "hard_gate_status"),
            "fold_review_class": _count_by(triage_rows, "fold_review_class"),
            "sae_window_status": _count_by(triage_rows, "sae_window_status"),
        },
        "selected_candidate_ids": [str(row["candidate_id"]) for row in panel_rows],
        "panel_coverage": panel_coverage_summary(
            panel_rows,
            expected_design_classes=[spec.design_class_id for spec in ALL_SPECS],
        ),
        "handoff_readiness": _handoff_readiness(path=path, panel_rows=panel_rows),
        "hard_gate_allowed_fold_classes": ["strong_fold_preserved", "good_fold_preserved"],
        "default_excluded_fold_classes": ["low_confidence", "review_band"],
        "panel_tie_break_order": [
            "fold review class",
            "selection-support MSA observed fraction",
            "selection-support MSA mean alternate-residue frequency",
            "selection-support unobserved mutation count",
            "near retained DNA/RNA or thumb-track mutation count",
            "near retained DNA/RNA or thumb-track chemistry warning count",
            "nearest selected sequence distance",
            "fold metrics",
            "mutation count",
            "sequence hash",
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _count_by(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "missing") for row in rows)
    return {value: counts[value] for value in sorted(counts)}


def _handoff_readiness(*, path: Path, panel_rows: list[dict[str, object]]) -> dict[str, object]:
    candidate_handoff_path = path.parent / "candidate_handoff.yaml"
    handoff_sequence_csv_path = path.parent / CANDIDATE_HANDOFF_SEQUENCE_CSV_FILE_NAME
    return {
        "handoff_kind": "rt_only_candidate_handoff",
        "panel_selected": bool(panel_rows),
        "candidate_handoff_path": candidate_handoff_path.name,
        "candidate_handoff_sequence_csv_path": handoff_sequence_csv_path.name,
        "candidate_handoff_sequence_csv_materialized": handoff_sequence_csv_path.exists(),
        "candidate_handoff_materialized": candidate_handoff_path.exists(),
        "construct_subject_created": False,
    }


_HANDOFF_SEQUENCE_CSV_FIELDS = [
    "candidate_id",
    "selection_slot",
    "design_class_id",
    "sequence_scope",
    "protein_sequence",
    "sequence_hash",
    "protein_sequence_length",
    "protein_sequence_sha256",
    "mapped_rt_chain_length",
    "canonical_rt_length",
    "canonical_sequence_status",
    "canonical_sequence_sha256",
    "fold_review_class",
    "feasibility_status",
    "eligible_for_handoff",
    "codon_policy_id",
    "dna_design_status",
    "dna_sequence_status",
    "codon_optimization_status",
    "restriction_screen_status",
    "handoff_scope_note",
    "source_candidate_pool_sha256",
    "source_panel_sha256",
]


def _write_candidate_handoff_sequence_csv(
    path: Path,
    *,
    panel_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    source_candidate_pool_sha256: str,
    source_panel_sha256: str,
) -> list[dict[str, object]]:
    """Write a flat selected-protein sequence table for review and handoff planning."""

    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    output_rows: list[dict[str, object]] = []
    for panel_row in panel_rows:
        candidate_id = str(panel_row["candidate_id"])
        candidate_row = candidate_by_id.get(candidate_id)
        if candidate_row is None:
            raise ValueError(f"Selected panel candidate is absent from candidate pool: {candidate_id}")
        sequence = str(candidate_row.get("sequence") or "").strip().upper()
        if not sequence:
            raise ValueError(f"Selected panel candidate has no protein sequence: {candidate_id}")
        candidate_hash = str(candidate_row.get("sequence_hash") or "")
        panel_hash = str(panel_row.get("sequence_hash") or "")
        if candidate_hash != panel_hash:
            raise ValueError(
                "Selected panel sequence hash does not match candidate pool for "
                f"{candidate_id}: panel={panel_hash!r} candidate_pool={candidate_hash!r}"
            )
        output_rows.append(
            {
                "candidate_id": candidate_id,
                "selection_slot": str(panel_row.get("selection_slot") or ""),
                "design_class_id": str(panel_row.get("design_class_id") or ""),
                "sequence_scope": "mapped_rt_chain_protein",
                "protein_sequence": sequence,
                "sequence_hash": candidate_hash,
                "protein_sequence_length": len(sequence),
                "protein_sequence_sha256": _sequence_sha256(sequence),
                "mapped_rt_chain_length": len(sequence),
                "canonical_rt_length": 320,
                "canonical_sequence_status": "not_exported_in_this_slice",
                "canonical_sequence_sha256": "",
                "fold_review_class": str(panel_row.get("fold_review_class") or ""),
                "feasibility_status": str(panel_row.get("feasibility_status") or ""),
                "eligible_for_handoff": str(bool(panel_row.get("eligible_for_handoff"))).lower(),
                "codon_policy_id": CODON_POLICY_ID,
                "dna_design_status": "not_materialized",
                "dna_sequence_status": "not_dna",
                "codon_optimization_status": "not_codon_optimized",
                "restriction_screen_status": "not_screened",
                "handoff_scope_note": (
                    "RT protein sequence only; not DNA, codon optimized, restriction screened, or construct ready."
                ),
                "source_candidate_pool_sha256": source_candidate_pool_sha256,
                "source_panel_sha256": source_panel_sha256,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_HANDOFF_SEQUENCE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    return output_rows


def _sequence_sha256(sequence: str) -> str:
    return "sha256:" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def _resolve(repo_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (repo_root / expanded).resolve()


def _plot_manifest_row(row: dict[str, object], *, manifest_root: Path) -> dict[str, object]:
    normalized = dict(row)
    normalized["path"] = str(Path(str(row["path"])).relative_to(manifest_root))
    return normalized
