"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/structure_browser.py

Interactive candidate-fold structure-browser manifest for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
    SECTION_PANEL_SELECTION,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from .molecular_scene_contract import (
    REFERENCE_MODEL_ID,
    molecular_visual_contract,
    reference_complex_molecule_styles,
)
from .structure_browser_common import (
    CANDIDATE_LOW_CONFIDENCE_COLOR,
    CANDIDATE_PASS_COLOR,
    PROTEIN_CLASS_COLOR,
    REFERENCE_COLOR,
    RESIDUE_CATEGORY_HIGHLIGHT_COLOR,
    display_label,
    nullable_float,
    nullable_int,
    relative_path,
    repo_relative_hint,
)
from .structure_sequences import sequence_by_candidate

STRUCTURE_BROWSER_MANIFEST_FILE_NAME = "interactive_structure_browser_manifest.yaml"
SELECTED_PANEL_STRUCTURE_BROWSER_MANIFEST_FILE_NAME = "selected_panel_structure_browser_manifest.yaml"
_MUTATION_PATTERN = re.compile(r"^[A-Z](?P<position>\d+)[A-Z]$")


def write_interactive_structure_browser_manifest(
    *,
    panel_root: Path,
    full_structure_set_path: Path,
    foldcheck_ranking_path: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    alignment_reference_path: Path,
    candidate_table_path: Path | None = None,
    foldcheck_fasta_path: Path | None = None,
    candidate_preference_table_path: Path | None = None,
    selection_panel_table_path: Path | None = None,
    triage_table_path: Path | None = None,
    manifest_file_name: str = STRUCTURE_BROWSER_MANIFEST_FILE_NAME,
    deliverable_id: str = "interactive_structure_browser_manifest",
    section: str = SECTION_DESIGNS_AND_FOLD_TRIAGE,
    title: str = "Reference-fitted ColabFold structures can be inspected one at a time",
    alt_text: str = "Manifest for interactive browser review of reference-fitted local ColabFold structure models.",
    description: str = (
        "Lists the local reference and fold-check PDB paths used by the browser-native "
        "structure view. Query coordinates are aligned in memory "
        "over mapped C-alpha atoms before rendering; the local raw PDB files "
        "remain unchanged."
    ),
    source_table_prefix: str = "foldcheck_review",
) -> dict[str, Any]:
    """Write a compact manifest that lets marimo browse local fold structures."""

    panel_root.mkdir(parents=True, exist_ok=True)
    manifest_path = panel_root / manifest_file_name
    if not full_structure_set_path.exists():
        return _missing_row(
            manifest_path,
            full_structure_set_path,
            deliverable_id=deliverable_id,
            section=section,
            source_table_prefix=source_table_prefix,
        )

    source = yaml.safe_load(full_structure_set_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict) or source.get("schema_id") != "eco1_rt.foldcheck_full_structure_set":
        raise ValueError(f"Expected eco1_rt.foldcheck_full_structure_set at {full_structure_set_path}")

    ranking = _ranking_by_candidate(foldcheck_ranking_path)
    sequences = sequence_by_candidate(candidate_table_path, foldcheck_fasta_path=foldcheck_fasta_path)
    selection = _selection_by_candidate(selection_panel_table_path)
    triage = _triage_by_candidate(triage_table_path)
    if not reference_structure_path.exists():
        raise ValueError(f"interactive structure browser reference path is missing: {reference_structure_path}")
    if not alignment_reference_path.exists():
        raise ValueError(
            f"interactive structure browser alignment reference path is missing: {alignment_reference_path}"
        )
    query_start_residue = 3
    reference_start_residue = 1
    structures = _structure_rows(
        source_rows=list(source.get("structures") or []),
        source_root=full_structure_set_path.parent,
        manifest_root=manifest_path.parent,
        ranking=ranking,
        sequences=sequences,
        selection=selection,
        triage=triage,
        mutations_by_candidate=_mutation_rows_by_candidate(candidate_table_path),
        esmc_scores_by_candidate=_esmc_scores_by_candidate(candidate_preference_table_path),
        selected_candidate_ids=set(selection),
    )
    source_tables = [
        repo_relative_hint(full_structure_set_path),
        repo_relative_hint(foldcheck_ranking_path),
    ]
    input_hash_paths = {
        "full_structure_set": full_structure_set_path,
        "foldcheck_ranking": foldcheck_ranking_path,
        "reference_structure": reference_structure_path,
        "alignment_reference": alignment_reference_path,
    }
    if candidate_table_path is not None:
        source_tables.append(repo_relative_hint(candidate_table_path))
        input_hash_paths["candidate_table"] = candidate_table_path
    if foldcheck_fasta_path is not None and foldcheck_fasta_path.exists():
        source_tables.append(repo_relative_hint(foldcheck_fasta_path))
        input_hash_paths["foldcheck_fasta"] = foldcheck_fasta_path
    if candidate_preference_table_path is not None and candidate_preference_table_path.exists():
        source_tables.append(repo_relative_hint(candidate_preference_table_path))
        input_hash_paths["candidate_preference_table"] = candidate_preference_table_path
    if selection_panel_table_path is not None and selection_panel_table_path.exists():
        source_tables.append(repo_relative_hint(selection_panel_table_path))
        input_hash_paths["selection_panel_table"] = selection_panel_table_path
    if triage_table_path is not None and triage_table_path.exists():
        source_tables.append(repo_relative_hint(triage_table_path))
        input_hash_paths["candidate_triage_table"] = triage_table_path
    payload = {
        "schema_id": "eco1_rt.interactive_structure_browser_manifest",
        "schema_version": 1,
        "status": "materialized",
        "title": title,
        "alt_text": alt_text,
        "description": description,
        "viewer_contract": "dnadesign.thread.structure_views",
        "backend_kind": "browser_structure_view",
        "default_backend": "py3dmol",
        "visual_contract": molecular_visual_contract(),
        "protein_surface_default": False,
        "path_policy": "paths_relative_to_this_manifest",
        "source_tables": source_tables,
        "reference": {
            "model_id": REFERENCE_MODEL_ID,
            "display_label": _reference_display_label(reference_structure_path, reference_structure_format),
            "local_path": relative_path(reference_structure_path, manifest_path.parent),
            "structure_format": reference_structure_format,
            "color": REFERENCE_COLOR,
        },
        "alignment": {
            "status": "enabled",
            "method": "mapped_ca_kabsch",
            "query_start_residue": query_start_residue,
            "reference_start_residue": reference_start_residue,
            "residue_count": 309,
            "reference_local_path": relative_path(alignment_reference_path, manifest_path.parent),
            "reference_structure_format": "pdb",
            "output_policy": "query coordinates are aligned in memory for browser viewing; local PDB files stay raw",
        },
        "structures": structures,
        "structure_count": len(structures),
        "interpretation_limit": (
            "The browser view is for interactive review. ChimeraX remains the publication-still and pose-capture path."
        ),
    }
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return make_deliverable_row(
        deliverable_id=deliverable_id,
        section=section,
        artifact_kind="structure_browser_manifest",
        status="rendered",
        path=manifest_path,
        source_tables=source_tables,
        input_hashes=file_hashes(input_hash_paths),
        alt_text=alt_text,
        description=description,
        interpretation_limit=payload["interpretation_limit"],
        title=title,
        role="interactive_review",
    )


def write_selected_panel_structure_browser_manifest(
    *,
    panel_root: Path,
    full_structure_set_path: Path,
    foldcheck_ranking_path: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    alignment_reference_path: Path,
    candidate_table_path: Path,
    selection_panel_table_path: Path,
    triage_table_path: Path,
    foldcheck_fasta_path: Path | None = None,
    candidate_preference_table_path: Path | None = None,
    source_table_prefix: str = "foldcheck_review",
) -> dict[str, Any]:
    """Write the selected-panel structure-browser manifest from the active selection root."""

    return write_interactive_structure_browser_manifest(
        panel_root=panel_root,
        full_structure_set_path=full_structure_set_path,
        foldcheck_ranking_path=foldcheck_ranking_path,
        reference_structure_path=reference_structure_path,
        reference_structure_format=reference_structure_format,
        alignment_reference_path=alignment_reference_path,
        candidate_table_path=candidate_table_path,
        foldcheck_fasta_path=foldcheck_fasta_path,
        candidate_preference_table_path=candidate_preference_table_path,
        selection_panel_table_path=selection_panel_table_path,
        triage_table_path=triage_table_path,
        manifest_file_name=SELECTED_PANEL_STRUCTURE_BROWSER_MANIFEST_FILE_NAME,
        deliverable_id="selected_panel_structure_browser_manifest",
        section=SECTION_PANEL_SELECTION,
        title="Selected Eco1 protein hypotheses can be inspected one at a time",
        alt_text=(
            "Manifest for interactive browser review of WT and the eight selected Eco1 ColabFold structure models."
        ),
        description=(
            "Lists WT and the eight selected sequences from the active fold-check structure set. The side summary "
            "shows fold metrics, mutation count, MSA support, chemistry "
            "near retained DNA/RNA or the thumb track, and selection context beside the py3Dmol viewer."
        ),
        source_table_prefix=source_table_prefix,
    )


def _reference_display_label(reference_structure_path: Path, reference_structure_format: str) -> str:
    if reference_structure_format == "mmcif" or "all_atom" in reference_structure_path.stem:
        return "Ec86/7V9U all-atom reference"
    return "ec86kit/7V9U reference"


def _missing_row(
    manifest_path: Path,
    missing_path: Path,
    *,
    deliverable_id: str,
    section: str,
    source_table_prefix: str,
) -> dict[str, Any]:
    return make_deliverable_row(
        deliverable_id=deliverable_id,
        section=section,
        artifact_kind="structure_browser_manifest",
        status="skipped_missing_input",
        path=manifest_path,
        source_tables=[f"{source_table_prefix}/foldcheck_full_structure_set.yaml"],
        input_hashes={},
        alt_text="Structure browsing is unavailable because the local fold structure set is missing.",
        description="Interactive structure browsing is skipped until the local fold structure set exists.",
        interpretation_limit="Missing structure paths cannot support interactive structure review.",
        title="Interactive structure browser is skipped until local PDBs are available",
        role="interactive_review",
        skip_reason=f"Missing input manifest: {missing_path}",
    )


def _ranking_by_candidate(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    table = pq.read_table(
        path,
        columns=[
            "candidate_id",
            "review_rank",
            "review_class",
            "plddt",
            "wt_runtime_ca_rmsd",
            "cryoem_mapped_ca_rmsd",
            "seq_recovery",
            "mutation_count",
        ],
    )
    return {str(row["candidate_id"]): row for row in table.to_pylist()}


def _mutation_rows_by_candidate(
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = pq.read_table(path, columns=["candidate_id", "canonical_mutations"]).to_pylist()
    mutations_by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = str(row["candidate_id"])
        canonical_mutations = [str(mutation) for mutation in row.get("canonical_mutations") or []]
        canonical_positions: list[int] = []
        for mutation in canonical_mutations:
            match = _MUTATION_PATTERN.match(mutation)
            if not match:
                raise ValueError(f"Malformed canonical mutation for {candidate_id}: {mutation!r}")
            canonical_positions.append(int(match.group("position")))
        mutations_by_candidate[candidate_id] = {
            "canonical_mutations": canonical_mutations,
            "canonical_positions": sorted(set(canonical_positions)),
            # ColabFold models contain the canonical 320-aa RT, so model residue
            # numbers and canonical Eco1 positions are identical.
            "query_residue_numbers": sorted(set(canonical_positions)),
        }
    return mutations_by_candidate


def _structure_rows(
    *,
    source_rows: list[Any],
    source_root: Path,
    manifest_root: Path,
    ranking: dict[str, dict[str, Any]],
    sequences: dict[str, dict[str, Any]],
    selection: dict[str, dict[str, Any]],
    triage: dict[str, dict[str, Any]],
    mutations_by_candidate: dict[str, dict[str, Any]],
    esmc_scores_by_candidate: dict[str, dict[str, Any]],
    selected_candidate_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(source_rows):
        if not isinstance(row, dict):
            raise ValueError(f"full structure-set row {row_index} is not a mapping")
        candidate_id = str(row.get("candidate_id") or "")
        local_value = str(row.get("local_model_artifact_path") or "")
        local_path = source_root / local_value
        if not candidate_id:
            raise ValueError(f"full structure-set row {row_index} is missing candidate_id")
        if not local_value:
            raise ValueError(f"full structure-set row {row_index} is missing local_model_artifact_path")
        if not local_path.exists():
            raise ValueError(f"declared structure path is missing for {candidate_id}: {local_path}")
        if selected_candidate_ids and candidate_id != "wild_type" and candidate_id not in selected_candidate_ids:
            continue
        rank_row = ranking.get(candidate_id, {})
        sequence_row = sequences.get(candidate_id, {})
        selection_row = selection.get(candidate_id, {})
        triage_row = triage.get(candidate_id, {})
        mutation_payload = mutations_by_candidate.get(candidate_id, {})
        esmc_payload = esmc_scores_by_candidate.get(candidate_id, {})
        rows.append(
            {
                "candidate_id": candidate_id,
                "display_label": display_label(candidate_id, row),
                "group": _structure_group(candidate_id, rank_row, selection_row),
                "local_path": relative_path(local_path, manifest_root),
                "structure_format": "pdb",
                "color": _structure_color(candidate_id, rank_row),
                "molecule_styles": reference_complex_molecule_styles(include_protein_surface=False),
                "review_rank": nullable_int(rank_row.get("review_rank")),
                "review_class": str(rank_row.get("review_class") or ""),
                "plddt": nullable_float(rank_row.get("plddt")),
                "wt_runtime_ca_rmsd": nullable_float(
                    row.get("wt_runtime_ca_rmsd") or rank_row.get("wt_runtime_ca_rmsd")
                ),
                "cryoem_mapped_ca_rmsd": nullable_float(rank_row.get("cryoem_mapped_ca_rmsd")),
                "full_sequence_identity_percent": nullable_float(row.get("full_sequence_identity_percent")),
                "design_position_recovery_percent": nullable_float(row.get("design_position_recovery_percent")),
                "protein_sequence": str(sequence_row.get("protein_sequence") or ""),
                "protein_sequence_hash": str(sequence_row.get("sequence_hash") or ""),
                "protein_sequence_length": nullable_int(sequence_row.get("amino_acid_length")),
                "protein_sequence_source": str(sequence_row.get("sequence_source") or ""),
                "mutation_count": nullable_int(rank_row.get("mutation_count"))
                or len(mutation_payload.get("canonical_positions") or []),
                "canonical_mutations": mutation_payload.get("canonical_mutations", []),
                "mutation_residue_numbers": mutation_payload.get("query_residue_numbers", []),
                **_selection_payload(selection_row, triage_row),
                "esmc_llr_total": nullable_float(esmc_payload.get("llr_total")),
                "esmc_llr_per_mutation": nullable_float(esmc_payload.get("llr_per_mutation")),
                "esmc_mutations_scored_count": nullable_int(esmc_payload.get("mutations_scored_count")),
                "esmc_scoring_method_id": str(esmc_payload.get("scoring_method_id") or ""),
            }
        )
    return sorted(rows, key=_structure_sort_key)


def _esmc_scores_by_candidate(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    table = pq.read_table(
        path,
        columns=[
            "candidate_id",
            "scoring_method_id",
            "mutation_count",
            "mutations_scored_count",
            "llr_total",
            "llr_per_mutation",
            "status",
        ],
    )
    return {str(row["candidate_id"]): row for row in table.to_pylist()}


def _selection_by_candidate(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    table = pq.read_table(path)
    return {str(row["candidate_id"]): row for row in table.to_pylist()}


def _triage_by_candidate(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    table = pq.read_table(path)
    return {str(row["candidate_id"]): row for row in table.to_pylist()}


def _selection_payload(selection_row: dict[str, Any], triage_row: dict[str, Any]) -> dict[str, Any]:
    if not selection_row and not triage_row:
        return {}
    return {
        "selection_slot": str(selection_row.get("selection_slot") or ""),
        "selection_rank": nullable_int(selection_row.get("selection_rank")),
        "selection_reason": str(selection_row.get("selection_reason") or ""),
        "nearest_selected_distance_aa": nullable_int(selection_row.get("nearest_selected_distance_aa")),
        "selection_support_alt_observed_fraction": nullable_float(
            triage_row.get("selection_support_alt_observed_fraction")
        ),
        "selection_support_unobserved_mutation_count": nullable_int(
            triage_row.get("selection_support_unobserved_mutation_count")
        ),
        "nucleic_acid_facing_mutation_count": nullable_int(triage_row.get("nucleic_acid_facing_mutation_count")),
        "nucleic_acid_facing_charge_delta": nullable_int(triage_row.get("nucleic_acid_facing_charge_delta")),
        "nucleic_acid_facing_chemistry_warning_count": nullable_int(
            triage_row.get("nucleic_acid_facing_chemistry_warning_count")
        ),
        "catalytic_or_direct_contact_mutation_count": nullable_int(
            triage_row.get("catalytic_or_direct_contact_mutation_count")
        ),
        "thumb_contact_track_mutation_count": nullable_int(triage_row.get("thumb_contact_track_mutation_count")),
        "c_terminal_primer_rna_recognition_mutation_count": nullable_int(
            triage_row.get("c_terminal_primer_rna_recognition_mutation_count")
        ),
        "distal_scaffold_mutation_count": nullable_int(triage_row.get("distal_scaffold_mutation_count")),
    }


def _structure_group(candidate_id: str, rank_row: dict[str, Any], selection_row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return "0 WT ColabFold baseline"
    selection_slot = str(selection_row.get("selection_slot") or "")
    if selection_slot:
        selection_rank = nullable_int(selection_row.get("selection_rank")) or 0
        candidate_label = candidate_id.removeprefix("thread_candidate_")
        return f"{selection_rank} Selected hypothesis: {candidate_label}"
    review_class = str(rank_row.get("review_class") or "")
    if review_class in {"strong_fold_preserved", "good_fold_preserved"}:
        return "1 Passing fold triage (CA RMSD <= 2.0 A; pLDDT >= 90)"
    if review_class in {"structural_outlier", "low_confidence"}:
        return "3 Hold for review (CA RMSD > 5.0 A or pLDDT < 90)"
    return "2 Intermediate fold review band"


def _structure_color(candidate_id: str, rank_row: dict[str, Any]) -> str:
    if candidate_id == "wild_type":
        return PROTEIN_CLASS_COLOR
    review_class = str(rank_row.get("review_class") or "")
    if review_class == "structural_outlier":
        return RESIDUE_CATEGORY_HIGHLIGHT_COLOR
    if review_class == "low_confidence":
        return CANDIDATE_LOW_CONFIDENCE_COLOR
    return CANDIDATE_PASS_COLOR


def _structure_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    if row["candidate_id"] == "wild_type":
        return (0, 0, str(row["candidate_id"]))
    rank = row.get("review_rank")
    return (1, int(rank) if rank is not None else 10_000, str(row["candidate_id"]))
