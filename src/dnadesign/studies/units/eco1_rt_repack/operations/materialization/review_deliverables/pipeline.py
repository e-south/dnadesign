"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/pipeline.py

Materialize Eco1 review-deliverable artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
)

from .biohub_esmc_sae_interpretation import write_biohub_esmc_sae_interpretation_panels
from .constants import (
    ALIGNED_FASTA_RELATIVE_PATH,
    BIOHUB_ESMC_FEATURE_CATALOG_FILE_NAME,
    BIOHUB_ESMC_MUTATION_SCORING_RELATIVE_PATH,
    BIOHUB_ESMC_PROTEIN_FEATURES_FILE_NAME,
    BIOHUB_ESMC_REQUEST_MANIFEST_FILE_NAME,
    BIOHUB_ESMC_RESIDUE_FEATURES_FILE_NAME,
    BIOHUB_ESMC_SAE_INTERPRETATION_DIR_NAME,
    BIOHUB_ESMC_SAE_PROFILE_FILE_NAME,
    BIOHUB_ESMC_WT_SUBSTITUTION_LLR_RELATIVE_PATH,
    CANDIDATE_TABLE_FILE_NAME,
    CONSERVATION_PROFILE_FILE_NAME,
    DEFAULT_OUTPUT_ROOT,
    DELIVERABLE_DIR_NAME,
    FOLDCHECK_FULL_STRUCTURE_SET_RELATIVE_PATH,
    FOLDCHECK_REPORT_FILE_NAME,
    FOLDCHECK_REVIEW_MANIFEST_RELATIVE_PATH,
    FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
    MANIFEST_FILE_NAME,
    MASK_CONTEXT_DIR_NAME,
    MASK_SET_FILE_NAME,
    MSA_PANEL_DIR_NAME,
    NOTEBOOK_FILE_NAME,
    NOTEBOOKS_DIR_NAME,
    PROTEINMPNN_DIR_NAME,
    REFERENCE_BACKBONE_RELATIVE_PATH,
    STRUCTURE_BROWSER_DIR_NAME,
    WT_MODEL_CONSTRAINT_DIR_NAME,
)
from .esmc_model_constraint import write_esmc_model_constraint_audit_panels
from .manifest import file_hashes, make_deliverable_row, write_manifest
from .mask_rows import read_mask_residues
from .mask_tracks import write_linear_mask_tracks, write_mask_structure_context
from .models import MaterializedReviewDeliverables
from .msa_panel import write_msa_plurality_mask_panel
from .notebook import write_review_deliverables_notebook
from .proteinmpnn_diversity import write_proteinmpnn_diversity_panels
from .structure_browser import write_interactive_structure_browser_manifest, write_mask_structure_browser_manifest


def materialize_review_deliverables(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    render_chimerax_png: bool = True,
) -> MaterializedReviewDeliverables:
    """Materialize the first Eco1 manuscript/review deliverable bundle."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    deliverable_root = out_root / DELIVERABLE_DIR_NAME
    deliverable_root.mkdir(parents=True, exist_ok=True)

    aligned_fasta_path = out_root / ALIGNED_FASTA_RELATIVE_PATH
    conservation_profile_path = out_root / CONSERVATION_PROFILE_FILE_NAME
    mask_set_path = out_root / MASK_SET_FILE_NAME
    candidate_table_path = out_root / CANDIDATE_TABLE_FILE_NAME
    reference_backbone_path = out_root / REFERENCE_BACKBONE_RELATIVE_PATH
    for required_path in (
        aligned_fasta_path,
        conservation_profile_path,
        mask_set_path,
        candidate_table_path,
        reference_backbone_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    mask_residues = read_mask_residues(mask_set_path)
    deliverables: list[dict[str, Any]] = [
        write_msa_plurality_mask_panel(
            panel_root=deliverable_root / MSA_PANEL_DIR_NAME,
            aligned_fasta_path=aligned_fasta_path,
            conservation_profile_path=conservation_profile_path,
            mask_set_path=mask_set_path,
            mask_residues=mask_residues,
        ),
        write_linear_mask_tracks(
            panel_root=deliverable_root / MASK_CONTEXT_DIR_NAME,
            mask_set_path=mask_set_path,
            mask_residues=mask_residues,
        ),
    ]
    deliverables.extend(
        write_mask_structure_context(
            panel_root=deliverable_root / MASK_CONTEXT_DIR_NAME,
            mask_set_path=mask_set_path,
            reference_backbone_path=reference_backbone_path,
            mask_residues=mask_residues,
            render_png=render_chimerax_png,
        )
    )
    deliverables.append(
        write_mask_structure_browser_manifest(
            panel_root=deliverable_root / STRUCTURE_BROWSER_DIR_NAME,
            mask_set_path=mask_set_path,
            reference_backbone_path=reference_backbone_path,
            mask_residues=mask_residues,
        )
    )
    deliverables.extend(
        write_proteinmpnn_diversity_panels(
            panel_root=deliverable_root / PROTEINMPNN_DIR_NAME,
            candidate_table_path=candidate_table_path,
            foldcheck_ranking_path=out_root / FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
            mask_residues=mask_residues,
        )
    )
    deliverables.extend(_linked_foldcheck_review_rows(out_root / FOLDCHECK_REVIEW_MANIFEST_RELATIVE_PATH))
    deliverables.append(
        write_interactive_structure_browser_manifest(
            panel_root=deliverable_root / STRUCTURE_BROWSER_DIR_NAME,
            full_structure_set_path=out_root / FOLDCHECK_FULL_STRUCTURE_SET_RELATIVE_PATH,
            foldcheck_ranking_path=out_root / FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
        )
    )
    deliverables.extend(
        write_esmc_model_constraint_audit_panels(
            panel_root=deliverable_root / WT_MODEL_CONSTRAINT_DIR_NAME,
            mutation_scoring_root=out_root / BIOHUB_ESMC_MUTATION_SCORING_RELATIVE_PATH,
        )
    )
    deliverables.extend(
        write_biohub_esmc_sae_interpretation_panels(
            panel_root=deliverable_root / BIOHUB_ESMC_SAE_INTERPRETATION_DIR_NAME,
            profile_path=out_root / BIOHUB_ESMC_SAE_PROFILE_FILE_NAME,
            protein_features_path=out_root / BIOHUB_ESMC_PROTEIN_FEATURES_FILE_NAME,
            residue_features_path=out_root / BIOHUB_ESMC_RESIDUE_FEATURES_FILE_NAME,
            feature_catalog_path=out_root / BIOHUB_ESMC_FEATURE_CATALOG_FILE_NAME,
            request_manifest_path=out_root / BIOHUB_ESMC_REQUEST_MANIFEST_FILE_NAME,
            candidate_table_path=candidate_table_path,
            foldcheck_report_path=out_root / FOLDCHECK_REPORT_FILE_NAME,
            foldcheck_ranking_path=out_root / FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
            wt_substitution_llr_path=out_root / BIOHUB_ESMC_WT_SUBSTITUTION_LLR_RELATIVE_PATH,
        )
    )

    notebook_path = deliverable_root / NOTEBOOKS_DIR_NAME / NOTEBOOK_FILE_NAME
    write_review_deliverables_notebook(notebook_path)
    manifest_path = deliverable_root / MANIFEST_FILE_NAME
    write_manifest(manifest_path, deliverables=deliverables, notebook_path=notebook_path)
    return MaterializedReviewDeliverables(
        manifest_path=manifest_path,
        notebook_path=notebook_path,
        deliverable_count=len(deliverables),
    )


def _linked_foldcheck_review_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return [
            make_deliverable_row(
                deliverable_id="foldcheck_review_visuals",
                section="fold_review",
                artifact_kind="manifest",
                status="skipped_missing_input",
                path=manifest_path,
                source_tables=["foldcheck_review/review_visual_manifest.yaml"],
                input_hashes={},
                alt_text="Fold-review visuals were not linked because their manifest was missing.",
                description="Fold-review links are skipped until foldcheck_review is materialized.",
                interpretation_limit="Missing fold-review visuals cannot support candidate review.",
                role="review_only",
                skip_reason=f"Missing input manifest: {manifest_path}",
            )
        ]
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {manifest_path}")
    rows: list[dict[str, Any]] = []
    for plot in loaded.get("plots", []):
        if not isinstance(plot, dict):
            continue
        plot_id = str(plot.get("plot_id") or "plot")
        plot_path = _resolve_linked_manifest_path(manifest_path, str(plot.get("path") or ""))
        plot_status = str(plot.get("status") or "rendered")
        linked_status = "linked_existing" if plot_status == "rendered" and plot_path.exists() else plot_status
        if plot_status == "rendered" and not plot_path.exists():
            linked_status = "skipped_missing_input"
        rows.append(
            make_deliverable_row(
                deliverable_id=f"foldcheck_review_{plot_id}",
                section="fold_review",
                artifact_kind="linked_visual",
                status=linked_status,
                path=plot_path,
                source_tables=[str(source) for source in plot.get("data_sources", [])],
                input_hashes=file_hashes({"foldcheck_review_manifest": manifest_path, "linked_plot": plot_path}),
                alt_text=str(plot.get("alt_text") or ""),
                description=str(plot.get("description") or ""),
                interpretation_limit=str(plot.get("interpretation_limit") or ""),
                title=str(plot.get("title") or ""),
                role="review_only",
                skip_reason=str(plot.get("skip_reason") or "")
                if linked_status != "skipped_missing_input"
                else f"Missing linked fold-review visual: {plot_path}",
            )
        )
    return rows


def _resolve_linked_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
