"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/pipeline.py

Materialize Eco1 review-deliverable artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.contact_geometry.paths import (
    find_repo_root,
    resolve_output_root,
)

from .biohub_esmc_sae_interpretation import write_biohub_esmc_sae_interpretation_panels
from .biohub_esmc_sequence_preference import (
    TITLE_6B,
    VARIANT_LLR_FILE_NAME,
    write_biohub_esmc_sequence_preference_deliverables,
)
from .biohub_esmc_sequence_preference_model_agreement import write_biohub_esmc_model_agreement_deliverables
from .constants import (
    ALIGNED_FASTA_RELATIVE_PATH,
    BIOHUB_ESMC_6B_MUTATION_SCORING_RELATIVE_PATH,
    BIOHUB_ESMC_FEATURE_CATALOG_FILE_NAME,
    BIOHUB_ESMC_FEATURE_HEATMAP_DIR_NAME,
    BIOHUB_ESMC_MUTATION_SCORING_RELATIVE_PATH,
    BIOHUB_ESMC_PROTEIN_FEATURES_FILE_NAME,
    BIOHUB_ESMC_REQUEST_MANIFEST_FILE_NAME,
    BIOHUB_ESMC_RESIDUE_FEATURES_FILE_NAME,
    BIOHUB_ESMC_SAE_INTERPRETATION_DIR_NAME,
    BIOHUB_ESMC_SAE_PROFILE_FILE_NAME,
    BIOHUB_ESMC_SEQUENCE_SCORING_DIR_NAME,
    BIOHUB_ESMC_WT_SUBSTITUTION_LLR_RELATIVE_PATH,
    CANDIDATE_TABLE_FILE_NAME,
    CONSERVATION_PROFILE_FILE_NAME,
    CONSERVATION_SOURCE_MANIFEST_RELATIVE_PATH,
    DEFAULT_OUTPUT_ROOT,
    DELIVERABLE_DIR_NAME,
    FOLDCHECK_FULL_STRUCTURE_SET_RELATIVE_PATH,
    FOLDCHECK_REQUEST_INPUT_FASTA_RELATIVE_PATH,
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
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
    SECTION_ESMC_FEATURE_REVIEW,
    STRUCTURE_BROWSER_DIR_NAME,
    SUBTYPE_ALIGNED_FASTA_RELATIVE_PATH,
    SUBTYPE_CONSERVATION_SOURCE_MANIFEST_RELATIVE_PATH,
    WT_MODEL_CHECK_DIR_NAME,
)
from .design_class_masks import write_design_class_mask_overview
from .esmc_model_check import write_esmc_model_check_panels
from .manifest import file_hashes, make_deliverable_row, write_manifest
from .mask_rows import read_mask_residues
from .mask_structure_browser import write_mask_structure_browser_manifest
from .mask_tracks import write_mask_structure_context
from .models import MaterializedReviewDeliverables
from .msa_panel import CLADE9_MSA_PANEL, SUBTYPE_MSA_PANEL, write_msa_plurality_mask_panel
from .msa_panel_data import source_manifest_accessions
from .notebook import write_review_deliverables_notebook
from .proteinmpnn_diversity import write_proteinmpnn_diversity_panels
from .proteinmpnn_fold_validation import write_expanded_design_class_fold_validation
from .rt_annotation_context import (
    MANUAL_MASK_AUTHORITY_SOURCE_LABEL,
    RT_ANNOTATION_TRACKS_SOURCE_LABEL,
    load_rt_annotation_context,
)
from .sae_structure_browser import write_sae_structure_browser_manifest
from .selection_readiness import linked_selection_readiness_rows
from .structure_browser import (
    write_interactive_structure_browser_manifest,
    write_selected_panel_structure_browser_manifest,
)
from .structure_browser_common import REFERENCE_STRUCTURE_RELATIVE_PATH, stage_browser_reference_structure


def materialize_review_deliverables(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
    render_chimerax_png: bool = False,
) -> MaterializedReviewDeliverables:
    """Materialize the first Eco1 manuscript/review deliverable bundle."""

    root = (repo_root or find_repo_root(Path.cwd())).expanduser().resolve()
    out_root = resolve_output_root(root, output_root or DEFAULT_OUTPUT_ROOT)
    deliverable_root = out_root / DELIVERABLE_DIR_NAME
    deliverable_root.mkdir(parents=True, exist_ok=True)
    _remove_retired_deliverables(deliverable_root)

    aligned_fasta_path = out_root / ALIGNED_FASTA_RELATIVE_PATH
    subtype_aligned_fasta_path = out_root / SUBTYPE_ALIGNED_FASTA_RELATIVE_PATH
    conservation_source_manifest_path = out_root / CONSERVATION_SOURCE_MANIFEST_RELATIVE_PATH
    subtype_conservation_source_manifest_path = out_root / SUBTYPE_CONSERVATION_SOURCE_MANIFEST_RELATIVE_PATH
    conservation_profile_path = out_root / CONSERVATION_PROFILE_FILE_NAME
    mask_set_path = out_root / MASK_SET_FILE_NAME
    candidate_table_path = out_root / CANDIDATE_TABLE_FILE_NAME
    reference_backbone_path = out_root / REFERENCE_BACKBONE_RELATIVE_PATH
    for required_path in (
        aligned_fasta_path,
        subtype_aligned_fasta_path,
        conservation_source_manifest_path,
        subtype_conservation_source_manifest_path,
        conservation_profile_path,
        mask_set_path,
        candidate_table_path,
        reference_backbone_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)
    _validate_subtype_source_subset(
        clade_source_manifest_path=conservation_source_manifest_path,
        subtype_source_manifest_path=subtype_conservation_source_manifest_path,
    )

    mask_residues = read_mask_residues(mask_set_path)
    rt_annotation_context = load_rt_annotation_context(
        annotation_tracks_path=root / RT_ANNOTATION_TRACKS_SOURCE_LABEL,
        manual_mask_authority_source_path=root / MANUAL_MASK_AUTHORITY_SOURCE_LABEL,
    )
    foldcheck_review_root = out_root / Path(FOLDCHECK_FULL_STRUCTURE_SET_RELATIVE_PATH).parent
    foldcheck_reference_backbone_path = foldcheck_review_root / REFERENCE_STRUCTURE_RELATIVE_PATH
    browser_reference = stage_browser_reference_structure(
        repo_root=root,
        reference_backbone_path=foldcheck_reference_backbone_path,
    )
    deliverables: list[dict[str, Any]] = [
        write_msa_plurality_mask_panel(
            panel_root=deliverable_root / MSA_PANEL_DIR_NAME,
            panel_profile=CLADE9_MSA_PANEL,
            aligned_fasta_path=aligned_fasta_path,
            source_manifest_path=conservation_source_manifest_path,
            conservation_profile_path=conservation_profile_path,
            mask_set_path=mask_set_path,
            mask_residues=mask_residues,
            subtype_source_manifest_path=subtype_conservation_source_manifest_path,
            rt_annotation_context=rt_annotation_context,
        ),
        write_msa_plurality_mask_panel(
            panel_root=deliverable_root / MSA_PANEL_DIR_NAME,
            panel_profile=SUBTYPE_MSA_PANEL,
            aligned_fasta_path=subtype_aligned_fasta_path,
            source_manifest_path=subtype_conservation_source_manifest_path,
            conservation_profile_path=conservation_profile_path,
            mask_set_path=mask_set_path,
            mask_residues=mask_residues,
            rt_annotation_context=rt_annotation_context,
        ),
        write_design_class_mask_overview(
            panel_root=deliverable_root / MASK_CONTEXT_DIR_NAME,
            baseline_mask_set_path=mask_set_path,
            design_classes_root=out_root / "design_classes",
            rt_annotation_context=rt_annotation_context,
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
            design_classes_root=out_root / "design_classes",
            reference_structure_path=browser_reference.local_path,
            reference_structure_format=browser_reference.structure_format,
            mask_residues=mask_residues,
            rt_annotation_context=rt_annotation_context,
        )
    )
    deliverables.extend(
        write_proteinmpnn_diversity_panels(
            panel_root=deliverable_root / PROTEINMPNN_DIR_NAME,
            candidate_table_path=candidate_table_path,
            candidate_pool_path=out_root / "design_classes" / "candidate_pool.parquet",
            design_classes_root=out_root / "design_classes",
            foldcheck_ranking_path=out_root / FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
            mask_set_path=mask_set_path,
        )
    )
    deliverables.append(
        write_expanded_design_class_fold_validation(
            panel_root=deliverable_root / PROTEINMPNN_DIR_NAME,
            candidate_pool_path=out_root / "design_classes" / "candidate_pool.parquet",
            foldcheck_ranking_path=out_root
            / "design_classes"
            / "foldcheck_review"
            / "foldcheck_candidate_ranking.parquet",
            selection_panel_table_path=out_root / "design_classes" / "selection" / "candidate_selection_panel.parquet",
        )
    )
    deliverables.extend(_linked_foldcheck_review_rows(out_root / FOLDCHECK_REVIEW_MANIFEST_RELATIVE_PATH))
    sequence_preference_deliverables = write_biohub_esmc_sequence_preference_deliverables(
        panel_root=deliverable_root / BIOHUB_ESMC_SEQUENCE_SCORING_DIR_NAME,
        candidate_table_path=candidate_table_path,
        wt_substitution_llr_path=out_root / BIOHUB_ESMC_WT_SUBSTITUTION_LLR_RELATIVE_PATH,
        wt_mutation_scoring_manifest_path=out_root
        / BIOHUB_ESMC_MUTATION_SCORING_RELATIVE_PATH
        / "wt_mutation_scoring_manifest.yaml",
        foldcheck_ranking_path=out_root / FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
    )
    deliverables.extend(sequence_preference_deliverables)
    six_b_sequence_scoring_root = deliverable_root / BIOHUB_ESMC_SEQUENCE_SCORING_DIR_NAME / "esmc_6b_2024_12"
    six_b_wt_substitution_llr_path = (
        out_root / BIOHUB_ESMC_6B_MUTATION_SCORING_RELATIVE_PATH / "wt_substitution_llr.parquet"
    )
    six_b_wt_manifest_path = (
        out_root / BIOHUB_ESMC_6B_MUTATION_SCORING_RELATIVE_PATH / "wt_mutation_scoring_manifest.yaml"
    )
    if six_b_wt_substitution_llr_path.exists() or six_b_wt_manifest_path.exists():
        six_b_sequence_preference_deliverables = write_biohub_esmc_sequence_preference_deliverables(
            panel_root=six_b_sequence_scoring_root,
            candidate_table_path=candidate_table_path,
            wt_substitution_llr_path=six_b_wt_substitution_llr_path,
            wt_mutation_scoring_manifest_path=six_b_wt_manifest_path,
            foldcheck_ranking_path=out_root / FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
            deliverable_id_prefix="biohub_esmc_6b",
            title=TITLE_6B,
            source_tables=[
                "candidate_table.parquet",
                "foldcheck_review/foldcheck_candidate_ranking.parquet",
                "biohub_esmc/mutation_scoring/esmc_6b_2024_12/wt_substitution_llr.parquet",
                "biohub_esmc/mutation_scoring/esmc_6b_2024_12/wt_mutation_scoring_manifest.yaml",
            ],
        )
        deliverables.extend(six_b_sequence_preference_deliverables)
        deliverables.extend(
            write_biohub_esmc_model_agreement_deliverables(
                panel_root=deliverable_root / BIOHUB_ESMC_SEQUENCE_SCORING_DIR_NAME,
                left_table_path=deliverable_root / BIOHUB_ESMC_SEQUENCE_SCORING_DIR_NAME / VARIANT_LLR_FILE_NAME,
                right_table_path=six_b_sequence_scoring_root / VARIANT_LLR_FILE_NAME,
            )
        )
    candidate_preference_table_path = (
        six_b_sequence_scoring_root / VARIANT_LLR_FILE_NAME
        if (six_b_sequence_scoring_root / VARIANT_LLR_FILE_NAME).exists()
        else deliverable_root / BIOHUB_ESMC_SEQUENCE_SCORING_DIR_NAME / VARIANT_LLR_FILE_NAME
    )
    deliverables.append(
        write_interactive_structure_browser_manifest(
            panel_root=deliverable_root / STRUCTURE_BROWSER_DIR_NAME,
            full_structure_set_path=out_root / FOLDCHECK_FULL_STRUCTURE_SET_RELATIVE_PATH,
            foldcheck_ranking_path=out_root / FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
            reference_structure_path=browser_reference.local_path,
            reference_structure_format=browser_reference.structure_format,
            alignment_reference_path=foldcheck_reference_backbone_path,
            candidate_table_path=candidate_table_path,
            foldcheck_fasta_path=out_root / FOLDCHECK_REQUEST_INPUT_FASTA_RELATIVE_PATH,
            candidate_preference_table_path=candidate_preference_table_path,
        )
    )
    design_class_foldcheck_root = out_root / "design_classes" / "foldcheck_review"
    selected_panel_table_path = out_root / "design_classes" / "selection" / "candidate_selection_panel.parquet"
    candidate_triage_table_path = out_root / "design_classes" / "selection" / "candidate_triage_table.parquet"
    candidate_pool_path = out_root / "design_classes" / "candidate_pool.parquet"
    selected_alignment_reference_path = design_class_foldcheck_root / REFERENCE_STRUCTURE_RELATIVE_PATH
    if selected_panel_table_path.exists() and candidate_pool_path.exists():
        deliverables.append(
            write_selected_panel_structure_browser_manifest(
                panel_root=deliverable_root / STRUCTURE_BROWSER_DIR_NAME,
                full_structure_set_path=design_class_foldcheck_root / "foldcheck_full_structure_set.yaml",
                foldcheck_ranking_path=design_class_foldcheck_root / "foldcheck_candidate_ranking.parquet",
                reference_structure_path=browser_reference.local_path,
                reference_structure_format=browser_reference.structure_format,
                alignment_reference_path=selected_alignment_reference_path,
                candidate_table_path=candidate_pool_path,
                selection_panel_table_path=selected_panel_table_path,
                triage_table_path=candidate_triage_table_path,
                foldcheck_fasta_path=out_root / "design_classes" / "foldcheck_request" / "input_sequences.fasta",
                candidate_preference_table_path=out_root
                / "design_classes"
                / "review_deliverables"
                / BIOHUB_ESMC_SEQUENCE_SCORING_DIR_NAME
                / "esmc_6b_2024_12"
                / VARIANT_LLR_FILE_NAME,
            )
        )
    deliverables.extend(
        write_esmc_model_check_panels(
            panel_root=deliverable_root / WT_MODEL_CHECK_DIR_NAME,
            mutation_scoring_root=out_root / BIOHUB_ESMC_MUTATION_SCORING_RELATIVE_PATH,
        )
    )
    deliverables.extend(
        write_biohub_esmc_sae_interpretation_panels(
            panel_root=deliverable_root / BIOHUB_ESMC_SAE_INTERPRETATION_DIR_NAME,
            heatmap_root=deliverable_root / BIOHUB_ESMC_FEATURE_HEATMAP_DIR_NAME,
            profile_path=out_root / BIOHUB_ESMC_SAE_PROFILE_FILE_NAME,
            protein_features_path=out_root / BIOHUB_ESMC_PROTEIN_FEATURES_FILE_NAME,
            residue_features_path=out_root / BIOHUB_ESMC_RESIDUE_FEATURES_FILE_NAME,
            feature_catalog_path=out_root / BIOHUB_ESMC_FEATURE_CATALOG_FILE_NAME,
            request_manifest_path=out_root / BIOHUB_ESMC_REQUEST_MANIFEST_FILE_NAME,
            foldcheck_ranking_path=out_root / FOLDCHECK_REVIEW_RANKING_RELATIVE_PATH,
            candidate_preference_table_path=candidate_preference_table_path,
            mask_residues=mask_residues,
        )
    )
    deliverables.append(
        write_sae_structure_browser_manifest(
            panel_root=deliverable_root / STRUCTURE_BROWSER_DIR_NAME,
            top_feature_table_path=deliverable_root
            / BIOHUB_ESMC_SAE_INTERPRETATION_DIR_NAME
            / "protein_top_sae_features.parquet",
            residue_features_path=out_root / BIOHUB_ESMC_RESIDUE_FEATURES_FILE_NAME,
            full_structure_set_path=out_root / FOLDCHECK_FULL_STRUCTURE_SET_RELATIVE_PATH,
            reference_structure_path=browser_reference.local_path,
            reference_structure_format=browser_reference.structure_format,
            alignment_reference_path=foldcheck_reference_backbone_path,
        )
    )
    deliverables.extend(linked_selection_readiness_rows(out_root))

    notebook_path = deliverable_root / NOTEBOOKS_DIR_NAME / NOTEBOOK_FILE_NAME
    write_review_deliverables_notebook(notebook_path)
    manifest_path = deliverable_root / MANIFEST_FILE_NAME
    write_manifest(manifest_path, deliverables=deliverables, notebook_path=notebook_path)
    return MaterializedReviewDeliverables(
        manifest_path=manifest_path,
        notebook_path=notebook_path,
        deliverable_count=len(deliverables),
    )


def _remove_retired_deliverables(deliverable_root: Path) -> None:
    """Remove generated artifacts retired by renamed review deliverables."""

    for dirname in ("wt_model_constraint_audit",):
        retired = deliverable_root / dirname
        if retired.is_dir():
            shutil.rmtree(retired)
    for relative_path in (
        Path(MASK_CONTEXT_DIR_NAME) / "linear_mask_tracks.svg",
        Path(PROTEINMPNN_DIR_NAME) / "proteinmpnn_mutation_density.svg",
        Path(PROTEINMPNN_DIR_NAME) / "proteinmpnn_variant_similarity_heatmap.svg",
    ):
        retired_file = deliverable_root / relative_path
        if retired_file.is_file():
            retired_file.unlink()


def _validate_subtype_source_subset(*, clade_source_manifest_path: Path, subtype_source_manifest_path: Path) -> None:
    """Fail fast if the declared Eco1 subtype source set stops being a clade-9 subset."""

    clade_accessions = source_manifest_accessions(clade_source_manifest_path)
    subtype_accessions = source_manifest_accessions(subtype_source_manifest_path)
    missing = sorted(subtype_accessions - clade_accessions)
    if missing:
        shown = ", ".join(missing[:8])
        extra = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(
            "Eco1 subtype MSA source accessions must be a subset of the clade 9 MSA source accessions; "
            f"missing from clade 9: {shown}{extra}"
        )


def _linked_foldcheck_review_rows(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return [
            make_deliverable_row(
                deliverable_id="foldcheck_review_visuals",
                section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
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
        section = (
            SECTION_ESMC_FEATURE_REVIEW if plot_id == "biohub_esmc_sae_coverage" else SECTION_DESIGNS_AND_FOLD_TRIAGE
        )
        rows.append(
            make_deliverable_row(
                deliverable_id=f"foldcheck_review_{plot_id}",
                section=section,
                artifact_kind="linked_visual",
                status=linked_status,
                path=plot_path,
                source_tables=[str(source) for source in plot.get("data_sources", [])],
                input_hashes=file_hashes({"foldcheck_review_manifest": manifest_path, "linked_plot": plot_path}),
                alt_text=str(plot.get("alt_text") or ""),
                description=str(plot.get("description") or ""),
                interpretation_limit=str(plot.get("interpretation_limit") or ""),
                title=str(plot.get("title") or ""),
                role="manuscript_facing" if plot_id == "review_class_counts" else "review_only",
                skip_reason=str(plot.get("skip_reason") or "")
                if linked_status != "skipped_missing_input"
                else f"Missing linked fold-review visual: {plot_path}",
            )
        )
    return rows


def _resolve_linked_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path
