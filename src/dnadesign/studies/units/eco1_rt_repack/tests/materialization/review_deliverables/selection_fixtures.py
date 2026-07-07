"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/selection_fixtures.py

Panel-selection fixtures for Eco1 review-deliverable tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_PLAIN_TITLES,
)

from .selection_plot_fixtures import plot_row, write_svg
from .selection_sequence_fixtures import write_handoff_sequence_csv
from .selection_table_fixtures import panel_row, triage_row


def write_selection_readiness_manifest(selection_root: Path) -> None:
    plot_root = selection_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    panel_rows = [
        panel_row(
            slot="primary_panel_01",
            design_class_id=ALL_SPECS[-1].design_class_id,
            candidate_id="thread_candidate_alpha",
            mutation_count=2,
            msa_fraction=0.75,
            na_facing=1,
            chemistry_warnings=0,
        ),
        panel_row(
            slot="primary_panel_02",
            design_class_id=ALL_SPECS[-1].design_class_id,
            candidate_id="thread_candidate_beta",
            mutation_count=3,
            msa_fraction=0.6,
            na_facing=2,
            chemistry_warnings=1,
        ),
    ]
    pq.write_table(pa.Table.from_pylist(panel_rows), selection_root / "candidate_selection_panel.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "selection_policy_id": "eco1_rt_primary_conservative_panel_v1",
                    "stage_order": index,
                    "stage_id": stage_id,
                    "stage_label": stage_label,
                    "selector_role": selector_role,
                    "filter_rule": "Fixture primary-panel funnel stage.",
                    "input_count": input_count,
                    "removed_count": 0,
                    "remaining_count": remaining_count,
                    "is_hard_gate": is_hard_gate,
                }
                for index, (
                    stage_id,
                    stage_label,
                    selector_role,
                    input_count,
                    remaining_count,
                    is_hard_gate,
                ) in enumerate(
                    (
                        ("candidate_pool", "Accepted candidate pool", "input_pool", 2, 2, False),
                        ("broad_contract_pool", "Broad protein contract", "hard_gate", 2, 2, True),
                        (
                            "primary_panel_candidate_pool",
                            "Primary candidate pool",
                            "preservation_contract",
                            2,
                            2,
                            False,
                        ),
                        (
                            "global_conservative_diverse_selection",
                            "Conservative-diverse six-row selection",
                            "global_rank",
                            2,
                            2,
                            False,
                        ),
                    ),
                    start=1,
                )
            ]
        ),
        selection_root / "primary_panel_selection_trace.parquet",
    )
    write_handoff_sequence_csv(selection_root / "candidate_handoff_sequences.csv", panel_rows)
    pq.write_table(
        pa.Table.from_pylist(
            [
                triage_row(candidate_id="thread_candidate_alpha", msa_fraction=0.75, charge_delta=1),
                triage_row(candidate_id="thread_candidate_beta", msa_fraction=0.6, charge_delta=-1),
            ]
        ),
        selection_root / "candidate_triage_table.parquet",
    )
    plots = SELECTION_PLOT_PLAIN_TITLES
    for plot_id, title in plots.items():
        write_svg(plot_root / f"{plot_id}.svg", plot_id=plot_id, title=title)
    payload = {
        "schema_id": "eco1_rt.selection_readiness_manifest",
        "schema_version": 1,
        "status": "materialized",
        "selection_policy_id": "eco1_rt_primary_conservative_panel_v1",
        "governing_rule": (
            "Select primary conservative candidates globally after broad protein-contract and stricter "
            "primary-panel checks. Do not use ESMC or SAE as positive selection evidence."
        ),
        "sae_window_policy": "SAE windows are retained for review evidence and are not panel-selection inputs.",
        "esmc_policy": "ESMC additive LLR rows are retained for review and are not panel-selection tie-breaks.",
        "path_policy": "paths_relative_to_selection_manifest",
        "artifacts": {
            "candidate_triage_table": "candidate_triage_table.parquet",
            "primary_panel_selection_trace": "primary_panel_selection_trace.parquet",
            "candidate_selection_panel": "candidate_selection_panel.parquet",
            "candidate_handoff_sequences": "candidate_handoff_sequences.csv",
        },
        "row_counts": {
            "candidate_triage_table": 2,
            "primary_panel_selection_trace": 4,
            "candidate_selection_panel": 2,
            "candidate_handoff_sequences": 2,
        },
        "gate_counts": {
            "hard_gate_status": {"eligible": 2},
            "sae_window_status": {"wt_like_not_used_for_selection": 2},
        },
        "selection_funnel_stages": [
            {
                "stage_id": "candidate_pool",
                "stage_label": "Accepted candidate pool",
                "selector_role": "input_pool",
                "filter_rule": "Accepted ProteinMPNN candidate rows before protein-level selection checks.",
                "input_count": 2,
                "removed_count": 0,
                "remaining_count": 2,
                "is_hard_gate": False,
            },
            {
                "stage_id": "broad_contract_pool",
                "stage_label": "Broad protein contract",
                "selector_role": "hard_gate",
                "filter_rule": "Keep rows passing protein preservation checks.",
                "input_count": 2,
                "removed_count": 0,
                "remaining_count": 2,
                "is_hard_gate": True,
            },
            {
                "stage_id": "primary_panel_candidate_pool",
                "stage_label": "Primary candidate pool",
                "selector_role": "preservation_contract",
                "filter_rule": "Keep rows passing the stricter C-terminal/thumb local RMSD check.",
                "input_count": 2,
                "removed_count": 0,
                "remaining_count": 2,
                "is_hard_gate": False,
            },
            {
                "stage_id": "global_conservative_diverse_selection",
                "stage_label": "Conservative-diverse six-row selection",
                "selector_role": "global_rank",
                "filter_rule": (
                    "Select primary-panel candidates globally by conservative rank fields and mutation-set "
                    "dissimilarity; design class is context, not a quota."
                ),
                "input_count": 2,
                "removed_count": 0,
                "remaining_count": 2,
                "is_hard_gate": False,
            },
        ],
        "selected_candidate_ids": ["thread_candidate_alpha", "thread_candidate_beta"],
        "panel_tie_break_order": [
            "fewest proximal unsupported substitutions",
            "fewest acidic gains near retained DNA/RNA or thumb-track",
            "fewest basic losses near retained DNA/RNA or thumb-track",
            "fewest Pro/Gly gains near retained DNA/RNA or thumb-track",
            "largest nearest selected mutation-position Jaccard distance",
            "largest nearest selected exact-substitution Jaccard distance",
            "lowest C-terminal primer-RNA recognition-region C-alpha RMSD",
            "lowest substrate-relevant local C-alpha RMSD",
            "fold metrics",
            "sequence hash",
        ],
        "handoff_readiness": {
            "handoff_kind": "rt_only_candidate_handoff",
            "panel_selected": True,
            "candidate_handoff_path": "../../candidate_handoff.yaml",
            "candidate_handoff_sequence_csv_path": "candidate_handoff_sequences.csv",
            "candidate_handoff_sequence_csv_materialized": True,
            "candidate_handoff_file_present": False,
            "candidate_handoff_materialized": False,
            "construct_subject_created": False,
        },
        "plots": [
            plot_row(
                plot_id="selection_design_class_gate_counts",
                title=plots["selection_design_class_gate_counts"],
                path="plots/selection_design_class_gate_counts.svg",
                alt_text="Fixture gate-count panel-selection plot.",
                description="Shows candidate pass counts by design class.",
                interpretation_limit="Gate counts do not measure activity.",
                input_hash_tail="a",
            ),
            plot_row(
                plot_id="selection_design_class_contrast",
                title=plots["selection_design_class_contrast"],
                path="plots/selection_design_class_contrast.svg",
                alt_text="Fixture design-class contrast summary.",
                description="Shows mask-policy contrasts for the declared design classes.",
                interpretation_limit="Design-class contrast does not measure activity.",
                input_hash_tail="b",
            ),
            plot_row(
                plot_id="selection_primary_panel_sankey",
                title=plots["selection_primary_panel_sankey"],
                path="plots/selection_primary_panel_sankey.svg",
                alt_text="Fixture primary-panel funnel Sankey.",
                description="Shows broad, primary, boundary, and selected rows.",
                interpretation_limit="The primary-panel funnel does not measure activity.",
                input_hash_tail="i",
            ),
            plot_row(
                plot_id="selection_local_structure_stratification",
                title=plots["selection_local_structure_stratification"],
                path="plots/selection_local_structure_stratification.svg",
                alt_text="Fixture local-RMSD threshold stratification plot.",
                description="Shows local RMSD thresholds against candidate distributions.",
                interpretation_limit="Local RMSD thresholds do not measure activity.",
                input_hash_tail="s",
            ),
            plot_row(
                plot_id="selection_local_structure_threshold_sensitivity",
                title=plots["selection_local_structure_threshold_sensitivity"],
                path="plots/selection_local_structure_threshold_sensitivity.svg",
                alt_text="Fixture local-RMSD threshold sensitivity plot.",
                description="Shows failure counts under tighter, declared, and looser local RMSD thresholds.",
                interpretation_limit="Threshold sensitivity does not measure activity.",
                input_hash_tail="t",
            ),
            plot_row(
                plot_id="selection_local_structure_by_region",
                title=plots["selection_local_structure_by_region"],
                path="plots/selection_local_structure_by_region.svg",
                alt_text="Fixture local-structure heatmap.",
                description="Shows local backbone shifts by RT region.",
                interpretation_limit="Local backbone shifts do not measure activity.",
                input_hash_tail="h",
            ),
            plot_row(
                plot_id="selection_premise_alignment",
                title=plots["selection_premise_alignment"],
                path="plots/selection_premise_alignment.svg",
                alt_text="Fixture selected-panel premise checklist plot.",
                description="Shows selected rows against the core review premise.",
                interpretation_limit="The premise checklist does not measure activity.",
                input_hash_tail="p",
            ),
            plot_row(
                plot_id="selection_selected_substitutions_across_rt",
                title=plots["selection_selected_substitutions_across_rt"],
                path="plots/selection_selected_substitutions_across_rt.svg",
                alt_text="Fixture selected-substitutions heatmap.",
                description="Shows selected substitutions across Eco1 RT.",
                interpretation_limit="Substitution context does not measure activity.",
                input_hash_tail="e",
            ),
            plot_row(
                plot_id="selection_regional_mutation_burden",
                title=plots["selection_regional_mutation_burden"],
                path="plots/selection_regional_mutation_burden.svg",
                alt_text="Fixture regional mutation-burden heatmap.",
                description="Shows mutation burden by RT region.",
                interpretation_limit="Regional mutation burden does not measure activity.",
                input_hash_tail="f",
            ),
            plot_row(
                plot_id="selection_na_facing_chemistry_balance",
                title=plots["selection_na_facing_chemistry_balance"],
                path="plots/selection_na_facing_chemistry_balance.svg",
                alt_text="Fixture near-DNA/RNA chemistry-balance heatmap.",
                description="Shows chemistry changes near retained DNA/RNA or thumb-track.",
                interpretation_limit="Chemistry balance does not measure activity.",
                input_hash_tail="g",
            ),
            plot_row(
                plot_id="selection_regionwise_msa_support",
                title=plots["selection_regionwise_msa_support"],
                path="plots/selection_regionwise_msa_support.svg",
                alt_text="Fixture region-wise MSA support heatmap.",
                description="Shows selected substitution support by mutation region.",
                interpretation_limit="Region-wise MSA support does not measure activity.",
                input_hash_tail="m",
            ),
            plot_row(
                plot_id="selection_six_sequence_distance",
                title=plots["selection_six_sequence_distance"],
                path="plots/selection_six_sequence_distance.svg",
                alt_text="Fixture selected mutation-set dissimilarity heatmap.",
                description="Shows selected candidate mutation-set dissimilarity.",
                interpretation_limit="Mutation-set dissimilarity does not measure activity.",
                input_hash_tail="d",
            ),
        ],
    }
    (selection_root / "selection_readiness_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
