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
            slot="clade9_p25_contact5a",
            candidate_id="thread_candidate_alpha",
            mutation_count=2,
            msa_fraction=0.75,
            na_facing=1,
            chemistry_warnings=0,
        ),
        panel_row(
            slot="clade9_p25_contact6a",
            candidate_id="thread_candidate_beta",
            mutation_count=3,
            msa_fraction=0.6,
            na_facing=2,
            chemistry_warnings=1,
        ),
    ]
    pq.write_table(pa.Table.from_pylist(panel_rows), selection_root / "candidate_selection_panel.parquet")
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
        "selection_policy_id": "eco1_rt_structure_evolution_class_representative_panel_v1",
        "governing_rule": (
            "Select one feasible fold-preserved representative from each design class. Do not use ESMC or SAE "
            "as positive selection evidence."
        ),
        "sae_window_policy": "SAE windows are retained for review evidence and are not panel-selection inputs.",
        "esmc_policy": "ESMC additive LLR rows are retained for review and are not panel-selection tie-breaks.",
        "path_policy": "paths_relative_to_selection_manifest",
        "artifacts": {
            "candidate_triage_table": "candidate_triage_table.parquet",
            "candidate_selection_panel": "candidate_selection_panel.parquet",
            "candidate_handoff_sequences": "candidate_handoff_sequences.csv",
        },
        "row_counts": {
            "candidate_triage_table": 2,
            "candidate_selection_panel": 2,
            "candidate_handoff_sequences": 2,
        },
        "gate_counts": {
            "hard_gate_status": {"eligible": 2},
            "sae_window_status": {"wt_like_not_used_for_selection": 2},
        },
        "selected_candidate_ids": ["thread_candidate_alpha", "thread_candidate_beta"],
        "panel_tie_break_order": [
            "fold review class",
            "selection-support MSA observed fraction",
            "selection-support unobserved mutation count",
            "near retained DNA/RNA or thumb-track mutation count",
            "near retained DNA/RNA or thumb-track chemistry warning count",
            "nearest selected sequence distance",
        ],
        "handoff_readiness": {
            "handoff_kind": "rt_only_candidate_handoff",
            "panel_selected": True,
            "candidate_handoff_path": "../../candidate_handoff.yaml",
            "candidate_handoff_sequence_csv_path": "candidate_handoff_sequences.csv",
            "candidate_handoff_sequence_csv_materialized": True,
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
                plot_id="selection_population_stratification",
                title=plots["selection_population_stratification"],
                path="plots/selection_population_stratification.svg",
                alt_text="Fixture full candidate population stratification plot with selected rows highlighted.",
                description="Shows the six selected candidates relative to the full candidate population.",
                interpretation_limit="Population stratification is not an activity measurement.",
                input_hash_tail="b",
            ),
            plot_row(
                plot_id="selection_class_local_percentiles",
                title=plots["selection_class_local_percentiles"],
                path="plots/selection_class_local_percentiles.svg",
                alt_text="Fixture class-local percentile plot.",
                description="Shows selected rows against their own design classes.",
                interpretation_limit="Class-local review variables do not measure activity.",
                input_hash_tail="c",
            ),
            plot_row(
                plot_id="selection_six_sequence_distance",
                title=plots["selection_six_sequence_distance"],
                path="plots/selection_six_sequence_distance.svg",
                alt_text="Fixture selected-six sequence-distance heatmap.",
                description="Shows selected candidate sequence distances.",
                interpretation_limit="Sequence distance does not measure activity.",
                input_hash_tail="d",
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
        ],
    }
    (selection_root / "selection_readiness_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
