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

from .selection_plot_fixtures import plot_row, write_svg


def write_selection_readiness_manifest(selection_root: Path) -> None:
    plot_root = selection_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                _panel_row(
                    slot="clade9_p25_contact5a",
                    candidate_id="thread_candidate_alpha",
                    mutation_count=2,
                    msa_fraction=0.75,
                    na_facing=1,
                    chemistry_warnings=0,
                ),
                _panel_row(
                    slot="clade9_p25_contact6a",
                    candidate_id="thread_candidate_beta",
                    mutation_count=3,
                    msa_fraction=0.6,
                    na_facing=2,
                    chemistry_warnings=1,
                ),
            ]
        ),
        selection_root / "candidate_selection_panel.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                _triage_row(candidate_id="thread_candidate_alpha", msa_fraction=0.75, charge_delta=1),
                _triage_row(candidate_id="thread_candidate_beta", msa_fraction=0.6, charge_delta=-1),
            ]
        ),
        selection_root / "candidate_triage_table.parquet",
    )
    plots = {
        "selection_design_class_gate_counts": "Eco1 design classes retain fold-preserved candidates",
        "selection_panel_review_axes": "The six selected candidates balance MSA support with mutation geography",
        "selection_panel_sequence_differences": "Selected Eco1 candidates vary only at designable protein positions",
        "selection_panel_mutation_geography_chemistry": "Selected candidates change distal scaffold chemistry",
    }
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
        "path_policy": "manifest_relative_for_plots",
        "artifacts": {
            "candidate_triage_table": "candidate_triage_table.parquet",
            "candidate_selection_panel": "candidate_selection_panel.parquet",
        },
        "row_counts": {
            "candidate_triage_table": 2,
            "candidate_selection_panel": 2,
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
            "nucleic-acid-facing mutation count",
            "nucleic-acid-facing chemistry warning count",
            "nearest selected sequence distance",
        ],
        "handoff_readiness": {
            "handoff_kind": "rt_only_candidate_handoff",
            "panel_selected": True,
            "candidate_handoff_path": "candidate_handoff.yaml",
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
                plot_id="selection_panel_review_axes",
                title=plots["selection_panel_review_axes"],
                path="plots/selection_panel_review_axes.svg",
                alt_text="Fixture panel review-axis plot.",
                description="Shows panel review axes.",
                interpretation_limit="Review axes do not measure strand displacement.",
                input_hash_tail="b",
            ),
            plot_row(
                plot_id="selection_panel_sequence_differences",
                title=plots["selection_panel_sequence_differences"],
                path="plots/selection_panel_sequence_differences.svg",
                alt_text="Fixture selected-panel sequence-difference heatmap.",
                description="Shows WT-match and designed-difference positions for selected panel rows.",
                interpretation_limit="Sequence differences do not measure activity.",
                input_hash_tail="c",
            ),
            plot_row(
                plot_id="selection_panel_mutation_geography_chemistry",
                title=plots["selection_panel_mutation_geography_chemistry"],
                path="plots/selection_panel_mutation_geography_chemistry.svg",
                alt_text="Fixture selected-panel mutation-chemistry heatmap.",
                description="Shows chemistry classes for selected panel substitutions.",
                interpretation_limit="Mutation chemistry categories do not measure activity.",
                input_hash_tail="d",
            ),
        ],
    }
    (selection_root / "selection_readiness_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _panel_row(
    *,
    slot: str,
    candidate_id: str,
    mutation_count: int,
    msa_fraction: float,
    na_facing: int,
    chemistry_warnings: int,
) -> dict[str, object]:
    trace_json = (
        f'{{"selection_support_alt_observed_fraction": {msa_fraction}, '
        f'"selection_support_unobserved_mutation_count": 1, '
        f'"mutation_count_total": {mutation_count}, '
        f'"mean_plddt": 92.4, '
        f'"wt_runtime_ca_rmsd": 0.82, '
        f'"cryoem_mapped_ca_rmsd": 1.23, '
        f'"nucleic_acid_facing_mutation_count": {na_facing}, '
        f'"nucleic_acid_facing_charge_delta": 1, '
        f'"nucleic_acid_facing_chemistry_warning_count": {chemistry_warnings}, '
        f'"catalytic_or_direct_contact_mutation_count": 0, '
        f'"thumb_contact_track_mutation_count": 0, '
        f'"distal_scaffold_mutation_count": {mutation_count}}}'
    )
    return {
        "selection_slot": slot,
        "candidate_id": candidate_id,
        "design_class_id": slot,
        "fold_review_class": "strong_fold_preserved",
        "feasibility_status": "pass",
        "nearest_selected_distance_aa": 4,
        "selection_reason": "fixture selected panel row",
        "tie_break_trace_json": trace_json,
    }


def _triage_row(*, candidate_id: str, msa_fraction: float, charge_delta: int) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "selection_support_alt_observed_fraction": msa_fraction,
        "selection_support_unobserved_mutation_count": 1,
        "nucleic_acid_facing_mutation_count": 2,
        "nucleic_acid_facing_charge_delta": charge_delta,
        "nucleic_acid_facing_chemistry_warning_count": 1,
        "catalytic_or_direct_contact_mutation_count": 0,
        "thumb_contact_track_mutation_count": 0,
        "distal_scaffold_mutation_count": 2,
        "hard_gate_status": "eligible",
        "sae_window_status": "wt_like_not_used_for_selection",
    }
