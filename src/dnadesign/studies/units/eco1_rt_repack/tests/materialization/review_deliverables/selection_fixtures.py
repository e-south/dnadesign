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
        _write_svg(plot_root / f"{plot_id}.svg", plot_id=plot_id, title=title)
    payload = {
        "schema_id": "eco1_rt.selection_readiness_manifest",
        "schema_version": 1,
        "status": "materialized",
        "path_policy": "manifest_relative_for_plots",
        "artifacts": {"candidate_selection_panel": "candidate_selection_panel.parquet"},
        "plots": [
            _plot_row(
                plot_id="selection_design_class_gate_counts",
                title=plots["selection_design_class_gate_counts"],
                path="plots/selection_design_class_gate_counts.svg",
                alt_text="Fixture gate-count panel-selection plot.",
                description="Shows candidate pass counts by design class.",
                interpretation_limit="Gate counts do not measure activity.",
                input_hash_tail="a",
            ),
            _plot_row(
                plot_id="selection_panel_review_axes",
                title=plots["selection_panel_review_axes"],
                path="plots/selection_panel_review_axes.svg",
                alt_text="Fixture panel review-axis plot.",
                description="Shows panel review axes.",
                interpretation_limit="Review axes do not measure strand displacement.",
                input_hash_tail="b",
            ),
            _plot_row(
                plot_id="selection_panel_sequence_differences",
                title=plots["selection_panel_sequence_differences"],
                path="plots/selection_panel_sequence_differences.svg",
                alt_text="Fixture selected-panel sequence-difference heatmap.",
                description="Shows WT-match and designed-difference positions for selected panel rows.",
                interpretation_limit="Sequence differences do not measure activity.",
                input_hash_tail="c",
            ),
            _plot_row(
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
    }


def _plot_row(
    *,
    plot_id: str,
    title: str,
    path: str,
    alt_text: str,
    description: str,
    interpretation_limit: str,
    input_hash_tail: str,
) -> dict[str, object]:
    return {
        "plot_id": plot_id,
        "title": title,
        "artifact_kind": "svg",
        "status": "rendered",
        "path": path,
        "data_sources": ["selection/candidate_selection_panel.parquet"],
        "input_hashes": {"candidate_selection_panel": "sha256:" + input_hash_tail * 64},
        "alt_text": alt_text,
        "description": description,
        "interpretation_limit": interpretation_limit,
        "role": "manuscript_facing",
        "render_mode": "wide_visual",
    }


def _write_svg(path: Path, *, plot_id: str, title: str) -> None:
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" role="img" width="320" height="180">
<title>{title}</title>
<desc>Fixture panel-selection visual for review-deliverable linking.</desc>
<text x="20" y="90">{plot_id}</text>
</svg>
""",
        encoding="utf-8",
    )
