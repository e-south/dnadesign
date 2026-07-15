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

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.constants import (
    COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
    NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_contract import (
    SELECTION_POLICY_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_PLAIN_TITLES,
)

from .selection_manifest_contract import (
    PANEL_TIE_BREAK_ORDER,
    SELECTION_FUNNEL_STAGES,
    selection_trace_rows,
)
from .selection_plot_fixtures import plot_row, write_svg
from .selection_sequence_fixtures import write_handoff_sequence_csv
from .selection_table_fixtures import panel_row, triage_row


def write_selection_readiness_manifest(selection_root: Path) -> None:
    plot_root = selection_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    panel_rows = [
        panel_row(
            slot="selected_hypothesis_01",
            policy_id=COMBINED_NEAR_PLUS_DISTAL_POLICY_ID,
            candidate_id="thread_candidate_alpha",
            mutation_count=2,
            msa_fraction=0.75,
            na_facing=1,
            chemistry_warnings=0,
        ),
        panel_row(
            slot="selected_hypothesis_02",
            policy_id=NEAR_DNA_RNA_ACID_FREE_POLICY_ID,
            candidate_id="thread_candidate_beta",
            mutation_count=3,
            msa_fraction=0.6,
            na_facing=2,
            chemistry_warnings=1,
        ),
    ]
    pq.write_table(pa.Table.from_pylist(panel_rows), selection_root / "candidate_selection_panel.parquet")
    pq.write_table(
        pa.Table.from_pylist(selection_trace_rows()),
        selection_root / "hypothesis_panel_selection_trace.parquet",
    )
    write_handoff_sequence_csv(selection_root / "candidate_handoff_sequences.csv", panel_rows)
    pq.write_table(
        pa.Table.from_pylist(
            [
                triage_row(
                    candidate_id="thread_candidate_alpha",
                    msa_fraction=0.75,
                    charge_delta=1,
                    mutation_count=2,
                ),
                triage_row(
                    candidate_id="thread_candidate_beta",
                    msa_fraction=0.6,
                    charge_delta=-1,
                    mutation_count=3,
                ),
            ]
        ),
        selection_root / "candidate_triage_table.parquet",
    )
    plots = SELECTION_PLOT_PLAIN_TITLES
    for plot_id, title in plots.items():
        write_svg(plot_root / f"{plot_id}.svg", plot_id=plot_id, title=title)
    payload = {
        "schema_id": "eco1_rt.selection_readiness_manifest",
        "schema_version": 3,
        "status": "materialized",
        "selection_policy_id": SELECTION_POLICY_ID,
        "governing_rule": (
            "Select policy-defined hypotheses by within-group mutation-set distance. Design groups define "
            "experimental comparisons, not quality tiers. Do not use ESMC or SAE as selection evidence."
        ),
        "path_policy": "paths_relative_to_selection_manifest",
        "artifacts": {
            "candidate_triage_table": "candidate_triage_table.parquet",
            "hypothesis_panel_selection_trace": "hypothesis_panel_selection_trace.parquet",
            "candidate_selection_panel": "candidate_selection_panel.parquet",
            "candidate_handoff_sequences": "candidate_handoff_sequences.csv",
        },
        "row_counts": {
            "candidate_triage_table": 2,
            "hypothesis_panel_selection_trace": len(SELECTION_FUNNEL_STAGES),
            "candidate_selection_panel": 2,
            "candidate_handoff_sequences": 2,
        },
        "gate_counts": {
            "hard_gate_status": {"eligible": 2},
            "sae_window_status": {"wt_like_not_used_for_selection": 2},
        },
        "selection_funnel_stages": SELECTION_FUNNEL_STAGES,
        "selected_candidate_ids": ["thread_candidate_alpha", "thread_candidate_beta"],
        "panel_coverage": {"selected_panel_size": 2},
        "panel_tie_break_order": PANEL_TIE_BREAK_ORDER,
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
        "plots": _selection_plot_rows(plots),
    }
    (selection_root / "selection_readiness_manifest.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _selection_plot_rows(plots: dict[str, str]) -> list[dict[str, object]]:
    return [
        plot_row(
            plot_id=plot_id,
            title=title,
            path=f"plots/{plot_id}.svg",
            alt_text=f"Fixture selection plot for {title}.",
            description=f"Fixture metadata for the {title} selection-readiness plot.",
            interpretation_limit="Fixture selection plots do not measure activity.",
            input_hash_tail=chr(97 + index),
        )
        for index, (plot_id, title) in enumerate(plots.items())
    ]
