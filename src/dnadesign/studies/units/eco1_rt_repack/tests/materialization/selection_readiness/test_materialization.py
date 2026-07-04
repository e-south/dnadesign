"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_materialization.py

Panel-selection materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    materialize_selection_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.feasibility import (
    build_feasibility_rows,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._fixtures import (
    sequence,
    write_inputs,
)


def test_selection_readiness_writes_feasibility_triage_and_one_per_class_panel(tmp_path: Path) -> None:
    repo_root = tmp_path
    class_root = repo_root / "outputs/thread/design_classes"
    selection_root = class_root / "selection"
    source_root = repo_root / "outputs/thread"
    inputs = write_inputs(class_root, source_root)

    result = materialize_selection_readiness(
        repo_root=repo_root,
        output_root=class_root,
        source_output_root=source_root,
        selection_root=selection_root,
        created_at="2026-07-02T00:00:00Z",
    )

    assert result.feasibility_report_path == selection_root / "feasibility_report.parquet"
    assert result.candidate_triage_table_path == selection_root / "candidate_triage_table.parquet"
    assert result.candidate_selection_panel_path == selection_root / "candidate_selection_panel.parquet"
    assert result.candidate_handoff_sequence_csv_path == selection_root / "candidate_handoff_sequences.csv"
    assert result.plots_root == selection_root / "plots"
    assert result.manifest_path == selection_root / "selection_readiness_manifest.yaml"

    feasibility = pq.read_table(result.feasibility_report_path).to_pylist()
    assert {row["candidate_id"] for row in feasibility} == {row["candidate_id"] for row in inputs["candidate_pool"]}
    blocked = next(row for row in feasibility if row["candidate_id"] == "candidate_blocked_by_mask")
    assert blocked["feasibility_status"] == "blocked"
    assert blocked["protected_mutation_violation_count"] == 1

    triage = pq.read_table(result.candidate_triage_table_path).to_pylist()
    low_conf = next(row for row in triage if row["candidate_id"] == "candidate_low_conf")
    assert low_conf["hard_gate_status"] == "ineligible"
    assert next(row for row in triage if row["candidate_id"] == "candidate_blocked_by_mask")["hard_gate_status"] == (
        "ineligible"
    )
    assert {row["sae_window_status"] for row in triage} == {"wt_like_not_used_for_selection"}
    assert all(row["sae_mechanistic_contrast_window_id"] is None for row in triage)
    assert all(row["selection_support_alt_observed_fraction"] is not None for row in triage)
    assert all(row["nucleic_acid_facing_mutation_count"] is not None for row in triage)
    assert all(row["nucleic_acid_facing_chemistry_warning_count"] is not None for row in triage)

    panel = pq.read_table(result.candidate_selection_panel_path).to_pylist()
    assert len(panel) == len(ALL_SPECS)
    assert {row["selection_slot"] for row in panel} == {spec.design_class_id for spec in ALL_SPECS}
    assert {row["design_class_id"] for row in panel} == {spec.design_class_id for spec in ALL_SPECS}
    assert {row["fold_review_class"] for row in panel} == {"strong_fold_preserved"}
    assert all(row["selected_for_panel"] for row in panel)
    assert all(row["eligible_for_handoff"] for row in panel)
    assert "esmc_penalty_rank" not in panel[0]
    assert "sae_window_contrast_rank" not in panel[0]
    assert "MSA support" in panel[0]["selection_reason"]
    assert "not used for selection" in panel[0]["selection_reason"]
    assert "esmc_6b_additive_llr_total" not in panel[0]["tie_break_trace_json"]
    assert "selection_support_alt_observed_fraction" in panel[0]["tie_break_trace_json"]
    assert "mutation_count_total" in panel[0]["tie_break_trace_json"]
    assert "distal_scaffold_mutation_count" in panel[0]["tie_break_trace_json"]

    with result.candidate_handoff_sequence_csv_path.open(encoding="utf-8", newline="") as handle:
        handoff_sequence_rows = list(csv.DictReader(handle))
    assert len(handoff_sequence_rows) == len(panel)
    assert {row["candidate_id"] for row in handoff_sequence_rows} == {row["candidate_id"] for row in panel}
    assert all(
        row["protein_sequence"] == sequence(int(row["candidate_id"].split("_")[-1])) for row in handoff_sequence_rows
    )
    assert {row["dna_design_status"] for row in handoff_sequence_rows} == {"not_materialized"}
    assert {row["restriction_site_screen_status"] for row in handoff_sequence_rows} == {
        "not_applicable_until_dna_sequence_materialized"
    }

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["gate_counts"]["hard_gate_status"] == {"eligible": len(panel), "ineligible": 2}
    assert manifest["gate_counts"]["sae_window_status"] == {"wt_like_not_used_for_selection": len(triage)}
    assert manifest["selected_candidate_ids"] == [row["candidate_id"] for row in panel]
    assert manifest["handoff_readiness"] == {
        "handoff_kind": "rt_only_candidate_handoff",
        "panel_selected": True,
        "candidate_handoff_path": "candidate_handoff.yaml",
        "candidate_handoff_sequence_csv_path": "candidate_handoff_sequences.csv",
        "candidate_handoff_sequence_csv_materialized": True,
        "candidate_handoff_materialized": False,
        "construct_subject_created": False,
    }
    assert manifest["row_counts"]["candidate_handoff_sequences"] == len(panel)
    assert "candidate_handoff_sequences" in manifest["artifact_hashes"]
    assert [plot["plot_id"] for plot in manifest["plots"]] == [
        "selection_design_class_gate_counts",
        "selection_panel_review_axes",
        "selection_panel_sequence_differences",
        "selection_panel_mutation_geography_chemistry",
    ]
    for plot in manifest["plots"]:
        plot_path = result.manifest_path.parent / plot["path"]
        assert plot_path.exists()
        assert "<title" in plot_path.read_text(encoding="utf-8")
        assert plot["alt_text"].strip()
        assert plot["interpretation_limit"].strip()


def test_feasibility_preserves_positions_from_serialized_mutation_lists() -> None:
    candidate_rows = [
        {
            "candidate_id": "candidate_serialized",
            "sequence_hash": "sha256:" + "c" * 64,
            "sequence": sequence(4),
            "status": "accepted",
            "mutation_count": 2,
            "mutable_mutation_count": 2,
            "protected_mutation_count": 0,
            "outside_mutable_positions": [],
            "canonical_mutations": "['A7G', 'L21V']",
        }
    ]

    rows = build_feasibility_rows(
        candidate_rows=candidate_rows,
        foldcheck_report_rows=[
            {"candidate_id": "wild_type", "input_sequence_hash": "sha256:" + "0" * 64, "status": "accepted"},
            {"candidate_id": "candidate_serialized", "status": "accepted"},
        ],
        input_candidate_pool_hash="sha256:pool",
        input_mask_policy_hash="sha256:mask",
        input_foldcheck_report_hash="sha256:fold",
        created_at="2026-07-02T00:00:00Z",
    )

    assert rows[0]["max_mutation_window_mutation_count"] == 1
    assert rows[0]["mutation_windows_json"] == (
        '[{"density": 1.0, "end": 7, "length": 1, "mutation_count": 1, "start": 7}, '
        '{"density": 1.0, "end": 21, "length": 1, "mutation_count": 1, "start": 21}]'
    )
