"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_materialization.py

Panel-selection materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import (
    ALL_SPECS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    cli as selection_readiness_cli,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    materialize_selection_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    CURRENT_SELECTION_PLOT_IDS,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._fixtures import (
    write_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._handoff_fixture import (
    candidate_handoff_payload,
)


def test_selection_readiness_writes_feasibility_triage_and_one_per_class_panel(tmp_path: Path) -> None:
    repo_root = tmp_path
    class_root = repo_root / "outputs/thread/design_classes"
    selection_root = class_root / "selection"
    source_root = repo_root / "outputs/thread"
    inputs = write_inputs(class_root, source_root)
    root_handoff_path = source_root / "candidate_handoff.yaml"
    root_handoff_path.write_text(yaml.safe_dump(candidate_handoff_payload(), sort_keys=False), encoding="utf-8")
    selection_local_handoff_path = selection_root / "candidate_handoff.yaml"
    selection_local_handoff_path.parent.mkdir(parents=True, exist_ok=True)
    selection_local_handoff_path.write_text("handoff_kind: wrong_local_path\n", encoding="utf-8")
    retired_plot = selection_root / "plots" / "selection_panel_review_axes.svg"
    retired_plot.parent.mkdir(parents=True, exist_ok=True)
    retired_plot.write_text("<svg>retired selected-only scatter</svg>\n", encoding="utf-8")

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

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["path_policy"] == "paths_relative_to_selection_manifest"
    assert all(not Path(value).is_absolute() for value in manifest["source_tables"].values())
    assert all(not Path(value).is_absolute() for value in manifest["artifacts"].values())
    assert manifest["gate_counts"]["hard_gate_status"] == {"eligible": len(panel), "ineligible": 2}
    assert manifest["gate_counts"]["sae_window_status"] == {"wt_like_not_used_for_selection": len(triage)}
    assert manifest["selected_candidate_ids"] == [row["candidate_id"] for row in panel]
    assert manifest["handoff_readiness"] == {
        "handoff_kind": "rt_only_candidate_handoff",
        "panel_selected": True,
        "candidate_handoff_path": "../../candidate_handoff.yaml",
        "candidate_handoff_sequence_csv_path": "candidate_handoff_sequences.csv",
        "candidate_handoff_sequence_csv_materialized": True,
        "candidate_handoff_file_present": True,
        "candidate_handoff_materialized": True,
        "construct_subject_created": False,
    }
    assert manifest["panel_coverage"] == {
        "expected_design_class_count": len(ALL_SPECS),
        "selected_row_count": len(panel),
        "required_rows_per_class": 1,
        "missing_design_classes": [],
        "duplicate_design_classes": [],
        "unexpected_design_classes": [],
        "valid": True,
    }
    assert manifest["row_counts"]["candidate_handoff_sequences"] == len(panel)
    assert "candidate_handoff_sequences" in manifest["artifact_hashes"]
    assert [plot["plot_id"] for plot in manifest["plots"]] == list(CURRENT_SELECTION_PLOT_IDS)
    plot_text_by_id: dict[str, str] = {}
    for plot in manifest["plots"]:
        plot_path = result.manifest_path.parent / plot["path"]
        assert plot_path.exists()
        plot_text = plot_path.read_text(encoding="utf-8")
        plot_text_by_id[str(plot["plot_id"])] = plot_text
        assert "<title" in plot_text
        assert plot["alt_text"].strip()
        assert plot["interpretation_limit"].strip()
    gate_count_text = plot_text_by_id["selection_design_class_gate_counts"]
    assert "Passes protein gate" in gate_count_text
    assert "Fold-review reserve" in gate_count_text
    assert "Blocked by gate" in gate_count_text
    assert "Missing fold or feasibility input" in gate_count_text
    assert "Manual reserve" not in gate_count_text
    assert "Excluded" not in gate_count_text
    _assert_svg_has_square_panel(gate_count_text)
    _assert_heatmap_cells_are_square(
        plot_text_by_id["selection_class_local_percentiles"],
        row_count=len(ALL_SPECS),
        column_count=6,
    )
    _assert_heatmap_cells_are_square(
        plot_text_by_id["selection_regional_mutation_burden"],
        row_count=len(ALL_SPECS),
        column_count=4,
    )
    assert not retired_plot.exists()


def test_selection_readiness_cli_reports_handoff_sequence_csv_path(tmp_path: Path, capsys) -> None:
    repo_root = tmp_path
    class_root = repo_root / "outputs/thread/design_classes"
    selection_root = class_root / "selection"
    source_root = repo_root / "outputs/thread"
    write_inputs(class_root, source_root)

    exit_code = selection_readiness_cli.main(
        [
            "--repo-root",
            str(repo_root),
            "--output-root",
            str(class_root),
            "--source-output-root",
            str(source_root),
            "--selection-root",
            str(selection_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["candidate_handoff_sequence_csv_path"] == str(selection_root / "candidate_handoff_sequences.csv")
    assert Path(payload["candidate_handoff_sequence_csv_path"]).exists()


def _svg_clip_rects(svg_text: str) -> list[tuple[float, float, float, float]]:
    return [
        tuple(float(value) for value in match)
        for match in re.findall(
            r'<clipPath id="[^"]+">\s*<rect x="([0-9.]+)" y="([0-9.]+)" width="([0-9.]+)" height="([0-9.]+)"',
            svg_text,
        )
    ]


def _assert_svg_has_square_panel(svg_text: str) -> None:
    assert any(
        width > 100.0 and height > 100.0 and abs(width - height) <= 1.0
        for _x, _y, width, height in _svg_clip_rects(svg_text)
    )


def _assert_heatmap_cells_are_square(svg_text: str, *, row_count: int, column_count: int) -> None:
    expected_ratio = column_count / row_count
    assert any(
        width > 80.0 and height > 80.0 and abs((width / height) - expected_ratio) <= 0.03
        for _x, _y, width, height in _svg_clip_rects(svg_text)
    )
