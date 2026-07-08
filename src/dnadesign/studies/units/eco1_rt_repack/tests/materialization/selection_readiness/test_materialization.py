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
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    cli as selection_readiness_cli,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    materialize_selection_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness import (
    _selection_manifest_assertions as manifest_assertions,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness import (
    _selection_table_assertions as table_assertions,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._fixtures import (
    write_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._handoff_fixture import (
    candidate_handoff_payload,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._source_basis_fixture import (
    write_manual_mask_authority_source_basis,
)


def test_selection_readiness_writes_feasibility_triage_and_primary_panel(tmp_path: Path) -> None:
    repo_root = tmp_path
    class_root = repo_root / "outputs/thread/design_classes"
    selection_root = class_root / "selection"
    source_root = repo_root / "outputs/thread"
    inputs = write_inputs(class_root, source_root)
    write_manual_mask_authority_source_basis(repo_root)
    _write_handoff_fixture(source_root=source_root, selection_root=selection_root)
    retired_plot = _write_retired_plot(selection_root)

    result = materialize_selection_readiness(
        repo_root=repo_root,
        output_root=class_root,
        source_output_root=source_root,
        selection_root=selection_root,
        created_at="2026-07-02T00:00:00Z",
    )

    assert result.feasibility_report_path == selection_root / "feasibility_report.parquet"
    assert result.candidate_triage_table_path == selection_root / "candidate_triage_table.parquet"
    assert result.local_structure_region_metrics_path == selection_root / "local_structure_region_metrics.parquet"
    assert result.local_structure_threshold_sensitivity_path == (
        selection_root / "local_structure_threshold_sensitivity.parquet"
    )
    assert result.region_msa_support_path == selection_root / "region_msa_support.parquet"
    assert result.primary_panel_selection_trace_path == selection_root / "primary_panel_selection_trace.parquet"
    assert result.candidate_selection_panel_path == selection_root / "candidate_selection_panel.parquet"
    assert result.candidate_handoff_sequence_csv_path == selection_root / "candidate_handoff_sequences.csv"
    assert result.plots_root == selection_root / "plots"
    assert result.manifest_path == selection_root / "selection_readiness_manifest.yaml"

    triage, panel = table_assertions.assert_materialized_selection_tables(result, inputs)
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    manifest_assertions.assert_materialized_selection_manifest(
        result=result,
        manifest=manifest,
        triage=triage,
        panel=panel,
        retired_plot=retired_plot,
    )


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


def _write_handoff_fixture(*, source_root: Path, selection_root: Path) -> None:
    root_handoff_path = source_root / "candidate_handoff.yaml"
    root_handoff_path.write_text(yaml.safe_dump(candidate_handoff_payload(), sort_keys=False), encoding="utf-8")
    selection_local_handoff_path = selection_root / "candidate_handoff.yaml"
    selection_local_handoff_path.parent.mkdir(parents=True, exist_ok=True)
    selection_local_handoff_path.write_text("handoff_kind: wrong_local_path\n", encoding="utf-8")


def _write_retired_plot(selection_root: Path) -> Path:
    retired_plot = selection_root / "plots" / "selection_panel_review_axes.svg"
    retired_plot.parent.mkdir(parents=True, exist_ok=True)
    retired_plot.write_text("<svg>retired selected-only scatter</svg>\n", encoding="utf-8")
    return retired_plot
