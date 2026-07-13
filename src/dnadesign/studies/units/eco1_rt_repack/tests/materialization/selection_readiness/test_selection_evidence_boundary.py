"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_selection_evidence_boundary.py

Selection-evidence boundary tests for Eco1 selection readiness.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    materialize_selection_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.panel_ranking import (
    choose_farthest_candidate as _choose_farthest_candidate,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._fixtures import (
    write_inputs,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._panel_contract_fixtures import (
    candidate_row,
)


def test_missing_model_review_sources_do_not_degrade_selection_manifest(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs/thread/design_classes"
    selection_root = output_root / "selection"
    source_root = tmp_path / "outputs/thread"
    write_inputs(output_root, source_root)
    for path in (
        output_root / "review_deliverables/biohub_esmc_sequence_scoring/biohub_esmc_variant_llr_scores.parquet",
        output_root
        / "review_deliverables/biohub_esmc_sequence_scoring/esmc_6b_2024_12/biohub_esmc_variant_llr_scores.parquet",
        output_root / "biohub_esmc/sae_feature_window_summary.parquet",
    ):
        path.unlink()

    result = materialize_selection_readiness(
        repo_root=tmp_path,
        output_root=output_root,
        source_output_root=source_root,
        selection_root=selection_root,
        created_at="2026-07-02T00:00:00Z",
    )

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "materialized"
    assert "optional_review_sources" not in manifest
    assert "missing_optional_review_sources" not in manifest
    assert manifest["handoff_readiness"]["candidate_handoff_materialized"] is False


def test_distinct_mutation_set_can_precede_soft_chemistry_penalties() -> None:
    selected = [candidate_row("already_selected", na_facing_mutation_count=0)]
    rows = [
        candidate_row("overlapping_no_soft_warning", na_facing_mutation_count=0),
        candidate_row(
            "distinct_with_soft_warning",
            na_facing_mutation_count=3,
            basic_loss_count=1,
            proline_glycine_gain_count=1,
        ),
    ]

    chosen, _nearest_distance = _choose_farthest_candidate(
        candidate_rows=rows,
        selected_rows=selected,
        sequence_by_id={
            "already_selected": "A" * 12,
            "overlapping_no_soft_warning": "A" * 11 + "C",
            "distinct_with_soft_warning": "A" * 10 + "CC",
        },
        mutation_tokens_by_id={
            "already_selected": frozenset({"A10G", "L20V"}),
            "overlapping_no_soft_warning": frozenset({"A10G", "L20V"}),
            "distinct_with_soft_warning": frozenset({"A30K", "L40R"}),
        },
        mutation_positions_by_id={
            "already_selected": frozenset({10, 20}),
            "overlapping_no_soft_warning": frozenset({10, 20}),
            "distinct_with_soft_warning": frozenset({30, 40}),
        },
    )

    assert chosen["candidate_id"] == "distinct_with_soft_warning"
