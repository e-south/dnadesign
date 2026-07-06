"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/selection_readiness/test_sequence_export_contract.py

Sequence-export contract tests for Eco1 RT panel selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness import (
    materialize_selection_readiness,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.selection_readiness._fixtures import (
    sequence,
    write_inputs,
)


def test_handoff_sequence_csv_declares_protein_scope_and_non_dna_status(tmp_path: Path) -> None:
    class_root = tmp_path / "outputs/thread/design_classes"
    selection_root = class_root / "selection"
    source_root = tmp_path / "outputs/thread"
    write_inputs(class_root, source_root)

    result = materialize_selection_readiness(
        repo_root=tmp_path,
        output_root=class_root,
        source_output_root=source_root,
        selection_root=selection_root,
        created_at="2026-07-02T00:00:00Z",
    )

    panel = pq.read_table(result.candidate_selection_panel_path).to_pylist()
    with result.candidate_handoff_sequence_csv_path.open(encoding="utf-8", newline="") as handle:
        handoff_sequence_rows = list(csv.DictReader(handle))
    assert len(handoff_sequence_rows) == len(panel)
    assert {row["candidate_id"] for row in handoff_sequence_rows} == {row["candidate_id"] for row in panel}
    assert all(
        row["protein_sequence"] == sequence(int(row["candidate_id"].split("_")[-1])) for row in handoff_sequence_rows
    )
    assert {row["dna_design_status"] for row in handoff_sequence_rows} == {"not_materialized"}
    assert {row["sequence_scope"] for row in handoff_sequence_rows} == {"mapped_rt_chain_protein"}
    assert {row["mapped_rt_chain_length"] for row in handoff_sequence_rows} == {"64"}
    assert {row["canonical_rt_length"] for row in handoff_sequence_rows} == {"320"}
    assert {row["canonical_sequence_status"] for row in handoff_sequence_rows} == {"not_exported_in_this_slice"}
    assert {row["dna_sequence_status"] for row in handoff_sequence_rows} == {"not_dna"}
    assert {row["codon_optimization_status"] for row in handoff_sequence_rows} == {"not_codon_optimized"}
    assert {row["restriction_screen_status"] for row in handoff_sequence_rows} == {"not_screened"}
    assert all(row["handoff_scope_note"].startswith("RT protein sequence only") for row in handoff_sequence_rows)
    assert all(int(row["protein_sequence_length"]) == len(row["protein_sequence"]) for row in handoff_sequence_rows)
    assert all(
        row["protein_sequence_sha256"]
        == "sha256:" + hashlib.sha256(row["protein_sequence"].encode("utf-8")).hexdigest()
        for row in handoff_sequence_rows
    )
    assert all(row["source_candidate_pool_sha256"].startswith("sha256:") for row in handoff_sequence_rows)
    assert all(row["source_panel_sha256"].startswith("sha256:") for row in handoff_sequence_rows)
