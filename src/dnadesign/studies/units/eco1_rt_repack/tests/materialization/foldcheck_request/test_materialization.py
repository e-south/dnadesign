"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/foldcheck_request/test_materialization.py

Eco1 fold-check request materialization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_request import (
    materialize_foldcheck_request,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.foldcheck_request._fixtures import (
    write_minimal_foldcheck_inputs,
)


def test_foldcheck_request_materializes_wt_and_candidate_fasta(tmp_path: Path) -> None:
    write_minimal_foldcheck_inputs(tmp_path)

    result = materialize_foldcheck_request(repo_root=Path.cwd(), output_root=tmp_path)

    manifest = yaml.safe_load(result.request_manifest_path.read_text(encoding="utf-8"))
    fasta_text = result.input_fasta_path.read_text(encoding="utf-8")
    assert manifest["backend_kind"] == "colabfold"
    assert manifest["execution_status"] == "planned_not_run"
    assert manifest["sequence_count"] == 2
    assert manifest["wt_sequence_id"] == "wild_type"
    assert ">wild_type\n" in fasta_text
    assert ">thread_candidate_test\n" in fasta_text
    assert "AAAE" in fasta_text
    assert all(row["length"] == 320 for row in manifest["sequences"])
