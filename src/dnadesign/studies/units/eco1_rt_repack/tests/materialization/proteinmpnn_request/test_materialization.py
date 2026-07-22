"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/proteinmpnn_request/test_materialization.py

ProteinMPNN request adapter tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.sampling import (
    validate_proteinmpnn_request_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.mask_set import materialize_mask_set
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.proteinmpnn_request import (
    materialize_proteinmpnn_request,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.thread_plan import materialize_thread_plan
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import ec86kit_source_artifacts_available, repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.mask_set._fixtures import (
    materialize_upstream_artifacts,
)

pytestmark = pytest.mark.skipif(
    not ec86kit_source_artifacts_available(),
    reason="requires sibling ec86kit structure-authority artifacts",
)


def test_proteinmpnn_request_materializer_writes_helper_compatible_sidecars(tmp_path: Path) -> None:
    materialize_upstream_artifacts(tmp_path)
    materialize_mask_set(repo_root=repo_root(), output_root=tmp_path)
    thread_result = materialize_thread_plan(repo_root=repo_root(), output_root=tmp_path)

    result = materialize_proteinmpnn_request(repo_root=repo_root(), output_root=tmp_path)

    plan = _load_yaml(thread_result.thread_plan_path)
    manifest = _load_yaml(result.request_manifest_path)
    parsed = _load_jsonl(result.parsed_pdbs_path)[0]
    assigned = _load_jsonl(result.assigned_chains_path)[0]
    fixed = _load_jsonl(result.fixed_positions_path)[0]

    assert result.chain_a_backbone_pdb_path.exists()
    assert manifest["schema_id"] == "proteinmpnn.fixed_backbone_request"
    assert manifest["status"] == "materialized"
    assert manifest["execution_status"] == "planned_not_run"
    assert manifest["proteinmpnn_design_chain"] == "A"
    assert manifest["omit_aas"] == ["C"]
    assert manifest["fallback_policy"] == "explicit_no_fallback"
    assert manifest["source_thread_plan"]["path"] == str(thread_result.thread_plan_path)
    assert manifest["proteinmpnn_position_basis"] == "chain_local_1_indexed_after_export"
    assert manifest["canonical_position_count"] == 309
    assert manifest["fixed_position_count"] == len(plan["fixed_positions"])
    assert manifest["mutable_position_count"] == len(plan["mutable_positions"])
    assert manifest["excluded_missing_backbone_positions"] == [1, 2, 312, 313, 314, 315, 316, 317, 318, 319, 320]

    assert parsed["name"] == "chain_a_backbone"
    assert parsed["num_of_chains"] == 1
    assert len(parsed["seq_chain_A"]) == 309
    assert parsed["seq"] == parsed["seq_chain_A"]
    assert set(parsed["coords_chain_A"]) == {"N_chain_A", "CA_chain_A", "C_chain_A", "O_chain_A"}
    assert len(parsed["coords_chain_A"]["CA_chain_A"]) == 309

    assert assigned == {"chain_a_backbone": [["A"], []]}
    assert set(fixed) == {"chain_a_backbone"}
    assert set(fixed["chain_a_backbone"]) == {"A"}
    assert len(fixed["chain_a_backbone"]["A"]) == len(plan["fixed_positions"])

    mapping = manifest["canonical_to_proteinmpnn_position"]
    assert mapping["3"] == 1
    assert mapping["311"] == 309
    assert "1" not in mapping
    assert "312" not in mapping
    expected_fixed_positions = sorted(mapping[str(position)] for position in plan["fixed_positions"])
    assert fixed["chain_a_backbone"]["A"] == expected_fixed_positions
    assert manifest["run_commands"][0]["argv"][:2] == ["python", "helper_scripts/parse_multiple_chains.py"]
    assert manifest["run_commands"][-1]["argv"][1] == "protein_mpnn_run.py"

    issues = validate_proteinmpnn_request_content(
        result.request_manifest_path,
        repo_root=repo_root(),
        output_root=tmp_path,
    )
    assert issues == []


def _load_yaml(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
