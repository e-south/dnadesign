"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/generation_policies/test_request_materialization.py

Generation-policy request materialization tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies import (
    GENERATION_POLICY_VERSION,
    PRIMARY_POLICY_IDS,
    materialize_generation_policies,
    materialize_generation_policy_requests,
    request_materialization,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.generation_policies.cli import (
    main as generation_policy_main,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import require_ec86kit_source_artifacts
from dnadesign.thread.adapters.proteinmpnn import validate_request_manifest


def test_generation_policy_requests_materialize_one_request_per_complete_policy(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    policy_result = materialize_generation_policies(repo_root=Path.cwd(), output_root=tmp_path)
    request_result = materialize_generation_policy_requests(repo_root=Path.cwd(), generation_policy_root=tmp_path)
    positions = pq.read_table(policy_result.positions_path).to_pylist()

    assert {path.parent.parent.name for path in request_result.request_manifest_paths} == set(PRIMARY_POLICY_IDS)
    assert request_result.policy_manifest_path == policy_result.manifest_path
    assert request_result.request_manifest_paths

    for manifest_path in request_result.request_manifest_paths:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        policy_id = manifest["policy_id"]
        open_positions = {
            int(row["eco1_position"]) for row in positions if row["policy_id"] == policy_id and row["is_open_position"]
        }
        mapped_positions = {
            int(row["eco1_position"]) for row in positions if row["policy_id"] == policy_id and row["is_mapped"]
        }
        fixed_positions = set(manifest["canonical_fixed_positions"])

        assert manifest["schema_id"] == "proteinmpnn.fixed_backbone_request"
        assert manifest["generation_policy_version"] == GENERATION_POLICY_VERSION
        assert manifest["policy_version"] == GENERATION_POLICY_VERSION
        assert manifest["policy_manifest_hash"].startswith("sha256:")
        assert manifest["requested_variants"] == 336
        assert manifest["expected_sample_count"] == 336
        assert manifest["mask_policy_id"] is None
        assert "design_class_id" not in manifest
        assert manifest["canonical_open_positions"] == sorted(open_positions)
        assert fixed_positions == mapped_positions - open_positions
        assert fixed_positions.isdisjoint(open_positions)
        assert manifest["mutable_position_count"] == len(open_positions)
        assert validate_request_manifest(manifest_path) == []

        if policy_id == "distal_scaffold_repack_v1":
            assert manifest["alphabet_enforcement_modes"] == ["upstream_omit_AAs_C"]
            assert "omit_AA_jsonl" not in manifest["sidecar_paths"]
        else:
            assert "upstream_omit_AA_jsonl" in manifest["alphabet_enforcement_modes"]
            assert "omit_AA_jsonl" in manifest["sidecar_paths"]
            omit_path = manifest_path.parent / Path(manifest["sidecar_paths"]["omit_AA_jsonl"]).name
            omit_payload = json.loads(omit_path.read_text(encoding="utf-8"))
            assert set(omit_payload) == {"chain_a_backbone"}
            assert set(omit_payload["chain_a_backbone"]) == {"A"}
            omit_groups = omit_payload["chain_a_backbone"]["A"]
            assert omit_groups
            omitted_by_position = {
                int(position): set(aa_text) for positions, aa_text in omit_groups for position in positions
            }
            near_open_positions = {
                int(row["eco1_position"])
                for row in positions
                if row["policy_id"] == policy_id and row["is_open_position"] and row["is_near_region_gt5_le10a"]
            }
            near_open_wt_by_position = {
                int(row["eco1_position"]): str(row["wt_aa"])
                for row in positions
                if row["policy_id"] == policy_id and row["is_open_position"] and row["is_near_region_gt5_le10a"]
            }
            near_open_mpnn_positions = {
                int(manifest["canonical_to_proteinmpnn_position"][str(position)]) for position in near_open_positions
            }
            assert near_open_mpnn_positions
            assert near_open_mpnn_positions <= set(omitted_by_position)
            for canonical_position, wt_aa in near_open_wt_by_position.items():
                mpnn_position = int(manifest["canonical_to_proteinmpnn_position"][str(canonical_position)])
                assert {"D", "E"} - {wt_aa} <= omitted_by_position[mpnn_position]
            for command in manifest["run_commands"]:
                if command["name"].startswith("protein_mpnn_run_seed_"):
                    assert "--omit_AA_jsonl" in command["argv"]


def test_generation_policy_request_materialization_rejects_legacy_manifest_policy_id(tmp_path: Path) -> None:
    manifest_path = tmp_path / "generation_policy_manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_id": "eco1_rt.generation_policy_manifest",
                "generation_policy_version": GENERATION_POLICY_VERSION,
                "policy_manifest_hash": "sha256:test",
                "position_manifest_path": str(tmp_path / "generation_policy_positions.parquet"),
                "alphabet_manifest_path": str(tmp_path / "generation_policy_alphabets.parquet"),
                "generation_policies": [
                    {
                        "policy_id": "eco1_rt_clade9_plurality25_contact8a_v1",
                        "policy_version": GENERATION_POLICY_VERSION,
                        "requested_variants": 336,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="legacy design-class id"):
        materialize_generation_policy_requests(repo_root=Path.cwd(), generation_policy_root=tmp_path)


def test_generation_policy_request_materializer_does_not_import_design_class_specs() -> None:
    source = Path(request_materialization.__file__).read_text(encoding="utf-8")

    assert "design_classes.specs" not in source


def test_generation_policy_cli_materializes_policy_and_request_roots(tmp_path: Path) -> None:
    require_ec86kit_source_artifacts()
    exit_code = generation_policy_main(["--repo-root", str(Path.cwd()), "--output-root", str(tmp_path), "all"])

    assert exit_code == 0
    assert (tmp_path / "generation_policy_manifest.yaml").exists()
    for policy_id in PRIMARY_POLICY_IDS:
        assert (tmp_path / policy_id / "proteinmpnn_request" / "request_manifest.yaml").exists()


def test_bu_scc_generation_policy_job_template_is_policy_first() -> None:
    path = Path("docs/bu-scc/jobs/eco1-proteinmpnn-generation-policy.qsub")

    text = path.read_text(encoding="utf-8")

    assert "ECO1_GENERATION_POLICIES_ROOT" in text
    assert "GENERATION_POLICY_ID" in text
    assert "generation_policies_v3" in text
    assert "materialization.generation_policies" in text
    assert "design_classes" not in text
    assert "DESIGN_CLASS_ID" not in text
