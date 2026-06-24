"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/contracts/structure/test_preprocessing.py

Structure-preprocessing manifest contract tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.structure import (
    validate_structure_preprocessing_manifest_content,
)


def test_structure_preprocessing_validator_accepts_hash_linked_manifest(tmp_path: Path) -> None:
    manifest_path = _write_preprocessing_contract_fixture(tmp_path)

    issues = validate_structure_preprocessing_manifest_content(
        manifest_path,
        repo_root=tmp_path,
        structure_root=tmp_path / "outputs/thread/eco1_rt_conservative_v1",
    )

    assert issues == []


def test_structure_preprocessing_validator_rejects_stale_upstream_hash(tmp_path: Path) -> None:
    manifest_path = _write_preprocessing_contract_fixture(tmp_path)
    (tmp_path / "docs/studies/eco1_rt_repack/workbench/provenance/structure-preprocessing.yaml").write_text(
        "preprocessing_id: changed\n",
        encoding="utf-8",
    )

    issues = validate_structure_preprocessing_manifest_content(
        manifest_path,
        repo_root=tmp_path,
        structure_root=tmp_path / "outputs/thread/eco1_rt_conservative_v1",
    )

    assert "eco1_rt.structure.structure_preprocessing_upstream_hash_mismatch" in {issue.check_id for issue in issues}


def test_structure_preprocessing_validator_rejects_implicit_dimerization_objective(tmp_path: Path) -> None:
    manifest_path = _write_preprocessing_contract_fixture(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["design_objectives"]["preserve_paired_protomer_dimerization"] = True
    _write_yaml(manifest_path, manifest)

    issues = validate_structure_preprocessing_manifest_content(
        manifest_path,
        repo_root=tmp_path,
        structure_root=tmp_path / "outputs/thread/eco1_rt_conservative_v1",
    )

    assert "eco1_rt.structure.structure_preprocessing_design_objective_mismatch" in {issue.check_id for issue in issues}


def _write_preprocessing_contract_fixture(root: Path) -> Path:
    docs_root = root / "docs/studies/eco1_rt_repack/workbench/provenance"
    output_root = root / "outputs/thread/eco1_rt_conservative_v1"
    docs_root.mkdir(parents=True)
    output_root.mkdir(parents=True)
    ec86kit_manifest = root / "ec86kit_manifest.yaml"
    ec86kit_model = root / "ec86kit_model.cif"
    ec86kit_manifest.write_text("model: ec86kit\n", encoding="utf-8")
    ec86kit_model.write_text("data_ec86kit\n", encoding="utf-8")

    structure_sources = {
        "selected_source": {
            "source_id": "ec86kit_7v9u_protomer1",
            "rt_chain_id": "A",
            "retained_context_chains": ["D", "E", "F"],
            "retained_context_policy": "retain_msdna_msrna_context_remove_effector_context",
            "ec86kit_manifest_ref": str(ec86kit_manifest),
            "ec86kit_model_ref": str(ec86kit_model),
            "ec86kit_manifest_sha256": _sha256(ec86kit_manifest),
            "ec86kit_model_sha256": _sha256(ec86kit_model),
        }
    }
    _write_yaml(docs_root / "structure-sources.yaml", structure_sources)
    _write_yaml(docs_root / "residue-numbering-policy.yaml", {"policy_id": "ec86kit_residue_map_v1"})
    design_objectives = {
        "preserve_monomeric_rt_msdna_msrna_context": True,
        "preserve_paired_protomer_dimerization": False,
        "paired_protomer_interface_policy": "not_mask_authoritative_unless_retained_context_manual_or_conserved",
        "pre_rt1_policy": "pre-RT1 residues may be designable if they pass retained-context gates",
    }
    _write_yaml(
        docs_root / "structure-preprocessing.yaml",
        {
            "preprocessing_id": "ec86kit_7v9u_protomer1_v1",
            "design_objectives": design_objectives,
        },
    )
    _write_yaml(output_root / "backbone_bundle.yaml", {"chain_inventory": _chain_inventory()})
    manifest = {
        "schema_id": "eco1_rt_repack.structure_preprocessing_manifest",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.structure_preprocessing_manifest",
        "status": "materialized",
        "created_by": "test",
        "created_at": "2026-06-22T00:00:00Z",
        "preprocessing_id": "ec86kit_7v9u_protomer1_v1",
        "raw_structure": {"source_ref": "pdb:7v9u"},
        "selected_protomer": {
            "source_id": "ec86kit_7v9u_protomer1",
            "protomer_id": 1,
            "rt_chain_id": "A",
            "retained_context_chains": ["D", "E", "F"],
            "retained_context_policy": "retain_msdna_msrna_context_remove_effector_context",
        },
        "design_objectives": design_objectives,
        "chain_inventory": _chain_inventory(),
        "acceptance_rules": ["selected_chains_must_match_structure_authority"],
        "upstream_artifact_hashes": {
            "structure_preprocessing_yaml": "sha256:" + _sha256(docs_root / "structure-preprocessing.yaml"),
            "structure_sources_yaml": "sha256:" + _sha256(docs_root / "structure-sources.yaml"),
            "residue_numbering_policy_yaml": "sha256:" + _sha256(docs_root / "residue-numbering-policy.yaml"),
            "backbone_bundle": "sha256:" + _sha256(output_root / "backbone_bundle.yaml"),
            "ec86kit_manifest": "sha256:" + _sha256(ec86kit_manifest),
            "ec86kit_model": "sha256:" + _sha256(ec86kit_model),
        },
        "numbering_policy": {"policy_id": "ec86kit_residue_map_v1"},
    }
    manifest_path = output_root / "structure_preprocessing_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    return manifest_path


def _chain_inventory() -> list[dict[str, str]]:
    return [
        {"chain_id": "A", "molecule_type": "protein", "thread_role": "design_backbone", "retention": "retained"},
        {"chain_id": "D", "molecule_type": "dna", "thread_role": "retained_context", "retention": "retained"},
        {"chain_id": "E", "molecule_type": "rna", "thread_role": "retained_context", "retention": "retained"},
        {"chain_id": "F", "molecule_type": "rna", "thread_role": "retained_context", "retention": "retained"},
    ]


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
