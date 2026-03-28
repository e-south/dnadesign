"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/contracts/test_runtime_lane_contracts.py

Public runtime-lane contract tests for infer study config surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.infer.contracts import resolve_infer_runtime_lane_contracts


def test_resolve_infer_runtime_lane_contracts_prefers_requested_family(tmp_path: Path) -> None:
    infer_config_paths = {
        "anchor_only_7b": tmp_path / "config.anchor_only.evo2_7b.yaml",
        "anchor_plus_template_7b": tmp_path / "config.anchor_plus_template.evo2_7b.yaml",
        "anchor_only_20b": tmp_path / "config.anchor_only.evo2_20b.yaml",
        "anchor_plus_template_20b": tmp_path / "config.anchor_plus_template.evo2_20b.yaml",
    }

    contracts = resolve_infer_runtime_lane_contracts(
        infer_config_paths,
        preferred_model_family="evo2_20b",
    )

    assert [contract.runtime_label for contract in contracts] == [
        "anchor_only_20b",
        "anchor_plus_template_20b",
        "anchor_only_7b",
        "anchor_plus_template_7b",
    ]
    assert [contract.phase_id for contract in contracts] == [
        "infer_anchor_only_20b",
        "infer_anchor_plus_template_20b",
        "infer_anchor_only_7b",
        "infer_anchor_plus_template_7b",
    ]


def test_resolve_infer_runtime_lane_contracts_ignores_full_lane_configs(tmp_path: Path) -> None:
    infer_config_paths = {
        "full_lane_set_20b": tmp_path / "config.full_lane_set.evo2_20b.yaml",
        "anchor_only_20b": tmp_path / "config.anchor_only.evo2_20b.yaml",
    }

    contracts = resolve_infer_runtime_lane_contracts(infer_config_paths)

    assert [contract.runtime_label for contract in contracts] == ["anchor_only_20b"]


def test_resolve_infer_runtime_lane_contracts_uses_config_name_for_runtime_identity(tmp_path: Path) -> None:
    infer_config_paths = {
        "preferred_lane": tmp_path / "config.anchor_plus_template.evo2_20b.yaml",
    }

    contracts = resolve_infer_runtime_lane_contracts(infer_config_paths)

    assert len(contracts) == 1
    assert contracts[0].config_label == "preferred_lane"
    assert contracts[0].runtime_label == "anchor_plus_template_20b"
    assert contracts[0].lane_kind == "anchor_plus_template"
    assert contracts[0].model_family == "evo2_20b"
