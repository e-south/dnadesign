"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/snapback/test_load.py

Load and schema tests for snapback explicit and solve specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.snapback.errors import SnapbackSpecError
from dnadesign.cruncher.snapback.load import load_snapback_solve_spec, load_snapback_spec


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _workspace_root(tmp_path: Path) -> Path:
    return tmp_path / "workspaces" / "demo_snapback"


def _explicit_path(tmp_path: Path) -> Path:
    return _workspace_root(tmp_path) / "configs" / "snapback" / "demo.snapback.yaml"


def _solve_path(tmp_path: Path) -> Path:
    return _workspace_root(tmp_path) / "configs" / "snapback" / "demo.snapback.solve.yaml"


def _explicit_payload() -> dict[str, object]:
    return {
        "snapback": {
            "schema_version": 2,
            "contract": "single_nick_snapback_v2",
            "name": "demo_snapback",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": "CCTCAGCA",
                "protected_region": {"start": 0, "end": 8},
                "pre_nick_duplex_window": {"start": 0, "end": 8},
            }
        },
        "design": {
            "nickase": {
                "variant_id": "Nt.Bpu10I",
                "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
            },
            "orientation_policy": {
                "normalize_to_top_strand_nick": True,
                "release_direction": "left_to_right_from_nick",
            },
            "single_nick_goal": {"nick_boundary_window": {"min": 2, "max": 2}},
            "topology": {
                "retained_homology_window": {"start": 2, "end": 6},
                "cap_sequence": "T",
                "foldback_arm": "CTGA",
                "homology_policy": {"max_mismatches": 0, "min_paired_bp": 4, "max_paired_bp": 4},
            },
            "constraints": {
                "terminal_ligatable_duplex_bp": {"min": 4, "max": 4},
                "max_uninterrupted_duplex_bp": 4,
                "max_added_nt": 5,
                "forbid_additional_target_strand_nicks": False,
                "forbid_any_additional_nicks": False,
            },
            "sequence_quality": {
                "gc_fraction": {"min": 0.25, "max": 0.75},
                "max_homopolymer_run": 2,
            },
        },
        "output": {"run_dir": "outputs/design", "emit_visual_contracts": True, "emit_baserender_jobs": True},
    }


def _solve_payload() -> dict[str, object]:
    return {
        "snapback_solve": {
            "schema_version": 3,
            "contract": "single_nick_snapback_solve_v3",
            "name": "demo_snapback_solve",
        },
        "input": {
            "canonical_top_strand": {
                "sequence": "CCTCAGCA",
                "protected_region": {"start": 0, "end": 8},
                "pre_nick_duplex_window": {"start": 0, "end": 8},
            }
        },
        "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
        "orientation_policy": {"normalize_to_top_strand_nick": True},
        "goal": {"nick_boundary_window": {"min": 2, "max": 2}},
        "search": {
            "retained_homology_length": {"min": 4, "max": 4},
            "max_added_nt": 5,
            "max_mismatches": 0,
            "max_enumerated_candidates": 64,
            "max_search_nodes": 64,
            "max_hits": 4,
            "materialize_top_k": 2,
        },
        "constraints": {
            "terminal_ligatable_duplex_bp": {"min": 4, "max": 4},
            "max_uninterrupted_duplex_bp": 4,
            "forbid_additional_target_strand_nicks": False,
            "forbid_any_additional_nicks": False,
        },
        "sequence_quality": {
            "gc_fraction": {"min": 0.0, "max": 0.75},
            "max_homopolymer_run": 3,
        },
        "output": {
            "run_dir": "outputs/solve",
            "emit_visual_contracts": True,
            "emit_baserender_jobs": True,
        },
    }


def test_load_snapback_spec_accepts_valid_v2_explicit_contract(tmp_path: Path) -> None:
    spec_path = _write_yaml(_explicit_path(tmp_path), _explicit_payload())

    spec, resolved, workspace_root = load_snapback_spec(spec_path)

    assert resolved == spec_path.resolve()
    assert workspace_root == _workspace_root(tmp_path).resolve()
    assert spec.snapback.contract == "single_nick_snapback_v2"
    assert spec.name == "demo_snapback"
    assert spec.designed_sequence == "CCTCAGCATCTGA"
    assert spec.added_nt == 5


def test_load_snapback_spec_defaults_homology_floor_to_three_when_omitted(tmp_path: Path) -> None:
    payload = _explicit_payload()
    payload["design"]["topology"]["homology_policy"] = {"max_mismatches": 0}
    spec_path = _write_yaml(_explicit_path(tmp_path), payload)

    spec, _resolved, _workspace_root = load_snapback_spec(spec_path)

    assert spec.design.topology.homology_policy.min_paired_bp == 3
    assert spec.design.topology.homology_policy.max_paired_bp >= 3


def test_load_snapback_spec_rejects_retained_window_outside_input_sequence(tmp_path: Path) -> None:
    payload = _explicit_payload()
    payload["design"]["topology"]["retained_homology_window"] = {"start": 8, "end": 20}
    spec_path = _write_yaml(_explicit_path(tmp_path), payload)

    with pytest.raises(SnapbackSpecError, match="retained_homology_window must stay inside"):
        load_snapback_spec(spec_path)


def test_load_snapback_spec_rejects_blank_output_run_dir(tmp_path: Path) -> None:
    payload = _explicit_payload()
    payload["output"]["run_dir"] = ""
    spec_path = _write_yaml(_explicit_path(tmp_path), payload)

    with pytest.raises(SnapbackSpecError, match="output.run_dir must be non-empty"):
        load_snapback_spec(spec_path)


def test_load_snapback_spec_rejects_baserender_jobs_without_visual_contracts(tmp_path: Path) -> None:
    payload = _explicit_payload()
    payload["output"]["emit_visual_contracts"] = False
    payload["output"]["emit_baserender_jobs"] = True
    spec_path = _write_yaml(_explicit_path(tmp_path), payload)

    with pytest.raises(SnapbackSpecError, match="emit_baserender_jobs requires output.emit_visual_contracts"):
        load_snapback_spec(spec_path)


def test_load_snapback_spec_accepts_preset_only_catalog_sources(tmp_path: Path) -> None:
    payload = _explicit_payload()
    payload["design"]["nickase"] = {
        "variant_id": "Nt.AlwI",
        "catalog": {"preset": "neb_nicking_v1"},
    }
    payload["input"]["canonical_top_strand"]["sequence"] = "GGATCAGTC"
    payload["input"]["canonical_top_strand"]["protected_region"] = {"start": 0, "end": 9}
    payload["input"]["canonical_top_strand"]["pre_nick_duplex_window"] = {"start": 0, "end": 9}
    payload["design"]["single_nick_goal"]["nick_boundary_window"] = {"min": 4, "max": 4}
    payload["design"]["topology"]["retained_homology_window"] = {"start": 5, "end": 9}
    spec_path = _write_yaml(_explicit_path(tmp_path), payload)

    spec, _resolved, _workspace_root = load_snapback_spec(spec_path)

    assert spec.design.nickase.catalog.preset == "neb_nicking_v1"
    assert spec.design.nickase.catalog.additional_presets == []
    assert spec.design.nickase.catalog.additional_paths == []


def test_load_snapback_solve_spec_accepts_additional_presets(tmp_path: Path) -> None:
    payload = _solve_payload()
    payload["catalog"] = {
        "preset": "neb_nicking_v1",
        "additional_presets": ["thermo_nicking_v1"],
    }
    spec_path = _write_yaml(_solve_path(tmp_path), payload)

    spec, _resolved, _workspace_root = load_snapback_solve_spec(spec_path)

    assert spec.catalog.resolved_preset_ids() == ["neb_nicking_v1", "thermo_nicking_v1"]


def test_load_snapback_solve_spec_accepts_valid_v3_solve_contract(tmp_path: Path) -> None:
    spec_path = _write_yaml(_solve_path(tmp_path), _solve_payload())

    spec, resolved, workspace_root = load_snapback_solve_spec(spec_path)

    assert resolved == spec_path.resolve()
    assert workspace_root == _workspace_root(tmp_path).resolve()
    assert spec.snapback_solve.contract == "single_nick_snapback_solve_v3"
    assert spec.search.materialize_top_k == 2


def test_load_snapback_solve_spec_defaults_compact_ranges_when_omitted(tmp_path: Path) -> None:
    payload = _solve_payload()
    payload.pop("goal")
    payload["search"].pop("retained_homology_length")
    payload["search"]["min_paired_bp"] = 3
    payload["constraints"].pop("terminal_ligatable_duplex_bp")
    payload["constraints"].pop("max_uninterrupted_duplex_bp")
    spec_path = _write_yaml(_solve_path(tmp_path), payload)

    spec, _resolved, _workspace_root = load_snapback_solve_spec(spec_path)
    resolved = spec.resolved_search_space()

    assert resolved.nick_boundary_window.min == 0
    assert resolved.nick_boundary_window.max == 8
    assert resolved.retained_homology_length.min == 3
    assert resolved.retained_homology_length.max == 8
    assert resolved.terminal_ligatable_duplex_bp.min == 3
    assert resolved.terminal_ligatable_duplex_bp.max == 8
    assert resolved.max_uninterrupted_duplex_bp == 8


def test_load_snapback_solve_spec_rejects_materialize_top_k_above_max_hits(tmp_path: Path) -> None:
    payload = _solve_payload()
    payload["search"]["materialize_top_k"] = 5
    payload["search"]["max_hits"] = 4
    spec_path = _write_yaml(_solve_path(tmp_path), payload)

    with pytest.raises(SnapbackSpecError, match="materialize_top_k must be <= max_hits"):
        load_snapback_solve_spec(spec_path)


def test_load_snapback_solve_spec_rejects_blank_output_run_dir(tmp_path: Path) -> None:
    payload = _solve_payload()
    payload["output"]["run_dir"] = ""
    spec_path = _write_yaml(_solve_path(tmp_path), payload)

    with pytest.raises(SnapbackSpecError, match="output.run_dir must be non-empty"):
        load_snapback_solve_spec(spec_path)


def test_load_snapback_solve_spec_rejects_baserender_jobs_without_visual_contracts(tmp_path: Path) -> None:
    payload = _solve_payload()
    payload["output"]["emit_visual_contracts"] = False
    payload["output"]["emit_baserender_jobs"] = True
    spec_path = _write_yaml(_solve_path(tmp_path), payload)

    with pytest.raises(SnapbackSpecError, match="emit_baserender_jobs requires output.emit_visual_contracts"):
        load_snapback_solve_spec(spec_path)
