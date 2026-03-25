"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cassette/test_load.py

Load/normalization tests for cassette specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.cassette.errors import CassetteSpecError
from dnadesign.cruncher.cassette.load import load_cassette_spec


def _write_spec(tmp_path: Path, cassette_payload: dict[str, object]) -> Path:
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump({"cassette": cassette_payload}), encoding="utf-8")
    return spec_path


def test_load_cassette_spec_normalizes_v2_fields(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        {
            "schema_version": 2,
            "name": "demo_hairpin",
            "topology": {
                "stem5p_arm": "AACGAT",
                "loop": "TT",
                "stem3p_arm_mode": "derived_reverse_complement",
            },
            "construct_context": {"left_flank": "AA", "right_flank": "TT"},
            "nicking": {
                "target_strand": "primary",
                "left": {"nickase": "Nt.demo", "nick_window": {"start": 0, "end": 2}},
                "right": {"nickase": "Nb.demo", "nick_window": {"start": 12, "end": 14}},
                "require_exactly_two_intended_nicks": True,
                "bounded_segment_length": {"min": 10, "max": 20},
            },
            "site_policy": {
                "forbid_additional_designated_strand_nicks": True,
                "scan_scope": "catalog",
            },
            "hairpin_validation": {
                "require_topological_hairpin": True,
                "require_energetic_hairpin": False,
            },
            "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
            "output": {"run_dir": "outputs/cassettes", "write_render_contract": True},
        },
    )

    spec, _resolved, _workspace_root = load_cassette_spec(spec_path)

    assert spec.schema_version == 2
    assert spec.construct_context.left_flank == "AA"
    assert spec.construct_context.right_flank == "TT"
    assert spec.nicking.target_strand == "primary"
    assert spec.site_policy.scan_scope == "catalog"
    assert spec.topology.stem3p_arm_mode == "derived_reverse_complement"


def test_load_cassette_spec_rejects_alias_conflicts(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        {
            "schema_version": 2,
            "name": "demo_hairpin",
            "topology": {
                "stem5p_arm": "AACGAT",
                "loop": "TT",
                "stem3p_arm_mode": "derived_reverse_complement",
            },
            "construct_context": {"left_flank": "", "right_flank": ""},
            "duplex_context": {"upstream": "", "downstream": ""},
            "nicking": {
                "target_strand": "primary",
                "left": {"nickase": "Nt.demo", "nick_window": {"start": 0, "end": 2}},
                "right": {"nickase": "Nb.demo", "nick_window": {"start": 12, "end": 14}},
            },
            "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
        },
    )

    with pytest.raises(CassetteSpecError, match="SCHEMA_ALIAS_CONFLICT"):
        load_cassette_spec(spec_path)


def test_load_cassette_spec_normalizes_legacy_nicking_aliases(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        {
            "schema_version": 2,
            "name": "demo_hairpin",
            "topology": {
                "stem5p_arm": "AACGAT",
                "loop": "TT",
                "stem3p_arm_mode": "fixed",
            },
            "duplex_context": {"upstream": "AA", "downstream": "TT"},
            "nicking": {
                "designated_strand": "complement_strand",
                "left": {"nickase": "Nt.demo", "nick_window": {"start": 0, "end": 2}},
                "right": {"nickase": "Nb.demo", "nick_window": {"start": 12, "end": 14}},
                "forbid_additional_designated_strand_nicks": True,
            },
            "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
        },
    )

    spec, _resolved, _workspace_root = load_cassette_spec(spec_path)

    assert spec.construct_context.left_flank == "AA"
    assert spec.construct_context.right_flank == "TT"
    assert spec.nicking.target_strand == "complement"
    assert spec.site_policy.forbid_additional_designated_strand_nicks is True
    assert spec.site_policy.scan_scope == "requested_variants"
    assert spec.topology.stem3p_arm_mode == "derived_reverse_complement"


def test_load_cassette_spec_rejects_non_tracer_bullet_intended_nick_mode(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        {
            "schema_version": 2,
            "name": "demo_hairpin",
            "topology": {
                "stem5p_arm": "AACGAT",
                "loop": "TT",
                "stem3p_arm_mode": "derived_reverse_complement",
            },
            "construct_context": {"left_flank": "", "right_flank": ""},
            "nicking": {
                "target_strand": "primary",
                "left": {"nickase": "Nt.demo", "nick_window": {"start": 0, "end": 2}},
                "right": {"nickase": "Nb.demo", "nick_window": {"start": 12, "end": 14}},
                "require_exactly_two_intended_nicks": False,
            },
            "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
        },
    )

    with pytest.raises(CassetteSpecError, match="UNSUPPORTED_INTENDED_NICK_COUNT_MODE"):
        load_cassette_spec(spec_path)


def test_load_cassette_spec_rejects_non_topological_hairpin_mode(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        {
            "schema_version": 2,
            "name": "demo_hairpin",
            "topology": {
                "stem5p_arm": "AACGAT",
                "loop": "TT",
                "stem3p_arm_mode": "derived_reverse_complement",
            },
            "construct_context": {"left_flank": "", "right_flank": ""},
            "nicking": {
                "target_strand": "primary",
                "left": {"nickase": "Nt.demo", "nick_window": {"start": 0, "end": 2}},
                "right": {"nickase": "Nb.demo", "nick_window": {"start": 12, "end": 14}},
            },
            "hairpin_validation": {"require_topological_hairpin": False},
            "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
        },
    )

    with pytest.raises(CassetteSpecError, match="UNSUPPORTED_TOPOLOGICAL_HAIRPIN_MODE"):
        load_cassette_spec(spec_path)


def test_load_cassette_spec_rejects_energetic_hairpin_flag_until_supported(tmp_path: Path) -> None:
    spec_path = _write_spec(
        tmp_path,
        {
            "schema_version": 2,
            "name": "demo_hairpin",
            "topology": {
                "stem5p_arm": "AACGAT",
                "loop": "TT",
                "stem3p_arm_mode": "derived_reverse_complement",
            },
            "construct_context": {"left_flank": "", "right_flank": ""},
            "nicking": {
                "target_strand": "primary",
                "left": {"nickase": "Nt.demo", "nick_window": {"start": 0, "end": 2}},
                "right": {"nickase": "Nb.demo", "nick_window": {"start": 12, "end": 14}},
            },
            "hairpin_validation": {"require_energetic_hairpin": True},
            "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
        },
    )

    with pytest.raises(CassetteSpecError, match="ENERGETIC_HAIRPIN_VALIDATION_NOT_SUPPORTED"):
        load_cassette_spec(spec_path)
