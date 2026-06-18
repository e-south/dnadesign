"""
--------------------------------------------------------------------------------
dnadesign
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
            "output": {
                "run_dir": "outputs/cassettes",
                "emit_visual_contracts": True,
                "emit_baserender_jobs": True,
            },
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


def test_load_cassette_spec_rejects_baserender_jobs_without_visual_contracts(tmp_path: Path) -> None:
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
            "catalog": {"path": "inputs/nickases/demo.nickases.yaml"},
            "output": {
                "emit_visual_contracts": False,
                "emit_baserender_jobs": True,
            },
        },
    )

    message = "output.emit_baserender_jobs requires output.emit_visual_contracts=true"
    with pytest.raises(CassetteSpecError, match=message):
        load_cassette_spec(spec_path)


def test_load_cassette_solve_spec_rejects_baserender_jobs_without_visual_contracts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_cassette"
    spec_path = workspace / "configs" / "cassettes" / "demo_hairpin.cassette.solve.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "cassette_solve": {
                    "schema_version": 1,
                    "topology": {
                        "stem5p_arm_pattern": "NNNNNCCTCAGC",
                        "loop_pattern": "TTT",
                    },
                    "construct_context": {
                        "left_flank": "",
                        "right_flank": "",
                        "evaluation_scope": "cassette_plus_flanks",
                    },
                    "nick_goal": {
                        "target_strand": "primary",
                        "left_nick_window": {"start": 7, "end": 7},
                        "right_nick_window": {"start": 17, "end": 17},
                        "bounded_segment_length": {"min": 10, "max": 10},
                    },
                    "assignment_policy": {
                        "allowed_left_variant_ids": ["Nt.BbvCI"],
                        "allowed_right_variant_ids": ["Nb.BbvCI"],
                        "forbidden_intended_variant_ids": [],
                        "forbidden_intended_specificity_ids": [],
                        "allow_same_variant": True,
                        "allow_same_specificity_opposite_variant": True,
                    },
                    "site_blacklist": {
                        "forbidden_any_site_specificity_ids": [],
                        "forbidden_unintended_site_specificity_ids": [],
                        "forbidden_any_site_variant_ids": [],
                        "scope": "evaluation_context",
                    },
                    "sequence_blacklist": {
                        "forbidden_literals": [],
                        "forbidden_iupac_motifs": [],
                        "forbid_reverse_complements": True,
                        "scope": "evaluation_context",
                    },
                    "sequence_quality": {},
                    "catalog": {"preset": "neb_nicking_v1", "additional_paths": []},
                    "search": {
                        "max_hits": 3,
                        "max_enumerated_candidates": 256,
                        "selection": {
                            "policy": "greedy_hamming",
                            "pool_size": 16,
                            "distance_metric": "hamming",
                            "min_pairwise_distance": 2,
                        },
                        "bounded_segment_target": 10,
                        "gc_target": 0.5,
                        "materialize_top_k": 2,
                    },
                    "output": {
                        "emit_visual_contracts": False,
                        "emit_baserender_jobs": True,
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    from dnadesign.cruncher.cassette.load import load_cassette_solve_spec

    message = "output.emit_baserender_jobs requires output.emit_visual_contracts=true"
    with pytest.raises(CassetteSpecError, match=message):
        load_cassette_solve_spec(spec_path)
