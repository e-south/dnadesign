"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/yiu/test_load.py

Strict-load contracts for YIU workflow specs.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.yiu.load import load_yiu_solve_spec, load_yiu_spec, resolve_base_spec_path_for_yiu_solve_spec


def _base_yiu_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol": "yiu_v1",
        "name": "demo_yiu",
        "source_oligo": {
            "sequence": "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACGTTTTT",
            "primer_sites": [
                {"id": "fwd_primer", "start": 0, "end": 4, "strand": "primary"},
                {"id": "rev_primer", "start": 36, "end": 40, "strand": "complement"},
            ],
            "restriction_sites": [
                {
                    "id": "left_digest",
                    "enzyme": "BsaI",
                    "recognition_sequence": "GGTCTC",
                    "start": 4,
                    "orientation": "forward",
                    "top_cut_offset": 6,
                    "bottom_cut_offset": 10,
                },
                {
                    "id": "right_digest",
                    "enzyme": "BsaI",
                    "recognition_sequence": "GGTCTC",
                    "start": 26,
                    "orientation": "forward",
                    "top_cut_offset": 6,
                    "bottom_cut_offset": 10,
                },
            ],
            "nickase_sites": [
                {
                    "id": "nick_1",
                    "enzyme": "Nt.Mock",
                    "recognition_sequence": "GGGG",
                    "start": 18,
                    "orientation": "forward",
                    "top_cut_offset": 2,
                }
            ],
            "payload_windows": [
                {"id": "left_half", "start": 14, "end": 18},
                {"id": "right_half", "start": 22, "end": 26},
            ],
            "homology_windows": [
                {"id": "left_fold", "start": 14, "end": 18},
                {"id": "right_fold", "start": 14, "end": 18},
            ],
            "retained_regions": [
                {"id": "retained_left", "start": 14, "end": 18},
                {"id": "retained_right", "start": 22, "end": 26},
            ],
            "sacrificial_regions": [{"id": "sacrificial_center", "start": 18, "end": 22}],
        },
        "step_graph": {
            "steps": [
                {
                    "kind": "pcr",
                    "id": "pcr_linear_duplex",
                    "forward_primer_site": "fwd_primer",
                    "reverse_primer_site": "rev_primer",
                },
                {
                    "kind": "restriction_digest",
                    "id": "digested_linear_duplex",
                    "left_site": "left_digest",
                    "right_site": "right_digest",
                    "expected_left_overhang": "ACGT",
                    "expected_right_overhang": "ACGT",
                },
                {
                    "kind": "circularization",
                    "id": "circularization_candidate",
                    "compatibility": "exact_complement",
                },
                {"kind": "exonuclease_selection", "id": "post_exonuclease_enriched_pool"},
                {
                    "kind": "nickase_digest",
                    "id": "post_nickase_fragmentation",
                    "site_ids": ["nick_1"],
                    "sacrificial_region_ids": ["sacrificial_center"],
                    "retained_region_ids": ["retained_left", "retained_right"],
                },
                {"kind": "size_selection", "id": "post_size_selection"},
                {
                    "kind": "foldback",
                    "id": "foldback_or_cap_intermediate",
                    "left_homology_window": "left_fold",
                    "right_homology_window": "right_fold",
                    "min_complementary_bases": 4,
                },
                {
                    "kind": "adapter_ligation",
                    "id": "y_adapter_ligated_product",
                    "adapter_sequence": "AGATCGGA",
                },
                {
                    "kind": "amplification",
                    "id": "downstream_amplifiable_product",
                    "forward_primer_requirement": "AGAT",
                    "reverse_primer_requirement": "CCGG",
                },
            ]
        },
        "payload_goal": {
            "assembled_payload": "TTAACCGG",
            "left_half_ref": "left_half",
            "right_half_ref": "right_half",
            "junction_rule": "contiguous_after_ligation",
        },
        "cleanup_policy": {
            "linear_depletion": {"enabled": True, "enzyme": "T5 exonuclease"},
            "size_selection": {
                "max_retained_sacrificial_fragment_nt": 4,
                "min_retained_product_nt": 8,
            },
        },
        "adapter_policy": {
            "adapter_sequence": "AGATCGGA",
            "primer_binding_requirements": [
                {"id": "amp_fwd", "sequence": "AGAT"},
                {"id": "amp_rev", "sequence": "CCGG"},
            ],
        },
        "output": {"run_dir": "outputs/yiu/explicit", "emit_view_contracts": True},
    }


def _base_yiu_v2_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "family": "yiu",
        "protocol_template": "msd_hop_retron_eco1_v1",
        "workflow_scope": "core_insert_generation",
        "name": "demo_yiu_v2",
        "source_oligo": {
            "sequence": "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACGTTTTT",
            "annotations": {
                "primer_binding_cores": [
                    {"id": "source_fwd_core", "start": 0, "end": 4, "strand": "primary"},
                    {"id": "source_rev_core", "start": 36, "end": 40, "strand": "complement"},
                ],
                "primer_tails": [
                    {"id": "source_fwd_tail", "primer_binding_core_id": "source_fwd_core", "sequence": "GG"},
                    {"id": "source_rev_tail", "primer_binding_core_id": "source_rev_core", "sequence": "CC"},
                ],
                "nickase_sites": [
                    {
                        "id": "nick_1",
                        "enzyme": "Nt.Mock",
                        "recognition_sequence": "GGGG",
                        "start": 18,
                        "orientation": "forward",
                        "top_cut_offset": 2,
                    }
                ],
                "payload_windows": [
                    {"id": "payload_left", "start": 14, "end": 18, "projection_mode": "compound_required"},
                    {"id": "payload_right", "start": 22, "end": 26, "projection_mode": "compound_required"},
                ],
                "homology_windows": [
                    {"id": "stem_left", "start": 14, "end": 18, "projection_mode": "compound_allowed"},
                    {"id": "stem_right", "start": 22, "end": 26, "projection_mode": "compound_allowed"},
                ],
                "retained_regions": [
                    {"id": "retained_left", "start": 14, "end": 18, "projection_mode": "compound_allowed"},
                    {"id": "retained_right", "start": 22, "end": 26, "projection_mode": "compound_allowed"},
                ],
                "sacrificial_regions": [
                    {"id": "sacrificial_center", "start": 18, "end": 22, "projection_mode": "atomic_required"}
                ],
            },
        },
        "steps": {
            "source_pcr": {"forward_primer_id": "oES790", "reverse_primer_id": "oES791"},
            "double_nicking_digest": {"enzymes": ["Nt.Mock"]},
            "heat_cleanup": {"enabled": True, "min_retained_nt": 8},
            "adapter_anneal": {
                "adapter_id": "oES792",
                "compatibility_mode": "partial_complement",
                "partial_complement": {
                    "min_paired_nt": 4,
                    "allow_left_tail": True,
                    "allow_right_tail": True,
                },
            },
            "hairpin_ligation": {
                "ligase": "T4_DNA_ligase",
                "require_5p_phosphate": True,
                "compatibility_mode": "partial_complement",
                "partial_complement": {
                    "min_paired_nt": 4,
                    "allow_left_tail": True,
                    "allow_right_tail": True,
                },
            },
            "hairpin_pcr": {
                "forward_primer_id": "oES793",
                "reverse_primer_id": "oES794",
                "single_primer_precycles": {"enabled": True, "primer_id": "oES794", "cycles": 6},
                "x_structure_resolution_cycle": {"enabled": True},
            },
            "insert_cleanup": {"enabled": False},
            "backbone_pcr": {"enabled": False},
            "golden_gate_assembly": {"enabled": False},
        },
        "payload_goal": {
            "assembled_payload_pattern": "TTAACCGG",
            "left_half_ref": "payload_left",
            "right_half_ref": "payload_right",
            "assembly_space": "post_ligation",
            "evidence_policy": "require_guaranteed",
        },
        "catalogs": {
            "enzymes": "catalogs/enzymes.yaml",
            "oligo_parts": "catalogs/oligo_parts.yaml",
            "backbones": "catalogs/backbones.yaml",
        },
        "output": {
            "run_dir": "outputs/yiu/explicit",
            "emit_view_contracts": True,
            "publish_contract_version": 2,
        },
    }


def _base_split_yiu_v2_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "family": "yiu",
        "protocol_template": "yiu_split_payload_circularized_v1",
        "workflow_scope": "core_insert_generation",
        "name": "demo_yiu_split_v2",
        "source_oligo": {
            "authored_sequence": "ccgatgTCCCTATCAgGtctcGTGATAGAGAGGGGAAAGGGGCCCTCAGCCCGCTGA",
            "annotations": {
                "primer_binding_cores": [
                    {"id": "source_fwd_core", "start": 0, "end": 6, "strand": "primary"},
                    {"id": "source_rev_core", "start": 51, "end": 57, "strand": "complement"},
                ],
                "restriction_sites": [
                    {
                        "id": "split_payload_digest",
                        "enzyme": "BsaI",
                        "recognition_sequence": "GGTCTC",
                        "start": 15,
                        "orientation": "forward",
                        "top_cut_offset": 1,
                        "bottom_cut_offset": 5,
                    }
                ],
                "nickase_sites": [
                    {
                        "id": "nick_1",
                        "enzyme": "Nb.BssSI",
                        "recognition_sequence": "GGGG",
                        "start": 31,
                        "orientation": "forward",
                        "bottom_cut_offset": 2,
                    },
                    {
                        "id": "nick_2",
                        "enzyme": "Nb.BssSI",
                        "recognition_sequence": "GGGG",
                        "start": 38,
                        "orientation": "forward",
                        "bottom_cut_offset": 2,
                    },
                ],
                "payload_windows": [
                    {
                        "id": "payload_left",
                        "start": 6,
                        "end": 15,
                        "projection_mode": "compound_required",
                        "annotation_class": "payload_half_left",
                    },
                    {
                        "id": "payload_right",
                        "start": 21,
                        "end": 31,
                        "projection_mode": "compound_required",
                        "annotation_class": "payload_half_right",
                    },
                ],
                "retained_regions": [
                    {
                        "id": "retained_payload_left",
                        "start": 6,
                        "end": 15,
                        "projection_mode": "compound_allowed",
                        "annotation_class": "retained_region",
                    },
                    {
                        "id": "retained_payload_right",
                        "start": 21,
                        "end": 31,
                        "projection_mode": "compound_allowed",
                        "annotation_class": "retained_region",
                    },
                ],
                "sacrificial_regions": [
                    {
                        "id": "sacrificial_tract",
                        "start": 31,
                        "end": 43,
                        "projection_mode": "atomic_required",
                        "annotation_class": "sacrificial_region",
                    }
                ],
                "named_regions": [
                    {
                        "id": "snapback_seed",
                        "start": 43,
                        "end": 57,
                        "projection_mode": "compound_allowed",
                        "annotation_class": "snapback_seed",
                    },
                    {
                        "id": "downstream_of_payload",
                        "start": 31,
                        "end": 57,
                        "projection_mode": "compound_allowed",
                        "annotation_class": "fixed_scaffold",
                    },
                ],
            },
        },
        "steps": {
            "source_pcr": {"forward_primer_id": "oES790", "reverse_primer_id": "oES791"},
            "type_iis_digest": {"enzyme_id": "BsaI", "site_ids": ["split_payload_digest"]},
            "circularization": {
                "ligation_rule": {
                    "mode": "exact_complement",
                    "min_contiguous_core_bp": 4,
                    "max_left_tail_nt": 0,
                    "max_right_tail_nt": 0,
                    "max_bulge_nt": 0,
                    "min_left_flank_bp": 0,
                    "min_right_flank_bp": 0,
                    "bulge_owner": "either",
                }
            },
            "exonuclease_cleanup": {"enabled": True, "enzyme": "T5 exonuclease"},
            "sacrificial_digest": {
                "enzyme_ids": ["Nb.BssSI"],
                "site_ids": ["nick_1", "nick_2"],
                "sacrificial_region_ids": ["sacrificial_tract"],
                "retained_region_ids": ["retained_payload_left", "retained_payload_right", "snapback_seed"],
            },
            "fragment_cleanup": {"enabled": True, "max_fragment_nt": 6, "min_retained_nt": 20},
            "snapback_adapter_engagement": {
                "adapter_id": "oES792",
                "ligation_rule": {
                    "mode": "partial_complement",
                    "min_contiguous_core_bp": 4,
                    "max_left_tail_nt": 1,
                    "max_right_tail_nt": 1,
                    "max_bulge_nt": 0,
                    "min_left_flank_bp": 0,
                    "min_right_flank_bp": 0,
                    "bulge_owner": "either",
                },
            },
            "hairpin_ligation": {
                "ligase": "T4_DNA_ligase",
                "require_5p_phosphate": True,
                "ligation_rule": {
                    "mode": "partial_complement",
                    "min_contiguous_core_bp": 4,
                    "max_left_tail_nt": 1,
                    "max_right_tail_nt": 1,
                    "max_bulge_nt": 0,
                    "min_left_flank_bp": 0,
                    "min_right_flank_bp": 0,
                    "bulge_owner": "either",
                },
            },
            "hairpin_pcr": {
                "forward_primer_id": "oES793",
                "reverse_primer_id": "oES794",
            },
            "insert_cleanup": {"enabled": False},
            "backbone_pcr": {"enabled": False},
            "golden_gate_assembly": {"enabled": False},
        },
        "payload_goal": {
            "assembled_payload_pattern": "TCCCTATCAGTGATAGAGA",
            "left_half_ref": "payload_left",
            "right_half_ref": "payload_right",
            "assembly_space": "circularized_payload_junction",
            "evidence_policy": "require_guaranteed",
        },
        "template_bindings": {
            "source_forward_primer_core_ref": "source_fwd_core",
            "source_reverse_primer_core_ref": "source_rev_core",
            "snapback_seed_region_ref": "snapback_seed",
            "retained_left_region_ref": "retained_payload_left",
            "retained_right_region_ref": "retained_payload_right",
            "primary_sacrificial_region_refs": ["sacrificial_tract"],
            "circularization_left_overhang_ref": "split_payload_digest",
            "circularization_right_overhang_ref": "split_payload_digest",
        },
        "compound_regions": [
            {
                "id": "assembled_payload",
                "segments": [
                    {
                        "source_state": "source_oligo_ssdna",
                        "source_region_ref": "payload_left",
                        "orientation": "forward",
                    },
                    {
                        "source_state": "source_oligo_ssdna",
                        "source_region_ref": "payload_right",
                        "orientation": "forward",
                    },
                ],
                "join_policy": "junction_assemble",
            }
        ],
        "hard_invariants": [
            {
                "id": "payload_assembly",
                "class": "payload_assembly",
                "transform_ref": "circularization",
                "space_kind": "assembly_junction",
                "region_ref": "assembled_payload",
                "evidence_policy": "require_guaranteed",
                "params": {"expected_pattern": "TCCCTATCAGTGATAGAGA"},
            },
            {
                "id": "sacrificial_fragmentation",
                "class": "sacrificial_fragmentation",
                "transform_ref": "sacrificial_digest",
                "space_kind": "fragment_pool",
                "region_ref": "sacrificial_tract",
                "strand_scope": "complement",
                "evidence_policy": "require_guaranteed",
                "params": {
                    "enzyme_id": "Nb.BssSI",
                    "max_fragment_nt": 6,
                    "min_site_count": 2,
                    "require_full_region_cover": True,
                },
            },
        ],
        "catalogs": {
            "enzymes": "catalogs/enzymes.yaml",
            "oligo_parts": "catalogs/oligo_parts.yaml",
            "backbones": "catalogs/backbones.yaml",
        },
        "output": {
            "run_dir": "outputs/yiu/explicit",
            "emit_view_contracts": True,
            "emit_baserender_jobs": True,
            "publish_contract_version": 3,
        },
    }


def _write_spec(tmp_path: Path, payload: dict[str, object]) -> Path:
    workspace = tmp_path / "workspaces" / "demo_yiu"
    spec_path = workspace / "configs" / "yiu" / "example.yiu.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump({"yiu": payload}, sort_keys=False), encoding="utf-8")
    return spec_path


def test_load_yiu_spec_returns_workspace_root_and_ordered_steps(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path, _base_yiu_payload())

    spec, resolved_spec, workspace_root = load_yiu_spec(spec_path)

    assert resolved_spec == spec_path.resolve()
    assert workspace_root == spec_path.parents[2]
    assert spec.name == "demo_yiu"
    assert [step.kind for step in spec.step_graph.steps] == [
        "pcr",
        "restriction_digest",
        "circularization",
        "exonuclease_selection",
        "nickase_digest",
        "size_selection",
        "foldback",
        "adapter_ligation",
        "amplification",
    ]


def test_load_yiu_spec_rejects_paths_outside_configs_yiu_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_yiu"
    spec_path = workspace / "configs" / "other" / "example.yiu.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump({"yiu": _base_yiu_payload()}, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"<workspace>/configs/yiu/<name>\.yiu\.yaml"):
        load_yiu_spec(spec_path)


def test_load_yiu_spec_rejects_duplicate_annotation_ids(tmp_path: Path) -> None:
    payload = _base_yiu_payload()
    payload["source_oligo"]["payload_windows"].append({"id": "left_half", "start": 30, "end": 34})
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="duplicate annotation id"):
        load_yiu_spec(spec_path)


def test_load_yiu_spec_accepts_adapter_sequence_from_adapter_policy(tmp_path: Path) -> None:
    payload = _base_yiu_payload()
    payload["step_graph"]["steps"][7].pop("adapter_sequence")
    spec_path = _write_spec(tmp_path, payload)

    spec, _resolved_spec, _workspace_root = load_yiu_spec(spec_path)

    assert spec.step_graph.steps[7].kind == "adapter_ligation"
    assert spec.step_graph.steps[7].adapter_sequence is None
    assert spec.adapter_policy.adapter_sequence == "AGATCGGA"


def test_load_yiu_spec_rejects_adapter_ligation_without_any_adapter_source(tmp_path: Path) -> None:
    payload = _base_yiu_payload()
    payload["step_graph"]["steps"][7].pop("adapter_sequence")
    payload["adapter_policy"].pop("adapter_sequence")
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="adapter_ligation requires an adapter sequence source"):
        load_yiu_spec(spec_path)


def test_load_yiu_v2_spec_accepts_protocol_template_and_publish_contract_version(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path, _base_yiu_v2_payload())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec, _resolved_spec, _workspace_root = load_yiu_spec(spec_path)

    assert spec.schema_version == 2
    assert spec.family == "yiu"
    assert spec.protocol_template == "yiu_adapter_hairpin_v1"
    assert spec.workflow_scope == "core_insert_generation"
    assert spec.output.publish_contract_version == 2
    assert any("msd_hop_retron_eco1_v1" in str(item.message) for item in caught)


def test_load_yiu_v2_spec_rejects_unknown_protocol_template(tmp_path: Path) -> None:
    payload = _base_yiu_v2_payload()
    payload["protocol_template"] = "unknown_template"
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="protocol_template"):
        load_yiu_spec(spec_path)


def test_load_yiu_v2_split_template_accepts_authored_sequence_invariants_and_v3_publication(tmp_path: Path) -> None:
    spec_path = _write_spec(tmp_path, _base_split_yiu_v2_payload())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec, _resolved_spec, _workspace_root = load_yiu_spec(spec_path)

    assert spec.protocol_template == "yiu_circularized_payload_v1"
    assert spec.source_oligo.authored_sequence == "ccgatgTCCCTATCAgGtctcGTGATAGAGAGGGGAAAGGGGCCCTCAGCCCGCTGA"
    assert spec.source_oligo.sequence == "CCGATGTCCCTATCAGGTCTCGTGATAGAGAGGGGAAAGGGGCCCTCAGCCCGCTGA"
    assert spec.payload_goal.assembly_space == "circularized_payload_junction"
    assert spec.output.publish_contract_version == 3
    assert spec.output.emit_baserender_jobs is True
    assert spec.compound_regions[0].join_policy == "junction_assemble"
    assert spec.hard_invariants[0].evidence_policy == "require_guaranteed"
    assert any("yiu_split_payload_circularized_v1" in str(item.message) for item in caught)


def test_load_yiu_v2_split_template_requires_template_bindings(tmp_path: Path) -> None:
    payload = _base_split_yiu_v2_payload()
    payload.pop("template_bindings")
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="YIU_TEMPLATE_BINDING_MISSING"):
        load_yiu_spec(spec_path)


def test_load_yiu_v2_rejects_baserender_jobs_without_view_contracts(tmp_path: Path) -> None:
    payload = _base_split_yiu_v2_payload()
    payload["output"]["emit_view_contracts"] = False
    payload["output"]["emit_baserender_jobs"] = True
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="output.emit_baserender_jobs requires output.emit_view_contracts=true"):
        load_yiu_spec(spec_path)


def test_load_yiu_v2_core_insert_rejects_cloning_geometry_invariant(tmp_path: Path) -> None:
    payload = _base_split_yiu_v2_payload()
    payload["hard_invariants"].append(
        {
            "id": "unsupported_cloning_geometry",
            "class": "cloning_geometry",
            "transform_ref": "golden_gate_assembly",
            "space_kind": "assembly_junction",
            "evidence_policy": "require_guaranteed",
            "params": {"note": "unsupported in core_insert_generation"},
        }
    )
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="cloning_geometry"):
        load_yiu_spec(spec_path)


def test_load_yiu_v2_compat_template_rejects_hard_invariants_not_supported_for_template(tmp_path: Path) -> None:
    payload = _base_yiu_v2_payload()
    payload["hard_invariants"] = [
        {
            "id": "compat_payload_check",
            "class": "payload_assembly",
            "transform_ref": "hairpin_pcr",
            "space_kind": "assembly_junction",
            "region_ref": "payload_left",
            "evidence_policy": "require_guaranteed",
            "params": {"expected_pattern": "TTAA"},
        }
    ]
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="YIU_INVARIANT_CLASS_NOT_ALLOWED_FOR_TEMPLATE"):
        load_yiu_spec(spec_path)


def test_load_yiu_v2_rejects_overlap_override_unknown_annotation(tmp_path: Path) -> None:
    payload = _base_split_yiu_v2_payload()
    payload["source_oligo"]["annotations"]["overlap_overrides"] = [
        {
            "left_annotation_id": "missing_region",
            "right_annotation_id": "payload_left",
            "mode": "allow_partial",
            "rationale": "test",
        }
    ]
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="YIU_OVERLAP_OVERRIDE_REF_UNKNOWN"):
        load_yiu_spec(spec_path)


def _write_solve_spec(
    tmp_path: Path,
    *,
    solve_payload: dict[str, object],
    filename: str = "example.yiu.solve.yaml",
) -> Path:
    workspace = tmp_path / "workspaces" / "demo_yiu"
    spec_path = workspace / "configs" / "yiu" / filename
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump({"yiu_solve": solve_payload}, sort_keys=False), encoding="utf-8")
    return spec_path


def _base_yiu_solve_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "base_spec": "configs/yiu/example.yiu.yaml",
        "search": {
            "max_hits": 4,
            "materialize_top_k": 2,
            "max_search_nodes": 1024,
            "max_enumerated_candidates": 64,
        },
        "variables": {
            "source_windows": [
                {
                    "id": "payload_left",
                    "span_ref": "payload_left",
                    "alphabet": "iupac_dna",
                    "allowed_patterns": ["TCCCTATCA"],
                },
                {
                    "id": "payload_right",
                    "span_ref": "payload_right",
                    "alphabet": "iupac_dna",
                    "allowed_patterns": ["GTGATAGAGA"],
                },
            ]
        },
        "candidate_policy": {
            "require_guaranteed_hard_invariants": True,
            "forbid_possible_hits": True,
        },
        "output": {
            "run_dir": "outputs/yiu/solve",
            "emit_view_contracts": True,
            "emit_baserender_jobs": True,
            "publish_contract_version": 3,
        },
    }


def test_load_yiu_solve_spec_rejects_baserender_jobs_without_view_contracts(tmp_path: Path) -> None:
    solve_payload = _base_yiu_solve_payload()
    solve_payload["output"]["emit_view_contracts"] = False
    solve_payload["output"]["emit_baserender_jobs"] = True
    _write_spec(tmp_path, _base_split_yiu_v2_payload())
    spec_path = _write_solve_spec(tmp_path, solve_payload=solve_payload)

    with pytest.raises(ValueError, match="output.emit_baserender_jobs requires output.emit_view_contracts=true"):
        load_yiu_solve_spec(spec_path)


def test_load_yiu_solve_spec_resolves_workspace_root_and_base_spec(tmp_path: Path) -> None:
    explicit_spec_path = _write_spec(tmp_path, _base_split_yiu_v2_payload())
    spec_path = _write_solve_spec(tmp_path, solve_payload=_base_yiu_solve_payload())

    solve_spec, resolved_spec, workspace_root = load_yiu_solve_spec(spec_path)
    base_spec_path = resolve_base_spec_path_for_yiu_solve_spec(solve_spec, workspace_root=workspace_root)

    assert resolved_spec == spec_path.resolve()
    assert workspace_root == explicit_spec_path.parents[2].resolve()
    assert base_spec_path == explicit_spec_path.resolve()
    assert solve_spec.output.publish_contract_version == 3
    assert solve_spec.variables.source_windows[0].span_ref == "payload_left"


def test_load_yiu_solve_spec_rejects_bad_placement(tmp_path: Path) -> None:
    bad_path = tmp_path / "demo.yiu.solve.yaml"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text(yaml.safe_dump({"yiu_solve": _base_yiu_solve_payload()}, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"<workspace>/configs/yiu/<name>\.yiu\.solve\.yaml"):
        load_yiu_solve_spec(bad_path)
