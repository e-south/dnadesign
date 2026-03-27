"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/yiu/test_load.py

Strict-load contracts for YIU workflow specs.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.yiu.load import load_yiu_spec


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

    spec, _resolved_spec, _workspace_root = load_yiu_spec(spec_path)

    assert spec.schema_version == 2
    assert spec.family == "yiu"
    assert spec.protocol_template == "msd_hop_retron_eco1_v1"
    assert spec.workflow_scope == "core_insert_generation"
    assert spec.output.publish_contract_version == 2


def test_load_yiu_v2_spec_rejects_unknown_protocol_template(tmp_path: Path) -> None:
    payload = _base_yiu_v2_payload()
    payload["protocol_template"] = "unknown_template"
    spec_path = _write_spec(tmp_path, payload)

    with pytest.raises(ValueError, match="protocol_template"):
        load_yiu_spec(spec_path)
