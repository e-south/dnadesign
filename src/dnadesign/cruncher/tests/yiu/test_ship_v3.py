"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/yiu/test_ship_v3.py

Ship-readiness contracts for the canonical YIU v4 workflow.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.app.yiu_solve_workflow import run_yiu_solve
from dnadesign.cruncher.app.yiu_workflow import run_yiu_trace, yiu_show_payload
from dnadesign.cruncher.yiu.load import (
    load_yiu_solve_spec,
    load_yiu_spec,
    resolve_base_spec_path_for_yiu_solve_spec,
)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _owner_projection(state: str, strand: str, provenance_mode: str) -> dict[str, object]:
    return {
        "state": state,
        "strand": strand,
        "provenance_mode": provenance_mode,
    }


def _owner_lifecycle(
    owner_id: str, *, projected_to: list[dict[str, object]], disappears_after: str | None
) -> dict[str, object]:
    return {
        "owner_id": owner_id,
        "appears_in": ["source_oligo_ssdna"]
        if owner_id
        not in {
            "y_adapter_complementary_arm",
            "y_adapter_noncomplementary_arm",
            "hairpin_pcr_forward_binding_region",
            "hairpin_pcr_reverse_binding_region",
            "retained_region",
            "sacrificial_region_short",
        }
        else [],
        "projected_to": projected_to,
        "disappears_after": disappears_after,
    }


def _canonical_v4_payload() -> dict[str, object]:
    sequence = "CACGAGAGGTCTCACGAGAAAAAAAAACCTCAGCCCGCTGAACCTATAGAGGAGACC"
    owner_spans = {
        "source_fwd_primer_binding_region": (0, 6),
        "payload_left_half": (6, 15),
        "sacrificial_region_long": (15, 27),
        "tether_dock_complement": (27, 31),
        "tether_cap": (31, 35),
        "tether_dock": (35, 39),
        "snapback_stem": (39, 41),
        "payload_right_half": (41, 51),
        "source_rev_primer_binding_region": (51, 57),
    }
    structural_owners = [{"id": owner_id, "start": start, "end": end} for owner_id, (start, end) in owner_spans.items()]
    effect_tags = [
        {"id": "source_forward_primer_bindable", "class": "primer_bindable_by_source_forward", "start": 0, "end": 6},
        {"id": "left_nb_bsssi_member", "class": "nb_bsssi_array_member", "start": 0, "end": 6},
        {"id": "left_bsssi_bsai_overlap_unit", "class": "left_bsssi_bsai_overlap_unit", "start": 0, "end": 18},
        {"id": "payload_overhang_left", "class": "payload_overhang_left", "start": 6, "end": 10},
        {"id": "type_iis_recognition_left", "class": "type_iis_recognition_left", "start": 7, "end": 13},
        {"id": "right_nb_bsssi_member", "class": "nb_bsssi_array_member", "start": 12, "end": 18},
        {"id": "sacrificial_region_long_tag", "class": "sacrificial", "start": 15, "end": 27},
        {"id": "nt_bpu10i_snapback_site", "class": "nt_bpu10i_snapback_site", "start": 27, "end": 41},
        {"id": "payload_overhang_right", "class": "payload_overhang_right", "start": 41, "end": 45},
        {"id": "source_reverse_primer_bindable", "class": "primer_bindable_by_source_reverse", "start": 51, "end": 57},
        {"id": "type_iis_recognition_right", "class": "type_iis_recognition_right", "start": 51, "end": 57},
    ]
    owner_lifecycle = [
        _owner_lifecycle(
            "source_fwd_primer_binding_region",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
            ],
            disappears_after="pcr_linear_duplex",
        ),
        _owner_lifecycle(
            "payload_left_half",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
                _owner_projection("type_iis_cut_product_duplex", "primary", "cut_product_projection"),
                _owner_projection("type_iis_cut_product_duplex", "complement", "cut_product_projection"),
                _owner_projection("circularized_payload_candidate", "primary", "ligation_assembly"),
                _owner_projection("circularized_payload_candidate", "complement", "ligation_assembly"),
                _owner_projection("post_sacrificial_fragmentation", "primary", "retained_projection"),
                _owner_projection("post_fragment_cleanup", "primary", "retained_projection"),
                _owner_projection("snapback_adapter_complex", "primary", "retained_projection"),
                _owner_projection("ligated_ssdna_hairpin", "primary", "ligated_projection"),
                _owner_projection("hairpin_pcr_linear_insert", "primary", "amplification_projection"),
            ],
            disappears_after=None,
        ),
        _owner_lifecycle(
            "sacrificial_region_long",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
                _owner_projection("type_iis_cut_product_duplex", "primary", "cut_product_projection"),
                _owner_projection("type_iis_cut_product_duplex", "complement", "cut_product_projection"),
                _owner_projection("circularized_payload_candidate", "primary", "ligation_assembly"),
                _owner_projection("circularized_payload_candidate", "complement", "ligation_assembly"),
                _owner_projection("post_sacrificial_fragmentation", "primary", "sacrificial_projection"),
            ],
            disappears_after="post_sacrificial_fragmentation",
        ),
        _owner_lifecycle(
            "tether_dock_complement",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
                _owner_projection("type_iis_cut_product_duplex", "primary", "cut_product_projection"),
                _owner_projection("type_iis_cut_product_duplex", "complement", "cut_product_projection"),
                _owner_projection("circularized_payload_candidate", "primary", "ligation_assembly"),
                _owner_projection("circularized_payload_candidate", "complement", "ligation_assembly"),
                _owner_projection("post_sacrificial_fragmentation", "primary", "retained_projection"),
                _owner_projection("post_fragment_cleanup", "primary", "retained_projection"),
                _owner_projection("snapback_adapter_complex", "primary", "retained_projection"),
                _owner_projection("ligated_ssdna_hairpin", "primary", "ligated_projection"),
                _owner_projection("hairpin_pcr_linear_insert", "primary", "amplification_projection"),
            ],
            disappears_after=None,
        ),
        _owner_lifecycle(
            "tether_cap",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
                _owner_projection("type_iis_cut_product_duplex", "primary", "cut_product_projection"),
                _owner_projection("type_iis_cut_product_duplex", "complement", "cut_product_projection"),
                _owner_projection("circularized_payload_candidate", "primary", "ligation_assembly"),
                _owner_projection("circularized_payload_candidate", "complement", "ligation_assembly"),
                _owner_projection("post_sacrificial_fragmentation", "primary", "retained_projection"),
                _owner_projection("post_fragment_cleanup", "primary", "retained_projection"),
                _owner_projection("snapback_adapter_complex", "primary", "retained_projection"),
                _owner_projection("ligated_ssdna_hairpin", "primary", "ligated_projection"),
                _owner_projection("hairpin_pcr_linear_insert", "primary", "amplification_projection"),
            ],
            disappears_after=None,
        ),
        _owner_lifecycle(
            "tether_dock",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
                _owner_projection("type_iis_cut_product_duplex", "primary", "cut_product_projection"),
                _owner_projection("type_iis_cut_product_duplex", "complement", "cut_product_projection"),
                _owner_projection("circularized_payload_candidate", "primary", "ligation_assembly"),
                _owner_projection("circularized_payload_candidate", "complement", "ligation_assembly"),
                _owner_projection("post_sacrificial_fragmentation", "primary", "retained_projection"),
                _owner_projection("post_fragment_cleanup", "primary", "retained_projection"),
                _owner_projection("snapback_adapter_complex", "primary", "retained_projection"),
                _owner_projection("ligated_ssdna_hairpin", "primary", "ligated_projection"),
                _owner_projection("hairpin_pcr_linear_insert", "primary", "amplification_projection"),
            ],
            disappears_after=None,
        ),
        _owner_lifecycle(
            "snapback_stem",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
                _owner_projection("type_iis_cut_product_duplex", "primary", "cut_product_projection"),
                _owner_projection("type_iis_cut_product_duplex", "complement", "cut_product_projection"),
                _owner_projection("circularized_payload_candidate", "primary", "ligation_assembly"),
                _owner_projection("circularized_payload_candidate", "complement", "ligation_assembly"),
                _owner_projection("post_sacrificial_fragmentation", "primary", "retained_projection"),
                _owner_projection("post_fragment_cleanup", "primary", "retained_projection"),
                _owner_projection("snapback_adapter_complex", "primary", "retained_projection"),
                _owner_projection("ligated_ssdna_hairpin", "primary", "ligated_projection"),
                _owner_projection("hairpin_pcr_linear_insert", "primary", "amplification_projection"),
            ],
            disappears_after=None,
        ),
        _owner_lifecycle(
            "payload_right_half",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
                _owner_projection("type_iis_cut_product_duplex", "primary", "cut_product_projection"),
                _owner_projection("type_iis_cut_product_duplex", "complement", "cut_product_projection"),
                _owner_projection("circularized_payload_candidate", "primary", "ligation_assembly"),
                _owner_projection("circularized_payload_candidate", "complement", "ligation_assembly"),
                _owner_projection("post_sacrificial_fragmentation", "primary", "retained_projection"),
                _owner_projection("post_fragment_cleanup", "primary", "retained_projection"),
                _owner_projection("snapback_adapter_complex", "primary", "retained_projection"),
                _owner_projection("ligated_ssdna_hairpin", "primary", "ligated_projection"),
                _owner_projection("hairpin_pcr_linear_insert", "primary", "amplification_projection"),
            ],
            disappears_after=None,
        ),
        _owner_lifecycle(
            "source_rev_primer_binding_region",
            projected_to=[
                _owner_projection("pcr_linear_duplex", "primary", "literal_source"),
                _owner_projection("pcr_linear_duplex", "complement", "amplification_projection"),
            ],
            disappears_after="pcr_linear_duplex",
        ),
        _owner_lifecycle(
            "retained_region",
            projected_to=[
                _owner_projection("post_sacrificial_fragmentation", "primary", "retained_projection"),
                _owner_projection("post_fragment_cleanup", "primary", "retained_projection"),
                _owner_projection("hairpin_pcr_linear_insert", "complement", "amplification_projection"),
            ],
            disappears_after=None,
        ),
        _owner_lifecycle(
            "sacrificial_region_short",
            projected_to=[_owner_projection("post_sacrificial_fragmentation", "complement", "sacrificial_projection")],
            disappears_after="post_sacrificial_fragmentation",
        ),
        _owner_lifecycle(
            "y_adapter_complementary_arm",
            projected_to=[
                _owner_projection("snapback_adapter_complex", "primary", "late_introduction"),
                _owner_projection("ligated_ssdna_hairpin", "primary", "late_introduction"),
            ],
            disappears_after="ligated_ssdna_hairpin",
        ),
        _owner_lifecycle(
            "y_adapter_noncomplementary_arm",
            projected_to=[
                _owner_projection("snapback_adapter_complex", "primary", "late_introduction"),
                _owner_projection("ligated_ssdna_hairpin", "primary", "late_introduction"),
            ],
            disappears_after="ligated_ssdna_hairpin",
        ),
        _owner_lifecycle(
            "hairpin_pcr_forward_binding_region",
            projected_to=[_owner_projection("hairpin_pcr_linear_insert", "primary", "derived_binding_region")],
            disappears_after=None,
        ),
        _owner_lifecycle(
            "hairpin_pcr_reverse_binding_region",
            projected_to=[_owner_projection("hairpin_pcr_linear_insert", "primary", "derived_binding_region")],
            disappears_after=None,
        ),
    ]
    return {
        "yiu": {
            "schema_version": 4,
            "family": "yiu",
            "protocol_template": "yiu_circularized_payload_v1",
            "name": "example_reference_circularized",
            "source_oligo": {
                "authored_sequence": sequence,
                "structural_owners": structural_owners,
                "effect_tags": effect_tags,
            },
            "owner_lifecycle": owner_lifecycle,
            "external_parts": {
                "primer_source_forward": "oES790",
                "primer_source_reverse": "oES791",
                "hairpin_pcr_forward": "oES793",
                "hairpin_pcr_reverse": "oES794",
                "y_adapter": "oES792",
            },
            "enzymes": {
                "left_type_iis": "BsmBI",
                "right_type_iis": "BsmBI",
                "snapback_nickase": "Nt.Bpu10I",
                "sacrificial_nickase": "Nb.BssSI",
            },
            "payload": {
                "target_sequence": "AGGTCTCACACCTATAGAG",
                "bulge_mask": [],
            },
            "catalogs": {
                "enzymes": "catalogs/enzymes.yaml",
                "oligo_parts": "catalogs/oligo_parts.yaml",
                "backbones": "catalogs/backbones.yaml",
            },
            "output": {
                "run_dir": "outputs/yiu/explicit",
                "emit_view_contracts": True,
                "publish_contract_version": 4,
                "persist_render_jobs_debug": False,
            },
        }
    }


def _legacy_v3_payload() -> dict[str, object]:
    payload = _canonical_v4_payload()
    payload["yiu"]["schema_version"] = 3
    return payload


def _canonical_v4_solve_payload() -> dict[str, object]:
    return {
        "yiu_solve": {
            "schema_version": 1,
            "base_spec": "configs/yiu/example_reference_circularized.yiu.yaml",
            "target": {
                "payload_pattern": "AGGTCTCACACCTATAGAG",
                "bulge_mask": [],
            },
            "scaffold_windows": [
                {
                    "id": "sacrificial_spacing_window",
                    "owner_id": "sacrificial_region_long",
                    "relative_start": 3,
                    "relative_end": 12,
                    "allowed_patterns": ["AAAAAAAAA", "AAAATAAAA"],
                }
            ],
            "search": {
                "max_search_nodes": 16,
                "max_enumerated_candidates": 16,
            },
            "solve": {
                "compare_solutions": False,
                "max_solutions": 1,
            },
            "output": {
                "run_dir": "outputs/yiu/solve",
                "emit_view_contracts": True,
                "publish_contract_version": 4,
                "persist_render_jobs_debug": False,
            },
        }
    }


def _write_catalogs(workspace: Path) -> None:
    _write_yaml(
        workspace / "catalogs" / "enzymes.yaml",
        {
            "enzymes": {
                "entries": [
                    {"id": "BsmBI", "recognition_sequence": "GGTCTC", "top_cut_offset": 6, "bottom_cut_offset": 10},
                    {"id": "Nt.Bpu10I", "recognition_sequence": "CCTCAGC", "top_cut_offset": 2},
                    {"id": "Nb.BssSI", "recognition_sequence": "CACGAG"},
                ]
            }
        },
    )
    _write_yaml(
        workspace / "catalogs" / "oligo_parts.yaml",
        {
            "oligo_parts": {
                "entries": [
                    {"id": "oES790", "part_kind": "primer", "sequence": "GGTCTCAA"},
                    {"id": "oES791", "part_kind": "primer", "sequence": "GGTCTCAA"},
                    {"id": "oES792", "part_kind": "adapter", "sequence": "TCAGCGGGCTGAGG", "phosphorylated_5p": True},
                    {"id": "oES793", "part_kind": "primer", "sequence": "TCCCTA"},
                    {"id": "oES794", "part_kind": "primer", "sequence": "CTCTAT"},
                ]
            }
        },
    )
    _write_yaml(workspace / "catalogs" / "backbones.yaml", {"backbones": {"entries": []}})


def _write_canonical_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspaces" / "demo_yiu_ship_v4"
    spec_path = workspace / "configs" / "yiu" / "example_reference_circularized.yiu.yaml"
    solve_path = workspace / "configs" / "yiu" / "example_reference_circularized.yiu.solve.yaml"
    _write_yaml(spec_path, _canonical_v4_payload())
    _write_yaml(solve_path, _canonical_v4_solve_payload())
    _write_catalogs(workspace)
    return workspace, spec_path, solve_path


def _row_owner_annotations(state, *, row_id: str) -> list[dict[str, object]]:
    return [
        annotation
        for annotation in state.annotations
        if annotation.get("annotation_layer") == "structural_owner" and annotation.get("row_id") == row_id
    ]


def _assert_single_owner_partition(state, *, row_id: str, sequence: str) -> None:
    annotations = _row_owner_annotations(state, row_id=row_id)
    assert annotations, f"{state.state_id} is missing structural owners for row {row_id}"
    coverage = [0] * len(sequence)
    for annotation in annotations:
        start = int(annotation["start"])
        end = int(annotation["end"])
        for index in range(start, end):
            coverage[index] += 1
    assert coverage, f"{state.state_id} row {row_id} has no emitted owner coverage"
    assert all(value == 1 for value in coverage), f"{state.state_id} row {row_id} owner coverage was {coverage}"


def _state_by_id(report, state_id: str):
    return next(state for state in report.states if state.state_id == state_id)


def _hard_invariant_by_id(state, invariant_id: str) -> dict[str, object]:
    invariants = state.metadata.get("hard_invariants")
    assert isinstance(invariants, list)
    for invariant in invariants:
        if invariant.get("id") == invariant_id:
            return invariant
    raise AssertionError(f"missing hard invariant {invariant_id} in {state.state_id}")


def test_load_yiu_v4_accepts_canonical_schema_and_rejects_removed_semantic_fields(tmp_path: Path) -> None:
    _workspace, spec_path, _solve_path = _write_canonical_workspace(tmp_path)

    spec, _resolved_spec_path, _workspace_root = load_yiu_spec(spec_path)

    assert spec.schema_version == 4
    assert spec.protocol_template == "yiu_circularized_payload_v1"

    payload = _canonical_v4_payload()
    payload["yiu"]["source_oligo"]["structural_owners"][0]["projection_mode"] = "compound_required"
    decorated_path = tmp_path / "workspaces" / "decorated" / "configs" / "yiu" / "decorated.yiu.yaml"
    _write_yaml(decorated_path, payload)

    with pytest.raises(ValueError, match="projection_mode"):
        load_yiu_spec(decorated_path)

    payload = _canonical_v4_payload()
    payload["yiu"]["compound_regions"] = [{"id": "legacy_compound", "join_policy": "ordered_concat"}]
    removed_field_path = tmp_path / "workspaces" / "removed_field" / "configs" / "yiu" / "removed_field.yiu.yaml"
    _write_yaml(removed_field_path, payload)

    with pytest.raises(ValueError, match="compound_regions"):
        load_yiu_spec(removed_field_path)


def test_load_yiu_v4_rejects_unknown_effect_tags_and_unspecified_overlaps(tmp_path: Path) -> None:
    payload = _canonical_v4_payload()
    payload["yiu"]["source_oligo"]["effect_tags"][0]["class"] = "unknown_tag_class"
    unknown_class_path = tmp_path / "workspaces" / "unknown_tag" / "configs" / "yiu" / "unknown_tag.yiu.yaml"
    _write_yaml(unknown_class_path, payload)

    with pytest.raises(ValueError, match="effect_tag.class"):
        load_yiu_spec(unknown_class_path)

    payload = _canonical_v4_payload()
    payload["yiu"]["source_oligo"]["effect_tags"].append(
        {
            "id": "illegal_payload_bulge",
            "class": "payload_bulge_position",
            "start": 0,
            "end": 2,
        }
    )
    overlap_path = tmp_path / "workspaces" / "illegal_overlap" / "configs" / "yiu" / "illegal_overlap.yiu.yaml"
    _write_yaml(overlap_path, payload)

    with pytest.raises(ValueError, match="overlap"):
        load_yiu_spec(overlap_path)


def test_load_yiu_solve_spec_resolves_workspace_root_and_base_spec(tmp_path: Path) -> None:
    workspace, spec_path, solve_path = _write_canonical_workspace(tmp_path)

    solve_spec, resolved_solve_path, workspace_root = load_yiu_solve_spec(solve_path)

    assert resolved_solve_path == solve_path.resolve()
    assert workspace_root == workspace.resolve()
    assert resolve_base_spec_path_for_yiu_solve_spec(solve_spec, workspace_root=workspace_root) == spec_path.resolve()


def test_load_yiu_spec_rejects_legacy_schema_versions(tmp_path: Path) -> None:
    legacy_v3_path = tmp_path / "workspaces" / "legacy_v3" / "configs" / "yiu" / "legacy_v3.yiu.yaml"
    _write_yaml(legacy_v3_path, _legacy_v3_payload())

    with pytest.raises(ValueError, match="schema_version 4"):
        load_yiu_spec(legacy_v3_path)


def test_run_yiu_trace_v4_emits_owner_closed_states_real_cut_product_and_clean_bundle_artifacts(tmp_path: Path) -> None:
    _workspace, spec_path, _solve_path = _write_canonical_workspace(tmp_path)

    run_dir, report = run_yiu_trace(spec_path)

    assert report.metadata.spec_schema_version == 4
    assert [state.state_id for state in report.states] == [
        "source_oligo_ssdna",
        "pcr_linear_duplex",
        "type_iis_cut_product_duplex",
        "circularized_payload_candidate",
        "post_sacrificial_fragmentation",
        "post_fragment_cleanup",
        "snapback_adapter_complex",
        "ligated_ssdna_hairpin",
        "hairpin_pcr_linear_insert",
    ]

    assert (run_dir / "report.json").exists()
    assert (run_dir / "status.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "state_trace.jsonl").exists()
    assert (run_dir / "tables" / "state_sequences.csv").exists()
    assert (run_dir / "tables" / "state_owners.csv").exists()
    assert (run_dir / "tables" / "effect_tags.csv").exists()
    assert (run_dir / "tables" / "fragment_summary.csv").exists()
    assert (run_dir / "contracts" / "visuals").exists()
    assert not (run_dir / "published").exists()
    assert not (run_dir / "hits").exists()

    inventory_path = run_dir / "visual_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["bundle_kind"] == "explicit"
    assert inventory["protocol_template"] == "yiu_circularized_payload_v1"
    assert inventory["view_count"] == 9
    assert inventory["render_status"] == "not_requested"
    assert inventory["renderer_kind"] == "nucleotide_evidence_map"
    assert all(view["contract_kind"] == "sequence_evidence_map_v1" for view in inventory["views"])

    for state in report.states:
        if state.primary_sequence:
            _assert_single_owner_partition(state, row_id="primary", sequence=state.primary_sequence)
        if state.complement_sequence:
            _assert_single_owner_partition(state, row_id="complement", sequence=state.complement_sequence)

    source_state = _state_by_id(report, "source_oligo_ssdna")
    cut_state = _state_by_id(report, "type_iis_cut_product_duplex")
    assert cut_state.primary_sequence != source_state.primary_sequence
    assert len(cut_state.primary_sequence) < len(source_state.primary_sequence)
    assert "source_fwd_primer_binding_region" not in {
        item["id"] for item in _row_owner_annotations(cut_state, row_id="primary")
    }
    assert "source_rev_primer_binding_region" not in {
        item["id"] for item in _row_owner_annotations(cut_state, row_id="primary")
    }

    ligated_hairpin = _state_by_id(report, "ligated_ssdna_hairpin")
    ligated_owner_ids = {item["id"] for item in _row_owner_annotations(ligated_hairpin, row_id="primary")}
    assert "y_adapter_complementary_arm" in ligated_owner_ids
    assert "y_adapter_noncomplementary_arm" in ligated_owner_ids

    final_insert = _state_by_id(report, "hairpin_pcr_linear_insert")
    assert final_insert.complement_sequence
    final_primary_owner_ids = {item["id"] for item in _row_owner_annotations(final_insert, row_id="primary")}
    final_complement_owner_ids = {item["id"] for item in _row_owner_annotations(final_insert, row_id="complement")}
    assert "hairpin_pcr_forward_binding_region" in final_primary_owner_ids
    assert "hairpin_pcr_reverse_binding_region" in final_primary_owner_ids
    assert "retained_region" in final_complement_owner_ids

    post_fragment_cleanup = _state_by_id(report, "post_fragment_cleanup")
    nt_bpu10i = _hard_invariant_by_id(post_fragment_cleanup, "nt_bpu10i_snapback_invariant")
    assert nt_bpu10i["status"] == "guaranteed"
    assert nt_bpu10i["observed"]["recognized_sequence"] == "CCTCAGC"
    assert nt_bpu10i["observed"]["nick_boundary"] == 33
    assert nt_bpu10i["subchecks"]["downstream_exposed_tether_geometry"]["status"] == "guaranteed"

    circularized = _state_by_id(report, "circularized_payload_candidate")
    payload_assembly = _hard_invariant_by_id(circularized, "payload_assembly_invariant")
    assert payload_assembly["status"] == "guaranteed"
    assert payload_assembly["observed"]["assembled_payload_sequence"] == "AGGTCTCACACCTATAGAG"

    with (run_dir / "tables" / "fragment_summary.csv").open(newline="", encoding="utf-8") as handle:
        fragment_rows = list(csv.DictReader(handle))
    assert fragment_rows
    assert any(row["state_id"] == "post_sacrificial_fragmentation" for row in fragment_rows)
    assert {row["max_fragment_nt"] for row in fragment_rows if row["state_id"] == "post_sacrificial_fragmentation"} == {
        "12"
    }

    show_payload = yiu_show_payload(run_dir)
    assert show_payload["bundle_kind"] == "explicit"
    assert show_payload["protocol_template"] == "yiu_circularized_payload_v1"
    assert show_payload["schema_version"] == 4
    assert show_payload["explicit_final_state"] == "hairpin_pcr_linear_insert"
    assert show_payload["state_count"] == 9
    assert show_payload["visual_inventory_path"] == str(inventory_path.resolve())


def test_run_yiu_solve_v4_returns_single_canonical_solution_and_clean_bundle(tmp_path: Path) -> None:
    _workspace, _spec_path, solve_path = _write_canonical_workspace(tmp_path)

    run_dir, report = run_yiu_solve(solve_path)

    assert report.status == "solved"
    assert report.satisfying_solution_count == 2
    assert report.selected_solution_path is not None
    assert report.selected_source_sequence is not None
    assert "AAAAAAAAA" in report.selected_source_sequence
    assert (run_dir / "solve_report.json").exists()
    assert (run_dir / "solve_status.json").exists()
    assert (run_dir / "solve_manifest.json").exists()
    assert (run_dir / "solution" / "report.json").exists()
    assert (run_dir / "visual_inventory.json").exists()
    assert not (run_dir / "hits").exists()
    assert not (run_dir / "alternatives").exists()
    assert not (run_dir / "comparison").exists()

    inventory = json.loads((run_dir / "visual_inventory.json").read_text(encoding="utf-8"))
    assert inventory["bundle_kind"] == "solve"
    assert inventory["render_status"] == "not_requested"
    assert all(str(view["view_contract_path"]).startswith("solution/") for view in inventory["views"])

    show_payload = yiu_show_payload(run_dir)
    assert show_payload["bundle_kind"] == "solve"
    assert show_payload["protocol_template"] == "yiu_circularized_payload_v1"
    assert show_payload["solve_status"] == "solved"
    assert show_payload["satisfying_solution_count"] == 2
    assert show_payload["comparison_solution_count"] == 0
    assert show_payload["selected_canonical_solution_path"].endswith("/solution")


def test_run_yiu_solve_v4_incomplete_search_fails_closed(tmp_path: Path) -> None:
    _workspace, _spec_path, solve_path = _write_canonical_workspace(tmp_path)
    payload = _canonical_v4_solve_payload()
    payload["yiu_solve"]["search"]["max_enumerated_candidates"] = 1
    _write_yaml(solve_path, payload)

    run_dir, report = run_yiu_solve(solve_path)

    assert report.status == "incomplete_search"
    assert report.selected_solution_path is None
    assert report.selected_source_sequence is None
    assert not (run_dir / "solution").exists()
    assert not (run_dir / "visual_inventory.json").exists()


def test_run_yiu_solve_rejects_forbidden_mutation_window(tmp_path: Path) -> None:
    _workspace, _spec_path, solve_path = _write_canonical_workspace(tmp_path)
    payload = _canonical_v4_solve_payload()
    payload["yiu_solve"]["scaffold_windows"][0]["owner_id"] = "source_fwd_primer_binding_region"
    _write_yaml(solve_path, payload)

    with pytest.raises(ValueError, match="owner_id"):
        run_yiu_solve(solve_path)
