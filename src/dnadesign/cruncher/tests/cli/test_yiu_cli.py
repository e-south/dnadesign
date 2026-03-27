"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_yiu_cli.py

CLI contract tests for the YIU workflow family.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _yiu_payload(*, expected_right_overhang: str = "ACGT") -> dict[str, object]:
    return {
        "yiu": {
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
                        "expected_right_overhang": expected_right_overhang,
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
    }


def _write_yiu_workspace(tmp_path: Path, *, expected_right_overhang: str = "ACGT") -> tuple[Path, Path]:
    workspace = tmp_path / "workspaces" / "demo_yiu"
    spec_path = workspace / "configs" / "yiu" / "example.yiu.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(_yiu_payload(expected_right_overhang=expected_right_overhang), sort_keys=False),
        encoding="utf-8",
    )
    return workspace, spec_path


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _yiu_v2_payload(
    *,
    source_sequence: str = "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACGTTTTT",
    evidence_policy: str = "require_guaranteed",
    workflow_scope: str = "core_insert_generation",
) -> dict[str, object]:
    return {
        "yiu": {
            "schema_version": 2,
            "family": "yiu",
            "protocol_template": "msd_hop_retron_eco1_v1",
            "workflow_scope": workflow_scope,
            "name": "demo_yiu_v2",
            "source_oligo": {
                "sequence": source_sequence,
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
                        {
                            "id": "sacrificial_center",
                            "start": 18,
                            "end": 22,
                            "projection_mode": "atomic_required",
                        }
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
                "evidence_policy": evidence_policy,
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
    }


def _write_yiu_v2_workspace(
    tmp_path: Path,
    *,
    source_sequence: str = "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACGTTTTT",
    evidence_policy: str = "require_guaranteed",
) -> tuple[Path, Path]:
    workspace = tmp_path / "workspaces" / "demo_yiu_v2"
    spec_path = workspace / "configs" / "yiu" / "example.yiu.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            _yiu_v2_payload(source_sequence=source_sequence, evidence_policy=evidence_policy),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_yaml(
        workspace / "catalogs" / "enzymes.yaml",
        {
            "enzymes": {
                "entries": [
                    {"id": "Nt.Mock", "recognition_sequence": "GGGG", "top_cut_offset": 2},
                ]
            }
        },
    )
    _write_yaml(
        workspace / "catalogs" / "oligo_parts.yaml",
        {
            "oligo_parts": {
                "entries": [
                    {"id": "oES790", "part_kind": "primer", "sequence": "GGAAAA"},
                    {"id": "oES791", "part_kind": "primer", "sequence": "CCAAAA"},
                    {"id": "oES792", "part_kind": "adapter", "sequence": "ACCGGTTAA", "phosphorylated_5p": True},
                    {"id": "oES793", "part_kind": "primer", "sequence": "TTAA"},
                    {"id": "oES794", "part_kind": "primer", "sequence": "CCGG"},
                ]
            }
        },
    )
    _write_yaml(workspace / "catalogs" / "backbones.yaml", {"backbones": {"entries": []}})
    return workspace, spec_path


def test_root_help_includes_yiu_group() -> None:
    result = runner.invoke(app, ["--help"], color=False)

    assert result.exit_code == 0
    assert "yiu" in result.output
    assert "hairpin oligo" in result.output.lower()


def test_yiu_help_describes_validate_design_trace_show_surface() -> None:
    result = runner.invoke(app, ["yiu", "--help"], color=False)

    assert result.exit_code == 0
    assert "init-workspace" in result.output
    assert "validate" in result.output
    assert "design" in result.output
    assert "trace" in result.output
    assert "show" in result.output


def test_yiu_validate_json_reports_step_trace(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "satisfied"
    assert payload["protocol"] == "yiu_v1"
    assert payload["states"][0]["state_id"] == "source_oligo_ssdna"
    assert payload["states"][-1]["state_id"] == "downstream_amplifiable_product"
    assert payload["metadata"]["emitted_view_count"] == 0
    pcr_state = next(state for state in payload["states"] if state["state_id"] == "pcr_linear_duplex")
    assert pcr_state["metadata"]["amplicon_start"] == 0
    assert pcr_state["metadata"]["amplicon_end"] == 40
    assert pcr_state["metadata"]["amplicon_length_nt"] == 40
    assert pcr_state["primary_sequence"] == "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACGTTTTT"
    nickase_state = next(state for state in payload["states"] if state["state_id"] == "post_nickase_fragmentation")
    assert nickase_state["metadata"]["retained_product"] == "TTAACCGG"
    assert nickase_state["metadata"]["retained_components"] == [
        {
            "id": "retained_left",
            "source_start": 14,
            "source_end": 18,
            "state_start": 0,
            "state_end": 4,
            "sequence": "TTAA",
        },
        {
            "id": "retained_right",
            "source_start": 22,
            "source_end": 26,
            "state_start": 4,
            "state_end": 8,
            "sequence": "CCGG",
        },
    ]
    assert payload["sequence_mode"] == "concrete"
    assert payload["validation_mode"] == "concrete_realization"
    assert payload["states"][0]["sequence_mode"] == "concrete"
    assert payload["states"][0]["validation_mode"] == "concrete_realization"


def test_yiu_design_writes_bundle_and_show_reads_it(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_workspace(tmp_path)

    result = runner.invoke(app, ["yiu", "design", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 0
    run_root = workspace / "outputs" / "yiu" / "explicit" / "demo_yiu"
    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    assert (run_dir / "yiu_manifest.json").exists()
    assert (run_dir / "yiu_status.json").exists()
    assert (run_dir / "yiu_report.json").exists()
    assert (run_dir / "yiu_trace.jsonl").exists()
    assert (run_dir / "yiu_trace_manifest.json").exists()
    assert (run_dir / "yiu_published_views_manifest.json").exists()
    assert (run_dir / "yiu_parts.csv").exists()
    assert (run_dir / "yiu_annotations.csv").exists()
    assert (run_dir / "yiu_fragments.csv").exists()
    assert (run_dir / "published" / "views" / "source_oligo_ssdna.json").exists()
    assert (run_dir / "published" / "views" / "downstream_amplifiable_product.json").exists()

    show_result = runner.invoke(app, ["yiu", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 0
    assert "demo_yiu" in show_result.output
    assert "Manifest ->" in show_result.output
    assert "Trace ->" in show_result.output
    assert "published/views" in show_result.output


def test_yiu_validate_reports_pattern_compatibility_for_iupac_source(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["source_oligo"]["sequence"] = "AAAAGGTCTCACGTNTAAGGGGCCGGGGTCTCACGTTTTT"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["sequence_mode"] == "iupac_pattern"
    assert report["validation_mode"] == "pattern_compatibility"
    retained_state = next(state for state in report["states"] if state["state_id"] == "post_nickase_fragmentation")
    assert retained_state["sequence_mode"] == "iupac_pattern"
    assert retained_state["validation_mode"] == "pattern_compatibility"


def test_yiu_design_writes_mode_and_reproducibility_manifests(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_workspace(tmp_path)

    result = runner.invoke(app, ["yiu", "design", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 0
    run_root = workspace / "outputs" / "yiu" / "explicit" / "demo_yiu"
    run_dir = next(run_root.iterdir())

    manifest = json.loads((run_dir / "yiu_manifest.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "yiu_status.json").read_text(encoding="utf-8"))
    trace_manifest = json.loads((run_dir / "yiu_trace_manifest.json").read_text(encoding="utf-8"))
    views_manifest = json.loads((run_dir / "yiu_published_views_manifest.json").read_text(encoding="utf-8"))
    state_view = json.loads((run_dir / "published" / "views" / "source_oligo_ssdna.json").read_text(encoding="utf-8"))

    assert manifest["family"] == "yiu"
    assert manifest["protocol"] == "yiu_v1"
    assert manifest["state_count"] == 10
    assert manifest["sequence_mode"] == "concrete"
    assert manifest["validation_mode"] == "concrete_realization"
    assert len(manifest["input_fingerprint"]) == 64
    assert len(manifest["catalog_fingerprint"]) == 64
    assert manifest["engine_contract_version"]

    assert status["family"] == "yiu"
    assert status["protocol"] == "yiu_v1"
    assert status["state_count"] == 10
    assert status["sequence_mode"] == "concrete"
    assert status["validation_mode"] == "concrete_realization"
    assert len(status["input_fingerprint"]) == 64
    assert len(status["catalog_fingerprint"]) == 64
    assert status["engine_contract_version"]

    assert trace_manifest["family"] == "yiu"
    assert trace_manifest["protocol"] == "yiu_v1"
    assert trace_manifest["state_count"] == 10
    assert trace_manifest["sequence_mode"] == "concrete"
    assert trace_manifest["validation_mode"] == "concrete_realization"
    assert trace_manifest["states"][0]["state_id"] == "source_oligo_ssdna"
    assert trace_manifest["states"][-1]["path"] == "published/views/downstream_amplifiable_product.json"

    assert views_manifest["family"] == "yiu"
    assert views_manifest["protocol"] == "yiu_v1"
    assert views_manifest["schema_version"] == 1
    assert views_manifest["view_count"] == 10
    assert views_manifest["views"][0]["state_id"] == "source_oligo_ssdna"

    assert state_view["schema_version"] == 1
    assert state_view["family"] == "yiu"
    assert state_view["protocol"] == "yiu_v1"
    assert state_view["sequence_mode"] == "concrete"
    assert state_view["validation_mode"] == "concrete_realization"


def test_yiu_validate_reports_structured_digest_issue_codes(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path, expected_right_overhang="TTTT")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "DIGEST_OVERHANG_MISMATCH" in result.output


def test_yiu_validate_reports_missing_payload_region_reference(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["payload_goal"]["left_half_ref"] = "missing_left_half"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "PAYLOAD_REGION_MISSING" in result.output


def test_yiu_validate_reports_annotations_outside_pcr_amplicon(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["source_oligo"]["primer_sites"][1]["start"] = 10
    payload["yiu"]["source_oligo"]["primer_sites"][1]["end"] = 14
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "PCR_AMPLICON_EXCLUDES_ANNOTATION" in result.output


def test_yiu_validate_reports_size_selection_removed_fragment_threshold(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["cleanup_policy"]["size_selection"]["min_removed_fragment_nt"] = 3
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "SIZE_SELECTION_FRAGMENT_TOO_SHORT_TO_REMOVE" in result.output


def test_yiu_validate_projects_digest_state_and_removed_flanks(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    report = json.loads(result.output)
    digest_state = next(state for state in report["states"] if state["state_id"] == "digested_linear_duplex")
    assert digest_state["primary_sequence"] == "ACGTTTAAGGGGCCGGGGTCTC"
    assert digest_state["metadata"]["left_primary_cut_boundary"] == 10
    assert digest_state["metadata"]["right_primary_cut_boundary"] == 32
    assert digest_state["metadata"]["removed_primary_flanks"] == [
        {"start": 0, "end": 10, "length_nt": 10},
        {"start": 32, "end": 40, "length_nt": 8},
    ]
    projected = {item["id"]: item for item in digest_state["metadata"]["projected_annotations"]}
    assert projected["left_half"]["start"] == 4
    assert projected["left_half"]["end"] == 8
    assert projected["right_half"]["start"] == 12
    assert projected["right_half"]["end"] == 16


def test_yiu_validate_publishes_circularization_payload_junction_geometry(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    report = json.loads(result.output)
    circular_state = next(state for state in report["states"] if state["state_id"] == "circularization_candidate")
    assert circular_state["metadata"]["assembled_payload"] == "TTAACCGG"
    assert circular_state["metadata"]["payload_junction_segments"] == [
        {
            "id": "left_half",
            "source_start": 14,
            "source_end": 18,
            "payload_start": 0,
            "payload_end": 4,
            "sequence": "TTAA",
        },
        {
            "id": "right_half",
            "source_start": 22,
            "source_end": 26,
            "payload_start": 4,
            "payload_end": 8,
            "sequence": "CCGG",
        },
    ]
    assert circular_state["metadata"]["payload_junction"] == {
        "left_region_id": "left_half",
        "right_region_id": "right_half",
        "payload_join_index": 4,
        "junction_rule": "contiguous_after_ligation",
    }


def test_yiu_validate_reports_uncut_sacrificial_region(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["source_oligo"]["sacrificial_regions"] = [{"id": "sacrificial_center", "start": 26, "end": 30}]
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "NICKASE_SACRIFICIAL_REGION_UNCUT" in result.output


def test_yiu_validate_projects_foldback_windows_on_retained_product(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    report = json.loads(result.output)
    foldback_state = next(state for state in report["states"] if state["state_id"] == "foldback_or_cap_intermediate")
    assert foldback_state["primary_sequence"] == "TTAACCGG"
    assert foldback_state["metadata"]["left_homology"] == "TTAA"
    assert foldback_state["metadata"]["right_homology"] == "TTAA"
    assert foldback_state["metadata"]["complementary_bases"] == 4
    assert foldback_state["metadata"]["paired_nt"] == 4
    assert foldback_state["metadata"]["overlap_start"] == 0
    assert foldback_state["metadata"]["overlap_end"] == 4
    assert foldback_state["metadata"]["sequence_mode"] == "concrete"
    assert foldback_state["metadata"]["topology_compatibility"] is True
    assert foldback_state["metadata"]["projected_homology_windows"] == [
        {
            "id": "left_fold",
            "source_start": 14,
            "source_end": 18,
            "sequence": "TTAA",
            "spans_junction": False,
            "parts": [{"segment_id": "retained_left", "start": 0, "end": 4}],
            "state_start": 0,
            "state_end": 4,
        },
        {
            "id": "right_fold",
            "source_start": 14,
            "source_end": 18,
            "sequence": "TTAA",
            "spans_junction": False,
            "parts": [{"segment_id": "retained_left", "start": 0, "end": 4}],
            "state_start": 0,
            "state_end": 4,
        },
    ]


def test_yiu_validate_accepts_partial_complement_when_exact_complement_fails(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["source_oligo"]["restriction_sites"][1]["bottom_cut_offset"] = 11
    payload["yiu"]["step_graph"]["steps"][1]["expected_right_overhang"] = "ACGTT"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    exact_result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert exact_result.exit_code == 1
    exact_report = json.loads(exact_result.output)
    assert any(issue["code"] == "CIRCULARIZATION_COMPATIBILITY_FAIL" for issue in exact_report["issues"])

    payload["yiu"]["step_graph"]["steps"][2]["compatibility"] = "partial_complement"
    payload["yiu"]["step_graph"]["steps"][2]["min_paired_nt"] = 4
    payload["yiu"]["step_graph"]["steps"][2]["max_unpaired_tail_nt"] = 1
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    partial_result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert partial_result.exit_code == 0
    partial_report = json.loads(partial_result.output)
    circular_state = next(
        state for state in partial_report["states"] if state["state_id"] == "circularization_candidate"
    )
    assert circular_state["metadata"]["compatibility"] == "partial_complement"
    assert circular_state["metadata"]["paired_nt"] == 4
    assert circular_state["metadata"]["unpaired_tail_nt"] == 1
    assert circular_state["metadata"]["bulge_nt"] == 0


def test_yiu_validate_accepts_bulged_sticky_end_only_within_bulge_budget(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["source_oligo"]["sequence"] = "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACCGTTTT"
    payload["yiu"]["source_oligo"]["restriction_sites"][1]["bottom_cut_offset"] = 11
    payload["yiu"]["step_graph"]["steps"][1]["expected_right_overhang"] = "ACCGT"
    payload["yiu"]["step_graph"]["steps"][2]["compatibility"] = "bulged"
    payload["yiu"]["step_graph"]["steps"][2]["min_paired_nt"] = 4
    payload["yiu"]["step_graph"]["steps"][2]["max_unpaired_tail_nt"] = 0
    payload["yiu"]["step_graph"]["steps"][2]["max_bulge_nt"] = 1
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    bulged_result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert bulged_result.exit_code == 0
    bulged_report = json.loads(bulged_result.output)
    circular_state = next(
        state for state in bulged_report["states"] if state["state_id"] == "circularization_candidate"
    )
    assert circular_state["metadata"]["compatibility"] == "bulged"
    assert circular_state["metadata"]["paired_nt"] == 4
    assert circular_state["metadata"]["unpaired_tail_nt"] == 0
    assert circular_state["metadata"]["bulge_nt"] == 1
    assert circular_state["metadata"]["bulge_side"] == "right"

    payload["yiu"]["step_graph"]["steps"][2]["max_bulge_nt"] = 0
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    failing_result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert failing_result.exit_code == 1
    assert "CIRCULARIZATION_COMPATIBILITY_FAIL" in failing_result.output


def test_yiu_validate_rejects_retained_sacrificial_overlap(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["source_oligo"]["sacrificial_regions"] = [{"id": "sacrificial_center", "start": 16, "end": 20}]
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "RETAINED_SACRIFICIAL_OVERLAP" in result.output


def test_yiu_validate_represents_junction_spanning_homology_projection(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["source_oligo"]["homology_windows"][1] = {"id": "junction_span", "start": 16, "end": 24}
    payload["yiu"]["step_graph"]["steps"][6]["right_homology_window"] = "junction_span"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert any(issue["code"] == "HOMOLOGY_WINDOW_SPANS_JUNCTION" for issue in report["issues"])
    foldback_state = next(state for state in report["states"] if state["state_id"] == "foldback_or_cap_intermediate")
    projected = next(
        item for item in foldback_state["metadata"]["projected_homology_windows"] if item["id"] == "junction_span"
    )
    assert projected["spans_junction"] is True
    assert projected["parts"] == [
        {"segment_id": "retained_left", "start": 2, "end": 4},
        {"segment_id": "retained_right", "start": 4, "end": 6},
    ]


def test_yiu_validate_publishes_branched_adapter_geometry(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    report = json.loads(result.output)
    adapter_state = next(state for state in report["states"] if state["state_id"] == "y_adapter_ligated_product")
    assert adapter_state["primary_sequence"] == "TTAACCGG|AGATCGGA"
    assert adapter_state["metadata"]["topology"] == "branched_y"
    assert adapter_state["metadata"]["arms"] == [
        {
            "id": "retained_product",
            "role": "payload",
            "state_start": 0,
            "state_end": 8,
            "sequence": "TTAACCGG",
        },
        {
            "id": "y_adapter",
            "role": "adapter",
            "state_start": 9,
            "state_end": 17,
            "sequence": "AGATCGGA",
        },
    ]
    assert adapter_state["metadata"]["branch_junction"] == {
        "payload_arm_id": "retained_product",
        "payload_state_index": 8,
        "adapter_arm_id": "y_adapter",
        "adapter_state_index": 9,
        "separator": "|",
    }


def test_yiu_validate_reports_foldback_window_excluded_from_retained_product(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["source_oligo"]["homology_windows"][1]["start"] = 32
    payload["yiu"]["source_oligo"]["homology_windows"][1]["end"] = 36
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "HOMOLOGY_WINDOW_EXCLUDED_FROM_CURRENT_STATE" in result.output


def test_yiu_validate_errors_when_catalog_path_is_missing(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_workspace(tmp_path)
    payload = _yiu_payload()
    payload["yiu"]["catalogs"] = {"restriction_enzymes": "catalogs/missing_restriction_enzymes.yaml"}
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "catalogs.restriction_enzymes not found" in result.output


def test_yiu_validate_reports_missing_restriction_catalog_entry(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_workspace(tmp_path)
    _write_yaml(
        workspace / "catalogs" / "restriction_enzymes.yaml",
        {"restriction_enzymes": {"entries": [{"id": "BsmBI", "recognition_sequence": "CGTCTC"}]}},
    )
    payload = _yiu_payload()
    payload["yiu"]["catalogs"] = {"restriction_enzymes": "catalogs/restriction_enzymes.yaml"}
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "RESTRICTION_CATALOG_ENTRY_MISSING" in result.output


def test_yiu_validate_reports_nickase_catalog_mismatch(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_workspace(tmp_path)
    _write_yaml(
        workspace / "catalogs" / "nickases.yaml",
        {"nickases": {"entries": [{"id": "Nt.Mock", "recognition_sequence": "CCCC", "top_cut_offset": 2}]}},
    )
    payload = _yiu_payload()
    payload["yiu"]["catalogs"] = {"nickases": "catalogs/nickases.yaml"}
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "NICKASE_CATALOG_MISMATCH" in result.output


def test_yiu_validate_reports_missing_adapter_catalog_entry(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_workspace(tmp_path)
    _write_yaml(
        workspace / "catalogs" / "adapters.yaml",
        {"adapters": {"entries": [{"id": "demo_y_adapter", "sequence": "AGATCGGA"}]}},
    )
    payload = _yiu_payload()
    payload["yiu"]["adapter_policy"]["y_adapter_id"] = "missing_adapter"
    payload["yiu"]["catalogs"] = {"adapters": "catalogs/adapters.yaml"}
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "ADAPTER_CATALOG_ENTRY_MISSING" in result.output


def test_yiu_validate_accepts_adapter_sequence_from_catalog_only(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_workspace(tmp_path)
    _write_yaml(
        workspace / "catalogs" / "adapters.yaml",
        {"adapters": {"entries": [{"id": "demo_y_adapter", "sequence": "AGATCGGA"}]}},
    )
    payload = _yiu_payload()
    payload["yiu"]["step_graph"]["steps"][7].pop("adapter_sequence")
    payload["yiu"]["adapter_policy"].pop("adapter_sequence")
    payload["yiu"]["adapter_policy"]["y_adapter_id"] = "demo_y_adapter"
    payload["yiu"]["catalogs"] = {"adapters": "catalogs/adapters.yaml"}
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    report = json.loads(result.output)
    adapter_state = next(state for state in report["states"] if state["state_id"] == "y_adapter_ligated_product")
    assert adapter_state["metadata"]["y_adapter_id"] == "demo_y_adapter"
    assert adapter_state["metadata"]["adapter_sequence"] == "AGATCGGA"


def test_yiu_validate_errors_when_catalog_schema_is_invalid(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_workspace(tmp_path)
    _write_yaml(workspace / "catalogs" / "restriction_enzymes.yaml", {"restriction_enzymes": {"entries": [{}]}})
    payload = _yiu_payload()
    payload["yiu"]["catalogs"] = {"restriction_enzymes": "catalogs/restriction_enzymes.yaml"}
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert result.exit_code == 1
    assert "YIU restriction catalog validation failed" in result.output


def test_yiu_init_workspace_scaffolds_family_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "demo_yiu_scaffold"

    result = runner.invoke(app, ["yiu", "init-workspace", "--output", str(workspace_root)], color=False)

    assert result.exit_code == 0
    assert (workspace_root / "configs" / "runbook.yaml").exists()
    assert (workspace_root / "configs" / "yiu" / "example.yiu.yaml").exists()
    assert (workspace_root / "catalogs" / "restriction_enzymes.yaml").exists()
    assert (workspace_root / "catalogs" / "nickases.yaml").exists()
    assert (workspace_root / "catalogs" / "adapters.yaml").exists()

    list_result = runner.invoke(
        app,
        ["workspaces", "list", "--root", str(workspace_root.parent)],
        env={"COLUMNS": "240"},
        color=False,
    )

    assert list_result.exit_code == 0
    assert "demo_yiu_scaffold" in list_result.output
    assert "yiu" in list_result.output


def test_workspaces_list_discovers_spec_only_yiu_workspace_via_family_registry(tmp_path: Path) -> None:
    workspace_root, spec_path = _write_yiu_workspace(tmp_path)
    validate_result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path)], color=False)

    assert validate_result.exit_code == 0
    list_result = runner.invoke(
        app,
        ["workspaces", "list", "--root", str(workspace_root.parent)],
        env={"COLUMNS": "240"},
        color=False,
    )

    assert list_result.exit_code == 0
    assert "demo_yiu" in list_result.output
    assert "family-spec" in list_result.output
    assert "yiu" in list_result.output


def test_yiu_init_workspace_scaffolded_spec_validates(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "demo_yiu_scaffold"

    result = runner.invoke(app, ["yiu", "init-workspace", "--output", str(workspace_root)], color=False)

    assert result.exit_code == 0
    validate_result = runner.invoke(
        app,
        ["yiu", "validate", "--spec", str(workspace_root / "configs" / "yiu" / "example.yiu.yaml"), "--json"],
        color=False,
    )

    assert validate_result.exit_code == 0
    payload = json.loads(validate_result.output)
    assert payload["status"] == "satisfied"
    assert len(payload["metadata"]["catalog_paths"]) == 3


def test_yiu_validate_v2_reports_template_state_order_and_compound_payload_projection(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_v2_workspace(tmp_path)

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert report["protocol_template"] == "msd_hop_retron_eco1_v1"
    assert report["sequence_mode"] == "concrete"
    assert [state["state_id"] for state in report["states"]] == [
        "source_oligo_ssdna",
        "source_amplicon_dsdna",
        "post_double_nicking_fragment_pool",
        "post_heat_cleanup_fragment_pool",
        "adapter_annealed_complex",
        "ligated_ssdna_hairpin",
        "hairpin_pcr_linear_insert",
    ]
    hairpin_insert = next(state for state in report["states"] if state["state_id"] == "hairpin_pcr_linear_insert")
    assert hairpin_insert["topology_kind"] == "linear_dsdna"
    compound_payload = next(
        annotation for annotation in hairpin_insert["annotations"] if annotation["id"] == "assembled_payload"
    )
    assert compound_payload["projection_kind"] == "compound"
    assert compound_payload["assembled_coordinate_space"] == "post_ligation"
    assert compound_payload["pieces"] == [
        {"segment_id": "retained_left", "start": 0, "end": 4},
        {"segment_id": "retained_right", "start": 4, "end": 8},
    ]


def test_yiu_validate_v2_reports_pattern_evidence_honestly_for_iupac_inputs(tmp_path: Path) -> None:
    _workspace, spec_path = _write_yiu_v2_workspace(
        tmp_path,
        source_sequence="AAAAGGTCTCACGTTTAANGGGCCGGGGTCTCACGTTTTT",
    )

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 1
    report = json.loads(result.output)
    assert report["sequence_mode"] == "pattern"
    assert report["validation_mode"] == "pattern_compatibility"
    assert any(issue["code"] == "PATTERN_CHECK_NOT_GUARANTEED" for issue in report["issues"])
    digest_state = next(state for state in report["states"] if state["state_id"] == "post_double_nicking_fragment_pool")
    assert digest_state["sequence_mode"] == "pattern"
    assert digest_state["pattern_evidence_summary"]["possible_checks"] >= 1


def test_yiu_design_v2_writes_additive_view_contract_and_show_reports_it(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_v2_workspace(tmp_path)

    design_result = runner.invoke(app, ["yiu", "design", "--spec", str(spec_path)], color=False)

    assert design_result.exit_code == 0
    run_root = workspace / "outputs" / "yiu" / "explicit" / "demo_yiu_v2"
    run_dir = next(run_root.iterdir())
    views_manifest = json.loads((run_dir / "yiu_published_views_manifest.json").read_text(encoding="utf-8"))
    state_view = json.loads(
        (run_dir / "published" / "views" / "hairpin_pcr_linear_insert.json").read_text(encoding="utf-8")
    )

    assert views_manifest["schema_version"] == 2
    assert views_manifest["view_contract_version"] == 2
    assert views_manifest["protocol_template"] == "msd_hop_retron_eco1_v1"
    assert state_view["view_contract_version"] == 2
    assert state_view["state_kind"] == "hairpin_pcr_linear_insert"
    assert state_view["topology_kind"] == "linear_dsdna"
    assert "segments" in state_view
    assert "annotations" in state_view
    assert "junctions" in state_view
    assert state_view["sequence_mode"] == "concrete"

    show_result = runner.invoke(app, ["yiu", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 0
    assert "View contract -> 2" in show_result.output
    assert "Protocol template -> msd_hop_retron_eco1_v1" in show_result.output


def test_yiu_validate_v2_includes_optional_cleanup_and_cloning_states_only_when_enabled(tmp_path: Path) -> None:
    workspace, spec_path = _write_yiu_v2_workspace(tmp_path)
    payload = _yiu_v2_payload(workflow_scope="insert_plus_backbone_cloning")
    payload["yiu"]["steps"]["insert_cleanup"]["enabled"] = True
    payload["yiu"]["steps"]["backbone_pcr"] = {
        "enabled": True,
        "backbone_id": "demo_backbone",
        "forward_primer_id": "oES790",
        "reverse_primer_id": "oES791",
    }
    payload["yiu"]["steps"]["golden_gate_assembly"] = {
        "enabled": True,
        "enzyme": "BsaI",
        "backbone_id": "demo_backbone",
    }
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    _write_yaml(
        workspace / "catalogs" / "backbones.yaml",
        {"backbones": {"entries": [{"id": "demo_backbone", "sequence": "GGTCTCAGATCGGA"}]}},
    )

    result = runner.invoke(app, ["yiu", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    report = json.loads(result.output)
    assert [state["state_id"] for state in report["states"]][-4:] == [
        "post_insert_cleanup_linear_insert",
        "backbone_amplicon",
        "assembly_reaction",
        "assembled_plasmid_candidate",
    ]
