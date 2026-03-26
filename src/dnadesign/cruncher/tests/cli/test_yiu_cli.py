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
                    {"id": "left_fold", "start": 10, "end": 14},
                    {"id": "right_fold", "start": 32, "end": 36},
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
    pcr_state = next(state for state in payload["states"] if state["state_id"] == "pcr_linear_duplex")
    assert pcr_state["metadata"]["amplicon_start"] == 0
    assert pcr_state["metadata"]["amplicon_end"] == 40
    assert pcr_state["metadata"]["amplicon_length_nt"] == 40
    assert pcr_state["primary_sequence"] == "AAAAGGTCTCACGTTTAAGGGGCCGGGGTCTCACGTTTTT"


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
