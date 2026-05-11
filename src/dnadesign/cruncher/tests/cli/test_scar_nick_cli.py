"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_scar_nick_cli.py

CLI smoke tests for the scar-nick command group.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

import dnadesign.baserender as baserender
from dnadesign.cruncher.cli.app import app

runner = CliRunner()


def _write_spec(tmp_path: Path, *, materialize_top_k: int = 8) -> tuple[Path, Path]:
    workspace = tmp_path / "workspaces" / "demo_scar_nick"
    nick_catalog = workspace / "inputs" / "nickases" / "terminal.nickases.yaml"
    nick_catalog.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Test.TerminalBottomNickase",
                            "specificity_id": "TerminalBottomNickase",
                            "motif_top_5to3": "GGTCTCGNNNN",
                            "vendor_diagram_top_5to3": "GGTCTCGNNNN",
                            "bottom_cut_offset": 11,
                            "vendor": "dnadesign test fixture",
                            "source_url": "https://example.invalid/dnadesign/scar-nick-terminal-fixture",
                            "source_family": "nicking_endonuclease",
                            "commercial_confidence": "primary_vendor_current",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path = workspace / "configs" / "scar_nick" / "teto_upstream_processing.scar_nick.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        yaml.safe_dump(
            {
                "scar_nick": {
                    "schema_version": 1,
                    "contract": "terminal_type_iis_scar_nick_v1",
                    "name": "teto_upstream_processing",
                },
                "junction": {
                    "left_base": "CGGG",
                    "right_base": "ACAG",
                    "profile_order": "S3_S2_S1_S0",
                    "s0_match_required": True,
                    "overhang_length": 4,
                },
                "processing": {
                    "release": {
                        "variant_id": "BsaI-HFv2",
                        "catalog": {"preset": "type_iis_release_v1"},
                        "required_terminal_scar_nt": 4,
                        "recognition_site_must_be_excised": True,
                    },
                    "nick": {
                        "target_strand": "bottom",
                        "terminal_nick_required": True,
                        "downstream_protected_nt_allowed": 0,
                        "downstream_must_be_degenerate": True,
                        "catalog": {"additional_paths": ["inputs/nickases/terminal.nickases.yaml"]},
                    },
                },
                "ranking_context": {
                    "anchor_mode": "profile_analog",
                    "optional_reference_profiles": {
                        "working_control": {
                            "id": "retron_26",
                            "left_base": "CGGG",
                            "right_base": "ACAG",
                            "profile_s3s2s1s0": "MXMX",
                        }
                    },
                    "target_profile_buckets": [
                        "MXMM",
                        "WXMM",
                        "XWMM",
                        "MWXM",
                        "MXWM",
                        "XMWM",
                        "WMMM",
                        "MWMM",
                        "MMWM",
                        "WWMM",
                        "WMWM",
                        "MWWM",
                        "XXMM",
                        "XMXM",
                    ],
                    "reject_profiles": ["MMMM"],
                    "allow_gt_wobble": True,
                    "active_max_hard_mismatches": 2,
                    "active_max_non_watson_crick_pairs": 2,
                    "forbid_active_middle_middle_double_hard": True,
                    "min_ligation_support": 2.0,
                    "max_effective_disruption": 2.5,
                    "reduce_gc_when_tied": True,
                },
                "search": {
                    "mode": "curated_panel",
                    "max_hits": 16,
                    "materialize_top_k": materialize_top_k,
                },
                "output": {"run_dir": "outputs/scar_nick/teto_upstream_processing"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace, spec_path


def test_root_help_includes_scar_nick_group() -> None:
    result = runner.invoke(app, ["--help"], color=False)

    assert result.exit_code == 0
    assert "scar-nick" in result.output


def test_scar_nick_validate_is_read_only(tmp_path: Path) -> None:
    workspace, spec_path = _write_spec(tmp_path)

    result = runner.invoke(app, ["scar-nick", "validate", "--spec", str(spec_path), "--json"], color=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "satisfied"
    assert not (workspace / "outputs").exists()


def test_scar_nick_design_writes_bundle_and_show_reads_it(tmp_path: Path) -> None:
    workspace, spec_path = _write_spec(tmp_path)

    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    run_dir = Path(payload["run_dir"])
    assert run_dir == workspace / "outputs" / "scar_nick" / "teto_upstream_processing"
    assert (run_dir / "meta" / "scar_nick_manifest.json").exists()
    assert (run_dir / "meta" / "scar_nick_status.json").exists()
    assert (run_dir / "provenance" / "spec.snapshot.yaml").exists()
    assert (run_dir / "provenance" / "nickase_catalog.yaml").exists()
    assert (run_dir / "provenance" / "release_catalog.yaml").exists()
    assert (run_dir / "analysis" / "report.json").exists()
    assert (run_dir / "analysis" / "report.md").exists()
    assert (run_dir / "analysis" / "candidate_profiles.json").exists()
    assert (run_dir / "analysis" / "nickase_geometry_audit.json").exists()
    assert (run_dir / "export" / "table__scar_nick_candidates.csv").exists()
    candidate_rows = list(
        csv.DictReader((run_dir / "export" / "table__scar_nick_candidates.csv").open(encoding="utf-8"))
    )
    assert set(candidate_rows[0]) >= {
        "release_variant_id",
        "release_recognition_sequence",
        "release_top_cut_boundary",
        "release_bottom_cut_boundary",
        "nickase_variant_id",
        "nickase_motif_top_5to3",
        "nickase_strand",
        "nickase_exact_terminal",
        "non_watson_crick_count",
        "non_wc_count",
    }
    assert candidate_rows[0]["release_recognition_sequence"]
    assert candidate_rows[0]["nickase_motif_top_5to3"]
    pair_call_table = run_dir / "export" / "table__scar_nick_candidate_pair_calls.csv"
    assert pair_call_table.exists()
    assert (run_dir / "export" / "table__scar_nick_nickase_geometry_audit.csv").exists()
    manifest = json.loads((run_dir / "meta" / "scar_nick_manifest.json").read_text(encoding="utf-8"))
    artifact_names = {artifact["name"] for artifact in manifest["artifacts"]}
    assert "nickase_geometry_audit" in artifact_names
    assert "candidate_pair_call_table" in artifact_names
    assert "nickase_geometry_audit_table" in artifact_names
    pair_call_rows = list(csv.DictReader(pair_call_table.open(encoding="utf-8")))
    assert len(pair_call_rows) == 4 * payload["metadata"]["accepted_candidate_count"]
    assert set(pair_call_rows[0]) >= {
        "rank",
        "left_base",
        "right_base",
        "profile_order",
        "profile_policy_status",
        "profile_policy_reason",
        "site",
        "left_nt",
        "right_nt",
        "aligned_right_nt",
        "pair_identity",
        "aligned_pair_identity",
        "class_label",
        "is_watson_crick",
        "is_wobble",
        "is_hard_mismatch",
        "non_watson_crick_count",
        "non_wc_count",
    }
    assert all(row["non_wc_count"] == row["non_watson_crick_count"] for row in candidate_rows)
    assert {row["site"] for row in pair_call_rows if row["rank"] == "1"} == {"S3", "S2", "S1", "S0"}
    assert any(row["class_label"] in {"W", "X"} for row in pair_call_rows)
    assert all(row["pair_identity"] == f"{row['left_nt']}:{row['right_nt']}" for row in pair_call_rows)
    assert all(row["aligned_pair_identity"] == f"{row['left_nt']}:{row['aligned_right_nt']}" for row in pair_call_rows)
    assert all(row["non_wc_count"] == row["non_watson_crick_count"] for row in pair_call_rows)

    show_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)
    assert show_result.exit_code == 0
    assert "teto_upstream_processing" in show_result.output
    assert "Report JSON ->" in show_result.output
    assert "Candidate table ->" in show_result.output
    assert "Candidate pair-call table ->" in show_result.output
    assert "Nickase geometry audit ->" in show_result.output


def test_scar_nick_design_writes_unique_terminal_nick_visuals_and_baserender_job(tmp_path: Path) -> None:
    workspace, spec_path = _write_spec(tmp_path, materialize_top_k=2)

    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )

    assert result.exit_code == 0
    run_dir = Path(json.loads(result.output)["run_dir"])
    views_dir = run_dir / "analysis" / "views"
    post_path = views_dir / "post_terminal_nick.scar_nick_visual.v1.json"
    jsonl_path = views_dir / "scar_nick_terminal_nick.scar_nick_visual.v1.jsonl"
    job_path = run_dir / "baserender_jobs" / "scar_nick_terminal_nick.job.yaml"

    assert (views_dir / "views_manifest.v1.json").exists()
    assert not (views_dir / "pre_terminal_nick.scar_nick_visual.v1.json").exists()
    assert post_path.exists()
    assert jsonl_path.exists()
    assert job_path.exists()

    post = json.loads(post_path.read_text(encoding="utf-8"))
    assert post["contract_kind"] == "scar_nick_visual_v1"
    assert post["state_kind"] == "pre_post_terminal_nick"
    assert post["event_scope"] == "terminal_nick"
    assert post["nick_state"] == "pre_post"
    assert [panel["panel_id"] for panel in post["panels"]] == ["pre_release", "post_release"]
    assert post["panels"][1]["fragment_spans"][0]["row"] in {"primary", "complement"}
    assert post["title"] and f"L={post['left_base']}/R={post['right_base']}" in post["title"]

    release_span = post["release_site_span"]
    assert post["primary_sequence"][release_span["start"] : release_span["end"]] == "GGTCTC"
    assert set(post["primary_sequence"][post["terminal_boundary"] :]) <= {"N"}
    assert post["junction_partner_span"] is None
    assert post["release_placement"]["variant_id"] == "BsaI-HFv2"
    assert post["nickase"]["variant_id"] == "Test.TerminalBottomNickase"
    assert post["nickase"]["orientation"] == "forward"
    assert post["nickase"]["canonical_read_row"] == "primary"
    assert post["nickase"]["recognition_nt"] == 7
    assert post["nickase"]["source_family"] == "nicking_endonuclease"
    assert post["nickase"]["vendor"] == "dnadesign test fixture"
    assert post["nickase"]["strand"] == "bottom"
    assert post["nickase"]["exact_terminal"] is True
    assert post["terminal_boundary"] == post["retained_scar_span"]["end"]
    assert post["nick_boundary"] == post["terminal_boundary"]
    assert post["retained_scar_span"]["end"] - post["retained_scar_span"]["start"] == 4
    fills_by_semantic: dict[str, list[dict[str, object]]] = {}
    for fill in post["rectangular_fills"]:
        fills_by_semantic.setdefault(fill["semantic"], []).append(fill)
    assert set(fills_by_semantic) >= {"type_iis_release_site", "retained_type_iis_scar", "nickase_footprint"}
    assert "junction_partner" not in fills_by_semantic
    scar_fill = next(
        fill
        for fill in fills_by_semantic["retained_type_iis_scar"]
        if fill["start"] == post["retained_scar_span"]["start"]
    )
    assert scar_fill["semantic"] == "retained_type_iis_scar"
    assert scar_fill["start"] == post["retained_scar_span"]["start"]
    assert scar_fill["end"] == post["retained_scar_span"]["end"]
    assert scar_fill["cover_rows"] == "both"
    assert scar_fill["corner_radius"] == 0.0
    assert any(
        fill["start"] == post["release_site_span"]["start"] and fill["end"] == post["release_site_span"]["end"]
        for fill in fills_by_semantic["type_iis_release_site"]
    )
    assert len(fills_by_semantic["nickase_footprint"]) == 1
    assert fills_by_semantic["nickase_footprint"][0]["start"] == post["panels"][0]["nickase_site_span"]["start"]
    assert fills_by_semantic["nickase_footprint"][0]["end"] == post["panels"][0]["nickase_site_span"]["end"]
    assert post["meta"]["panel_transition_arrows"] == [
        {"start": post["panels"][0]["end"], "end": post["panels"][1]["start"]}
    ]
    assert post["meta"]["processing_event_scope"] == "terminal_nick"
    assert post["meta"]["release_site_role"] == "excised_provenance"
    assert post["meta"]["profile_order"] == "S3_S2_S1_S0"
    assert post["meta"]["type_iis_label"] == "BsaI-HFv2 GGTCTC"
    assert post["meta"]["nickase_label"] == "Test.TerminalBottomNickase GGTCTCGNNNN"
    assert post["meta"]["junction_label"] == ""

    materialized_dirs = sorted((run_dir / "analysis" / "materialized_candidates").iterdir())
    assert [path.name for path in materialized_dirs] == ["candidate_01", "candidate_02"]
    assert (materialized_dirs[0] / "meta" / "scar_nick_candidate_manifest.json").exists()
    assert (materialized_dirs[0] / "analysis" / "views" / "post_terminal_nick.scar_nick_visual.v1.json").exists()
    assert (materialized_dirs[0] / "baserender_jobs" / "scar_nick_terminal_nick.job.yaml").exists()

    jsonl_records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(jsonl_records) == 2
    assert {record["state_kind"] for record in jsonl_records} == {"pre_post_terminal_nick"}
    assert len({record["primary_sequence"] for record in jsonl_records}) == len(jsonl_records)
    assert baserender.validate_job(job_path, kind="render_job_v3", caller_root=job_path.parent)

    show_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)
    assert show_result.exit_code == 0
    assert "Views manifest ->" in show_result.output
    assert "BaseRender job ->" in show_result.output


def test_scar_nick_show_fails_on_missing_report_and_provenance_drift(tmp_path: Path) -> None:
    _workspace, spec_path = _write_spec(tmp_path)
    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )
    assert result.exit_code == 0
    run_dir = Path(json.loads(result.output)["run_dir"])

    report_path = run_dir / "analysis" / "report.json"
    report_path.unlink()
    missing_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)
    assert missing_result.exit_code == 1
    assert "Missing scar-nick report" in missing_result.output

    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )
    assert result.exit_code == 0
    (run_dir / "provenance" / "spec.snapshot.yaml").write_text("scar_nick: drifted\n", encoding="utf-8")
    drift_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)
    assert drift_result.exit_code == 1
    assert "provenance drift" in drift_result.output.lower()


def test_scar_nick_show_fails_on_missing_advertised_visual(tmp_path: Path) -> None:
    _workspace, spec_path = _write_spec(tmp_path)
    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )
    assert result.exit_code == 0
    run_dir = Path(json.loads(result.output)["run_dir"])

    (run_dir / "analysis" / "views" / "post_terminal_nick.scar_nick_visual.v1.json").unlink()

    show_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "Missing scar-nick visual artifact" in show_result.output


def test_scar_nick_show_fails_on_report_content_drift(tmp_path: Path) -> None:
    _workspace, spec_path = _write_spec(tmp_path, materialize_top_k=2)
    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )
    assert result.exit_code == 0
    run_dir = Path(json.loads(result.output)["run_dir"])
    report_path = run_dir / "analysis" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["metadata"]["materialized_candidate_count"] = 0
    report_path.write_text(json.dumps(report), encoding="utf-8")

    show_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "artifact drift for report.json" in show_result.output


def test_scar_nick_show_fails_on_materialized_candidate_payload_drift(tmp_path: Path) -> None:
    _workspace, spec_path = _write_spec(tmp_path, materialize_top_k=2)
    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )
    assert result.exit_code == 0
    run_dir = Path(json.loads(result.output)["run_dir"])
    candidate_json_path = (
        run_dir / "analysis" / "materialized_candidates" / "candidate_01" / "analysis" / "candidate.json"
    )
    candidate_payload = json.loads(candidate_json_path.read_text(encoding="utf-8"))
    candidate_payload["left_base"] = "CCCC"
    candidate_json_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

    show_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "materialized candidate payload drift" in show_result.output


def test_scar_nick_show_fails_on_missing_materialized_candidate_visual(tmp_path: Path) -> None:
    _workspace, spec_path = _write_spec(tmp_path, materialize_top_k=2)
    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )
    assert result.exit_code == 0
    run_dir = Path(json.loads(result.output)["run_dir"])

    (
        run_dir
        / "analysis"
        / "materialized_candidates"
        / "candidate_01"
        / "analysis"
        / "views"
        / "post_terminal_nick.scar_nick_visual.v1.json"
    ).unlink()

    show_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "Missing scar-nick visual artifact" in show_result.output


def test_scar_nick_show_fails_on_visual_manifest_content_drift(tmp_path: Path) -> None:
    _workspace, spec_path = _write_spec(tmp_path, materialize_top_k=2)
    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )
    assert result.exit_code == 0
    run_dir = Path(json.loads(result.output)["run_dir"])
    views_manifest_path = run_dir / "analysis" / "views" / "views_manifest.v1.json"
    views_manifest = json.loads(views_manifest_path.read_text(encoding="utf-8"))
    views_manifest["views"] = views_manifest["views"][:-1]
    views_manifest_path.write_text(json.dumps(views_manifest), encoding="utf-8")

    show_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "views manifest drift" in show_result.output


def test_scar_nick_show_fails_on_baserender_job_content_drift(tmp_path: Path) -> None:
    _workspace, spec_path = _write_spec(tmp_path, materialize_top_k=2)
    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )
    assert result.exit_code == 0
    run_dir = Path(json.loads(result.output)["run_dir"])
    job_path = run_dir / "baserender_jobs" / "scar_nick_terminal_nick.job.yaml"
    job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    job["run"]["strict"] = False
    job_path.write_text(yaml.safe_dump(job), encoding="utf-8")

    show_result = runner.invoke(app, ["scar-nick", "show", "--run", str(run_dir)], color=False)

    assert show_result.exit_code == 1
    assert "BaseRender job drift" in show_result.output


def test_scar_nick_design_fails_fast_without_outputs_for_unsatisfied_spec(tmp_path: Path) -> None:
    workspace, spec_path = _write_spec(tmp_path)
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["processing"]["release"]["variant_id"] = "MissingI"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = runner.invoke(
        app,
        ["scar-nick", "design", "--spec", str(spec_path), "--force-overwrite", "--json"],
        color=False,
    )

    assert result.exit_code == 1
    assert "Scar-nick design is unsatisfied" in result.output
    assert not (workspace / "outputs").exists()
