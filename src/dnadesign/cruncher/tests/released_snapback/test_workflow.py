"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_workflow.py

Bundle and show-path tests for released-product snapback workflows.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import dnadesign.cruncher.snapback.released_target_search as released_target_search
from dnadesign.cruncher.app.snapback_released_show import released_show_payload
from dnadesign.cruncher.app.snapback_released_solve_workflow import run_released_snapback_solve
from dnadesign.cruncher.app.snapback_released_target_search_workflow import run_released_snapback_target_search
from dnadesign.cruncher.app.snapback_released_workflow import (
    run_released_snapback_design,
    validate_released_snapback_spec,
)
from dnadesign.cruncher.nickases.catalog import load_merged_nickase_catalog
from dnadesign.cruncher.nickases.selection import matching_nickase_warning_codes
from dnadesign.cruncher.release_enzymes.catalog import load_merged_release_enzyme_catalog
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.released_artifacts import RELEASED_SUMMARY_FIELDNAMES
from dnadesign.cruncher.snapback.released_hit_plot import build_released_hit_plot_context
from dnadesign.cruncher.snapback.released_models import (
    ReleaseCatalogSources,
    ReleasedFinalTargetGeometry,
    ReleasedSolveOutputConfig,
    ReleasedTargetSearchConfig,
    SingleNickReleasedTargetSearchRequest,
)


def _write_workspace(tmp_path: Path, *, precursor_top_strand: str = "AACGTTGTTCCAA") -> Path:
    workspace = tmp_path / "workspaces" / "demo_released"
    spec_path = workspace / "configs" / "snapback" / "demo.released.snapback.yaml"
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace / "inputs" / "release_enzymes" / "local.release.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nx.Exact7",
                            "specificity_id": "Nx.Exact7",
                            "motif_top_5to3": "AACGTTG",
                            "top_cut_offset": 0,
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    release_catalog_path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "Re.Exact",
                            "display_name": "Re.Exact",
                            "recognition_sequence": "CCAA",
                            "top_cut_offset": 1,
                            "bottom_cut_offset": 0,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "released_snapback": {
                    "schema_version": 1,
                    "kind": "single_nick_released_snapback_v1",
                    "name": "demo_released",
                },
                "input": {
                    "precursor_top_strand": precursor_top_strand,
                },
                "nick_stage": {
                    "nickase_variant_id": "Nx.Exact7",
                    "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                    "normalized_to_top_strand_nick": True,
                    "require_site_sequence_preserved_pre_nick": True,
                },
                "release_stage": {
                    "release_variant_id": "Re.Exact",
                    "catalog": {"additional_paths": ["inputs/release_enzymes/local.release.yaml"]},
                    "retained_side": "upstream",
                    "stage_order": "nick_then_release",
                    "require_site_sequence_preserved_pre_release": True,
                },
                "final_target": {
                    "nick_boundary_from_left": 0,
                    "paired_bp": 3,
                    "cap_nt": 3,
                },
                "constraints": {
                    "allow_post_release_loss_of_nickase_site": True,
                    "allow_post_release_loss_of_release_site": True,
                    "require_nick_survives_in_retained_product": False,
                    "require_release_site_downstream_of_nick": True,
                    "require_complete_downstream_fragment_separation": True,
                },
                "output": {"run_dir": "outputs/released_design"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return spec_path


def test_released_design_writes_bundle_and_released_show_revalidates_it(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, report = run_released_snapback_design(spec_path)

    assert report.status == "satisfied"
    assert (run_dir / "meta" / "released_snapback_manifest.json").exists()
    assert (run_dir / "meta" / "released_snapback_status.json").exists()
    assert (run_dir / "analysis" / "report.json").exists()
    assert (run_dir / "analysis" / "released_product_projection.json").exists()
    assert (run_dir / "analysis" / "pre_nick_site.json").exists()
    assert (run_dir / "analysis" / "release_site.json").exists()
    assert (run_dir / "export" / "released_design_summary.csv").exists()

    payload = released_show_payload(run_dir)

    assert payload["kind"] == "released_explicit"
    assert payload["status"] == "satisfied"
    projection_payload = json.loads(
        (run_dir / "analysis" / "released_product_projection.json").read_text(encoding="utf-8")
    )
    assert projection_payload["release_top_cut_precursor"] == 10
    assert projection_payload["release_bottom_cut_precursor"] == 9


def test_released_design_rejects_left_of_origin_outside_site_exact_bundle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "de033"
    spec_path = workspace / "configs" / "snapback" / "de033.released.snapback.yaml"
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace / "inputs" / "release_enzymes" / "local.release.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.BsmAI",
                            "specificity_id": "BsmAI",
                            "motif_top_5to3": "GTCTC",
                            "top_cut_offset": 6,
                            "selection": {"outside_site": True},
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    release_catalog_path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "Re.Exact",
                            "display_name": "Re.Exact",
                            "recognition_sequence": "CCAA",
                            "top_cut_offset": 1,
                            "bottom_cut_offset": 0,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "released_snapback": {
                    "schema_version": 1,
                    "kind": "single_nick_released_snapback_v1",
                    "name": "de033_left_prefix",
                },
                "input": {
                    "precursor_top_strand": "GTCTCAAACGTTGTTCCAA",
                },
                "nick_stage": {
                    "nickase_variant_id": "Nt.BsmAI",
                    "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                    "intended_site_sequence": "GTCTC",
                    "normalized_to_top_strand_nick": True,
                    "require_site_sequence_preserved_pre_nick": True,
                },
                "release_stage": {
                    "release_variant_id": "Re.Exact",
                    "catalog": {"additional_paths": ["inputs/release_enzymes/local.release.yaml"]},
                    "intended_site_sequence": "CCAA",
                    "retained_side": "upstream",
                    "stage_order": "nick_then_release",
                    "require_site_sequence_preserved_pre_release": True,
                },
                "final_target": {
                    "nick_boundary_from_left": 0,
                    "paired_bp": 3,
                    "cap_nt": 3,
                },
                "constraints": {
                    "allow_post_release_loss_of_nickase_site": True,
                    "allow_post_release_loss_of_release_site": True,
                    "require_nick_survives_in_retained_product": False,
                    "require_release_site_downstream_of_nick": True,
                    "require_complete_downstream_fragment_separation": True,
                },
                "output": {"run_dir": "outputs/released_design"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    run_dir, report = run_released_snapback_design(spec_path)

    assert report.status == "invalid_precursor"
    assert any(issue.code == "PRE_NICK_SITE_LEFT_OF_ORIGIN" for issue in report.issues)
    payload = released_show_payload(run_dir)
    assert payload["status"] == "invalid_precursor"


def test_released_show_uses_snapshots_when_source_spec_drifts_or_is_missing(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    spec_path.write_text("# drift\n", encoding="utf-8")
    payload = released_show_payload(run_dir)
    assert payload["status"] == "satisfied"
    spec_path.unlink()

    payload = released_show_payload(run_dir)
    assert payload["status"] == "satisfied"


def test_released_show_detects_report_run_dir_drift(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    report_path = run_dir / "analysis" / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["run_dir"] = "/tmp/drifted"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product report run_dir drift detected."):
        released_show_payload(run_dir)


def test_released_show_detects_summary_csv_drift(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    summary_path = run_dir / "export" / "released_design_summary.csv"
    summary_path.write_text(
        ",".join(RELEASED_SUMMARY_FIELDNAMES) + "\nbad,drifted,Nx.Bad,Re.Bad,99,99,99,99,99,99,99,99\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Released-product summary CSV content drift detected."):
        released_show_payload(run_dir)


def test_released_show_rejects_hollowed_satisfied_bundle(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    report_path = run_dir / "analysis" / "report.json"
    projection_path = run_dir / "analysis" / "released_product_projection.json"
    pre_nick_path = run_dir / "analysis" / "pre_nick_site.json"
    release_path = run_dir / "analysis" / "release_site.json"
    summary_path = run_dir / "export" / "released_design_summary.csv"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["candidate"] = None
    report_payload["projection"] = None
    report_payload["pre_nick_site"] = None
    report_payload["pre_nick_event"] = None
    report_payload["release_site"] = None
    report_payload["release_event"] = None
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    projection_path.write_text("null\n", encoding="utf-8")
    pre_nick_path.write_text(json.dumps({"site": None, "event": None}, indent=2), encoding="utf-8")
    release_path.write_text(json.dumps({"site": None, "event": None}, indent=2), encoding="utf-8")
    summary_path.write_text(",".join(RELEASED_SUMMARY_FIELDNAMES) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product satisfied report candidate drift detected."):
        released_show_payload(run_dir)


def test_released_show_detects_status_issue_count_drift(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    status_path = run_dir / "meta" / "released_snapback_status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["issue_count"] = 99
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product status/report issue_count drift detected."):
        released_show_payload(run_dir)


def test_released_show_detects_report_final_target_drift_for_candidate_free_bundle(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path, precursor_top_strand="AACGTTG")

    run_dir, report = run_released_snapback_design(spec_path)
    assert report.candidate is None
    report_path = run_dir / "analysis" / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["metadata"]["final_target"] = {
        "nick_boundary_from_left": 99,
        "paired_bp": 7,
        "cap_nt": 3,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product report final_target drift detected."):
        released_show_payload(run_dir)


def test_released_show_detects_catalog_source_spoofing(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    report_path = run_dir / "analysis" / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["metadata"]["nick_catalog_source"] = "preset:spoofed"
    report_payload["metadata"]["release_catalog_source"] = "preset:spoofed"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product report nick_catalog_source drift detected."):
        released_show_payload(run_dir)


def test_released_show_detects_disallowed_warning_code_policy_drift(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    report_path = run_dir / "analysis" / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["metadata"]["disallowed_nickase_warning_codes"] = []
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product report disallowed_nickase_warning_codes drift detected."):
        released_show_payload(run_dir)


@pytest.mark.parametrize(
    ("relative_path", "replacement_text", "expected_message"),
    [
        ("meta/released_snapback_manifest.json", "[]\n", "Released-product manifest must be a JSON object."),
        ("meta/released_snapback_status.json", "[]\n", "Released-product status record must be a JSON object."),
        ("analysis/report.json", "[]\n", "Released-product report must be a JSON object."),
    ],
)
def test_released_show_rejects_non_object_json(
    tmp_path: Path,
    relative_path: str,
    replacement_text: str,
    expected_message: str,
) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    (run_dir / relative_path).write_text(replacement_text, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        released_show_payload(run_dir)


@pytest.mark.parametrize(
    ("relative_path", "replacement_text", "expected_message"),
    [
        ("provenance/spec.snapshot.yaml", "# drift\n", "Released-product spec snapshot integrity drift detected."),
        (
            "provenance/nickase_catalog.yaml",
            "nickases: {schema_version: 1, entries: []}\n",
            "Released-product nickase catalog snapshot integrity drift detected.",
        ),
        (
            "provenance/release_catalog.yaml",
            "release_enzymes: {schema_version: 1, entries: []}\n",
            "Released-product release catalog snapshot integrity drift detected.",
        ),
    ],
)
def test_released_show_detects_provenance_snapshot_drift(
    tmp_path: Path,
    relative_path: str,
    replacement_text: str,
    expected_message: str,
) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    (run_dir / relative_path).write_text(replacement_text, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        released_show_payload(run_dir)


def test_released_solve_materializes_hits_and_emits_per_hit_plots(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)
    workspace_root = spec_path.parent.parent.parent

    run_dir, report = run_released_snapback_solve(
        request=SingleNickReleasedTargetSearchRequest(
            target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
            nick_sources=CatalogSources(additional_paths=[Path("inputs/nickases/local.nickases.yaml")]),
            release_sources=ReleaseCatalogSources(additional_paths=[Path("inputs/release_enzymes/local.release.yaml")]),
            search=ReleasedTargetSearchConfig(max_results=2, near_boundary_search_limit=2),
        ),
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve"),
            materialize_top_k=2,
            render_format="pdf",
            emit_renders=True,
        ),
        workspace_root=workspace_root,
        force_overwrite=True,
    )

    assert report.status == "exact_hits_materialized"
    assert report.metadata.materialized_hit_count == 1
    assert report.metadata.selected_hit_kind == "exact"
    assert report.metadata.evaluated_pair_count > 0
    assert report.issues == []
    assert (run_dir / "meta" / "released_solve_manifest.json").exists()
    assert (run_dir / "meta" / "released_solve_status.json").exists()
    assert (run_dir / "analysis" / "solve_report.json").exists()
    assert (run_dir / "export" / "table__hits.csv").exists()
    for hit in report.hits:
        hit_run_dir = workspace_root / hit.materialized_run_dir
        assert hit.render_job_path is None
        assert hit.rendered_plot_path is not None
        assert hit_run_dir.exists()
        assert (workspace_root / hit.rendered_plot_path).exists()
        assert (hit_run_dir / "analysis" / "target_search_hit.json").exists()
        assert (hit_run_dir / "analysis" / "released_hit_plot_context.json").exists()
        assert (hit_run_dir / "analysis" / "released_product_projection.json").exists()
        assert (hit_run_dir / "analysis" / "pre_nick_site.json").exists()
        assert (hit_run_dir / "analysis" / "release_site.json").exists()
    first_context = json.loads(
        (
            workspace_root / report.hits[0].materialized_run_dir / "analysis" / "released_hit_plot_context.json"
        ).read_text(encoding="utf-8")
    )
    assert first_context["foldback"]["foldback_sequence"] == "CAA"
    assert first_context["foldback"]["foldback_partner_sequence"] == "AAC"


def test_released_design_rejects_frequent_cutter_nickase_by_default(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)
    nick_catalog_path = spec_path.parent.parent.parent / "inputs" / "nickases" / "local.nickases.yaml"
    nick_payload = yaml.safe_load(nick_catalog_path.read_text(encoding="utf-8"))
    nick_payload["nickases"]["entries"][0]["selection"] = {"warning_codes": ["FREQUENT_CUTTER"]}
    nick_catalog_path.write_text(yaml.safe_dump(nick_payload, sort_keys=False), encoding="utf-8")

    report = validate_released_snapback_spec(spec_path)

    assert report.status == "invalid_catalog"
    assert report.issues[0].code == "DISALLOWED_NICKASE_WARNING_CODE"
    assert report.metadata.disallowed_nickase_warning_codes == ["FREQUENT_CUTTER"]


def test_released_solve_real_presets_materializes_exact_hits_with_bottom_strand_context(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces" / "de033"
    workspace_root.mkdir(parents=True, exist_ok=True)
    request = SingleNickReleasedTargetSearchRequest(
        target=ReleasedFinalTargetGeometry(nick_boundary_from_left=0, paired_bp=3, cap_nt=3),
        nick_sources=CatalogSources(preset="neb_nicking_v1", additional_presets=["thermo_nicking_v1"]),
        release_sources=ReleaseCatalogSources(preset="type_iis_release_v1"),
        search=ReleasedTargetSearchConfig(max_results=8, near_boundary_search_limit=8),
    )

    search_report = run_released_snapback_target_search(
        request=request,
        workspace_root=workspace_root,
    )
    nick_catalog, _nick_paths = load_merged_nickase_catalog(
        preset_id="neb_nicking_v1",
        additional_preset_ids=["thermo_nicking_v1"],
        additional_paths=[],
        workspace_root=workspace_root,
    )
    release_catalog, _release_paths = load_merged_release_enzyme_catalog(
        preset_id="type_iis_release_v1",
        additional_paths=[],
        workspace_root=workspace_root,
    )
    disallowed_nick_placement_count = len(
        [
            placement
            for placement in released_target_search._nick_placements(
                nick_catalog,
                normalize_to_top_strand_nick=True,
            )
            if matching_nickase_warning_codes(
                placement.entry,
                warning_codes=request.search.disallowed_nickase_warning_codes,
            )
        ]
    )
    release_placement_count = len(
        released_target_search._release_placements(
            release_catalog,
            target=request.target,
        )
    )
    assert search_report.status == "near_hits_only"
    assert search_report.metadata.pre_truncation_exact_hit_count == 0
    assert search_report.metadata.disallowed_nickase_warning_codes == ["FREQUENT_CUTTER"]
    assert (
        search_report.metadata.blocker_counts["DISALLOWED_NICKASE_WARNING_CODE"]
        == disallowed_nick_placement_count * release_placement_count
    )
    assert not search_report.exact_hits
    assert search_report.near_hits
    assert all(
        hit.pre_nick_site.local_start is not None and hit.pre_nick_site.local_start >= 0
        for hit in search_report.near_hits
    )

    run_dir, solve_report = run_released_snapback_solve(
        request=request,
        output=ReleasedSolveOutputConfig(
            run_dir=Path("outputs/released_solve"),
            materialize_top_k=8,
            render_format="pdf",
            emit_renders=False,
        ),
        workspace_root=workspace_root,
        force_overwrite=True,
    )

    assert run_dir.exists()
    assert solve_report.status == "near_hits_materialized"
    assert solve_report.metadata.materialized_hit_count == min(8, len(search_report.near_hits))
    assert solve_report.metadata.available_exact_hit_count == 0
    assert solve_report.metadata.selected_hit_kind == "nearest"
    assert solve_report.hits[0].nickase_variant_id != "Nt.BspQI"
    assert solve_report.hits[0].release_variant_id == "BsaI-HFv2"
    assert solve_report.hits[0].target_search_hit.sacrificial_downstream_tail_nt == 7
    plot_context = build_released_hit_plot_context(solve_report.hits[0].target_search_hit)
    assert plot_context["precursor"]["nick_site"]["local_start"] >= 0
    assert plot_context["precursor"]["nick_site"]["local_end"] >= 0
    assert plot_context["released_product"]["retained_top_span"]["start"] >= 0
    assert plot_context["released_product"]["active_bottom_span"]["start"] >= 0
    assert plot_context["released_product"]["duplex_overlap_span"]["start"] >= 0
    assert plot_context["released_product"]["duplex_mismatch_positions"] == []
    assert plot_context["foldback"]["foldback_mismatch_positions"] == []
    first_context = json.loads(
        (
            workspace_root / solve_report.hits[0].materialized_run_dir / "analysis" / "released_hit_plot_context.json"
        ).read_text(encoding="utf-8")
    )
    assert first_context["precursor"]["nick_site"]["local_start"] >= 0
    assert first_context["precursor"]["nick_site"]["local_end"] >= 0
    assert first_context["released_product"]["retained_top_span"]["start"] >= 0
    assert first_context["released_product"]["active_bottom_span"]["start"] >= 0
    assert first_context["released_product"]["duplex_overlap_span"]["start"] >= 0
    assert first_context["released_product"]["duplex_mismatch_positions"] == []
    assert first_context["released_product"]["nick_origin_boundary"] >= 0
    assert (
        first_context["released_product"]["retained_top_span"]["end"]
        == first_context["released_product"]["nick_origin_boundary"]
    )
    assert first_context["released_product"]["nickase_site_survives_post_release"] is False
