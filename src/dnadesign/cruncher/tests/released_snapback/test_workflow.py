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

from dnadesign.cruncher.app.snapback_released_show import released_show_payload
from dnadesign.cruncher.app.snapback_released_workflow import run_released_snapback_design
from dnadesign.cruncher.snapback.released_artifacts import RELEASED_SUMMARY_FIELDNAMES


def _write_workspace(tmp_path: Path) -> Path:
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
                            "top_cut_offset": 0,
                            "bottom_cut_offset": 1,
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
                    "precursor_top_strand": "AACGTTGTTCCAA",
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
                    "require_nick_survives_in_retained_product": True,
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
    assert projection_payload["release_top_cut_precursor"] == 9


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


def test_released_show_detects_status_issue_count_drift(tmp_path: Path) -> None:
    spec_path = _write_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(spec_path)
    status_path = run_dir / "meta" / "released_snapback_status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["issue_count"] = 99
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product status/report issue_count drift detected."):
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
