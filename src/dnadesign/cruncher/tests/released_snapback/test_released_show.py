"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/released_snapback/test_released_show.py

Focused released-product snapback readback and drift tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dnadesign.cruncher.app.snapback_released_show import released_show_payload
from dnadesign.cruncher.app.snapback_released_workflow import run_released_snapback_design
from dnadesign.cruncher.snapback.released_artifacts import RELEASED_SUMMARY_FIELDNAMES
from dnadesign.cruncher.tests.released_snapback.builders import write_released_workspace


def test_released_show_uses_snapshots_when_source_spec_drifts_or_is_missing(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
    fixture.spec_path.write_text("# drift\n", encoding="utf-8")
    payload = released_show_payload(run_dir)
    assert payload["status"] == "satisfied"
    assert payload["final_target"] == {"nick_boundary_from_left": 0, "paired_bp": 3, "cap_nt": 3}
    assert payload["final_geometry_source"] == "exposed_bottom_strand"
    assert str(payload["nick_catalog_source"]).endswith("inputs/nickases/local.nickases.yaml")
    assert str(payload["release_catalog_source"]).endswith("inputs/release_enzymes/local.release.yaml")
    assert Path(str(payload["spec_snapshot"])).exists()
    assert Path(str(payload["nickase_catalog_snapshot"])).exists()
    assert Path(str(payload["release_catalog_snapshot"])).exists()
    fixture.spec_path.unlink()

    payload = released_show_payload(run_dir)
    assert payload["status"] == "satisfied"


def test_released_show_detects_report_run_dir_drift(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
    report_path = run_dir / "analysis" / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["run_dir"] = "/tmp/drifted"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product report run_dir drift detected."):
        released_show_payload(run_dir)


def test_released_show_detects_summary_csv_drift(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
    summary_path = run_dir / "export" / "released_design_summary.csv"
    summary_path.write_text(
        ",".join(RELEASED_SUMMARY_FIELDNAMES) + "\nbad,drifted,Nx.Bad,Re.Bad,99,99,99,99,99,99,99,99,99,99\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Released-product summary CSV content drift detected."):
        released_show_payload(run_dir)


def test_released_show_rejects_hollowed_satisfied_bundle(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
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
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
    status_path = run_dir / "meta" / "released_snapback_status.json"
    status_payload = json.loads(status_path.read_text(encoding="utf-8"))
    status_payload["issue_count"] = 99
    status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product status/report issue_count drift detected."):
        released_show_payload(run_dir)


def test_released_show_detects_report_final_target_drift_for_candidate_free_bundle(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path, precursor_top_strand="AACGTTG")

    run_dir, report = run_released_snapback_design(fixture.spec_path)
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
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
    report_path = run_dir / "analysis" / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["metadata"]["nick_catalog_source"] = "preset:spoofed"
    report_payload["metadata"]["release_catalog_source"] = "preset:spoofed"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product report nick_catalog_source drift detected."):
        released_show_payload(run_dir)


def test_released_show_detects_disallowed_warning_code_policy_drift(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
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
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
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
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
    (run_dir / relative_path).write_text(replacement_text, encoding="utf-8")

    with pytest.raises(ValueError, match=expected_message):
        released_show_payload(run_dir)


def test_released_show_detects_candidate_route_family_drift(tmp_path: Path) -> None:
    fixture = write_released_workspace(tmp_path)

    run_dir, _report = run_released_snapback_design(fixture.spec_path)
    report_path = run_dir / "analysis" / "report.json"
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    report_payload["candidate"]["route_family"] = "top_active_from_bottom_nick"
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Released-product report payload drift detected."):
        released_show_payload(run_dir)
