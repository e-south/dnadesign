"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/snapback/test_show.py

Focused preserved-site Snapback readback and drift tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dnadesign.cruncher.app.snapback_solve_workflow import run_snapback_solve
from dnadesign.cruncher.app.snapback_workflow import run_snapback_design, snapback_show_payload
from dnadesign.cruncher.tests.snapback.builders import write_snapback_workspace


def test_snapback_show_reads_explicit_bundle_without_source_spec(tmp_path: Path) -> None:
    fixture = write_snapback_workspace(tmp_path)

    run_dir, _report = run_snapback_design(fixture.explicit_path)
    fixture.explicit_path.unlink()

    payload = snapback_show_payload(run_dir)
    assert payload["kind"] == "explicit"
    assert payload["status"] == "satisfied"


def test_snapback_show_detects_explicit_foldback_visual_drift(tmp_path: Path) -> None:
    fixture = write_snapback_workspace(tmp_path)

    run_dir, _report = run_snapback_design(fixture.explicit_path)
    foldback_visual_path = run_dir / "analysis" / "views" / "post_nick_foldback.snapback_visual.v1.json"
    foldback_visual = json.loads(foldback_visual_path.read_text(encoding="utf-8"))
    foldback_visual["primary_sequence"] = "AAAAAAA"
    foldback_visual_path.write_text(json.dumps(foldback_visual, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Snapback foldback visual primary_sequence drift detected."):
        snapback_show_payload(run_dir)


def test_snapback_show_detects_views_manifest_inventory_drift(tmp_path: Path) -> None:
    fixture = write_snapback_workspace(tmp_path)

    run_dir, _report = run_snapback_design(fixture.explicit_path)
    views_manifest_path = run_dir / "analysis" / "views" / "views_manifest.v1.json"
    views_manifest = json.loads(views_manifest_path.read_text(encoding="utf-8"))
    views_manifest["views"] = views_manifest["views"][:-1]
    views_manifest["recommended_jobs"][0]["path"] = "../../baserender_jobs/missing.job.yaml"
    views_manifest_path.write_text(json.dumps(views_manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Snapback views manifest content drift detected."):
        snapback_show_payload(run_dir)


def test_snapback_show_detects_solve_report_workspace_root_drift(tmp_path: Path) -> None:
    fixture = write_snapback_workspace(tmp_path)

    run_dir, _report = run_snapback_solve(fixture.solve_path)
    solve_report_path = run_dir / "analysis" / "reports" / "solve_report.json"
    solve_report = json.loads(solve_report_path.read_text(encoding="utf-8"))
    solve_report["workspace_root"] = "/tmp/drifted"
    solve_report_path.write_text(json.dumps(solve_report, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Snapback solve report workspace_root drift detected."):
        snapback_show_payload(run_dir)


def test_snapback_show_detects_missing_materialized_hit_bundle(tmp_path: Path) -> None:
    fixture = write_snapback_workspace(tmp_path)

    run_dir, _report = run_snapback_solve(fixture.solve_path)
    shutil.rmtree(run_dir / "analysis" / "materialized_hits" / "hit_01")

    with pytest.raises(FileNotFoundError, match="Materialized snapback hit bundle missing"):
        snapback_show_payload(run_dir)


def test_snapback_show_detects_materialized_hit_path_reuse(tmp_path: Path) -> None:
    fixture = write_snapback_workspace(tmp_path)

    run_dir, _report = run_snapback_solve(fixture.solve_path)
    solve_report_path = run_dir / "analysis" / "reports" / "solve_report.json"
    solve_report = json.loads(solve_report_path.read_text(encoding="utf-8"))
    solve_report["hits"][0]["materialized_run_dir"] = solve_report["hits"][1]["materialized_run_dir"]
    solve_report_path.write_text(json.dumps(solve_report, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="Snapback solve materialized hit path/rank drift detected."):
        snapback_show_payload(run_dir)
