"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_artifacts.py

Regression tests for CLI artifacts OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.core.utils import now_iso, write_json
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def _setup_artifact_campaign(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    workdir = tmp_path / ".var" / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    plots_dir = workdir / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    current_plot = plots_dir / "current.png"
    current_plot.write_bytes(b"current-plot")
    current_csv = plots_dir / "current.csv"
    current_csv.write_text("round,value\n0,1.0\n", encoding="utf-8")
    stale_plot = plots_dir / "old.png"
    stale_plot.write_bytes(b"stale-plot")

    plot_manifest = {
        "schema_version": "opal.plot_artifact.v1",
        "plot_id": "current",
        "name": "current",
        "kind": "metric_over_rounds",
        "status": "written",
        "generated_at": now_iso(),
        "outputs": [
            {"role": "media", "path": str(current_plot), "exists": True},
            {"role": "tidy_csv", "path": str(current_csv), "exists": True},
        ],
        "manifest_path": str(plots_dir / "current.manifest.json"),
    }
    write_json(plots_dir / "current.manifest.json", plot_manifest)
    write_json(
        plots_dir / "plot_manifest.json",
        {
            "schema_version": "opal.plot_manifest_index.v1",
            "generated_at": now_iso(),
            "output_dir": str(plots_dir),
            "plot_count": 1,
            "manifests": [plot_manifest],
        },
    )

    review_plots_dir = workdir / "outputs" / "review" / "selection_views" / "primary" / "plots"
    review_plots_dir.mkdir(parents=True, exist_ok=True)
    stale_review_plot = review_plots_dir / "old_review.png"
    stale_review_plot.write_bytes(b"stale-review-plot")
    return (
        workdir,
        campaign,
        {
            "current_plot": current_plot,
            "current_csv": current_csv,
            "stale_plot": stale_plot,
            "stale_review_plot": stale_review_plot,
        },
    )


def test_artifacts_audit_reports_stale_manifest_absent_files_and_bytes(tmp_path: Path) -> None:
    _, campaign, paths = _setup_artifact_campaign(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "artifacts", "audit", "-c", str(campaign), "--json"])

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "opal.artifact_garden.v1"
    assert payload["local_only"] is True
    assert payload["active_manifests"]
    assert payload["bytes"]["stale_artifacts"] == (
        paths["stale_plot"].stat().st_size + paths["stale_review_plot"].stat().st_size
    )
    stale_paths = {row["path"] for row in payload["stale_artifacts"]}
    assert str(paths["stale_plot"]) in stale_paths
    assert str(paths["stale_review_plot"]) in stale_paths
    assert str(paths["current_plot"]) not in stale_paths
    assert str(paths["current_csv"]) not in stale_paths
    assert payload["prune_plan"]["requires_apply"] is True
    assert payload["prune_plan"]["item_count"] == 2


def test_artifacts_audit_reports_nested_stale_plot_outputs(tmp_path: Path) -> None:
    _, campaign, _ = _setup_artifact_campaign(tmp_path)
    workdir = campaign.parent
    nested_stale = workdir / "outputs" / "plots" / "metric_over_rounds" / "old_nested.png"
    nested_stale.parent.mkdir(parents=True, exist_ok=True)
    nested_stale.write_bytes(b"nested-stale")
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "artifacts", "audit", "-c", str(campaign), "--json"])

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    stale_paths = {row["path"] for row in payload["stale_artifacts"]}
    assert str(nested_stale) in stale_paths


def test_artifacts_audit_ignores_hidden_plot_runtime_cache(tmp_path: Path) -> None:
    _, campaign, _ = _setup_artifact_campaign(tmp_path)
    workdir = campaign.parent
    hidden_cache = workdir / "outputs" / "plots" / ".opal" / "tmp" / "mpl" / "fontlist-v390.json"
    hidden_cache.parent.mkdir(parents=True, exist_ok=True)
    hidden_cache.write_text("{}", encoding="utf-8")
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "artifacts", "audit", "-c", str(campaign), "--json"])

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    stale_paths = {row["path"] for row in payload["stale_artifacts"]}
    assert str(hidden_cache) not in stale_paths


def test_artifacts_audit_does_not_read_records_parquet(tmp_path: Path, monkeypatch) -> None:
    _, campaign, _ = _setup_artifact_campaign(tmp_path)
    from dnadesign.opal.src.storage import records_io

    def fail_read_parquet_df(*args, **kwargs):
        raise AssertionError("artifact gardening must not read records.parquet")

    monkeypatch.setattr(records_io, "read_parquet_df", fail_read_parquet_df)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "artifacts", "audit", "-c", str(campaign), "--json"])

    assert res.exit_code == 0, res.stdout


def test_artifacts_prune_is_dry_run_by_default_and_apply_removes_only_stale_files(tmp_path: Path) -> None:
    _, campaign, paths = _setup_artifact_campaign(tmp_path)
    app = _build()
    runner = CliRunner()

    dry = runner.invoke(app, ["--no-color", "artifacts", "prune", "-c", str(campaign), "--json"])

    assert dry.exit_code == 0, dry.stdout
    dry_payload = json.loads(dry.stdout)
    assert dry_payload["applied"] is False
    assert paths["stale_plot"].exists()
    assert paths["stale_review_plot"].exists()

    applied = runner.invoke(app, ["--no-color", "artifacts", "prune", "-c", str(campaign), "--apply", "--json"])

    assert applied.exit_code == 0, applied.stdout
    applied_payload = json.loads(applied.stdout)
    assert applied_payload["applied"] is True
    assert applied_payload["deleted_count"] == 2
    assert not paths["stale_plot"].exists()
    assert not paths["stale_review_plot"].exists()
    assert paths["current_plot"].exists()
    assert paths["current_csv"].exists()
