"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/analysis/test_manifest_and_report.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest

from dnadesign.cruncher.analysis.layout import (
    analysis_manifest_path,
    analysis_plot_path,
    analysis_root,
    analysis_table_path,
    plot_manifest_path,
    report_json_path,
    report_md_path,
    summary_path,
    table_manifest_path,
)
from dnadesign.cruncher.analysis.report import build_report_payload, ensure_report
from dnadesign.cruncher.artifacts.layout import (
    elites_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import build_run_manifest, write_manifest
from dnadesign.cruncher.config.schema_v3 import CatalogConfig, CruncherConfig, WorkspaceConfig
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.lockfile import LockedMotif


def _make_config() -> CruncherConfig:
    return CruncherConfig(
        schema_version=3,
        workspace=WorkspaceConfig(
            out_dir=Path("results"),
            regulator_sets=[["lexA"]],
        ),
        catalog=CatalogConfig(
            root=Path(".cruncher"),
            source_preference=["regulondb"],
            allow_ambiguous=False,
            pwm_source="matrix",
            min_sites_for_pwm=2,
        ),
        sample=None,
        analysis=None,
    )


def test_build_manifest_and_report(tmp_path: Path) -> None:
    cfg = _make_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: config")

    catalog_root = tmp_path / ".cruncher"
    cfg.catalog.root = catalog_root
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM0001",
        tf_name="lexA",
        kind="PFM",
        matrix_length=4,
        matrix_source="alignment",
        matrix_semantics="probabilities",
        has_matrix=True,
        has_sites=False,
        site_count=0,
        site_total=0,
    )
    catalog = CatalogIndex(entries={entry.key: entry})
    catalog.save(catalog_root)

    from dnadesign.cruncher.utils.paths import resolve_lock_path

    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pwm_source": "matrix",
                "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM0001", "sha256": "abc"}},
            }
        )
    )
    lockmap = {"lexA": LockedMotif(source="regulondb", motif_id="RBM0001", sha256="abc")}

    run_dir = tmp_path / "results" / "sample" / "sample_test"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        stage="sample",
        cfg=cfg,
        config_path=config_path,
        lock_path=lock_path,
        lockmap=lockmap,
        catalog=catalog,
        run_dir=run_dir,
        artifacts=["trace.nc", "sequences.parquet"],
        extra={"sequence_length": 6},
    )
    write_manifest(run_dir, manifest)

    # minimal trace + sequences + elites
    idata = az.from_dict(posterior={"score": np.array([[0.1, 0.2]])})
    trace_file = trace_path(run_dir)
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(trace_file)

    seq_path = sequences_path(run_dir)
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "chain": 0,
                "draw": 0,
                "phase": "draw",
                "sequence": "ACGTAC",
                "score_lexA": 1.0,
            }
        ]
    ).to_parquet(seq_path, index=False)

    pd.DataFrame(
        [
            {
                "sequence": "ACGTAC",
                "rank": 1,
                "score_lexA": 1.0,
                "per_tf_json": json.dumps({"lexA": {"scaled_score": 1.0}}),
            }
        ]
    ).to_parquet(elites_path(run_dir), index=False)

    analysis_dir = analysis_root(run_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "analysis_id": "analysis_test",
        "run": "sample_test",
        "tf_names": ["lexA"],
        "diagnostics": {},
        "objective_components": {},
        "overlap_summary": {},
    }
    summary_path(analysis_dir).parent.mkdir(parents=True, exist_ok=True)
    summary_path(analysis_dir).write_text(json.dumps(summary_payload))

    ensure_report(analysis_root=analysis_dir, summary_payload=summary_payload, refresh=True)

    report_root = analysis_root(run_dir)
    assert report_json_path(report_root).exists()
    assert report_md_path(report_root).exists()

    report = json.loads(report_json_path(report_root).read_text())
    assert report["run"]["run"] == "sample_test"


def test_report_includes_learning_highlights(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "sample" / "sample_learning"
    analysis_dir = analysis_root(run_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "analysis_id": "analysis_learning",
        "run": "sample_learning",
        "tf_names": ["lexA"],
        "diagnostics": {},
        "objective_components": {
            "learning": {
                "best_score_draw": 5,
                "best_score_chain": 1,
                "last_improvement_draw": 4,
                "plateau_draws": 2,
                "early_stop": {
                    "earliest_draw": 6,
                    "stopped_chains": 1,
                },
            }
        },
        "overlap_summary": {},
    }
    summary_path(analysis_dir).parent.mkdir(parents=True, exist_ok=True)
    summary_path(analysis_dir).write_text(json.dumps(summary_payload))

    ensure_report(analysis_root=analysis_dir, summary_payload=summary_payload, refresh=True)

    report = json.loads(report_json_path(analysis_dir).read_text())
    learning = report["highlights"]["learning"]
    assert learning["best_score_draw"] == 5
    assert learning["best_score_chain"] == 1
    assert learning["early_stop_earliest_draw"] == 6


def test_report_payload_preserves_zero_highlights() -> None:
    payload = build_report_payload(
        analysis_root=Path("."),
        summary_payload={
            "run": "sample_zero",
            "analysis_id": "analysis_zero",
            "tf_names": ["lexA"],
        },
        diagnostics_payload={
            "status": "ok",
            "warnings": [],
            "metrics": {"sequences": {"unique_fraction": 0.9}},
        },
        objective_components={
            "unique_fraction_canonical": 0.0,
            "unique_fraction_raw": 0.8,
            "overlap_total_bp_median": 0.0,
        },
        overlap_summary={"overlap_rate_median": 0.0, "overlap_total_bp_median": 0.0},
        analysis_used_payload={"analysis": {"table_format": "parquet", "plot_format": "png"}},
    )
    diversity = payload["highlights"]["diversity"]
    overlap = payload["highlights"]["overlap"]
    assert diversity["unique_fraction"] == 0.0
    assert overlap["overlap_rate_median"] == 0.0
    assert overlap["overlap_total_bp_median"] == 0.0


def test_report_payload_paths_use_flat_output_and_plots_schema(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "run"
    analysis_plot_path(analysis_dir, "run_summary", "png").parent.mkdir(parents=True, exist_ok=True)
    analysis_plot_path(analysis_dir, "run_summary", "png").write_text("png")
    analysis_table_path(analysis_dir, "diagnostics_summary", "json").parent.mkdir(parents=True, exist_ok=True)
    analysis_table_path(analysis_dir, "diagnostics_summary", "json").write_text("{}")
    analysis_table_path(analysis_dir, "objective_components", "json").write_text("{}")
    analysis_manifest_path(analysis_dir).write_text("{}")
    plot_manifest_path(analysis_dir).write_text('{"plots": []}')
    table_manifest_path(analysis_dir).write_text('{"tables": []}')

    payload = build_report_payload(
        analysis_root=analysis_dir,
        summary_payload={"run": "sample_one", "analysis_id": "analysis_one", "tf_names": ["lexA"]},
        diagnostics_payload={"status": "ok", "warnings": [], "metrics": {}},
        objective_components={},
        overlap_summary={},
        analysis_used_payload={"analysis": {"table_format": "parquet", "plot_format": "png"}},
    )
    pointers = payload["paths"]
    assert pointers["start_here_plot"] == "plots/plot__run_summary.png"
    assert pointers["diagnostics"] == "output/table__diagnostics_summary.json"
    assert pointers["objective_components"] == "output/table__objective_components.json"
    assert pointers["manifest"] == "output/manifest.json"
    assert pointers["plot_manifest"] == "output/plot_manifest.json"
    assert pointers["table_manifest"] == "output/table_manifest.json"


def test_ensure_report_requires_summary_json_when_payload_not_provided(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "sample" / "sample_missing_summary"
    analysis_dir = analysis_root(run_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Missing analysis summary JSON"):
        ensure_report(analysis_root=analysis_dir, refresh=True)
