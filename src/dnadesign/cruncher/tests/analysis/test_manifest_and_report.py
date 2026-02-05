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

from dnadesign.cruncher.analysis.layout import analysis_root, summary_path
from dnadesign.cruncher.analysis.report import ensure_report
from dnadesign.cruncher.artifacts.layout import (
    elites_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import build_run_manifest, write_manifest
from dnadesign.cruncher.config.schema_v2 import (
    CruncherConfig,
    IngestConfig,
    InitConfig,
    MotifStoreConfig,
    ParseConfig,
    PlotConfig,
    SampleComputeConfig,
    SampleConfig,
    SampleElitesConfig,
    SampleMovesConfig,
    SampleObjectiveConfig,
    SampleRngConfig,
)
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.lockfile import LockedMotif


def _make_config() -> CruncherConfig:
    return CruncherConfig(
        out_dir=Path("results"),
        regulator_sets=[["lexA"]],
        motif_store=MotifStoreConfig(
            catalog_root=Path(".cruncher"),
            source_preference=["regulondb"],
            allow_ambiguous=False,
            pwm_source="matrix",
            min_sites_for_pwm=2,
        ),
        ingest=IngestConfig(),
        parse=ParseConfig(plot=PlotConfig(logo=False, bits_mode="information", dpi=100)),
        sample=SampleConfig(
            mode="sample",
            rng=SampleRngConfig(seed=1, deterministic=True),
            sequence_length=6,
            compute=SampleComputeConfig(total_sweeps=3, adapt_sweep_frac=0.34),
            init=InitConfig(kind="random"),
            objective=SampleObjectiveConfig(bidirectional=True, score_scale="llr"),
            elites=SampleElitesConfig(k=1),
            moves=SampleMovesConfig(),
        ),
        analysis=None,
    )


def test_build_manifest_and_report(tmp_path: Path) -> None:
    cfg = _make_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: config")

    catalog_root = tmp_path / ".cruncher"
    cfg.motif_store.catalog_root = catalog_root
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
        artifacts=["artifacts/trace.nc", "artifacts/sequences.parquet"],
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
    summary_path(analysis_dir).write_text(json.dumps(summary_payload))

    ensure_report(analysis_root=analysis_dir, summary_payload=summary_payload, refresh=True)

    report_root = analysis_root(run_dir)
    assert (report_root / "report.json").exists()
    assert (report_root / "report.md").exists()

    report = json.loads((report_root / "report.json").read_text())
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
    summary_path(analysis_dir).write_text(json.dumps(summary_payload))

    ensure_report(analysis_root=analysis_dir, summary_payload=summary_payload, refresh=True)

    report = json.loads((analysis_dir / "report.json").read_text())
    learning = report["highlights"]["learning"]
    assert learning["best_score_draw"] == 5
    assert learning["best_score_chain"] == 1
    assert learning["early_stop_earliest_draw"] == 6
