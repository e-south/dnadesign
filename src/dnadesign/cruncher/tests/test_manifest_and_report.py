from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from dnadesign.cruncher.config.schema_v2 import (
    CoolingLinear,
    CruncherConfig,
    IngestConfig,
    InitConfig,
    MotifStoreConfig,
    MoveConfig,
    OptimiserConfig,
    ParseConfig,
    PlotConfig,
    SampleConfig,
)
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.store.lockfile import LockedMotif
from dnadesign.cruncher.utils.manifest import build_run_manifest, write_manifest
from dnadesign.cruncher.workflows.report_workflow import run_report


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
            bidirectional=True,
            seed=1,
            record_tune=False,
            init=InitConfig(kind="random", length=6),
            draws=2,
            tune=1,
            chains=1,
            min_dist=1,
            top_k=1,
            moves=MoveConfig(),
            optimiser=OptimiserConfig(
                kind="gibbs",
                scorer_scale="llr",
                cooling=CoolingLinear(beta=(0.1, 0.2)),
                swap_prob=0.1,
            ),
            save_sequences=True,
            pwm_sum_threshold=0.0,
        ),
        analysis=None,
    )


def test_build_manifest_and_report(tmp_path: Path) -> None:
    cfg = _make_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dummy: config")

    catalog_root = tmp_path / ".cruncher"
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

    lock_path = catalog_root / "locks" / "config.lock.json"
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

    run_dir = tmp_path / "results" / "sample_test"
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
    idata.to_netcdf(run_dir / "trace.nc")

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
    ).to_parquet(run_dir / "sequences.parquet", index=False)

    elite_dir = run_dir / "cruncher_elites_test"
    elite_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "sequence": "ACGTAC",
                "rank": 1,
                "score_lexA": 1.0,
                "per_tf_json": json.dumps({"lexA": {"scaled_score": 1.0}}),
            }
        ]
    ).to_parquet(elite_dir / "cruncher_elites_test.parquet", index=False)

    run_report(cfg, config_path, "sample_test")

    assert (run_dir / "report.json").exists()
    assert (run_dir / "report.md").exists()
