"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_plot_manifest.py

Plot manifest coverage for plot generation outputs.
Dunlop Lab.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.adapters.outputs import OutputRecord, ParquetSink
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.viz.plotting import run_plots_from_config
from dnadesign.densegen.tests.meta_fixtures import output_meta


def _write_config(path: Path, *, plots_default: list[str]) -> None:
    path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.5"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/dense_arrays.parquet
              generation:
                sequence_length: 10
                quota: 1
                plan:
                  - name: demo_plan
                    quota: 1
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: parquet
              out_dir: outputs/plots
              format: png
              default: PLACEHOLDER_DEFAULT
            """
        )
        .strip()
        .replace("PLACEHOLDER_DEFAULT", json.dumps(plots_default))
        + "\n"
    )


def _write_output_record(run_root: Path) -> None:
    out_file = run_root / "outputs" / "tables" / "dense_arrays.parquet"
    sink = ParquetSink(path=str(out_file), chunk_size=1)
    meta = output_meta(library_hash="abc123", library_index=1)
    meta["compression_ratio"] = 1.0
    rec = OutputRecord.from_sequence(
        sequence="ATGCATGCAT",
        meta=meta,
        source="densegen:demo",
        bio_type="dna",
        alphabet="dna_4",
    )
    sink.add(rec)
    sink.finalize()


def _write_pool_manifest(run_root: Path) -> None:
    pools_dir = run_root / "outputs" / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "input_name": ["demo_input"] * 3,
            "tf": ["tfA", "tfA", "tfB"],
            "tfbs": ["AAAA", "AAAAT", "AAAAAA"],
            "best_hit_score": [7.0, 9.0, 5.5],
            "tier": [1, 0, 2],
            "rank_within_regulator": [1, 0, 0],
            "motif_id": ["m1", "m1", "m2"],
            "tfbs_id": ["id1", "id2", "id3"],
        }
    )
    pool_path = pools_dir / "demo_input__pool.parquet"
    df.to_parquet(pool_path, index=False)
    manifest = {
        "schema_version": "1.3",
        "run_id": "demo",
        "run_root": ".",
        "config_path": "config.yaml",
        "inputs": [
            {
                "name": "demo_input",
                "type": "binding_sites",
                "pool_path": "demo_input__pool.parquet",
                "rows": int(len(df)),
                "columns": list(df.columns),
                "pool_mode": "tfbs",
                "stage_a_sampling": {
                    "backend": "fimo",
                    "tier_scheme": "pct_1_9_90",
                    "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
                    "retention_rule": "top_n_sites_by_best_hit_score",
                    "fimo_thresh": 1.0,
                    "eligible_score_hist": [
                        {
                            "regulator": "tfA",
                            "edges": [4.0, 6.0, 8.0, 10.0],
                            "counts": [0, 1, 1],
                            "tier0_score": 9.0,
                            "tier1_score": 7.0,
                        },
                        {
                            "regulator": "tfB",
                            "edges": [4.0, 6.0, 8.0],
                            "counts": [1, 0],
                            "tier0_score": 5.5,
                            "tier1_score": None,
                        },
                    ],
                },
            }
        ],
    }
    (pools_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=2))


def test_plot_manifest_written(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["compression_ratio"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_output_record(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    manifest_path = run_root / "outputs" / "plots" / "plot_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    names = {item["name"] for item in payload.get("plots", [])}
    assert "compression_ratio" in names
    paths = {item["path"] for item in payload.get("plots", [])}
    assert "compression_ratio.png" in paths


def test_stage_a_plots_without_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    _write_config(cfg_path, plots_default=["stage_a_strata_overview"])
    (run_root / "inputs.csv").write_text("tf,tfbs\n")
    _write_pool_manifest(run_root)

    loaded = load_config(cfg_path)
    run_plots_from_config(loaded.root, cfg_path)

    plots_dir = run_root / "outputs" / "plots"
    overview_plot = plots_dir / "stage_a_strata_overview__demo_input.png"
    assert overview_plot.exists()
