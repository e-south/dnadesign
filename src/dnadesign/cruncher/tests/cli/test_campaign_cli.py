"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_campaign_cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.app.campaign_service import expand_campaign
from dnadesign.cruncher.app.run_service import load_run_index, save_run_index
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.config.load import load_config

runner = CliRunner()


def test_campaign_generate_cli(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [["A"]],
                "regulator_categories": {"CatA": ["A", "B"], "CatB": ["C", "D"]},
            },
            "catalog": {"root": str(tmp_path / ".cruncher")},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    out_path = tmp_path / "expanded.yaml"
    result = runner.invoke(
        app,
        ["campaign", "generate", "--campaign", "demo", "--out", str(out_path), str(config_path)],
    )
    assert result.exit_code == 0
    assert out_path.exists()
    manifest_path = out_path.with_suffix(".campaign_manifest.json")
    assert manifest_path.exists()

    payload = yaml.safe_load(out_path.read_text())["cruncher"]
    assert payload["campaign"]["name"] == "demo"
    assert payload["workspace"]["regulator_sets"] == [
        ["A", "B"],
        ["C", "D"],
        ["A", "C"],
        ["A", "D"],
        ["B", "C"],
        ["B", "D"],
    ]

    manifest = json.loads(manifest_path.read_text())
    assert manifest["campaign_name"] == "demo"
    assert manifest["expanded_count"] == 6


def test_campaign_generate_keeps_workspace_relative_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [["A"]],
                "regulator_categories": {"CatA": ["A"], "CatB": ["B"]},
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
            "catalog": {"root": ".cruncher"},
            "ingest": {
                "genome_fasta": "data/genome.fna",
                "genome_cache": ".cruncher/genomes",
                "regulondb": {"ca_bundle": "certs/ca.pem"},
                "local_sources": [
                    {
                        "source_id": "local",
                        "root": "data/motifs",
                        "format_map": {".txt": "MEME"},
                    }
                ],
            },
        }
    }
    config_path = workspace / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    out_path = workspace / "expanded.yaml"
    result = runner.invoke(
        app, ["campaign", "generate", "--campaign", "demo", "--out", "expanded.yaml", str(config_path)]
    )
    assert result.exit_code == 0
    assert out_path.exists()
    payload = yaml.safe_load(out_path.read_text())["cruncher"]

    assert payload["workspace"]["out_dir"] == "runs"
    assert payload["catalog"]["root"] == ".cruncher"
    assert payload["ingest"]["genome_cache"] == ".cruncher/genomes"
    assert payload["ingest"]["genome_fasta"] == "data/genome.fna"
    assert payload["ingest"]["regulondb"]["ca_bundle"] == "certs/ca.pem"
    assert payload["ingest"]["local_sources"][0]["root"] == "data/motifs"


def test_campaign_summarize_cli(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [["A"]],
                "regulator_categories": {"CatA": ["A", "B"], "CatB": ["C", "D"]},
            },
            "catalog": {"root": str(tmp_path / ".cruncher")},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    runs_root = tmp_path / "runs"
    run_a = runs_root / "sample" / "sample_a"
    run_b = runs_root / "sample" / "sample_b"
    for run_dir, tfs in (
        (run_a, ["A", "B"]),
        (run_b, ["C", "D"]),
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps({"analysis_id": "analysis-1", "tf_names": tfs}))
        (run_dir / "table_manifest.json").write_text(
            json.dumps(
                {
                    "analysis_id": "analysis-1",
                    "tables": [
                        {"key": "scores_summary", "path": "tables/scores_summary.parquet", "exists": True},
                        {"key": "metrics_joint", "path": "tables/metrics_joint.parquet", "exists": True},
                    ],
                }
            )
        )
        (run_dir / "tables").mkdir(parents=True, exist_ok=True)
        score_summary_df = pd.DataFrame(
            [
                {"tf": tfs[0], "mean": 1.0, "median": 1.0, "std": 0.1, "min": 0.8, "max": 1.2},
                {"tf": tfs[1], "mean": 0.9, "median": 0.9, "std": 0.1, "min": 0.7, "max": 1.1},
            ]
        )
        score_summary_df.to_parquet(run_dir / "tables" / "scores_summary.parquet", index=False)
        joint_metrics_df = pd.DataFrame(
            [
                {
                    "tf_names": ",".join(tfs),
                    "joint_min": 0.8,
                    "joint_mean": 1.0,
                    "joint_hmean": 0.9,
                    "balance_index": 0.8,
                    "pareto_front_size": 1,
                    "pareto_fraction": 0.5,
                }
            ]
        )
        joint_metrics_df.to_parquet(run_dir / "tables" / "metrics_joint.parquet", index=False)
        manifest_path = run_dir / "run_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"stage": "sample", "run_dir": str(run_dir), "regulator_set": {"tfs": tfs}})
        )
        report_path = run_dir / "report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"run": {"n_sequences": 2, "n_elites": 1}}))

    result = runner.invoke(
        app,
        [
            "campaign",
            "summarize",
            "--campaign",
            "demo",
            "--no-metrics",
            "--runs",
            str(run_a),
            str(run_b),
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0

    cfg = load_config(config_path)
    expansion = expand_campaign(cfg=cfg, config_path=config_path, campaign_name="demo", include_metrics=False)
    out_dir = runs_root / "campaigns" / expansion.campaign_id
    assert (out_dir / "campaign_summary.csv").exists()
    assert (out_dir / "campaign_best.csv").exists()
    assert (out_dir / "campaign_manifest.json").exists()
    assert (out_dir / "plot__best_jointscore_bar.png").exists()
    assert (out_dir / "plot__tf_coverage_heatmap.png").exists()
    assert (out_dir / "plot__pairgrid_overview.png").exists()
    assert (out_dir / "plot__joint_trend.png").exists()
    assert (out_dir / "plot__pareto_projection.png").exists()


def test_campaign_summarize_uses_table_manifest_contract(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [["A"]],
                "regulator_categories": {"CatA": ["A", "B"], "CatB": ["C", "D"]},
            },
            "catalog": {"root": str(tmp_path / ".cruncher")},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = tmp_path / "runs" / "sample" / "sample_manifest_tables"
    analysis_dir = run_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    tfs = ["A", "B"]
    (analysis_dir / "summary.json").write_text(
        json.dumps({"analysis_id": "analysis-1", "tf_names": tfs, "analysis_config": {"table_format": "parquet"}})
    )
    (analysis_dir / "table_manifest.json").write_text(
        json.dumps(
            {
                "analysis_id": "analysis-1",
                "tables": [
                    {"key": "scores_summary", "path": "tables/scores_summary.parquet", "exists": True},
                    {"key": "metrics_joint", "path": "tables/metrics_joint.parquet", "exists": True},
                ],
            }
        )
    )
    (analysis_dir / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"tf": tfs[0], "mean": 1.0, "median": 1.0, "std": 0.1, "min": 0.8, "max": 1.2},
            {"tf": tfs[1], "mean": 0.9, "median": 0.9, "std": 0.1, "min": 0.7, "max": 1.1},
        ]
    ).to_parquet(analysis_dir / "tables" / "scores_summary.parquet", index=False)
    pd.DataFrame(
        [
            {
                "tf_names": ",".join(tfs),
                "joint_min": 0.8,
                "joint_mean": 1.0,
                "joint_hmean": 0.9,
                "balance_index": 0.8,
                "pareto_front_size": 1,
                "pareto_fraction": 0.5,
            }
        ]
    ).to_parquet(analysis_dir / "tables" / "metrics_joint.parquet", index=False)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"stage": "sample", "run_dir": str(run_dir), "regulator_set": {"tfs": tfs}}))
    report_path = run_dir / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"run": {"n_sequences": 2, "n_elites": 1}}))

    result = runner.invoke(
        app,
        [
            "campaign",
            "summarize",
            "--campaign",
            "demo",
            "--no-metrics",
            "--runs",
            str(run_dir),
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0


def test_campaign_summarize_auto_repairs_stale_index_entries(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [["A"]],
                "regulator_categories": {"CatA": ["A", "B"], "CatB": ["C", "D"]},
            },
            "catalog": {"root": str(tmp_path / ".cruncher")},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run_dir = tmp_path / "runs" / "sample" / "sample_valid"
    tfs = ["A", "B"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"analysis_id": "analysis-1", "tf_names": tfs, "analysis_config": {"table_format": "parquet"}})
    )
    (run_dir / "table_manifest.json").write_text(
        json.dumps(
            {
                "analysis_id": "analysis-1",
                "tables": [
                    {"key": "scores_summary", "path": "tables/scores_summary.parquet", "exists": True},
                    {"key": "metrics_joint", "path": "tables/metrics_joint.parquet", "exists": True},
                ],
            }
        )
    )
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"tf": tfs[0], "mean": 1.0, "median": 1.0, "std": 0.1, "min": 0.8, "max": 1.2},
            {"tf": tfs[1], "mean": 0.9, "median": 0.9, "std": 0.1, "min": 0.7, "max": 1.1},
        ]
    ).to_parquet(run_dir / "tables" / "scores_summary.parquet", index=False)
    pd.DataFrame(
        [
            {
                "tf_names": ",".join(tfs),
                "joint_min": 0.8,
                "joint_mean": 1.0,
                "joint_hmean": 0.9,
                "balance_index": 0.8,
                "pareto_front_size": 1,
                "pareto_fraction": 0.5,
            }
        ]
    ).to_parquet(run_dir / "tables" / "metrics_joint.parquet", index=False)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "stage": "sample",
                "run_dir": str(run_dir),
                "regulator_set": {"tfs": tfs},
            }
        )
    )
    (run_dir / "report.json").write_text(json.dumps({"run": {"n_sequences": 2, "n_elites": 1}}))

    stale_dir = tmp_path / "runs" / "sample" / "deleted_run"
    save_run_index(
        config_path,
        {
            "sample/deleted_run": {"stage": "sample", "run_dir": str(stale_dir)},
            "sample/sample_valid": {"stage": "sample", "run_dir": str(run_dir)},
        },
        catalog_root=str(tmp_path / ".cruncher"),
    )

    result = runner.invoke(
        app,
        [
            "campaign",
            "summarize",
            "--campaign",
            "demo",
            "--no-metrics",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0

    index = load_run_index(config_path, catalog_root=str(tmp_path / ".cruncher"))
    assert "sample/deleted_run" not in index
    assert "sample/sample_valid" in index


def test_campaign_validate_cli_no_selectors(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [["A"]],
                "regulator_categories": {"CatA": ["A", "B"], "CatB": ["C"]},
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = runner.invoke(
        app,
        ["campaign", "validate", "--campaign", "demo", "--no-selectors", "--no-metrics", str(config_path)],
    )
    assert result.exit_code == 0
    assert "Campaign validation" in result.output


def test_campaign_validate_requires_catalog(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {
                "out_dir": "runs",
                "regulator_sets": [["A"]],
                "regulator_categories": {"CatA": ["A", "B"], "CatB": ["C"]},
            },
            "catalog": {"root": str(tmp_path / ".cruncher")},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                    "selectors": {"min_site_count": 1},
                }
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    result = runner.invoke(
        app,
        ["campaign", "validate", "--campaign", "demo", str(config_path)],
    )
    assert result.exit_code == 1
    assert "Catalog root not found" in result.output
