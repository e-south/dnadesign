"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_campaign_cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.campaign_service import expand_campaign

runner = CliRunner()


def test_campaign_generate_cli(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [],
            "regulator_categories": {
                "CatA": ["A", "B"],
                "CatB": ["C", "D"],
            },
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
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
    assert payload["regulator_sets"] == [
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


def test_campaign_generate_rebases_relative_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [],
            "regulator_categories": {"CatA": ["A"], "CatB": ["B"]},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
            "motif_store": {"catalog_root": ".cruncher"},
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
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = workspace / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    out_dir = tmp_path / "exports"
    out_dir.mkdir()
    out_path = out_dir / "expanded.yaml"
    result = runner.invoke(
        app,
        ["campaign", "generate", "--campaign", "demo", "--out", str(out_path), str(config_path)],
    )
    assert result.exit_code == 0
    payload = yaml.safe_load(out_path.read_text())["cruncher"]

    def rel(target: str) -> str:
        return os.path.relpath((workspace / target).resolve(), out_dir)

    assert payload["out_dir"] == rel("runs")
    assert payload["motif_store"]["catalog_root"] == rel(".cruncher")
    assert payload["ingest"]["genome_cache"] == rel(".cruncher/genomes")
    assert payload["ingest"]["genome_fasta"] == rel("data/genome.fna")
    assert payload["ingest"]["regulondb"]["ca_bundle"] == rel("certs/ca.pem")
    assert payload["ingest"]["local_sources"][0]["root"] == rel("data/motifs")


def test_campaign_summarize_cli(tmp_path: Path) -> None:
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [],
            "regulator_categories": {"CatA": ["A", "B"], "CatB": ["C", "D"]},
            "campaigns": [
                {
                    "name": "demo",
                    "categories": ["CatA", "CatB"],
                    "within_category": {"sizes": [2]},
                    "across_categories": {"sizes": [2], "max_per_category": 1},
                }
            ],
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    runs_root = tmp_path / "runs"
    run_a = runs_root / "sample_a"
    run_b = runs_root / "sample_b"
    for run_dir, tfs in (
        (run_a, ["A", "B"]),
        (run_b, ["C", "D"]),
    ):
        (run_dir / "analysis" / "tables").mkdir(parents=True, exist_ok=True)
        (run_dir / "analysis" / "summary.json").write_text(json.dumps({"analysis_id": "analysis-1", "tf_names": tfs}))
        score_summary = f"tf,mean,median,std,min,max\n{tfs[0]},1.0,1.0,0.1,0.8,1.2\n{tfs[1]},0.9,0.9,0.1,0.7,1.1\n"
        (run_dir / "analysis" / "tables" / "score_summary.csv").write_text(score_summary)
        joint_metrics = (
            "tf_names,joint_min,joint_mean,joint_hmean,balance_index,pareto_front_size,pareto_fraction\n"
            f"{','.join(tfs)},0.8,1.0,0.9,0.8,1,0.5\n"
        )
        (run_dir / "analysis" / "tables" / "joint_metrics.csv").write_text(joint_metrics)
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"stage": "sample", "run_dir": str(run_dir), "regulator_set": {"tfs": tfs}})
        )
        (run_dir / "report.json").write_text(json.dumps({"n_sequences": 2, "n_elites": 1}))

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
    assert (out_dir / "plots" / "best_jointscore_bar.png").exists()
    assert (out_dir / "plots" / "tf_coverage_heatmap.png").exists()
    assert (out_dir / "plots" / "pairgrid_overview.png").exists()
