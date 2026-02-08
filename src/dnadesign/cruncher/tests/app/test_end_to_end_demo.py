"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_end_to_end_demo.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.app.fetch_service import fetch_motifs, fetch_sites
from dnadesign.cruncher.app.lock_service import resolve_lock
from dnadesign.cruncher.app.parse_workflow import run_parse
from dnadesign.cruncher.app.sample_workflow import run_sample
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_path,
    logos_root,
    manifest_path,
    out_root,
    parse_manifest_path,
    pwm_summary_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.ingest.adapters.regulondb import RegulonDBAdapter, RegulonDBAdapterConfig
from dnadesign.cruncher.tests.fixtures.regulondb_payloads import (
    CPXR_ID,
    HT_DATASETS,
    HT_SOURCES,
    HT_TF_BINDING,
    LEXA_ID,
    REGULON_DETAIL,
    regulon_list_for_search,
)
from dnadesign.cruncher.utils.paths import resolve_lock_path


def _sample_block() -> dict:
    return {
        "seed": 7,
        "sequence_length": 12,
        "budget": {"tune": 1, "draws": 2},
        "pt": {"n_temps": 2, "temp_max": 10.0},
        "objective": {"bidirectional": True, "score_scale": "normalized-llr", "combine": "min"},
        "elites": {
            "k": 1,
            "filter": {"min_per_tf_norm": None, "require_all_tfs": True, "pwm_sum_min": 0.0},
            "select": {"alpha": 0.85, "pool_size": "auto"},
        },
        "moves": {
            "profile": "balanced",
            "overrides": {
                "block_len_range": [2, 2],
                "multi_k_range": [2, 2],
                "slide_max_shift": 1,
                "swap_len_range": [2, 2],
                "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
            },
        },
        "output": {
            "save_trace": False,
            "save_sequences": True,
            "include_tune_in_sequences": False,
            "live_metrics": False,
        },
    }


runner = CliRunner()


def _fixture_transport(query: str, variables: dict) -> dict:
    if "listAllHTSources" in query:
        return HT_SOURCES
    if "getDatasetsWithMetadata" in query:
        source = variables.get("source")
        return HT_DATASETS.get(source, {"getDatasetsWithMetadata": {"datasets": []}})
    if "getAllTFBindingOfDataset" in query:
        dataset_id = variables.get("datasetId")
        page = variables.get("page", 0)
        if page:
            return {"getAllTFBindingOfDataset": []}
        return HT_TF_BINDING.get(dataset_id, {"getAllTFBindingOfDataset": []})
    if "regulatoryInteractions" in query:
        search = (variables.get("search") or "").lower()
        if search in {LEXA_ID.lower(), "lexa"}:
            return REGULON_DETAIL[LEXA_ID]
        if search in {CPXR_ID.lower(), "cpxr"}:
            return REGULON_DETAIL[CPXR_ID]
        return {"getRegulonBy": {"data": []}}
    return regulon_list_for_search(variables.get("search"))


def test_end_to_end_sites_pipeline(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {
                "root": str(catalog_root),
                "source_preference": ["regulondb"],
                "allow_ambiguous": False,
                "pwm_source": "sites",
                "min_sites_for_pwm": 2,
                "allow_low_sites": False,
            },
            "ingest": {
                "regulondb": {
                    "base_url": "https://regulondb.ccg.unam.mx/graphql",
                    "verify_ssl": True,
                    "timeout_seconds": 30,
                    "motif_matrix_source": "alignment",
                    "alignment_matrix_semantics": "probabilities",
                    "min_sites_for_pwm": 2,
                    "allow_low_sites": False,
                    "curated_sites": True,
                    "ht_sites": False,
                    "ht_dataset_sources": None,
                    "ht_dataset_type": "TFBINDING",
                    "uppercase_binding_site_only": True,
                }
            },
            "sample": _sample_block(),
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=True, ht_sites=False),
        transport=_fixture_transport,
    )
    fetch_sites(adapter, catalog_root, names=["lexA", "cpxR"])
    fetch_motifs(adapter, catalog_root, names=["lexA", "cpxR"])

    lock_path = resolve_lock_path(config_path)
    resolve_lock(
        names=["lexA", "cpxR"],
        catalog_root=catalog_root,
        pwm_source="sites",
        lock_path=lock_path,
    )

    cfg = load_config(config_path)
    run_parse(cfg, config_path)
    run_sample(cfg, config_path)

    sample_dir = tmp_path / "results" / "latest"
    assert sample_dir.exists()
    assert manifest_path(sample_dir).exists()
    assert parse_manifest_path(sample_dir).exists()
    assert pwm_summary_path(sample_dir).exists()

    result = runner.invoke(app, ["catalog", "logos", "--set", "1", "-c", str(config_path)])
    assert result.exit_code == 0
    elites_df = pd.read_parquet(elites_path(sample_dir), engine="fastparquet")
    assert len(elites_df) <= config["cruncher"]["sample"]["elites"]["k"]
    logo_root = logos_root(out_root(config_path, cfg.out_dir)) / "catalog"
    logos = list(logo_root.glob("**/*_logo.png"))
    assert any("lexA" in path.name for path in logos)
    assert any("cpxR" in path.name for path in logos)
    assert config_used_path(sample_dir).exists()
    assert sequences_path(sample_dir).exists()
    assert not trace_path(sample_dir).exists()


def test_demo_workspace_cli_without_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "regulondb_ecoli"
    workspace.mkdir()
    catalog_root = workspace / ".cruncher"
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {
                "root": str(catalog_root),
                "source_preference": ["regulondb"],
                "allow_ambiguous": False,
                "pwm_source": "sites",
                "min_sites_for_pwm": 2,
                "allow_low_sites": False,
            },
            "ingest": {
                "regulondb": {
                    "base_url": "https://regulondb.ccg.unam.mx/graphql",
                    "verify_ssl": True,
                    "timeout_seconds": 30,
                    "motif_matrix_source": "alignment",
                    "alignment_matrix_semantics": "probabilities",
                    "min_sites_for_pwm": 2,
                    "allow_low_sites": False,
                    "curated_sites": True,
                    "ht_sites": False,
                    "ht_dataset_sources": None,
                    "ht_dataset_type": "TFBINDING",
                    "uppercase_binding_site_only": True,
                }
            },
            "sample": _sample_block(),
        }
    }
    config_path = workspace / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=True, ht_sites=False),
        transport=_fixture_transport,
    )
    fetch_sites(adapter, catalog_root, names=["lexA", "cpxR"])
    fetch_motifs(adapter, catalog_root, names=["lexA", "cpxR"])

    lock_path = resolve_lock_path(config_path)
    resolve_lock(
        names=["lexA", "cpxR"],
        catalog_root=catalog_root,
        pwm_source="sites",
        lock_path=lock_path,
    )

    monkeypatch.chdir(workspace)

    result = runner.invoke(app, ["lock"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["sample"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["runs", "list"])
    assert result.exit_code == 0


def test_demo_campaign_pair_local_only_generates_plots(tmp_path: Path) -> None:
    package_root = Path(__file__).resolve().parents[2]
    demo_workspace = package_root / "workspaces" / "demo_campaigns_multi_tf"
    workspace = tmp_path / "demo_campaigns_multi_tf"
    shutil.copytree(demo_workspace, workspace)
    shutil.rmtree(workspace / "outputs", ignore_errors=True)
    shutil.rmtree(workspace / ".cruncher", ignore_errors=True)
    (workspace / "outputs").mkdir(parents=True, exist_ok=True)

    config_path = workspace / "config.yaml"
    config_payload = yaml.safe_load(config_path.read_text())
    cruncher_cfg = config_payload["cruncher"]
    cruncher_cfg["catalog"]["root"] = str(workspace / ".cruncher" / "demo_campaigns_multi_tf")
    cruncher_cfg["sample"]["budget"]["tune"] = 240
    cruncher_cfg["sample"]["budget"]["draws"] = 480
    cruncher_cfg["sample"]["elites"]["k"] = 3
    cruncher_cfg["sample"]["elites"]["filter"]["min_per_tf_norm"] = 0.0
    cruncher_cfg["analysis"]["max_points"] = 1000
    config_path.write_text(yaml.safe_dump(config_payload))

    result = runner.invoke(
        app,
        [
            "fetch",
            "sites",
            "--source",
            "demo_local_meme",
            "--tf",
            "lexA",
            "--tf",
            "cpxR",
            "--update",
            "-c",
            str(config_path),
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(
        app,
        [
            "campaign",
            "generate",
            "--campaign",
            "demo_pair",
            "--out",
            "campaign_demo_pair.yaml",
            "-c",
            str(config_path),
        ],
    )
    assert result.exit_code == 0

    derived_config = workspace / "campaign_demo_pair.yaml"
    for command in (
        ["lock", "-c", str(derived_config)],
        ["parse", "-c", str(derived_config)],
        ["sample", "-c", str(derived_config)],
        ["analyze", "--summary", "-c", str(derived_config)],
        ["campaign", "summarize", "--campaign", "demo_pair", "-c", str(derived_config)],
    ):
        result = runner.invoke(app, command)
        assert result.exit_code == 0

    analysis_dir = workspace / "outputs" / "latest"
    assert analysis_dir.is_dir()
    assert manifest_path(analysis_dir).exists()
    assert (analysis_dir / "plots" / "plot__run_summary.png").exists()
    assert (analysis_dir / "plots" / "plot__opt_trajectory.png").exists()
    assert (analysis_dir / "plots" / "plot__elites_nn_distance.png").exists()
    assert (analysis_dir / "plots" / "plot__overlap_panel.png").exists()
    assert (analysis_dir / "plots" / "plot__health_panel.png").exists()

    campaign_dir = workspace / "outputs" / "campaign" / "demo_pair" / "latest"
    assert campaign_dir.is_dir()
    assert (campaign_dir / "plot__best_jointscore_bar.png").exists()
    assert (campaign_dir / "plot__tf_coverage_heatmap.png").exists()
    assert (campaign_dir / "plot__pairgrid_overview.png").exists()
    assert (campaign_dir / "plot__joint_trend.png").exists()
    assert (campaign_dir / "plot__pareto_projection.png").exists()
