"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_end_to_end_demo.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.ingest.adapters.regulondb import RegulonDBAdapter, RegulonDBAdapterConfig
from dnadesign.cruncher.services.fetch_service import fetch_motifs, fetch_sites
from dnadesign.cruncher.services.lock_service import resolve_lock
from dnadesign.cruncher.tests.fixtures.regulondb_payloads import (
    CPXR_ID,
    HT_DATASETS,
    HT_SOURCES,
    HT_TF_BINDING,
    LEXA_ID,
    REGULON_DETAIL,
    regulon_list_for_search,
)
from dnadesign.cruncher.utils.run_layout import (
    config_used_path,
    logos_dir_for_run,
    manifest_path,
    out_root,
    sequences_path,
)
from dnadesign.cruncher.workflows.parse_workflow import run_parse
from dnadesign.cruncher.workflows.sample_workflow import run_sample


def _sample_block() -> dict:
    return {
        "mode": "sample",
        "rng": {"seed": 7, "deterministic": True},
        "budget": {"draws": 2, "tune": 1, "restarts": 1},
        "init": {"kind": "random", "length": 12, "pad_with": "background"},
        "objective": {"bidirectional": True, "score_scale": "llr"},
        "elites": {"k": 1, "min_hamming": 0, "filters": {"pwm_sum_min": 0.0}},
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
        "optimizer": {"name": "gibbs"},
        "optimizers": {"gibbs": {"beta_schedule": {"kind": "fixed", "beta": 1.0}, "apply_during": "tune"}},
        "auto_opt": {"enabled": False},
        "output": {"trace": {"save": False}, "save_sequences": True},
        "ui": {"progress_bar": False, "progress_every": 0},
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
    config = {
        "cruncher": {
            "out_dir": "results",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": ".cruncher",
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
            "parse": {"plot": {"logo": True, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(),
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=True, ht_sites=False),
        transport=_fixture_transport,
    )
    catalog_root = tmp_path / ".cruncher"
    fetch_sites(adapter, catalog_root, names=["lexA", "cpxR"])
    fetch_motifs(adapter, catalog_root, names=["lexA", "cpxR"])

    lock_path = catalog_root / "locks" / "config.lock.json"
    resolve_lock(
        names=["lexA", "cpxR"],
        catalog_root=catalog_root,
        pwm_source="sites",
        lock_path=lock_path,
    )

    cfg = load_config(config_path)
    run_parse(cfg, config_path)
    run_sample(cfg, config_path)

    results_dir = tmp_path / "results"

    def _find_runs(stage_dir: Path) -> list[Path]:
        runs: list[Path] = []
        for child in stage_dir.iterdir():
            if not child.is_dir():
                continue
            if manifest_path(child).exists():
                runs.append(child)
                continue
            for grand in child.iterdir():
                if grand.is_dir() and manifest_path(grand).exists():
                    runs.append(grand)
        return runs

    parse_runs = _find_runs(results_dir / "parse")
    sample_runs = _find_runs(results_dir / "sample")
    assert parse_runs
    assert sample_runs
    parse_dir = parse_runs[0]
    sample_dir = sample_runs[0]

    logo_dir = logos_dir_for_run(out_root(config_path, cfg.out_dir), "parse", parse_dir.name)
    assert (logo_dir / "lexA_logo.png").exists()
    assert (logo_dir / "cpxR_logo.png").exists()
    assert config_used_path(sample_dir).exists()
    assert sequences_path(sample_dir).exists()
    assert not (sample_dir / "artifacts" / "trace.nc").exists()


def test_demo_workspace_cli_without_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "regulondb_ecoli"
    workspace.mkdir()
    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {
                "catalog_root": ".cruncher",
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
            "parse": {"plot": {"logo": True, "bits_mode": "information", "dpi": 72}},
            "sample": _sample_block(),
        }
    }
    config_path = workspace / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    adapter = RegulonDBAdapter(
        RegulonDBAdapterConfig(curated_sites=True, ht_sites=False),
        transport=_fixture_transport,
    )
    catalog_root = workspace / ".cruncher"
    fetch_sites(adapter, catalog_root, names=["lexA", "cpxR"])
    fetch_motifs(adapter, catalog_root, names=["lexA", "cpxR"])

    lock_path = catalog_root / "locks" / "config.lock.json"
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
