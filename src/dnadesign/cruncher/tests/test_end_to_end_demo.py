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
from dnadesign.cruncher.workflows.parse_workflow import run_parse
from dnadesign.cruncher.workflows.sample_workflow import run_sample

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
            "sample": {
                "bidirectional": True,
                "seed": 7,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": False,
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "draws": 2,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 1,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.1,
                },
                "save_sequences": True,
            },
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
    parse_runs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("parse_")]
    sample_runs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("sample_")]
    assert parse_runs
    assert sample_runs
    parse_dir = parse_runs[0]
    sample_dir = sample_runs[0]

    assert (parse_dir / "lexA_logo.png").exists()
    assert (parse_dir / "cpxR_logo.png").exists()
    assert (sample_dir / "config_used.yaml").exists()
    assert (sample_dir / "sequences.parquet").exists()
    assert not (sample_dir / "trace.nc").exists()


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
            "sample": {
                "bidirectional": True,
                "seed": 7,
                "record_tune": False,
                "progress_bar": False,
                "progress_every": 0,
                "save_trace": False,
                "init": {"kind": "random", "length": 12, "pad_with": "background"},
                "draws": 2,
                "tune": 1,
                "chains": 1,
                "min_dist": 0,
                "top_k": 1,
                "moves": {
                    "block_len_range": [2, 2],
                    "multi_k_range": [2, 2],
                    "slide_max_shift": 1,
                    "swap_len_range": [2, 2],
                    "move_probs": {"S": 0.8, "B": 0.1, "M": 0.1},
                },
                "optimiser": {
                    "kind": "gibbs",
                    "scorer_scale": "llr",
                    "cooling": {"kind": "fixed", "beta": 1.0},
                    "swap_prob": 0.1,
                },
                "save_sequences": True,
            },
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
