"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_regulator_sets_runs.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.cruncher.app.fetch_service import write_motif_record
from dnadesign.cruncher.app.sample_workflow import run_sample
from dnadesign.cruncher.artifacts.layout import config_used_path, manifest_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.ingest.normalize import build_motif_record
from dnadesign.cruncher.store.catalog_index import CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_lock_path


def _write_motif(catalog_root: Path, *, tf_name: str, motif_id: str) -> None:
    record = build_motif_record(
        source="regulondb",
        motif_id=motif_id,
        tf_name=tf_name,
        matrix=[[0.25, 0.25, 0.25, 0.25]],
        matrix_semantics="probabilities",
        organism=None,
        raw_payload="{}",
    )
    write_motif_record(catalog_root, record)
    catalog = CatalogIndex.load(catalog_root)
    catalog.upsert_from_record(record)
    catalog.save(catalog_root)


def _motif_checksum(catalog_root: Path, motif_id: str) -> str:
    payload = json.loads((catalog_root / "normalized" / "motifs" / "regulondb" / f"{motif_id}.json").read_text())
    return payload["checksums"]["sha256_norm"]


def test_sample_runs_split_by_regulator_set(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    _write_motif(catalog_root, tf_name="lexA", motif_id="RBM1")
    _write_motif(catalog_root, tf_name="cpxR", motif_id="RBM2")

    lock_payload = {
        "pwm_source": "matrix",
        "resolved": {
            "lexA": {
                "source": "regulondb",
                "motif_id": "RBM1",
                "sha256": _motif_checksum(catalog_root, "RBM1"),
            },
            "cpxR": {
                "source": "regulondb",
                "motif_id": "RBM2",
                "sha256": _motif_checksum(catalog_root, "RBM2"),
            },
        },
    }
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"], ["cpxR"]]},
            "catalog": {
                "root": str(catalog_root),
                "source_preference": ["regulondb"],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
                "min_sites_for_pwm": 2,
                "allow_low_sites": False,
            },
            "ingest": {"regulondb": {"min_sites_for_pwm": 2}},
            "sample": {
                "seed": 11,
                "sequence_length": 6,
                "budget": {"tune": 1, "draws": 2},
                "pt": {"n_temps": 2, "temp_max": 10.0},
                "objective": {"bidirectional": True, "score_scale": "llr", "combine": "min"},
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
                        "move_probs": {"S": 1.0, "B": 0.0, "M": 0.0},
                    },
                },
                "output": {
                    "save_trace": False,
                    "save_sequences": True,
                    "include_tune_in_sequences": False,
                    "live_metrics": False,
                },
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    cfg = load_config(config_path)
    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock_payload))
    run_sample(cfg, config_path)

    results_dir = tmp_path / "results" / "runs"
    runs = sorted(path.parent for path in results_dir.rglob("run_manifest.json"))
    assert len(runs) == 2
    for run_dir in runs:
        manifest_file = manifest_path(run_dir)
        cfg_path = config_used_path(run_dir)
        assert manifest_file.exists()
        assert cfg_path.exists()
        manifest = json.loads(manifest_file.read_text())
        assert "regulator_set" in manifest
        config_used = yaml.safe_load(cfg_path.read_text())["cruncher"]
        assert "active_regulator_set" in config_used
