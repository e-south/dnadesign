"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_sample_abort.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.app.fetch_service import write_motif_record
from dnadesign.cruncher.app.run_service import load_run_index
from dnadesign.cruncher.app.sample_workflow import run_sample
from dnadesign.cruncher.artifacts.layout import status_path
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


def test_sample_abort_marks_run_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    catalog_root = tmp_path / ".cruncher"
    _write_motif(catalog_root, tf_name="lexA", motif_id="RBM1")

    lock_payload = {
        "pwm_source": "matrix",
        "resolved": {
            "lexA": {
                "source": "regulondb",
                "motif_id": "RBM1",
                "sha256": _motif_checksum(catalog_root, "RBM1"),
            }
        },
    }
    config = {
        "cruncher": {
            "schema_version": 3,
            "workspace": {"out_dir": "results", "regulator_sets": [["lexA"]]},
            "catalog": {
                "root": str(catalog_root),
                "source_preference": ["regulondb"],
                "allow_ambiguous": False,
                "pwm_source": "matrix",
                "min_sites_for_pwm": 1,
                "allow_low_sites": True,
            },
            "sample": {
                "seed": 11,
                "sequence_length": 6,
                "budget": {"tune": 1, "draws": 1},
                "objective": {"bidirectional": True, "score_scale": "llr", "combine": "min"},
                "elites": {
                    "k": 1,
                    "select": {"diversity": 0.0, "pool_size": "auto"},
                },
                "moves": {"profile": "balanced", "overrides": {"move_probs": {"S": 1.0, "B": 0.0, "M": 0.0}}},
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

    class AbortOptimizer:
        def __init__(self, **_kwargs) -> None:
            pass

        def optimise(self) -> None:
            raise KeyboardInterrupt("test interrupt")

    def _factory(_name: str):
        return AbortOptimizer

    monkeypatch.setattr("dnadesign.cruncher.core.optimizers.registry.get_optimizer", _factory)

    with pytest.raises(KeyboardInterrupt):
        run_sample(cfg, config_path)

    run_index = load_run_index(config_path)
    assert run_index
    entry = next(iter(run_index.values()))
    assert entry.get("status") == "aborted"
    status_file = status_path(Path(entry["run_dir"]))
    assert status_file.exists()
    status_payload = json.loads(status_file.read_text())
    assert status_payload.get("status") == "aborted"
