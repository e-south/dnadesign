"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_parse_logo_flag.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.workflows.parse_workflow import run_parse


def _write_motif(path: Path, *, source: str, motif_id: str, tf_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": {"source": source, "motif_id": motif_id, "tf_name": tf_name},
        "matrix": [[0.25, 0.25, 0.25, 0.25]],
        "checksums": {"sha256_norm": "good"},
    }
    path.write_text(json.dumps(payload))


def test_parse_skips_logos_when_disabled(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
        kind="PFM",
        has_matrix=True,
        matrix_source="alignment",
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)

    motif_path = catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json"
    _write_motif(motif_path, source="regulondb", motif_id="RBM1", tf_name="lexA")

    lock_path = catalog_root / "locks" / "config.lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pwm_source": "matrix",
                "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "good"}},
            }
        )
    )

    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": ".cruncher", "pwm_source": "matrix"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    run_parse(cfg, config_path)

    out_dir = tmp_path / "runs"
    parse_runs = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("parse_")]
    assert len(parse_runs) == 1
    parse_dir = parse_runs[0]
    assert (parse_dir / "run_manifest.json").exists()
    assert (parse_dir / "run_status.json").exists()
    assert not list(parse_dir.glob("*_logo.png"))
