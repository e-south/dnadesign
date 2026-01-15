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

from dnadesign.cruncher.app.parse_workflow import run_parse
from dnadesign.cruncher.artifacts.layout import logos_dir_for_run, manifest_path, out_root, status_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_lock_path


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

    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": str(catalog_root), "pwm_source": "matrix"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pwm_source": "matrix",
                "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "good"}},
            }
        )
    )

    run_parse(cfg, config_path)

    out_dir = tmp_path / "runs" / "parse"
    parse_runs = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        if manifest_path(child).exists():
            parse_runs.append(child)
            continue
        for grand in child.iterdir():
            if grand.is_dir() and manifest_path(grand).exists():
                parse_runs.append(grand)
    assert len(parse_runs) == 1
    parse_dir = parse_runs[0]
    assert manifest_path(parse_dir).exists()
    assert status_path(parse_dir).exists()
    logo_dir = logos_dir_for_run(out_root(config_path, cfg.out_dir), "parse", parse_dir.name)
    assert not logo_dir.exists() or not list(logo_dir.glob("*_logo.png"))


def test_parse_is_idempotent_when_inputs_match(tmp_path: Path) -> None:
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

    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA"]],
            "motif_store": {"catalog_root": str(catalog_root), "pwm_source": "matrix"},
            "parse": {"plot": {"logo": True, "bits_mode": "information", "dpi": 72}},
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cfg = load_config(config_path)

    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pwm_source": "matrix",
                "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "good"}},
            }
        )
    )

    run_parse(cfg, config_path)
    lock_path.write_text(
        json.dumps(
            {
                "pwm_source": "matrix",
                "generated_at": "2026-01-14T12:00:00Z",
                "resolved": {"lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "good"}},
            }
        )
    )
    run_parse(cfg, config_path)

    out_dir = tmp_path / "runs" / "parse"
    parse_runs = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        if manifest_path(child).exists():
            parse_runs.append(child)
            continue
        for grand in child.iterdir():
            if grand.is_dir() and manifest_path(grand).exists():
                parse_runs.append(grand)
    assert len(parse_runs) == 1
    manifest = json.loads(manifest_path(parse_runs[0]).read_text())
    assert manifest.get("parse_signature")
