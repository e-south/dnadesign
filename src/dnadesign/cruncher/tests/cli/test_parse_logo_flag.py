"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_parse_logo_flag.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.app.parse_workflow import run_parse
from dnadesign.cruncher.artifacts.layout import (
    build_run_dir,
    lockfile_snapshot_path,
    logos_root,
    out_root,
    parse_manifest_path,
    pwm_summary_path,
    run_optimize_dir,
    run_output_dir,
    run_plots_dir,
)
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
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix"},
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

    parse_dir = build_run_dir(
        config_path=config_path,
        out_dir=cfg.out_dir,
        stage="parse",
        tfs=["lexA"],
        set_index=1,
        include_set_index=False,
    )
    assert parse_manifest_path(parse_dir).exists()
    assert pwm_summary_path(parse_dir).exists()
    assert lockfile_snapshot_path(parse_dir).exists()
    assert not (tmp_path / "runs").exists()
    assert not run_optimize_dir(parse_dir).exists()
    assert not run_output_dir(parse_dir).exists()
    assert not run_plots_dir(parse_dir).exists()
    logo_dir = logos_root(out_root(config_path, cfg.out_dir))
    assert not logo_dir.exists() or not list(logo_dir.glob("*_logo.png"))


def test_parse_requires_force_overwrite_when_outputs_exist(tmp_path: Path) -> None:
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
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix"},
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
    with pytest.raises(ValueError, match="--force-overwrite"):
        run_parse(cfg, config_path)
    run_parse(cfg, config_path, force_overwrite=True)

    parse_dir = build_run_dir(
        config_path=config_path,
        out_dir=cfg.out_dir,
        stage="parse",
        tfs=["lexA"],
        set_index=1,
        include_set_index=False,
    )
    manifest = json.loads(parse_manifest_path(parse_dir).read_text())
    assert manifest.get("parse_signature")
