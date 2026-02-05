"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_sample_smoke.py

CLI smoke test for a minimal two-TF sampling run (matrix PWMs).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.artifacts.layout import manifest_path
from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_lock_path

runner = CliRunner()


def _write_motif(path: Path, *, source: str, motif_id: str, tf_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": {"source": source, "motif_id": motif_id, "tf_name": tf_name},
        "matrix": [[0.7, 0.1, 0.1, 0.1]] * 3,
        "checksums": {"sha256_norm": "good"},
    }
    path.write_text(json.dumps(payload))


def _find_runs(stage_dir: Path) -> list[Path]:
    runs: list[Path] = []
    if not stage_dir.exists():
        return runs
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


def test_sample_cli_smoke_matrix(tmp_path: Path) -> None:
    catalog_root = tmp_path / ".cruncher"
    entries = {
        "regulondb:RBM1": CatalogEntry(
            source="regulondb",
            motif_id="RBM1",
            tf_name="lexA",
            kind="PFM",
            has_matrix=True,
            matrix_source="alignment",
        ),
        "regulondb:RBM2": CatalogEntry(
            source="regulondb",
            motif_id="RBM2",
            tf_name="cpxR",
            kind="PFM",
            has_matrix=True,
            matrix_source="alignment",
        ),
    }
    CatalogIndex(entries=entries).save(catalog_root)
    _write_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM1.json",
        source="regulondb",
        motif_id="RBM1",
        tf_name="lexA",
    )
    _write_motif(
        catalog_root / "normalized" / "motifs" / "regulondb" / "RBM2.json",
        source="regulondb",
        motif_id="RBM2",
        tf_name="cpxR",
    )

    config = {
        "cruncher": {
            "out_dir": "runs",
            "regulator_sets": [["lexA", "cpxR"]],
            "motif_store": {"catalog_root": str(catalog_root), "pwm_source": "matrix"},
            "parse": {"plot": {"logo": False, "bits_mode": "information", "dpi": 72}},
            "sample": {
                "mode": "sample",
                "rng": {"seed": 3, "deterministic": True},
                "budget": {"draws": 2, "tune": 1, "restarts": 1},
                "early_stop": {"enabled": False},
                "init": {"kind": "random", "length": 6, "pad_with": "background"},
                "objective": {"bidirectional": True, "score_scale": "normalized-llr"},
                "elites": {"k": 1, "filters": {"pwm_sum_min": 0.0}},
                "moves": {"profile": "balanced"},
                "optimizer": {"name": "pt"},
                "optimizers": {"pt": {"beta_ladder": {"kind": "geometric", "betas": [1.0, 0.5]}}},
                "auto_opt": {"enabled": False},
                "output": {"trace": {"save": False}, "save_sequences": True},
                "ui": {"progress_bar": False, "progress_every": 0},
            },
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    lock_path = resolve_lock_path(config_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(
        json.dumps(
            {
                "pwm_source": "matrix",
                "resolved": {
                    "lexA": {"source": "regulondb", "motif_id": "RBM1", "sha256": "good"},
                    "cpxR": {"source": "regulondb", "motif_id": "RBM2", "sha256": "good"},
                },
            }
        )
    )

    result = runner.invoke(app, ["sample", "-c", str(config_path)])
    assert result.exit_code == 0
    sample_runs = _find_runs(tmp_path / "runs" / "sample")
    assert sample_runs
