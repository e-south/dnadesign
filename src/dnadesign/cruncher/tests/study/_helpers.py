"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/_helpers.py

Shared fixtures and setup helpers for Study workflow tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.utils.paths import resolve_lock_path


def _write_motif(path: Path, *, source: str, motif_id: str, tf_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": {"source": source, "motif_id": motif_id, "tf_name": tf_name},
        "matrix": [[0.7, 0.1, 0.1, 0.1]] * 3,
        "checksums": {"sha256_norm": "good"},
    }
    path.write_text(json.dumps(payload))


def write_workspace_config(tmp_path: Path) -> Path:
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
            "schema_version": 3,
            "workspace": {"out_dir": "runs", "regulator_sets": [["lexA", "cpxR"]]},
            "catalog": {"root": str(catalog_root), "pwm_source": "matrix"},
            "sample": {
                "seed": 3,
                "sequence_length": 6,
                "budget": {"tune": 1, "draws": 1},
                "optimizer": {
                    "kind": "gibbs_anneal",
                    "chains": 1,
                    "cooling": {"kind": "fixed", "beta": 1.0},
                },
                "objective": {"bidirectional": True, "score_scale": "normalized-llr", "combine": "min"},
                "elites": {
                    "k": 1,
                    "select": {"diversity": 0.0, "pool_size": "auto"},
                },
                "moves": {
                    "profile": "balanced",
                    "overrides": {"move_probs": {"S": 1.0, "B": 0.0, "M": 0.0}},
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
    return config_path


def write_study_spec(
    spec_path: Path,
    *,
    profile: str,
    mmr_enabled: bool,
    seeds: list[int],
    trials: list[dict[str, object]],
    parallelism: int = 1,
    on_trial_error: str = "continue",
) -> None:
    payload = {
        "study": {
            "schema_version": 3,
            "name": "smoke_study",
            "base_config": "config.yaml",
            "target": {"kind": "regulator_set", "set_index": 1},
            "execution": {
                "parallelism": int(parallelism),
                "on_trial_error": str(on_trial_error),
                "exit_code_policy": "nonzero_if_any_error",
                "summarize_after_run": True,
            },
            "artifacts": {"trial_output_profile": profile},
            "replicates": {"seed_path": "sample.seed", "seeds": seeds},
            "trials": trials,
            "replays": {
                "mmr_sweep": {
                    "enabled": mmr_enabled,
                    "pool_size_values": ["auto"],
                    "diversity_values": [0.0, 0.5, 1.0],
                }
            },
        }
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(payload))
