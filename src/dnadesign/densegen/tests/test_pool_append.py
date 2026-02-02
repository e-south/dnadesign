from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from dnadesign.densegen.src.adapters.sources import data_source_factory
from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.artifacts.pool import build_pool_artifact, load_pool_artifact


class _Deps:
    def __init__(self) -> None:
        self.source_factory = data_source_factory


def _write_config(path: Path, input_path: Path) -> None:
    cfg = {
        "densegen": {
            "schema_version": "2.8",
            "run": {"id": "demo", "root": "."},
            "inputs": [
                {
                    "name": "demo_input",
                    "type": "binding_sites",
                    "path": str(input_path),
                    "format": "csv",
                }
            ],
            "output": {
                "targets": ["parquet"],
                "schema": {"bio_type": "dna", "alphabet": "dna_4"},
                "parquet": {"path": "outputs/tables/dense_arrays.parquet"},
            },
            "generation": {
                "sequence_length": 3,
                "quota": 1,
                "plan": [
                    {
                        "name": "demo_plan",
                        "quota": 1,
                        "regulator_constraints": {
                            "groups": [
                                {
                                    "name": "all",
                                    "members": ["TF1"],
                                    "min_required": 1,
                                }
                            ]
                        },
                    }
                ],
            },
            "solver": {"backend": "CBC", "strategy": "iterate"},
            "runtime": {"random_seed": 1},
            "postprocess": {"pad": {"mode": "off"}},
            "logging": {"log_dir": "outputs/logs", "level": "INFO"},
        }
    }
    path.write_text(yaml.safe_dump(cfg))


def test_build_pool_appends_without_overwrite(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF1,CCC\n")
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, csv_path)

    loaded = load_config(cfg_path)
    deps = _Deps()
    rng = np.random.default_rng(1)
    outputs_root = tmp_path / "outputs"
    out_dir = outputs_root / "pools"

    build_pool_artifact(
        cfg=loaded.root.densegen,
        cfg_path=cfg_path,
        deps=deps,
        rng=rng,
        outputs_root=outputs_root,
        out_dir=out_dir,
        overwrite=False,
    )

    build_pool_artifact(
        cfg=loaded.root.densegen,
        cfg_path=cfg_path,
        deps=deps,
        rng=rng,
        outputs_root=outputs_root,
        out_dir=out_dir,
        overwrite=False,
    )

    artifact = load_pool_artifact(out_dir)
    entry = artifact.entry_for("demo_input")
    df = pd.read_parquet(out_dir / entry.pool_path)
    assert set(df["tfbs"].tolist()) == {"AAA", "CCC"}


def test_build_pool_requires_fresh_on_input_change(tmp_path: Path) -> None:
    csv_path = tmp_path / "sites.csv"
    csv_path.write_text("tf,tfbs\nTF1,AAA\nTF1,CCC\n")
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, csv_path)

    loaded = load_config(cfg_path)
    deps = _Deps()
    rng = np.random.default_rng(1)
    outputs_root = tmp_path / "outputs"
    out_dir = outputs_root / "pools"

    build_pool_artifact(
        cfg=loaded.root.densegen,
        cfg_path=cfg_path,
        deps=deps,
        rng=rng,
        outputs_root=outputs_root,
        out_dir=out_dir,
        overwrite=False,
    )

    csv_path.write_text("tf,tfbs\nTF1,GGG\n")
    with pytest.raises(ValueError, match="fresh"):
        build_pool_artifact(
            cfg=loaded.root.densegen,
            cfg_path=cfg_path,
            deps=deps,
            rng=rng,
            outputs_root=outputs_root,
            out_dir=out_dir,
            overwrite=False,
        )
