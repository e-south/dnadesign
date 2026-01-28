from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.config import load_config
from dnadesign.densegen.src.core.reporting import write_report
from dnadesign.densegen.tests.config_fixtures import write_minimal_config


def test_report_does_not_write_assets_dir(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True)
    cfg_path = run_root / "config.yaml"
    write_minimal_config(cfg_path)
    (run_root / "inputs.csv").write_text("tf,tfbs\n")

    tables_root = run_root / "outputs" / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "solution_id": "s1",
                "attempt_id": "a1",
                "placement_index": 0,
                "tf": "lexA",
                "tfbs": "AAA",
                "motif_id": "m1",
                "tfbs_id": "t1",
                "orientation": "fwd",
                "offset": 0,
                "library_hash": "abc123",
            }
        ]
    ).to_parquet(tables_root / "composition.parquet", index=False)

    loaded = load_config(cfg_path)
    write_report(loaded.root, cfg_path)

    assets_root = run_root / "outputs" / "report" / "assets"
    assert not assets_root.exists()
