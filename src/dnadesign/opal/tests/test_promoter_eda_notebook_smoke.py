from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import polars as pl

NOTEBOOK_PATH = Path("src/dnadesign/opal/notebooks/prom60_eda.py")


def _load_notebook_module() -> object:
    spec = importlib.util.spec_from_file_location("prom60_eda_smoke", NOTEBOOK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load prom60_eda notebook module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prom60_eda_headless(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr"
    dataset_dir = usr_root / "demo"
    dataset_dir.mkdir(parents=True)
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "cluster__ldn_v1__umap_x": [0.1, 0.2],
            "cluster__ldn_v1__umap_y": [0.2, 0.3],
        }
    )
    df.write_parquet(dataset_dir / "records.parquet")

    env_backup = os.environ.copy()
    os.environ["DNADESIGN_USR_ROOT"] = str(usr_root)
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    os.environ["MPLBACKEND"] = "Agg"
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    try:
        module = _load_notebook_module()
        outputs, defs = module.app.run(
            defs={
                "active_record": None,
                "active_record_id": None,
            }
        )
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    assert outputs is not None
    assert defs.get("dataset_name") == "demo"
