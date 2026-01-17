from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.artifacts.layout import elites_path


def find_elites_parquet(run_dir: Path) -> Path:
    path = elites_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `cruncher sample` to generate elites.parquet.")
    return path
