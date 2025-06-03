"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/main.py

Main entry point for the cruncher package. Loads the config, computes the batch
path, and calls into the appropriate run_parse, run_sample, or run_analyse subprocess.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from dnadesign.cruncher.analyse.run_analyse import run_analyse
from dnadesign.cruncher.parse.run_parse import run_parse
from dnadesign.cruncher.sample.run_sample import run_sample
from dnadesign.cruncher.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
)

# Compute PROJECT_ROOT (four levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Default config.yaml inside the cruncher folder
CFG_PATH = PROJECT_ROOT / "dnadesign" / "src" / "dnadesign" / "cruncher" / "config.yaml"


def main(cfg_path: str | Path | None = None) -> None:
    """
    Dispatch entrypoint: load config, compute the batch directory name,
    and call the appropriate run_<mode> function.
    """
    cfg_file = Path(cfg_path) if cfg_path else CFG_PATH
    cfg = load_config(cfg_file)

    # Build a unique batch name: <mode>_<tf1>-<tf2>-<YYYYMMDD>
    today = datetime.now().strftime("%Y%m%d")
    regs = [r for group in cfg.regulator_sets for r in group]
    regs_label = "-".join(regs)
    batch_name = f"{cfg.mode}_{regs_label}_{today}"

    # Create (or reuse) batch directory under:
    # <PROJECT_ROOT>/dnadesign/src/dnadesign/cruncher/<cfg.out_dir>/<batch_name>
    batch = PROJECT_ROOT / "dnadesign" / "src" / "dnadesign" / "cruncher" / cfg.out_dir / batch_name
    # If it already exists, overwrite is okay (exist_ok=True)
    batch.mkdir(parents=True, exist_ok=True)

    match cfg.mode:
        case "parse":
            run_parse(cfg, batch)
        case "sample":
            run_sample(cfg, batch)
        case "analyse" | "analyze":
            run_analyse(cfg, batch)
        case other:
            raise ValueError(f"Unknown mode '{other}'")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
