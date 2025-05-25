"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/main.py

    python -m dnadesign.cruncher        # uses configs/example.yaml
    python -m dnadesign.cruncher path/to/other.yaml

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Set

import numpy as np

from dnadesign.cruncher.config import load_yaml
from dnadesign.cruncher.motif.registry import Registry
from dnadesign.cruncher.plots.pssm import plot_pwm


# helpers
def _run_parse(cfg):
    """Dry-run: read PWMs, draw logos, print stats."""
    reg = Registry(cfg.motif.root, cfg.motif.formats)

    tf_names: Set[str] = set()
    if cfg.sample and "pairs" in cfg.sample:
        for a, b in cfg.sample["pairs"]:
            tf_names.update((a, b))

    if not tf_names:
        print("No TF names found in config → nothing to parse.")
        return

    out_dir = Path("results/motif_logos")
    out_dir.mkdir(parents=True, exist_ok=True)

    for tf in sorted(tf_names):
        pwm = reg.load(tf)
        plot_pwm(
            pwm,
            mode=cfg.motif.plot.bits_mode,
            out=out_dir / f"{tf}_logo.png",
            dpi=cfg.motif.plot.dpi,
        )
        print(f"[OK] {tf:8s}  L={pwm.length:2d}  bits={pwm.information_bits():4.1f}")

    print("Dry-run complete.  No sampling executed.")


def _run_sample(cfg):
    """Placeholder for upcoming optimiser stage."""
    print("Sample mode selected - optimiser not implemented yet.")
    # Here you would import Scorer / optimiser etc.


def _run_analyse(cfg):
    """Placeholder for analysis stage."""
    print("Analyse mode selected - analysis not implemented yet.")


# dispatcher
def main(config_path: str | Path | None = None):
    if config_path is None:
        # default to repo_root/configs/example.yaml
        config_path = (
            Path(__file__).resolve().parents[2] / "dnadesign" / "configs" / "example.yaml"
        )
    cfg = load_yaml(Path(config_path))

    if cfg.mode == "parse":
        _run_parse(cfg)
    elif cfg.mode == "sample":
        _run_sample(cfg)
    elif cfg.mode == "analyse":
        _run_analyse(cfg)
    else:  # should never happen
        raise ValueError(f"Unknown mode {cfg.mode!r}")


# allow  `python -m dnadesign.cruncher <optional_yaml>`
if __name__ == "__main__":
    chosen_cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(chosen_cfg)
