"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/parse/run_parse.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.cruncher.parse.plots.pssm import plot_pwm
from dnadesign.cruncher.parse.registry import Registry
from dnadesign.cruncher.utils.config import CruncherConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MOTIF_ROOT = (
    PROJECT_ROOT.parent
    / "dnadesign-data"
    / "primary_literature"
    / "OMalley_et_al"
    / "escherichia_coli_motifs"
)


def _motif_root(cfg: CruncherConfig) -> Path:
    if cfg.parse.motif_root is not None:
        return cfg.parse.motif_root
    return DEFAULT_MOTIF_ROOT


def run_parse(cfg: CruncherConfig, base_out: Path) -> None:
    """
    Load each PWM for all regulators, generate a logo (PNG) and print summary
    (length, information bits, and full log-odds matrix) to stdout.

    Inputs:
      - cfg.parse.formats        : mapping of file-extensions → parser names
      - cfg.regulator_sets       : list of TF groups; we flatten all TF names
      - cfg.parse.plot.bits_mode : "information" or "probability" (passed to plot_pwm)
      - cfg.parse.plot.dpi       : DPI for each saved logo

    Outputs (flat under `base_out`):
      - <base_out>/motif_logos/<tf>_logo.png  (one per TF)
      - Console: "[OK] <tf> L=<length> bits=<info_bits>"
                 followed by the log-odds matrix (pandas DataFrame) printout

    Error behavior:
      - If `Registry(PWM_PATH, cfg.parse.formats)` fails (e.g. PWM_PATH not found),
        we allow that exception to propagate (abort immediately).
      - If loading a particular TF's PWM raises FileNotFoundError or ValueError,
        we abort immediately.
    """
    # Initialize the Registry (will raise if PWM_PATH does not exist)
    reg = Registry(_motif_root(cfg), cfg.parse.formats)

    # Prepare output folder
    out_base = base_out
    out_base.mkdir(parents=True, exist_ok=True)

    # Flatten all TF names and remove duplicates
    unique_regs = set(sum(cfg.regulator_sets, []))

    for tf in sorted(unique_regs):
        # Attempt to load that TF’s PWM (will raise if file not found or malformed)
        pwm = reg.load(tf)

        # Save a sequence-logo plot for this PWM:
        plot_pwm(
            pwm,
            mode=cfg.parse.plot.bits_mode,
            out=out_base / f"{tf}_logo.png",
            dpi=cfg.parse.plot.dpi,
        )

        # Print a short summary line (length, information bits)
        length = pwm.length
        bits = pwm.information_bits()
        print(f"[OK] {tf:8s} L={length:2d} bits={bits:4.1f}")

        # Compute or retrieve the log-odds matrix and print it as a DataFrame
        log_odds = pwm.log_odds()
        df_lo = pd.DataFrame(
            log_odds,
            columns=list(pwm.alphabet),
            index=[f"pos{i+1}" for i in range(log_odds.shape[0])],
        )
        print(f"Log-odds matrix for {tf}:")
        print(df_lo.to_string(float_format=lambda x: f"{x:.3f}"))
        print()

    print("Parse stage complete.")
