"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/plots/pssm.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations
from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import pandas as pd

from ..motif.model import PWM


def plot_pwm(pwm: PWM, mode: str = "information", out: Path | None = None, dpi: int = 150):
    """Render a sequence logo of the PWM."""
    df = pd.DataFrame(pwm.matrix, columns=list(pwm.alphabet))
    if mode == "information":
        df = logomaker.transform_matrix(df, from_type="probability", to_type="information")
    ax = logomaker.Logo(df, shade_below=0.5)
    ax.set_title(pwm.name)
    ax.axhline(0, color="black", lw=0.5)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=dpi)
    plt.close()
