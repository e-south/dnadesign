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


def plot_pwm(
    pwm: PWM,
    mode: str = "information",
    out: Path | None = None,
    dpi: int = 150,
) -> None:
    """Render a sequence logo of the PWM and save to disk."""
    df = pd.DataFrame(pwm.matrix, columns=list(pwm.alphabet))

    if mode == "information":
        df = logomaker.transform_matrix(
            df,
            from_type="probability",
            to_type="information",
        )

    fig, ax = plt.subplots()
    logomaker.Logo(df, ax=ax, shade_below=0.5)

    # strip top & right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title and labels
    title = pwm.name
    if pwm.nsites is not None and pwm.evalue is not None:
        title += f" (n={pwm.nsites}, E={pwm.evalue:.1e})"
    ax.set_title(title)
    ax.set_xlabel("Position")
    ylabel = "Information Content (bits)" if mode == "information" else "Probability"
    ax.set_ylabel(ylabel)

    ax.axhline(0, color="black", lw=0.5)

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi)
    plt.close(fig)
