"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/io/plots/pssm.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.cruncher.core.pwm import PWM


def plot_pwm(
    pwm: PWM,
    mode: str = "information",
    out: Path | None = None,
    dpi: int = 150,
) -> None:
    """Render a sequence logo of the PWM and save to disk, using the _filename_ for the title’s case."""
    # Build DataFrame in float64 (pandas default) so LogoMaker's internal assignments
    # stay in float64 and avoid dtype incompatibility warnings.
    df = pd.DataFrame(
        pwm.matrix,
        columns=list(pwm.alphabet),
        dtype=float,
    )

    if mode == "information":
        df = logomaker.transform_matrix(
            df,
            from_type="probability",
            to_type="information",
        )

    # Draw logo
    fig, ax = plt.subplots(figsize=(8, 3))
    logomaker.Logo(df, ax=ax, shade_below=0.5)

    # Remove top & right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title: prefer the output filename (preserves Finder-case), else pwm.name
    if out:
        # e.g. "cpxR_logo.png" → stem "cpxR_logo" → strip trailing "_logo"
        stem = out.stem
        if stem.lower().endswith("_logo"):
            title_text = stem[: -len("_logo")]
        else:
            title_text = stem
    else:
        title_text = pwm.name

    if pwm.nsites is not None:
        title_text += f" (n={pwm.nsites})"
    if pwm.evalue is not None:
        title_text += f" E={pwm.evalue:.1e}"

    ax.set_title(title_text)
    ax.set_xlabel("Position")
    ax.set_ylabel("Information Content (bits)" if mode == "information" else "Probability")

    # Zero‐reference line
    ax.axhline(0, color="black", lw=0.5)

    fig.tight_layout()

    # Save or close
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
