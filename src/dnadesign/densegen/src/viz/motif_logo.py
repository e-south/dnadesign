"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/motif_logo.py

Render sequence logos from DenseGen pwm_artifact inputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.stage_a.stage_a_types import PWMMotif
from .plot_common import _apply_style, _save_figure, _style


def render_pwm_logo(
    motif: PWMMotif,
    out_path: Path,
    *,
    title: str,
    subtitle: str | None = None,
    mode: str = "information",
    style: Optional[dict] = None,
) -> tuple[Path, Path]:
    """Render one PWM logo and write the primary artifact plus its SVG sibling."""
    if mode not in {"information", "probability"}:
        raise ValueError(f"Unsupported PWM logo mode: {mode}")

    raw_style = style or {}
    logo_style = _style(raw_style)
    if "figsize" not in raw_style:
        logo_style["figsize"] = (8.0, 3.0)
    logo_style.setdefault("save_dpi", 220.0)

    df = pd.DataFrame(motif.matrix, columns=["A", "C", "G", "T"], dtype=float)
    if mode == "information":
        df = logomaker.transform_matrix(
            df,
            from_type="probability",
            to_type="information",
        )

    data = df.to_numpy(dtype=float)
    flat = False
    min_val = max_val = 0.0
    if data.size:
        min_val = float(data.min())
        max_val = float(data.max())
        flat = np.isclose(min_val, max_val)
    if flat:
        # Avoid singular y-limits in Logomaker for flat matrices.
        df.iloc[0, 0] = max_val + 1.0e-3

    fig, ax = plt.subplots(figsize=logo_style["figsize"])
    try:
        logomaker.Logo(df, ax=ax, shade_below=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, pad=20 if subtitle else None)
        if subtitle:
            ax.text(
                0.5,
                1.02,
                subtitle,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="dimgray",
            )
        ax.set_xlabel("Position")
        ax.set_ylabel("Information Content (bits)" if mode == "information" else "Probability")
        ax.axhline(0, color="black", lw=0.5)
        if flat:
            ax.set_ylim(min_val - 1.0, max_val + 1.0)
        _apply_style(ax, logo_style)
        ax.grid(False)
        fig.tight_layout()
        _save_figure(fig, out_path, style=logo_style)
    finally:
        plt.close(fig)

    return out_path, out_path.with_suffix(".svg")
