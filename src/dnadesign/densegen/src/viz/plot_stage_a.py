"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/viz/plot_stage_a.py

Stage-A summary plotting (tiers, yield/bias, and core diversity).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..core.artifacts.pool import TFBSPoolArtifact
from .plot_common import _safe_filename, _style
from .plot_stage_a_diversity import _build_stage_a_diversity_figure
from .plot_stage_a_strata import _build_stage_a_strata_overview_figure
from .plot_stage_a_yield import _build_stage_a_yield_bias_figure


def plot_stage_a_summary(
    df: pd.DataFrame,
    out_path: Path,
    *,
    pools: dict[str, pd.DataFrame] | None = None,
    pool_manifest: TFBSPoolArtifact | None = None,
    style: Optional[dict] = None,
) -> list[Path]:
    if pools is None or pool_manifest is None:
        raise ValueError("Stage-A summary requires pool manifests; run stage-a build-pool first.")
    raw_style = style or {}
    style = _style(raw_style)
    style["seaborn_style"] = False
    if "figsize" not in raw_style:
        style["figsize"] = (11, 4)
    paths: list[Path] = []
    for input_name, pool_df in pools.items():
        entry = pool_manifest.entry_for(input_name)
        sampling = entry.stage_a_sampling
        if sampling is None:
            raise ValueError(f"Stage-A sampling metadata missing for input '{input_name}'.")
        eligible_hist = sampling.get("eligible_score_hist") or []
        if not eligible_hist:
            raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
        for row in eligible_hist:
            if row.get("diversity") is None:
                raise ValueError(
                    f"Stage-A diversity metrics missing for input '{input_name}' ({row.get('regulator')}). "
                    "Rebuild Stage-A pools."
                )
        fig, _, _ = _build_stage_a_strata_overview_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)
        paths.append(path)

        fig2, _, _, _ = _build_stage_a_yield_bias_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__yield_bias{out_path.suffix}"
        path2 = out_path.parent / fname
        fig2.savefig(path2, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig2)
        paths.append(path2)

        fig3, _, _ = _build_stage_a_diversity_figure(
            input_name=input_name,
            pool_df=pool_df,
            sampling=sampling,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__diversity{out_path.suffix}"
        path3 = out_path.parent / fname
        fig3.savefig(path3, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig3)
        paths.append(path3)
    return paths
