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

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..core.artifacts.pool import TFBSPoolArtifact
from .plot_common import _apply_style, _safe_filename, _style
from .plot_stage_a_diversity import _build_stage_a_diversity_figure
from .plot_stage_a_strata import _build_stage_a_strata_overview_figure
from .plot_stage_a_yield import _build_stage_a_yield_bias_figure


def _background_pwm_from_sequences(sequences: list[str]) -> np.ndarray:
    if not sequences:
        raise ValueError("Background logo requires at least one sequence.")
    length = len(sequences[0])
    if length <= 0:
        raise ValueError("Background logo sequences must have positive length.")
    base_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    counts = np.zeros((length, 4), dtype=float)
    for seq in sequences:
        if len(seq) != length:
            raise ValueError("Background logo sequences must have a consistent length per panel.")
        for idx, base in enumerate(seq.upper()):
            if base not in base_index:
                raise ValueError(f"Background logo sequence contains invalid base: {base}")
            counts[idx, base_index[base]] += 1.0
    totals = counts.sum(axis=1, keepdims=True)
    if np.any(totals == 0):
        raise ValueError("Background logo sequences cannot have empty position counts.")
    return counts / totals


def _build_background_logo_figure(
    *,
    input_name: str,
    sequences: list[str],
    style: dict,
) -> tuple[plt.Figure, list[plt.Axes]]:
    length_groups: dict[int, list[str]] = {}
    for seq in sequences:
        length_groups.setdefault(len(seq), []).append(seq)
    lengths = sorted(length_groups)
    if not lengths:
        raise ValueError("Background logo requires at least one sequence length.")
    n_lengths = len(lengths)
    if n_lengths <= 2:
        ncols = max(1, n_lengths)
    else:
        ncols = 2
    nrows = int(np.ceil(n_lengths / ncols))
    width = max(7.2, 3.6 * ncols)
    height = max(3.2, 2.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharey=True)
    axes_list = list(np.atleast_1d(axes).ravel())
    used_axes: list[plt.Axes] = []
    for ax, length in zip(axes_list, lengths):
        seqs = length_groups[length]
        pwm = _background_pwm_from_sequences(seqs)
        df = pd.DataFrame(pwm, columns=["A", "C", "G", "T"], dtype=float)
        logomaker.Logo(df, ax=ax, shade_below=0.5)
        ax.set_title(f"L={length} (n={len(seqs)})")
        ax.set_xlabel("Position")
        ax.set_ylabel("Probability")
        _apply_style(ax, style)
        used_axes.append(ax)
    for ax in axes_list[len(lengths) :]:
        ax.set_axis_off()
    fig.suptitle(f"{input_name} background logo", fontsize=float(style.get("title_size", 14)))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, used_axes


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
    pwm_inputs: list[tuple[str, pd.DataFrame, dict]] = []
    background_inputs: list[tuple[str, pd.DataFrame]] = []

    def _regulator_count(input_name: str, sampling: dict) -> int:
        eligible_hist = sampling.get("eligible_score_hist") or []
        if not eligible_hist:
            raise ValueError(f"Stage-A sampling missing eligible score histogram for input '{input_name}'.")
        regs: list[str] = []
        for row in eligible_hist:
            if "regulator" not in row:
                raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
            regs.append(str(row["regulator"]))
        if not regs:
            raise ValueError(f"Stage-A sampling missing regulator labels for input '{input_name}'.")
        return len(regs)

    for input_name, pool_df in pools.items():
        entry = pool_manifest.entry_for(input_name)
        if entry.input_type == "background_pool":
            background_inputs.append((input_name, pool_df))
            continue
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
        pwm_inputs.append((input_name, pool_df, sampling))

    if pwm_inputs:
        fig_width = float(style.get("figsize", (11, 4))[0])
        strata_heights = [
            max(3.8, 1.35 * _regulator_count(input_name, sampling) + 1.2)
            for input_name, _pool_df, sampling in pwm_inputs
        ]
        fig = plt.figure(figsize=(fig_width, float(sum(strata_heights))), constrained_layout=False)
        outer = fig.add_gridspec(
            nrows=len(pwm_inputs),
            ncols=1,
            height_ratios=strata_heights,
            hspace=0.4,
        )
        for idx, (input_name, pool_df, sampling) in enumerate(pwm_inputs):
            _build_stage_a_strata_overview_figure(
                input_name=input_name,
                pool_df=pool_df,
                sampling=sampling,
                style=style,
                fig=fig,
                slot=outer[idx, 0],
            )
        path = out_path.parent / f"{out_path.stem}__pool_tiers{out_path.suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)
        paths.append(path)

        base_height = float(style.get("figsize", (11, 4.2))[1])
        yield_heights = [
            max(4.8, base_height, 1.75 * _regulator_count(input_name, sampling) + 0.8)
            for input_name, _pool_df, sampling in pwm_inputs
        ]
        fig2 = plt.figure(figsize=(fig_width, float(sum(yield_heights))), constrained_layout=False)
        outer = fig2.add_gridspec(
            nrows=len(pwm_inputs),
            ncols=1,
            height_ratios=yield_heights,
            hspace=0.45,
        )
        for idx, (input_name, pool_df, sampling) in enumerate(pwm_inputs):
            _build_stage_a_yield_bias_figure(
                input_name=input_name,
                pool_df=pool_df,
                sampling=sampling,
                style=style,
                fig=fig2,
                slot=outer[idx, 0],
            )
        path2 = out_path.parent / f"{out_path.stem}__yield_bias{out_path.suffix}"
        fig2.savefig(path2, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig2)
        paths.append(path2)

        diversity_heights = [
            max(4.0, 1.5 * _regulator_count(input_name, sampling) + 1.1)
            for input_name, _pool_df, sampling in pwm_inputs
        ]
        fig3 = plt.figure(figsize=(fig_width, float(sum(diversity_heights))), constrained_layout=False)
        outer = fig3.add_gridspec(
            nrows=len(pwm_inputs),
            ncols=1,
            height_ratios=diversity_heights,
            hspace=0.4,
        )
        for idx, (input_name, pool_df, sampling) in enumerate(pwm_inputs):
            _build_stage_a_diversity_figure(
                input_name=input_name,
                pool_df=pool_df,
                sampling=sampling,
                style=style,
                fig=fig3,
                slot=outer[idx, 0],
            )
        path3 = out_path.parent / f"{out_path.stem}__diversity{out_path.suffix}"
        fig3.savefig(path3, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig3)
        paths.append(path3)

    for input_name, pool_df in background_inputs:
        if "tfbs" in pool_df.columns:
            sequences = pool_df["tfbs"].astype(str).tolist()
        elif "sequence" in pool_df.columns:
            sequences = pool_df["sequence"].astype(str).tolist()
        else:
            raise ValueError(f"Background pool '{input_name}' is missing tfbs/sequence columns.")
        fig, _axes = _build_background_logo_figure(
            input_name=input_name,
            sequences=sequences,
            style=style,
        )
        fname = f"{out_path.stem}__{_safe_filename(input_name)}__background_logo{out_path.suffix}"
        path = out_path.parent / fname
        fig.savefig(path, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths
