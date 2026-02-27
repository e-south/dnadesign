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
from .plot_common import _apply_style, _safe_filename, _save_figure, _style
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


def _background_logo_title(input_name: str) -> str:
    normalized = " ".join(str(input_name or "").strip().split())
    if not normalized:
        return "Background sequence logo"
    if normalized.lower() == "background":
        return "Background sequence logo"
    return f"{normalized} sequence logo"


def _build_background_logo_figure(
    *,
    input_name: str,
    sequences: list[str],
    style: dict,
) -> tuple[plt.Figure, list[plt.Axes]]:
    def _coerce_float(value: object, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    length_groups: dict[int, list[str]] = {}
    for seq in sequences:
        length_groups.setdefault(len(seq), []).append(seq)
    lengths = sorted(length_groups)
    if not lengths:
        raise ValueError("Background logo requires at least one sequence length.")
    n_lengths = len(lengths)
    nrows = n_lengths
    base_font_size = _coerce_float(style.get("font_size", 13.0), 13.0)
    logo_style = dict(style)
    logo_style["tick_size"] = max(_coerce_float(logo_style.get("tick_size", base_font_size), base_font_size), 14.0)
    logo_style["label_size"] = max(_coerce_float(logo_style.get("label_size", base_font_size), base_font_size), 14.5)
    logo_style["title_size"] = max(
        _coerce_float(logo_style.get("title_size", base_font_size * 1.1), base_font_size * 1.1),
        16.0,
    )

    subplot_height = 1.75
    width = max(6.2, min(10.6, 0.12 * float(max(lengths)) + 4.4))
    height = max(2.4, subplot_height * float(nrows) + 0.35)
    fig, axes = plt.subplots(nrows, 1, figsize=(width, height), sharex=True, sharey=True)
    axes_list = list(np.atleast_1d(axes).ravel())
    used_axes: list[plt.Axes] = []
    for ax, length in zip(axes_list, lengths):
        seqs = length_groups[length]
        pwm = _background_pwm_from_sequences(seqs)
        df = pd.DataFrame(pwm, columns=["A", "C", "G", "T"], dtype=float)
        logomaker.Logo(df, ax=ax, shade_below=0.5)
        ax.set_title(f"L={length} (n={len(seqs)})")
        ax.set_ylabel("Probability")
        _apply_style(ax, logo_style)
        ax.grid(False)
        used_axes.append(ax)
    if used_axes:
        for ax in used_axes[:-1]:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        used_axes[-1].set_xlabel("Position")
    for ax in axes_list[len(lengths) :]:
        ax.set_axis_off()
    suptitle_size = max(_coerce_float(logo_style.get("title_size", 16.0), 16.0) + 2.0, 18.0)
    if int(nrows) <= 1:
        top_margin = 0.82
        bottom_margin = 0.19
        title_y = 0.905
    else:
        top_margin = 0.88
        bottom_margin = 0.11
        title_y = 0.955
    fig.tight_layout(rect=(0.0, bottom_margin, 1.0, top_margin), h_pad=0.22)
    fig.suptitle(_background_logo_title(input_name), fontsize=suptitle_size, y=title_y)
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
        style["figsize"] = (15.8, 2.6)
    base_dir = out_path.parent / "stage_a"
    base_dir.mkdir(parents=True, exist_ok=True)
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
            if entry.input_type == "pwm_artifact":
                raise ValueError(f"Stage-A sampling metadata missing for input '{input_name}'.")
            continue
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
            max(2.45, 0.9 * _regulator_count(input_name, sampling) + 0.6)
            for input_name, _pool_df, sampling in pwm_inputs
        ]
        fig = plt.figure(figsize=(fig_width, float(sum(strata_heights))), constrained_layout=False)
        outer = fig.add_gridspec(
            nrows=len(pwm_inputs),
            ncols=1,
            height_ratios=strata_heights,
            hspace=0.34,
        )
        for idx, (input_name, pool_df, sampling) in enumerate(pwm_inputs):
            _build_stage_a_strata_overview_figure(
                input_name=input_name,
                pool_df=pool_df,
                sampling=sampling,
                style=style,
                fig=fig,
                slot=outer[idx, 0],
                show_column_titles=(idx == 0),
            )
        path = base_dir / f"pool_tiers{out_path.suffix}"
        _save_figure(fig, path, style=style)
        plt.close(fig)
        paths.append(path)

        base_height = float(style.get("figsize", (11, 4.2))[1])
        yield_heights = [
            max(2.9, base_height, 0.95 * _regulator_count(input_name, sampling) + 0.5)
            for input_name, _pool_df, sampling in pwm_inputs
        ]
        fig2 = plt.figure(figsize=(fig_width, float(sum(yield_heights))), constrained_layout=False)
        outer = fig2.add_gridspec(
            nrows=len(pwm_inputs),
            ncols=1,
            height_ratios=yield_heights,
            hspace=0.36,
        )
        for idx, (input_name, pool_df, sampling) in enumerate(pwm_inputs):
            _build_stage_a_yield_bias_figure(
                input_name=input_name,
                pool_df=pool_df,
                sampling=sampling,
                style=style,
                fig=fig2,
                slot=outer[idx, 0],
                show_column_titles=(idx == 0),
            )
        path2 = base_dir / f"yield_bias{out_path.suffix}"
        _save_figure(fig2, path2, style=style)
        plt.close(fig2)
        paths.append(path2)

        diversity_heights = [
            max(2.65, 0.9 * _regulator_count(input_name, sampling) + 0.65)
            for input_name, _pool_df, sampling in pwm_inputs
        ]
        fig3 = plt.figure(figsize=(fig_width, float(sum(diversity_heights))), constrained_layout=False)
        outer = fig3.add_gridspec(
            nrows=len(pwm_inputs),
            ncols=1,
            height_ratios=diversity_heights,
            hspace=0.34,
        )
        for idx, (input_name, pool_df, sampling) in enumerate(pwm_inputs):
            _build_stage_a_diversity_figure(
                input_name=input_name,
                pool_df=pool_df,
                sampling=sampling,
                style=style,
                fig=fig3,
                slot=outer[idx, 0],
                show_column_titles=(idx == 0),
            )
        path3 = base_dir / f"diversity{out_path.suffix}"
        _save_figure(fig3, path3, style=style)
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
        input_segment = _safe_filename(input_name)
        if input_segment.lower() == "background":
            fname = f"background_logo{out_path.suffix}"
        else:
            fname = f"{input_segment}__background_logo{out_path.suffix}"
        path = base_dir / fname
        logo_save_style = dict(style)
        try:
            logo_save_style["save_dpi"] = min(float(logo_save_style.get("save_dpi", 300.0)), 220.0)
        except Exception:
            logo_save_style["save_dpi"] = 220.0
        _save_figure(fig, path, style=logo_save_style)
        plt.close(fig)
        paths.append(path)
    if not paths:
        fig, ax = plt.subplots(figsize=(8.2, 2.8), constrained_layout=False)
        ax.axis("off")
        ax.text(
            0.5,
            0.65,
            "No Stage-A diagnostic panels were generated for this run.",
            ha="center",
            va="center",
            fontsize=float(style.get("label_size", style.get("font_size", 13.0) * 0.95)),
        )
        ax.text(
            0.5,
            0.35,
            "Common causes: binding-site-only inputs, no PWM pool summaries, or no background pools.",
            ha="center",
            va="center",
            fontsize=float(style.get("tick_size", style.get("font_size", 13.0) * 0.7)),
            alpha=0.9,
        )
        fallback_path = base_dir / f"no_stage_a_panels{out_path.suffix}"
        _save_figure(fig, fallback_path, style=style)
        plt.close(fig)
        paths.append(fallback_path)
    return paths
