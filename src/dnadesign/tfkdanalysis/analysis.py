"""
--------------------------------------------------------------------------------
<dnadesign project>

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white.
    amount=0 returns the original color, amount=1 returns white.
    """
    try:
        c = mc.to_rgb(color)
    except ValueError:
        c = (0.5, 0.5, 0.5)
    white = np.array([1, 1, 1])
    c = np.array(c)
    return tuple((1 - amount) * c + amount * white)


def generate_volcano_plot(
    df: pd.DataFrame, fc_column: str, logp_column: str, config: dict, output_path: Path
) -> pd.DataFrame:
    sns.set(style="ticks")
    try:
        fc_threshold = float(config.get("threshold", 1.5))
        pval_threshold = float(config.get("pval_threshold", 2))
    except ValueError as e:
        raise ValueError("Threshold values in config must be numeric.") from e

    df = df.copy()
    df["significant"] = (df[fc_column].abs() >= fc_threshold) & (df[logp_column] >= pval_threshold)
    if "significant" not in df.columns:
        raise KeyError("The 'significant' column was not created in the DataFrame.")

    # Handle point sizing.
    size_column = config.get("volcano_point_size")
    if isinstance(size_column, str) and size_column.lower() == "none":
        size_column = None
    size_scale = 20
    if size_column is not None:
        if size_column not in df.columns:
            raise ValueError(f"Expected size column '{size_column}' not found in the dataset for volcano plot.")
        sizes = np.log1p(df[size_column]) * size_scale
    else:
        sizes = None

    # Identify user-defined regulators.
    regulators = config.get("regulators", [])
    if regulators:
        reg_mask = df["tf_gene"].str.lower().str.strip().isin([r.lower().strip() for r in regulators])
    else:
        reg_mask = pd.Series(False, index=df.index)

    plt.figure(figsize=(8, 6))

    # Compute x-axis range for annotation offset.
    x_min, x_max = df[fc_column].min(), df[fc_column].max()
    x_range = x_max - x_min
    x_offset = 0.02 * x_range  # slight offset for annotations

    # Define groups.
    groupA = df[~df["significant"] & ~reg_mask]  # Not significant, not user-defined regulator.
    groupB = df[df["significant"] & ~reg_mask]  # Significant, not user-defined regulator.
    groupC = df[~df["significant"] & reg_mask]  # Not significant, user-defined regulator.
    groupD = df[df["significant"] & reg_mask]  # Significant, user-defined regulator.

    # Plot groups A and B without edges.
    if sizes is not None:
        plt.scatter(
            groupA[fc_column],
            groupA[logp_column],
            s=sizes[groupA.index],
            color="lightgray",
            alpha=0.35,
            marker="o",
            edgecolor="none",
        )
        plt.scatter(
            groupB[fc_column],
            groupB[logp_column],
            s=sizes[groupB.index],
            color="gray",
            alpha=0.35,
            marker="o",
            edgecolor="none",
        )
    else:
        plt.scatter(groupA[fc_column], groupA[logp_column], color="lightgray", alpha=0.35, marker="o", edgecolor="none")
        plt.scatter(groupB[fc_column], groupB[logp_column], color="gray", alpha=0.35, marker="o", edgecolor="none")

    # Marker shapes to cycle through.
    markers = ["o", "s", "^"]

    # Plot user-defined regulators.
    if regulators:
        palette = sns.color_palette("colorblind", n_colors=len(regulators))
        for i, reg in enumerate(regulators):
            marker_style = markers[i % len(markers)]
            mask = df["tf_gene"].str.lower().str.strip() == reg.lower().strip()
            reg_data = df[mask]
            if reg_data.empty:
                logging.warning(f"Regulator '{reg}' not found in the dataset for volcano plot.")
                continue
            reg_non_sig = reg_data[~reg_data["significant"]]
            reg_sig = reg_data[reg_data["significant"]]
            light_color = sns.desaturate(palette[i], 0.5)
            if sizes is not None:
                if not reg_non_sig.empty and reg_sig.empty:
                    plt.scatter(
                        reg_non_sig[fc_column],
                        reg_non_sig[logp_column],
                        s=sizes[reg_non_sig.index],
                        color=light_color,
                        alpha=0.65,
                        marker=marker_style,
                        label=reg,
                        edgecolor="none",
                    )
                else:
                    plt.scatter(
                        reg_non_sig[fc_column],
                        reg_non_sig[logp_column],
                        s=sizes[reg_non_sig.index],
                        color=light_color,
                        alpha=0.65,
                        marker=marker_style,
                        edgecolor="none",
                    )
                    plt.scatter(
                        reg_sig[fc_column],
                        reg_sig[logp_column],
                        s=sizes[reg_sig.index],
                        color=palette[i],
                        alpha=0.65,
                        marker=marker_style,
                        label=reg,
                        edgecolor="none",
                    )
            else:
                if not reg_non_sig.empty and reg_sig.empty:
                    plt.scatter(
                        reg_non_sig[fc_column],
                        reg_non_sig[logp_column],
                        color=light_color,
                        alpha=0.65,
                        marker=marker_style,
                        label=reg,
                        edgecolor="none",
                    )
                else:
                    plt.scatter(
                        reg_non_sig[fc_column],
                        reg_non_sig[logp_column],
                        color=light_color,
                        alpha=0.65,
                        marker=marker_style,
                        edgecolor="none",
                    )
                    plt.scatter(
                        reg_sig[fc_column],
                        reg_sig[logp_column],
                        color=palette[i],
                        alpha=0.65,
                        marker=marker_style,
                        label=reg,
                        edgecolor="none",
                    )
            # Annotate only if flag is true.
            if config.get("annotate_operon", True):
                for _, row in reg_sig.iterrows():
                    plt.text(
                        row[fc_column] + x_offset,
                        row[logp_column],
                        str(row.get("operon", "")),
                        fontsize=8,
                        ha="left",
                        va="bottom",
                    )

    # Add dashed threshold lines.
    plt.axvline(x=-fc_threshold, linestyle="--", color="gray", linewidth=1)
    plt.axvline(x=fc_threshold, linestyle="--", color="gray", linewidth=1)
    plt.axhline(y=pval_threshold, linestyle="--", color="gray", linewidth=1)

    pad = 0.1 * x_range
    plt.xlim(x_min - pad, x_max + pad)

    media = config.get("media", "glu")
    plt.title("Promoter activity changes by TF Knockdown (M9-glucose)", fontsize=12)
    plt.xlabel("log2(fold change)")
    plt.ylabel("-log10(p-value)")

    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()

    return df


def export_regulator_csvs(
    df: pd.DataFrame, regulators: list, fc_column: str, threshold: float, output_dir: Path
) -> None:
    """
    For each user-defined regulator, export two CSV files containing rows (from the volcano data)
    for that regulator: one for up-regulated (fc >= threshold) and one for down-regulated (fc <= -threshold).
    Before saving, drop any columns that have missing values in the filtered data.
    """
    for reg in regulators:
        mask = df["tf_gene"].str.lower().str.strip() == reg.lower().strip()
        reg_df = df[mask]
        if reg_df.empty:
            logging.warning(f"No data for regulator '{reg}' to export CSVs.")
            continue
        up_df = reg_df[reg_df[fc_column] >= threshold].dropna(axis=1)
        down_df = reg_df[reg_df[fc_column] <= -threshold].dropna(axis=1)
        up_path = output_dir / f"{reg}_upregulated.csv"
        down_path = output_dir / f"{reg}_downregulated.csv"
        up_df.to_csv(up_path, index=False)
        down_df.to_csv(down_path, index=False)
        logging.info(f"Exported CSVs for {reg}: {up_path}, {down_path}")


def compute_histogram_xlim(df: pd.DataFrame, fc_column: str, regulators: list) -> tuple:
    mask = df["tf_gene"].str.lower().str.strip().isin([r.lower().strip() for r in regulators])
    if not mask.any():
        logging.warning("No matching regulators found for computing x-limits; using default limits.")
        x_min, x_max = df[fc_column].min(), df[fc_column].max()
    else:
        x_min, x_max = df.loc[mask, fc_column].min(), df.loc[mask, fc_column].max()
    pad = (x_max - x_min) * 0.1
    return (x_min - pad, x_max + pad)


def generate_regulator_scatter_plot(
    df: pd.DataFrame, regulators: list, fc_column: str, global_xlim: tuple, config: dict, output_path: Path
) -> None:
    """
    Generate a scatter plot where each point represents a regulated promoter.
    The x-axis is the fold change, and the y-axis is the regulator name (with jitter).
    Points are colored based on the regulator, with full color for points surpassing the threshold
    and a lighter version (mixed with white) for those that do not. Significant points are annotated with the operon.
    Vertical dashed lines indicate the positive and negative fold-change thresholds (and x=0).
    The plot also annotates the number of unique regulated promoters per regulator.
    """
    sns.set(style="ticks")
    plt.figure(figsize=(8, 6))

    try:
        fc_threshold = float(config.get("threshold", 1.5))
    except ValueError as e:
        raise ValueError("Fold change threshold in config must be numeric.") from e

    y_positions = {reg: i for i, reg in enumerate(regulators)}
    palette = sns.color_palette("colorblind", n_colors=len(regulators))
    promoter_counts = {}
    scatter_data = []

    for reg in regulators:
        reg_mask = df["tf_gene"].str.lower().str.strip() == reg.lower().strip()
        reg_data = df[reg_mask]
        if reg_data.empty:
            logging.warning(f"No data found for regulator '{reg}' in scatter plot.")
            continue
        count = reg_data["operon"].nunique() if "operon" in reg_data.columns else len(reg_data)
        promoter_counts[reg] = count
        for _, row in reg_data.iterrows():
            jitter = np.random.uniform(-0.2, 0.2)
            scatter_data.append(
                {
                    "regulator": reg,
                    "fc": row[fc_column],
                    "y": y_positions[reg] + jitter,
                    "significant": abs(row[fc_column]) >= fc_threshold,
                    "operon": row.get("operon", ""),
                }
            )

    scatter_df = pd.DataFrame(scatter_data)

    for i, reg in enumerate(regulators):
        reg_df = scatter_df[scatter_df["regulator"] == reg]
        if reg_df.empty:
            continue
        full_color = palette[i]
        # For non-significant, mix in white further to reduce emphasis.
        light_color = lighten_color(full_color, amount=0.7)
        sig_df = reg_df[reg_df["significant"]]
        nonsig_df = reg_df[~reg_df["significant"]]
        if not nonsig_df.empty and sig_df.empty:
            plt.scatter(nonsig_df["fc"], nonsig_df["y"], color=light_color, alpha=0.8, marker="o", label=reg)
        else:
            plt.scatter(nonsig_df["fc"], nonsig_df["y"], color=light_color, alpha=0.8, marker="o")
            plt.scatter(sig_df["fc"], sig_df["y"], color=full_color, alpha=0.9, marker="o", label=reg)
            if config.get("annotate_operon", True):
                for _, point in sig_df.iterrows():
                    if point["operon"]:
                        plt.text(
                            point["fc"] + 0.02, point["y"], str(point["operon"]), fontsize=8, ha="left", va="bottom"
                        )

    plt.axvline(x=fc_threshold, color="gray", linestyle="--", linewidth=1)
    plt.axvline(x=-fc_threshold, color="gray", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="gray", linestyle="--", linewidth=1)

    plt.xlim(global_xlim)
    plt.xlabel("log2(fold change)")
    plt.yticks(list(y_positions.values()), list(y_positions.keys()))
    plt.ylabel("Transcription Factor")

    ymin = min(y_positions.values()) - 0.5
    ymax = max(y_positions.values()) + 0.5
    plt.ylim(ymin, ymax)

    for reg, pos in y_positions.items():
        count = promoter_counts.get(reg, 0)
        plt.text(global_xlim[1], pos, f" ({count})", va="center", fontsize=9, color="black")

    media = config.get("media", "glu")
    plt.title("Promoter activity changes by TF Knockdown (M9-glucose)", fontsize=12)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
    logging.info(f"Regulator scatter plot saved to {output_path}")
