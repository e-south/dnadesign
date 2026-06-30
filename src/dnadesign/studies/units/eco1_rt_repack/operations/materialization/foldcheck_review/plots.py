"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/plots.py

SVG plot rendering for Eco1 fold-check review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors as mcolors
import pyarrow.parquet as pq
from matplotlib import rc_context

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.constants import (
    BIOHUB_ESMC_PROFILE_FILE_NAME,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_CLASS_COLORS = {
    "strong_fold_preserved": "#2f7d5b",
    "good_fold_preserved": "#6aa84f",
    "review_band": "#6c8ebf",
    "low_confidence": "#c9a227",
    "structural_outlier": "#b45f5f",
    "metric_missing": "#8a8a8a",
}
_CLASS_ORDER = (
    "strong_fold_preserved",
    "good_fold_preserved",
    "review_band",
    "low_confidence",
    "structural_outlier",
    "metric_missing",
)
_TITLE_SIZE = 16
_LABEL_SIZE = 13.5
_TICK_SIZE = 12
_LEGEND_SIZE = 12


def write_review_plot_rows(
    *,
    plot_root: Path,
    output_root: Path,
    ranking_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Write review SVGs and return manifest-ready plot rows."""

    plot_root.mkdir(parents=True, exist_ok=True)
    plots = [
        _write_review_class_counts(plot_root, ranking_rows),
        _write_fold_metric_scatter(plot_root, ranking_rows),
        _write_cryoem_metric_scatter(plot_root, ranking_rows),
    ]
    biohub_profile_path = output_root / BIOHUB_ESMC_PROFILE_FILE_NAME
    if biohub_profile_path.exists():
        plots.append(_write_biohub_profile_summary(plot_root, biohub_profile_path))
    return plots


def _write_review_class_counts(plot_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row.get("review_class") or "metric_missing") for row in rows)
    labels = [label for label in _CLASS_ORDER if counts.get(label)]
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    y_positions = list(range(len(labels)))
    ax.barh(y_positions, values, color=[_CLASS_COLORS[label] for label in labels])
    ax.set_yticks(y_positions, [_review_class_tick_label(label) for label in labels], fontsize=10.5)
    ax.invert_yaxis()
    ax.set_xlabel("Candidate count", fontsize=_LABEL_SIZE)
    ax.set_title("Fold-review bins are threshold summaries", fontsize=_TITLE_SIZE, pad=10)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=_TICK_SIZE)
    fig.subplots_adjust(left=0.43, right=0.98, top=0.9, bottom=0.12)
    path = plot_root / "review_class_counts.svg"
    alt = (
        f"Bar chart of Eco1 fold-review classes for {len(rows)} candidates. "
        + ", ".join(f"{_review_class_plain_label(label)}: {counts[label]}" for label in labels)
        + "."
    )
    _save_accessible_svg(
        fig,
        path,
        title="Fold-review bins are threshold summaries",
        description=alt,
    )
    return _plot_row(
        plot_id="review_class_counts",
        path=path,
        title="Fold-review bins are threshold summaries",
        alt_text=alt,
        description=(
            "Counts candidates in the fold-review bins used for structural inspection. "
            "Use the continuous RMSD and pLDDT plots for metric-level interpretation."
        ),
        interpretation_limit="Review labels are triage summaries, not candidate acceptance decisions.",
        data_sources=["foldcheck_review/foldcheck_candidate_ranking.parquet"],
    )


def _write_fold_metric_scatter(plot_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    valid_rows = [
        row
        for row in rows
        if row.get("wt_runtime_ca_rmsd") is not None
        and row.get("plddt") is not None
        and row.get("seq_recovery") is not None
    ]
    sequence_identity = [_float(row.get("seq_recovery")) * 100.0 for row in valid_rows]
    scatter = ax.scatter(
        [_float(row.get("wt_runtime_ca_rmsd")) for row in valid_rows],
        [_float(row.get("plddt")) for row in valid_rows],
        c=sequence_identity,
        cmap="viridis",
        norm=_color_norm(sequence_identity),
        s=44,
        alpha=0.88,
        edgecolors="#ffffff",
        linewidths=0.45,
    )
    ax.set_xlabel("WT-runtime C-alpha RMSD (A)", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Mean pLDDT", fontsize=_LABEL_SIZE)
    ax.set_title("ColabFold metrics show continuous review signals", fontsize=_TITLE_SIZE, pad=10)
    _style_scatter_axis(ax)
    colorbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", fraction=0.055, pad=0.13)
    colorbar.set_label("Sequence identity to Ec86 WT (%)", fontsize=_LEGEND_SIZE)
    colorbar.ax.tick_params(labelsize=_LEGEND_SIZE)
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    path = plot_root / "fold_metric_scatter.svg"
    alt = (
        "Scatter plot of WT-runtime C-alpha RMSD versus mean pLDDT for Eco1 "
        f"ProteinMPNN candidates. The plot contains {len(rows)} candidate points "
        "colored by sequence identity to the Ec86 WT reference."
    )
    _save_accessible_svg(fig, path, title="ColabFold metrics show continuous review signals", description=alt)
    return _plot_row(
        plot_id="fold_metric_scatter",
        path=path,
        title="ColabFold metrics show continuous review signals",
        alt_text=alt,
        description=(
            "Shows confidence and within-run structural drift as quantitative axes, "
            "with point color showing sequence identity to Ec86 WT."
        ),
        interpretation_limit="This is a structural-fidelity summary, not activity or processivity evidence.",
        data_sources=["foldcheck_review/foldcheck_candidate_ranking.parquet"],
    )


def _write_cryoem_metric_scatter(plot_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if str(row.get("cryoem_mapped_ca_rmsd_status")) == "available"]
    fig, ax = plt.subplots(figsize=(6.8, 7.2))
    for label in _CLASS_ORDER:
        selected = [row for row in available if str(row.get("review_class")) == label]
        if selected:
            _scatter_review_rows(
                ax,
                selected,
                x_key="wt_runtime_ca_rmsd",
                y_key="cryoem_mapped_ca_rmsd",
                label=label,
            )
    ax.set_xlabel("WT-runtime C-alpha RMSD (A)", fontsize=_LABEL_SIZE)
    ax.set_ylabel("cryoEM-reference mapped C-alpha RMSD (A)", fontsize=_LABEL_SIZE)
    ax.set_title("WT-model similarity and cryoEM similarity are checked separately", fontsize=_TITLE_SIZE, pad=10)
    _style_scatter_axis(ax)
    if not available:
        ax.text(0.5, 0.5, "CryoEM-reference RMSD unavailable", transform=ax.transAxes, ha="center", va="center")
    _add_legend_below(fig, ax, ncol=3)
    fig.tight_layout(rect=(0, 0.15, 1, 0.98))
    path = plot_root / "cryoem_vs_runtime_rmsd.svg"
    alt = (
        "Scatter plot comparing each candidate's RMSD to the WT ColabFold runtime model "
        "against its mapped-residue RMSD to the ec86kit/7V9U cryoEM-backed reference. "
        f"{len(available)} candidates have available cryoEM-reference RMSD."
    )
    _save_accessible_svg(
        fig,
        path,
        title="WT-model similarity and cryoEM similarity are checked separately",
        description=alt,
    )
    return _plot_row(
        plot_id="cryoem_vs_runtime_rmsd",
        path=path,
        title="Runtime RMSD and cryoEM-reference RMSD are separate checks",
        alt_text=alt,
        description=(
            "Makes the two RMSD reference frames visible so within-run ColabFold similarity "
            "is not confused with direct comparison to the cryoEM-backed scaffold."
        ),
        interpretation_limit="High or low RMSD here is a review signal; final selection still requires feasibility.",
        data_sources=["foldcheck_review/foldcheck_candidate_ranking.parquet"],
    )


def _write_biohub_profile_summary(plot_root: Path, profile_path: Path) -> dict[str, Any]:
    rows = pq.read_table(
        profile_path,
        columns=["candidate_id", "status", "protein_feature_count", "residue_feature_count", "encoded_sae_bytes"],
    ).to_pylist()
    accepted = [row for row in rows if str(row.get("status")) == "accepted"]
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(
        [_float(row.get("protein_feature_count")) for row in accepted],
        [_float(row.get("encoded_sae_bytes")) / 1024.0 for row in accepted],
        s=28,
        alpha=0.7,
        color="#5b7fa6",
        edgecolors="none",
    )
    ax.set_xlabel("Protein-level nonzero SAE features", fontsize=_LABEL_SIZE)
    ax.set_ylabel("Encoded SAE payload size (KiB)", fontsize=_LABEL_SIZE)
    title = "Biohub ESMC coverage is available for accepted query rows"
    ax.set_title(title, fontsize=_TITLE_SIZE, pad=10)
    _style_scatter_axis(ax)
    fig.tight_layout()
    path = plot_root / "biohub_esmc_sae_coverage.svg"
    residue_counts = sorted({int(row.get("residue_feature_count") or 0) for row in accepted})
    alt = (
        f"Scatter plot summarizing Biohub ESMC SAE output coverage for {len(accepted)} accepted query rows. "
        f"Per-residue activation counts observed: {residue_counts}."
    )
    _save_accessible_svg(
        fig,
        path,
        title=title,
        description=alt,
    )
    return _plot_row(
        plot_id="biohub_esmc_sae_coverage",
        path=path,
        title=title,
        alt_text=alt,
        description="Confirms query-time SAE coverage without loading the full sparse residue table into the UI.",
        interpretation_limit="SAE features are semantic annotations and are not activity measurements.",
        data_sources=["biohub_esmc_sae_profile.parquet"],
    )


def _scatter_review_rows(ax: Any, rows: list[dict[str, Any]], *, x_key: str, y_key: str, label: str) -> None:
    ax.scatter(
        [_float(row.get(x_key)) for row in rows],
        [_float(row.get(y_key)) for row in rows],
        s=36,
        alpha=0.82,
        label=_review_class_plain_label(label),
        color=_CLASS_COLORS[label],
        edgecolors="#ffffff",
        linewidths=0.35,
    )


def _plot_row(
    *,
    plot_id: str,
    path: Path,
    title: str,
    alt_text: str,
    description: str,
    interpretation_limit: str,
    data_sources: list[str],
) -> dict[str, Any]:
    return {
        "plot_id": plot_id,
        "status": "rendered",
        "path": str(path),
        "title": title,
        "alt_text": alt_text,
        "description": description,
        "interpretation_limit": interpretation_limit,
        "data_sources": data_sources,
        "skip_reason": "",
    }


def _save_accessible_svg(fig: Any, path: Path, *, title: str, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rc_context({"svg.fonttype": "none"}):
        fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    _inject_svg_accessibility(path, title=title, description=description)


def _inject_svg_accessibility(path: Path, *, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    title_id = f"{path.stem}-title"
    desc_id = f"{path.stem}-desc"
    if "<title" not in text and "<svg " in text:
        text = text.replace("<svg ", f'<svg role="img" aria-labelledby="{title_id} {desc_id}" ', 1)
        svg_start = text.find("<svg ")
        svg_end = text.find(">", svg_start)
        if svg_start != -1 and svg_end != -1:
            accessible = (
                f'\n<title id="{escape(title_id)}">{escape(title)}</title>'
                f'\n<desc id="{escape(desc_id)}">{escape(description)}</desc>'
            )
            text = text[: svg_end + 1] + accessible + text[svg_end + 1 :]
    path.write_text(text, encoding="utf-8")


def _human_label(value: str) -> str:
    return _review_class_plain_label(value)


def _review_class_tick_label(value: str) -> str:
    labels = {
        "strong_fold_preserved": "RMSD <= 1.25 A\nand pLDDT >= 91.5",
        "good_fold_preserved": "RMSD <= 2.0 A\nand pLDDT >= 90",
        "review_band": "Intermediate review band",
        "low_confidence": "pLDDT < 90",
        "structural_outlier": "RMSD > 5.0 A",
        "metric_missing": "Metric missing",
    }
    return labels.get(value, value.replace("_", " "))


def _review_class_plain_label(value: str) -> str:
    labels = {
        "strong_fold_preserved": "WT-runtime CA RMSD <= 1.25 A and mean pLDDT >= 91.5",
        "good_fold_preserved": "WT-runtime CA RMSD <= 2.0 A and mean pLDDT >= 90",
        "review_band": "Intermediate fold-review band",
        "low_confidence": "Mean pLDDT < 90",
        "structural_outlier": "WT-runtime CA RMSD > 5.0 A",
        "metric_missing": "Metric missing",
    }
    return labels.get(value, value.replace("_", " "))


def _float(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def _style_scatter_axis(ax: Any) -> None:
    ax.set_axisbelow(True)
    ax.grid(color="#d0d7de", alpha=0.42, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=_TICK_SIZE)
    ax.set_box_aspect(1)


def _add_legend_below(fig: Any, ax: Any, *, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=_LEGEND_SIZE,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=ncol,
        )


def _color_norm(values: list[float]) -> mcolors.Normalize | None:
    finite = [value for value in values if value == value]
    if not finite:
        return None
    minimum = min(finite)
    maximum = max(finite)
    if minimum == maximum:
        return mcolors.Normalize(vmin=minimum - 1.0, vmax=maximum + 1.0)
    return mcolors.Normalize(vmin=minimum, vmax=maximum)
