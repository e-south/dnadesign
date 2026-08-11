"""Render a compact synthesis purchase-price comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .contracts import PricingSnapshot, SynthesisScenario
from .model import build_purchase_price_rows

_GENE_COLOR = "#28788F"
_JUNCTION_COLOR = "#C05B3D"
_GRID_COLOR = "#DDE3E8"
_AXIS_COLOR = "#9AA5AE"


@dataclass(frozen=True, slots=True)
class RenderedPurchasePriceComparison:
    svg_path: Path
    png_path: Path


def render_purchase_price_comparison(
    snapshot: PricingSnapshot,
    scenario: SynthesisScenario,
    *,
    output_stem: Path,
) -> RenderedPurchasePriceComparison:
    """Render one square log-scale figure from a validated snapshot and scenario."""

    rows = build_purchase_price_rows(snapshot, scenario)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output_stem.with_suffix(".svg")
    png_path = output_stem.with_suffix(".png")
    target_counts = [row.target_count for row in rows]
    gene_prices = [float(row.gene_fragments_usd) for row in rows]
    pool_prices = [float(row.oligo_pool_usd) for row in rows]
    with matplotlib.rc_context(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "svg.fonttype": "path",
            "svg.hashsalt": "junction-purchase-price-comparison-v1",
        }
    ):
        figure = Figure(figsize=(7.2, 7.2))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        figure.subplots_adjust(left=0.16, right=0.96, top=0.90, bottom=0.14)
        axis.set_box_aspect(1)
        axis.plot(target_counts, gene_prices, color=_GENE_COLOR, linewidth=2.7, label="Gene fragments")
        axis.plot(
            target_counts,
            pool_prices,
            color=_JUNCTION_COLOR,
            linewidth=2.7,
            drawstyle="steps-post",
            label="Junction oligo pool",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1, rows[-1].target_count)
        axis.set_ylim(bottom=min(gene_prices[0], pool_prices[0]) * 0.8)
        axis.set_title("Gene fragments and Junction oligo pools", loc="center", pad=16)
        axis.set_xlabel("Target sequences", labelpad=10)
        axis.set_ylabel("Purchase price (USD)", labelpad=10)
        axis.grid(which="major", color=_GRID_COLOR, linewidth=0.8)
        axis.grid(which="minor", color=_GRID_COLOR, linewidth=0.45, alpha=0.45)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", color=_AXIS_COLOR, labelcolor="#38424D", width=0.9, length=4)
        axis.spines["left"].set_color(_AXIS_COLOR)
        axis.spines["bottom"].set_color(_AXIS_COLOR)
        axis.yaxis.set_major_formatter(FuncFormatter(_currency_tick))
        axis.legend(frameon=False, loc="upper left")
        metadata = {
            "Title": "Gene fragments and Junction oligo pools",
            "Description": (
                f"Purchase-price comparison for {scenario.target_length_nt}-nt targets using "
                f"{scenario.oligos_per_target} oligos per Junction target. Snapshot dated "
                f"{snapshot.retrieved_on.isoformat()}."
            ),
            "Creator": "dnadesign junction",
            "Date": None,
        }
        figure.savefig(svg_path, format="svg", metadata=metadata)
        _normalize_svg_whitespace(svg_path)
        figure.savefig(png_path, format="png", metadata={"Software": "dnadesign junction"})
        figure.clear()
    return RenderedPurchasePriceComparison(svg_path=svg_path, png_path=png_path)


def _currency_tick(value: float, _position: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:g}M"
    if value >= 1_000:
        return f"${value / 1_000:g}k"
    return f"${value:g}"


def _normalize_svg_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


__all__ = ["RenderedPurchasePriceComparison", "render_purchase_price_comparison"]
