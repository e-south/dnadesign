"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/plots/edge_bundling.py

Hierarchical edge-bundling style plot for AA mutation co-occurrence
among selected multisite variants.

Nodes: AA mutation tokens (e.g., G16F, L17I, N21H, …)
Edges: undirected, token_a -- token_b if they co-occur in ≥1 selected variant
       weight_count = number of variants where {a,b} ⊂ combo
       avg_k       = mean mut_count across those variants

Layout:
  • circular node ring, ordered by position then alt AA (clockwise)
  • simple spline-like chords; grouped by position to suggest hierarchy

Color:
  • configurable via `color_by` ('edge_avg_k' or 'node_avg_k') and `edge_cmap`

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Slightly larger, consistent font sizes across this plot
_TITLE_FONTSIZE = 12
_NODE_LABEL_FONTSIZE = 10
_LEGEND_TITLE_FONTSIZE = 10
_LEGEND_TEXT_FONTSIZE = 10
_COLORBAR_LABEL_FONTSIZE = 10
_COLORBAR_TICK_FONTSIZE = 10


@dataclass(frozen=True)
class EdgeRecord:
    token_a: str
    token_b: str
    weight_count: int
    avg_k: float


def _ensure_path(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _parse_combo_tokens(combo: str) -> List[str]:
    s = (combo or "").strip()
    if not s:
        return []
    return [tok.strip() for tok in s.split("|") if tok.strip()]


def _position_key(token: str) -> Tuple[int, str, str]:
    """
    Sort key: (position, altAA, full token).
    Token is like 'G16F'. Malformed tokens are placed at the end deterministically.
    """
    import re

    m = re.match(r"^[A-Z\*](\d+)([A-Z\*])$", token)
    if not m:
        # place malformed tokens at the end deterministically
        return (10_000_000, token, token)
    pos = int(m.group(1))
    alt = m.group(2)
    return (pos, alt, token)


def build_edge_table(
    aa_combo_strs: Sequence[str],
    k_values: Sequence[int],
    *,
    min_cooccur_count: int = 1,
) -> pd.DataFrame:
    """
    Build a co-occurrence edge table from AA combo strings and mutation counts k.
    Returns columns: token_a, token_b, weight_count, avg_k.
    """
    if len(aa_combo_strs) != len(k_values):
        raise ValueError("edge_bundling.build_edge_table: aa_combo_strs and k_values must have same length")

    co_counts: Counter[Tuple[str, str]] = Counter()
    sum_k: Dict[Tuple[str, str], float] = defaultdict(float)

    for combo, k in zip(aa_combo_strs, k_values):
        tokens = sorted(set(_parse_combo_tokens(combo)))
        if len(tokens) < 2:
            continue
        k_val = float(k)
        # all unordered pairs
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                a, b = tokens[i], tokens[j]
                key = (a, b) if a <= b else (b, a)
                co_counts[key] += 1
                sum_k[key] += k_val

    rows: List[EdgeRecord] = []
    for (a, b), cnt in co_counts.items():
        if cnt < int(min_cooccur_count):
            continue
        avg_k = sum_k[(a, b)] / float(cnt)
        rows.append(EdgeRecord(token_a=a, token_b=b, weight_count=int(cnt), avg_k=avg_k))

    if not rows:
        return pd.DataFrame(columns=["token_a", "token_b", "weight_count", "avg_k"], dtype=float)

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df[["token_a", "token_b", "weight_count", "avg_k"]]


def _circular_node_positions(tokens: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """
    Place tokens on a unit circle using their sorted order.

    Nodes are ordered clockwise by their sorted token order so that increasing
    residue positions appear in clockwise order around the circle.
    """
    n = len(tokens)
    if n == 0:
        return {}
    angles = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    coords: Dict[str, Tuple[float, float]] = {}
    for tok, ang in zip(tokens, angles):
        coords[tok] = (float(np.cos(ang)), float(-np.sin(ang)))
    return coords


def _edge_widths(
    weight: np.ndarray,
    *,
    scale: str = "sqrt",
    min_width: float = 0.5,
    max_width: float = 8.0,
) -> np.ndarray:
    w = np.asarray(weight, dtype=float)
    if w.size == 0:
        return w
    if scale == "sqrt":
        x = np.sqrt(w)
    else:
        x = w
    x_min, x_max = float(x.min()), float(x.max())
    if x_max <= x_min:
        return np.full_like(x, (min_width + max_width) / 2.0)
    x_norm = (x - x_min) / (x_max - x_min)
    return min_width + x_norm * (max_width - min_width)


def _edge_colors_from_avg_k(
    avg_k: np.ndarray,
    cmap_name: str = "viridis",
) -> List[Tuple[float, float, float, float]]:
    cmap = plt.get_cmap(cmap_name)
    k = np.asarray(avg_k, dtype=float)
    if k.size == 0:
        return []
    k_min, k_max = float(k.min()), float(k.max())
    if k_max <= k_min:
        return [cmap(0.5) for _ in k]
    k_norm = (k - k_min) / (k_max - k_min)
    return [cmap(v) for v in k_norm]


def plot_edge_bundle(
    edges_df: pd.DataFrame,
    *,
    out_png: str | Path,
    out_pdf: str | Path | None = None,
    figsize_in: float = 10.0,
    dpi: int = 200,
    width_scale: str = "sqrt",
    node_counts: Dict[str, int] | None = None,
    edge_cmap: str = "viridis",
    color_by: str = "edge_avg_k",
    node_avg_k: Dict[str, float] | None = None,
    title: str | None = None,
) -> None:
    """
    Render a circular edge-bundling style chord plot from an EDGE_BUNDLE_TABLE
    (token_a, token_b, weight_count, avg_k).

    Visual encodings (configurable via `color_by`):
      • Edge width  ~ co-occurrence count
      • Colorbar    ~ continuous metric (edge-average or per-node avg k)
      • Node size   ~ token frequency across selected variants (if node_counts provided)
    """
    if edges_df.empty:
        return

    out_png = _ensure_path(out_png)
    if out_pdf is not None:
        out_pdf = _ensure_path(out_pdf)

    # Node set, ordered by (position, alt AA, token)
    tokens = sorted(
        set(edges_df["token_a"]).union(set(edges_df["token_b"])),
        key=_position_key,
    )
    pos = _circular_node_positions(tokens)

    # Edge widths from co-occurrence counts
    weights = edges_df["weight_count"].to_numpy(dtype=float)
    widths = _edge_widths(weights, scale=width_scale)

    # Node sizes: proportional to how often each mutation token appears across
    # the selected variants (if node_counts provided).
    if node_counts:
        counts = np.array([node_counts.get(tok, 0) for tok in tokens], dtype=float)
    else:
        # uniform size when per-token frequencies are unavailable
        counts = np.ones(len(tokens), dtype=float)

    c_min, c_max = float(counts.min()), float(counts.max())
    min_size, max_size = 100.0, 500.0
    if c_max > c_min:
        sizes = min_size + (max_size - min_size) * (counts - c_min) / (c_max - c_min)
    else:
        sizes = np.full_like(counts, (min_size + max_size) / 2.0)

    # Color mapping (edges and/or nodes) and associated colorbar configuration.
    cmap = plt.get_cmap(edge_cmap)
    color_by_norm = (color_by or "edge_avg_k").lower()

    node_rgba: List[Tuple[float, float, float, float]]
    edge_rgba: List[Tuple[float, float, float, float]] | None = None
    colorbar_norm: Normalize | None = None
    colorbar_label: str = ""

    if color_by_norm == "edge_avg_k":
        # Original behavior: edge hue encodes avg mut_count k per edge.
        avg_k = edges_df["avg_k"].to_numpy(dtype=float)
        if avg_k.size:
            finite = avg_k[np.isfinite(avg_k)]
            if finite.size == 0:
                raise ValueError("plot_edge_bundle: 'avg_k' column has no finite values")
            vmin, vmax = float(finite.min()), float(finite.max())
            if vmin == vmax:
                vmin, vmax = vmin - 0.5, vmax + 0.5
            colorbar_norm = Normalize(vmin=vmin, vmax=vmax)
            edge_rgba = [cmap(colorbar_norm(v)) for v in avg_k]
        else:
            edge_rgba = []
            colorbar_norm = None
        node_rgba = [cmap(0.5) for _ in tokens]
        colorbar_label = "Edge hue: avg mut_count k"
    elif color_by_norm == "node_avg_k":
        # New behavior: node hue encodes avg mut_count k for that mutation token.
        if node_avg_k is None:
            raise ValueError("plot_edge_bundle: color_by='node_avg_k' requires node_avg_k mapping")
        node_vals = np.array([float(node_avg_k.get(tok, np.nan)) for tok in tokens], dtype=float)
        finite = node_vals[np.isfinite(node_vals)]
        if finite.size == 0:
            raise ValueError("plot_edge_bundle: node_avg_k mapping produced no finite values")
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        colorbar_norm = Normalize(vmin=vmin, vmax=vmax)
        node_rgba = [cmap(colorbar_norm(v)) if np.isfinite(v) else cmap(0.5) for v in node_vals]
        edge_rgba = None
        colorbar_label = "Node hue: avg mut_count k"
    else:
        raise ValueError(f"plot_edge_bundle: unsupported color_by={color_by!r} (expected 'edge_avg_k' or 'node_avg_k')")

    # ------------------------------------------------------------------
    # Figure + layout: main chord axis on the left, legends on the right
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=dpi)
    # Main chord-plot axis plus a legend column to the right
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[4.0, 1.5],
        wspace=0.25,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")

    # Sub-grid in the right column for the three legends
    legend_gs = gs[0, 1].subgridspec(
        nrows=3,
        ncols=1,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.1,
    )
    ax_edge_legend = fig.add_subplot(legend_gs[0, 0])
    ax_node_legend = fig.add_subplot(legend_gs[1, 0])
    ax_cbar = fig.add_subplot(legend_gs[2, 0])

    # Use a [0,1]x[0,1] coordinate system for manual legends
    for lax in (ax_edge_legend, ax_node_legend):
        lax.set_xlim(0.0, 1.0)
        lax.set_ylim(0.0, 1.0)
        lax.axis("off")

    if title:
        fig.suptitle(title, fontsize=_TITLE_FONTSIZE, y=0.96)

    # ------------------------------------------------------------------
    # 1. Draw edges (thin → thick) so larger weights sit higher in z-order
    # ------------------------------------------------------------------
    rows = list(edges_df.itertuples(index=False))
    order = np.argsort(widths)  # thin → thick

    # Normalize co-occurrence counts for alpha/z‑ordering (heavier edges = more emphasis)
    if weights.size:
        w_min_val = float(weights.min())
        w_max_val = float(weights.max())
        if w_max_val > w_min_val:
            weight_norm = (weights - w_min_val) / (w_max_val - w_min_val)
        else:
            weight_norm = np.zeros_like(weights, dtype=float)
    else:
        weight_norm = np.zeros_like(weights, dtype=float)

    for idx in order:
        row = rows[int(idx)]
        a, b = row.token_a, row.token_b
        xa, ya = pos[a]
        xb, yb = pos[b]
        mx, my = (xa + xb) / 2.0, (ya + yb) / 2.0
        # Stronger bundling: pull midpoint towards the origin
        ctrl = (0.5 * mx, 0.5 * my)

        w = float(widths[int(idx)])
        strength = float(weight_norm[int(idx)])  # 0 → weakest, 1 → strongest
        if color_by_norm == "edge_avg_k" and edge_rgba is not None:
            edge_color = edge_rgba[int(idx)]
            base_alpha, max_alpha = 0.40, 0.90
        else:
            edge_color = "0.6"
            base_alpha, max_alpha = 0.30, 0.70

        edge_alpha = base_alpha + (max_alpha - base_alpha) * strength

        path = matplotlib.path.Path(
            [(xa, ya), ctrl, (xb, yb)],
            [
                matplotlib.path.Path.MOVETO,
                matplotlib.path.Path.CURVE3,
                matplotlib.path.Path.CURVE3,
            ],
        )
        patch = matplotlib.patches.PathPatch(
            path,
            facecolor="none",
            edgecolor=edge_color,
            lw=w,
            alpha=edge_alpha,
            zorder=1.0 + w + 2.0 * strength,
        )
        ax.add_patch(patch)

    # ------------------------------------------------------------------
    # 2. Draw nodes on top of edges
    # ------------------------------------------------------------------
    for tok, s, c in zip(tokens, sizes, node_rgba):
        x, y = pos[tok]
        ax.scatter(
            [x],
            [y],
            s=float(s),
            c=[c],
            edgecolor="none",
            zorder=20,
        )

    # ------------------------------------------------------------------
    # 3. Residue labels at the very top of the z-stack
    # ------------------------------------------------------------------
    for tok in tokens:
        x, y = pos[tok]
        ax.text(
            1.20 * x,
            1.20 * y,
            tok,
            fontsize=_NODE_LABEL_FONTSIZE,
            ha="center",
            va="center",
            rotation=np.degrees(np.arctan2(y, x)),
            zorder=21,
        )

    # Fix limits and aspect (no extra margin below; legends live in separate axes).
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal", "box")

    # ------------------------------------------------------------------
    # 4. Edge-width legend: numeric co-occurrence counts (right column)
    # ------------------------------------------------------------------
    if weights.size:
        w_min = int(weights.min())
        w_max = int(weights.max())
        w_med = int(np.median(weights))
        demo_counts = np.unique([w_min, w_med, w_max])
        demo_widths = _edge_widths(demo_counts.astype(float), scale=width_scale)

        if len(demo_counts) == 1:
            xs = [0.5]
        else:
            xs = np.linspace(0.25, 0.75, num=len(demo_counts))

        title_y = 0.9
        line_y = 0.55
        label_y = 0.25

        ax_edge_legend.text(
            0.5,
            title_y,
            "Edge co-occurrence count",
            ha="center",
            va="top",
            fontsize=_LEGEND_TITLE_FONTSIZE,
        )

        for (count, lw), x in zip(zip(demo_counts, demo_widths), xs):
            ax_edge_legend.plot(
                [x - 0.12, x + 0.12],
                [line_y, line_y],
                lw=float(lw),
                color="gray",
            )
            ax_edge_legend.text(
                x,
                label_y,
                f"{int(count)}×",
                ha="center",
                va="top",
                fontsize=_LEGEND_TEXT_FONTSIZE,
            )

    # ------------------------------------------------------------------
    # 5. Node-size legend: numeric token frequencies (right column)
    # ------------------------------------------------------------------
    if counts.size:
        f_min = int(counts.min())
        f_max = int(counts.max())
        f_med = int(np.median(counts))
        demo_freq = np.unique([f_min, f_med, f_max])

        def _size_for_freq(v: float) -> float:
            if c_max > c_min:
                return min_size + (max_size - min_size) * (v - c_min) / (c_max - c_min)
            return (min_size + max_size) / 2.0

        demo_sizes = np.array([_size_for_freq(float(v)) for v in demo_freq])

        if len(demo_freq) == 1:
            xs2 = [0.5]
        else:
            xs2 = np.linspace(0.25, 0.75, num=len(demo_freq))

        title_y2 = 0.9
        scatter_y = 0.55
        label_y2 = 0.25

        ax_node_legend.text(
            0.5,
            title_y2,
            "Node frequency (# variants)",
            ha="center",
            va="top",
            fontsize=_LEGEND_TITLE_FONTSIZE,
        )

        for freq, size, x in zip(demo_freq, demo_sizes, xs2):
            ax_node_legend.scatter(
                [x],
                [scatter_y],
                s=float(size),
                c=[cmap(0.5)],
                edgecolor="none",
            )
            ax_node_legend.text(
                x,
                label_y2,
                f"{int(freq)}×",
                ha="center",
                va="top",
                fontsize=_LEGEND_TEXT_FONTSIZE,
            )

    # ------------------------------------------------------------------
    # 6. Colorbar: configured metric (edge or node) in the right column
    # ------------------------------------------------------------------
    if colorbar_norm is not None:
        sm = ScalarMappable(norm=colorbar_norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            cax=ax_cbar,
            orientation="vertical",
        )
        cbar.set_label(colorbar_label, fontsize=_COLORBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=_COLORBAR_TICK_FONTSIZE)

    # Final layout and export
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def edge_bundle_from_combos(
    *,
    aa_combo_strs: Sequence[str],
    k_values: Sequence[int],
    min_cooccur_count: int = 1,
    width_scale: str = "sqrt",
    out_png: str | Path,
    out_pdf: str | Path | None = None,
    figsize_in: float = 10.0,
    dpi: int = 200,
    edge_cmap: str = "viridis",
    color_by: str = "node_avg_k",
) -> pd.DataFrame:
    """
    Convenience wrapper used by the multisite_select protocol.

    1. Build a co-occurrence edge table from AA combo strings + k values.
    2. Render the circular edge-bundling plot.
    3. Return the edge table DataFrame (for parquet export by the caller).
    """
    # Build edge table; if it's empty, skip plotting but still return it.
    edges_df = build_edge_table(
        aa_combo_strs=aa_combo_strs,
        k_values=k_values,
        min_cooccur_count=min_cooccur_count,
    )
    if edges_df.empty:
        return edges_df

    # Node sizes reflect how often each mutation token appears across variants,
    # and node hue (when color_by='node_avg_k') encodes the average mut_count k
    # among variants in which that mutation appears.
    node_counts: Dict[str, int] = defaultdict(int)
    node_k_sum: Dict[str, float] = defaultdict(float)
    for combo, k in zip(aa_combo_strs, k_values):
        tokens = set(_parse_combo_tokens(combo))
        if not tokens:
            continue
        k_val = float(k)
        for tok in tokens:
            node_counts[tok] += 1
            node_k_sum[tok] += k_val
    node_avg_k = {tok: (node_k_sum[tok] / float(node_counts[tok])) for tok in node_counts}

    plot_edge_bundle(
        edges_df,
        out_png=out_png,
        out_pdf=out_pdf,
        figsize_in=figsize_in,
        dpi=dpi,
        width_scale=width_scale,
        node_counts=dict(node_counts),
        edge_cmap=edge_cmap,
        color_by=color_by,
        node_avg_k=node_avg_k,
        title=f"Mutation co-occurrence among {len(aa_combo_strs)} selected sequences",
    )
    return edges_df
