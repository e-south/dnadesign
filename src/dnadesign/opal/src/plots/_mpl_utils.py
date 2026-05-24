"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/_mpl_utils.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json as _json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.tmpdir import resolve_opal_tmpdir
from ..core.utils import OpalError

if TYPE_CHECKING:
    import numpy as np


COLORBLIND_PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
CATEGORY_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "P", "X", "v", "*")
CATEGORY_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")
DEFAULT_SQUARE_FIGSIZE: tuple[float, float] = (7.2, 7.2)
DEFAULT_LANDSCAPE_FIGSIZE: tuple[float, float] = (11.0, 3.8)
DEFAULT_PANORAMA_FIGSIZE: tuple[float, float] = (12.0, 3.6)

_PRETTY_LABELS = {
    "as_of_round": "Round",
    "observed_round": "Observed round",
    "round": "Round",
    "id": "ID",
    "run_id": "Run ID",
    "pred__score_selected": "Predicted selected score",
    "pred__score_ref": "Predicted reference score",
    "pred__y_hat_model": "Predicted response vector",
    "obj__logic_fidelity": "Logic fidelity",
    "obj__effect_raw": "Raw effect",
    "obj__effect_scaled": "Scaled effect",
    "obj__diag__setpoint": "Setpoint",
    "sel__is_selected": "Selected",
    "sel__rank_competition": "Selection rank",
    "rank__sequential": "Sequential rank",
    "logic_fidelity": "Logic fidelity",
    "fold_change": "Fold change",
    "effect_raw": "Raw effect",
    "effect_scaled": "Scaled effect",
    "score": "Score",
    "dist_to_labeled_logic": "Distance to labeled logic",
    "uncertainty": "Model uncertainty",
    "denom_used": "Denominator used",
    "clip_lo": "Lower clipping",
    "clip_hi": "Upper clipping",
    "clip_lo_fraction": "Lower clipping fraction",
    "clip_hi_fraction": "Upper clipping fraction",
    "E_raw": "Raw effect (E_raw)",
    "e_raw": "Raw effect (E_raw)",
    "mse": "MSE",
    "iqr": "IQR",
    "q25": "Q25",
    "q75": "Q75",
    "all_pool": "All pool",
    "top_k": "Top K",
    "selected": "Selected",
    "mean": "Mean",
    "median": "Median",
    "count": "Count",
    "importance": "Importance",
    "rf_feature_importance": "RF importance",
    "feature_index": "Feature index",
    "feature_id": "Feature ID",
    "AB interaction": "AB interaction",
    "reference_mse": "MSE to reference vector",
}

_ACRONYMS = {
    "ab": "AB",
    "api": "API",
    "dna": "DNA",
    "id": "ID",
    "iqr": "IQR",
    "mse": "MSE",
    "rf": "RF",
    "sfxi": "SFXI",
    "and": "AND",
    "vs": "vs",
    "x": "X",
    "y": "Y",
}

_MATH_LABELS = {
    "pred__score_selected": r"Predicted objective score, $\hat{S}$",
    "obj__logic_fidelity": r"Logic fidelity, $F_{\ell}=1-\|\hat{\ell}-s\|_2/D$",
    "obj__effect_scaled": r"Scaled effect, $E_{\mathrm{scaled}}=\mathrm{clip}(E_{\mathrm{raw}}/d,0,1)$",
    "obj__effect_raw": r"Raw effect, $E_{\mathrm{raw}}=\sum_i w_i\,\max(0,2^{y_i^*}-\delta)$",
    "logic_fidelity": r"Logic fidelity, $F_{\ell}=1-\|\hat{\ell}-s\|_2/D$",
    "objective_score": r"Objective score, $S=F_{\ell}^{\beta}E_{\mathrm{scaled}}^{\gamma}$",
    "effect_scaled": r"Scaled effect, $E_{\mathrm{scaled}}=\mathrm{clip}(E_{\mathrm{raw}}/d,0,1)$",
    "effect_raw": r"Raw effect, $E_{\mathrm{raw}}=\sum_i w_i\,\max(0,2^{y_i^*}-\delta)$",
    "mse_to_reference": r"MSE to reference, $d^{-1}\sum_i(\bar{y}_i-r_i)^2$",
    "support_distance": r"Nearest labeled logic distance, $\min_j\|\hat{\ell}-\ell_j\|_2$",
    "score_uncertainty": r"Score uncertainty, $\sigma(\hat{S}_{\mathrm{tree}})$",
    "factorial_a": r"A effect, $\frac{(v_{10}+v_{11})-(v_{00}+v_{01})}{2}$",
    "factorial_b": r"B effect, $\frac{(v_{01}+v_{11})-(v_{00}+v_{10})}{2}$",
    "factorial_ab": r"AB interaction, $\frac{v_{11}+v_{00}-v_{10}-v_{01}}{2}$",
}


def ensure_mpl_config_dir(*, workdir: Path | None = None) -> Path:
    """
    Ensure Matplotlib has a writable config/cache directory.
    Only called by plotting workflows and must run before importing matplotlib.
    """
    env_val = os.getenv("MPLCONFIGDIR", "").strip()
    if env_val:
        path = Path(env_val).expanduser()
    else:
        path = resolve_opal_tmpdir(workdir=workdir) / "mpl"
        os.environ["MPLCONFIGDIR"] = str(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise OpalError(f"Matplotlib config dir is not writable: {path}") from exc
    if not path.is_dir():
        raise OpalError(f"Matplotlib config dir is not a directory: {path}")
    if not os.access(path, os.W_OK):
        raise OpalError(f"Matplotlib config dir is not writable: {path}")
    return path


def _apply_perf_rcparams() -> None:
    # Cheap wins for large point clouds
    ensure_mpl_config_dir()
    import matplotlib.pyplot as plt

    plt.rcParams["agg.path.chunksize"] = int(os.getenv("OPAL_MPL_PATH_CHUNKSIZE", "10000"))
    plt.rcParams["path.simplify"] = True
    plt.rcParams["path.simplify_threshold"] = 0.0  # keep geometry intact


def apply_plot_style(*, variant: str = "diagnostic") -> None:
    """
    Apply consistent plot styling for OPAL diagnostics.
    """
    ensure_mpl_config_dir()
    import matplotlib.pyplot as plt

    if variant != "diagnostic":
        raise ValueError(f"Unknown plot style variant: {variant}")

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "figure.titlesize": 18,
            "axes.titlepad": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "axes.titlecolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "axes.edgecolor": "#111111",
            "legend.frameon": False,
            "axes.axisbelow": True,
            "axes.grid": True,
            "grid.color": "#E6E6E6",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
            "axes.prop_cycle": plt.cycler(color=COLORBLIND_PALETTE),
        }
    )


def apply_notebook_axes_style(ax, *, grid: bool = True, square: bool = False) -> None:
    """Apply the notebook-review axes contract for static OPAL plot artifacts."""

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#B8B8B8")
        ax.spines[spine].set_linewidth(0.8)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, color="#E6E6E6", linewidth=0.8, alpha=0.85)
    else:
        ax.grid(False)
    ax.tick_params(axis="both", direction="out", length=4, width=0.8)
    if square:
        try:
            ax.set_box_aspect(1.0)
        except Exception:
            ax.set_aspect("equal", adjustable="box")


def save_notebook_square_figure(fig, out: Path, *, dpi: int, tight: bool = True) -> None:
    """Save a square notebook artifact without legend/crop-driven shape drift."""

    if tight:
        try:
            fig.tight_layout(pad=0.35)
        except Exception:
            pass
    fig.savefig(out, dpi=dpi, facecolor="white")


def categorical_color(index: int) -> str:
    return COLORBLIND_PALETTE[int(index) % len(COLORBLIND_PALETTE)]


def categorical_marker(index: int) -> str:
    return CATEGORY_MARKERS[int(index) % len(CATEGORY_MARKERS)]


def categorical_linestyle(index: int) -> str:
    return CATEGORY_LINESTYLES[int(index) % len(CATEGORY_LINESTYLES)]


def categorical_style(index: int) -> dict[str, str]:
    return {
        "color": categorical_color(index),
        "marker": categorical_marker(index),
        "linestyle": categorical_linestyle(index),
    }


def categorical_colormap(name: str | None = None, *, n: int | None = None):
    """Return a categorical colormap, defaulting to the OPAL colorblind palette."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    label = str(name or "okabe_ito").strip().lower().replace("-", "_")
    if label in {"okabe_ito", "okabeito", "colorblind", "opal"}:
        size = max(int(n or len(COLORBLIND_PALETTE)), 1)
        colors = [categorical_color(index) for index in range(size)]
        return ListedColormap(colors, name="opal_okabe_ito")
    return plt.get_cmap(str(name), n) if n is not None else plt.get_cmap(str(name))


def sequential_colormap(name: str | None = None, *, n: int | None = None):
    """Return an OPAL sequential colormap with quiet low values and readable highs."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    label = str(name or "opal_importance").strip().lower().replace("-", "_")
    palettes = {
        "opal_importance": ("#FFFFFF", "#E7F0F7", "#9EC2DE", "#2E75B6", "#073B5B"),
        "importance": ("#FFFFFF", "#E7F0F7", "#9EC2DE", "#2E75B6", "#073B5B"),
        "rf_importance": ("#FFFFFF", "#E7F0F7", "#9EC2DE", "#2E75B6", "#073B5B"),
        "seafoam": ("#FFFFFF", "#DDF5ED", "#93D8C6", "#2CA58D", "#005F56"),
        "opal_seafoam": ("#FFFFFF", "#DDF5ED", "#93D8C6", "#2CA58D", "#005F56"),
        "round_progression": ("#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"),
    }
    if label in palettes:
        cmap = LinearSegmentedColormap.from_list(f"opal_{label}", palettes[label], N=int(n or 256))
        cmap.set_bad(color="#F2F2F2")
        return cmap
    return plt.get_cmap(str(name), n) if n is not None else plt.get_cmap(str(name))


def math_label(key: str, *, fallback: object | None = None) -> str:
    """Return a compact math-aware label for objective quantities."""

    token = str(key or "").strip()
    if token in _MATH_LABELS:
        return _MATH_LABELS[token]
    if fallback is not None:
        return pretty_label(fallback)
    return pretty_label(token)


def pretty_label(value: object, *, raw: bool = False) -> str:
    """Humanize OPAL field slugs while optionally retaining the exact raw token."""

    token = str(value or "").strip()
    if not token:
        return ""
    pretty = _PRETTY_LABELS.get(token)
    if pretty is None:
        pretty = _PRETTY_LABELS.get(token.replace(".", "__"))
    if pretty is None:
        pretty = _humanize_slug(token)
    if raw and token and token != pretty:
        return f"{pretty} ({token})"
    return pretty


def pretty_title(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "_" not in text and "__" not in text and "-" not in text:
        lower = text.lower()
        if lower in _ACRONYMS:
            return _ACRONYMS[lower]
        return text[:1].upper() + text[1:] if text.islower() else text
    text = text.replace("__", " ").replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split(" "):
        lower = word.lower()
        if lower in _ACRONYMS:
            words.append(_ACRONYMS[lower])
        elif len(lower) == 1 and lower.isalpha():
            words.append(lower.upper())
        elif lower == "vec8":
            words.append("vec8")
        else:
            words.append(lower)
    label = " ".join(words)
    label = re.sub(r"\btop n\b", "top-N", label, flags=re.IGNORECASE)
    return label[:1].upper() + label[1:]


def legend_below_single_row(fig, ax, *, handles=None, labels=None, bottom: float = 0.11) -> bool:
    """Place a legend below the plot as one compact row and reserve minimal space."""

    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    pairs = [(handle, label) for handle, label in zip(handles, labels) if str(label) and not str(label).startswith("_")]
    if not pairs:
        return False
    handles, labels = zip(*pairs)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(labels),
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, bottom, 1, 1), pad=0.35)
    return True


def add_flush_colorbar(
    fig,
    ax,
    mappable,
    *,
    label: str | None = None,
    size: str = "4%",
    pad: float = 0.08,
    ticklabelsize: float | None = None,
):
    """Add a vertical colorbar whose top and bottom align with the target axes."""

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cax = inset_axes(
        ax,
        width=size,
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.0 + float(pad), 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    cax.set_in_layout(False)
    cbar = fig.colorbar(mappable, cax=cax)
    if label:
        cbar.set_label(label)
    if ticklabelsize is not None:
        cbar.ax.tick_params(labelsize=ticklabelsize)
    try:
        from matplotlib.ticker import MaxNLocator

        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    except Exception:
        pass
    return cbar


def _humanize_slug(value: str) -> str:
    text = value
    for prefix in ("records.", "sfxi_ref__", "obj__diag__", "obj__", "pred__", "sel__", "opal__"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("__", "_")
    parts = [part for part in text.replace("-", "_").split("_") if part]
    if not parts:
        return value
    words = []
    for part in parts:
        lower = part.lower()
        words.append(_ACRONYMS.get(lower, part.upper() if part.startswith("v") and part[1:].isdigit() else lower))
    label = " ".join(words)
    return label[:1].upper() + label[1:]


def scatter_smart(ax, x, y, *, s=16, alpha=0.85, rasterize_at=None, edgecolors="none", **kw):
    """
    Always deterministic; switches to rasterized draw above 'rasterize_at' points
    to prevent vector-graphics ballooning and crashes in backends like PDF/PS.

    No downsampling here (no fallbacks); just a drawing-mode choice.
    """
    import numpy as np

    _apply_perf_rcparams()
    x = np.asarray(x, dtype=np.float32)  # halves memory vs float64
    y = np.asarray(y, dtype=np.float32)
    # Opt-in rasterization: None/0/negative → never rasterize.
    thr = None
    if rasterize_at is not None:
        try:
            thr = int(rasterize_at)
        except Exception as exc:
            raise ValueError("rasterize_at must be an int or None.") from exc
        if thr < 0:
            raise ValueError("rasterize_at must be >= 0.")
    rasterized = (thr is not None) and (thr > 0) and (x.size >= thr)
    # Respect explicit overrides without passing the same kw twice.
    # Allow both linewidths/linewidth alias; default to 0 when unspecified.
    lw = kw.pop("linewidths", None)
    lw_single = kw.pop("linewidth", None)
    if lw is None and lw_single is not None:
        lw = lw_single
    if lw is None:
        lw = 0
    # If callers mistakenly put edgecolors in **kw, let that win.
    edgecolors = kw.pop("edgecolors", edgecolors)
    # Drop cmap if no color data provided.
    if "cmap" in kw and (("c" not in kw) or (kw.get("c") is None)):
        kw.pop("cmap", None)
    return ax.scatter(
        x,
        y,
        s=s,
        alpha=alpha,
        linewidths=lw,
        edgecolors=edgecolors,
        rasterized=rasterized,
        **kw,
    )


def scale_to_sizes(values, *, s_min: float = 10.0, s_max: float = 60.0, clip=None) -> np.ndarray:
    """
    Map a numeric vector to point sizes in [s_min, s_max].
    Non-finite → s_min. If the vector is (near-)constant, return s_min.
    """
    import numpy as np

    v = np.asarray(values, dtype=np.float32).ravel()
    mask = np.isfinite(v)
    v = v.copy()
    if clip is not None and np.all(np.isfinite(clip)) and len(clip) == 2:
        v[mask] = np.clip(v[mask], float(clip[0]), float(clip[1]))
    if not np.any(mask) or np.nanmax(v) <= np.nanmin(v):
        return np.full(v.shape, float(s_min), dtype=np.float32)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    out = s_min + ((v - lo) / max(hi - lo, 1e-12)) * (s_max - s_min)
    out[~mask] = float(s_min)
    return out


def annotate_plot_meta(
    ax,
    *,
    hue: str | None = None,
    size_by: str | None = None,
    alpha: float | None = None,
    rasterized: bool = False,
    extras: dict | None = None,
    loc: str = "upper left",
    fontsize: float = 9.0,
) -> None:
    """Small, unobtrusive top-left text to document what drove color/size and draw mode."""
    lines = []
    lines.append(f"hue: {hue or '—'}")
    lines.append(f"size: {size_by or '—'}")
    if alpha is not None:
        lines.append(f"alpha: {alpha:.2f}")
    lines.append(f"rasterized: {'yes' if rasterized else 'no'}")
    if extras:
        for k in sorted(extras.keys()):
            lines.append(f"{k}: {extras[k]}")
    # location
    x = 0.01 if "left" in loc else 0.99
    y = 0.99 if "upper" in loc else 0.01
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"),
    )


def log_kv(logger, plot: str, **kwargs) -> None:
    """Uniform one-line key/value logging for plots."""
    items = {k: kwargs[k] for k in sorted(kwargs.keys())}
    logger.info("[%s] %s", plot, _json.dumps(items, separators=(",", ":"), sort_keys=True))


def swarm_smart(
    ax,
    x_positions: list[float] | np.ndarray,
    y_by_group: list[np.ndarray] | list[list[float]],
    *,
    jitter: float = 0.08,
    max_points_per_group: int = 3000,
    s: float = 10.0,
    sizes_by_group: list[np.ndarray] | None = None,
    hue_by_group: list[np.ndarray] | None = None,
    cmap: str | None = None,
    alpha: float = 0.25,
    seed: int = 0,
    rasterize_at: int = 20000,
) -> int:
    """
    Memory-conscious jittered swarm. Deterministically subsamples per group and
    draws with rasterization when large.
    """
    import numpy as np

    rng = np.random.default_rng(int(seed))
    total = 0
    for gi, (xi, yy) in enumerate(zip(list(x_positions), list(y_by_group))):
        y = np.asarray(yy, dtype=np.float32).ravel()
        if y.size == 0:
            continue
        m = min(int(max_points_per_group), y.size)
        if y.size > m:
            idx = rng.choice(y.size, size=m, replace=False)
            y = y[idx]
        else:
            idx = slice(None)  # no subsample
        x = np.full(y.shape[0], float(xi), dtype=np.float32)
        x += rng.uniform(-jitter, jitter, size=y.shape[0]).astype(np.float32)
        total += y.shape[0]
        # Per-point sizes
        s_kw = s
        if sizes_by_group is not None and gi < len(sizes_by_group):
            gsz = np.asarray(sizes_by_group[gi], dtype=np.float32).ravel()
            gsz = gsz[idx] if isinstance(idx, np.ndarray) else gsz
            if gsz.size == y.size:
                s_kw = gsz
        # Optional hue coloring
        kw = {}
        if hue_by_group is not None and gi < len(hue_by_group):
            gh = np.asarray(hue_by_group[gi], dtype=np.float32).ravel()
            gh = gh[idx] if isinstance(idx, np.ndarray) else gh
            if gh.size == y.size:
                kw = {"c": gh}
                if cmap:
                    kw["cmap"] = cmap
        scatter_smart(ax, x, y, s=s_kw, alpha=alpha, rasterize_at=rasterize_at, **kw)
    return total
