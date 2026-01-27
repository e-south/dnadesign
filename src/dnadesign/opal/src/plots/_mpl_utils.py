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
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.tmpdir import resolve_opal_tmpdir
from ..core.utils import OpalError

if TYPE_CHECKING:
    import numpy as np


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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "axes.titlepad": 6,
        }
    )


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
