"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/feature_importance_bars.py

Aggregates per-round feature importance artifacts into plots. Discovers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._mpl_utils import (
    apply_notebook_axes_style,
    apply_plot_style,
    ensure_mpl_config_dir,
    log_kv,
    pretty_label,
    pretty_title,
    save_notebook_square_figure,
    sequential_colormap,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

DEFAULT_FEATURE_IMPORTANCE_BARS_FIGSIZE: tuple[float, float] = (14.0, 4.4)

# -----------------------------
# Helpers (pure, testable)
# -----------------------------

_ROUND_DIR_RE = re.compile(r"^round_(\d+)$")


def _discover_round_fi_files(outputs_dir: Path) -> Dict[int, Path]:
    """
    Return {round_index: feature_importance.csv path} for every round_* dir
    that contains the file. Strictly requires the file to exist.
    """
    found: Dict[int, Path] = {}
    rounds_dir = outputs_dir / "rounds"
    if not rounds_dir.exists():
        return {}
    for child in rounds_dir.iterdir():
        if not child.is_dir():
            continue
        m = _ROUND_DIR_RE.match(child.name)
        if not m:
            continue
        r = int(m.group(1))
        p = child / "model" / "feature_importance.csv"
        if p.exists():
            found[r] = p.resolve()
    return dict(sorted(found.items()))


def _read_fi_csv(path: Path, round_idx: int) -> pd.DataFrame:
    """
    Strict CSV loader: requires columns {'feature_index','importance'}, no NaNs,
    and unique feature_index. Adds 'as_of_round' and '__order__' columns.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path)
    want = {"feature_index", "importance"}
    missing = sorted(list(want - set(df.columns)))
    if missing:
        raise ValueError(f"[feature_importance_bars] {path}: missing required columns {missing}")

    df = df.loc[:, ["feature_index", "importance"]].copy()
    # Coerce types and validate
    df["feature_index"] = pd.to_numeric(df["feature_index"], errors="raise").astype(int)
    df["importance"] = pd.to_numeric(df["importance"], errors="raise").astype(float)

    if df["feature_index"].duplicated().any():
        dups = df.loc[df["feature_index"].duplicated(), "feature_index"].unique().tolist()
        raise ValueError(f"[feature_importance_bars] {path}: duplicate feature_index values: {dups}")
    if not np.isfinite(df["importance"].to_numpy()).all():
        bad = df.loc[~np.isfinite(df["importance"]), "feature_index"].tolist()
        raise ValueError(f"[feature_importance_bars] {path}: non-finite importance values at feature_index={bad}")

    df["as_of_round"] = int(round_idx)
    df["__order__"] = np.arange(len(df), dtype=int)  # preserve file order
    return df


def _select_rounds(available: List[int], rounds_sel) -> List[int]:
    """
    Decide the target rounds from context.rounds, assertively.
    """
    if not available:
        raise FileNotFoundError("No round_* folders with feature_importance.csv were found under outputs/rounds/.")

    if rounds_sel in ("unspecified", "latest"):
        return [max(available)]
    if rounds_sel == "all":
        return available

    # explicit list or single int
    req = rounds_sel if isinstance(rounds_sel, list) else [rounds_sel]
    try:
        req = sorted(set(int(x) for x in req))
    except Exception as e:
        raise ValueError(f"Invalid --round selector: {rounds_sel!r}") from e

    missing = [r for r in req if r not in available]
    if missing:
        raise FileNotFoundError(
            f"Requested rounds {missing} do not have feature_importance.csv. Available: {available}"
        )
    return req


def _resolve_order(frames: List[pd.DataFrame], policy: str) -> List[int]:
    """
    Return the canonical feature_index order according to policy.

    Policies:
      - "preserve"   : require identical order across rounds; use order of first frame
      - "sort_index" : require identical feature sets; order by ascending feature_index
    """
    import numpy as np

    policy = str(policy or "preserve").strip().lower()
    if policy not in {"preserve", "sort_index"}:
        raise ValueError("order_policy must be 'preserve' or 'sort_index'.")

    first = frames[0]
    set0 = set(first["feature_index"].tolist())

    # Validate sets & (optionally) order
    for f in frames[1:]:
        s = set(f["feature_index"].tolist())
        if s != set0:
            extra = sorted(list(s - set0))
            missing = sorted(list(set0 - s))
            msg = []
            if extra:
                msg.append(f"extra={extra}")
            if missing:
                msg.append(f"missing={missing}")
            raise ValueError("[feature_importance_bars] Mismatched feature sets across rounds: " + "; ".join(msg))

        if policy == "preserve":
            if not np.array_equal(f["feature_index"].to_numpy(), first["feature_index"].to_numpy()):
                raise ValueError(
                    "[feature_importance_bars] order_policy='preserve' requires "
                    "identical feature_index order across rounds. "
                    "Set params.order_policy: 'sort_index' if you prefer a stable sort."
                )

    if policy == "preserve":
        return first.sort_values("__order__")["feature_index"].tolist()
    else:  # sort_index
        return sorted(list(set0))


# -----------------------------
# Plot plugin
# -----------------------------


@register_plot(
    "feature_importance_bars",
    meta=PlotMeta(
        summary="Overlaid feature-importance bars across rounds.",
        params={
            "order_policy": "preserve|sort_index (default preserve).",
            "alpha": "Bar transparency (default 0.45).",
            "figsize_in": "Figure size in inches (default landscape [14.0, 4.4]).",
        },
        requires=["outputs/rounds/round_<k>/model/feature_importance.csv"],
        notes=["Reads per-round outputs, not ledger."],
        data_shape="attribution matrix",
        tidy_schema=["as_of_round", "feature_index", "importance"],
        objective_family="generic",
        data_layer="model_artifact",
        round_scope="round_history",
        requires_model_artifact=True,
        failure_modes=[
            "missing feature_importance.csv",
            "duplicate or inconsistent feature IDs",
            "non-finite importances",
            "requested round has no feature-importance artifact",
        ],
    ),
)
def render(context, params: dict) -> None:
    """
    Params (all optional, assertively validated):
      - alpha: float in (0,1], transparency for overlaid bars (default 0.45)
      - bar_width: float (default 0.80)
      - cmap: str, Matplotlib colormap (default "round_progression")
      - figsize_in: [W, H] inches (default [14.0, 4.4])
      - xtick_step: int, draw every Nth x tick (default: auto ≤ ~30 ticks)
      - title: str
      - ylabel: str
      - order_policy: "preserve" | "sort_index"  (default "preserve")
    """
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # ---- Parameters
    alpha = float(params.get("alpha", 0.45))
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    bar_width = float(params.get("bar_width", 1.05))
    if not (0.05 <= bar_width <= 1.5):
        raise ValueError("bar_width must be in [0.05, 1.5].")

    cmap_name = str(params.get("cmap", "round_progression"))
    figsize = tuple(params.get("figsize_in", DEFAULT_FEATURE_IMPORTANCE_BARS_FIGSIZE))
    if len(figsize) != 2:
        raise ValueError("figsize_in must be a 2-element [W, H] list.")

    title = pretty_title(params.get("title", "Feature importance by round"))
    ylabel = str(params.get("ylabel", "rf_feature_importance"))
    xtick_step_cfg = params.get("xtick_step", None)
    xtick_step = int(xtick_step_cfg) if xtick_step_cfg is not None else None
    order_policy = str(params.get("order_policy", "preserve")).strip().lower()

    # ---- Discover files from outputs/rounds/round_* directories (decoupled from runs parquet)
    outputs_dir = resolve_outputs_dir(context)
    fi_map = _discover_round_fi_files(outputs_dir)  # {round: path}
    available_rounds = sorted(fi_map.keys())
    target_rounds = _select_rounds(available_rounds, context.rounds)

    # ---- Load CSVs for selected rounds
    frames: List[pd.DataFrame] = []
    for r in target_rounds:
        frames.append(_read_fi_csv(fi_map[r], r))

    # ---- Resolve canonical order (assertive; no silent unions)
    order = _resolve_order(frames, policy=order_policy)
    n_features = len(order)
    x = np.arange(n_features, dtype=float)

    # ---- Build aligned arrays per round (strict mapping by feature_index)
    Ys: List[Tuple[int, np.ndarray]] = []
    ymax = 0.0
    for f in frames:
        r = int(f["as_of_round"].iloc[0])
        m = dict(zip(f["feature_index"].tolist(), f["importance"].tolist()))
        y = np.array([float(m[fi]) for fi in order], dtype=float)
        ymax = max(ymax, float(np.nanmax(y)))
        Ys.append((r, y))

    # ---- Figure
    apply_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    apply_notebook_axes_style(ax, square=False)

    cmap = sequential_colormap(cmap_name)
    denom = max(len(Ys) - 1, 1)
    round_min = min(r for r, _ in Ys)
    round_max = max(r for r, _ in Ys)
    for i, (r, y) in enumerate(Ys):
        ax.bar(
            x,
            y,
            width=bar_width,
            alpha=alpha,
            label=f"Round {r}",
            color=cmap(i / denom),
            edgecolor="none",
            align="center",
        )

    # ---- Axes, ticks, legend, labels
    if xtick_step is None:
        # Aim for ~30 ticks max by default (assertive, deterministic)
        max_xticks = int(params.get("max_xticks", 18))
        xtick_step = max(1, int(np.ceil(n_features / max(1, max_xticks))))
    ax.set_xticks(x[::xtick_step])
    ax.set_xticklabels([str(order[i]) for i in range(0, n_features, xtick_step)], rotation=0)
    ax.set_xlim(-0.5, n_features - 0.5)
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.05)

    ax.set_xlabel("Feature index")
    ylabel_text = "RF importance" if ylabel == "rf_feature_importance" else pretty_label(ylabel)
    ax.set_ylabel(ylabel_text)
    ax.set_title(title)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.34, top=0.80)
    sm = ScalarMappable(norm=Normalize(vmin=round_min, vmax=round_max), cmap=cmap)
    sm.set_array([])
    fig.canvas.draw()
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x0, max(0.07, bbox.y0 - 0.18), bbox.width, 0.032])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Round")
    if len(Ys) <= 12:
        cbar.set_ticks([r for r, _ in Ys])
    else:
        cbar.set_ticks([round_min, round_max])

    # ---- Log + annotate
    log_kv(
        context.logger,
        "feature_importance_bars",
        rounds=target_rounds,
        alpha=float(alpha),
        bar_width=float(bar_width),
        cmap=cmap_name,
        features=int(n_features),
        order_policy=order_policy,
    )

    # ---- Save
    out = context.output_dir / context.filename
    save_notebook_square_figure(fig, out, dpi=context.dpi, tight=False)
    plt.close(fig)

    # Optional tidy export (long form)
    if context.save_data:
        tidy = (
            pd.concat(frames, ignore_index=True)
            .loc[:, ["as_of_round", "feature_index", "importance", "__order__"]]
            .sort_values(["as_of_round", "__order__"])
            .drop(columns="__order__")
        )
        context.save_df(tidy)
