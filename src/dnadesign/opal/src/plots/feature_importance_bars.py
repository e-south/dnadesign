"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/feature_importance_bars.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._mpl_utils import annotate_plot_meta, log_kv

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

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
    for child in outputs_dir.iterdir():
        if not child.is_dir():
            continue
        m = _ROUND_DIR_RE.match(child.name)
        if not m:
            continue
        r = int(m.group(1))
        p = child / "feature_importance.csv"
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
        raise FileNotFoundError("No round_* folders with feature_importance.csv were found under outputs/.")

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
        summary="Overlayed feature-importance bars across rounds.",
        params={
            "order_policy": "preserve|sort_index (default preserve).",
            "alpha": "Bar transparency (default 0.45).",
            "figsize_in": "Figure size in inches (default [12, 5]).",
        },
        requires=["outputs/round_<k>/feature_importance.csv"],
        notes=["Reads per-round outputs, not ledger."],
    ),
)
def render(context, params: dict) -> None:
    """
    Params (all optional, assertively validated):
      - alpha: float in (0,1], transparency for overlaid bars (default 0.45)
      - bar_width: float (default 0.80)
      - cmap: str, Matplotlib colormap (default "tab10")
      - figsize_in: [W, H] inches (default [12, 5])
      - xtick_step: int, draw every Nth x tick (default: auto ≤ ~30 ticks)
      - title: str
      - ylabel: str
      - order_policy: "preserve" | "sort_index"  (default "preserve")
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # ---- Parameters
    alpha = float(params.get("alpha", 0.45))
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1].")

    bar_width = float(params.get("bar_width", 0.80))
    if not (0.05 <= bar_width <= 1.5):
        raise ValueError("bar_width must be in [0.05, 1.5].")

    cmap_name = str(params.get("cmap", "tab10"))
    figsize = tuple(params.get("figsize_in", (12, 5)))
    if len(figsize) != 2:
        raise ValueError("figsize_in must be a 2-element [W, H] list.")

    title = str(params.get("title", "Feature importance (overlay by round)"))
    ylabel = str(params.get("ylabel", "Importance"))
    xtick_step_cfg = params.get("xtick_step", None)
    xtick_step = int(xtick_step_cfg) if xtick_step_cfg is not None else None
    order_policy = str(params.get("order_policy", "preserve")).strip().lower()

    # ---- Discover files from round_* directories (decoupled from runs parquet)
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
    plt.rcParams.update(
        {
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    cmap = plt.cm.get_cmap(cmap_name, max(1, len(Ys)))
    for i, (r, y) in enumerate(Ys):
        ax.bar(
            x,
            y,
            width=bar_width,
            alpha=alpha,
            label=f"r{r}",
            color=cmap(i),
            edgecolor="none",
            align="center",
        )

    # ---- Axes, ticks, legend, labels
    if xtick_step is None:
        # Aim for ~30 ticks max by default (assertive, deterministic)
        xtick_step = max(1, int(np.ceil(n_features / 30)))
    ax.set_xticks(x[::xtick_step])
    ax.set_xticklabels([str(order[i]) for i in range(0, n_features, xtick_step)], rotation=0)
    ax.set_xlim(-0.5, n_features - 0.5)
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.05)

    ax.set_xlabel("Feature index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    lg = ax.legend(title="round", frameon=False, loc="upper right")
    if lg and lg.get_title():
        lg.get_title().set_fontsize(11)

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
    annotate_plot_meta(
        ax,
        hue="round",
        size_by=None,
        alpha=alpha,
        rasterized=False,
        extras={"rounds": f"{len(target_rounds)}", "bars": "overlay"},
    )

    # ---- Save
    out = context.output_dir / context.filename
    fig.savefig(out, dpi=context.dpi, bbox_inches="tight")
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
