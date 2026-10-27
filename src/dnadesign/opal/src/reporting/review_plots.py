"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/review_plots.py

Renders compact diagnostic plots for portable OPAL campaign reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..core.utils import ExitCodes, OpalError
from ..plots._mpl_utils import apply_plot_style, ensure_mpl_config_dir, pretty_label, scatter_smart


def write_score_vs_rank_plot(
    predictions: pd.DataFrame,
    selected_mask: pd.Series,
    path: Path,
    *,
    campaign_name: str,
    selection_view_id: str,
    round_index: int,
) -> None:
    ensure_mpl_config_dir(workdir=path.parent)
    apply_plot_style()
    import matplotlib.pyplot as plt

    if len(selected_mask) != len(predictions) or not selected_mask.index.equals(predictions.index):
        raise ValueError("Campaign review selected mask must align exactly with prediction rows.")
    df = predictions.copy()
    df["view__selection_score"] = pd.to_numeric(df["view__selection_score"], errors="raise")
    df["view__rank_competition"] = pd.to_numeric(df["view__rank_competition"], errors="raise").astype(int)
    df["_selected"] = selected_mask.to_numpy(dtype=bool)
    df = df.sort_values(["view__rank_competition", "id"])
    selected = df["_selected"]
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    scatter_smart(
        ax,
        df["view__rank_competition"].to_numpy(),
        df["view__selection_score"].to_numpy(),
        s=12,
        alpha=0.35,
        rasterize_at=20_000,
        label="Prediction pool",
    )
    if selected.any():
        scatter_smart(
            ax,
            df.loc[selected, "view__rank_competition"].to_numpy(),
            df.loc[selected, "view__selection_score"].to_numpy(),
            s=34,
            alpha=0.9,
            edgecolors="black",
            rasterize_at=20_000,
            label="Allocated to view",
            zorder=3,
        )
    ax.set_xlabel(r"Competition rank, $k$ (rank 1 at right)")
    ax.set_ylabel("Predicted objective score")
    ax.set_title(
        f"{campaign_name}\n{pretty_label(selection_view_id)} selection view · round {round_index}",
        loc="left",
    )
    maximum_rank = int(df["view__rank_competition"].max())
    ax.set_xscale("log")
    ax.set_xlim(left=maximum_rank * 1.08, right=0.8)
    ax.legend(loc="upper left", frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_feature_importance_plot(
    source_path: Path,
    output_path: Path,
    *,
    round_index: int,
    top_n: int = 30,
) -> dict[str, object]:
    if not source_path.exists():
        return {
            "name": "feature_importance_top",
            "status": "not_available",
            "reason": f"feature importance artifact missing: {source_path}",
        }
    df = pd.read_csv(source_path)
    missing = sorted({"feature_index", "importance"} - set(df.columns))
    if missing:
        raise OpalError(
            f"feature_importance.csv missing campaign review column(s): {missing}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    df["feature_index"] = pd.to_numeric(df["feature_index"], errors="raise").astype(int)
    df["importance"] = pd.to_numeric(df["importance"], errors="raise")
    df = df.sort_values(["importance", "feature_index"], ascending=[False, True]).head(int(top_n))
    ensure_mpl_config_dir(workdir=output_path.parent)
    apply_plot_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    labels = [str(idx) for idx in df["feature_index"].tolist()]
    ax.bar(labels, df["importance"].to_numpy(), color="#446A8C")
    ax.set_xlabel("Sequence feature index")
    ax.set_ylabel("Model importance")
    ax.set_title(
        (
            f"Top {len(df)} model feature importances · round {round_index}\n"
            "Shared model diagnostic across selection views"
        ),
        loc="left",
    )
    ax.tick_params(axis="x", rotation=90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {
        "name": "feature_importance_top",
        "status": "written",
        "scope": "shared_model_diagnostic",
        "path": str(output_path),
    }


__all__ = ["write_feature_importance_plot", "write_score_vs_rank_plot"]
