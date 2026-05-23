"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/feature_importance_heatmap.py

Generic attribution-matrix heatmap over OPAL rounds.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._mpl_utils import apply_notebook_axes_style, apply_plot_style, ensure_mpl_config_dir, save_notebook_square_figure
from .feature_importance_bars import _discover_round_fi_files, _read_fi_csv, _resolve_order, _select_rounds


@register_plot(
    "feature_importance_heatmap",
    meta=PlotMeta(
        summary="Feature-importance heatmap with stable feature rows and round columns.",
        params={
            "order_policy": "preserve|sort_index (default preserve).",
            "top_n": "Optional top N features by max importance.",
            "sort": "preserve|sort_index|max_importance (default preserve).",
            "cluster": "Optional clustering flag; defaults false and is currently unsupported when true.",
            "cmap": "Matplotlib colormap (default viridis).",
        },
        requires=["outputs/rounds/round_<k>/model/feature_importance.csv"],
        notes=["Preserves feature identity by default; clustering is intentionally off by default."],
        data_shape="attribution matrix",
        tidy_schema=["round", "feature_id", "importance", "rank", "source_path"],
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
    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    apply_plot_style()
    if bool(params.get("cluster", False)):
        raise ValueError("feature_importance_heatmap does not cluster by default; set cluster: false.")
    order_policy = str(params.get("order_policy", params.get("sort", "preserve"))).strip().lower()
    if order_policy == "max_importance":
        strict_order_policy = "sort_index"
    else:
        strict_order_policy = order_policy
    top_n = params.get("top_n")
    top_n_int = int(top_n) if top_n is not None else None
    if top_n_int is not None and top_n_int <= 0:
        raise ValueError("top_n must be positive when provided.")

    outputs_dir = resolve_outputs_dir(context)
    fi_map = _discover_round_fi_files(outputs_dir)
    target_rounds = _select_rounds(sorted(fi_map.keys()), context.rounds)
    frames = []
    for round_index in target_rounds:
        frame = _read_fi_csv(fi_map[round_index], round_index)
        frame["source_path"] = str(fi_map[round_index])
        frames.append(frame)

    order = _resolve_order(frames, policy=strict_order_policy)
    tidy = pd.concat(frames, ignore_index=True)
    tidy["rank"] = tidy.groupby("as_of_round")["importance"].rank(method="min", ascending=False).astype(int)
    if order_policy == "max_importance":
        max_by_feature = tidy.groupby("feature_index")["importance"].max().sort_values(ascending=False)
        order = [int(feature) for feature in max_by_feature.index.to_list()]
    if top_n_int is not None:
        by_max = tidy.groupby("feature_index")["importance"].max().sort_values(ascending=False)
        keep = set(int(feature) for feature in by_max.head(top_n_int).index.to_list())
        order = [feature for feature in order if int(feature) in keep]
        tidy = tidy[tidy["feature_index"].isin(keep)]
    if not order:
        raise ValueError("feature_importance_heatmap has no features after filtering.")

    matrix = []
    for feature_index in order:
        row = []
        sub = tidy[tidy["feature_index"] == feature_index]
        by_round = dict(zip(sub["as_of_round"].astype(int), sub["importance"].astype(float)))
        for round_index in target_rounds:
            if round_index not in by_round:
                raise ValueError(f"feature {feature_index} is missing for round {round_index}.")
            row.append(float(by_round[round_index]))
        matrix.append(row)
    arr = np.asarray(matrix, dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("feature_importance_heatmap contains non-finite importances.")

    figsize = tuple(params.get("figsize_in", (7.2, 7.2)))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=str(params.get("cmap", "viridis")))
    apply_notebook_axes_style(ax)
    ax.set_xticks(range(len(target_rounds)))
    ax.set_xticklabels([str(round_index) for round_index in target_rounds])
    max_yticks = int(params.get("max_yticks", 40))
    step = max(1, int(np.ceil(len(order) / max_yticks)))
    yticks = list(range(0, len(order), step))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(order[index]) for index in yticks])
    ax.set_xlabel("Round")
    ax.set_ylabel("Feature ID")
    ax.set_title(str(params.get("title", "Feature importance heatmap")))
    fig.colorbar(im, ax=ax, label="Importance")
    fig.tight_layout()
    out = context.output_dir / context.filename
    save_notebook_square_figure(fig, out, dpi=context.dpi)
    plt.close(fig)

    if context.save_data:
        export = tidy.rename(columns={"as_of_round": "round", "feature_index": "feature_id"})
        export = export.loc[:, ["round", "feature_id", "importance", "rank", "source_path"]]
        context.save_df(export.sort_values(["round", "feature_id"]))
