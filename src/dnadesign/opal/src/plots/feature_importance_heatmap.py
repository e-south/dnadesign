"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/feature_importance_heatmap.py

Generic attribution-matrix heatmap over OPAL rounds.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..registries.plots import PlotMeta, register_plot
from ._events_util import resolve_outputs_dir
from ._mpl_utils import (
    add_flush_colorbar,
    apply_notebook_axes_style,
    apply_plot_style,
    ensure_mpl_config_dir,
    pretty_label,
    pretty_title,
    save_notebook_square_figure,
    sequential_colormap,
)
from .feature_importance_bars import _discover_round_fi_files, _read_fi_csv, _resolve_order, _select_rounds

DEFAULT_FEATURE_IMPORTANCE_HEATMAP_FIGSIZE: tuple[float, float] = (14.0, 4.4)


@register_plot(
    "feature_importance_heatmap",
    meta=PlotMeta(
        summary="Feature-importance heatmap with stable feature columns and round rows.",
        params={
            "order_policy": "preserve|sort_index|max_importance (default sort_index).",
            "top_n": "Optional positive debugging cap; omit for the full ordinal feature heatmap.",
            "cluster": "Optional clustering flag; defaults false and is currently unsupported when true.",
            "cmap": "Matplotlib colormap (default opal_importance: low values white, high values dark blue).",
            "colorbar_label": "Colorbar label (default Random forest feature importance).",
            "contrast_gamma": "PowerNorm gamma for dense importance contrast; lower emphasizes weak nonzero signal.",
            "max_xticks": "Maximum feature-index tick labels (default 28).",
            "rasterized": "Rasterize the heatmap image artist while keeping axes/text vector-ready (default true).",
        },
        requires=["outputs/rounds/round_<k>/model/feature_importance.csv"],
        notes=[
            "Sorts by ascending feature_index by default so dense ordinal X surfaces keep all feature columns.",
            "Clustering is intentionally off by default.",
        ],
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
    if "sort" in params:
        raise ValueError("feature_importance_heatmap does not accept parameter 'sort'; use 'order_policy'.")
    order_policy = str(params.get("order_policy", "sort_index")).strip().lower()
    if order_policy == "max_importance":
        strict_order_policy = "sort_index"
    else:
        strict_order_policy = order_policy
    top_n = params.get("top_n")
    top_n_int = int(top_n) if top_n is not None else None
    if top_n_int is not None and top_n_int <= 0:
        raise ValueError("top_n must be positive when provided.")
    contrast_gamma = float(params.get("contrast_gamma", 0.55))
    if contrast_gamma <= 0:
        raise ValueError("contrast_gamma must be positive.")

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
    arr = np.asarray(matrix, dtype=float).T
    if not np.isfinite(arr).all():
        raise ValueError("feature_importance_heatmap contains non-finite importances.")

    figsize = tuple(params.get("figsize_in", DEFAULT_FEATURE_IMPORTANCE_HEATMAP_FIGSIZE))
    fig, ax = plt.subplots(figsize=figsize)
    cmap = sequential_colormap(params.get("cmap", "opal_importance"))
    from matplotlib.colors import Normalize, PowerNorm

    vmax = float(np.nanmax(arr)) if arr.size else 0.0
    norm = PowerNorm(gamma=contrast_gamma, vmin=0.0, vmax=vmax) if vmax > 0 else Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    if bool(params.get("rasterized", True)):
        im.set_rasterized(True)
    apply_notebook_axes_style(ax, grid=False, square=False)
    max_xticks = int(params.get("max_xticks", params.get("max_yticks", 28)))
    step = max(1, int(np.ceil(len(order) / max_xticks)))
    xticks = list(range(0, len(order), step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(order[index]) for index in xticks], rotation=0)
    ax.set_yticks(range(len(target_rounds)))
    ax.set_yticklabels([str(round_index) for round_index in target_rounds])
    ax.set_xlabel(f"{pretty_label('feature_index')} ({len(order)} features)")
    ax.set_ylabel("Round")
    ax.set_title(pretty_title(params.get("title", "RF feature importance by round")))
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.24, top=0.80)
    add_flush_colorbar(fig, ax, im, label=pretty_label(params.get("colorbar_label", "rf_feature_importance")), pad=0.03)
    out = context.output_dir / context.filename
    save_notebook_square_figure(fig, out, dpi=context.dpi, tight=False)
    plt.close(fig)

    if context.save_data:
        export = tidy.rename(columns={"as_of_round": "round", "feature_index": "feature_id"})
        export = export.loc[:, ["round", "feature_id", "importance", "rank", "source_path"]]
        context.save_df(export.sort_values(["round", "feature_id"]))
