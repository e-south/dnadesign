"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/response_magnitude_feasibility.py

Operative plots for Response-Magnitude Feasibility.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..registries.plots import PlotMeta, register_plot
from ._mpl_utils import (
    DEFAULT_SQUARE_FIGSIZE,
    add_flush_colorbar,
    apply_notebook_axes_style,
    apply_plot_style,
    ensure_mpl_config_dir,
    observed_batch_marker_map,
    pretty_batch_label,
    pretty_label,
    scatter_smart,
    wrap_plot_title,
)
from .response_magnitude_feasibility_aliases import (
    resolve_candidate_display_aliases,
    short_candidate_id,
)
from .response_magnitude_feasibility_data import (
    FEASIBILITY_REF,
    OFF_MAGNITUDE_REF,
    ON_MAGNITUDE_REF,
    RESPONSE_REF,
    ResponseMagnitudeFeasibilityPlotData,
    load_response_magnitude_feasibility_plot_data,
)

_FRONTIER_KIND = "response_magnitude_feasibility_frontier"
_DECOMPOSITION_KIND = "response_magnitude_feasibility_constraint_decomposition"


@register_plot(
    _FRONTIER_KIND,
    meta=PlotMeta(
        summary="Predicted candidate constraints with observed labels and selected candidates identified.",
        premise="Predicted RMF components locate selections relative to the configured campaign boundaries.",
        decision_value=(
            "Shows whether predicted selections satisfy all three configured requirements and where tradeoffs remain."
        ),
        rationale="The three requirements remain visible instead of being hidden inside one scalar score.",
        alt_text=(
            "Scatter plot of predicted target-ON/OFF response separation against the target-ON fluorescence floor. "
            "Color encodes the signed target-OFF constraint; all three directions improve upward, and zero marks "
            "each configured boundary. Distinct marker shapes identify observed batches, and filled diamonds "
            "identify selected candidates without hiding their target-OFF values."
        ),
        non_claim_boundary="Predicted feasibility does not establish measured response or fluorescence.",
        tier="decision",
        params={
            "title": "Optional complete-sentence title.",
            "response_label": "Display label for the response-separation axis.",
            "magnitude_label": "Display label for reference-relative magnitude.",
            "off_constraint_label": "Display label for the signed OFF-state color scale.",
            "target_name": "Optional display name for the target set point.",
            "state_labels": "Optional exact state-id to display-label mapping.",
            "figsize_in": "Two-item figure size in inches.",
            "point_size": "Candidate point size (default 10).",
            "point_alpha": "Candidate point alpha (default 0.35).",
            "rasterize_at": "Rasterize points at or above this count (default 10000).",
            "surface_label": "Optional notebook-facing label.",
        },
        requires=[
            "as_of_round",
            "run_id",
            "id",
            "pred__score_channels",
            "view__rank_competition",
            "view__is_selected",
            "labels.parquet",
        ],
        notes=[
            "Reads one response_magnitude_feasibility_v1 run and recomputes feasibility from canonical public math."
        ],
        data_shape="candidate constraint frontier",
        tidy_schema=[
            "id",
            "response_separation",
            "on_magnitude_floor",
            "off_magnitude_ceiling",
            "off_constraint_margin",
            "feasibility_margin",
            "feasible",
            "selected",
            "rank",
            "record_kind",
            "observed_round",
            "batch_id",
            "batch_key",
            "display_label",
        ],
        failure_modes=[
            "ambiguous round or run",
            "objective or selection score reference mismatch",
            "missing or malformed score channels",
            "persisted feasibility disagrees with canonical objective math",
        ],
        objective_family="response_magnitude_feasibility",
        data_layer="predictions_plus_labels",
        round_scope="single_round",
        label_requirement="required",
        notebook_view={
            "adapter": "layered_scatter_v1",
            "record_kind_column": "record_kind",
            "prediction_value": "prediction",
            "observed_value": "observed_label",
            "selection_column": "selected",
            "batch_column": "batch_key",
            "label_column": "display_label",
            "x_column": "response_separation",
            "y_column": "on_magnitude_floor",
            "color_column": "off_constraint_margin",
        },
    ),
)
def render_frontier(context: Any, params: dict) -> None:
    """Render candidate feasibility without collapsing the three constraints."""

    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    apply_plot_style()
    data = load_response_magnitude_feasibility_plot_data(context)
    frame = data.frame
    observed = data.observed_frame
    response_label = str(params.get("response_label", r"Response separation, $d_R$")).strip()
    magnitude_label = str(
        params.get(
            "magnitude_label",
            r"ON fluorescence floor, $f_{\mathrm{ON}}$",
        )
    ).strip()
    off_label = str(
        params.get(
            "off_constraint_label",
            r"OFF fluorescence clearance, $q_{\mathrm{OFF}}$",
        )
    ).strip()
    if not response_label or not magnitude_label or not off_label:
        raise ValueError("RMF plot labels must be non-empty.")
    figsize = _figsize(params.get("figsize_in", DEFAULT_SQUARE_FIGSIZE))
    point_size = _positive_float(params.get("point_size", 10.0), name="point_size")
    point_alpha = _unit_float(params.get("point_alpha", 0.35), name="point_alpha")
    rasterize_at = _nonnegative_int(params.get("rasterize_at", 10_000), name="rasterize_at")
    off_constraint = frame["off_magnitude_constraint_margin"].to_numpy(dtype=float)
    observed_off_constraint = observed["off_magnitude_constraint_margin"].to_numpy(dtype=float)
    visible_off_constraint = np.concatenate((off_constraint, observed_off_constraint))
    color_extent = max(float(np.max(np.abs(visible_off_constraint))), 1.0e-9)
    norm = TwoSlopeNorm(vmin=-color_extent, vcenter=0.0, vmax=color_extent)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    apply_notebook_axes_style(ax, square=True)
    points = scatter_smart(
        ax,
        frame[RESPONSE_REF],
        frame[ON_MAGNITUDE_REF],
        c=off_constraint,
        cmap="RdBu",
        norm=norm,
        s=point_size,
        alpha=point_alpha,
        rasterize_at=rasterize_at,
        label=f"Predicted pool (n={len(frame):,})",
        zorder=2,
    )
    selected = frame["view__is_selected"].to_numpy(dtype=bool)
    if not selected.any():
        raise ValueError("RMF frontier has no selected candidates.")
    observed_batch_ids = sorted(observed["batch_key"].astype(str).unique().tolist())
    observed_markers = observed_batch_marker_map(tuple(observed_batch_ids))
    for batch_key, batch in observed.groupby("batch_key", sort=True):
        batch_key = str(batch_key)
        batch_label = f"Observed · {pretty_batch_label(batch_key)} (n={len(batch)})"
        ax.scatter(
            batch[RESPONSE_REF],
            batch[ON_MAGNITUDE_REF],
            c=batch["off_magnitude_constraint_margin"],
            cmap="RdBu",
            norm=norm,
            edgecolors="#111111",
            marker=observed_markers[batch_key],
            s=max(30.0, point_size * 2.0),
            linewidths=0.9,
            label=batch_label,
            zorder=3,
        )
    ax.scatter(
        frame.loc[selected, RESPONSE_REF],
        frame.loc[selected, ON_MAGNITUDE_REF],
        c=off_constraint[selected],
        cmap="RdBu",
        norm=norm,
        marker="D",
        s=max(36.0, point_size * 2.5),
        edgecolors="#111111",
        linewidths=1.2,
        label=f"Selected (n={int(selected.sum())})",
        zorder=4,
    )
    ax.axvline(data.calibration["response_separation_min"], color="#555555", linestyle="--", linewidth=1.1, zorder=1)
    ax.axhline(data.calibration["on_magnitude_min"], color="#555555", linestyle="--", linewidth=1.1, zorder=1)
    ax.set_xlabel(response_label, fontsize=9.5, labelpad=7)
    ax.set_ylabel(magnitude_label, fontsize=9.5, labelpad=7)
    title = _selection_view_title(
        params.get("title", "RMF candidate constraint landscape"),
        context=context,
    )
    ax.tick_params(axis="both", labelsize=8.5)
    ax.set_title(
        f"{wrap_plot_title(title, width=62)}\n{_target_context(data, params)}",
        loc="left",
        fontweight="semibold",
        fontsize=10.5,
        pad=8,
        linespacing=1.35,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=7.6,
        ncol=3,
        frameon=False,
        handletextpad=0.45,
        columnspacing=0.8,
    )
    colorbar = add_flush_colorbar(
        fig,
        ax,
        points,
        label=f"{off_label}\n0 = boundary",
        pad=0.065,
        ticklabelsize=8.5,
    )
    colorbar.ax.yaxis.label.set_size(9)
    records_path = context.data_paths.get("records")
    alias_ids = list(
        dict.fromkeys([*frame.loc[selected, "id"].astype(str).tolist(), *observed["id"].astype(str).tolist()])
    )
    aliases = (
        resolve_candidate_display_aliases(records_path, alias_ids)
        if records_path is not None
        else {candidate_id: short_candidate_id(candidate_id) for candidate_id in alias_ids}
    )
    context.artifact_metadata["notebook_view"] = {
        "title": title,
        "context": _target_context(data, params),
        "x_label": response_label,
        "y_label": magnitude_label,
        "color_label": off_label,
        "x_boundary": float(data.calibration["response_separation_min"]),
        "y_boundary": float(data.calibration["on_magnitude_min"]),
        "color_extent": color_extent,
        "x_limits": [float(value) for value in ax.get_xlim()],
        "y_limits": [float(value) for value in ax.get_ylim()],
    }
    _save(context, fig)
    if context.save_data:
        context.save_df(_frontier_tidy(frame, observed, aliases=aliases))
    plt.close(fig)


@register_plot(
    _DECOMPOSITION_KIND,
    meta=PlotMeta(
        summary="Predicted requirement margins and maximin score for selected candidates.",
        premise="The weakest standardized requirement determines each selected candidate's feasibility score.",
        decision_value="Identifies which requirement limits every selected candidate before experimental handoff.",
        rationale="A component heatmap makes the non-compensatory maximin rule directly inspectable.",
        alt_text=(
            "Heatmap with selected candidates as rows and standardized response, target-ON magnitude, target-OFF, "
            "and feasibility margins as columns. Zero is the configured boundary; the feasibility column equals "
            "the smallest of the first three signed requirements. The campaign ON/OFF set point is printed above "
            "the panel, and an outline marks the component that limits each selected candidate."
        ),
        non_claim_boundary="Constraint margins are model predictions until the candidates are measured.",
        tier="decision",
        params={
            "title": "Optional complete-sentence title.",
            "target_name": "Optional display name for the target set point.",
            "state_labels": "Optional exact state-id to display-label mapping.",
            "figsize_in": "Optional two-item figure size in inches.",
            "max_selected": "Maximum selected rows permitted in one readable heatmap (default 24).",
            "candidate_label_mode": "Candidate row labels: short_id (default) or alias.",
        },
        requires=[
            "as_of_round",
            "run_id",
            "id",
            "pred__score_channels",
            "view__rank_competition",
            "view__is_selected",
        ],
        notes=["The fourth column is verified against the minimum of the three canonical signed constraints."],
        data_shape="selected candidate by constraint matrix",
        tidy_schema=["id", "rank", "constraint", "signed_margin", "limiting", "feasible"],
        failure_modes=[
            "ambiguous round or run",
            "no selected candidates",
            "selected row count exceeds the declared review limit",
            "persisted feasibility disagrees with canonical objective math",
        ],
        objective_family="response_magnitude_feasibility",
        data_layer="predictions_objective",
        round_scope="single_round",
    ),
)
def render_constraint_decomposition(context: Any, params: dict) -> None:
    """Render the maximin score as an inspectable selected-candidate matrix."""

    ensure_mpl_config_dir(workdir=getattr(context.workspace, "workdir", None))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    apply_plot_style()
    data = load_response_magnitude_feasibility_plot_data(context)
    selected = data.frame.loc[data.frame["view__is_selected"]].copy()
    if selected.empty:
        raise ValueError("RMF decomposition has no selected candidates.")
    max_selected = _nonnegative_int(params.get("max_selected", 24), name="max_selected")
    if max_selected == 0 or len(selected) > max_selected:
        raise ValueError(f"Selected-candidate heatmap has {len(selected)} rows; max_selected={max_selected}.")

    columns = (
        "response_constraint_margin",
        "on_magnitude_constraint_margin",
        "off_magnitude_constraint_margin",
        FEASIBILITY_REF,
    )
    matrix = selected.loc[:, columns].to_numpy(dtype=float)
    extent = max(float(np.max(np.abs(matrix))), 1.0e-9)
    norm = TwoSlopeNorm(vmin=-extent, vcenter=0.0, vmax=extent)
    default_height = max(4.8, min(10.0, 2.5 + 0.55 * len(selected)))
    figsize = _figsize(params.get("figsize_in", (7.4, default_height)))
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    image = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="equal")
    apply_notebook_axes_style(ax, grid=False, square=False)
    ax.set_xticks(
        np.arange(4),
        [
            r"$q_R$",
            r"$q_{\mathrm{ON}}$",
            r"$q_{\mathrm{OFF}}$",
            r"$S_{\mathrm{RMF}}$",
        ],
    )
    candidate_label_mode = str(params.get("candidate_label_mode", "short_id")).strip().lower()
    if candidate_label_mode not in {"short_id", "alias"}:
        raise ValueError("candidate_label_mode must be 'short_id' or 'alias'.")
    if candidate_label_mode == "alias":
        records_path = context.data_paths.get("records")
        if records_path is None:
            raise ValueError("candidate_label_mode='alias' requires the built-in records.parquet input.")
        row_labels = resolve_candidate_display_aliases(records_path, selected["id"].astype(str).tolist())
    else:
        row_labels = {str(value): short_candidate_id(str(value)) for value in selected["id"]}
    ax.set_yticks(
        np.arange(len(selected)),
        [f"#{int(row.view__rank_competition)}  {row_labels[str(row.id)]}" for row in selected.itertuples(index=False)],
    )
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel(
        r"$S_{\mathrm{RMF}}=\min(q_R,q_{\mathrm{ON}},q_{\mathrm{OFF}})$"
        "\n0 marks each configured boundary",
        fontsize=9,
        labelpad=7,
    )
    ax.set_ylabel("Competition rank · candidate", fontsize=9.5, labelpad=7)
    title = _selection_view_title(
        params.get("title", "Predicted RMF margins for selected candidates"),
        context=context,
    )
    ax.set_title(
        f"{wrap_plot_title(title, width=62)}\n{_target_context(data, params)}",
        loc="left",
        fontweight="semibold",
        fontsize=10.5,
        pad=8,
        linespacing=1.35,
    )
    for row in range(matrix.shape[0]):
        limiting_column = int(np.argmin(matrix[row, :3]))
        ax.add_patch(
            Rectangle(
                (limiting_column - 0.5, row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#111111",
                linewidth=2.0,
                zorder=4,
            )
        )
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(value) > 0.55 * extent else "#111111",
            )
    colorbar = add_flush_colorbar(
        fig,
        ax,
        image,
        label="Standardized margin\n0 = boundary; outline = limiting",
        pad=0.06,
        ticklabelsize=8.5,
    )
    colorbar.ax.yaxis.label.set_size(9)
    _save(context, fig)
    if context.save_data:
        context.save_df(_decomposition_tidy(selected))
    plt.close(fig)


def _frontier_tidy(
    frame: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    aliases: Mapping[str, str],
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "id": frame["id"].astype(str),
            "record_kind": "prediction",
            "response_separation": frame[RESPONSE_REF].astype(float),
            "on_magnitude_floor": frame[ON_MAGNITUDE_REF].astype(float),
            "off_magnitude_ceiling": frame[OFF_MAGNITUDE_REF].astype(float),
            "off_constraint_margin": frame["off_magnitude_constraint_margin"].astype(float),
            "feasibility_margin": frame[FEASIBILITY_REF].astype(float),
            "feasible": frame["feasible"].astype(bool),
            "selected": frame["view__is_selected"].astype(bool),
            "rank": frame["view__rank_competition"].astype(int),
            "observed_round": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "batch_id": pd.Series([pd.NA] * len(frame), dtype="string"),
            "batch_key": pd.Series([pd.NA] * len(frame), dtype="string"),
            "display_label": frame["id"].astype(str).map(aliases).astype("string"),
        }
    )
    labels = pd.DataFrame(
        {
            "id": observed["id"].astype(str),
            "record_kind": "observed_label",
            "response_separation": observed[RESPONSE_REF].astype(float),
            "on_magnitude_floor": observed[ON_MAGNITUDE_REF].astype(float),
            "off_magnitude_ceiling": observed[OFF_MAGNITUDE_REF].astype(float),
            "off_constraint_margin": observed["off_magnitude_constraint_margin"].astype(float),
            "feasibility_margin": observed[FEASIBILITY_REF].astype(float),
            "feasible": observed["feasible"].astype(bool),
            "selected": False,
            "rank": pd.Series([pd.NA] * len(observed), dtype="Int64"),
            "observed_round": observed["observed_round"].astype("Int64"),
            "batch_id": observed["batch_id"].astype("string"),
            "batch_key": observed["batch_key"].astype("string"),
            "display_label": _observed_display_labels(observed, fallbacks=aliases),
        }
    )
    return pd.concat([predictions, labels], ignore_index=True)


def _observed_display_labels(observed: pd.DataFrame, *, fallbacks: Mapping[str, str]) -> pd.Series:
    labels = observed["display_label"].astype("string")
    missing = labels.isna() | labels.str.strip().eq("")
    if missing.any():
        labels = labels.where(~missing, observed["id"].astype(str).map(fallbacks).astype("string"))
    candidate_labels = pd.DataFrame({"id": observed["id"].astype(str), "label": labels}).drop_duplicates()
    ambiguous = set(candidate_labels.loc[candidate_labels["label"].duplicated(keep=False), "label"].astype(str))
    if ambiguous:
        labels = pd.Series(
            [
                f"{label} · {candidate_id[:6]}" if str(label) in ambiguous else str(label)
                for candidate_id, label in zip(observed["id"].astype(str), labels, strict=True)
            ],
            index=observed.index,
            dtype="string",
        )
    return labels.astype("string")


def _decomposition_tidy(selected: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "response_constraint_margin": "response",
        "on_magnitude_constraint_margin": "target_on_magnitude",
        "off_magnitude_constraint_margin": "target_off_control",
        FEASIBILITY_REF: "feasibility_minimum",
    }
    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        values = {column: float(row[column]) for column in labels}
        limiting_value = min(values[column] for column in tuple(labels)[:3])
        for column, label in labels.items():
            value = values[column]
            rows.append(
                {
                    "id": str(row["id"]),
                    "rank": int(row["view__rank_competition"]),
                    "constraint": label,
                    "signed_margin": value,
                    "limiting": bool(column != FEASIBILITY_REF and np.isclose(value, limiting_value)),
                    "feasible": bool(row["feasible"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _save(context: Any, fig: Any) -> None:
    context.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        context.output_dir / context.filename,
        dpi=context.dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def _target_context(data: ResponseMagnitudeFeasibilityPlotData, params: Mapping[str, Any]) -> str:
    state_ids = tuple(str(value) for value in data.state_ids)
    target_mask = tuple(int(value) for value in data.target_mask)
    if len(state_ids) != len(target_mask):
        raise ValueError("RMF plot state_ids and target_mask must have equal length.")

    configured_labels = params.get("state_labels")
    if configured_labels is None:
        labels = dict(zip(state_ids, state_ids, strict=True))
    else:
        if not isinstance(configured_labels, Mapping):
            raise ValueError("state_labels must be a mapping from exact state IDs to display labels.")
        labels = {str(key).strip(): str(value).strip() for key, value in configured_labels.items()}
        missing = sorted(set(state_ids) - set(labels))
        extra = sorted(set(labels) - set(state_ids))
        if missing or extra:
            raise ValueError(f"state_labels must match state_ids exactly; missing={missing}, extra={extra}.")
        if any(not labels[state_id] for state_id in state_ids):
            raise ValueError("state_labels values must be non-empty.")

    on_labels = [labels[state_id] for state_id, enabled in zip(state_ids, target_mask, strict=True) if enabled]
    off_labels = [labels[state_id] for state_id, enabled in zip(state_ids, target_mask, strict=True) if not enabled]
    if not on_labels or not off_labels:
        raise ValueError("RMF plot target_mask must contain at least one ON and one OFF state.")
    target_name = str(params.get("target_name") or "").strip()
    prefix = f"{target_name} target" if target_name else "Target"
    return f"{prefix} ON: {', '.join(on_labels)} | OFF: {', '.join(off_labels)}"


def _selection_view_title(value: object, *, context: Any) -> str:
    title = str(value or "").strip()
    if not title:
        raise ValueError("RMF plot title must be non-empty.")
    view_id = str(getattr(context, "selection_view_id", "") or "").strip()
    if not view_id:
        return title
    view_label = "AND" if view_id.lower() == "and" else pretty_label(view_id)
    return f"{title} · {view_label} view"


def _figsize(value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("figsize_in must contain exactly two values.")
    size = tuple(float(item) for item in value)
    if not all(np.isfinite(size)) or min(size) <= 0.0:
        raise ValueError("figsize_in values must be finite and positive.")
    return size


def _positive_float(value: object, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return parsed


def _unit_float(value: object, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise ValueError(f"{name} must be in (0, 1].")
    return parsed


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    parsed = int(value)
    if float(value) != parsed or parsed < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return parsed


__all__ = ["render_constraint_decomposition", "render_frontier"]
