from __future__ import annotations

from typing import Any, Mapping

from ...plots._mpl_utils import plot_metric_expression, plot_metric_label, pretty_label
from ._support import mapping, sequence


def plot_alt_text(
    *,
    title: str,
    kind: Any,
    summary: str,
    params: Any,
    metadata: Any,
    rounds: Any,
    run_id: Any,
    freshness: Any,
    warning_count: int,
) -> str:
    kind_text = str(kind or "plot").replace("_", " ")
    summary_text = str(summary or "").strip()
    scope = rounds_text(rounds)
    run_text = "all runs" if run_id in (None, "") else f"run {run_id}"
    field_text = plot_field_text(kind=str(kind or ""), params=mapping(params), metadata=mapping(metadata))
    quality = f"freshness {freshness or 'unknown'}"
    if int(warning_count) > 0:
        quality += f", {int(warning_count)} warnings"
    if summary_text:
        return f"{title}. {summary_text} {field_text} Scope: {scope}, {run_text}; {quality}."
    return f"{title}. OPAL {kind_text} visual. {field_text} Scope: {scope}, {run_text}; {quality}."


def rounds_text(value: Any) -> str:
    if value in (None, ""):
        return "the selected round"
    if value == "all":
        return "all rounds"
    items = sequence(value)
    if len(items) == 1:
        return f"round {items[0]}"
    return "rounds " + ", ".join(str(item) for item in items)


def plot_math_description(kind: str, params: Mapping[str, Any] | None = None) -> str:
    params_map = mapping(params)
    expression = plot_metric_expression(params_map)
    descriptions = {
        "metric_over_rounds": (
            "For each round and cohort, OPAL filters prediction rows, extracts the configured numeric metric, "
            "then computes requested summaries. mean = sum(x) / n; quantile summaries use the requested order "
            "statistic; count = n. When band=iqr, q25 and q75 are encoded as a shaded interquartile band rather "
            "than separate summary lines. Cohort n is available from the tidy count rows when count is requested."
        ),
        "feature_importance_heatmap": (
            "OPAL builds a feature-by-round matrix from random-forest feature_importance.csv artifacts. Each cell "
            "is the model-reported RF feature importance for one feature in one round. The default ordering is "
            "ascending feature ID on the x-axis so dense X surfaces can retain the full ordinal feature axis; "
            "top_n is an explicit debugging cap only. Values are model attributions, not label counts or replicate "
            "measurements."
        ),
        "vector_summary_heatmap": (
            "For each round, cohort, and vector channel, OPAL aggregates the configured vector prediction field. "
            "The current primitive computes the mean channel value across matching prediction rows, can include a "
            "reference-vector row for target baselines, and can show MSE = mean((mean_vector - reference)^2) by round. "
            "The tidy n column records the row count behind each round/cohort/channel mean."
        ),
        "scatter_score_vs_rank": (
            "OPAL plots prediction rows by rank and the configured score field. sequential rank is recomputed within "
            "each round after sorting score descending; competition rank uses the persisted sel__rank_competition "
            "field so selected-order ties remain visible. Selected count is represented by rows with "
            "sel__is_selected=true under the chosen round/run scope."
        ),
        "percent_high_activity_over_rounds": (
            "OPAL first chooses one fixed cutoff for the plotted metric. With threshold_quantile=0.9, P90 means "
            "the 90th percentile of all finite plotted metric values under the current round/run scope, not a "
            "separate per-round cutoff. For each round, high = count(metric >= cutoff), total = count(metric), "
            "and percent_high = 100 * high / total. The line therefore shows enrichment into the same top-tail "
            "score band over active-learning rounds."
        ),
        "feature_importance_bars": (
            "OPAL reads per-round random-forest feature_importance.csv artifacts. Bars encode model-reported RF "
            "feature importance, with ordering controlled by the configured order policy and round encoded by a "
            "monotonic progression palette. The bars are per-model attribution values rather than replicate "
            "summaries."
        ),
        "fold_change_vs_logic_fidelity": (
            "For SFXI, logic fidelity is clipped 1 - ||v - p||2 / D, where v is the predicted logic vector, "
            "p is the setpoint, and D is the worst-corner distance. The y-axis is the configured effect or score "
            "field. Optional reference overlays are validated records-table points and are counted separately from "
            "predictions."
        ),
        "sfxi_logic_fidelity_closeness": (
            "For each observed label, OPAL uses the first four SFXI components as logic values and computes "
            "mean squared error against the setpoint: MSE = mean((v - p)^2). Lower MSE means closer logic behavior. "
            "Replicate semantics are per observed label row unless the label source has already collapsed replicates."
        ),
        "sfxi_factorial_effects": (
            "With state order 00,10,01,11, A effect = ((v10 + v11) - (v00 + v01)) / 2, "
            "B effect = ((v01 + v11) - (v00 + v10)) / 2, and interaction = ((v11 + v00) - (v10 + v01)) / 2. "
            "Optional label overlays use observed label rows for the selected round."
        ),
        "sfxi_setpoint_sweep": (
            "OPAL evaluates current-round label vectors against a setpoint library. Logic fidelity is "
            "1 - ||v - p||2 / D, effect is scaled by a percentile denominator, and score is "
            "logic_fidelity^beta * effect_scaled^gamma unless the setpoint disables intensity. Each plotted value "
            "summarizes the current-round label rows that meet the configured min_n gate."
        ),
        "sfxi_support_diagnostics": (
            "For each candidate, OPAL computes Euclidean distance in four-channel logic space to the nearest "
            "labeled point available as of the selected round. Larger distances flag extrapolation risk; label "
            "support comes from observed labels, not from prediction rows."
        ),
        "sfxi_uncertainty": _sfxi_uncertainty_math_description(params_map),
        "sfxi_intensity_scaling": (
            "OPAL recovers linear intensity as max(0, 2^y_star - delta), computes weighted raw effect, "
            "uses a configured percentile denominator, then clips effect_scaled = effect_raw / denom to [0, 1]. "
            "The denominator is label-derived for the current round and requires the configured min_n labels."
        ),
    }
    description = descriptions.get(
        kind,
        "See the plot kind metadata for the exact data shape, required fields, parameters, and failure modes.",
    )
    if expression:
        description = f"{description} Configured score/loss expression: {expression}."
    return description


def compact_params(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return "not recorded"
    return "; ".join(f"{key}={item}" for key, item in value.items())


def capability_text(value: Any) -> str:
    capability = mapping(value)
    if not capability:
        return "not recorded"
    return "; ".join(
        [
            f"objective_family={capability.get('objective_family', 'unknown')}",
            f"data_layer={capability.get('data_layer', 'unspecified')}",
            f"round_scope={capability.get('round_scope', 'unspecified')}",
            f"label_requirement={capability.get('label_requirement', 'none')}",
            f"requires_model_artifact={bool(capability.get('requires_model_artifact'))}",
            f"tidy_available={bool(capability.get('tidy_available'))}",
        ]
    )


def plot_field_text(*, kind: str, params: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    fields: list[str] = []
    kind_key = str(kind or "")
    encoding = plot_encoding_text(kind=kind_key, params=params)
    if encoding:
        fields.append(encoding)
    label_field = params.get("metric") or params.get("score_field") or params.get("y_axis")
    has_explicit_label = any(
        params.get(key) not in (None, "") for key in ("metric_label", "score_label", "y_label", "axis_label")
    )
    if label_field not in (None, "") or has_explicit_label:
        metric_label = plot_metric_label(params, label_field or "pred__score_selected")
        if metric_label:
            fields.append(f"metric_label={metric_label}")
    metric_expression = plot_metric_expression(params)
    if metric_expression:
        fields.append(f"metric_expression={metric_expression}")
    for key in (
        "metric",
        "summary",
        "summaries",
        "cohort",
        "score_field",
        "rank_mode",
        "threshold",
        "y_axis",
        "hue",
        "hue_field",
        "size_by",
        "vector_field",
        "aggregation",
    ):
        value = params.get(key)
        if value not in (None, "", []):
            fields.append(f"{key}={_pretty_param_value(value)}")
    shape = metadata.get("data_shape")
    if shape not in (None, ""):
        fields.append(f"data_shape={shape}")
    if not fields:
        return f"Plot kind: {str(kind or 'unknown').replace('_', ' ')}."
    return "Encoded fields: " + "; ".join(str(item) for item in fields[:8]) + "."


def plot_encoding_text(*, kind: str, params: Mapping[str, Any]) -> str:
    """Describe the primary visual channels for notebook alt text."""

    metric = str(params.get("metric") or params.get("score_field") or "pred__score_selected")
    y_axis = str(params.get("y_axis") or params.get("y") or "score")
    hue = str(params.get("hue") or params.get("hue_field") or "none")
    size_by = str(params.get("size_by") or "none")
    vector_field = str(params.get("vector_field") or "pred__y_hat_model")
    cohort = params.get("cohort", "selected")
    if kind == "metric_over_rounds":
        band = str(params.get("band") or "none")
        return (
            f"x=round; y={plot_metric_label(params, metric)} summary; "
            f"series=cohort and summary; band={pretty_label(band)}"
        )
    if kind == "scatter_score_vs_rank":
        rank_mode = str(params.get("rank_mode") or "sequential")
        score_field = str(params.get("score_field") or "pred__score_selected")
        return f"x={rank_mode} rank; y={plot_metric_label(params, score_field)}; selected rows outlined when present"
    if kind == "percent_high_activity_over_rounds":
        threshold = params.get("threshold", None)
        if threshold is None and params.get("threshold_quantile") is not None:
            threshold = f"fixed P{100 * float(params.get('threshold_quantile')):g} cutoff of the plotted metric"
        if threshold is None:
            threshold = 0.8
        mode = str(params.get("mode") or "both")
        return (
            f"x=round; y=percent of rows where {plot_metric_label(params, metric)} is at/above {threshold}; mode={mode}"
        )
    if kind == "feature_importance_heatmap":
        return "x=feature index; y=round; color=model-reported RF feature importance"
    if kind == "feature_importance_bars":
        return "x=feature index; y=model-reported importance; color=round"
    if kind == "vector_summary_heatmap":
        return (
            "x=vector channel; y=reference plus round/cohort rows; "
            f"color=mean {_pretty_with_raw(vector_field)}; cohort={pretty_label(cohort)}"
        )
    if kind == "fold_change_vs_logic_fidelity":
        return (
            f"x=logic fidelity; y={_pretty_with_raw(y_axis)}; hue={_pretty_with_raw(hue)}; "
            f"size={_pretty_with_raw(size_by)}; selected rows use marker styling"
        )
    if kind == "sfxi_logic_fidelity_closeness":
        return (
            "left panel x=logic component and y=target/observed round; "
            "right panel x=observed round and y=MSE to setpoint"
        )
    if kind == "sfxi_factorial_effects":
        return f"x=A effect; y=B effect; color=AB interaction; size={_pretty_with_raw(size_by)}"
    if kind == "sfxi_setpoint_sweep":
        return "x=setpoint; y=metric; color=median label-derived metric value"
    if kind == "sfxi_support_diagnostics":
        return f"x=distance to nearest labeled logic vector; y={_pretty_with_raw(y_axis)}; hue={_pretty_with_raw(hue)}"
    if kind == "sfxi_uncertainty":
        sample_n = params.get("sample_n") or params.get("sample") or params.get("n")
        sample_clause = f"; deterministic sample up to n={sample_n}" if sample_n is not None else ""
        return f"x=ensemble score uncertainty; y={_pretty_with_raw(y_axis)}; hue={_pretty_with_raw(hue)}{sample_clause}"
    if kind == "sfxi_intensity_scaling":
        return "panels=denominator by setpoint, clipping fraction by setpoint, and E_raw distribution"
    return ""


def _pretty_param_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_pretty_param_value(item) for item in value) + "]"
    return _pretty_with_raw(value)


def _pretty_with_raw(value: Any) -> str:
    text = str(value or "")
    if text in {"", "none", "None"}:
        return "none"
    return pretty_label(text)


def _sfxi_uncertainty_math_description(params: Mapping[str, Any]) -> str:
    sample_n = params.get("sample_n") or params.get("sample") or params.get("n")
    sample_clause = (
        f" on a deterministic sample of up to {sample_n} candidates"
        if sample_n is not None
        else " on every plotted candidate"
    )
    return (
        "For models with ensemble predictions, OPAL recomputes SFXI score per estimator after inverse y-ops, "
        "then reports the population standard deviation (ddof=0) across estimator scores using the run's "
        f"fixed setpoint and denominator{sample_clause}."
    )
