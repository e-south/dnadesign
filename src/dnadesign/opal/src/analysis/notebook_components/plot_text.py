from __future__ import annotations

from typing import Any, Mapping

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


def plot_math_description(kind: str) -> str:
    descriptions = {
        "metric_over_rounds": (
            "For each round and cohort, OPAL filters prediction rows, extracts the configured numeric metric, "
            "then computes requested summaries. mean = sum(x) / n; quantile summaries use the requested order "
            "statistic; count = n."
        ),
        "feature_importance_heatmap": (
            "OPAL builds a feature-by-round matrix from model feature_importance.csv artifacts. Each cell is the "
            "model-reported importance for one feature in one round. The default ordering is ascending feature ID "
            "so dense X surfaces can retain the full ordinal feature axis; top_n is an explicit debugging cap only."
        ),
        "vector_summary_heatmap": (
            "For each round, cohort, and vector channel, OPAL aggregates the configured vector prediction field. "
            "The current primitive computes the mean channel value across matching prediction rows, with an "
            "optional reference-vector row for target baselines."
        ),
        "scatter_score_vs_rank": (
            "OPAL plots selected prediction rows by rank and score. Rank is selection order; score is the configured "
            "selection score or objective channel."
        ),
        "percent_high_activity_over_rounds": (
            "For each round, OPAL counts rows where the configured metric >= threshold. "
            "percent_high = 100 * high / total; violin and swarm layers show the underlying metric distribution."
        ),
        "feature_importance_bars": (
            "OPAL reads per-round model feature_importance.csv artifacts. Bars encode model-reported importance, "
            "with ordering controlled by the configured order policy."
        ),
        "fold_change_vs_logic_fidelity": (
            "For SFXI, logic fidelity is clipped 1 - ||v - p||2 / D, where v is the predicted logic vector, "
            "p is the setpoint, and D is the worst-corner distance. The y-axis is the configured effect or score field."
        ),
        "sfxi_logic_fidelity_closeness": (
            "For each observed label, OPAL uses the first four SFXI components as logic values and computes "
            "mean squared error against the setpoint: MSE = mean((v - p)^2). Lower MSE means closer logic behavior."
        ),
        "sfxi_factorial_effects": (
            "With state order 00,10,01,11, A effect = ((v10 + v11) - (v00 + v01)) / 2, "
            "B effect = ((v01 + v11) - (v00 + v10)) / 2, and interaction = ((v11 + v00) - (v10 + v01)) / 2."
        ),
        "sfxi_setpoint_sweep": (
            "OPAL evaluates label vectors against a setpoint library. Logic fidelity is 1 - ||v - p||2 / D, "
            "effect is scaled by a percentile denominator, and score = logic_fidelity^beta * effect_scaled^gamma."
        ),
        "sfxi_support_diagnostics": (
            "For each candidate, OPAL computes Euclidean distance in four-channel logic space to the nearest "
            "labeled point. Larger distances flag extrapolation risk."
        ),
        "sfxi_uncertainty": (
            "For models with ensemble predictions, OPAL computes the standard deviation of objective scores "
            "across estimators after inverse y-ops and SFXI scoring."
        ),
        "sfxi_intensity_scaling": (
            "OPAL recovers linear intensity as max(0, 2^y_star - delta), computes weighted raw effect, "
            "uses a configured percentile denominator, then clips effect_scaled = effect_raw / denom to [0, 1]."
        ),
    }
    return descriptions.get(
        kind,
        "See the plot kind metadata for the exact data shape, required fields, parameters, and failure modes.",
    )


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
            fields.append(f"{key}={value}")
    shape = metadata.get("data_shape")
    if shape not in (None, ""):
        fields.append(f"data_shape={shape}")
    if not fields:
        return f"Plot kind: {str(kind or 'unknown').replace('_', ' ')}."
    return "Encoded fields: " + "; ".join(str(item) for item in fields[:8]) + "."
