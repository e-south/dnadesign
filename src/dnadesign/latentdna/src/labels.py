"""User-facing label helpers for latentdna notebook and plot surfaces."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .metrics.definitions import resolve_metric_definition

_REFERENCE_NEIGHBOR_CENSORED_RANK_LABEL = resolve_metric_definition(
    "reference_neighbor_topk_censored_rank_median"
).display_name
_GEOMETRY_DISTANCE_CORRELATION_LABEL = resolve_metric_definition("geometry_distance_correlation").display_name

_DIRECT_LABELS = {
    "appendix_umap_gallery": "UMAP gallery",
    "candidate_decision_frontier": "Representation tradeoff",
    "dataset_overview": "Dataset inventory by cohort dimension",
    "representation_health_summary": "Representation health summary",
    "design_structure_summary": "Design-structure summary",
    "context_robustness_summary": "Context robustness summary",
    "context_pair_summary": "Anchor vs 1 kb anchor-mean context shift",
    "design_centroid_margin_gallery": "Design-family margin scatter",
    "reference_alignment_summary": "Reference collapse summary",
    "reference_core60_strength_umap": "Reference core60 strength UMAP",
    "reference_core60_pca_scree": "Reference core60 PCA scree",
    "representation_scree_diagnostic": "PCA variance decay",
    "intermediate_embedding": "Intermediate block mean",
    "output_layer_mean": "Output-layer mean",
    "anchor_60bp": "Anchor-source insert",
    "merged_anchor_insert_seq_mean": "Mixed-length anchor-source insert",
    "full_context_1kb": "1 kb construct context",
    "native_source_record": "Native source record",
    "core60_tss_upstream": "Core60 TSS-upstream",
    "1 kb context anchor mean": "1 kb context anchor mean",
    "anchor + anchor-mean concat": "Anchor + anchor-mean concat",
    "anchor + 1 kb context concat": "Anchor + 1 kb context concat",
    "anchor_vs_context": "Anchor vs 1 kb context",
    "spacer_length": "Spacer length",
    "emitted_length_bp": "Emitted length (bp)",
    "design_family": "Design family",
    "style": "Design family",
    "background_only": "Background only",
    "ethanol": "Ethanol",
    "cipro": "Ciprofloxacin",
    "ciprofloxacin": "Ciprofloxacin",
    "ethanol_ciprofloxacin": "Ethanol + ciprofloxacin",
    "control": "Control",
    "densegen": "DenseGen",
    "manual_or_wildtype": "Manual/wildtype control",
    "spyp": "spyP",
    "spyP": "spyP",
    "sulap": "sulAp",
    "sulAp": "sulAp",
    "j23105": "J23105",
    "J23105": "J23105",
    "evo2_7b": "Evo 2 7B",
    "evo2_20b": "Evo 2 20B",
    "intermediate_embedding_7b_anchor_60bp": "Anchor-source insert mean",
    "intermediate_embedding_7b_full_context_1kb": "1 kb seq mean",
    "intermediate_embedding_7b_full_context_anchor_mean": "1 kb anchor mean",
    "intermediate_embedding_7b_anchor_plus_full_context_concat": "Anchor + 1 kb concat",
    "intermediate_embedding_7b_anchor_plus_anchor_mean_concat": "Anchor + anchor-mean concat",
    "output_layer_mean_7b_anchor_60bp": "Anchor-source insert output-layer mean",
    "output_layer_mean_7b_full_context_1kb": "1 kb output-layer mean",
    "context_self_cosine": "Context self-cosine",
    "context_shift_l2": "Context-shift L2 distance",
    "context_self_cosine_median": "Median context self-cosine",
    "reference_in_knn_rate": "Reference in-kNN rate",
    "reference_neighbor_topk_censored_rank_median": _REFERENCE_NEIGHBOR_CENSORED_RANK_LABEL,
    "effective_rank": "Effective rank",
    "pc1_variance_fraction": "PC1 variance fraction",
    "pairwise_cosine_distance_median": "Median pairwise cosine distance",
    "pairwise_cosine_distance_iqr": "Pairwise cosine distance IQR",
    "design_family_separation_ratio": "Design-family separation ratio",
    "design_family_balanced_separation_ratio": "Balanced design-family separation ratio",
    "design_regulator_composition_separation_ratio": "Regulator-composition separation ratio",
    "spacer_length_separation_ratio": "Spacer-length separation ratio",
    "cohort_separation_ratio": "Cohort separation ratio",
    "ordinal_axis_spearman": "Ordinal-axis Spearman",
    "ordinal_axis_kendall": "Ordinal-axis Kendall",
    "ordinal_axis_balanced_spearman": "Balanced ordinal-axis Spearman",
    "ordinal_axis_within_group_mean_spearman": "Within-group ordinal-axis Spearman",
    "ordinal_axis_label_permutation_pvalue": "Ordinal-axis permutation p-value",
    "design_family_retention_correlation": "Design-family retention",
    "design_regulator_composition_retention_correlation": "Regulator-composition retention",
    "reference_alignment_ethanol_background_relative": "Ethanol reference alignment",
    "reference_alignment_cipro_background_relative": "Ciprofloxacin reference alignment",
    "synthetic_margin_ethanol_vs_background": "Design-centroid ethanol margin",
    "synthetic_margin_cipro_vs_background": "Design-centroid ciprofloxacin margin",
    "synthetic_margin_dual_vs_background": "Design-centroid dual margin",
    "synthetic_best_stress_margin": "Best stress-family margin",
    "context_margin_delta_ethanol": "Δ ethanol margin",
    "context_margin_delta_cipro": "Δ ciprofloxacin margin",
    "generation_plan": "Generation plan",
    "provenance": "Provenance",
    "neighbor_overlap_fraction": "Neighbor-overlap fraction",
    "geometry_distance_correlation": _GEOMETRY_DISTANCE_CORRELATION_LABEL,
    "pairwise distance correlation": _GEOMETRY_DISTANCE_CORRELATION_LABEL,
    "pairwise distance pearson correlation": _GEOMETRY_DISTANCE_CORRELATION_LABEL,
    "pairwise distance spearman": _GEOMETRY_DISTANCE_CORRELATION_LABEL,
    "x_metric_value": "Evidence metric value",
    "y_metric_value": "Median context self-cosine",
    "candidate_family": "Representation family",
    "candidate_model": "Model size",
    "candidate_scope": "Sequence context",
    "health_status": "Health status",
    "selection_role": "Decision role",
    "comparison_role": "Context comparison",
    "selected": "Chosen",
    "baseline": "Baseline",
    "challenger": "Challenger",
    "orientation": "Orientation",
    "context_anchor_mean": "1 kb anchor mean",
    "whole_sequence_context": "1 kb seq mean",
    "pass": "Pass",
    "warn": "Warn",
    "fail": "Fail",
    "source_class": "Provenance",
    "fraction": "Percent of N",
    "percent": "Percent of N",
    "row_count": "Count",
    "dimension": "Dimension",
    "dimension_label": "Dimension",
    "category": "Category",
    "category_label": "Category",
}

_TOKEN_LABELS = {
    "20b": "20B",
    "7b": "7B",
    "60bp": "60 bp",
    "1kb": "1 kb",
    "kb": "kb",
    "l2": "L2",
    "umap": "UMAP",
    "pca": "PCA",
    "ari": "ARI",
    "nmi": "NMI",
    "cka": "CKA",
    "knn": "kNN",
    "auprc": "AUPRC",
    "id": "ID",
    "evo2": "Evo 2",
    "spyp": "spyP",
    "sulap": "sulAp",
    "j23105": "J23105",
}


def _normalize_key(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _direct_label(value: object) -> str | None:
    text = _normalize_key(value)
    if not text:
        return None
    if text in _DIRECT_LABELS:
        return _DIRECT_LABELS[text]
    lowered = text.lower()
    if lowered in _DIRECT_LABELS:
        return _DIRECT_LABELS[lowered]
    underscored = lowered.replace(" ", "_")
    return _DIRECT_LABELS.get(underscored)


def _split_tokens(value: str) -> list[str]:
    normalized = value.replace("__", " ").replace("_", " ").replace("-", " ")
    return [token for token in re.split(r"\s+", normalized) if token]


def _title_token(token: str) -> str:
    direct = _direct_label(token)
    if direct is not None:
        return direct
    lowered = token.lower()
    if lowered in _TOKEN_LABELS:
        return _TOKEN_LABELS[lowered]
    if any(character.isdigit() for character in token):
        return token.upper() if token.islower() else token
    if token.isupper():
        return token
    return token.capitalize()


def _apply_common_phrase_rewrites(text: str) -> str:
    normalized = text.replace("_", " ")
    normalized = re.sub(r"\bintermediate embedding\b", "Intermediate block mean", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\boutput[- ]layer mean\b", "Output-layer mean", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bfull context anchor mean\b", "1 kb context anchor mean", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\banchor plus anchor mean concat\b",
        "Anchor + anchor-mean concat",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\banchor plus full context concat\b",
        "Anchor + 1 kb context concat",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\banchor 60 ?bp\b", "anchor-source insert mean", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\bfull context 1 ?kb\b",
        "1 kb construct context",
        normalized,
        flags=re.IGNORECASE,
    )
    return normalized


def humanize_label(value: object) -> str:
    direct = _direct_label(value)
    if direct is not None:
        return direct

    text = _normalize_key(value)
    if not text:
        return ""

    normalized = _apply_common_phrase_rewrites(text)
    normalized = re.sub(r"\bevo2 20b\b", "Evo 2 20B", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bevo2 7b\b", "Evo 2 7B", normalized, flags=re.IGNORECASE)
    direct = _direct_label(normalized)
    if direct is not None:
        return direct

    tokens = _split_tokens(normalized)
    if not tokens:
        return ""
    return " ".join(_title_token(token) for token in tokens)


def humanize_column_name(column: str) -> str:
    return humanize_label(column)


def humanize_plot_title(slug: str) -> str:
    return humanize_label(slug)


def humanize_candidate(candidate_key: str | Mapping[str, str]) -> str:
    if isinstance(candidate_key, Mapping):
        model = humanize_label(candidate_key.get("model") or candidate_key.get("candidate_model") or "")
        family = humanize_label(candidate_key.get("family") or candidate_key.get("candidate_family") or "")
        scope = humanize_label(candidate_key.get("scope") or candidate_key.get("candidate_scope") or "")
        pieces = [piece for piece in (model, scope, family) if piece]
        return " · ".join(pieces)

    text = _normalize_key(candidate_key)
    direct = _direct_label(text)
    if direct is not None:
        return direct
    normalized = _apply_common_phrase_rewrites(text)
    normalized = re.sub(r"\b20b\b", "Evo 2 20B", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b7b\b", "Evo 2 7B", normalized, flags=re.IGNORECASE)
    return humanize_label(normalized)
