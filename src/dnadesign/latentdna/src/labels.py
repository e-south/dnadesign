"""User-facing label helpers for latentdna notebook and plot surfaces."""

from __future__ import annotations

import re
from collections.abc import Mapping

_DIRECT_LABELS = {
    "appendix_umap_gallery": "UMAP gallery",
    "dataset_overview": "Dataset inventory by cohort dimension",
    "reference_margin_gallery_wildtype": "Wildtype reference-margin gallery",
    "reference_neighbor_evidence": "Reference-neighborhood evidence",
    "context_shift_reference_plane": "Context-shift margin plane",
    "context_delta_distributions": "Context-shift distributions",
    "context_geometry_summary": "Context stability summary",
    "representation_tradeoff_scatter": "Candidate evidence vs context stability",
    "representation_scree_diagnostic": "PCA variance-decay diagnostic",
    "reference_margin_gallery_synthetic_centroids": "Synthetic proxy margin gallery",
    "intermediate_embedding": "Intermediate block mean",
    "pooled_logits": "Output-layer mean",
    "anchor_60bp": "60 bp anchor",
    "full_context_1kb": "1 kb construct context",
    "sig35_variant": "Sigma-35 variant",
    "sigma35_variant": "Sigma-35 variant",
    "design_family": "Design family",
    "style": "Design family",
    "background_only": "Background only",
    "ethanol": "Ethanol",
    "cipro": "Ciprofloxacin",
    "ciprofloxacin": "Ciprofloxacin",
    "ethanol_ciprofloxacin": "Ethanol + ciprofloxacin",
    "ethanol_plus_cipro": "Ethanol + ciprofloxacin",
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
    "context_self_cosine": "Context self-cosine",
    "context_shift_l2": "Context-shift L2 distance",
    "context_self_cosine_median": "Median context self-cosine",
    "reference_in_knn_rate": "Reference in-kNN rate",
    "reference_neighbor_rank_median": "Median reference-neighbor rank",
    "wildtype_margin_ethanol_auprc": "Wildtype ethanol-margin AUPRC",
    "wildtype_margin_cipro_auprc": "Wildtype ciprofloxacin-margin AUPRC",
    "wildtype_margin_dual_joint_auprc": "Joint wildtype-margin AUPRC",
    "effective_rank": "Effective rank",
    "wildtype_margin_ethanol_vs_control": "Ethanol reference margin",
    "wildtype_margin_cipro_vs_control": "Ciprofloxacin reference margin",
    "synthetic_margin_ethanol_vs_background": "Synthetic ethanol margin",
    "synthetic_margin_cipro_vs_background": "Synthetic ciprofloxacin margin",
    "context_margin_delta_ethanol": "Δ ethanol margin",
    "context_margin_delta_cipro": "Δ ciprofloxacin margin",
    "generation_plan": "Generation plan",
    "provenance": "Provenance",
    "variant b": "Variant b",
    "variant c": "Variant c",
    "variant d": "Variant d",
    "variant e": "Variant e",
    "variant f": "Variant f",
    "neighbor_overlap_fraction": "Neighbor-overlap fraction",
    "geometry_distance_correlation": "Geometry-distance correlation",
    "pairwise distance spearman": "Geometry-distance correlation",
    "x_metric_value": "Evidence metric value",
    "y_metric_value": "Median context self-cosine",
    "candidate_family": "Representation family",
    "candidate_model": "Model size",
    "candidate_scope": "Sequence context",
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


def humanize_label(value: object) -> str:
    direct = _direct_label(value)
    if direct is not None:
        return direct

    text = _normalize_key(value)
    if not text:
        return ""

    normalized = text.replace("_", " ")
    normalized = re.sub(r"\bintermediate embedding\b", "Intermediate block mean", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bpooled logits\b", "Output-layer mean", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\banchor 60 ?bp\b", "60 bp anchor", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\bfull context 1 ?kb\b",
        "1 kb construct context",
        normalized,
        flags=re.IGNORECASE,
    )
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
    normalized = text.replace("_", " ")
    normalized = re.sub(r"\bintermediate embedding\b", "Intermediate block mean", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bpooled logits\b", "Output-layer mean", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\banchor 60 ?bp\b", "60 bp anchor", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\bfull context 1 ?kb\b",
        "1 kb construct context",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\b20b\b", "Evo 2 20B", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b7b\b", "Evo 2 7B", normalized, flags=re.IGNORECASE)
    return humanize_label(normalized)
