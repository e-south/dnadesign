"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/metrics/definitions.py

Canonical comparison-metric definitions for LatentDNA artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from ..contracts.errors import ContractViolationError
from ..io.hashing import sha256_payload


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    metric_id: str
    display_name: str
    mathematical_definition: str
    metric_family: str
    evidence_tier: str
    unit: str
    direction: str
    aggregation_level: str
    task_id: str | None = None
    definition_version: str = "2026-04-17"

    @property
    def higher_is_better(self) -> bool | None:
        if self.direction == "higher_is_better":
            return True
        if self.direction == "lower_is_better":
            return False
        return None


def _metric(
    metric_id: str,
    display_name: str,
    mathematical_definition: str,
    *,
    metric_family: str,
    evidence_tier: str,
    unit: str,
    direction: str,
    aggregation_level: str,
    task_id: str | None = None,
) -> MetricDefinition:
    return MetricDefinition(
        metric_id=metric_id,
        display_name=display_name,
        mathematical_definition=mathematical_definition,
        metric_family=metric_family,
        evidence_tier=evidence_tier,
        unit=unit,
        direction=direction,
        aggregation_level=aggregation_level,
        task_id=task_id,
    )


def _model_or_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def metric_definitions_from_config(config: object | None) -> dict[str, MetricDefinition]:
    configured = getattr(config, "metric_definitions", {}) if config is not None else {}
    definitions: dict[str, MetricDefinition] = {}
    for metric_id, raw_definition in dict(configured or {}).items():
        payload = _model_or_mapping(raw_definition)
        definitions[str(metric_id)] = MetricDefinition(
            metric_id=str(metric_id),
            display_name=str(payload["display_name"]),
            mathematical_definition=str(payload["mathematical_definition"]),
            metric_family=str(payload["metric_family"]),
            evidence_tier=str(payload["evidence_tier"]),
            unit=str(payload["unit"]),
            direction=str(payload["direction"]),
            aggregation_level=str(payload["aggregation_level"]),
            task_id=str(payload["task_id"]) if payload.get("task_id") is not None else None,
            definition_version=str(payload.get("definition_version") or "workspace_config.v1"),
        )
    return definitions


_METRICS = [
    _metric(
        "wildtype_margin_ethanol_auroc",
        "Wildtype ethanol AUROC",
        "AUROC over ethanol_present using the wildtype ethanol-vs-control margin.",
        metric_family="biology_signal",
        evidence_tier="primary",
        unit="auroc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="ethanol_present",
    ),
    _metric(
        "wildtype_margin_ethanol_auprc",
        "Wildtype ethanol AUPRC",
        "AUPRC over ethanol_present using the wildtype ethanol-vs-control margin.",
        metric_family="biology_signal",
        evidence_tier="primary",
        unit="auprc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="ethanol_present",
    ),
    _metric(
        "wildtype_margin_cipro_auroc",
        "Wildtype cipro AUROC",
        "AUROC over cipro_present using the wildtype cipro-vs-control margin.",
        metric_family="biology_signal",
        evidence_tier="primary",
        unit="auroc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="cipro_present",
    ),
    _metric(
        "wildtype_margin_cipro_auprc",
        "Wildtype cipro AUPRC",
        "AUPRC over cipro_present using the wildtype cipro-vs-control margin.",
        metric_family="biology_signal",
        evidence_tier="primary",
        unit="auprc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="cipro_present",
    ),
    _metric(
        "wildtype_margin_dual_joint_auroc",
        "Wildtype dual AUROC",
        "AUROC over dual_only using min(wildtype ethanol margin, wildtype cipro margin).",
        metric_family="biology_signal",
        evidence_tier="primary",
        unit="auroc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="dual_only",
    ),
    _metric(
        "wildtype_margin_dual_joint_auprc",
        "Wildtype dual AUPRC",
        "AUPRC over dual_only using min(wildtype ethanol margin, wildtype cipro margin).",
        metric_family="biology_signal",
        evidence_tier="primary",
        unit="auprc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="dual_only",
    ),
    _metric(
        "synthetic_margin_ethanol_auroc",
        "Synthetic ethanol AUROC",
        "AUROC over ethanol_present using the synthetic ethanol-vs-background margin.",
        metric_family="biology_signal",
        evidence_tier="appendix",
        unit="auroc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="ethanol_present",
    ),
    _metric(
        "synthetic_margin_ethanol_auprc",
        "Synthetic ethanol AUPRC",
        "AUPRC over ethanol_present using the synthetic ethanol-vs-background margin.",
        metric_family="biology_signal",
        evidence_tier="appendix",
        unit="auprc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="ethanol_present",
    ),
    _metric(
        "synthetic_margin_cipro_auroc",
        "Synthetic cipro AUROC",
        "AUROC over cipro_present using the synthetic cipro-vs-background margin.",
        metric_family="biology_signal",
        evidence_tier="appendix",
        unit="auroc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="cipro_present",
    ),
    _metric(
        "synthetic_margin_cipro_auprc",
        "Synthetic cipro AUPRC",
        "AUPRC over cipro_present using the synthetic cipro-vs-background margin.",
        metric_family="biology_signal",
        evidence_tier="appendix",
        unit="auprc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="cipro_present",
    ),
    _metric(
        "synthetic_margin_dual_joint_auroc",
        "Synthetic dual AUROC",
        "AUROC over dual_only using min(synthetic ethanol margin, synthetic cipro margin).",
        metric_family="biology_signal",
        evidence_tier="appendix",
        unit="auroc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="dual_only",
    ),
    _metric(
        "synthetic_margin_dual_joint_auprc",
        "Synthetic dual AUPRC",
        "AUPRC over dual_only using min(synthetic ethanol margin, synthetic cipro margin).",
        metric_family="biology_signal",
        evidence_tier="appendix",
        unit="auprc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="dual_only",
    ),
    _metric(
        "scalar_ethanol_auroc",
        "Scalar ethanol AUROC",
        "AUROC over ethanol_present using a scalar evidence surface.",
        metric_family="biology_signal",
        evidence_tier="secondary",
        unit="auroc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="ethanol_present",
    ),
    _metric(
        "scalar_ethanol_auprc",
        "Scalar ethanol AUPRC",
        "AUPRC over ethanol_present using a scalar evidence surface.",
        metric_family="biology_signal",
        evidence_tier="secondary",
        unit="auprc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="ethanol_present",
    ),
    _metric(
        "scalar_cipro_auroc",
        "Scalar cipro AUROC",
        "AUROC over cipro_present using a scalar evidence surface.",
        metric_family="biology_signal",
        evidence_tier="secondary",
        unit="auroc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="cipro_present",
    ),
    _metric(
        "scalar_cipro_auprc",
        "Scalar cipro AUPRC",
        "AUPRC over cipro_present using a scalar evidence surface.",
        metric_family="biology_signal",
        evidence_tier="secondary",
        unit="auprc",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
        task_id="cipro_present",
    ),
    _metric(
        "knn_design_family_enrichment_delta",
        "kNN design-family enrichment",
        "Mean within-neighborhood enrichment delta for design_family labels.",
        metric_family="biology_signal",
        evidence_tier="secondary",
        unit="delta_fraction",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "reference_in_knn_rate",
        "Reference in-kNN rate",
        "Fraction of rows whose top-k neighborhood contains the intended reference anchor.",
        metric_family="reference_neighborhood",
        evidence_tier="primary",
        unit="fraction",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "reference_neighbor_topk_censored_rank_median",
        "Reference neighbor top-k-censored rank median",
        "Median observed rank of the intended reference anchor inside the retained top-k neighborhood, "
        "with absent references reported as k+1.",
        metric_family="reference_neighborhood",
        evidence_tier="primary",
        unit="rank",
        direction="lower_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "context_self_cosine_median",
        "Context self cosine",
        "Median cosine similarity between anchor and context embeddings for aligned pairs.",
        metric_family="context_stability",
        evidence_tier="primary",
        unit="cosine",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "context_shift_l2_median",
        "Context shift L2",
        "Median L2 shift between anchor and context embeddings for aligned pairs.",
        metric_family="context_stability",
        evidence_tier="primary",
        unit="l2",
        direction="lower_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "geometry_distance_correlation",
        "Pairwise distance Spearman correlation",
        "Spearman correlation between pairwise cosine distances in anchor and context spaces.",
        metric_family="context_stability",
        evidence_tier="secondary",
        unit="correlation",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "neighbor_overlap_fraction",
        "Neighbor overlap fraction",
        "|N_a ∩ N_b| / k for aligned anchor-context neighborhoods.",
        metric_family="context_stability",
        evidence_tier="primary",
        unit="fraction",
        direction="higher_is_better",
        aggregation_level="pair",
    ),
    _metric(
        "neighbor_set_jaccard",
        "Neighbor-set Jaccard",
        "|N_a ∩ N_b| / |N_a ∪ N_b| for aligned anchor-context neighborhoods.",
        metric_family="context_stability",
        evidence_tier="secondary",
        unit="fraction",
        direction="higher_is_better",
        aggregation_level="pair",
    ),
    _metric(
        "landmark_neighbor_jaccard",
        "Reference-neighbor Jaccard",
        "Mean Jaccard overlap between landmark-seeded neighborhoods across anchor and context spaces.",
        metric_family="reference_neighborhood",
        evidence_tier="secondary",
        unit="fraction",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "selected_rank",
        "Retained reduced rank",
        "Number of retained dimensions in the reducer output.",
        metric_family="representation_health",
        evidence_tier="secondary",
        unit="dims",
        direction="descriptive",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "explained_variance_captured",
        "Explained variance captured",
        "Cumulative explained variance captured by retained reducer components.",
        metric_family="representation_health",
        evidence_tier="secondary",
        unit="fraction",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "effective_rank",
        "Effective rank",
        "Entropy-derived effective rank across the retained PCA components reported by the reducer summary.",
        metric_family="representation_health",
        evidence_tier="primary",
        unit="dims",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "pc1_variance_fraction",
        "PC1 variance fraction",
        "Fraction of retained variance explained by the first principal component.",
        metric_family="representation_health",
        evidence_tier="primary",
        unit="fraction",
        direction="lower_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "pairwise_cosine_distance_median",
        "Median pairwise cosine distance",
        "Median pairwise cosine distance after view-level standardization and row L2 normalization.",
        metric_family="representation_health",
        evidence_tier="primary",
        unit="distance",
        direction="descriptive",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "pairwise_cosine_distance_iqr",
        "Pairwise cosine distance IQR",
        "Interquartile range of pairwise cosine distances after view-level standardization and row L2 normalization.",
        metric_family="representation_health",
        evidence_tier="primary",
        unit="distance",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "design_family_separation_ratio",
        "Design-family separation ratio",
        "Mean between-centroid cosine distance divided by mean within-centroid cosine distance for "
        "design-family cohorts.",
        metric_family="design_structure",
        evidence_tier="primary",
        unit="ratio",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "design_family_balanced_separation_ratio",
        "Balanced design-family separation ratio",
        "Design-family separation ratio after balancing by config-declared cohort columns.",
        metric_family="design_structure",
        evidence_tier="primary",
        unit="ratio",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "design_regulator_composition_separation_ratio",
        "Regulator-composition separation ratio",
        "Mean between-centroid cosine distance divided by mean within-centroid cosine distance for "
        "regulator-composition cohorts.",
        metric_family="design_structure",
        evidence_tier="primary",
        unit="ratio",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "spacer_length_separation_ratio",
        "Spacer-length separation ratio",
        "Mean between-centroid cosine distance divided by mean within-centroid cosine distance for "
        "realized spacer-length cohorts.",
        metric_family="design_structure",
        evidence_tier="primary",
        unit="ratio",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "cohort_separation_ratio",
        "Cohort separation ratio",
        "Mean between-centroid cosine distance divided by mean within-centroid cosine distance for a "
        "config-declared metadata cohort.",
        metric_family="cohort_structure",
        evidence_tier="primary",
        unit="ratio",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "ordinal_axis_spearman",
        "Ordinal-axis Spearman",
        "Spearman correlation between configured ordinal-rank gaps and observed centroid distances.",
        metric_family="ordinal_structure",
        evidence_tier="primary",
        unit="correlation",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "ordinal_axis_kendall",
        "Ordinal-axis Kendall",
        "Kendall tau correlation between configured ordinal-rank gaps and observed centroid distances.",
        metric_family="ordinal_structure",
        evidence_tier="primary",
        unit="correlation",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "ordinal_axis_balanced_spearman",
        "Balanced ordinal-axis Spearman",
        "Spearman correlation between configured ordinal-rank gaps and observed centroid distances after "
        "config-declared cohort balancing.",
        metric_family="ordinal_structure",
        evidence_tier="primary",
        unit="correlation",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "ordinal_axis_within_group_mean_spearman",
        "Within-group ordinal-axis Spearman",
        "Mean within-group Spearman correlation between configured ordinal-rank gaps and observed centroid distances.",
        metric_family="ordinal_structure",
        evidence_tier="primary",
        unit="correlation",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "ordinal_axis_label_permutation_pvalue",
        "Ordinal-axis permutation p-value",
        "Permutation p-value for the global ordinal-axis Spearman statistic under shuffled ordinal labels.",
        metric_family="ordinal_structure",
        evidence_tier="primary",
        unit="p_value",
        direction="lower_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "design_family_retention_correlation",
        "Design-family retention",
        "Pearson correlation between anchor and context centroid-distance matrices for design-family cohorts.",
        metric_family="context_stability",
        evidence_tier="primary",
        unit="correlation",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "design_regulator_composition_retention_correlation",
        "Regulator-composition retention",
        "Pearson correlation between anchor and context centroid-distance matrices for regulator-composition cohorts.",
        metric_family="context_stability",
        evidence_tier="primary",
        unit="correlation",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "reference_alignment_ethanol_background_relative",
        "Ethanol reference alignment",
        "Background-relative cosine alignment between the ethanol cohort centroid and the spyP reference centroid.",
        metric_family="reference_alignment",
        evidence_tier="appendix",
        unit="margin",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "reference_alignment_cipro_background_relative",
        "Ciprofloxacin reference alignment",
        "Background-relative cosine alignment between the ciprofloxacin cohort centroid and the "
        "sulAp reference centroid.",
        metric_family="reference_alignment",
        evidence_tier="appendix",
        unit="margin",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "reference_group_size",
        "Reference group size",
        "Number of labeled reference rows in a configured reference set or metadata reference group.",
        metric_family="reference_alignment",
        evidence_tier="appendix",
        unit="rows",
        direction="descriptive",
        aggregation_level="group_summary",
    ),
    _metric(
        "reference_group_pairwise_cosine_distance_median",
        "Reference group median distance",
        (
            "Median pairwise cosine distance within a configured reference set or metadata reference group "
            "after view-level normalization."
        ),
        metric_family="reference_alignment",
        evidence_tier="appendix",
        unit="distance",
        direction="descriptive",
        aggregation_level="group_summary",
    ),
    _metric(
        "reference_group_pairwise_cosine_distance_iqr",
        "Reference group distance IQR",
        (
            "Interquartile range of pairwise cosine distances within a labeled reference group "
            "or configured reference set after view-level normalization."
        ),
        metric_family="reference_alignment",
        evidence_tier="appendix",
        unit="distance",
        direction="descriptive",
        aggregation_level="group_summary",
    ),
    _metric(
        "reference_to_centroid_similarity",
        "Reference-to-centroid similarity",
        "Cosine similarity between a reference row or reference-set centroid and a configured cohort centroid.",
        metric_family="reference_alignment",
        evidence_tier="primary",
        unit="cosine_similarity",
        direction="descriptive",
        aggregation_level="reference_to_centroid",
    ),
    _metric(
        "reference_to_centroid_margin_median",
        "Reference-to-centroid margin median",
        "Median best-minus-second-best centroid similarity margin across configured reference entities.",
        metric_family="reference_alignment",
        evidence_tier="primary",
        unit="cosine_similarity_delta",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
    _metric(
        "reference_strength_ordinal_spearman_median",
        "Reference strength Spearman median",
        "Median collection-specific ordinal Spearman across configured reference-standard strength collections.",
        metric_family="ordinal_structure",
        evidence_tier="primary",
        unit="correlation",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
]

METRIC_DEFINITIONS = {metric.metric_id: metric for metric in _METRICS}


def _combined_metric_definitions(config: object | None = None) -> dict[str, MetricDefinition]:
    workspace_definitions = metric_definitions_from_config(config)
    overlap = sorted(set(METRIC_DEFINITIONS).intersection(workspace_definitions))
    if overlap:
        raise ContractViolationError(f"workspace metric_definitions cannot override global metric ids: {overlap}")
    return {**METRIC_DEFINITIONS, **workspace_definitions}


def resolve_metric_definition(metric_id: str, *, config: object | None = None) -> MetricDefinition:
    definitions = _combined_metric_definitions(config)
    try:
        return definitions[metric_id]
    except KeyError as exc:  # pragma: no cover - enforced by tests
        raise ContractViolationError(f"no metric definition is registered for {metric_id!r}") from exc


def metric_definition_digest(metric_id: str, *, config: object | None = None) -> str:
    definition = resolve_metric_definition(metric_id, config=config)
    return sha256_payload(asdict(definition))


def metric_definition_digests(
    metric_ids: list[str] | set[str] | tuple[str, ...], *, config: object | None = None
) -> dict[str, str]:
    return {
        metric_id: metric_definition_digest(metric_id, config=config)
        for metric_id in sorted({str(metric_id) for metric_id in metric_ids})
    }


def validate_metric_registry(config: object | None = None) -> None:
    definitions = _combined_metric_definitions(config)
    if len(METRIC_DEFINITIONS) != len(_METRICS):
        raise ContractViolationError("metric registry reuses metric identifiers")
    display_names = [metric.display_name for metric in definitions.values()]
    if len(display_names) != len(set(display_names)):
        raise ContractViolationError("metric registry reuses display names")
