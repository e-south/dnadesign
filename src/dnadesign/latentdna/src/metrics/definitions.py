"""Canonical comparison-metric definitions for LatentDNA artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.errors import ContractViolationError


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
    def math_definition(self) -> str:
        return self.mathematical_definition

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
        "knn_sig35_enrichment_delta",
        "kNN sig35 enrichment",
        "Mean within-neighborhood enrichment delta for sig35_variant labels.",
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
        "reference_neighbor_rank_median",
        "Reference neighbor rank median",
        "Median rank position of the intended reference anchor in the full high-dimensional neighborhood order.",
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
        "Pairwise distance Spearman",
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
        "Entropy-derived effective rank of the centered representation spectrum.",
        metric_family="representation_health",
        evidence_tier="primary",
        unit="dims",
        direction="higher_is_better",
        aggregation_level="candidate_summary",
    ),
]

METRIC_DEFINITIONS = {metric.metric_id: metric for metric in _METRICS}


def resolve_metric_definition(metric_id: str) -> MetricDefinition:
    try:
        return METRIC_DEFINITIONS[metric_id]
    except KeyError as exc:  # pragma: no cover - enforced by tests
        raise ContractViolationError(f"no metric definition is registered for {metric_id!r}") from exc


def validate_metric_registry() -> None:
    if len(METRIC_DEFINITIONS) != len(_METRICS):
        raise ContractViolationError("metric registry reuses metric identifiers")
    display_names = [metric.display_name for metric in _METRICS]
    if len(display_names) != len(set(display_names)):
        raise ContractViolationError("metric registry reuses display names")
