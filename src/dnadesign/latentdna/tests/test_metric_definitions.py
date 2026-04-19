from __future__ import annotations

from dnadesign.latentdna.src.metrics.definitions import METRIC_DEFINITIONS, resolve_metric_definition


def test_metric_registry_uses_comparison_metadata_and_drops_selection_state() -> None:
    assert "selection_state_code" not in METRIC_DEFINITIONS

    overlap = resolve_metric_definition("neighbor_overlap_fraction")
    assert overlap.display_name == "Neighbor overlap fraction"
    assert overlap.direction == "higher_is_better"
    assert overlap.unit == "fraction"
    assert overlap.aggregation_level == "pair"
    assert "|N_a ∩ N_b| / k" in overlap.mathematical_definition

    jaccard = resolve_metric_definition("neighbor_set_jaccard")
    assert jaccard.display_name == "Neighbor-set Jaccard"
    assert jaccard.direction == "higher_is_better"
    assert jaccard.unit == "fraction"
    assert jaccard.aggregation_level == "pair"

    reference_rate = resolve_metric_definition("reference_in_knn_rate")
    assert reference_rate.metric_family == "reference_neighborhood"
    assert reference_rate.direction == "higher_is_better"
    assert reference_rate.unit == "fraction"

    reference_rank = resolve_metric_definition("reference_neighbor_topk_censored_rank_median")
    assert reference_rank.metric_family == "reference_neighborhood"
    assert reference_rank.direction == "lower_is_better"
    assert reference_rank.unit == "rank"

    distance_spearman = resolve_metric_definition("geometry_distance_correlation")
    assert distance_spearman.display_name == "Pairwise distance Spearman"
    assert "Spearman correlation" in distance_spearman.mathematical_definition
