"""Label humanization coverage for notebook-facing text."""

from __future__ import annotations

from dnadesign.latentdna.src.labels import humanize_candidate, humanize_label, humanize_plot_title


def test_humanize_label_preserves_required_public_labels() -> None:
    assert humanize_label("intermediate_embedding") == "Intermediate block mean"
    assert humanize_label("pooled_logits") == "Pooled logits"
    assert humanize_label("output_layer_mean") == "Output-layer mean"
    assert humanize_label("anchor_60bp") == "60 bp anchor"
    assert humanize_label("full_context_1kb") == "1 kb construct context"
    assert humanize_label("sig35_variant") == "Sigma-35 variant"
    assert humanize_label("variant f") == "TTGACA (f)"


def test_humanize_plot_title_removes_public_appendix_language() -> None:
    assert humanize_plot_title("appendix_umap_gallery") == "UMAP gallery"


def test_humanize_candidate_preserves_anchor_mean_and_concat_labels() -> None:
    assert (
        humanize_candidate(
            {
                "candidate_model": "evo2_7b",
                "candidate_scope": "full_context_anchor_mean",
                "candidate_family": "intermediate_embedding",
            }
        )
        == "Evo 2 7B · 1 kb context anchor mean · Intermediate block mean"
    )
    assert (
        humanize_candidate(
            {
                "candidate_model": "evo2_7b",
                "candidate_scope": "anchor_plus_anchor_mean_concat",
                "candidate_family": "intermediate_embedding",
            }
        )
        == "Evo 2 7B · Anchor + anchor-mean concat · Intermediate block mean"
    )


def test_humanize_label_keeps_distance_correlation_labels_consistent() -> None:
    assert humanize_label("geometry_distance_correlation") == "Pairwise distance Spearman correlation"
    assert humanize_label("pairwise distance correlation") == "Pairwise distance Spearman correlation"
    assert humanize_label("pairwise distance spearman") == "Pairwise distance Spearman correlation"
