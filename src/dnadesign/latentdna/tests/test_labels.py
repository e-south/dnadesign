"""Label humanization coverage for notebook-facing text."""

from __future__ import annotations

from dnadesign.latentdna.src.labels import humanize_label, humanize_plot_title


def test_humanize_label_preserves_required_public_labels() -> None:
    assert humanize_label("intermediate_embedding") == "Intermediate block mean"
    assert humanize_label("pooled_logits") == "Pooled logits"
    assert humanize_label("anchor_60bp") == "60 bp anchor"
    assert humanize_label("full_context_1kb") == "1 kb construct context"
    assert humanize_label("sig35_variant") == "Sigma-35 variant"


def test_humanize_plot_title_removes_public_appendix_language() -> None:
    assert humanize_plot_title("appendix_umap_gallery") == "UMAP gallery"
