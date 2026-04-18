from __future__ import annotations

from types import SimpleNamespace

from dnadesign.latentdna.src.services.workspace_snapshot_service import _decision_ladder


def _deliverable(*, plots: list[str] | None = None, exports: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(outputs={"plots": list(plots or []), "exports": list(exports or [])})


def _plot(*, visibility_tier: str) -> SimpleNamespace:
    return SimpleNamespace(visibility_tier=visibility_tier)


def test_decision_ladder_excludes_appendix_only_deliverables() -> None:
    context = SimpleNamespace(
        config=SimpleNamespace(
            deliverables={
                "dataset_overview": _deliverable(plots=["dataset_overview"]),
                "appendix_umap_gallery": _deliverable(
                    plots=["reference_margin_gallery_synthetic_centroids", "appendix_umap_gallery"]
                ),
                "representation_comparison": _deliverable(
                    plots=["representation_tradeoff_scatter", "appendix_umap_gallery"]
                ),
                "workspace_snapshot_export": _deliverable(exports=["workspace_snapshot"]),
            },
            plots={
                "dataset_overview": _plot(visibility_tier="primary"),
                "reference_margin_gallery_synthetic_centroids": _plot(visibility_tier="appendix"),
                "appendix_umap_gallery": _plot(visibility_tier="appendix"),
                "representation_tradeoff_scatter": _plot(visibility_tier="primary"),
            },
        )
    )

    assert _decision_ladder(context) == [
        "dataset_overview",
        "representation_comparison",
        "workspace_snapshot_export",
    ]
