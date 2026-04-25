from __future__ import annotations

from types import SimpleNamespace

from dnadesign.latentdna.src.services.workspace_snapshot_service import _decision_ladder


def _deliverable(
    *,
    plots: list[str] | None = None,
    exports: list[str] | None = None,
    section: str = "Summary",
) -> SimpleNamespace:
    return SimpleNamespace(section=section, outputs={"plots": list(plots or []), "exports": list(exports or [])})


def _plot(*, visibility_tier: str) -> SimpleNamespace:
    return SimpleNamespace(visibility_tier=visibility_tier)


def test_decision_ladder_excludes_appendix_only_deliverables() -> None:
    context = SimpleNamespace(
        config=SimpleNamespace(
            deliverables={
                "dataset_overview": _deliverable(plots=["dataset_overview"]),
                "appendix_geometry_audit": _deliverable(
                    plots=["design_centroid_margin_gallery", "appendix_umap_gallery"],
                    section="Appendix",
                ),
                "representation_health_summary": _deliverable(
                    plots=["representation_health_summary", "appendix_umap_gallery"],
                    section="Gate",
                ),
                "workspace_snapshot_export": _deliverable(exports=["workspace_snapshot"]),
            },
            plots={
                "dataset_overview": _plot(visibility_tier="primary"),
                "design_centroid_margin_gallery": _plot(visibility_tier="appendix"),
                "appendix_umap_gallery": _plot(visibility_tier="appendix"),
                "representation_health_summary": _plot(visibility_tier="primary"),
            },
        )
    )

    assert _decision_ladder(context) == [
        "dataset_overview",
        "workspace_snapshot_export",
    ]
