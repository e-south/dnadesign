from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.latentdna.src.services.workspace_snapshot_service import (
    _model_families,
    decision_ladder,
    workspace_snapshot,
)


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
                "appendix_geometry_review": _deliverable(
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

    assert decision_ladder(context) == [
        "dataset_overview",
        "workspace_snapshot_export",
    ]


def test_model_families_accept_encoder_model_and_fully_qualified_model_tags() -> None:
    context = SimpleNamespace(
        config=SimpleNamespace(
            views={
                "short_model": SimpleNamespace(tags={"encoder": "evo2", "model": "7b"}),
                "qualified_model": SimpleNamespace(tags={"model": "evo2_7b"}),
                "other_model": SimpleNamespace(tags={"model": "demo"}),
            }
        )
    )

    assert _model_families(context) == ["demo", "evo2_7b"]


def test_workspace_snapshot_omits_missing_planned_sources(tmp_path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inputs_dir = workspace_dir / "inputs"
    inputs_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "id": ["row_01"],
                "embedding": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32())),
            }
        ),
        inputs_dir / "native.parquet",
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: snapshot_planned_source_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
  plot_formats: [svg]
  neighbor_backend: auto
sources:
  native:
    kind: parquet
    path: ./inputs/native.parquet
    record_key: id
    subject_key: id
  future_core60_features:
    kind: parquet
    path: ./inputs/not_yet_materialized.parquet
    record_key: id
    subject_key: id
    role: planned
views:
  native_geometry:
    source: native
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
    tags: {model: demo}
    role: planned
  future_core60_geometry:
    source: future_core60_features
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
    tags: {model: demo}
    role: planned
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    snapshot = workspace_snapshot(workspace_dir, write=False)

    assert "native" in snapshot["sources"]
    assert "future_core60_features" not in snapshot["sources"]
    assert not (workspace_dir / "outputs" / "status" / "workspace_snapshot.json").exists()

    workspace_snapshot(workspace_dir)
    assert (workspace_dir / "outputs" / "status" / "workspace_snapshot.json").is_file()
