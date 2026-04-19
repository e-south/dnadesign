from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.latentdna.src.io.hashing import sha256_file
from dnadesign.latentdna.src.services._artifact_inputs import (
    artifact_kind_for_input_dependency,
    dependency_artifact_input,
)
from dnadesign.latentdna.src.services._artifacts import prune_retired_artifact_dirs
from dnadesign.latentdna.src.services.freshness_service import FreshnessCache, evaluate_artifact_freshness
from dnadesign.latentdna.src.sources import provenance as provenance_module
from dnadesign.latentdna.src.sources.provenance import OVERLAY_INVENTORY_DIGEST_MODE, overlay_inventory_digest
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def test_dependency_artifact_input_uses_manifest_for_managed_artifacts(tmp_path: Path) -> None:
    context = SimpleNamespace(output_root=tmp_path)
    manifest_path = tmp_path / "views" / "demo_view" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"artifact_id":"demo_view"}', encoding="utf-8")
    matrix_path = manifest_path.parent / "matrix.npy"
    matrix_path.write_bytes(b"matrix-bytes")

    input_entry = dependency_artifact_input(
        context,
        kind="view_matrix",
        artifact_id="demo_view",
        path=matrix_path,
    )

    assert input_entry.path == manifest_path.as_posix()
    assert input_entry.digest == sha256_file(manifest_path)


def test_dependency_artifact_input_preserves_raw_path_inputs(tmp_path: Path) -> None:
    context = SimpleNamespace(output_root=tmp_path)
    source_path = tmp_path / "usr" / "records.parquet"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"raw-source")

    input_entry = dependency_artifact_input(
        context,
        kind="landmark_source",
        artifact_id="anchor60",
        path=source_path,
    )

    assert input_entry.path == source_path.as_posix()
    assert input_entry.digest == sha256_file(source_path)


def test_artifact_kind_for_input_dependency_maps_shared_input_kinds() -> None:
    assert artifact_kind_for_input_dependency("view_matrix") == "view"
    assert artifact_kind_for_input_dependency("neighbor_rows") == "neighbor_set"
    assert artifact_kind_for_input_dependency("landmark_source") is None


def test_freshness_accepts_legacy_managed_input_paths_via_upstream_manifest(tmp_path: Path) -> None:
    source_path = tmp_path / "usr" / "records.parquet"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"source-bytes")

    view_manifest_path = tmp_path / "views" / "demo_view" / "manifest.json"
    view_manifest_path.parent.mkdir(parents=True)
    view_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "demo_view",
                "status": "ok",
                "source_provenance": [
                    {
                        "path": source_path.as_posix(),
                        "digest": sha256_file(source_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    sample_manifest_path = tmp_path / "samples" / "demo_sample" / "manifest.json"
    sample_manifest_path.parent.mkdir(parents=True)
    sample_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "sample_set",
                "artifact_id": "demo_sample",
                "status": "ok",
                "inputs": [
                    {
                        "kind": "view_rows",
                        "id": "demo_view",
                        "path": (tmp_path / "views" / "demo_view" / "rows.parquet").as_posix(),
                        "digest": "sha256:legacy-payload-digest",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    context = SimpleNamespace(
        output_root=tmp_path,
        read_manifest=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="sample_set", artifact_id="demo_sample")

    assert freshness["status"] == "ok"


def test_freshness_uses_overlay_inventory_digest_for_new_source_provenance(tmp_path: Path) -> None:
    records_path = tmp_path / "usr" / "records.parquet"
    records_path.parent.mkdir(parents=True)
    records_path.write_bytes(b"records-bytes")

    overlay_dir = tmp_path / "usr" / "_derived" / "infer"
    overlay_dir.mkdir(parents=True)
    overlay_part = overlay_dir / "part-000.parquet"
    overlay_part.write_bytes(b"infer-overlay")

    view_manifest_path = tmp_path / "views" / "demo_view" / "manifest.json"
    view_manifest_path.parent.mkdir(parents=True)
    view_manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "demo_view",
                "status": "ok",
                "source_provenance": [
                    {
                        "kind": "file",
                        "id": "records.parquet",
                        "path": records_path.as_posix(),
                        "role": "records",
                        "digest": sha256_file(records_path),
                    },
                    {
                        "kind": "directory",
                        "id": "infer",
                        "path": overlay_dir.as_posix(),
                        "role": "overlay",
                        "namespace": "infer",
                        "digest_mode": OVERLAY_INVENTORY_DIGEST_MODE,
                        "digest": overlay_inventory_digest(overlay_dir),
                    },
                    {
                        "kind": "file",
                        "id": "infer:part-000.parquet",
                        "path": overlay_part.as_posix(),
                        "role": "overlay_part",
                        "namespace": "infer",
                        "digest": sha256_file(overlay_part),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    context = SimpleNamespace(
        output_root=tmp_path,
        read_manifest=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="demo_view")

    assert freshness["status"] == "ok"

    extra_part = overlay_dir / "part-001.parquet"
    extra_part.write_bytes(b"new-overlay")

    stale = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="demo_view")

    assert stale["status"] == "attention"
    assert "overlay inventory" in str(stale["reason"])


def test_freshness_cache_reuses_overlay_inventory_digest_across_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records_path = tmp_path / "usr" / "records.parquet"
    records_path.parent.mkdir(parents=True)
    records_path.write_bytes(b"records-bytes")

    overlay_dir = tmp_path / "usr" / "_derived" / "infer"
    overlay_dir.mkdir(parents=True)
    overlay_part = overlay_dir / "part-000.parquet"
    overlay_part.write_bytes(b"infer-overlay")

    source_provenance = [
        {
            "kind": "file",
            "id": "records.parquet",
            "path": records_path.as_posix(),
            "role": "records",
            "digest": sha256_file(records_path),
        },
        {
            "kind": "directory",
            "id": "infer",
            "path": overlay_dir.as_posix(),
            "role": "overlay",
            "namespace": "infer",
            "digest_mode": OVERLAY_INVENTORY_DIGEST_MODE,
            "digest": overlay_inventory_digest(overlay_dir),
        },
    ]

    for view_id in ["demo_view_a", "demo_view_b"]:
        manifest_path = tmp_path / "views" / view_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "artifact_kind": "view",
                    "artifact_id": view_id,
                    "status": "ok",
                    "source_provenance": source_provenance,
                }
            ),
            encoding="utf-8",
        )

    overlay_counts: dict[str, int] = {}
    original_overlay_inventory_digest = provenance_module.overlay_inventory_digest

    def counting_overlay_inventory_digest(path: Path) -> str:
        key = Path(path).resolve().as_posix()
        overlay_counts[key] = overlay_counts.get(key, 0) + 1
        return original_overlay_inventory_digest(path)

    monkeypatch.setattr(provenance_module, "overlay_inventory_digest", counting_overlay_inventory_digest)

    context = SimpleNamespace(
        output_root=tmp_path,
        read_manifest=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    freshness_cache = FreshnessCache()

    first = evaluate_artifact_freshness(
        context,
        artifact_kind="view",
        artifact_id="demo_view_a",
        cache=freshness_cache,
    )
    second = evaluate_artifact_freshness(
        context,
        artifact_kind="view",
        artifact_id="demo_view_b",
        cache=freshness_cache,
    )

    assert first["status"] == "ok"
    assert second["status"] == "ok"


def test_view_freshness_detects_row_column_drift_from_workspace_config(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    source_path = workspace_dir / "inputs" / "anchor.parquet"
    source_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["row_a"],
                "subject_id": ["row_a"],
                "embedding": [[0.1, 0.2]],
            }
        ),
        source_path,
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: freshness_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
sources:
  anchor_60bp:
    kind: parquet
    path: inputs/anchor.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: [spacer_length]
views:
  demo_view:
    source: anchor_60bp
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
cohorts:
  spacer_length:
    kind: promoter_metadata
    source: anchor_60bp
    derive: spacer_length
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path = workspace_dir / "outputs" / "views" / "demo_view" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "demo_view",
                "status": "ok",
                "source_provenance": [
                    {
                        "path": source_path.as_posix(),
                        "digest": sha256_file(source_path),
                    }
                ],
                "params": {
                    "source": "anchor_60bp",
                    "coordinate_space_id": "demo_space",
                    "record_key": "id",
                    "subject_key": "subject_id",
                    "context_key": None,
                    "row_columns": ["id", "subject_id"],
                },
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    freshness = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="demo_view")

    assert freshness["status"] == "attention"
    assert "missing row columns ['spacer_length']" in str(freshness["reason"])


def test_view_freshness_ignores_promoter_metadata_cohorts_from_other_sources(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    anchor_path = workspace_dir / "inputs" / "anchor.parquet"
    control_path = workspace_dir / "inputs" / "control.parquet"
    anchor_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["anchor_a"],
                "subject_id": ["anchor_a"],
                "embedding": [[0.1, 0.2]],
                "densegen__used_tfbs_detail": ['[{"spacer_length":17}]'],
                "usr_label__primary": ["anchor_a"],
            }
        ),
        anchor_path,
    )
    pq.write_table(
        pa.table(
            {
                "id": ["control_a"],
                "subject_id": ["control_a"],
                "embedding": [[0.3, 0.4]],
            }
        ),
        control_path,
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: freshness_scope_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
sources:
  anchor_60bp:
    kind: parquet
    path: inputs/anchor.parquet
    record_key: id
    subject_key: subject_id
  controls:
    kind: parquet
    path: inputs/control.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  control_view:
    source: controls
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
cohorts:
  spacer_length:
    kind: promoter_metadata
    source: anchor_60bp
    derive: spacer_length
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path = workspace_dir / "outputs" / "views" / "control_view" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "control_view",
                "status": "ok",
                "source_provenance": [
                    {
                        "path": control_path.as_posix(),
                        "digest": sha256_file(control_path),
                    }
                ],
                "params": {
                    "source": "controls",
                    "coordinate_space_id": "demo_space",
                    "record_key": "id",
                    "subject_key": "subject_id",
                    "context_key": None,
                    "row_columns": ["id", "subject_id"],
                },
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    freshness = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="control_view")

    assert freshness["status"] == "ok"


def test_view_freshness_marks_source_resolution_failures_as_attention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    source_path = workspace_dir / "inputs" / "anchor.parquet"
    source_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": ["row_a"],
                "subject_id": ["row_a"],
                "embedding": [[0.1, 0.2]],
            }
        ),
        source_path,
    )
    (workspace_dir / "config.yaml").write_text(
        """
schema_version: latentdna.workspace.v1
workspace:
  id: freshness_resolution_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
sources:
  anchor_60bp:
    kind: parquet
    path: inputs/anchor.parquet
    record_key: id
    subject_key: subject_id
metadata:
  include: []
views:
  demo_view:
    source: anchor_60bp
    vector:
      kind: column
      name: embedding
    coordinate_space_id: demo_space
        """.strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path = workspace_dir / "outputs" / "views" / "demo_view" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "demo_view",
                "status": "ok",
                "source_provenance": [
                    {
                        "path": source_path.as_posix(),
                        "digest": sha256_file(source_path),
                    }
                ],
                "params": {
                    "source": "anchor_60bp",
                    "coordinate_space_id": "demo_space",
                    "record_key": "id",
                    "subject_key": "subject_id",
                    "context_key": None,
                    "row_columns": ["id", "subject_id"],
                },
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    monkeypatch.setattr(
        "dnadesign.latentdna.src.services.freshness_service.inspect_source_schema",
        lambda resolved: (_ for _ in ()).throw(RuntimeError("schema probe failed")),
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="demo_view")

    assert freshness["status"] == "attention"
    assert freshness["known"] is False
    assert "schema probe failed" in str(freshness["reason"])


def test_prune_retired_artifact_dirs_removes_only_unconfigured_manifests(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    current_view = output_root / "views" / "current_view"
    retired_view = output_root / "views" / "retired_view"
    scratch_dir = output_root / "views" / "_scratch"
    for path in (current_view, retired_view, scratch_dir):
        path.mkdir(parents=True, exist_ok=True)
    (current_view / "manifest.json").write_text("{}", encoding="utf-8")
    (retired_view / "manifest.json").write_text("{}", encoding="utf-8")
    (scratch_dir / "note.txt").write_text("keep", encoding="utf-8")

    context = SimpleNamespace(
        output_root=output_root,
        config=SimpleNamespace(views={"current_view": object()}),
    )

    removed = prune_retired_artifact_dirs(context, artifact_kind="view")

    assert removed == [retired_view.as_posix()]
    assert current_view.is_dir()
    assert not retired_view.exists()
    assert scratch_dir.is_dir()
