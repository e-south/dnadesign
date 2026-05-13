from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.latentdna.src.io.hashing import sha256_file
from dnadesign.latentdna.src.metrics.definitions import metric_definition_digests
from dnadesign.latentdna.src.services._artifact_inputs import (
    artifact_kind_for_input_dependency,
    dependency_artifact_input,
)
from dnadesign.latentdna.src.services._artifacts import prune_retired_artifact_dirs
from dnadesign.latentdna.src.services.freshness_service import FreshnessCache, evaluate_artifact_freshness
from dnadesign.latentdna.src.sources import provenance as provenance_module
from dnadesign.latentdna.src.sources.provenance import (
    OVERLAY_INVENTORY_DIGEST_MODE,
    OVERLAY_LEDGER_PAYLOAD_DIGEST_MODE,
    overlay_inventory_digest,
    overlay_ledger_payload_digest,
)
from dnadesign.latentdna.src.workspaces.loader import load_workspace_config


def _write_metric_definition_workspace(workspace_dir: Path, *, display_name: str = "Demo workspace metric") -> None:
    (workspace_dir / "config.yaml").write_text(
        f"""
schema_version: latentdna.workspace.v1
workspace:
  id: scalar_metric_definition_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
metadata:
  include: []
sources: {{}}
metric_definitions:
  demo_workspace_metric:
    display_name: {display_name!r}
    mathematical_definition: Mean configured demo score over materialized rows.
    metric_family: demo
    evidence_tier: appendix
    unit: score
    direction: descriptive
    aggregation_level: scalar_summary
        """.strip()
        + "\n",
        encoding="utf-8",
    )


def _write_scalar_metric_artifact(
    workspace_dir: Path,
    *,
    params: dict[str, object] | None = None,
) -> Path:
    stable_input = workspace_dir / "inputs" / "stable-source.txt"
    stable_input.parent.mkdir(parents=True, exist_ok=True)
    stable_input.write_text("stable\n", encoding="utf-8")
    scalar_dir = workspace_dir / "outputs" / "scalars" / "demo_scalar"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "metric_id": ["demo_workspace_metric"],
                "metric_value": [1.0],
                "display_name": ["Demo workspace metric"],
            }
        ),
        scalar_dir / "table.parquet",
    )
    (scalar_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "scalar_table",
                "artifact_id": "demo_scalar",
                "status": "ok",
                "inputs": [
                    {
                        "kind": "source",
                        "id": "stable-source",
                        "path": stable_input.as_posix(),
                        "digest": sha256_file(stable_input),
                    }
                ],
                "params": params or {},
                "outputs": [{"path": "table.parquet", "media_type": "application/x-parquet"}],
            }
        ),
        encoding="utf-8",
    )
    return scalar_dir / "manifest.json"


def _write_scalar_build_recipe_workspace(workspace_dir: Path, *, pairwise_max_rows: int) -> None:
    (workspace_dir / "config.yaml").write_text(
        f"""
schema_version: latentdna.workspace.v1
workspace:
  id: scalar_build_freshness_demo
  output_root: ./outputs
defaults:
  analysis_dtype: float32
  metric: cosine
  random_seed: 17
sources: {{}}
recipes:
  demo_recipe:
    steps:
      - id: build_demo_scalar
        op: scalar.build
        params:
          scalar: demo_scalar
          kind: representation_health_summary
          pairwise_max_rows: {pairwise_max_rows}
          pairwise_seed: 17
          candidates: []
        """.strip()
        + "\n",
        encoding="utf-8",
    )


def _write_scalar_build_artifact(workspace_dir: Path, *, pairwise_max_rows: int) -> Path:
    stable_input = workspace_dir / "inputs" / "stable-source.txt"
    stable_input.parent.mkdir(parents=True, exist_ok=True)
    stable_input.write_text("stable\n", encoding="utf-8")
    scalar_dir = workspace_dir / "outputs" / "scalars" / "demo_scalar"
    scalar_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"label": ["demo"], "value": [1.0]}), scalar_dir / "table.parquet")
    (scalar_dir / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_kind": "scalar_table",
                "artifact_id": "demo_scalar",
                "status": "ok",
                "inputs": [
                    {
                        "kind": "source",
                        "id": "stable-source",
                        "path": stable_input.as_posix(),
                        "digest": sha256_file(stable_input),
                    }
                ],
                "params": {
                    "builder_kind": "representation_health_summary",
                    "pairwise_max_rows": pairwise_max_rows,
                    "pairwise_seed": 17,
                    "candidates": [],
                },
                "outputs": [{"path": "table.parquet", "media_type": "application/x-parquet"}],
            }
        ),
        encoding="utf-8",
    )
    return scalar_dir / "manifest.json"


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


def test_scalar_freshness_requires_metric_definition_provenance(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_metric_definition_workspace(workspace_dir)
    _write_scalar_metric_artifact(workspace_dir)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    freshness = evaluate_artifact_freshness(context, artifact_kind="scalar_table", artifact_id="demo_scalar")

    assert freshness["status"] == "attention"
    assert "metric definition provenance" in str(freshness["reason"])


def test_scalar_freshness_detects_metric_definition_drift(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_metric_definition_workspace(workspace_dir)
    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    _write_scalar_metric_artifact(
        workspace_dir,
        params={
            "metric_definition_digests": metric_definition_digests(
                ["demo_workspace_metric"],
                config=context.config,
            )
        },
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="scalar_table", artifact_id="demo_scalar")
    assert freshness["status"] == "ok"

    _write_metric_definition_workspace(workspace_dir, display_name="Renamed demo workspace metric")
    drifted_context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    drifted = evaluate_artifact_freshness(drifted_context, artifact_kind="scalar_table", artifact_id="demo_scalar")

    assert drifted["status"] == "attention"
    assert "demo_workspace_metric" in str(drifted["reason"])


def test_scalar_freshness_detects_build_recipe_param_drift(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write_scalar_build_recipe_workspace(workspace_dir, pairwise_max_rows=64)
    _write_scalar_build_artifact(workspace_dir, pairwise_max_rows=64)

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    freshness = evaluate_artifact_freshness(context, artifact_kind="scalar_table", artifact_id="demo_scalar")
    assert freshness["status"] == "ok"

    _write_scalar_build_recipe_workspace(workspace_dir, pairwise_max_rows=128)
    drifted_context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    drifted = evaluate_artifact_freshness(drifted_context, artifact_kind="scalar_table", artifact_id="demo_scalar")

    assert drifted["status"] == "attention"
    assert "stale scalar build config for scalar_table:demo_scalar" in str(drifted["reason"])


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


def test_freshness_uses_overlay_ledger_payload_digest_for_in_place_overlay_mutation(tmp_path: Path) -> None:
    records_path = tmp_path / "usr" / "records.parquet"
    records_path.parent.mkdir(parents=True)
    records_path.write_bytes(b"records-bytes")

    overlay_dir = tmp_path / "usr" / "_derived" / "infer"
    overlay_dir.mkdir(parents=True)
    overlay_part = overlay_dir / "part-000.parquet"
    pq.write_table(pa.table({"id": ["row_a"], "value": [1.0]}), overlay_part)
    ledger_path = overlay_dir / "digest_ledger.json"
    ledger_path.write_text("{}", encoding="utf-8")

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
                        "id": "infer:digest_ledger",
                        "path": ledger_path.as_posix(),
                        "role": "overlay_ledger",
                        "namespace": "infer",
                        "digest_mode": OVERLAY_LEDGER_PAYLOAD_DIGEST_MODE,
                        "digest": overlay_ledger_payload_digest(ledger_path),
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

    pq.write_table(pa.table({"id": ["row_a"], "value": [2.0]}), overlay_part)

    stale = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="demo_view")
    assert stale["status"] == "attention"
    assert "infer:digest_ledger" in str(stale["reason"])


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
                "usr_label__primary": ["row_a"],
                "densegen__plan": ["ethanol__sig35=b"],
                "densegen__used_tfbs_detail": ['[{"part_kind":"fixed_element","spacer_length":17}]'],
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
  derivations:
    spacer_length:
      kind: annotation
      source: row
      handler: dnadesign.latentdna.src.views.promoter_metadata:derive_promoter_metadata_value
      derive: spacer_length
      required_columns: [densegen__plan, densegen__used_tfbs_detail, usr_label__primary]
      missing_policy: error
      value_type: int64
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
                    "analysis_dtype": "float32",
                    "source": "anchor_60bp",
                    "coordinate_space_id": "demo_space",
                    "record_key": "id",
                    "subject_key": "subject_id",
                    "context_key": None,
                    "vector_kind": "column",
                    "vector_column": "embedding",
                    "row_columns": ["id", "subject_id"],
                    "role": None,
                    "tags": {},
                },
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    freshness = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="demo_view")

    assert freshness["status"] == "attention"
    assert "missing row columns ['spacer_length']" in str(freshness["reason"])


def test_derived_view_freshness_accepts_matching_concat_config(tmp_path: Path) -> None:
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
  id: derived_freshness_demo
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
  left_view:
    source: anchor_60bp
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
  right_view:
    source: anchor_60bp
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
  concat_view:
    derive:
      kind: concatenate
      inputs: [left_view, right_view]
    coordinate_space_id: concat_space
    role: appendix
    tags:
      scope: anchor_plus_context_concat
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    for view_id in ["left_view", "right_view"]:
        manifest_path = workspace_dir / "outputs" / "views" / view_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "artifact_kind": "view",
                    "artifact_id": view_id,
                    "status": "ok",
                    "source_provenance": [
                        {
                            "path": source_path.as_posix(),
                            "digest": sha256_file(source_path),
                        }
                    ],
                    "params": {
                        "analysis_dtype": "float32",
                        "source": "anchor_60bp",
                        "coordinate_space_id": "shared_space",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": None,
                        "vector_kind": "column",
                        "vector_column": "embedding",
                        "row_columns": ["id", "subject_id"],
                        "role": None,
                        "tags": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    left_manifest = workspace_dir / "outputs" / "views" / "left_view" / "manifest.json"
    right_manifest = workspace_dir / "outputs" / "views" / "right_view" / "manifest.json"
    concat_manifest = workspace_dir / "outputs" / "views" / "concat_view" / "manifest.json"
    concat_manifest.parent.mkdir(parents=True, exist_ok=True)
    concat_manifest.write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "concat_view",
                "status": "ok",
                "inputs": [
                    {
                        "kind": "view_matrix",
                        "id": "left_view",
                        "path": left_manifest.as_posix(),
                        "digest": sha256_file(left_manifest),
                    },
                    {
                        "kind": "view_matrix",
                        "id": "right_view",
                        "path": right_manifest.as_posix(),
                        "digest": sha256_file(right_manifest),
                    },
                ],
                "params": {
                    "analysis_dtype": "float32",
                    "coordinate_space_id": "concat_space",
                    "derive_kind": "concatenate",
                    "row_columns": ["id", "subject_id"],
                    "role": "appendix",
                    "tags": {"scope": "anchor_plus_context_concat"},
                    "input_views": ["left_view", "right_view"],
                },
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    freshness = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="concat_view")

    assert freshness["status"] == "ok"
    assert freshness["known"] is True


def test_view_freshness_scopes_metadata_requirements_to_materialized_source(tmp_path: Path) -> None:
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
                    "analysis_dtype": "float32",
                    "source": "controls",
                    "coordinate_space_id": "demo_space",
                    "record_key": "id",
                    "subject_key": "subject_id",
                    "context_key": None,
                    "vector_kind": "column",
                    "vector_column": "embedding",
                    "row_columns": ["id", "subject_id"],
                    "role": None,
                    "tags": {},
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


def test_view_freshness_detects_source_backed_analysis_dtype_drift(tmp_path: Path) -> None:
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
  id: dtype_source_demo
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
                "source_provenance": [{"path": source_path.as_posix(), "digest": sha256_file(source_path)}],
                "params": {
                    "analysis_dtype": "float16",
                    "source": "anchor_60bp",
                    "coordinate_space_id": "demo_space",
                    "record_key": "id",
                    "subject_key": "subject_id",
                    "context_key": None,
                    "vector_kind": "column",
                    "vector_column": "embedding",
                    "row_columns": ["id", "subject_id"],
                    "role": None,
                    "tags": {},
                },
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    freshness = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="demo_view")

    assert freshness["status"] == "attention"
    assert "analysis_dtype" in str(freshness["reason"])


def test_view_freshness_detects_derived_analysis_dtype_drift(tmp_path: Path) -> None:
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
  id: dtype_derived_demo
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
  left_view:
    source: anchor_60bp
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
  right_view:
    source: anchor_60bp
    vector:
      kind: column
      name: embedding
    coordinate_space_id: shared_space
  concat_view:
    derive:
      kind: concatenate
      inputs: [left_view, right_view]
    coordinate_space_id: concat_space
        """.strip()
        + "\n",
        encoding="utf-8",
    )

    for view_id in ["left_view", "right_view"]:
        manifest_path = workspace_dir / "outputs" / "views" / view_id / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "artifact_kind": "view",
                    "artifact_id": view_id,
                    "status": "ok",
                    "source_provenance": [{"path": source_path.as_posix(), "digest": sha256_file(source_path)}],
                    "params": {
                        "analysis_dtype": "float32",
                        "source": "anchor_60bp",
                        "coordinate_space_id": "shared_space",
                        "record_key": "id",
                        "subject_key": "subject_id",
                        "context_key": None,
                        "vector_kind": "column",
                        "vector_column": "embedding",
                        "row_columns": ["id", "subject_id"],
                        "role": None,
                        "tags": {},
                    },
                }
            ),
            encoding="utf-8",
        )

    left_manifest = workspace_dir / "outputs" / "views" / "left_view" / "manifest.json"
    right_manifest = workspace_dir / "outputs" / "views" / "right_view" / "manifest.json"
    concat_manifest = workspace_dir / "outputs" / "views" / "concat_view" / "manifest.json"
    concat_manifest.parent.mkdir(parents=True, exist_ok=True)
    concat_manifest.write_text(
        json.dumps(
            {
                "artifact_kind": "view",
                "artifact_id": "concat_view",
                "status": "ok",
                "inputs": [
                    {
                        "kind": "view_matrix",
                        "id": "left_view",
                        "path": left_manifest.as_posix(),
                        "digest": sha256_file(left_manifest),
                    },
                    {
                        "kind": "view_matrix",
                        "id": "right_view",
                        "path": right_manifest.as_posix(),
                        "digest": sha256_file(right_manifest),
                    },
                ],
                "params": {
                    "analysis_dtype": "float16",
                    "coordinate_space_id": "concat_space",
                    "derive_kind": "concatenate",
                    "row_columns": ["id", "subject_id"],
                    "role": None,
                    "tags": {},
                    "input_views": ["left_view", "right_view"],
                },
            }
        ),
        encoding="utf-8",
    )

    context = load_workspace_config(workspace_dir, validate_plot_semantics=False)
    freshness = evaluate_artifact_freshness(context, artifact_kind="view", artifact_id="concat_view")

    assert freshness["status"] == "attention"
    assert "analysis_dtype" in str(freshness["reason"])


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


def test_freshness_requires_declared_output_payloads_to_exist(tmp_path: Path) -> None:
    source_path = tmp_path / "inputs" / "source.txt"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("source", encoding="utf-8")
    manifest_path = tmp_path / "views" / "demo_view" / "manifest.json"
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
                "outputs": [
                    {"path": "matrix.npy", "media_type": "application/x-npy"},
                    {"path": "rows.parquet", "media_type": "application/x-parquet"},
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

    assert freshness["status"] == "attention"
    assert freshness["known"] is True
    assert "artifact payload is missing for view:demo_view: matrix.npy" in str(freshness["reason"])


def test_notebook_freshness_consults_health_contract(tmp_path: Path) -> None:
    source_path = tmp_path / "inputs" / "workspace.yaml"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("workspace: demo\n", encoding="utf-8")
    notebook_dir = tmp_path / "notebooks" / "atlas_review"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "notebook.py").write_text("import marimo\n", encoding="utf-8")
    (notebook_dir / "controls.json").write_text("{}", encoding="utf-8")
    manifest_path = notebook_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "notebook",
                "artifact_id": "atlas_review",
                "status": "ok",
                "source_provenance": [
                    {
                        "path": source_path.as_posix(),
                        "digest": sha256_file(source_path),
                    }
                ],
                "outputs": [
                    {"path": "notebook.py", "media_type": "text/x-python"},
                    {"path": "controls.json", "media_type": "application/json"},
                ],
            }
        ),
        encoding="utf-8",
    )
    health_path = notebook_dir / "health.json"
    health_path.write_text(
        json.dumps(
            {
                "workspace_id": "freshness_demo",
                "notebook_id": "atlas_review",
                "status": "error",
                "checks": {
                    "notebook_exists": True,
                    "control_plane_loads": False,
                    "imports_resolve": True,
                    "plot_catalog_loads": True,
                    "default_deliverable_ready": True,
                    "static_links_resolve": True,
                },
                "warnings": ["control_plane_loads failed: invalid controls payload"],
            }
        ),
        encoding="utf-8",
    )

    context = SimpleNamespace(
        output_root=tmp_path,
        workspace_id="freshness_demo",
        read_manifest=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="notebook", artifact_id="atlas_review")

    assert freshness["status"] == "attention"
    assert freshness["known"] is True
    assert "control_plane_loads failed" in str(freshness["reason"])


def test_notebook_freshness_rejects_workspace_mismatched_health_payload(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks" / "atlas_review"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "manifest.json").write_text(
        json.dumps({"artifact_kind": "notebook", "artifact_id": "atlas_review", "status": "ok", "outputs": []}),
        encoding="utf-8",
    )
    (notebook_dir / "health.json").write_text(
        json.dumps(
            {
                "workspace_id": "other_workspace",
                "notebook_id": "atlas_review",
                "status": "ok",
                "checks": {"notebook_exists": True},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    context = SimpleNamespace(
        output_root=tmp_path,
        workspace_id="freshness_demo",
        read_manifest=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="notebook", artifact_id="atlas_review")

    assert freshness["status"] == "attention"
    assert "workspace_id=other_workspace" in str(freshness["reason"])


def test_notebook_freshness_rejects_ok_status_with_failing_checks(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks" / "atlas_review"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "manifest.json").write_text(
        json.dumps({"artifact_kind": "notebook", "artifact_id": "atlas_review", "status": "ok", "outputs": []}),
        encoding="utf-8",
    )
    (notebook_dir / "health.json").write_text(
        json.dumps(
            {
                "workspace_id": "freshness_demo",
                "notebook_id": "atlas_review",
                "status": "ok",
                "checks": {
                    "notebook_exists": True,
                    "control_plane_loads": False,
                    "imports_resolve": True,
                },
                "warnings": ["control_plane_loads failed: invalid controls payload"],
            }
        ),
        encoding="utf-8",
    )

    context = SimpleNamespace(
        output_root=tmp_path,
        workspace_id="freshness_demo",
        read_manifest=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="notebook", artifact_id="atlas_review")

    assert freshness["status"] == "attention"
    assert "control_plane_loads" in str(freshness["reason"])
    assert "invalid controls payload" in str(freshness["reason"])


def test_notebook_freshness_rejects_invalid_checks_payload_shape(tmp_path: Path) -> None:
    notebook_dir = tmp_path / "notebooks" / "atlas_review"
    notebook_dir.mkdir(parents=True)
    (notebook_dir / "manifest.json").write_text(
        json.dumps({"artifact_kind": "notebook", "artifact_id": "atlas_review", "status": "ok", "outputs": []}),
        encoding="utf-8",
    )
    (notebook_dir / "health.json").write_text(
        json.dumps(
            {
                "workspace_id": "freshness_demo",
                "notebook_id": "atlas_review",
                "status": "ok",
                "checks": ["not", "a", "mapping"],
            }
        ),
        encoding="utf-8",
    )

    context = SimpleNamespace(
        output_root=tmp_path,
        workspace_id="freshness_demo",
        read_manifest=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="notebook", artifact_id="atlas_review")

    assert freshness["status"] == "attention"
    assert "invalid checks payload" in str(freshness["reason"])


def test_notebook_freshness_is_isolated_per_notebook_health_file(tmp_path: Path) -> None:
    source_path = tmp_path / "inputs" / "workspace.yaml"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("workspace: demo\n", encoding="utf-8")
    for notebook_id, status in [("atlas_review", "ok"), ("comparison_review", "error")]:
        notebook_dir = tmp_path / "notebooks" / notebook_id
        notebook_dir.mkdir(parents=True)
        (notebook_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "artifact_kind": "notebook",
                    "artifact_id": notebook_id,
                    "status": "ok",
                    "outputs": [],
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
        (notebook_dir / "health.json").write_text(
            json.dumps(
                {
                    "workspace_id": "freshness_demo",
                    "notebook_id": notebook_id,
                    "status": status,
                    "checks": {"notebook_exists": True},
                    "warnings": ([] if status == "ok" else ["broken"]),
                }
            ),
            encoding="utf-8",
        )

    context = SimpleNamespace(
        output_root=tmp_path,
        workspace_id="freshness_demo",
        read_manifest=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )

    freshness = evaluate_artifact_freshness(context, artifact_kind="notebook", artifact_id="atlas_review")

    assert freshness["status"] == "ok"
