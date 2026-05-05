"""Contracts for notebook context-audit summary aggregation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.latentdna.src.services.notebook_context_audit import build_workspace_notebook_context_audit


def _write_context_audit_table(output_root: Path, *, scalar_id: str) -> None:
    artifact_dir = output_root / "scalars" / scalar_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "context_shift_l2": [0.2, 0.4, 0.6],
                "context_self_cosine": [0.95, 0.9, 0.85],
            }
        ),
        artifact_dir / "table.parquet",
    )


def test_context_audit_summary_aggregates_configured_context_shift_scalars(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    _write_context_audit_table(output_root, scalar_id="context_delta_distribution_intermediate_embedding_20b")
    _write_context_audit_table(output_root, scalar_id="context_delta_distribution_output_layer_mean_20b")

    context = SimpleNamespace(
        output_root=output_root,
        config=SimpleNamespace(
            notebooks={
                "latent_geometry_browser": SimpleNamespace(
                    context_audit_scalar_ids=[
                        "context_delta_distribution_intermediate_embedding_20b",
                        "context_delta_distribution_intermediate_embedding_7b",
                        "context_delta_distribution_output_layer_mean_20b",
                    ]
                )
            }
        ),
    )

    payload = build_workspace_notebook_context_audit(context)

    assert payload.status == "ok"
    assert payload.decision == "structured_context_shift"
    assert payload.rows == 6
    assert payload.metrics is not None
    assert payload.metrics["configured_scalar_panel_count"] == 3
    assert payload.metrics["scalar_panel_count"] == 2
    assert payload.metrics["scalar_panel_ids"] == [
        "context_delta_distribution_intermediate_embedding_20b",
        "context_delta_distribution_intermediate_embedding_7b",
        "context_delta_distribution_output_layer_mean_20b",
    ]
    assert payload.metrics["missing_scalar_table_ids"] == ["context_delta_distribution_intermediate_embedding_7b"]
    assert payload.metrics["context_shift_l2_median"] == 0.4
    assert payload.metrics["context_shift_l2_p95"] == 0.6
    assert payload.metrics["context_self_cosine_median"] == 0.9
    assert payload.metrics["context_self_cosine_p05"] == 0.85
    assert payload.metrics["table_paths"] == [
        "scalars/context_delta_distribution_intermediate_embedding_20b/table.parquet",
        "scalars/context_delta_distribution_output_layer_mean_20b/table.parquet",
    ]
