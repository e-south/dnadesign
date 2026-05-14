"""Scalar builder for native regulator plan-margin enrichment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...enrichments.regulatory_plan_margin import build_regulatory_plan_margin_artifacts
from ...io.matrix_io import read_matrix
from ...io.parquet_io import read_table, write_table
from ...workspaces.loader import WorkspaceContext
from ..common import BuiltScalarArtifact, ScalarInputRef, _optional_param, _require_param

_CONTRACT_NAME = "native_regulator_plan_margin_enrichment"


def _mapping_param(params: dict[str, object], key: str) -> dict[str, object]:
    raw_value = _require_param(params, key)
    if not isinstance(raw_value, dict):
        raise ContractViolationError(f"{_CONTRACT_NAME} requires mapping param {key!r}")
    return dict(raw_value)


def _view_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path]:
    matrix_path = context.output_root / "views" / view_id / "matrix.npy"
    rows_path = context.output_root / "views" / view_id / "rows.parquet"
    if not matrix_path.is_file() or not rows_path.is_file():
        raise MissingArtifactError(f"view artifact is missing for {_CONTRACT_NAME}: {view_id}")
    return matrix_path, rows_path


def _workspace_table_path(
    context: WorkspaceContext,
    raw_path: object,
    *,
    param_name: str,
    contract_name: str,
) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ContractViolationError(f"{contract_name} requires {param_name}")
    path = Path(text)
    if not path.is_absolute():
        path = context.workspace_dir / path
    if not path.is_file():
        raise MissingArtifactError(f"{contract_name} is missing: {path}")
    return path


def build_native_regulator_plan_margin_enrichment_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, object],
) -> BuiltScalarArtifact:
    """Build the scalar artifact and side tables for native regulator margins."""

    view_id = str(_require_param(params, "view_id"))
    matrix_path, rows_path = _view_paths(context, view_id)
    regulatory_interactions = _mapping_param(params, "regulatory_interactions")
    interactions_path = _workspace_table_path(
        context,
        regulatory_interactions.get("path"),
        param_name="path",
        contract_name=f"{_CONTRACT_NAME} regulatory_interactions",
    )
    native_parent_column = str(
        _optional_param(
            params,
            "native_parent_column",
            default=regulatory_interactions.get("row_key")
            or regulatory_interactions.get("join_key")
            or "derived__parent_id",
        )
    )
    artifacts = build_regulatory_plan_margin_artifacts(
        matrix=np.asarray(read_matrix(matrix_path), dtype=np.float32),
        rows_table=read_table(rows_path),
        relations_table=read_table(interactions_path),
        view_id=view_id,
        cohort_column=str(_require_param(params, "cohort_column")),
        centroid_groups=_mapping_param(params, "centroid_groups"),
        native_filter=_mapping_param(params, "native_filter"),
        native_parent_column=native_parent_column,
        relation_key=str(
            regulatory_interactions.get("relation_key") or regulatory_interactions.get("join_key") or "usr_id"
        ),
        regulator_column=str(regulatory_interactions.get("regulator_column") or "regulator_abbrev"),
        required_relation_columns=regulatory_interactions.get("required_columns") or [],
        thresholds=_optional_param(params, "thresholds", default=[0.05, 0.10]),
        tail_modes=_optional_param(params, "tail_modes", default=["margin_top_quantile"]),
        min_global_promoters=int(_optional_param(params, "min_global_promoters", default=10)),
        min_tail_hits=int(_optional_param(params, "min_tail_hits", default=3)),
        fdr_method=str(_optional_param(params, "fdr_method", default="benjamini_hochberg")),
        common_regulators=_optional_param(params, "common_regulators", default=[]),
        plan_order=_optional_param(params, "plan_order", default=None),
        native_metadata_columns=_optional_param(params, "native_metadata_columns", default=[]),
        expected_output_rows=(
            int(_optional_param(params, "expected_output_rows", default=0))
            if "expected_output_rows" in params
            else None
        ),
    )
    write_table(artifacts.enrichment_table, artifact_dir / "table.parquet")
    write_table(artifacts.scores_table, artifact_dir / "native_plan_margin_scores.parquet")
    write_table(artifacts.tail_membership_table, artifact_dir / "native_plan_margin_tail_membership.parquet")
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=artifacts.enrichment_table.num_rows,
        columns=artifacts.enrichment_table.column_names,
        inputs=[
            ScalarInputRef(kind="view_matrix", artifact_id=view_id, path=matrix_path),
            ScalarInputRef(kind="view_rows", artifact_id=view_id, path=rows_path),
            ScalarInputRef(
                kind="regulatory_interactions",
                artifact_id=interactions_path.stem,
                path=interactions_path,
            ),
        ],
        outputs=[
            ("native_plan_margin_scores.parquet", "application/x-parquet"),
            ("native_plan_margin_tail_membership.parquet", "application/x-parquet"),
        ],
        stats=artifacts.stats,
    )
