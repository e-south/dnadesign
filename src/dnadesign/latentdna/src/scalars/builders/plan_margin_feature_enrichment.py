"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/scalars/builders/plan_margin_feature_enrichment.py

Scalar builder for plan-margin feature-term enrichment.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ...contracts.errors import ContractViolationError, MissingArtifactError
from ...enrichments.plan_margin_feature_enrichment import build_plan_margin_feature_enrichment_artifact
from ...io.json_io import read_json
from ...io.parquet_io import read_table, write_table
from ...workspaces.loader import WorkspaceContext
from ..common import BuiltScalarArtifact, ScalarInputRef, _optional_param, _require_param, _workspace_input_path

_CONTRACT_NAME = "plan_margin_feature_enrichment"


def _mapping_param(params: dict[str, object], key: str) -> dict[str, object]:
    raw_value = _require_param(params, key)
    if not isinstance(raw_value, dict):
        raise ContractViolationError(f"{_CONTRACT_NAME} requires mapping param {key!r}")
    return dict(raw_value)


def _source_scalar_table(context: WorkspaceContext, source_scalar: str, filename: str) -> Path:
    path = context.output_root / "scalars" / source_scalar / filename
    if not path.is_file():
        raise MissingArtifactError(f"{_CONTRACT_NAME} is missing source scalar table: {path}")
    return path


def _validate_source_scalar_manifest(context: WorkspaceContext, source_scalar: str, filenames: list[str]) -> None:
    manifest_path = context.output_root / "scalars" / source_scalar / "manifest.json"
    if not manifest_path.is_file():
        raise MissingArtifactError(f"{_CONTRACT_NAME} is missing source scalar manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("artifact_kind") != "scalar_table":
        raise ContractViolationError(f"{_CONTRACT_NAME} source_scalar {source_scalar!r} is not a scalar_table artifact")
    if manifest.get("artifact_id") != source_scalar:
        raise ContractViolationError(
            f"{_CONTRACT_NAME} source_scalar manifest id mismatch: "
            f"expected {source_scalar!r}, found {manifest.get('artifact_id')!r}"
        )
    if manifest.get("status", "ok") != "ok":
        raise ContractViolationError(
            f"{_CONTRACT_NAME} source_scalar {source_scalar!r} manifest is not ok: {manifest.get('status')!r}"
        )
    output_paths = {
        str(output.get("path") or "").strip() for output in manifest.get("outputs", []) if isinstance(output, dict)
    }
    missing = sorted(filename for filename in filenames if filename not in output_paths)
    if missing:
        raise ContractViolationError(
            f"{_CONTRACT_NAME} source_scalar {source_scalar!r} manifest does not declare required outputs: {missing}"
        )


def _workspace_table_path(context: WorkspaceContext, raw_path: object, *, param_name: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise ContractViolationError(f"{_CONTRACT_NAME} requires {param_name}")
    path = _workspace_input_path(context, text)
    if not path.is_file():
        raise MissingArtifactError(f"{_CONTRACT_NAME} is missing feature table: {path}")
    return path


def build_plan_margin_feature_enrichment_scalar(
    context: WorkspaceContext,
    *,
    artifact_dir: Path,
    params: dict[str, object],
) -> BuiltScalarArtifact:
    """Build a feature-term enrichment scalar from persisted plan-margin tails."""

    source_scalar = str(_require_param(params, "source_scalar"))
    scores_table = str(_require_param(params, "scores_table"))
    tail_membership_table = str(_require_param(params, "tail_membership_table"))
    _validate_source_scalar_manifest(
        context,
        source_scalar,
        [scores_table, tail_membership_table],
    )
    scores_path = _source_scalar_table(
        context,
        source_scalar,
        scores_table,
    )
    tails_path = _source_scalar_table(
        context,
        source_scalar,
        tail_membership_table,
    )
    feature_membership = _mapping_param(params, "feature_membership")
    feature_path = _workspace_table_path(context, feature_membership.get("path"), param_name="feature_membership.path")
    namespace_filter = feature_membership.get("namespace_filter")
    artifacts = build_plan_margin_feature_enrichment_artifact(
        scores_table=read_table(scores_path),
        tail_membership_table=read_table(tails_path),
        feature_table=read_table(feature_path),
        subject_column=str(feature_membership.get("subject_column") or "usr_id"),
        feature_id_column=str(feature_membership.get("feature_id_column") or "feature_id"),
        feature_label_column=str(feature_membership.get("feature_label_column") or "feature_label"),
        feature_namespace_column=(
            str(feature_membership.get("feature_namespace_column"))
            if feature_membership.get("feature_namespace_column") is not None
            else None
        ),
        namespace_filter=(str(namespace_filter) if namespace_filter is not None else None),
        exclude_label_prefixes=feature_membership.get("exclude_label_prefixes") or [],
        source_metadata_columns=feature_membership.get("source_metadata_columns") or [],
        min_global_subjects=int(_optional_param(params, "min_global_subjects", default=10)),
        min_tail_hits=int(_optional_param(params, "min_tail_hits", default=3)),
        rank_test_alternative=str(_optional_param(params, "rank_test_alternative", default="greater")),
        common_features=_optional_param(params, "common_features", default=[]),
    )
    write_table(artifacts.enrichment_table, artifact_dir / "table.parquet")
    rank_tests_filename = "plan_margin_feature_rank_tests.parquet"
    write_table(artifacts.rank_tests_table, artifact_dir / rank_tests_filename)
    return BuiltScalarArtifact(
        artifact_dir=artifact_dir,
        rows=artifacts.enrichment_table.num_rows,
        columns=artifacts.enrichment_table.column_names,
        inputs=[
            ScalarInputRef(kind="scalar_table", artifact_id=source_scalar, path=scores_path),
            ScalarInputRef(kind="scalar_table", artifact_id=source_scalar, path=tails_path),
            ScalarInputRef(kind="feature_membership_table", artifact_id=feature_path.stem, path=feature_path),
        ],
        outputs=[(rank_tests_filename, "application/x-parquet")],
        stats={"source_scalar": source_scalar, **artifacts.stats},
    )
