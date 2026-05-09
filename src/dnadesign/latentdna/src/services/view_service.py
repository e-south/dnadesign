"""
View services for latentdna.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError, ContractViolationError
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.result import CommandResult
from ..contracts.workspace import DerivedViewConfig
from ..io.artifact_dirs import commit_staged_artifact_dirs, stage_artifact_dir
from ..io.manifest_io import write_manifest
from ..io.parquet_io import read_table
from ..runs.recorder import record_audit
from ..sources.resolver import resolve_source, source_digest
from ..version import __version__
from ..views.derive import derive_view_artifact
from ..views.materialize import materialize_grouped_infer_sidecar_view_artifacts, materialize_view_artifact
from ..views.reduce import fit_pca_reducer_artifacts
from ..views.stats import compute_view_stats
from ..workspaces.loader import load_workspace_config
from ._artifact_inputs import artifact_input_from_manifest, dependency_artifact_input
from .memory_service import apply_memory_preflight, evaluate_materialize_preflight, evaluate_reduce_preflight
from .operation_lock_service import acquire_workspace_operation_lock


def materialize_view(
    workspace: str | Path,
    view_id: str,
    *,
    allow_memory_overage: bool = False,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    with acquire_workspace_operation_lock(
        context.output_root,
        operation="view_materialize",
        owner_id=view_id,
    ):
        view_dir = context.output_root / "views" / view_id
        if view_dir.exists() and not force:
            raise ArtifactConflictError(f"view artifact already exists: {view_dir}")
        preflight = evaluate_materialize_preflight(context, view_id=view_id)
        status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)
        staging_dir = stage_artifact_dir(context.output_root / "views", view_id)

        try:
            materialized = materialize_view_artifact(context, view_id=view_id, artifact_dir=staging_dir)
            artifact_dir = materialized[0]
            assert artifact_dir == staging_dir
            result = _write_materialized_view_manifest_and_result(
                context,
                view_id=view_id,
                materialized=materialized,
                staging_dir=staging_dir,
                status=status,
                preflight_payload=preflight.as_payload(),
                warnings=warnings,
            )
        except Exception:
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        commit_staged_artifact_dirs([(staging_dir, view_dir)], force=force)
        record_audit(
            context.output_root / "logs" / "audit",
            payload=result.model_dump(mode="json"),
            command="view_materialize",
            artifact_id=view_id,
        )
        return result


def _write_materialized_view_manifest_and_result(
    context,
    *,
    view_id: str,
    materialized: tuple[Path, int, int, str, list[str], list[str]],
    staging_dir: Path,
    status: str,
    preflight_payload: dict[str, object],
    warnings: list[str],
) -> CommandResult:
    _, rows, dims, record_key, row_columns, provenance_row_columns = materialized
    view = context.require_source_view(view_id)
    source = context.require_source(view.source)
    resolved_source = resolve_source(view.source, source, workspace_dir=context.workspace_dir)
    input_columns = (
        provenance_row_columns if view.vector.kind == "bundle_matrix" else [*provenance_row_columns, view.vector.name]
    )
    source_input_digest, source_provenance, input_digests = source_digest(resolved_source, columns=input_columns)
    source_input = ArtifactInput(
        kind="source",
        id=view.source,
        digest=source_input_digest,
    )
    manifest = ArtifactManifest(
        artifact_kind="view",
        artifact_id=view_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="view materialize",
        status=status,
        inputs=[source_input],
        input_digests=input_digests,
        freshness_basis={"kind": "source_provenance", "known": True},
        source_provenance=source_provenance,
        params={
            "analysis_dtype": context.analysis_dtype,
            "coordinate_space_id": view.coordinate_space_id,
            "record_key": record_key,
            "subject_key": source.subject_key,
            "context_key": source.context_key,
            "row_columns": row_columns,
            "vector_kind": view.vector.kind,
            "vector_column": getattr(view.vector, "name", None),
            "source": view.source,
            "role": view.role,
            "tags": view.tags,
            "memory_preflight": preflight_payload,
        },
        outputs=[
            ArtifactOutput(path="matrix.npy", media_type="application/x-npy"),
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
        ],
        stats={"rows": rows, "dims": dims},
        warnings=warnings,
    )
    write_manifest(staging_dir / "manifest.json", manifest.model_dump(mode="json"))
    final_dir = context.output_root / "views" / view_id
    return CommandResult(
        command="view materialize",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="view",
        artifact_id=view_id,
        outputs=[final_dir.as_posix()],
        inputs={"view": view_id, "source": view.source},
        input_digests=input_digests,
        warnings=warnings,
        metrics={"rows": rows, "dims": dims, "memory_preflight": preflight_payload},
        freshness_known=True,
    )


def materialize_views(
    workspace: str | Path,
    view_ids: list[str],
    *,
    allow_memory_overage: bool = False,
    force: bool = False,
) -> dict[str, CommandResult]:
    context = load_workspace_config(workspace)
    if len(set(view_ids)) != len(view_ids):
        raise ContractViolationError(f"grouped view materialization received duplicate view ids: {view_ids}")
    with acquire_workspace_operation_lock(
        context.output_root,
        operation="view_materialize_group",
        owner_id="+".join(view_ids[:3]),
    ):
        for view_id in view_ids:
            view_dir = context.output_root / "views" / view_id
            if view_dir.exists() and not force:
                raise ArtifactConflictError(f"view artifact already exists: {view_dir}")
        preflights = {view_id: evaluate_materialize_preflight(context, view_id=view_id) for view_id in view_ids}
        statuses_and_warnings = {
            view_id: apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)
            for view_id, preflight in preflights.items()
        }
        staging_dirs = {view_id: stage_artifact_dir(context.output_root / "views", view_id) for view_id in view_ids}
        try:
            materialized = materialize_grouped_infer_sidecar_view_artifacts(
                context,
                view_ids=view_ids,
                artifact_dirs=staging_dirs,
            )
            results = {
                view_id: _write_materialized_view_manifest_and_result(
                    context,
                    view_id=view_id,
                    materialized=materialized[view_id],
                    staging_dir=staging_dirs[view_id],
                    status=statuses_and_warnings[view_id][0],
                    preflight_payload=preflights[view_id].as_payload(),
                    warnings=statuses_and_warnings[view_id][1],
                )
                for view_id in view_ids
            }
        except Exception:
            import shutil

            for staging_dir in staging_dirs.values():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        commit_staged_artifact_dirs(
            [(staging_dirs[view_id], context.output_root / "views" / view_id) for view_id in view_ids],
            force=force,
        )
        for view_id, result in results.items():
            record_audit(
                context.output_root / "logs" / "audit",
                payload=result.model_dump(mode="json"),
                command="view_materialize",
                artifact_id=view_id,
            )
        return results


def derive_view(workspace: str | Path, view_id: str, *, force: bool = False) -> CommandResult:
    context = load_workspace_config(workspace)
    view = context.require_view(view_id)
    if not isinstance(view, DerivedViewConfig):
        raise ContractViolationError(f"view {view_id} is not a derived view declaration")

    view_dir = context.output_root / "views" / view_id
    if view_dir.exists() and not force:
        raise ArtifactConflictError(f"view artifact already exists: {view_dir}")
    staging_dir = stage_artifact_dir(context.output_root / "views", view_id)

    try:
        artifact_dir, rows, dims, record_key, row_columns = derive_view_artifact(
            context,
            view_id=view_id,
            artifact_dir=staging_dir,
        )
        assert artifact_dir == staging_dir
    except Exception:
        import shutil

        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    input_entries: list[ArtifactInput]
    params = {
        "analysis_dtype": context.analysis_dtype,
        "coordinate_space_id": view.coordinate_space_id,
        "record_key": record_key,
        "row_columns": row_columns,
        "derive_kind": view.derive.kind,
        "role": view.role,
        "tags": view.tags,
    }
    input_payload: dict[str, object] = {"view": view_id}
    if view.derive.kind == "vector_difference":
        input_entries = [
            dependency_artifact_input(
                context,
                kind="view_matrix",
                artifact_id=view.derive.left,
                path=context.output_root / "views" / view.derive.left / "matrix.npy",
            ),
            dependency_artifact_input(
                context,
                kind="view_matrix",
                artifact_id=view.derive.right,
                path=context.output_root / "views" / view.derive.right / "matrix.npy",
            ),
            dependency_artifact_input(
                context,
                kind="alignment_set",
                artifact_id=view.derive.alignment,
                path=context.output_root / "alignments" / view.derive.alignment / "mapping.parquet",
            ),
        ]
        params.update(
            {
                "left_view": view.derive.left,
                "right_view": view.derive.right,
                "alignment": view.derive.alignment,
            }
        )
        input_payload = {
            "view": view_id,
            "left": view.derive.left,
            "right": view.derive.right,
            "alignment": view.derive.alignment,
        }
    elif view.derive.kind == "normalize":
        input_entries = [
            dependency_artifact_input(
                context,
                kind="view_matrix",
                artifact_id=view.derive.view,
                path=context.output_root / "views" / view.derive.view / "matrix.npy",
            )
        ]
        params.update({"input_view": view.derive.view, "method": view.derive.method})
        input_payload = {"view": view_id, "input_view": view.derive.view}
    elif view.derive.kind == "aggregate_by_key":
        input_entries = [
            dependency_artifact_input(
                context,
                kind="view_matrix",
                artifact_id=view.derive.view,
                path=context.output_root / "views" / view.derive.view / "matrix.npy",
            ),
            dependency_artifact_input(
                context,
                kind="view_rows",
                artifact_id=view.derive.view,
                path=context.output_root / "views" / view.derive.view / "rows.parquet",
            ),
        ]
        params.update({"input_view": view.derive.view, "key": view.derive.key, "aggregation": view.derive.aggregation})
        input_payload = {"view": view_id, "input_view": view.derive.view, "key": view.derive.key}
    elif view.derive.kind == "apply_reducer":
        input_entries = [
            dependency_artifact_input(
                context,
                kind="view_matrix",
                artifact_id=view.derive.view,
                path=context.output_root / "views" / view.derive.view / "matrix.npy",
            ),
            dependency_artifact_input(
                context,
                kind="reducer",
                artifact_id=view.derive.reducer,
                path=context.output_root / "reducers" / view.derive.reducer / "state.npz",
            ),
        ]
        params.update({"input_view": view.derive.view, "reducer": view.derive.reducer})
        input_payload = {"view": view_id, "input_view": view.derive.view, "reducer": view.derive.reducer}
    elif view.derive.kind in {"concatenate", "block_normalized_concatenate"}:
        input_entries = [
            dependency_artifact_input(
                context,
                kind="view_matrix",
                artifact_id=input_view,
                path=context.output_root / "views" / input_view / "matrix.npy",
            )
            for input_view in view.derive.inputs
        ]
        params.update({"input_views": view.derive.inputs})
        if view.derive.kind == "block_normalized_concatenate":
            params.update(
                {
                    "center": view.derive.center,
                    "scale": view.derive.scale,
                    "block_norm": view.derive.block_norm,
                    "nonfinite_policy": view.derive.nonfinite_policy,
                    "zero_variance_policy": view.derive.zero_variance_policy,
                    "zero_row_policy": view.derive.zero_row_policy,
                }
            )
        input_payload = {"view": view_id, "input_views": view.derive.inputs}
    else:  # pragma: no cover - constrained by workspace schema
        raise ContractViolationError(f"unsupported derived view kind: {view.derive.kind}")

    manifest = ArtifactManifest(
        artifact_kind="view",
        artifact_id=view_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="view derive",
        inputs=input_entries,
        params=params,
        outputs=[
            ArtifactOutput(path="matrix.npy", media_type="application/x-npy"),
            ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
        ],
        stats={"rows": rows, "dims": dims},
    )
    write_manifest(staging_dir / "manifest.json", manifest.model_dump(mode="json"))
    commit_staged_artifact_dirs([(staging_dir, view_dir)], force=force)
    result = CommandResult(
        command="view derive",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="view",
        artifact_id=view_id,
        outputs=[view_dir.as_posix()],
        inputs=input_payload,
        metrics={"rows": rows, "dims": dims},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="view_derive",
        artifact_id=view_id,
    )
    return result


def reduce_view(
    workspace: str | Path,
    view_id: str,
    *,
    reducer_id: str,
    dims: int,
    sample_id: str | None,
    alignment_id: str | None,
    reduced_view_id: str | None,
    allow_memory_overage: bool = False,
    force: bool = False,
) -> CommandResult:
    context = load_workspace_config(workspace)
    reducer_dir = context.output_root / "reducers" / reducer_id
    if reducer_dir.exists() and not force:
        raise ArtifactConflictError(f"reducer artifact already exists: {reducer_dir}")

    reduced_view_dir = None if reduced_view_id is None else context.output_root / "reduced_views" / reduced_view_id
    if reduced_view_dir is not None and reduced_view_dir.exists() and not force:
        raise ArtifactConflictError(f"reduced view artifact already exists: {reduced_view_dir}")
    reducer_staging_dir = stage_artifact_dir(context.output_root / "reducers", reducer_id)
    reduced_view_staging_dir = (
        None if reduced_view_id is None else stage_artifact_dir(context.output_root / "reduced_views", reduced_view_id)
    )
    preflight = evaluate_reduce_preflight(
        context,
        view_id=view_id,
        dims=dims,
        sample_id=sample_id,
        alignment_id=alignment_id,
        reduced_view_id=reduced_view_id,
    )
    status, warnings = apply_memory_preflight(preflight, allow_memory_overage=allow_memory_overage)

    try:
        (
            staged_reducer_dir,
            staged_reduced_view_dir,
            fit_rows,
            output_dims,
            scope_kind,
            scope_id,
            pca_method,
        ) = fit_pca_reducer_artifacts(
            context,
            view_id=view_id,
            reducer_id=reducer_id,
            dims=dims,
            sample_id=sample_id,
            alignment_id=alignment_id,
            reduced_view_id=reduced_view_id,
            reducer_dir=reducer_staging_dir,
            reduced_view_dir=reduced_view_staging_dir,
        )
        assert staged_reducer_dir == reducer_staging_dir
        assert staged_reduced_view_dir == reduced_view_staging_dir
    except Exception:
        import shutil

        shutil.rmtree(reducer_staging_dir, ignore_errors=True)
        if reduced_view_staging_dir is not None:
            shutil.rmtree(reduced_view_staging_dir, ignore_errors=True)
        raise

    reducer_manifest = ArtifactManifest(
        artifact_kind="reducer",
        artifact_id=reducer_id,
        workspace_id=context.workspace_id,
        created_at=datetime.now(UTC).isoformat(),
        tool_version=__version__,
        command="view reduce",
        status=status,
        inputs=[
            dependency_artifact_input(
                context,
                kind="view_matrix",
                artifact_id=view_id,
                path=context.output_root / "views" / view_id / "matrix.npy",
            )
        ],
        params={
            "method": "pca",
            "pca_method": pca_method,
            "view_id": view_id,
            "fit_scope_kind": scope_kind,
            "fit_scope_id": scope_id,
            "output_dims": output_dims,
            "memory_preflight": preflight.as_payload(),
        },
        outputs=[
            ArtifactOutput(path="state.npz", media_type="application/x-npz"),
            ArtifactOutput(path="summary.json", media_type="application/json"),
        ],
        stats={"rows": fit_rows, "dims": output_dims},
        warnings=warnings,
    )
    write_manifest(reducer_staging_dir / "manifest.json", reducer_manifest.model_dump(mode="json"))

    outputs = [reducer_dir.as_posix()]
    metrics: dict[str, int | str | None] = {"fit_rows": fit_rows, "dims": output_dims}
    staged_pairs: list[tuple[Path, Path]] = [(reducer_staging_dir, reducer_dir)]
    if reduced_view_staging_dir is not None and reduced_view_dir is not None and reduced_view_id is not None:
        reduced_rows = read_table(reduced_view_staging_dir / "rows.parquet").num_rows
        reduced_manifest = ArtifactManifest(
            artifact_kind="reduced_view",
            artifact_id=reduced_view_id,
            workspace_id=context.workspace_id,
            created_at=datetime.now(UTC).isoformat(),
            tool_version=__version__,
            command="view reduce",
            inputs=[
                artifact_input_from_manifest(
                    kind="reducer",
                    artifact_id=reducer_id,
                    digest_path=reducer_staging_dir / "manifest.json",
                    recorded_path=reducer_dir / "manifest.json",
                ),
                dependency_artifact_input(
                    context,
                    kind="view_matrix",
                    artifact_id=view_id,
                    path=context.output_root / "views" / view_id / "matrix.npy",
                ),
            ],
            params={
                "source_view_id": view_id,
                "reducer_id": reducer_id,
                "coordinate_space_id": f"pca_{reducer_id}",
                "fit_scope_kind": scope_kind,
                "fit_scope_id": scope_id,
                "memory_preflight": preflight.as_payload(),
            },
            outputs=[
                ArtifactOutput(path="matrix.npy", media_type="application/x-npy"),
                ArtifactOutput(path="rows.parquet", media_type="application/x-parquet"),
            ],
            stats={"rows": reduced_rows, "dims": output_dims},
            status=status,
            warnings=warnings,
        )
        write_manifest(reduced_view_staging_dir / "manifest.json", reduced_manifest.model_dump(mode="json"))
        staged_pairs.append((reduced_view_staging_dir, reduced_view_dir))
        outputs.append(reduced_view_dir.as_posix())
        metrics["reduced_view_rows"] = reduced_rows

    commit_staged_artifact_dirs(staged_pairs, force=force)

    result = CommandResult(
        command="view reduce",
        workspace_id=context.workspace_id,
        status=status,
        artifact_kind="reducer",
        artifact_id=reducer_id,
        outputs=outputs,
        inputs={"view": view_id, "sample": sample_id, "alignment": alignment_id, "reduced_view": reduced_view_id},
        warnings=warnings,
        metrics=metrics,
    )
    result.metrics["memory_preflight"] = preflight.as_payload()
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="view_reduce",
        artifact_id=reducer_id,
    )
    return result


def view_stats(workspace: str | Path, view_id: str) -> dict[str, object]:
    context = load_workspace_config(workspace)
    return compute_view_stats(context, view_id=view_id)
