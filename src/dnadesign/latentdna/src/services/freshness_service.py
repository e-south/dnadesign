"""
Freshness evaluation helpers for latentdna artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..contracts.errors import ContractViolationError
from ..contracts.workspace import DerivedViewConfig, SourceBackedViewConfig
from ..io.json_io import read_json
from ..io.parquet_io import read_table
from ..metrics.definitions import metric_definition_digests
from ..sources.provenance import (
    OVERLAY_INVENTORY_DIGEST_MODE,
    OVERLAY_LEDGER_PAYLOAD_DIGEST_MODE,
    source_provenance_digest,
)
from ..sources.resolver import inspect_source_schema, resolve_source
from ..views.row_contracts import source_backed_view_row_contract
from ..workspaces.loader import WorkspaceContext
from ._artifact_inputs import artifact_kind_for_input_dependency
from ._artifacts import artifact_exists, artifact_manifest_path


@dataclass(slots=True)
class FreshnessCache:
    artifact_results: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)
    path_digests: dict[str, tuple[bool, str | None]] = field(default_factory=dict)
    overlay_inventory_digests: dict[str, str] = field(default_factory=dict)
    overlay_ledger_payload_digests: dict[str, str] = field(default_factory=dict)


def _resolve_path_digest(path: Path, *, cache: FreshnessCache) -> tuple[bool, str | None]:
    key = path.as_posix()
    cached = cache.path_digests.get(key)
    if cached is not None:
        return cached
    if not path.exists():
        result = (False, None)
    else:
        result = (True, source_provenance_digest({"path": path.as_posix()}))
    cache.path_digests[key] = result
    return result


def _resolve_overlay_inventory_digest(path: Path, *, cache: FreshnessCache) -> str:
    key = path.resolve().as_posix()
    cached = cache.overlay_inventory_digests.get(key)
    if cached is not None:
        return cached
    digest = source_provenance_digest({"path": key, "digest_mode": OVERLAY_INVENTORY_DIGEST_MODE})
    cache.overlay_inventory_digests[key] = digest
    return digest


def _resolve_overlay_ledger_payload_digest(path: Path, *, cache: FreshnessCache) -> str:
    key = path.resolve().as_posix()
    cached = cache.overlay_ledger_payload_digests.get(key)
    if cached is not None:
        return cached
    digest = source_provenance_digest({"path": key, "digest_mode": OVERLAY_LEDGER_PAYLOAD_DIGEST_MODE})
    cache.overlay_ledger_payload_digests[key] = digest
    return digest


def _view_config_freshness_reasons(
    context: WorkspaceContext,
    *,
    artifact_id: str,
    manifest: dict[str, object],
) -> tuple[list[str], bool]:
    if not all(hasattr(context, attribute) for attribute in ("require_view", "config")):
        return [], True

    params = manifest.get("params")
    if not isinstance(params, dict):
        return [f"freshness unknown: view manifest lacks params for view:{artifact_id}"], False

    try:
        view = context.require_view(artifact_id)
    except ContractViolationError as exc:
        return [f"stale view config for view:{artifact_id}: {exc}"], True
    except Exception as exc:
        return [f"freshness unknown: could not resolve view config for view:{artifact_id}: {exc}"], False

    if isinstance(view, SourceBackedViewConfig):
        try:
            source = context.require_source(view.source)
            resolved = resolve_source(view.source, source, workspace_dir=context.workspace_dir)
            row_contract = source_backed_view_row_contract(
                context,
                source_id=view.source,
                source=source,
                available_columns=inspect_source_schema(resolved)["columns"],
            )
        except ContractViolationError as exc:
            return [f"stale view config for view:{artifact_id}: {exc}"], True
        except Exception as exc:
            return [f"freshness unknown: could not resolve view config for view:{artifact_id}: {exc}"], False

        reasons: list[str] = []
        expected_pairs = {
            "analysis_dtype": context.analysis_dtype,
            "coordinate_space_id": view.coordinate_space_id,
            "record_key": source.record_key,
            "subject_key": source.subject_key,
            "context_key": source.context_key,
            "vector_kind": view.vector.kind,
            "vector_column": getattr(view.vector, "name", None),
            "source": view.source,
            "role": view.role,
            "tags": view.tags,
        }
        for key, expected in expected_pairs.items():
            if params.get(key) != expected:
                reasons.append(f"stale view config for view:{artifact_id}: {key}")

        expected_row_columns = row_contract.materialized_row_columns
        recorded_row_columns = [str(value) for value in params.get("row_columns", []) if value is not None]
        missing_row_columns = [column for column in expected_row_columns if column not in recorded_row_columns]
        if missing_row_columns:
            reasons.append(f"stale view config for view:{artifact_id}: missing row columns {missing_row_columns}")
        return reasons, True

    if not isinstance(view, DerivedViewConfig):
        return [f"freshness unknown: unsupported view config type for view:{artifact_id}"], False

    reasons: list[str] = []
    expected_pairs = {
        "analysis_dtype": context.analysis_dtype,
        "coordinate_space_id": view.coordinate_space_id,
        "derive_kind": view.derive.kind,
        "role": view.role,
        "tags": view.tags,
    }
    if view.derive.kind == "vector_difference":
        expected_pairs.update(
            {
                "left_view": view.derive.left,
                "right_view": view.derive.right,
                "alignment": view.derive.alignment,
            }
        )
    elif view.derive.kind == "normalize":
        expected_pairs.update({"input_view": view.derive.view, "method": view.derive.method})
    elif view.derive.kind == "aggregate_by_key":
        expected_pairs.update(
            {
                "input_view": view.derive.view,
                "key": view.derive.key,
                "aggregation": view.derive.aggregation,
            }
        )
    elif view.derive.kind == "apply_reducer":
        expected_pairs.update({"input_view": view.derive.view, "reducer": view.derive.reducer})
    elif view.derive.kind == "concatenate":
        expected_pairs.update({"input_views": list(view.derive.inputs)})

    for key, expected in expected_pairs.items():
        if params.get(key) != expected:
            reasons.append(f"stale view config for view:{artifact_id}: {key}")
    return reasons, True


def _notebook_plot_ids(context: WorkspaceContext, notebook_id: str) -> list[str]:
    notebook = context.require_notebook(notebook_id)
    return list(
        notebook.ordered_plots or context.require_deliverable(notebook.default_deliverable).outputs.get("plots", [])
    )


def _notebook_missing_plot_ids(context: WorkspaceContext, notebook_id: str) -> list[str]:
    missing_plot_ids: list[str] = []
    for plot_id in _notebook_plot_ids(context, notebook_id):
        manifest_path = artifact_manifest_path(context, artifact_kind="plot", artifact_id=plot_id)
        if not manifest_path.exists():
            missing_plot_ids.append(plot_id)
    return missing_plot_ids


def _notebook_config_freshness_reasons(
    context: WorkspaceContext,
    *,
    artifact_id: str,
    manifest: dict[str, object],
) -> tuple[list[str], bool]:
    if not hasattr(context, "require_notebook"):
        return [], True
    try:
        notebook = context.require_notebook(artifact_id)
        expected_plot_ids = _notebook_plot_ids(context, artifact_id)
        expected_missing_plot_ids = _notebook_missing_plot_ids(context, artifact_id)
    except ContractViolationError as exc:
        return [f"stale notebook config for notebook:{artifact_id}: {exc}"], True
    except Exception as exc:
        return [f"freshness unknown: could not resolve notebook config for notebook:{artifact_id}: {exc}"], False

    params = manifest.get("params")
    if not isinstance(params, dict):
        return [f"freshness unknown: notebook manifest lacks params for notebook:{artifact_id}"], False

    reasons: list[str] = []
    expected_pairs = {
        "kind": notebook.kind,
        "title": notebook.title,
        "default_deliverable": notebook.default_deliverable,
        "default_surface": notebook.default_surface,
        "ordered_plot_ids": expected_plot_ids,
        "missing_ordered_plots": expected_missing_plot_ids,
    }
    for key, expected in expected_pairs.items():
        if params.get(key) != expected:
            reasons.append(f"stale notebook config for notebook:{artifact_id}: {key}")
    return reasons, True


def _declared_output_freshness_reasons(
    *,
    artifact_kind: str,
    artifact_id: str,
    manifest: dict[str, object],
    manifest_path: Path,
) -> tuple[list[str], bool]:
    reasons: list[str] = []
    for output in manifest.get("outputs", []) or []:
        if not isinstance(output, dict):
            continue
        path_text = str(output.get("path") or "").strip()
        if not path_text:
            continue
        if not (manifest_path.parent / path_text).exists():
            reasons.append(f"artifact payload is missing for {artifact_kind}:{artifact_id}: {path_text}")
    return reasons, True


def _notebook_health_freshness_reasons(
    context: WorkspaceContext,
    *,
    artifact_id: str,
) -> tuple[list[str], bool]:
    health_path = context.output_root / "notebooks" / artifact_id / "health.json"
    if not health_path.is_file():
        return [f"notebook health artifact is missing for notebook:{artifact_id}"], True
    try:
        payload = read_json(health_path)
    except Exception as exc:
        return [f"freshness unknown: could not read notebook health for notebook:{artifact_id}: {exc}"], False
    workspace_id = str(payload.get("workspace_id") or "")
    if workspace_id and workspace_id != context.workspace_id:
        return [f"stale notebook health for notebook:{artifact_id}: workspace_id={workspace_id}"], True
    notebook_id = str(payload.get("notebook_id") or "")
    if notebook_id and notebook_id != artifact_id:
        return [f"stale notebook health for notebook:{artifact_id}: notebook_id={notebook_id}"], True
    checks_payload = payload.get("checks")
    invalid_checks_payload = checks_payload is not None and not isinstance(checks_payload, dict)
    checks_dict = checks_payload if isinstance(checks_payload, dict) else {}
    failing_checks = [str(name) for name, value in checks_dict.items() if str(name).strip() and not bool(value)]
    warnings = [str(item).strip() for item in payload.get("warnings", []) or [] if str(item).strip()]
    status = str(payload.get("status") or "")
    if status == "ok":
        contradictions: list[str] = []
        if invalid_checks_payload:
            contradictions.append("invalid checks payload")
        if failing_checks:
            contradictions.append("failing checks: " + ", ".join(sorted(failing_checks)))
        if warnings:
            contradictions.append("; ".join(warnings))
        if contradictions:
            return [
                f"stale notebook health for notebook:{artifact_id}: "
                + "; ".join(part for part in contradictions if part)
            ], True
        return [], True
    detail_parts: list[str] = []
    if invalid_checks_payload:
        detail_parts.append("invalid checks payload")
    if failing_checks:
        detail_parts.append("failing checks: " + ", ".join(sorted(failing_checks)))
    if warnings:
        detail_parts.append("; ".join(warnings))
    detail = "; ".join(detail_parts) if detail_parts else (status or "status unavailable")
    return [f"notebook health requires attention for notebook:{artifact_id}: {detail}"], True


def _plot_config_freshness_reasons(
    context: WorkspaceContext,
    *,
    artifact_id: str,
    manifest: dict[str, object],
) -> tuple[list[str], bool]:
    if artifact_id not in context.config.plots:
        return [], True
    from ..plots.recipes import resolve_plot_spec
    from ._plot_payloads import manifest_params_for_plot

    current_spec = resolve_plot_spec(
        plots=context.config.plots,
        plot_id=artifact_id,
        kind=None,
        projection_ids=[],
        panel_titles=[],
        enrichment_id=None,
        distance_id=None,
        scalar_id=None,
        scalar_ids=[],
        agreement_id=None,
        agreement_ids=[],
        reducer_id=None,
        left_cluster_id=None,
        right_cluster_id=None,
        value_column=None,
        x_column=None,
        y_column=None,
        color_column=None,
        shape_column=None,
        render_mode=None,
        label_column=None,
        label_values=[],
    )
    recorded_params = manifest.get("params")
    if not isinstance(recorded_params, dict):
        return [f"freshness unknown: plot manifest lacks params for plot:{artifact_id}"], False
    if recorded_params != manifest_params_for_plot(current_spec):
        return [f"stale plot config for plot:{artifact_id}"], True
    return [], True


def _scalar_metric_definition_freshness_reasons(
    context: WorkspaceContext,
    *,
    artifact_id: str,
    manifest: dict[str, object],
    manifest_path: Path,
) -> tuple[list[str], bool]:
    table_path = manifest_path.parent / "table.parquet"
    if not table_path.is_file():
        return [], True
    try:
        table = read_table(table_path)
    except Exception as exc:
        return [f"freshness unknown: could not read scalar table for scalar_table:{artifact_id}: {exc}"], False
    if "metric_id" not in table.column_names:
        return [], True
    metric_ids = {
        str(value).strip() for value in table["metric_id"].to_pylist() if value is not None and str(value).strip()
    }
    if not metric_ids:
        return [], True
    params = manifest.get("params")
    if not isinstance(params, dict):
        return [f"freshness unknown: scalar manifest lacks params for scalar_table:{artifact_id}"], False
    recorded = params.get("metric_definition_digests")
    if not isinstance(recorded, dict):
        return [
            f"stale scalar metric definition provenance for scalar_table:{artifact_id}: "
            "missing metric_definition_digests"
        ], True
    try:
        current = metric_definition_digests(metric_ids, config=context.config)
    except ContractViolationError as exc:
        return [f"stale scalar metric definition for scalar_table:{artifact_id}: {exc}"], True

    reasons: list[str] = []
    for metric_id in sorted(metric_ids):
        if str(recorded.get(metric_id) or "") != current[metric_id]:
            reasons.append(f"stale scalar metric definition for scalar_table:{artifact_id}: {metric_id}")
    extra_recorded_ids = sorted(str(metric_id) for metric_id in set(recorded) - metric_ids)
    if extra_recorded_ids:
        reasons.append(
            f"stale scalar metric definition provenance for scalar_table:{artifact_id}: "
            f"extra metric ids {extra_recorded_ids}"
        )
    return reasons, True


def _scalar_build_recipe_freshness_reasons(
    context: WorkspaceContext,
    *,
    artifact_id: str,
    manifest: dict[str, object],
) -> tuple[list[str], bool]:
    if not all(hasattr(context, attribute) for attribute in ("config",)):
        return [], True

    expected_params: list[tuple[str, dict[str, object]]] = []
    for recipe_id, recipe in context.config.recipes.items():
        for step in recipe.steps:
            if step.op != "scalar.build":
                continue
            step_params = dict(step.params)
            step_scalar = str(step_params.get("scalar_id") or step_params.get("scalar") or "").strip()
            if step_scalar != artifact_id:
                continue
            builder_kind = str(step_params.get("kind") or "").strip()
            if not builder_kind:
                return [
                    f"freshness unknown: scalar.build recipe step {recipe_id}.{step.id} "
                    f"lacks kind for scalar_table:{artifact_id}"
                ], False
            builder_params = {
                key: value for key, value in step_params.items() if key not in {"scalar_id", "scalar", "kind"}
            }
            expected_params.append((f"{recipe_id}.{step.id}", {"builder_kind": builder_kind, **builder_params}))

    if not expected_params:
        return [], True

    recorded_params = manifest.get("params")
    if not isinstance(recorded_params, dict):
        return [f"freshness unknown: scalar manifest lacks params for scalar_table:{artifact_id}"], False

    recorded_build_params = {key: value for key, value in recorded_params.items() if key != "metric_definition_digests"}
    if any(recorded_build_params == expected for _, expected in expected_params):
        return [], True

    expected_locations = ", ".join(location for location, _ in expected_params)
    return [
        f"stale scalar build config for scalar_table:{artifact_id}: "
        f"manifest params do not match current recipe step(s) {expected_locations}"
    ], True


def evaluate_artifact_freshness(
    context: WorkspaceContext,
    *,
    artifact_kind: str,
    artifact_id: str,
    _stack: set[tuple[str, str]] | None = None,
    cache: FreshnessCache | None = None,
) -> dict[str, object]:
    if _stack is None:
        _stack = set()
    if cache is None:
        cache = FreshnessCache()
    key = (artifact_kind, artifact_id)
    if key in _stack:
        return {
            "status": "attention",
            "reason": f"freshness unknown: recursive dependency detected for {artifact_kind}:{artifact_id}",
            "known": False,
        }
    cached = cache.artifact_results.get(key)
    if cached is not None:
        return cached
    if not artifact_exists(context, artifact_kind=artifact_kind, artifact_id=artifact_id):
        result = {"status": "missing", "reason": f"artifact is missing: {artifact_kind}:{artifact_id}", "known": True}
        cache.artifact_results[key] = result
        return result

    manifest_path = artifact_manifest_path(context, artifact_kind=artifact_kind, artifact_id=artifact_id)
    manifest = context.read_manifest(manifest_path)
    stack = set(_stack)
    stack.add(key)
    result = evaluate_manifest_freshness(
        context,
        manifest=manifest,
        manifest_path=manifest_path,
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        _stack=stack,
        cache=cache,
    )
    cache.artifact_results[key] = result
    return result


def evaluate_manifest_freshness(
    context: WorkspaceContext,
    *,
    manifest: dict[str, object],
    manifest_path: Path,
    artifact_kind: str,
    artifact_id: str,
    _stack: set[tuple[str, str]],
    cache: FreshnessCache,
) -> dict[str, object]:
    if manifest.get("status") == "error":
        return {
            "status": "error",
            "reason": f"artifact manifest is marked error: {artifact_kind}:{artifact_id}",
            "known": True,
        }

    reasons: list[str] = []
    known = True
    checked_any = False

    output_reasons, output_known = _declared_output_freshness_reasons(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    if output_reasons:
        reasons.extend(output_reasons)
    known = known and output_known

    for entry in manifest.get("source_provenance", []) or []:
        if not isinstance(entry, dict):
            continue
        if artifact_kind == "plot" and entry.get("id") == "workspace_config":
            continue
        path_text = str(entry.get("path") or "")
        recorded_digest = str(entry.get("digest") or "")
        if not path_text or not recorded_digest:
            known = False
            reasons.append(f"freshness unknown: incomplete source provenance for {artifact_kind}:{artifact_id}")
            continue
        checked_any = True
        path = Path(path_text)
        digest_mode = str(entry.get("digest_mode") or "")
        if digest_mode == OVERLAY_INVENTORY_DIGEST_MODE:
            if not path.exists():
                known = False
                reasons.append(f"freshness unknown: source path is missing: {path_text}")
                continue
            current_digest = _resolve_overlay_inventory_digest(path, cache=cache)
            if current_digest != recorded_digest:
                namespace = str(entry.get("namespace") or entry.get("id") or path.name)
                reasons.append(
                    f"stale freshness: source overlay inventory for {artifact_kind}:{artifact_id}: {namespace}"
                )
            continue
        if digest_mode == OVERLAY_LEDGER_PAYLOAD_DIGEST_MODE:
            if not path.exists():
                known = False
                reasons.append(f"freshness unknown: source path is missing: {path_text}")
                continue
            current_digest = _resolve_overlay_ledger_payload_digest(path, cache=cache)
            if current_digest != recorded_digest:
                reasons.append(f"stale source freshness for {artifact_kind}:{artifact_id}: {entry.get('id')}")
            continue
        path_exists, current_digest = _resolve_path_digest(path, cache=cache)
        if not path_exists:
            known = False
            reasons.append(f"freshness unknown: source path is missing: {path_text}")
            continue
        assert current_digest is not None
        if current_digest != recorded_digest:
            reasons.append(f"stale source freshness for {artifact_kind}:{artifact_id}: {entry.get('id')}")

    for input_entry in manifest.get("inputs", []) or []:
        if not isinstance(input_entry, dict):
            continue
        input_kind = str(input_entry.get("kind") or "")
        input_id = str(input_entry.get("id") or "")
        path_text = input_entry.get("path")
        recorded_digest = str(input_entry.get("digest") or "")
        upstream_kind = artifact_kind_for_input_dependency(input_kind)
        use_recorded_path_digest = (
            path_text is not None
            and recorded_digest
            and (upstream_kind is None or Path(str(path_text)).name == "manifest.json")
        )
        if use_recorded_path_digest:
            checked_any = True
            path = Path(str(path_text))
            path_exists, current_digest = _resolve_path_digest(path, cache=cache)
            if not path_exists:
                known = False
                reasons.append(f"freshness unknown: input path is missing: {path}")
            else:
                assert current_digest is not None
                if current_digest != recorded_digest:
                    reasons.append(f"stale input digest for {input_kind}:{input_id}")

        if upstream_kind is None:
            if not path_text and input_kind not in {"source", "landmark_source"}:
                known = False
                reasons.append(f"freshness unknown for input {input_kind}:{input_id}")
            continue
        checked_any = True
        upstream = evaluate_artifact_freshness(
            context,
            artifact_kind=upstream_kind,
            artifact_id=input_id,
            _stack=_stack,
            cache=cache,
        )
        if upstream["status"] != "ok":
            known = known and bool(upstream.get("known"))
            reason = str(upstream.get("reason") or f"{upstream_kind}:{input_id} is not fresh")
            reasons.append(f"freshness depends on {upstream_kind}:{input_id}: {reason}")

    if artifact_kind == "plot":
        checked_any = True
        plot_config_reasons, plot_config_known = _plot_config_freshness_reasons(
            context,
            artifact_id=artifact_id,
            manifest=manifest,
        )
        if plot_config_reasons:
            reasons.extend(plot_config_reasons)
        known = known and plot_config_known

    if reasons:
        return {"status": "attention", "reason": reasons[0], "known": known, "reasons": reasons}
    if artifact_kind == "scalar_table":
        scalar_config_reasons, scalar_config_known = _scalar_build_recipe_freshness_reasons(
            context,
            artifact_id=artifact_id,
            manifest=manifest,
        )
        if scalar_config_reasons:
            return {
                "status": "attention",
                "reason": scalar_config_reasons[0],
                "known": scalar_config_known,
                "reasons": scalar_config_reasons,
            }
        scalar_reasons, scalar_known = _scalar_metric_definition_freshness_reasons(
            context,
            artifact_id=artifact_id,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        if scalar_reasons:
            return {
                "status": "attention",
                "reason": scalar_reasons[0],
                "known": scalar_known,
                "reasons": scalar_reasons,
            }
    if artifact_kind == "view":
        view_reasons, view_known = _view_config_freshness_reasons(
            context,
            artifact_id=artifact_id,
            manifest=manifest,
        )
        if view_reasons:
            return {
                "status": "attention",
                "reason": view_reasons[0],
                "known": view_known,
                "reasons": view_reasons,
            }
    if artifact_kind == "notebook":
        notebook_reasons, notebook_known = _notebook_config_freshness_reasons(
            context,
            artifact_id=artifact_id,
            manifest=manifest,
        )
        if notebook_reasons:
            return {
                "status": "attention",
                "reason": notebook_reasons[0],
                "known": notebook_known,
                "reasons": notebook_reasons,
            }
        health_reasons, health_known = _notebook_health_freshness_reasons(context, artifact_id=artifact_id)
        if health_reasons:
            return {
                "status": "attention",
                "reason": health_reasons[0],
                "known": health_known,
                "reasons": health_reasons,
            }
    if not checked_any:
        return {
            "status": "attention",
            "reason": f"freshness unknown: manifest lacks recorded input provenance for {artifact_kind}:{artifact_id}",
            "known": False,
        }
    return {"status": "ok", "reason": None, "known": True}
