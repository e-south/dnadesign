"""
Plot services for latentdna.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError
from ..contracts.ids import validate_identifier
from ..contracts.manifest import ArtifactManifest, ArtifactOutput
from ..contracts.plot import ResolvedPlotSpec
from ..contracts.result import CommandResult
from ..io.json_io import write_json
from ..io.manifest_io import write_manifest
from ..plots.recipes import resolve_plot_spec
from ..plots.render import render_plot_artifact
from ..runs.recorder import record_audit
from ..sources.provenance import source_provenance_digest
from ..version import __version__
from ..workspaces.loader import WorkspaceContext, load_workspace_config
from ..workspaces.plot_semantics import inline_plot_semantics, resolve_plot_semantics
from ._plot_payloads import manifest_params_for_plot, plot_artifact_inputs, plot_input_payload
from .freshness_service import FreshnessCache, evaluate_artifact_freshness


def _plot_deliverable_id(context: WorkspaceContext, *, plot_id: str) -> str | None:
    for deliverable_id, deliverable in context.config.deliverables.items():
        if plot_id in deliverable.outputs.get("plots", []):
            return deliverable_id
    return None


def write_plot_index(
    context: WorkspaceContext,
    *,
    freshness_cache: FreshnessCache | None = None,
) -> dict[str, object]:
    plots_root = context.output_root / "plots"
    cache = freshness_cache or FreshnessCache()
    items: list[dict[str, object]] = []
    if plots_root.is_dir():
        for plot_dir in sorted(candidate for candidate in plots_root.iterdir() if candidate.is_dir()):
            manifest_path = plot_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = context.read_manifest(manifest_path)
            plot_id = str(manifest.get("artifact_id") or plot_dir.name)
            freshness = evaluate_artifact_freshness(context, artifact_kind="plot", artifact_id=plot_id, cache=cache)
            rendered_formats = [
                Path(str(output.get("path") or "")).suffix.lstrip(".")
                for output in manifest.get("outputs", [])
                if isinstance(output, dict) and output.get("path")
            ]
            output_paths = [
                (plot_dir / str(output.get("path"))).relative_to(context.output_root).as_posix()
                for output in manifest.get("outputs", [])
                if isinstance(output, dict) and output.get("path")
            ]
            items.append(
                {
                    "plot_id": plot_id,
                    "deliverable_id": _plot_deliverable_id(context, plot_id=plot_id),
                    "status": freshness["status"],
                    "manifest_status": manifest.get("status"),
                    "rendered_formats": rendered_formats,
                    "output_paths": output_paths,
                    "input_artifact_ids": [
                        f"{entry.get('kind')}:{entry.get('id')}"
                        for entry in manifest.get("inputs", [])
                        if isinstance(entry, dict)
                    ],
                    "question": manifest.get("semantics", {}).get("question")
                    if isinstance(manifest.get("semantics"), dict)
                    else None,
                    "decision_role": manifest.get("semantics", {}).get("decision_role")
                    if isinstance(manifest.get("semantics"), dict)
                    else None,
                    "created_at": manifest.get("created_at"),
                    "stale": freshness["status"] != "ok",
                }
            )
    payload = {"workspace_id": context.workspace_id, "plots": items}
    write_json(plots_root / "index.json", payload)
    return payload


def _stage_plot_dir(parent_dir: Path, plot_id: str) -> Path:
    output_root = parent_dir.parent
    staging_root = output_root / "runs" / "_staging" / parent_dir.name
    staging_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{plot_id}_", dir=staging_root))


def resolve_plot_request(
    workspace: str | Path,
    plot_id: str,
    *,
    kind: str | None,
    projection_ids: list[str],
    panel_titles: list[str],
    enrichment_id: str | None,
    distance_id: str | None,
    scalar_id: str | None,
    agreement_id: str | None,
    reducer_id: str | None,
    left_cluster_id: str | None,
    right_cluster_id: str | None,
    value_column: str | None,
    x_column: str | None,
    y_column: str | None,
    color_column: str | None,
    render_mode: str | None,
    label_column: str | None,
    label_values: list[str],
    shape_column: str | None = None,
    scalar_ids: list[str] | None = None,
    agreement_ids: list[str] | None = None,
) -> tuple[WorkspaceContext, ResolvedPlotSpec]:
    validate_identifier(plot_id, label="plot id")
    context = load_workspace_config(workspace)
    spec = resolve_plot_spec(
        plots=context.config.plots,
        plot_id=plot_id,
        kind=kind,
        projection_ids=projection_ids,
        panel_titles=panel_titles,
        enrichment_id=enrichment_id,
        distance_id=distance_id,
        scalar_id=scalar_id,
        scalar_ids=list(scalar_ids or []),
        agreement_id=agreement_id,
        agreement_ids=list(agreement_ids or []),
        reducer_id=reducer_id,
        left_cluster_id=left_cluster_id,
        right_cluster_id=right_cluster_id,
        value_column=value_column,
        x_column=x_column,
        y_column=y_column,
        color_column=color_column,
        shape_column=shape_column,
        render_mode=render_mode,
        label_column=label_column,
        label_values=label_values,
    )
    return context, spec


def render_plot(
    workspace: str | Path,
    plot_id: str,
    *,
    kind: str | None,
    projection_ids: list[str],
    panel_titles: list[str],
    enrichment_id: str | None,
    distance_id: str | None,
    scalar_id: str | None,
    agreement_id: str | None,
    reducer_id: str | None,
    left_cluster_id: str | None,
    right_cluster_id: str | None,
    value_column: str | None,
    x_column: str | None,
    y_column: str | None,
    color_column: str | None,
    render_mode: str | None,
    label_column: str | None,
    label_values: list[str],
    shape_column: str | None = None,
    scalar_ids: list[str] | None = None,
    agreement_ids: list[str] | None = None,
    force: bool = False,
) -> CommandResult:
    context, spec = resolve_plot_request(
        workspace,
        plot_id,
        kind=kind,
        projection_ids=projection_ids,
        panel_titles=panel_titles,
        enrichment_id=enrichment_id,
        distance_id=distance_id,
        scalar_id=scalar_id,
        scalar_ids=scalar_ids,
        agreement_id=agreement_id,
        agreement_ids=agreement_ids,
        reducer_id=reducer_id,
        left_cluster_id=left_cluster_id,
        right_cluster_id=right_cluster_id,
        value_column=value_column,
        x_column=x_column,
        y_column=y_column,
        color_column=color_column,
        shape_column=shape_column,
        render_mode=render_mode,
        label_column=label_column,
        label_values=label_values,
    )
    plot_dir = context.output_root / "plots" / plot_id
    if plot_dir.exists() and not force:
        raise ArtifactConflictError(f"plot artifact already exists: {plot_dir}")

    staging_dir = _stage_plot_dir(context.output_root / "plots", plot_id)
    try:
        semantics = (
            inline_plot_semantics(spec.plot_id)
            if spec.config_id is None
            else resolve_plot_semantics(context, plot_id=spec.config_id)
        )
        source_provenance = [
            {
                "id": "workspace_config",
                "role": "workspace_config",
                "path": context.config_path.as_posix(),
                "digest": source_provenance_digest({"path": context.config_path.as_posix()}),
            }
        ]
        plot_config = context.config.plots.get(spec.config_id or spec.plot_id)
        semantics_ref = getattr(plot_config, "semantics_ref", None) if plot_config is not None else None
        if isinstance(semantics_ref, str) and semantics_ref.strip():
            semantics_path = (context.workspace_dir / semantics_ref).resolve()
            source_provenance.append(
                {
                    "id": f"plot_semantics:{spec.config_id or spec.plot_id}",
                    "role": "plot_semantics",
                    "path": semantics_path.as_posix(),
                    "digest": source_provenance_digest({"path": semantics_path.as_posix()}),
                }
            )
        _, outputs, plot_metadata = render_plot_artifact(
            context,
            spec=spec,
            output_dir=staging_dir,
            semantics=semantics,
        )
        inputs = plot_artifact_inputs(context, spec)
        manifest = ArtifactManifest(
            artifact_kind="plot",
            artifact_id=plot_id,
            workspace_id=context.workspace_id,
            created_at=datetime.now(UTC).isoformat(),
            tool_version=__version__,
            command="plot render",
            inputs=inputs,
            source_provenance=source_provenance,
            params=manifest_params_for_plot(spec),
            outputs=[
                ArtifactOutput(
                    path=Path(output).name,
                    media_type=(
                        "image/svg+xml"
                        if output.endswith(".svg")
                        else "application/pdf"
                        if output.endswith(".pdf")
                        else "image/png"
                    ),
                )
                for output in outputs
            ],
            stats={"outputs": len(outputs), **plot_metadata},
            semantics=semantics.model_dump(mode="json"),
        )
        write_manifest(staging_dir / "manifest.json", manifest.model_dump(mode="json"))
        if force and plot_dir.exists():
            shutil.rmtree(plot_dir)
        if plot_dir.exists():
            raise ArtifactConflictError(f"plot artifact already exists: {plot_dir}")
        plot_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.rename(plot_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    result = CommandResult(
        command="plot render",
        workspace_id=context.workspace_id,
        status="ok",
        artifact_kind="plot",
        artifact_id=plot_id,
        outputs=[plot_dir.as_posix()],
        inputs=plot_input_payload(spec),
        metrics={"outputs": len(outputs)},
    )
    record_audit(
        context.output_root / "logs" / "audit",
        payload=result.model_dump(mode="json"),
        command="plot_render",
        artifact_id=plot_id,
    )
    write_plot_index(context)
    return result
