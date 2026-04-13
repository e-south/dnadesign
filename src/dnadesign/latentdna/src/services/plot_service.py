"""
Plot services for latentdna.
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ..contracts.errors import ArtifactConflictError, ContractViolationError
from ..contracts.ids import validate_identifier
from ..contracts.manifest import ArtifactInput, ArtifactManifest, ArtifactOutput
from ..contracts.plot import ResolvedPlotSpec
from ..contracts.result import CommandResult
from ..io.hashing import sha256_file
from ..io.json_io import write_json
from ..io.manifest_io import write_manifest
from ..plots.recipes import resolve_plot_spec
from ..plots.render import render_plot_artifact
from ..runs.recorder import record_audit
from ..version import __version__
from ..workspaces.loader import WorkspaceContext, load_workspace_config
from .freshness_service import FreshnessCache, evaluate_artifact_freshness


def _artifact_input(kind: str, artifact_id: str, path: Path) -> ArtifactInput:
    return ArtifactInput(
        kind=kind,
        id=artifact_id,
        digest=sha256_file(path),
        path=path.as_posix(),
    )


def _artifact_inputs_for_plot(context, spec: ResolvedPlotSpec) -> list[ArtifactInput]:
    if spec.kind in {"projection_scatter", "projection_grid"}:
        return [
            _artifact_input(
                "projection",
                projection_id,
                context.output_root / "projections" / projection_id / "coords.parquet",
            )
            for projection_id in spec.projection_ids
        ]
    if spec.kind == "heatmap":
        assert spec.enrichment_id is not None
        return [
            _artifact_input(
                "enrichment_set",
                spec.enrichment_id,
                context.output_root / "enrichments" / spec.enrichment_id / "table.parquet",
            )
        ]
    if spec.kind == "distance_scatter":
        assert spec.distance_id is not None
        return [
            _artifact_input(
                "distance_set",
                spec.distance_id,
                context.output_root / "distances" / spec.distance_id / "table.parquet",
            )
        ]
    if spec.kind == "xy_scatter":
        if spec.distance_id is not None:
            return [
                _artifact_input(
                    "distance_set",
                    spec.distance_id,
                    context.output_root / "distances" / spec.distance_id / "table.parquet",
                )
            ]
        assert spec.scalar_id is not None
        return [
            _artifact_input(
                "scalar_table",
                spec.scalar_id,
                context.output_root / "scalars" / spec.scalar_id / "table.parquet",
            )
        ]
    if spec.kind == "distribution":
        table_inputs = [
            (
                "scalar_table",
                spec.scalar_id,
                context.output_root / "scalars" / spec.scalar_id / "table.parquet"
                if spec.scalar_id is not None
                else None,
            ),
            (
                "distance_set",
                spec.distance_id,
                context.output_root / "distances" / spec.distance_id / "table.parquet"
                if spec.distance_id is not None
                else None,
            ),
            (
                "enrichment_set",
                spec.enrichment_id,
                context.output_root / "enrichments" / spec.enrichment_id / "table.parquet"
                if spec.enrichment_id is not None
                else None,
            ),
            (
                "agreement_set",
                spec.agreement_id,
                context.output_root / "agreements" / spec.agreement_id / "table.parquet"
                if spec.agreement_id is not None
                else None,
            ),
        ]
        selected = [
            _artifact_input(input_kind, str(artifact_id), input_path)
            for input_kind, artifact_id, input_path in table_inputs
            if artifact_id is not None and input_path is not None
        ]
        if len(selected) != 1:
            raise ContractViolationError("distribution rendering requires exactly one table-backed artifact input")
        return selected
    if spec.kind == "curve":
        assert spec.reducer_id is not None
        return [
            _artifact_input(
                "reducer",
                spec.reducer_id,
                context.output_root / "reducers" / spec.reducer_id / "summary.json",
            )
        ]
    if spec.kind == "correspondence_heatmap":
        assert spec.left_cluster_id is not None and spec.right_cluster_id is not None
        return [
            _artifact_input(
                "cluster_set",
                spec.left_cluster_id,
                context.output_root / "clusters" / spec.left_cluster_id / "assignments.parquet",
            ),
            _artifact_input(
                "cluster_set",
                spec.right_cluster_id,
                context.output_root / "clusters" / spec.right_cluster_id / "assignments.parquet",
            ),
        ]

    assert spec.agreement_id is not None
    return [
        _artifact_input(
            "agreement_set",
            spec.agreement_id,
            context.output_root / "agreements" / spec.agreement_id / "summary.json",
        )
    ]


def _input_payload(spec: ResolvedPlotSpec) -> dict[str, object]:
    payload: dict[str, object] = {"kind": spec.kind}
    if spec.projection_ids:
        payload["projections"] = spec.projection_ids
    if spec.panel_titles:
        payload["panel_titles"] = spec.panel_titles
    if spec.enrichment_id is not None:
        payload["enrichment"] = spec.enrichment_id
    if spec.distance_id is not None:
        payload["distance"] = spec.distance_id
    if spec.scalar_id is not None:
        payload["scalar"] = spec.scalar_id
    if spec.agreement_id is not None:
        payload["agreement"] = spec.agreement_id
    if spec.reducer_id is not None:
        payload["reducer"] = spec.reducer_id
    if spec.left_cluster_id is not None:
        payload["left_cluster"] = spec.left_cluster_id
    if spec.right_cluster_id is not None:
        payload["right_cluster"] = spec.right_cluster_id
    if spec.render_mode is not None:
        payload["render_mode"] = spec.render_mode
    if spec.label_column is not None:
        payload["label_column"] = spec.label_column
    if spec.label_values:
        payload["label_values"] = spec.label_values
    if spec.config_id is not None:
        payload["plot_recipe"] = spec.config_id
    return payload


def _manifest_params(spec: ResolvedPlotSpec) -> dict[str, object]:
    params: dict[str, object] = {"plot_kind": spec.kind}
    if spec.projection_ids:
        params["projection_ids"] = spec.projection_ids
    if spec.enrichment_id is not None:
        params["enrichment_id"] = spec.enrichment_id
    if spec.distance_id is not None:
        params["distance_id"] = spec.distance_id
    if spec.scalar_id is not None:
        params["scalar_id"] = spec.scalar_id
    if spec.agreement_id is not None:
        params["agreement_id"] = spec.agreement_id
    if spec.reducer_id is not None:
        params["reducer_id"] = spec.reducer_id
    if spec.left_cluster_id is not None:
        params["left_cluster_id"] = spec.left_cluster_id
    if spec.right_cluster_id is not None:
        params["right_cluster_id"] = spec.right_cluster_id
    if spec.value_column is not None:
        params["value_column"] = spec.value_column
    if spec.x_column is not None:
        params["x_column"] = spec.x_column
    if spec.y_column is not None:
        params["y_column"] = spec.y_column
    if spec.color_column is not None:
        params["color_column"] = spec.color_column
    if spec.render_mode is not None:
        params["render_mode"] = spec.render_mode
    if spec.label_column is not None:
        params["label_column"] = spec.label_column
    if spec.label_values:
        params["label_values"] = spec.label_values
    if spec.panel_titles:
        params["panel_titles"] = spec.panel_titles
    if spec.kind == "distribution":
        if spec.scalar_id is not None:
            params["input_kind"] = "scalar_table"
            params["input_id"] = spec.scalar_id
        elif spec.distance_id is not None:
            params["input_kind"] = "distance_set"
            params["input_id"] = spec.distance_id
        elif spec.enrichment_id is not None:
            params["input_kind"] = "enrichment_set"
            params["input_id"] = spec.enrichment_id
        elif spec.agreement_id is not None:
            params["input_kind"] = "agreement_set"
            params["input_id"] = spec.agreement_id
    if spec.config_id is not None:
        params["plot_config_id"] = spec.config_id
    return params


def _plot_deliverable_id(context: WorkspaceContext, *, plot_id: str) -> str | None:
    for deliverable_id, deliverable in context.config.deliverables.items():
        if plot_id in deliverable.outputs.get("plots", []):
            return deliverable_id
    return None


def write_plot_index(context: WorkspaceContext) -> dict[str, object]:
    plots_root = context.output_root / "plots"
    cache = FreshnessCache()
    items: list[dict[str, object]] = []
    if plots_root.is_dir():
        for plot_dir in sorted(candidate for candidate in plots_root.iterdir() if candidate.is_dir()):
            manifest_path = plot_dir / "manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = context.read_manifest(manifest_path)
            plot_id = str(manifest["artifact_id"])
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
                    "status": manifest.get("status", freshness["status"]),
                    "rendered_formats": rendered_formats,
                    "output_paths": output_paths,
                    "input_artifact_ids": [
                        f"{entry.get('kind')}:{entry.get('id')}"
                        for entry in manifest.get("inputs", [])
                        if isinstance(entry, dict)
                    ],
                    "created_at": manifest.get("created_at"),
                    "stale": freshness["status"] != "ok",
                }
            )
    payload = {"workspace_id": context.workspace_id, "plots": items}
    write_json(plots_root / "index.json", payload)
    return payload


def _stage_plot_dir(parent_dir: Path, plot_id: str) -> Path:
    parent_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{plot_id}_", dir=parent_dir))


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
        agreement_id=agreement_id,
        reducer_id=reducer_id,
        left_cluster_id=left_cluster_id,
        right_cluster_id=right_cluster_id,
        value_column=value_column,
        x_column=x_column,
        y_column=y_column,
        color_column=color_column,
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
        agreement_id=agreement_id,
        reducer_id=reducer_id,
        left_cluster_id=left_cluster_id,
        right_cluster_id=right_cluster_id,
        value_column=value_column,
        x_column=x_column,
        y_column=y_column,
        color_column=color_column,
        render_mode=render_mode,
        label_column=label_column,
        label_values=label_values,
    )
    plot_dir = context.output_root / "plots" / plot_id
    if plot_dir.exists() and not force:
        raise ArtifactConflictError(f"plot artifact already exists: {plot_dir}")

    staging_dir = _stage_plot_dir(context.output_root / "plots", plot_id)
    try:
        _, outputs = render_plot_artifact(context, spec=spec, output_dir=staging_dir)
        inputs = _artifact_inputs_for_plot(context, spec)
        manifest = ArtifactManifest(
            artifact_kind="plot",
            artifact_id=plot_id,
            workspace_id=context.workspace_id,
            created_at=datetime.now(UTC).isoformat(),
            tool_version=__version__,
            command="plot render",
            inputs=inputs,
            params=_manifest_params(spec),
            outputs=[
                ArtifactOutput(
                    path=Path(output).name,
                    media_type="image/svg+xml" if output.endswith(".svg") else "image/png",
                )
                for output in outputs
            ],
            stats={"outputs": len(outputs)},
        )
        write_manifest(staging_dir / "manifest.json", manifest.model_dump(mode="json"))
        if force and plot_dir.exists():
            shutil.rmtree(plot_dir)
        if plot_dir.exists():
            raise ArtifactConflictError(f"plot artifact already exists: {plot_dir}")
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
        inputs=_input_payload(spec),
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
