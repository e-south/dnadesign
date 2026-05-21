"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/notebook_components.py

Reusable generated-cell components for OPAL marimo notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from typing import Any, Iterable, Mapping


def build_notebook_campaign_summary_row(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact campaign row for notebook overview tables."""

    campaign = _mapping(view_model.get("campaign"))
    status = _mapping(view_model.get("status"))
    stale_count = len(_sequence(view_model.get("stale_artifacts")))
    warning_count = len(_sequence(view_model.get("warnings")))
    label = f"{campaign.get('slug') or 'unknown'} | {status.get('progress_status') or 'unknown'}"
    return {
        "label": label,
        "campaign": campaign.get("slug"),
        "status": status.get("progress_status"),
        "round_count": status.get("round_count"),
        "latest_run_id": status.get("latest_run_id"),
        "x_column": campaign.get("x_column"),
        "label_source": campaign.get("label_source"),
        "plots": len(_sequence(view_model.get("plot_manifests"))),
        "stale": stale_count,
        "warnings": warning_count,
    }


def build_notebook_at_a_glance_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build first-viewport campaign status lines from a notebook view model."""

    row = build_notebook_campaign_summary_row(view_model)
    campaign = _mapping(view_model.get("campaign"))
    status = _mapping(view_model.get("status"))
    selected_count = _selection_count(view_model)
    lines = [
        "### At a glance",
        "",
        f"- Campaign: `{row['campaign']}`",
        f"- Status: `{row['status']}`",
        f"- Round selector: `{status.get('round_selector')}`",
        f"- Round count: `{row['round_count']}`",
        f"- Latest run ID: `{row['latest_run_id']}`",
        f"- X column: `{row['x_column']}`",
        f"- Label source: `{row['label_source']}`",
        f"- Config: `{campaign.get('config_path')}`",
        f"- Workdir: `{campaign.get('workdir')}`",
    ]
    if selected_count is not None:
        lines.append(f"- Selected count: `{selected_count}`")
    lines.extend(
        [
            f"- Manifest-backed plots: `{row['plots']}`",
            f"- Warnings: `{row['warnings']}`",
            f"- Stale artifacts: `{row['stale']}`",
        ]
    )
    return lines


def build_notebook_validity_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build explicit trust-state lines for generated notebooks."""

    status = _mapping(view_model.get("status"))
    progress = _mapping(view_model.get("progress"))
    state = _mapping(progress.get("state"))
    gallery = build_notebook_plot_gallery_model(view_model)
    plot_manifests = _sequence(view_model.get("plot_manifests"))
    stale = _sequence(view_model.get("stale_artifacts"))
    warnings = [
        item
        for item in (*_sequence(view_model.get("warnings")), *_sequence(progress.get("warnings")))
        if isinstance(item, Mapping)
    ]
    blocking_count = sum(1 for item in warnings if item.get("severity") == "error")
    artifact_garden = _mapping(view_model.get("artifact_garden"))
    prune_plan = _mapping(artifact_garden.get("prune_plan"))
    review_state = "present" if isinstance(view_model.get("review_manifest"), Mapping) else "missing"
    state_text = "present" if state.get("exists") else "missing"
    artifact_schema = artifact_garden.get("schema_version") or "unavailable"
    return [
        "### Validity",
        "",
        f"- Campaign status: `{status.get('progress_status') or 'unknown'}`",
        f"- Progress schema: `{progress.get('schema_version') or 'missing'}`",
        f"- State file: `{state_text}`",
        f"- Review manifest: `{review_state}`",
        f"- Plot manifests: `{len(plot_manifests)}`",
        f"- Written plot media choices: `{len(gallery['choices'])}`",
        f"- Missing plot outputs: `{len(gallery['missing_outputs'])}`",
        f"- Warnings: `{len(warnings)}`",
        f"- Stale artifacts: `{len(stale)}`",
        f"- Artifact garden: `{artifact_schema}`",
        f"- Prune requires apply: `{prune_plan.get('requires_apply', True)}`",
        f"- Blocking issues: `{blocking_count}`",
    ]


def build_notebook_change_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build a compact round/run change summary for generated notebooks."""

    status = _mapping(view_model.get("status"))
    progress = _mapping(view_model.get("progress"))
    event_contract = _mapping(progress.get("event_contract"))
    rounds = _sequence(progress.get("rounds"))
    lines = [
        "### Changes",
        "",
        f"- Round selector: `{status.get('round_selector') or progress.get('round_selector') or 'latest'}`",
        f"- Rounds visible: `{len(rounds)}`",
        f"- Latest run ID: `{status.get('latest_run_id')}`",
        (
            "- Event phases: "
            f"command=`{event_contract.get('command_events', 0)}`, "
            f"preflight=`{event_contract.get('preflight_events', 0)}`, "
            f"run=`{event_contract.get('run_events', 0)}`, "
            f"finalize=`{event_contract.get('finalize_events', 0)}`"
        ),
        f"- Aborted rounds: `{len(_sequence(event_contract.get('aborted_rounds')))}`",
        f"- Ambiguous run-scope rounds: `{len(_sequence(event_contract.get('ambiguous_rounds')))}`",
    ]
    if not rounds:
        lines.append("- Round history: `not started`")
    return lines


def build_notebook_change_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return progress-derived round/run rows for notebook change tables."""

    progress = _mapping(view_model.get("progress"))
    rows: list[dict[str, Any]] = []
    for round_row in _sequence(progress.get("rounds")):
        if not isinstance(round_row, Mapping):
            continue
        summary = _mapping(round_row.get("summary"))
        run_scope = _mapping(summary.get("run_scope"))
        predict = _mapping(round_row.get("predict"))
        rows.append(
            {
                "round": round_row.get("round_index"),
                "status": round_row.get("status"),
                "last_stage": round_row.get("last_stage"),
                "run_id": _resolved_run_id(run_scope),
                "attempts": len(_sequence(run_scope.get("attempt_ids"))),
                "events": round_row.get("events"),
                "elapsed_sec": round_row.get("elapsed_sec"),
                "predict": _predict_progress_text(predict),
                "aborted": bool(summary.get("aborted")),
                "ambiguous_run_scope": bool(run_scope.get("ambiguous_run_scope")),
                "log_path": round_row.get("path"),
            }
        )
    return rows


def build_notebook_distrust_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build a compact distrust/limitations panel for generated notebooks."""

    review_manifest = view_model.get("review_manifest")
    gallery = build_notebook_plot_gallery_model(view_model)
    warnings = _sequence(view_model.get("warnings"))
    stale = _sequence(view_model.get("stale_artifacts"))
    lines = [
        "### Distrust and limitations",
        "",
        "- OPAL notebooks are inspection surfaces; execution and mutation stay in the CLI.",
        "- Producer-specific representation browsers and study benchmark reports are outside this notebook.",
    ]
    if review_manifest is None:
        lines.append("- Review manifest: `missing`")
    else:
        lines.append("- Review manifest: `present`")
    if not gallery["choices"]:
        lines.append("- Plot evidence: no written manifest-backed plot media.")
    if warnings:
        lines.append(f"- Warnings: `{len(warnings)}`")
    if stale:
        lines.append(f"- Stale artifacts ignored by active manifests: `{len(stale)}`")
    return lines


def build_notebook_evidence_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return warning and stale-artifact rows for notebook evidence tables."""

    rows: list[dict[str, Any]] = []
    for warning in _sequence(view_model.get("warnings")):
        if not isinstance(warning, Mapping):
            continue
        rows.append(
            {
                "source": "warning",
                "category": warning.get("category"),
                "severity": warning.get("severity"),
                "message": warning.get("message"),
                "path": warning.get("path"),
            }
        )
    for artifact in _sequence(view_model.get("stale_artifacts")):
        if not isinstance(artifact, Mapping):
            continue
        rows.append(
            {
                "source": "stale_artifact",
                "category": artifact.get("category"),
                "severity": artifact.get("severity"),
                "message": artifact.get("message"),
                "path": artifact.get("path"),
            }
        )
    return rows


def build_notebook_metric_definition_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return plot metric/data-shape definitions for notebook evidence tables."""

    rows: list[dict[str, Any]] = []
    for manifest in _sequence(view_model.get("plot_manifests")):
        if not isinstance(manifest, Mapping):
            continue
        metadata = _mapping(manifest.get("metadata"))
        freshness = _mapping(manifest.get("freshness"))
        purpose = manifest.get("review_purpose") or manifest.get("caption") or metadata.get("summary") or "not recorded"
        rows.append(
            {
                "plot": manifest.get("name"),
                "kind": manifest.get("kind"),
                "data_shape": metadata.get("data_shape") or "not recorded",
                "tidy_schema": _join_list(metadata.get("tidy_schema"), sep=", "),
                "failure_modes": _join_list(metadata.get("failure_modes"), sep="; "),
                "freshness": freshness.get("status") or "unknown",
                "purpose": purpose,
            }
        )
    return rows


def build_notebook_artifact_garden_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build artifact-garden status lines for generated notebooks."""

    audit = _mapping(view_model.get("artifact_garden"))
    if not audit:
        return [
            "### Artifacts",
            "",
            "- Artifact garden audit: `unavailable`",
            "- Run `uv run opal artifacts audit -c <campaign.yaml>` for a manifest-authoritative artifact inventory.",
        ]
    bytes_row = _mapping(audit.get("bytes"))
    prune_plan = _mapping(audit.get("prune_plan"))
    roots = _sequence(audit.get("artifact_roots"))
    active_manifests = _sequence(audit.get("active_manifests"))
    stale = _sequence(audit.get("stale_artifacts"))
    local_only = "yes (local-only)" if audit.get("local_only") else "no"
    return [
        "### Artifacts",
        "",
        f"- Artifact garden schema: `{audit.get('schema_version')}`",
        f"- Root: `{audit.get('root')}`",
        f"- Local-only root: `{local_only}`",
        f"- Artifact roots: `{len(roots)}`",
        f"- Active manifests: `{len(active_manifests)}`",
        f"- Stale artifacts: `{len(stale)}`",
        f"- Artifact bytes: `{bytes_row.get('artifact_roots', 0)}`",
        f"- Stale bytes: `{bytes_row.get('stale_artifacts', 0)}`",
        f"- Prune plan items: `{prune_plan.get('item_count', 0)}`",
        f"- Prune requires apply: `{prune_plan.get('requires_apply', True)}`",
    ]


def build_notebook_artifact_garden_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return artifact root, stale artifact, and prune-plan rows for notebooks."""

    audit = _mapping(view_model.get("artifact_garden"))
    if not audit:
        return []
    rows: list[dict[str, Any]] = []
    for root in _sequence(audit.get("artifact_roots")):
        if not isinstance(root, Mapping):
            continue
        rows.append(
            {
                "source": "artifact_root",
                "name": root.get("name"),
                "path": root.get("path"),
                "exists": root.get("exists"),
                "file_count": root.get("file_count"),
                "size_bytes": root.get("size_bytes"),
                "scope": None,
                "reason": None,
            }
        )
    for artifact in _sequence(audit.get("stale_artifacts")):
        if not isinstance(artifact, Mapping):
            continue
        rows.append(
            {
                "source": "stale_artifact",
                "name": None,
                "path": artifact.get("path"),
                "exists": True,
                "file_count": None,
                "size_bytes": artifact.get("size_bytes"),
                "scope": artifact.get("scope"),
                "reason": artifact.get("reason"),
            }
        )
    prune_plan = _mapping(audit.get("prune_plan"))
    if prune_plan:
        rows.append(
            {
                "source": "prune_plan",
                "name": "stale_artifacts_only",
                "path": "",
                "exists": None,
                "file_count": prune_plan.get("item_count"),
                "size_bytes": prune_plan.get("bytes_to_delete"),
                "scope": prune_plan.get("mode"),
                "reason": "dry-run unless --apply is explicit",
            }
        )
    return rows


def build_notebook_plot_gallery_model(
    view_model: Mapping[str, Any],
    *,
    plot_entries: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a manifest-authoritative plot gallery model for marimo templates."""

    campaign = _mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir") or ""
    plots_dir = str(Path(str(workdir)) / "outputs" / "plots") if workdir else "outputs/plots"
    manifest_rows = [
        manifest
        for manifest in _sequence(view_model.get("plot_manifests"))
        if isinstance(manifest, Mapping) and manifest.get("status") == "written"
    ]
    active_by_name = {str(row.get("name")): row for row in manifest_rows}
    configured_entries = _plot_entries_from_manifests(manifest_rows) if plot_entries is None else list(plot_entries)

    choices: list[dict[str, Any]] = []
    missing_outputs: list[str] = []
    for entry in configured_entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or "")
        if not name:
            continue
        manifest = active_by_name.get(name)
        if manifest is None:
            missing_outputs.append(name)
            continue
        media_output = _first_media_output(manifest)
        if media_output is None:
            missing_outputs.append(name)
            continue
        path = str(media_output.get("path"))
        label = f"{name} ({Path(path).name})"
        choices.append(
            {
                "label": label,
                "path": path,
                "entry": dict(entry),
                "manifest": dict(manifest),
            }
        )
    return {
        "plots_dir": plots_dir,
        "choices": choices,
        "missing_outputs": missing_outputs,
        "stale_artifacts": list(_sequence(view_model.get("stale_artifacts"))),
    }


def build_notebook_plot_card_lines(choice: Mapping[str, Any]) -> list[str]:
    """Build manifest-backed plot-card detail lines for generated notebooks."""

    entry = _mapping(choice.get("entry"))
    manifest = _mapping(choice.get("manifest"))
    freshness = _mapping(manifest.get("freshness"))
    inputs = [
        item
        for item in _sequence(manifest.get("inputs"))
        if isinstance(item, Mapping) and (item.get("path") or item.get("role"))
    ]
    source_data = "; ".join(f"{item.get('role') or 'input'}={item.get('path') or 'unrecorded'}" for item in inputs[:5])
    if not source_data:
        source_data = "not recorded"
    warnings = _sequence(manifest.get("warnings"))
    tags = ", ".join(str(tag) for tag in _sequence(entry.get("tags"))) or "none"
    return [
        "### Plot deliverables",
        "",
        f"**Plot**: `{entry.get('name') or manifest.get('name')}`",
        f"Kind: `{entry.get('kind') or manifest.get('kind')}`",
        f"Tags: `{tags}`",
        f"Status: `{manifest.get('status')}`",
        f"Freshness: `{freshness.get('status') or manifest.get('stale_state') or 'unknown'}`",
        f"Generated: `{manifest.get('generated_at')}`",
        f"Run ID: `{manifest.get('run_id')}`",
        f"Rounds: `{manifest.get('rounds')}`",
        f"Media: `{choice.get('path')}`",
        f"Tidy CSV: `{manifest.get('tidy_csv') or 'none'}`",
        f"Source data: `{source_data}`",
        f"Params: `{manifest.get('params') or {}}`",
        f"Warnings: `{len(warnings)}`",
    ]


def render_plot_gallery_cells() -> str:
    """Render generated cells for manifest-backed plot selection."""

    return dedent(
        """
        @app.cell
        def _(Path, build_notebook_plot_gallery_model, notebook_view_model, plot_entries):
            plot_gallery_model = build_notebook_plot_gallery_model(
                notebook_view_model,
                plot_entries=plot_entries,
            )
            plots_dir = Path(plot_gallery_model["plots_dir"])
            plot_choices = plot_gallery_model["choices"]
            missing_outputs = plot_gallery_model["missing_outputs"]
            stale_plot_artifacts = plot_gallery_model["stale_artifacts"]
            return plots_dir, plot_choices, missing_outputs, stale_plot_artifacts


        @app.cell
        def _(mo, plot_cfg_error, plot_choices, plots_dir, missing_outputs, stale_plot_artifacts):
            plot_ui = None
            gallery_scope = "All configured plots with written manifests."
            if plot_cfg_error:
                plot_gallery_note = (
                    "### Plot artifacts (`outputs/plots`)\\n\\n"
                    f"Plot config unavailable: `{plot_cfg_error}`"
                )
            elif not plot_choices:
                _lines = [
                    "### Plot artifacts (`outputs/plots`)",
                    "",
                    f"No manifest-backed plot outputs found in `{plots_dir}`.",
                    "Run `uv run opal plot -c <campaign.yaml>` to generate plots.",
                    gallery_scope,
                ]
                if missing_outputs:
                    _lines.append(
                        f"Configured plots without outputs: {', '.join(missing_outputs)}"
                    )
                if stale_plot_artifacts:
                    _lines.append(f"Stale artifact warnings: `{len(stale_plot_artifacts)}`")
                plot_gallery_note = "\\n".join(_lines)
            else:
                labels = [plot_choice["label"] for plot_choice in plot_choices]
                plot_ui = mo.ui.dropdown(labels, value=labels[0], label="Plot")
                plot_gallery_note = "### Plot artifacts (`outputs/plots`)\\n\\n" + gallery_scope
                if stale_plot_artifacts:
                    plot_gallery_note += f"\\n\\nStale artifact warnings: `{len(stale_plot_artifacts)}`"
            return plot_ui, plot_gallery_note


        @app.cell
        def _(Path, build_notebook_plot_card_lines, mo, plot_choices, plot_gallery_note, plot_ui):
            if plot_ui is None:
                plot_panel = mo.md(plot_gallery_note)
            else:
                selected = str(plot_ui.value)
                choice = next(
                    (
                        plot_choice
                        for plot_choice in plot_choices
                        if plot_choice["label"] == selected
                    ),
                    None,
                )
                if choice is None:
                    raise ValueError(f"Plot selection not found: {selected}")
                details = [plot_gallery_note, "", *build_notebook_plot_card_lines(choice)]
                plot_panel = mo.vstack(
                    [
                        mo.md("\\n".join(details)),
                        plot_ui,
                        mo.image(Path(choice["path"]).read_bytes()),
                    ]
                )
            return plot_panel
        """
    ).strip("\n")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _join_list(value: Any, *, sep: str) -> str:
    items = [str(item) for item in _sequence(value) if str(item)]
    return sep.join(items) if items else "not recorded"


def _selection_count(view_model: Mapping[str, Any]) -> int | None:
    review_manifest = _mapping(view_model.get("review_manifest"))
    selection = _mapping(review_manifest.get("selection"))
    for key in ("selected_count", "selection_count", "count"):
        value = selection.get(key)
        if isinstance(value, int):
            return value
    for key in ("selected_records", "preview", "rows"):
        value = selection.get(key)
        if isinstance(value, list):
            return len(value)
    return None


def _resolved_run_id(run_scope: Mapping[str, Any]) -> str | None:
    value = run_scope.get("resolved_run_id")
    if value not in (None, ""):
        return str(value)
    run_ids = _sequence(run_scope.get("run_ids"))
    return str(run_ids[-1]) if run_ids else None


def _predict_progress_text(predict: Mapping[str, Any]) -> str:
    batch = predict.get("batch")
    of = predict.get("of")
    rows = predict.get("rows")
    if batch is not None and of is not None:
        text = f"{batch}/{of} batches"
        if rows is not None:
            text += f", {rows} rows"
        return text
    if rows is not None:
        return f"{rows} rows"
    return "not recorded"


def _plot_entries_from_manifests(manifests: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for manifest in manifests:
        name = manifest.get("name")
        if not name:
            continue
        entries.append(
            {
                "name": str(name),
                "kind": manifest.get("kind") or "unknown",
                "tags": list(_sequence(manifest.get("tags"))),
            }
        )
    return entries


def _first_media_output(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for output in _sequence(manifest.get("outputs")):
        if not isinstance(output, Mapping):
            continue
        if output.get("role") == "media" and output.get("exists") and output.get("path"):
            return output
    return None
