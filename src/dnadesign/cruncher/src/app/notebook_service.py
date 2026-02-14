"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/notebook_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dnadesign.cruncher.analysis.layout import (
    load_table_manifest,
    plot_manifest_path,
    resolve_analysis_dir,
    resolve_required_table_paths,
    summary_path,
    table_manifest_path,
)
from dnadesign.cruncher.app.run_service import update_run_index_from_manifest
from dnadesign.cruncher.artifacts.entries import (
    append_artifacts,
    artifact_entry,
    normalize_artifacts,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest, write_manifest


def ensure_marimo() -> None:
    try:
        import marimo as _  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "marimo is not installed. Install with:\n"
            "  uv add --group notebooks marimo\n"
            "Then re-run: cruncher notebook <run_dir>"
        ) from exc


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc


def _validate_notebook_inputs(analysis_dir: Path) -> None:
    summary_file = summary_path(analysis_dir)
    plot_manifest_file = plot_manifest_path(analysis_dir)
    table_manifest_file = table_manifest_path(analysis_dir)
    missing = [path for path in (summary_file, plot_manifest_file, table_manifest_file) if not path.exists()]
    if missing:
        missing_blob = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing analysis artifacts required for the notebook: {missing_blob}. "
            "Run `cruncher analyze <config>` to regenerate."
        )
    summary = _read_json(summary_file, "summary.json")
    if not isinstance(summary, dict):
        raise ValueError("summary.json must be a JSON object.")
    tf_names = summary.get("tf_names")
    if not isinstance(tf_names, list) or not tf_names:
        raise ValueError("summary.json must include a non-empty 'tf_names' list.")
    plot_manifest = _read_json(plot_manifest_file, "plot_manifest.json")
    if not isinstance(plot_manifest, dict):
        raise ValueError("plot_manifest.json must be a JSON object.")
    plots = plot_manifest.get("plots")
    if not isinstance(plots, list):
        raise ValueError("plot_manifest.json must include a 'plots' list.")
    load_table_manifest(table_manifest_file, required=True)
    resolve_required_table_paths(
        analysis_dir,
        keys=("scores_summary", "metrics_joint", "elites_topk"),
    )


def _record_notebook_artifact(
    manifest: dict,
    *,
    run_dir: Path,
    notebook_path: Path,
    config_path_str: str | None,
) -> None:
    existing_count = len(normalize_artifacts(manifest.get("artifacts")))
    append_artifacts(
        manifest,
        [
            artifact_entry(
                notebook_path,
                run_dir,
                kind="notebook",
                label="Analysis overview notebook (marimo)",
                stage="analysis",
            )
        ],
    )
    if len(manifest.get("artifacts", [])) == existing_count:
        return
    write_manifest(run_dir, manifest)
    if config_path_str:
        config_path = Path(config_path_str)
        if config_path.exists():
            update_run_index_from_manifest(config_path, run_dir, manifest)


def generate_notebook(
    run_dir: Path,
    *,
    analysis_id: Optional[str] = None,
    latest: bool = False,
    force: bool = False,
) -> Path:
    if analysis_id is not None and not str(analysis_id).strip():
        raise ValueError("analysis_id must be a non-empty string.")
    if analysis_id is not None and latest:
        raise ValueError("Use either --analysis-id or --latest, not both.")
    ensure_marimo()
    run_dir = run_dir.resolve()
    manifest = load_manifest(run_dir)
    analysis_dir, resolved_id = resolve_analysis_dir(run_dir, analysis_id=analysis_id, latest=latest)
    _validate_notebook_inputs(analysis_dir)

    notebook_path = analysis_dir / "notebook__run_overview.py"
    if notebook_path.exists() and not force:
        _record_notebook_artifact(
            manifest,
            run_dir=run_dir,
            notebook_path=notebook_path,
            config_path_str=manifest.get("config_path"),
        )
        return notebook_path

    template = _render_template(resolved_id)
    notebook_path.write_text(template)

    _record_notebook_artifact(
        manifest,
        run_dir=run_dir,
        notebook_path=notebook_path,
        config_path_str=manifest.get("config_path"),
    )
    return notebook_path


def _render_template(default_analysis_id: str | None) -> str:
    return f"""import marimo as mo

app = mo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    return mo, json, Path, pd, plt


@app.cell
def _(Path, json, mo, refresh_button):
    _ = refresh_button.value
    notebook_path = Path(__file__).resolve()
    analysis_dir = notebook_path.parent
    if analysis_dir.name == "analysis":
        run_dir = analysis_dir.parent
    elif "_archive" in analysis_dir.parts:
        parts = analysis_dir.parts
        if "analysis" in parts:
            analysis_idx = parts.index("analysis")
            run_dir = Path(*parts[:analysis_idx])
        else:
            run_dir = analysis_dir
    else:
        run_dir = analysis_dir.parent
    if not run_dir.exists():
        mo.stop(True, mo.md(f"Run directory not found: {{run_dir}}"))
    analysis_root = run_dir / "analysis"
    from dnadesign.cruncher.analysis.layout import list_analysis_entries_verbose

    analysis_entries = list_analysis_entries_verbose(run_dir)
    analysis_labels = [entry["label"] for entry in analysis_entries]
    latest_label = next((entry["label"] for entry in analysis_entries if entry.get("kind") == "latest"), None)
    default_id_hint = {default_analysis_id!r}
    default_label = None
    if default_id_hint:
        for entry in analysis_entries:
            if entry.get("id") == default_id_hint:
                default_label = entry.get("label")
                break
    if default_label is None:
        default_label = latest_label
    if default_label is None and analysis_labels:
        default_label = analysis_labels[0]
    return run_dir, analysis_root, analysis_entries, analysis_labels, default_label


@app.cell
def _(mo):
    refresh_button = mo.ui.button(label="Refresh analysis list", kind="neutral")
    return refresh_button


@app.cell
def _(Path, json, mo):
    def _load_json(path: Path, *, default: dict):
        if not path.exists():
            return default, f"Missing JSON at {{path}}"
        try:
            return json.loads(path.read_text()), None
        except Exception as exc:  # pragma: no cover - notebook UX
            return default, f"Failed to parse JSON at {{path}}: {{exc}}"
    return _load_json


@app.cell
def _(analysis_labels, mo):
    if not analysis_labels:
        mo.stop(True, mo.md("No analysis runs found. Run `cruncher analyze <config>` first."))
    return


@app.cell
def _(analysis_labels, default_label, mo):
    analysis_picker = mo.ui.dropdown(options=analysis_labels, value=default_label, label="analysis")
    return analysis_picker


@app.cell
def _(analysis_picker, mo, refresh_button):
    if analysis_picker is None:
        return None
    analysis_controls = mo.hstack([analysis_picker, refresh_button])
    return analysis_controls


@app.cell
def _(analysis_entries, analysis_picker, _load_json, Path):
    from dnadesign.cruncher.analysis.layout import summary_path as analysis_summary_path

    selected = analysis_picker.value or ""
    entry = next((item for item in analysis_entries if item.get("label") == selected), None)
    analysis_id = entry.get("id") if entry else ""
    analysis_dir = Path(entry.get("path")) if entry else Path()
    entry_warnings = entry.get("warnings", []) if entry else []
    summary_file = analysis_summary_path(analysis_dir)
    summary, summary_error = _load_json(summary_file, default={{}})
    tf_names = summary.get("tf_names", []) if isinstance(summary, dict) else []
    return analysis_id, analysis_dir, summary, summary_file, tf_names, summary_error, entry_warnings


@app.cell
def _(analysis_dir, _load_json, run_dir):
    from dnadesign.cruncher.analysis.layout import plot_manifest_path as analysis_plot_manifest_path
    from dnadesign.cruncher.analysis.layout import resolve_required_table_paths as analysis_resolve_required_table_paths
    from dnadesign.cruncher.analysis.layout import table_manifest_path as analysis_table_manifest_path
    from dnadesign.cruncher.artifacts.layout import manifest_path as run_manifest_path

    manifest_path = run_manifest_path(run_dir)
    manifest, manifest_error = _load_json(manifest_path, default={{}})
    plot_manifest_file = analysis_plot_manifest_path(analysis_dir)
    plot_manifest, plot_manifest_error = _load_json(plot_manifest_file, default={{"plots": []}})
    # table_manifest.json defines table keys used by this notebook.
    table_manifest_file = analysis_table_manifest_path(analysis_dir)
    table_manifest, table_manifest_error = _load_json(table_manifest_file, default={{"tables": []}})
    table_paths = {{}}
    table_paths_error = None
    try:
        table_paths = analysis_resolve_required_table_paths(
            analysis_dir,
            keys=("scores_summary", "metrics_joint", "elites_topk"),
        )
    except Exception as exc:
        table_paths_error = str(exc)
    return (
        manifest,
        manifest_path,
        plot_manifest,
        plot_manifest_file,
        table_manifest,
        table_manifest_file,
        table_paths,
        manifest_error,
        plot_manifest_error,
        table_manifest_error,
        table_paths_error,
    )


@app.cell
def _(analysis_dir, mo, pd, table_paths, table_paths_error):
    if table_paths_error:
        mo.stop(True, mo.md(f"Table contract error: {{table_paths_error}}"))

    def _read_table(path):
        if not path.exists():
            return pd.DataFrame()
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    summary_table_path = table_paths["scores_summary"]
    summary_df = _read_table(summary_table_path)
    joint_metrics_path = table_paths["metrics_joint"]
    joint_metrics_df = _read_table(joint_metrics_path)
    topk_path = table_paths["elites_topk"]
    topk_df = _read_table(topk_path)
    return summary_df, summary_table_path, joint_metrics_df, joint_metrics_path, topk_df, topk_path


@app.cell
def _(mo, topk_df):
    if topk_df.empty:
        return None
    max_k = max(1, min(len(topk_df), 50))
    topk_slider = mo.ui.slider(1, max_k, value=min(10, max_k), label="Top-K")
    return topk_slider


@app.cell
def _(topk_df, topk_slider):
    if topk_slider is None:
        topk_view = topk_df
    else:
        topk_view = topk_df.head(int(topk_slider.value))
    return topk_view


@app.cell
def _(analysis_dir, pd, plot_manifest):
    plot_entries = plot_manifest.get("plots", []) if isinstance(plot_manifest, dict) else []
    rows = []
    for entry in plot_entries:
        outputs = []
        for o in entry.get("outputs", []):
            path = o.get("path", "") if isinstance(o, dict) else ""
            full_path = analysis_dir / path if path else None
            outputs.append(
                {{
                    "path": path,
                    "exists": bool(full_path and full_path.exists()),
                }}
            )
        manifest_generated = bool(entry.get("generated"))
        manifest_skipped = bool(entry.get("skipped"))
        generated = bool(manifest_generated and any(o["exists"] for o in outputs))
        missing = [o["path"] for o in outputs if manifest_generated and not o["exists"] and o.get("path")]
        rows.append(
            dict(
                key=entry.get("key"),
                label=entry.get("label"),
                group=entry.get("group"),
                manifest_generated=manifest_generated,
                skipped=manifest_skipped,
                generated=generated,
                skip_reason=entry.get("skip_reason"),
                missing_outputs="; ".join(missing),
                outputs="; ".join([o.get("path", "") for o in outputs if o.get("path")]),
            )
        )
        entry["outputs"] = outputs
        entry["missing_outputs"] = missing
        entry["generated"] = generated
    plot_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    plot_options = (
        dict((f"{{e.get('label')}} ({{e.get('key')}})", e.get("key")) for e in plot_entries)
        if plot_entries
        else dict()
    )
    default_key = plot_entries[0].get("key") if plot_entries else None
    return plot_entries, plot_df, plot_options, default_key


@app.cell
def _(default_key, mo, plot_options):
    plot_picker = None
    if plot_options:
        plot_picker = mo.ui.dropdown(options=plot_options, value=default_key, label="plot")
    return plot_picker


@app.cell
def _(analysis_dir, mo, plot_entries, plot_picker):
    plot_key = plot_picker.value if plot_picker else None
    selected = next((p for p in plot_entries if p.get("key") == plot_key), None)
    if not selected:
        return None, mo.md("No plot metadata available.")
    blocks = [
        mo.md(
            f"### {{selected.get('label')}} (`{{selected.get('key')}}`)\\n\\n"
            f"{{selected.get('description', '')}}"
        )
    ]
    requires = selected.get("requires", [])
    if requires:
        blocks.append(mo.md("Requires: " + ", ".join(requires)))
    outputs = selected.get("outputs", [])
    if outputs:
        lines = []
        for output in outputs:
            path = output.get("path", "")
            full_path = analysis_dir / path if path else None
            exists = bool(full_path and full_path.exists())
            lines.append(f"- {{path}} ({{'ok' if exists else 'missing'}})")
        blocks.append(mo.md("Outputs:\\n" + "\\n".join(lines)))
    missing = selected.get("missing_outputs", [])
    if missing:
        blocks.append(mo.md("Missing outputs: " + ", ".join(missing)))

    for output in outputs:
        path = output.get("path", "")
        if not path:
            continue
        full_path = analysis_dir / path
        if full_path.exists() and full_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            blocks.append(mo.image(full_path))
            continue
        if full_path.exists() and full_path.suffix.lower() in {".txt", ".log"}:
            try:
                text = full_path.read_text()
            except Exception as exc:
                blocks.append(mo.md(f"Failed to read {{path}}: {{exc}}"))
            else:
                max_chars = 4000
                preview = text
                truncated = False
                if len(preview) > max_chars:
                    preview = preview[:max_chars]
                    truncated = True
                blocks.append(mo.md(f"**Text output:** {{path}}"))
                if truncated:
                    blocks.append(mo.md("_preview truncated_"))
                blocks.append(mo.md("```\\n" + preview + "\\n```"))
    plot_preview = mo.vstack(blocks)
    return selected, plot_preview


@app.cell
def _(analysis_dir, manifest, run_dir):
    artifacts = manifest.get("artifacts", []) if isinstance(manifest, dict) else []
    try:
        prefix = str(analysis_dir.relative_to(run_dir)).rstrip("/") + "/"
    except Exception:
        return artifacts
    return [a for a in artifacts if isinstance(a, dict) and str(a.get("path", "")).startswith(prefix)]


@app.cell
def _(
    analysis_id,
    entry_warnings,
    manifest,
    manifest_error,
    mo,
    plot_manifest_error,
    plot_manifest_path,
    summary,
    summary_error,
    summary_path,
    table_manifest_error,
    table_manifest_path,
    table_paths_error,
    tf_names,
):
    motif_store = manifest.get("motif_store") or dict() if isinstance(manifest, dict) else dict()
    motif_count = len(manifest.get("motifs", [])) if isinstance(manifest, dict) else 0
    tf_blob = ", ".join(tf_names) if tf_names else "-"
    warning_sources = (
        summary_error,
        manifest_error,
        plot_manifest_error,
        table_manifest_error,
        table_paths_error,
    )
    warnings = [err for err in warning_sources if err]
    if entry_warnings:
        warnings.extend([str(warn) for warn in entry_warnings if warn])
    md = [
        "# Cruncher analysis overview",
        "",
    ]
    if warnings:
        md.append("Warnings:")
        md.extend([f"- {{warn}}" for warn in warnings])
        md.append("")
    md.extend(
        [
            f"Run: {{summary.get('run', '-') }}",
        f"Stage: {{manifest.get('stage', '-') }}",
        f"Analysis ID: {{analysis_id}}",
        f"Created at: {{summary.get('created_at', manifest.get('created_at', '-') )}}",
        f"TFs: {{tf_blob}}",
        f"Motifs: {{motif_count}}",
        f"PWM source: {{motif_store.get('pwm_source', '-') }}",
        f"Site kinds: {{motif_store.get('site_kinds', '-') }}",
        f"Combine sites: {{motif_store.get('combine_sites', '-') }}",
        "",
        "Inputs:",
        f"- Config used: {{summary.get('config_used', '-') }}",
        f"- Analysis settings: {{summary.get('analysis_used', '-') }}",
        f"- Plot manifest: {{plot_manifest_path}}",
        f"- Table manifest: {{table_manifest_path}}",
        f"- Summary JSON: {{summary_path}}",
        ]
    )
    overview_md = mo.md("\\n".join(md))
    return overview_md


@app.cell
def _(analysis_controls, mo, overview_md):
    blocks = []
    if analysis_controls is not None:
        blocks.append(analysis_controls)
    blocks.append(overview_md)
    overview_block = mo.vstack(blocks)
    return overview_block


@app.cell
def _(
    joint_metrics_df,
    joint_metrics_path,
    mo,
    summary_df,
    summary_table_path,
    topk_path,
    topk_slider,
    topk_view,
):
    blocks = []
    if summary_df.empty:
        blocks.append(mo.md("No summary table found."))
    else:
        blocks.append(mo.ui.dataframe(summary_df))
    if joint_metrics_df.empty:
        blocks.append(mo.md("No joint metrics table found."))
    else:
        blocks.append(mo.ui.dataframe(joint_metrics_df))
    if topk_slider is not None:
        blocks.append(topk_slider)
    if not topk_view.empty:
        blocks.append(mo.ui.dataframe(topk_view))
    blocks.append(mo.md(f"Summary table: {{summary_table_path}}"))
    blocks.append(mo.md(f"Joint metrics table: {{joint_metrics_path}}"))
    blocks.append(mo.md(f"Top-K table: {{topk_path}}"))
    tables_block = mo.vstack(blocks)
    return tables_block


@app.cell
def _(mo, plot_df, plot_preview):
    blocks = []
    if not plot_df.empty:
        blocks.append(mo.ui.dataframe(plot_df))
    blocks.append(plot_preview)
    plots_block = mo.vstack(blocks)
    return plots_block


@app.cell
def _(mo, artifacts):
    if not artifacts:
        artifacts_block = mo.md("No artifacts recorded for this analysis.")
    else:
        artifacts_block = mo.ui.dataframe(artifacts)
    return artifacts_block


@app.cell
def _(mo, overview_block, tables_block, plots_block, artifacts_block):
    tabs = mo.ui.tabs(
        {{
            "Overview": overview_block,
            "Tables": tables_block,
            "Plots": plots_block,
            "Artifacts": artifacts_block,
        }}
    )
    return tabs


if __name__ == "__main__":
    app.run()
"""
