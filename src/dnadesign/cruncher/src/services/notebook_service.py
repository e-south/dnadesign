"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/services/notebook_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dnadesign.cruncher.services.run_service import update_run_index_from_manifest
from dnadesign.cruncher.utils.analysis_layout import resolve_analysis_dir
from dnadesign.cruncher.utils.artifacts import append_artifacts, artifact_entry, normalize_artifacts
from dnadesign.cruncher.utils.manifest import load_manifest, write_manifest


def _ensure_marimo() -> None:
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
    summary_path = analysis_dir / "summary.json"
    plot_manifest_path = analysis_dir / "plot_manifest.json"
    missing = [path for path in (summary_path, plot_manifest_path) if not path.exists()]
    if missing:
        missing_blob = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing analysis artifacts required for the notebook: {missing_blob}. "
            "Run `cruncher analyze <config>` to regenerate."
        )
    summary = _read_json(summary_path, "summary.json")
    if not isinstance(summary, dict):
        raise ValueError("summary.json must be a JSON object.")
    tf_names = summary.get("tf_names")
    if not isinstance(tf_names, list) or not tf_names:
        raise ValueError("summary.json must include a non-empty 'tf_names' list.")
    plot_manifest = _read_json(plot_manifest_path, "plot_manifest.json")
    if not isinstance(plot_manifest, dict):
        raise ValueError("plot_manifest.json must be a JSON object.")
    plots = plot_manifest.get("plots")
    if not isinstance(plots, list):
        raise ValueError("plot_manifest.json must include a 'plots' list.")


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
    strict: bool = True,
) -> Path:
    if analysis_id is not None and not str(analysis_id).strip():
        raise ValueError("analysis_id must be a non-empty string.")
    if analysis_id is not None and latest:
        raise ValueError("Use either --analysis-id or --latest, not both.")
    _ensure_marimo()
    run_dir = run_dir.resolve()
    manifest = load_manifest(run_dir)
    analysis_root = run_dir / "analysis"
    try:
        analysis_dir, resolved_id = resolve_analysis_dir(run_dir, analysis_id=analysis_id, latest=latest)
    except (FileNotFoundError, ValueError):
        if strict or analysis_id is not None or not analysis_root.exists():
            raise
        analysis_dir, resolved_id = analysis_root, None
    if strict:
        _validate_notebook_inputs(analysis_dir)

    notebooks_dir = analysis_dir / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebooks_dir / "run_overview.py"
    if notebook_path.exists() and not force:
        _record_notebook_artifact(
            manifest,
            run_dir=run_dir,
            notebook_path=notebook_path,
            config_path_str=manifest.get("config_path"),
        )
        return notebook_path

    template = _render_template(run_dir, analysis_root, resolved_id)
    notebook_path.write_text(template)

    _record_notebook_artifact(
        manifest,
        run_dir=run_dir,
        notebook_path=notebook_path,
        config_path_str=manifest.get("config_path"),
    )
    return notebook_path


def _render_template(run_dir: Path, analysis_root: Path, default_analysis_id: str | None) -> str:
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
def _(Path, json):
    run_dir = Path({json.dumps(str(run_dir))})
    analysis_root = Path({json.dumps(str(analysis_root))})
    unindexed_label = "analysis (unindexed)"
    from dnadesign.cruncher.utils.analysis_layout import list_analysis_entries_verbose

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
def _(analysis_entries, analysis_picker, _load_json, Path):
    selected = analysis_picker.value or ""
    entry = next((item for item in analysis_entries if item.get("label") == selected), None)
    analysis_id = entry.get("id") if entry else ""
    analysis_dir = Path(entry.get("path")) if entry else Path()
    entry_warnings = entry.get("warnings", []) if entry else []
    summary_path = analysis_dir / "summary.json"
    summary, summary_error = _load_json(summary_path, default={{}})
    tf_names = summary.get("tf_names", []) if isinstance(summary, dict) else []
    return analysis_id, analysis_dir, summary, summary_path, tf_names, summary_error, entry_warnings


@app.cell
def _(analysis_dir, _load_json, run_dir):
    manifest_path = run_dir / "run_manifest.json"
    manifest, manifest_error = _load_json(manifest_path, default={{}})
    plot_manifest_path = analysis_dir / "plot_manifest.json"
    plot_manifest, plot_manifest_error = _load_json(plot_manifest_path, default={{"plots": []}})
    return manifest, manifest_path, plot_manifest, plot_manifest_path, manifest_error, plot_manifest_error


@app.cell
def _(mo, tf_names):
    if not tf_names:
        tf_warning = mo.md("No TFs found in summary.json; scatter controls disabled.")
        return None, None, tf_warning
    default_x = tf_names[0] if len(tf_names) > 0 else None
    default_y = tf_names[1] if len(tf_names) > 1 else default_x
    tf_x = mo.ui.dropdown(options=tf_names, value=default_x, label="x_tf")
    tf_y = mo.ui.dropdown(options=tf_names, value=default_y, label="y_tf")
    tf_warning = None
    return tf_x, tf_y, tf_warning


@app.cell
def _(analysis_dir, pd):
    per_pwm_path = analysis_dir / "tables" / "gathered_per_pwm_everyN.csv"
    per_pwm_df = pd.read_csv(per_pwm_path) if per_pwm_path.exists() else pd.DataFrame()
    summary_table_path = analysis_dir / "tables" / "score_summary.csv"
    summary_df = pd.read_csv(summary_table_path) if summary_table_path.exists() else pd.DataFrame()
    topk_path = analysis_dir / "tables" / "elite_topk.csv"
    topk_df = pd.read_csv(topk_path) if topk_path.exists() else pd.DataFrame()
    return per_pwm_df, per_pwm_path, summary_df, summary_table_path, topk_df, topk_path


@app.cell
def _(mo, per_pwm_df):
    if per_pwm_df.empty or "chain" not in per_pwm_df.columns or "draw" not in per_pwm_df.columns:
        chain_picker = None
        draw_slider = None
        draw_limits = None
        points_slider = None
        return chain_picker, draw_slider, draw_limits, points_slider
    chain_ids = sorted(per_pwm_df["chain"].unique().tolist())
    chain_options = {{"all": None}}
    for cid in chain_ids:
        chain_options[f"chain {{cid}}"] = int(cid)
    chain_picker = mo.ui.dropdown(options=chain_options, value=None, label="chain")
    min_draw = int(per_pwm_df["draw"].min())
    max_draw = int(per_pwm_df["draw"].max())
    draw_slider = None
    if min_draw != max_draw:
        draw_slider = mo.ui.range_slider(
            min_draw,
            max_draw,
            value=(min_draw, max_draw),
            step=1,
            label="draw range",
        )
    draw_limits = (min_draw, max_draw)
    max_points = min(2000, len(per_pwm_df))
    points_slider = None
    if max_points > 0:
        slider_min = 1 if max_points < 50 else 50
        slider_step = 1 if max_points < 50 else 10
        points_slider = mo.ui.slider(
            slider_min,
            max_points,
            value=min(500, max_points),
            step=slider_step,
            label="max points",
        )
    return chain_picker, draw_slider, draw_limits, points_slider


@app.cell
def _(chain_picker, draw_slider, per_pwm_df, points_slider, draw_limits):
    if per_pwm_df.empty:
        filtered_pwm_df = per_pwm_df
        scatter_info = "No per-PWM rows to plot."
        return filtered_pwm_df, scatter_info
    if draw_limits is None:
        filtered_pwm_df = per_pwm_df
        scatter_info = "Per-PWM table missing chain/draw columns."
        return filtered_pwm_df, scatter_info
    min_draw, max_draw = draw_limits
    chain_value = None if chain_picker is None else chain_picker.value
    if draw_slider is None:
        draw_range = (min_draw, max_draw)
    else:
        draw_range = draw_slider.value
    max_points = None if points_slider is None else int(points_slider.value)

    filtered_pwm_df = per_pwm_df
    if chain_value is not None:
        filtered_pwm_df = filtered_pwm_df[filtered_pwm_df["chain"] == chain_value]
    filtered_pwm_df = filtered_pwm_df[
        (filtered_pwm_df["draw"] >= draw_range[0]) & (filtered_pwm_df["draw"] <= draw_range[1])
    ]
    if max_points is not None and len(filtered_pwm_df) > max_points:
        filtered_pwm_df = filtered_pwm_df.sample(n=max_points, random_state=0)
    scatter_info = f"Scatter points: {{len(filtered_pwm_df)}} / {{len(per_pwm_df)}}"
    return filtered_pwm_df, scatter_info


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
def _(filtered_pwm_df, plt, tf_x, tf_y):
    fig = None
    if not filtered_pwm_df.empty and tf_x is not None and tf_y is not None and tf_x.value and tf_y.value:
        x_col = f"score_{{tf_x.value}}"
        y_col = f"score_{{tf_y.value}}"
        if x_col in filtered_pwm_df.columns and y_col in filtered_pwm_df.columns:
            fig, ax = plt.subplots(figsize=(5, 5))
            data = filtered_pwm_df[[x_col, y_col]].dropna()
            ax.scatter(data[x_col].to_numpy(), data[y_col].to_numpy(), s=6, alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Per-PWM score scatter")
    return fig


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
        enabled = bool(entry.get("enabled"))
        missing = [o["path"] for o in outputs if enabled and not o["exists"] and o.get("path")]
        generated = bool(enabled and any(o["exists"] for o in outputs))
        rows.append(
            dict(
                key=entry.get("key"),
                label=entry.get("label"),
                group=entry.get("group"),
                enabled=entry.get("enabled"),
                generated=generated,
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
    tf_names,
):
    motif_store = manifest.get("motif_store") or dict() if isinstance(manifest, dict) else dict()
    motif_count = len(manifest.get("motifs", [])) if isinstance(manifest, dict) else 0
    tf_blob = ", ".join(tf_names) if tf_names else "-"
    warnings = [err for err in (summary_error, manifest_error, plot_manifest_error) if err]
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
        f"- Summary JSON: {{summary_path}}",
        ]
    )
    overview_md = mo.md("\\n".join(md))
    return overview_md


@app.cell
def _(mo, per_pwm_df, summary_df, topk_view, topk_slider, topk_path, per_pwm_path, summary_table_path):
    blocks = []
    if summary_df.empty:
        blocks.append(mo.md("No summary table found."))
    else:
        blocks.append(mo.ui.dataframe(summary_df))
    if topk_slider is not None:
        blocks.append(topk_slider)
    if not topk_view.empty:
        blocks.append(mo.ui.dataframe(topk_view))
    if per_pwm_df.empty:
        blocks.append(mo.md("No per-PWM table found."))
    else:
        blocks.append(mo.ui.data_explorer(per_pwm_df))
    blocks.append(mo.md(f"Summary table: {{summary_table_path}}"))
    blocks.append(mo.md(f"Top-K table: {{topk_path}}"))
    blocks.append(mo.md(f"Per-PWM table: {{per_pwm_path}}"))
    tables_block = mo.vstack(blocks)
    return tables_block


@app.cell
def _(chain_picker, draw_slider, fig, mo, plot_df, plot_preview, points_slider, scatter_info, tf_warning):
    blocks = []
    controls = [c for c in (chain_picker, draw_slider, points_slider) if c is not None]
    if controls:
        blocks.append(mo.hstack(controls))
    if tf_warning is not None:
        blocks.append(tf_warning)
    if scatter_info:
        blocks.append(mo.md(scatter_info))
    if fig is not None:
        blocks.append(fig)
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
def _(mo, overview_md, tables_block, plots_block, artifacts_block):
    tabs = mo.ui.tabs(
        {{
            "Overview": overview_md,
            "Tables": tables_block,
            "Plots": plots_block,
            "Artifacts": artifacts_block,
        }}
    )
    return tabs


if __name__ == "__main__":
    app.run()
"""
