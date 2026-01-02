"""Notebook helpers for analysis runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from dnadesign.cruncher.services.run_service import update_run_index_from_manifest
from dnadesign.cruncher.utils.artifacts import append_artifacts, artifact_entry
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


def _analysis_ids(analysis_root: Path) -> list[str]:
    if not analysis_root.exists():
        return []
    ids = [p.name for p in analysis_root.iterdir() if p.is_dir()]
    return sorted(ids)


def _latest_analysis_id(analysis_root: Path) -> Optional[str]:
    latest_path = analysis_root / "latest.txt"
    if latest_path.exists():
        val = latest_path.read_text().strip()
        return val or None
    ids = _analysis_ids(analysis_root)
    if not ids:
        return None
    ids.sort(key=lambda name: (analysis_root / name).stat().st_mtime, reverse=True)
    return ids[0]


def generate_notebook(
    run_dir: Path,
    *,
    analysis_id: Optional[str] = None,
    latest: bool = False,
    force: bool = False,
) -> Path:
    _ensure_marimo()
    run_dir = run_dir.resolve()
    manifest = load_manifest(run_dir)
    analysis_root = run_dir / "analysis"
    if analysis_id is None or latest:
        analysis_id = _latest_analysis_id(analysis_root)
    if analysis_id is None:
        raise ValueError("No analysis runs found. Run `cruncher analyze <config>` first.")
    analysis_dir = analysis_root / analysis_id
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis run not found: {analysis_dir}")

    notebooks_dir = analysis_dir / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebooks_dir / "run_overview.py"
    if notebook_path.exists() and not force:
        return notebook_path

    template = _render_template(run_dir, analysis_root)
    notebook_path.write_text(template)

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
    write_manifest(run_dir, manifest)
    config_path = Path(manifest.get("config_path", ""))
    if config_path and config_path.exists():
        update_run_index_from_manifest(config_path, run_dir, manifest)
    return notebook_path


def _render_template(run_dir: Path, analysis_root: Path) -> str:
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
def _(Path):
    run_dir = Path({json.dumps(str(run_dir))})
    analysis_root = Path({json.dumps(str(analysis_root))})
    analysis_ids = sorted([p.name for p in analysis_root.iterdir() if p.is_dir()]) if analysis_root.exists() else []
    return run_dir, analysis_root, analysis_ids


@app.cell
def _(analysis_ids, mo):
    if not analysis_ids:
        mo.stop(True, mo.md("No analysis runs found. Run `cruncher analyze <config>` first."))
    return


@app.cell
def _(analysis_ids, mo):
    default_id = analysis_ids[-1] if analysis_ids else None
    analysis_picker = mo.ui.dropdown(options=analysis_ids, value=default_id, label="analysis_id")
    return analysis_picker


@app.cell
def _(analysis_picker, analysis_root, json):
    analysis_id = analysis_picker.value or ""
    analysis_dir = analysis_root / analysis_id
    summary_path = analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {{}}
    tf_names = summary.get("tf_names", [])
    return analysis_id, analysis_dir, summary, summary_path, tf_names


@app.cell
def _(mo, tf_names):
    default_x = tf_names[0] if len(tf_names) > 0 else None
    default_y = tf_names[1] if len(tf_names) > 1 else default_x
    tf_x = mo.ui.dropdown(options=tf_names, value=default_x, label="x_tf")
    tf_y = mo.ui.dropdown(options=tf_names, value=default_y, label="y_tf")
    return tf_x, tf_y


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
def _(per_pwm_df, plt, tf_x, tf_y):
    fig = None
    if not per_pwm_df.empty and tf_x.value and tf_y.value:
        x_col = f"score_{{tf_x.value}}"
        y_col = f"score_{{tf_y.value}}"
        if x_col in per_pwm_df.columns and y_col in per_pwm_df.columns:
            fig, ax = plt.subplots(figsize=(5, 5))
            data = per_pwm_df[[x_col, y_col]].dropna()
            ax.scatter(data[x_col].to_numpy(), data[y_col].to_numpy(), s=6, alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title("Per-PWM score scatter")
    return fig


@app.cell
def _(analysis_dir, mo):
    plot_dir = analysis_dir / "plots"
    plot_paths = sorted([p for p in plot_dir.glob("*.png")]) if plot_dir.exists() else []
    plot_options = [p.name for p in plot_paths]
    return plot_dir, plot_paths, plot_options


@app.cell
def _(mo, plot_options):
    plot_picker = mo.ui.dropdown(options=plot_options, value=plot_options[0] if plot_options else None, label="plot")
    show_previews = mo.ui.checkbox(label="Show all plot previews", value=False)
    return plot_picker, show_previews


@app.cell
def _(mo, plot_dir, plot_paths, plot_picker, show_previews):
    blocks = []
    selected_name = plot_picker.value
    if selected_name:
        selected_path = plot_dir / selected_name
        if selected_path.exists():
            blocks.append(mo.image(selected_path))
    if show_previews.value and plot_paths:
        blocks.append(mo.md("## Plot previews"))
        preview_paths = [p for p in plot_paths if p.name != selected_name]
        blocks.extend([mo.image(p) for p in preview_paths])
    plot_preview = mo.vstack(blocks) if blocks else mo.md("No plots available.")
    return plot_preview


@app.cell
def _(analysis_id, json, run_dir):
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text())
    artifacts = payload.get("artifacts", [])
    if not analysis_id:
        return artifacts
    prefix = f"analysis/{{analysis_id}}/"
    return [a for a in artifacts if isinstance(a, dict) and str(a.get("path", "")).startswith(prefix)]


@app.cell
def _(mo, analysis_id, summary, summary_path):
    md = [
        "# Cruncher analysis overview",
        "",
        f"Run: {{summary.get('run', '-') }}",
        f"Analysis ID: {{analysis_id}}",
        f"Created at: {{summary.get('created_at', '-') }}",
        f"Summary: {{summary_path}}",
        f"Config used: {{summary.get('config_used', '-') }}",
        f"Analysis settings: {{summary.get('analysis_used', '-') }}",
    ]
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
def _(mo, fig, plot_preview):
    blocks = []
    if fig is not None:
        blocks.append(mo.ui.pyplot(fig))
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
