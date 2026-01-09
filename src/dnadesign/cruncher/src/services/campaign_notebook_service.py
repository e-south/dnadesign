"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/services/campaign_notebook_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.services.notebook_service import ensure_marimo


def _validate_campaign_summary_inputs(summary_dir: Path) -> None:
    required = [
        summary_dir / "campaign_summary.csv",
        summary_dir / "campaign_best.csv",
        summary_dir / "campaign_manifest.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_blob = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing campaign summary artifacts required for the notebook: "
            f"{missing_blob}. Run `cruncher campaign summarize` first."
        )


def generate_campaign_notebook(
    summary_dir: Path,
    *,
    force: bool = False,
    strict: bool = True,
) -> Path:
    ensure_marimo()
    summary_dir = summary_dir.resolve()
    if strict:
        _validate_campaign_summary_inputs(summary_dir)

    notebooks_dir = summary_dir / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebooks_dir / "campaign_overview.py"
    if notebook_path.exists() and not force:
        return notebook_path

    template = _render_template()
    notebook_path.write_text(template)
    return notebook_path


def _render_template() -> str:
    return """import marimo as mo

app = mo.App(width=\"full\")

@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import pandas as pd
    return mo, json, Path, pd


@app.cell
def _(Path, mo):
    notebook_path = Path(__file__).resolve()
    summary_dir = notebook_path.parent.parent
    if summary_dir.name == "notebooks":
        summary_dir = summary_dir.parent
    if not summary_dir.exists():
        mo.stop(True, mo.md(f"Campaign summary directory not found: {summary_dir}"))
    summary_path = summary_dir / "campaign_summary.csv"
    best_path = summary_dir / "campaign_best.csv"
    manifest_path = summary_dir / "campaign_manifest.json"
    if not summary_path.exists():
        mo.stop(True, mo.md(f"Missing summary CSV: {summary_path}"))
    return summary_dir, summary_path, best_path, manifest_path


@app.cell
def _(Path, json):
    def _load_json(path: Path):
        if not path.exists():
            return {}, f"Missing JSON at {path}"
        try:
            return json.loads(path.read_text()), None
        except Exception as exc:  # pragma: no cover - notebook UX
            return {}, f"Failed to parse JSON at {path}: {exc}"
    return _load_json


@app.cell
def _(best_path, manifest_path, pd, summary_path, _load_json):
    summary_df = pd.read_csv(summary_path)
    best_df = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()
    manifest, manifest_error = _load_json(manifest_path)
    return best_df, manifest, manifest_error, summary_df


@app.cell
def _(manifest, manifest_error, mo):
    if manifest_error:
        manifest_summary = mo.md(f"Warning: {manifest_error}")
    else:
        name = manifest.get("campaign_name", "-")
        campaign_id = manifest.get("campaign_id", "-")
        categories = manifest.get("categories", {})
        selectors = manifest.get("selectors", {})
        rules = manifest.get("rules", {})
        lines = [
            "# Campaign overview",
            f"**Campaign:** {name}",
            f"**Campaign ID:** {campaign_id}",
            f"**Categories:** {', '.join(categories.keys()) if categories else '-'}",
            f"**Selectors:** {selectors if selectors else '-'}",
            f"**Rules:** {rules if rules else '-'}",
        ]
        manifest_summary = mo.md("\n".join(lines))
    return manifest_summary


@app.cell
def _(summary_df, mo):
    if summary_df.empty:
        mo.stop(True, mo.md("campaign_summary.csv is empty."))
    set_options = ["all"] + [str(idx) for idx in sorted(summary_df["set_index"].unique())]
    set_picker = mo.ui.dropdown(options=set_options, value="all", label="set_index")
    return set_picker


@app.cell
def _(set_picker, summary_df):
    selected = set_picker.value
    if selected and selected != "all":
        filtered_df = summary_df[summary_df["set_index"] == int(selected)]
    else:
        filtered_df = summary_df
    return filtered_df


@app.cell
def _(best_df, filtered_df, mo, set_picker):
    controls = mo.hstack([set_picker])
    summary_table = mo.ui.dataframe(filtered_df)
    best_table = mo.ui.dataframe(best_df) if not best_df.empty else mo.md("No best rows found.")
    blocks = [controls, summary_table, best_table]
    overview_block = mo.vstack(blocks)
    return overview_block


@app.cell
def _(manifest_summary, mo, overview_block):
    overview = mo.vstack([manifest_summary, overview_block])
    return overview


@app.cell
def _(summary_dir, mo):
    plots_dir = summary_dir / "plots"
    images = []
    for name in ["best_jointscore_bar.png", "tf_coverage_heatmap.png", "pairgrid_overview.png"]:
        path = plots_dir / name
        if path.exists():
            images.append(mo.image(path))
    if images:
        plots_block = mo.vstack(images)
    else:
        plots_block = mo.md("No campaign plots found under plots/")
    return plots_block


@app.cell
def _(mo, overview, plots_block):
    tabs = mo.ui.tabs({
        "Overview": overview,
        "Plots": plots_block,
    })
    return tabs


if __name__ == "__main__":
    app.run()
"""
