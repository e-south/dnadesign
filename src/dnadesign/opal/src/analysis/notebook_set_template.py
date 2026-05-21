"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/notebook_set_template.py

Renders marimo notebook templates for OPAL campaign sets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent


def render_campaign_set_notebook(config_paths: list[Path], *, round_selector: str) -> str:
    """Render a marimo notebook template for a campaign set."""

    try:
        import marimo as _marimo
    except Exception:
        _marimo = None
    marimo_version = "unknown" if _marimo is None else getattr(_marimo, "__version__", "unknown")
    path_literals = repr([str(Path(path)) for path in config_paths])
    template = dedent(
        """
        import marimo

        __generated_with = "__GENERATED_WITH__"

        app = marimo.App(width="medium")


        @app.cell
        def _():
            from pathlib import Path

            import marimo as mo
            import polars as pl

            from dnadesign.opal import build_campaign_set_notebook_view_model
            return Path, build_campaign_set_notebook_view_model, mo, pl


        @app.cell
        def _(Path, build_campaign_set_notebook_view_model):
            config_paths = [Path(path) for path in __CONFIG_PATHS__]
            campaign_set_view_model = build_campaign_set_notebook_view_model(
                config_paths,
                round_selector=__DEFAULT_ROUND__,
            )
            campaigns = campaign_set_view_model["campaigns"]
            return campaign_set_view_model, campaigns, config_paths


        @app.cell
        def _(campaign_set_view_model, campaigns, mo, pl):
            _rows = []
            _labels = []
            for campaign_model in campaigns:
                _campaign = campaign_model["campaign"]
                _status = campaign_model.get("status") or {}
                _plot_count = len(campaign_model.get("plot_manifests") or [])
                _stale_count = len(campaign_model.get("stale_artifacts") or [])
                _warning_count = len(campaign_model.get("warnings") or [])
                _label = f"{_campaign['slug']} | {_status.get('progress_status') or 'unknown'}"
                _labels.append(_label)
                _rows.append(
                    {
                        "label": _label,
                        "campaign": _campaign["slug"],
                        "status": _status.get("progress_status"),
                        "round_count": _status.get("round_count"),
                        "latest_run_id": _status.get("latest_run_id"),
                        "x_column": _campaign.get("x_column"),
                        "label_source": _campaign.get("label_source"),
                        "plots": _plot_count,
                        "stale": _stale_count,
                        "warnings": _warning_count,
                    }
                )
            campaign_ui = mo.ui.dropdown(_labels, value=_labels[0], label="Campaign")
            campaign_summary_df = pl.DataFrame(_rows)
            header_md = mo.md(
                "# OPAL Campaign Set Notebook\\n\\n"
                f"Campaigns: `{campaign_set_view_model['campaign_count']}`  "
                f"Round selector: `{campaign_set_view_model['round_selector']}`  "
                f"Generated with marimo: `{__generated_with}`"
            )
            return campaign_summary_df, campaign_ui, header_md


        @app.cell
        def _(campaign_ui, campaigns):
            selected_label = str(campaign_ui.value)
            selected_campaign_model = next(
                campaign_model
                for campaign_model in campaigns
                if f"{campaign_model['campaign']['slug']} | "
                f"{campaign_model.get('status', {}).get('progress_status') or 'unknown'}"
                == selected_label
            )
            return selected_campaign_model, selected_label


        @app.cell
        def _(mo, selected_campaign_model):
            _campaign = selected_campaign_model["campaign"]
            _status = selected_campaign_model.get("status") or {}
            _progress = selected_campaign_model.get("progress") or {}
            _event_contract = _progress.get("event_contract") or {}
            _overview_lines = [
                "### Selected campaign",
                "",
                f"- Campaign: `{_campaign['slug']}`",
                f"- Config: `{_campaign['config_path']}`",
                f"- Workdir: `{_campaign['workdir']}`",
                f"- Status: `{_status.get('progress_status')}`",
                f"- Round count: `{_status.get('round_count')}`",
                f"- Latest run ID: `{_status.get('latest_run_id')}`",
                f"- X column: `{_campaign.get('x_column')}`",
                f"- Label source: `{_campaign.get('label_source')}`",
                f"- Event contract: `{_event_contract.get('schema_version')}`",
            ]
            selected_overview_md = mo.md("\\n".join(_overview_lines))
            return selected_overview_md


        @app.cell
        def _(selected_campaign_model):
            _plot_choices = []
            for _manifest in selected_campaign_model.get("plot_manifests", []):
                if _manifest.get("status") != "written":
                    continue
                _media_outputs = [
                    output
                    for output in _manifest.get("outputs", [])
                    if output.get("role") == "media" and output.get("exists")
                ]
                if not _media_outputs:
                    continue
                _path = _media_outputs[0]["path"]
                _label = f"{_manifest.get('name')} ({_manifest.get('kind')})"
                _plot_choices.append(
                    {
                        "label": _label,
                        "path": _path,
                        "manifest": _manifest,
                    }
                )
            plot_choices = _plot_choices
            return plot_choices


        @app.cell
        def _(mo, plot_choices):
            if plot_choices:
                _labels = [choice["label"] for choice in plot_choices]
                plot_ui = mo.ui.dropdown(_labels, value=_labels[0], label="Plot")
            else:
                plot_ui = None
            return plot_ui


        @app.cell
        def _(Path, mo, plot_choices, plot_ui):
            if plot_ui is None:
                plot_panel = mo.md(
                    "### Plot deliverables\\n\\n"
                    "No written manifest-backed plot media are available for this campaign."
                )
            else:
                _selected = str(plot_ui.value)
                _choice = next(choice for choice in plot_choices if choice["label"] == _selected)
                _manifest = _choice["manifest"]
                _details = [
                    "### Plot deliverables",
                    "",
                    f"- Plot: `{_manifest.get('name')}`",
                    f"- Kind: `{_manifest.get('kind')}`",
                    f"- Status: `{_manifest.get('status')}`",
                    f"- Generated: `{_manifest.get('generated_at')}`",
                    f"- Freshness: `{(_manifest.get('freshness') or {}).get('status')}`",
                    f"- Tidy CSV: `{_manifest.get('tidy_csv') or 'none'}`",
                    f"- Path: `{_choice['path']}`",
                ]
                plot_panel = mo.vstack(
                    [
                        mo.md("\\n".join(_details)),
                        plot_ui,
                        mo.image(Path(_choice["path"]).read_bytes()),
                    ]
                )
            return plot_panel


        @app.cell
        def _(mo, pl, selected_campaign_model):
            _warnings = selected_campaign_model.get("warnings") or []
            _stale = selected_campaign_model.get("stale_artifacts") or []
            _rows = []
            for _warning in _warnings:
                _rows.append(
                    {
                        "source": "warning",
                        "category": _warning.get("category"),
                        "severity": _warning.get("severity"),
                        "message": _warning.get("message"),
                        "path": _warning.get("path"),
                    }
                )
            for _artifact in _stale:
                _rows.append(
                    {
                        "source": "stale_artifact",
                        "category": _artifact.get("category"),
                        "severity": _artifact.get("severity"),
                        "message": _artifact.get("message"),
                        "path": _artifact.get("path"),
                    }
                )
            if _rows:
                evidence_panel = mo.ui.table(pl.DataFrame(_rows), page_size=10)
            else:
                evidence_panel = mo.md("No warnings or stale artifacts reported for this campaign.")
            return evidence_panel


        @app.cell
        def _(
            campaign_summary_df,
            campaign_ui,
            evidence_panel,
            header_md,
            mo,
            plot_panel,
            selected_overview_md,
        ):
            mo.vstack(
                [
                    header_md,
                    campaign_ui,
                    mo.accordion(
                        {
                            "Campaigns at a glance": mo.ui.table(campaign_summary_df, page_size=12),
                            "Selected campaign": selected_overview_md,
                            "Plot deliverables": plot_panel,
                            "Warnings and stale artifacts": evidence_panel,
                        },
                        multiple=True,
                        lazy=True,
                    ),
                ]
            )
            return


        if __name__ == "__main__":
            app.run()
        """
    ).strip("\n")
    return (
        template.replace("__CONFIG_PATHS__", path_literals)
        .replace("__DEFAULT_ROUND__", repr(str(round_selector)))
        .replace("__GENERATED_WITH__", str(marimo_version))
        + "\n"
    )
