"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/notebook.py

Generate the single-viewport Marimo review surface for a metastudy bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from .notebook_copy import COMPARATOR_GUIDE_MARKDOWN


def write_review_notebook(out_dir: Path) -> Path:
    """Write a generated notebook that reads only manifest-backed bundle artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "review.py"
    source = _NOTEBOOK_SOURCE.replace("__MARIMO_VERSION__", version("marimo")).replace(
        "__COMPARATOR_GUIDE_MARKDOWN__",
        COMPARATOR_GUIDE_MARKDOWN,
    )
    path.write_text(source, encoding="utf-8")
    if path.stat().st_size <= 0:
        raise RuntimeError("generated response metric metastudy review notebook is empty.")
    return path


_NOTEBOOK_SOURCE = '''import marimo

__generated_with = "__MARIMO_VERSION__"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal import response_metastudy
    from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.campaign_navigation import (
        discover_current_campaign_navigation,
    )
    from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import publication

    return Path, discover_current_campaign_navigation, mo, pd, publication, response_metastudy


@app.cell
def _(Path, discover_current_campaign_navigation, pd, publication):
    bundle_root = Path(__file__).resolve().parent
    manifest_path = bundle_root / "manifest.json"
    plot_manifest_path = bundle_root / "tables" / "plot_manifest.csv"
    if not manifest_path.is_file() or not plot_manifest_path.is_file():
        raise FileNotFoundError("The review notebook requires manifest.json and tables/plot_manifest.csv.")
    bundle_manifest = publication.verify_bundle_artifacts(bundle_root)
    plot_catalog = pd.read_csv(plot_manifest_path)
    required_plot_fields = {
        "plot_id",
        "tier",
        "review_section",
        "section_order",
        "review_step",
        "title",
        "premise",
        "decision_value",
        "rationale",
        "alt_text",
        "non_claim_boundary",
        "data_table",
        "path",
    }
    missing_plot_fields = sorted(required_plot_fields - set(plot_catalog.columns))
    if missing_plot_fields:
        raise ValueError(f"Plot manifest is missing review fields: {missing_plot_fields}")
    if plot_catalog.empty or plot_catalog["plot_id"].duplicated().any():
        raise ValueError("Plot manifest must contain unique deliverables.")
    campaign_navigation = discover_current_campaign_navigation(bundle_root)
    return bundle_manifest, bundle_root, campaign_navigation, plot_catalog


@app.cell
def _(mo):
    section_options = {
        "Assay and labels": "assay_and_labels",
        "Historical model screens": "historical_model_screens",
        "RMF comparator": "rmf_comparator",
        "SFXI comparator": "sfxi_comparator",
    }
    review_section = mo.ui.dropdown(
        options=section_options,
        value=next(iter(section_options)),
        label="Review section",
        full_width=True,
    )
    return (review_section,)


@app.cell
def _(mo, plot_catalog, review_section):
    section_catalog = plot_catalog.loc[plot_catalog["review_section"].eq(review_section.value)].copy()
    if section_catalog.empty:
        raise ValueError(f"Review section {review_section.value!r} has no figures.")
    if section_catalog["section_order"].isna().any():
        raise ValueError("Review-section figures require an explicit order.")
    section_catalog = section_catalog.sort_values("section_order", kind="mergesort")
    deliverable_options = {
        f"{int(row.section_order)}. {row.title}": row.plot_id
        for row in section_catalog.itertuples(index=False)
    }
    deliverable = mo.ui.dropdown(
        options=deliverable_options,
        value=next(iter(deliverable_options)),
        label="Figure",
        searchable=True,
        full_width=True,
    )
    return deliverable, section_catalog


@app.cell
def _(deliverable, mo, review_section):
    controls = mo.hstack(
        [review_section, deliverable],
        widths=[1, 2],
        gap=0.75,
        align="end",
        wrap=True,
    )
    return (controls,)


@app.cell
def _(bundle_manifest, campaign_navigation, mo, response_metastudy):
    review_summary = response_metastudy.build_review_summary(bundle_manifest)
    study_context = mo.md(
        f"""
        **Scope:** {review_summary.scope}

        **Observed labels:** {review_summary.label_state}

        **Predictor support:** {review_summary.predictor_support}

        **Basis:** {review_summary.basis}

        **Primary assay summary:** {review_summary.primary_assay_summary}

        **Evidence base:** {review_summary.evidence_base}

        **Prospective hill climb:** {review_summary.prospective_hill_climb}
        """
    )
    if campaign_navigation is None:
        campaign_review = mo.md("Current OPAL navigation is unavailable outside a source checkout.")
    else:
        objective_label = "Objective" if len(campaign_navigation.objective_names) == 1 else "Objectives"
        notebook_status = (
            "generated" if campaign_navigation.notebook_materialized else "generate with the command below"
        )
        campaign_review = mo.md(
            f"""
            **Campaign:** `{campaign_navigation.campaign_slug}`

            **Selection views:** {", ".join(campaign_navigation.selection_view_ids)}

            **{objective_label}:** {", ".join(campaign_navigation.objective_names)}

            **Config:** `{campaign_navigation.config_path}`

            **Notebook target:** `{campaign_navigation.notebook_path}` ({notebook_status})

            ```bash
            {campaign_navigation.run_command}
            ```
            """
        )
    review_context = mo.accordion(
        {
            "Study context": study_context,
            "Current OPAL review — outside this evidence bundle": campaign_review,
            "Objective comparators": mo.md(
                """__COMPARATOR_GUIDE_MARKDOWN__"""
            ),
        }
    )
    return (review_context,)


@app.cell
def _(Path, bundle_root, controls, deliverable, mo, pd, plot_catalog, review_context):
    selected_rows = plot_catalog.loc[plot_catalog["plot_id"].eq(deliverable.value)]
    if len(selected_rows) != 1:
        raise ValueError(f"Deliverable selection must resolve exactly once: {deliverable.value!r}")
    selected = selected_rows.iloc[0]
    plot_path = bundle_root / selected["path"]
    table_path = bundle_root / selected["data_table"]
    if not plot_path.is_file() or not table_path.is_file():
        raise FileNotFoundError(f"Missing plot or source table for {selected['plot_id']!r}.")
    evidence_table = pd.read_csv(table_path)
    metadata = mo.md(
        f"""
        **Premise:** {selected['premise']}

        **Decision value:** {selected['decision_value']}

        **Rationale:** {selected['rationale']}

        **Limit:** {selected['non_claim_boundary']}
        """
    )
    details = mo.accordion(
        {
            "Evidence contract": metadata,
            f"Source table · {Path(table_path).name}": mo.ui.dataframe(evidence_table, page_size=12),
        }
    )
    viewport = mo.vstack(
        [
            controls,
            mo.image(
                str(plot_path),
                alt=str(selected["alt_text"]),
                width="100%",
                style={
                    "max-width": "100%",
                    "max-height": "68vh",
                    "height": "auto",
                    "object-fit": "contain",
                },
            ),
            details,
            review_context,
        ],
        gap=1.0,
    )
    viewport
    return


if __name__ == "__main__":
    app.run()
'''


__all__ = ["write_review_notebook"]
