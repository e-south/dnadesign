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


def write_review_notebook(out_dir: Path) -> Path:
    """Write a generated notebook that reads only manifest-backed bundle artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "review.py"
    source = _NOTEBOOK_SOURCE.replace("__MARIMO_VERSION__", version("marimo"))
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
    from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import publication

    return Path, mo, pd, publication, response_metastudy


@app.cell
def _(Path, pd, publication):
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
    return bundle_manifest, bundle_root, plot_catalog


@app.cell
def _(mo):
    tier_options = {
        "Primary evidence": "primary_decision",
        "Metric diagnostics": "metric_diagnostic",
        "Screen appendix": "screen_appendix",
    }
    tier = mo.ui.dropdown(
        options=tier_options,
        value=next(iter(tier_options)),
        label="Evidence tier",
        full_width=True,
    )
    return (tier,)


@app.cell
def _(mo, plot_catalog, tier):
    tier_catalog = plot_catalog.loc[plot_catalog["tier"].eq(tier.value)].copy()
    if tier_catalog.empty:
        raise ValueError(f"Evidence tier {tier.value!r} has no figures.")
    if tier.value == "primary_decision":
        if tier_catalog["review_step"].isna().any():
            raise ValueError("Primary evidence figures require an explicit review step.")
        tier_catalog = tier_catalog.sort_values("review_step", kind="mergesort")
    deliverable_options = {
        (
            f"{int(row.review_step)}. {row.title}"
            if tier.value == "primary_decision"
            else str(row.title)
        ): row.plot_id
        for row in tier_catalog.itertuples(index=False)
    }
    deliverable = mo.ui.dropdown(
        options=deliverable_options,
        value=next(iter(deliverable_options)),
        label="Figure",
        searchable=True,
        full_width=True,
    )
    return deliverable, tier_catalog


@app.cell
def _(bundle_manifest, deliverable, mo, response_metastudy, tier):
    review_summary = response_metastudy.build_review_summary(bundle_manifest)
    summary = mo.md(
        f"""
        # Stress response metric review

        **Decision:** {review_summary.decision}
        **Basis:** {review_summary.basis}
        **Primary assay summary:** {review_summary.primary_assay_summary}
        **Evidence base:** {review_summary.evidence_base}
        **Prospective hill climb:** {review_summary.prospective_hill_climb}
        """
    )
    metric_rule = mo.md(
        """
        **RMF selection rule**

        A target mask declares which measured states should be ON and OFF. RMF converts response separation,
        target-ON fluorescence, and target-OFF control into three calibrated signed margins. Positive values
        clear the configured requirement and zero is its boundary. The score is their minimum, so top-K selects
        candidates with the largest weakest requirement; a strong component cannot compensate for a failed one.

        Scores are ordered within one fixed target mask and calibration. Changing the mask changes the question,
        so scores from different target views are not directly comparable.
        """
    )
    metric_guide = mo.accordion(
        {
            "How the target mask changes the score": mo.md(
                """
                Reader publishes the same eight measured values for every target view:

                - `r_i` is the median across design wells of the 6-12 h mean
                  `log2[(YFP / CFP)_design,i(t)]`.
                - `b_i` is the design-well median 6-12 h mean
                  `log2[(YFP / OD600)_design,i(t)]` minus the same-state pDual-10 well median.

                The study target mask changes only which states are treated as ON and OFF:

                | Target | ON conditions | OFF conditions |
                | --- | --- | --- |
                | Ethanol | ethanol; ethanol + ciprofloxacin | no stress; ciprofloxacin |
                | Ciprofloxacin | ciprofloxacin; ethanol + ciprofloxacin | no stress; ethanol |
                | AND | ethanol + ciprofloxacin | no stress; ethanol; ciprofloxacin |
                | OR (screen only) | ethanol; ciprofloxacin; ethanol + ciprofloxacin | no stress |

                `m_response = min_ON(r) - max_OFF(r)`
                `b_on = min_ON(b)`
                `b_off = max_OFF(b)`
                `S_RMF = min(q_response, q_on, q_off)`

                A new mask can therefore change all four score channels for the same measured design.
                Scores order candidates within one target; they are not calibrated for direct comparison
                between targets.
                """
            )
        }
    )
    controls = mo.hstack(
        [tier, deliverable],
        widths=[1, 2],
        gap=0.75,
        align="end",
        wrap=True,
    )
    mo.vstack([summary, metric_rule, metric_guide, controls], gap=1.0)
    return (controls,)


@app.cell
def _(Path, bundle_root, deliverable, mo, pd, plot_catalog):
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
                caption=str(selected["decision_value"]),
            ),
            details,
        ],
        gap=1.0,
    )
    viewport
    return


if __name__ == "__main__":
    app.run()
'''


__all__ = ["write_review_notebook"]
