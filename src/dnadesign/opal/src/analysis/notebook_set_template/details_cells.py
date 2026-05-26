from __future__ import annotations

from ._support import block


def render_details_cells() -> str:
    """Render evidence, secondary detail panels, layout, and app entrypoint cells."""

    return "\n\n".join((_evidence_cell(), _details_cell(), _layout_cell(), _main_cell()))


def _evidence_cell() -> str:
    return block(
        """
        @app.cell
        def _(build_notebook_evidence_rows, mo, opal_table, pl, selected_campaign_model):
            _rows = build_notebook_evidence_rows(selected_campaign_model)
            if _rows:
                evidence_panel = opal_table(pl.DataFrame(_rows), page_size=10)
            else:
                evidence_panel = mo.md("No warnings or stale artifacts reported for this campaign.")
            return evidence_panel
        """
    )


def _details_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            build_notebook_artifact_garden_rows,
            build_notebook_artifact_garden_summary_rows,
            build_notebook_change_rows,
            build_notebook_change_summary_rows,
            build_notebook_metric_definition_rows,
            mo,
            opal_table,
            pl,
            selected_campaign_model,
        ):
            _metric_rows = build_notebook_metric_definition_rows(selected_campaign_model)
            if _metric_rows:
                metric_definitions_panel = opal_table(pl.DataFrame(_metric_rows), page_size=10)
            else:
                metric_definitions_panel = mo.md("No manifest-backed plot metric definitions are available.")

            _change_rows = build_notebook_change_rows(selected_campaign_model)
            if _change_rows:
                changes_table = opal_table(pl.DataFrame(_change_rows), page_size=10)
            else:
                changes_table = mo.md("No round changes are available yet.")
            changes_panel = mo.vstack(
                [
                    opal_table(pl.DataFrame(build_notebook_change_summary_rows(selected_campaign_model)), page_size=8),
                    changes_table,
                ]
            )

            _artifact_rows = build_notebook_artifact_garden_rows(selected_campaign_model)
            _artifact_summary_rows = build_notebook_artifact_garden_summary_rows(selected_campaign_model)
            artifact_rows_panel = (
                opal_table(pl.DataFrame(_artifact_rows), page_size=10)
                if _artifact_rows
                else mo.md("No artifact garden rows are available.")
            )
            artifact_garden_panel = mo.vstack(
                [
                    opal_table(pl.DataFrame(_artifact_summary_rows), page_size=10),
                    artifact_rows_panel,
                ]
            )
            return artifact_garden_panel, changes_panel, metric_definitions_panel
        """
    )


def _layout_cell() -> str:
    return block(
        """
        @app.cell
        def _(
            active_view_mode,
            artifact_garden_panel,
            campaign_summary_df,
            collection_set_ui,
            campaign_ui,
            changes_panel,
            evidence_panel,
            header_md,
            metric_definitions_panel,
            mo,
            opal_table,
            plot_panel,
            selected_campaign_header_md,
            selected_overview_panel,
            selected_validity_md,
            view_mode_ui,
        ):
            _items = [header_md]
            _top_control_items = [view_mode_ui]
            if active_view_mode != "Campaign set":
                _top_control_items.append(campaign_ui)
            elif collection_set_ui is not None:
                _top_control_items.append(collection_set_ui)
            _items.append(mo.vstack(_top_control_items, gap=0.20))
            if active_view_mode != "Campaign set":
                _items.append(selected_campaign_header_md)
            _accordion_items = {
                "OPAL campaigns at a glance": opal_table(campaign_summary_df, page_size=12),
            }
            if active_view_mode != "Campaign set":
                _accordion_items.update(
                    {
                        "Selected OPAL campaign": selected_overview_panel,
                        "Validity": selected_validity_md,
                        "Changes": changes_panel,
                        "Metric definitions": metric_definitions_panel,
                        "Artifacts": artifact_garden_panel,
                        "Warnings and stale artifacts": evidence_panel,
                    }
                )
            _items.extend([plot_panel, mo.accordion(_accordion_items, multiple=True, lazy=True)])
            mo.vstack(_items)
            return
        """
    )


def _main_cell() -> str:
    return block(
        """
        if __name__ == "__main__":
            app.run()
        """
    )


__all__ = ["render_details_cells"]
