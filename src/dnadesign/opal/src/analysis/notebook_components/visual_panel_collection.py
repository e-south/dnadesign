"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/visual_panel_collection.py

Campaign-set visual panels for generated OPAL notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping


def render_collection_plot_panel(
    *,
    build_notebook_collection_visual_card_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None,
    collection_visual_description: Callable[[Mapping[str, Any]], str] | None,
    controls: Any | None,
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    render_notebook_plot_choice_image: Callable[..., Any],
    selected_visual_choice: Mapping[str, Any],
) -> Any:
    details = mo.vstack(
        [
            mo.md(
                _require_callable(collection_visual_description, "collection_visual_description")(
                    selected_visual_choice
                )
            ),
            opal_table(
                pl.DataFrame(
                    _require_callable(
                        build_notebook_collection_visual_card_rows,
                        "build_notebook_collection_visual_card_rows",
                    )(selected_visual_choice)
                ),
                page_size=12,
            ),
        ],
        gap=0.35,
    )
    return _panel_stack(
        mo=mo,
        items=[
            controls,
            render_notebook_plot_choice_image(selected_visual_choice, mo=mo),
            mo.accordion({"Collection plot evidence": details}, multiple=True),
        ],
    )


def render_selection_overlap_panel(
    *,
    build_notebook_campaign_set_selection_overlap_card_rows: Callable[[Mapping[str, Any]], list[dict[str, Any]]] | None,
    controls: Any | None,
    mo: Any,
    opal_table: Callable[..., Any],
    pl: Any,
    render_notebook_campaign_set_selection_overlap_image: Callable[..., Mapping[str, Any] | None] | None,
    selected_visual_choice: Mapping[str, Any],
) -> Any:
    payload = _require_callable(
        render_notebook_campaign_set_selection_overlap_image,
        "render_notebook_campaign_set_selection_overlap_image",
    )(selected_visual_choice)
    if payload is None:
        visual = mo.md("No selection overlap is available for the selected campaigns.")
    else:
        visual = mo.image(
            payload["image_bytes"],
            alt=str(payload.get("alt_text") or "Pooled selection overlap."),
            caption=str(payload.get("caption") or "") or None,
            rounded=True,
            style={
                "width": "auto",
                "max-height": "min(62vh, 720px)",
                "max-width": "100%",
                "height": "auto",
                "object-fit": "contain",
                "margin": "0 auto",
                "display": "block",
                "background": "white",
            },
        )
    details = opal_table(
        pl.DataFrame(
            _require_callable(
                build_notebook_campaign_set_selection_overlap_card_rows,
                "build_notebook_campaign_set_selection_overlap_card_rows",
            )(selected_visual_choice)
        ),
        page_size=12,
    )
    return _panel_stack(
        mo=mo,
        items=[controls, visual, mo.accordion({"Selection overlap evidence": details}, multiple=True)],
    )


def _panel_stack(*, mo: Any, items: list[Any | None]) -> Any:
    return mo.vstack([item for item in items if item is not None], gap=0.45)


def _require_callable(value: Callable[..., Any] | None, name: str) -> Callable[..., Any]:
    if value is None:
        raise ValueError(f"{name} is required for this OPAL notebook visual surface.")
    return value


__all__ = ["render_collection_plot_panel", "render_selection_overlap_panel"]
