"""Notebook runtime assembly helpers."""

from __future__ import annotations

from dnadesign.latentdna.src.notebooks.browser_runtime import (
    _parse_deliverable_markdown,
    resolve_plot_doc_block,
    resolve_runtime_hue_kinds,
)


def test_resolve_plot_doc_block_uses_plot_specific_subsection() -> None:
    markdown = """# Reference-margin analysis

Short deliverable summary.

## Why this deliverable exists

Deliverable context.

## Plot guide

Guide text.

### reference_neighbor_evidence | Reference-neighbor evidence

#### Plot details

**Data.** Neighbor evidence details.

**Definition.** Read the neighborhood metrics directly.
"""

    parsed = _parse_deliverable_markdown(markdown)
    block = resolve_plot_doc_block(
        plot_id="reference_neighbor_evidence",
        deliverable_summary="Fallback summary.",
        parsed_markdown=parsed,
    )

    assert parsed["summary_markdown"] == "Short deliverable summary."
    assert block["title"] == "Reference-neighbor evidence"
    assert "Neighbor evidence details." in str(block["markdown"])
    assert "**Data.** Neighbor evidence details." in block["plot_details_md"]
    assert block["warning"] is None


def test_resolve_plot_doc_block_warns_when_subsection_is_missing() -> None:
    markdown = """# Context geometry audit

Deliverable fallback summary.

## Why this deliverable exists

Deliverable context.
"""

    parsed = _parse_deliverable_markdown(markdown)
    block = resolve_plot_doc_block(
        plot_id="context_geometry_summary",
        deliverable_summary="Fallback summary from deliverable contract.",
        parsed_markdown=parsed,
    )

    assert block["title"] == "Context stability summary"
    assert block["markdown"] == "Deliverable fallback summary."
    assert block["plot_details_md"] == "Deliverable fallback summary."
    assert block["warning"] == "Missing plot-specific study-doc subsection for `context_geometry_summary`."


def test_resolve_runtime_hue_kinds_preserves_binary_entries() -> None:
    assert resolve_runtime_hue_kinds(
        ["design_family", "is_control", "context_shift_l2", "ignored_metric"],
        {
            "design_family": "categorical",
            "is_control": "binary",
            "context_shift_l2": "continuous",
            "ignored_metric": "unknown",
        },
    ) == {
        "design_family": "categorical",
        "is_control": "binary",
        "context_shift_l2": "continuous",
    }
