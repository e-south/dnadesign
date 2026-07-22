"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/reporting/review/rendering/outcome.py

Outcome-section rendering for DenseGen axis probe reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from .formatting import _e


def markdown_outcome_lines(outcome_summary: Mapping[str, Any]) -> list[str]:
    """Return the Markdown outcome section for a probe review."""

    return [
        "## Probe Outcome",
        "",
        f"- headline: {outcome_summary.get('headline', 'not recorded')}",
        f"- operator read: {outcome_summary.get('operator_read', 'not recorded')}",
        f"- next action: {outcome_summary.get('next_action', 'not recorded')}",
        f"- interpretation boundary: {outcome_summary.get('interpretation_boundary', 'not recorded')}",
        "",
    ]


def html_outcome_section(outcome_summary: Mapping[str, Any]) -> str:
    """Return the HTML outcome section for a probe review."""

    return f"""
      <section>
        <h2>Probe Outcome</h2>
        <dl>
          <dt>Headline</dt><dd>{_e(outcome_summary.get("headline", "not recorded"))}</dd>
          <dt>Operator read</dt><dd>{_e(outcome_summary.get("operator_read", "not recorded"))}</dd>
          <dt>Next action</dt><dd>{_e(outcome_summary.get("next_action", "not recorded"))}</dd>
          <dt>Boundary</dt><dd>{_e(outcome_summary.get("interpretation_boundary", "not recorded"))}</dd>
        </dl>
      </section>
    """
