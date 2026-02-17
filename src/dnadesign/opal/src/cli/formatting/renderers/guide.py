"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting/renderers/guide.py

Renders guided workflow reports and next-step recommendations for OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Iterable

from ...guidance import GuidanceReport, NextGuidance


def _bullets(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def render_guide_text(report: GuidanceReport) -> str:
    lines: list[str] = []
    lines.append("Guided Workflow")
    lines.append("")
    lines.append(f"Campaign: {report.campaign['name']} ({report.campaign['slug']})")
    lines.append(f"Workflow: {report.workflow_key}")
    lines.append(f"Config: {report.campaign['config_path']}")
    lines.append(f"Workdir: {report.campaign['workdir']}")
    lines.append("")
    lines.append("Plugin wiring")
    lines.append(f"- model: {report.plugins['model']['name']}")
    objective_names = [str(row.get("name")) for row in report.plugins["objectives"]]
    lines.append(f"- objectives: {', '.join(objective_names)}")
    lines.append(f"- selection: {report.plugins['selection']['name']}")
    lines.append("")
    lines.append("Round semantics")
    lines.append(f"- observed_round: {report.round_semantics['observed_round']}")
    lines.append(f"- labels_as_of: {report.round_semantics['labels_as_of']}")
    lines.append("")
    lines.append("Runbook")
    for idx, step in enumerate(report.steps, start=1):
        lines.append(f"{idx}. {step.title}")
        lines.append(f"   Why: {step.why}")
        lines.append(f"   Command: {step.command}")
    lines.append("")
    lines.append("Common errors")
    lines.extend([f"- {item}" for item in report.common_errors])
    lines.append("")
    lines.append("Learn more (docs)")
    lines.extend([f"- {path}" for path in report.learn_more["docs"]])
    lines.append("")
    lines.append("Learn more (source)")
    lines.extend([f"- {path}" for path in report.learn_more["source"]])
    return "\n".join(lines)


def render_guide_markdown(report: GuidanceReport) -> str:
    lines: list[str] = []
    lines.append("## Guided Workflow")
    lines.append("")
    lines.append(f"- **Campaign:** `{report.campaign['name']}` (`{report.campaign['slug']}`)")
    lines.append(f"- **Workflow key:** `{report.workflow_key}`")
    lines.append(f"- **Config:** `{report.campaign['config_path']}`")
    lines.append("")
    lines.append("### Plugin Wiring")
    lines.append("")
    lines.append(f"- model: `{report.plugins['model']['name']}`")
    objective_names = [str(row.get("name")) for row in report.plugins["objectives"]]
    lines.append(f"- objectives: `{', '.join(objective_names)}`")
    lines.append(f"- selection: `{report.plugins['selection']['name']}`")
    lines.append("")
    lines.append("### Round Semantics")
    lines.append("")
    lines.append(f"- `--observed-round`: {report.round_semantics['observed_round']}")
    lines.append(f"- `--labels-as-of`: {report.round_semantics['labels_as_of']}")
    lines.append("")
    lines.append("### Runbook")
    lines.append("")
    for idx, step in enumerate(report.steps, start=1):
        lines.append(f"{idx}. **{step.title}**")
        lines.append(f"   - Why: {step.why}")
        lines.append("   - Command:")
        lines.append("")
        lines.append("```bash")
        lines.append(step.command)
        lines.append("```")
    lines.append("")
    lines.append("### Common Errors")
    lines.append("")
    lines.append(_bullets(report.common_errors))
    lines.append("")
    lines.append("### Learn More")
    lines.append("")
    lines.append("Docs:")
    lines.append(_bullets(report.learn_more["docs"]))
    lines.append("")
    lines.append("Source:")
    lines.append(_bullets(report.learn_more["source"]))
    lines.append("")
    return "\n".join(lines)


def render_next_human(next_report: NextGuidance) -> str:
    lines: list[str] = []
    lines.append("Guided Next Step")
    lines.append("")
    lines.append(f"stage: {next_report.stage}")
    lines.append(f"reason: {next_report.reason}")
    lines.append(
        f"context: labels_as_of={next_report.labels_as_of}, "
        f"observed_round={next_report.observed_round}, "
        f"labels_in_observed_round={next_report.labels_in_observed_round}"
    )
    lines.append("")
    lines.append("Next commands")
    lines.extend([f"- {cmd}" for cmd in next_report.next_commands])
    lines.append("")
    lines.append("Learn more")
    lines.extend([f"- {path}" for path in next_report.learn_more])
    return "\n".join(lines)
