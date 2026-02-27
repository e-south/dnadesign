"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/cli/notebook_template.py

Marimo notebook template sections and renderer for DenseGen CLI notebook scaffolding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

from ..config.root import load_config
from .notebook_cells_template import notebook_template_cells
from .notebook_template_cells import baserender_export_cell_template as _baserender_export_cell_template
from .notebook_template_cells import records_export_cell_template as _records_export_cell_template
from .run_intro import (
    RunDetailsPathsContext,
    build_run_details_payload,
    extract_contract,
    extract_outcome,
)


@dataclass(frozen=True)
class NotebookTemplateContext:
    run_root: Path
    cfg_path: Path
    records_path: Path
    output_source: str
    usr_root: Path | None
    usr_dataset: str | None
    notebook_path: Path | None = None


_KNOWN_ACRONYMS = {
    "cbc": "CBC",
    "dna": "DNA",
    "fimo": "FIMO",
    "gurobi": "GUROBI",
    "pwm": "PWM",
    "tfbs": "TFBS",
    "usr": "USR",
}


def _format_workspace_heading(raw_name: str) -> str:
    text = str(raw_name or "").strip()
    if not text:
        return "DenseGen Workspace"
    tokens = [token for token in re.split(r"[_\-\s]+", text) if token]
    if not tokens:
        return "DenseGen Workspace"
    words: list[str] = []
    for token in tokens:
        normalized = token.strip()
        lower = normalized.lower()
        if lower in _KNOWN_ACRONYMS:
            words.append(_KNOWN_ACRONYMS[lower])
            continue
        words.append(normalized[:1].upper() + normalized[1:])
    return " ".join(words)


def _build_workspace_intro(context: NotebookTemplateContext) -> str:
    payload = _build_workspace_intro_payload(context)
    summary_lines = payload.get("summary_lines", [])
    sections = payload.get("sections", [])
    rendered_lines: list[str] = ["## Run details"]
    rendered_lines.extend(str(line) for line in summary_lines if str(line).strip())
    for section in sections:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or "").strip()
        body = str(section.get("body_md") or "").strip()
        if title:
            rendered_lines.append("")
            rendered_lines.append(f"### {title}")
            if body:
                rendered_lines.append(body)
    return "\n".join(rendered_lines).strip()


def _build_workspace_intro_payload(context: NotebookTemplateContext) -> dict[str, object]:
    payload: dict[str, object] = {}
    config_error: str | None = None
    try:
        loaded = load_config(context.cfg_path)
        parsed = loaded.root.model_dump(mode="python")
        if isinstance(parsed, dict):
            payload = parsed
    except Exception as exc:
        config_error = (
            f"config could not be validated against schema at notebook generation time ({exc.__class__.__name__})."
        )

    manifest_path = context.run_root / "outputs" / "meta" / "run_manifest.json"
    manifest_payload: dict[str, object] | None = None
    outcome_error: str | None = None
    if not manifest_path.exists():
        outcome_error = (
            "run outcomes are not available yet; run `dense run` to materialize `outputs/meta/run_manifest.json`."
        )
    else:
        try:
            parsed = json.loads(manifest_path.read_text())
            if isinstance(parsed, dict):
                manifest_payload = parsed
            else:
                outcome_error = "run outcomes are unavailable because run_manifest.json is not a JSON object."
        except Exception:
            outcome_error = "run outcomes are unavailable because run_manifest.json could not be parsed."

    contract = extract_contract(payload, config_error=config_error)
    outcome = extract_outcome(
        manifest_payload,
        plan_order=[plan.name for plan in contract.plans],
        error_message=outcome_error,
    )
    return build_run_details_payload(
        contract,
        outcome,
        paths_context=RunDetailsPathsContext(
            run_root=context.run_root,
            config_path=context.cfg_path,
            records_path=context.records_path,
            manifest_path=manifest_path,
            notebook_path=context.notebook_path,
        ),
    )


def _template_header() -> str:
    return """import marimo

__generated_with = __GENERATED_WITH__

app = marimo.App(width=\"medium\")

"""


def _template_cells() -> str:
    return notebook_template_cells(
        baserender_export_cell_template=_baserender_export_cell_template,
        records_export_cell_template=_records_export_cell_template,
    )


def _template_footer() -> str:
    return """
if __name__ == "__main__":
    app.run()
"""


def _template_sections() -> tuple[str, ...]:
    return (
        _template_header(),
        _template_cells(),
        _template_footer(),
    )


def render_notebook_template(context: NotebookTemplateContext) -> str:
    run_root_text = json.dumps(str(context.run_root.resolve()))
    cfg_path_text = json.dumps(str(context.cfg_path.resolve()))
    records_path_text = json.dumps(str(context.records_path.resolve()))
    output_source_text = json.dumps(str(context.output_source))
    usr_root_text = json.dumps(str(context.usr_root.resolve()) if context.usr_root is not None else "")
    usr_dataset_text = json.dumps(str(context.usr_dataset or ""))
    workspace_raw_name = str(context.cfg_path.parent.name or context.run_root.name)
    workspace_heading_text = json.dumps(_format_workspace_heading(workspace_raw_name))
    workspace_intro_payload_text = json.dumps(_build_workspace_intro_payload(context))
    generated_with = "unknown"
    try:
        generated_with = importlib_metadata.version("marimo")
    except importlib_metadata.PackageNotFoundError:
        pass
    generated_with_text = json.dumps(generated_with)

    template = "".join(_template_sections())
    return (
        template.replace("__GENERATED_WITH__", generated_with_text)
        .replace("__RUN_ROOT__", run_root_text)
        .replace("__CFG_PATH__", cfg_path_text)
        .replace("__RECORDS_PATH__", records_path_text)
        .replace("__OUTPUT_SOURCE__", output_source_text)
        .replace("__USR_ROOT__", usr_root_text)
        .replace("__USR_DATASET__", usr_dataset_text)
        .replace("__WORKSPACE_HEADING__", workspace_heading_text)
        .replace("__WORKSPACE_RUN_DETAILS_PAYLOAD__", workspace_intro_payload_text)
    )
