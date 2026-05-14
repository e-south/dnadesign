"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/cli.py

CLI entrypoint for secondary-structure folding.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from .src import (
    FoldingConfigError,
    FoldingError,
    enrich_prediction_pairing_qa,
    load_prediction_request,
    preflight_request,
    publish_viennarna_structure_svg,
    run_prediction_request,
)

_BUNDLE_MANIFEST = "manifest.json"
_BUNDLE_PLOT_ARTIFACT = "viennarna_structure_plot"
_BUNDLE_PLOT_DIR = Path("visual") / "viennarna_secondary_structure"

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Run backend-neutral secondary-structure folding requests.",
)


@dataclass(frozen=True)
class _FoldingBundle:
    root: Path
    manifest_path: Path
    artifacts: dict[str, Any]


def _output_dir_for(request_path: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return request_path.parent
    if output_dir.is_absolute():
        return output_dir
    return (request_path.parent / output_dir).resolve()


def _plot_output_dir_for(prediction_path: Path, output_dir: Path) -> Path:
    if output_dir.is_absolute():
        return output_dir
    if output_dir.parts and output_dir.parts[0] == "..":
        return (prediction_path.parent / output_dir).resolve()
    return output_dir.expanduser().resolve()


def _load_bundle(bundle: Path) -> _FoldingBundle:
    root = bundle.expanduser().resolve()
    if not root.is_dir():
        raise FoldingConfigError(f"Folding bundle does not exist or is not a directory: {root}")
    manifest_path = root / _BUNDLE_MANIFEST
    if not manifest_path.is_file():
        raise FoldingConfigError(f"Folding bundle mode requires manifest.json: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FoldingConfigError(f"Folding bundle manifest is not valid JSON: {manifest_path}") from exc
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FoldingConfigError(f"Folding bundle manifest has no artifacts map: {manifest_path}")
    return _FoldingBundle(root=root, manifest_path=manifest_path, artifacts=artifacts)


def _bundle_artifact_path(
    bundle: _FoldingBundle,
    artifact_key: str,
    *,
    required: bool,
    must_exist: bool = True,
) -> Path | None:
    value = bundle.artifacts.get(artifact_key)
    if value is None:
        if required:
            raise FoldingConfigError(
                f"Folding bundle manifest missing required artifact '{artifact_key}': {bundle.manifest_path}"
            )
        return None
    if not isinstance(value, str) or not value.strip():
        raise FoldingConfigError(f"Folding bundle artifact '{artifact_key}' must be a relative path string.")
    artifact_path = Path(value)
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        raise FoldingConfigError(f"Folding bundle artifact '{artifact_key}' must stay inside the bundle: {value}")
    resolved = (bundle.root / artifact_path).resolve()
    try:
        resolved.relative_to(bundle.root)
    except ValueError as exc:
        raise FoldingConfigError(f"Folding bundle artifact '{artifact_key}' escapes the bundle: {value}") from exc
    if must_exist and not resolved.is_file():
        raise FoldingConfigError(f"Folding bundle artifact '{artifact_key}' does not exist: {resolved}")
    return resolved


def _bundle_plot_output_dir(bundle: _FoldingBundle) -> Path:
    plot_manifest = _bundle_artifact_path(bundle, _BUNDLE_PLOT_ARTIFACT, required=False, must_exist=False)
    if plot_manifest is not None:
        return plot_manifest.parent
    return (bundle.root / _BUNDLE_PLOT_DIR).resolve()


def _emit(payload: object, *, output_format: str) -> None:
    if output_format == "json":
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if isinstance(payload, dict):
        typer.echo(f"status: {payload.get('status')}")
        return
    typer.echo(str(payload))


def _format_option(output_format: str) -> str:
    format_norm = str(output_format or "").strip().lower()
    if format_norm not in {"text", "json"}:
        raise typer.BadParameter("Output format must be text or json.")
    return format_norm


@app.command("preflight")
def preflight_command(
    request: Path | None = typer.Option(
        None,
        "--request",
        exists=True,
        readable=True,
        help="Folding request YAML/JSON.",
    ),
    bundle: Path | None = typer.Option(
        None,
        "--bundle",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Construct output bundle containing manifest.json and folding artifacts.",
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Directory for folding artifacts."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = _format_option(output_format)
    try:
        if bundle is not None:
            if request is not None:
                raise FoldingConfigError("Use either --bundle or --request, not both.")
            folding_bundle = _load_bundle(bundle)
            request = _bundle_artifact_path(folding_bundle, "folding_request", required=True)
        if request is None:
            raise FoldingConfigError("--request is required unless --bundle is provided.")
        loaded, request_path = load_prediction_request(request)
        result = preflight_request(loaded, output_dir=_output_dir_for(request_path, output_dir))
    except FoldingError as exc:
        if format_norm == "json":
            _emit({"status": "error", "error": str(exc)}, output_format=format_norm)
        else:
            typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit(result.to_dict(), output_format=format_norm)


@app.command("run")
def run_command(
    request: Path | None = typer.Option(
        None,
        "--request",
        exists=True,
        readable=True,
        help="Folding request YAML/JSON.",
    ),
    bundle: Path | None = typer.Option(
        None,
        "--bundle",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Construct output bundle containing manifest.json and folding artifacts.",
    ),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Directory for folding artifacts."),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = _format_option(output_format)
    try:
        if bundle is not None:
            if request is not None:
                raise FoldingConfigError("Use either --bundle or --request, not both.")
            folding_bundle = _load_bundle(bundle)
            request = _bundle_artifact_path(folding_bundle, "folding_request", required=True)
        if request is None:
            raise FoldingConfigError("--request is required unless --bundle is provided.")
        loaded, request_path = load_prediction_request(request)
        prediction = run_prediction_request(
            loaded,
            output_dir=_output_dir_for(request_path, output_dir),
            request_path=request_path,
        )
    except FoldingError as exc:
        if format_norm == "json":
            _emit({"status": "error", "error": str(exc)}, output_format=format_norm)
        else:
            typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit(prediction.model_dump(mode="json"), output_format=format_norm)


@app.command("plot")
def plot_command(
    prediction: Path | None = typer.Option(
        None,
        "--prediction",
        exists=True,
        readable=True,
        help="Folding prediction JSON.",
    ),
    assembled_sequence: Path | None = typer.Option(
        None,
        "--assembled-sequence",
        exists=True,
        readable=True,
        help="Assembled sequence JSON artifact.",
    ),
    bundle: Path | None = typer.Option(
        None,
        "--bundle",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Construct output bundle containing manifest.json and folding/visual artifacts.",
    ),
    visual_contract: Path | None = typer.Option(
        None,
        "--visual-contract",
        exists=True,
        readable=True,
        help="Optional sequence_evidence_map_v1 JSON for dnadesign annotations.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help=(
            "Directory for ViennaRNA SVG plot artifacts. Plain relative paths resolve from the current working "
            "directory; ../ paths resolve relative to the prediction artifact directory."
        ),
    ),
    python_module: str = typer.Option("RNA", "--python-module", help="ViennaRNA Python module name."),
    layout: str = typer.Option(
        "naview", "--layout", help="ViennaRNA layout: simple, naview, circular, turtle, puzzler."
    ),
    emphasize_stem_bases: bool = typer.Option(
        True,
        "--emphasize-stem-bases/--no-emphasize-stem-bases",
        help="Bold and lightly stroke nucleotides tagged as left or right stem bases.",
    ),
    output_format: str = typer.Option("text", "--format", help="Output format: text or json."),
) -> None:
    format_norm = _format_option(output_format)
    try:
        if bundle is not None:
            if prediction is not None or assembled_sequence is not None:
                raise FoldingConfigError("Use either --bundle or explicit artifact paths, not both.")
            folding_bundle = _load_bundle(bundle)
            prediction = _bundle_artifact_path(folding_bundle, "folding_prediction", required=True)
            assembled_sequence = _bundle_artifact_path(folding_bundle, "folding_input_sequence", required=True)
            if visual_contract is None:
                visual_contract = _bundle_artifact_path(folding_bundle, "visual_contract", required=True)
            if output_dir is None:
                output_dir = _bundle_plot_output_dir(folding_bundle)
        if prediction is None:
            raise FoldingConfigError("--prediction is required unless --bundle is provided.")
        if assembled_sequence is None:
            raise FoldingConfigError("--assembled-sequence is required unless --bundle is provided.")
        if output_dir is None:
            raise FoldingConfigError("--output-dir is required unless --bundle is provided.")
        prediction_path = prediction.expanduser().resolve()
        if visual_contract is not None:
            enrich_prediction_pairing_qa(
                prediction_path,
                visual_contract_path=visual_contract,
                output_path=prediction_path,
            )
        plot = publish_viennarna_structure_svg(
            prediction_path,
            assembled_sequence_path=assembled_sequence,
            visual_contract_path=visual_contract,
            output_dir=_plot_output_dir_for(prediction_path, output_dir),
            python_module=python_module,
            layout_algorithm=layout,
            emphasize_stem_base_nucleotides=emphasize_stem_bases,
        )
    except FoldingError as exc:
        if format_norm == "json":
            _emit({"status": "error", "error": str(exc)}, output_format=format_norm)
        else:
            typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    _emit(plot.model_dump(mode="json"), output_format=format_norm)


def main() -> None:
    app()


__all__ = ["app", "main"]
