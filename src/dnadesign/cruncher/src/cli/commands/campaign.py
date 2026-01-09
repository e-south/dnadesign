"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/campaign.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml
from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, resolve_config_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.campaign_notebook_service import generate_campaign_notebook
from dnadesign.cruncher.services.campaign_service import build_campaign_manifest, expand_campaign
from dnadesign.cruncher.workflows.campaign_summary import summarize_campaign
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Generate or summarize category campaigns.")
console = Console()


@app.command("generate", help="Expand a campaign into explicit regulator_sets.")
def generate(
    config: Path | None = typer.Argument(None, help="Path to cruncher config.yaml.", metavar="CONFIG"),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    campaign: str = typer.Option(..., "--campaign", "-n", help="Campaign name to expand."),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output path for derived config (default: <config_stem>.<campaign>.yaml).",
    ),
    manifest: Path | None = typer.Option(
        None,
        "--manifest",
        help="Optional path for campaign manifest (default: alongside output config).",
    ),
    include_metrics: bool = typer.Option(
        True,
        "--metrics/--no-metrics",
        help="Include per-TF metrics in the manifest when selectors are used.",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        expansion = expand_campaign(
            cfg=cfg,
            config_path=config_path,
            campaign_name=campaign,
            include_metrics=include_metrics,
        )
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)

    out_path = out or config_path.with_name(f"{config_path.stem}.{campaign}.yaml")
    manifest_path = manifest or out_path.with_suffix(".campaign_manifest.json")
    generated_at = datetime.now(timezone.utc).isoformat()

    data = cfg.model_dump(mode="json")
    data["regulator_sets"] = expansion.regulator_sets
    data["campaign"] = {
        "name": expansion.name,
        "campaign_id": expansion.campaign_id,
        "manifest_path": str(manifest_path),
        "generated_at": generated_at,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        yaml.safe_dump({"cruncher": data}, fh, sort_keys=False, default_flow_style=False)

    payload = build_campaign_manifest(expansion=expansion, config_path=config_path)
    payload["created_at"] = generated_at
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))

    console.print(str(out_path))
    console.print(str(manifest_path))


@app.command("summarize", help="Aggregate campaign runs into summary tables and plots.")
def summarize(
    config: Path | None = typer.Argument(None, help="Path to cruncher config.yaml.", metavar="CONFIG"),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    campaign: str = typer.Option(..., "--campaign", "-n", help="Campaign name to summarize."),
    runs: list[str] | None = typer.Option(
        None,
        "--runs",
        help="Run directories or glob patterns (repeatable). Defaults to all sample runs.",
    ),
    analysis_id: str | None = typer.Option(
        None,
        "--analysis-id",
        help="Specific analysis id to summarize (defaults to latest per run).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Output directory for campaign summary artifacts.",
    ),
    include_metrics: bool = typer.Option(
        True,
        "--metrics/--no-metrics",
        help="Include per-TF quality metrics (requires local catalog).",
    ),
    skip_missing: bool = typer.Option(
        False,
        "--skip-missing",
        help="Skip runs with missing analysis tables instead of failing.",
    ),
    skip_non_campaign: bool = typer.Option(
        False,
        "--skip-non-campaign",
        help="Skip runs whose regulator_set does not match the campaign.",
    ),
    top_k: int = typer.Option(10, "--top-k", help="Number of top runs to include in campaign_best.csv."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    try:
        result = summarize_campaign(
            cfg=cfg,
            config_path=config_path,
            campaign_name=campaign,
            run_inputs=runs or None,
            analysis_id=analysis_id,
            out_dir=out,
            include_metrics=include_metrics,
            skip_missing=skip_missing,
            skip_non_campaign=skip_non_campaign,
            top_k=top_k,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)

    console.print(str(result.summary_path))
    console.print(str(result.best_path))
    for path in result.plot_paths:
        console.print(str(path))
    if result.skipped:
        console.print("Skipped runs:")
        for item in result.skipped:
            console.print(f"- {item}")


@app.command("notebook", help="Generate a marimo notebook for a campaign summary.")
def notebook(
    config: Path | None = typer.Argument(None, help="Path to cruncher config.yaml.", metavar="CONFIG"),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
    campaign: str = typer.Option(..., "--campaign", "-n", help="Campaign name to open."),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Campaign summary directory (defaults to runs/campaigns/<campaign_id>).",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite the notebook if it already exists."),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Require campaign summary artifacts to exist before generating.",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    if out is None:
        try:
            expansion = expand_campaign(
                cfg=cfg,
                config_path=config_path,
                campaign_name=campaign,
                include_metrics=False,
            )
        except ValueError as exc:
            console.print(f"Error: {exc}")
            raise typer.Exit(code=1)
        runs_root = config_path.parent / cfg.out_dir
        out_dir = runs_root / "campaigns" / expansion.campaign_id
    else:
        out_dir = out
    try:
        notebook_path = generate_campaign_notebook(summary_dir=out_dir, force=force, strict=strict)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    console.print(str(notebook_path))
