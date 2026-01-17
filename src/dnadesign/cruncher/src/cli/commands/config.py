"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/config.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import click
import typer
from dnadesign.cruncher.app.config_service import summarize_config
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="Summarize effective config and sampling settings.",
)
console = Console()


@app.callback(invoke_without_command=True)
def config_main(
    ctx: typer.Context,
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    if ctx.invoked_subcommand is not None:
        if config_option is None and config is None:
            return
        try:
            ctx.obj = {"config_path": resolve_config_path(config_option or config)}
        except ConfigResolutionError as exc:
            console.print(str(exc))
            raise typer.Exit(code=1)
        return
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    summary(config_path, None)


@app.command("summary", help="Show resolved config and key sampling settings.")
def summary(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    ctx = click.get_current_context(silent=True)
    try:
        if config_option is not None or config is not None:
            config_path = resolve_config_path(config_option or config)
        elif ctx.obj and isinstance(ctx.obj, dict) and ctx.obj.get("config_path"):
            config_path = ctx.obj["config_path"]
        else:
            config_path = resolve_config_path(None)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    summary = summarize_config(cfg)

    table = Table(title="Cruncher config summary", header_style="bold")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("out_dir", summary["out_dir"])
    table.add_row("regulator_sets", str(summary["regulator_sets"]))
    table.add_row("regulators_flat", ", ".join(summary["regulators_flat"]))
    table.add_row("regulator_categories", str(summary.get("regulator_categories") or {}))
    campaign_names = summary.get("campaign_names") or []
    table.add_row("campaigns", ", ".join(campaign_names) if campaign_names else "-")
    campaign_meta = summary.get("campaign")
    if campaign_meta:
        table.add_row("campaign.name", campaign_meta.get("name", "-"))
        table.add_row("campaign.id", campaign_meta.get("campaign_id", "-"))
        table.add_row("campaign.manifest_path", campaign_meta.get("manifest_path", "-"))
    table.add_row("io.parsers.extra_modules", str(summary["io"]["parsers"]["extra_modules"]))
    table.add_row("pwm_source", summary["motif_store"]["pwm_source"])
    table.add_row("site_kinds", str(summary["motif_store"]["site_kinds"]))
    table.add_row("combine_sites", str(summary["motif_store"]["combine_sites"]))
    table.add_row("dataset_preference", str(summary["motif_store"]["dataset_preference"]))
    table.add_row("dataset_map", str(summary["motif_store"]["dataset_map"]))
    table.add_row("site_window_lengths", str(summary["motif_store"]["site_window_lengths"]))
    table.add_row("site_window_center", summary["motif_store"]["site_window_center"])
    table.add_row("min_sites_for_pwm", str(summary["motif_store"]["min_sites_for_pwm"]))
    table.add_row("source_preference", str(summary["motif_store"]["source_preference"]))
    table.add_row("allow_ambiguous", str(summary["motif_store"]["allow_ambiguous"]))
    table.add_row("ingest.genome_source", summary["ingest"]["genome_source"])
    table.add_row("ingest.genome_fasta", summary["ingest"]["genome_fasta"] or "-")
    table.add_row("ingest.genome_cache", summary["ingest"]["genome_cache"])
    table.add_row("ingest.genome_assembly", summary["ingest"]["genome_assembly"] or "-")
    table.add_row("ingest.contig_aliases", str(summary["ingest"]["contig_aliases"] or "{}"))
    table.add_row("ingest.ncbi_email", summary["ingest"]["ncbi_email"] or "-")
    table.add_row("ingest.ncbi_tool", summary["ingest"]["ncbi_tool"])
    table.add_row("ingest.ncbi_timeout", str(summary["ingest"]["ncbi_timeout_seconds"]))
    table.add_row("ingest.http.retries", str(summary["ingest"]["http"]["retries"]))
    table.add_row("ingest.http.backoff_seconds", str(summary["ingest"]["http"]["backoff_seconds"]))
    table.add_row(
        "ingest.http.max_backoff_seconds",
        str(summary["ingest"]["http"]["max_backoff_seconds"]),
    )
    local_sources = summary["ingest"].get("local_sources") or []
    if local_sources:
        rendered = ", ".join(f"{src['source_id']}@{src['root']}" for src in local_sources)
    else:
        rendered = "-"
    table.add_row("ingest.local_sources", rendered)
    table.add_row(
        "ingest.regulondb.curated_sites",
        str(summary["ingest"]["regulondb"]["curated_sites"]),
    )
    table.add_row("ingest.regulondb.ht_sites", str(summary["ingest"]["regulondb"]["ht_sites"]))
    table.add_row(
        "ingest.regulondb.ht_dataset_type",
        str(summary["ingest"]["regulondb"].get("ht_dataset_type", "-")),
    )
    table.add_row(
        "ingest.regulondb.ht_binding_mode",
        summary["ingest"]["regulondb"]["ht_binding_mode"],
    )

    sample = summary.get("sample")
    if sample is None:
        table.add_row("sample", "None")
    else:
        table.add_row("mode", sample["mode"])
        table.add_row("rng.seed", str(sample["rng"]["seed"]))
        table.add_row("rng.deterministic", str(sample["rng"]["deterministic"]))
        table.add_row("budget.tune", str(sample["budget"]["tune"]))
        table.add_row("budget.draws", str(sample["budget"]["draws"]))
        table.add_row("budget.restarts", str(sample["budget"]["restarts"]))
        table.add_row("init.kind", sample["init"]["kind"])
        table.add_row("init.length", str(sample["init"]["length"]))
        table.add_row("init.regulator", str(sample["init"].get("regulator")))
        table.add_row("objective.score_scale", sample["objective"]["score_scale"])
        table.add_row("objective.bidirectional", str(sample["objective"]["bidirectional"]))
        table.add_row("elites.k", str(sample["elites"]["k"]))
        table.add_row("elites.min_hamming", str(sample["elites"]["min_hamming"]))
        table.add_row("elites.filters.pwm_sum_min", str(sample["elites"]["filters"]["pwm_sum_min"]))
        table.add_row("optimizer.name", sample["optimizer"]["name"])
        table.add_row("optimizers.gibbs.beta_schedule", str(sample["optimizers"]["gibbs"]["beta_schedule"]))
        table.add_row("optimizers.pt.beta_ladder", str(sample["optimizers"]["pt"]["beta_ladder"]))
        table.add_row("optimizers.pt.swap_prob", str(sample["optimizers"]["pt"]["swap_prob"]))
        table.add_row("output.trace.save", str(sample["output"]["trace"]["save"]))
        table.add_row("output.save_sequences", str(sample["output"]["save_sequences"]))
        table.add_row("ui.progress_bar", str(sample["ui"]["progress_bar"]))
        table.add_row("ui.progress_every", str(sample["ui"]["progress_every"]))

    analysis = summary.get("analysis")
    if analysis is None:
        table.add_row("analysis", "None")
    else:
        table.add_row("analysis.runs", str(analysis["runs"]))
        table.add_row("analysis.plots", str(analysis["plots"]))
        table.add_row("analysis.scatter_scale", str(analysis["scatter_scale"]))
        table.add_row("analysis.subsampling_epsilon", str(analysis["subsampling_epsilon"]))
        table.add_row("analysis.scatter_style", str(analysis["scatter_style"]))
        table.add_row("analysis.scatter_background", str(analysis["scatter_background"]))
        table.add_row("analysis.scatter_background_samples", str(analysis["scatter_background_samples"]))
        table.add_row("analysis.scatter_background_seed", str(analysis["scatter_background_seed"]))
        table.add_row("analysis.tf_pair", str(analysis["tf_pair"]))
        table.add_row("analysis.archive", str(analysis.get("archive")))

    console.print(table)
