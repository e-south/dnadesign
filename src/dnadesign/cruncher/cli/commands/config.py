"""Config inspection command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.config_service import summarize_config

app = typer.Typer(no_args_is_help=True, help="Summarize effective config and sampling settings.")
console = Console()


@app.command("summary", help="Show resolved config and key sampling settings.")
def summary(config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG")) -> None:
    cfg = load_config(config)
    summary = summarize_config(cfg)

    table = Table(title="Cruncher config summary", header_style="bold")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("out_dir", summary["out_dir"])
    table.add_row("regulator_sets", str(summary["regulator_sets"]))
    table.add_row("regulators_flat", ", ".join(summary["regulators_flat"]))
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
    table.add_row("ingest.http.max_backoff_seconds", str(summary["ingest"]["http"]["max_backoff_seconds"]))
    table.add_row("ingest.regulondb.curated_sites", str(summary["ingest"]["regulondb"]["curated_sites"]))
    table.add_row("ingest.regulondb.ht_sites", str(summary["ingest"]["regulondb"]["ht_sites"]))
    table.add_row("ingest.regulondb.ht_dataset_type", str(summary["ingest"]["regulondb"].get("ht_dataset_type", "-")))
    table.add_row("ingest.regulondb.ht_binding_mode", summary["ingest"]["regulondb"]["ht_binding_mode"])

    sample = summary.get("sample")
    if sample is None:
        table.add_row("sample", "None")
    else:
        table.add_row("init.kind", sample["init"]["kind"])
        table.add_row("init.length", str(sample["init"]["length"]))
        table.add_row("init.regulator", str(sample["init"]["regulator"]))
        table.add_row("draws", str(sample["draws"]))
        table.add_row("tune", str(sample["tune"]))
        table.add_row("chains", str(sample["chains"]))
        table.add_row("top_k", str(sample["top_k"]))
        table.add_row("min_dist", str(sample["min_dist"]))
        table.add_row("seed", str(sample["seed"]))
        table.add_row("record_tune", str(sample["record_tune"]))
        table.add_row("progress_bar", str(sample["progress_bar"]))
        table.add_row("progress_every", str(sample["progress_every"]))
        table.add_row("save_trace", str(sample["save_trace"]))
        table.add_row("optimizer.kind", sample["optimiser"]["kind"])
        table.add_row("scorer_scale", sample["optimiser"]["scorer_scale"])
        table.add_row("cooling", str(sample["optimiser"]["cooling"]))
        table.add_row("swap_prob", str(sample["optimiser"]["swap_prob"]))

    analysis = summary.get("analysis")
    if analysis is None:
        table.add_row("analysis", "None")
    else:
        table.add_row("analysis.runs", str(analysis["runs"]))
        table.add_row("analysis.plots", str(analysis["plots"]))
        table.add_row("analysis.scatter_scale", str(analysis["scatter_scale"]))
        table.add_row("analysis.subsampling_epsilon", str(analysis["subsampling_epsilon"]))

    console.print(table)
