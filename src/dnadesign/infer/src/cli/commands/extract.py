"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/extract.py

Registration and implementation of the infer `extract` CLI command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from ...api import run_job
from ..builders import run_with_progress
from ..common import guard_pickle, raise_cli_error
from ..console import console, render_config_summary, render_outputs_spec_table, render_outputs_summary
from ..ingest import build_extract_ingest
from ..requests import build_extract_request


def register(app: typer.Typer) -> None:
    @app.command(help="Ad-hoc extract (single or multi-output via --preset).")
    def extract(
        model_id: Optional[str] = typer.Option(None, "--model-id"),
        device: Optional[str] = typer.Option(None, "--device"),
        precision: Optional[str] = typer.Option(None, "--precision"),
        alphabet: Optional[str] = typer.Option(None, "--alphabet"),
        batch_size: Optional[int] = typer.Option(None, "--batch-size"),
        preset: Optional[str] = typer.Option(None, "--preset", help="Use a named preset (extract)."),
        fn: Optional[str] = typer.Option(None, "--fn", help="namespaced fn, e.g., evo2.log_likelihood"),
        format: Optional[str] = typer.Option(None, "--format", help="float|list|numpy|tensor"),
        out_id: str = typer.Option("out", "--out-id", help="Column id for output (ignored if --preset)."),
        seq: List[str] = typer.Option(None, "--seq", help="One or more sequences.", show_default=False),
        seq_file: Optional[Path] = typer.Option(None, "--seq-file", help="Text file (one per line)."),
        usr: Optional[str] = typer.Option(None, "--usr", help="USR dataset name."),
        field: str = typer.Option("sequence", "--field"),
        ids: Optional[str] = typer.Option(None, "--ids", help="Path/CSV of ids (USR)."),
        usr_root: Optional[Path] = typer.Option(None, "--usr-root"),
        pt: Optional[Path] = typer.Option(None, "--pt", help=".pt file path (pickle)."),
        records_jsonl: Optional[Path] = typer.Option(None, "--records-jsonl", help="JSONL records path."),
        pool_method: Optional[str] = typer.Option(None, "--pool-method", help="mean|sum|max (for logits/embedding)"),
        pool_dim: Optional[int] = typer.Option(None, "--pool-dim"),
        layer: Optional[str] = typer.Option(None, "--layer", help="Layer override (embedding)."),
        write_back: bool = typer.Option(False, "--write-back/--no-write-back"),
        overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
        progress: bool = typer.Option(True, "--progress/--no-progress"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print summary and exit."),
        i_know_this_is_pickle: bool = typer.Option(False, "--i-know-this-is-pickle"),
    ):
        try:
            request = build_extract_request(
                model_id=model_id,
                device=device,
                precision=precision,
                alphabet=alphabet,
                batch_size=batch_size,
                preset=preset,
                fn=fn,
                format=format,
                out_id=out_id,
                pool_method=pool_method,
                pool_dim=pool_dim,
                layer=layer,
                write_back=write_back,
                overwrite=overwrite,
            )
            model = request.model
            job = request.job

            ingest_request = build_extract_ingest(
                seq=seq,
                seq_file=seq_file,
                usr=usr,
                field=field,
                ids=ids,
                usr_root=usr_root,
                pt=pt,
                records_jsonl=records_jsonl,
                i_know_this_is_pickle=i_know_this_is_pickle,
                guard_pickle=guard_pickle,
            )
            job.ingest = ingest_request.ingest
            inputs = ingest_request.inputs

            if dry_run:
                render_config_summary(model, [job])
                render_outputs_spec_table(request.output_rows)
                console.print("[green]✔ Extract validated (dry run).[/green]")
                raise typer.Exit(code=0)

            result = run_with_progress(
                progress=progress,
                runner=lambda progress_factory: run_job(
                    inputs=inputs,
                    model=model,
                    job=job,
                    progress_factory=progress_factory,
                ),
            )
            render_outputs_summary(job, result)

        except typer.Exit:
            raise
        except Exception as error:
            raise_cli_error(error)
