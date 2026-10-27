"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/generate.py

Registration and implementation of the infer `generate` CLI command.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from ...api import run_job
from ..builders import run_with_progress
from ..common import raise_cli_error
from ..console import console, render_config_summary, render_outputs_spec_table
from ..ingest import build_generate_ingest
from ..requests import build_generate_request


def register(app: typer.Typer) -> None:
    @app.command(help="Ad-hoc generation (use params or --preset).")
    def generate(
        model_id: Optional[str] = typer.Option(None, "--model-id"),
        device: Optional[str] = typer.Option(None, "--device"),
        precision: Optional[str] = typer.Option(None, "--precision"),
        alphabet: Optional[str] = typer.Option(None, "--alphabet"),
        batch_size: Optional[int] = typer.Option(None, "--batch-size"),
        prompt: List[str] = typer.Option(None, "--prompt", help="One or more prompts.", show_default=False),
        prompt_file: Optional[Path] = typer.Option(None, "--prompt-file", help="Text file (one prompt per line)."),
        usr: Optional[str] = typer.Option(None, "--usr", help="USR dataset as prompts."),
        field: str = typer.Option("sequence", "--field"),
        ids: Optional[str] = typer.Option(None, "--ids"),
        usr_root: Optional[Path] = typer.Option(None, "--usr-root"),
        preset: Optional[str] = typer.Option(None, "--preset", help="Use a named preset (generate)."),
        max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens"),
        temperature: Optional[float] = typer.Option(None, "--temperature"),
        top_k: Optional[int] = typer.Option(None, "--top-k"),
        top_p: Optional[float] = typer.Option(None, "--top-p"),
        seed: Optional[int] = typer.Option(None, "--seed"),
        out: Optional[Path] = typer.Option(None, "--out", help="Write generated sequences (.json or .txt)."),
        progress: bool = typer.Option(True, "--progress/--no-progress"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print summary and exit."),
    ):
        try:
            request = build_generate_request(
                model_id=model_id,
                device=device,
                precision=precision,
                alphabet=alphabet,
                batch_size=batch_size,
                preset=preset,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                seed=seed,
            )
            model = request.model
            job = request.job

            ingest_request = build_generate_ingest(
                prompt=prompt,
                prompt_file=prompt_file,
                usr=usr,
                field=field,
                ids=ids,
                usr_root=usr_root,
            )
            job.ingest = ingest_request.ingest
            inputs = ingest_request.inputs

            if dry_run:
                render_config_summary(model, [job])
                render_outputs_spec_table([request.output_row])
                console.print("[green]✔ Generate validated (dry run).[/green]")
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

            generated = result.get("gen_seqs", [])
            console.print(f"[green]Generated {len(generated)} sequence(s).[/green]")
            if out:
                if out.suffix.lower() == ".json":
                    out.write_text(json.dumps(generated, indent=2))
                else:
                    out.write_text("\n".join(generated) + "\n")
                console.print(f"[accent]Wrote:[/accent] {out}")

        except typer.Exit:
            raise
        except Exception as error:
            raise_cli_error(error)
