"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/cli.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml

from ._console import (
    RichProgressManager,
    console,
    render_adapters_table,
    render_config_summary,
    render_functions_table,
    render_outputs_spec_table,
    render_outputs_summary,
    render_preset_detail,
    render_presets_table,
    rich_tracebacks,
    setup_console_logging,
)
from .api import run_job
from .config import IngestConfig, JobConfig, ModelConfig, OutputSpec, RootConfig
from .engine import clear_adapter_cache
from .errors import (
    CapabilityError,
    ConfigError,
    IOErrorInfer,
    ModelLoadError,
    RuntimeOOMError,
    UnsafeInputError,
    ValidationError,
    WriteBackError,
)
from .presets import list_presets, load_preset
from .registry import list_fns, list_models

app = typer.Typer(
    add_completion=True,
    no_args_is_help=True,
    help="Model-agnostic sequence inference CLI.",
)


def _exit_for(e: Exception) -> int:
    mapping = {
        ConfigError: 2,
        ValidationError: 3,
        ModelLoadError: 4,
        CapabilityError: 5,
        RuntimeOOMError: 6,
        WriteBackError: 7,
        IOErrorInfer: 8,
        UnsafeInputError: 8,
    }
    for etype, code in mapping.items():
        if isinstance(e, etype):
            return code
    return 1


def _discovery_config(provided: Optional[Path]) -> Path:
    if provided:
        return provided.resolve()
    cwd_cfg = Path.cwd() / "config.yaml"
    if cwd_cfg.exists():
        return cwd_cfg.resolve()
    module_cfg = Path(__file__).with_name("config.yaml")
    if module_cfg.exists():
        return module_cfg.resolve()
    raise ConfigError("No config found. Pass --config or place config.yaml in the current directory.")


def _read_ids_arg(ids: Optional[str]) -> Optional[List[str]]:
    if not ids:
        return None
    p = Path(ids)
    if p.exists():
        text = p.read_text().strip()
        if "\n" in text:
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
        return [x.strip() for x in text.split(",") if x.strip()]
    return [x.strip() for x in ids.split(",") if x.strip()]


def _guard_pickle(i_know: bool) -> None:
    allow = os.environ.get("INFER_ALLOW_PICKLE", "0").lower() in {"1", "true", "yes"}
    if not i_know and not allow:
        raise UnsafeInputError(
            "Refusing to load a .pt file without explicit consent. "
            "Re-run with --i-know-this-is-pickle or set INFER_ALLOW_PICKLE=1."
        )


@app.callback()
def _root(
    log_level: str = typer.Option(
        os.environ.get("INFER_LOG_LEVEL", "INFO"),
        "--log-level",
        help="Console log level.",
    ),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON logs."),
    trace: bool = typer.Option(False, "--trace", help="Rich tracebacks on errors."),
):
    setup_console_logging(log_level, json_logs)
    rich_tracebacks(enabled=trace)


# ───────────────────────────────────────────────────────────────────────────────
# RUN (config or preset)
# ───────────────────────────────────────────────────────────────────────────────


@app.command(help="Run jobs from a config OR a single job from a preset.")
def run(
    # config or preset
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Run a single job from a named preset."),
    job: List[str] = typer.Option([], "--job", help="If --config is used: one or more job ids to run."),
    # common overrides
    device: Optional[str] = typer.Option(None, "--device", help="Override model.device."),
    precision: Optional[str] = typer.Option(None, "--precision", help="Override model.precision."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override model.batch_size."),
    overwrite: Optional[bool] = typer.Option(None, "--overwrite/--no-overwrite", help="Override job.io.overwrite."),
    progress: bool = typer.Option(True, "--progress/--no-progress", help="Use progress bars."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate and print summary, then exit."),
    # preset ingest (USR-centric for run)
    usr: Optional[str] = typer.Option(None, "--usr", help="USR dataset (preset mode)."),
    field: str = typer.Option("sequence", "--field", help="USR field/column."),
    ids: Optional[str] = typer.Option(None, "--ids", help="CSV or file of ids (USR)."),
    usr_root: Optional[Path] = typer.Option(None, "--usr-root", help="USR datasets root."),
    write_back: bool = typer.Option(
        False,
        "--write-back/--no-write-back",
        help="Preset mode: attach outputs to USR.",
    ),
    # safety
    i_know_this_is_pickle: bool = typer.Option(False, "--i-know-this-is-pickle", help="Needed only for pt_file."),
):
    try:
        if preset:
            # Build a single job from a preset (extract or generate)
            p = load_preset(preset)
            # model defaults (CLI overrides last)
            model_d = {
                "id": p.get("model", {}).get("id") or "evo2_7b",
                "device": device or "cpu",
                "precision": precision or p.get("model", {}).get("precision", "fp32"),
                "alphabet": p.get("model", {}).get("alphabet", "dna"),
                "batch_size": batch_size,
            }
            model = ModelConfig(**model_d)

            if p["kind"] == "extract":
                outputs = [OutputSpec(**o) for o in p.get("outputs", [])]
                job_cfg = JobConfig(
                    id=f"preset__{p['id'].split('/')[-1]}",
                    operation="extract",
                    ingest={
                        "source": "usr",
                        "dataset": usr,
                        "root": usr_root.as_posix() if usr_root else None,
                        "field": field,
                        "ids": _read_ids_arg(ids),
                    },
                    outputs=outputs,
                    io={"write_back": write_back, "overwrite": bool(overwrite)},
                )
            else:
                job_cfg = JobConfig(
                    id=f"preset__{p['id'].split('/')[-1]}",
                    operation="generate",
                    ingest={
                        "source": "usr",
                        "dataset": usr,
                        "root": usr_root.as_posix() if usr_root else None,
                        "field": field,
                        "ids": _read_ids_arg(ids),
                    },
                    params=p.get("params") or {},
                    io={"write_back": False, "overwrite": False},
                )

            if dry_run:
                render_config_summary(model, [job_cfg])
                if p["kind"] == "extract":
                    render_outputs_spec_table(p.get("outputs", []))
                console.print("[green]✔ Preset validated (dry run).[/green]")
                raise typer.Exit(code=0)

            if not progress:
                os.environ["DNADESIGN_PROGRESS"] = "0"
            pm = RichProgressManager(enabled=progress)
            with pm:
                res = run_job(
                    inputs=None,
                    model=model,
                    job=job_cfg,
                    progress_factory=pm.factory if progress else None,
                )
            if p["kind"] == "extract":
                render_outputs_summary(job_cfg, res)
            else:
                console.print(f"[green]Generated {len(res.get('gen_seqs', []))} sequence(s).[/green]")
            raise typer.Exit(code=0)

        # --config path
        cfg_path = _discovery_config(config)
        root = RootConfig(**yaml.safe_load(cfg_path.read_text()))
        jobs = root.jobs if not job else [j for j in root.jobs if j.id in set(job)]
        if not jobs:
            raise ConfigError("No jobs selected. Check --job or the config file.")
        model = root.model
        if device:
            model.device = device
        if precision:
            model.precision = precision  # type: ignore[assignment]
        if batch_size is not None:
            model.batch_size = batch_size
        if overwrite is not None:
            for j in jobs:
                j.io.overwrite = overwrite

        if dry_run:
            render_config_summary(model, jobs)
            console.print("[green]✔ Config validated (dry run).[/green]")
            raise typer.Exit(code=0)

        if not progress:
            os.environ["DNADESIGN_PROGRESS"] = "0"
        pm = RichProgressManager(enabled=progress)

        with pm:
            for j in jobs:
                inputs = None
                if j.ingest.source == "pt_file":
                    _guard_pickle(i_know_this_is_pickle)
                    inputs = (cfg_path.parent / f"{j.id}.pt").as_posix()
                res = run_job(
                    inputs=inputs,
                    model=model,
                    job=j,
                    progress_factory=pm.factory if progress else None,
                )
                if j.operation == "extract":
                    render_outputs_summary(j, res)
                else:
                    console.print(f"[green]Generated {len(res.get('gen_seqs', []))} sequence(s).[/green]")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=_exit_for(e))


# ───────────────────────────────────────────────────────────────────────────────
# EXTRACT (ad-hoc; now supports --preset and --dry-run)
# ───────────────────────────────────────────────────────────────────────────────


@app.command(help="Ad-hoc extract (single or multi-output via --preset).")
def extract(
    # model (optional; can come from preset)
    model_id: Optional[str] = typer.Option(None, "--model-id"),
    device: Optional[str] = typer.Option(None, "--device"),
    precision: Optional[str] = typer.Option(None, "--precision"),
    alphabet: Optional[str] = typer.Option(None, "--alphabet"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    # output or preset
    preset: Optional[str] = typer.Option(None, "--preset", help="Use a named preset (extract)."),
    fn: Optional[str] = typer.Option(None, "--fn", help="namespaced fn, e.g., evo2.log_likelihood"),
    format: Optional[str] = typer.Option(None, "--format", help="float|list|numpy|tensor"),
    out_id: str = typer.Option("out", "--out-id", help="Column id for output (ignored if --preset)."),
    # source
    seq: List[str] = typer.Option(None, "--seq", help="One or more sequences.", show_default=False),
    seq_file: Optional[Path] = typer.Option(None, "--seq-file", help="Text file (one per line)."),
    usr: Optional[str] = typer.Option(None, "--usr", help="USR dataset name."),
    field: str = typer.Option("sequence", "--field"),
    ids: Optional[str] = typer.Option(None, "--ids", help="Path/CSV of ids (USR)."),
    usr_root: Optional[Path] = typer.Option(None, "--usr-root"),
    pt: Optional[Path] = typer.Option(None, "--pt", help=".pt file path (pickle)."),
    records_jsonl: Optional[Path] = typer.Option(None, "--records-jsonl", help="JSONL records path."),
    # params (single-output path)
    pool_method: Optional[str] = typer.Option(None, "--pool-method", help="mean|sum|max (for logits/embedding)"),
    pool_dim: Optional[int] = typer.Option(None, "--pool-dim"),
    layer: Optional[str] = typer.Option(None, "--layer", help="Layer (embedding)"),
    # IO
    write_back: bool = typer.Option(False, "--write-back/--no-write-back"),
    overwrite: bool = typer.Option(False, "--overwrite/--no-overwrite"),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print summary and exit."),
    i_know_this_is_pickle: bool = typer.Option(False, "--i-know-this-is-pickle"),
):
    try:
        outputs: List[OutputSpec] = []
        if preset:
            p = load_preset(preset)
            if p["kind"] != "extract":
                raise ConfigError(f"Preset '{preset}' is not an extract preset.")
            # model defaults come from preset if not specified
            model_d = {
                "id": model_id or p.get("model", {}).get("id") or "evo2_7b",
                "device": device or "cpu",
                "precision": precision or p.get("model", {}).get("precision", "fp32"),
                "alphabet": alphabet or p.get("model", {}).get("alphabet", "dna"),
                "batch_size": batch_size,
            }
            outputs = [OutputSpec(**o) for o in p.get("outputs", [])]
        else:
            # single-output path must have fn+format
            if not (fn and format):
                raise ConfigError("Provide --fn and --format, or use --preset.")
            params: Dict[str, Any] = {}
            if fn.split(".")[-1] in {"logits", "embedding"}:
                if pool_method or pool_dim is not None:
                    params["pool"] = {}
                    if pool_method:
                        params["pool"]["method"] = pool_method
                    if pool_dim is not None:
                        params["pool"]["dim"] = pool_dim
                if layer:
                    params["layer"] = layer
            elif fn.split(".")[-1] == "log_likelihood":
                params.setdefault("method", "native")
                params.setdefault("reduction", "sum")
            outputs = [OutputSpec(id=out_id, fn=fn, params=params, format=format)]
            model_d = {
                "id": model_id or "evo2_7b",
                "device": device or "cpu",
                "precision": precision or "fp32",
                "alphabet": alphabet or "dna",
                "batch_size": batch_size,
            }

        model = ModelConfig(**model_d)

        # Ingest
        job = JobConfig(
            id="adhoc_extract",
            operation="extract",
            ingest={"source": "sequences"},
            outputs=outputs,
            io={"write_back": write_back, "overwrite": overwrite},
        )

        inputs = None

        def _load_lines(path: Path) -> List[str]:
            return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]

        if usr:
            job.ingest = IngestConfig(
                source="usr",
                dataset=usr,
                field=field,
                root=(usr_root.as_posix() if usr_root else None),
                ids=_read_ids_arg(ids),
            )
        elif pt:
            _guard_pickle(i_know_this_is_pickle)
            job.ingest = IngestConfig(source="pt_file", field=field)
            inputs = pt.as_posix()
        elif records_jsonl:
            job.ingest = IngestConfig(source="records", field=field)
            records = [json.loads(ln) for ln in _load_lines(records_jsonl)]
            inputs = records  # type: ignore[assignment]
        elif seq_file:
            job.ingest = IngestConfig(source="sequences")
            inputs = _load_lines(seq_file)
        elif seq:
            job.ingest = IngestConfig(source="sequences")
            inputs = seq
        else:
            raise ConfigError("Provide one of --seq/--seq-file/--usr/--pt/--records-jsonl")

        if dry_run:
            render_config_summary(model, [job])
            render_outputs_spec_table([o.model_dump() for o in outputs])
            console.print("[green]✔ Extract validated (dry run).[/green]")
            raise typer.Exit(code=0)

        if not progress:
            os.environ["DNADESIGN_PROGRESS"] = "0"
        pm = RichProgressManager(enabled=progress)
        with pm:
            res = run_job(
                inputs=inputs,
                model=model,
                job=job,
                progress_factory=pm.factory if progress else None,
            )
        render_outputs_summary(job, res)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=_exit_for(e))


# ───────────────────────────────────────────────────────────────────────────────
# GENERATE (ad-hoc; now supports --preset and --dry-run)
# ───────────────────────────────────────────────────────────────────────────────


@app.command(help="Ad-hoc generation (use params or --preset).")
def generate(
    model_id: Optional[str] = typer.Option(None, "--model-id"),
    device: Optional[str] = typer.Option(None, "--device"),
    precision: Optional[str] = typer.Option(None, "--precision"),
    alphabet: Optional[str] = typer.Option(None, "--alphabet"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    # prompts
    prompt: List[str] = typer.Option(None, "--prompt", help="One or more prompts.", show_default=False),
    prompt_file: Optional[Path] = typer.Option(None, "--prompt-file", help="Text file (one prompt per line)."),
    usr: Optional[str] = typer.Option(None, "--usr", help="USR dataset as prompts."),
    field: str = typer.Option("sequence", "--field"),
    ids: Optional[str] = typer.Option(None, "--ids"),
    usr_root: Optional[Path] = typer.Option(None, "--usr-root"),
    # params or preset
    preset: Optional[str] = typer.Option(None, "--preset", help="Use a named preset (generate)."),
    max_new_tokens: Optional[int] = typer.Option(None, "--max-new-tokens"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    top_p: Optional[float] = typer.Option(None, "--top-p"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    # output
    out: Optional[Path] = typer.Option(None, "--out", help="Write generated sequences (.json or .txt)."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print summary and exit."),
):
    try:
        params: Dict[str, Any] = {}

        if preset:
            p = load_preset(preset)
            if p["kind"] != "generate":
                raise ConfigError(f"Preset '{preset}' is not a generate preset.")
            params.update(p.get("params") or {})
            model_d = {
                "id": model_id or p.get("model", {}).get("id") or "evo2_7b",
                "device": device or "cpu",
                "precision": precision or p.get("model", {}).get("precision", "fp32"),
                "alphabet": alphabet or p.get("model", {}).get("alphabet", "dna"),
                "batch_size": batch_size,
            }
        else:
            if max_new_tokens is None:
                max_new_tokens = 64
            if temperature is None:
                temperature = 1.0
            params.update(
                {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                }
            )
            if top_k is not None:
                params["top_k"] = top_k
            if top_p is not None:
                params["top_p"] = top_p
            if seed is not None:
                params["seed"] = seed
            model_d = {
                "id": model_id or "evo2_7b",
                "device": device or "cpu",
                "precision": precision or "fp32",
                "alphabet": alphabet or "dna",
                "batch_size": batch_size,
            }

        model = ModelConfig(**model_d)

        job = JobConfig(
            id="adhoc_generate",
            operation="generate",
            ingest={"source": "sequences"},
            params=params,
        )

        inputs = None

        def _load_lines(path: Path) -> List[str]:
            return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]

        if usr:
            job.ingest = IngestConfig(
                source="usr",
                dataset=usr,
                field=field,
                root=(usr_root.as_posix() if usr_root else None),
                ids=_read_ids_arg(ids),
            )
        elif prompt_file:
            job.ingest = IngestConfig(source="sequences")
            inputs = _load_lines(prompt_file)
        elif prompt:
            job.ingest = IngestConfig(source="sequences")
            inputs = prompt
        else:
            raise ConfigError("Provide prompts via --prompt/--prompt-file or use --usr.")

        if dry_run:
            render_config_summary(model, [job])
            t = {"id": "gen", "fn": "generate", "format": "—", "params": params}
            render_outputs_spec_table([t])  # simple visualization
            console.print("[green]✔ Generate validated (dry run).[/green]")
            raise typer.Exit(code=0)

        if not progress:
            os.environ["DNADESIGN_PROGRESS"] = "0"
        pm = RichProgressManager(enabled=progress)
        with pm:
            res = run_job(
                inputs=inputs,
                model=model,
                job=job,
                progress_factory=pm.factory if progress else None,
            )

        seqs = res.get("gen_seqs", [])
        console.print(f"[green]Generated {len(seqs)} sequence(s).[/green]")
        if out:
            if out.suffix.lower() == ".json":
                out.write_text(json.dumps(seqs, indent=2))
            else:
                out.write_text("\n".join(seqs) + "\n")
            console.print(f"[accent]Wrote:[/accent] {out}")

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=_exit_for(e))


# ───────────────────────────────────────────────────────────────────────────────
# PRESETS subcommands
# ───────────────────────────────────────────────────────────────────────────────

presets_app = typer.Typer(no_args_is_help=True, help="Presets registry.")
app.add_typer(presets_app, name="presets")


@presets_app.command("list", help="List available presets.")
def presets_list():
    render_presets_table(list_presets())


@presets_app.command("show", help="Show a preset details.")
def presets_show(preset: str = typer.Argument(...)):
    try:
        p = load_preset(preset)
        render_preset_detail(p)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)


# Existing groups
adapters_app = typer.Typer(no_args_is_help=True, help="Adapter utilities.")
app.add_typer(adapters_app, name="adapters")


@adapters_app.command("list", help="List registered model_ids and capabilities.")
def adapters_list():
    models = list_models()
    render_adapters_table(models)


@adapters_app.command("fns", help="List registered namespaced functions.")
def adapters_fns():
    render_functions_table(list_fns())


@adapters_app.command("cache-clear", help="Clear in-process adapter cache.")
def adapters_cache_clear():
    clear_adapter_cache()
    console.print("[green]✔ Adapter cache cleared.[/green]")


validate_app = typer.Typer(no_args_is_help=False, help="Validation utilities.")
app.add_typer(validate_app, name="validate")


@validate_app.command("config", help="Validate a config file (default discovery if omitted).")
def validate_config(config: Optional[Path] = typer.Option(None, "--config")):
    try:
        cfg_path = _discovery_config(config)
        root = RootConfig(**yaml.safe_load(cfg_path.read_text()))
        render_config_summary(root.model, root.jobs)
        console.print("[green]✔ Config validated.[/green]")
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=_exit_for(e))


@validate_app.command("usr", help="Validate a USR dataset can be read (id + field).")
def validate_usr(
    dataset: str = typer.Option(..., "--dataset"),
    field: str = typer.Option("sequence", "--field"),
    usr_root: Optional[Path] = typer.Option(None, "--usr-root"),
    ids: Optional[str] = typer.Option(None, "--ids", help="Path or CSV of ids to subset"),
):
    try:
        from .ingest.sources import load_usr_input

        seqs, id_list, ds = load_usr_input(
            dataset_name=dataset,
            field=field,
            root=(usr_root.as_posix() if usr_root else None),
            ids=_read_ids_arg(ids),
        )
        console.print(f"[green]✔ USR OK[/green]  dataset={dataset}  rows={len(seqs)}  field={field}")
        console.print(f"[accent]records:[/accent] {ds.records_path}")
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=_exit_for(e))
