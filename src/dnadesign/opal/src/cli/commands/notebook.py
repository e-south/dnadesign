"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/notebook.py

Provides notebook CLI commands for campaign analysis workflows. Generates and.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...analysis.campaign import CampaignAnalysis
from ...analysis.notebook_set_template import render_campaign_set_notebook
from ...analysis.notebook_template import render_campaign_notebook
from ...core.rounds import resolve_round_index_from_runs
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...reporting.campaign_set_artifacts import materialize_campaign_set_collection_visuals
from ...reporting.notebook import smoke_check_notebook
from ...reporting.notebook_set import build_campaign_set_notebook_view_model
from ..registry import cli_group
from ..tui import tui_enabled
from ._common import internal_error, json_error, json_out, opal_error, print_config_context, prompt_confirm
from .notebook_generation import (
    NOTEBOOK_GENERATE_SCHEMA_VERSION,
    notebook_generate_payload,
    resolve_generation_run_scope,
)
from .notebook_support import (
    launch_marimo_notebook,
    list_notebooks,
    notebook_rows,
    parse_notebook_round_selector,
    print_rich,
    resolve_notebook_name,
    resolve_notebook_path,
    rich_kv_table,
    rich_list_table,
)

notebook_app = typer.Typer(no_args_is_help=False, help="Notebook workflows (marimo).")
cli_group("notebook", help="Notebook workflows (marimo).")(notebook_app)


@notebook_app.callback(invoke_without_command=True)
def notebook_root(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
) -> None:
    if ctx.invoked_subcommand:
        return
    try:
        analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
        ws = analysis.workspace
        notebooks_dir = ws.workdir / "notebooks"
        notebooks = list_notebooks(notebooks_dir)
        if not notebooks:
            if tui_enabled():
                table = rich_kv_table(
                    "Notebook",
                    {
                        "Status": "No notebooks found",
                        "Next step": f"uv run opal notebook generate -c {analysis.config_path} --round latest",
                        "Tip": "use --name to customize the notebook filename",
                    },
                )
                if print_rich(table):
                    return
            print_stdout(
                "No notebooks found. Generate one with:\n"
                f"  uv run opal notebook generate -c {analysis.config_path} --round latest\n"
                "Tip: use --name to customize the notebook filename."
            )
            return
        if len(notebooks) == 1:
            if tui_enabled():
                table = rich_kv_table(
                    "Notebook",
                    {
                        "Notebook": notebooks[0].name,
                        "Run": f"uv run opal notebook run -c {analysis.config_path}",
                    },
                )
                if print_rich(table):
                    return
            print_stdout(
                "Notebook available:\n"
                f"  {notebooks[0].name}\n"
                "Run it with:\n"
                f"  uv run opal notebook run -c {analysis.config_path}"
            )
            return
        rows = notebook_rows(notebooks)
        if tui_enabled():
            table = rich_list_table("Notebooks", rows)
            if print_rich(table):
                hint = rich_kv_table(
                    "Next steps",
                    {
                        "Run": f"uv run opal notebook run -c {analysis.config_path}",
                        "Pick": "Or specify a file with --path",
                    },
                )
                print_rich(hint)
                return
        print_stdout(
            "Multiple notebooks found:\n" + "\n".join(rows) + "\nRun with:\n"
            f"  uv run opal notebook run -c {analysis.config_path}\n"
            "Or specify a file with --path."
        )
    except OpalError as e:
        opal_error("notebook", e)
        raise typer.Exit(code=e.exit_code)


@notebook_app.command("generate", help="Generate a campaign or campaign-set marimo notebook.")
def cmd_notebook_generate(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
    campaign: Optional[list[Path]] = typer.Option(
        None,
        "--campaign",
        help="Campaign config or directory for campaign-set notebooks. Repeat for each campaign.",
    ),
    round: Optional[str] = typer.Option(
        "latest",
        "--round",
        "-r",
        help="Default round selector (int or 'latest'; campaign-set notebooks also support 'all').",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Pin the generated single-campaign notebook to a run_id from outputs/ledger/runs.parquet.",
    ),
    collection: Optional[Path] = typer.Option(
        None,
        "--collection",
        help="Optional opal.campaign_collection.v2 manifest for campaign-set comparison views.",
    ),
    collection_visual_index: Optional[Path] = typer.Option(
        None,
        "--collection-visual-index",
        help=(
            "Existing opal.collection_visual_manifest_index.v1 to embed in a campaign-set notebook. "
            "Use with --no-materialize-collection-visuals."
        ),
    ),
    materialize_collection_visuals: bool = typer.Option(
        True,
        "--materialize-collection-visuals/--no-materialize-collection-visuals",
        help="Materialize campaign-set comparison CSV/PNG/manifests before writing the notebook.",
    ),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help=(
            "Output notebook path "
            "(default: <workdir>/notebooks/opal_<slug>_analysis.py or opal_campaign_set_analysis.py)."
        ),
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Notebook file name (defaults to opal_<slug>_analysis.py).",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite if the notebook already exists."),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate requested round against existing runs before generating the notebook.",
    ),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON summary."),
) -> None:
    try:
        campaign_set_paths = list(campaign or [])
        if campaign_set_paths:
            if run_id is not None:
                raise OpalError(
                    "--run-id is only supported for single-campaign notebook generation.",
                    ExitCodes.BAD_ARGS,
                )
            _generate_campaign_set_notebook(
                config=config,
                campaign_paths=campaign_set_paths,
                round=round,
                out=out,
                name=name,
                collection=collection,
                collection_visual_index=collection_visual_index,
                materialize_collection_visuals=materialize_collection_visuals,
                force=force,
                validate=validate,
                json=json,
            )
            return

        if collection is not None:
            raise OpalError("--collection is only supported for campaign-set notebook generation.", ExitCodes.BAD_ARGS)
        if collection_visual_index is not None:
            raise OpalError(
                "--collection-visual-index is only supported for campaign-set notebook generation.",
                ExitCodes.BAD_ARGS,
            )

        analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
        cfg = analysis.config
        ws = analysis.workspace
        store = analysis.records_store()
        if not store.records_path.exists():
            raise OpalError(f"records.parquet not found: {store.records_path}", ExitCodes.BAD_ARGS)
        if out is not None and name is not None:
            raise OpalError("Use --out or --name, not both.", ExitCodes.BAD_ARGS)

        round_sel = parse_notebook_round_selector(round, allow_all=False)

        if run_id is not None:
            round_sel, resolved_run_id = resolve_generation_run_scope(
                analysis,
                round_selector=round_sel,
                run_id=run_id,
            )
        else:
            resolved_run_id = None

        if validate and ws.ledger_runs_path.exists():
            runs_df = analysis.read_runs()
            # Validate requested round exists (or at least that runs are available for "latest").
            resolve_round_index_from_runs(runs_df, round_sel)

        default_name = f"opal_{cfg.campaign.slug}_analysis.py"
        notebook_name = resolve_notebook_name(name, default_name)
        default_out = ws.workdir / "notebooks" / notebook_name
        out_path = Path(out) if out is not None else default_out
        overwritten = out_path.exists()
        if out_path.exists() and not force:
            msg = (
                f"Notebook already exists: {out_path}. "
                "Use --force to overwrite or --name to choose a different filename."
            )
            try:
                confirmed = prompt_confirm(
                    f"{msg}\nOverwrite? (y/N): ",
                    non_interactive_hint=msg,
                )
            except OpalError:
                raise
            if not confirmed:
                if json:
                    json_out(
                        {
                            "schema_version": NOTEBOOK_GENERATE_SCHEMA_VERSION,
                            "ok": False,
                            "status": "aborted",
                            "notebook_path": str(out_path),
                        }
                    )
                else:
                    print_stdout("Aborted.")
                return

        content = render_campaign_notebook(analysis.config_path, round_selector=round_sel, run_id=resolved_run_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content)
        smoke_check_notebook(out_path, run_marimo_check=True)
        if json:
            json_out(
                notebook_generate_payload(
                    kind="campaign",
                    out_path=out_path,
                    round_selector=round_sel,
                    run_id=resolved_run_id,
                    validate=validate,
                    force=force,
                    overwritten=overwritten,
                    analyses=[analysis],
                )
            )
            return

        if tui_enabled():
            table = rich_kv_table(
                "Notebook Generated",
                {
                    "Config": analysis.config_path,
                    "Workdir": ws.workdir,
                    "Notebook": out_path,
                    "Run ID": resolved_run_id or "",
                },
            )
            if print_rich(table):
                hint = rich_list_table(
                    "Next steps",
                    [f"uv run opal notebook run -c {analysis.config_path}"],
                )
                print_rich(hint)
            else:
                print_config_context(analysis.config_path, cfg=cfg)
                print_stdout(f"Notebook written: {out_path}")
        else:
            print_config_context(analysis.config_path, cfg=cfg)
            print_stdout(f"Notebook written: {out_path}")
    except OpalError as e:
        if json:
            json_error("notebook.generate", e)
        else:
            opal_error("notebook.generate", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("notebook.generate", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


def _generate_campaign_set_notebook(
    *,
    config: Optional[Path],
    campaign_paths: list[Path],
    round: Optional[str],
    out: Optional[Path],
    name: Optional[str],
    collection: Optional[Path],
    collection_visual_index: Optional[Path],
    materialize_collection_visuals: bool,
    force: bool,
    validate: bool,
    json: bool,
) -> None:
    config_paths = ([config] if config is not None else []) + campaign_paths
    if not config_paths:
        raise OpalError("Notebook generation requires at least one campaign config.", ExitCodes.BAD_ARGS)
    resolved = [str(Path(path).resolve()) for path in config_paths]
    duplicates = sorted({path for path in resolved if resolved.count(path) > 1})
    if duplicates:
        raise OpalError(
            "Campaign-set notebook generation requires distinct campaign configs; duplicates: " + ", ".join(duplicates),
            ExitCodes.BAD_ARGS,
        )
    if out is not None and name is not None:
        raise OpalError("Use --out or --name, not both.", ExitCodes.BAD_ARGS)
    if collection_visual_index is not None:
        if collection is None:
            raise OpalError("--collection-visual-index requires --collection.", ExitCodes.BAD_ARGS)
        if materialize_collection_visuals:
            raise OpalError(
                "Use --no-materialize-collection-visuals when passing --collection-visual-index.",
                ExitCodes.BAD_ARGS,
            )

    analyses = [CampaignAnalysis.from_config_path(path, allow_dir=True) for path in config_paths]
    round_sel = parse_notebook_round_selector(round, allow_all=True)

    for analysis in analyses:
        store = analysis.records_store()
        if not store.records_path.exists():
            raise OpalError(f"records.parquet not found: {store.records_path}", ExitCodes.BAD_ARGS)
        if validate and round_sel != "all" and analysis.workspace.ledger_runs_path.exists():
            resolve_round_index_from_runs(analysis.read_runs(), round_sel)

    default_name = "opal_campaign_set_analysis.py"
    notebook_name = resolve_notebook_name(name, default_name)
    default_out = analyses[0].workspace.workdir / "notebooks" / notebook_name
    out_path = Path(out) if out is not None else default_out
    overwritten = out_path.exists()
    if out_path.exists() and not force:
        msg = f"Notebook already exists: {out_path}. Use --force to overwrite or --name to choose a different filename."
        try:
            confirmed = prompt_confirm(f"{msg}\nOverwrite? (y/N): ", non_interactive_hint=msg)
        except OpalError:
            raise
        if not confirmed:
            if json:
                json_out(
                    {
                        "schema_version": NOTEBOOK_GENERATE_SCHEMA_VERSION,
                        "ok": False,
                        "status": "aborted",
                        "notebook_path": str(out_path),
                    }
                )
            else:
                print_stdout("Aborted.")
            return

    if collection_visual_index is not None:
        build_campaign_set_notebook_view_model(
            [analysis.config_path for analysis in analyses],
            round_selector=round_sel,
            run_id=None,
            collection_manifest_path=collection,
            collection_visual_index_path=collection_visual_index,
        )

    collection_visual_index_path: Path | None = collection_visual_index
    if collection is not None and materialize_collection_visuals:
        view_model = build_campaign_set_notebook_view_model(
            [analysis.config_path for analysis in analyses],
            round_selector=round_sel,
            collection_manifest_path=collection,
        )
        visual_index = materialize_campaign_set_collection_visuals(
            view_model["campaigns"],
            collection=view_model["collection"],
            output_dir=out_path.parent / "collection_visuals",
        )
        collection_visual_index_path = Path(str(visual_index["output_dir"])) / "collection_visual_manifest.json"

    content = render_campaign_set_notebook(
        [analysis.config_path for analysis in analyses],
        round_selector=round_sel,
        collection_manifest_path=collection,
        collection_visual_index_path=collection_visual_index_path,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    smoke_check_notebook(out_path, run_marimo_check=True)
    if json:
        json_out(
            notebook_generate_payload(
                kind="campaign_set",
                out_path=out_path,
                round_selector=round_sel,
                run_id=None,
                validate=validate,
                force=force,
                overwritten=overwritten,
                analyses=analyses,
                collection_manifest_path=collection,
                collection_visual_index_path=collection_visual_index_path,
            )
        )
        return

    if tui_enabled():
        table = rich_kv_table(
            "Campaign Set Notebook Generated",
            {
                "Campaigns": len(analyses),
                "Notebook": out_path,
                "Round": round_sel,
                "Collection": collection or "",
                "Collection visuals": collection_visual_index_path or "",
            },
        )
        if print_rich(table):
            return
    print_stdout(f"Campaign-set notebook written: {out_path}")


@notebook_app.command("run", help="Launch a generated notebook in read-only marimo app mode.")
def cmd_notebook_run(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
    path: Optional[Path] = typer.Option(None, "--path", help="Notebook path (defaults to generated path)."),
    host: Optional[str] = typer.Option(None, "--host", help="Host passed to `marimo run`."),
    port: Optional[int] = typer.Option(None, "--port", help="Port passed to `marimo run`."),
    headless: bool = typer.Option(False, "--headless", help="Run marimo without launching a browser."),
) -> None:
    try:
        analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
        nb_path = resolve_notebook_path(analysis, path)
        print_stdout(f"Launching marimo app: {nb_path}")
        launch_marimo_notebook(mode="run", notebook_path=nb_path, host=host, port=port, headless=headless)
    except OpalError as e:
        opal_error("notebook.run", e)
        raise typer.Exit(code=e.exit_code)
    except FileNotFoundError:
        opal_error(
            "notebook.run",
            OpalError("marimo CLI not found on PATH. Install marimo or use `uv run marimo run ...`."),
        )
        raise typer.Exit(code=ExitCodes.BAD_ARGS)
    except Exception as e:
        internal_error("notebook.run", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@notebook_app.command("edit", help="Launch a generated notebook in editable marimo mode.")
def cmd_notebook_edit(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
    path: Optional[Path] = typer.Option(None, "--path", help="Notebook path (defaults to generated path)."),
    host: Optional[str] = typer.Option(None, "--host", help="Host passed to `marimo edit`."),
    port: Optional[int] = typer.Option(None, "--port", help="Port passed to `marimo edit`."),
    headless: bool = typer.Option(False, "--headless", help="Run marimo without launching a browser."),
) -> None:
    try:
        analysis = CampaignAnalysis.from_config_path(config, allow_dir=True)
        nb_path = resolve_notebook_path(analysis, path)
        print_stdout(f"Launching marimo editor: {nb_path}")
        launch_marimo_notebook(mode="edit", notebook_path=nb_path, host=host, port=port, headless=headless)
    except OpalError as e:
        opal_error("notebook.edit", e)
        raise typer.Exit(code=e.exit_code)
    except FileNotFoundError:
        opal_error(
            "notebook.edit",
            OpalError("marimo CLI not found on PATH. Install marimo or use `uv run marimo edit ...`."),
        )
        raise typer.Exit(code=ExitCodes.BAD_ARGS)
    except Exception as e:
        internal_error("notebook.edit", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
