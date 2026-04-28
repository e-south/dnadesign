"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/cli/commands/migrate.py

Registration for infer migration utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from ...config import RootConfig
from ...features.legacy_alias_migration import migrate_legacy_overlay_aliases
from ...features.legacy_payload_retirement import prune_stale_infer_overlay_columns, retire_legacy_overlay_payloads
from ...features.sequence_views import bundle_uses_sequence_views
from ..common import discovery_config, raise_cli_error
from ..config_inputs import resolve_config_sequence_view_roots


def register(app: typer.Typer) -> None:
    migrate_app = typer.Typer(no_args_is_help=True, help="Data-preserving migration utilities.")
    app.add_typer(migrate_app, name="migrate")

    @migrate_app.command(
        "legacy-overlay-aliases",
        help="Backfill sequence-view feature alias/vector sidecars from verified legacy row overlays.",
    )
    def legacy_overlay_aliases(
        config: Path = typer.Option(..., "--config"),
        job: str = typer.Option(..., "--job", help="Sequence-view job id to migrate."),
        legacy_job_id: str = typer.Option(..., "--legacy-job-id", help="Legacy row-overlay job id to read."),
        write: bool = typer.Option(False, "--write", help="Write alias/vector sidecars. Default is dry-run."),
        verify_payloads: bool = typer.Option(
            False,
            "--verify-payloads",
            help="During dry-run, read legacy feature payload columns. --write always reads payloads.",
        ),
        max_views: int | None = typer.Option(
            None,
            "--max-views",
            help="Optional smoke-test limit. Full migration should omit this.",
        ),
        fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    ) -> None:
        try:
            cfg_path = discovery_config(config)
            root = RootConfig(**yaml.safe_load(cfg_path.read_text(encoding="utf-8")))
            selected_jobs = [selected_job for selected_job in root.jobs if str(selected_job.id) == str(job)]
            if not selected_jobs:
                raise ValueError(f"No job selected for migration: {job}")
            selected_job = selected_jobs[0]
            if selected_job.feature_bundle is None or not bundle_uses_sequence_views(selected_job.feature_bundle):
                raise ValueError("legacy-overlay-aliases requires a job with feature_bundle.sequence_view_inputs.")
            resolve_config_sequence_view_roots(job=selected_job, config_dir=cfg_path.parent)
            result = migrate_legacy_overlay_aliases(
                bundle=selected_job.feature_bundle,
                model_id=root.model.id,
                legacy_job_id=legacy_job_id,
                write=write,
                max_views=max_views,
                verify_payloads=verify_payloads,
            ).to_dict()
            result["job_id"] = str(selected_job.id)
            result["mode"] = "write" if write else "dry_run"
            result["max_views"] = max_views
            if fmt == "json":
                typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":")))
                return
            if fmt != "text":
                raise ValueError("format must be one of: text, json.")
            typer.echo(
                "legacy-overlay-aliases "
                f"mode={result['mode']} job={selected_job.id} legacy_job_id={legacy_job_id} "
                f"required_vectors={result['required_vectors']} reusable={result['reusable_vectors']} "
                f"payload_unverified={result['payload_unverified_vectors']} "
                f"missing={result['missing_vectors']} unclassified={result['unclassified_vectors']} "
                f"orientation_blocked={result['orientation_blocked_vectors']} "
                f"vectors_written={result['vectors_written']} aliases_written={result['aliases_written']}"
            )
        except Exception as error:
            raise_cli_error(error)

    @migrate_app.command(
        "retire-legacy-payloads",
        help="Drop duplicated legacy row-overlay payload columns once canonical sequence-view vectors exist.",
    )
    def retire_legacy_payloads(
        config: Path = typer.Option(..., "--config"),
        job: str = typer.Option(..., "--job", help="Sequence-view job id whose canonical vectors protect cleanup."),
        legacy_job_id: str = typer.Option(..., "--legacy-job-id", help="Legacy row-overlay job id to prune."),
        write: bool = typer.Option(False, "--write", help="Rewrite/delete legacy payload columns. Default is dry-run."),
        keep_empty_parts: bool = typer.Option(
            False,
            "--keep-empty-parts",
            help="Keep part files that contain only id after payload retirement.",
        ),
        fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    ) -> None:
        try:
            cfg_path = discovery_config(config)
            root = RootConfig(**yaml.safe_load(cfg_path.read_text(encoding="utf-8")))
            selected_jobs = [selected_job for selected_job in root.jobs if str(selected_job.id) == str(job)]
            if not selected_jobs:
                raise ValueError(f"No job selected for payload retirement: {job}")
            selected_job = selected_jobs[0]
            if selected_job.feature_bundle is None or not bundle_uses_sequence_views(selected_job.feature_bundle):
                raise ValueError("retire-legacy-payloads requires a job with feature_bundle.sequence_view_inputs.")
            resolve_config_sequence_view_roots(job=selected_job, config_dir=cfg_path.parent)
            result = retire_legacy_overlay_payloads(
                bundle=selected_job.feature_bundle,
                model_id=root.model.id,
                legacy_job_id=legacy_job_id,
                write=write,
                delete_empty_parts=not keep_empty_parts,
            ).to_dict()
            result["job_id"] = str(selected_job.id)
            if fmt == "json":
                typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":")))
                return
            if fmt != "text":
                raise ValueError("format must be one of: text, json.")
            typer.echo(
                "retire-legacy-payloads "
                f"mode={result['mode']} job={selected_job.id} legacy_job_id={legacy_job_id} "
                f"required_vectors={result['required_vectors']} protected={result['protected_vectors']} "
                f"missing_modern={result['missing_modern_vectors']} "
                f"parts_scanned={result['legacy_parts_scanned']} "
                f"parts_with_payload={result['legacy_parts_with_payload']} "
                f"bytes_reclaimable={result['bytes_reclaimable']} "
                f"bytes_reclaimed={result['bytes_reclaimed']} "
                f"files_rewritten={result['files_rewritten']} files_deleted={result['files_deleted']}"
            )
        except Exception as error:
            raise_cli_error(error)

    @migrate_app.command(
        "prune-stale-overlay-columns",
        help="Drop explicitly approved stale Infer overlay columns by prefix/name without reading payloads.",
    )
    def prune_stale_overlay_columns(
        usr_root: Path = typer.Option(..., "--usr-root", help="USR datasets root."),
        dataset: str = typer.Option(..., "--dataset", help="Dataset id containing _derived/infer parts."),
        namespace: str = typer.Option("infer", "--namespace", help="Overlay namespace. Only infer is supported."),
        column_prefix: list[str] | None = typer.Option(
            None,
            "--column-prefix",
            help="Column prefix to prune. May be repeated.",
        ),
        column_name: list[str] | None = typer.Option(
            None,
            "--column-name",
            help="Exact column name to prune. May be repeated.",
        ),
        reason: str = typer.Option("", "--reason", help="Short audit reason recorded on write."),
        write: bool = typer.Option(False, "--write", help="Rewrite/delete selected columns. Default is dry-run."),
        keep_empty_parts: bool = typer.Option(
            False,
            "--keep-empty-parts",
            help="Keep part files that contain only id after stale-column pruning.",
        ),
        fmt: str = typer.Option("text", "--format", help="Output format: text or json."),
    ) -> None:
        try:
            result = prune_stale_infer_overlay_columns(
                dataset_root=usr_root,
                dataset_id=dataset,
                namespace=namespace,
                column_prefixes=column_prefix or (),
                column_names=column_name or (),
                reason=reason,
                write=write,
                delete_empty_parts=not keep_empty_parts,
            ).to_dict()
            if fmt == "json":
                typer.echo(json.dumps(result, sort_keys=True, separators=(",", ":")))
                return
            if fmt != "text":
                raise ValueError("format must be one of: text, json.")
            typer.echo(
                "prune-stale-overlay-columns "
                f"mode={result['mode']} dataset={dataset} namespace={namespace} "
                f"parts_scanned={result['parts_scanned']} "
                f"parts_with_columns={result['parts_with_columns']} "
                f"columns_removed={len(result['removed_columns'])} "
                f"bytes_reclaimable={result['bytes_reclaimable']} "
                f"bytes_reclaimed={result['bytes_reclaimed']} "
                f"files_rewritten={result['files_rewritten']} files_deleted={result['files_deleted']}"
            )
        except Exception as error:
            raise_cli_error(error)
