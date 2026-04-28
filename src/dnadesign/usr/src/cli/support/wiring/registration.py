"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/support/wiring/registration.py

CLI app registration helpers for the USR entrypoint facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Callable

import typer

from ....contracts import SequencesError
from ...commands import tooling as tooling_commands
from ...commands.genbank.cli import register_genbank_commands
from ...commands.lifecycle import register_lifecycle_commands
from ...commands.maintenance import register_maintenance_commands
from ...commands.namespace.cli import register_namespace_commands
from ...commands.query import register_query_commands
from ...commands.remotes.cli import register_remotes_commands
from ...commands.sync.cli import register_sync_commands
from .surface import CliApps


def ctx_args(ctx: typer.Context, **kwargs) -> NS:
    base = {
        "root": ctx.obj["root"],
        "rich": ctx.obj["rich"],
        "remotes_config": ctx.obj.get("remotes_config"),
    }
    base.update(kwargs)
    return NS(**base)


def build_root_callback(
    *,
    default_usr_root: Callable[[], Path],
    normalize_usr_root: Callable[[Path], Path],
    assert_supported_root: Callable[[Path], None],
) -> Callable[..., None]:
    def _root(
        ctx: typer.Context,
        root: Path = typer.Option(
            default_usr_root(),
            "--root",
            help="Datasets root folder",
            readable=True,
            exists=True,
            dir_okay=True,
            file_okay=False,
            path_type=Path,
        ),
        rich: bool = typer.Option(True, "--rich/--no-rich", help="Use Rich formatting for supported commands"),
        remotes_config: Path | None = typer.Option(
            None,
            "--remotes-config",
            help="Explicit remotes config path for this invocation (sets USR_REMOTES_PATH).",
            path_type=Path,
            dir_okay=False,
            file_okay=True,
        ),
    ) -> None:
        if remotes_config is not None:
            os.environ["USR_REMOTES_PATH"] = str(remotes_config.expanduser())
        try:
            normalized_root = normalize_usr_root(root)
            assert_supported_root(normalized_root)
        except SequencesError as exc:
            raise typer.BadParameter(str(exc), param_hint="--root") from exc
        ctx.obj = {"root": normalized_root, "rich": rich, "remotes_config": remotes_config}

    return _root


def register_cli_surface(*, apps: CliApps, root_callback: Callable[..., None], ctx_args_builder, bindings) -> None:
    apps.app.callback()(root_callback)

    register_sync_commands(
        apps.app,
        sync_args_builder=ctx_args_builder,
        cmd_diff=bindings.cmd_diff,
        cmd_pull=bindings.cmd_pull,
        cmd_push=bindings.cmd_push,
    )

    register_maintenance_commands(
        apps.maintenance_app,
        ctx_args_builder=ctx_args_builder,
        cmd_dedupe_sequences=bindings.cmd_dedupe_sequences,
        cmd_registry_freeze=bindings.cmd_registry_freeze,
        cmd_overlay_compact=bindings.cmd_overlay_compact,
        cmd_overlay_project=bindings.cmd_overlay_project,
        cmd_overlay_remove=bindings.cmd_overlay_remove,
        cmd_event_log_garden=bindings.cmd_event_log_garden,
        cmd_merge_datasets=bindings.cmd_merge_datasets,
    )

    register_genbank_commands(
        apps.genbank_app,
        ctx_args_builder=ctx_args_builder,
    )

    tooling_commands.register_ops_commands(
        apps.densegen_app,
        apps.dev_app,
        apps.legacy_app,
        ctx_args_builder=ctx_args_builder,
        cmd_repair_densegen=bindings.cmd_repair_densegen,
        cmd_make_mock=bindings.cmd_make_mock,
        cmd_add_demo=bindings.cmd_add_demo,
        cmd_convert_legacy=bindings.cmd_convert_legacy,
    )

    register_query_commands(
        apps.app,
        apps.events_app,
        ctx_args_builder=ctx_args_builder,
        cmd_ls=bindings.cmd_ls,
        cmd_info=bindings.cmd_info,
        cmd_schema=bindings.cmd_schema,
        cmd_head=bindings.cmd_head,
        cmd_cols=bindings.cmd_cols,
        cmd_describe=bindings.cmd_describe,
        cmd_cell=bindings.cmd_cell,
        cmd_validate=bindings.cmd_validate,
        cmd_events_tail=bindings.cmd_events_tail,
        cmd_get=bindings.cmd_get,
        cmd_grep=bindings.cmd_grep,
        cmd_export=bindings.cmd_export,
    )

    register_lifecycle_commands(
        apps.app,
        apps.state_app,
        ctx_args_builder=ctx_args_builder,
        cmd_init=bindings.cmd_init,
        cmd_import=bindings.cmd_import,
        cmd_attach=bindings.cmd_attach,
        cmd_delete=bindings.cmd_delete,
        cmd_restore=bindings.cmd_restore,
        cmd_state_set=bindings.cmd_state_set,
        cmd_state_clear=bindings.cmd_state_clear,
        cmd_state_get=bindings.cmd_state_get,
        cmd_materialize=bindings.cmd_materialize,
        cmd_snapshot=bindings.cmd_snapshot,
    )

    register_remotes_commands(
        apps.remotes_app,
        ctx_args_builder=ctx_args_builder,
        cmd_remotes_list=bindings.cmd_remotes_list,
        cmd_remotes_show=bindings.cmd_remotes_show,
        cmd_remotes_add=bindings.cmd_remotes_add,
        cmd_remotes_wizard=bindings.cmd_remotes_wizard,
        cmd_remotes_doctor=bindings.cmd_remotes_doctor,
        cmd_remotes_status=bindings.cmd_remotes_status,
        cmd_remotes_warm_auth=bindings.cmd_remotes_warm_auth,
    )

    register_namespace_commands(
        apps.namespace_app,
        ctx_args_builder=ctx_args_builder,
        cmd_namespace_list=bindings.cmd_namespace_list,
        cmd_namespace_show=bindings.cmd_namespace_show,
        cmd_namespace_register=bindings.cmd_namespace_register,
    )


__all__ = ["build_root_callback", "ctx_args", "register_cli_surface"]
