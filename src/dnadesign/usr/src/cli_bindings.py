"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_bindings.py

Command binding helpers that adapt the USR CLI entrypoint to decomposed modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .cli_commands import maintenance as maintenance_commands
from .cli_commands import materialize as materialize_commands
from .cli_commands import merge as merge_commands
from .cli_commands import namespace_handlers as namespace_handlers_commands
from .cli_commands import read as read_commands
from .cli_commands import read_views as read_views_commands
from .cli_commands import remotes as remotes_commands
from .cli_commands import runtime as runtime_commands
from .cli_commands import sync as sync_commands
from .cli_commands import tooling as tooling_commands
from .cli_commands import write as write_commands

CommandBinding = Callable[[object], None]


@dataclass(frozen=True)
class CliBindings:
    cmd_repair_densegen: CommandBinding
    cmd_ls: CommandBinding
    cmd_init: CommandBinding
    cmd_import: CommandBinding
    cmd_attach: CommandBinding
    cmd_info: CommandBinding
    cmd_schema: CommandBinding
    cmd_head: CommandBinding
    cmd_cols: CommandBinding
    cmd_describe: CommandBinding
    cmd_cell: CommandBinding
    cmd_validate: CommandBinding
    cmd_registry_freeze: CommandBinding
    cmd_overlay_compact: CommandBinding
    cmd_overlay_remove: CommandBinding
    cmd_events_tail: CommandBinding
    cmd_get: CommandBinding
    cmd_grep: CommandBinding
    cmd_export: CommandBinding
    cmd_delete: CommandBinding
    cmd_restore: CommandBinding
    cmd_state_set: CommandBinding
    cmd_state_clear: CommandBinding
    cmd_state_get: CommandBinding
    cmd_materialize: CommandBinding
    cmd_snapshot: CommandBinding
    cmd_convert_legacy: CommandBinding
    cmd_make_mock: CommandBinding
    cmd_add_demo: CommandBinding
    cmd_merge_datasets: CommandBinding
    cmd_remotes_list: CommandBinding
    cmd_remotes_show: CommandBinding
    cmd_remotes_add: CommandBinding
    cmd_remotes_wizard: CommandBinding
    cmd_remotes_doctor: CommandBinding
    cmd_namespace_list: CommandBinding
    cmd_namespace_show: CommandBinding
    cmd_namespace_register: CommandBinding
    cmd_diff: CommandBinding
    cmd_pull: CommandBinding
    cmd_push: CommandBinding
    cmd_dedupe_sequences: CommandBinding


def build_cli_bindings(
    *,
    resolve_path_anywhere: Callable[[Path], Path],
    resolve_output_format: Callable[[object], str],
    print_json: Callable[[Any], None],
    output_version: int,
    resolve_dataset_for_read: Callable[[Path, str], object],
    read_view_deps: Callable[[], object],
    runtime_deps: Callable[[], object],
    materialize_deps: Callable[[], object],
    maintenance_deps: Callable[[], object],
    merge_deps: Callable[[], object],
    namespace_deps: Callable[[], object],
    tooling_deps: Callable[[], object],
    get_shutil_module: Callable[[], Any],
    get_ssh_remote_class: Callable[[], Any],
) -> CliBindings:
    def cmd_repair_densegen(args):
        tooling_commands.cmd_repair_densegen(args, deps=tooling_deps())

    def cmd_ls(args):
        read_commands.cmd_ls(
            args,
            resolve_output_format=resolve_output_format,
            print_json=print_json,
            output_version=output_version,
        )

    def cmd_init(args):
        write_commands.cmd_init(args)

    def cmd_import(args):
        write_commands.cmd_import(args, resolve_path_anywhere=resolve_path_anywhere)

    def cmd_attach(args):
        write_commands.cmd_attach(args, resolve_path_anywhere=resolve_path_anywhere)

    def cmd_info(args):
        read_commands.cmd_info(
            args,
            resolve_output_format=resolve_output_format,
            print_json=print_json,
            output_version=output_version,
            resolve_dataset_for_read=resolve_dataset_for_read,
        )

    def cmd_schema(args):
        read_commands.cmd_schema(
            args,
            resolve_output_format=resolve_output_format,
            print_json=print_json,
            output_version=output_version,
        )

    def cmd_head(args):
        read_views_commands.cmd_head(args, deps=read_view_deps())

    def cmd_cols(args):
        read_views_commands.cmd_cols(args, deps=read_view_deps())

    def cmd_describe(args):
        read_views_commands.cmd_describe(args, deps=read_view_deps())

    def cmd_cell(args):
        read_views_commands.cmd_cell(args, deps=read_view_deps())

    def cmd_validate(args):
        runtime_commands.cmd_validate(args, deps=runtime_deps())

    def cmd_registry_freeze(args):
        maintenance_commands.cmd_registry_freeze(args, deps=maintenance_deps())

    def cmd_overlay_compact(args):
        maintenance_commands.cmd_overlay_compact(args, deps=maintenance_deps())

    def cmd_overlay_remove(args):
        maintenance_commands.cmd_overlay_remove(args, deps=maintenance_deps())

    def cmd_events_tail(args):
        runtime_commands.cmd_events_tail(args, deps=runtime_deps())

    def cmd_get(args):
        runtime_commands.cmd_get(args, deps=runtime_deps())

    def cmd_grep(args):
        runtime_commands.cmd_grep(args, deps=runtime_deps())

    def cmd_export(args):
        runtime_commands.cmd_export(args, deps=runtime_deps())

    def cmd_delete(args):
        runtime_commands.cmd_delete(args, deps=runtime_deps())

    def cmd_restore(args):
        runtime_commands.cmd_restore(args, deps=runtime_deps())

    def cmd_state_set(args):
        runtime_commands.cmd_state_set(args, deps=runtime_deps())

    def cmd_state_clear(args):
        runtime_commands.cmd_state_clear(args, deps=runtime_deps())

    def cmd_state_get(args):
        runtime_commands.cmd_state_get(args, deps=runtime_deps())

    def cmd_materialize(args):
        materialize_commands.cmd_materialize(args, deps=materialize_deps())

    def cmd_snapshot(args):
        maintenance_commands.cmd_snapshot(args, deps=maintenance_deps())

    def cmd_convert_legacy(args):
        tooling_commands.cmd_convert_legacy(args, deps=tooling_deps())

    def cmd_make_mock(args):
        tooling_commands.cmd_make_mock(args, deps=tooling_deps())

    def cmd_add_demo(args):
        tooling_commands.cmd_add_demo(args, deps=tooling_deps())

    def cmd_merge_datasets(args):
        merge_commands.cmd_merge_datasets(args, deps=merge_deps())

    def cmd_remotes_list(args):
        remotes_commands.cmd_remotes_list(args)

    def cmd_remotes_show(args):
        remotes_commands.cmd_remotes_show(args)

    def cmd_remotes_add(args):
        remotes_commands.cmd_remotes_add(args)

    def cmd_remotes_wizard(args):
        remotes_commands.cmd_remotes_wizard(args)

    def cmd_remotes_doctor(args):
        remotes_commands.shutil = get_shutil_module()
        remotes_commands.SSHRemote = get_ssh_remote_class()
        remotes_commands.cmd_remotes_doctor(args)

    def cmd_namespace_list(args):
        namespace_handlers_commands.cmd_namespace_list(args, deps=namespace_deps())

    def cmd_namespace_show(args):
        namespace_handlers_commands.cmd_namespace_show(args, deps=namespace_deps())

    def cmd_namespace_register(args):
        namespace_handlers_commands.cmd_namespace_register(args, deps=namespace_deps())

    def cmd_diff(args):
        sync_commands.cmd_diff(
            args,
            resolve_output_format=resolve_output_format,
            print_json=print_json,
            output_version=output_version,
        )

    def cmd_pull(args):
        sync_commands.cmd_pull(args)

    def cmd_push(args):
        sync_commands.cmd_push(args)

    def cmd_dedupe_sequences(args):
        maintenance_commands.cmd_dedupe_sequences(args, deps=maintenance_deps())

    return CliBindings(
        cmd_repair_densegen=cmd_repair_densegen,
        cmd_ls=cmd_ls,
        cmd_init=cmd_init,
        cmd_import=cmd_import,
        cmd_attach=cmd_attach,
        cmd_info=cmd_info,
        cmd_schema=cmd_schema,
        cmd_head=cmd_head,
        cmd_cols=cmd_cols,
        cmd_describe=cmd_describe,
        cmd_cell=cmd_cell,
        cmd_validate=cmd_validate,
        cmd_registry_freeze=cmd_registry_freeze,
        cmd_overlay_compact=cmd_overlay_compact,
        cmd_overlay_remove=cmd_overlay_remove,
        cmd_events_tail=cmd_events_tail,
        cmd_get=cmd_get,
        cmd_grep=cmd_grep,
        cmd_export=cmd_export,
        cmd_delete=cmd_delete,
        cmd_restore=cmd_restore,
        cmd_state_set=cmd_state_set,
        cmd_state_clear=cmd_state_clear,
        cmd_state_get=cmd_state_get,
        cmd_materialize=cmd_materialize,
        cmd_snapshot=cmd_snapshot,
        cmd_convert_legacy=cmd_convert_legacy,
        cmd_make_mock=cmd_make_mock,
        cmd_add_demo=cmd_add_demo,
        cmd_merge_datasets=cmd_merge_datasets,
        cmd_remotes_list=cmd_remotes_list,
        cmd_remotes_show=cmd_remotes_show,
        cmd_remotes_add=cmd_remotes_add,
        cmd_remotes_wizard=cmd_remotes_wizard,
        cmd_remotes_doctor=cmd_remotes_doctor,
        cmd_namespace_list=cmd_namespace_list,
        cmd_namespace_show=cmd_namespace_show,
        cmd_namespace_register=cmd_namespace_register,
        cmd_diff=cmd_diff,
        cmd_pull=cmd_pull,
        cmd_push=cmd_push,
        cmd_dedupe_sequences=cmd_dedupe_sequences,
    )
