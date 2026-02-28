"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/deps.py

Dependency builder helpers for USR CLI command handler wiring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..dataset import Dataset
from . import maintenance as maintenance_commands
from . import materialize as materialize_commands
from . import merge as merge_commands
from . import namespace_handlers as namespace_handlers_commands
from . import read_views as read_views_commands
from . import runtime as runtime_commands
from . import tooling as tooling_commands


def build_read_view_deps(
    *,
    is_explicit_path_target: Callable[[str | None], bool],
    exit_missing_path_target: Callable[[str], None],
    resolve_existing_dataset_id: Callable[[Path, str], str],
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    assert_not_legacy_dataset_path: Callable[[Path, Path | None], None],
    legacy_dataset_prefix: str,
    legacy_dataset_path_error: str,
) -> read_views_commands.ReadViewDeps:
    return read_views_commands.ReadViewDeps(
        is_explicit_path_target=is_explicit_path_target,
        exit_missing_path_target=exit_missing_path_target,
        resolve_existing_dataset_id=resolve_existing_dataset_id,
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        assert_not_legacy_dataset_path=assert_not_legacy_dataset_path,
        legacy_dataset_prefix=legacy_dataset_prefix,
        legacy_dataset_path_error=legacy_dataset_path_error,
    )


def build_runtime_deps(
    *,
    resolve_dataset_for_read: Callable[[Path, str], Dataset],
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    resolve_output_format: Callable[[object], str],
    print_json: Callable[[dict], None],
    output_version: int,
) -> runtime_commands.RuntimeDeps:
    return runtime_commands.RuntimeDeps(
        resolve_dataset_for_read=resolve_dataset_for_read,
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        resolve_output_format=resolve_output_format,
        print_json=print_json,
        output_version=output_version,
    )


def build_materialize_deps(
    *,
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    is_interactive: Callable[[], bool],
    confirm: Callable[[str], bool],
) -> materialize_commands.MaterializeDeps:
    return materialize_commands.MaterializeDeps(
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        is_interactive=is_interactive,
        confirm=confirm,
    )


def build_maintenance_deps(
    *,
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    prompt: Callable[[str], str],
) -> maintenance_commands.MaintenanceDeps:
    return maintenance_commands.MaintenanceDeps(
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        prompt=prompt,
    )


def build_merge_deps(
    *,
    resolve_merge_policy: Callable[[str], object],
    merge_usr_to_usr: Callable[..., object],
    mode_require_same: object,
    mode_union: object,
) -> merge_commands.MergeDeps:
    return merge_commands.MergeDeps(
        resolve_merge_policy=resolve_merge_policy,
        merge_usr_to_usr=merge_usr_to_usr,
        mode_require_same=mode_require_same,
        mode_union=mode_union,
        dataset_factory=Dataset,
    )


def build_namespace_deps(
    *,
    load_registry: Callable[[Path], object],
    parse_columns_spec: Callable[[str], list[tuple[str, str]]],
    register_namespace: Callable[..., object],
) -> namespace_handlers_commands.NamespaceDeps:
    return namespace_handlers_commands.NamespaceDeps(
        load_registry=load_registry,
        parse_columns_spec=parse_columns_spec,
        register_namespace=register_namespace,
    )


def build_tooling_deps(
    *,
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    resolve_path_anywhere: Callable[[Path], Path],
    create_mock_dataset: Callable[..., object],
    add_demo_columns: Callable[..., object],
) -> tooling_commands.ToolingDeps:
    return tooling_commands.ToolingDeps(
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        resolve_path_anywhere=resolve_path_anywhere,
        create_mock_dataset=create_mock_dataset,
        add_demo_columns=add_demo_columns,
        dataset_factory=Dataset,
    )
