"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/support/wiring/dependencies.py

USR CLI dependency-builder wiring for the entrypoint facade.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ....dataset import ARCHIVE_DATASET_PREFIX, Dataset
from ....datasets.merge import MergeColumnsMode
from ...commands import deps as deps_commands
from ...commands import lifecycle as lifecycle_commands
from ...commands import maintenance as maintenance_commands
from ...commands import namespace as namespace_commands
from ...commands import query as query_commands
from ...commands import read_views as read_views_commands
from ...commands import tooling as tooling_commands


def build_read_view_deps(
    *,
    is_explicit_path_target: Callable[[str | None], bool],
    exit_missing_path_target: Callable[[str], None],
    resolve_existing_dataset_id: Callable[[Path, str], str],
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    assert_not_legacy_dataset_path: Callable[[Path, Path | None], None],
    legacy_dataset_path_error: str,
) -> read_views_commands.ReadViewDeps:
    return deps_commands.build_read_view_deps(
        is_explicit_path_target=is_explicit_path_target,
        exit_missing_path_target=exit_missing_path_target,
        resolve_existing_dataset_id=resolve_existing_dataset_id,
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        assert_not_legacy_dataset_path=assert_not_legacy_dataset_path,
        legacy_dataset_prefix=ARCHIVE_DATASET_PREFIX,
        legacy_dataset_path_error=legacy_dataset_path_error,
    )


def build_runtime_deps(
    *,
    resolve_dataset_for_read: Callable[[Path, str], Dataset],
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
) -> query_commands.RuntimeDeps:
    return deps_commands.build_runtime_deps(
        resolve_dataset_for_read=resolve_dataset_for_read,
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
    )


def build_materialize_deps(
    *,
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    is_interactive: Callable[[], bool],
    confirm: Callable[[str], bool],
) -> lifecycle_commands.MaterializeDeps:
    return deps_commands.build_materialize_deps(
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        is_interactive=is_interactive,
        confirm=confirm,
    )


def build_snapshot_deps(
    *,
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
) -> lifecycle_commands.SnapshotDeps:
    return deps_commands.build_snapshot_deps(
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
    )


def build_maintenance_deps(
    *,
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None],
    prompt: Callable[[str], str],
) -> maintenance_commands.MaintenanceDeps:
    return deps_commands.build_maintenance_deps(
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        prompt=prompt,
    )


def build_merge_deps(
    *,
    resolve_merge_policy: Callable[[str], object],
    get_merge_usr_to_usr: Callable[[], Callable[..., object]],
) -> maintenance_commands.MergeDeps:
    return deps_commands.build_merge_deps(
        resolve_merge_policy=resolve_merge_policy,
        merge_usr_to_usr=get_merge_usr_to_usr(),
        mode_require_same=MergeColumnsMode.REQUIRE_SAME,
        mode_union=MergeColumnsMode.UNION,
    )


def build_namespace_deps(
    *,
    load_registry: Callable[[Path], object],
    parse_columns_spec: Callable[[str], list[tuple[str, str]]],
    register_namespace: Callable[..., object],
) -> namespace_commands.NamespaceDeps:
    return deps_commands.build_namespace_deps(
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
    return deps_commands.build_tooling_deps(
        resolve_dataset_name_interactive=resolve_dataset_name_interactive,
        resolve_path_anywhere=resolve_path_anywhere,
        create_mock_dataset=create_mock_dataset,
        add_demo_columns=add_demo_columns,
    )


__all__ = [
    "build_maintenance_deps",
    "build_materialize_deps",
    "build_merge_deps",
    "build_namespace_deps",
    "build_read_view_deps",
    "build_runtime_deps",
    "build_snapshot_deps",
    "build_tooling_deps",
]
