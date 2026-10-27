"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/__init__.py

Typer CLI entrypoint facade for USR dataset operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import typer

from ..contracts import SequencesError, UserAbort
from ..datasets.demo.mock import add_demo_columns, create_mock_dataset
from ..datasets.merge import merge_usr_to_usr
from ..registry import load_registry, parse_columns_spec, register_namespace
from .commands import error_output as error_output_commands
from .commands import remotes as remotes_commands
from .support.presentation import runtime as runtime_support
from .support.resolution import dataset_targets as dataset_target_support
from .support.resolution.merge_policy import resolve_merge_policy
from .support.resolution.paths import LEGACY_DATASET_PATH_ERROR as _LEGACY_DATASET_PATH_ERROR
from .support.resolution.paths import assert_not_legacy_dataset_path as _assert_not_legacy_dataset_path_impl
from .support.resolution.paths import assert_supported_root as _assert_supported_root_impl
from .support.resolution.paths import resolve_dataset_for_read as _resolve_dataset_for_read_impl
from .support.resolution.paths import resolve_path_anywhere as _resolve_path_anywhere_impl
from .support.resolution.roots import default_usr_root as _default_usr_root_impl
from .support.resolution.roots import normalize_usr_root as _normalize_usr_root_impl
from .support.resolution.roots import pkg_usr_root as _pkg_usr_root_impl
from .support.resolution.roots import resolve_usr_root_from_env as _resolve_usr_root_from_env_impl
from .support.wiring import dependencies as dependency_support
from .support.wiring import registration as registration_support
from .support.wiring.bindings import build_cli_bindings
from .support.wiring.surface import build_cli_apps

# Compatibility exports kept for existing monkeypatch-based tests.
shutil = remotes_commands.shutil
SSHRemote = remotes_commands.SSHRemote

USR_OUTPUT_VERSION = 1
LEGACY_DATASET_PATH_ERROR = _LEGACY_DATASET_PATH_ERROR


def _resolve_output_format(args, *, default: str = "auto") -> str:
    return runtime_support.resolve_output_format(args, default=default)


def _print_json(payload) -> None:
    runtime_support.print_json(payload)


def _is_interactive() -> bool:
    return runtime_support.is_interactive()


def _normalize_dataset_id(dataset: str) -> str:
    return dataset_target_support.normalize_dataset_id(dataset)


def _resolve_existing_dataset_id(root: Path, dataset: str) -> str:
    return dataset_target_support.resolve_existing_dataset_id(root, dataset)


def _resolve_dataset_name_interactive(root: Path, dataset: str | None, use_rich: bool) -> str | None:
    return dataset_target_support.resolve_dataset_name_interactive(root, dataset, use_rich)


def _is_explicit_path_target(target: str | None) -> bool:
    return dataset_target_support.is_explicit_path_target(target)


def _exit_missing_path_target(target: str) -> None:
    dataset_target_support.exit_missing_path_target(target)


def _pkg_usr_root() -> Path:
    return _pkg_usr_root_impl()


def _resolve_dataset_for_read(root: Path, dataset_arg: str):
    return dataset_target_support.resolve_dataset_for_read(
        root,
        dataset_arg,
        resolve_dataset_for_read_impl=_resolve_dataset_for_read_impl,
        resolve_existing_dataset_id_impl=_resolve_existing_dataset_id,
        normalize_dataset_id_impl=_normalize_dataset_id,
        pkg_root=_pkg_usr_root,
    )


def _assert_not_legacy_dataset_path(path: Path, *, root: Path | None = None) -> None:
    _assert_not_legacy_dataset_path_impl(path, root=root, pkg_root=_pkg_usr_root())


def _assert_not_legacy_dataset_path_for_read_views(path: Path, root: Path | None) -> None:
    _assert_not_legacy_dataset_path(path, root=root)


def _default_usr_root() -> Path:
    return _default_usr_root_impl(pkg_root=_pkg_usr_root())


def _normalize_usr_root(root: Path) -> Path:
    return _normalize_usr_root_impl(root, pkg_root=_pkg_usr_root())


def _resolve_usr_root_from_env() -> Path | None:
    return _resolve_usr_root_from_env_impl(pkg_root=_pkg_usr_root())


def _assert_supported_root(root: Path) -> None:
    _assert_supported_root_impl(root, pkg_root=_pkg_usr_root())


def _resolve_path_anywhere(path: Path) -> Path:
    return _resolve_path_anywhere_impl(path, pkg_root=_pkg_usr_root())


def _read_view_deps():
    return dependency_support.build_read_view_deps(
        is_explicit_path_target=_is_explicit_path_target,
        exit_missing_path_target=_exit_missing_path_target,
        resolve_existing_dataset_id=_resolve_existing_dataset_id,
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        assert_not_legacy_dataset_path=_assert_not_legacy_dataset_path_for_read_views,
        legacy_dataset_path_error=LEGACY_DATASET_PATH_ERROR,
    )


def _runtime_deps():
    return dependency_support.build_runtime_deps(
        resolve_dataset_for_read=_resolve_dataset_for_read,
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
    )


def _materialize_deps():
    return dependency_support.build_materialize_deps(
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        is_interactive=_is_interactive,
        confirm=lambda message: typer.confirm(message, default=False),
    )


def _snapshot_deps():
    return dependency_support.build_snapshot_deps(
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
    )


def _maintenance_deps():
    return dependency_support.build_maintenance_deps(
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        prompt=input,
    )


def _merge_deps():
    return dependency_support.build_merge_deps(
        resolve_merge_policy=resolve_merge_policy,
        get_merge_usr_to_usr=lambda: merge_usr_to_usr,
    )


def _namespace_deps():
    return dependency_support.build_namespace_deps(
        load_registry=load_registry,
        parse_columns_spec=parse_columns_spec,
        register_namespace=register_namespace,
    )


def _tooling_deps():
    return dependency_support.build_tooling_deps(
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        resolve_path_anywhere=_resolve_path_anywhere,
        create_mock_dataset=create_mock_dataset,
        add_demo_columns=add_demo_columns,
    )


list_datasets = dataset_target_support.list_datasets


def _print_user_error(error: SequencesError) -> None:
    error_output_commands.print_user_error(error)


_bindings = build_cli_bindings(
    resolve_path_anywhere=_resolve_path_anywhere,
    resolve_output_format=_resolve_output_format,
    print_json=_print_json,
    output_version=USR_OUTPUT_VERSION,
    resolve_dataset_for_read=_resolve_dataset_for_read,
    read_view_deps=_read_view_deps,
    runtime_deps=_runtime_deps,
    materialize_deps=_materialize_deps,
    snapshot_deps=_snapshot_deps,
    maintenance_deps=_maintenance_deps,
    merge_deps=_merge_deps,
    namespace_deps=_namespace_deps,
    tooling_deps=_tooling_deps,
    get_shutil_module=lambda: shutil,
    get_ssh_remote_class=lambda: SSHRemote,
)

cmd_repair_densegen = _bindings.cmd_repair_densegen
cmd_ls = _bindings.cmd_ls
cmd_init = _bindings.cmd_init
cmd_import = _bindings.cmd_import
cmd_attach = _bindings.cmd_attach
cmd_info = _bindings.cmd_info
cmd_schema = _bindings.cmd_schema
cmd_head = _bindings.cmd_head
cmd_cols = _bindings.cmd_cols
cmd_describe = _bindings.cmd_describe
cmd_cell = _bindings.cmd_cell
cmd_validate = _bindings.cmd_validate
cmd_registry_freeze = _bindings.cmd_registry_freeze
cmd_overlay_compact = _bindings.cmd_overlay_compact
cmd_overlay_project = _bindings.cmd_overlay_project
cmd_overlay_remove = _bindings.cmd_overlay_remove
cmd_events_tail = _bindings.cmd_events_tail
cmd_get = _bindings.cmd_get
cmd_grep = _bindings.cmd_grep
cmd_export = _bindings.cmd_export
cmd_delete = _bindings.cmd_delete
cmd_restore = _bindings.cmd_restore
cmd_state_set = _bindings.cmd_state_set
cmd_state_clear = _bindings.cmd_state_clear
cmd_state_get = _bindings.cmd_state_get
cmd_materialize = _bindings.cmd_materialize
cmd_snapshot = _bindings.cmd_snapshot
cmd_convert_legacy = _bindings.cmd_convert_legacy
cmd_make_mock = _bindings.cmd_make_mock
cmd_add_demo = _bindings.cmd_add_demo
cmd_merge_datasets = _bindings.cmd_merge_datasets
cmd_remotes_list = _bindings.cmd_remotes_list
cmd_remotes_show = _bindings.cmd_remotes_show
cmd_remotes_add = _bindings.cmd_remotes_add
cmd_remotes_wizard = _bindings.cmd_remotes_wizard
cmd_remotes_doctor = _bindings.cmd_remotes_doctor
cmd_remotes_status = _bindings.cmd_remotes_status
cmd_remotes_warm_auth = _bindings.cmd_remotes_warm_auth
cmd_namespace_list = _bindings.cmd_namespace_list
cmd_namespace_show = _bindings.cmd_namespace_show
cmd_namespace_register = _bindings.cmd_namespace_register
cmd_diff = _bindings.cmd_diff
cmd_pull = _bindings.cmd_pull
cmd_push = _bindings.cmd_push
cmd_dedupe_sequences = _bindings.cmd_dedupe_sequences

_cli_apps = build_cli_apps(show_dev_commands=os.getenv("USR_SHOW_DEV_COMMANDS") == "1")
app = _cli_apps.app
remotes_app = _cli_apps.remotes_app
legacy_app = _cli_apps.legacy_app
maintenance_app = _cli_apps.maintenance_app
densegen_app = _cli_apps.densegen_app
dev_app = _cli_apps.dev_app
namespace_app = _cli_apps.namespace_app
events_app = _cli_apps.events_app
state_app = _cli_apps.state_app

_ctx_args = registration_support.ctx_args
_root = registration_support.build_root_callback(
    default_usr_root=_default_usr_root,
    resolve_usr_root_from_env=_resolve_usr_root_from_env,
    normalize_usr_root=_normalize_usr_root,
    assert_supported_root=_assert_supported_root,
)
registration_support.register_cli_surface(
    apps=_cli_apps,
    root_callback=_root,
    ctx_args_builder=_ctx_args,
    bindings=_bindings,
)


def main() -> None:
    from .support.presentation.stderr_filter import maybe_install_pyarrow_sysctl_filter

    maybe_install_pyarrow_sysctl_filter()
    try:
        app()
    except UserAbort:
        raise SystemExit(130)
    except SequencesError as error:
        _print_user_error(error)
        raise SystemExit(2)
    except FileExistsError as error:
        print(f"ERROR: {error}")
        raise SystemExit(3)
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        raise SystemExit(4)


if __name__ == "__main__":
    main()
