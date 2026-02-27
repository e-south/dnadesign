"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/cli.py

Typer CLI entrypoint for USR dataset operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import typer

from .cli_commands import datasets as dataset_commands
from .cli_commands import error_output as error_output_commands
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
from .cli_commands.lifecycle_cli import register_lifecycle_commands
from .cli_commands.namespace_cli import register_namespace_commands
from .cli_commands.ops_cli import register_ops_commands
from .cli_commands.query_cli import register_query_commands
from .cli_commands.remotes_cli import register_remotes_commands
from .cli_commands.sync_cli import register_sync_commands
from .cli_merge_policy import resolve_merge_policy
from .cli_paths import (
    LEGACY_DATASET_PATH_ERROR as _LEGACY_DATASET_PATH_ERROR,
)
from .cli_paths import (
    assert_not_legacy_dataset_path as _assert_not_legacy_dataset_path_impl,
)
from .cli_paths import (
    assert_supported_root as _assert_supported_root_impl,
)
from .cli_paths import (
    pkg_usr_root as _pkg_usr_root_impl,
)
from .cli_paths import (
    resolve_dataset_for_read as _resolve_dataset_for_read_impl,
)
from .cli_paths import (
    resolve_path_anywhere as _resolve_path_anywhere_impl,
)
from .cli_surface import build_cli_apps
from .dataset import LEGACY_DATASET_PREFIX, Dataset
from .errors import SequencesError, UserAbort
from .merge_datasets import (
    MergeColumnsMode,
    merge_usr_to_usr,
)
from .mock import add_demo_columns, create_mock_dataset
from .registry import load_registry, parse_columns_spec, register_namespace

# Compatibility exports kept for existing monkeypatch-based tests.
shutil = remotes_commands.shutil
SSHRemote = remotes_commands.SSHRemote

USR_OUTPUT_VERSION = 1
LEGACY_DATASET_PATH_ERROR = _LEGACY_DATASET_PATH_ERROR


def _resolve_output_format(args, *, default: str = "auto") -> str:
    fmt = str(getattr(args, "format", default) or default).lower()
    if fmt not in {"auto", "rich", "plain", "json"}:
        raise SequencesError(f"Unsupported format '{fmt}'. Use auto|rich|plain|json.")
    if fmt == "auto":
        if _is_interactive() and bool(getattr(args, "rich", True)):
            return "rich"
        return "plain"
    return fmt


def _print_json(payload) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def cmd_repair_densegen(args):
    tooling_commands.cmd_repair_densegen(args, deps=_tooling_deps())


# --------- dataset guessing (path-first) ----------
def _normalize_dataset_id(dataset: str) -> str:
    return dataset_commands._normalize_dataset_id(dataset)  # noqa: SLF001


def _resolve_existing_dataset_id(root: Path, dataset: str) -> str:
    return dataset_commands.resolve_existing_dataset_id(root, dataset)


def _resolve_dataset_name_interactive(root: Path, dataset: str | None, use_rich: bool) -> str | None:
    return dataset_commands.resolve_dataset_name_interactive(root, dataset, use_rich)


def _is_explicit_path_target(target: str | None) -> bool:
    text = str(target or "").strip()
    if text in {"", ".", "./", "..", "../"}:
        return True
    if text.startswith("./") or text.startswith("../") or text.startswith("~/"):
        return True
    if Path(text).is_absolute():
        return True
    if text.lower().endswith(".parquet"):
        return True
    if "/" in text or "\\" in text:
        return Path(text).expanduser().exists()
    return False


def _exit_missing_path_target(target: str) -> None:
    print(f"ERROR: Path target not found: {target}")
    raise typer.Exit(code=4)


def _resolve_dataset_for_read(root: Path, dataset_arg: str) -> Dataset:
    return _resolve_dataset_for_read_impl(
        root,
        dataset_arg,
        resolve_existing_dataset_id=_resolve_existing_dataset_id,
        normalize_dataset_id=_normalize_dataset_id,
        pkg_root=_pkg_usr_root(),
    )


def _assert_not_legacy_dataset_path(path: Path, *, root: Path | None = None) -> None:
    _assert_not_legacy_dataset_path_impl(path, root=root, pkg_root=_pkg_usr_root())


def _assert_not_legacy_dataset_path_for_read_views(path: Path, root: Path | None) -> None:
    _assert_not_legacy_dataset_path(path, root=root)


def _read_view_deps() -> read_views_commands.ReadViewDeps:
    return read_views_commands.ReadViewDeps(
        is_explicit_path_target=_is_explicit_path_target,
        exit_missing_path_target=_exit_missing_path_target,
        resolve_existing_dataset_id=_resolve_existing_dataset_id,
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        assert_not_legacy_dataset_path=_assert_not_legacy_dataset_path_for_read_views,
        legacy_dataset_prefix=LEGACY_DATASET_PREFIX,
        legacy_dataset_path_error=LEGACY_DATASET_PATH_ERROR,
    )


def _runtime_deps() -> runtime_commands.RuntimeDeps:
    return runtime_commands.RuntimeDeps(
        resolve_dataset_for_read=_resolve_dataset_for_read,
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
    )


def _materialize_deps() -> materialize_commands.MaterializeDeps:
    return materialize_commands.MaterializeDeps(
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        is_interactive=_is_interactive,
        confirm=lambda message: typer.confirm(message, default=False),
    )


def _maintenance_deps() -> maintenance_commands.MaintenanceDeps:
    return maintenance_commands.MaintenanceDeps(
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        prompt=input,
    )


def _merge_deps() -> merge_commands.MergeDeps:
    return merge_commands.MergeDeps(
        resolve_merge_policy=resolve_merge_policy,
        merge_usr_to_usr=merge_usr_to_usr,
        mode_require_same=MergeColumnsMode.REQUIRE_SAME,
        mode_union=MergeColumnsMode.UNION,
        dataset_factory=Dataset,
    )


def _namespace_deps() -> namespace_handlers_commands.NamespaceDeps:
    return namespace_handlers_commands.NamespaceDeps(
        load_registry=load_registry,
        parse_columns_spec=parse_columns_spec,
        register_namespace=register_namespace,
    )


def _tooling_deps() -> tooling_commands.ToolingDeps:
    return tooling_commands.ToolingDeps(
        resolve_dataset_name_interactive=_resolve_dataset_name_interactive,
        resolve_path_anywhere=_resolve_path_anywhere,
        create_mock_dataset=create_mock_dataset,
        add_demo_columns=add_demo_columns,
        dataset_factory=Dataset,
    )


# ---------------- path helpers: resolve paths relative to the installed package ----------------


def _pkg_usr_root() -> Path:
    """
    Return the installed dnadesign/usr package directory.
    This is stable no matter where the user runs 'usr' from.
    """
    return _pkg_usr_root_impl()


def _assert_supported_root(root: Path) -> None:
    _assert_supported_root_impl(root, pkg_root=_pkg_usr_root())


def _resolve_path_anywhere(p: Path) -> Path:
    """
    Make file arguments robust:
      1) absolute path → as-is
      2) relative path existing under CWD → as-is
      3) otherwise, try to resolve relative to the installed dnadesign/usr package,
         including common repo-style prefixes like 'src/dnadesign/usr/...'
         or 'usr/...'.
    """
    return _resolve_path_anywhere_impl(p, pkg_root=_pkg_usr_root())


# ---------- helpers & command impls ----------
def list_datasets(root: Path):
    return dataset_commands.list_datasets(root)


def cmd_ls(args):
    read_commands.cmd_ls(
        args,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
    )


def cmd_init(args):
    write_commands.cmd_init(args)


def cmd_import(args):
    write_commands.cmd_import(
        args,
        resolve_path_anywhere=_resolve_path_anywhere,
    )


def cmd_attach(args):
    write_commands.cmd_attach(
        args,
        resolve_path_anywhere=_resolve_path_anywhere,
    )


def _print_user_error(e: SequencesError) -> None:
    error_output_commands.print_user_error(e)


def cmd_info(args):
    read_commands.cmd_info(
        args,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
        resolve_dataset_for_read=_resolve_dataset_for_read,
    )


def cmd_schema(args):
    read_commands.cmd_schema(
        args,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
    )


def _resolve_parquet_from_dir(dir_path: Path, glob: str | None = None) -> Path:
    return read_views_commands._resolve_parquet_from_dir(dir_path, glob=glob)


def _resolve_parquet_target(path_like: Path, glob: str | None = None) -> Path:
    return read_views_commands._resolve_parquet_target(path_like, glob=glob)


def cmd_head(args):
    read_views_commands.cmd_head(args, deps=_read_view_deps())


def cmd_cols(args):
    read_views_commands.cmd_cols(args, deps=_read_view_deps())


def cmd_describe(args):
    read_views_commands.cmd_describe(args, deps=_read_view_deps())


def cmd_cell(args):
    read_views_commands.cmd_cell(args, deps=_read_view_deps())


def cmd_validate(args):
    runtime_commands.cmd_validate(args, deps=_runtime_deps())


def cmd_registry_freeze(args) -> None:
    maintenance_commands.cmd_registry_freeze(args, deps=_maintenance_deps())


def cmd_overlay_compact(args) -> None:
    maintenance_commands.cmd_overlay_compact(args, deps=_maintenance_deps())


def _emit_event_line(line: str, fmt: str) -> None:
    runtime_commands._emit_event_line(line, fmt)


def cmd_events_tail(args) -> None:
    runtime_commands.cmd_events_tail(args, deps=_runtime_deps())


def cmd_get(args):
    runtime_commands.cmd_get(args, deps=_runtime_deps())


def cmd_grep(args):
    runtime_commands.cmd_grep(args, deps=_runtime_deps())


def _default_export_filename(dataset_name: str, fmt: str) -> str:
    return runtime_commands._default_export_filename(dataset_name, fmt)


def _resolve_export_target(out_path: Path, *, dataset_name: str, fmt: str) -> Path:
    return runtime_commands._resolve_export_target(out_path, dataset_name=dataset_name, fmt=fmt)


def cmd_export(args):
    runtime_commands.cmd_export(args, deps=_runtime_deps())


def _collect_ids(ids: list[str] | None, id_file: Path | None) -> list[str]:
    return runtime_commands._collect_ids(ids, id_file)


def _collect_list(vals: list[str] | None) -> list[str]:
    return runtime_commands._collect_list(vals)


def cmd_delete(args):
    runtime_commands.cmd_delete(args, deps=_runtime_deps())


def cmd_restore(args):
    runtime_commands.cmd_restore(args, deps=_runtime_deps())


def cmd_state_set(args):
    runtime_commands.cmd_state_set(args, deps=_runtime_deps())


def cmd_state_clear(args):
    runtime_commands.cmd_state_clear(args, deps=_runtime_deps())


def cmd_state_get(args):
    runtime_commands.cmd_state_get(args, deps=_runtime_deps())


def cmd_materialize(args):
    materialize_commands.cmd_materialize(args, deps=_materialize_deps())


def cmd_snapshot(args):
    maintenance_commands.cmd_snapshot(args, deps=_maintenance_deps())


def cmd_convert_legacy(args):
    tooling_commands.cmd_convert_legacy(args, deps=_tooling_deps())


# ---------- make-mock ----------
def cmd_make_mock(args):
    tooling_commands.cmd_make_mock(args, deps=_tooling_deps())


# ---------- add-demo-cols ----------
def cmd_add_demo(args):
    tooling_commands.cmd_add_demo(args, deps=_tooling_deps())


# ---------- MERGE DATASETS ----------
def cmd_merge_datasets(args):
    merge_commands.cmd_merge_datasets(args, deps=_merge_deps())


# ---------- remotes commands ----------
def cmd_remotes_list(args):
    remotes_commands.cmd_remotes_list(args)


def cmd_remotes_show(args):
    remotes_commands.cmd_remotes_show(args)


def cmd_remotes_add(args):
    remotes_commands.cmd_remotes_add(args)


def cmd_remotes_wizard(args):
    remotes_commands.cmd_remotes_wizard(args)


def cmd_remotes_doctor(args):
    remotes_commands.shutil = shutil
    remotes_commands.SSHRemote = SSHRemote
    remotes_commands.cmd_remotes_doctor(args)


# ---------- namespace registry ----------
def cmd_namespace_list(args):
    namespace_handlers_commands.cmd_namespace_list(args, deps=_namespace_deps())


def cmd_namespace_show(args):
    namespace_handlers_commands.cmd_namespace_show(args, deps=_namespace_deps())


def cmd_namespace_register(args):
    namespace_handlers_commands.cmd_namespace_register(args, deps=_namespace_deps())


# ---------- diff/pull/push ----------
def cmd_diff(args):
    sync_commands.cmd_diff(
        args,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
    )


def cmd_pull(args):
    sync_commands.cmd_pull(args)


def cmd_push(args):
    sync_commands.cmd_push(args)


def cmd_dedupe_sequences(args):
    maintenance_commands.cmd_dedupe_sequences(args, deps=_maintenance_deps())


# ---------- Typer CLI (library-first adapter) ----------
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


def _ctx_args(ctx: typer.Context, **kwargs) -> NS:
    base = {"root": ctx.obj["root"], "rich": ctx.obj["rich"]}
    base.update(kwargs)
    return NS(**base)


@app.callback()
def _root(
    ctx: typer.Context,
    root: Path = typer.Option(
        (_pkg_usr_root() / "datasets").resolve(),
        "--root",
        help="Datasets root folder",
        readable=True,
        exists=True,
        dir_okay=True,
        file_okay=False,
        path_type=Path,
    ),
    rich: bool = typer.Option(True, "--rich/--no-rich", help="Use Rich formatting for supported commands"),
) -> None:
    try:
        _assert_supported_root(root)
    except SequencesError as exc:
        raise typer.BadParameter(str(exc), param_hint="--root") from exc
    ctx.obj = {"root": root, "rich": rich}


register_sync_commands(
    app,
    sync_args_builder=_ctx_args,
    cmd_diff=cmd_diff,
    cmd_pull=cmd_pull,
    cmd_push=cmd_push,
)

register_ops_commands(
    maintenance_app,
    densegen_app,
    dev_app,
    legacy_app,
    ctx_args_builder=_ctx_args,
    cmd_dedupe_sequences=cmd_dedupe_sequences,
    cmd_registry_freeze=cmd_registry_freeze,
    cmd_overlay_compact=cmd_overlay_compact,
    cmd_repair_densegen=cmd_repair_densegen,
    cmd_make_mock=cmd_make_mock,
    cmd_add_demo=cmd_add_demo,
    cmd_convert_legacy=cmd_convert_legacy,
    cmd_merge_datasets=cmd_merge_datasets,
)

register_query_commands(
    app,
    events_app,
    ctx_args_builder=_ctx_args,
    cmd_ls=cmd_ls,
    cmd_info=cmd_info,
    cmd_schema=cmd_schema,
    cmd_head=cmd_head,
    cmd_cols=cmd_cols,
    cmd_describe=cmd_describe,
    cmd_cell=cmd_cell,
    cmd_validate=cmd_validate,
    cmd_events_tail=cmd_events_tail,
    cmd_get=cmd_get,
    cmd_grep=cmd_grep,
    cmd_export=cmd_export,
)

register_lifecycle_commands(
    app,
    state_app,
    ctx_args_builder=_ctx_args,
    cmd_init=cmd_init,
    cmd_import=cmd_import,
    cmd_attach=cmd_attach,
    cmd_delete=cmd_delete,
    cmd_restore=cmd_restore,
    cmd_state_set=cmd_state_set,
    cmd_state_clear=cmd_state_clear,
    cmd_state_get=cmd_state_get,
    cmd_materialize=cmd_materialize,
    cmd_snapshot=cmd_snapshot,
)

register_remotes_commands(
    remotes_app,
    ctx_args_builder=_ctx_args,
    cmd_remotes_list=cmd_remotes_list,
    cmd_remotes_show=cmd_remotes_show,
    cmd_remotes_add=cmd_remotes_add,
    cmd_remotes_wizard=cmd_remotes_wizard,
    cmd_remotes_doctor=cmd_remotes_doctor,
)

register_namespace_commands(
    namespace_app,
    ctx_args_builder=_ctx_args,
    cmd_namespace_list=cmd_namespace_list,
    cmd_namespace_show=cmd_namespace_show,
    cmd_namespace_register=cmd_namespace_register,
)


def main() -> None:
    from .stderr_filter import maybe_install_pyarrow_sysctl_filter

    maybe_install_pyarrow_sysctl_filter()
    try:
        app()
    except UserAbort:
        raise SystemExit(130)
    except SequencesError as e:
        _print_user_error(e)
        raise SystemExit(2)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        raise SystemExit(3)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise SystemExit(4)


if __name__ == "__main__":
    main()
