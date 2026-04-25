"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/catalog.py

Compatibility wrapper around the shared nickase catalog seam.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

from dnadesign.cruncher.cassette.errors import NickaseCatalogError
from dnadesign.cruncher.cassette.models import HairpinCassetteSpec
from dnadesign.cruncher.nickases.catalog import (
    dump_nickase_catalog_payload as _dump_nickase_catalog_payload,
)
from dnadesign.cruncher.nickases.catalog import (
    dump_nickase_catalog_yaml as _dump_nickase_catalog_yaml,
)
from dnadesign.cruncher.nickases.catalog import (
    load_builtin_nickase_catalog_preset as _load_builtin_nickase_catalog_preset,
)
from dnadesign.cruncher.nickases.catalog import (
    load_merged_nickase_catalog as _load_merged_nickase_catalog,
)
from dnadesign.cruncher.nickases.catalog import (
    load_nickase_catalog as _load_nickase_catalog,
)
from dnadesign.cruncher.nickases.catalog import (
    merge_nickase_catalogs as _merge_nickase_catalogs,
)
from dnadesign.cruncher.nickases.catalog import (
    read_builtin_nickase_catalog_preset_text as _read_builtin_nickase_catalog_preset_text,
)
from dnadesign.cruncher.nickases.catalog import (
    resolve_workspace_relative_path as _resolve_workspace_relative_path,
)
from dnadesign.cruncher.nickases.errors import NickaseCatalogError as SharedNickaseCatalogError

T = TypeVar("T")


def _translate_catalog_error(fn: Callable[..., T], /, *args, **kwargs) -> T:
    try:
        return fn(*args, **kwargs)
    except SharedNickaseCatalogError as exc:
        raise NickaseCatalogError(str(exc)) from exc


def resolve_catalog_path(spec: HairpinCassetteSpec, *, workspace_root: Path) -> Path:
    return _translate_catalog_error(
        _resolve_workspace_relative_path,
        spec.catalog.path,
        workspace_root=workspace_root,
        label="catalog.path",
    )


def load_builtin_nickase_catalog_preset(preset_id: str):
    return _translate_catalog_error(_load_builtin_nickase_catalog_preset, preset_id)


def read_builtin_nickase_catalog_preset_text(preset_id: str) -> str:
    return _translate_catalog_error(_read_builtin_nickase_catalog_preset_text, preset_id)


def load_nickase_catalog(path: Path):
    return _translate_catalog_error(_load_nickase_catalog, path)


def merge_nickase_catalogs(*catalogs):
    return _translate_catalog_error(_merge_nickase_catalogs, *catalogs)


def load_merged_nickase_catalog(
    *,
    preset_id: str | None,
    additional_preset_ids: list[str] | None = None,
    additional_paths: list[Path],
    workspace_root: Path,
):
    return _translate_catalog_error(
        _load_merged_nickase_catalog,
        preset_id=preset_id,
        additional_preset_ids=additional_preset_ids,
        additional_paths=additional_paths,
        workspace_root=workspace_root,
    )


def dump_nickase_catalog_payload(catalog) -> dict[str, Any]:
    return _translate_catalog_error(_dump_nickase_catalog_payload, catalog)


def dump_nickase_catalog_yaml(catalog) -> str:
    return _translate_catalog_error(_dump_nickase_catalog_yaml, catalog)
