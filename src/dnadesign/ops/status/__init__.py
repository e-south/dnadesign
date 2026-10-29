"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/status/__init__.py

Neutral status/observation public surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .models import (
    STATE_SEVERITY,
    STATUS_STATES,
    CampaignScaffold,
    CampaignScaffoldStep,
    CampaignStatus,
    InputFieldSpec,
    ProcedureStatus,
    StatusKindSpec,
    StatusState,
    combine_states,
    state_counts,
)
from .parsing import (
    optional_positive_int,
    required_metadata_text,
    required_text,
    string_list_or_empty,
    string_or_none,
)
from .path_ref import PathBase, resolve_path_ref
from .paths import (
    flatten_named_paths,
    path_or_none,
    required_path,
    resolve_input_path,
    resolve_named_path_mapping,
    resolve_repo_relative_path,
)


def file_count(*args, **kwargs):
    from .artifacts import file_count as _file_count

    return _file_count(*args, **kwargs)


def line_count(*args, **kwargs):
    from .artifacts import line_count as _line_count

    return _line_count(*args, **kwargs)


def load_yaml_mapping(*args, **kwargs):
    from .artifacts import load_yaml_mapping as _load_yaml_mapping

    return _load_yaml_mapping(*args, **kwargs)


def list_status_kind_specs(*args, **kwargs):
    from .registry_loader import list_status_kind_specs as _list_status_kind_specs

    return _list_status_kind_specs(*args, **kwargs)


def list_status_kind_specs_for_repo(*args, **kwargs):
    from .registry_loader import list_status_kind_specs_for_repo as _list_status_kind_specs_for_repo

    return _list_status_kind_specs_for_repo(*args, **kwargs)


def namespace_column_counts(*args, **kwargs):
    from .artifacts import namespace_column_counts as _namespace_column_counts

    return _namespace_column_counts(*args, **kwargs)


def overlay_namespace_names(*args, **kwargs):
    from .artifacts import overlay_namespace_names as _overlay_namespace_names

    return _overlay_namespace_names(*args, **kwargs)


def parquet_row_count(*args, **kwargs):
    from .artifacts import parquet_row_count as _parquet_row_count

    return _parquet_row_count(*args, **kwargs)


__all__ = [
    "CampaignScaffold",
    "CampaignScaffoldStep",
    "CampaignStatus",
    "InputFieldSpec",
    "PathBase",
    "ProcedureStatus",
    "STATE_SEVERITY",
    "STATUS_STATES",
    "StatusKindSpec",
    "StatusState",
    "combine_states",
    "file_count",
    "flatten_named_paths",
    "line_count",
    "list_status_kind_specs",
    "list_status_kind_specs_for_repo",
    "load_yaml_mapping",
    "namespace_column_counts",
    "optional_positive_int",
    "overlay_namespace_names",
    "parquet_row_count",
    "path_or_none",
    "required_metadata_text",
    "required_path",
    "required_text",
    "resolve_input_path",
    "resolve_named_path_mapping",
    "resolve_path_ref",
    "resolve_repo_relative_path",
    "state_counts",
    "string_list_or_empty",
    "string_or_none",
]
