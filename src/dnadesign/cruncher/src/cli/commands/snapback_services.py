"""
Lazy app/workspace service accessors for Snapback CLI commands.
"""

from __future__ import annotations


def validate_snapback_spec(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import validate_snapback_spec as _validate_snapback_spec

    return _validate_snapback_spec(*args, **kwargs)


def run_snapback_design(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import run_snapback_design as _run_snapback_design

    return _run_snapback_design(*args, **kwargs)


def run_snapback_solve(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_solve_workflow import run_snapback_solve as _run_snapback_solve

    return _run_snapback_solve(*args, **kwargs)


def run_snapback_target_search(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_target_search_workflow import (
        run_snapback_target_search as _run_snapback_target_search,
    )

    return _run_snapback_target_search(*args, **kwargs)


def validate_released_snapback_spec(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_workflow import (
        validate_released_snapback_spec as _validate_released_snapback_spec,
    )

    return _validate_released_snapback_spec(*args, **kwargs)


def run_released_snapback_design(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_workflow import (
        run_released_snapback_design as _run_released_snapback_design,
    )

    return _run_released_snapback_design(*args, **kwargs)


def run_released_snapback_target_search(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_target_search_workflow import (
        run_released_snapback_target_search as _run_released_snapback_target_search,
    )

    return _run_released_snapback_target_search(*args, **kwargs)


def run_released_snapback_solve(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_solve_workflow import (
        run_released_snapback_solve as _run_released_snapback_solve,
    )

    return _run_released_snapback_solve(*args, **kwargs)


def snapback_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workflow import snapback_show_payload as _snapback_show_payload

    return _snapback_show_payload(*args, **kwargs)


def released_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_released_show import released_show_payload as _released_show_payload

    return _released_show_payload(*args, **kwargs)


def init_snapback_workspace(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workspace_service import init_snapback_workspace as _init_snapback_workspace

    return _init_snapback_workspace(*args, **kwargs)


def snapback_workspace_path(*args, **kwargs):
    from dnadesign.cruncher.app.snapback_workspace_service import snapback_workspace_path as _snapback_workspace_path

    return _snapback_workspace_path(*args, **kwargs)


__all__ = [
    "init_snapback_workspace",
    "released_show_payload",
    "run_released_snapback_design",
    "run_released_snapback_solve",
    "run_released_snapback_target_search",
    "run_snapback_design",
    "run_snapback_solve",
    "run_snapback_target_search",
    "snapback_show_payload",
    "snapback_workspace_path",
    "validate_released_snapback_spec",
    "validate_snapback_spec",
]
