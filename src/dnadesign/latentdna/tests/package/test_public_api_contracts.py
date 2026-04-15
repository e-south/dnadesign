"""
Public facade contracts for latentdna.
"""

from __future__ import annotations

from dnadesign.latentdna.src import api as public_api
from dnadesign.latentdna.src import workspaces


def test_workspace_package_keeps_narrow_explicit_surface() -> None:
    assert workspaces.__all__ == [
        "CoordinateSpaceError",
        "WorkspaceContext",
        "WorkspaceValidationError",
        "load_workspace_config",
    ]


def test_execution_helper_surface_does_not_reexport_workspace_contracts() -> None:
    assert "CoordinateSpaceError" not in public_api.__all__
    assert "WorkspaceValidationError" not in public_api.__all__
    assert "load_workspace_config" not in public_api.__all__
