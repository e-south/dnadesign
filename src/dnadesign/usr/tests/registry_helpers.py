"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/registry_helpers.py

Test helpers for registering USR overlay namespaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dnadesign.usr.src.registry import parse_columns_spec, register_namespace


def register_test_namespace(
    root: Path,
    *,
    namespace: str,
    columns_spec: str,
    owner: Optional[str] = "tests",
    description: Optional[str] = "test namespace",
    overwrite: bool = True,
) -> Path:
    cols = parse_columns_spec(columns_spec, namespace=namespace)
    return register_namespace(
        root,
        namespace=namespace,
        columns=cols,
        owner=owner,
        description=description,
        overwrite=overwrite,
    )
