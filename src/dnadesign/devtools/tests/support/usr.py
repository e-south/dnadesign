"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/support/usr.py

Repo-level USR fixture helpers shared across sibling package tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.usr import (
    SchemaError,
    parse_columns_spec,
    register_namespace,
)


def register_test_namespace(
    root: Path,
    *,
    namespace: str,
    columns_spec: str,
    owner: str | None = "tests",
    description: str | None = "test namespace",
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


def ensure_registry(root: Path) -> None:
    try:
        register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64", overwrite=False)
    except SchemaError as exc:
        if "already registered" not in str(exc):
            raise


__all__ = ["ensure_registry", "register_test_namespace"]
