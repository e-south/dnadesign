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

from dnadesign.usr.src.registry import (
    USR_STATE_COLUMNS,
    USR_STATE_NAMESPACE,
    load_registry,
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
    entries = load_registry(root, required=False)
    if USR_STATE_NAMESPACE not in entries:
        register_namespace(
            root,
            namespace=USR_STATE_NAMESPACE,
            columns=USR_STATE_COLUMNS,
            owner="usr",
            description="Reserved record-state overlay (tests).",
            overwrite=False,
        )
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
    entries = load_registry(root, required=False)
    if "mock" in entries:
        return
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64", overwrite=False)


__all__ = ["ensure_registry", "register_test_namespace"]
