"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/packaged.py

Build and read portable catalog metadata for installed OPS commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

from .models import (
    CatalogProcedureEntry,
    CatalogProcedureRelation,
    CatalogProcedureToolRouteReference,
    CatalogToolRouteEntry,
    CatalogToolSourceEntry,
)


def packaged_catalog_path() -> Path:
    return Path(str(files("dnadesign.ops").joinpath("catalog", "builtin.json")))


def compile_catalog(repo_root: Path, *, version: str) -> str:
    """Compile checked-in metadata only, without importing external providers."""
    from .loader import load_runbook_catalog

    catalog = load_runbook_catalog(repo_root=repo_root, include_external=False)
    payload = {
        "schema_version": 1,
        "procedures": [asdict(entry) for entry in catalog.procedures],
        "tool_sources": [asdict(entry) for entry in catalog.tool_sources],
        "procedure_relations": {
            key: [asdict(relation) for relation in values] for key, values in catalog.procedure_relations.items()
        },
    }

    def public_link(relative_path: str) -> str:
        path = (catalog.catalog_path.parent / relative_path).resolve().relative_to(repo_root.resolve())
        return f"https://github.com/e-south/dnadesign/blob/v{version}/{path.as_posix()}"

    for entry in payload["procedures"] + payload["tool_sources"]:
        entry["doc_path"] = public_link(entry["doc_path"])
        for route in entry.get("routes", ()):
            route["doc_path"] = public_link(route["doc_path"])
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def read_catalog(
    path: Path,
) -> tuple[
    tuple[CatalogProcedureEntry, ...],
    tuple[CatalogToolSourceEntry, ...],
    dict[str, tuple[CatalogProcedureRelation, ...]],
]:
    """Restore typed catalog entries; runtime discovery validates their relations."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported packaged catalog schema_version")
    procedures = tuple(
        CatalogProcedureEntry(
            **{
                **entry,
                "keywords": tuple(entry["keywords"]),
                "related_tools": tuple(entry["related_tools"]),
                "related_tool_routes": tuple(
                    CatalogProcedureToolRouteReference(**route) for route in entry["related_tool_routes"]
                ),
            }
        )
        for entry in payload["procedures"]
    )
    tool_sources = tuple(
        CatalogToolSourceEntry(
            **{
                **entry,
                "keywords": tuple(entry["keywords"]),
                "routes": tuple(CatalogToolRouteEntry(**route) for route in entry["routes"]),
            }
        )
        for entry in payload["tool_sources"]
    )
    relations = {
        key: tuple(CatalogProcedureRelation(**relation) for relation in values)
        for key, values in payload["procedure_relations"].items()
    }
    return procedures, tool_sources, relations
