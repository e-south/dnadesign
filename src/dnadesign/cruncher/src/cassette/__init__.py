"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/__init__.py

Public cassette workflow contracts for Cruncher.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.cruncher.cassette.catalog import load_nickase_catalog, resolve_catalog_path
from dnadesign.cruncher.cassette.errors import (
    CassetteError,
    CassettePlanningError,
    CassetteSpecError,
    NickaseCatalogError,
)
from dnadesign.cruncher.cassette.load import load_cassette_spec, resolve_workspace_root_for_spec
from dnadesign.cruncher.cassette.models import (
    CassetteEvaluationReport,
    HairpinCassetteSpec,
    NickaseCatalog,
    NickaseCatalogEntry,
)
from dnadesign.cruncher.cassette.planner import build_cassette_report, render_markdown_report

__all__ = [
    "CassetteError",
    "CassetteEvaluationReport",
    "CassettePlanningError",
    "CassetteSpecError",
    "HairpinCassetteSpec",
    "NickaseCatalog",
    "NickaseCatalogEntry",
    "NickaseCatalogError",
    "build_cassette_report",
    "load_cassette_spec",
    "load_nickase_catalog",
    "render_markdown_report",
    "resolve_catalog_path",
    "resolve_workspace_root_for_spec",
]
