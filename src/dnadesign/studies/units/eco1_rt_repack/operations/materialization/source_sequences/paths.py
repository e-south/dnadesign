"""Path constants for Eco1 conservation source-sequence bundles."""

from __future__ import annotations

from pathlib import Path

DOCS_ROOT = Path("docs/studies/eco1_rt_repack")
CONSERVATION_SOURCES = DOCS_ROOT / "workbench/provenance/conservation-sources.yaml"
DEFAULT_OUTPUT_ROOT = Path("outputs/thread/eco1_rt_conservative_v1")
DEFAULT_SOURCE_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "conservation_source_cache"
DEFAULT_SOURCE_BUNDLE_ROOT = DEFAULT_OUTPUT_ROOT / "conservation_sources"
DEFAULT_CREATED_AT = "2026-06-20T00:00:00Z"
