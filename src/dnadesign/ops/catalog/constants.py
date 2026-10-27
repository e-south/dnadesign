"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/catalog/constants.py

Shared constants for the Ops runbook catalog loader.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re

LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
TITLE_HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
PROCEDURES_SECTION_HEADING = "### Cross-tool procedures"
PROCEDURES_SECTION_INTRO = (
    "This table is generated from `*.registry.yaml` sidecars. Edit those files instead of hand-editing rows here."
)
TOOL_SOURCES_SECTION_HEADING = "### Tool docs"
TOOL_SOURCES_SECTION_INTRO = (
    "This table is generated from `*.tool-source.yaml` sidecars. Edit those files instead of hand-editing rows here."
)
REGISTRY_METADATA_SUFFIX = ".registry.yaml"
TOOL_SOURCE_METADATA_SUFFIX = ".tool-source.yaml"
ALLOWED_RELATION_TYPES = frozenset(
    {
        "alternative-to",
        "depends-on",
        "execution-support",
        "handoff-to",
        "see-also",
    }
)
METADATA_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9-]*(?:-[a-z0-9]+)*$")
TOOL_SOURCE_KEYWORD_PATTERN = re.compile(r"^[a-z0-9][a-z0-9 _-]*$")
