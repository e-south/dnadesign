"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/contracts/visual/cassette_views_manifest_v1.py

Discovery manifest for cassette visual-contract bundles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .common import VisualContractModel


class ViewReference(VisualContractModel):
    view_kind: Literal["linear_duplex_v1", "ssdna_hairpin_v1"]
    path: str


class RecommendedJobReference(VisualContractModel):
    name: str
    path: str


class CassetteViewsManifestV1(VisualContractModel):
    version: Literal[1] = 1
    kind: Literal["cassette_views_manifest_v1"] = "cassette_views_manifest_v1"
    solution_id: str
    rank: int | None = Field(default=None, ge=1)
    views: list[ViewReference]
    recommended_jobs: list[RecommendedJobReference] = Field(default_factory=list)
