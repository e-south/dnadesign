"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/motif_annotation/__init__.py

Provide optional motif annotation transforms for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import IntegrationProvider, StyleProfileDescriptor, TransformDescriptor

MOTIF_SHOWCASE_PROFILE = "motif_showcase.v1"


def _motif_showcase_style() -> dict[str, object]:
    from .style import motif_showcase_style_overrides

    return motif_showcase_style_overrides()


def _build_library_transform(params: Mapping[str, Any]):
    from .motifs_from_library import AttachMotifsFromLibraryTransform

    return AttachMotifsFromLibraryTransform(**params)


def _build_sigma70_transform(params: Mapping[str, Any]):
    from .sigma70 import Sigma70Transform

    return Sigma70Transform(**params)


PROVIDER = IntegrationProvider(
    name="motif_annotation",
    transforms=(
        TransformDescriptor(
            name="attach_motifs_from_library",
            owner_tool=None,
            factory=_build_library_transform,
            docs_slug="motif-library",
            allowed_params=("library_path", "tf_tag_prefix", "require_effect"),
            required_params=("library_path",),
            path_params=("library_path",),
        ),
        TransformDescriptor(
            name="sigma70",
            owner_tool=None,
            factory=_build_sigma70_transform,
            docs_slug="sigma70-annotation",
            allowed_params=(
                "variants",
                "spacer_min",
                "spacer_max",
                "label_mode",
                "label_text",
                "inner_margin_bp",
                "on_multiple_matches",
            ),
        ),
    ),
    style_profiles=(
        StyleProfileDescriptor(
            name=MOTIF_SHOWCASE_PROFILE,
            owner_tool=None,
            docs_slug="motif-showcase",
            style_factory=_motif_showcase_style,
        ),
    ),
)

__all__ = ["MOTIF_SHOWCASE_PROFILE", "PROVIDER"]
