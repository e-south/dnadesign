"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/promoter_panel/__init__.py

Provide the promoter sequence-panel presentation profile.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts import IntegrationProvider, StyleProfileDescriptor

PROMOTER_COMPACT_SLIDE_PROFILE = "promoter_compact_slide.v1"


def _style():
    from .style import promoter_compact_slide_style

    return promoter_compact_slide_style()


PROVIDER = IntegrationProvider(
    name="promoter_panel",
    style_profiles=(
        StyleProfileDescriptor(
            name=PROMOTER_COMPACT_SLIDE_PROFILE,
            owner_tool=None,
            docs_slug="promoter-compact-slide",
            style_factory=_style,
        ),
    ),
)

__all__ = ["PROMOTER_COMPACT_SLIDE_PROFILE", "PROVIDER"]
