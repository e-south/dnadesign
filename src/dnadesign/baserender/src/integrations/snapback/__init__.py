"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/snapback/__init__.py

Adapt Snapback visual records for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts import AdapterDescriptor, IntegrationProvider


def _build_adapter(cfg, alphabet: str):
    from .visual_v1 import SnapbackVisualV1Adapter

    return SnapbackVisualV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


PROVIDER = IntegrationProvider(
    name="snapback",
    adapters=(
        AdapterDescriptor(
            kind="snapback_visual_v1",
            owner_tool="snapback",
            contract_kind="snapback_visual_v1",
            supported_renderers=("snapback_map",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_adapter,
            docs_slug="snapback-visual-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
    ),
)

__all__ = ["PROVIDER"]
