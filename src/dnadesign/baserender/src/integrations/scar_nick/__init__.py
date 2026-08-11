"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/scar_nick/__init__.py

Adapt scar-nick visual records for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts import AdapterDescriptor, IntegrationProvider


def _build_adapter(cfg, alphabet: str):
    from .visual_v1 import ScarNickVisualV1Adapter

    return ScarNickVisualV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


PROVIDER = IntegrationProvider(
    name="scar_nick",
    adapters=(
        AdapterDescriptor(
            kind="scar_nick_visual_v1",
            owner_tool="scar_nick",
            contract_kind="scar_nick_visual_v1",
            supported_renderers=("nucleotide_evidence_map",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_adapter,
            docs_slug="scar-nick-visual-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
    ),
)

__all__ = ["PROVIDER"]
