"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/yiu/__init__.py

Adapt YIU visual records for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts import AdapterDescriptor, IntegrationProvider


def _build_linear_state(cfg, alphabet: str):
    from .linear_state_v1 import YiuLinearStateV1Adapter

    return YiuLinearStateV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_payload_visual(cfg, alphabet: str):
    from .payload_visual_v1 import YiuPayloadVisualV1Adapter

    return YiuPayloadVisualV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_hairpin_topology(cfg, alphabet: str):
    from .hairpin_topology_v1 import YiuHairpinTopologyV1Adapter

    return YiuHairpinTopologyV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_topology_cartoon(cfg, alphabet: str):
    from .topology_cartoon_v1 import YiuTopologyCartoonV1Adapter

    return YiuTopologyCartoonV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


PROVIDER = IntegrationProvider(
    name="yiu",
    adapters=(
        AdapterDescriptor(
            kind="yiu_linear_state_v1",
            owner_tool="yiu",
            contract_kind="yiu_linear_state_v1",
            supported_renderers=("sequence_rows",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_linear_state,
            docs_slug="yiu-linear-state-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
        AdapterDescriptor(
            kind="yiu_payload_visual_v1",
            owner_tool="yiu",
            contract_kind="yiu_payload_visual_v1",
            supported_renderers=("nucleotide_evidence_map",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_payload_visual,
            docs_slug="yiu-payload-visual-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
        AdapterDescriptor(
            kind="yiu_hairpin_topology_v1",
            owner_tool="yiu",
            contract_kind="yiu_hairpin_topology_v1",
            supported_renderers=("hairpin_cartoon",),
            supported_alphabets=("DNA",),
            factory=_build_hairpin_topology,
            docs_slug="yiu-hairpin-topology-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
        AdapterDescriptor(
            kind="yiu_topology_cartoon_v1",
            owner_tool="yiu",
            contract_kind="yiu_topology_cartoon_v1",
            supported_renderers=("topology_cartoon",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_topology_cartoon,
            docs_slug="yiu-topology-cartoon-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
    ),
)

__all__ = ["PROVIDER"]
