"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/cassette/__init__.py

Adapt cassette-owned duplex and hairpin records for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts import AdapterDescriptor, IntegrationProvider


def _build_duplex_sequence(cfg, alphabet: str):
    from .duplex_sequence_v1 import DuplexSequenceV1Adapter

    return DuplexSequenceV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_hairpin_topology(cfg, alphabet: str):
    from .hairpin_topology_v1 import HairpinTopologyV1Adapter

    return HairpinTopologyV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


PROVIDER = IntegrationProvider(
    name="cassette",
    adapters=(
        AdapterDescriptor(
            kind="duplex_sequence_v1",
            owner_tool="cassette",
            contract_kind="duplex_sequence_v1",
            supported_renderers=("sequence_rows",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_duplex_sequence,
            docs_slug="duplex-sequence-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
        AdapterDescriptor(
            kind="hairpin_topology_v1",
            owner_tool="cassette",
            contract_kind="hairpin_topology_v1",
            supported_renderers=("hairpin_cartoon",),
            supported_alphabets=("DNA",),
            factory=_build_hairpin_topology,
            docs_slug="hairpin-topology-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
    ),
)

__all__ = ["PROVIDER"]
