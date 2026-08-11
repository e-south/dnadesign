"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/integrations/generic/__init__.py

Adapt producer-neutral sequence records for BaseRender.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..contracts import AdapterDescriptor, IntegrationProvider


def _build_generic(cfg, alphabet: str):
    from .generic_features import GenericFeaturesAdapter

    return GenericFeaturesAdapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_sequence_windows(cfg, alphabet: str):
    from .sequence_windows_v1 import SequenceWindowsV1Adapter

    return SequenceWindowsV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


def _build_sequence_evidence_map(cfg, alphabet: str):
    from .sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter

    return SequenceEvidenceMapV1Adapter(columns=cfg.columns, policies=cfg.policies, alphabet=alphabet)


PROVIDER = IntegrationProvider(
    name="generic",
    adapters=(
        AdapterDescriptor(
            kind="generic_features",
            owner_tool=None,
            contract_kind="generic_features",
            supported_renderers=("sequence_rows",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_generic,
            docs_slug="generic-features",
            allowed_config_columns=("sequence", "features", "effects", "display", "id"),
            required_config_columns=("sequence", "features"),
            required_source_columns=("sequence", "features"),
            optional_source_columns=("effects", "display", "id"),
        ),
        AdapterDescriptor(
            kind="sequence_windows_v1",
            owner_tool=None,
            contract_kind="sequence_windows_v1",
            supported_renderers=("sequence_rows",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_sequence_windows,
            docs_slug="sequence-windows-v1",
            allowed_config_columns=("id", "sequence", "regulator_windows", "motifs", "display"),
            required_config_columns=("sequence", "regulator_windows"),
            required_source_columns=("sequence", "regulator_windows"),
            optional_source_columns=("id", "motifs", "display"),
        ),
        AdapterDescriptor(
            kind="sequence_evidence_map_v1",
            owner_tool=None,
            contract_kind="sequence_evidence_map_v1",
            supported_renderers=("nucleotide_evidence_map",),
            supported_alphabets=("DNA", "IUPAC_DNA"),
            factory=_build_sequence_evidence_map,
            docs_slug="sequence-evidence-map-v1",
            allowed_config_columns=(),
            required_config_columns=(),
            required_source_columns=(),
        ),
    ),
)

__all__ = ["PROVIDER"]
