"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/job_contracts.py

Explicit render-contract descriptors for BaseRender job-like YAML contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..core import InputEnvelope, RenderContractDescriptor, SchemaError
from ..integrations import registered_render_contracts

_DESCRIPTORS: dict[str, RenderContractDescriptor] = {
    "render_job_v4": RenderContractDescriptor(
        kind="render_job_v4",
        schema_version=4,
        display_name="BaseRender orchestration contract",
        purpose="Generic adapter -> renderer -> output orchestration for BaseRender records.",
        accepted_renderers=(
            "sequence_rows",
            "nucleotide_evidence_map",
            "hairpin_cartoon",
            "topology_cartoon",
            "junction_annealed_fragments",
            "junction_three_way_assembly",
        ),
        docs_slug="render-job-v4",
    ),
    "sequence_rows_render_v3": RenderContractDescriptor(
        kind="sequence_rows_render_v3",
        schema_version=3,
        display_name="Sequence rows render contract",
        purpose="Linear sequence-row visualization for sequence features, motifs, and interval annotations.",
        accepted_renderers=("sequence_rows",),
        docs_slug="sequence-rows-render-v3",
    ),
    "nucleotide_evidence_map_render_v3": RenderContractDescriptor(
        kind="nucleotide_evidence_map_render_v3",
        schema_version=3,
        display_name="Nucleotide evidence map render contract",
        purpose="Payload/evidence-map visualization for nucleotide-level ownership, effects, and boundaries.",
        accepted_renderers=("nucleotide_evidence_map",),
        docs_slug="nucleotide-evidence-map-render-v3",
    ),
    "hairpin_cartoon_render_v3": RenderContractDescriptor(
        kind="hairpin_cartoon_render_v3",
        schema_version=3,
        display_name="Hairpin cartoon render contract",
        purpose="Hairpin topology visualization from explicit hairpin topology payloads.",
        accepted_renderers=("hairpin_cartoon",),
        docs_slug="hairpin-cartoon-render-v3",
    ),
    "topology_cartoon_render_v3": RenderContractDescriptor(
        kind="topology_cartoon_render_v3",
        schema_version=3,
        display_name="Topology cartoon render contract",
        purpose="Topology cartoon visualization for explicit segment-geometry payloads.",
        accepted_renderers=("topology_cartoon",),
        docs_slug="topology-cartoon-render-v3",
    ),
}

for _integration_descriptor in registered_render_contracts():
    if _integration_descriptor.kind in _DESCRIPTORS:
        raise RuntimeError(f"Duplicate BaseRender render contract: {_integration_descriptor.kind}")
    _DESCRIPTORS[_integration_descriptor.kind] = _integration_descriptor

_ALIASES: dict[str, str] = {
    alias: descriptor.kind for descriptor in _DESCRIPTORS.values() for alias in descriptor.compatibility_aliases
}


def render_contract_descriptors() -> tuple[RenderContractDescriptor, ...]:
    return tuple(_DESCRIPTORS[kind] for kind in sorted(_DESCRIPTORS))


def render_contract_kinds(*, include_aliases: bool = False) -> tuple[str, ...]:
    kinds = set(_DESCRIPTORS)
    if include_aliases:
        kinds.update(_ALIASES)
    return tuple(sorted(kinds))


def render_contract_renderer_kinds() -> tuple[str, ...]:
    """Return renderer names declared by at least one render contract."""

    return tuple(
        sorted({renderer for descriptor in _DESCRIPTORS.values() for renderer in descriptor.accepted_renderers})
    )


def render_contract_descriptor(kind: str) -> RenderContractDescriptor:
    raw = str(kind).strip()
    canonical = _ALIASES.get(raw, raw)
    descriptor = _DESCRIPTORS.get(canonical)
    if descriptor is None:
        allowed = ", ".join(render_contract_kinds(include_aliases=True))
        raise SchemaError(f"Unsupported render contract kind: {kind!r}; allowed values: {allowed}")
    return descriptor


def validate_render_contract_renderer(kind: str, renderer: str, *, field: str) -> None:
    descriptor = render_contract_descriptor(kind)
    if renderer not in descriptor.accepted_renderers:
        allowed = ", ".join(descriptor.accepted_renderers)
        raise SchemaError(
            f"{field} {kind!r} is not compatible with render.renderer {renderer!r}; "
            f"supported render.renderer values: {allowed}"
        )


DEFAULT_RENDER_CONTRACT_KIND = "render_job_v4"


__all__ = [
    "DEFAULT_RENDER_CONTRACT_KIND",
    "InputEnvelope",
    "RenderContractDescriptor",
    "render_contract_descriptor",
    "render_contract_descriptors",
    "render_contract_kinds",
    "render_contract_renderer_kinds",
    "validate_render_contract_renderer",
]
