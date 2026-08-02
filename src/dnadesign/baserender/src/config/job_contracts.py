"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/config/job_contracts.py

Explicit render-contract descriptors for BaseRender job-like YAML contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..core import SchemaError


@dataclass(frozen=True)
class InputEnvelope:
    max_bytes: int
    max_records: int
    max_bases: int
    base_field_path: tuple[str, ...]
    accepted_input_kinds: tuple[str, ...]


THREE_WAY_JUNCTION_REVIEW_INPUT_ENVELOPE = InputEnvelope(
    max_bytes=64 * 1024 * 1024,
    max_records=2_000,
    max_bases=10_000_000,
    base_field_path=("target", "sequence_5to3"),
    accepted_input_kinds=("json",),
)


@dataclass(frozen=True)
class RenderContractDescriptor:
    kind: str
    schema_version: int
    display_name: str
    purpose: str
    accepted_renderers: tuple[str, ...]
    compatibility_aliases: tuple[str, ...] = ()
    docs_slug: str | None = None
    sensitivity: Literal["public", "private"] = "public"
    input_envelope: InputEnvelope | None = None


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
            "snapback_map",
            "three_way_junction_review",
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
    "usr_genbank_annotation_render_v1": RenderContractDescriptor(
        kind="usr_genbank_annotation_render_v1",
        schema_version=1,
        display_name="USR GenBank annotation render contract",
        purpose=("Linear sequence-row visualization for USR datasets with seq_annot GenBank feature overlays."),
        accepted_renderers=("sequence_rows",),
        docs_slug="usr-genbank-annotation-render-v1",
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
    "snapback_map_render_v3": RenderContractDescriptor(
        kind="snapback_map_render_v3",
        schema_version=3,
        display_name="Snapback map render contract",
        purpose="Snapback map visualization for explicit snapback visual payload contracts.",
        accepted_renderers=("snapback_map",),
        docs_slug="snapback-map-render-v3",
    ),
    "three_way_junction_review_render_v1": RenderContractDescriptor(
        kind="three_way_junction_review_render_v1",
        schema_version=1,
        display_name="Three-way-junction review render contract",
        purpose="Four-panel QA rendering from explicit three-way-junction design evidence.",
        accepted_renderers=("three_way_junction_review",),
        docs_slug="three-way-junction-review-render-v1",
        sensitivity="private",
        input_envelope=THREE_WAY_JUNCTION_REVIEW_INPUT_ENVELOPE,
    ),
}

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
    "THREE_WAY_JUNCTION_REVIEW_INPUT_ENVELOPE",
    "RenderContractDescriptor",
    "render_contract_descriptor",
    "render_contract_descriptors",
    "render_contract_kinds",
    "render_contract_renderer_kinds",
    "validate_render_contract_renderer",
]
