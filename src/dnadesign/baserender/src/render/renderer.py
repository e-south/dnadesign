"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/renderer.py

Renderer registry and pre-render contract enforcement for Record v1.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..config import Style
from ..core import ContractError, Record, RenderingError, validate_record_kinds
from .palette import Palette


class Renderer(Protocol):
    def render(self, record: Record, style: Style, palette: Palette): ...


RendererFactory = Callable[[], Renderer]


@dataclass(frozen=True)
class RendererDescriptor:
    name: str
    topology_kinds: tuple[str, ...]
    accepted_alphabets: tuple[str, ...]
    required_record_features: tuple[str, ...]
    optional_record_features: tuple[str, ...]
    docs_slug: str


@dataclass(frozen=True)
class _RegisteredRenderer:
    descriptor: RendererDescriptor
    factory: RendererFactory


def _build_sequence_rows_renderer() -> Renderer:
    from .sequence_rows import SequenceRowsRenderer

    return SequenceRowsRenderer()


def _build_nucleotide_evidence_map_renderer() -> Renderer:
    from .sequence_rows import SequenceRowsRenderer

    return SequenceRowsRenderer()


def _build_hairpin_cartoon_renderer() -> Renderer:
    from .hairpin_cartoon import HairpinCartoonRenderer

    return HairpinCartoonRenderer()


def _build_topology_cartoon_renderer() -> Renderer:
    from .topology_cartoon import TopologyCartoonRenderer

    return TopologyCartoonRenderer()


def _build_snapback_map_renderer() -> Renderer:
    from .snapback_map import SnapbackMapRenderer

    return SnapbackMapRenderer()


@dataclass(frozen=True)
class _RendererRegistry:
    renderers: dict[str, _RegisteredRenderer]

    def get(self, name: str) -> Renderer:
        registered = self.renderers.get(name)
        if registered is None:
            raise RenderingError(f"Unknown renderer: {name}")
        return registered.factory()

    def descriptor(self, name: str) -> RendererDescriptor:
        registered = self.renderers.get(name)
        if registered is None:
            raise RenderingError(f"Unknown renderer: {name}")
        return registered.descriptor


_REGISTRY = _RendererRegistry(
    renderers={
        "sequence_rows": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="sequence_rows",
                topology_kinds=("linear_ssdna", "linear_dsdna", "fragment_pool"),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=(),
                optional_record_features=("interval_annotation", "boundary_marker"),
                docs_slug="sequence-rows",
            ),
            factory=_build_sequence_rows_renderer,
        ),
        "nucleotide_evidence_map": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="nucleotide_evidence_map",
                topology_kinds=(
                    "linear_ssdna",
                    "linear_dsdna",
                    "fragment_pool",
                    "circularized_linearized",
                    "hairpin_folded",
                    "branched_adapter",
                ),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=(),
                optional_record_features=("interval_annotation", "boundary_marker", "span_link"),
                docs_slug="nucleotide-evidence-map",
            ),
            factory=_build_nucleotide_evidence_map_renderer,
        ),
        "hairpin_cartoon": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="hairpin_cartoon",
                topology_kinds=("hairpin_ssdna", "ssdna_hairpin"),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=("pair_map",),
                optional_record_features=("interval_annotation",),
                docs_slug="hairpin-cartoon",
            ),
            factory=_build_hairpin_cartoon_renderer,
        ),
        "topology_cartoon": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="topology_cartoon",
                topology_kinds=("circular_dsdna_candidate", "branched_y", "fragment_pool", "circular_duplex"),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=(),
                optional_record_features=("interval_annotation",),
                docs_slug="topology-cartoon",
            ),
            factory=_build_topology_cartoon_renderer,
        ),
        "snapback_map": _RegisteredRenderer(
            descriptor=RendererDescriptor(
                name="snapback_map",
                topology_kinds=("linear_dsdna", "linear_ssdna", "hairpin_folded"),
                accepted_alphabets=("DNA", "IUPAC_DNA"),
                required_record_features=(),
                optional_record_features=(),
                docs_slug="snapback-map",
            ),
            factory=_build_snapback_map_renderer,
        ),
    }
)


def get_renderer(name: str) -> Renderer:
    return _REGISTRY.get(name)


def renderer_descriptors() -> tuple[RendererDescriptor, ...]:
    return tuple(_REGISTRY.descriptor(name) for name in sorted(_REGISTRY.renderers))


def get_renderer_descriptor(name: str) -> RendererDescriptor:
    return _REGISTRY.descriptor(name)


def render_record(record: Record, *, renderer_name: str, style: Style, palette: Palette):
    try:
        validated = record.validate()
        validate_record_kinds(validated)
    except ContractError as exc:
        raise RenderingError(str(exc)) from exc

    renderer = get_renderer(renderer_name)
    return renderer.render(validated, style, palette)
